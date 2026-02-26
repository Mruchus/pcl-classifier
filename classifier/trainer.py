import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    TrainerCallback,
    get_linear_schedule_with_warmup
)
import itertools
import copy

from data_utils import prepare_data, prepare_test_data, load_span_data, tokenize_with_spans
from model import PCLClassifier

def set_seed(seed=127):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(127)   # use seed 127

# ---------- Custom Trainer (multi‑task) ----------
class PCLTrainer(Trainer):
    def __init__(self, alpha=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")               # paragraph labels
        token_labels = inputs.pop("token_labels")   # token labels
        seq_logits, token_logits = model(**inputs)

        # paragraph loss (binary cross‑entropy)
        loss_fct_seq = nn.BCEWithLogitsLoss()
        seq_loss = loss_fct_seq(seq_logits.squeeze(-1), labels.float())

        # token loss (only over active tokens)
        loss_fct_token = nn.BCEWithLogitsLoss(reduction='none')
        token_loss = loss_fct_token(token_logits, token_labels.float())
        token_loss = (token_loss * inputs['attention_mask']).sum() / inputs['attention_mask'].sum()

        # combined loss
        loss = self.alpha * seq_loss + (1 - self.alpha) * token_loss

        return (loss, (seq_logits, token_logits)) if return_outputs else loss

# ---------- Callback to check NaN gradients ----------
class CheckNaNGradCallback(TrainerCallback):
    def on_step_end(self, args, state, control, model=None, **kwargs):
        # check for NaN gradients after each step
        # stop training if any are found
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"NaN gradient detected in {name} at step {state.global_step}")
                control.should_training_stop = True
                return control
        return control

if __name__ == "__main__":
    # load and split data according to official train/dev IDs
    train_df, dev_df = prepare_data(
        "dontpatronizeme_pcl.tsv",
        "SemEval 2022 Train Labels.csv",
        "Semeval 2022 Dev Labels.csv"
    )

    # load span annotations from the categories file
    spans_by_par = load_span_data("Dont Patronize Me Categories.tsv")

    # reset index to bring 'id' back as a column, but ensure it's named 'id'
    train_df_reset = train_df.reset_index()   # no drop=True
    dev_df_reset = dev_df.reset_index()

    # if the index became a column named 'index', rename it to 'id'
    if 'index' in train_df_reset.columns:
        train_df_reset.rename(columns={'index': 'id'}, inplace=True)
    if 'index' in dev_df_reset.columns:
        dev_df_reset.rename(columns={'index': 'id'}, inplace=True)

    # optional: print columns to confirm (remove after debugging)
    print("Train columns:", train_df_reset.columns.tolist())
    print("Dev columns:", dev_df_reset.columns.tolist())

    raw_datasets = DatasetDict({
        "train": Dataset.from_pandas(train_df_reset),
        "validation": Dataset.from_pandas(dev_df_reset),
    })

    MODEL_NAME = "microsoft/deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Ensure pad token is set (DeBERTa already has one, but just in case)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # tokenise both splits, now with token‑level labels from spans
    max_length = 256
    tokenized_datasets = raw_datasets.map(
        lambda batch: tokenize_with_spans(batch, tokenizer, max_length, spans_by_par),
        batched=True,
        # DO NOT remove columns here – we'll select later
    )

    # keep only columns needed for training
    keep_columns = ['input_ids', 'attention_mask', 'labels', 'token_labels']
    tokenized_datasets = tokenized_datasets.select_columns(keep_columns)

    # compute weights for each class for the sequence loss (optional, can also tune alpha)
    labels_train = tokenized_datasets["train"]["labels"]
    class_counts = torch.bincount(torch.tensor(labels_train))
    class_weights_seq = 1.0 / class_counts.float()
    print(f"Class counts: {class_counts}, weights: {class_weights_seq}")

    # prepare test set (no token labels needed)
    test_df = prepare_test_data("Task 4 Test.tsv")
    test_dataset = Dataset.from_pandas(test_df[['text']])
    def tokenize_test(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)
    test_tokenized = test_dataset.map(tokenize_test, batched=True, remove_columns=['text'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define hyperparameter grid
    lr_list = [8e-7, 9e-7, 1e-6, 1.2e-6]
    wd_list = [0.01, 0.05, 0.1, 0.2]
    accum_list = [1, 2, 4]                # gradient accumulation steps -> effective batch sizes 16,32,64
    warmup_list = [500, 1000, 1500, 2000]
    alpha_list = [0.3, 0.5, 0.7]           # weight of sequence loss vs token loss

    # generate all combinations
    param_combinations = list(itertools.product(lr_list, wd_list, accum_list, warmup_list, alpha_list))
    print(f"Total combinations: {len(param_combinations)}")

    # For quick testing, you can restrict to one combination
    # param_combinations = [(1e-6, 0.1, 2, 1500, 0.5)]

    best_f1_overall = 0.0
    best_params = None
    best_dev_preds = None
    best_test_preds = None
    best_true_labels = None
    best_threshold = None

    # check every hyperparameter combination
    for lr, wd, accum, warmup, alpha in param_combinations:
        print(f"\n--- Testing params: lr={lr}, wd={wd}, accum={accum}, warmup={warmup}, alpha={alpha} ---")

        # create a fresh model for each run (custom multi‑task model)
        model = PCLClassifier(MODEL_NAME)   # <-- direct instantiation, not from_pretrained
        model.to(device)

        # feed parameters to optimiser
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
            eps=1e-6
        )

        # compute total training steps and set warmup
        batch_size = 16
        num_epochs = 10
        total_steps = (len(tokenized_datasets["train"]) // batch_size // accum) * num_epochs
        warmup_steps = warmup

        # learning rate scheduler: linear warmup + linear decay
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # training arguments (different output dir for each run to avoid collisions)
        training_args = TrainingArguments(
            output_dir=f"./runs/lr{lr}_wd{wd}_acc{accum}_warm{warmup}_a{alpha}",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=accum,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            warmup_steps=warmup_steps,
            logging_steps=50,
            max_grad_norm=1.0,                     # gradient clipping for stability
            fp16=False,
            bf16=False,
            remove_unused_columns=False,
        )

        trainer = PCLTrainer(
            alpha=alpha,
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),   # will pad token_labels as well
            optimizers=(optimizer, scheduler),
            callbacks=[CheckNaNGradCallback()],
            compute_metrics=lambda p: {
                "f1": f1_score(p.label_ids, (p.predictions.squeeze(-1) > 0).astype(int))
                "num_pos_pred": int(np.sum(p.predictions.squeeze(-1) > 0))
            },
        )

        trainer.train()

        # after training, get predictions on validation set and find optimal threshold
        # note: predictions[0] are the sequence logits (the token logits are also returned but we ignore them here)
        predictions = trainer.predict(tokenized_datasets["validation"])
        probs = torch.sigmoid(torch.tensor(predictions.predictions[0])).squeeze(-1).numpy()  # shape (n,)
        true_labels = predictions.label_ids

        # loop through thresholds from 0.20 to 0.69 (in steps of 0.01) 
        # keep track of threshold that gives highest F1
        best_t, best_f1 = 0.5, 0
        for t in np.arange(0.2, 0.7, 0.01):
            f1 = f1_score(true_labels, (probs > t).astype(int))
            if f1 > best_f1:
                best_f1, best_t = f1, t

        # generate test set predictions using the best threshold from validation
        test_predictions = trainer.predict(test_tokenized)
        test_probs = torch.sigmoid(torch.tensor(test_predictions.predictions[0])).numpy()
        test_preds = (test_probs > best_t).astype(int)

        # also record validation predictions at that threshold
        dev_preds = (probs > best_t).astype(int)

        print(f"--> Best F1 = {best_f1:.4f} at threshold {best_t:.2f}")

        if best_f1 > best_f1_overall:
            best_f1_overall = best_f1
            best_params = (lr, wd, accum, warmup, alpha)
            best_threshold = best_t
            best_dev_preds = dev_preds.copy()
            best_test_preds = test_preds.copy()
            best_true_labels = true_labels.copy()

        # clean up to free memory
        del model, trainer, optimizer, scheduler
        torch.cuda.empty_cache()

    # after all runs, report best overall result
    print("\n" + "="*50)
    print(f"Best F1 overall: {best_f1_overall:.4f}")
    print(f"Best parameters: lr={best_params[0]}, wd={best_params[1]}, "
          f"accum={best_params[2]}, warmup={best_params[3]}, alpha={best_params[4]}")
    print(f"Best threshold: {best_threshold:.2f}")

    # compute final metrics on dev set for best run
    precision = precision_score(best_true_labels, best_dev_preds)
    recall = recall_score(best_true_labels, best_dev_preds)
    cm = confusion_matrix(best_true_labels, best_dev_preds)
    print(f"\nMetrics at best threshold {best_threshold:.2f}:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {best_f1_overall:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    # save the best dev and test predictions
    with open("dev.txt", "w") as f:
        for p in best_dev_preds:
            f.write(f"{p}\n")
    with open("test.txt", "w") as f:
        for p in best_test_preds:
            f.write(f"{p}\n")

    print("\nBest predictions saved to dev.txt and test.txt")

    # Verify line counts
    if len(best_dev_preds) != len(dev_df):
        print(f"ERROR: dev.txt has {len(best_dev_preds)} lines, expected {len(dev_df)}")
    else:
        print("dev.txt OK")
    if len(best_test_preds) != len(test_df):
        print(f"ERROR: test.txt has {len(best_test_preds)} lines, expected {len(test_df)}")
    else:
        print("test.txt OK")