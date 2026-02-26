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

from data_utils import prepare_data, prepare_test_data, load_span_data, tokenize_with_spans
from model import PCLClassifier


def set_seed(seed=127):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class PCLTrainer(Trainer):
    def __init__(self, alpha=0.5, pos_weight=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.pos_weight = pos_weight
        self.train_labels = train_labels


    def _get_train_sampler(self):
        if self.train_labels is None:
            return super()._get_train_sampler()
        class_sample_counts = np.bincount(self.train_labels)
        sample_weights = 1.0 / class_sample_counts[self.train_labels]
        return torch.utils.data.WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights),
            num_samples=len(sample_weights),
            replacement=True
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        token_labels = inputs.pop("token_labels")

        seq_logits, token_logits = model(**inputs)

        loss_fct_seq = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        seq_loss = loss_fct_seq(seq_logits.squeeze(-1), labels.float())

        loss_fct_token = nn.BCEWithLogitsLoss(reduction='none')
        token_loss = loss_fct_token(token_logits, token_labels.float())
        token_loss = (token_loss * inputs['attention_mask']).sum() / inputs['attention_mask'].sum()

        loss = self.alpha * seq_loss + (1 - self.alpha) * token_loss
        return (loss, seq_logits) if return_outputs else loss

    def prediction_step(
    self,
    model,
    inputs,
    prediction_loss_only,
    ignore_keys=None
    ):
        labels = inputs.get("labels")

        if "token_labels" in inputs:
            inputs = inputs.copy()
            inputs.pop("token_labels")

        with torch.no_grad():
            seq_logits, _ = model(**inputs)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(seq_logits.squeeze(-1), labels.float())

        return (loss, seq_logits, labels)

class CheckNaNGradCallback(TrainerCallback):
    def on_step_end(self, args, state, control, model=None, **kwargs):
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"NaN gradient in {name} at step {state.global_step}")
                control.should_training_stop = True
                return control
        return control 

def compute_metrics(p):
    preds = (p.predictions.squeeze(-1) > 0).astype(int)  # p.predictions is now cleanly (n, 1)
    return {
        "f1": f1_score(p.label_ids, preds, zero_division=0),
        "num_pos_pred": int(np.sum(preds))
    }


if __name__ == "__main__":
    set_seed(127)

    train_df, dev_df = prepare_data(
        "dontpatronizeme_pcl.tsv",
        "SemEval 2022 Train Labels.csv",
        "Semeval 2022 Dev Labels.csv"
    )

    print("Train columns:", train_df.columns.tolist())
    print("Dev columns:",   dev_df.columns.tolist())
    print(f"Train size: {len(train_df)}, Dev size: {len(dev_df)}")

    # Compute pos_weight on original label distribution
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1]),
        y=train_df['label'].values
    )
    pos_weight = torch.tensor(class_weights[1], dtype=torch.float)
    print(f"pos_weight: {pos_weight:.4f}")

    raw_datasets = DatasetDict({
        "train":      Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(dev_df),
    })

    MODEL_NAME = "microsoft/deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    max_length = 256
    tokenized_datasets = raw_datasets.map(
        lambda batch: tokenize_with_spans(batch, tokenizer, max_length),
        batched=True,
    )

    keep_columns = ['input_ids', 'attention_mask', 'labels', 'token_labels']
    tokenized_datasets = tokenized_datasets.select_columns(keep_columns)

    labels_train  = tokenized_datasets["train"]["labels"]
    class_counts  = torch.bincount(torch.tensor(labels_train))
    class_weights_seq = 1.0 / class_counts.float()
    print(f"Class counts: {class_counts}, weights: {class_weights_seq}")

    test_df = prepare_test_data("Task 4 Test.tsv")
    test_dataset = Dataset.from_pandas(test_df[['text']])
    def tokenize_test(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)
    test_tokenized = test_dataset.map(tokenize_test, batched=True, remove_columns=['text'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr_list     = [8e-7, 9e-7, 1e-6, 1.2e-6]
    wd_list     = [0.01, 0.05, 0.1, 0.2]
    accum_list  = [1, 2, 4]
    warmup_list = [500, 1000, 1500, 2000]
    alpha_list  = [0.3, 0.5, 0.7]

    param_combinations = list(itertools.product(lr_list, wd_list, accum_list, warmup_list, alpha_list))
    print(f"Total combinations: {len(param_combinations)}")

    param_combinations = [(9e-7, 0.1, 2, 1500, 0.5)]

    best_f1_overall  = 0.0
    best_params      = None
    best_dev_preds   = None
    best_test_preds  = None
    best_true_labels = None
    best_threshold   = None

    for lr, wd, accum, warmup, alpha in param_combinations:
        print(f"\n--- lr={lr}, wd={wd}, accum={accum}, warmup={warmup}, alpha={alpha} ---")

        set_seed(127)

        model = PCLClassifier(MODEL_NAME)
        model.to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=wd, eps=1e-6
        )

        batch_size = 16
        num_epochs = 10

        total_steps   = (len(tokenized_datasets["train"]) // batch_size // accum) * num_epochs
        warmup_steps  = warmup

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

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
            max_grad_norm=1.0,
            fp16=False,
            bf16=False,
            remove_unused_columns=False,
        )

        from sklearn.utils.class_weight import compute_class_weight

        labels_train = tokenized_datasets["train"]["labels"]
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.array([0, 1]),
            y=np.array(labels_train)
        )

        pos_weight = torch.tensor(class_weights[1], dtype=torch.float).to(device)

        trainer = PCLTrainer(
            alpha=alpha,
            pos_weight=pos_weight,
            train_labels=np.array(train_df['label'].values),
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            optimizers=(optimizer, scheduler),
            callbacks=[CheckNaNGradCallback()],
            compute_metrics=compute_metrics,
        )

        trainer.train()

        # find optimal threshold on validation set
        predictions = trainer.predict(tokenized_datasets["validation"])
        probs = torch.sigmoid(torch.tensor(predictions.predictions)).squeeze(-1).numpy()
        print("logits shape:", predictions.predictions.shape)
        print("probs shape:", probs.shape)
        true_labels = predictions.label_ids

        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0, 1, 0.01):
            f1 = f1_score(true_labels, (probs > t).astype(int))
            if f1 > best_f1:
                best_f1, best_t = f1, t

        # test predictions using best validation threshold
        test_predictions = trainer.predict(test_tokenized)
        test_probs = torch.sigmoid(torch.tensor(test_predictions.predictions)).squeeze(-1).numpy()
        test_preds = (test_probs > best_t).astype(int)
        dev_preds  = (probs > best_t).astype(int)

        print(f"--> Best F1 = {best_f1:.4f} at threshold {best_t:.2f}")

        if best_f1 > best_f1_overall:
            best_f1_overall  = best_f1
            best_params      = (lr, wd, accum, warmup, alpha)
            best_threshold   = best_t
            best_dev_preds   = dev_preds.copy()
            best_test_preds  = test_preds.copy()
            best_true_labels = true_labels.copy()

        del model, trainer, optimizer, scheduler
        torch.cuda.empty_cache()

    print("\n" + "=" * 50)
    print(f"Best F1 overall: {best_f1_overall:.4f}")
    print(f"Best parameters: lr={best_params[0]}, wd={best_params[1]}, "
          f"accum={best_params[2]}, warmup={best_params[3]}, alpha={best_params[4]}")
    print(f"Best threshold: {best_threshold:.2f}")

    precision = precision_score(best_true_labels, best_dev_preds)
    recall    = recall_score(best_true_labels, best_dev_preds)
    cm        = confusion_matrix(best_true_labels, best_dev_preds)
    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {best_f1_overall:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    with open("dev.txt", "w") as f:
        for p in best_dev_preds:
            f.write(f"{p}\n")
    with open("test.txt", "w") as f:
        for p in best_test_preds:
            f.write(f"{p}\n")

    print("\nBest predictions saved to dev.txt and test.txt")

    if len(best_dev_preds) != len(dev_df):
        print(f"ERROR: dev.txt has {len(best_dev_preds)} lines, expected {len(dev_df)}")
    else:
        print("dev.txt OK")
    if len(best_test_preds) != len(test_df):
        print(f"ERROR: test.txt has {len(best_test_preds)} lines, expected {len(test_df)}")
    else:
        print("test.txt OK")