import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    TrainerCallback,
    get_linear_schedule_with_warmup
)
import itertools
import copy

def prepare_comprehensive_data(pcl_file, train_labels_file, dev_labels_file):
    cols = ['id', 'public_id', 'keyword', 'country', 'text', 'label']
    df = pd.read_csv(
        pcl_file, sep='\t', skipinitialspace=True,
        names=cols, index_col='id', quoting=3
    )
    df = df.dropna(subset=['text', 'label'])
    df['label'] = df['label'].apply(lambda x: 1 if x >= 2 else 0)
    df['text'] = (
        df['keyword'].fillna('') + " [SEP] " +
        df['text'].str.replace(r'@@\d+', '', regex=True).str.strip()
    )
    df = df[df['text'].str.strip() != '']
    df = df[df['text'].str.len() > 10]

    train_ids = pd.read_csv(train_labels_file)['par_id'].tolist()
    dev_ids = pd.read_csv(dev_labels_file)['par_id'].tolist()

    train_df = df[df.index.isin(train_ids)]
    dev_df = df[df.index.isin(dev_ids)]
    return train_df, dev_df

def prepare_test_data(test_file):
    test_df = pd.read_csv(test_file, sep='\t', header=None,
                          names=['t_id', 'user_id', 'keyword', 'country', 'text'])
    test_df['text'] = (
        test_df['keyword'].fillna('') + " [SEP] " +
        test_df['text'].str.replace(r'@@\d+', '', regex=True).str.strip()
    )
    test_df = test_df[test_df['text'].str.strip() != '']
    test_df = test_df[test_df['text'].str.len() > 10]
    return test_df

class CheckNaNGradCallback(TrainerCallback):
    def on_step_end(self, args, state, control, model=None, **kwargs):
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"NaN gradient detected in {name} at step {state.global_step}")
                control.should_training_stop = True
                return control
        return control

class PCLComprehensiveTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device, dtype=logits.dtype))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def train_and_evaluate(params, train_dataset, eval_dataset, test_tokenized, class_weights, device):
    # create a fresh model for each run
    model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base", num_labels=2).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params['lr'],
        weight_decay=params['weight_decay'],
        eps=1e-6
    )

    # calculate steps
    batch_size = 16
    num_epochs = 10
    total_steps = len(train_dataset) // batch_size * num_epochs
    warmup_steps = params['warmup_steps']

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    training_args = TrainingArguments(
        output_dir=f"./pcl_final_lr{params['lr']}_wd{params['weight_decay']}_acc{params['grad_accum']}_warm{params['warmup_steps']}",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=params['grad_accum'],
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

    trainer = PCLComprehensiveTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        optimizers=(optimizer, scheduler),
        callbacks=[CheckNaNGradCallback()],
        compute_metrics=lambda p: {
            "f1": f1_score(p.label_ids, np.argmax(p.predictions, axis=-1)),
            "num_pos_pred": np.sum(np.argmax(p.predictions, axis=-1))
        },
    )

    trainer.train()

    # validation predictions + threshold tuning
    predictions = trainer.predict(eval_dataset)
    probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=-1)[:, 1].numpy()
    true_labels = predictions.label_ids

    best_t, best_f1 = 0.5, 0
    for t in np.arange(0.2, 0.7, 0.01):
        f1 = f1_score(true_labels, (probs > t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t

    # test set predictions with best threshold
    test_predictions = trainer.predict(test_tokenized)
    test_probs = torch.nn.functional.softmax(torch.tensor(test_predictions.predictions), dim=-1)[:, 1].numpy()
    test_preds = (test_probs > best_t).astype(int)

    # dev set predictions at best threshold (for later saving)
    dev_preds = (probs > best_t).astype(int)

    # clean up to free memory
    del model
    torch.cuda.empty_cache()

    return best_f1, best_t, dev_preds, test_preds, predictions.label_ids

if __name__ == "__main__":
    train_df, dev_df = prepare_comprehensive_data(
        "dontpatronizeme_pcl.tsv",
        "SemEval 2022 Train Labels.csv",
        "Semeval 2022 Dev Labels.csv"
    )

    raw_datasets = DatasetDict({
        "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
        "validation": Dataset.from_pandas(dev_df.reset_index(drop=True)),
    })

    MODEL_NAME = "microsoft/deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        tokenized = tokenizer(batch["text"], truncation=True, max_length=256)
        tokenized["labels"] = batch["label"]
        return tokenized

    tokenized_datasets = raw_datasets.map(tokenize, batched=True)

    keep_columns = ['input_ids', 'attention_mask', 'labels']
    if 'token_type_ids' in tokenized_datasets["train"].column_names:
        keep_columns.append('token_type_ids')
    for split in tokenized_datasets.keys():
        tokenized_datasets[split] = tokenized_datasets[split].select_columns(keep_columns)

    # precompute class weights
    labels_train = tokenized_datasets["train"]["labels"]
    class_counts = torch.bincount(torch.tensor(labels_train))
    class_weights = 1.0 / class_counts.float()
    print(f"Class counts: {class_counts}, weights: {class_weights}")

    # prepare test tokenised dataset
    test_df = prepare_test_data("Task 4 Test.tsv")
    test_dataset = Dataset.from_pandas(test_df[['text']])
    def tokenize_test(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256)
    test_tokenized = test_dataset.map(tokenize_test, batched=True, remove_columns=['text'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # define hyperparameter grid
    # lr_list = [8e-7, 9e-7, 1e-6, 1.2e-6]
    # wd_list = [0.01, 0.05, 0.1, 0.2]
    # accum_list = [1, 2, 4] # gradient accumulation steps -> effective batch sizes 16,32,64
    # warmup_list = [500, 1000, 1500, 2000]

    # # generate all combinations
    # param_combinations = list(itertools.product(lr_list, wd_list, accum_list, warmup_list))
    # print(f"Total combinations: {len(param_combinations)}")
     
    # param_combinations = [(1e-6, 0.1, 2, 1500)]

    # best_f1_overall = 0.0
    # best_params = None
    # best_dev_preds = None
    # best_test_preds = None
    # best_true_labels = None
    # best_threshold = None

    # for lr, wd, accum, warmup in param_combinations:
    #     params = {
    #         'lr': lr,
    #         'weight_decay': wd,
    #         'grad_accum': accum,
    #         'warmup_steps': warmup
    #     }
    #     print(f"\n--- Testing params: {params} ---")
    #     try:
    #         f1, thresh, dev_preds, test_preds, true_labels = train_and_evaluate(
    #             params,
    #             tokenized_datasets["train"],
    #             tokenized_datasets["validation"],
    #             test_tokenized,
    #             class_weights,
    #             device
    #         )
    #         print(f"--> Best F1 = {f1:.4f} at threshold {thresh:.2f}")
    #         if f1 > best_f1_overall:
    #             best_f1_overall = f1
    #             best_params = params.copy()
    #             best_threshold = thresh
    #             best_dev_preds = dev_preds.copy()
    #             best_test_preds = test_preds.copy()
    #             best_true_labels = true_labels.copy()
    #     except Exception as e:
    #         print(f"Run failed: {e}")
    #         continue

    # print("\n" + "="*50)
    # print(f"Best F1 overall: {best_f1_overall:.4f}")
    # print(f"Best parameters: {best_params}")
    # print(f"Best threshold: {best_threshold:.2f}")

    # # compute final metrics on dev set for best run
    # precision = precision_score(best_true_labels, best_dev_preds)
    # recall = recall_score(best_true_labels, best_dev_preds)
    # cm = confusion_matrix(best_true_labels, best_dev_preds)
    # print(f"\nMetrics at best threshold {best_threshold:.2f}:")
    # print(f"Precision: {precision:.4f}")
    # print(f"Recall:    {recall:.4f}")
    # print(f"F1:        {best_f1_overall:.4f}")
    # print("\nConfusion Matrix:")
    # print(cm)

    # # save the best dev and test predictions
    # with open("dev.txt", "w") as f:
    #     for p in best_dev_preds:
    #         f.write(f"{p}\n")
    # with open("test.txt", "w") as f:
    #     for p in best_test_preds:
    #         f.write(f"{p}\n")

    # print("\nBest predictions saved to dev.txt and test.txt")

    lr = 1e-6
    wd = 0.1
    accum = 2
    warmup = 1500

    seeds = [1221]
    print(f"Running {len(seeds)} trials with different seeds")

    best_f1_overall = 0.0
    best_params = None
    best_dev_preds = None
    best_test_preds = None
    best_true_labels = None
    best_threshold = None

    for trial_idx, seed in enumerate(seeds):
        print(f"\n{'='*50}")
        print(f"Trial {trial_idx+1}/{len(seeds)} | seed = {seed}")
        print(f"{'='*50}")

        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base", num_labels=2).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
            eps=1e-6
        )

        batch_size = 16
        num_epochs = 10
        total_steps = len(tokenized_datasets["train"]) // batch_size * num_epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup,
            num_training_steps=total_steps
        )

        # unique output dir for this trial
        output_dir = f"./pcl_final_lr{lr}_wd{wd}_acc{accum}_warm{warmup}_seed{seed}"

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=accum,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            warmup_steps=warmup,
            logging_steps=50,
            max_grad_norm=1.0,
            fp16=False,
            bf16=False,
            remove_unused_columns=False,
        )

        trainer = PCLComprehensiveTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            optimizers=(optimizer, scheduler),
            callbacks=[CheckNaNGradCallback()],
            compute_metrics=lambda p: {
                "f1": f1_score(p.label_ids, np.argmax(p.predictions, axis=-1)),
                "num_pos_pred": np.sum(np.argmax(p.predictions, axis=-1))
            },
        )

        trainer.train()

        # validation predictions + threshold tuning
        predictions = trainer.predict(tokenized_datasets["validation"])
        probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=-1)[:, 1].numpy()
        true_labels = predictions.label_ids

        best_t, best_f1 = 0.5, 0
        for t in np.arange(0.2, 0.7, 0.01):
            f1 = f1_score(true_labels, (probs > t).astype(int))
            if f1 > best_f1:
                best_f1, best_t = f1, t

        # test set predictions with best threshold
        test_predictions = trainer.predict(test_tokenized)
        test_probs = torch.nn.functional.softmax(torch.tensor(test_predictions.predictions), dim=-1)[:, 1].numpy()
        test_preds = (test_probs > best_t).astype(int)

        # dev set predictions at best threshold
        dev_preds = (probs > best_t).astype(int)

        print(f"--> Trial {trial_idx+1} best F1 = {best_f1:.4f} at threshold {best_t:.2f}")

        # update best overall
        if best_f1 > best_f1_overall:
            best_f1_overall = best_f1
            best_params = {'lr': lr, 'weight_decay': wd, 'grad_accum': accum, 'warmup_steps': warmup}
            best_threshold = best_t
            best_dev_preds = dev_preds.copy()
            best_test_preds = test_preds.copy()
            best_true_labels = true_labels.copy()

        # clean up to free memory
        del model
        torch.cuda.empty_cache()

    print("\n" + "="*50)
    print(f"Best F1 overall: {best_f1_overall:.4f}")
    print(f"Best parameters: {best_params}")
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