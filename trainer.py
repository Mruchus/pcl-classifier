import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.metrics import f1_score
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

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # Compute class weights from the imbalanced training set
    labels_train = tokenized_datasets["train"]["labels"]
    class_counts = torch.bincount(torch.tensor(labels_train))
    class_weights = 1.0 / class_counts.float()
    print(f"Class counts: {class_counts}, weights: {class_weights}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-6,               # very low LR
        weight_decay=0.1,       # strong regularization
        eps=1e-6
    )

    batch_size = 16
    num_epochs = 1
    total_steps = len(tokenized_datasets["train"]) // batch_size * num_epochs
    warmup_steps = 1000

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    training_args = TrainingArguments(
        output_dir="./pcl_final",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,      # effective batch size 32
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

    print("\n--- Training: Imbalanced data + Class Weights (LR=1e-5, wd=0.1, grad accum=2) ---")
    trainer.train()

    # Dev set predictions + threshold tuning
    predictions = trainer.predict(tokenized_datasets["validation"])
    probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=-1)[:, 1].numpy()
    true_labels = predictions.label_ids

    best_t, best_f1 = 0.5, 0
    for t in np.arange(0.2, 0.7, 0.01):
        f1 = f1_score(true_labels, (probs > t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t

    print(f"Optimal Threshold: {best_t:.2f} | Max F1: {best_f1:.4f}")
    final_preds = (probs > best_t).astype(int)

    with open("dev.txt", "w") as f:
        for p in final_preds:
            f.write(f"{p}\n")

    # Test set prediction
    test_df = prepare_test_data("Task 4 Test.tsv")
    test_dataset = Dataset.from_pandas(test_df[['text']])
    def tokenize_test(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256)
    test_tokenized = test_dataset.map(tokenize_test, batched=True, remove_columns=['text'])

    test_predictions = trainer.predict(test_tokenized)
    test_probs = torch.nn.functional.softmax(torch.tensor(test_predictions.predictions), dim=-1)[:, 1].numpy()
    test_preds = (test_probs > best_t).astype(int)

    with open("test.txt", "w") as f:
        for p in test_preds:
            f.write(f"{p}\n")