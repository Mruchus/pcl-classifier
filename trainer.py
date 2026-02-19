import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
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

def prepare_comprehensive_data(file_path):
    cols = ['id', 'public_id', 'keyword', 'country', 'text', 'label',
            'unbalanced_power_relations', 'authority_voice', 'shallow_solutions',
            'presupposition', 'compassion', 'metaphor', 'the_people_the_merrier']
    df = pd.read_csv(
        file_path, sep='\t', skipinitialspace=True,
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

    # Fill missing category values with 0 and convert to int
    category_cols = [
        'unbalanced_power_relations', 'authority_voice', 'shallow_solutions',
        'presupposition', 'compassion', 'metaphor', 'the_people_the_merrier'
    ]
    for col in category_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    train_df, dev_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    return train_df, dev_df

categories = [
    "unbalanced_power_relations",
    "authority_voice",
    "shallow_solutions",
    "presupposition",
    "compassion",
    "metaphor",
    "the_people_the_merrier"
]

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
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self):
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        labels = train_dataset["labels"]
        class_counts = torch.bincount(torch.tensor(labels))
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

if __name__ == "__main__":
    train_df, dev_df = prepare_comprehensive_data("dontpatronizeme_pcl.tsv")

    raw_datasets = DatasetDict({
        "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
        "validation": Dataset.from_pandas(dev_df.reset_index(drop=True)),
    })

    MODEL_NAME = "microsoft/deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        tokenized = tokenizer(batch["text"], truncation=True, max_length=256)
        tokenized["labels"] = batch["label"]
        for cat in categories:
            tokenized[cat] = batch[cat]
        return tokenized

    tokenized_datasets = raw_datasets.map(tokenize, batched=True)

    # Keep only the columns we need (tokenizer outputs, labels, and categories)
    keep_columns = ['input_ids', 'attention_mask', 'labels'] + categories
    if 'token_type_ids' in tokenized_datasets["train"].column_names:
        keep_columns.append('token_type_ids')
    for split in tokenized_datasets.keys():
        tokenized_datasets[split] = tokenized_datasets[split].select_columns(keep_columns)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model = model.float()

    labels_train = tokenized_datasets["train"]["labels"]
    class_counts = torch.bincount(torch.tensor(labels_train))
    class_weights = 1.0 / class_counts.float()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5,
        weight_decay=0.01,
        eps=1e-6
    )

    batch_size = 16
    num_epochs = 5
    total_steps = len(tokenized_datasets["train"]) // batch_size * num_epochs
    warmup_steps = 500

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    training_args = TrainingArguments(
        output_dir="./pcl_final",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
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

    print("\n--- Training with Weighted Cross-Entropy and Balanced Batches ---")
    trainer.train()

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

    print("\n--- Per-Category Results (Binary Model) ---")
    print(f"{'Category':<25} {'P':>6} {'R':>6} {'F1':>6}")
    for cat in categories:
        y_true = tokenized_datasets["validation"][cat]
        y_pred = final_preds
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f = f1_score(y_true, y_pred, zero_division=0)
        print(f"{cat:<25} {p*100:5.1f} {r*100:5.1f} {f*100:5.1f}")