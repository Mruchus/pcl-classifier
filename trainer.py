import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)


def prepare_comprehensive_data(file_path):
    cols = ['id', 'public_id', 'keyword', 'country', 'text', 'label']
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
    train_df, dev_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    df_minority = train_df[train_df.label == 1]
    train_df_balanced = pd.concat([train_df, df_minority, df_minority])
    return train_df_balanced, dev_df


train_df, dev_df = prepare_comprehensive_data("dontpatronizeme_pcl.tsv")

raw_datasets = DatasetDict({
    "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
    "validation": Dataset.from_pandas(dev_df.reset_index(drop=True)),
})


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits, labels):
        logits = logits.float()
        labels = labels.long()

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        log_pt = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        pt = torch.exp(log_pt).clamp(self.eps, 1.0 - self.eps)
        focal_weight = self.alpha * (1.0 - pt) ** self.gamma
        loss = -focal_weight * log_pt
        return loss.mean()


class PCLComprehensiveTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = FocalLoss(alpha=0.8, gamma=2.0)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


MODEL_NAME = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)


def get_optimizer(model):
    params = [
        {
            'params': [p for n, p in model.named_parameters() if "classifier" in n],
            'lr': 1e-4,
        },
        {
            'params': [p for n, p in model.named_parameters() if "deberta" in n],
            'lr': 2e-5,
        },
    ]
    return torch.optim.AdamW(params, weight_decay=0.01)


def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=256)


tokenized_datasets = raw_datasets.map(tokenize, batched=True)

training_args = TrainingArguments(
    output_dir="./pcl_final",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    warmup_steps=200,
    bf16=True,
    logging_steps=50,
    max_grad_norm=1.0,
)

optimizer = get_optimizer(model)

num_training_steps = (
    len(tokenized_datasets["train"])
    // training_args.per_device_train_batch_size
    * training_args.num_train_epochs
)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=training_args.warmup_steps,
    num_training_steps=num_training_steps,
)

trainer = PCLComprehensiveTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    optimizers=(optimizer, scheduler),
    compute_metrics=lambda p: {
        "f1": f1_score(p.label_ids, np.argmax(p.predictions, axis=-1))
    },
)

print("\n--- Training Comprehensive Model ---")
trainer.train()

predictions = trainer.predict(tokenized_datasets["validation"])
probs = (
    torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=-1)[:, 1]
    .numpy()
)
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