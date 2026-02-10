import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
import csv

# Config
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 192
BATCH_SIZE = 16
LR = 2e-5
EPOCHS = 6
PATIENCE = 3
WARMUP_RATIO = 0.1
SAVE_DIR = "best_deberta_pcl"

# Reproducibility
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# Dataset
class PCLDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),         # [L]
            "attention_mask": enc["attention_mask"].squeeze(0),# [L]
            "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


# Data loading
def load_data(file_path):
    df = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=["id", "para_id", "keyword", "country", "text", "label"],
        engine="python",
        quoting=csv.QUOTE_NONE,
    )

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    # binary mapping: 0/1 -> 0, 2/3/4 -> 1
    df["label_bin"] = (df["label"] >= 2).astype(int)

    texts = df["text"].tolist()
    labels = df["label_bin"].tolist()
    return texts, labels

def make_splits(texts, labels, test_size=0.2):
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=SEED,
        stratify=labels,
    )
    return train_texts, val_texts, train_labels, val_labels

def compute_class_weights(train_labels):
    # weights[c] = max_count / count[c]
    counts = np.bincount(np.asarray(train_labels, dtype=np.int64), minlength=2)
    weights = counts.max() / np.maximum(counts, 1)
    return torch.tensor(weights, dtype=torch.float32)

# Train / eval
def train_one_epoch(model, loader, optimizer, scheduler, class_weights):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        targets = batch["labels"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.float()  # force fp32 loss

        loss = F.cross_entropy(logits, targets, weight=class_weights)

        if not torch.isfinite(loss):
            print("Non-finite loss detected!")
            print("targets unique:", torch.unique(targets))
            print("logits min/max:", torch.nanmin(outputs.logits).item(), torch.nanmax(outputs.logits).item())
            raise RuntimeError("Stopping due to NaN/Inf loss")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_true = []
    all_pred = []

    for batch in tqdm(loader, desc="Validating", leave=False):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        targets = batch["labels"].to(DEVICE)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        preds = torch.argmax(logits, dim=1)

        all_true.extend(targets.cpu().numpy().tolist())
        all_pred.extend(preds.cpu().numpy().tolist())

    f1 = f1_score(all_true, all_pred, pos_label=1)
    report = classification_report(all_true, all_pred, target_names=["Non-PCL", "PCL"], zero_division=0)
    return f1, report

def main():
    print("Device:", DEVICE)

    texts, labels = load_data("dontpatronizeme_pcl.tsv")
    train_texts, val_texts, train_labels, val_labels = make_splits(texts, labels)

    # show class balance
    tr_counts = np.bincount(np.asarray(train_labels, dtype=np.int64), minlength=2)
    va_counts = np.bincount(np.asarray(val_labels, dtype=np.int64), minlength=2)
    print("Train counts:", tr_counts, " (pos% =", tr_counts[1] / tr_counts.sum() * 100, ")")
    print("Val counts:  ", va_counts, " (pos% =", va_counts[1] / va_counts.sum() * 100, ")")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)

    train_ds = PCLDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_ds = PCLDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    class_weights = compute_class_weights(train_labels).to(DEVICE).float()
    print("Class weights:", class_weights.detach().cpu().numpy())

    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_f1 = -1.0
    patience = 0

    for epoch in range(1, EPOCHS + 1):
        print("\n" + "=" * 60)
        print(f"Epoch {epoch}/{EPOCHS}")
        print("=" * 60)

        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, class_weights)
        val_f1, val_report = evaluate(model, val_loader)

        print(f"Train loss: {train_loss:.4f}")
        print(f"Val F1 (PCL=1): {val_f1:.4f}")
        print(val_report)

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience = 0
            os.makedirs(SAVE_DIR, exist_ok=True)
            model.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)
            print(f"Saved new best model to {SAVE_DIR} (F1={best_f1:.4f})")
        else:
            patience += 1
            print(f"No improvement. Patience {patience}/{PATIENCE}")
            if patience >= PATIENCE:
                print("Early stopping.")
                break

    print("\nDone. Best val F1:", best_f1)

if __name__ == "__main__":
    main()
