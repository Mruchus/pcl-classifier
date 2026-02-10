import os
import csv
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

# --- CONFIG ---
os.environ['HF_HOME'] = '/tmp/mnn23_hf_cache'
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

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

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
        enc = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }

def load_data(file_path):
    df = pd.read_csv(file_path, sep="\t", header=None, 
                     names=["id", "para_id", "keyword", "country", "text", "label"],
                     engine="python", quoting=csv.QUOTE_NONE)
    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label_bin"] = (df["label"] >= 2).astype(int)
    return df["text"].astype(str).tolist(), df["label_bin"].tolist()

# --- THE SAMPLER IMPLEMENTATION ---
def get_sampler(labels):
    """Creates a sampler that ensures the model sees PCL examples as often as non-PCL."""
    class_counts = np.bincount(labels)
    # Weight is 1 / frequency
    class_weights = 1.0 / class_counts
    # Assign the weight of its class to each individual sample
    weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler

def train_one_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        targets = batch["labels"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.float() # Stability fix
        
        # We use standard CrossEntropy because the Sampler balances the batch for us
        loss = F.cross_entropy(logits, targets)

        if not torch.isfinite(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_true, all_pred = [], []
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
    print(f"Device: {DEVICE}")
    texts, labels = load_data("dontpatronizeme_pcl.tsv")
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=SEED, stratify=labels)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)

    # Prepare Sampler for the training set
    train_sampler = get_sampler(train_labels)

    train_ds = PCLDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_ds = PCLDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

    # Use the sampler here - shuffle MUST be False when using a sampler
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    optimizer = AdamW(model.parameters(), lr=LR, eps=1e-6)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP_RATIO * total_steps), total_steps)

    best_f1 = -1.0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler)
        val_f1, report = evaluate(model, val_loader)

        print(f"Loss: {train_loss:.4f} | Val F1: {val_f1:.4f}")
        print(report)

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            model.save_pretrained(SAVE_DIR)
            print(f"🚀 New best model saved (F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping.")
                break

if __name__ == "__main__":
    main()