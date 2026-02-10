import os
import csv
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

# --- 1. PREVENT DISK QUOTA ERRORS ---
# Redirect Hugging Face cache to /tmp so it doesn't fill your home folder
os.environ['HF_HOME'] = '/tmp/mnn23_hf_cache'

# --- 2. CONFIGURATION ---
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256  # Increased to capture full context
BATCH_SIZE = 16
LR = 1e-5         # Lowered slightly for stability
EPOCHS = 6
PATIENCE = 3
WARMUP_RATIO = 0.1
SAVE_DIR = "best_deberta_pcl"

# --- 3. REPRODUCIBILITY ---
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # These two lines ensure deterministic behavior on GPU (slightly slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# --- 4. DATASET CLASS ---
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
        # Using [0] is safer than squeeze(0) for consistency
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }

# --- 5. DATA LOADING ---
def load_data(file_path):
    # quoting=3 (csv.QUOTE_NONE) is crucial for the PCL dataset format
    df = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=["id", "para_id", "keyword", "country", "text", "label"],
        engine="python",
        quoting=csv.QUOTE_NONE,
    )

    # Clean empty rows
    df = df.dropna(subset=["text", "label"]).copy()
    
    # Ensure text is string and not just whitespace
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip().str.len() > 1].copy()

    # Clean labels
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    # Binary Mapping: 0,1 -> 0 (Negative); 2,3,4 -> 1 (Positive)
    df["label_bin"] = (df["label"] >= 2).astype(int)

    texts = df["text"].tolist()
    labels = df["label_bin"].tolist()
    return texts, labels

def make_splits(texts, labels, test_size=0.2):
    return train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=SEED,
        stratify=labels,
    )

def compute_class_weights(train_labels):
    # Compute inverse frequency weights
    counts = np.bincount(np.asarray(train_labels, dtype=np.int64), minlength=2)
    # weights[c] = max_count / count[c]
    weights = counts.max() / np.maximum(counts, 1)
    return torch.tensor(weights, dtype=torch.float32)

# --- 6. TRAINING & EVALUATION ---
def train_one_epoch(model, loader, optimizer, scheduler, class_weights):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        targets = batch["labels"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # FIX: Ensure logits are float32 for stability
        logits = outputs.logits.float()

        loss = F.cross_entropy(logits, targets, weight=class_weights)

        # FIX: Robust NaN check that doesn't crash on old PyTorch versions
        if torch.isnan(loss) or torch.isinf(loss):
            print("!!! Non-finite loss detected !!!")
            print("Targets unique:", torch.unique(targets))
            # Use .min() and .max() instead of nanmin/nanmax
            print("Logits range:", logits.min().item(), "to", logits.max().item())
            raise RuntimeError("Stopping due to NaN/Inf loss")

        loss.backward()
        
        # Gradient Clipping helps prevent DeBERTa explosions
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

# --- 7. MAIN LOOP ---
def main():
    print(f"Using Device: {DEVICE}")

    # Load Data
    print("Loading data...")
    texts, labels = load_data("dontpatronizeme_pcl.tsv")
    train_texts, val_texts, train_labels, val_labels = make_splits(texts, labels)

    # Class Balance check
    tr_counts = np.bincount(np.asarray(train_labels, dtype=np.int64), minlength=2)
    print(f"Train Class Dist: {tr_counts} (Pos ratio: {tr_counts[1] / tr_counts.sum():.2%})")

    # Initialization
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)

    train_ds = PCLDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_ds = PCLDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

    # num_workers=0 is safer on shared cluster nodes
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Weights
    class_weights = compute_class_weights(train_labels).to(DEVICE).float()
    print(f"Using Class Weights: {class_weights.cpu().numpy()}")

    # OPTIMIZER FIX: eps=1e-6 is critical for DeBERTa stability
    optimizer = AdamW(model.parameters(), lr=LR, eps=1e-6)
    
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print("Starting Training...")
    best_f1 = -1.0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*20} Epoch {epoch}/{EPOCHS} {'='*20}")

        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, class_weights)
        val_f1, val_report = evaluate(model, val_loader)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val F1 (PCL): {val_f1:.4f}")
        print(val_report)

        # Checkpointing
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            os.makedirs(SAVE_DIR, exist_ok=True)
            model.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)
            print(f"🚀 New Best Model Saved! (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"\nTraining Complete. Best F1: {best_f1:.4f}")

if __name__ == "__main__":
    main()