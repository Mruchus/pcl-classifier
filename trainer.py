import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
from imblearn.over_sampling import RandomOverSampler

# configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 10  # Increased from 4
PATIENCE = 3  # Stop if no improvement for 3 epochs

# data preparation
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
        encoding = self.tokenizer(
            text, 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_len, 
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def load_and_split_data(file_path, test_size=0.2, seed=42):
    # load tsv (quoting=3 ignores quotes in text)
    df = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=["id", "para_id", "keyword", "country", "text", "label"],
        engine="python",
        quoting=3
    )

    # clean
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    # binary mapping: 0/1 -> 0, 2/3/4 -> 1
    df["label_bin"] = (df["label"] >= 2).astype(int)

    texts = df["text"].tolist()
    labels = df["label_bin"].tolist()

    # split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels
    )

    print(f"Before oversampling - Class distribution:")
    unique, counts = np.unique(train_labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Class {u}: {c} samples ({c/len(train_labels)*100:.1f}%)")

    # oversample minority class to 50% ratio
    ros = RandomOverSampler(random_state=seed, sampling_strategy=0.5)
    train_texts_arr = np.array(train_texts).reshape(-1, 1)
    train_labels_arr = np.array(train_labels)
    train_texts_resampled, train_labels_resampled = ros.fit_resample(train_texts_arr, train_labels_arr)
    train_texts = train_texts_resampled.flatten().tolist()
    train_labels = train_labels_resampled.tolist()

    print(f"\nAfter oversampling - Class distribution:")
    unique, counts = np.unique(train_labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Class {u}: {c} samples ({c/len(train_labels)*100:.1f}%)")

    # calculate class weights (gentler since we oversampled)
    counts = np.bincount(np.array(train_labels, dtype=np.int64), minlength=2)
    weights = counts.max() / np.maximum(counts, 1)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    
    print(f"\nClass weights: {class_weights}")

    return train_texts, val_texts, train_labels, val_labels, class_weights


# main training execution
def main():
    print(f"Using device: {DEVICE}")
    
    # load data
    train_texts, val_texts, train_labels, val_labels, class_weights = load_and_split_data("dontpatronizeme_pcl.tsv")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)
    
    train_ds = PCLDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_ds = PCLDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    
    # setup optimiser & scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    # use crossentropy instead of focal loss
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # training loop with early stopping
    best_f1 = 0
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"{'='*60}")
        
        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            targets = batch['labels'].to(DEVICE)
            
            outputs = model(ids, attention_mask=mask)
            loss = criterion(outputs.logits, targets)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Average training loss: {avg_loss:.4f}")

        # validation
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                ids = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                targets = batch['labels'].to(DEVICE)
                
                outputs = model(ids, attention_mask=mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(targets.cpu().numpy())
        
        f1 = f1_score(all_true, all_preds, pos_label=1)
        print(f"\nValidation F1 (PCL Class): {f1:.4f}")
        print(classification_report(all_true, all_preds, target_names=['Non-PCL', 'PCL']))
        
        # check for model collapse
        if f1 == 0:
            print("⚠️  WARNING: Model collapsed (F1=0)! Stopping training.")
            break
        
        # save best model and early stopping
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            model.save_pretrained("BestModel_DeBERTa_Focal")
            tokenizer.save_pretrained("BestModel_DeBERTa_Focal")
            print(f"✓ New best F1: {f1:.4f} - Model saved!")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best F1 score: {best_f1:.4f}")
                break

    print(f"\n{'='*60}")
    print(f"Training complete! Best F1: {best_f1:.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()