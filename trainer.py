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

# configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5 # low LR to prevent forgetting in transformer layers
EPOCHS = 4
GAMMA = 2.0 # focal loss hyperparameter
ALPHA = 1.0 # focal loss scaling

# focal loss implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        # targets are expected to be long/indices
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss) 
        # down-weight easy examples, scale up hard ones
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()

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

def load_and_split_data(file_path):
    # standard tsv loading for the pcl dataset
    df = pd.read_csv(file_path, sep='\t', header=None, quoting=3)
    df.columns = ["row_id", "para_id", "keyword", "country", "text", "label"]
    
    # drop rows without text or labels
    df = df.dropna(subset=['text', 'label'])
    
    # binary mapping: 0,1 -> 0; 2,3,4 -> 1
    df['label_bin'] = df['label'].apply(lambda x: 1 if int(x) >= 2 else 0)
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].values, 
        df['label_bin'].values, 
        test_size=0.2, 
        stratify=df['label_bin'].values,
        random_state=42
    )
    
    # calculate weights for imbalanced classes (from stage 2 stats)
    # approx 9:1 ratio -> weight pcl class higher
    weights = torch.tensor([1.0, 8.5]).to(DEVICE)
    
    return train_texts, val_texts, train_labels, val_labels, weights

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
    
    criterion = FocalLoss(alpha=ALPHA, gamma=GAMMA, weight=class_weights)

    # training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        for batch in tqdm(train_loader):
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

        # local evaluation
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                targets = batch['labels'].to(DEVICE)
                
                outputs = model(ids, attention_mask=mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(targets.cpu().numpy())
        
        f1 = f1_score(all_true, all_preds, pos_label=1)
        print(f"Validation F1 (PCL Class): {f1:.4f}")
        print(classification_report(all_true, all_preds, target_names=['Non-PCL', 'PCL']))

    # save for stage 4/5
    model.save_pretrained("BestModel_DeBERTa_Focal")
    tokenizer.save_pretrained("BestModel_DeBERTa_Focal")

if __name__ == "__main__":
    main()