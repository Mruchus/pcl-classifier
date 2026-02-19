import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import resample
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)

def prepare_data(file_path):
    print("--- Loading and Cleaning Data ---")
    cols = ['id', 'public_id', 'keyword', 'country', 'text', 'label']
    df = pd.read_csv(file_path, sep='\t', skipinitialspace=True, 
                     names=cols, index_col='id', quoting=3)
    
    df = df.dropna(subset=['text', 'label'])
    df['label'] = df['label'].apply(lambda x: 1 if x >= 2 else 0)
    df['text'] = df['text'].str.replace(r'@@\d+', '', regex=True).str.strip()
    
    train_df, dev_df = train_test_split(
        df[['text', 'label']], 
        test_size=0.2, 
        random_state=42, 
        stratify=df['label']
    )

    # separate majority and minority classes
    df_majority = train_df[train_df.label == 0]
    df_minority = train_df[train_df.label == 1]
    
    # upsample minority class to match majority class size
    df_minority_upsampled = resample(df_minority, 
                                     replace=True,
                                     n_samples=len(df_majority),
                                     random_state=42)
    
    train_df_balanced = pd.concat([df_majority, df_minority_upsampled])
    
    print(f"Original train size: {len(train_df)}")
    print(f"Balanced train size: {len(train_df_balanced)}")
    
    return train_df_balanced, dev_df

# load the raw data
train_df, dev_df = prepare_data("dontpatronizeme_pcl.tsv")

# convert to huggingface dataset format
raw_datasets = DatasetDict({
    "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
    "validation": Dataset.from_pandas(dev_df.reset_index(drop=True)),
})

# model implementation: deberta + weighted loss

MODEL_NAME = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=False, max_length=256)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# calculate class weights (for pcl imbalance)
num_pos = sum(train_df['label'])
num_neg = len(train_df) - num_pos
weights = torch.tensor([len(train_df)/(2*num_neg), len(train_df)/(2*num_pos)], dtype=torch.float)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = weights.to(device)

class PCLTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        loss_fct = nn.CrossEntropyLoss(weight=weights.to(device=logits.device, dtype=logits.dtype))
        
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# model & metrics
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"f1": f1_score(labels, preds, pos_label=1)}

# training arguments

training_args = TrainingArguments(
    output_dir="./pcl_model",
    num_train_epochs=4,           
    per_device_train_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.05,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    warmup_ratio=0.2,
    lr_scheduler_type="cosine",
    bf16=True,
)

trainer = PCLTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# start training
print("\n--- Starting Training ---")
trainer.train()

# final evaluation & export

print("\n--- Generating dev.txt for submission ---")
predictions = trainer.predict(tokenized_datasets["validation"])
preds = np.argmax(predictions.predictions, axis=-1)

# write to file in the format required by the spec
with open("dev.txt", "w") as f:
    for p in preds:
        f.write(f"{p}\n")

print("Done! Model trained and 'dev.txt' created.")