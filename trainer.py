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
    DataCollatorWithPadding
)

def prepare_data(file_path):
    print("--- Loading and Cleaning Data ---")
    cols = ['id', 'public_id', 'keyword', 'country', 'text', 'label']
    # quoting=3 ensures we don't break on quotes inside paragraphs
    df = pd.read_csv(file_path, sep='\t', skipinitialspace=True, 
                     names=cols, index_col='id', quoting=3)
    
    # drop empty rows
    df = df.dropna(subset=['text', 'label'])
    
    # label mapping: 0,1 -> 0 (No PCL) | 2,3,4 -> 1 (PCL)
    df['label'] = df['label'].apply(lambda x: 1 if x >= 2 else 0)
    
    # clean text: remove the @@ID tags and strip whitespace
    df['text'] = df['text'].str.replace(r'@@\d+', '', regex=True).str.strip()
    
    # split into 80% train, 20% dev (stratified to keep class balance)
    train_df, dev_df = train_test_split(
        df[['text', 'label']], 
        test_size=0.2, 
        random_state=42, 
        stratify=df['label']
    )
    return train_df, dev_df

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
        outputs = model(**inputs)3
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
    learning_rate=1e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=10,


    bf16=True,
    fp16=False,
    max_grad_norm=1.0,
    warmup_ratio=0.1,
    adam_epsilon=1e-6,
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