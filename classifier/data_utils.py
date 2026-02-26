import pandas as pd
import torch
from datasets import Dataset, DatasetDict

def prepare_data(pcl_file, train_labels_file, dev_labels_file):
    # load the main dataset
    cols = ['id', 'public_id', 'keyword', 'country', 'text', 'label']
    df = pd.read_csv(
        pcl_file, sep='\t', skipinitialspace=True,
        names=cols, index_col='id', quoting=3
    )

    # drop rows were text or label is missing
    df = df.dropna(subset=['text', 'label'])

    # converts the original multi‑class labels (0–4) into binary labels
    df['label'] = df['label'].apply(lambda x: 1 if x >= 2 else 0)

    # builds the final text field by concatenating the keyword (if present)
    # separated by the special token [SEP]
    # removes any occurrences of @@\d+ (user mentions or annotations), strip extra whitespace
    df['text'] = (
        df['keyword'].fillna('') + " [SEP] " +
        df['text'].str.replace(r'@@\d+', '', regex=True).str.strip()
    )

    # loads train and dev IDs
    train_ids = pd.read_csv(train_labels_file)['par_id'].tolist()
    dev_ids = pd.read_csv(dev_labels_file)['par_id'].tolist()

    print(f"df.index.dtype: {df.index.dtype}")
    print(f"First 5 dev_ids: {dev_ids[:5]}")
    print(f"Type of first dev_id: {type(dev_ids[0])}")
    print(f"Is 8640 in df.index? {8640 in df.index}")
    print(f"Total dev IDs: {len(dev_ids)}")
    print(f"Dev IDs present in df: {sum(df.index.isin(dev_ids))}")

    # split DataFrame (currently whole dataset) into train and dev
    # based on the paragraph IDs
    train_df = df[df.index.isin(train_ids)]
    dev_df = df[df.index.isin(dev_ids)]

    missing = set(dev_ids) - set(dev_df.index)
    if missing:
        print(f"WARNING: Missing dev IDs: {missing}")
    else:
        print("All dev IDs present.")

    # ensure dev set is in the exact order that the ids are listed
    dev_df = dev_df.loc[dev_ids]

    # cut out very short texts (less than 10 characters)
    train_df = train_df[train_df['text'].str.strip() != '']
    train_df = train_df[train_df['text'].str.len() > 10]

    return train_df, dev_df

def prepare_test_data(test_file):
    # similar cleaning process to above train and dev
    test_df = pd.read_csv(test_file, sep='\t', header=None,
                          names=['t_id', 'user_id', 'keyword', 'country', 'text'])
    test_df['text'] = (
        test_df['keyword'].fillna('') + " [SEP] " +
        test_df['text'].str.replace(r'@@\d+', '', regex=True).str.strip()
    )

    return test_df

def load_span_data(categories_file):
    span_df = pd.read_csv(categories_file, sep='\t')
    spans_by_par = {}
    for _, row in span_df.iterrows():
        pid = row['par_id']
        spans_by_par.setdefault(pid, []).append((row['span_start'], row['span_finish']))
    return spans_by_par

def create_token_labels(text, spans, tokenizer, max_length):
    encoding = tokenizer(text, truncation=True, max_length=max_length, return_offsets_mapping=True)
    offsets = encoding['offset_mapping']
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    
    token_labels = [0] * len(input_ids)
    for start, end in spans:
        for i, (token_start, token_end) in enumerate(offsets):
            if token_start is None or token_end is None:
                continue
            if token_start >= start and token_end <= end:
                token_labels[i] = 1
    return input_ids, attention_mask, token_labels

def tokenize_with_spans(batch, tokenizer, max_length, spans_by_par):
    input_ids_list = []
    attention_mask_list = []
    token_labels_list = []
    labels_list = []
    for text, label, pid in zip(batch['text'], batch['label'], batch['id']):
        spans = spans_by_par.get(pid, [])
        input_ids, attention_mask, token_labels = create_token_labels(text, spans, tokenizer, max_length)
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        token_labels_list.append(token_labels)
        labels_list.append(label)
    return {
        'input_ids': input_ids_list,
        'attention_mask': attention_mask_list,
        'labels': labels_list,
        'token_labels': token_labels_list,
    }