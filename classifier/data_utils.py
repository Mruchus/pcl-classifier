import pandas as pd
import numpy as np

def prepare_data(pcl_file, train_labels_file, dev_labels_file):
    # load the main dataset
    cols = ['id', 'public_id', 'keyword', 'country', 'text', 'label']
    df = pd.read_csv(
        pcl_file, sep='\t', skipinitialspace=True,
        names=cols, index_col='id', quoting=3
    )

    # fill missing text with empty string so we can still build the final field
    df['text'] = df['text'].fillna('')

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

    # split DataFrame (currently whole dataset) into train and dev
    # based on the paragraph IDs
    train_df = df[df.index.isin(train_ids)]
    dev_df = df[df.index.isin(dev_ids)]

    # for training only: cut out very short texts (less than 10 characters)
    train_df = train_df[train_df['text'].str.strip() != '']
    train_df = train_df[train_df['text'].str.len() > 10]

    # for dev: we keep all rows, but we must ensure the order matches the original list
    # first filter dev_ids to only those that exist in dev_df (all should exist now)
    present_dev_ids = [id for id in dev_ids if id in dev_df.index]
    dev_df = dev_df.loc[present_dev_ids]

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
    # columns: par_id, art_id, text, keyword, country_code, span_start, span_finish, span_text, pcl_category, num_annotators
    span_df = pd.read_csv(
        categories_file,
        sep='\t',
        header=None,
        names=['par_id', 'art_id', 'text', 'keyword', 'country_code',
               'span_start', 'span_finish', 'span_text', 'pcl_category', 'num_annotators'],
        engine='python',
        quoting=3,
        on_bad_lines='warn'
    )
    # Group by par_id and collect spans as list of (start, end) tuples
    spans_by_par = {}
    for _, row in span_df.iterrows():
        pid = row['par_id']
        start = row['span_start']
        end = row['span_finish']
        spans_by_par.setdefault(pid, []).append((start, end))
    return spans_by_par

def create_token_labels(text, spans, tokenizer, max_length):
    # tokenise with offsets
    encoding = tokenizer(text, truncation=True, max_length=max_length, return_offsets_mapping=True)
    offsets = encoding['offset_mapping']
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    
    token_labels = [0] * len(input_ids)
    for start, end in spans:
        for i, (token_start, token_end) in enumerate(offsets):
            # skip special tokens (offset (0,0) for [CLS], etc.)
            if token_start is None or token_end is None:
                continue
            if token_start >= start and token_end <= end:
                token_labels[i] = 1

    # Pad to max_length so all sequences have uniform length
    pad_len = max_length - len(input_ids)
    if pad_len > 0:
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        input_ids = input_ids + [pad_token_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len
        token_labels = token_labels + [0] * pad_len

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