import pandas as pd
import numpy as np

def prepare_data(pcl_file, train_labels_file, dev_labels_file):
    cols = ['id', 'public_id', 'keyword', 'country', 'text', 'label']
    df = pd.read_csv(
        pcl_file, sep='\t', skipinitialspace=True,
        names=cols, index_col=False, quoting=3
    )
    df.columns = ['par_id', 'public_id', 'keyword', 'country', 'text', 'label']
    df['label'] = df['label'].apply(lambda x: 0 if x in [0, 1] else 1)
    df['text'] = df['text'].fillna('').astype(str)

    span_df = pd.read_csv(
        'Dont Patronize Me Categories.tsv', sep='\t', header=None,
        names=['par_id', 'art_id', 'text', 'keyword', 'country_code',
               'span_start', 'span_finish', 'span_text', 'pcl_category', 'num_annotators'],
        engine='python', quoting=3, on_bad_lines='warn'
    )

    train_ids = pd.read_csv(train_labels_file)['par_id'].tolist()
    dev_ids   = pd.read_csv(dev_labels_file)['par_id'].tolist()

    def build_split(ids):
        span_data = span_df[span_df['par_id'].isin(ids)][['par_id', 'span_text']]
        span_data.columns = ['par_id', 'span_label']

        main = df[df['par_id'].isin(ids)][['par_id', 'text', 'label']]
        main['text'] = main['text'].str.replace(r'@@\d+', '', regex=True).str.strip()

        # Right join: one row per span annotation (annotated examples duplicated)
        with_spans    = pd.merge(main, span_data, on='par_id', how='right')
        # Left join remainder: examples with no spans
        without_spans = pd.merge(main, span_data, on='par_id', how='left')
        without_spans = without_spans[without_spans['span_label'].isna()]

        merged = pd.concat([with_spans, without_spans], ignore_index=True)
        merged['text'] = merged['text'].fillna('').astype(str)
        return merged

    train_df = build_split(train_ids)
    dev_df   = build_split(dev_ids)

    return train_df, dev_df


def prepare_test_data(test_file):
    test_df = pd.read_csv(test_file, sep='\t', header=None,
                          names=['t_id', 'user_id', 'keyword', 'country', 'text'])
    test_df['text'] = (
        test_df['keyword'].fillna('') + " [SEP] " +
        test_df['text'].str.replace(r'@@\d+', '', regex=True).str.strip()
    )
    return test_df


def create_token_labels(text, span_label, tokenizer, max_length):
    encoding = tokenizer(
        text, truncation=True, max_length=max_length,
        return_offsets_mapping=True, padding='max_length'
    )
    input_ids      = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    offsets        = encoding['offset_mapping']

    if span_label is None or (isinstance(span_label, float) and np.isnan(span_label)):
        token_labels = [0] * max_length
    else:
        span_text  = str(span_label)
        span_start = text.find(span_text)
        if span_start == -1:
            token_labels = [0] * max_length
        else:
            span_end     = span_start + len(span_text)
            token_labels = [
                1 if (s < span_end and e > span_start) else 0
                for s, e in offsets
            ]

    return input_ids, attention_mask, token_labels


def tokenize_with_spans(batch, tokenizer, max_length, spans_by_par=None):
    input_ids_list, attention_mask_list, token_labels_list, labels_list = [], [], [], []

    for text, label, span_label in zip(batch['text'], batch['label'], batch['span_label']):
        input_ids, attention_mask, token_labels = create_token_labels(
            text, span_label, tokenizer, max_length
        )
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        token_labels_list.append(token_labels)
        labels_list.append(label)

    return {
        'input_ids':      input_ids_list,
        'attention_mask': attention_mask_list,
        'labels':         labels_list,
        'token_labels':   token_labels_list,
    }