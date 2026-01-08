# Load the dataset from HuggingFace, if the split include all three train, validation, and test, then use it as it is.
# If the dataset only contain train and test -> we split the train set into train and validation sets in the 80/20 ratio.

# Also optionally cleanup the dataset if needed(remove unnecessary columns, multi-labels, etc.)



from datasets import load_dataset
import torch
import torch.nn.functional as F
import config as CFG
import pandas as pd
from datasets import Dataset, Value, ClassLabel

from config import *



def train_val_test_split(dataset_name, label_column, ratio=0.2, has_val=False):
    train_dataset = load_dataset(dataset_name, split="train")
    test_dataset = load_dataset(dataset_name, split="test")
    val_dataset = None

    if has_val:
        try:
            val_dataset = load_dataset(dataset_name, split="validation")
            print("Successfully loaded 'validation' split from source.")
        except Exception as e:
            print(f"Warning: Could not load 'validation' split ({str(e)}). Switching to manual split.")
            val_dataset = None

    if val_dataset is None:
        print("Using part of training set as validation set. Ratio 80/20 for train/validation.")
        
        labels = concepts_from_labels.get(dataset_name)
        
        if labels is None:
            print(f"No labels provided for {dataset_name}, extracting unique labels from the dataset.")
            labels = train_dataset.unique(label_column)
            labels = [str(label) for label in labels]

        train_dataset = train_dataset.cast_column(label_column, ClassLabel(names=labels))
        test_dataset = test_dataset.cast_column(label_column, ClassLabel(names=labels))
        
        train_val_split = train_dataset.train_test_split(test_size=ratio, seed=42, stratify_by_column=label_column)

        train_dataset = train_val_split["train"]
        val_dataset = train_val_split["test"]

    print(f"Train dataset size: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    print(f"Successfully loaded dataset {dataset_name}")

    return train_dataset, val_dataset, test_dataset


def preprocess(dataset, dataset_name, text_column, label_column):
    dataset = clean_dataset_with_extra_columns(dataset, text_column, label_column)
    # dataset = clean_dataset_with_multiple_labels_per_row(dataset, text_column, label_column)
    dataset = preprocess_label_column(dataset, dataset_name, label_column)

    dataset = dataset.filter(lambda x: x[text_column] is not None and x[text_column] != "")

    return dataset

def preprocess_label_column(dataset, dataset_name, label_column):

    def reformat_label(rows):
        return {label_column: [label - 1 for label in rows[label_column]]}

    def cast_type(dataset, column_name, dtype):
        new_features = dataset.features.copy()
        new_features[column_name] = Value(dtype)
        dataset = dataset.cast(new_features)
        return dataset

    if dataset_name == "Duyacquy/Single_label_medical_abstract":
        dataset = dataset.map(reformat_label, batched=True)

    elif dataset_name == "dd-n-kk/uci-drug-review-cleaned":
        # convert the rating to integers
        dataset = cast_type(dataset, label_column, "int32")
        dataset = dataset.map(reformat_label, batched=True)
    
    elif dataset_name == "Duyacquy/Pubmed_20k":
        label_map = {
            "BACKGROUND": 0, 
            "OBJECTIVE": 1, 
            "METHODS": 2, 
            "RESULTS": 3, 
            "CONCLUSIONS": 4
        }
        
        def map_pubmed_labels(rows):
            first_label = rows[label_column][0]
            
            if isinstance(first_label, int):
                return {label_column: rows[label_column]}
            else:
                return {label_column: [label_map[l.strip()] for l in rows[label_column]]}

        print(f"Mapping text labels for {dataset_name} to integers...")
        dataset = dataset.map(map_pubmed_labels, batched=True)
        # Cast sang int32 để đảm bảo tương thích với torch
        dataset = cast_type(dataset, label_column, "int32")

    return dataset


def clean_dataset_with_extra_columns(dataset, text_column, label_column):
    # Convert HF dataset to pandas DataFrame
    df = dataset.to_pandas()

    print(f"Rows before removing extra columns: {len(df)}")

    # Check if the text column and label column exist
    if text_column not in df.columns or label_column not in df.columns:
        raise ValueError(f"Columns '{text_column}' or '{label_column}' not found in the dataset.")

    # Keep only the specified text and label columns
    clean_df = df[[text_column, label_column]].reset_index(drop=True)

    # Convert back to Hugging Face Dataset
    dataset = Dataset.from_pandas(clean_df)

    print(f"Rows after removing extra columns: {len(dataset)}")

    return dataset


def clean_dataset_with_multiple_labels_per_row(dataset, text_column, label_column):

    # Convert HF dataset to pandas DataFrame
    df = dataset.to_pandas()

    print(f"Rows before removing contaminated: {len(df)}")

    # Count unique labels per text
    label_counts = df.groupby(text_column)[label_column].nunique()

    # Find contaminated texts (with more than one label)
    contaminated_texts = label_counts[label_counts > 1].index

    # Count number of contaminated rows
    contaminated_rows = df[df[text_column].isin(contaminated_texts)].shape[0]

    # Keep only clean rows
    clean_df = df[~df[text_column].isin(contaminated_texts)].reset_index(drop=True)

    # Convert back to Hugging Face Dataset
    dataset = Dataset.from_pandas(clean_df)

    print(f"Rows after removing contaminated: {len(dataset)}")

    return dataset
