from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding
from dataset.dataset import LIAR2Dataset

@dataclass
class RobertaDataConfig:
    model_name: str = "roberta-base"
    max_length: int = 512
    label_scheme: str = "binary"
    include_metadata: bool = True

#COnvert one Liar2Dataset into a hugging face dataset obkect
def build_hf_split(split: str, label_scheme: str = "binary", include_metadata: bool=True):
    dataset = LIAR2Dataset(
        split=split, 
        task_type="classification", 
        label_scheme=label_scheme, 
        include_metadata=include_metadata
    )

    records: dict[str, list[Any]] = {
        "text": [], 
        "label": [], 
        "statement" : [], 
        "metadata_text" : [], 
        "credit_vector" : [], 
    }

    #Populating the records
    for i in range(len(dataset)):
        sample = dataset[i]
        records["text"].append(sample["text"])
        records["label"].append(int(sample["label"]))
        records["statement"].append(sample["statement"])
        records["metadata_text"].append(sample["metadata_text"])
        records["credit_vector"].append(sample["credit_vector"].tolist())

    return Dataset.from_dict(records)

#Dataset builder handles:
#   - loading train/validation/test splits
#   - tokenizing the text for RoBERTa
#   - creating a collator that bads batches dynamically
class RobertaDatasetBuilder:
    def __init__(self, config: RobertaDataConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    #Tokenize one batch of examples
    def tokenize_batch(self, batch: dict[str, list[Any]]):
        return self.tokenizer(
            batch["text"], 
            truncation=True, 
            max_length=self.config.max_length,
        )

    #Building the raw, untokenized DatasetDict (train, validation, test)
    def build_raw_dataset_dict(self):
        train_dataset = build_hf_split(
            split="train",
            label_scheme=self.config.label_scheme,
            include_metadata=self.config.include_metadata,
        )

        validation_dataset = build_hf_split(
            split="validation",
            label_scheme=self.config.label_scheme,
            include_metadata=self.config.include_metadata,
        )

        test_dataset = build_hf_split(
            split="test",
            label_scheme=self.config.label_scheme,
            include_metadata=self.config.include_metadata,
        )

        return DatasetDict(
            {
                "train": train_dataset,
                "validation": validation_dataset,
                "test": test_dataset,
            }
        )
    
    #Function to build the DatasetDict that RoBERTa will train on
    def build_tokenized_dataset_dict(self):
        raw_datasets = self.build_raw_dataset_dict()

        tokenized_datasets = raw_datasets.map(
            self.tokenize_batch, 
            batched=True, 
            desc="Tokenizing LIAR2 for RoBERTa"
        )

        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

        #The trainer only needs tokenized inputs and labels for the baseline.
        tokenized_datasets = tokenized_datasets.remove_columns(
            ["text", "statement", "metadata_text", "credit_vector"]
        )

        return tokenized_datasets
    
    #Dynamic padding
    def get_data_collator(self):
        return DataCollatorWithPadding(tokenizer=self.tokenizer)
        