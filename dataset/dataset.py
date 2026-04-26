from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import Dataset

#Type aliases for readability
LIAR2SPLIT = Literal["train", "validation", "test"]
TASKTYPE = Literal["classification", "generation"]
LABELSCHEME = Literal["binary", "six_class"]

#Speak History columns in Liar 2
CREDIT_VECTOR_COLUMNS = [
    "true_counts",
    "mostly_true_counts",
    "half_true_counts",
    "mostly_false_counts",
    "false_counts",
    "pants_on_fire_counts",
]

#Class to keep a sample organized before convertin to exact format a model needs
@dataclass
class Liar2Example:
    statement: str
    original_label: int
    label: int
    subject: Optional[str]
    speaker: Optional[str]
    speaker_description: Optional[str]
    state_info: Optional[str]
    context: Optional[str]
    metadata_text: str
    credit_vector: list[float]
    conditioning_prompt: str
    training_text: str

#Binary Label Collapse
def collapse_binary_label(label: int):
    if label in {0, 1, 2}:
        return 0
    return 1

#Builds the normalized 6-dimensional credit vector
def normalize_credit_vector(example: Dict[str, Any]):
    raw_counts = [float(example.get(column, 0) or 0) for column in CREDIT_VECTOR_COLUMNS]
    total = sum(raw_counts)

    if total == 0:
        return [0.0] * len(raw_counts)
    
    return [count / total for count in raw_counts]

#Conver metadata columns into one readable text block
def build_metadata_text(example: Dict[str, Any]):
    metadata_fields = {
        "speaker": example.get("speaker"),
        "speaker_description": example.get("speaker_description"),
        "state_info": example.get("state_info"),
        "subject": example.get("subject"),
        "context": example.get("context"),
    }

    parts = []

    for field_name, value in metadata_fields.items():
        if value is None:
            continue

        value = str(value).strip()

        if not value or value.lower() == "null":
            continue

        #replace underscores so that the text is easier to read
        pretty_name = field_name.replace("_", " ")
        parts.append(f"{pretty_name}: {value}")

    return " | ".join(parts)

# Small helper to load one LIAR2 split directly from Hugging Face.
def load_liar2_split(split: LIAR2SPLIT):
    return load_dataset("chengxuphd/liar2", split=split)

#Helper to clean text fields
def clean_text(value: Any):
    if value is None:
        return None
    
    text = str(value).strip()

    if not text or text.lower() == "null":
        return None
    
    return text

#Build the prompt format for generation experiments
def build_conditioning_prompt(example: Dict[str, Any]):
    binary_label = collapse_binary_label(int(example["label"]))
    label_name = "FAKE" if binary_label == 0 else "REAL"

    #Getting speaker, subject and context
    speaker = clean_text(example.get("speaker")) or "unknown"
    subject = clean_text(example.get("subject")) or "unknown"
    context = clean_text(example.get("context")) or "unknown"

    return (
        f"[CONDITION] Label: {label_name} | "
        f"Speaker: {speaker} | "
        f"Subject: {subject} | "
        f"Context: {context}\n"
        f"[CLAIM]"
    )

def build_training_text(example: Dict[str, Any]) -> str:
    conditioning_prompt = build_conditioning_prompt(example)
    return f"{conditioning_prompt} {example['statement']}"

#Pytorch dataset wrapper
class LIAR2Dataset(Dataset):
    def __init__(self, split:LIAR2SPLIT, task_type: TASKTYPE="classification", 
                 label_scheme: LABELSCHEME="binary", include_metadata: bool=True):
        self.split = split
        self.task_type = task_type
        self.label_scheme = label_scheme
        self.include_metadata = include_metadata

        #Load the split during initialization
        self.dataset = load_liar2_split(split)

    def __len__(self):
        return len(self.dataset)
        
    #Generate labels for the Binary labels or 6 classes
    def _make_label(self, raw_label: int):
        if self.label_scheme == "binary":
            return collapse_binary_label(raw_label)
        return raw_label
    
    def __getitem__(self, index: int):
        row = self.dataset[index]

        #Original LIAR2 Label from 0 to 5
        raw_label = int(row["label"])

        #Project-specific label representation
        label = self._make_label(raw_label)

        #Build text features
        metadata_text = build_metadata_text(row) if self.include_metadata else ""
        credit_vector = normalize_credit_vector(row)
        conditioning_prompt = build_conditioning_prompt(row)
        training_text = build_training_text(row)

        #Getting features
        speaker = clean_text(row.get("speaker"))
        speaker_description = clean_text(row.get("speaker_description"))
        state_info = clean_text(row.get("state_info"))
        subject = clean_text(row.get("subject"))
        context = clean_text(row.get("context"))

        #Package the sample in a structured way first.
        example = Liar2Example(
            statement=row["statement"],
            original_label=raw_label,
            label=label,
            speaker=speaker,
            speaker_description=speaker_description,
            state_info=state_info,
            subject=subject,
            context=context,
            metadata_text=metadata_text,
            credit_vector=credit_vector,
            conditioning_prompt=conditioning_prompt,
            training_text=training_text,
        )

        #Generation mode is for fine-tuning a model that produces claims
        if self.task_type == "generation":
            return {
                "conditioning_prompt": example.conditioning_prompt,
                "training_text": example.training_text,
                "target_text": example.statement,
                "label": example.label,
                "speaker": example.speaker,
                "speaker_description": example.speaker_description,
                "state_info": example.state_info,
                "subject": example.subject,
                "context": example.context,
                "credit_vector": torch.tensor(example.credit_vector, dtype=torch.float32),
            }
        
        #Classification mode is for detector models, we combine the claims with metadata text
        #for models that consume text, while stilll returning the structured field separately
        text = example.statement
        if metadata_text:
            text = f"{text}\n\n[METADATA] {metadata_text}"

        return {
            "text": text,
            "statement": example.statement,
            "speaker": example.speaker,
            "speaker_description": example.speaker_description,
            "state_info": example.state_info,
            "subject": example.subject,
            "context": example.context,
            "metadata_text": example.metadata_text,
            "label": example.label,
            "original_label": example.original_label,
            "credit_vector": torch.tensor(example.credit_vector, dtype=torch.float32),
        }
