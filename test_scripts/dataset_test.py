from __future__ import annotations
import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from dataset.dataset import LIAR2Dataset

def print_divider(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def test_classification_dataset():
    print_divider("TESTING CLASSIFICATION DATASET")

    dataset = LIAR2Dataset(
        split="train",
        task_type="classification",
        label_scheme="binary",
        include_metadata=True,
    )

    print(f"Dataset length: {len(dataset)}")

    sample = dataset[0]

    expected_keys = {
        "text",
        "statement",
        "speaker",
        "speaker_description",
        "state_info",
        "subject",
        "context",
        "metadata_text",
        "label",
        "original_label",
        "credit_vector",
    }

    print("\nReturned keys:")
    print(sample.keys())

    missing_keys = expected_keys - set(sample.keys())
    assert not missing_keys, f"Missing keys in classification sample: {missing_keys}"

    assert isinstance(sample["text"], str), "text should be a string"
    assert isinstance(sample["statement"], str), "statement should be a string"
    assert isinstance(sample["label"], int), "label should be an int"
    assert isinstance(sample["original_label"], int), "original_label should be an int"
    assert isinstance(sample["credit_vector"], torch.Tensor), "credit_vector should be a tensor"

    assert sample["credit_vector"].shape[0] == 6, "credit_vector should have length 6"
    assert sample["label"] in {0, 1}, "binary label should be 0 or 1"
    assert 0 <= sample["original_label"] <= 5, "original label should be between 0 and 5"

    vector_sum = float(sample["credit_vector"].sum().item())
    print(f"\nCredit vector sum: {vector_sum:.4f}")

    # If the vector is nonzero, it should sum to about 1.
    if vector_sum > 0:
        assert abs(vector_sum - 1.0) < 1e-5, "nonzero credit_vector should sum to 1"

    print("\nSample text:")
    print(sample["text"])

    print("\nSample label info:")
    print(f"Binary label: {sample['label']}")
    print(f"Original label: {sample['original_label']}")

    print("\nSample credit vector:")
    print(sample["credit_vector"])

    print("\nClassification dataset test passed.")

def test_generation_dataset():
    print_divider("TESTING GENERATION DATASET")

    dataset = LIAR2Dataset(
        split="train",
        task_type="generation",
        label_scheme="binary",
        include_metadata=True,
    )

    print(f"Dataset length: {len(dataset)}")

    sample = dataset[0]

    expected_keys = {
        "conditioning_prompt",
        "training_text",
        "target_text",
        "label",
        "speaker",
        "speaker_description",
        "state_info",
        "subject",
        "context",
        "credit_vector",
    }


    print("\nReturned keys:")
    print(sample.keys())

    missing_keys = expected_keys - set(sample.keys())
    assert not missing_keys, f"Missing keys in generation sample: {missing_keys}"

    assert isinstance(sample["training_text"], str), "training_text should be a string"
    assert isinstance(sample["target_text"], str), "target_text should be a string"
    assert isinstance(sample["label"], int), "label should be an int"
    assert isinstance(sample["credit_vector"], torch.Tensor), "credit_vector should be a tensor"

    assert sample["credit_vector"].shape[0] == 6, "credit_vector should have length 6"
    assert sample["label"] in {0, 1}, "binary label should be 0 or 1"

    print("\nSample conditioning prompt:")
    print(sample["conditioning_prompt"])

    print("\nSample training text:")
    print(sample["training_text"])

    print("\nSample target text:")
    print(sample["target_text"])

    print("\nSample credit vector:")
    print(sample["credit_vector"])

    print("\nGeneration dataset test passed.")


def test_six_class_labels():
    print_divider("TESTING SIX-CLASS LABEL MODE")

    dataset = LIAR2Dataset(
        split="train",
        task_type="classification",
        label_scheme="six_class",
        include_metadata=True,
    )

    sample = dataset[0]

    assert 0 <= sample["label"] <= 5, "six-class label should be between 0 and 5"
    assert sample["label"] == sample["original_label"], (
        "in six_class mode, label should match original_label"
    )

    print(f"Six-class label: {sample['label']}")
    print("Six-class label test passed.")


if __name__ == "__main__":
    test_classification_dataset()
    test_generation_dataset()
    test_six_class_labels()

    print("\nAll dataset tests passed successfully.")