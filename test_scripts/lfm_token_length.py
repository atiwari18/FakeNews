from transformers import AutoTokenizer
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from dataset.dataset import LIAR2Dataset

tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2.5-350M")

dataset = LIAR2Dataset(
    split="train",
    task_type="generation",
    label_scheme="binary",
    include_metadata=True,
)

lengths = []

for i in range(len(dataset)):
    sample = dataset[i]
    token_ids = tokenizer(sample["training_text"], truncation=False)["input_ids"]
    lengths.append(len(token_ids))

    if i < 3:
        print("\n" + "=" * 80)
        print(f"Example {i}")
        print("=" * 80)

        print("\nConditioning prompt:")
        print(sample["conditioning_prompt"])

        print("\nFull training text:")
        print(sample["training_text"])

        print("\nTarget claim only:")
        print(sample["target_text"])

        print("\nToken length:")
        print(len(token_ids))

lengths = np.array(lengths)

print("\nNumber of examples:", len(lengths))
print("Min tokens:", lengths.min())
print("Mean tokens:", lengths.mean())
print("Median tokens:", np.percentile(lengths, 50))
print("90th percentile:", np.percentile(lengths, 90))
print("95th percentile:", np.percentile(lengths, 95))
print("99th percentile:", np.percentile(lengths, 99))
print("Max tokens:", lengths.max())
