from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from dataset.dataset import LIAR2Dataset  # noqa: E402


def summarize_lengths(lengths: list[int], name: str) -> None:
    lengths_array = np.array(lengths)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Count           : {len(lengths_array)}")
    print(f"Min tokens      : {lengths_array.min()}")
    print(f"Mean tokens     : {lengths_array.mean():.2f}")
    print(f"Median tokens   : {np.median(lengths_array):.2f}")
    print(f"90th percentile : {np.percentile(lengths_array, 90):.2f}")
    print(f"95th percentile : {np.percentile(lengths_array, 95):.2f}")
    print(f"99th percentile : {np.percentile(lengths_array, 99):.2f}")
    print(f"Max tokens      : {lengths_array.max()}")

    for cutoff in [128, 256, 384, 512]:
        truncated = int((lengths_array > cutoff).sum())
        pct = 100 * truncated / len(lengths_array)
        print(f"Above {cutoff:>3} tokens : {truncated:>5} ({pct:.2f}%)")


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    dataset = LIAR2Dataset(
        split="train",
        task_type="classification",
        label_scheme="binary",
        include_metadata=True,
    )

    statement_lengths = []
    full_text_lengths = []

    longest_statement = ("", 0)
    longest_full_text = ("", 0)

    for i in range(len(dataset)):
        sample = dataset[i]

        statement = sample["statement"]
        full_text = sample["text"]

        statement_tokens = tokenizer(
            statement,
            truncation=False,
            add_special_tokens=True,
        )["input_ids"]

        full_text_tokens = tokenizer(
            full_text,
            truncation=False,
            add_special_tokens=True,
        )["input_ids"]

        statement_len = len(statement_tokens)
        full_text_len = len(full_text_tokens)

        statement_lengths.append(statement_len)
        full_text_lengths.append(full_text_len)

        if statement_len > longest_statement[1]:
            longest_statement = (statement, statement_len)

        if full_text_len > longest_full_text[1]:
            longest_full_text = (full_text, full_text_len)

    summarize_lengths(statement_lengths, "Statement Token Lengths")
    summarize_lengths(full_text_lengths, "Statement + Metadata Token Lengths")

    print("\nLongest statement example")
    print("-------------------------")
    print(f"Token length: {longest_statement[1]}")
    print(longest_statement[0][:1000])

    print("\nLongest full text example")
    print("-------------------------")
    print(f"Token length: {longest_full_text[1]}")
    print(longest_full_text[0][:1500])


if __name__ == "__main__":
    main()