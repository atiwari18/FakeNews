from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from safetensors.torch import load_file
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.logistic_regression import LogisticRegressionDetector, build_real_only_bundle
from models.roberta_baseline import RoBERTaWithCreditVector, softmax_numpy

CREDIT_VECTOR_COLUMNS = [
    "true_counts",
    "mostly_true_counts",
    "half_true_counts",
    "mostly_false_counts",
    "false_counts",
    "pants_on_fire_counts",
]


@dataclass
class SyntheticEvaluationResult:
    model_name: str
    num_examples: int
    accuracy: float
    fake_recall: float
    false_real_rate: float
    macro_f1: float
    confusion_matrix: list[list[int]]
    classification_report: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate real-trained classifiers on synthetic FAKE claims.")
    parser.add_argument("--synthetic-csv", default="Results/synthetic_lfm25_claims.csv")
    parser.add_argument("--output-json", default="Results/synthetic_classifier_evaluation.json")
    parser.add_argument("--roberta-checkpoint", default=None)
    parser.add_argument("--roberta-base-model", default="roberta-base")
    parser.add_argument("--max-length", type=int, default=512)
    return parser.parse_args()


def load_synthetic_rows(csv_path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    texts: list[str] = []
    credit_vectors: list[list[float]] = []
    labels: list[int] = []

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            text = row.get("text") or row.get("synthetic_claim") or row.get("claim")
            if not text:
                continue

            texts.append(text)
            labels.append(int(row.get("label") or 0))
            credit_vectors.append([float(row.get(column) or 0.0) for column in CREDIT_VECTOR_COLUMNS])

    if not texts:
        raise ValueError(f"No synthetic examples found in {csv_path}")

    return texts, np.asarray(credit_vectors, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def summarize_predictions(model_name: str, labels: np.ndarray, preds: np.ndarray) -> SyntheticEvaluationResult:
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    fake_total = int(np.sum(labels == 0))
    false_real = int(np.sum((labels == 0) & (preds == 1)))
    fake_recall = 0.0 if fake_total == 0 else 1.0 - (false_real / fake_total)

    return SyntheticEvaluationResult(
        model_name=model_name,
        num_examples=int(len(labels)),
        accuracy=float(np.mean(preds == labels)),
        fake_recall=float(fake_recall),
        false_real_rate=float(1.0 - fake_recall),
        macro_f1=float(f1_score(labels, preds, labels=[0, 1], average="macro", zero_division=0)),
        confusion_matrix=cm.tolist(),
        classification_report=classification_report(
            labels,
            preds,
            labels=[0, 1],
            target_names=["FAKE", "REAL"],
            zero_division=0,
        ),
    )


def evaluate_logistic_regression(texts: list[str], credit_vectors: np.ndarray, labels: np.ndarray) -> SyntheticEvaluationResult:
    data = build_real_only_bundle()
    model = LogisticRegressionDetector(max_features=20000, C=1.0, random_state=42)
    model.fit(data.train_texts, data.train_credit_vectors, data.train_labels)
    preds = model.predict(texts, credit_vectors)
    return summarize_predictions("logistic_regression", labels, preds)


def find_roberta_checkpoint(path_arg: str | None) -> Path:
    if path_arg:
        checkpoint = Path(path_arg)
        if not checkpoint.is_absolute():
            checkpoint = PROJECT_ROOT / checkpoint
        return checkpoint

    candidates = sorted((PROJECT_ROOT / "Results").glob("**/trainer_output/checkpoint-*"))
    candidates = [candidate for candidate in candidates if (candidate / "model.safetensors").exists()]
    if not candidates:
        raise FileNotFoundError("No RoBERTa checkpoint with model.safetensors found under Results.")

    return max(candidates, key=lambda path: path.stat().st_mtime)


def build_roberta_dataset(
    texts: list[str],
    credit_vectors: np.ndarray,
    labels: np.ndarray,
    tokenizer,
    max_length: int,
) -> Dataset:
    dataset = Dataset.from_dict(
        {
            "text": texts,
            "labels": labels.tolist(),
            "credit_vector": credit_vectors.tolist(),
        }
    )

    tokenized = dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
        batched=True,
        desc="Tokenizing synthetic claims for RoBERTa",
    )
    tokenized = tokenized.remove_columns(["text"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "credit_vector"])
    return tokenized


def evaluate_roberta(
    texts: list[str],
    credit_vectors: np.ndarray,
    labels: np.ndarray,
    checkpoint: Path,
    base_model: str,
    max_length: int,
) -> SyntheticEvaluationResult:
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    dataset = build_roberta_dataset(texts, credit_vectors, labels, tokenizer, max_length)

    model = RoBERTaWithCreditVector(model_name=base_model, num_labels=2)
    state_dict = load_file(str(checkpoint / "model.safetensors"))
    model.load_state_dict(state_dict)

    args = TrainingArguments(
        output_dir=str(PROJECT_ROOT / "Results" / "synthetic_eval_trainer_output"),
        per_device_eval_batch_size=16,
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args, processing_class=tokenizer)
    predictions = trainer.predict(dataset)
    logits = np.asarray(predictions.predictions)
    probs = softmax_numpy(logits)
    preds = np.argmax(probs, axis=1)

    result = summarize_predictions(f"roberta_credit_vector:{checkpoint}", labels, preds)
    return result


def print_result(result: SyntheticEvaluationResult) -> None:
    print(f"\n{result.model_name}")
    print("-" * len(result.model_name))
    print(f"Examples        : {result.num_examples}")
    print(f"Accuracy        : {result.accuracy:.4f}")
    print(f"Fake recall     : {result.fake_recall:.4f}")
    print(f"False-real rate : {result.false_real_rate:.4f}")
    print(f"Macro F1        : {result.macro_f1:.4f}")
    print("Confusion matrix [[FAKE->FAKE, FAKE->REAL], [REAL->FAKE, REAL->REAL]]:")
    print(result.confusion_matrix)
    print("\nClassification Report:")
    print(result.classification_report)


def main() -> None:
    args = parse_args()
    set_seed(42)

    synthetic_csv = Path(args.synthetic_csv)
    if not synthetic_csv.is_absolute():
        synthetic_csv = PROJECT_ROOT / synthetic_csv

    output_json = Path(args.output_json)
    if not output_json.is_absolute():
        output_json = PROJECT_ROOT / output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)

    texts, credit_vectors, labels = load_synthetic_rows(synthetic_csv)
    results = [evaluate_logistic_regression(texts, credit_vectors, labels)]

    checkpoint = find_roberta_checkpoint(args.roberta_checkpoint)
    results.append(evaluate_roberta(texts, credit_vectors, labels, checkpoint, args.roberta_base_model, args.max_length))

    for result in results:
        print_result(result)

    with output_json.open("w", encoding="utf-8") as output_file:
        json.dump([asdict(result) for result in results], output_file, indent=2)

    print(f"\nSaved synthetic evaluation to: {output_json}")


if __name__ == "__main__":
    main()
