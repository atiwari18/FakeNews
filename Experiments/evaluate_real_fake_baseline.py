from __future__ import annotations
import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
from datasets import Dataset
from safetensors.torch import load_file
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from dataset.dataset import LIAR2Dataset
from models.logistic_regression import LogisticRegressionDetector, build_real_only_bundle
from models.roberta_baseline import RoBERTaWithCreditVector, softmax_numpy

@dataclass
class BaselineEvaluationResult:
    model_name: str
    num_examples: int
    accuracy: float
    fake_recall: float
    false_real_rate: float
    macro_f1: float
    confusion_matrix: list[list[int]]
    classification_report: str

def parse_args() :
    parser = argparse.ArgumentParser(description="Evaluate real-trained classifiers on 1,000 real FAKE LIAR2 test claims.")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--num-examples", type=int, default=1000)
    parser.add_argument("--output-json", default="Results/synthetic/real_fake_1000_classifier_baseline.json",)
    parser.add_argument("--roberta-checkpoint", required=True)
    parser.add_argument("--roberta-base-model", default="roberta-base")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def load_real_fake_examples(split: str, num_examples: int):
    dataset = LIAR2Dataset(
        split=split,
        task_type="classification",
        label_scheme="binary",
        include_metadata=True,
    )

    texts: list[str] = []
    credit_vectors: list[list[float]] = []
    labels: list[int] = []

    for index in range(len(dataset)):
        sample = dataset[index]

        if int(sample["label"]) != 0:
            continue

        texts.append(sample["text"])
        credit_vectors.append(sample["credit_vector"].tolist())
        labels.append(int(sample["label"]))

        if len(texts) >= num_examples:
            break

    if len(texts) < num_examples:
        raise ValueError(
            f"Requested {num_examples} FAKE examples, but only found {len(texts)} in split={split}."
        )

    return (
        texts,
        np.asarray(credit_vectors, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
    )

def summarize_predictions(model_name: str, labels: np.ndarray, preds: np.ndarray):
    cm = confusion_matrix(labels, preds, labels=[0, 1])

    fake_total = int(np.sum(labels == 0))
    false_real = int(np.sum((labels == 0) & (preds == 1)))
    fake_recall = 0.0 if fake_total == 0 else 1.0 - (false_real / fake_total)

    return BaselineEvaluationResult(
        model_name=model_name,
        num_examples=int(len(labels)),
        accuracy=float(np.mean(preds == labels)),
        fake_recall=float(fake_recall),
        false_real_rate=float(1.0 - fake_recall),
        
        macro_f1=float(
            f1_score(
                labels,
                preds,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
        
        confusion_matrix=cm.tolist(),
        
        classification_report=classification_report(
            labels,
            preds,
            labels=[0, 1],
            target_names=["FAKE", "REAL"],
            zero_division=0,
        )
    )

def evaluate_logistic_regression(texts: list[str], credit_vectors: np.ndarray, labels: np.ndarray):
    data = build_real_only_bundle()

    model = LogisticRegressionDetector(
        max_features=20000,
        C=1.0,
        random_state=42,
    )

    model.fit(
        texts=data.train_texts,
        credit_vectors=data.train_credit_vectors,
        labels=data.train_labels,
    )

    preds = model.predict(texts, credit_vectors)

    return summarize_predictions(
        model_name="logistic_regression_real_fake_baseline",
        labels=labels,
        preds=preds,
    )

def build_roberta_dataset(texts: list[str], credit_vectors: np.ndarray, labels: np.ndarray, tokenizer, max_length: int):
    dataset = Dataset.from_dict(
        {
            "text": texts,
            "labels": labels.tolist(),
            "credit_vector": credit_vectors.tolist(),
        }
    )

    tokenized = dataset.map(
        lambda batch: tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        ),
        batched=True,
        desc="Tokenizing real FAKE claims for RoBERTa",
    )

    tokenized = tokenized.remove_columns(["text"])
    tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels", "credit_vector"],
    )

    return tokenized


def evaluate_roberta(texts: list[str], credit_vectors: np.ndarray, labels: np.ndarray, checkpoint: Path, base_model: str, max_length: int) :
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    dataset = build_roberta_dataset(
        texts=texts,
        credit_vectors=credit_vectors,
        labels=labels,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    model = RoBERTaWithCreditVector(
        model_name=base_model,
        num_labels=2,
    )

    state_dict = load_file(str(checkpoint / "model.safetensors"))
    model.load_state_dict(state_dict)

    args = TrainingArguments(
        output_dir=str(PROJECT_ROOT / "Results" / "real_fake_baseline_eval_trainer_output"),
        per_device_eval_batch_size=16,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        processing_class=tokenizer,
    )

    predictions = trainer.predict(dataset)
    logits = np.asarray(predictions.predictions)
    probs = softmax_numpy(logits)
    preds = np.argmax(probs, axis=1)

    return summarize_predictions(
        model_name=f"roberta_credit_vector_real_fake_baseline:{checkpoint}",
        labels=labels,
        preds=preds,
    )


def print_result(result: BaselineEvaluationResult):
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


def main():
    args = parse_args()
    set_seed(args.seed)

    output_json = Path(args.output_json)
    if not output_json.is_absolute():
        output_json = PROJECT_ROOT / output_json

    output_json.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = Path(args.roberta_checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint

    texts, credit_vectors, labels = load_real_fake_examples(
        split=args.split,
        num_examples=args.num_examples,
    )

    results = [
        evaluate_logistic_regression(
            texts=texts,
            credit_vectors=credit_vectors,
            labels=labels,
        ),
        evaluate_roberta(
            texts=texts,
            credit_vectors=credit_vectors,
            labels=labels,
            checkpoint=checkpoint,
            base_model=args.roberta_base_model,
            max_length=args.max_length,
        ),
    ]

    for result in results:
        print_result(result)

    with output_json.open("w", encoding="utf-8") as output_file:
        json.dump([asdict(result) for result in results], output_file, indent=2)

    print(f"\nSaved real FAKE baseline evaluation to: {output_json}")


if __name__ == "__main__":
    main()