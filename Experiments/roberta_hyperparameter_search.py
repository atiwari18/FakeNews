from __future__ import annotations
import csv
import itertools
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np

from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from dataset.roberta_data import RobertaDataConfig, RobertaDatasetBuilder
from models.roberta_baseline import compute_metrics, evaluate_split

#Object to store hyperparameters for one run
@dataclass
class SearchConfig:
    learning_rate: float
    num_train_epochs: int
    weight_decay: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    max_length: int
    warmup_ratio: float

#Class to store the final validation metrics for one completed run
@dataclass
class SearchResult:
    learning_rate: float
    num_train_epochs: int
    weight_decay: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    max_length: int
    warmup_ratio: float
    accuracy: float
    macro_f1: float
    roc_auc: float
    pr_auc: float

#Build the training arguments for one hyperparameter trial.
def build_training_arguments(output_dir: str, config: SearchConfig) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
        seed=42,
    )

#Train and evaluate one configuration on the validation split.
def run_single_experiment(config: SearchConfig, run_name: str) -> SearchResult:
    print("\n" + "=" * 80)
    print(f"Starting run: {run_name}")
    print(asdict(config))
    print("=" * 80)

    data_config = RobertaDataConfig(
        model_name="roberta-base",
        max_length=config.max_length,
        label_scheme="binary",
        include_metadata=True,
    )

    builder = RobertaDatasetBuilder(data_config)
    tokenized_datasets = builder.build_tokenized_dataset_dict()
    data_collator = builder.get_data_collator()

    model = AutoModelForSequenceClassification.from_pretrained(
        data_config.model_name,
        num_labels=2,
    )

    training_args = build_training_arguments(
        output_dir=f"roberta_hparam_runs/{run_name}",
        config=config,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=builder.tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    validation_results = evaluate_split(trainer, tokenized_datasets["validation"])

    print("\nValidation summary")
    print(f"Accuracy : {validation_results.accuracy:.4f}")
    print(f"Macro F1 : {validation_results.macro_f1:.4f}")
    print(f"ROC AUC  : {validation_results.roc_auc:.4f}")
    print(f"PR AUC   : {validation_results.pr_auc:.4f}")

    return SearchResult(
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        weight_decay=config.weight_decay,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        max_length=config.max_length,
        warmup_ratio=config.warmup_ratio,
        accuracy=validation_results.accuracy,
        macro_f1=validation_results.macro_f1,
        roc_auc=validation_results.roc_auc,
        pr_auc=validation_results.pr_auc,
    )


#Save all search results so you can inspect them later in Excel, pandas
def save_results_csv(results: list[SearchResult], output_path: str) -> None:
    if not results:
        return

    fieldnames = list(asdict(results[0]).keys())

    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def save_best_result_json(best_result: SearchResult, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(asdict(best_result), json_file, indent=2)


def main() -> None:
    set_seed(42)

    #Search space
    learning_rates = [1e-5, 2e-5, 3e-5, 5e-5]
    num_train_epochs_list = [3, 6, 10]
    weight_decays = [0.0, 0.01, 0.05]
    train_batch_sizes = [8]
    eval_batch_size = 16
    max_lengths = [256, 384]
    warmup_ratios = [0.0, 0.06, 0.1]

    all_configs = []

    for values in itertools.product(
        learning_rates,
        num_train_epochs_list,
        weight_decays,
        train_batch_sizes,
        max_lengths,
        warmup_ratios,
    ):
        config = SearchConfig(
            learning_rate=values[0],
            num_train_epochs=values[1],
            weight_decay=values[2],
            per_device_train_batch_size=values[3],
            per_device_eval_batch_size=eval_batch_size,
            max_length=values[4],
            warmup_ratio=values[5],
        )
        all_configs.append(config)

    results: list[SearchResult] = []

    print(f"Total runs to execute: {len(all_configs)}")

    for index, config in enumerate(all_configs, start=1):
        run_name = f"run_{index:03d}"

        try:
            result = run_single_experiment(config=config, run_name=run_name)
            results.append(result)
        except Exception as exc:
            print(f"\nRun failed: {run_name}")
            print(asdict(config))
            print(f"Error: {exc}")

    if not results:
        print("\nNo successful runs completed.")
        return

    # Choose the best run by validation macro F1.
    best_result = max(results, key=lambda result: result.macro_f1)

    print("\n" + "=" * 80)
    print("Best validation configuration")
    print("=" * 80)
    print(json.dumps(asdict(best_result), indent=2))

    save_results_csv(results, "roberta_hyperparameter_search_results.csv")
    save_best_result_json(best_result, "roberta_best_hyperparameters.json")

if __name__ == "__main__":
    main()