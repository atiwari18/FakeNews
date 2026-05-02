from __future__ import annotations
import csv
import json
import math
import sys
import itertools
from dataclasses import asdict, dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from Experiments.train_lfm25_lora_generator import (
    LoraTrainingConfig,
    build_generation_split,
    build_lora_model,
    build_tokenize_function,
    causal_lm_collator
)
from Experiments.roberta_hyperparameter_search import save_results_csv, save_best_result_json

@dataclass
class SearchConfig:
    learning_rate: float
    num_train_epochs: float
    lora_r: int
    lora_alpha: int
    max_length: int
    lora_dropout: float
    warmup_ratio: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int

@dataclass
class SearchResult:
    run_name: str
    learning_rate: float
    num_train_epochs: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    max_length: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    warmup_ratio: float

    # Store the metrics we care about.
    final_train_loss: float | None
    eval_loss: float
    eval_ppl: float
    train_runtime: float | None


def safe_perplexity(loss: float):
    #perplexity = exp(loss) but very large losses can overflow.
    return math.exp(loss) if loss < 20 else float("inf")

def extract_final_train_loss(log_history: list[dict]):
    #Trainer logs many dictionaries; training-loss entries have "loss".
    train_losses = [entry["loss"] for entry in log_history if "loss" in entry]

    #If no training loss was logged, return None instead of crashing.
    if not train_losses:
        return None

    #The final logged training loss is a useful optimization summary.
    return train_losses[-1]

def plot_run_history(log_history: list[dict], output_path: Path, title: str) -> None:
    #Collect training-loss points.
    train_steps = []
    train_losses = []

    #Collect validation-loss points.
    eval_steps = []
    eval_losses = []

    #Walk through Trainer's log history.
    for entry in log_history:
        if "loss" in entry:
            train_steps.append(entry["step"])
            train_losses.append(entry["loss"])

        if "eval_loss" in entry:
            eval_steps.append(entry["step"])
            eval_losses.append(entry["eval_loss"])

    #Create a clean loss plot for this run.
    plt.figure(figsize=(8, 5))

    if train_steps:
        plt.plot(train_steps, train_losses, label="Training loss")

    if eval_steps:
        plt.plot(eval_steps, eval_losses, marker="o", label="Validation loss")

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_search_summary(results: list[SearchResult], output_path: Path) -> None:
    #Sort by validation loss so the plot shows best-to-worst runs.
    sorted_results = sorted(results, key=lambda result: result.eval_loss)

    #Use run names as x-axis labels.
    run_names = [result.run_name for result in sorted_results]

    #Pull validation losses and perplexities.
    eval_losses = [result.eval_loss for result in sorted_results]
    eval_ppls = [result.eval_ppl for result in sorted_results]

    #Plot validation loss.
    plt.figure(figsize=(10, 5))
    plt.bar(run_names, eval_losses)
    plt.xlabel("Run")
    plt.ylabel("Validation loss")
    plt.title("LFM2.5 LoRA Hyperparameter Search: Validation Loss")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path.with_name("lfm25_lora_search_eval_loss.png"), dpi=200)
    plt.close()

    #Plot validation perplexity.
    plt.figure(figsize=(10, 5))
    plt.bar(run_names, eval_ppls)
    plt.xlabel("Run")
    plt.ylabel("Validation perplexity")
    plt.title("LFM2.5 LoRA Hyperparameter Search: Validation PPL")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path.with_name("lfm25_lora_search_eval_ppl.png"), dpi=200)
    plt.close()

def run_single_experiment(config: SearchConfig, run_name: str, tokenizer: AutoTokenizer, train_dataset, eval_dataset, output_root):
    print("\n" + "=" * 80)
    print(f"Starting run: {run_name}")
    print(asdict(config))
    print("=" * 80)

    #convert this search config into the config expected by the LoRA training helpers.
    lora_config = LoraTrainingConfig(
        model_name="LiquidAI/LFM2.5-350M",
        output_dir=str(output_root / run_name),
        max_length=config.max_length,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
    )

    #build a tokenizer for this run's max_length
    tokenize_function = build_tokenize_function(tokenizer, max_length=config.max_length)

    #tokenize the training split.
    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc=f"Tokenizing train for {run_name}",
    )

    tokenized_eval = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc=f"Tokenizing validation for {run_name}",
    )

    #Fresh model and Lora adater
    model = build_lora_model(lora_config)

    training_args = TrainingArguments(
        output_dir=str(output_root / run_name),
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=25,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=0.0,
        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=True,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=causal_lm_collator(tokenizer),
    )

    train_output = trainer.train()

    #run final validation pass after training
    eval_metrics = trainer.evaluate()

    #save this run's loss curve.
    plot_run_history(
        log_history=trainer.state.log_history,
        output_path=output_root / f"{run_name}_loss_history.png",
        title=f"{run_name} Loss History",
    )

    #pull validation loss and convert to perplexity.
    eval_loss = float(eval_metrics["eval_loss"])
    eval_ppl = safe_perplexity(eval_loss)

    #pull the last logged training loss.
    final_train_loss = extract_final_train_loss(trainer.state.log_history)

    #return a structued result for csv/json
    return SearchResult(
        run_name=run_name,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_length=config.max_length,
        warmup_ratio=config.warmup_ratio,
        final_train_loss=final_train_loss,
        eval_loss=eval_loss,
        eval_ppl=eval_ppl,
        train_runtime=train_output.metrics.get("train_runtime"),
    )

def main():
    set_seed(42)

    output_root = Path("lfm25_lora_hparam_runs")
    output_root.mkdir(exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2.5-350M")

    #Some causal LMs do not define a pad token, so reuse EOS if needed.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    #Load raw generation splits once.
    train_dataset = build_generation_split("train")
    eval_dataset = build_generation_split("validation")

    #Search Space
    learning_rates = [1e-4, 2e-4]
    num_train_epochs = [1, 3]
    lora_ranks = [16, 32]
    lora_alphas = {
        16: [32], 
        32: [64]
    }
    lora_dropout = [0.05]
    train_batch_sizes = [4]

    #Effective batch_size = train_batch * gradient_accumulation_steps
    gradient_accumulation_steps = [4]
    warmup_ratios = [0.06]

    #build all hyperparameter combinations
    configs: list[SearchConfig] = []

    for (
        learning_rate,
        num_train_epochs,
        lora_rank,
        lora_dropout,
        train_batch_size,
        gradient_accumulation_steps,
        warmup_ratio,
    ) in itertools.product(
        learning_rates,
        num_train_epochs,
        lora_ranks,
        lora_dropout,
        train_batch_sizes,
        gradient_accumulation_steps,
        warmup_ratios,
    ):
        for lora_alpha in lora_alphas[lora_rank]:
            config = SearchConfig(
                learning_rate=learning_rate,
                num_train_epochs=num_train_epochs,
                lora_r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                per_device_train_batch_size=train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_length=160,
                warmup_ratio=warmup_ratio,)

            configs.append(config)

    print(f"Total runs to execute: {len(configs)}")

    # Store successful runs here.
    results: list[SearchResult] = []

    # Run each configuration.
    for index, config in enumerate(configs, start=1):
        run_name = f"run_{index:03d}"

        try:
            result = run_single_experiment(
                config=config,
                run_name=run_name,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                output_root=output_root,
            )

            results.append(result)

        except Exception as exc:
            print(f"\nRun failed: {run_name}")
            print(asdict(config))
            print(f"Error: {exc}")

    # Stop cleanly if every run failed.
    if not results:
        print("\nNo successful runs completed.")
        return

    # Lower validation loss is better.
    best_result = min(results, key=lambda result: result.eval_loss)

    # Save all results.
    save_results_csv(
        results=results,
        output_path=output_root / "lfm25_lora_hyperparameter_search_results.csv",)

    # Save the best result.
    save_best_result_json(
        best_result=best_result,
        output_path=output_root / "lfm25_lora_best_hyperparameters.json",)

    # Save summary plots across all runs.
    plot_search_summary(results=results, output_root=output_root)

    print("\n" + "=" * 80)
    print("Best validation configuration")
    print("=" * 80)
    print(json.dumps(asdict(best_result), indent=2))


if __name__ == "__main__":
    main()