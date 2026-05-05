from __future__ import annotations
import argparse
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any
import torch
import json
import json
import math
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from dataset.dataset import LIAR2Dataset


@dataclass
class LoraTrainingConfig:
    #The 350M instruction model is the most practical first pass on a local GPU.
    model_name: str = "LiquidAI/LFM2.5-350M"

    #Where the trained LoRA adapter and tokenizer files will be saved.
    output_dir: str = "lfm25_lora_claim_generator"

    #Maximum total token length for prompt + claim.
    max_length: int = 600

    #Small batch sizes are friendlier to consumer GPUs.
    per_device_train_batch_size: int = 2

    #This simulates a larger batch by accumulating gradients before each optimizer step.
    gradient_accumulation_steps: int = 8

    #One epoch is a safe starting point while you are learning the workflow.
    num_train_epochs: float = 1.0

    #LoRA often uses a slightly higher learning rate than full fine-tuning.
    learning_rate: float = 2e-4

    #LoRA rank controls the size of the trainable adapter matrices.
    lora_r: int = 16

    #LoRA alpha scales the adapter update; common practice is alpha = 2 * r.
    lora_alpha: int = 32

    #Dropout adds regularization so the adapter does not memorize as easily.
    lora_dropout: float = 0.05

    #Fraction of training used for learning-rate warmup.
    warmup_ratio: float = 0.03

    #Set this to a small number like 200 for a quick smoke test.
    max_train_samples: int | None = None

    #Set this to a small number like 100 for a quick validation smoke test.
    max_eval_samples: int | None = None


def parse_args() -> LoraTrainingConfig:
    # Create a command-line parser with defaults from LoraTrainingConfig.
    parser = argparse.ArgumentParser(description="Train a LoRA adapter on LFM2.5 for synthetic claim generation.")

    defaults = LoraTrainingConfig()
    parser.add_argument("--model-name", default=defaults.model_name)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--max-length", type=int, default=defaults.max_length)
    parser.add_argument("--per-device-train-batch-size", type=int, default=defaults.per_device_train_batch_size)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=defaults.gradient_accumulation_steps)
    parser.add_argument("--num-train-epochs", type=float, default=defaults.num_train_epochs)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--lora-r", type=int, default=defaults.lora_r)
    parser.add_argument("--lora-alpha", type=int, default=defaults.lora_alpha)
    parser.add_argument("--lora-dropout", type=float, default=defaults.lora_dropout)
    parser.add_argument("--warmup-ratio", type=float, default=defaults.warmup_ratio)
    parser.add_argument("--max-train-samples", type=int, default=defaults.max_train_samples)
    parser.add_argument("--max-eval-samples", type=int, default=defaults.max_eval_samples)

    args = parser.parse_args()

    #Return the namespace as our typed dataclass config.
    return LoraTrainingConfig(**vars(args))


def build_generation_split(split: str, max_samples: int | None = None) -> Dataset:
    #Load the project dataset in generation mode instead of classification mode.
    liar2_dataset = LIAR2Dataset(
        split=split,
        task_type="generation",
        label_scheme="binary",
        include_metadata=True,
    )

    #Trainer wants a column-oriented Hugging Face Dataset, so we collect columns in lists.
    records: dict[str, list[Any]] = {
        "conditioning_prompt": [],
        "training_text": [],
        "target_text": [],
        "label": [],
        "speaker": [],
        "subject": [],
        "context": [],
    }

    #Decide how many rows to copy; max_samples is useful for quick test runs.
    limit = len(liar2_dataset) if max_samples is None else min(max_samples, len(liar2_dataset))

    #Copy each LIAR2 example into the simple table structure above.
    for index in range(limit):
        # Pull one generation example from the existing dataset wrapper.
        sample = liar2_dataset[index]

        #Keep only FAKE examples.
        if int(sample["label"]) != 0:
            continue

        #The prompt contains the control information: label, speaker, subject, and context.
        records["conditioning_prompt"].append(sample["conditioning_prompt"])

        #The training text is prompt + real claim, which is what the causal LM learns to continue.
        records["training_text"].append(sample["training_text"])

        #The target text is the real claim alone; we keep it for inspection/debugging.
        records["target_text"].append(sample["target_text"])

        #Keep the binary label so generated datasets can preserve FAKE/REAL balance later.
        records["label"].append(int(sample["label"]))

        #Keep metadata columns so we can audit what conditions the model saw.
        records["speaker"].append(sample["speaker"])
        records["subject"].append(sample["subject"])
        records["context"].append(sample["context"])

    #Convert the Python lists into a Hugging Face Dataset.
    return Dataset.from_dict(records)


def build_tokenize_function(tokenizer: AutoTokenizer, max_length: int):
    #This nested function closes over tokenizer and max_length.
    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        #Tokenize the full prompt + claim text.
        full_text = tokenizer(
            batch["training_text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        #Tokenize just the conditioning prompts so we can hide prompt tokens from the loss.
        prompts = tokenizer(
            batch["conditioning_prompt"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        #Create the labels list that Trainer will use for language-model loss.
        labels = []

        #Loop over each example in the batch.
        for input_ids, prompt_ids in zip(full_text["input_ids"], prompts["input_ids"]):
            #Start labels as a copy of input_ids because causal LM labels are next-token targets.
            example_labels = input_ids.copy()

            #Count prompt tokens so the model is graded only on the claim continuation.
            prompt_length = min(len(prompt_ids), len(example_labels))

            #Replace prompt labels with -100; PyTorch ignores -100 in cross-entropy loss.
            example_labels[:prompt_length] = [-100] * prompt_length

            #Store the masked labels for this example.
            labels.append(example_labels)

        #Attach labels to the tokenized output.
        full_text["labels"] = labels

        #Return input_ids, attention_mask, and labels.
        return full_text

    # Return the configured batch tokenization function.
    return tokenize_batch


def causal_lm_collator(tokenizer: AutoTokenizer):
    #This collator pads input_ids, attention_mask, and labels to the longest example in a batch.
    def collate(features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        #Extract labels before tokenizer.pad because labels need a different padding value.
        labels = [feature.pop("labels") for feature in features]

        #Pad input_ids and attention_mask with the tokenizer's normal padding behavior.
        batch = tokenizer.pad(features, padding=True, return_tensors="pt")

        #Find the padded sequence length used for input_ids.
        max_length = batch["input_ids"].shape[1]

        #Pad labels with -100 so padding tokens do not contribute to the loss.
        padded_labels = [label + [-100] * (max_length - len(label)) for label in labels]

        #Convert the padded labels to a PyTorch tensor and attach them to the batch.
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        #Return the complete batch dictionary expected by AutoModelForCausalLM.
        return batch

    return collate


def build_lora_model(config: LoraTrainingConfig):
    #Use bf16 on CUDA when available; it is memory efficient and stable on modern GPUs.
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    #Load the base causal language model; only LoRA adapter weights will be trainable.
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    #Disable cache during training because gradient checkpointing and Trainer expect it off.
    model.config.use_cache = False

    #These target names cover common attention/projection modules in modern causal LMs.
    lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    #Define the LoRA adapter: small trainable matrices added to selected Linear layers.
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_targets,
    )

    #Wrap the base model with PEFT so the original model weights are frozen.
    model = get_peft_model(model, lora_config)

    #Print a summary showing how few parameters are trainable.
    model.print_trainable_parameters()

   
    return model

def safe_perplexity(loss: float):
    return math.exp(loss) if loss < 20 else float("inf")


def extract_final_train_loss(log_history: list[dict]):
    train_losses = [entry["loss"] for entry in log_history if "loss" in entry]

    if not train_losses:
        return None

    return train_losses[-1]


def plot_run_history(log_history: list[dict], output_path: Path, title: str) -> None:
    train_steps = []
    train_losses = []
    eval_steps = []
    eval_losses = []

    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(entry["step"])
            train_losses.append(entry["loss"])

        if "eval_loss" in entry:
            eval_steps.append(entry["step"])
            eval_losses.append(entry["eval_loss"])

    plt.figure(figsize=(8, 5))

    if train_steps:
        plt.plot(train_steps, train_losses, label="Training loss")

    if eval_steps:
        plt.plot(eval_steps, eval_losses, marker="o", label="Validation loss")

    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    
def main() -> None:
    #Read command-line options into our config object.
    config = parse_args()

    #Fix random seeds so repeated runs are more comparable.
    set_seed(42)

    #Print basic hardware information before training starts.
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
    print()

    #load the tokenizer for the chosen LFM2.5 model.
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    #Some causal LMs do not define a pad token, so we reuse the EOS token for padding.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    #Build the real training split; synthetic data is not involved in LoRA training yet.
    train_dataset = build_generation_split("train", max_samples=config.max_train_samples)
    eval_dataset = build_generation_split("validation", max_samples=config.max_eval_samples)
    tokenize_function = build_tokenize_function(tokenizer, max_length=config.max_length)

    #tokenize train examples and drop raw text columns before training.
    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing LIAR2 train examples",
    )

    #Tokenize validation examples the same way.
    tokenized_eval = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizing LIAR2 validation examples",
    )

    #Build the LFM2.5 causal LM with LoRA adapters attached.
    model = build_lora_model(config)

    
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
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
        save_total_limit=2,
        report_to="none",
        seed=42,
    )

    # Trainer connects the model, data, tokenizer/collator, and training arguments.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=causal_lm_collator(tokenizer),
    )

    # Run the actual LoRA fine-tuning loop.
    train_output = trainer.train()

    # Run a final validation pass so we can report final eval loss/perplexity.
    eval_metrics = trainer.evaluate()

    # Collect the final metrics we care about.
    eval_loss = float(eval_metrics["eval_loss"])
    eval_ppl = safe_perplexity(eval_loss)
    final_train_loss = extract_final_train_loss(trainer.state.log_history)

    # Make sure the output directory exists before writing plots/metrics.
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save a training/validation loss plot.
    plot_run_history(
        log_history=trainer.state.log_history,
        output_path=output_path / "training_validation_loss.png",
        title="Final LFM2.5 LoRA Training History",
    )

    # Save final metrics and the exact training config.
    metrics = {
        "config": asdict(config),
        "final_train_loss": final_train_loss,
        "eval_loss": eval_loss,
        "eval_ppl": eval_ppl,
        "train_runtime": train_output.metrics.get("train_runtime"),
    }

    with open(output_path / "final_metrics.json", "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    # Save the small LoRA adapter files, not a full copy of the base model.
    trainer.model.save_pretrained(config.output_dir)

    # Save the tokenizer beside the adapter so generation scripts use the same tokenization.
    tokenizer.save_pretrained(config.output_dir)

    # Print the final location and final validation metrics.
    print(f"\nSaved LoRA adapter and tokenizer to: {config.output_dir}")
    print(f"Final train loss: {final_train_loss}")
    print(f"Final eval loss : {eval_loss:.4f}")
    print(f"Final eval PPL  : {eval_ppl:.4f}")


if __name__ == "__main__":
    main()
