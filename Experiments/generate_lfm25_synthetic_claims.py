from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path
import torch

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from dataset.dataset import LIAR2Dataset


def parse_args() -> argparse.Namespace:
    # Create a parser for command-line generation options.
    parser = argparse.ArgumentParser(description="Generate synthetic LIAR2-style claims with a trained LFM2.5 LoRA adapter.")

    parser.add_argument("--base-model-name", default="LiquidAI/LFM2.5-350M")
    parser.add_argument("--adapter-dir", default="lfm25_lora_claim_generator")
    parser.add_argument("--output-csv", default="synthetic_lfm25_claims.csv")
    parser.add_argument("--conditioning-split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--num-claims", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def load_model_and_tokenizer(base_model_name: str, adapter_dir: str):
    #Use bf16 on CUDA when available to reduce memory use.
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    #Load the tokenizer saved with the adapter; fall back to the base tokenizer if needed.
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

    #Ensure generation can pad inputs cleanly.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    #Load the frozen base causal LM.
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    #Attach the LoRA adapter that learned the claim-generation task.
    model = PeftModel.from_pretrained(base_model, adapter_dir)

    #Put the model in evaluation mode so dropout is disabled.
    model.eval()

    return model, tokenizer


def generate_one_claim(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    #Convert the conditioning prompt into model input tensors.
    inputs = tokenizer(prompt, return_tensors="pt")

    #Move tensors to the same device as the model.
    inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}

    #Generate a continuation after the prompt without tracking gradients.
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    #Slice away the prompt tokens so we keep only the generated claim text.
    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]

    #Decode token ids back into readable text.
    claim = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return " ".join(claim.strip().split())


def main() -> None:
    #read command-line options.
    args = parse_args()

    #fix random seed for reproducible sampling.
    set_seed(args.seed)

    #load the trained adapter on top of the base model.
    model, tokenizer = load_model_and_tokenizer(args.base_model_name, args.adapter_dir)

    #load real LIAR2 prompts to condition generation.
    conditioning_dataset = LIAR2Dataset(
        split=args.conditioning_split,
        task_type="generation",
        label_scheme="binary",
        include_metadata=True,
    )

    #open the output CSV file.
    with open(args.output_csv, "w", newline="", encoding="utf-8") as csv_file:
        #define the columns we want to save.
        fieldnames = ["synthetic_claim", "label", "speaker", "subject", "context", "conditioning_prompt"]

        #create a DictWriter so each row is named by column.
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        #write the header row once.
        writer.writeheader()

        #generate the requested number of synthetic claims.
        for index in range(args.num_claims):
            #cycle through real prompts if num_claims is larger than the conditioning split.
            sample = conditioning_dataset[index % len(conditioning_dataset)]

            #pull out the prompt that tells the model what kind of claim to generate.
            prompt = sample["conditioning_prompt"]

            #Generate one synthetic claim from that prompt.
            synthetic_claim = generate_one_claim(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            writer.writerow(
                {
                    "synthetic_claim": synthetic_claim,
                    "label": int(sample["label"]),
                    "speaker": sample["speaker"],
                    "subject": sample["subject"],
                    "context": sample["context"],
                    "conditioning_prompt": prompt,
                }
            )

            print(f"[{index + 1}/{args.num_claims}] {synthetic_claim}")

    print(f"\nSaved synthetic claims to: {args.output_csv}")


if __name__ == "__main__":
    main()
