from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from dataset.dataset import CREDIT_VECTOR_COLUMNS, LIAR2Dataset

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train or sample a conditional text VAE for LIAR2-style fake claim generation. "
            "Training uses every fake claim without claim-length filtering or truncation."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train the conditional VAE on LIAR2 fake claims.")
    train.add_argument("--output-dir", default="Results/vae/conditional_text_vae")
    train.add_argument("--train-split", default="train", choices=["train", "validation", "test"])
    train.add_argument("--eval-split", default="validation", choices=["train", "validation", "test"])
    train.add_argument("--min-token-frequency", type=int, default=1)
    train.add_argument("--max-vocab-size", type=int, default=30000)
    train.add_argument("--embedding-dim", type=int, default=192)
    train.add_argument("--hidden-dim", type=int, default=256)
    train.add_argument("--latent-dim", type=int, default=96)
    train.add_argument("--condition-dim", type=int, default=128)
    train.add_argument("--dropout", type=float, default=0.2)
    train.add_argument("--epochs", type=int, default=25)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--beta", type=float, default=0.15)
    train.add_argument("--kl-warmup-epochs", type=int, default=8)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--early-stopping-patience", type=int, default=10)
    train.add_argument(
        "--early-stopping-min-epochs",
        type=int,
        default=None,
        help="Do not early-stop before this epoch. Defaults to --kl-warmup-epochs.",
    )
    train.add_argument("--seed", type=int, default=42)

    generate = subparsers.add_parser("generate", help="Sample synthetic fake claims from a trained VAE.")
    generate.add_argument("--checkpoint", default="Results/vae/conditional_text_vae/best_model.pt")
    generate.add_argument("--output-csv", default="Results/synthetic/raw_vae_2000_claims.csv")
    generate.add_argument("--conditioning-split", default="train", choices=["train", "validation", "test"])
    generate.add_argument("--num-claims", type=int, default=2000)
    generate.add_argument("--max-new-tokens", type=int, default=90)
    generate.add_argument("--temperature", type=float, default=0.9)
    generate.add_argument("--top-k", type=int, default=50)
    generate.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def claim_tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\w\s]", str(text or "").lower())


def metadata_text(sample: dict[str, Any]) -> str:
    parts = []
    for field_name in ["speaker", "subject", "context"]:
        value = sample.get(field_name)
        if value:
            parts.append(f"{field_name}: {value}")
    return " | ".join(parts)


def detokenize(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?%)])", r"\1", text)
    text = re.sub(r"([(])\s+", r"\1", text)
    text = re.sub(r"\s+'\s+", "'", text)
    text = re.sub(r'\s+(")', r"\1", text)
    text = re.sub(r'(")\s+', r"\1", text)
    return text.strip()


@dataclass
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[BOS]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK]

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids = [self.token_to_id.get(token, self.unk_id) for token in claim_tokenize(text)]
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        tokens = []
        for token_id in ids:
            if token_id == self.eos_id:
                break
            if token_id in {self.pad_id, self.bos_id}:
                continue
            token = self.id_to_token[token_id]
            if token != UNK:
                tokens.append(token)
        return detokenize(tokens)


def build_vocab(samples: list[dict[str, Any]], min_frequency: int, max_vocab_size: int) -> Vocab:
    counts: Counter[str] = Counter()
    for sample in samples:
        counts.update(claim_tokenize(sample["target_text"]))
        counts.update(claim_tokenize(metadata_text(sample)))

    kept_tokens = [
        token
        for token, count in counts.most_common(max(0, max_vocab_size - len(SPECIAL_TOKENS)))
        if count >= min_frequency
    ]
    id_to_token = SPECIAL_TOKENS + kept_tokens
    token_to_id = {token: index for index, token in enumerate(id_to_token)}
    return Vocab(token_to_id=token_to_id, id_to_token=id_to_token)


def load_fake_generation_samples(split: str) -> list[dict[str, Any]]:
    dataset = LIAR2Dataset(
        split=split,
        task_type="generation",
        label_scheme="binary",
        include_metadata=True,
    )
    return [dataset[index] for index in range(len(dataset)) if int(dataset[index]["label"]) == 0]


class FakeClaimVAEDataset(Dataset):
    def __init__(self, samples: list[dict[str, Any]], vocab: Vocab):
        self.samples = samples
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        claim_ids = self.vocab.encode(sample["target_text"], add_eos=True)
        condition_ids = self.vocab.encode(metadata_text(sample), add_eos=True)

        if not claim_ids:
            claim_ids = [self.vocab.eos_id]
        if not condition_ids:
            condition_ids = [self.vocab.eos_id]

        return {
            "claim_ids": claim_ids,
            "condition_ids": condition_ids,
            "sample": sample,
        }


def pad_sequences(sequences: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    max_length = int(lengths.max().item())
    padded = torch.full((len(sequences), max_length), pad_id, dtype=torch.long)

    for row_index, sequence in enumerate(sequences):
        padded[row_index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)

    return padded, lengths


class VAECollator:
    def __init__(self, vocab: Vocab):
        self.vocab = vocab

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        claim_ids = [item["claim_ids"] for item in batch]
        condition_ids = [item["condition_ids"] for item in batch]

        target_ids, target_lengths = pad_sequences(claim_ids, self.vocab.pad_id)
        decoder_inputs = torch.cat(
            [
                torch.full((target_ids.size(0), 1), self.vocab.bos_id, dtype=torch.long),
                target_ids[:, :-1],
            ],
            dim=1,
        )
        condition_tensor, condition_lengths = pad_sequences(condition_ids, self.vocab.pad_id)

        return {
            "target_ids": target_ids,
            "target_lengths": target_lengths,
            "decoder_inputs": decoder_inputs,
            "condition_ids": condition_tensor,
            "condition_lengths": condition_lengths,
            "samples": [item["sample"] for item in batch],
        }


class ConditionalTextVAE(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        embedding_dim: int,
        hidden_dim: int,
        latent_dim: int,
        condition_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.encoder = nn.GRU(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.condition_projection = nn.Sequential(
            nn.Linear(embedding_dim, condition_dim),
            nn.Tanh(),
        )
        encoder_out_dim = hidden_dim * 2 + condition_dim
        self.to_mu = nn.Linear(encoder_out_dim, latent_dim)
        self.to_logvar = nn.Linear(encoder_out_dim, latent_dim)

        decoder_input_dim = embedding_dim + latent_dim + condition_dim
        self.decoder = nn.GRU(decoder_input_dim, hidden_dim, batch_first=True)
        self.decoder_initial = nn.Linear(latent_dim + condition_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def encode_condition(self, condition_ids: torch.Tensor) -> torch.Tensor:
        mask = (condition_ids != self.pad_id).unsqueeze(-1)
        embedded = self.embedding(condition_ids)
        summed = (embedded * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp_min(1)
        mean_embedding = summed / lengths
        return self.condition_projection(mean_embedding)

    def encode_claim(
        self,
        target_ids: torch.Tensor,
        target_lengths: torch.Tensor,
        condition_vector: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(target_ids))
        packed = pack_padded_sequence(
            embedded,
            target_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.encoder(packed)
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        encoded = torch.cat([forward_hidden, backward_hidden, condition_vector], dim=1)
        return self.to_mu(encoded), self.to_logvar(encoded)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)
        return mu + noise * std

    def decode(
        self,
        decoder_inputs: torch.Tensor,
        z: torch.Tensor,
        condition_vector: torch.Tensor,
    ) -> torch.Tensor:
        embedded = self.dropout(self.embedding(decoder_inputs))
        repeated_z = z.unsqueeze(1).expand(-1, decoder_inputs.size(1), -1)
        repeated_condition = condition_vector.unsqueeze(1).expand(-1, decoder_inputs.size(1), -1)
        decoder_input = torch.cat([embedded, repeated_z, repeated_condition], dim=2)
        initial_hidden = torch.tanh(self.decoder_initial(torch.cat([z, condition_vector], dim=1))).unsqueeze(0)
        decoded, _ = self.decoder(decoder_input, initial_hidden)
        return self.output_projection(decoded)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        condition_vector = self.encode_condition(batch["condition_ids"])
        mu, logvar = self.encode_claim(batch["target_ids"], batch["target_lengths"], condition_vector)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(batch["decoder_inputs"], z, condition_vector)
        return {"logits": logits, "mu": mu, "logvar": logvar}

    @torch.no_grad()
    def sample(
        self,
        condition_ids: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> list[int]:
        self.eval()
        condition_vector = self.encode_condition(condition_ids)
        z = torch.randn(1, self.latent_dim, device=condition_ids.device)
        hidden = torch.tanh(self.decoder_initial(torch.cat([z, condition_vector], dim=1))).unsqueeze(0)
        current = torch.tensor([[bos_id]], dtype=torch.long, device=condition_ids.device)
        generated: list[int] = []

        for _ in range(max_new_tokens):
            embedded = self.embedding(current)
            decoder_input = torch.cat([embedded, z.unsqueeze(1), condition_vector.unsqueeze(1)], dim=2)
            decoded, hidden = self.decoder(decoder_input, hidden)
            logits = self.output_projection(decoded[:, -1, :]) / max(temperature, 1e-6)

            if top_k > 0 and top_k < logits.size(-1):
                values, indices = torch.topk(logits, top_k, dim=-1)
                filtered = torch.full_like(logits, float("-inf"))
                filtered.scatter_(1, indices, values)
                logits = filtered

            probabilities = F.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probabilities, num_samples=1).item())
            if next_id == eos_id:
                break
            generated.append(next_id)
            current = torch.tensor([[next_id]], dtype=torch.long, device=condition_ids.device)

        return generated


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def vae_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    pad_id: int,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reconstruction = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_ids.reshape(-1),
        ignore_index=pad_id,
        reduction="mean",
    )
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction + beta * kl, reconstruction, kl


def run_epoch(
    model: ConditionalTextVAE,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    beta: float,
    grad_clip: float,
    pad_id: int,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_reconstruction = 0.0
    total_kl = 0.0
    total_batches = 0

    for batch in tqdm(dataloader, leave=False):
        batch = move_batch_to_device(batch, device)

        with torch.set_grad_enabled(is_train):
            output = model(batch)
            loss, reconstruction, kl = vae_loss(
                output["logits"],
                batch["target_ids"],
                output["mu"],
                output["logvar"],
                pad_id=pad_id,
                beta=beta,
            )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        total_loss += float(loss.item())
        total_reconstruction += float(reconstruction.item())
        total_kl += float(kl.item())
        total_batches += 1

    return {
        "loss": total_loss / max(total_batches, 1),
        "reconstruction_loss": total_reconstruction / max(total_batches, 1),
        "kl_loss": total_kl / max(total_batches, 1),
    }


def plot_training_history(history: list[dict[str, Any]], output_dir: Path) -> None:
    epochs = [record["epoch"] for record in history]
    train_losses = [record["train"]["loss"] for record in history]
    eval_losses = [record["eval"]["loss"] for record in history]
    train_reconstruction = [record["train"]["reconstruction_loss"] for record in history]
    eval_reconstruction = [record["eval"]["reconstruction_loss"] for record in history]
    train_kl = [record["train"]["kl_loss"] for record in history]
    eval_kl = [record["eval"]["kl_loss"] for record in history]
    betas = [record["beta"] for record in history]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="train total loss")
    plt.plot(epochs, eval_losses, label="validation total loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Conditional Text VAE Total Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "vae_total_loss.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_reconstruction, label="train reconstruction")
    plt.plot(epochs, eval_reconstruction, label="validation reconstruction")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy")
    plt.title("Conditional Text VAE Reconstruction Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "vae_reconstruction_loss.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_kl, label="train KL")
    plt.plot(epochs, eval_kl, label="validation KL")
    plt.plot(epochs, betas, label="beta", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Conditional Text VAE KL Loss and Beta Warmup")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "vae_kl_beta.png", dpi=200)
    plt.close()


def train_vae(args: argparse.Namespace) -> None:
    set_random_seed(args.seed)
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_samples = load_fake_generation_samples(args.train_split)
    eval_samples = load_fake_generation_samples(args.eval_split)
    if not train_samples:
        raise ValueError(f"No fake training samples found in split: {args.train_split}")
    if not eval_samples:
        raise ValueError(f"No fake evaluation samples found in split: {args.eval_split}")

    vocab = build_vocab(
        train_samples,
        min_frequency=args.min_token_frequency,
        max_vocab_size=args.max_vocab_size,
    )
    collator = VAECollator(vocab)
    train_loader = DataLoader(
        FakeClaimVAEDataset(train_samples, vocab),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    eval_loader = DataLoader(
        FakeClaimVAEDataset(eval_samples, vocab),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalTextVAE(
        vocab_size=len(vocab.id_to_token),
        pad_id=vocab.pad_id,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        condition_dim=args.condition_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_eval_loss = math.inf
    best_eval_reconstruction = math.inf
    epochs_without_improvement = 0
    history = []

    config = {
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "condition_dim": args.condition_dim,
        "dropout": args.dropout,
        "pad_id": vocab.pad_id,
    }

    for epoch in range(1, args.epochs + 1):
        beta = args.beta * min(1.0, epoch / max(args.kl_warmup_epochs, 1))
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            beta=beta,
            grad_clip=args.grad_clip,
            pad_id=vocab.pad_id,
        )
        eval_metrics = run_epoch(
            model,
            eval_loader,
            device,
            optimizer=None,
            beta=beta,
            grad_clip=args.grad_clip,
            pad_id=vocab.pad_id,
        )

        record = {
            "epoch": epoch,
            "beta": beta,
            "train": train_metrics,
            "eval": eval_metrics,
        }
        history.append(record)
        print(json.dumps(record, indent=2))

        if eval_metrics["loss"] < best_eval_loss:
            best_eval_loss = eval_metrics["loss"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": asdict(vocab),
                    "config": config,
                    "train_args": vars(args),
                },
                output_dir / "best_model.pt",
            )
        else:
            epochs_without_improvement += 1

        if eval_metrics["reconstruction_loss"] < best_eval_reconstruction:
            best_eval_reconstruction = eval_metrics["reconstruction_loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": asdict(vocab),
                    "config": config,
                    "train_args": vars(args),
                },
                output_dir / "best_reconstruction_model.pt",
            )

        early_stopping_min_epochs = args.early_stopping_min_epochs or args.kl_warmup_epochs
        can_stop = epoch >= early_stopping_min_epochs
        if (
            can_stop
            and args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                "Early stopping triggered after "
                f"{epochs_without_improvement} epochs without validation loss improvement."
            )
            break

    with (output_dir / "training_history.json").open("w", encoding="utf-8") as history_file:
        json.dump(history, history_file, indent=2)
    plot_training_history(history, output_dir)

    print(f"Saved best checkpoint to: {output_dir / 'best_model.pt'}")


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[ConditionalTextVAE, Vocab]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vocab_data = checkpoint["vocab"]
    vocab = Vocab(token_to_id=vocab_data["token_to_id"], id_to_token=vocab_data["id_to_token"])
    config = checkpoint["config"]
    model = ConditionalTextVAE(
        vocab_size=len(vocab.id_to_token),
        pad_id=vocab.pad_id,
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        latent_dim=config["latent_dim"],
        condition_dim=config["condition_dim"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, vocab


def build_classifier_text(claim: str, sample: dict[str, Any]) -> str:
    metadata_parts = []
    for field_name in ["speaker", "subject", "context"]:
        value = sample.get(field_name)
        if value:
            metadata_parts.append(f"{field_name}: {value}")
    if metadata_parts:
        return f"{claim}\n\n[METADATA] {' | '.join(metadata_parts)}"
    return claim


def generate_claims(args: argparse.Namespace) -> None:
    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = PROJECT_ROOT / args.checkpoint
    model, vocab = load_checkpoint(checkpoint_path, device)

    fake_conditioning_samples = load_fake_generation_samples(args.conditioning_split)
    if not fake_conditioning_samples:
        raise ValueError(f"No fake conditioning samples found in split: {args.conditioning_split}")

    output_csv = PROJECT_ROOT / args.output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "synthetic_claim",
        "text",
        "label",
        "speaker",
        "subject",
        "context",
        "conditioning_metadata",
        *CREDIT_VECTOR_COLUMNS,
    ]

    with output_csv.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        for index in range(args.num_claims):
            sample = fake_conditioning_samples[index % len(fake_conditioning_samples)]
            condition_ids = vocab.encode(metadata_text(sample), add_eos=True)
            if not condition_ids:
                condition_ids = [vocab.eos_id]
            condition_tensor = torch.tensor([condition_ids], dtype=torch.long, device=device)
            generated_ids = model.sample(
                condition_tensor,
                bos_id=vocab.bos_id,
                eos_id=vocab.eos_id,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            synthetic_claim = vocab.decode(generated_ids)
            credit_vector = sample["credit_vector"].tolist()

            row = {
                "synthetic_claim": synthetic_claim,
                "text": build_classifier_text(synthetic_claim, sample),
                "label": 0,
                "speaker": sample["speaker"],
                "subject": sample["subject"],
                "context": sample["context"],
                "conditioning_metadata": metadata_text(sample),
            }
            for column, value in zip(CREDIT_VECTOR_COLUMNS, credit_vector):
                row[column] = value

            writer.writerow(row)
            print(f"[{index + 1}/{args.num_claims}] {synthetic_claim}")

    print(f"Saved VAE synthetic claims to: {output_csv}")


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train_vae(args)
    elif args.command == "generate":
        generate_claims(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
