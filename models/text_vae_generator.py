from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


def simple_tokenize(text: str) -> list[str]:
    text = str(text or "").lower()
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?|[^\w\s]", text)


def simple_detokenize(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?%)])", r"\1", text)
    text = re.sub(r"([(])\s+", r"\1", text)
    text = text.replace(" n't", "n't")
    text = text.replace(" 's", "'s")
    text = text.replace(" 're", "'re")
    text = text.replace(" 've", "'ve")
    text = text.replace(" 'll", "'ll")
    text = text.replace(" 'd", "'d")
    return text.strip()


@dataclass
class TextVAEConfig:
    vocab_size: int
    embedding_dim: int = 256
    hidden_dim: int = 384
    latent_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.15
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3


class Vocab:
    def __init__(self, token_to_id: dict[str, int]):
        self.token_to_id = token_to_id
        self.id_to_token = {idx: token for token, idx in token_to_id.items()}

    @classmethod
    def build(cls, texts: list[str], max_vocab_size: int, min_freq: int) -> "Vocab":
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(simple_tokenize(text))

        token_to_id = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
        for token, count in counts.most_common(max_vocab_size - len(SPECIAL_TOKENS)):
            if count < min_freq:
                continue
            if token not in token_to_id:
                token_to_id[token] = len(token_to_id)

        return cls(token_to_id)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Vocab":
        return cls({str(token): int(idx) for token, idx in payload["token_to_id"].items()})

    def to_dict(self) -> dict[str, Any]:
        return {"token_to_id": self.token_to_id}

    def __len__(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str, max_length: int, add_bos_eos: bool = True) -> list[int]:
        ids = [self.token_to_id.get(token, self.token_to_id[UNK_TOKEN]) for token in simple_tokenize(text)]
        if add_bos_eos:
            ids = [self.token_to_id[BOS_TOKEN]] + ids + [self.token_to_id[EOS_TOKEN]]
        return ids[:max_length]

    def decode(self, ids: list[int]) -> str:
        tokens = []
        for idx in ids:
            token = self.id_to_token.get(int(idx), UNK_TOKEN)
            if token in SPECIAL_TOKENS:
                continue
            tokens.append(token)
        return simple_detokenize(tokens)


def sequence_mask(token_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    return token_ids.ne(pad_token_id).float()


def masked_mean(embeddings: torch.Tensor, token_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    mask = sequence_mask(token_ids, pad_token_id).unsqueeze(-1)
    summed = (embeddings * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return summed / counts


class ConditionalTextVAE(nn.Module):
    def __init__(self, config: TextVAEConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=config.pad_token_id)
        self.encoder = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(config.hidden_dim, config.latent_dim)
        self.logvar = nn.Linear(config.hidden_dim, config.latent_dim)
        self.latent_to_hidden = nn.Linear(config.latent_dim + config.embedding_dim, config.hidden_dim * config.num_layers)
        self.decoder = nn.GRU(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(config.hidden_dim, config.vocab_size)

    def encode(self, claim_ids: torch.Tensor, condition_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        claim_emb = self.embedding(claim_ids)
        condition_emb = self.embedding(condition_ids)
        claim_mean = masked_mean(claim_emb, claim_ids, self.config.pad_token_id)
        condition_mean = masked_mean(condition_emb, condition_ids, self.config.pad_token_id)
        encoded = self.encoder(torch.cat([claim_mean, condition_mean], dim=1))
        return self.mu(encoded), self.logvar(encoded)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_logits(self, decoder_input_ids: torch.Tensor, condition_ids: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        condition_emb = self.embedding(condition_ids)
        condition_mean = masked_mean(condition_emb, condition_ids, self.config.pad_token_id)
        hidden = self.latent_to_hidden(torch.cat([z, condition_mean], dim=1))
        hidden = hidden.view(self.config.num_layers, decoder_input_ids.size(0), self.config.hidden_dim).contiguous()
        decoder_emb = self.embedding(decoder_input_ids)
        outputs, _ = self.decoder(decoder_emb, hidden)
        return self.output(outputs)

    def forward(
        self,
        claim_ids: torch.Tensor,
        condition_ids: torch.Tensor,
        beta: float = 0.1,
    ) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(claim_ids, condition_ids)
        z = self.reparameterize(mu, logvar)
        decoder_input_ids = claim_ids[:, :-1]
        target_ids = claim_ids[:, 1:]
        logits = self.decode_logits(decoder_input_ids, condition_ids, z)
        reconstruction_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            ignore_index=self.config.pad_token_id,
        )
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        loss = reconstruction_loss + beta * kl_loss
        return {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "mu": mu,
            "logvar": logvar,
        }

    @torch.no_grad()
    def generate(
        self,
        condition_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.9,
        top_k: int = 40,
        z: torch.Tensor | None = None,
    ) -> list[list[int]]:
        self.eval()
        device = condition_ids.device
        batch_size = condition_ids.size(0)
        if z is None:
            z = torch.randn(batch_size, self.config.latent_dim, device=device)

        generated = torch.full((batch_size, 1), self.config.bos_token_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            logits = self.decode_logits(generated, condition_ids, z)[:, -1, :]
            logits[:, self.config.pad_token_id] = -float("inf")
            logits[:, self.config.bos_token_id] = -float("inf")
            logits = logits / max(temperature, 1e-5)

            if top_k > 0 and top_k < logits.size(-1):
                top_values, _ = torch.topk(logits, top_k)
                cutoff = top_values[:, -1].unsqueeze(1)
                logits = torch.where(logits < cutoff, torch.full_like(logits, -float("inf")), logits)

            probs = torch.softmax(logits, dim=-1)
            next_ids = torch.multinomial(probs, num_samples=1)
            next_ids = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_ids, self.config.eos_token_id),
                next_ids,
            )
            generated = torch.cat([generated, next_ids], dim=1)
            finished |= next_ids.squeeze(1).eq(self.config.eos_token_id)
            if bool(finished.all()):
                break

        return generated[:, 1:].tolist()


def config_to_dict(config: TextVAEConfig) -> dict[str, Any]:
    return asdict(config)


def config_from_dict(payload: dict[str, Any]) -> TextVAEConfig:
    return TextVAEConfig(**payload)


def kl_beta_for_step(step: int, warmup_steps: int, max_beta: float) -> float:
    if warmup_steps <= 0:
        return max_beta
    return max_beta * min(1.0, step / warmup_steps)


def safe_perplexity(loss: float) -> float:
    return math.exp(loss) if loss < 20 else float("inf")
