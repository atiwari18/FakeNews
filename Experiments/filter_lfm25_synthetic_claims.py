from __future__ import annotations
import argparse
import csv
import json
import re
from pathlib import Path
from difflib import SequenceMatcher

CREDIT_VECTOR_COLUMNS = [
    "true_counts",
    "mostly_true_counts",
    "half_true_counts",
    "mostly_false_counts",
    "false_counts",
    "pants_on_fire_counts",
]

def parse_args():
    parser = argparse.ArgumentParser(description="Quality-filter LFM2.5 synthetic claims before classifier evaluation.")
    parser.add_argument("--input-csv", default="Results/synthetic/raw_lfm25_2000_claims.csv")
    parser.add_argument("--output-csv",default="Results/synthetic/filtered_lfm25_1000_claims.csv",)
    parser.add_argument("--report-json", default="Results/synthetic/filtered_lfm25_1000_report.json",)
    parser.add_argument("--target-size", type=int, default=1000)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.92)
    return parser.parse_args()

def normalize_whitespace(text: str) -> str:
    return " ".join(str(text or "").split())

def strip_prompt_artifacts(text: str) -> str:
    claim = text

    for marker in ["[CONDITION]", "[CLAIM]", "[METADATA]"]:
        if marker in claim:
            claim = claim.split(marker)[0]

    claim = claim.replace("\r", " ").replace("\n", " ")
    return normalize_whitespace(claim)

def clean_claim(raw_claim: str) -> str:
    claim = strip_prompt_artifacts(raw_claim)

    # Remove CSV/viewer-style wrapping quotes if they survived parsing.
    claim = claim.strip()
    claim = claim.strip("`")
    claim = claim.strip()

    # Normalize repeated punctuation spacing.
    claim = re.sub(r"\s+([,.;:!?])", r"\1", claim)
    claim = re.sub(r"([,.;:!?])([A-Za-z])", r"\1 \2", claim)

    # Remove dangling hash fragments sometimes produced by generation.
    claim = re.sub(r"\s+#\s*$", "", claim)

    # If the model produced a very obvious unfinished parenthetical, drop it.
    if claim.count("(") > claim.count(")"):
        last_open = claim.rfind("(")
        if last_open > len(claim) * 0.65:
            claim = claim[:last_open].strip()

    # If the model produced one unmatched double quote, remove double quotes.
    # This is conservative: it avoids rejecting many otherwise usable claims.
    if claim.count('"') % 2 == 1:
        claim = claim.replace('"', "")

    # Same for single quotes only when there is exactly one.
    if claim.count("'") == 1:
        claim = claim.replace("'", "")

    return normalize_whitespace(claim)

def token_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())

def has_repeated_ngram(text: str, n: int = 4) -> bool:
    tokens = token_words(text)
    if len(tokens) < n * 2:
        return False

    grams = [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
    return len(grams) != len(set(grams))

def max_repeated_content_word_count(text: str) -> int:
    stopwords = {
        "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for",
        "with", "is", "are", "was", "were", "be", "been", "being", "that",
        "this", "it", "he", "she", "they", "we", "you", "i", "his", "her",
        "their", "our", "your", "as", "by", "from", "at", "about",
    }

    counts: dict[str, int] = {}
    for word in token_words(text):
        if word in stopwords or len(word) <= 2:
            continue
        counts[word] = counts.get(word, 0) + 1

    return max(counts.values(), default=0)

def ends_cleanly(text: str) -> bool:
    return bool(re.search(r'[.!?]"?$', text.strip()))

def quality_score(claim: str) -> tuple[int, list[str]]:
    reasons: list[str] = []

    score = 100

    if not claim:
        return 0, ["empty"]

    if any(marker in claim for marker in ["[CONDITION]", "[CLAIM]", "[METADATA]"]):
        score -= 80
        reasons.append("prompt_artifact")

    if has_repeated_ngram(claim, n=4):
        score -= 45
        reasons.append("repeated_4gram")

    if has_repeated_ngram(claim, n=3):
        score -= 25
        reasons.append("repeated_3gram")

    max_repeat = max_repeated_content_word_count(claim)
    if max_repeat >= 5:
        score -= 25
        reasons.append("repeated_content_word")

    if not ends_cleanly(claim):
        score -= 20
        reasons.append("unfinished")

    if claim.count('"') % 2 == 1:
        score -= 20
        reasons.append("unbalanced_double_quote")

    if claim.count("(") != claim.count(")"):
        score -= 10
        reasons.append("unbalanced_parentheses")

    if "..." in claim or "…" in claim:
        score -= 8
        reasons.append("ellipsis")

    return max(score, 0), reasons


def canonical_for_dedup(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return normalize_whitespace(text)


def is_near_duplicate(candidate: str, accepted_canonicals: list[str], threshold: float) -> bool:
    candidate_canonical = canonical_for_dedup(candidate)

    for existing in accepted_canonicals:
        if candidate_canonical == existing:
            return True

        ratio = SequenceMatcher(None, candidate_canonical, existing).ratio()
        if ratio >= threshold:
            return True

    return False


def build_classifier_text(claim: str, row: dict[str, str]) -> str:
    metadata_parts = []

    for field_name in ["speaker", "subject", "context"]:
        value = normalize_whitespace(row.get(field_name, ""))
        if value:
            metadata_parts.append(f"{field_name}: {value}")

    if metadata_parts:
        return f"{claim}\n\n[METADATA] {' | '.join(metadata_parts)}"

    return claim

def main() -> None:
    args = parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    report_path = Path(args.report_json)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))

    scored_rows = []
    reject_counts: dict[str, int] = {}

    for row_index, row in enumerate(rows):
        raw_claim = row.get("synthetic_claim", "")
        cleaned_claim = clean_claim(raw_claim)
        score, reasons = quality_score(cleaned_claim)

        for reason in reasons:
            reject_counts[reason] = reject_counts.get(reason, 0) + 1

        new_row = dict(row)
        new_row["synthetic_claim"] = cleaned_claim
        new_row["text"] = build_classifier_text(cleaned_claim, row)
        new_row["label"] = "0"

        scored_rows.append(
            {
                "row_index": row_index,
                "score": score,
                "reasons": reasons,
                "row": new_row,
            }
        )

    scored_rows.sort(
        key=lambda item: (
            item["score"],
        ),
        reverse=True,
    )

    accepted = []
    accepted_canonicals = []
    duplicate_count = 0

    for item in scored_rows:
        claim = item["row"]["synthetic_claim"]

        if item["score"] <= 0:
            continue

        if is_near_duplicate(
            claim,
            accepted_canonicals,
            threshold=args.near_duplicate_threshold,
        ):
            duplicate_count += 1
            continue

        accepted.append(item)
        accepted_canonicals.append(canonical_for_dedup(claim))

        if len(accepted) >= args.target_size:
            break

    if not accepted:
        raise ValueError("No usable synthetic claims were accepted after filtering.")

    fieldnames = list(rows[0].keys())

    # Keep compatibility if the input did not have these columns.
    for required_column in ["synthetic_claim", "text", "label"]:
        if required_column not in fieldnames:
            fieldnames.insert(0, required_column)

    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        for item in accepted:
            writer.writerow({field: item["row"].get(field, "") for field in fieldnames})

    report = {
        "input_csv": str(input_path),
        "output_csv": str(output_path),
        "target_size": args.target_size,
        "raw_rows": len(rows),
        "accepted_rows": len(accepted),
        "duplicate_or_near_duplicate_rows_skipped": duplicate_count,
        "quality_reason_counts_before_selection": reject_counts,
        "min_score_accepted": min(item["score"] for item in accepted),
        "max_score_accepted": max(item["score"] for item in accepted),
        "mean_score_accepted": sum(item["score"] for item in accepted) / len(accepted),
        "accepted_examples": [
            {
                "score": item["score"],
                "reasons": item["reasons"],
                "synthetic_claim": item["row"]["synthetic_claim"],
            }
            for item in accepted[:10]
        ],
    }

    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)

    print(f"Raw rows: {len(rows)}")
    print(f"Accepted rows: {len(accepted)}")
    print(f"Skipped near duplicates: {duplicate_count}")
    print(f"Saved filtered CSV to: {output_path}")
    print(f"Saved filtering report to: {report_path}")


if __name__ == "__main__":
    main()