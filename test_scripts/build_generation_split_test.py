from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from Experiments.train_lfm25_lora_generator import build_generation_split
from collections import Counter

dataset = build_generation_split("validation")

label_counts = Counter(dataset["label"])

print("Labels from build_generation_split:")
print(label_counts)