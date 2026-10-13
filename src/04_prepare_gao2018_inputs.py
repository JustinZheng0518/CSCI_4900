"""
Utility: Prepare Gao et al. (2018) CLS Input Representations

Purpose:
    Prepare and cache the fixed input representations needed for the
    Gao et al. (2018) VUA verb-classification model without ELMo.

Method:
    1. Read VUA train, validation, and test splits.
    2. Build the vocabulary using whitespace tokenization, following
       Gao et al.'s original preprocessing.
    3. Load only the GloVe 840B 300d vectors needed by the VUA vocabulary.
    4. Randomly initialize OOV embeddings using the mean and standard
       deviation of the loaded pretrained GloVe vectors.
    5. Create two 50-dimensional target-indicator embeddings:
         0 = non-target token
         1 = target verb
    6. Cache all processed objects so later model training does not need
       to scan the 5.26 GB GloVe file again.

Outputs:
    data/processed/gao2018/
        vocab.json
        glove_matrix.npy
        target_indicator.npy

Reproduction Notes:
    - Gao's original implementation builds vocabulary from
      train + validation + test.
    - Gao uses sentence.split() for the formatted VUA data.
    - Gao randomly initializes missing GloVe words from a distribution
      matching the mean/std of the available pretrained embeddings.
    - Gao does not specify a random seed. We use seed=42 so our
      preprocessing is reproducible.
    - Vocabulary words are sorted here to make word IDs deterministic.
      Gao's original code iterates over a Python set, so its IDs were
      not guaranteed to be deterministic.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# Configuration
# ============================================================

SEED = 42

TRAIN_PATH = "data/VUA/VUA_formatted_train.csv"
VAL_PATH = "data/VUA/VUA_formatted_val.csv"
TEST_PATH = "data/VUA/VUA_formatted_test.csv"

# Change "Glove" to "glove" here if your folder is lowercase.
GLOVE_PATH = "embeddings/Glove/glove.840B.300d.txt"

OUTPUT_DIR = Path("data/processed/gao2018")

CSV_ENCODING = "latin1"

GLOVE_DIM = 300
TARGET_INDICATOR_DIM = 50


# ============================================================
# Reproducibility
# ============================================================

np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Helper functions
# ============================================================

def load_split(path):
    """Load one formatted VUA CSV split."""
    return pd.read_csv(path, encoding=CSV_ENCODING)


def build_vocab(*dataframes):
    """
    Build vocabulary using whitespace tokenization.

    Gao's original implementation uses sentence.split().
    """
    vocab = set()

    for df in dataframes:
        for sentence in df["sentence"]:
            vocab.update(sentence.split())

    return vocab


# ============================================================
# 1. Load VUA data
# ============================================================

print("===== Loading VUA =====")

train = load_split(TRAIN_PATH)
val = load_split(VAL_PATH)
test = load_split(TEST_PATH)

print(f"Train:      {len(train)}")
print(f"Validation: {len(val)}")
print(f"Test:       {len(test)}")


# ============================================================
# 2. Build vocabulary
# ============================================================

print("\n===== Building Vocabulary =====")

# Gao uses train + val + test when constructing the vocabulary.
vocab = build_vocab(train, val, test)

# Reserve:
#   0 = PAD
#   1 = UNK
word2idx = {
    "<PAD>": 0,
    "<UNK>": 1,
}

# Sorting is our reproducibility improvement.
for word in sorted(vocab):
    word2idx[word] = len(word2idx)

print(f"Raw VUA vocabulary: {len(vocab)}")
print(f"Vocabulary with PAD/UNK: {len(word2idx)}")


# ============================================================
# 3. Scan GloVe
# ============================================================

print("\n===== Scanning GloVe =====")
print("This may take a while because the source file is ~5.26 GB.")

glove_vectors = {}

with open(GLOVE_PATH, "r", encoding="utf-8") as glove_file:
    for line in glove_file:
        parts = line.rstrip().split()

        # Valid GloVe line:
        # word + 300 floating-point values
        if len(parts) != GLOVE_DIM + 1:
            continue

        word = parts[0]

        # Only keep vectors actually needed by VUA.
        if word not in word2idx:
            continue

        vector = np.asarray(
            parts[1:],
            dtype=np.float32
        )

        glove_vectors[word] = vector


print(f"Pretrained vectors loaded: {len(glove_vectors)}")


# ============================================================
# 4. Calculate GloVe statistics
# ============================================================

print("\n===== GloVe Statistics =====")

if len(glove_vectors) == 0:
    raise RuntimeError(
        "No GloVe vectors were loaded. Check GLOVE_PATH."
    )

all_pretrained_vectors = np.stack(
    list(glove_vectors.values())
)

embedding_mean = float(all_pretrained_vectors.mean())
embedding_std = float(all_pretrained_vectors.std())

print(f"Mean: {embedding_mean:.6f}")
print(f"Std:  {embedding_std:.6f}")


# ============================================================
# 5. Build complete GloVe embedding matrix
# ============================================================

print("\n===== Building Embedding Matrix =====")

vocab_size = len(word2idx)

# Gao initializes the whole matrix from a normal distribution
# matching the pretrained GloVe distribution.
embedding_matrix = torch.empty(
    vocab_size,
    GLOVE_DIM
)

embedding_matrix.normal_(
    mean=embedding_mean,
    std=embedding_std
)

# Replace random vectors with pretrained GloVe vectors
# wherever one exists.
for word, vector in glove_vectors.items():
    word_index = word2idx[word]

    embedding_matrix[word_index] = torch.from_numpy(vector)


# Convert to NumPy for caching.
embedding_matrix = (
    embedding_matrix
    .cpu()
    .numpy()
    .astype(np.float32)
)

print("GloVe matrix shape:", embedding_matrix.shape)

assert embedding_matrix.shape == (
    vocab_size,
    GLOVE_DIM
)


# ============================================================
# 6. Calculate OOV statistics
# ============================================================

# Ignore PAD and UNK when calculating lexical coverage.
vocab_words = set(vocab)

found_words = vocab_words.intersection(
    glove_vectors.keys()
)

oov_words = vocab_words - found_words

coverage = len(found_words) / len(vocab_words)
oov_rate = len(oov_words) / len(vocab_words)

print("\n===== Vocabulary Coverage =====")
print(f"Found in GloVe: {len(found_words)}")
print(f"OOV words:      {len(oov_words)}")
print(f"Coverage:       {coverage:.2%}")
print(f"OOV rate:       {oov_rate:.2%}")


# ============================================================
# 7. Create target-indicator embeddings
# ============================================================

print("\n===== Target Indicator =====")

# Gao's original code:
# nn.Embedding(2, 50)
#
# index 0 = ordinary/non-target token
# index 1 = target verb
target_indicator = nn.Embedding(
    2,
    TARGET_INDICATOR_DIM
)

target_indicator_matrix = (
    target_indicator
    .weight
    .detach()
    .cpu()
    .numpy()
    .astype(np.float32)
)

print(
    "Target indicator shape:",
    target_indicator_matrix.shape
)

assert target_indicator_matrix.shape == (
    2,
    TARGET_INDICATOR_DIM
)


# ============================================================
# 8. Save cached preprocessing artifacts
# ============================================================

print("\n===== Saving Cache =====")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# word -> integer ID
with open(
    OUTPUT_DIR / "vocab.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        word2idx,
        f,
        ensure_ascii=False,
        indent=2
    )

# integer ID -> 300d GloVe vector
np.save(
    OUTPUT_DIR / "glove_matrix.npy",
    embedding_matrix
)

# target/non-target -> 50d indicator vector
np.save(
    OUTPUT_DIR / "target_indicator.npy",
    target_indicator_matrix
)


# ============================================================
# 9. Final sanity checks
# ============================================================

print("\n===== Sanity Check =====")

for example_word in ["turn", "attack", "fail"]:

    if example_word not in word2idx:
        print(f"{example_word}: not in VUA vocabulary")
        continue

    idx = word2idx[example_word]
    vector = embedding_matrix[idx]

    print(
        f"{example_word:>10} "
        f"-> ID {idx:>5} "
        f"-> shape {vector.shape}"
    )


print("\n===== Done =====")

print("Saved:")
print(f"  {OUTPUT_DIR / 'vocab.json'}")
print(f"  {OUTPUT_DIR / 'glove_matrix.npy'}")
print(f"  {OUTPUT_DIR / 'target_indicator.npy'}")