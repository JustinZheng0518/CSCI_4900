"""
Utility: VUA Vocabulary and GloVe Coverage

Purpose:
    Build the vocabulary used by the VUA verb-classification dataset
    and measure how much of that vocabulary is covered by
    GloVe 840B 300d embeddings.

Data:
    - VUA_formatted_train.csv
    - VUA_formatted_val.csv
    - VUA_formatted_test.csv

Method:
    1. Tokenize formatted VUA sentences using whitespace split(),
       following Gao et al. (2018)'s preprocessing behavior.
    2. Build a vocabulary from train + validation + test.
    3. Scan GloVe and keep only words required by VUA.
    4. Report vocabulary coverage and OOV rate.

Research / Reproduction Goal:
    Verify the embedding coverage before constructing the input
    representations for the Gao et al. (2018) BiLSTM model.

Note:
    This is a preprocessing utility, not a model experiment.
"""

import pandas as pd


TRAIN_PATH = "data/VUA/VUA_formatted_train.csv"
VAL_PATH = "data/VUA/VUA_formatted_val.csv"
TEST_PATH = "data/VUA/VUA_formatted_test.csv"

GLOVE_PATH = "embeddings/Glove/glove.840B.300d.txt"

ENCODING = "latin1"
GLOVE_DIM = 300


def load_data(path):
    return pd.read_csv(path, encoding=ENCODING)


def build_vocab(*dataframes):
    vocab = set()

    for df in dataframes:
        for sentence in df["sentence"]:
            tokens = sentence.split()
            vocab.update(tokens)

    return vocab


train = load_data(TRAIN_PATH)
val = load_data(VAL_PATH)
test = load_data(TEST_PATH)

print("===== Dataset Sizes =====")
print("Train:", len(train))
print("Validation:", len(val))
print("Test:", len(test))

vocab = build_vocab(train, val, test)

print("\n===== Vocabulary =====")
print("Unique VUA tokens:", len(vocab))


found_words = set()

print("\nScanning GloVe...")

with open(GLOVE_PATH, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip().split()

        # 1 word + 300 numbers
        if len(parts) != GLOVE_DIM + 1:
            continue

        word = parts[0]

        if word in vocab:
            found_words.add(word)


oov_words = vocab - found_words

coverage = len(found_words) / len(vocab)
oov_rate = len(oov_words) / len(vocab)

print("\n===== GloVe Coverage =====")
print("Found in GloVe:", len(found_words))
print("OOV words:", len(oov_words))
print(f"Coverage: {coverage:.2%}")
print(f"OOV rate: {oov_rate:.2%}")

print("\n===== Example OOV Words =====")
for word in sorted(oov_words)[:20]:
    print(word)