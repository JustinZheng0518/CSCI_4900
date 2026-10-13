"""
Utility: GloVe Embedding Sanity Check

Purpose:
    Verify that the pretrained GloVe 840B 300d embedding file
    can be read correctly before using it in the Gao et al. (2018)
    BiLSTM reproduction.

Checks:
    1. Find several VUA-relevant words in GloVe.
    2. Confirm each embedding has 300 dimensions.
    3. Print the first 10 values of each vector.

Note:
    This is a utility check, not a formal experiment,
    so its output does not need to be added to results.csv.
"""

GLOVE_PATH = "embeddings/Glove/glove.840B.300d.txt"

TARGET_WORDS = {"turn", "attack", "fail"}

found = {}

with open(GLOVE_PATH, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip().split(" ")

        word = parts[0]

        if word in TARGET_WORDS:
            vector = [float(x) for x in parts[1:]]
            found[word] = vector

        # 找齐以后立即停止，不需要扫完整个 5.26 GB 文件
        if len(found) == len(TARGET_WORDS):
            break

for word in sorted(TARGET_WORDS):
    print(f"\nWord: {word}")

    if word not in found:
        print("Not found")
        continue

    vector = found[word]

    print("Dimensions:", len(vector))
    print("First 10 values:", vector[:10])