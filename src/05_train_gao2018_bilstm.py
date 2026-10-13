"""
Experiment 05: Gao et al. (2018) BiLSTM + Attention Reproduction (No ELMo)

Purpose:
    Reproduce the VUA verb-classification architecture from
    Gao et al. (2018), using GloVe embeddings and the target-verb
    indicator, but WITHOUT ELMo.

Dataset:
    VUA 2018 verb classification:
        data/VUA/VUA_formatted_train.csv
        data/VUA/VUA_formatted_val.csv
        data/VUA/VUA_formatted_test.csv

Model:
    300d GloVe
      + 50d target indicator
      -> BiLSTM (hidden size 300, bidirectional)
      -> Attention
      -> Binary classifier

Research question:
    How well can the Gao et al. BiLSTM architecture identify
    metaphorical verbs using static GloVe representations before
    contextual ELMo embeddings are introduced?

Outputs:
    results/gao2018_bilstm_no_elmo_history.csv
    results/gao2018_bilstm_no_elmo_test.csv
    checkpoints/gao2018_bilstm_no_elmo_best.pt
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Configuration
# ============================================================

CONFIG_PATH = Path("configs/bilstm_glove_vua.yaml")
CACHE_DIR = Path("data/processed/gao2018")
RESULTS_DIR = Path("results")
CHECKPOINT_DIR = Path("checkpoints")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

SEED = config["training"]["seed"]
BATCH_SIZE = config["training"]["batch_size"]
LR = config["training"]["learning_rate"]
MOMENTUM = config["training"]["momentum"]
EPOCHS = config["training"]["epochs"]

MAX_LEN = config["data"]["max_sequence_length"]

HIDDEN_SIZE = config["model"]["hidden_size"]
INPUT_DROPOUT = config["model"]["input_dropout"]
OUTPUT_DROPOUT = config["model"]["output_dropout"]

TRAIN_PATH = config["data"]["train_path"]
VAL_PATH = config["data"]["val_path"]
TEST_PATH = config["data"]["test_path"]
CSV_ENCODING = config["data"]["encoding"]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


print("===== Device =====")
print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# Load preprocessing artifacts
# ============================================================

print("\n===== Loading Cached Inputs =====")

with open(
    CACHE_DIR / "vocab.json",
    "r",
    encoding="utf-8",
) as f:
    word2idx = json.load(f)

glove_matrix = np.load(
    CACHE_DIR / "glove_matrix.npy"
)

target_indicator_matrix = np.load(
    CACHE_DIR / "target_indicator.npy"
)

print("Vocabulary size:", len(word2idx))
print("GloVe matrix:", glove_matrix.shape)
print("Target indicator:", target_indicator_matrix.shape)


# ============================================================
# Dataset helpers
# ============================================================

def resolve_column(df, candidates, fallback_index):
    """
    Find a column by name if possible.
    Otherwise fall back to Gao's original CSV column position.
    """
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    return df.columns[fallback_index]


class VUADataset(Dataset):

    def __init__(self, path):

        self.df = pd.read_csv(
            path,
            encoding=CSV_ENCODING,
        )

        # Gao original CSV:
        # column 3 = sentence
        # column 4 = target verb index
        # column 5 = label
        self.sentence_col = resolve_column(
            self.df,
            ["sentence"],
            3,
        )

        self.target_col = resolve_column(
            self.df,
            ["verb_idx", "target_idx", "verb_index"],
            4,
        )

        self.label_col = resolve_column(
            self.df,
            ["label"],
            5,
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        sentence = str(row[self.sentence_col])
        target_idx = int(row[self.target_col])
        label = int(row[self.label_col])

        words = sentence.split()

        token_ids = [
            word2idx.get(word, 1)
            for word in words
        ]

        target_flags = [0] * len(words)

        if 0 <= target_idx < len(words):
            target_flags[target_idx] = 1
        else:
            raise ValueError(
                f"Invalid target index {target_idx} "
                f"for sentence with {len(words)} tokens."
            )

        # Match Gao's maximum sequence length.
        token_ids = token_ids[:MAX_LEN]
        target_flags = target_flags[:MAX_LEN]

        length = len(token_ids)

        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(target_flags, dtype=torch.long),
            length,
            label,
        )


# ============================================================
# Batch padding
# ============================================================

def collate_fn(batch):

    token_ids, target_flags, lengths, labels = zip(*batch)

    max_length = max(lengths)
    batch_size = len(batch)

    padded_tokens = torch.zeros(
        batch_size,
        max_length,
        dtype=torch.long,
    )

    padded_targets = torch.zeros(
        batch_size,
        max_length,
        dtype=torch.long,
    )

    for i in range(batch_size):

        length = lengths[i]

        padded_tokens[i, :length] = token_ids[i]
        padded_targets[i, :length] = target_flags[i]

    return (
        padded_tokens,
        padded_targets,
        torch.tensor(lengths, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


# ============================================================
# Load datasets
# ============================================================

print("\n===== Loading VUA =====")

train_dataset = VUADataset(TRAIN_PATH)
val_dataset = VUADataset(VAL_PATH)
test_dataset = VUADataset(TEST_PATH)

print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
)


# ============================================================
# Gao-style BiLSTM + Attention
# ============================================================

class GaoBiLSTMAttention(nn.Module):

    def __init__(
        self,
        glove_matrix,
        target_indicator_matrix,
        hidden_size=300,
    ):
        super().__init__()

        glove_tensor = torch.tensor(
            glove_matrix,
            dtype=torch.float32,
        )

        target_tensor = torch.tensor(
            target_indicator_matrix,
            dtype=torch.float32,
        )

        # Gao's preprocessing uses fixed input embeddings.
        self.glove = nn.Embedding.from_pretrained(
            glove_tensor,
            freeze=True,
            padding_idx=0,
        )

        self.target_indicator = nn.Embedding.from_pretrained(
            target_tensor,
            freeze=True,
        )

        input_dim = (
            glove_tensor.shape[1]
            + target_tensor.shape[1]
        )

        # 300 GloVe + 50 target indicator = 350
        print("BiLSTM input dimension:", input_dim)

        self.input_dropout = nn.Dropout(
            INPUT_DROPOUT
        )

        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # BiLSTM output = 300 forward + 300 backward
        output_dim = hidden_size * 2

        self.attention = nn.Linear(
            output_dim,
            1,
        )

        self.output_dropout = nn.Dropout(
            OUTPUT_DROPOUT
        )

        self.classifier = nn.Linear(
            output_dim,
            2,
        )

    def forward(
        self,
        token_ids,
        target_flags,
        lengths,
    ):

        # --------------------------------------------
        # Embeddings
        # --------------------------------------------

        glove_vectors = self.glove(token_ids)

        target_vectors = self.target_indicator(
            target_flags
        )

        x = torch.cat(
            [glove_vectors, target_vectors],
            dim=-1,
        )

        # Make padding positions exactly zero.
        sequence_length = token_ids.size(1)

        mask = (
            torch.arange(
                sequence_length,
                device=token_ids.device,
            )[None, :]
            < lengths[:, None]
        )

        x = x * mask.unsqueeze(-1)

        x = self.input_dropout(x)

        # --------------------------------------------
        # BiLSTM
        # --------------------------------------------

        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_output, _ = self.bilstm(
            packed
        )

        lstm_output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=sequence_length,
        )

        # --------------------------------------------
        # Attention
        # --------------------------------------------

        attention_logits = self.attention(
            lstm_output
        ).squeeze(-1)

        attention_logits = attention_logits.masked_fill(
            ~mask,
            float("-inf"),
        )

        attention_weights = torch.softmax(
            attention_logits,
            dim=1,
        )

        sentence_representation = torch.sum(
            lstm_output
            * attention_weights.unsqueeze(-1),
            dim=1,
        )

        # --------------------------------------------
        # Classification
        # --------------------------------------------

        sentence_representation = self.output_dropout(
            sentence_representation
        )

        logits = self.classifier(
            sentence_representation
        )

        return logits


# ============================================================
# Initialize model
# ============================================================

model = GaoBiLSTMAttention(
    glove_matrix,
    target_indicator_matrix,
    hidden_size=HIDDEN_SIZE,
).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LR,
    momentum=MOMENTUM,
)


# ============================================================
# Evaluation
# ============================================================

def evaluate(model, loader):

    model.eval()

    total_loss = 0.0
    predictions = []
    labels_all = []

    with torch.no_grad():

        for (
            tokens,
            targets,
            lengths,
            labels,
        ) in loader:

            tokens = tokens.to(DEVICE)
            targets = targets.to(DEVICE)
            lengths = lengths.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(
                tokens,
                targets,
                lengths,
            )

            loss = criterion(
                logits,
                labels,
            )

            total_loss += (
                loss.item()
                * labels.size(0)
            )

            batch_predictions = torch.argmax(
                logits,
                dim=1,
            )

            predictions.extend(
                batch_predictions.cpu().tolist()
            )

            labels_all.extend(
                labels.cpu().tolist()
            )

    average_loss = total_loss / len(loader.dataset)

    accuracy = accuracy_score(
        labels_all,
        predictions,
    )

    precision = precision_score(
        labels_all,
        predictions,
        pos_label=1,
        zero_division=0,
    )

    recall = recall_score(
        labels_all,
        predictions,
        pos_label=1,
        zero_division=0,
    )

    f1 = f1_score(
        labels_all,
        predictions,
        pos_label=1,
        zero_division=0,
    )

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# Training
# ============================================================

print("\n===== Training =====")

history = []

best_val_f1 = -1.0

checkpoint_path = (
    CHECKPOINT_DIR
    / "gao2018_bilstm_no_elmo_best.pt"
)

for epoch in range(1, EPOCHS + 1):

    model.train()

    running_loss = 0.0

    for (
        tokens,
        targets,
        lengths,
        labels,
    ) in train_loader:

        tokens = tokens.to(DEVICE)
        targets = targets.to(DEVICE)
        lengths = lengths.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        logits = model(
            tokens,
            targets,
            lengths,
        )

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * labels.size(0)
        )

    train_loss = (
        running_loss
        / len(train_dataset)
    )

    val_metrics = evaluate(
        model,
        val_loader,
    )

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_metrics['loss']:.4f} | "
        f"P: {val_metrics['precision']:.4f} | "
        f"R: {val_metrics['recall']:.4f} | "
        f"F1: {val_metrics['f1']:.4f}"
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_metrics["loss"],
        "val_accuracy": val_metrics["accuracy"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_f1": val_metrics["f1"],
    })

    if val_metrics["f1"] > best_val_f1:

        best_val_f1 = val_metrics["f1"]

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": best_val_f1,
            },
            checkpoint_path,
        )

        print(
            f"  -> Saved new best model "
            f"(Val F1={best_val_f1:.4f})"
        )


# ============================================================
# Save training history
# ============================================================

history_df = pd.DataFrame(history)

history_df.to_csv(
    RESULTS_DIR
    / "gao2018_bilstm_no_elmo_history.csv",
    index=False,
)


# ============================================================
# Test best checkpoint
# ============================================================

print("\n===== Loading Best Checkpoint =====")

checkpoint = torch.load(
    checkpoint_path,
    map_location=DEVICE,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print(
    "Best epoch:",
    checkpoint["epoch"],
)

print(
    "Best validation F1:",
    f"{checkpoint['val_f1']:.4f}",
)


print("\n===== Test Evaluation =====")

test_metrics = evaluate(
    model,
    test_loader,
)

print(
    f"Accuracy:  {test_metrics['accuracy']:.4f}"
)
print(
    f"Precision: {test_metrics['precision']:.4f}"
)
print(
    f"Recall:    {test_metrics['recall']:.4f}"
)
print(
    f"F1:        {test_metrics['f1']:.4f}"
)


# ============================================================
# Save final test result
# ============================================================

test_result = pd.DataFrame([
    {
        "Model":
            "Gao 2018 BiLSTM + GloVe + Target (No ELMo)",
        "Accuracy":
            test_metrics["accuracy"],
        "Precision":
            test_metrics["precision"],
        "Recall":
            test_metrics["recall"],
        "F1":
            test_metrics["f1"],
        "Best Epoch":
            checkpoint["epoch"],
    }
])

test_result.to_csv(
    RESULTS_DIR
    / "gao2018_bilstm_no_elmo_test.csv",
    index=False,
)

print("\n===== Done =====")
print(
    "Training history saved to:",
    RESULTS_DIR
    / "gao2018_bilstm_no_elmo_history.csv",
)
print(
    "Test results saved to:",
    RESULTS_DIR
    / "gao2018_bilstm_no_elmo_test.csv",
)