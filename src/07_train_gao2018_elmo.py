"""
Experiment 07: Gao et al. (2018) BiLSTM + GloVe + ELMo

Purpose:
    Reproduce the contextualized VUA verb-classification model from
    Gao et al. (2018) using:

        - 300d GloVe embeddings
        - 1024d pretrained ELMo contextual representations
        - 50d target-verb indicator
        - Bidirectional LSTM
        - Attention
        - Binary metaphor classifier

Dataset:
    VUA 2018 verb classification.

Research question:
    How much does adding contextualized ELMo representation improve
    metaphor detection relative to the GloVe-only BiLSTM baseline?

Comparison baseline:
    Experiment 05:
        BiLSTM + GloVe + target indicator
        Test F1 = 0.5588

Outputs:
    results/gao2018_bilstm_elmo_history.csv
    results/gao2018_bilstm_elmo_test.csv

Checkpoint:
    checkpoints/gao2018_bilstm_elmo_best.pt
"""

import json
import random
from pathlib import Path

import h5py
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

from torch.utils.data import (
    Dataset,
    DataLoader,
)


# ============================================================
# Paths and configuration
# ============================================================

CONFIG_PATH = Path(
    "configs/bilstm_elmo_vua.yaml"
)

CACHE_DIR = Path(
    "data/processed/gao2018"
)

RESULTS_DIR = Path("results")
CHECKPOINT_DIR = Path("checkpoints")

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8",
) as f:

    config = yaml.safe_load(f)


# ============================================================
# Training configuration
# ============================================================

SEED = config["training"]["seed"]

BATCH_SIZE = config["training"]["batch_size"]

LR = config["training"]["learning_rate"]

MOMENTUM = config["training"]["momentum"]

EPOCHS = config["training"]["epochs"]


# ============================================================
# Model configuration
# ============================================================

HIDDEN_SIZE = config["model"]["hidden_size"]

INPUT_DROPOUT = config["model"]["input_dropout"]

OUTPUT_DROPOUT = config["model"]["output_dropout"]

MAX_LEN = config["data"]["max_sequence_length"]


# ============================================================
# Dataset paths
# ============================================================

TRAIN_PATH = Path(
    config["data"]["train_path"]
)

VAL_PATH = Path(
    config["data"]["val_path"]
)

TEST_PATH = Path(
    config["data"]["test_path"]
)

CSV_ENCODING = config["data"]["encoding"]


ELMO_TRAIN_PATH = Path(
    config["embeddings"]["elmo_train_path"]
)

ELMO_VAL_PATH = Path(
    config["embeddings"]["elmo_val_path"]
)

ELMO_TEST_PATH = Path(
    config["embeddings"]["elmo_test_path"]
)


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("===== Device =====")
print("Device:", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Load cached GloVe inputs
# ============================================================

print(
    "\n===== Loading Cached Inputs ====="
)

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


print(
    "Vocabulary size:",
    len(word2idx),
)

print(
    "GloVe matrix:",
    glove_matrix.shape,
)

print(
    "Target indicator:",
    target_indicator_matrix.shape,
)


# ============================================================
# Dataset
# ============================================================

class VUAELMoDataset(Dataset):

    def __init__(
        self,
        csv_path,
        elmo_path,
    ):

        self.df = pd.read_csv(
            csv_path,
            encoding=CSV_ENCODING,
        )

        self.elmo_path = elmo_path

        # Open lazily on first access.
        # This avoids unnecessary interaction with
        # HDF5 during object construction.
        self.h5 = None


    def _get_h5(self):

        if self.h5 is None:

            self.h5 = h5py.File(
                self.elmo_path,
                "r",
            )

        return self.h5


    def __len__(self):

        return len(self.df)


    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        sentence = str(
            row["sentence"]
        )

        target_idx = int(
            row["verb_idx"]
        )

        label = int(
            row["label"]
        )

        words = sentence.split()

        # ----------------------------------------------------
        # Word IDs for GloVe
        # ----------------------------------------------------

        token_ids = [
            word2idx.get(word, 1)
            for word in words
        ]

        # ----------------------------------------------------
        # Target indicator
        # ----------------------------------------------------

        target_flags = [
            0
            for _ in words
        ]

        target_flags[target_idx] = 1


        # ----------------------------------------------------
        # Read ELMo
        #
        # Shape:
        #   [sequence_length, 1024]
        # ----------------------------------------------------

        h5 = self._get_h5()

        elmo = np.asarray(
            h5[sentence],
            dtype=np.float32,
        )


        # ----------------------------------------------------
        # Alignment safeguard
        # ----------------------------------------------------

        if len(words) != elmo.shape[0]:

            raise ValueError(
                "ELMo alignment error:\n"
                f"sentence={sentence}\n"
                f"tokens={len(words)}\n"
                f"ELMo={elmo.shape}"
            )


        # ----------------------------------------------------
        # Maximum length handling
        # ----------------------------------------------------

        token_ids = token_ids[:MAX_LEN]

        target_flags = target_flags[:MAX_LEN]

        elmo = elmo[:MAX_LEN]

        length = len(token_ids)


        # ----------------------------------------------------
        # Critical safeguard:
        # target must survive truncation
        # ----------------------------------------------------

        if target_idx >= length:

            raise ValueError(
                "Target verb removed by truncation.\n"
                f"target_idx={target_idx}\n"
                f"length={length}\n"
                f"sentence={sentence}"
            )


        return (
            torch.tensor(
                token_ids,
                dtype=torch.long,
            ),

            torch.tensor(
                target_flags,
                dtype=torch.long,
            ),

            torch.tensor(
                elmo,
                dtype=torch.float32,
            ),

            length,

            label,
        )


# ============================================================
# Collate / padding
# ============================================================

def collate_fn(batch):

    (
        token_ids,
        target_flags,
        elmo_vectors,
        lengths,
        labels,
    ) = zip(*batch)


    max_length = max(lengths)

    batch_size = len(batch)

    elmo_dim = elmo_vectors[0].shape[-1]


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

    padded_elmo = torch.zeros(
        batch_size,
        max_length,
        elmo_dim,
        dtype=torch.float32,
    )


    for i in range(batch_size):

        length = lengths[i]

        padded_tokens[
            i,
            :length
        ] = token_ids[i]

        padded_targets[
            i,
            :length
        ] = target_flags[i]

        padded_elmo[
            i,
            :length
        ] = elmo_vectors[i]


    return (
        padded_tokens,
        padded_targets,
        padded_elmo,

        torch.tensor(
            lengths,
            dtype=torch.long,
        ),

        torch.tensor(
            labels,
            dtype=torch.long,
        ),
    )


# ============================================================
# Load datasets
# ============================================================

print(
    "\n===== Loading VUA + ELMo ====="
)


train_dataset = VUAELMoDataset(
    TRAIN_PATH,
    ELMO_TRAIN_PATH,
)

val_dataset = VUAELMoDataset(
    VAL_PATH,
    ELMO_VAL_PATH,
)

test_dataset = VUAELMoDataset(
    TEST_PATH,
    ELMO_TEST_PATH,
)


print(
    "Train:",
    len(train_dataset),
)

print(
    "Validation:",
    len(val_dataset),
)

print(
    "Test:",
    len(test_dataset),
)


# IMPORTANT:
# num_workers=0 keeps HDF5 access simple and safe
# on Windows.

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0,
)


# ============================================================
# Model
# ============================================================

class GaoELMoBiLSTM(nn.Module):

    def __init__(
        self,
        glove_matrix,
        target_indicator_matrix,
        hidden_size,
    ):

        super().__init__()


        # ----------------------------------------------------
        # GloVe
        # ----------------------------------------------------

        glove_tensor = torch.tensor(
            glove_matrix,
            dtype=torch.float32,
        )

        self.glove = (
            nn.Embedding.from_pretrained(
                glove_tensor,
                freeze=True,
                padding_idx=0,
            )
        )


        # ----------------------------------------------------
        # Target indicator
        # ----------------------------------------------------

        target_tensor = torch.tensor(
            target_indicator_matrix,
            dtype=torch.float32,
        )

        self.target_indicator = (
            nn.Embedding.from_pretrained(
                target_tensor,
                freeze=True,
            )
        )


        glove_dim = glove_tensor.shape[1]

        target_dim = target_tensor.shape[1]

        elmo_dim = config["model"]["elmo_dim"]


        input_dim = (
            glove_dim
            + elmo_dim
            + target_dim
        )


        print(
            "GloVe dimension:",
            glove_dim,
        )

        print(
            "ELMo dimension:",
            elmo_dim,
        )

        print(
            "Target dimension:",
            target_dim,
        )

        print(
            "BiLSTM input dimension:",
            input_dim,
        )


        # ----------------------------------------------------
        # Dropout
        # ----------------------------------------------------

        self.input_dropout = nn.Dropout(
            INPUT_DROPOUT
        )


        # ----------------------------------------------------
        # BiLSTM
        # ----------------------------------------------------

        self.bilstm = nn.LSTM(
            input_size=input_dim,

            hidden_size=hidden_size,

            num_layers=1,

            batch_first=True,

            bidirectional=True,
        )


        output_dim = (
            hidden_size * 2
        )


        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------

        self.attention = nn.Linear(
            output_dim,
            1,
        )


        self.output_dropout = nn.Dropout(
            OUTPUT_DROPOUT
        )


        # ----------------------------------------------------
        # Binary classifier
        # ----------------------------------------------------

        self.classifier = nn.Linear(
            output_dim,
            2,
        )


    def forward(
        self,
        token_ids,
        target_flags,
        elmo_vectors,
        lengths,
    ):

        # ----------------------------------------------------
        # Static GloVe
        # ----------------------------------------------------

        glove_vectors = self.glove(
            token_ids
        )


        # ----------------------------------------------------
        # Target indicator
        # ----------------------------------------------------

        target_vectors = (
            self.target_indicator(
                target_flags
            )
        )


        # ----------------------------------------------------
        # Concatenate representations
        #
        # 300 + 1024 + 50 = 1374
        # ----------------------------------------------------

        x = torch.cat(
            [
                glove_vectors,
                elmo_vectors,
                target_vectors,
            ],
            dim=-1,
        )


        sequence_length = (
            token_ids.size(1)
        )


        mask = (
            torch.arange(
                sequence_length,
                device=token_ids.device,
            )[None, :]
            <
            lengths[:, None]
        )


        x = (
            x
            * mask.unsqueeze(-1)
        )


        x = self.input_dropout(x)


        # ----------------------------------------------------
        # Packed BiLSTM
        # ----------------------------------------------------

        packed = pack_padded_sequence(
            x,

            lengths.cpu(),

            batch_first=True,

            enforce_sorted=False,
        )


        packed_output, _ = (
            self.bilstm(packed)
        )


        lstm_output, _ = (
            pad_packed_sequence(
                packed_output,

                batch_first=True,

                total_length=sequence_length,
            )
        )


        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------

        attention_logits = (
            self.attention(
                lstm_output
            ).squeeze(-1)
        )


        attention_logits = (
            attention_logits.masked_fill(
                ~mask,
                float("-inf"),
            )
        )


        attention_weights = (
            torch.softmax(
                attention_logits,
                dim=1,
            )
        )


        sentence_representation = (
            torch.sum(
                lstm_output
                * attention_weights.unsqueeze(-1),

                dim=1,
            )
        )


        sentence_representation = (
            self.output_dropout(
                sentence_representation
            )
        )


        logits = self.classifier(
            sentence_representation
        )


        return logits


# ============================================================
# Initialize model
# ============================================================

model = GaoELMoBiLSTM(
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

def evaluate(
    model,
    loader,
):

    model.eval()

    total_loss = 0.0

    predictions = []

    labels_all = []


    with torch.no_grad():

        for (
            tokens,
            targets,
            elmo,
            lengths,
            labels,
        ) in loader:

            tokens = tokens.to(DEVICE)

            targets = targets.to(DEVICE)

            elmo = elmo.to(DEVICE)

            lengths = lengths.to(DEVICE)

            labels = labels.to(DEVICE)


            logits = model(
                tokens,
                targets,
                elmo,
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


            batch_predictions = (
                torch.argmax(
                    logits,
                    dim=1,
                )
            )


            predictions.extend(
                batch_predictions
                .cpu()
                .tolist()
            )


            labels_all.extend(
                labels
                .cpu()
                .tolist()
            )


    average_loss = (
        total_loss
        / len(loader.dataset)
    )


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

print(
    "\n===== Training Gao 2018 ELMo Model ====="
)


history = []

best_val_f1 = -1.0


checkpoint_path = (
    CHECKPOINT_DIR
    / "gao2018_bilstm_elmo_best.pt"
)


for epoch in range(
    1,
    EPOCHS + 1,
):

    model.train()

    running_loss = 0.0


    for (
        tokens,
        targets,
        elmo,
        lengths,
        labels,
    ) in train_loader:


        tokens = tokens.to(DEVICE)

        targets = targets.to(DEVICE)

        elmo = elmo.to(DEVICE)

        lengths = lengths.to(DEVICE)

        labels = labels.to(DEVICE)


        optimizer.zero_grad()


        logits = model(
            tokens,
            targets,
            elmo,
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


    history.append(
        {
            "epoch": epoch,
            "train_loss": train_loss,

            "val_loss":
                val_metrics["loss"],

            "val_accuracy":
                val_metrics["accuracy"],

            "val_precision":
                val_metrics["precision"],

            "val_recall":
                val_metrics["recall"],

            "val_f1":
                val_metrics["f1"],
        }
    )


    # --------------------------------------------------------
    # Save best validation checkpoint
    # --------------------------------------------------------

    if (
        val_metrics["f1"]
        > best_val_f1
    ):

        best_val_f1 = (
            val_metrics["f1"]
        )


        torch.save(
            {
                "epoch":
                    epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "val_f1":
                    best_val_f1,
            },

            checkpoint_path,
        )


        print(
            "  -> Saved new best model "
            f"(Val F1={best_val_f1:.4f})"
        )


# ============================================================
# Save history
# ============================================================

history_df = pd.DataFrame(
    history
)


history_path = (
    RESULTS_DIR
    / "gao2018_bilstm_elmo_history.csv"
)


history_df.to_csv(
    history_path,
    index=False,
)


# ============================================================
# Load best checkpoint
# ============================================================

print(
    "\n===== Loading Best Checkpoint ====="
)


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


# ============================================================
# Final test evaluation
# ============================================================

print(
    "\n===== Test Evaluation ====="
)


test_metrics = evaluate(
    model,
    test_loader,
)


print(
    f"Accuracy:  "
    f"{test_metrics['accuracy']:.4f}"
)

print(
    f"Precision: "
    f"{test_metrics['precision']:.4f}"
)

print(
    f"Recall:    "
    f"{test_metrics['recall']:.4f}"
)

print(
    f"F1:        "
    f"{test_metrics['f1']:.4f}"
)


# ============================================================
# Save final test result
# ============================================================

test_result = pd.DataFrame(
    [
        {
            "Model":
                "Gao 2018 BiLSTM + GloVe + ELMo + Target",

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
    ]
)


test_path = (
    RESULTS_DIR
    / "gao2018_bilstm_elmo_test.csv"
)


test_result.to_csv(
    test_path,
    index=False,
)


print("\n===== Done =====")

print(
    "Training history saved to:",
    history_path,
)

print(
    "Test results saved to:",
    test_path,
)

print(
    "Best checkpoint saved to:",
    checkpoint_path,
)