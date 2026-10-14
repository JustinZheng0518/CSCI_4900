## Baseline Results

The following experiments establish a progression from simple non-contextual baselines to contextual neural models on the VUA verb-classification task.

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Majority | 0.700 | 0.000 | 0.000 | 0.000 |
| Lexical (train only) | 0.764 | 0.665 | 0.429 | 0.522 |
| Gao 2018 Lexical | 0.765 | 0.679 | 0.407 | 0.509 |
| BiLSTM + GloVe + Target Indicator | 0.7572 | 0.6139 | 0.5128 | 0.5588 |
| BiLSTM + GloVe + ELMo + Target Indicator | 0.8027 | 0.6783 | 0.6502 | 0.6640 |

### Majority Baseline

The majority baseline always predicts the most frequent class in the training data.

Because literal usages are more common than metaphorical usages in the VUA verb-classification dataset, this model predicts every example as literal.

Although the model achieves an accuracy of 0.700, its metaphor precision, recall, and F1 are all 0.000. This illustrates why accuracy alone can be misleading for an imbalanced classification task.

### Lexical Baseline (Train Only)

The lexical baseline uses only the identity of the target verb and does not use sentence context.

For each verb, the proportion of metaphorical occurrences in the training set is calculated. A verb is predicted as metaphorical when more than half of its observed training occurrences are metaphorical; otherwise, it is predicted as literal.

This version constructs the lexical statistics using only the training split.

The model achieves an F1 score of 0.522, showing that lexical identity alone contains useful information for metaphor classification. However, it cannot distinguish between literal and metaphorical uses of the same verb in different contexts.

### Gao 2018 Lexical Baseline

This baseline follows the lexical-baseline setup used in Gao et al. (2018) more closely.

Like the train-only lexical baseline, prediction is based on the historical metaphor rate of the target verb rather than on the sentence context.

The Gao-style setup uses the available training and validation data to construct the lexical statistics before evaluation on the test set.

The resulting F1 score of 0.509 provides a historical reference point for the contextual neural models reproduced later in the project.

---

## Gao et al. (2018) Reproduction

We reproduced Gao et al. (2018)-style neural metaphor classification models on the VUA verb-classification dataset.

The purpose of this stage is to study the transition from static word embeddings to contextualized word representations before moving to modern Transformer-based models.

### BiLSTM + GloVe + Target Indicator

The first neural model introduces sentence context through a bidirectional LSTM.

Each token is represented using:

- a 300-dimensional pretrained GloVe embedding;
- a 50-dimensional target indicator identifying the verb being classified.

These representations are concatenated to form a 350-dimensional input vector for each token.

The sequence is then processed by:

1. a bidirectional LSTM;
2. an attention mechanism;
3. a binary classifier predicting literal or metaphorical usage.

Unlike the lexical baselines, this model can use information from surrounding words when classifying the target verb.

Test performance:

- Accuracy: 0.7572
- Precision: 0.6139
- Recall: 0.5128
- F1: **0.5588**

### BiLSTM + GloVe + ELMo + Target Indicator

The second neural model extends the previous architecture by adding pretrained ELMo contextual representations.

Each token is represented using:

- a 300-dimensional GloVe embedding;
- a 1024-dimensional ELMo contextual embedding;
- a 50-dimensional target indicator.

These features are concatenated into a 1374-dimensional representation:

`300 + 1024 + 50 = 1374`

The same BiLSTM, attention, and classification architecture is then used.

Test performance:

- Accuracy: 0.8027
- Precision: 0.6783
- Recall: 0.6502
- F1: **0.6640**

Adding ELMo improves test F1 from **0.5588** to **0.6640**, an absolute improvement of **10.52 F1 points**.

The largest improvement is also visible in recall:

`0.5128 → 0.6502`

This suggests that contextualized representations help the model identify substantially more metaphorical usages than static GloVe representations alone.

### Reproduction Summary

The experiments show a clear progression:

`Lexical information → sentence context → contextualized representations`

The lexical baselines demonstrate that target-word identity alone contains useful information about metaphor usage.

The GloVe BiLSTM introduces sentence context and improves over the Gao lexical baseline.

Adding ELMo produces a substantially larger improvement, supporting the importance of contextualized word representations for metaphor identification.

These experiments establish the historical baseline for the next stage of the project, which will investigate Transformer-based representations such as BERT and RoBERTa.

### Reproduction Pipeline

- `src/04_prepare_gao2018_inputs.py` — builds the vocabulary and prepares GloVe and target-indicator inputs.
- `src/05_train_gao2018_bilstm.py` — trains the BiLSTM + GloVe + Target Indicator model.
- `src/06_check_elmo_alignment.py` — verifies alignment between VUA sentences and the precomputed Gao 2018 ELMo representations.
- `src/07_train_gao2018_elmo.py` — trains the BiLSTM + GloVe + ELMo + Target Indicator model.

### Generated Results

The main experiment outputs are stored in the `results/` directory:

- `gao2018_bilstm_no_elmo_history.csv`
- `gao2018_bilstm_no_elmo_test.csv`
- `gao2018_bilstm_elmo_history.csv`
- `gao2018_bilstm_elmo_test.csv`
- `baseline_results.csv`

Large pretrained embeddings and model checkpoints are excluded from version control through `.gitignore`.