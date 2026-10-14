| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Majority | 0.700 | 0.000 | 0.000 | 0.000 |
| Lexical (train only) | 0.764 | 0.665 | 0.429 | 0.522 |
| Gao 2018 Lexical | 0.765 | 0.679 | 0.407 | 0.509 |



## Gao et al. (2018) Reproduction

We reproduced Gao et al. (2018)-style metaphor classification models on the VUA verb classification dataset.

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| BiLSTM + GloVe + Target Indicator | 0.7572 | 0.6139 | 0.5128 | 0.5588 |
| BiLSTM + GloVe + ELMo + Target Indicator | 0.8027 | 0.6783 | 0.6502 | 0.6640 |

Adding contextualized ELMo representations improved test F1 from **0.5588** to **0.6640**, an absolute gain of **10.52 F1 points**.

### Reproduction pipeline

- `src/04_prepare_gao2018_inputs.py` — prepares vocabulary and GloVe inputs
- `src/05_train_gao2018_bilstm.py` — trains the GloVe-only BiLSTM baseline
- `src/06_check_elmo_alignment.py` — verifies VUA/ELMo alignment
- `src/07_train_gao2018_elmo.py` — trains the BiLSTM + ELMo model