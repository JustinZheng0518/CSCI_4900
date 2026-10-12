import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# majority baseline: always predict literal

train = pd.read_csv(
    "data/VUA/VUA_formatted_train.csv",
    encoding="latin1"
)

test = pd.read_csv(
    "data/VUA/VUA_formatted_test.csv",
    encoding="latin1"
)

y_true = test["label"]

# majority baseline: always predict literal
y_pred = [0] * len(test)

print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred, zero_division=0))
print("Recall:", recall_score(y_true, y_pred, zero_division=0))
print("F1:", f1_score(y_true, y_pred, zero_division=0))

