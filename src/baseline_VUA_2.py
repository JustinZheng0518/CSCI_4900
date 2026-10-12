import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# if the work shows that the majority of the words are literal, then we can use a majority baseline 
# to predict all words as literal. This will give us a baseline accuracy to compare our model against.

train = pd.read_csv(
    "data/VUA/VUA_formatted_train.csv",
    encoding="latin1"
)

test = pd.read_csv(
    "data/VUA/VUA_formatted_test.csv",
    encoding="latin1"
)

# 每个 verb 在训练集中成为 metaphor 的比例
verb_metaphor_rate = train.groupby("verb")["label"].mean()

def lexical_predict(verb):
    # 如果这个 verb 在训练集见过
    if verb in verb_metaphor_rate.index:
        return 1 if verb_metaphor_rate[verb] >= 0.5 else 0

    # 没见过的 verb 默认预测 literal
    return 0

y_true = test["label"]
y_pred = test["verb"].apply(lexical_predict)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("Lexical baseline")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)

unseen = ~test["verb"].isin(train["verb"].unique())

print("\nTest examples:", len(test))
print("Unseen verb examples:", unseen.sum())
print("Unseen verb rate:", unseen.mean())