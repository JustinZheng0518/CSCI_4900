import pandas as pd

train = pd.read_csv(
    "data/VUA/VUA_formatted_train.csv",
    encoding="latin1"
)

print(train.head())

print("\nColumns:")
print(train.columns)

print("\nDataset shape:")
print(train.shape)

print("\nLabel counts:")
print(train["label"].value_counts())

print("\nLabel percentages:")
print(train["label"].value_counts(normalize=True))

print("\nMissing values:")
print(train.isnull().sum())