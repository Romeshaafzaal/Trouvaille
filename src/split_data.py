import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/processed/products_clean.csv")

train, temp = train_test_split(df, test_size=0.30, random_state=42, stratify=df["category"])
val, test = train_test_split(temp, test_size=0.50, random_state=42, stratify=temp["category"])

train.to_csv("data/processed/train.csv", index=False)
val.to_csv("data/processed/val.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)

print(len(train), len(val), len(test))