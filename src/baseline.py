import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

def make_text(row):
    # strip the label part so the model can't read the answer
    desc = re.sub(r"^.*?occasions:[^.]*\.\s*", "", str(row["text_for_embedding"]))
    return f"{row['title']}. {desc}"

def load(name):
    df = pd.read_csv(f"data/processed/{name}.csv")
    df["occasions"] = df["occasions"].fillna("casual")
    return df

train, val = load("train"), load("val")
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(train["occasions"].str.split("|"))
y_val = mlb.transform(val["occasions"].str.split("|"))

model = SentenceTransformer("all-MiniLM-L6-v2")
model.max_seq_length = 128
X_train = model.encode(train.apply(make_text, axis=1).tolist(), batch_size=64, show_progress_bar=True)
X_val = model.encode(val.apply(make_text, axis=1).tolist(), batch_size=64, show_progress_bar=True)
np.save("data/processed/X_train.npy", X_train)
np.save("data/processed/X_val.npy", X_val)

clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced"))
clf.fit(X_train, y_train)

pred = clf.predict(X_val)
proba = clf.predict_proba(X_val)
p, r, f, _ = precision_recall_fscore_support(y_val, pred, average="macro", zero_division=0)
print("Classes:", list(mlb.classes_))
print(f"Precision {p:.3f}  Recall {r:.3f}  F1 {f:.3f}  AUC-ROC {roc_auc_score(y_val, proba, average='macro'):.3f}")