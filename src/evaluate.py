import os, re
import numpy as np, pandas as pd, torch, torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, classification_report

def make_text(row):
    desc = re.sub(r"^.*?occasions:[^.]*\.\s*", "", str(row["text_for_embedding"]))
    return f"{row['title']}. {desc}"

def load(n):
    d = pd.read_csv(f"data/processed/{n}.csv"); d["occasions"] = d["occasions"].fillna("casual"); return d

train, test = load("train"), load("test")
mlb = MultiLabelBinarizer()
mlb.fit(train["occasions"].str.split("|"))
y = mlb.transform(test["occasions"].str.split("|"))

if os.path.exists("data/processed/X_test.npy"):
    X = np.load("data/processed/X_test.npy")
else:
    st = SentenceTransformer("all-MiniLM-L6-v2"); st.max_seq_length = 128
    X = st.encode(test.apply(make_text, axis=1).tolist(), batch_size=64, show_progress_bar=True)
    np.save("data/processed/X_test.npy", X)

net = nn.Sequential(nn.Linear(384, 256), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(128, y.shape[1]))
net.load_state_dict(torch.load("models/mlp_occasion.pt")); net.eval()
with torch.no_grad():
    proba = torch.sigmoid(net(torch.tensor(X, dtype=torch.float32))).numpy()
pred = proba > 0.5

p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
print(f"TEST  Precision {p:.3f}  Recall {r:.3f}  F1 {f:.3f}  AUC {roc_auc_score(y, proba, average='macro'):.3f}\n")
print(classification_report(y, pred, target_names=mlb.classes_, zero_division=0))

# Bias check: macro F1 per subgroup
rows = []
for col in ["brand", "gender", "category", "has_plus_sizes"]:
    for val, g in test.groupby(col):
        idx = g.index.values
        if len(idx) < 30: continue
        gf = precision_recall_fscore_support(y[idx], pred[idx], average="macro", zero_division=0)[2]
        rows.append(dict(group=col, value=val, n=len(idx), f1=round(gf, 3)))
bias = pd.DataFrame(rows)
print(bias.to_string(index=False))
bias.to_csv("data/processed/bias_check.csv", index=False)