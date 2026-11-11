import os
import numpy as np, pandas as pd, torch, torch.nn as nn

def load(n):
    d = pd.read_csv(f"data/processed/{n}.csv"); d["occasions"] = d["occasions"].fillna("casual"); return d

train, val, test = load("train"), load("val"), load("test")
classes = sorted({o for s in train["occasions"] for o in s.split("|")})

df = pd.concat([train, val, test], ignore_index=True)
X = np.vstack([np.load(f"data/processed/X_{n}.npy") for n in ["train", "val", "test"]])

net = nn.Sequential(nn.Linear(384, 256), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(128, len(classes)))
net.load_state_dict(torch.load("models/mlp_occasion.pt")); net.eval()
with torch.no_grad():
    proba = torch.sigmoid(net(torch.tensor(X, dtype=torch.float32))).numpy()

for i, c in enumerate(classes):
    df[f"p_{c}"] = proba[:, i].round(3)

keep = ["brand", "title", "category", "gender", "color", "price", "compare_at_price",
        "discount_pct", "available", "image_url", "url"] + [f"p_{c}" for c in classes]
os.makedirs("app_data", exist_ok=True)
df[keep].to_csv("app_data/products_app.csv", index=False)
print(df[keep].shape)