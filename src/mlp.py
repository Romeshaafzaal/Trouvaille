import itertools, json
import numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

torch.manual_seed(42)
def load(n):
    d = pd.read_csv(f"data/processed/{n}.csv"); d["occasions"] = d["occasions"].fillna("casual"); return d
train, val = load("train"), load("val")
mlb = MultiLabelBinarizer()
y_tr = mlb.fit_transform(train["occasions"].str.split("|")).astype("float32")
y_va = mlb.transform(val["occasions"].str.split("|")).astype("float32")
X_tr = torch.tensor(np.load("data/processed/X_train.npy"))
X_va = torch.tensor(np.load("data/processed/X_val.npy"))
Y_tr = torch.tensor(y_tr)

def run(lr, batch, dropout, epochs=30):
    net = nn.Sequential(nn.Linear(384, 256), nn.ReLU(), nn.Dropout(dropout),
                        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
                        nn.Linear(128, y_tr.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    tr_losses, va_losses = [], []
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(perm), batch):
            idx = perm[i:i+batch]
            opt.zero_grad(); loss_fn(net(X_tr[idx]), Y_tr[idx]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            tr_losses.append(loss_fn(net(X_tr), Y_tr).item())
            va_losses.append(loss_fn(net(X_va), torch.tensor(y_va)).item())
    with torch.no_grad():
        proba = torch.sigmoid(net(X_va)).numpy()
    p, r, f, _ = precision_recall_fscore_support(y_va, proba > 0.5, average="macro", zero_division=0)
    return net, dict(lr=lr, batch=batch, dropout=dropout, precision=p, recall=r, f1=f,
                     auc=roc_auc_score(y_va, proba, average="macro")), tr_losses, va_losses

results, best = [], None
for lr, batch, dropout in itertools.product([1e-2, 1e-3, 1e-4], [64, 256], [0.0, 0.3]):
    net, res, trl, val_l = run(lr, batch, dropout)
    results.append(res); print(res)
    if best is None or res["f1"] > best[1]["f1"]:
        best = (net, res, trl, val_l)

pd.DataFrame(results).to_csv("data/processed/mlp_tuning.csv", index=False)
torch.save(best[0].state_dict(), "models/mlp_occasion.pt")
plt.plot(best[2], label="train"); plt.plot(best[3], label="val")
plt.xlabel("epoch"); plt.ylabel("BCE loss"); plt.legend(); plt.savefig("figures/mlp_loss_curve.png", dpi=150)
print("BEST:", json.dumps(best[1], indent=2))