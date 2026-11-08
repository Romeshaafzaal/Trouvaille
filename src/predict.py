import pandas as pd, torch, torch.nn as nn
from sentence_transformers import SentenceTransformer

train = pd.read_csv("data/processed/train.csv")
train["occasions"] = train["occasions"].fillna("casual")
classes = sorted({o for s in train["occasions"] for o in s.split("|")})

net = nn.Sequential(nn.Linear(384, 256), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(128, len(classes)))
net.load_state_dict(torch.load("models/mlp_occasion.pt")); net.eval()
st = SentenceTransformer("all-MiniLM-L6-v2")

while True:
    text = input("\nDescribe a product (or 'q' to quit): ")
    if text.lower() == "q": break
    with torch.no_grad():
        p = torch.sigmoid(net(torch.tensor(st.encode([text]), dtype=torch.float32)))[0]
    for c, v in sorted(zip(classes, p.tolist()), key=lambda x: -x[1]):
        print(f"{c:13s} {v:.2f} {'<-- predicted' if v > 0.5 else ''}")