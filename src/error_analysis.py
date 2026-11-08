"""Phase 4 error analysis: per-class metrics, ROC curves, confusion matrices and the
most confident mistakes of the occasion classifier on the held-out TEST set.

Run from the repo root (after src/mlp.py has produced models/mlp_occasion.pt):
    python src/error_analysis.py
Outputs: figures/roc_per_class.png, figures/confusion_per_class.png,
         report/tables/*.tex, data/processed/misclassified_examples.csv,
         data/processed/test_overall.json
"""
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (precision_recall_fscore_support, roc_auc_score, roc_curve,
                             multilabel_confusion_matrix, hamming_loss)
from report_utils import df_to_tex

THRESHOLD = 0.5


def load(name):
    df = pd.read_csv(f"data/processed/{name}.csv")
    df["occasions"] = df["occasions"].fillna("casual")
    return df


def build_net(n_classes):
    # must match the architecture trained in src/mlp.py
    return nn.Sequential(nn.Linear(384, 256), nn.ReLU(), nn.Dropout(0.3),
                         nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
                         nn.Linear(128, n_classes))


def main():
    train, test = load("train"), load("test")
    mlb = MultiLabelBinarizer()
    mlb.fit(train["occasions"].str.split("|"))
    classes = list(mlb.classes_)
    y = mlb.transform(test["occasions"].str.split("|"))
    X = np.load("data/processed/X_test.npy")  # created by src/evaluate.py

    net = build_net(len(classes))
    net.load_state_dict(torch.load("models/mlp_occasion.pt"))
    net.eval()
    with torch.no_grad():
        proba = torch.sigmoid(net(torch.tensor(X, dtype=torch.float32))).numpy()
    pred = (proba > THRESHOLD).astype(int)

    # ---- overall numbers -------------------------------------------------------------
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    overall = dict(precision=p, recall=r, f1=f, auc=roc_auc_score(y, proba, average="macro"),
                   micro_f1=precision_recall_fscore_support(y, pred, average="micro", zero_division=0)[2],
                   hamming=hamming_loss(y, pred), exact_match=float((pred == y).all(axis=1).mean()),
                   n_test=len(test))
    json.dump(overall, open("data/processed/test_overall.json", "w"), indent=2)
    with open("report/tables/numbers.tex", "w") as fh:
        fh.write("\n".join([
            rf"\renewcommand{{\testPrec}}{{{p:.3f}}}", rf"\renewcommand{{\testRec}}{{{r:.3f}}}",
            rf"\renewcommand{{\testFone}}{{{f:.3f}}}", rf"\renewcommand{{\testAUC}}{{{overall['auc']:.3f}}}",
            rf"\renewcommand{{\testMicroF}}{{{overall['micro_f1']:.3f}}}",
            rf"\renewcommand{{\testHamming}}{{{overall['hamming']:.3f}}}",
            rf"\renewcommand{{\testExact}}{{{overall['exact_match']:.3f}}}",
            rf"\renewcommand{{\testN}}{{{len(test)}}}"]) + "\n")

    # ---- per-class table -------------------------------------------------------------
    cms = multilabel_confusion_matrix(y, pred)  # per class: [[TN, FP], [FN, TP]]
    rows = []
    for i, c in enumerate(classes):
        tn, fp, fn, tp = cms[i].ravel()
        pr, rc, f1, _ = precision_recall_fscore_support(y[:, i], pred[:, i], average="binary", zero_division=0)
        rows.append(dict(Class=c, Support=int(y[:, i].sum()), Prevalence=y[:, i].mean(),
                         Precision=pr, Recall=rc, F1=f1, AUC=roc_auc_score(y[:, i], proba[:, i]),
                         FP=int(fp), FN=int(fn)))
    per_class = pd.DataFrame(rows)
    per_class.to_csv("data/processed/per_class_metrics.csv", index=False)
    df_to_tex(per_class, "report/tables/per_class.tex", col_format="lrrrrrrrr")
    print(per_class.to_string(index=False))

    # ---- ROC curves ------------------------------------------------------------------
    plt.figure(figsize=(5.5, 4.5))
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y[:, i], proba[:, i])
        plt.plot(fpr, tpr, label=f"{c} (AUC {per_class.loc[i, 'AUC']:.2f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("Per-class ROC curves (test set)"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig("figures/roc_per_class.png", dpi=150); plt.close()

    # ---- confusion matrices ----------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    axes = axes.ravel()
    for ax, c, cm in zip(axes, classes, cms):
        ax.imshow(cm, cmap="Blues")
        for (a, b), v in np.ndenumerate(cm):
            ax.text(b, a, str(v), ha="center", va="center", fontsize=11,
                    color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred 0", "pred 1"], fontsize=9)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["true 0", "true 1"], fontsize=9)
        ax.set_title(c, fontsize=11)
    for ax in axes[len(classes):]:
        ax.axis("off")
    plt.tight_layout(); plt.savefig("figures/confusion_per_class.png", dpi=150); plt.close()

    # ---- most confident mistakes -----------------------------------------------------
    err = np.abs(proba - y)
    wrong = (pred != y).any(axis=1)
    worst_class = err.argmax(axis=1)
    order = np.argsort(-err.max(axis=1))
    order = [i for i in order if wrong[i]][:20]
    ex = pd.DataFrame(dict(
        title=test["title"].values[order], brand=test["brand"].values[order],
        true=[", ".join(np.array(classes)[y[i] == 1]) for i in order],
        predicted=[", ".join(np.array(classes)[pred[i] == 1]) or "-" for i in order],
        worst_label=[classes[worst_class[i]] for i in order],
        prob=[round(float(proba[i, worst_class[i]]), 3) for i in order]))
    ex.to_csv("data/processed/misclassified_examples.csv", index=False)
    short = ex.head(8).copy()
    short["title"] = short["title"].str.slice(0, 38)
    short["brand"] = short["brand"].str.slice(0, 12)
    short = short[["title", "brand", "true", "predicted"]]
    short.columns = ["Product title", "Brand", "True labels", "Predicted"]
    df_to_tex(short, "report/tables/errors.tex", col_format="p{4.2cm}p{2.4cm}p{3.4cm}p{3.6cm}")
    print(f"\nTEST macro F1 {f:.3f}, AUC {overall['auc']:.3f}; {int(wrong.sum())} of {len(test)} products "
          f"have at least one wrong label")


if __name__ == "__main__":
    main()