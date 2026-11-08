"""Phase 3/4 plots and tables: hyperparameter search results and subgroup bias check.

Run from the repo root after src/mlp.py, src/evaluate.py and src/error_analysis.py:
    python src/plot_results.py
Outputs: figures/tuning_f1.png, figures/bias_check.png, report/tables/tuning.tex,
         report/tables/bias.tex, report/tables/best_params.tex
"""
import json
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from report_utils import df_to_tex


def tuning():
    t = pd.read_csv("data/processed/mlp_tuning.csv").sort_values("f1", ascending=False).reset_index(drop=True)
    best = t.iloc[0]
    with open("report/tables/best_params.tex", "w") as fh:
        fh.write("\n".join([
            rf"\renewcommand{{\bestLR}}{{{best['lr']:g}}}", rf"\renewcommand{{\bestBatch}}{{{int(best['batch'])}}}",
            rf"\renewcommand{{\bestDrop}}{{{best['dropout']:g}}}", rf"\renewcommand{{\bestValF}}{{{best['f1']:.3f}}}",
            rf"\renewcommand{{\bestValAUC}}{{{best['auc']:.3f}}}", rf"\renewcommand{{\nConfigs}}{{{len(t)}}}"]) + "\n")
    table = t.rename(columns={"lr": "LR", "batch": "Batch", "dropout": "Dropout", "precision": "Precision",
                              "recall": "Recall", "f1": "F1", "auc": "AUC"})
    table["Batch"] = table["Batch"].astype(int)
    table["LR"] = table["LR"].map(lambda v: f"{v:g}")
    table["Dropout"] = table["Dropout"].map(lambda v: f"{v:g}")
    df_to_tex(table, "report/tables/tuning.tex", col_format="lrrrrrr", bold_row=0)

    labels = [f"lr={r.lr:g}, bs={int(r.batch)}, do={r.dropout:g}" for r in t.itertuples()]
    plt.figure(figsize=(6.5, 4.2))
    colors = ["tab:green"] + ["tab:blue"] * (len(t) - 1)
    plt.barh(labels[::-1], t["f1"][::-1], color=colors[::-1])
    plt.xlabel("Validation macro F1"); plt.title("Hyperparameter search (validation set)")
    plt.tight_layout(); plt.savefig("figures/tuning_f1.png", dpi=150); plt.close()


def bias():
    b = pd.read_csv("data/processed/bias_check.csv")
    overall = None
    if os.path.exists("data/processed/test_overall.json"):
        overall = json.load(open("data/processed/test_overall.json"))["f1"]
    b["label"] = b["group"] + " = " + b["value"].astype(str)
    b = b.sort_values(["group", "f1"]).reset_index(drop=True)
    plt.figure(figsize=(6.5, 0.28 * len(b) + 1.2))
    plt.barh(b["label"], b["f1"], color="tab:purple")
    if overall is not None:
        plt.axvline(overall, color="k", ls="--", lw=1, label=f"overall test F1 = {overall:.3f}")
        plt.legend(fontsize=8, loc="lower right")
    plt.xlabel("Macro F1 on subgroup"); plt.title("Bias check: F1 by subgroup (test set)")
    plt.tight_layout(); plt.savefig("figures/bias_check.png", dpi=150); plt.close()

    out = b[["group", "value", "n", "f1"]].copy()
    out.columns = ["Attribute", "Group", "n", "Macro F1"]
    if overall is not None:
        out["Gap vs overall"] = out["Macro F1"] - overall
    df_to_tex(out, "report/tables/bias.tex", col_format="llrrr" if overall is not None else "llrr")
    spread = b["f1"].max() - b["f1"].min()
    print(f"Subgroup F1 range: {b['f1'].min():.3f} - {b['f1'].max():.3f} (spread {spread:.3f})")


if __name__ == "__main__":
    os.makedirs("report/tables", exist_ok=True)
    tuning()
    bias()