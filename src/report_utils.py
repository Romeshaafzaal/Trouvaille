"""Small helpers that turn pandas tables into LaTeX so report numbers are never typed by hand."""
import os

_SPECIAL = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#", "$": r"\$", "{": r"\{", "}": r"\}"}


def tex_escape(value):
    """Escape characters that LaTeX treats as special."""
    return "".join(_SPECIAL.get(ch, ch) for ch in str(value))


def df_to_tex(df, path, col_format=None, decimals=3, bold_row=None):
    """Write a DataFrame as a booktabs tabular to `path`.

    bold_row: index label of a row to print in bold (e.g. the selected hyperparameters).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    col_format = col_format or "l" + "r" * (len(df.columns) - 1)
    lines = [rf"\begin{{tabular}}{{{col_format}}}", r"\toprule",
             " & ".join(tex_escape(c) for c in df.columns) + r" \\", r"\midrule"]
    for idx, row in df.iterrows():
        cells = []
        for v in row:
            cells.append(f"{v:.{decimals}f}" if isinstance(v, float) else tex_escape(v))
        line = " & ".join(cells)
        if bold_row is not None and idx == bold_row:
            line = " & ".join(rf"\textbf{{{c}}}" for c in cells)
        lines.append(line + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")