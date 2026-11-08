"""Build the report's Contribution Table straight from the Git history.

Run from the repo root:  python src/contrib_table.py [--since 2026-10-25]
Output: report/tables/contributions.tex  (one row per commit: author, hash, files, message)
"""
import argparse
import os
import subprocess
from report_utils import tex_escape

# map e-mail (lower case) -> display name, so one person never appears twice
AUTHORS = {"romeshaafzaal72@gmail.com": "Romesha Afzaal", "nadeemaliza1512@gmail.com": "Aliza Nadeem"}
IGNORED = ("report/Deliverable", "figures/", "data/")  # generated artefacts are not "code modules"


def main(since):
    out = subprocess.run(["git", "log", f"--since={since}", "--reverse", "--name-only",
                          "--format=@@%h|%ae|%s"], capture_output=True, text=True, check=True).stdout
    commits, cur = [], None
    for line in out.splitlines():
        if line.startswith("@@"):
            h, email, subject = line[2:].split("|", 2)
            cur = dict(hash=h, author=AUTHORS.get(email.lower(), email), subject=subject, files=[])
            commits.append(cur)
        elif line.strip() and cur is not None and not line.startswith(IGNORED):
            cur["files"].append(line.strip())
    rows = [r"\begin{tabular}{p{2.6cm}p{1.3cm}p{4.4cm}p{6.2cm}}", r"\toprule",
            r"Author & Commit & Files / module & Message \\", r"\midrule"]
    for c in commits:
        files = ", ".join(c["files"]) or "(none)"
        rows.append(" & ".join([tex_escape(c["author"]), r"\texttt{" + c["hash"] + "}",
                                tex_escape(files), tex_escape(c["subject"])]) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    os.makedirs("report/tables", exist_ok=True)
    with open("report/tables/contributions.tex", "w", encoding="utf-8") as fh:
        fh.write("\n".join(rows) + "\n")
    by = {}
    for c in commits:
        by[c["author"]] = by.get(c["author"], 0) + 1
    print(f"{len(commits)} commits since {since}:", by)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-11-01")
    main(ap.parse_args().since)