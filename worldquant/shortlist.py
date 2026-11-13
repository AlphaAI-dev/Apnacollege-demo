#!/usr/bin/env python3
"""
Rank the alphas brain_auto.py has run, the way the WorldQuant session said to
judge them (the Good/Excellent/Spectacular labels are only fitness bands):

  1. passed BRAIN's Check Submission (status READY TO SUBMIT);
  2. a dataset you haven't got a ready alpha from yet (more pyramids);
  3. turnover under 25%;
  4. lower self-correlation (a stand-in for production correlation, which the
     API doesn't report: check it on the alpha's page before submitting).

Alphas you've already submitted, and ones built on fields several submitted
alphas already use, are left out. Add IDs you submit to results/submitted.txt
(one per line) so they stop appearing.

Usage:
    python3 shortlist.py            # top 10
    python3 shortlist.py 20         # top 20
"""
import csv
import os
import re
import sys

FILES = ["results/all_alphas.csv", "results/all_alphas_old.csv"]
NEW = ["time", "alpha_id", "link", "rating", "fitness", "sharpe", "turnover", "self_corr", "status",
       "region", "dataset", "category", "fields", "operators", "universe", "neutralization", "decay",
       "truncation", "code"]
OLD = [c for c in NEW if c not in ("region", "dataset", "category")]
# Submitted before this list existed, and the fields they already use.
SUBMITTED = {"78LAxOA1", "mLovdwX9", "N1PXQb9g", "d5a2ZOlj", "KPg3blVp", "78LpvnO5"}
USED_FIELDS = ["operating_income", "cashflow_op", "liabilities", "enterprise_value", "assets"]


def num(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_rows():
    rows = []
    for path in FILES:
        if not os.path.exists(path):
            continue
        with open(path, newline="") as f:
            for raw in csv.reader(f):
                if not raw or raw[0] == "time":
                    continue
                # Rows from both file layouts can be mixed in one file: tell them apart by length.
                cols = NEW if len(raw) == len(NEW) else OLD if len(raw) == len(OLD) else None
                if cols:
                    rows.append(dict(zip(cols, raw)))
    return rows


def main():
    top = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    submitted = set(SUBMITTED)
    if os.path.exists("results/submitted.txt"):
        submitted |= {line.strip() for line in open("results/submitted.txt") if line.strip()}
    rows, seen_ids = [], set()
    for r in read_rows():
        if not r["status"].startswith("READY") or r["alpha_id"] in submitted or r["alpha_id"] in seen_ids:
            continue
        if any(re.search(rf"\b{f}\b", r["code"]) for f in USED_FIELDS):
            continue
        seen_ids.add(r["alpha_id"])
        rows.append(r)
    rows.sort(key=lambda r: (num(r.get("self_corr"), 1), -num(r.get("sharpe"))))
    seen = set()
    for r in rows:
        ds = r.get("dataset") or "older run"
        r["new_dataset"] = ds not in seen and ds != "older run"
        seen.add(ds)
    rows.sort(key=lambda r: (not r["new_dataset"], num(r.get("turnover")) > 25, num(r.get("self_corr"), 1)))
    if not rows:
        print("No new alpha has passed Check Submission yet. Keep brain_auto.py running.")
        return
    print(f"{len(rows)} alphas you haven't submitted passed Check Submission. Top {min(top, len(rows))}:\n")
    for i, r in enumerate(rows[:top], 1):
        print(f"{i}. {r['alpha_id']}  Sharpe {r['sharpe']}  fitness {r['fitness']}  turnover {r['turnover']}%  "
              f"self-corr {r['self_corr']}  dataset {r.get('dataset', 'older run')} ({r.get('category', '-')})"
              f"{'  NEW DATASET' if r['new_dataset'] else ''}\n   {r['link']}\n"
              f"   Settings: {r.get('region', 'USA')} | Delay 1 | {r['universe']} | {r['neutralization'].title()} | "
              f"Decay {r['decay']} | Truncation {r['truncation']}\n   {r['code']}\n")
    print("Before submitting: open the link, check Production Correlation (lower is better) and that no year\n"
          "has Sharpe far below 1, then click Check Submission. Submit one per dataset, and add each\n"
          "submitted ID to results/submitted.txt.")


if __name__ == "__main__":
    main()
