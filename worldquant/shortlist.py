#!/usr/bin/env python3
"""
Rank the alphas brain_auto.py has run, the way the WorldQuant session said to
judge them (the Good/Excellent/Spectacular labels are only fitness bands):

  1. passed BRAIN's Check Submission (status READY TO SUBMIT);
  2. lower self-correlation (a stand-in for production correlation, which the
     API doesn't report: check it on the alpha's page before submitting);
  3. a dataset you haven't got a ready alpha from yet (more pyramids);
  4. turnover under 25%;
  5. then Sharpe.

Usage:
    python3 shortlist.py            # top 10 from results/all_alphas.csv
    python3 shortlist.py 20         # top 20
"""
import csv
import sys

CSV_FILE = "results/all_alphas.csv"


def num(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main():
    top = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    with open(CSV_FILE) as f:
        rows = [r for r in csv.DictReader(f) if r.get("status", "").startswith("READY")]
    seen, ranked = set(), []
    # Best first by correlation and Sharpe, then pull one alpha per dataset to the front.
    rows.sort(key=lambda r: (num(r.get("self_corr"), 1), -num(r.get("sharpe"))))
    for r in rows:
        r["new_dataset"] = r.get("dataset") not in seen
        seen.add(r.get("dataset"))
        ranked.append(r)
    ranked.sort(key=lambda r: (not r["new_dataset"], num(r.get("turnover")) > 25, num(r.get("self_corr"), 1)))
    if not ranked:
        print("No alpha has passed Check Submission yet. Keep brain_auto.py running.")
        return
    print(f"{len(ranked)} alphas passed Check Submission. Top {min(top, len(ranked))}, best first:\n")
    for i, r in enumerate(ranked[:top], 1):
        print(f"{i}. {r['alpha_id']}  Sharpe {r['sharpe']}  fitness {r['fitness']}  turnover {r['turnover']}%  "
              f"self-corr {r['self_corr']}  dataset {r.get('dataset', '?')} ({r.get('category', '?')})"
              f"{'  NEW DATASET' if r['new_dataset'] else ''}\n   {r['link']}\n"
              f"   Settings: {r.get('region', 'USA')} | Delay 1 | {r['universe']} | {r['neutralization'].title()} | "
              f"Decay {r['decay']} | Truncation {r['truncation']}\n   {r['code']}\n")
    print("Before submitting: open the link, check Production Correlation (lower is better) and that no year\n"
          "has Sharpe far below 1, then click Check Submission. Submit one per dataset.")


if __name__ == "__main__":
    main()
