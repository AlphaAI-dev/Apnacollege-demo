#!/usr/bin/env python3
"""
Batch-simulate WorldQuant BRAIN alpha candidates through the BRAIN API and
rank them by fitness. Nothing is submitted unless you run --submit yourself.

Setup:
    pip install requests
    export BRAIN_EMAIL="you@example.com"
    export BRAIN_PASSWORD="your-password"     # or leave unset to be prompted

Usage:
    python brain_search.py                    # simulate every candidate
    python brain_search.py --only quality     # only candidates whose name contains "quality"
    python brain_search.py --list             # print candidate names and exit
    python brain_search.py --submit ALPHA_ID  # submit one alpha after checking it on the website

Results are written to results/brain_<timestamp>.csv, sorted by fitness.
"""
import argparse
import csv
import getpass
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

API = "https://api.worldquantbrain.com"
PLATFORM = "https://platform.worldquantbrain.com"
MAX_CONCURRENT = 3  # BRAIN runs at most 3 simulations at a time per user

# Fitness ratings from the "Simulation results" doc.
RATINGS = [(2.5, "Spectacular"), (2.0, "Excellent"), (1.5, "Good"), (1.0, "Average")]

DEFAULTS = dict(
    region="USA", delay=1, universe="TOP500", neutralization="INDUSTRY",
    decay=6, truncation=0.05, pasteurization="ON", nanHandling="ON",
)

# Variable names are chosen to avoid data-field names (e.g. "eps" is a field,
# so it is a reserved word in Fast Expression).

# BRAIN rejects unused variables, so each template only defines what its last line uses.
VR = "vr  = ts_backfill(implied_volatility_call_120 / {rv}, 5);\n"
VAL = "val = -ts_rank(enterprise_value / ts_backfill({cf}, 60), {win});\n"
STAB = "stab = -ts_std_dev(ts_backfill(scl12_buzz, 5), 5);\n"


def volval(w=0.5, rv="parkinson_volatility_120", cf="cashflow_op", win=126):
    """Round 1's best alpha (volval_tsrank: fitness 1.78, self-corr 0.66) with knobs."""
    return (VR.format(rv=rv) + VAL.format(cf=cf, win=win)
            + f"{w} * group_rank(ts_mean(vr, 5), industry) + {round(1 - w, 2)} * group_rank(val, industry) - 0.5")


# Round 1 settings for volval_tsrank, which every variant below changes one thing from.
VT = dict(neutralization="SECTOR", decay=4, truncation=0.05)

CANDIDATES = [
    # Change one setting at a time.
    dict(name="vt_trunc08", code=volval(), **{**VT, "truncation": 0.08}),
    dict(name="vt_decay3", code=volval(), **{**VT, "decay": 3}),
    dict(name="vt_decay6", code=volval(), **{**VT, "decay": 6}),
    dict(name="vt_industry", code=volval(), **{**VT, "neutralization": "INDUSTRY"}),
    dict(name="vt_top1000", code=volval(), universe="TOP1000", **VT),
    # Change one part of the expression at a time.
    dict(name="vt_w60_vol", code=volval(w=0.6), **VT),
    dict(name="vt_w40_vol", code=volval(w=0.4), **VT),
    dict(name="vt_win252", code=volval(win=252), **VT),
    dict(name="vt_rv60", code=volval(rv="parkinson_volatility_60"), **VT),
    dict(name="vt_cashflow", code=volval(cf="cashflow"), **VT),
    dict(name="vt_plus_stab",
         code=VR.format(rv="parkinson_volatility_120") + VAL.format(cf="cashflow_op", win=126) + STAB
         + "0.4 * group_rank(ts_mean(vr, 5), industry) + 0.4 * group_rank(val, industry)"
           " + 0.2 * group_rank(stab, industry) - 0.5", **VT),
    # Each leg alone, to see which one carries the Sharpe.
    dict(name="leg_vr", code=VR.format(rv="parkinson_volatility_120")
         + "group_rank(ts_mean(vr, 5), industry) - 0.5", **VT),
    dict(name="leg_val", code=VAL.format(cf="cashflow_op", win=126)
         + "group_rank(val, industry) - 0.5", **VT),
    # New family for future days: profitability, accruals and margin trend.
    dict(name="qualityc",
         code="""gp  = ts_backfill((revenue - cogs) / assets, 60);
acc = -ts_backfill((net_income - cashflow_op) / assets, 60);
imp = ts_delta(ts_backfill(operating_income / assets, 60), 252);
0.4 * group_rank(gp, industry) + 0.3 * group_rank(acc, industry) + 0.3 * group_rank(imp, industry) - 0.5""",
         decay=2),
]


def rating(fitness):
    for cutoff, label in RATINGS:
        if fitness is not None and fitness > cutoff:
            return label
    return "Below average"


class Brain:
    def __init__(self, email, password):
        self.session = requests.Session()
        self.session.auth = (email, password)
        self.lock = threading.Lock()
        self.login()

    def login(self):
        r = self.session.post(f"{API}/authentication")
        body = r.json() if r.content else {}
        if "user" in body:
            print("Logged in to BRAIN.")
            return
        if "inquiry" in body:
            # BRAIN sometimes asks for a biometric check before API access.
            input(f"Complete the biometric check at {r.url}/persona?inquiry={body['inquiry']} "
                  "then press Enter...")
            self.session.post(f"{r.url}/persona", json=body)
            print("Logged in to BRAIN.")
            return
        sys.exit(f"Login failed ({r.status_code}): {body}")

    def request(self, method, url, **kwargs):
        """Send a request, re-logging in once on 401 and waiting out 429s."""
        for _ in range(20):
            r = self.session.request(method, url, timeout=60, **kwargs)
            if r.status_code == 401:
                with self.lock:
                    self.login()
                continue
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 10)))
                continue
            return r
        raise RuntimeError(f"Gave up on {method} {url}")

    def wait(self, url):
        """Poll a progress URL until it stops returning Retry-After."""
        while True:
            r = self.request("GET", url)
            retry = float(r.headers.get("Retry-After", 0))
            if retry == 0:
                return r
            time.sleep(retry)

    def simulate(self, cand):
        settings = {**DEFAULTS, **{k: v for k, v in cand.items() if k not in ("name", "code")}}
        payload = {
            "type": "REGULAR",
            "regular": cand["code"],
            "settings": {
                "instrumentType": "EQUITY",
                "region": settings["region"],
                "universe": settings["universe"],
                "delay": settings["delay"],
                "decay": settings["decay"],
                "neutralization": settings["neutralization"],
                "truncation": settings["truncation"],
                "pasteurization": settings["pasteurization"],
                "unitHandling": "VERIFY",
                "nanHandling": settings["nanHandling"],
                "language": "FASTEXPR",
                "visualization": False,
            },
        }
        row = dict(name=cand["name"], universe=settings["universe"],
                   neutralization=settings["neutralization"], decay=settings["decay"],
                   truncation=settings["truncation"], code=cand["code"])

        r = self.request("POST", f"{API}/simulations", json=payload)
        if "Location" not in r.headers:
            return {**row, "error": f"{r.status_code} {r.text[:200]}"}
        result = self.wait(r.headers["Location"]).json()
        if "alpha" not in result:
            return {**row, "error": result.get("message") or result.get("status") or str(result)[:200]}

        alpha_id = result["alpha"]
        alpha = self.request("GET", f"{API}/alphas/{alpha_id}").json()
        stats = alpha.get("is", {})
        checks = stats.get("checks", [])
        failed = [c["name"] for c in checks if c.get("result") == "FAIL"]
        row.update(
            alpha_id=alpha_id,
            link=f"{PLATFORM}/alpha/{alpha_id}",
            sharpe=stats.get("sharpe"),
            fitness=stats.get("fitness"),
            turnover=round(100 * stats["turnover"], 2) if stats.get("turnover") is not None else None,
            returns=round(100 * stats["returns"], 2) if stats.get("returns") is not None else None,
            drawdown=round(100 * stats["drawdown"], 2) if stats.get("drawdown") is not None else None,
            failed_checks=" ".join(failed),
            self_corr=self.self_correlation(alpha_id),
        )
        row["rating"] = rating(row["fitness"])
        return row

    def self_correlation(self, alpha_id):
        try:
            data = self.wait(f"{API}/alphas/{alpha_id}/correlations/self").json()
            return data.get("max")
        except Exception:
            return None  # check it on the website instead

    def submit(self, alpha_id):
        self.request("POST", f"{API}/alphas/{alpha_id}/submit")
        r = self.wait(f"{API}/alphas/{alpha_id}/submit")
        if r.status_code == 404:
            print("Not found: the alpha may already be submitted.")
            return
        checks = r.json().get("is", {}).get("checks", []) if r.content else []
        for c in checks:
            print(f"  {c.get('result', '?'):8} {c.get('name')} {c.get('value', '')}")
        failed = [c for c in checks if c.get("result") == "FAIL"]
        print("Submission FAILED." if failed else "Submitted.")


FIELDS = ["name", "rating", "fitness", "sharpe", "turnover", "returns", "drawdown",
          "self_corr", "failed_checks", "universe", "neutralization", "decay",
          "truncation", "alpha_id", "link", "error", "code"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="only simulate candidates whose name contains this text")
    ap.add_argument("--list", action="store_true", help="list candidate names and exit")
    ap.add_argument("--submit", metavar="ALPHA_ID", help="submit one alpha and exit")
    args = ap.parse_args()

    if args.list:
        for c in CANDIDATES:
            print(c["name"])
        return

    email = os.environ.get("BRAIN_EMAIL") or input("BRAIN email: ")
    password = os.environ.get("BRAIN_PASSWORD") or getpass.getpass("BRAIN password: ")
    brain = Brain(email, password)

    if args.submit:
        if input(f"Submit {args.submit}? This uses today's submission. Type yes: ").strip() == "yes":
            brain.submit(args.submit)
        return

    todo = [c for c in CANDIDATES if not args.only or args.only in c["name"]]
    print(f"Simulating {len(todo)} candidates, {MAX_CONCURRENT} at a time...")
    rows = []
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
        futures = {pool.submit(brain.simulate, c): c["name"] for c in todo}
        for fut in as_completed(futures):
            try:
                row = fut.result()
            except Exception as e:
                row = dict(name=futures[fut], error=f"{type(e).__name__}: {e}")
            rows.append(row)
            if row.get("error"):
                print(f"  {row['name']:24} ERROR {row['error']}")
            else:
                print(f"  {row['name']:24} fitness {row['fitness']}  sharpe {row['sharpe']}  "
                      f"turnover {row['turnover']}%  self-corr {row['self_corr']}  {row['rating']}")

    rows.sort(key=lambda r: r.get("fitness") or -99, reverse=True)
    os.makedirs("results", exist_ok=True)
    out = f"results/brain_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"\nSaved {out}\n\nExcellent or better with no failed checks:")
    winners = [r for r in rows if (r.get("fitness") or 0) > 2 and not r.get("failed_checks")
               and (r.get("self_corr") is None or r["self_corr"] < 0.7)]
    for r in winners:
        print(f"  {r['name']:24} fitness {r['fitness']}  {r['link']}")
    if not winners:
        print("  none yet. Send the CSV back and pick the next round of changes from it.")


if __name__ == "__main__":
    main()
