#!/usr/bin/env python3
"""
Search for WorldQuant BRAIN alphas through the BRAIN API: simulate every idea,
tune the best few one setting at a time, then print only the alphas that pass
every check, with their code and settings ready to paste into BRAIN.
Nothing is submitted unless you run --submit yourself.

Setup:
    pip install requests
    export BRAIN_EMAIL="you@example.com"
    export BRAIN_PASSWORD="your-password"     # or leave unset to be prompted

Usage:
    python brain_search.py                    # search for Spectacular (fitness 2.5+) alphas
    python brain_search.py --target 2.0       # accept Excellent too
    python brain_search.py --only ovp         # only ideas whose name contains "ovp"
    python brain_search.py --list             # print idea names and exit
    python brain_search.py --submit ALPHA_ID  # submit one alpha after checking it on the website

The pasteable alphas are saved to results/winners_<timestamp>.txt and every
simulation to results/brain_<timestamp>.csv.
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


def volval(w=0.5, rv="parkinson_volatility_120", cf="cashflow_op", win=126, smooth=5, grp="industry"):
    """Round 1's best alpha (volval_tsrank: fitness 1.78, self-corr 0.66) with knobs."""
    return (VR.format(rv=rv) + VAL.format(cf=cf, win=win)
            + f"{w} * group_rank(ts_mean(vr, {smooth}), {grp}) + {round(1 - w, 2)} * group_rank(val, {grp}) - 0.5")


# Round 4's new ideas all had Sharpe under 0.75. Round 5 mixes the alphas that
# already worked (each one submitted). Mixing n weakly related alphas cuts the
# mix's correlation with each of them to roughly sqrt((1 + (n-1)*rho) / n) while
# Sharpe adds up, which is how a mix can pass self-correlation and score higher.
LEGS = {
    "vr":   "vr   = ts_backfill(implied_volatility_call_120 / parkinson_volatility_120, 5);",
    "val":  "val  = -ts_rank(enterprise_value / ts_backfill(cashflow_op, 60), 126);",
    "oey":  "oey  = ts_rank(ts_backfill(operating_income, 60) / cap, 252);",
    "lev":  "lev  = ts_backfill(liabilities, 60) / ts_backfill(assets, 60);",
    "stab": "stab = -ts_std_dev(ts_backfill(scl12_buzz, 5), 5);",
    "skew": "skew = ts_backfill((implied_volatility_call_180 - implied_volatility_put_180)"
            " / implied_volatility_mean_180, 5);",
    "fcfq": "fcfq = ts_decay_linear(ts_scale(est_cashflow_op, 252), 22)"
            " - ts_decay_linear(ts_scale(est_capex, 252), 22);",
    "ovp":  "ovp  = -ts_corr(ts_backfill(est_ptp, 20), ts_backfill(est_fcf, 20), 63);",
}
# Each family is a 0-1 score built the way its submitted alpha was.
FAMILIES = {
    "A": [("vr", 0.5, 5), ("val", 0.5, 0)],                    # vol/valuation (fitness 1.98)
    "B": [("oey", 0.5, 0), ("lev", 0.3, 0), ("stab", 0.2, 0)],  # quality (1.67)
    "C": [("skew", 0.5, 5), ("fcfq", 0.5, 0)],                  # skew + FCF (Sharpe 1.89)
    "E": [("ovp", 1.0, 0)],                                     # new, weak but uncorrelated
}


# Round 5's mixes were Excellent but over 0.7 self-correlation: parts A and B are exact
# copies of submitted alphas. Round 6 applies the doc's three decorrelation moves to
# every part: a similar field, a similar operator, and a different grouping.
# Every field here already ran without errors in earlier rounds.
LEGS_ALT = {
    **LEGS,
    "vr":   "vr   = ts_backfill(implied_volatility_mean_120 / parkinson_volatility_60, 5);",
    "val":  "val  = -ts_zscore(enterprise_value / ts_backfill(cashflow, 60), 252);",
    "oey":  "oey  = ts_zscore(ts_backfill(operating_income, 60) / cap, 252);",
    "lev":  "lev  = ts_rank(ts_backfill(liabilities, 60) / ts_backfill(assets, 60), 252);",
    "stab": "stab = -ts_std_dev(ts_backfill(scl12_buzz, 5), 10);",
}


def mix(alt=False, grp="industry", **weights):
    """Weighted mix of families, e.g. mix(A=0.3, B=0.3, C=0.4)."""
    legs = LEGS_ALT if alt else LEGS
    terms = []
    for fam, fw in weights.items():
        for leg, lw, smooth in FAMILIES[fam]:
            x = f"ts_mean({leg}, {smooth})" if smooth else leg
            terms.append((leg, round(fw * lw, 3), f"group_rank({x}, {grp})"))
    defs = "\n".join(legs[leg] for leg, _, _ in terms)
    body = " + ".join(f"{w} * {gr}" for _, w, gr in terms)
    return f"{defs}\n{body} - 0.5"


EQUAL = dict(A=0.25, B=0.25, C=0.25, E=0.25)
CANDIDATES = [
    dict(name="r6_alt_equal", code=mix(alt=True, **EQUAL)),
    dict(name="r6_alt_equal_sector", code=mix(alt=True, grp="sector", **EQUAL)),
    dict(name="r6_alt_equal_subind", code=mix(alt=True, grp="subindustry", **EQUAL),
         neutralization="SUBINDUSTRY"),
    dict(name="r6_alt_equal_market", code=mix(alt=True, **EQUAL), neutralization="MARKET"),
    dict(name="r6_alt_equal_top1000", code=mix(alt=True, **EQUAL), universe="TOP1000"),
    dict(name="r6_alt_lowAB", code=mix(alt=True, A=0.15, B=0.15, C=0.4, E=0.3)),
    dict(name="r6_alt_lowAB_sector", code=mix(alt=True, grp="sector", A=0.15, B=0.15, C=0.4, E=0.3)),
    dict(name="r6_alt_minAB", code=mix(alt=True, A=0.1, B=0.1, C=0.5, E=0.3)),
    dict(name="r6_alt_noE", code=mix(alt=True, A=0.34, B=0.33, C=0.33)),
    dict(name="r6_alt_lowA", code=mix(alt=True, A=0.1, B=0.25, C=0.35, E=0.3)),
]
for c in CANDIDATES:
    c.setdefault("decay", 4)


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
        settings = settings_of(cand)
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
        # Self-correlation above 0.7 still passes when Sharpe beats the correlated alpha
        # by 10%. Only BRAIN's submission check knows that, so ask it for strong alphas.
        if (row["self_corr"] or 0) >= 0.7 and (row["fitness"] or 0) >= 2.0:
            row["corr_check"] = self.corr_check(alpha_id)
        return row

    def corr_check(self, alpha_id):
        """Result of the SELF_CORRELATION row of the "Check Submission" test, or None."""
        try:
            data = self.wait(f"{API}/alphas/{alpha_id}/check").json()
            for c in data.get("is", {}).get("checks", []):
                if c.get("name") == "SELF_CORRELATION":
                    return c.get("result")
        except Exception:
            pass
        return None

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

# Settings the search tunes, one at a time, for the most promising families.
SEARCH_SPACE = [
    ("neutralization", ["MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY"]),
    ("decay", [2, 4, 6, 8]),
    ("universe", ["TOP200", "TOP500", "TOP1000"]),
]


def settings_of(cand):
    return {**DEFAULTS, **{k: v for k, v in cand.items() if k not in ("name", "code", "family")}}


def passes(row):
    """True when no check failed and self-correlation passes (below 0.7, or BRAIN's
    check says the Sharpe is 10% better than the alpha it correlates with)."""
    if row.get("error") or row.get("fitness") is None or row.get("failed_checks"):
        return False
    return row.get("self_corr") is None or row["self_corr"] < 0.7 or row.get("corr_check") == "PASS"


def score(row):
    if row.get("error") or row.get("fitness") is None:
        return -99
    return row["fitness"] if passes(row) else row["fitness"] - 10


def run_batch(brain, cands, cache):
    """Simulate candidates 3 at a time, skipping ones already simulated this run."""
    keys = [(c["code"], tuple(sorted(settings_of(c).items()))) for c in cands]
    todo = {k: c for k, c in zip(keys, cands) if k not in cache}
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
        futures = {pool.submit(brain.simulate, c): k for k, c in todo.items()}
        for fut in as_completed(futures):
            try:
                row = fut.result()
            except Exception as e:
                row = dict(name=todo[futures[fut]]["name"], error=f"{type(e).__name__}: {e}")
            cache[futures[fut]] = row
            if row.get("error"):
                print(f"  {row['name']:40} ERROR {row['error'][:80]}")
            else:
                print(f"  {row['name']:40} fitness {row['fitness']}  sharpe {row['sharpe']}  "
                      f"self-corr {row['self_corr']}  {'ok' if passes(row) else 'FAILS'}")
    return [cache[k] for k in keys]


def paste_block(cand, row):
    s = settings_of(cand)
    corr = row["self_corr"] if row.get("self_corr") is not None else "check on website"
    return (f"=== {row['rating'].upper()}  fitness {row['fitness']}  sharpe {row['sharpe']}  "
            f"turnover {row['turnover']}%  self-corr {corr}\n"
            f"Settings: Region {s['region']} | Delay {s['delay']} | Universe {s['universe']} | "
            f"Neutralization {s['neutralization'].title()} | Decay {s['decay']} | "
            f"Truncation {s['truncation']} | Pasteurization {s['pasteurization'].title()} | "
            f"Unit Handling Verify | NaN Handling {s['nanHandling'].title()} | Test Period 1 year\n"
            f"Code:\n{cand['code']}\n"
            f"Link: {row['link']}\n")


def search(brain, cands, target, top):
    cache = {}
    print(f"Step 1: simulating {len(cands)} alpha ideas...")
    rows = run_batch(brain, cands, cache)
    best = {c["name"]: (c, r) for c, r in zip(cands, rows)}

    # Tune the most promising ideas one setting at a time, keeping whatever scores best.
    leaders = sorted(best, key=lambda n: score(best[n][1]), reverse=True)[:top]
    best = {n: best[n] for n in leaders if score(best[n][1]) > -99}
    for key, values in SEARCH_SPACE:
        trials = []
        for fam, (cand, _) in best.items():
            for v in values:
                if v != settings_of(cand)[key]:
                    trials.append((fam, {**cand, key: v, "name": f"{fam} {key}={v}"}))
        print(f"\nStep 2: tuning {key} for {', '.join(best)} ({len(trials)} simulations)...")
        for (fam, cand), row in zip(trials, run_batch(brain, [c for _, c in trials], cache)):
            if score(row) > score(best[fam][1]):
                best[fam] = (cand, row)

    all_rows = sorted(cache.values(), key=lambda r: r.get("fitness") or -99, reverse=True)
    finals = sorted(best.values(), key=lambda cr: score(cr[1]), reverse=True)
    return all_rows, finals


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="only use alpha ideas whose name contains this text")
    ap.add_argument("--list", action="store_true", help="list alpha ideas and exit")
    ap.add_argument("--submit", metavar="ALPHA_ID", help="submit one alpha and exit")
    ap.add_argument("--target", type=float, default=2.5, help="fitness to aim for (default 2.5, Spectacular)")
    ap.add_argument("--top", type=int, default=4, help="how many ideas to tune after step 1 (default 4)")
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

    cands = [c for c in CANDIDATES if not args.only or args.only in c["name"]]
    all_rows, finals = search(brain, cands, args.target, args.top)

    os.makedirs("results", exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    with open(f"results/brain_{stamp}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    # One alpha per idea: alphas from the same idea correlate with each other once one is submitted.
    hits = [(c, r) for c, r in finals if passes(r) and r["fitness"] >= args.target]
    near = [(c, r) for c, r in finals if passes(r) and 2.0 <= r["fitness"] < args.target]
    lines = []
    if hits:
        lines.append(f"ALPHAS AT FITNESS {args.target}+ THAT PASS EVERY CHECK (one per idea):\n"
                     "Submit the top one. Ideas that share a part (e.g. ovp) correlate with each other,\n"
                     "so re-run before submitting a second one to re-check its self-correlation.\n")
        lines += [paste_block(c, r) for c, r in hits]
    else:
        lines.append(f"No alpha reached fitness {args.target} this run.\n")
    if near:
        lines.append("Excellent alphas that pass every check (best found below the target):\n")
        lines += [paste_block(c, r) for c, r in near]
    if not hits and not near:
        lines.append("Nothing passed at fitness 2.0+. Paste this output to Claude for the next round of ideas.")
    text = "\n".join(lines)
    with open(f"results/winners_{stamp}.txt", "w") as f:
        f.write(text)
    print("\n" + "=" * 70 + "\n" + text + f"\nSaved results/winners_{stamp}.txt and results/brain_{stamp}.csv")


if __name__ == "__main__":
    main()
