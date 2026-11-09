#!/usr/bin/env python3
"""
Run the day's BRAIN simulations and record every alpha, so nothing gets lost.

Every alpha it builds stays inside the programme rules: at most 4 data fields
and at most 8 operators (counted conservatively: every function call and every
+ - * / sign). It runs them in multi-simulations of up to 10, keeps changing the
best ones (a part, a window, a weight, the grouping or a setting), and stops
when today's simulation target is met (default 300, BRAIN day in US Eastern).

Nothing is submitted. Instead, for every alpha it:
  * names and tags it on BRAIN (e.g. "auto_SPECTACULAR_2.61", tags auto + rating),
    so the Alphas page can filter by name to find it;
  * adds a line to results/all_alphas.csv (link, scores, settings, code, status);
  * adds Excellent and Spectacular ones to results/excellent_alphas.txt with full
    settings and code, marked READY or with the reason they can't be submitted.
At the end, results/ready_to_submit.txt lists the ones that passed BRAIN's own
Check Submission (every row green, self-correlation included), best first.

Usage:
    python3 brain_auto.py                   # 300 simulations today
    python3 brain_auto.py --daily-sims 500
Stop any time with Control+C; a rerun the same day continues towards the target.
"""
import argparse
import copy
import csv
import getpass
import json
import os
import random
import re
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from zoneinfo import ZoneInfo

from brain_search import MAX_CONCURRENT, Brain, rating

STATE_FILE = "results/auto_state_v2.json"
LOG_FILE = "results/auto_log.txt"
CSV_FILE = "results/all_alphas.csv"
EXCELLENT_FILE = "results/excellent_alphas.txt"
READY_FILE = "results/ready_to_submit.txt"
MAX_FIELDS, MAX_OPS = 4, 8

# Each part is one small expression with at most 2 data fields and 2 operators, so
# two parts plus the combining "group_rank(..) + w * group_rank(..)" stay within
# 4 fields and 8 operators. Every field here has already run on this account.
PARTS = {
    "vr":     dict(expr="ts_mean(implied_volatility_call_120 / parkinson_volatility_120, {d})", d=[5, 10, 20]),
    "val":    dict(expr="{op}(cashflow_op / enterprise_value, {d})", op=["ts_rank", "ts_zscore"], d=[126, 252, 504]),
    "oey":    dict(expr="{op}(operating_income / cap, {d})", op=["ts_rank", "ts_zscore"], d=[126, 252, 504]),
    "lev":    dict(expr="ts_rank(liabilities / assets, {d})", d=[126, 252]),
    "fcf":    dict(expr="ts_rank(est_cashflow_op - est_capex, {d})", d=[126, 252]),
    "skew":   dict(expr="ts_mean(implied_volatility_call_180 - implied_volatility_put_180, {d})", d=[5, 10, 20]),
    "ovp":    dict(expr="-ts_corr(est_ptp, est_fcf, {d})", d=[63, 126]),
    "stab":   dict(expr="-ts_std_dev(scl12_buzz, {d})", d=[5, 10, 20]),
    "lowvol": dict(expr="-ts_std_dev(returns, {d})", d=[63, 126, 252]),
    "salesy": dict(expr="{op}(sales / cap, {d})", op=["ts_rank", "ts_zscore"], d=[126, 252]),
}
GROUPS = ["industry", "sector", "subindustry"]
WEIGHTS = [0.5, 0.75, 1, 1.5, 2]
SETTING_CHOICES = dict(universe=["TOP200", "TOP500", "TOP1000", "TOP3000"],
                       neutralization=["MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY"],
                       decay=[2, 3, 4, 6, 8], truncation=[0.05, 0.08])
GROUP_WORDS = {"industry", "sector", "subindustry", "market"}


def part(name, **params):
    p = {k: v[0] for k, v in PARTS[name].items() if k != "expr"}
    p.update(params)
    return [name, p]


def genome(p1, p2=None, w=1, grp="industry", **settings):
    base = dict(universe="TOP500", neutralization="INDUSTRY", decay=4, truncation=0.05)
    return dict(p1=p1, p2=p2, w=w, grp=grp, settings={**base, **settings})


# Starting points: the pairs behind the Excellent/Spectacular alphas from October,
# rebuilt to fit the 4-field / 8-operator limits, plus a few untried pairs.
SEEDS = [
    genome(part("vr"), part("oey", d=252), w=1.5, universe="TOP200"),
    genome(part("vr"), part("oey", d=252), w=1, neutralization="MARKET"),
    genome(part("val", d=252), part("oey", d=252), w=0.75),
    genome(part("skew"), part("fcf"), w=1, neutralization="MARKET"),
    genome(part("vr"), part("val", d=252), w=1.5),
    genome(part("oey", d=252), part("stab"), w=0.5, neutralization="SECTOR"),
    genome(part("val", d=252), part("lev"), w=0.5, neutralization="SUBINDUSTRY"),
    genome(part("lowvol", d=252), part("oey", d=252), w=1, grp="sector", neutralization="SECTOR"),
    genome(part("ovp"), part("val", d=252), w=1),
    genome(part("salesy", d=252), part("stab"), w=0.5),
]


def build(g):
    def expr(p):
        name, params = p
        return PARTS[name]["expr"].format(**params)
    first = f"group_rank({expr(g['p1'])}, {g['grp']})"
    if not g["p2"]:
        return first
    second = f"group_rank({expr(g['p2'])}, {g['grp']})"
    return f"{first} + {second}" if g["w"] == 1 else f"{first} + {g['w']} * {second}"


def count_fields_ops(code):
    """(data fields, operators) — operators counted conservatively."""
    calls = re.findall(r"\b[a-z_][a-z0-9_]*(?=\s*\()", code)
    names = set(re.findall(r"\b[a-z_][a-z0-9_]*\b(?!\s*\()", code))
    fields = {n for n in names if n not in GROUP_WORDS}
    signs = len(re.findall(r"[-+*/]", code))
    return len(fields), len(calls) + signs


def mutate(g):
    g = copy.deepcopy(g)
    for _ in range(random.choice([1, 1, 2])):
        kind = random.choice(["part", "part", "param", "param", "weight", "group", "setting", "setting", "drop"])
        slot = random.choice(["p1", "p2"]) if g["p2"] else "p1"
        if kind == "part":
            used = {g["p1"][0], (g["p2"] or [None])[0]}
            new = random.choice([n for n in PARTS if n not in used])
            params = {k: random.choice(v) for k, v in PARTS[new].items() if k != "expr"}
            g[slot if g["p2"] or random.random() < 0.5 else "p2"] = [new, params]
        elif kind == "param":
            name, params = g[slot]
            key = random.choice([k for k in PARTS[name] if k != "expr"])
            params[key] = random.choice(PARTS[name][key])
        elif kind == "weight" and g["p2"]:
            g["w"] = random.choice(WEIGHTS)
        elif kind == "group":
            g["grp"] = random.choice(GROUPS)
        elif kind == "drop" and g["p2"] and random.random() < 0.3:
            g["p2"] = None
        else:
            key = random.choice(list(SETTING_CHOICES))
            g["settings"][key] = random.choice(SETTING_CHOICES[key])
    return g


def brain_day():
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return dict(tried=[], pool=[], bad_tokens=[], sims_by_day={}, ready=[])


def save_state(state):
    with open(STATE_FILE + ".tmp", "w") as f:
        json.dump(state, f)
    os.replace(STATE_FILE + ".tmp", STATE_FILE)


def log(text, path=LOG_FILE):
    print(text)
    with open(path, "a") as f:
        f.write(text + "\n")


def corr_ok(row):
    return row.get("self_corr") is None or row["self_corr"] < 0.7 or row.get("corr_check") == "PASS"


def parent_score(entry):
    row = entry["row"]
    # Children of high-correlation alphas stay high-correlation, so discount those parents.
    return (row["fitness"] or 0) * (1.0 if corr_ok(row) else 0.4)


def pick_parent(state):
    pool = [e for e in state["pool"] if e["row"].get("fitness") is not None]
    if not pool or random.random() < 0.3:
        return random.choice(SEEDS)
    pool.sort(key=parent_score, reverse=True)
    return max(random.sample(pool[:20], min(4, len(pool[:20]))), key=parent_score)["genome"]


def next_candidate(state, tried):
    bad = set(state["bad_tokens"])
    for _ in range(500):
        g = mutate(pick_parent(state))
        code = build(g)
        fields, ops = count_fields_ops(code)
        if fields > MAX_FIELDS or ops > MAX_OPS:
            continue
        if bad and any(re.search(rf"\b{re.escape(t)}\b", code) for t in bad):
            continue
        k = code + "|" + json.dumps(g["settings"], sort_keys=True)
        if k not in tried:
            tried.add(k)
            return g, code, k
    return None


def full_check(brain, alpha_id):
    """(passed, names of rows that did not pass) from BRAIN's Check Submission."""
    for _ in range(4):
        checks = brain.check(alpha_id)
        results = {c.get("result") for c in checks}
        if checks and "PENDING" not in results:
            bad = [c.get("name") for c in checks if c.get("result") in ("FAIL", "ERROR")]
            return not bad, bad
        time.sleep(30)
    return False, ["CHECK_PENDING"]


def details(code, g, row, status):
    s = g["settings"]
    fields, ops = count_fields_ops(code)
    return (f"=== {status} | {row['rating'].upper()} fitness {row['fitness']} sharpe {row['sharpe']} "
            f"turnover {row.get('turnover')}% self-corr {row.get('self_corr')} | {fields} fields, "
            f"{row.get('op_count') or ops} operators\n{row['link']}\n"
            f"Settings: USA | Delay 1 | {s['universe']} | Neutralization {s['neutralization'].title()} | "
            f"Decay {s['decay']} | Truncation {s['truncation']} | Pasteurization On | "
            f"Unit Handling Verify | NaN Handling On | Test Period 1 year\n{code}\n")


CSV_COLUMNS = ["time", "alpha_id", "link", "rating", "fitness", "sharpe", "turnover", "self_corr", "status",
               "fields", "operators", "universe", "neutralization", "decay", "truncation", "code"]


def record_csv(row, g, code, status):
    new = not os.path.exists(CSV_FILE)
    fields, ops = count_fields_ops(code)
    with open(CSV_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(CSV_COLUMNS)
        s = g["settings"]
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M"), row.get("alpha_id"), row.get("link"),
                    row.get("rating"), row.get("fitness"), row.get("sharpe"), row.get("turnover"),
                    row.get("self_corr"), status, fields, row.get("op_count") or ops, s["universe"],
                    s["neutralization"], s["decay"], s["truncation"], code])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--daily-sims", type=int, default=300, help="simulations to run today (default 300)")
    ap.add_argument("--batch", type=int, default=10, help="alphas per multi-simulation (default 10, max 10)")
    args = ap.parse_args()
    args.batch = max(1, min(10, args.batch))

    os.makedirs("results", exist_ok=True)
    state = load_state()
    tried = set(state["tried"])
    email = os.environ.get("BRAIN_EMAIL") or input("BRAIN email: ")
    password = os.environ.get("BRAIN_PASSWORD") or getpass.getpass("BRAIN password: ")
    brain = Brain(email, password)
    brain.multi_ok = state.get("multi_ok", True)  # remembered from earlier runs

    def sims_today():
        return state["sims_by_day"].get(brain_day(), 0)

    log(f"\n--- {datetime.now():%Y-%m-%d %H:%M} start, BRAIN day {brain_day()}, "
        f"{sims_today()}/{args.daily_sims} simulations done today ---")
    log("Running. Each simulation takes 1-3 minutes, so the first results appear after a few minutes.")
    queued = errors_in_a_row = 0
    running = {}
    try:
        pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENT)  # no 'with': Control+C exits at once
        while running or sims_today() + queued < args.daily_sims:
            # Until we know whether multi-simulation is allowed, send one batch at a time.
            slots = MAX_CONCURRENT if "multi_ok" in state else 1
            while len(running) < slots and sims_today() + queued < args.daily_sims:
                batch = []
                size = args.batch if brain.multi_ok else 1  # without multi-simulation, 3 singles at a time
                while len(batch) < min(size, args.daily_sims - sims_today() - queued):
                    nxt = next_candidate(state, tried)
                    if nxt is None:
                        break
                    g, code, k = nxt
                    batch.append((g, code, k, dict(name="auto", code=code, **g["settings"])))
                if not batch:
                    break
                running[pool.submit(brain.simulate_many, [b[3] for b in batch])] = batch
                queued += len(batch)
            if not running:
                break
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in done:
                batch = running.pop(fut)
                queued -= len(batch)
                if len(batch) > 1:
                    state["multi_ok"] = brain.multi_ok
                try:
                    rows = fut.result()
                except Exception as e:
                    rows = [dict(error=f"{type(e).__name__}: {e}")] * len(batch)
                for (g, code, k, _), row in zip(batch, rows):
                    if row.get("error"):
                        # A failed request is not a simulation: don't count it, and allow a retry.
                        errors_in_a_row += 1
                        tried.discard(k)
                        log(f"  ERROR {' '.join(row['error'].split())[:100]}")
                        m = re.search(r'unknown (?:variable|operator) "(\w+)"', row["error"])
                        if m:
                            state["bad_tokens"].append(m.group(1))
                        continue
                    errors_in_a_row = 0
                    state["tried"].append(k)
                    state["sims_by_day"][brain_day()] = sims_today() + 1
                    state["pool"].append(dict(genome=g, row={x: row.get(x) for x in
                                                             ("fitness", "sharpe", "self_corr", "corr_check")}))
                    fit = row["fitness"] or 0
                    if (row.get("op_count") or 0) > MAX_OPS:
                        status = f"NOT ALLOWED: {row['op_count']} operators"
                    elif fit < 2.0:
                        status = "below Excellent"
                    elif row.get("failed_checks"):
                        status = "FAILS " + row["failed_checks"]
                    elif not corr_ok(row):
                        status = f"FAILS self-correlation {row['self_corr']}"
                    else:
                        ok, bad = full_check(brain, row["alpha_id"])
                        status = "READY TO SUBMIT" if ok else "FAILS " + " ".join(bad)
                    brain.tag(row["alpha_id"], f"auto_{row['rating'].upper().replace(' ', '_')}_{row['fitness']}",
                              ["auto", row["rating"].lower().replace(" ", "_")])
                    record_csv(row, g, code, status)
                    log(f"  [{sims_today():>3}/{args.daily_sims}] {row['rating']:<13} fitness {row['fitness']}  "
                        f"sharpe {row['sharpe']}  self-corr {row.get('self_corr')}  {row['alpha_id']}  {status}")
                    if fit >= 2.0:
                        log(details(code, g, row, status), EXCELLENT_FILE)
                    if status == "READY TO SUBMIT":
                        state["ready"].append(dict(day=brain_day(), fitness=row["fitness"],
                                                   text=details(code, g, row, status)))
                save_state(state)
                if errors_in_a_row >= 10:
                    log("  BRAIN keeps failing (gateway errors); pausing 5 minutes before trying again.")
                    time.sleep(300)
                    errors_in_a_row = 0
    except KeyboardInterrupt:
        log("Stopped by Control+C.")
    save_state(state)

    ready = sorted((r for r in state["ready"] if r["day"] == brain_day()), key=lambda r: -r["fitness"])
    with open(READY_FILE, "w") as f:
        f.write(f"{len(ready)} alphas passed every check today ({brain_day()}), best first. None submitted.\n"
                "Submit the first, then click Check Submission on the next before submitting it:\n"
                "once one is submitted, the others are compared with it too.\n\n"
                + "\n".join(r["text"] for r in ready))
    log(f"--- done: {sims_today()}/{args.daily_sims} simulations today, {len(ready)} ready to submit "
        f"(see {READY_FILE}); every Excellent+ alpha is in {EXCELLENT_FILE}, every alpha in {CSV_FILE} ---")


if __name__ == "__main__":
    main()
    os._exit(0)  # don't wait for simulations still running on BRAIN's side
