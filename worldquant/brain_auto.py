#!/usr/bin/env python3
"""
Keep searching for WorldQuant BRAIN alphas until today's target is reached.

Each round it takes the best alphas found so far, changes one or two things
(a weight, a part, a field, an operator, the grouping, or a setting), simulates
the result, and runs BRAIN's own "Check Submission" on anything rated Excellent
or better. With --auto-submit it submits every alpha that passes every check.

It stops when today's target is reached (default: 4 Spectacular or 5 Excellent-
or-better submissions, counted on the BRAIN day in US Eastern time), when it has
run --max-sims simulations, or when you press Ctrl+C. Progress is saved in
results/auto_state.json, so running it again continues where it left off.

Usage:
    python3 brain_auto.py                     # search, print passing alphas, submit nothing
    python3 brain_auto.py --auto-submit       # also submit every alpha that passes
    python3 brain_auto.py --auto-submit --spectacular 10 --excellent 10
"""
import argparse
import copy
import getpass
import json
import os
import random
import re
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from zoneinfo import ZoneInfo

from brain_search import MAX_CONCURRENT, PLATFORM, Brain, rating

STATE_FILE = "results/auto_state.json"
LOG_FILE = "results/auto_log.txt"

# Every alpha is built from these parts. Each part lists the choices a mutation
# may switch between; all fields here already ran on this account.
PARTS = {
    "vr": dict(opts=dict(num=["implied_volatility_call_120", "implied_volatility_mean_120"],
                         den=["parkinson_volatility_120", "parkinson_volatility_60"], smooth=[5, 10]),
               line="vr = ts_backfill({num} / {den}, 5);", use="ts_mean(vr, {smooth})"),
    "val": dict(opts=dict(op=["ts_rank", "ts_zscore"], cf=["cashflow_op", "cashflow"], win=[126, 252, 504]),
                line="val = -{op}(enterprise_value / ts_backfill({cf}, 60), {win});", use="val"),
    "oey": dict(opts=dict(op=["ts_rank", "ts_zscore"], win=[126, 252, 504]),
                line="oey = {op}(ts_backfill(operating_income, 60) / cap, {win});", use="oey"),
    "lev": dict(opts=dict(win=[0, 252]),
                line="lev = ts_backfill(liabilities, 60) / ts_backfill(assets, 60);", use="lev"),
    "fcfq": dict(opts={}, line="fcfq = ts_decay_linear(ts_scale(est_cashflow_op, 252), 22)"
                               " - ts_decay_linear(ts_scale(est_capex, 252), 22);", use="fcfq"),
    "skew": dict(opts=dict(smooth=[5, 10]),
                 line="skew = ts_backfill((implied_volatility_call_180 - implied_volatility_put_180)"
                      " / implied_volatility_mean_180, 5);", use="ts_mean(skew, {smooth})"),
    "ovp": dict(opts=dict(win=[63, 126]),
                line="ovp = -ts_corr(ts_backfill(est_ptp, 20), ts_backfill(est_fcf, 20), {win});", use="ovp"),
    "stab": dict(opts=dict(win=[5, 10]), line="stab = -ts_std_dev(ts_backfill(scl12_buzz, 5), {win});", use="stab"),
    "epsg": dict(opts={}, line="epsg = ts_delta(ts_backfill(eps, 60), 252) / close;", use="epsg"),
    "surp": dict(opts={}, line="surp = (ts_backfill(eps, 60) - ts_delay(ts_backfill(est_eps, 60), 63)) / close;",
                 use="surp"),
    "lowvol": dict(opts=dict(win=[126, 252]), line="lowvol = -ts_std_dev(returns, {win});", use="lowvol"),
}
CUSTOM_GROUPS = {
    "size": 'cgrp = densify(bucket(rank(cap), range="0.1,1,0.1"));',
    "vol": 'cgrp = densify(bucket(rank(ts_std_dev(returns, 60)), range="0.1,1,0.1"));',
}
GROUPS = ["industry", "sector", "subindustry", "size", "vol"]
SETTING_CHOICES = dict(universe=["TOP200", "TOP500", "TOP1000", "TOP3000"],
                       neutralization=["MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY"],
                       decay=[2, 3, 4, 5, 6, 8], truncation=[0.03, 0.05, 0.08])


def genome(legs, group="industry", gop="group_rank", invvol=0, **settings):
    params = {leg: {k: v[0] for k, v in PARTS[leg]["opts"].items()} for leg in legs}
    params.get("val", {}).update(win=252)
    params.get("oey", {}).update(win=252)
    base = dict(universe="TOP500", neutralization="INDUSTRY", decay=4, truncation=0.05)
    return dict(legs=legs, params=params, group=group, gop=gop, invvol=invvol, settings={**base, **settings})


# Starting points: the Spectacular/Excellent alphas from earlier rounds, plus a few
# mixes that lean on parts no submitted alpha uses.
SEEDS = [
    genome(dict(vr=0.25, val=0.45, oey=0.3)),                        # wp92bkV1, Sharpe 2.24
    genome(dict(vr=0.25, val=0.4, oey=0.2, lev=0.15)),               # 00JM83oJ, Sharpe 2.16
    genome(dict(vr=0.25, val=0.4, oey=0.2, fcfq=0.15)),              # Spectacular, best 2023
    genome(dict(vr=0.25, val=0.45, oey=0.3), group="vol"),           # Excellent with vol groups
    genome(dict(vr=0.12, val=0.2, oey=0.08, ovp=0.2, lowvol=0.2, surp=0.2)),
    genome(dict(val=0.3, oey=0.2, ovp=0.25, skew=0.25), neutralization="SUBINDUSTRY"),
]


def build(g):
    """Fast Expression code for a genome."""
    lines, terms = [], []
    grp = g["group"]
    for leg, w in g["legs"].items():
        part, p = PARTS[leg], g["params"].get(leg, {})
        line = part["line"].format(**p)
        if leg == "lev" and p.get("win"):
            line = f"lev = ts_rank(ts_backfill(liabilities, 60) / ts_backfill(assets, 60), {p['win']});"
        lines.append(line)
        terms.append(f"{w} * {g['gop']}({part['use'].format(**p)}, {'cgrp' if grp in CUSTOM_GROUPS else grp})")
    if grp in CUSTOM_GROUPS:
        lines.append(CUSTOM_GROUPS[grp])
    body = " + ".join(terms) + (" - 0.5" if g["gop"] == "group_rank" else "")
    if g["invvol"]:
        body = f"({body}) / ts_std_dev(returns, {g['invvol']})"
    return "\n".join(lines + [body])


def normalize(legs):
    total = sum(legs.values())
    return {k: round(v / total, 2) for k, v in legs.items()}


def mutate(g):
    g = copy.deepcopy(g)
    for _ in range(random.choice([1, 1, 2])):
        kind = random.choice(["weight", "weight", "add", "remove", "param", "param",
                              "group", "gop", "invvol", "setting", "setting"])
        legs = g["legs"]
        if kind == "weight":
            leg = random.choice(list(legs))
            legs[leg] = max(0.05, legs[leg] + random.choice([-0.1, -0.05, 0.05, 0.1]))
        elif kind == "add":
            new = [p for p in PARTS if p not in legs]
            if new:
                leg = random.choice(new)
                legs[leg] = random.choice([0.1, 0.15, 0.2])
                g["params"][leg] = {k: random.choice(v) for k, v in PARTS[leg]["opts"].items()}
        elif kind == "remove" and len(legs) > 2:
            leg = random.choice(list(legs))
            del legs[leg]
            g["params"].pop(leg, None)
        elif kind == "param":
            leg = random.choice([p for p in legs if PARTS[p]["opts"]] or [None])
            if leg:
                key = random.choice(list(PARTS[leg]["opts"]))
                g["params"].setdefault(leg, {})[key] = random.choice(PARTS[leg]["opts"][key])
        elif kind == "group":
            g["group"] = random.choice(GROUPS)
        elif kind == "gop":
            g["gop"] = "group_zscore" if g["gop"] == "group_rank" else "group_rank"
        elif kind == "invvol":
            g["invvol"] = random.choice([0, 20, 60, 120])
        else:
            key = random.choice(list(SETTING_CHOICES))
            g["settings"][key] = random.choice(SETTING_CHOICES[key])
        g["legs"] = normalize(legs)
    return g


def brain_day():
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return dict(tried=[], pool=[], submitted=[], bad_tokens=[])


def save_state(state):
    os.makedirs("results", exist_ok=True)
    with open(STATE_FILE + ".tmp", "w") as f:
        json.dump(state, f)
    os.replace(STATE_FILE + ".tmp", STATE_FILE)


def log(text):
    print(text)
    os.makedirs("results", exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text + "\n")


def key_of(code, settings):
    return code + "|" + json.dumps(settings, sort_keys=True)


def corr_ok(row):
    return row.get("self_corr") is None or row["self_corr"] < 0.7 or row.get("corr_check") == "PASS"


def parent_score(entry):
    row = entry["row"]
    # Alphas that already pass self-correlation make better parents than ones that don't.
    return row["fitness"] * (1.0 if corr_ok(row) else 0.75)


def pick_parent(state):
    pool = [e for e in state["pool"] if e["row"].get("fitness") is not None]
    if not pool or random.random() < 0.15:
        return random.choice(SEEDS)
    pool.sort(key=parent_score, reverse=True)
    return max(random.sample(pool[:15], min(4, len(pool[:15]))), key=parent_score)["genome"]


def next_candidate(state, tried):
    bad = set(state["bad_tokens"])
    for _ in range(200):
        g = mutate(pick_parent(state))
        code = build(g)
        if bad and any(re.search(rf"\b{re.escape(t)}\b", code) for t in bad):
            continue
        k = key_of(code, g["settings"])
        if k not in tried:
            tried.add(k)
            return g, code, k
    return None


def full_check(brain, alpha_id):
    """True when every row of BRAIN's Check Submission passes."""
    for _ in range(4):
        checks = brain.check(alpha_id)
        results = {c.get("result") for c in checks}
        if checks and "PENDING" not in results:
            return not results & {"FAIL", "ERROR"}, checks
        time.sleep(30)
    return False, checks


def paste_block(code, g, row):
    s = g["settings"]
    return (f"=== {row['rating'].upper()}  fitness {row['fitness']}  sharpe {row['sharpe']}  "
            f"self-corr {row.get('self_corr')}  {row.get('link')}\n"
            f"Settings: USA | Delay 1 | {s['universe']} | Neutralization {s['neutralization'].title()} | "
            f"Decay {s['decay']} | Truncation {s['truncation']} | Pasteurization On | "
            f"Unit Handling Verify | NaN Handling On | Test Period 1 year\n{code}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--auto-submit", action="store_true", help="submit every alpha that passes every check")
    ap.add_argument("--spectacular", type=int, default=4, help="stop after this many Spectacular today (default 4)")
    ap.add_argument("--excellent", type=int, default=5,
                    help="stop after this many Excellent-or-better today (default 5)")
    ap.add_argument("--min-fitness", type=float, default=2.0, help="lowest fitness to check/submit (default 2.0)")
    ap.add_argument("--max-sims", type=int, default=400, help="stop after this many simulations (default 400)")
    ap.add_argument("--find", type=int, default=0,
                    help="without --auto-submit: stop after finding this many passing alphas, each built "
                         "from a different mix of parts, and print them for you to submit")
    args = ap.parse_args()

    state = load_state()
    tried = set(state["tried"])
    email = os.environ.get("BRAIN_EMAIL") or input("BRAIN email: ")
    password = os.environ.get("BRAIN_PASSWORD") or getpass.getpass("BRAIN password: ")
    brain = Brain(email, password)

    def today_counts():
        today = [s for s in state["submitted"] if s["day"] == brain_day()]
        return (sum(s["fitness"] > 2.5 for s in today), sum(s["fitness"] > 2.0 for s in today))

    found = []  # passing alphas kept for you to submit (with --find)

    def target_reached():
        if args.find and not args.auto_submit:
            return len(found) >= args.find
        spec, exc = today_counts()
        return spec >= args.spectacular or exc >= args.excellent

    log(f"\n--- {datetime.now():%Y-%m-%d %H:%M} start, BRAIN day {brain_day()}, "
        f"auto-submit {'ON' if args.auto_submit else 'OFF'} ---")
    sims = finished = 0
    running = {}
    try:
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
            while not target_reached() and (running or sims < args.max_sims):
                while len(running) < MAX_CONCURRENT and sims < args.max_sims:
                    nxt = next_candidate(state, tried)
                    if nxt is None:
                        break
                    g, code, k = nxt
                    cand = dict(name=f"auto{len(tried)}", code=code, **g["settings"])
                    running[pool.submit(brain.simulate, cand)] = (g, code, k)
                    sims += 1
                if not running:
                    break
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for fut in done:
                    g, code, k = running.pop(fut)
                    finished += 1
                    try:
                        row = fut.result()
                    except Exception as e:
                        row = dict(error=f"{type(e).__name__}: {e}")
                    state["tried"].append(k)
                    if row.get("error"):
                        log(f"  ERROR {row['error'][:120]}")
                        m = re.search(r'unknown (?:variable|operator) "(\w+)"', row["error"])
                        if m:
                            state["bad_tokens"].append(m.group(1))
                        save_state(state)
                        continue
                    state["pool"].append(dict(genome=g, row={x: row.get(x) for x in
                                                             ("fitness", "sharpe", "self_corr", "corr_check")}))
                    log(f"  #{finished:<4} fitness {row['fitness']}  sharpe {row['sharpe']}  "
                        f"self-corr {row.get('self_corr')}  {row['rating']}")
                    if (row["fitness"] or 0) >= args.min_fitness and not row.get("failed_checks") and corr_ok(row):
                        ok, _ = full_check(brain, row["alpha_id"])
                        # Alphas from the same mix of parts would fail against each other once one
                        # is submitted, so --find keeps one per mix.
                        mix_id = (sorted(g["legs"]), g["group"])
                        if ok and args.find and not args.auto_submit:
                            if mix_id in [m for m, _ in found]:
                                ok = False
                            else:
                                found.append((mix_id, paste_block(code, g, row)))
                        if ok:
                            log("PASSES EVERY CHECK:\n" + paste_block(code, g, row))
                            if args.auto_submit and not target_reached() and brain.submit(row["alpha_id"]):
                                state["submitted"].append(dict(day=brain_day(), alpha_id=row["alpha_id"],
                                                               fitness=row["fitness"], code=code,
                                                               settings=g["settings"]))
                                spec, exc = today_counts()
                                log(f"SUBMITTED {row['alpha_id']} ({rating(row['fitness'])}). "
                                    f"Today: {spec} Spectacular, {exc} Excellent or better.")
                    save_state(state)
    except KeyboardInterrupt:
        log("Stopped by Ctrl+C.")
    save_state(state)
    spec, exc = today_counts()
    reason = "target reached" if target_reached() else f"{sims} simulations run"
    if args.find and not args.auto_submit:
        log(f"\n=== {len(found)} ALPHAS READY TO SUBMIT (submit the first; then run Check Submission on the next "
            f"before submitting it) ===\n" + "\n".join(block for _, block in found))
    log(f"--- done ({reason}). Today: {spec} Spectacular, {exc} Excellent or better submitted. "
        f"Passing alphas are in {LOG_FILE} ---")


if __name__ == "__main__":
    main()
