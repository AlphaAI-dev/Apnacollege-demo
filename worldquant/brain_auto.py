#!/usr/bin/env python3
"""
Run the day's BRAIN simulations as "atom" alphas, and record every one.

Built around the guidance from WorldQuant's own sessions:
  * one dataset and one idea per alpha (atom alphas), spread across many
    datasets so submissions fill more pyramids (dataset category x region);
  * low production correlation: prefer fields few other researchers use, and
    techniques that strip out common risk: vector neutralization against size,
    custom group neutralization, residuals against the group median, and an
    entry condition (trade_when) that only trades when it makes sense;
  * turnover under 25% without heavy decay (decay is kept at 8 or below);
  * the programme rules: at most 4 data fields and 8 operators (counted
    conservatively: every function, every + - * / < > sign).

The dataset and field list comes from BRAIN's data catalog for your account and
is cached in results/ for a week. An alpha whose Sharpe comes out strongly
negative is re-run with its sign flipped, since the opposite bet is then good.

It runs 3 simulations at a time (BRAIN's limit) until today's target is met
(default 300; BRAIN day in US Eastern). Nothing is submitted. Every alpha is
named/tagged on BRAIN, written to results/all_alphas.csv, Good+ ones go to
results/excellent_alphas.txt, and results/ready_to_submit.txt lists the ones that
passed BRAIN's Check Submission, best first, one per dataset at the top.

Usage:
    python3 brain_auto.py                         # USA, 300 simulations today
    python3 brain_auto.py --daily-sims 500 --regions USA,GLB
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

from brain_search import API, MAX_CONCURRENT, Brain, rating

STATE_FILE = "results/auto_state_v3.json"
LOG_FILE = "results/auto_log.txt"
CSV_FILE = "results/all_alphas.csv"
EXCELLENT_FILE = "results/excellent_alphas.txt"
READY_FILE = "results/ready_to_submit.txt"
MAX_FIELDS, MAX_OPS = 4, 8
BACKUP_FITNESS = 1.3   # BRAIN's own minimums are fitness 1.0 and Sharpe 1.25
MAX_TURNOVER = 25      # percent; the session's advice for production alphas

UNIVERSES = {"USA": ["TOP3000", "TOP1000", "TOP500", "TOP200"], "GLB": ["TOP3000", "MINVOL1M"]}
GROUPS = ["industry", "subindustry", "sector"]
WINDOWS = [20, 63, 126, 252]
DECAYS = [0, 2, 4, 6, 8]
TRUNCATIONS = [0.05, 0.08]
NEUTRALIZATIONS = ["MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY"]
# The neutralization the course docs suggest for each kind of data.
NEUT_BY_CATEGORY = {"fundamental": "INDUSTRY", "analyst": "INDUSTRY", "earnings": "INDUSTRY",
                    "option": "SECTOR", "pv": "MARKET", "price volume": "MARKET", "news": "SUBINDUSTRY",
                    "sentiment": "SUBINDUSTRY", "socialmedia": "SUBINDUSTRY", "model": "INDUSTRY"}

# One idea per template; X and Y always come from the same dataset.
TEMPLATES = {
    "rank":  "group_rank(ts_rank({X}, {d}), {G})",
    "zscore": "group_rank(ts_zscore({X}, {d}), {G})",
    "change": "group_rank(ts_delta({X}, {d}), {G})",
    # How far the stock is from its group's median over time (from the WorldQuant session).
    "resid": "group_rank(ts_regression({X}, group_median({X}, {G}), {d}), {G})",
    "ratio": "group_rank(ts_rank({X} / {Y}, {d}), {G})",
    # Remove the part explained by company size.
    "vecneut": "vector_neut(group_rank(ts_rank({X}, {d}), {G}), rank(cap))",
    # Only trade the more volatile stocks (entry condition), hold otherwise.
    "entry": "trade_when(ts_rank(ts_std_dev(returns, 22), 252) > 0.3, group_rank(ts_rank({X}, {d}), {G}), -1)",
    # Neutralize within size buckets instead of a standard group.
    "sizeneut": 'group_neutralize(ts_rank({X}, {d}), densify(bucket(rank(cap), range="0.1,1,0.1")))',
}

# Used only if BRAIN's data catalog can't be read: fields that already ran on this account.
FALLBACK_CATALOG = {
    "fundamental6": ("fundamental", ["operating_income", "sales", "cashflow_op", "enterprise_value",
                                     "liabilities", "assets"]),
    "analyst4": ("analyst", ["est_eps", "est_fcf", "est_ptp", "est_cashflow_op", "est_capex"]),
    "socialmedia12": ("sentiment", ["scl12_buzz"]),
    "option8": ("option", ["implied_volatility_call_120", "parkinson_volatility_120",
                           "implied_volatility_call_180", "implied_volatility_put_180"]),
}


# ---------------------------------------------------------------- data catalog

def pages(brain, path, params, max_items):
    out, offset = [], 0
    while offset < max_items:
        r = brain.request("GET", f"{API}/{path}", params={**params, "limit": 50, "offset": offset})
        if not r.ok:
            break
        batch = r.json().get("results", [])
        out += batch
        if len(batch) < 50:
            break
        offset += 50
    return out


def load_catalog(brain, region, universe):
    """{dataset_id: {"category": str, "fields": [{"id", "cov", "alphas"}]}} for one region."""
    path = f"results/fields_{region}_{universe}.json"
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < 7 * 86400:
        with open(path) as f:
            return json.load(f)
    log(f"Reading BRAIN's data catalog for {region} {universe} (first run of the week takes a few minutes)...")
    base = dict(instrumentType="EQUITY", region=region, delay=1, universe=universe)
    catalog = {}
    try:
        for ds in pages(brain, "data-sets", base, 400):
            fields = pages(brain, "data-fields", {**base, "dataset.id": ds["id"]}, 300)
            keep = [dict(id=f["id"], cov=f.get("coverage") or 0, alphas=f.get("alphaCount") or 0)
                    for f in fields if f.get("type", "MATRIX") == "MATRIX" and (f.get("coverage") or 0) >= 0.5]
            if keep:
                cat = (ds.get("category") or {}).get("id") or (ds.get("category") or {}).get("name") or "other"
                catalog[ds["id"]] = dict(category=str(cat).lower(), fields=keep)
    except Exception as e:
        log(f"  Couldn't read the catalog ({type(e).__name__}); using the built-in field list.")
    if not catalog:
        catalog = {ds: dict(category=cat, fields=[dict(id=f, cov=1, alphas=0) for f in fields])
                   for ds, (cat, fields) in FALLBACK_CATALOG.items()}
    else:
        log(f"  {len(catalog)} datasets, {sum(len(v['fields']) for v in catalog.values())} usable fields.")
    with open(path, "w") as f:
        json.dump(catalog, f)
    return catalog


# ---------------------------------------------------------------- alphas

def build(g):
    def field(name):
        return f"ts_backfill({name}, 60)" if g["bf"] else name
    core = TEMPLATES[g["tpl"]].format(X=field(g["x"]), Y=field(g["y"]) if g.get("y") else "", d=g["d"], G=g["grp"])
    return core if g["sign"] > 0 else f"-{core}"


def count_fields_ops(code):
    """(data fields, operators); operators counted conservatively."""
    body = re.sub(r'"[^"]*"', "", code)                 # bucket ranges
    body = re.sub(r"\b[a-z_]+\s*=", "", body)            # keyword arguments
    calls = re.findall(r"\b[a-z_][a-z0-9_]*(?=\s*\()", body)
    names = set(re.findall(r"\b[a-z_][a-z0-9_]*\b(?!\s*\()", body))
    fields = names - set(GROUPS) - {"market"}
    return len(fields), len(calls) + len(re.findall(r"[-+*/<>]", body))


def pick_field(fields):
    """Prefer well-covered fields that few alphas use: lower production correlation."""
    weights = [f["cov"] / (1 + f["alphas"] / 50) for f in fields]
    return random.choices(fields, weights=weights or None)[0]["id"]


def new_atom(state, catalogs, regions):
    region = random.choice(regions)
    universe, catalog = catalogs[region]
    # Datasets with fewer ready alphas so far get more turns: more pyramids, more diversity.
    ds_ids = list(catalog)
    weights = [1 / (1 + state["ready_by_ds"].get(ds, 0)) for ds in ds_ids]
    ds = random.choices(ds_ids, weights=weights)[0]
    fields = catalog[ds]["fields"]
    tpl = random.choice(list(TEMPLATES)) if len(fields) > 1 else random.choice(
        [t for t in TEMPLATES if t != "ratio"])
    x = pick_field(fields)
    y = pick_field([f for f in fields if f["id"] != x]) if tpl == "ratio" else None
    category = catalog[ds]["category"]
    return dict(region=region, ds=ds, cat=category, x=x, y=y, tpl=tpl, d=random.choice(WINDOWS),
                grp=random.choice(GROUPS), sign=1, bf=int(category in ("fundamental", "analyst", "earnings")),
                settings=dict(universe=universe, neutralization=NEUT_BY_CATEGORY.get(category, "INDUSTRY"),
                              decay=random.choice([2, 4, 6]), truncation=0.08))


def mutate(g, catalogs):
    g = copy.deepcopy(g)
    fields = catalogs[g["region"]][1].get(g["ds"], {}).get("fields", [])
    kind = random.choice(["tpl", "d", "d", "grp", "sign", "bf", "field", "neut", "decay", "trunc", "universe"])
    if kind == "tpl":
        g["tpl"] = random.choice([t for t in TEMPLATES if t != "ratio" or len(fields) > 1])
        if g["tpl"] == "ratio" and not g.get("y"):
            g["y"] = pick_field([f for f in fields if f["id"] != g["x"]])
    elif kind == "d":
        g["d"] = random.choice(WINDOWS)
    elif kind == "grp":
        g["grp"] = random.choice(GROUPS)
    elif kind == "sign":
        g["sign"] = -g["sign"]
    elif kind == "bf":
        g["bf"] = 1 - g["bf"]
    elif kind == "field" and fields:
        g["x"] = pick_field(fields)
    elif kind == "neut":
        g["settings"]["neutralization"] = random.choice(NEUTRALIZATIONS)
    elif kind == "decay":
        g["settings"]["decay"] = random.choice(DECAYS)
    elif kind == "trunc":
        g["settings"]["truncation"] = random.choice(TRUNCATIONS)
    elif kind == "universe":
        g["settings"]["universe"] = random.choice(UNIVERSES.get(g["region"], ["TOP3000"]))
    return g


# ---------------------------------------------------------------- state & output

# Fields that are already the main idea of several submitted alphas (mLovdwX9, KPg3blVp,
# 78LpvnO5, ...). Reusing them only makes "the same expression as an addition", which
# the WorldQuant session said lowers an alpha's value, and raises production correlation.
ALREADY_SUBMITTED_FIELDS = ["USA:operating_income", "USA:cashflow_op", "USA:liabilities",
                            "USA:enterprise_value", "USA:assets"]


def brain_day():
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")


def load_state():
    state = {}
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            state = json.load(f)
    for key, default in dict(tried=[], pool=[], bad_tokens=[], sims_by_day={}, ready=[], blocked=[],
                             hot={}, ready_by_ds={}, flips=[]).items():
        state.setdefault(key, default)
    for key in ALREADY_SUBMITTED_FIELDS:
        if key not in state["blocked"]:
            state["blocked"].append(key)
    return state


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


def field_key(g):
    return f"{g['region']}:{g['x']}"


def next_candidate(state, tried, catalogs, regions):
    bad = set(state["bad_tokens"])
    for _ in range(500):
        if state["flips"]:
            g = state["flips"].pop(0)                       # strongly negative alphas, sign flipped
        else:
            pool = [e for e in state["pool"] if (e["row"].get("fitness") or 0) > 0.5]
            if pool and random.random() < 0.6:
                pool.sort(key=lambda e: e["row"]["fitness"] * (1 if corr_ok(e["row"]) else 0.4), reverse=True)
                g = mutate(random.choice(pool[:25])["genome"], catalogs)
            else:
                g = new_atom(state, catalogs, regions)
        if g["region"] not in regions or field_key(g) in state["blocked"]:
            continue
        code = build(g)
        fields, ops = count_fields_ops(code)
        if fields > MAX_FIELDS or ops > MAX_OPS:
            continue
        if bad and any(re.search(rf"\b{re.escape(t)}\b", code) for t in bad):
            continue
        k = g["region"] + "|" + code + "|" + json.dumps(g["settings"], sort_keys=True)
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


def analyze(brain, row):
    """The slow part, run off the simulation slots: self-correlation and Check Submission."""
    fit = row["fitness"] or 0
    row["self_corr"] = brain.self_correlation(row["alpha_id"])
    if (row["self_corr"] or 0) >= 0.7 and fit >= 2.0:
        row["corr_check"] = brain.corr_check(row["alpha_id"])  # the 10% Sharpe rule
    if not corr_ok(row):
        return row, f"FAILS self-correlation {row['self_corr']}"
    ok, bad = full_check(brain, row["alpha_id"])
    if not ok:
        return row, "FAILS " + " ".join(bad)
    return row, "READY TO SUBMIT" if fit >= 2.0 else f"READY TO SUBMIT (backup: {row['rating']}, not Excellent)"


def details(code, g, row, status):
    s = g["settings"]
    fields, ops = count_fields_ops(code)
    return (f"=== {status} | {row['rating'].upper()} fitness {row['fitness']} sharpe {row['sharpe']} "
            f"turnover {row.get('turnover')}% self-corr {row.get('self_corr')} | dataset {g['ds']} "
            f"({g['cat']}) | {fields} fields, {row.get('op_count') or ops} operators\n{row['link']}\n"
            f"Settings: {g['region']} | Delay 1 | {s['universe']} | Neutralization {s['neutralization'].title()} | "
            f"Decay {s['decay']} | Truncation {s['truncation']} | Pasteurization On | "
            f"Unit Handling Verify | NaN Handling On | Test Period 1 year\n{code}\n")


CSV_COLUMNS = ["time", "alpha_id", "link", "rating", "fitness", "sharpe", "turnover", "self_corr", "status",
               "region", "dataset", "category", "fields", "operators", "universe", "neutralization", "decay",
               "truncation", "code"]


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
                    row.get("self_corr"), status, g["region"], g["ds"], g["cat"], fields,
                    row.get("op_count") or ops, s["universe"], s["neutralization"], s["decay"],
                    s["truncation"], code])


# ---------------------------------------------------------------- main loop

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--daily-sims", type=int, default=300, help="simulations to run today (default 300)")
    ap.add_argument("--regions", default="USA", help="comma-separated regions, e.g. USA,GLB (default USA)")
    args = ap.parse_args()
    regions = [r.strip().upper() for r in args.regions.split(",") if r.strip()]

    os.makedirs("results", exist_ok=True)
    state = load_state()
    tried = set(state["tried"])
    email = os.environ.get("BRAIN_EMAIL") or input("BRAIN email: ")
    password = os.environ.get("BRAIN_PASSWORD") or getpass.getpass("BRAIN password: ")
    brain = Brain(email, password)
    catalogs = {r: (UNIVERSES.get(r, ["TOP3000"])[0], load_catalog(brain, r, UNIVERSES.get(r, ["TOP3000"])[0]))
                for r in regions}

    def sims_today():
        return state["sims_by_day"].get(brain_day(), 0)

    def finalize(g, code, row, status):
        state["pool"].append(dict(genome=g, row={x: row.get(x) for x in ("fitness", "sharpe", "self_corr",
                                                                        "corr_check")}))
        fit = row["fitness"] or 0
        key = field_key(g)
        if fit >= 1.3 and (row.get("self_corr") or 0) >= 0.9:
            state["hot"][key] = state["hot"].get(key, 0) + 1
        ready = status.startswith("READY")
        if ready:
            state["ready_by_ds"][g["ds"]] = state["ready_by_ds"].get(g["ds"], 0) + 1
        if (ready or state["hot"].get(key, 0) >= 2) and key not in state["blocked"]:
            state["blocked"].append(key)   # one ready alpha per field; or the field is used up
        brain.tag(row["alpha_id"], f"auto_{g['cat']}_{row['rating'].upper().replace(' ', '_')}_{row['fitness']}",
                  ["auto", g["cat"], row["rating"].lower().replace(" ", "_")])
        record_csv(row, g, code, status)
        log(f"  [{sims_today():>3}/{args.daily_sims}] {row['rating']:<13} fit {row['fitness']}  sharpe "
            f"{row['sharpe']}  to {row.get('turnover')}%  corr {row.get('self_corr')}  {g['ds']:<14} "
            f"{row['alpha_id']}  {status}")
        if fit >= 1.6:  # full code and settings for Good and better
            log(details(code, g, row, status), EXCELLENT_FILE)
        if ready:
            state["ready"].append(dict(day=brain_day(), fitness=row["fitness"], ds=g["ds"],
                                       text=details(code, g, row, status)))
        save_state(state)

    log(f"\n--- {datetime.now():%Y-%m-%d %H:%M} start, BRAIN day {brain_day()}, regions {','.join(regions)}, "
        f"{sims_today()}/{args.daily_sims} simulations done today ---")
    log(f"Running {MAX_CONCURRENT} simulations at a time (BRAIN's limit). Each takes 1-3 minutes.")
    sim_pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENT)  # no 'with': Control+C exits at once
    check_pool = ThreadPoolExecutor(max_workers=6)
    sims, checks, check_rows = {}, {}, {}
    errors_in_a_row = 0
    try:
        while sims or checks or sims_today() + len(sims) < args.daily_sims:
            while len(sims) < MAX_CONCURRENT and sims_today() + len(sims) < args.daily_sims:
                nxt = next_candidate(state, tried, catalogs, regions)
                if nxt is None:
                    break
                g, code, k = nxt
                cand = dict(name="auto", code=code, region=g["region"], **g["settings"])
                sims[sim_pool.submit(brain.simulate, cand, False)] = (g, code, k)
            if not sims and not checks:
                break
            done, _ = wait(list(sims) + list(checks), return_when=FIRST_COMPLETED)
            for fut in done:
                if fut in checks:
                    g, code = checks.pop(fut)
                    try:
                        row, status = fut.result()
                    except Exception as e:
                        row, status = check_rows[fut], f"CHECK ERROR {type(e).__name__}"
                    finalize(g, code, row, status)
                    continue
                g, code, k = sims.pop(fut)
                try:
                    row = fut.result()
                except Exception as e:
                    row = dict(error=f"{type(e).__name__}: {e}")
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
                fit, sharpe = row["fitness"] or 0, row["sharpe"] or 0
                if sharpe <= -1.0:   # the opposite bet is good: queue it with the sign flipped
                    flipped = dict(copy.deepcopy(g), sign=-g["sign"])
                    state["flips"].append(flipped)
                if (row.get("op_count") or 0) > MAX_OPS:
                    finalize(g, code, row, f"NOT ALLOWED: {row['op_count']} operators")
                elif fit < BACKUP_FITNESS or sharpe < 1.25:
                    finalize(g, code, row, "flipped sign queued" if sharpe <= -1.0 else "too weak to submit")
                elif (row.get("turnover") or 0) > MAX_TURNOVER:
                    finalize(g, code, row, f"turnover {row['turnover']}% (over {MAX_TURNOVER}%)")
                elif row.get("failed_checks"):
                    finalize(g, code, row, "FAILS " + row["failed_checks"])
                else:
                    cf = check_pool.submit(analyze, brain, row)
                    checks[cf] = (g, code)
                    check_rows[cf] = row
            if errors_in_a_row >= 10:
                log("  BRAIN keeps failing (gateway errors); pausing 5 minutes before trying again.")
                time.sleep(300)
                errors_in_a_row = 0
    except KeyboardInterrupt:
        log("Stopped by Control+C.")
    save_state(state)

    today = sorted((r for r in state["ready"] if r["day"] == brain_day()), key=lambda r: -r["fitness"])
    first, rest, seen = [], [], set()
    for r in today:                       # best alpha from each dataset first
        (rest if r["ds"] in seen else first).append(r)
        seen.add(r["ds"])
    with open(READY_FILE, "w") as f:
        f.write(f"{len(today)} alphas passed every check today ({brain_day()}). None submitted.\n"
                "The first block has the best alpha from each dataset: submit from there first.\n"
                "Click Check Submission before each one: once one is submitted, the others are compared with it.\n\n"
                + "\n".join(r["text"] for r in first)
                + ("\n--- more from the same datasets ---\n\n" + "\n".join(r["text"] for r in rest) if rest else ""))
    log(f"--- done: {sims_today()}/{args.daily_sims} simulations today, {len(today)} ready to submit from "
        f"{len(first)} datasets (see {READY_FILE}) ---")


if __name__ == "__main__":
    main()
    os._exit(0)  # don't wait for simulations still running on BRAIN's side
