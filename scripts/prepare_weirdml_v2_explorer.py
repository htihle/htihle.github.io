#!/usr/bin/env python3
"""Build the compact data file behind weirdml_v2_explorer.html.

Reads the final WeirdML v2 export (data/weirdml_v2_results.json.gz) and writes
assets/data/weirdml_v2_explorer.json, keeping only the fields the explorer
shows. Also refreshes the numeric columns of data/weirdml_data.csv so the
older v2 views agree with the final export. Standard library only.

    python scripts/prepare_weirdml_v2_explorer.py
"""
import csv
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "weirdml_v2_results.json.gz"
OUT = ROOT / "assets" / "data" / "weirdml_v2_explorer.json"
CSV_PATH = ROOT / "data" / "weirdml_data.csv"

OUTCOMES = {
    "scored": "s",
    "exec_error_no_preds": "e",
    "timeout_no_preds": "t",
    "no_preds": "n",
    "no_code": "c",
    "preds_load_error": "l",
    "synthetic_api_failure": "a",
}
COMPANIES = {
    "openai": "OpenAI", "anthropic": "Anthropic", "google": "Google",
    "deepseek": "DeepSeek", "qwen": "Alibaba (Qwen)", "x-ai": "xAI",
    "moonshotai": "Moonshot", "z-ai": "Z.ai", "meta": "Meta",
    "nvidia": "NVIDIA", "mistral": "Mistral", "mistralai": "Mistral",
    "inception": "Inception", "minimax": "MiniMax",
    "thinkingmachines": "Thinking Machines",
}


def rnd(value, digits):
    if value is None:
        return None
    value = round(float(value), digits)
    return int(value) if value == int(value) else value


def build(data):
    meta = data["metadata"]
    tasks = meta["scored_tasks"]
    models = []
    for model in data["models"]:
        s = model["summary"]
        runs = []
        for task in tasks:
            task_runs = []
            for run in sorted(model["tasks"].get(task, []), key=lambda r: r["run_timestamp"]):
                iters = [
                    [
                        rnd(it["accuracy"], 4),
                        OUTCOMES[it["outcome"]],
                        it["code_lines"],
                        rnd(it["python_run_time_s"], 1),
                        it["total_output_tokens"],
                        rnd(it["cost_usd"], 4),
                    ]
                    for it in sorted(run["iterations"], key=lambda i: i["iteration"])
                ]
                task_runs.append({
                    "id": run["run_uid"],
                    "d": run["run_timestamp"][:10],
                    "f": run["flags"],
                    "i": iters,
                })
            runs.append(task_runs)
        models.append({
            "n": s["leaderboard_name"],
            "c": COMPANIES.get(s["company"].lower(), s["company"]),
            "d": s["release_date"],
            "rank": s["rank"],
            "a": rnd(s["avg_acc"], 5),
            "se": rnd(s["avg_acc_se"], 5),
            "cost": rnd(s["cost_per_run_usd"], 5),
            "est": bool(s["cost_is_estimate"]),
            "tok": rnd(s["mean_total_output_tokens"], 0),
            "len": rnd(s["code_len_p50"], 0),
            "t": [rnd(s[f"{t}_acc"], 5) for t in tasks],
            "r": runs,
        })
    return {
        "generated": meta["generated_utc"][:10],
        "tasks": tasks,
        "chance": [meta["random_chance_accuracy"][t] for t in tasks],
        "models": models,
    }


CSV_FIELDS = {
    "avg_acc": "avg_acc", "avg_acc_standard_error": "avg_acc_se",
    "cost_per_run_usd": "cost_per_run_usd",
    "mean_total_output_tokens": "mean_total_output_tokens",
    "code_len_p10": "code_len_p10", "code_len_p50": "code_len_p50",
    "code_len_p90": "code_len_p90", "exec_time_median_s": "exec_time_median_s",
}


def refresh_csv(data):
    summaries = {m["leaderboard_name"]: m["summary"] for m in data["models"]}
    with CSV_PATH.open(newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames
        rows = list(reader)
    missing = set(summaries) - {r["display_name"] for r in rows}
    if missing:
        sys.exit(f"models missing from CSV: {sorted(missing)}")
    changed = 0
    for row in rows:
        s = summaries.get(row["display_name"])
        if s is None:
            continue
        columns = {f"{t}_acc": f"{t}_acc" for t in data["metadata"]["scored_tasks"]}
        columns.update(CSV_FIELDS)
        for column, key in columns.items():
            new = s.get(key)
            if new is None:
                continue
            try:
                same = abs(float(row[column]) - float(new)) <= 1e-9 * max(1.0, abs(float(new)))
            except ValueError:
                same = False
            if not same:
                row[column] = new
                changed += 1
    with CSV_PATH.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return changed


def main():
    with gzip.open(SRC, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    OUT.write_text(json.dumps(build(data), separators=(",", ":"), ensure_ascii=False) + "\n")
    changed = refresh_csv(data)
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.2f} MB); updated {changed} values in {CSV_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
