#!/usr/bin/env python3
"""Run independent simulator seeds and summarize uncertainty across runs.

The confidence intervals use independent seed-level policy means, not individual
tasks as independent samples. All input workload and performance values remain
outputs of the software evaluation model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
SIMULATOR = APP_DIR / "edge_cloud_simulator.py"
T_CRITICAL_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def confidence_summary(records: list[dict], group_fields: tuple[str, ...],
                       excluded: set[str]) -> list[dict]:
    groups: dict[tuple[str, ...], list[dict]] = {}
    for row in records:
        key = tuple(str(row[field]) for field in group_fields)
        groups.setdefault(key, []).append(row)

    summaries = []
    for key, rows in sorted(groups.items()):
        n = len(rows)
        t_value = T_CRITICAL_95.get(n - 1, 1.96)
        result = {field: value for field, value in zip(group_fields, key)}
        result["n_seeds"] = n
        metric_fields = [field for field in rows[0]
                         if field not in excluded and all(is_number(r.get(field, "")) for r in rows)]
        for field in metric_fields:
            values = [float(row[field]) for row in rows]
            mean = sum(values) / n
            sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (n - 1)) if n > 1 else 0.0
            margin = t_value * sd / math.sqrt(n) if n > 1 else 0.0
            result[f"{field}_mean"] = round(mean, 6)
            result[f"{field}_sd"] = round(sd, 6)
            result[f"{field}_ci95_low"] = round(mean - margin, 6)
            result[f"{field}_ci95_high"] = round(mean + margin, 6)
        summaries.append(result)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="simulated results/robust evaluation")
    parser.add_argument("--seed-start", type=int, default=2026,
                        help="First seed; consecutive seeds are evaluated.")
    parser.add_argument("--seeds", type=int, default=30, help="Independent repetitions (minimum 2).")
    parser.add_argument("--scenarios", type=int, default=20)
    parser.add_argument("--tasks-per-scenario", type=int, default=400)
    parser.add_argument("--training-scenarios", type=int, default=12)
    parser.add_argument("--capacity-scale", type=float, default=1.0)
    parser.add_argument("--bandwidth-scale", type=float, default=1.0)
    parser.add_argument("--network-latency-scale", type=float, default=1.0)
    parser.add_argument("--nodes-per-tier", type=int, default=1)
    parser.add_argument("--workload-source", choices=("synthetic_archetypes", "reference_marginals"),
                        default="synthetic_archetypes")
    parser.add_argument("--reference-task-file", type=Path,
                        default=APP_DIR / "processed dataset" / "task_decision_dataset.csv")
    args = parser.parse_args()
    if (args.seeds < 2 or args.scenarios < 3 or args.tasks_per_scenario < 1 or args.training_scenarios < 1
            or args.capacity_scale <= 0 or args.bandwidth_scale <= 0 or args.network_latency_scale <= 0
            or args.nodes_per_tier < 1):
        parser.error("use at least 2 seeds, 3 scenarios, and positive task/training counts")
    if not SIMULATOR.exists():
        parser.error(f"Simulator not found: {SIMULATOR}")

    output = Path(args.output_dir).expanduser()
    if not output.is_absolute():
        output = APP_DIR / output
    output.mkdir(parents=True, exist_ok=True)

    overall_rows: list[dict] = []
    load_rows: list[dict] = []
    priority_rows: list[dict] = []
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    # Place transient sub-runs under output rather than the system temp area;
    # this also avoids sandbox ACL failures and long nested Windows paths.
    with tempfile.TemporaryDirectory(prefix="_r_", dir=output) as scratch:
        scratch_root = Path(scratch)
        for index, seed in enumerate(seeds, start=1):
            run_dir = scratch_root / f"seed_{seed}"
            command = [sys.executable, str(SIMULATOR), "--output-dir", str(run_dir),
                       "--seed", str(seed), "--scenarios", str(args.scenarios),
                       "--tasks-per-scenario", str(args.tasks_per_scenario),
                       "--training-scenarios", str(args.training_scenarios),
                       "--capacity-scale", str(args.capacity_scale),
                       "--bandwidth-scale", str(args.bandwidth_scale),
                       "--network-latency-scale", str(args.network_latency_scale),
                       "--nodes-per-tier", str(args.nodes_per_tier),
                       "--workload-source", args.workload_source,
                       "--reference-task-file", str(args.reference_task_file.expanduser().resolve()),
                       "--ai-energy-weight", "0", "--ai-utilization-weight", "20", "--priority-score-weight", "35"]
            completed = subprocess.run(command, cwd=APP_DIR, capture_output=True, text=True,
                                       encoding="utf-8", errors="replace")
            if completed.returncode:
                raise SystemExit(f"Seed {seed} failed.\n{completed.stdout}\n{completed.stderr}")

            for row in read_csv(run_dir / "policy_comparison_summary.csv"):
                overall_rows.append({"seed": seed, **row})
            for row in read_csv(run_dir / "policy_comparison_by_load.csv"):
                load_rows.append({"seed": seed, **row})
            for row in read_csv(run_dir / "policy_comparison_by_priority.csv"):
                priority_rows.append({"seed": seed, **row})
            print(f"Completed seed {index}/{len(seeds)} ({seed})")

    write_csv(output / "per_seed_policy_summary.csv", overall_rows)
    write_csv(output / "per_seed_load_metrics.csv", load_rows)
    write_csv(output / "per_seed_priority_metrics.csv", priority_rows)
    write_csv(output / "robust_policy_summary.csv",
              confidence_summary(overall_rows, ("policy",), {"seed", "policy", "scenarios", "task_count_per_scenario"}))
    write_csv(output / "robust_policy_by_load.csv",
              confidence_summary(load_rows, ("load_regime", "policy"), {"seed", "load_regime", "policy", "scenarios"}))
    write_csv(output / "robust_policy_by_priority.csv",
              confidence_summary(priority_rows, ("load_regime", "policy", "priority"),
                                 {"seed", "load_regime", "policy", "priority", "attempted_count"}))

    assumptions = {
        "evaluation_type": "software-based computational evaluation",
        "seed_start": args.seed_start,
        "seeds": seeds,
        "independent_repetitions": args.seeds,
        "scenarios_per_seed": args.scenarios,
        "tasks_per_scenario": args.tasks_per_scenario,
        "training_scenarios_per_seed": args.training_scenarios,
        "capacity_scale": args.capacity_scale,
        "bandwidth_scale": args.bandwidth_scale,
        "network_latency_scale": args.network_latency_scale,
        "nodes_per_tier": args.nodes_per_tier,
        "workload_source": args.workload_source,
        "reference_task_file": (args.reference_task_file.name if args.workload_source == "reference_marginals" else None),
        "ai_energy_weight_ms_per_j": 0,
        "ai_utilization_weight_ms": 20,
        "priority_score_weight_ms": 35,
        "confidence_interval": "Two-sided 95% Student t interval across independent seed-level means; normal approximation for more than 31 seeds.",
        "interpretation": "Interval estimates quantify variation across model-generated runs; they do not represent physical deployment measurement uncertainty.",
        "source_data_note": "Consolidated source datasets remain reference data and are not counterfactual outcomes for every policy or measured energy.",
    }
    (output / "robust_evaluation_assumptions.json").write_text(json.dumps(assumptions, indent=2), encoding="utf-8")
    print(f"\nRobust evaluation complete. Reports: {output}")


if __name__ == "__main__":
    main()
