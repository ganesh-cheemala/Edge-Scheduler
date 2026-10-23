#!/usr/bin/env python3
"""Evaluate horizontal edge/cloud node replication on matched workloads.

Each node tier is replicated 1, 2, and 4 times. Task generation is independent
of replica count, so each seed/profile receives the same offered workload.
Outcomes remain software-model estimates.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from multi_seed_evaluation import confidence_summary, read_csv, write_csv

APP_DIR = Path(__file__).resolve().parent
SIMULATOR = APP_DIR / "edge_cloud_simulator.py"
REPLICA_COUNTS = (1, 2, 4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="simulated results/scalability study")
    parser.add_argument("--seed-start", type=int, default=2026)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--scenarios", type=int, default=18)
    parser.add_argument("--tasks-per-scenario", type=int, default=400)
    parser.add_argument("--training-scenarios", type=int, default=12)
    parser.add_argument("--capacity-scale", type=float, default=2.25)
    parser.add_argument("--bandwidth-scale", type=float, default=5.0)
    parser.add_argument("--network-latency-scale", type=float, default=0.25)
    parser.add_argument("--ai-energy-weight", type=float, default=0.0)
    parser.add_argument("--ai-utilization-weight", type=float, default=20.0)
    parser.add_argument("--priority-score-weight", type=float, default=35.0)
    parser.add_argument("--workload-source", choices=("synthetic_archetypes", "reference_marginals"),
                        default="synthetic_archetypes")
    parser.add_argument("--reference-task-file", type=Path,
                        default=APP_DIR / "processed dataset" / "task_decision_dataset.csv")
    args = parser.parse_args()
    if (args.seeds < 2 or args.scenarios < 3 or args.tasks_per_scenario < 1
            or args.training_scenarios < 1 or args.capacity_scale <= 0
            or args.bandwidth_scale <= 0 or args.network_latency_scale <= 0):
        parser.error("use at least 2 seeds, 3 scenarios, and positive task/resource parameters")
    if args.workload_source == "reference_marginals" and not args.reference_task_file.expanduser().is_file():
        parser.error(f"Reference task table not found: {args.reference_task_file}")
    if not SIMULATOR.is_file():
        parser.error(f"Simulator not found: {SIMULATOR}")

    output = Path(args.output_dir).expanduser()
    if not output.is_absolute():
        output = APP_DIR / output
    output.mkdir(parents=True, exist_ok=True)

    overall: list[dict] = []
    by_load: list[dict] = []
    total_runs = args.seeds * len(REPLICA_COUNTS)
    index = 0
    # Keep scratch runs under the selected output folder: short names avoid
    # Windows path-length trouble and respect restricted temp-directory ACLs.
    with tempfile.TemporaryDirectory(prefix="_s_", dir=output) as scratch:
        scratch_root = Path(scratch)
        for replica_count in REPLICA_COUNTS:
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                index += 1
                run_dir = scratch_root / f"r{replica_count}_s{seed}"
                command = [sys.executable, str(SIMULATOR), "--output-dir", str(run_dir),
                           "--seed", str(seed), "--scenarios", str(args.scenarios),
                           "--tasks-per-scenario", str(args.tasks_per_scenario),
                           "--training-scenarios", str(args.training_scenarios),
                           "--capacity-scale", str(args.capacity_scale),
                           "--bandwidth-scale", str(args.bandwidth_scale),
                           "--network-latency-scale", str(args.network_latency_scale),
                           "--nodes-per-tier", str(replica_count),
                           "--workload-source", args.workload_source,
                           "--reference-task-file", str(args.reference_task_file.expanduser().resolve()),
                           "--ai-energy-weight", str(args.ai_energy_weight),
                           "--ai-utilization-weight", str(args.ai_utilization_weight),
                           "--priority-score-weight", str(args.priority_score_weight)]
                completed = subprocess.run(command, cwd=APP_DIR, capture_output=True, text=True,
                                           encoding="utf-8", errors="replace")
                if completed.returncode:
                    raise SystemExit(f"Replica count {replica_count}, seed {seed} failed.\n"
                                     f"{completed.stdout}\n{completed.stderr}")
                for row in read_csv(run_dir / "policy_comparison_summary.csv"):
                    overall.append({"seed": seed, "nodes_per_tier": replica_count, **row})
                for row in read_csv(run_dir / "policy_comparison_by_load.csv"):
                    by_load.append({"seed": seed, "nodes_per_tier": replica_count, **row})
                print(f"Completed {index}/{total_runs}: replicas={replica_count}, seed={seed}")

    write_csv(output / "scalability_per_seed.csv", overall)
    write_csv(output / "scalability_per_seed_by_load.csv", by_load)
    excluded = {"seed", "nodes_per_tier", "policy", "scenarios", "task_count_per_scenario"}
    write_csv(output / "scalability_summary.csv",
              confidence_summary(overall, ("nodes_per_tier", "policy"), excluded))
    excluded_load = {"seed", "nodes_per_tier", "load_regime", "policy", "scenarios"}
    write_csv(output / "scalability_by_load.csv",
              confidence_summary(by_load, ("load_regime", "nodes_per_tier", "policy"), excluded_load))
    assumptions = {
        "evaluation_type": "software-based horizontal-scaling sensitivity",
        "replicas_per_tier": list(REPLICA_COUNTS),
        "meaning": "Each replica adds a node with the same configured CPU, bandwidth, network delay, energy coefficient, and initial-utilization model; aggregate modeled capacity increases with replica count.",
        "matched_workloads": "Task generation depends on seed and scenario but not replica count. The same per-seed/scenario workload is offered to every replica-count condition.",
        "seed_start": args.seed_start,
        "seeds": list(range(args.seed_start, args.seed_start + args.seeds)),
        "independent_repetitions": args.seeds,
        "scenarios_per_seed": args.scenarios,
        "tasks_per_scenario": args.tasks_per_scenario,
        "training_scenarios_per_seed": args.training_scenarios,
        "capacity_scale": args.capacity_scale,
        "bandwidth_scale": args.bandwidth_scale,
        "network_latency_scale": args.network_latency_scale,
        "workload_source": args.workload_source,
        "reference_task_file": args.reference_task_file.name if args.workload_source == "reference_marginals" else None,
        "ai_weights": {"energy": args.ai_energy_weight,
                       "utilization": args.ai_utilization_weight,
                       "priority": args.priority_score_weight},
        "confidence_interval": "Two-sided 95% Student t intervals across independent seed-level means.",
        "limitations": "This tests modeled horizontal replication and additional aggregate resources, not overheads, coordination traffic, failure modes, deployment cost, or measured infrastructure scaling.",
    }
    (output / "scalability_assumptions.json").write_text(json.dumps(assumptions, indent=2), encoding="utf-8")
    print(f"\nScalability study complete: {output}")


if __name__ == "__main__":
    main()
