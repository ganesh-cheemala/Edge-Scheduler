#!/usr/bin/env python3
"""Reproducible software model for AI-assisted edge/cloud task scheduling.

Generated workload and performance values are model-based computational estimates.
Reference datasets are not treated as measured counterfactual outcomes.
Uses only Python's standard library.

Run from the project folder:
    python edge_cloud_simulator.py
    python edge_cloud_simulator.py --output-dir "simulated results" --seed 42
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


CONFIG = {
    "scenarios": 20,
    "tasks_per_scenario": 400,
    "training_scenarios": 12,
    "seed": 2026,
    "priority_score_weight_ms": 35.0,
    "waiting_age_weight": 0.04,
    "waiting_age_cap_ms": 4000.0,
    "max_queue_wait_target_ms": 2000.0,
    "ai_energy_weight_ms_per_j": 40.0,
    "ai_utilization_weight_ms": 20.0,
    "capacity_scale": 1.0,
    "bandwidth_scale": 1.0,
    "network_latency_scale": 1.0,
    "nodes_per_tier": 1,
    "workload_source": "synthetic_archetypes",
    "load_profiles": [
        {"name": "nominal", "mean_arrival_interval_ms": 250.0},
        {"name": "busy", "mean_arrival_interval_ms": 60.0},
        {"name": "overloaded", "mean_arrival_interval_ms": 12.0},
    ],
    "task_types": [
        # Payload MB, CPU work in million instructions, relative frequency, deadline ms
        {"name": "telemetry", "payload_mb": [0.005, 0.08], "cpu_mi": [2, 30], "weight": 0.45, "deadline_ms": [40, 180]},
        {"name": "image_inference", "payload_mb": [0.15, 1.2], "cpu_mi": [80, 700], "weight": 0.30, "deadline_ms": [100, 500]},
        {"name": "event_detection", "payload_mb": [0.02, 0.25], "cpu_mi": [20, 180], "weight": 0.25, "deadline_ms": [50, 250]},
    ],
    "nodes": [
        # name, edge/cloud, CPU MI/s, bandwidth Mbps, one-way RTT component ms, energy coefficient J/MI, baseline utilization
        {"name": "edge_near", "tier": "edge", "cpu_mi_s": 1800, "bandwidth_mbps": 80, "network_ms": 4, "joules_per_mi": 0.0009, "initial_utilization": 0.42},
        {"name": "edge_far", "tier": "edge", "cpu_mi_s": 2400, "bandwidth_mbps": 35, "network_ms": 13, "joules_per_mi": 0.0007, "initial_utilization": 0.28},
        {"name": "cloud", "tier": "cloud", "cpu_mi_s": 9000, "bandwidth_mbps": 120, "network_ms": 38, "joules_per_mi": 0.00035, "initial_utilization": 0.22},
    ],
    "energy_model_note": "Estimated task energy = CPU work (MI) × configured node joules_per_mi. It is a model-based estimate, not measured device energy.",
    "network_model_note": "Upload delay = payload MB × 8,000 / bandwidth Mbps + configured one-way network_ms. Download/result payload is omitted in v1.",
}


@dataclass
class Task:
    scenario_id: int
    load_regime: str
    task_id: str
    arrival_ms: float
    task_type: str
    payload_mb: float
    cpu_mi: float
    deadline_ms: float
    priority: int
    compute_intensity_class: str = ""
    workload_source: str = "synthetic_archetypes"
    reference_task_id: str = ""


REFERENCE_TASK_ROWS: list[dict[str, str]] = []
REFERENCE_TASK_SHA256 = ""
REFERENCE_TASK_FILE = ""


def load_reference_tasks(path: Path) -> list[dict[str, str]]:
    required = ("task_input_mb", "deadline_ms", "task_priority", "task_type")
    valid: list[dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = [field for field in required if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Reference task file lacks required fields: {', '.join(missing)}")
        for row in reader:
            try:
                payload = float(row["task_input_mb"])
                deadline = float(row["deadline_ms"])
                priority = int(float(row["task_priority"]))
            except (TypeError, ValueError):
                continue
            if payload <= 0 or deadline <= 0 or priority < 1 or priority > 5 or not row.get("task_type", "").strip():
                continue
            row["_payload_mb"] = str(payload)
            row["_deadline_ms"] = str(deadline)
            row["_priority"] = str(priority)
            valid.append(row)
    if len(valid) < 100:
        raise ValueError(f"Need at least 100 valid reference tasks; found {len(valid)} in {path}")
    return valid


@dataclass
class Node:
    name: str
    tier: str
    cpu_mi_s: float
    bandwidth_mbps: float
    network_ms: float
    joules_per_mi: float
    utilization: float
    available_ms: float = 0.0
    completed: int = 0
    service_total_ms: float = 0.0


def weighted_choice(rng: random.Random, items: list[dict]) -> dict:
    total = sum(x["weight"] for x in items)
    point = rng.random() * total
    for item in items:
        point -= item["weight"]
        if point <= 0:
            return item
    return items[-1]


def make_scenario(scenario_id: int, count: int, seed: int) -> list[Task]:
    rng = random.Random(seed + scenario_id * 7919)
    load = CONFIG["load_profiles"][scenario_id % len(CONFIG["load_profiles"])]
    now = 0.0
    tasks = []
    for i in range(count):
        now += rng.expovariate(1.0 / load["mean_arrival_interval_ms"])
        reference_task_id = ""
        if CONFIG["workload_source"] == "reference_marginals":
            row = REFERENCE_TASK_ROWS[rng.randrange(len(REFERENCE_TASK_ROWS))]
            payload_mb = float(row["_payload_mb"])
            deadline = float(row["_deadline_ms"])
            priority = int(row["_priority"])
            task_type = row["task_type"].strip()
            reference_task_id = row.get("task_id", row.get("decision_id", ""))
            # The merged task table has no CPU-demand label. Assign the documented
            # archetype by empirical payload tertile; this is an explicit model input.
            q1 = CONFIG["reference_payload_q1_mb"]
            q2 = CONFIG["reference_payload_q2_mb"]
            profile = (CONFIG["task_types"][0] if payload_mb <= q1 else
                       CONFIG["task_types"][1] if payload_mb <= q2 else CONFIG["task_types"][2])
        else:
            profile = weighted_choice(rng, CONFIG["task_types"])
            payload_mb = rng.uniform(*profile["payload_mb"])
            deadline = rng.uniform(*profile["deadline_ms"])
            priority = rng.randint(1, 5)
            task_type = profile["name"]
        tasks.append(Task(
            scenario_id=scenario_id,
            load_regime=load["name"],
            task_id=f"S{scenario_id:03d}-T{i:05d}",
            arrival_ms=now,
            task_type=task_type,
            payload_mb=payload_mb,
            cpu_mi=rng.uniform(*profile["cpu_mi"]),
            deadline_ms=deadline,
            priority=priority,
            compute_intensity_class=profile["name"],
            workload_source=CONFIG["workload_source"],
            reference_task_id=reference_task_id,
        ))
    return tasks


def fresh_nodes(seed: int, scenario_id: int) -> list[Node]:
    result = []
    for spec in CONFIG["nodes"]:
        jitter_key = f"{seed}:{scenario_id}:{spec['name']}".encode("utf-8")
        jitter_seed = int.from_bytes(hashlib.sha256(jitter_key).digest()[:8], "big")
        rng = random.Random(jitter_seed)
        jitter = rng.uniform(-0.08, 0.08)
        node_spec = {k: v for k, v in spec.items() if k != "initial_utilization"}
        node_spec["cpu_mi_s"] *= CONFIG["capacity_scale"]
        node_spec["bandwidth_mbps"] *= CONFIG["bandwidth_scale"]
        node_spec["network_ms"] *= CONFIG["network_latency_scale"]
        result.append(Node(**{**node_spec, "utilization": min(0.90, max(0.05, spec["initial_utilization"] + jitter))}))
    return result


def features(task: Task, node: Node, queued_ms: float) -> list[float]:
    # Inputs are observable task/node state transformed to the simulator's units.
    # The regression learns the mapping to simulated latency on separate seeds.
    upload_ms = task.payload_mb * 8000.0 / node.bandwidth_mbps + node.network_ms
    service_ms = task.cpu_mi / (node.cpu_mi_s * max(0.10, 1.0-node.utilization)) * 1000.0
    return [1.0, upload_ms, service_ms, queued_ms]


def outcome(task: Task, node: Node, start_ms: float, queued_ms: float) -> tuple[float, float, float]:
    upload_ms = task.payload_mb * 8000.0 / node.bandwidth_mbps + node.network_ms
    service_ms = task.cpu_mi / (node.cpu_mi_s * max(0.10, 1.0 - node.utilization)) * 1000.0
    completion_ms = max(task.arrival_ms, start_ms) + upload_ms + service_ms
    latency_ms = completion_ms - task.arrival_ms
    energy_j = task.cpu_mi * node.joules_per_mi
    return latency_ms, service_ms, energy_j


def train_latency_model(seed: int, scenarios: int, tasks_per_scenario: int) -> list[float]:
    """Fit ridge regression to simulated candidate outcomes on separate scenarios."""
    rng = random.Random(seed + 987654)
    rows: list[list[float]] = []
    targets: list[float] = []
    for sid in range(10000, 10000 + scenarios):
        tasks = make_scenario(sid, tasks_per_scenario, seed)
        nodes = fresh_nodes(seed, sid)
        for task in tasks:
            for node in nodes:
                queued_ms = rng.uniform(0, 300)
                x = features(task, node, queued_ms)
                # Supervised target is generated by this simulator's documented system model.
                y = sum(x[1:])
                rows.append(x)
                targets.append(y)
    n = len(rows[0])
    # Normal equations with a tiny ridge term; Gaussian elimination with pivoting.
    a = [[0.0] * (n + 1) for _ in range(n)]
    for x, y in zip(rows, targets):
        for i in range(n):
            for j in range(n):
                a[i][j] += x[i] * x[j]
            a[i][n] += x[i] * y
    for i in range(1, n):
        a[i][i] += 1e-6
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        a[col], a[pivot] = a[pivot], a[col]
        divisor = a[col][col] or 1e-12
        for j in range(col, n + 1):
            a[col][j] /= divisor
        for r in range(n):
            if r == col:
                continue
            factor = a[r][col]
            for j in range(col, n + 1):
                a[r][j] -= factor * a[col][j]
    return [a[i][n] for i in range(n)]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def choose(policy: str, task: Task, nodes: list[Node], weights: list[float]) -> Node:
    scored = []
    for node in nodes:
        queued_ms = max(0.0, node.available_ms - task.arrival_ms)
        latency, service, energy = outcome(task, node, node.available_ms, queued_ms)
        if policy == "edge_first":
            score = (0 if node.tier == "edge" else 1, latency)
        elif policy == "earliest_completion":
            score = (latency,)
        elif policy == "admission_control":
            score = (latency,)
        elif policy == "resource_aware":
            score = (latency * (1.0 + node.utilization) + 0.15 * energy,)
        elif policy == "priority_aware":
            late_penalty = max(0.0, latency-task.deadline_ms) * (0.5 + task.priority * 0.5)
            score = (latency + late_penalty + 0.05*energy,)
        elif policy == "aging_priority":
            score = (latency,)
        elif policy == "ai_ridge":
            predicted = dot(weights, features(task, node, queued_ms))
            # First prefer nodes predicted to meet the deadline; among them trade off
            # learned latency against estimated energy and current node utilization.
            feasible = predicted <= task.deadline_ms
            score = (0 if feasible else 1,
                     predicted + CONFIG["ai_energy_weight_ms_per_j"]*energy
                     + CONFIG["ai_utilization_weight_ms"]*node.utilization if feasible
                     else predicted)
        else:
            raise ValueError(f"Unknown policy: {policy}")
        scored.append((score, node, service, energy, latency))
    return min(scored, key=lambda x: x[0])[1]


POLICIES = ["edge_first", "earliest_completion", "resource_aware", "priority_aware", "aging_priority", "admission_control", "ai_ridge"]


def simulate_policy(tasks: list[Task], scenario_id: int, policy: str, seed: int,
                    weights: list[float]) -> tuple[list[dict], dict]:
    nodes = fresh_nodes(seed, scenario_id)
    runtimes = {n.name: {"queue": [], "busy_until": None, "completed": 0,
                         "service_total_ms": 0.0, "energy_total_j": 0.0} for n in nodes}
    decisions: list[dict] = []

    def start_queued(node: Node, runtime: dict, start_at: float) -> None:
        if not runtime["queue"]:
            return
        queue = runtime["queue"]
        aged_candidates = [i for i, item in enumerate(queue)
                           if start_at-item["task"].arrival_ms >= CONFIG["max_queue_wait_target_ms"]]
        # Fairness override: once a waiting task reaches the target, serve the oldest
        # eligible task at the next available non-preemptive dispatch opportunity.
        if policy in ("aging_priority", "ai_ridge") and aged_candidates:
            index = min(aged_candidates, key=lambda i: queue[i]["task"].arrival_ms)
        elif policy == "priority_aware":
            index = min(range(len(queue)), key=lambda i: (-queue[i]["task"].priority,
                         queue[i]["task"].deadline_ms, queue[i]["task"].arrival_ms))
        elif policy == "aging_priority":
            def aging_score(i: int) -> tuple[float, float]:
                task = queue[i]["task"]
                age_ms = min(CONFIG["waiting_age_cap_ms"], max(0.0, start_at-task.arrival_ms))
                effective_priority = task.priority + age_ms/1000.0
                return (-effective_priority, task.deadline_ms)
            index = min(range(len(queue)), key=aging_score)
        elif policy == "ai_ridge":
            # Learned latency, explicit task priority, and waiting time guide dispatch.
            def urgency(i: int) -> tuple[float, float]:
                task = queue[i]["task"]
                pred = dot(weights, features(task, node, 0.0))
                age_ms = min(CONFIG["waiting_age_cap_ms"], max(0.0, start_at-task.arrival_ms))
                score = task.deadline_ms-pred-task.priority*CONFIG["priority_score_weight_ms"]
                score -= age_ms*CONFIG["waiting_age_weight"]
                return (score, task.arrival_ms)
            index = min(range(len(queue)), key=urgency)
        else:
            index = min(range(len(queue)), key=lambda i: queue[i]["task"].arrival_ms)
        item = queue.pop(index)
        task: Task = item["task"]
        actual_start = max(start_at, task.arrival_ms)
        queue_wait = actual_start - task.arrival_ms
        latency, service_ms, energy_j = outcome(task, node, actual_start, queue_wait)
        finish = task.arrival_ms + latency
        runtime["busy_until"] = finish
        runtime["service_total_ms"] += service_ms
        runtime["energy_total_j"] += energy_j
        decisions.append({
            "scenario_id": scenario_id, "policy": policy, "task_id": task.task_id,
            "load_regime": task.load_regime, "task_type": task.task_type, "arrival_ms": round(task.arrival_ms, 4),
            "payload_mb": round(task.payload_mb, 6), "cpu_mi_assumed": round(task.cpu_mi, 4),
            "compute_intensity_class": task.compute_intensity_class,
            "workload_source": task.workload_source, "reference_task_id": task.reference_task_id,
            "deadline_ms": round(task.deadline_ms, 4), "priority": task.priority,
            "selected_node": node.name, "selected_tier": node.tier,
            "queue_wait_ms": round(queue_wait, 4), "latency_ms_simulated": round(latency, 4),
            "service_time_ms_simulated": round(service_ms, 4), "deadline_met": int(latency <= task.deadline_ms),
            "estimated_energy_j": round(energy_j, 8), "task_status": "accepted", "rejection_reason": "",
        })

    def advance_node(node: Node, until_ms: float) -> None:
        runtime = runtimes[node.name]
        while runtime["busy_until"] is not None and runtime["busy_until"] <= until_ms:
            finished_at = runtime["busy_until"]
            runtime["busy_until"] = None
            runtime["completed"] += 1
            start_queued(node, runtime, finished_at)

    for task in tasks:
        for node in nodes:
            advance_node(node, task.arrival_ms)
        node = choose(policy, task, nodes, weights)
        runtime = runtimes[node.name]
        # Keep a forecast of the tail of assigned work for placement decisions.
        queued_ms = max(0.0, node.available_ms-task.arrival_ms)
        if policy == "ai_ridge":
            predicted_latency = dot(weights, features(task, node, queued_ms))
        else:
            predicted_latency = outcome(task, node, node.available_ms, queued_ms)[0]
        if policy in ("admission_control", "ai_ridge") and predicted_latency > task.deadline_ms:
            reason = "learned_latency_exceeds_deadline" if policy == "ai_ridge" else "predicted_latency_exceeds_deadline"
            decisions.append({
                "scenario_id": scenario_id, "policy": policy, "task_id": task.task_id,
                "load_regime": task.load_regime, "task_type": task.task_type, "arrival_ms": round(task.arrival_ms, 4),
                "payload_mb": round(task.payload_mb, 6), "cpu_mi_assumed": round(task.cpu_mi, 4),
                "compute_intensity_class": task.compute_intensity_class,
                "workload_source": task.workload_source, "reference_task_id": task.reference_task_id,
                "deadline_ms": round(task.deadline_ms, 4), "priority": task.priority,
                "selected_node": "", "selected_tier": "", "queue_wait_ms": "",
                "latency_ms_simulated": "", "service_time_ms_simulated": "", "deadline_met": 0,
                "estimated_energy_j": 0.0, "task_status": "rejected",
                "rejection_reason": reason,
            })
            continue
        _, service_ms, _ = outcome(task, node, node.available_ms, queued_ms)
        node.available_ms = max(task.arrival_ms, node.available_ms) + service_ms + task.payload_mb*8000.0/node.bandwidth_mbps + node.network_ms
        runtime["queue"].append({"task": task})
        if runtime["busy_until"] is None:
            start_queued(node, runtime, task.arrival_ms)

    for node in nodes:
        runtime = runtimes[node.name]
        while runtime["busy_until"] is not None or runtime["queue"]:
            if runtime["busy_until"] is not None:
                runtime["completed"] += 1
                finished_at = runtime["busy_until"]
                runtime["busy_until"] = None
            else:
                finished_at = 0.0
            start_queued(node, runtime, finished_at)
    accepted = [r for r in decisions if r["task_status"] == "accepted"]
    latencies = [float(r["latency_ms_simulated"]) for r in accepted]
    total_elapsed_s = max((float(r["arrival_ms"]) + float(r["latency_ms_simulated"]) for r in accepted), default=0) / 1000.0
    horizon_ms = max((float(r["arrival_ms"]) + float(r["latency_ms_simulated"]) for r in accepted), default=1.0)
    node_rows = []
    for node in nodes:
        base_util = next(s["initial_utilization"] for s in CONFIG["nodes"] if s["name"] == node.name)
        runtime = runtimes[node.name]
        dynamic_util = min(1.0, base_util + runtime["service_total_ms"] / max(horizon_ms, 1.0) * (1.0-base_util))
        node_rows.append({"node": node.name, "tier": node.tier, "tasks_assigned": runtime["completed"],
                          "utilization_proxy": dynamic_util})
    metrics = {
        "scenario_id": scenario_id, "load_regime": tasks[0].load_regime if tasks else "unknown", "policy": policy, "task_count": len(tasks),
        "accepted_count": len(accepted), "rejected_count": len(tasks)-len(accepted),
        "acceptance_rate": round(len(accepted)/max(1, len(tasks)), 6),
        "rejection_rate": round((len(tasks)-len(accepted))/max(1, len(tasks)), 6),
        "mean_latency_ms": round(sum(latencies)/max(1, len(latencies)), 4),
        "p95_latency_ms": round(percentile(latencies, 0.95), 4),
        "max_queue_wait_ms": round(max((float(r["queue_wait_ms"]) for r in accepted), default=0.0), 4),
        "queue_wait_over_target_rate": round(sum(float(r["queue_wait_ms"]) > CONFIG["max_queue_wait_target_ms"] for r in accepted)/max(1, len(accepted)), 6),
        "deadline_success_rate": round(sum(int(r["deadline_met"]) for r in accepted)/max(1, len(accepted)), 6),
        "deadline_success_over_offered_rate": round(sum(int(r["deadline_met"]) for r in accepted)/max(1, len(tasks)), 6),
        "throughput_tasks_per_s": round(len(accepted)/max(total_elapsed_s, 1e-9), 4),
        "estimated_energy_j_total": round(sum(r["estimated_energy_j"] for r in decisions), 6),
        "mean_cpu_utilization_proxy": round(sum(r["utilization_proxy"] for r in node_rows)/len(node_rows), 6),
        "throughput_per_estimated_joule": round(len(accepted)/max(sum(r["estimated_energy_j"] for r in decisions), 1e-12), 6),
        "resource_balance_gini": round(gini([r["tasks_assigned"] for r in node_rows]), 6),
        "edge_task_share": round(sum(r["selected_tier"] == "edge" for r in accepted)/max(1, len(accepted)), 6),
    }
    return decisions, metrics


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered)-1)*p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return ordered[lo] + (ordered[hi]-ordered[lo])*(pos-lo)


def gini(values: list[int]) -> float:
    values = sorted(max(0, x) for x in values)
    total = sum(values)
    if not values or total == 0:
        return 0.0
    n = len(values)
    return sum((2*(i+1)-n-1)*x for i, x in enumerate(values))/(n*total)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="simulated results")
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--scenarios", type=int, default=CONFIG["scenarios"])
    parser.add_argument("--tasks-per-scenario", type=int, default=CONFIG["tasks_per_scenario"])
    parser.add_argument("--training-scenarios", type=int, default=CONFIG["training_scenarios"])
    parser.add_argument("--ai-energy-weight", type=float, default=CONFIG["ai_energy_weight_ms_per_j"])
    parser.add_argument("--ai-utilization-weight", type=float, default=CONFIG["ai_utilization_weight_ms"])
    parser.add_argument("--priority-score-weight", type=float, default=CONFIG["priority_score_weight_ms"])
    parser.add_argument("--training-seed", type=int, default=None,
                        help="Optional independent seed for fitting the AI latency model.")
    parser.add_argument("--capacity-scale", type=float, default=CONFIG["capacity_scale"],
                        help="Scale configured node CPU rates while keeping workload and network fixed.")
    parser.add_argument("--bandwidth-scale", type=float, default=CONFIG["bandwidth_scale"],
                        help="Scale configured link bandwidth while keeping workload fixed.")
    parser.add_argument("--network-latency-scale", type=float, default=CONFIG["network_latency_scale"],
                        help="Multiply configured network propagation latency by this positive factor.")
    parser.add_argument("--nodes-per-tier", type=int, default=CONFIG["nodes_per_tier"],
                        help="Replicate each configured node this many times for a horizontal-scaling scenario.")
    parser.add_argument("--workload-source", choices=("synthetic_archetypes", "reference_marginals"),
                        default=CONFIG["workload_source"],
                        help="Generate tasks from model archetypes or sample reference task size/deadline/priority/type marginals.")
    parser.add_argument("--reference-task-file", type=Path,
                        default=Path(__file__).resolve().parent / "processed dataset" / "task_decision_dataset.csv",
                        help="Consolidated task table used only when --workload-source=reference_marginals.")
    args = parser.parse_args()
    if (args.scenarios < 1 or args.tasks_per_scenario < 1 or args.training_scenarios < 1
            or args.capacity_scale <= 0 or args.bandwidth_scale <= 0 or args.network_latency_scale <= 0
            or args.nodes_per_tier < 1):
        parser.error("scenario/task counts and resource/network/node scale factors must be positive")

    global REFERENCE_TASK_ROWS, REFERENCE_TASK_SHA256, REFERENCE_TASK_FILE
    if args.workload_source == "reference_marginals":
        reference_path = args.reference_task_file.expanduser().resolve()
        if not reference_path.is_file():
            parser.error(f"Reference task table not found: {reference_path}")
        try:
            REFERENCE_TASK_ROWS = load_reference_tasks(reference_path)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        payloads = sorted(float(row["_payload_mb"]) for row in REFERENCE_TASK_ROWS)
        CONFIG["reference_payload_q1_mb"] = payloads[int((len(payloads)-1)/3)]
        CONFIG["reference_payload_q2_mb"] = payloads[int(2*(len(payloads)-1)/3)]
        REFERENCE_TASK_SHA256 = hashlib.sha256(reference_path.read_bytes()).hexdigest()
        REFERENCE_TASK_FILE = reference_path.name

    if args.nodes_per_tier > 1:
        expanded = []
        for spec in CONFIG["nodes"]:
            for replica in range(1, args.nodes_per_tier + 1):
                name = spec["name"] if replica == 1 else f"{spec['name']}_replica_{replica}"
                expanded.append({**spec, "name": name})
        CONFIG["nodes"] = expanded

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    CONFIG["ai_energy_weight_ms_per_j"] = args.ai_energy_weight
    CONFIG["ai_utilization_weight_ms"] = args.ai_utilization_weight
    CONFIG["priority_score_weight_ms"] = args.priority_score_weight
    CONFIG["capacity_scale"] = args.capacity_scale
    CONFIG["bandwidth_scale"] = args.bandwidth_scale
    CONFIG["network_latency_scale"] = args.network_latency_scale
    CONFIG["nodes_per_tier"] = args.nodes_per_tier
    CONFIG["workload_source"] = args.workload_source
    training_seed = args.training_seed if args.training_seed is not None else args.seed + 1_000_000
    weights = train_latency_model(training_seed, args.training_scenarios, min(args.tasks_per_scenario, 200))
    all_decisions: list[dict] = []
    all_metrics: list[dict] = []
    for sid in range(args.scenarios):
        # Identical arrivals and task attributes are replayed independently for each policy.
        tasks = make_scenario(sid, args.tasks_per_scenario, args.seed + 500000)
        for policy in POLICIES:
            decisions, metrics = simulate_policy(tasks, sid, policy, args.seed, weights)
            all_decisions.extend(decisions)
            all_metrics.append(metrics)
    write_csv(output / "simulated_task_decisions.csv", all_decisions)
    write_csv(output / "simulated_scenario_metrics.csv", all_metrics)
    aggregate = []
    for policy in POLICIES:
        rows = [x for x in all_metrics if x["policy"] == policy]
        aggregate.append({"policy": policy, "scenarios": len(rows),
                          **{key: round(sum(r[key] for r in rows)/len(rows), 4) for key in rows[0]
                             if key not in ("scenario_id", "load_regime", "policy", "task_count")},
                          "task_count_per_scenario": args.tasks_per_scenario})
    write_csv(output / "policy_comparison_summary.csv", aggregate)
    by_load = []
    for regime in [x["name"] for x in CONFIG["load_profiles"]]:
        for policy in POLICIES:
            rows = [x for x in all_metrics if x["load_regime"] == regime and x["policy"] == policy]
            if not rows:
                continue
            by_load.append({"load_regime": regime, "policy": policy, "scenarios": len(rows),
                            **{key: round(sum(float(r[key]) for r in rows)/len(rows), 4) for key in rows[0]
                               if key not in ("scenario_id", "load_regime", "policy", "task_count")}})
    write_csv(output / "policy_comparison_by_load.csv", by_load)
    by_priority = []
    for regime in [x["name"] for x in CONFIG["load_profiles"]]:
        for policy in POLICIES:
            for priority in range(1, 6):
                rows = [r for r in all_decisions if r["load_regime"] == regime
                        and r["policy"] == policy and r["priority"] == priority]
                if not rows:
                    continue
                accepted = [r for r in rows if r["task_status"] == "accepted"]
                latencies = [float(r["latency_ms_simulated"]) for r in accepted]
                by_priority.append({
                    "load_regime": regime, "policy": policy, "priority": priority,
                    "attempted_count": len(rows), "accepted_count": len(accepted),
                    "rejected_count": len(rows)-len(accepted),
                    "rejection_rate": round((len(rows)-len(accepted))/len(rows), 6),
                    "mean_latency_ms": round(sum(latencies)/max(1, len(latencies)), 4),
                    "p95_latency_ms": round(percentile(latencies, 0.95), 4),
                    "deadline_success_rate": round(sum(int(r["deadline_met"]) for r in accepted)/max(1, len(accepted)), 6),
                    "deadline_success_over_offered_rate": round(sum(int(r["deadline_met"]) for r in accepted)/len(rows), 6),
                    "mean_queue_wait_ms": round(sum(float(r["queue_wait_ms"]) for r in accepted)/max(1, len(accepted)), 4),
                    "p95_queue_wait_ms": round(percentile([float(r["queue_wait_ms"]) for r in accepted], 0.95), 4),
                    "queue_wait_over_target_rate": round(sum(float(r["queue_wait_ms"]) > CONFIG["max_queue_wait_target_ms"] for r in accepted)/max(1, len(accepted)), 6),
                })
    write_csv(output / "policy_comparison_by_priority.csv", by_priority)
    effective_config = {**CONFIG, "seed": args.seed, "training_seed": training_seed, "scenarios": args.scenarios,
                       "tasks_per_scenario": args.tasks_per_scenario,
                       "training_scenarios": args.training_scenarios,
                       "reference_task_table": ({"file_name": REFERENCE_TASK_FILE,
                           "sha256": REFERENCE_TASK_SHA256, "valid_rows": len(REFERENCE_TASK_ROWS),
                           "sampled_fields": ["task_input_mb", "deadline_ms", "task_priority", "task_type"],
                           "cpu_demand_assignment": "CPU MI is modeled from payload tertiles using the configured telemetry/event-detection/image-inference archetype ranges; source task categories are retained separately.",
                           "arrivals": "Inter-arrival times are generated from the selected nominal/busy/overloaded model regime, not replayed from source timestamps.",
                           "source_outcomes_used": False} if args.workload_source == "reference_marginals" else None),
                       "queue_model_note": "Each node is modeled as a non-preemptive single-server queue. FIFO baselines serve in arrival order; priority_aware serves higher priority then earlier deadline; aging_priority adds one effective priority level per 1,000 ms of waiting; aging_priority and ai_ridge promote the oldest queued task once it reaches max_queue_wait_target_ms, at the next non-preemptive dispatch opportunity.",
                       "ai_model_note": "Ridge regression is fit on separate scenarios. Training labels are generated by the same documented computational equations, never from source actual-latency, SLA, or selected-destination labels. The AI selects deadline-feasible nodes using learned latency, estimated energy and utilization.",
                       "admission_control_note": "Admission control rejects a task when its predicted completion latency on the currently fastest candidate node exceeds that task's deadline. Rejected tasks have no execution-latency outcome; attempted, accepted, rejected, and deadline-success-over-offered counts are reported separately.",
                       "outputs_note": "Policy decisions and performance outcomes are generated by the documented software model. In reference_marginals mode, observed source task size, deadline, priority, and category marginals shape generated workloads; arrivals, CPU work, node conditions, counterfactual outcomes, and energy remain modeled. Source archive provenance requires independent verification; no physical deployment results are claimed."}
    (output / "simulation_assumptions.json").write_text(json.dumps(effective_config, indent=2), encoding="utf-8")
    print(f"Wrote simulated_task_decisions.csv ({len(all_decisions):,} rows)")
    print(f"Wrote simulated_scenario_metrics.csv ({len(all_metrics):,} rows)")
    print(f"Wrote policy_comparison_summary.csv ({len(aggregate):,} rows)")
    print(f"Wrote policy_comparison_by_load.csv ({len(by_load):,} rows)")
    print(f"Wrote policy_comparison_by_priority.csv ({len(by_priority):,} rows)")
    print(f"Wrote simulation_assumptions.json")
    print("Performance results are model-generated estimates from a software-based computational evaluation; no physical IoT deployment measurements are used.")


if __name__ == "__main__":
    main()
