#!/usr/bin/env python3
"""Build initial project datasets from the downloaded ZIP archives.

Usage (from the project root after saving this file there):
    python build_project_datasets.py

Or specify paths explicitly:
    python build_project_datasets.py --raw-dir "D:/.../raw dataset" --output-dir "D:/.../processed dataset"

Uses only Python's standard library. Original ZIP files are read in place and
are not extracted or modified. This first ingestion pass creates observed
historical decisions and measured/supplied metrics; it does not fabricate
counterfactual outcomes, CPU-cycle demand, or energy measurements.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import io
import math
import statistics
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

PRIORITY_MAP = {"low": 1, "medium": 2, "high": 3, "critical": 4}
WINDOW_SECONDS = 900
DEFAULT_MAX_TELEMETRY_AGE_SECONDS = 1800


def find_archive(raw_dir: Path, token: str) -> Path | None:
    token = token.casefold()
    matches = [p for p in raw_dir.glob("*.zip") if token in p.name.casefold()]
    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(f"More than one archive matches {token!r}: {matches}")
    return matches[0]


def csv_rows_from_zip(zf: zipfile.ZipFile, member: str) -> list[dict[str, str]]:
    with zf.open(member) as stream:
        text = io.TextIOWrapper(stream, encoding="utf-8-sig", newline="")
        return list(csv.DictReader(text))


def find_nested_zip(zf: zipfile.ZipFile, suffix: str) -> str:
    matches = [n for n in zf.namelist() if n.casefold().endswith(suffix.casefold())]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one nested archive ending {suffix!r}; found {len(matches)}")
    return matches[0]


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        # The UCI reference CSV uses strings such as
        # "Thu Nov 26 14:47:38 2020" rather than ISO 8601.
        try:
            dt = datetime.strptime(value.strip(), "%a %b %d %H:%M:%S %Y")
        except ValueError:
            return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def iso_utc(dt: datetime | None) -> str:
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z") if dt else ""


def number(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def num_text(value: float | None) -> str:
    return "" if value is None else f"{value:.6g}"


def rounded_window(dt: datetime) -> datetime:
    epoch = int(dt.timestamp())
    return datetime.fromtimestamp(epoch - epoch % WINDOW_SECONDS, tz=timezone.utc)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def read_xstox(archive: Path, max_age_seconds: int) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, int]]:
    with zipfile.ZipFile(archive) as outer:
        nested = find_nested_zip(outer, "dataset.zip")
        with outer.open(nested) as nested_stream, zipfile.ZipFile(nested_stream) as data:
            names = data.namelist()
            task_members = [n for n in names if n.startswith("dataset/task_allocation/") and n.casefold().endswith(".csv")]
            task_members = [n for n in task_members if not n.casefold().endswith("readme.csv")]
            if not task_members:
                raise RuntimeError("No task allocation CSVs found in the Xstox archive")
            tasks: list[dict[str, str]] = []
            for member in sorted(task_members):
                tasks.extend(csv_rows_from_zip(data, member))

            inventory: dict[str, dict[str, str]] = {}
            for member in names:
                if member.endswith("/edge_node_inventory.csv") or member.endswith("/cloud_vm_inventory.csv"):
                    for row in csv_rows_from_zip(data, member):
                        if row.get("instance_id"):
                            inventory[row["instance_id"]] = row

            telemetry_by_node: dict[str, list[tuple[datetime, dict[str, str]]]] = defaultdict(list)
            telemetry_count = 0
            telemetry_members = [
                n for n in names
                if n.startswith(("dataset/edge/", "dataset/cloud/"))
                and n.casefold().endswith(".csv")
                and not n.endswith("inventory.csv")
            ]
            for member in sorted(telemetry_members):
                for row in csv_rows_from_zip(data, member):
                    dt = parse_dt(row.get("timestamp"))
                    node = row.get("instance_id", "")
                    if dt and node:
                        telemetry_by_node[node].append((dt, row))
                        telemetry_count += 1
            for node in telemetry_by_node:
                telemetry_by_node[node].sort(key=lambda pair: pair[0])

            time_index = {node: [pair[0] for pair in pairs] for node, pairs in telemetry_by_node.items()}
            output: list[dict[str, str]] = []
            task_by_id: dict[str, dict[str, str]] = {}
            for raw in tasks:
                task_id = raw.get("task_id", "")
                if not task_id:
                    continue
                if task_id in task_by_id:
                    raise RuntimeError(f"Duplicate task_id in task allocation files: {task_id}")
                task_by_id[task_id] = raw
                requested = parse_dt(raw.get("requested_at"))
                assigned = raw.get("assigned_instance_id", "")
                pairs = telemetry_by_node.get(assigned, [])
                times = time_index.get(assigned, [])
                snap: dict[str, str] = {}
                snap_dt: datetime | None = None
                age: float | None = None
                if requested and pairs:
                    idx = bisect.bisect_right(times, requested) - 1
                    if idx >= 0:
                        snap_dt, snap = pairs[idx]
                        age = (requested - snap_dt).total_seconds()
                        if age > max_age_seconds:
                            snap, snap_dt, age = {}, None, None

                inv = inventory.get(assigned, {})
                util_cpu = number(snap.get("cpu_utilization_pct"))
                util_mem = number(snap.get("memory_utilization_pct"))
                vcpu = number(snap.get("vcpu_count") or inv.get("vcpu_count"))
                mem_gb = number(snap.get("memory_gb") or inv.get("memory_gb"))
                available_vcpu = vcpu * (1 - util_cpu / 100) if vcpu is not None and util_cpu is not None else None
                available_mem = mem_gb * 1024 * (1 - util_mem / 100) if mem_gb is not None and util_mem is not None else None
                observed_latency = number(raw.get("observed_latency_ms"))
                expected_latency = number(raw.get("expected_latency_ms"))
                deadline_met = 1 if raw.get("sla_met", "").casefold() == "yes" else (0 if raw.get("sla_met", "").casefold() == "no" else "")
                priority = PRIORITY_MAP.get(raw.get("priority", "").casefold(), "")
                destination_type = (snap.get("deployment_tier") or inv.get("deployment_tier") or raw.get("source_tier") or "").casefold()
                if destination_type == "hybrid":
                    destination_type = "edge" if (inv.get("deployment_tier", "").casefold() == "edge") else "cloud"

                row = {
                    "scenario_id": "xstox_claimed_external",
                    "decision_id": task_id,
                    "task_id": task_id,
                    "candidate_id": assigned,
                    "timestamp_utc": iso_utc(requested),
                    "telemetry_timestamp_utc": iso_utc(snap_dt),
                    "telemetry_age_sec": num_text(age),
                    "task_type": raw.get("task_category", ""),
                    "task_priority": priority,
                    "task_priority_source_label": raw.get("priority", ""),
                    "task_input_mb": raw.get("data_size_mb", ""),
                    "task_output_mb": "",
                    "required_cpu_mi": "",
                    "required_memory_mb": "",
                    "deadline_ms": raw.get("expected_latency_ms", ""),
                    "source_node_id": "",
                    "source_node_type": raw.get("source_tier", ""),
                    "uplink_bandwidth_mbps": "",
                    "downlink_bandwidth_mbps": "",
                    "network_latency_ms": snap.get("network_latency_ms", ""),
                    "network_load_pct": "",
                    "node_network_in_mbps": snap.get("network_in_mbps", ""),
                    "node_network_out_mbps": snap.get("network_out_mbps", ""),
                    "destination_id": assigned,
                    "destination_type": destination_type,
                    "allocation_mode": raw.get("task_allocation", ""),
                    "partner_destination_id": raw.get("partner_instance_id", ""),
                    "available_cpu_mi_per_sec": "",
                    "estimated_available_vcpu": num_text(available_vcpu),
                    "available_memory_mb": num_text(available_mem),
                    "cpu_utilization_pct": snap.get("cpu_utilization_pct", ""),
                    "memory_utilization_pct": snap.get("memory_utilization_pct", ""),
                    "queue_length": snap.get("queue_depth", ""),
                    "estimated_queue_wait_ms": "",
                    "estimated_transfer_ms": "",
                    "estimated_execution_ms": "",
                    "estimated_total_latency_ms": "",
                    "selected": 1,
                    "actual_execution_ms": "",
                    "actual_transfer_ms": "",
                    "actual_total_latency_ms": num_text(observed_latency),
                    "deadline_met_source_label": raw.get("sla_met", ""),
                    "deadline_met": deadline_met,
                    "energy_joules": "",
                    "task_status": raw.get("status", ""),
                    "completion_pct": raw.get("completion_pct", ""),
                    "observed_latency_ms_source": raw.get("observed_latency_ms", ""),
                    "expected_latency_ms_source": raw.get("expected_latency_ms", ""),
                    "actual_start": raw.get("actual_start", ""),
                    "actual_end": raw.get("actual_end", ""),
                    "target_tier_source": raw.get("target_tier", ""),
                    "task_source": "xstox_claimed_external",
                    "network_source": "xstox_claimed_external" if snap else "missing_snapshot",
                    "resource_source": "xstox_claimed_external" if snap else "missing_snapshot",
                    "derived_field_source": "python_mapping_v1",
                    "execution_model_version": "not_modeled_v1",
                    "energy_model_version": "not_available_v1",
                }
                output.append(row)

            info = {
                "tasks": len(tasks),
                "unique_tasks": len(task_by_id),
                "telemetry_rows": telemetry_count,
                "telemetry_nodes": len(telemetry_by_node),
                "task_files": len(task_members),
                "inventory_nodes": len(inventory),
            }
            return output, tasks, info


def read_uci(archive: Path | None) -> list[dict[str, str]]:
    if archive is None:
        return []
    output: list[dict[str, str]] = []
    with zipfile.ZipFile(archive) as outer:
        nested = find_nested_zip(outer, "TATDescriptionDataset.zip")
        with outer.open(nested) as stream, zipfile.ZipFile(stream) as data:
            for member in sorted(n for n in data.namelist() if n.casefold().endswith(".csv")):
                server = Path(member).stem
                for raw in csv_rows_from_zip(data, member):
                    seconds = number(raw.get("Execution Time"))
                    dt = parse_dt(raw.get("Time"))
                    output.append({
                        "scenario_id": "uci_image_recognition_reference",
                        "server_profile": server,
                        "timestamp_utc": iso_utc(dt),
                        "turnaround_seconds": num_text(seconds),
                        "turnaround_ms": num_text(seconds * 1000 if seconds is not None else None),
                        "source_field_semantics": "turnaround from start of image transfer through receipt of result; not pure CPU execution time",
                        "source_dataset": "UCI Image Recognition Task Execution Times in Mobile Edge Computing",
                    })
    return output


def make_metrics(tasks: list[dict[str, str]], decisions: list[dict[str, str]]) -> list[dict[str, str]]:
    arrivals: dict[datetime, list[dict[str, str]]] = defaultdict(list)
    completions: dict[datetime, list[dict[str, str]]] = defaultdict(list)
    telemetry: dict[datetime, list[dict[str, str]]] = defaultdict(list)
    for task in tasks:
        dt = parse_dt(task.get("requested_at"))
        if dt:
            arrivals[rounded_window(dt)].append(task)
        end = parse_dt(task.get("actual_end"))
        if end and task.get("status", "").casefold() == "completed":
            completions[rounded_window(end)].append(task)
    for row in decisions:
        dt = parse_dt(row.get("telemetry_timestamp_utc"))
        if dt:
            telemetry[rounded_window(dt)].append(row)
    windows = sorted(set(arrivals) | set(completions) | set(telemetry))
    result = []
    for win in windows:
        arrived = arrivals.get(win, [])
        done = completions.get(win, [])
        states = telemetry.get(win, [])
        lats = [x for task in done if (x := number(task.get("observed_latency_ms"))) is not None]
        misses = 0
        for task in done:
            actual = number(task.get("observed_latency_ms"))
            deadline = number(task.get("expected_latency_ms"))
            if actual is not None and deadline is not None and deadline > 0 and actual > deadline:
                misses += 1
        cpu = [x for row in states if (x := number(row.get("cpu_utilization_pct"))) is not None]
        mem = [x for row in states if (x := number(row.get("memory_utilization_pct"))) is not None]
        sla_no = sum(task.get("sla_met", "").casefold() == "no" for task in arrived)
        result.append({
            "scenario_id": "xstox_claimed_external",
            "window_start_utc": iso_utc(win),
            "window_duration_sec": str(WINDOW_SECONDS),
            "tasks_arrived": str(len(arrived)),
            "tasks_completed": str(len(done)),
            "tasks_sla_failed_by_source_label": str(sla_no),
            "tasks_latency_over_target_among_completed": str(misses),
            "throughput_tasks_per_sec": num_text(len(done) / WINDOW_SECONDS),
            "mean_latency_ms_completed": num_text(statistics.fmean(lats) if lats else None),
            "p95_latency_ms_completed": num_text(percentile(lats, .95)),
            "p99_latency_ms_completed": num_text(percentile(lats, .99)),
            "mean_cpu_utilization_pct": num_text(statistics.fmean(cpu) if cpu else None),
            "mean_memory_utilization_pct": num_text(statistics.fmean(mem) if mem else None),
            "total_energy_joules": "",
            "energy_per_completed_task_joules": "",
            "energy_status": "not supplied by source; not inferred",
        })
    return result


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_manifest(raw_dir: Path, archive_status: list[dict[str, str]], outputs: list[tuple[str, int]]) -> list[dict[str, str]]:
    manifest = []
    for record in archive_status:
        manifest.append({"record_type": "source_archive", **record})
    for name, count in outputs:
        manifest.append({
            "record_type": "generated_output",
            "name": name,
            "status": "created",
            "details": f"{count} rows; output directory is separate from raw archives",
        })
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path(__file__).resolve().parent / "raw dataset",
                        help="Folder containing the downloaded ZIP archives")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "processed dataset",
                        help="Folder to receive generated CSV files")
    parser.add_argument("--max-telemetry-age-sec", type=int, default=DEFAULT_MAX_TELEMETRY_AGE_SECONDS,
                        help="Maximum age of prior node telemetry used for a task join (default: 1800)")
    args = parser.parse_args()

    raw_dir = args.raw_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not raw_dir.is_dir():
        parser.error(f"Raw dataset folder does not exist: {raw_dir}")

    xstox = find_archive(raw_dir, "xstoxservices")
    if xstox is None:
        parser.error("Could not find the Xstoxservices ZIP in --raw-dir")
    uci = find_archive(raw_dir, "image+recognition+task+execution")
    ecw = find_archive(raw_dir, "ecwdataset")
    edge = find_archive(raw_dir, "edgeworkloadstraces")
    alibaba = find_archive(raw_dir, "clusterdata-master")

    decisions, raw_tasks, info = read_xstox(xstox, args.max_telemetry_age_sec)
    metrics = make_metrics(raw_tasks, decisions)
    uci_rows = read_uci(uci)

    output_dir.mkdir(parents=True, exist_ok=True)
    task_path = output_dir / "task_decision_dataset.csv"
    metrics_path = output_dir / "scenario_metrics_dataset.csv"
    uci_path = output_dir / "execution_reference_dataset.csv"
    write_csv(task_path, decisions)
    write_csv(metrics_path, metrics)

    output_rows = [(task_path.name, len(decisions)), (metrics_path.name, len(metrics))]
    archive_status = [
        {"name": xstox.name, "status": "parsed", "details": f"{info['tasks']} task rows; {info['telemetry_rows']} telemetry rows; {info['inventory_nodes']} inventory nodes; provenance should be independently verified"},
        {"name": uci.name if uci else "UCI execution-time archive", "status": "parsed" if uci else "not_found", "details": f"{len(uci_rows)} turnaround reference rows" if uci else "optional archive absent"},
        {"name": ecw.name if ecw else "ECWDataset archive", "status": "available_not_merged" if ecw else "not_found", "details": "normalized NumPy/pickle workload arrays; retained as a separate scenario source" if ecw else "optional archive absent"},
        {"name": edge.name if edge else "EdgeWorkloadsTraces archive", "status": "documentation_only" if edge else "not_found", "details": "NEP raw trace CSVs are not present in the downloaded ZIP" if edge else "optional archive absent"},
        {"name": alibaba.name if alibaba else "Alibaba clusterdata archive", "status": "schemas_only_for_v2018" if alibaba else "not_found", "details": "raw v2018 trace tables are not present in the downloaded ZIP" if alibaba else "optional archive absent"},
    ]
    if uci_rows:
        write_csv(uci_path, uci_rows)
        output_rows.append((uci_path.name, len(uci_rows)))
    manifest = build_manifest(raw_dir, archive_status, output_rows)
    write_csv(output_dir / "data_manifest.csv", manifest)

    print(f"Task decisions: {task_path} ({len(decisions):,} rows)")
    print(f"Scenario metrics: {metrics_path} ({len(metrics):,} rows)")
    if uci_rows:
        print(f"UCI execution references: {uci_path} ({len(uci_rows):,} rows)")
    print(f"Source/output manifest: {output_dir / 'data_manifest.csv'}")
    print("Note: task rows contain only the historically selected destination. Counterfactual candidate outcomes and measured energy are not created by this script.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, zipfile.BadZipFile, RuntimeError, KeyError) as exc:
        print(f"Dataset build failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
