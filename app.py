from __future__ import annotations
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
SIMULATOR = APP_DIR / "edge_cloud_simulator.py"
MULTI_SEED_EVALUATOR = APP_DIR / "multi_seed_evaluation.py"
SCALABILITY_STUDY = APP_DIR / "scalability_study.py"
DATA_BUILDER = APP_DIR / "build_project_datasets.py"
DEFAULT_RESULTS = APP_DIR / "simulated results"
DEFAULT_PROCESSED = APP_DIR / "processed dataset"


def results_dir() -> Path:
    path = Path(st.session_state.get("results_dir", str(DEFAULT_RESULTS))).expanduser()
    return (path if path.is_absolute() else APP_DIR / path).resolve()

st.set_page_config(page_title="Edge Scheduler Lab", page_icon="⚡", layout="wide")
st.markdown("""
<style>
  .block-container {max-width: 1440px; padding-top: 1.5rem;}
  [data-testid="stMetric"] {background:#182535; border:1px solid #2b3d51; padding:14px 16px; border-radius:12px;}
  [data-testid="stMetricLabel"] {color:#c6d3e1 !important;}
  [data-testid="stMetricValue"] {color:#f7fbff !important;}
  [data-testid="stMetricDelta"] {color:#9bd4ff !important;}
  .hero {padding:24px 28px; border-radius:16px; background:linear-gradient(110deg,#102a43,#176b87); color:white; margin-bottom:18px;}
  .hero h1 {color:white; margin:0 0 6px 0; font-size:2rem;}
  .hero p {color:#d9edf5; margin:0; font-size:1rem;}
  .note {color:#d8e7f1; padding:12px 16px; border-left:4px solid #2e86ab; background:#142c3a; border-radius:6px;}
  .note b {color:#ffffff;}
</style>
""", unsafe_allow_html=True)


def csv_if_exists(name: str) -> pd.DataFrame:
    path = results_dir() / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def run_process(command: list[str], cwd: Path) -> tuple[int, str]:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return result.returncode, (result.stdout + "\n" + result.stderr).strip()


def reports_archive() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(results_dir().glob("*")):
            if path.is_file() and path.suffix.lower() in (".csv", ".json"):
                archive.write(path, arcname=path.name)
    return buffer.getvalue()


st.markdown("""
<div class="hero">
  <h1>Delay-Sensitive IoT Applications in Edge Computing Networks with AI- Empowered Fast Task Execution Decision</h1>
  <p>Prepare reference datasets and evaluate edge–cloud scheduling policies across latency, deadlines, energy, and resource balance.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Experiment controls")
    scenarios = st.number_input("Evaluation scenarios", min_value=3, max_value=500, value=20, step=1,
                                help="Use at least three scenarios to include nominal, busy, and overloaded load profiles.")
    tasks = st.number_input("Tasks per scenario", min_value=10, max_value=10000, value=400, step=50)
    training_scenarios = st.number_input("AI training scenarios", min_value=1, max_value=500, value=12, step=1)
    st.caption("Validated resource profile (software-model sensitivity)")
    cpu_scale = st.number_input("CPU capacity scale", min_value=0.25, max_value=10.0, value=2.25, step=0.25,
                                help="Multiplier applied to configured edge/cloud CPU rates. 1.0 is the original baseline.")
    bandwidth_scale = st.number_input("Network bandwidth scale", min_value=0.25, max_value=10.0, value=5.0, step=0.25,
                                      help="Multiplier applied to configured link bandwidth. 1.0 is the original baseline.")
    network_latency_scale = st.number_input("Network propagation latency scale", min_value=0.1, max_value=3.0, value=0.25, step=0.05,
                                            help="Multiplier applied to configured network propagation delay. This is a modeled scenario assumption.")
    nodes_per_tier = st.number_input("Replicas per node tier", min_value=1, max_value=8, value=1, step=1,
                                     help="Experimental horizontal-scaling control. Each replica adds another modeled node with the same tier configuration.")
    reference_table_available = (DEFAULT_PROCESSED / "task_decision_dataset.csv").is_file()
    workload_options = ["synthetic_archetypes"]
    if reference_table_available:
        workload_options.append("reference_marginals")
    workload_label = st.selectbox("Workload source", workload_options,
                                  format_func=lambda value: "Model-generated archetypes" if value == "synthetic_archetypes" else "Reference-informed task marginals",
                                  help="Reference-informed mode samples input size, deadline, priority, and task category from the consolidated task table; arrivals, CPU work, and outcomes remain model-generated.")
    reference_task_file = st.text_input("Consolidated task table", value=str(DEFAULT_PROCESSED / "task_decision_dataset.csv"),
                                        disabled=workload_label != "reference_marginals")
    if workload_label == "reference_marginals":
        st.caption("Uses source payload-size, SLA-target, priority, and task-category marginals. The dataset README does not define bandwidth path or latency measurement boundaries well enough to calibrate transfer latency; arrivals, CPU demand, resource states, policy outcomes, and energy remain model-generated.")
    evaluation_seeds = st.number_input("Independent evaluation seeds", min_value=2, max_value=100, value=30, step=1,
                                       help="Each seed trains and evaluates on separate generated scenarios. Confidence intervals are calculated across seed-level results.")
    seed = st.number_input("Random seed", min_value=0, max_value=2_000_000_000, value=2026, step=1)
    results_text = st.text_input("Results folder", value=str(DEFAULT_RESULTS))
    st.session_state["results_dir"] = results_text
    run_simulation = st.button("Run policy comparison", type="primary", width="stretch")
    run_robust_evaluation = st.button("Run multi-seed evaluation", width="stretch",
                                      help="Runs independent seeds and reports 95% confidence intervals. Results are saved in a robust evaluation subfolder.")
    scalability_seeds = st.number_input("Scalability evaluation seeds", min_value=2, max_value=100, value=30, step=1)
    run_scalability = st.button("Run scalability study (1, 2, 4 replicas)", width="stretch",
                                help="Compares horizontal node replication with the same generated task stream and reports seed-level confidence intervals.")

    st.divider()
    with st.expander("Prepare downloaded reference datasets"):
        raw_text = st.text_input("Raw dataset folder", value=str(APP_DIR / "raw dataset"))
        processed_text = st.text_input("Processed dataset folder", value=str(DEFAULT_PROCESSED))
        raw_folder_available = Path(raw_text).expanduser().is_dir()
        build_data = st.button("Build reference datasets", width="stretch", disabled=not raw_folder_available)
        if not raw_folder_available:
            st.info("Dataset archives are not included in this hosted deployment. The model-generated workload option remains available.")
        st.caption("This creates consolidated reference tables. Policy-comparison outcomes are generated by the documented software model; the reference tables do not provide counterfactual outcomes for every policy or measured energy values.")

if build_data:
    if not DATA_BUILDER.exists():
        st.error("Dataset builder file is missing from the application folder.")
    else:
        with st.spinner("Reading archives and building the reference tables…"):
            code, log = run_process([sys.executable, str(DATA_BUILDER), "--raw-dir", raw_text,
                                     "--output-dir", processed_text], APP_DIR)
        if code == 0:
            st.success("Reference datasets built.")
        else:
            st.error("Dataset preparation did not finish. Check the paths and details below.")
        if log:
            st.code(log)

if run_simulation:
    if not SIMULATOR.exists():
        st.error("Simulator file is missing from the application folder.")
    else:
        destination = Path(results_text).expanduser()
        if not destination.is_absolute():
            destination = APP_DIR / destination
        destination = destination.resolve()
        command = [sys.executable, str(SIMULATOR), "--output-dir", str(destination),
                   "--seed", str(int(seed)), "--scenarios", str(int(scenarios)),
                   "--tasks-per-scenario", str(int(tasks)), "--training-scenarios", str(int(training_scenarios)),
                   "--capacity-scale", str(float(cpu_scale)), "--bandwidth-scale", str(float(bandwidth_scale)),
                   "--network-latency-scale", str(float(network_latency_scale)),
                   "--nodes-per-tier", str(int(nodes_per_tier)), "--workload-source", workload_label,
                   "--reference-task-file", reference_task_file,
                   "--ai-energy-weight", "0", "--ai-utilization-weight", "20", "--priority-score-weight", "35"]
        with st.spinner("Running the scenarios and comparing scheduling policies…"):
            code, log = run_process(command, APP_DIR)
        if code == 0:
            st.session_state["last_run_log"] = log
            st.session_state["last_run_ok"] = True
            st.rerun()
        else:
            st.session_state["last_run_log"] = log
            st.session_state["last_run_ok"] = False

if run_robust_evaluation:
    if not MULTI_SEED_EVALUATOR.exists():
        st.error("Multi-seed evaluation file is missing from the application folder.")
    else:
        destination = Path(results_text).expanduser()
        if not destination.is_absolute():
            destination = APP_DIR / destination
        destination = destination.resolve() / "robust evaluation"
        command = [sys.executable, str(MULTI_SEED_EVALUATOR), "--output-dir", str(destination),
                   "--seed-start", str(int(seed)), "--seeds", str(int(evaluation_seeds)),
                   "--scenarios", str(int(scenarios)), "--tasks-per-scenario", str(int(tasks)),
                   "--training-scenarios", str(int(training_scenarios)),
                   "--capacity-scale", str(float(cpu_scale)), "--bandwidth-scale", str(float(bandwidth_scale)),
                   "--network-latency-scale", str(float(network_latency_scale))]
        command.extend(["--nodes-per-tier", str(int(nodes_per_tier)), "--workload-source", workload_label,
                        "--reference-task-file", reference_task_file])
        with st.spinner(f"Evaluating {int(evaluation_seeds)} independent seeds…"):
            code, log = run_process(command, APP_DIR)
        st.session_state["last_robust_log"] = log
        st.session_state["last_robust_ok"] = code == 0
        if code == 0:
            st.rerun()

if run_scalability:
    if not SCALABILITY_STUDY.exists():
        st.error("Scalability study file is missing from the application folder.")
    else:
        destination = Path(results_text).expanduser()
        if not destination.is_absolute():
            destination = APP_DIR / destination
        destination = destination.resolve() / "scalability study"
        command = [sys.executable, str(SCALABILITY_STUDY), "--output-dir", str(destination),
                   "--seed-start", str(int(seed)), "--seeds", str(int(scalability_seeds)),
                   "--scenarios", str(int(scenarios)), "--tasks-per-scenario", str(int(tasks)),
                   "--training-scenarios", str(int(training_scenarios)),
                   "--capacity-scale", str(float(cpu_scale)), "--bandwidth-scale", str(float(bandwidth_scale)),
                   "--network-latency-scale", str(float(network_latency_scale)),
                   "--workload-source", workload_label, "--reference-task-file", reference_task_file]
        with st.spinner(f"Evaluating 1, 2, and 4 replicas per node tier across {int(scalability_seeds)} seeds…"):
            code, log = run_process(command, APP_DIR)
        st.session_state["last_scalability_log"] = log
        st.session_state["last_scalability_ok"] = code == 0
        if code == 0:
            st.rerun()

if "last_robust_log" in st.session_state:
    if st.session_state.get("last_robust_ok"):
        st.success("Multi-seed evaluation completed.")
    else:
        st.error("Multi-seed evaluation failed. Review the details below.")
    with st.expander("Multi-seed run details"):
        st.code(st.session_state["last_robust_log"] or "No additional output.")

if "last_scalability_log" in st.session_state:
    if st.session_state.get("last_scalability_ok"):
        st.success("Scalability study completed.")
    else:
        st.error("Scalability study failed. Review the details below.")
    with st.expander("Scalability run details"):
        st.code(st.session_state["last_scalability_log"] or "No additional output.")

if "last_run_log" in st.session_state:
    if st.session_state.get("last_run_ok"):
        st.success("Simulation run completed.")
    else:
        st.error("Simulation run failed. Review the details below.")
    with st.expander("Run details"):
        st.code(st.session_state["last_run_log"] or "No additional output.")

summary = csv_if_exists("policy_comparison_summary.csv")
by_load = csv_if_exists("policy_comparison_by_load.csv")
by_priority = csv_if_exists("policy_comparison_by_priority.csv")
scenario_metrics = csv_if_exists("simulated_scenario_metrics.csv")
task_decisions = csv_if_exists("simulated_task_decisions.csv")
assumption_path = results_dir() / "simulation_assumptions.json"
robust_dir = results_dir() / "robust evaluation"
robust_summary_path = robust_dir / "robust_policy_summary.csv"
robust_load_path = robust_dir / "robust_policy_by_load.csv"
robust_priority_path = robust_dir / "robust_policy_by_priority.csv"
robust_summary = pd.read_csv(robust_summary_path) if robust_summary_path.exists() else pd.DataFrame()
robust_by_load = pd.read_csv(robust_load_path) if robust_load_path.exists() else pd.DataFrame()
scale_dir = results_dir() / "scalability study"
scale_summary_path = scale_dir / "scalability_summary.csv"
scale_load_path = scale_dir / "scalability_by_load.csv"
scale_summary = pd.read_csv(scale_summary_path) if scale_summary_path.exists() else pd.DataFrame()
scale_by_load = pd.read_csv(scale_load_path) if scale_load_path.exists() else pd.DataFrame()

if not summary.empty:
    try:
        assumptions = json.loads(assumption_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        assumptions = {}
    st.markdown('<div class="note"><b>Evaluation provenance:</b> This is a software-based computational evaluation. The policy-comparison workloads and performance metrics are model-generated estimates. CPU, bandwidth, and network-delay controls define the evaluated resource scenario; consolidated source datasets are retained as reference data and do not provide counterfactual outcomes for every policy. Source payload, bandwidth, and latency fields lack sufficient measurement-scope detail for latency calibration. No physical IoT deployment measurements are used.</div>', unsafe_allow_html=True)
    st.write("")
    tabs = st.tabs(["Overview", "Load analysis", "Priority analysis", "Task decisions", "Assumptions & export", "Robust evaluation", "Scalability"])

    with tabs[0]:
        metric = "deadline_success_over_offered_rate" if "deadline_success_over_offered_rate" in summary else "deadline_success_rate"
        best_row = summary.sort_values(metric, ascending=False).iloc[0]
        cols = st.columns(4)
        cols[0].metric("Scenarios", int(assumptions.get("scenarios", summary["scenarios"].max())))
        cols[1].metric("Tasks per policy", f"{int(assumptions.get('tasks_per_scenario', 0))*int(summary['scenarios'].max()):,}")
        cols[2].metric("Policies compared", summary.shape[0])
        cols[3].metric("Top offered deadline success", f"{float(best_row[metric])*100:.1f}%", str(best_row["policy"]))

        st.subheader("Policy comparison")
        st.caption("Latency and energy are averaged over accepted tasks for policies that reject work. Deadline success over offered tasks counts rejected tasks as unsuccessful.")
        display = summary.copy()
        if "deadline_success_over_offered_rate" in display:
            display["deadline_success_over_offered_%"] = (display["deadline_success_over_offered_rate"]*100).round(2)
        st.dataframe(display, width="stretch", hide_index=True)

        chart_metric = st.selectbox("Chart metric", [
            "deadline_success_over_offered_rate", "mean_latency_ms", "estimated_energy_j_total",
            "throughput_tasks_per_s", "rejection_rate", "mean_cpu_utilization_proxy", "resource_balance_gini"
        ], format_func=lambda x: x.replace("_", " ").title(), key="overview_metric")
        chart = summary.copy()
        if chart_metric.endswith("rate"):
            chart[chart_metric] = chart[chart_metric]*100
        fig = px.bar(chart, x="policy", y=chart_metric, color="policy", title=chart_metric.replace("_", " ").title())
        fig.update_layout(showlegend=False, xaxis_title="Scheduling policy", yaxis_title=chart_metric.replace("_", " ").title())
        # Streamlit's default Plotly chart width is stretch. Omitting width here
        # also avoids a deprecation warning in Streamlit 1.50.x.
        st.plotly_chart(fig)

    with tabs[1]:
        st.subheader("Performance by offered load")
        if by_load.empty:
            st.info("Load-specific results are not available yet.")
        else:
            load = st.selectbox("Load regime", sorted(by_load["load_regime"].unique()))
            metric = st.selectbox("Metric", ["deadline_success_over_offered_rate", "mean_latency_ms", "rejection_rate",
                                               "estimated_energy_j_total", "throughput_tasks_per_s", "queue_wait_over_target_rate"],
                                  format_func=lambda x: x.replace("_", " ").title(), key="load_metric")
            plot_data = by_load[by_load["load_regime"] == load].copy()
            if metric.endswith("rate"):
                plot_data[metric] *= 100
            fig = px.bar(plot_data, x="policy", y=metric, color="policy", title=f"{metric.replace('_',' ').title()} — {load}")
            fig.update_layout(showlegend=False, xaxis_title="Scheduling policy", yaxis_title=metric.replace("_", " ").title())
            st.plotly_chart(fig)
            st.dataframe(plot_data, width="stretch", hide_index=True)

    with tabs[2]:
        st.subheader("Outcomes by task priority")
        if by_priority.empty:
            st.info("Priority-specific results are not available yet.")
        else:
            load = st.selectbox("Load regime", sorted(by_priority["load_regime"].unique()), key="priority_load")
            metric = st.selectbox("Metric", ["deadline_success_over_offered_rate", "mean_latency_ms", "rejection_rate", "mean_queue_wait_ms"],
                                  format_func=lambda x: x.replace("_", " ").title(), key="priority_metric")
            plot_data = by_priority[by_priority["load_regime"] == load].copy()
            if metric.endswith("rate"):
                plot_data[metric] *= 100
            fig = px.line(plot_data, x="priority", y=metric, color="policy", markers=True,
                          title=f"{metric.replace('_',' ').title()} by priority — {load}")
            st.plotly_chart(fig)
            st.dataframe(plot_data, width="stretch", hide_index=True)

    with tabs[3]:
        st.subheader("Task-level decisions")
        if task_decisions.empty:
            st.info("Task-level rows are not available yet.")
        else:
            f1, f2, f3 = st.columns(3)
            selected_load = f1.selectbox("Load", ["All"]+sorted(task_decisions["load_regime"].unique().tolist()))
            selected_policy = f2.selectbox("Policy", ["All"]+sorted(task_decisions["policy"].unique().tolist()))
            selected_status = f3.selectbox("Status", ["All"]+sorted(task_decisions["task_status"].dropna().unique().tolist()) if "task_status" in task_decisions else ["All"])
            filtered = task_decisions
            if selected_load != "All": filtered = filtered[filtered["load_regime"] == selected_load]
            if selected_policy != "All": filtered = filtered[filtered["policy"] == selected_policy]
            if selected_status != "All" and "task_status" in filtered: filtered = filtered[filtered["task_status"] == selected_status]
            st.caption(f"Showing {len(filtered):,} of {len(task_decisions):,} task-decision rows.")
            st.dataframe(filtered.head(5000), width="stretch", hide_index=True)
            if len(filtered) > 5000: st.caption("Table display is limited to 5,000 rows. Download the full filtered dataset below.")
            st.download_button("Download visible task decisions", filtered.to_csv(index=False),
                               file_name="filtered_task_decisions.csv", mime="text/csv")

    with tabs[4]:
        st.subheader("Assumptions and exports")
        st.json(assumptions)
        available = [p for p in ["policy_comparison_summary.csv", "policy_comparison_by_load.csv",
                                  "policy_comparison_by_priority.csv", "simulated_scenario_metrics.csv",
                                  "simulated_task_decisions.csv", "simulation_assumptions.json"]
                     if (results_dir() / p).exists()]
        cols = st.columns(3)
        for i, name in enumerate(available):
            path = results_dir() / name
            mime = "application/json" if path.suffix == ".json" else "text/csv"
            cols[i % 3].download_button(f"Download {name}", path.read_bytes(), file_name=name, mime=mime, key=f"download_{name}")
        if available:
            st.download_button("Download all reports as ZIP", reports_archive(), file_name="edge_scheduler_reports.zip",
                               mime="application/zip", type="primary")
    with tabs[5]:
        st.subheader("Independent-seed evaluation")
        if robust_summary.empty:
            st.info("Run a multi-seed evaluation from the left panel to estimate performance variation and 95% confidence intervals.")
        else:
            st.caption("Intervals are calculated across independent seed-level means. They describe variation in the computational model, not physical deployment uncertainty.")
            st.dataframe(robust_summary, width="stretch", hide_index=True)
            if not robust_by_load.empty:
                robust_metric = st.selectbox("Metric with 95% interval", [
                    "deadline_success_over_offered_rate", "mean_latency_ms", "p95_latency_ms",
                    "throughput_tasks_per_s", "estimated_energy_j_total", "mean_cpu_utilization_proxy",
                    "resource_balance_gini"
                ], format_func=lambda x: x.replace("_", " ").title(), key="robust_metric")
                load_options = sorted(robust_by_load["load_regime"].unique())
                selected_robust_load = st.selectbox("Load regime", load_options, key="robust_load")
                plot_data = robust_by_load[robust_by_load["load_regime"] == selected_robust_load].copy()
                mean_col = f"{robust_metric}_mean"
                low_col = f"{robust_metric}_ci95_low"
                high_col = f"{robust_metric}_ci95_high"
                if robust_metric.endswith("rate"):
                    for column in (mean_col, low_col, high_col):
                        plot_data[column] *= 100
                plot_data["ci95_error"] = (plot_data[high_col] - plot_data[mean_col]).clip(lower=0)
                fig = px.bar(plot_data, x="policy", y=mean_col, color="policy", error_y="ci95_error",
                             title=f"{robust_metric.replace('_', ' ').title()} — {selected_robust_load}")
                fig.update_layout(showlegend=False, xaxis_title="Scheduling policy",
                                  yaxis_title=robust_metric.replace("_", " ").title())
                st.plotly_chart(fig)
                st.dataframe(plot_data, width="stretch", hide_index=True)
            export_files = [p for p in (robust_summary_path, robust_load_path, robust_priority_path,
                                         robust_dir / "per_seed_policy_summary.csv",
                                         robust_dir / "robust_evaluation_assumptions.json") if p.exists()]
            for path in export_files:
                mime = "application/json" if path.suffix == ".json" else "text/csv"
                st.download_button(f"Download {path.name}", path.read_bytes(), file_name=path.name,
                                   mime=mime, key=f"robust_download_{path.name}")
    with tabs[6]:
        st.subheader("Horizontal scaling analysis")
        if scale_summary.empty:
            st.info("Run the scalability study from the left panel to compare one, two, and four node replicas per tier.")
        else:
            st.caption("Replicas add modeled CPU and bandwidth capacity. Workload draws are held reproducible across scales; intervals summarize seed-level model variation.")
            st.dataframe(scale_summary, width="stretch", hide_index=True)
            if not scale_by_load.empty:
                scale_metric = st.selectbox("Scalability metric", ["deadline_success_over_offered_rate_mean", "mean_latency_ms_mean", "p95_latency_ms_mean", "throughput_tasks_per_s_mean", "estimated_energy_j_total_mean", "mean_cpu_utilization_proxy_mean"], format_func=lambda x: x.replace("_", " ").title(), key="scale_metric")
                scale_load = st.selectbox("Scalability load", sorted(scale_by_load["load_regime"].unique()), key="scale_load")
                plot_data = scale_by_load[scale_by_load["load_regime"] == scale_load].copy()
                fig = px.line(plot_data, x="nodes_per_tier", y=scale_metric, color="policy", markers=True,
                              title=f"{scale_metric.replace('_mean', '').replace('_', ' ').title()} — {scale_load}")
                st.plotly_chart(fig)
                st.dataframe(plot_data, width="stretch", hide_index=True)
            for path in [scale_summary_path, scale_load_path,
                         scale_dir / "scalability_assumptions.json"]:
                if path.exists():
                    mime = "application/json" if path.suffix == ".json" else "text/csv"
                    st.download_button(f"Download {path.name}", path.read_bytes(), file_name=path.name,
                                       mime=mime, key=f"scale_download_{path.name}")
else:
    st.info("Run a policy comparison from the left panel, or open this app from the project folder containing existing simulation results.")
    st.markdown("#### What this app provides")
    st.write("Prepare consolidated reference tables, run reproducible model-based scenarios, compare policies by load and priority, inspect decisions, and export CSV/JSON reports.")
    if not robust_summary.empty:
        st.subheader("Independent-seed evaluation")
        st.caption("Intervals summarize variation across model-generated runs, not physical deployment uncertainty.")
        st.dataframe(robust_summary, width="stretch", hide_index=True)
        if not robust_by_load.empty:
            st.markdown("#### Results by offered load")
            st.dataframe(robust_by_load, width="stretch", hide_index=True)
