#!/usr/bin/env python3
"""
HW2 Task 2 — live demonstration runner (real LLM, simulated focusing bench).

Runs the unchanged FocusingAgentSystem (hw2/task2_agent.py: the HW1 four-role
LangGraph system whose executor owns only FocusingSession.call) with the real
HW1 ModelCaller. The user supplies only the goal; the orchestrator LLM routes,
and the executor LLM chooses all 13 experiment settings. Nothing here selects a
route or a setting, and there is no fallback to a scripted model.

Bounds (system constraints, not LLM decisions):
  session   16 x 16 DMD, 32 x 32 camera, medium seed 7, default shot/read noise,
            at most 2000 camera acquisitions per tool call
  run       at most 3 completed experiments, 4000 camera acquisitions,
            8 orchestrator decisions, 3 argument proposals per executor turn
  API       at most --max-api-calls ModelCaller.call invocations (default 30);
            reaching the cap stops the run safely as incomplete, keeps every
            measurement already produced and reports no final result

Evidence (a NEW directory, default task6_runs/hw2_task2_live_<timestamp>/):
  dialogue.json, dialogue.txt, run_record.json, demo_evidence.json,
  experiments.json, experiments.csv, artifacts/<run_id>_callNN/ (tool artifacts),
  final_report.txt (only when the orchestrator finished with a completed experiment)
Values of credential-like environment variables are redacted from every file
this runner writes and from its terminal output.

Usage:
    python -m hw2.demo_task2_live [--model gpt-6-luna] [--max-api-calls 30]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from hw2 import task2_agent as t2
from hw2.focusing_tool import FocusingSession
from hw2.task1_hitl import SECRET_ENV_NAME, OverwriteRefused, ensure_fresh_workdir, write_new_file

DEFAULT_MODEL = "gpt-6-luna"
DEFAULT_REQUEST = (
    "Focus the beam through the simulated scattering medium. Choose the experimental settings "
    "autonomously, perform focusing experiments, evaluate the independently validated "
    "enhancement, and report the results."
)
SESSION_CONFIG = {
    "dmd_shape": (16, 16),
    "camera_shape": (32, 32),
    "medium_seed": 7,
    "noise": None,                 # the tool's existing default shot/read noise model
    "acquisition_limit": 2000,     # per tool call
}
MAX_COMPLETED_EXPERIMENTS = 3
RUN_ACQUISITION_LIMIT = 4000
DEFAULT_MAX_API_CALLS = 30
HARD_MAX_API_CALLS = 60
BACKEND_REAL = "real ModelCaller (task6/agentic_system.py, OpenAI Responses API)"

SETTING_COLUMNS = list(t2.REQUIRED_SETTINGS)
TABLE_COLUMNS = (
    ["invocation", "run_id", "status", "error", "camera_acquisitions"] + SETTING_COLUMNS + [
        "validated_initial_enhancement_mean", "validated_initial_enhancement_std",
        "validated_final_enhancement_mean", "validated_final_enhancement_std", "validation_count",
        "enhancement_gain", "focus_intensity_gain", "background_suppression_factor",
        "search_best_observed_enhancement_BIASED_not_a_result",
        "search_final_accepted_enhancement_BIASED_not_a_result",
        "artifact_dir", "committed_to_graph_state",
    ]
)


class ApiCallBudgetExceeded(RuntimeError):
    """The run's ModelCaller.call cap was reached; no further model call is made."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def secret_values() -> list[str]:
    """Values of credential-like environment variables (never printed or saved)."""
    return sorted({v for k, v in os.environ.items() if SECRET_ENV_NAME.search(k) and len(v) >= 8},
                  key=len, reverse=True)


def redact(text: str) -> str:
    for value in secret_values():
        text = text.replace(value, "[REDACTED]")
    return text


def write_text(path: Path, text: str) -> Path:
    return write_new_file(path, redact(text))


def write_json(path: Path, data) -> Path:
    return write_text(path, json.dumps(data, indent=2, ensure_ascii=False, default=str))


class BudgetedLLM:
    """Counts and records every ModelCaller.call; refuses calls beyond the cap.

    The cap is checked BEFORE the wrapped model is called, so the number of real
    API calls never exceeds max_calls.
    """

    def __init__(self, llm, max_calls: int, out: Callable[[str], None]):
        self.llm = llm
        self.max_calls = max_calls
        self.out = out
        self.calls: list[dict] = []
        self.refused = 0

    @property
    def count(self) -> int:
        return len(self.calls)

    def call(self, instructions: str, input_text: str) -> str:
        role = instructions.split()[3].lower() if instructions.startswith("You are the") else "unknown"
        if self.count >= self.max_calls:
            self.refused += 1
            raise ApiCallBudgetExceeded(
                f"API call budget of {self.max_calls} ModelCaller.call invocations reached "
                f"(refused a {role} call)"
            )
        started = time.perf_counter()
        output = self.llm.call(instructions, input_text)
        self.calls.append({
            "n": self.count + 1, "role": role, "time_utc": now(),
            "elapsed_s": round(time.perf_counter() - started, 2),
            "instructions_first_line": instructions.splitlines()[0],
            "input_text": input_text, "output_text": output,
        })
        if role == "executor":
            self.out(f"\n[LLM PLANNING] executor LLM proposed tool arguments "
                     f"(API call {self.count}/{self.max_calls}):\n{output}")
        return output


class ConsoleFocusingSession(FocusingSession):
    """The approved FocusingSession; call() also prints the invocation and its outcome."""

    out: Callable[[str], None] = print

    def call(self, arguments):
        self.out(f"\n[SIMULATED MEASUREMENT] FocusingSession.call({json.dumps(arguments)})")
        result = super().call(arguments)
        if result.get("ok") is True:
            v = result["validated"]
            self.out(
                f"[SIMULATED MEASUREMENT] tool result: SUCCESS run_id={result['run_name']} "
                f"camera_acquisitions={result['acquisitions']['counted_by_bench']}\n"
                f"[VALIDATED RESULT] initial enhancement {v['initial']['enhancement_mean']:.4g} "
                f"(std {v['initial']['enhancement_std']:.3g}) -> final enhancement "
                f"{v['final']['enhancement_mean']:.4g} (std {v['final']['enhancement_std']:.3g}); "
                f"enhancement gain {t2._fmt(v['enhancement_gain'])}x (local contrast); "
                f"focus intensity gain {t2._fmt(v['focus_intensity_gain'])}x (absolute brightening); "
                f"background suppression {t2._fmt(v['background_suppression_factor'])}x"
            )
        else:
            self.out(f"[SIMULATED MEASUREMENT] tool result: REJECTED (nothing measured): {result.get('error')}")
        return result


def experiment_rows(records: list[dict], committed: set[str]) -> list[dict]:
    rows = []
    for e in records:
        row = {c: None for c in TABLE_COLUMNS}
        row.update({
            "invocation": e["invocation"], "run_id": e["run_name"], "status": e["status"],
            "error": e["error"], "camera_acquisitions": e["acquisitions"],
            "committed_to_graph_state": e["run_name"] in committed,
        })
        row.update({k: e["arguments"].get(k) for k in SETTING_COLUMNS})
        if e["status"] == t2.COMPLETED:
            s, search = e["summary"], e["result"]["search"]
            row.update({k: s[k] for k in (
                "validated_initial_enhancement_mean", "validated_initial_enhancement_std",
                "validated_final_enhancement_mean", "validated_final_enhancement_std", "validation_count",
                "enhancement_gain", "focus_intensity_gain", "background_suppression_factor")})
            row["search_best_observed_enhancement_BIASED_not_a_result"] = search["best_observed_enhancement"]
            row["search_final_accepted_enhancement_BIASED_not_a_result"] = search["final_accepted_enhancement"]
            row["artifact_dir"] = Path(s["artifacts"]["result"]).parent.as_posix()
        rows.append(row)
    return rows


def rows_csv(rows: list[dict]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=TABLE_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def completion_status(outcome: dict) -> str:
    if outcome["status"] == "error" and "ApiCallBudgetExceeded" in (outcome["error"] or ""):
        return "incomplete_api_call_budget_exhausted"
    return {"completed": "completed",
            "decision_budget_exhausted": "incomplete_decision_budget_exhausted"}.get(outcome["status"], "error")


def make_turn_printer(out: Callable[[str], None]) -> Callable[[dict], None]:
    seen = {"decisions": 0, "completed": 0}

    def on_turn(turn: dict) -> None:
        role, text = turn["role"], turn["text"]
        if role == "user":
            out(f"\n===== USER REQUEST =====\n{text}")
        elif role == "orchestrator":
            seen["decisions"] += 1
            d = json.loads(text)
            out(f"\n===== [LLM PLANNING] ORCHESTRATOR decision {seen['decisions']} of {t2.MAX_DECISIONS} =====\n"
                f"LLM-selected destination: {d['next_role']}\ninstruction: {d['instruction']}\nreason: {d['reason']}")
            if seen["completed"]:
                another = {"executor": "YES (executor selected)",
                           "finish": "NO (orchestrator chose to finish)"}.get(d["next_role"], "not in this decision")
                out(f"Another experiment requested: {another}")
        elif role == "scientist":
            out(f"\n===== [LLM PLANNING] SCIENTIST plan (no tool) =====\n{text}")
        elif role == "general":
            out(f"\n===== [LLM PLANNING] GENERAL answer (no tool) =====\n{text}")
        elif role == "executor":
            if text.startswith(t2.COMPLETED_MARK):
                seen["completed"] += 1
            out(f"\n===== EXECUTOR turn (tool report; measured values are VALIDATED tool output) =====\n{text}")

    return on_turn


def run_live(llm, *, model: str, task_text: str, workdir: Path, thread_id: str,
             max_api_calls: int = DEFAULT_MAX_API_CALLS, backend: str = BACKEND_REAL,
             out: Callable[[str], None] = print) -> dict:
    """Run the bounded demonstration with `llm` and write the evidence. Returns the evidence."""
    if not 1 <= max_api_calls <= HARD_MAX_API_CALLS:
        raise ValueError(f"max_api_calls must be in 1..{HARD_MAX_API_CALLS}, got {max_api_calls}")
    workdir = ensure_fresh_workdir(Path(workdir).resolve())
    workdir.mkdir(parents=True, exist_ok=True)

    def say(text: str) -> None:
        out(redact(text))

    started_utc, started = now(), time.perf_counter()
    budgeted = BudgetedLLM(llm, max_api_calls, say)
    session = ConsoleFocusingSession(**SESSION_CONFIG, artifact_root=workdir / "artifacts")
    session.out = say
    run_id = f"live_{datetime.now():%Y%m%d_%H%M%S}"
    system = t2.FocusingAgentSystem(budgeted, session, run_id=run_id,
                                    max_experiments=MAX_COMPLETED_EXPERIMENTS,
                                    run_acquisition_limit=RUN_ACQUISITION_LIMIT)
    description = {k: v for k, v in session.describe().items() if k != "argument_schema"}
    limits = {
        "max_completed_experiments": MAX_COMPLETED_EXPERIMENTS,
        "run_camera_acquisition_limit": RUN_ACQUISITION_LIMIT,
        "per_call_camera_acquisition_limit": SESSION_CONFIG["acquisition_limit"],
        "max_orchestrator_decisions": t2.MAX_DECISIONS,
        "max_argument_proposals_per_executor_turn": system.max_argument_attempts,
        "max_api_calls": max_api_calls,
    }

    say("=" * 78)
    say(f"HW2 TASK 2 LIVE DEMONSTRATION - LLM backend: {backend}")
    say("Simulation only: the agent's only tool is FocusingSession.call on a simulated bench.")
    say("=" * 78)
    say(f"Model: {model}\nWorkdir: {workdir}\nThread: {thread_id}\nRun id: {run_id}")
    say(f"Session (system constraints): {json.dumps(description)}")
    say(f"Run limits: {json.dumps(limits)}")

    outcome = None
    crash = None
    try:
        outcome = t2.run_agent(system, task_text, thread_id=thread_id, on_turn=make_turn_printer(say))
    except Exception as exc:  # run_agent catches node errors; this guards the evidence anyway
        crash = f"{type(exc).__name__}: {exc}"
    finally:
        runtime_s = time.perf_counter() - started
        evidence = write_evidence(workdir, system, budgeted, outcome, crash, model=model, backend=backend,
                                  task_text=task_text, thread_id=thread_id, run_id=run_id,
                                  description=description, limits=limits, started_utc=started_utc,
                                  runtime_s=runtime_s)
    print_summary(evidence, workdir, say)
    return evidence


def write_evidence(workdir: Path, system, budgeted: BudgetedLLM, outcome: dict | None, crash: str | None,
                   **meta) -> dict:
    dialogue = outcome["dialogue"] if outcome else []
    committed_records = outcome["experiments"] if outcome else []
    committed = {e["run_name"] for e in committed_records}
    # Every tool call, including any from an executor turn interrupted before it committed.
    all_records = sorted(system._tool_ledger.values(), key=lambda e: e["invocation"])
    rows = experiment_rows(all_records, committed)
    status = completion_status(outcome) if outcome else "error"
    final_report = outcome["final_report"] if outcome else None
    session_now = system._session.describe()

    write_json(workdir / "dialogue.json", dialogue)
    write_text(workdir / "dialogue.txt",
               "\n\n".join(f"===== {t['role'].upper()} =====\n{t['text']}" for t in dialogue))
    if outcome:
        write_json(workdir / "run_record.json", t2.run_record(outcome))
    else:
        write_json(workdir / "run_record.json", {"status": "error", "error": crash})
    write_json(workdir / "experiments.json", rows)
    write_text(workdir / "experiments.csv", rows_csv(rows))
    if final_report:
        write_text(workdir / "final_report.txt", final_report["message"] + "\n")

    evidence = {
        "scenario": "HW2 Task 2 live demonstration: real LLM agent, simulated focusing bench",
        "llm_backend": meta["backend"],
        "model": meta["model"],
        "user_request": meta["task_text"],
        "thread_id": meta["thread_id"],
        "run_id": meta["run_id"],
        "workdir": workdir.as_posix(),
        "started_utc": meta["started_utc"],
        "finished_utc": now(),
        "runtime_s": round(meta["runtime_s"], 2),
        "completion_status": status,
        "error": crash or (outcome["error"] if outcome else None),
        "session_constraints": meta["description"],
        "limits": meta["limits"],
        "orchestrator_decisions": outcome["orchestrator_decisions"] if outcome else [],
        "num_orchestrator_decisions": outcome["num_orchestrator_decisions"] if outcome else 0,
        "llm_selected_parameter_sets": [
            {"invocation": e["invocation"], "run_id": e["run_name"], "arguments": e["arguments"],
             "rationale": e["rationale"], "status": e["status"]} for e in all_records
        ],
        "tool_calls": {
            "total": len(all_records),
            "completed": [e["run_name"] for e in all_records if e["status"] == t2.COMPLETED],
            "failed": [{"run_id": e["run_name"], "status": e["status"], "error": e["error"]}
                       for e in all_records if e["status"] != t2.COMPLETED],
            "not_committed_to_graph_state": sorted(set(system._tool_ledger) - committed),
        },
        "validated_results": [
            {"run_id": e["run_name"], **{k: v for k, v in e["summary"].items() if k != "arguments"}}
            for e in all_records if e["status"] == t2.COMPLETED
        ],
        "total_camera_acquisitions": sum(e["acquisitions"] for e in all_records),
        "session_camera_acquisitions": session_now["camera_acquisitions_so_far"],
        "transmission_fingerprint": session_now["transmission_fingerprint"],
        "api_call_count": budgeted.count,
        "api_calls_refused_by_cap": budgeted.refused,
        "api_calls_by_role": {r: sum(c["role"] == r for c in budgeted.calls)
                              for r in ("orchestrator", "scientist", "general", "executor")},
        "final_report": final_report,
        "partial_report": outcome["partial_report"] if outcome else None,
        "llm_calls": budgeted.calls,
        "files": ["dialogue.json", "dialogue.txt", "run_record.json", "demo_evidence.json",
                  "experiments.json", "experiments.csv", "artifacts/"]
                 + (["final_report.txt"] if final_report else []),
    }
    write_json(workdir / "demo_evidence.json", evidence)
    return evidence


def print_summary(evidence: dict, workdir: Path, say: Callable[[str], None]) -> None:
    say("\n===== TOOL CALLS (FocusingSession.call) =====")
    for p in evidence["llm_selected_parameter_sets"]:
        say(f"{p['run_id']}: status={p['status']} arguments={json.dumps(p['arguments'], sort_keys=True)}")
    if evidence["final_report"]:
        say("\n===== FINAL REPORT (program-built from validated tool output) =====")
        say(evidence["final_report"]["message"])
    else:
        say("\n===== PARTIAL REPORT (run incomplete; NO final result is reported) =====")
        say(json.dumps(evidence["partial_report"], indent=2, default=str))
    say("\n===== RUN SUMMARY =====")
    say(f"completion status: {evidence['completion_status']}")
    if evidence["error"]:
        say(f"stopped because: {evidence['error']}")
    say(f"orchestrator decisions ({evidence['num_orchestrator_decisions']}): {evidence['orchestrator_decisions']}")
    say(f"API calls: {evidence['api_call_count']} of at most {evidence['limits']['max_api_calls']} "
        f"{evidence['api_calls_by_role']}")
    say(f"tool calls: {evidence['tool_calls']['total']} (completed {len(evidence['tool_calls']['completed'])})")
    say(f"total simulated camera acquisitions: {evidence['total_camera_acquisitions']} "
        f"of at most {evidence['limits']['run_camera_acquisition_limit']}")
    say(f"transmission fingerprint (same medium for every call): {evidence['transmission_fingerprint']}")
    say(f"runtime: {evidence['runtime_s']} s")
    say(f"evidence: {(workdir / 'demo_evidence.json').as_posix()}")


def main(argv: list[str] | None = None) -> int:
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    parser = argparse.ArgumentParser(
        prog="python -m hw2.demo_task2_live",
        description="HW2 Task 2 live demonstration: the real LLM agent runs bounded simulated "
                    "focusing experiments (makes real OpenAI API calls).",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default {DEFAULT_MODEL}).")
    parser.add_argument("--prompt", default=DEFAULT_REQUEST, help="The user's goal (default: the Task 2 request).")
    parser.add_argument("--workdir", default=str(t2.REPO_ROOT / "task6_runs" / f"hw2_task2_live_{stamp}"),
                        help="NEW evidence directory (must not exist or must be empty).")
    parser.add_argument("--thread-id", default=f"hw2-task2-live-{stamp}", help="LangGraph thread_id.")
    parser.add_argument("--max-api-calls", type=int, default=DEFAULT_MAX_API_CALLS,
                        help=f"cap on ModelCaller.call invocations (1..{HARD_MAX_API_CALLS}, "
                             f"default {DEFAULT_MAX_API_CALLS}).")
    args = parser.parse_args(argv)

    if not 1 <= args.max_api_calls <= HARD_MAX_API_CALLS:
        print(f"ERROR: --max-api-calls must be in 1..{HARD_MAX_API_CALLS}.", file=sys.stderr)
        return 2
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set; the live demonstration has no offline fallback.",
              file=sys.stderr)
        return 2
    workdir = Path(args.workdir).resolve()
    try:
        ensure_fresh_workdir(workdir)
    except OverwriteRefused as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("OPENAI_API_KEY: set (value not shown)", flush=True)
    llm = t2.hw1.ModelCaller(args.model)
    evidence = run_live(llm, model=args.model, task_text=args.prompt, workdir=workdir,
                        thread_id=args.thread_id, max_api_calls=args.max_api_calls,
                        out=lambda text: print(text, flush=True))
    return 0 if evidence["completion_status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
