#!/usr/bin/env python3
"""
HW2 Task 1 demonstration — hard decision: human approval of a coding action.

Runs the unchanged HW2 graph (hw2/task1_hitl.py) with the real LLM for every
role decision and the real HW1 coding tool, and records what happens:

  * the orchestrator LLM decides; if it requests approval for the executor, the
    executor LLM prepares the exact script and the graph pauses in interrupt();
  * the human sees the exact code and approves or refuses it (terminal input);
  * a refused action never reaches the coding tool; the orchestrator sees the
    refusal and decides again;
  * an approved action runs exactly once, and the demonstration stops right
    after that execution.

Nothing is decided by this runner: it only answers interrupts with the human's
input and stops at the bounds below. If the LLM takes another route, that is
recorded as is.

Bounding: graph.stream(..., interrupt_after=<every node>) pauses after each node,
so the run stops immediately after the first executed action (no further LLM
call), and the LLM-call cap (--max-llm-calls) is checked after every node; it
can be exceeded only by orchestrator retries inside one node (at most 3 calls). The graph's own eight-decision budget
and the 300 s tool timeout are unchanged.

Evidence (new directory only; existing runs are never overwritten):
    dialogue.json, dialogue.txt, demo_evidence.json,
    silverbox_agent_attempt_NN/ (written by the coding tool for an executed action)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from langgraph.types import Command

from hw2.demo_task1 import RecordingLLM, now
from hw2.task1_hitl import (
    REJECTED_MARK,
    PREPARED_MARK,
    HumanInTheLoopSystem,
    ModelCaller,
    OverwriteRefused,
    PythonCodingTool,
    ask_in_terminal,
    code_sha256,
    ensure_fresh_workdir,
    hw1,
    load_historical_results,
    pending_interrupt,
    save_dialogue,
    thread_config,
    write_new_file,
)

SCENARIO_REQUEST = (
    "Create and run a Python script to calculate the zero-predictor Test RMSE using the "
    "Silverbox 1024-sample test split. Do not train any model. Use the existing dataset "
    "and print the actually computed TEST_RMSE."
)
STOP_AFTER = ["orchestrator", "scientist", "general", "executor", "human"]
DEFAULT_MAX_LLM_CALLS = 12


class LoggedCodingTool:
    """Wraps the coding tool and logs every call; it adds no permission of its own."""

    def __init__(self, tool):
        self.tool = tool
        self.calls: list[dict] = []

    def write_and_run(self, filename: str, code: str) -> dict:
        entry = {"time_utc": now(), "filename": filename, "code_sha256": code_sha256(code)}
        self.calls.append(entry)
        result = self.tool.write_and_run(filename=filename, code=code)
        entry.update({k: result.get(k) for k in
                      ("relpath", "exit_code", "timed_out", "elapsed_s", "train_rmse", "test_rmse")})
        return result


def run_hard_demo(llm, task_text: str, workdir: Path, thread_id: str, model: str, coding_tool,
                  decide: Callable[[dict], Any], history: dict | None = None,
                  max_llm_calls: int = DEFAULT_MAX_LLM_CALLS) -> dict:
    """Run the bounded approval demonstration, write the evidence, and return it.

    decide(payload) answers each interrupt: a string for a clarification, and
    approval(action_id, approved, text) for an approval request.
    """
    ensure_fresh_workdir(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    recorder = RecordingLLM(llm)
    tool = LoggedCodingTool(coding_tool)
    system = HumanInTheLoopSystem(recorder, workdir=workdir, data_path=workdir / "unused.mat",
                                  coding_tool=tool, history=history)
    graph = system.build()
    config = thread_config(thread_id)

    events: list[dict] = []
    actions: dict[str, dict] = {}  # action_id -> what happened to it
    seen = 0

    def record_new_turns() -> None:
        nonlocal seen
        values = graph.get_state(config).values
        dialogue = values.get("dialogue", [])
        action = values.get("pending_action")
        for index, turn in enumerate(dialogue[seen:], start=seen):
            # tool_calls_so_far: coding-tool calls logged when this turn was recorded
            # (at the breakpoint right after it, or the interrupt that followed it).
            event = {"index": index, "role": turn["role"], "tool_calls_so_far": len(tool.calls)}
            text = turn["text"]
            if turn["role"] == "orchestrator":
                event.update(event="orchestrator_decision", decision=json.loads(text))
            elif turn["role"] == "executor" and text.startswith(PREPARED_MARK):
                event.update(event="action_prepared", action_id=action["action_id"])
                actions[action["action_id"]] = {
                    "action_type": action["action_type"],
                    "operation": action["operation"],
                    "filename": action["filename"],
                    "code_sha256": action["code_sha256"],
                    "code": action["code"],
                    "status_history": ["awaiting_approval"],
                }
            elif turn["role"] == "executor" and text.startswith(REJECTED_MARK):
                event.update(event="action_rejected_by_program", text=text)
            elif turn["role"] == "executor":
                event.update(event="action_executed", action_id=action["action_id"], text=text,
                             llm_calls_so_far=len(recorder.calls))
                actions[action["action_id"]]["status_history"].append("executed")
                actions[action["action_id"]]["outcome"] = action.get("outcome")
            elif turn["role"] == "human":
                event.update(event="human_reply", text=text)
                if action is not None and action["action_id"] in actions:
                    status_history = actions[action["action_id"]]["status_history"]
                    # An approved action may already have run before this pause.
                    verdict = {"approved": "approved", "executed": "approved",
                               "refused": "refused"}.get(action["status"])
                    if verdict and status_history[-1] == "awaiting_approval":
                        status_history.append(verdict)
                        event.update(event=f"action_{verdict}", action_id=action["action_id"],
                                     refusal_reason=action.get("refusal_reason"))
            elif turn["role"] == "user":
                event.update(event="user_request", text=text)
            else:
                event.update(event="specialist_message", text=text)
            events.append(event)
        seen = len(dialogue)
        save_dialogue(dialogue, workdir)

    evidence: dict = {
        "scenario": "HW2 Task 1 — hard decision: human approval of a coding action",
        "initial_request": task_text,
        "model": model,
        "thread_id": thread_id,
        "workdir": str(workdir),
        "started_utc": now(),
        "bounding": f"graph.stream(..., interrupt_after={STOP_AFTER}); stop after the first "
                    f"executed action; at most {max_llm_calls} LLM calls",
        "historical_context_supplied": history is not None,
    }

    graph_input: Any = {"dialogue": [{"role": "user", "text": task_text}]}
    outcome = None
    while outcome is None:
        for _ in graph.stream(graph_input, config=config, stream_mode="values", interrupt_after=STOP_AFTER):
            pass
        graph_input = None
        record_new_turns()
        action = graph.get_state(config).values.get("pending_action")
        payload = pending_interrupt(graph, config)

        if action is not None and action["status"] == "executed":
            outcome = "approved_action_executed"
        elif payload is not None:
            events.append({"event": "interrupt", "kind": payload["kind"], "question": payload["question"],
                           "action_id": payload.get("action", {}).get("action_id"),
                           "tool_calls_so_far": len(tool.calls)})
            graph_input = Command(resume=decide(payload))
        elif not graph.get_state(config).next:
            outcome = "graph_ended_without_execution"
        elif len(recorder.calls) >= max_llm_calls:
            outcome = "llm_call_cap_reached"

    # ---- proofs, computed from the recorded sequence
    refused = [aid for aid, a in actions.items() if "refused" in a["status_history"]]
    approved = [aid for aid, a in actions.items() if "approved" in a["status_history"]]
    executed = [aid for aid, a in actions.items() if "executed" in a["status_history"]]
    first_approval_index = next(
        (i for i, e in enumerate(events) if e["event"] == "action_approved"), len(events)
    )
    evidence["refusal_proof"] = {
        "refused_action_ids": refused,
        "tool_calls_before_first_approval": max(
            [e["tool_calls_so_far"] for e in events[:first_approval_index]], default=0
        ),
        "tool_calls_running_refused_code": sum(
            c["code_sha256"] == actions[aid]["code_sha256"] for aid in refused for c in tool.calls
        ),
        "orchestrator_saw_refusal": [
            any(f"action_id={aid} status=refused" in call["input_text"] for call in recorder.calls)
            for aid in refused
        ],
    }
    evidence["approval_proof"] = {
        "approved_action_ids": approved,
        "executed_action_ids": executed,
        "tool_calls_total": len(tool.calls),
        "executed_code_matches_approved_hash": bool(executed) and len(tool.calls) == 1
        and tool.calls[0]["code_sha256"] == actions[executed[0]]["code_sha256"],
        "llm_calls_after_execution": next(
            (len(recorder.calls) - e["llm_calls_so_far"] for e in events if e["event"] == "action_executed"),
            None,
        ),
    }
    final_dialogue = graph.get_state(config).values.get("dialogue", [])
    save_dialogue(final_dialogue, workdir)
    evidence.update(
        outcome=outcome,
        events=events,
        actions=actions,
        tool_calls=tool.calls,
        dialogue_final=final_dialogue,
        llm_calls=recorder.calls,
        api_call_count=len(recorder.calls),
        finished_utc=now(),
    )
    write_new_file(workdir / "demo_evidence.json", json.dumps(evidence, indent=2, ensure_ascii=False))
    return evidence


def main(argv: list[str] | None = None) -> int:
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    parser = argparse.ArgumentParser(
        prog="python -m hw2.demo_task1_hard",
        description="HW2 Task 1 demonstration: human approval/refusal of an LLM-prepared coding action.",
    )
    parser.add_argument("--prompt", default=SCENARIO_REQUEST, help="Initial user request.")
    parser.add_argument("--model", default=os.environ.get("AGENT_MODEL", "gpt-6-luna"),
                        help="OpenAI API model used for all roles.")
    parser.add_argument("--data", default=None, help="Path to SNLS80mV.mat (default: located automatically).")
    parser.add_argument("--thread-id", default=f"hw2-task1-hard-{stamp}", help="LangGraph thread_id.")
    parser.add_argument("--workdir", default=f"task6_runs/hw2_task1_hard_{stamp}",
                        help="NEW evidence directory (must not exist or be empty).")
    parser.add_argument("--with-history", action="store_true",
                        help="Show the orchestrator the recorded Task 5 / Task 6 results (read-only).")
    parser.add_argument("--max-llm-calls", type=int, default=DEFAULT_MAX_LLM_CALLS,
                        help=f"Stop after this many LLM calls (default {DEFAULT_MAX_LLM_CALLS}).")
    args = parser.parse_args(argv)

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        return 2
    data_path = hw1.locate_data(args.data)
    if data_path is None or not data_path.exists():
        print("ERROR: SNLS80mV.mat not found; pass --data.", file=sys.stderr)
        return 2
    workdir = Path(args.workdir).resolve()
    try:
        ensure_fresh_workdir(workdir)
    except OverwriteRefused as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Model: {args.model}\nThread: {args.thread_id}\nData: {data_path}\nWorkdir: {workdir}", flush=True)
    history = load_historical_results() if args.with_history else None
    tool = PythonCodingTool(workdir=workdir, data_path=data_path)
    evidence = run_hard_demo(ModelCaller(args.model), args.prompt, workdir, args.thread_id, args.model,
                             tool, ask_in_terminal, history, args.max_llm_calls)

    print("\n===== DEMONSTRATION RESULT =====")
    print(f"Outcome: {evidence['outcome']}")
    for event in evidence["events"]:
        if event["event"] in ("orchestrator_decision",):
            d = event["decision"]
            print(f"- decision: next_role={d['next_role']} request={d['request']!r}")
        elif event["event"] in ("action_prepared", "action_refused", "action_approved", "action_executed"):
            print(f"- {event['event']}: {event['action_id']} (tool calls so far: {event['tool_calls_so_far']})")
    print(f"Refusal proof: {json.dumps(evidence['refusal_proof'])}")
    print(f"Approval proof: {json.dumps(evidence['approval_proof'])}")
    for call in evidence["tool_calls"]:
        print(f"Executed {call['relpath']}: exit_code={call['exit_code']} TEST_RMSE={call['test_rmse']}")
    print(f"LLM API calls: {evidence['api_call_count']}")
    print(f"Evidence: {workdir / 'demo_evidence.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
