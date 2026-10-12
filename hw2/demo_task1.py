#!/usr/bin/env python3
"""
HW2 Task 1 demonstration — ambiguous Silverbox request.

Runs the unchanged HW2 graph (hw2/task1_hitl.py) with the real orchestrator LLM
for a bounded clarification round trip, and records the evidence:

  1. the user request goes to the orchestrator, whose LLM decides the route;
  2. if it chooses the human, the graph pauses in a real interrupt();
  3. the human answers (terminal input, or a reply file) and the SAME thread_id
     is resumed with Command(resume=reply);
  4. the next orchestrator decision is recorded, and the demonstration stops.

Bounding: every stream uses LangGraph's run-time breakpoint
interrupt_after=["orchestrator"], so the run stops right after each orchestrator
decision. The decision chosen after the resume is recorded but its destination
node is never executed: no specialist, no code generation, no coding tool. The
coding tool is additionally replaced by one that refuses to run. If the LLM
does not choose the human, that decision is recorded as is and the run stops.

Historical context: by default the orchestrator also sees the recorded Task 5 /
Task 6 results read from the committed evidence files (read-only, with
provenance); --no-history turns this off. The evidence records whether it was
supplied, which files were read and the exact text passed to the orchestrator.

Evidence (new directory only; existing runs are never overwritten):
    dialogue.json, dialogue.txt, demo_evidence.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from langgraph.types import Command

from hw2.task1_hitl import (
    HumanInTheLoopSystem,
    ModelCaller,
    OverwriteRefused,
    ensure_fresh_workdir,
    format_history,
    latest_orchestrator_decision,
    load_historical_results,
    pending_interrupt,
    save_dialogue,
    thread_config,
    write_new_file,
)

SCENARIO_REQUEST = "Report the RMSE from the Silverbox work."
STOP_AFTER = ["orchestrator"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RecordingLLM:
    """Passes every call to the wrapped model and records input and output."""

    def __init__(self, llm):
        self.llm = llm
        self.calls: list[dict] = []

    def call(self, instructions: str, input_text: str) -> str:
        output = self.llm.call(instructions, input_text)
        self.calls.append({
            "role_prompt": instructions.splitlines()[0],
            "time_utc": now(),
            "input_text": input_text,
            "output_text": output,
        })
        return output


class DisabledCodingTool:
    """The demonstration never runs code; any attempt fails loudly."""

    def __init__(self):
        self.calls = 0

    def write_and_run(self, filename: str, code: str) -> dict:
        self.calls += 1
        raise RuntimeError("the coding tool is disabled in the ambiguity demonstration")


def run_demo(llm, task_text: str, workdir: Path, thread_id: str, model: str,
             get_reply: Callable[[dict], str], history: dict | None = None) -> dict:
    """Run the bounded demonstration, write the evidence, and return it.

    history: result of load_historical_results(), or None to give the
    orchestrator no recorded results.
    """
    ensure_fresh_workdir(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    recorder = RecordingLLM(llm)
    tool = DisabledCodingTool()
    system = HumanInTheLoopSystem(recorder, workdir=workdir, data_path=workdir / "unused.mat",
                                  coding_tool=tool, history=history)
    graph = system.build()
    config = thread_config(thread_id)

    def stream(graph_input) -> None:
        for _ in graph.stream(graph_input, config=config, stream_mode="values",
                              interrupt_after=STOP_AFTER):
            pass

    def dialogue() -> list[dict]:
        return graph.get_state(config).values["dialogue"]

    evidence: dict = {
        "scenario": "HW2 Task 1 — ambiguous Silverbox request, clarification by the human",
        "initial_request": task_text,
        "model": model,
        "thread_id": thread_id,
        "workdir": str(workdir),
        "started_utc": now(),
        "bounding": "graph.stream(..., interrupt_after=['orchestrator']); coding tool disabled",
        "historical_context": {
            "supplied": history is not None,
            "sources_read": history["sources_read"] if history else [],
            "problems": history["problems"] if history else [],
            "records": history["records"] if history else [],
            "context_text_passed_to_orchestrator": system.history_context,
        },
    }

    # Decision 1: the orchestrator LLM decides; the run stops right after it.
    stream({"dialogue": [{"role": "user", "text": task_text}]})
    first = latest_orchestrator_decision(dialogue())
    evidence["first_decision"] = first
    evidence["first_decision_raw_llm_output"] = recorder.calls[-1]["output_text"]

    if first["next_role"] == "human" and first.get("request") == "clarification":
        stream(None)  # into the human node, which pauses in interrupt()
        snapshot = graph.get_state(config)
        payload = pending_interrupt(graph, config)
        evidence["interrupt"] = {
            "payload": payload,
            "graph_next_while_paused": list(snapshot.next),
            "dialogue_while_paused": dialogue(),
        }
        save_dialogue(dialogue(), workdir)

        reply = get_reply(payload)
        evidence["human_reply"] = reply
        calls_before_resume = len(recorder.calls)

        # Resume the SAME thread; stop right after the next orchestrator decision.
        stream(Command(resume=reply))
        final_dialogue = dialogue()
        resumed_calls = recorder.calls[calls_before_resume:]
        evidence["post_resume_message"] = final_dialogue[-1]
        evidence["post_resume_decision"] = latest_orchestrator_decision(final_dialogue)
        evidence["post_resume_raw_llm_output"] = resumed_calls[-1]["output_text"] if resumed_calls else None
        evidence["human_turns_with_reply"] = sum(
            t["role"] == "human" and t["text"] == reply for t in final_dialogue
        )
        evidence["clarification_visible_to_resumed_orchestrator"] = bool(resumed_calls) and all(
            f"[HUMAN]\n{reply}" in c["input_text"] for c in resumed_calls
        )
        evidence["stopped_before_node_not_executed"] = list(graph.get_state(config).next)
        evidence["outcome"] = "human_clarification_resumed"
    else:
        evidence["outcome"] = "llm_did_not_choose_human_clarification"
        evidence["stopped_before_node_not_executed"] = list(graph.get_state(config).next)

    final_dialogue = dialogue()
    save_dialogue(final_dialogue, workdir)
    evidence["dialogue_final"] = final_dialogue
    evidence["llm_calls"] = recorder.calls
    evidence["api_call_count"] = len(recorder.calls)
    evidence["coding_tool_calls"] = tool.calls
    evidence["finished_utc"] = now()
    write_new_file(workdir / "demo_evidence.json", json.dumps(evidence, indent=2, ensure_ascii=False))
    return evidence


def terminal_reply(payload: dict) -> str:
    print(f"\n===== QUESTION FOR HUMAN ({payload['kind']}) =====\n{payload['question']}", flush=True)
    return input("Your reply> ")


def file_reply(workdir: Path, timeout_s: float = 1800.0) -> Callable[[dict], str]:
    """Publish the question to pending_question.json and wait for human_reply.txt."""
    def get(payload: dict) -> str:
        write_new_file(workdir / "pending_question.json", json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"\n===== QUESTION FOR HUMAN ({payload['kind']}) =====\n{payload['question']}", flush=True)
        reply_path = workdir / "human_reply.txt"
        print(f"Waiting for the reply in {reply_path} ...", flush=True)
        deadline = time.monotonic() + timeout_s
        while not reply_path.exists():
            if time.monotonic() > deadline:
                raise TimeoutError(f"no reply in {reply_path} within {timeout_s} s")
            time.sleep(1)
        time.sleep(0.5)  # let the writer finish
        return reply_path.read_text(encoding="utf-8").strip()
    return get


def main(argv: list[str] | None = None) -> int:
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    parser = argparse.ArgumentParser(
        prog="python -m hw2.demo_task1",
        description="HW2 Task 1 demonstration: ambiguous Silverbox request with a real orchestrator LLM.",
    )
    parser.add_argument("--prompt", default=SCENARIO_REQUEST, help="Initial user request.")
    parser.add_argument("--model", default=os.environ.get("AGENT_MODEL", "gpt-6-luna"),
                        help="OpenAI API model used for the orchestrator.")
    parser.add_argument("--thread-id", default=f"hw2-task1-ambiguity-{stamp}", help="LangGraph thread_id.")
    parser.add_argument("--workdir", default=f"task6_runs/hw2_task1_ambiguity_{stamp}",
                        help="NEW evidence directory (must not exist or be empty).")
    parser.add_argument("--reply-file", action="store_true",
                        help="Wait for <workdir>/human_reply.txt instead of reading the terminal.")
    parser.add_argument("--no-history", action="store_true",
                        help="Do not show the orchestrator the recorded Task 5 / Task 6 results.")
    args = parser.parse_args(argv)

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        return 2
    workdir = Path(args.workdir).resolve()
    try:
        ensure_fresh_workdir(workdir)
    except OverwriteRefused as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    history = None if args.no_history else load_historical_results()
    print(f"Model: {args.model}\nThread: {args.thread_id}\nWorkdir: {workdir}", flush=True)
    if history is not None:
        print(f"\nHistorical context supplied to the orchestrator:\n{format_history(history)}", flush=True)
    get_reply = file_reply(workdir) if args.reply_file else terminal_reply
    evidence = run_demo(ModelCaller(args.model), args.prompt, workdir, args.thread_id, args.model,
                        get_reply, history)

    print("\n===== DEMONSTRATION RESULT =====")
    print(f"Outcome: {evidence['outcome']}")
    print(f"First decision: {json.dumps(evidence['first_decision'], ensure_ascii=False)}")
    if "post_resume_decision" in evidence:
        print(f"Human reply: {evidence['human_reply']}")
        print(f"Post-resume decision: {json.dumps(evidence['post_resume_decision'], ensure_ascii=False)}")
        print(f"Clarification visible to resumed orchestrator: "
              f"{evidence['clarification_visible_to_resumed_orchestrator']}")
    print(f"Stopped before (not executed): {evidence['stopped_before_node_not_executed']}")
    print(f"LLM API calls: {evidence['api_call_count']}  coding tool calls: {evidence['coding_tool_calls']}")
    print(f"Evidence: {workdir / 'demo_evidence.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
