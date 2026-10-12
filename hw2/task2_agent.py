#!/usr/bin/env python3
"""
HW2 Task 2 — the HW1 four-role LangGraph system driving the simulated focusing tool.

The four HW1 roles (orchestrator, scientist, general, executor), the shared
dialogue, the LLM-based routing, the ModelCaller interface (any object with
call(instructions, input_text) -> str) and the eight-decision limit are reused
from task6/agentic_system.py, which is not modified. Only the role prompts and
the executor's tool change: instead of the HW1 Python coding tool, the executor
owns ONE bounded tool, hw2.focusing_tool.FocusingSession.call(arguments).

Graph (same topology as HW1):
    START -> orchestrator
    orchestrator -> scientist | general | executor | END ("finish")
    scientist / general / executor -> orchestrator, or END after decision 8

Simulation-only mode. The system has no coding tool, no shell, no file tool and
no hardware driver; the only action is a simulated focusing experiment on one
FocusingSession (one fixed transmission matrix T for the whole agent run). No
human approval is needed for that bounded simulated tool; HW2 Task 1 keeps its
own approval behaviour unchanged in hw2/task1_hitl.py.

Experiment selection. The executor's LLM receives the user request, the
orchestrator's instruction, the latest scientist plan, session.describe(), the
agent-facing argument schema, the run-level limits and all earlier experiment
results, and must reply with ONE JSON object
    {"arguments": {<all 13 settings>}, "rationale": "<short reason>"}
The settings are validated strictly (exact key set, integers, minimums, run
budget); nothing is filled in, clipped or replaced. A rejected proposal or a
tool-side rejection is returned to the same LLM with the error, for at most
MAX_ARGUMENT_ATTEMPTS proposals per executor turn. Session size, physics, noise
and hard limits are system constraints set when the session is built. The
artifact directory name (run_name) is assigned by the program, not the LLM.

Bookkeeping. Every FocusingSession.call is one record in state["experiments"]
(status completed | rejected_by_tool | tool_error) holding the exact arguments
and the full tool output. The executor appends a dialogue turn with the
settings and the VALIDATED results (never the noisy search maxima). A finish
is accepted only after at least one completed experiment; the final report is
then built by the program from the tool output, so it cannot contain an
invented number. If the decision budget runs out first, run_agent returns an
honest partial result with no final report.

Duplicate protection. The graph is checkpointed. Executor LLM proposals are
cached per (executor turn, attempt) and tool results per system-assigned
run_name, so re-entering the executor node after a crash or retry replays the
recorded call instead of running a second experiment; the tool itself also
refuses an existing run directory.

Offline demonstration (mocked LLM, real tool on a small simulated bench):
    python -m hw2.task2_agent --offline-demo
"""

from __future__ import annotations

import argparse
import json
import operator
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Callable, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

# Reuse the HW1 implementation without modifying it (task6/ is not a package).
TASK6_DIR = Path(__file__).resolve().parent.parent / "task6"
if str(TASK6_DIR) not in sys.path:
    sys.path.insert(0, str(TASK6_DIR))

import agentic_system as hw1  # noqa: E402
from agentic_system import (  # noqa: E402
    MAX_DECISIONS,
    AgenticSilverboxSystem,
    Turn,
    count_role,
    format_dialogue,
    latest_orchestrator_decision,
    latest_role_text,
    parse_json_object,
    save_dialogue,
)

from hw2.focusing_tool import ExperimentArguments, FocusingSession, FocusingToolError  # noqa: E402
from hw2.task1_hitl import OverwriteRefused, ensure_fresh_workdir, write_new_file  # noqa: E402

REPO_ROOT = TASK6_DIR.parent

# Orchestration-level limits (the tool's own validation and per-call budget still apply).
MAX_EXPERIMENTS = 4                 # completed experiments per agent run
DEFAULT_RUN_ACQUISITION_LIMIT = 10_000   # simulated camera frames per agent run
MAX_ARGUMENT_ATTEMPTS = 3           # executor LLM proposals per executor turn
MAX_ORCHESTRATOR_ATTEMPTS = 3       # as in HW1
# Smallest legal experiment: averaging 1 x (1 + iterations 1 + 2 x validation_count 1).
MIN_EXPERIMENT_ACQUISITIONS = 4
REQUIRED_SETTINGS = tuple(ExperimentArguments.json_schema()["required"])

COMPLETED = "completed"
REJECTED_BY_TOOL = "rejected_by_tool"
TOOL_ERROR = "tool_error"

COMPLETED_MARK = "EXPERIMENT_COMPLETED"
REJECTED_MARK = "EXPERIMENT_REJECTED"
NOT_RUN_MARK = "NO_EXPERIMENT_RUN"


class FocusingState(TypedDict, total=False):
    # Same shared dialogue as HW1: each node appends its new turn.
    dialogue: Annotated[list[Turn], operator.add]
    # One record per FocusingSession.call, in call order (machine-readable).
    experiments: Annotated[list[dict], operator.add]
    # Set only by an accepted finish; built by the program from tool output.
    final_report: dict | None


def agent_argument_schema() -> dict:
    """The tool schema the LLM must fill: every setting required, run_name not offered."""
    schema = json.loads(json.dumps(ExperimentArguments.json_schema()))
    schema["properties"].pop("run_name", None)
    return schema


def completed(experiments: list[dict]) -> list[dict]:
    return [e for e in experiments if e["status"] == COMPLETED]


def run_acquisitions(experiments: list[dict]) -> int:
    return sum(e["acquisitions"] for e in experiments)


def _fmt(value) -> str:
    return "not reported (see notes)" if value is None else f"{value:.6g}"


def validated_summary(result: dict) -> dict:
    """The validated figures of one successful tool output (no search values)."""
    v = result["validated"]
    d = v["decomposition"] or {}
    return {
        "run_name": result["run_name"],
        "arguments": result["arguments"],
        "validated_initial_enhancement_mean": v["initial"]["enhancement_mean"],
        "validated_initial_enhancement_std": v["initial"]["enhancement_std"],
        "validated_final_enhancement_mean": v["final"]["enhancement_mean"],
        "validated_final_enhancement_std": v["final"]["enhancement_std"],
        "validation_count": len(v["final"]["enhancement_values"]),
        "enhancement_gain": v["enhancement_gain"],
        "focus_intensity_gain": v["focus_intensity_gain"],
        "background_suppression_factor": v["background_suppression_factor"],
        "share_of_log_gain_from_focus_brightening": d.get("share_of_log_gain_from_focus_brightening"),
        "share_of_log_gain_from_background_suppression": d.get("share_of_log_gain_from_background_suppression"),
        "notes": list(v["notes"]),
        "camera_acquisitions": result["acquisitions"]["counted_by_bench"],
        "transmission_fingerprint": result["transmission_fingerprint"],
        "artifacts": result["artifacts"],
    }


def experiments_table(experiments: list[dict]) -> str:
    if not experiments:
        return "(no focusing tool calls yet)"
    rows = []
    for e in experiments:
        head = f"call {e['invocation']:02d} run_id={e['run_name']} status={e['status']}"
        if e["status"] == COMPLETED:
            s = e["summary"]
            rows.append(
                f"{head} validated_initial_enhancement={_fmt(s['validated_initial_enhancement_mean'])} "
                f"validated_final_enhancement={_fmt(s['validated_final_enhancement_mean'])} "
                f"(std {_fmt(s['validated_final_enhancement_std'])}) "
                f"enhancement_gain={_fmt(s['enhancement_gain'])} "
                f"focus_intensity_gain={_fmt(s['focus_intensity_gain'])} "
                f"background_suppression_factor={_fmt(s['background_suppression_factor'])} "
                f"acquisitions={e['acquisitions']} arguments={json.dumps(e['arguments'], sort_keys=True)}"
            )
        else:
            rows.append(f"{head} acquisitions={e['acquisitions']} error={e['error']} "
                        f"arguments={json.dumps(e['arguments'], sort_keys=True)}")
    return "\n".join(rows)


LIMITATIONS = (
    "Simulated bench only: one fixed random complex Gaussian transmission matrix, a binary "
    "amplitude DMD and a shot/read-noise camera model; no real optics or hardware were used.",
    "Enhancement is <I>_F / <I>_B against the proximal background strips chosen in the "
    "experiment settings, not against the whole camera frame.",
    "Enhancement gain is a contrast improvement; it is not an increase of the focus intensity. "
    "The absolute brightening of the focus is the separate focus_intensity_gain, and the rest of "
    "the contrast improvement comes from suppressing the local background.",
    "The tool does not test whether the focus is the brightest point of the camera frame; no "
    "such claim is made.",
    "Values are means of the tool's independent validation measurements (std reported); the "
    "noisy search maxima are biased upward and are not reported as results.",
)


def build_final_report(experiments: list[dict], reason: str) -> dict | None:
    """The final report, entirely from completed tool output, or None if there is none."""
    done = completed(experiments)
    if not done:
        return None
    best = max(done, key=lambda e: e["summary"]["validated_final_enhancement_mean"])
    s = best["summary"]
    limitations = list(LIMITATIONS)
    if len(done) > 1:
        limitations.append(
            f"The reported experiment was selected as the highest validated final enhancement among "
            f"{len(done)} completed experiments, which slightly favours a lucky validation set."
        )
    limitations += [f"Tool note: {n}" for n in s["notes"]]
    report = {
        "status": "completed",
        "source": "FocusingSession.call output (validated measurements); assembled by the program",
        "selection_rule": "completed experiment with the highest validated final enhancement mean",
        "selected_run_id": s["run_name"],
        "selected": s,
        "completed_experiments": [e["summary"] for e in done],
        "tool_calls": len(experiments),
        "total_camera_acquisitions": run_acquisitions(experiments),
        "limitations": limitations,
        "orchestrator_reason": reason,
    }
    report["message"] = format_final_message(report)
    return report


def format_final_message(report: dict) -> str:
    s = report["selected"]
    share_f = s["share_of_log_gain_from_focus_brightening"]
    share_b = s["share_of_log_gain_from_background_suppression"]
    shares = ("decomposition not reported (see notes)" if share_f is None else
              f"share of log(enhancement gain) from focus brightening {share_f:.1%}, "
              f"from background suppression {share_b:.1%}")
    lines = [
        "FINAL REPORT (values from the focusing tool's independent validation measurements; "
        "simulated bench only)",
        f"Selected experiment: run_id={s['run_name']} ({report['selection_rule']}; "
        f"{len(report['completed_experiments'])} completed of {report['tool_calls']} tool calls).",
        f"Settings: {json.dumps(s['arguments'], sort_keys=True)}",
        f"Validated enhancement: initial {_fmt(s['validated_initial_enhancement_mean'])} "
        f"(std {_fmt(s['validated_initial_enhancement_std'])}) -> final "
        f"{_fmt(s['validated_final_enhancement_mean'])} (std {_fmt(s['validated_final_enhancement_std'])}), "
        f"{s['validation_count']} validation measurements each.",
        f"Enhancement gain (local contrast): {_fmt(s['enhancement_gain'])}x.",
        f"Focus intensity gain (absolute brightening of the focus): {_fmt(s['focus_intensity_gain'])}x.",
        f"Background suppression factor: {_fmt(s['background_suppression_factor'])}x; {shares}.",
        f"Camera acquisitions: {s['camera_acquisitions']} for this experiment, "
        f"{report['total_camera_acquisitions']} in the whole run.",
        f"Artifacts: {s['artifacts']['result']} (and {len(s['artifacts']) - 1} more in {Path(s['artifacts']['result']).parent.as_posix()})",
        "Limitations:",
    ]
    lines += [f"- {item}" for item in report["limitations"]]
    return "\n".join(lines)


def build_partial_report(experiments: list[dict], why: str) -> dict:
    """What exists when the run ends without an accepted finish. Nothing is promoted to final."""
    return {
        "status": "partial",
        "reason": why,
        "note": "no accepted finish: the experiments below were measured but not finalized",
        "completed_experiments": [e["summary"] for e in completed(experiments)],
        "tool_calls": len(experiments),
        "total_camera_acquisitions": run_acquisitions(experiments),
    }


class _ReplayMismatch(RuntimeError):
    """A replayed executor step asked for a different tool call than the recorded one."""


class FocusingAgentSystem(AgenticSilverboxSystem):
    """The HW1 four-role system with the focusing tool as the executor's only tool.

    llm: any object with call(instructions, input_text) -> str (the runtime will
         pass a ModelCaller; offline tests pass a scripted model).
    session: the FocusingSession of this agent run (one fixed medium).
    run_id: prefix of the artifact directories; each tool call gets
         <run_id>_call<NN> under the session's artifact root.
    """

    def __init__(self, llm, session: FocusingSession, run_id: str | None = None,
                 max_experiments: int = MAX_EXPERIMENTS,
                 run_acquisition_limit: int = DEFAULT_RUN_ACQUISITION_LIMIT,
                 max_argument_attempts: int = MAX_ARGUMENT_ATTEMPTS):
        # AgenticSilverboxSystem.__init__ constructs ModelCaller (the OpenAI
        # client) and the HW1 coding tool, so it is not called. There is
        # deliberately no coding tool in this system.
        self.llm = llm
        self.coding_tool = None
        self.runs: list[dict] = []
        self._session = session
        self.run_id = run_id or f"agent_{datetime.now():%Y%m%d_%H%M%S}"
        self.max_experiments = max_experiments
        self.run_acquisition_limit = run_acquisition_limit
        self.max_argument_attempts = max_argument_attempts
        self._proposal_cache: dict[tuple[int, int], str] = {}
        self._tool_ledger: dict[str, dict] = {}
        self.replayed_tool_calls: list[str] = []

    @property
    def session_description(self) -> dict:
        """session.describe() with the agent-facing schema (run_name is program-assigned)."""
        description = self._session.describe()
        description["argument_schema"] = agent_argument_schema()
        return description

    def _limits(self, experiments: list[dict]) -> dict:
        used = run_acquisitions(experiments)
        remaining = self.run_acquisition_limit - used
        left = self.max_experiments - len(completed(experiments))
        return {
            "max_completed_experiments_per_run": self.max_experiments,
            "completed_experiments_left": left,
            "run_acquisition_limit": self.run_acquisition_limit,
            "run_acquisitions_used": used,
            "run_acquisitions_remaining": remaining,
            "per_call_acquisition_limit": self._session.acquisition_limit,
            "max_argument_proposals_per_executor_turn": self.max_argument_attempts,
            "experiment_allowed": left > 0 and remaining >= MIN_EXPERIMENT_ACQUISITIONS,
        }

    # -----------------------------
    # Orchestrator
    # -----------------------------

    def orchestrator(self, state: FocusingState) -> dict:
        dialogue = state["dialogue"]
        experiments = state.get("experiments", [])
        decision_no = count_role(dialogue, "orchestrator") + 1
        limits = self._limits(experiments)
        finish_allowed = bool(completed(experiments))

        instructions = """You are the ORCHESTRATOR in a four-role LangGraph system that runs
simulated wavefront-shaping experiments: focusing light through a simulated scattering
medium with a binary DMD and a noisy camera.

You see the complete dialogue and must choose exactly one next role.

Return ONLY a JSON object with exactly these three fields:
{
  "next_role": "scientist" | "general" | "executor" | "finish",
  "instruction": "specific instruction for that role",
  "reason": "one-sentence reason"
}

Rules:
- Your decision MUST be made from the dialogue; do not use a fixed role sequence.
- You run no experiments yourself.
- scientist owns the physics and the experiment strategy (grouping factor, focus and
  background regions, iterations, flips, averaging, validation, budget) and may propose
  settings; it has no tool.
- general answers side questions about terms, metrics or the record; it has no tool.
- executor is the ONLY role with a tool: one simulated focusing experiment per executor
  turn on the fixed simulated medium. The executor's own model chooses every setting.
- Experiments are allowed only while the program says experiment_allowed is true.
- After each experiment, read its validated results and decide whether another experiment
  is worthwhile (for example different settings) or whether to finish.
- You may choose finish ONLY when at least one experiment has status completed. The
  program then reports the validated results of the completed experiments; you add no
  numbers yourself.
- Keep enhancement gain (local contrast) and focus intensity gain (absolute brightening)
  separate. Never invent a measured value.
"""
        input_text = (
            f"This is orchestrator decision {decision_no} of at most {MAX_DECISIONS}.\n"
            f"finish currently allowed by the program: {finish_allowed}\n"
            f"RUN LIMITS: {json.dumps(limits)}\n"
            f"FOCUSING TOOL CALLS SO FAR:\n{experiments_table(experiments)}\n\n"
            f"DIALOGUE:\n{format_dialogue(dialogue)}"
        )

        # As in HW1: the decision is always an LLM call; a malformed or illegal
        # reply is sent back to the LLM with the problem stated.
        feedback = ""
        decision = None
        for _ in range(MAX_ORCHESTRATOR_ATTEMPTS):
            raw = self.llm.call(instructions, input_text + feedback)
            try:
                candidate = parse_json_object(raw)
                role = str(candidate.get("next_role", "")).strip().lower()
                if role not in hw1.ALLOWED_NEXT:
                    raise ValueError(f"next_role {role!r} is not one of {sorted(hw1.ALLOWED_NEXT)}")
                if role == "finish" and not finish_allowed:
                    raise ValueError("finish is forbidden: no focusing experiment has completed yet")
                if role == "executor" and not limits["experiment_allowed"]:
                    raise ValueError(
                        "executor is forbidden: the run's experiment or camera-acquisition limit "
                        "is exhausted; choose finish (if an experiment completed), scientist or general"
                    )
                decision = candidate
                break
            except Exception as exc:
                feedback = (
                    f"\n\nYOUR PREVIOUS REPLY WAS REJECTED ({exc}). Reply again with ONLY the "
                    f"JSON object with next_role, instruction, reason.\nPrevious reply:\n{raw[:2000]}"
                )
        if decision is None:
            raise RuntimeError(
                f"Orchestrator LLM did not return a valid decision in {MAX_ORCHESTRATOR_ATTEMPTS} attempts."
            )

        normalized = {
            "next_role": str(decision["next_role"]).strip().lower(),
            "instruction": str(decision.get("instruction") or "").strip(),
            "reason": str(decision.get("reason") or "").strip(),
        }
        update: dict = {
            "dialogue": [{"role": "orchestrator", "text": json.dumps(normalized, ensure_ascii=False)}]
        }
        if normalized["next_role"] == "finish":
            update["final_report"] = build_final_report(experiments, normalized["reason"])
        return update

    # -----------------------------
    # Scientist and general (no tools)
    # -----------------------------

    def scientist(self, state: FocusingState) -> dict:
        dialogue = state["dialogue"]
        decision = latest_orchestrator_decision(dialogue)
        instructions = """You are the SCIENTIST in a four-role system running simulated
wavefront-shaping (focusing through a scattering medium) experiments.

You own the physics and the experiment strategy. You have NO tool.

The simulated bench: a binary amplitude DMD -> fixed complex transmission matrix -> noisy
camera. The executor's tool runs iterative binary flipping on DMD macropixels to raise the
enhancement <I>_F / <I>_B of a focus rectangle against proximal background strips, and
validates the initial and final patterns with independent measurements.

When asked for a plan, explain the trade-offs (number of macropixels vs. grouping factor,
iterations and flips per proposal, averaging vs. noise, validation count, acquisition
budget, focus/background geometry) and propose concrete settings. Your settings are
suggestions; the executor chooses and is checked by the tool.

Keep these quantities separate: enhancement gain (local contrast) is not focus intensity
gain (absolute brightening); much of a contrast gain can come from background suppression.
Do not claim the focus is the brightest point of the frame. Never claim a measurement that
no EXECUTOR turn reported.
"""
        input_text = (
            f"ORCHESTRATOR INSTRUCTION:\n{decision.get('instruction', '')}\n\n"
            f"SESSION (system constraints):\n{json.dumps(self.session_description, indent=1)}\n\n"
            f"RUN LIMITS: {json.dumps(self._limits(state.get('experiments', [])))}\n\n"
            f"FOCUSING TOOL CALLS SO FAR:\n{experiments_table(state.get('experiments', []))}\n\n"
            f"DIALOGUE:\n{format_dialogue(dialogue)}"
        )
        return {"dialogue": [{"role": "scientist", "text": self.llm.call(instructions, input_text)}]}

    def general(self, state: FocusingState) -> dict:
        dialogue = state["dialogue"]
        decision = latest_orchestrator_decision(dialogue)
        instructions = """You are the GENERAL AGENT in a four-role system running simulated
wavefront-shaping experiments.

You answer only side questions: what a term or metric means (enhancement, enhancement
gain, focus intensity gain, background suppression, macropixel, acquisition), or what the
record shows. You have NO tool and you do NOT plan experiments. Questions about the
experiment strategy belong to the scientist; running experiments belongs to the executor.
Never invent measured results; quote only values an EXECUTOR turn reported.
"""
        input_text = (
            f"ORCHESTRATOR INSTRUCTION:\n{decision.get('instruction', '')}\n\n"
            f"DIALOGUE:\n{format_dialogue(dialogue)}"
        )
        return {"dialogue": [{"role": "general", "text": self.llm.call(instructions, input_text)}]}

    # -----------------------------
    # Executor: the only caller of the focusing tool
    # -----------------------------

    EXECUTOR_INSTRUCTIONS = """You are the EXECUTOR in a four-role LangGraph system running
simulated wavefront-shaping experiments.

You are the only role with a tool: ONE simulated focusing experiment per turn on a fixed
simulated scattering medium, FocusingSession.call(arguments). You choose every setting of
the experiment; nothing is filled in for you and invalid settings are not corrected for you.

Return ONLY one JSON object:
{
  "arguments": {
    "grouping_factor": int, "focus_row": int, "focus_col": int, "focus_height": int,
    "focus_width": int, "background_gap": int, "background_width": int, "iterations": int,
    "flips_per_proposal": int, "averaging": int, "validation_count": int, "seed": int,
    "max_acquisitions": int
  },
  "rationale": "one or two sentences on why these settings"
}

Constraints (checked before anything is measured):
- "arguments" contains exactly these 13 integer settings; no other keys (the artifact
  directory name is assigned by the program).
- grouping_factor must be one of the session's valid_grouping_factors.
- the focus rectangle and the background strips (background_gap, background_width) must fit
  inside the camera; flips_per_proposal <= number of macropixels.
- averaging * (1 + iterations + 2 * validation_count) <= max_acquisitions.
- max_acquisitions <= the per-call limit and <= the run's remaining acquisitions.
- Use the orchestrator instruction, the scientist plan (if any) and the earlier results.
  If an earlier proposal was rejected, correct exactly the stated problem.
"""

    def executor(self, state: FocusingState) -> dict:
        dialogue = state["dialogue"]
        experiments = list(state.get("experiments", []))
        turn_no = count_role(dialogue, "executor") + 1
        decision_no = count_role(dialogue, "orchestrator")
        limits = self._limits(experiments)

        if not limits["experiment_allowed"]:
            # Fail closed: the orchestrator's validation should already prevent this.
            text = (f"{NOT_RUN_MARK} (tool not called)\nreason=run limits exhausted\n"
                    f"run_limits={json.dumps(limits)}")
            return {"dialogue": [{"role": "executor", "text": text}]}

        decision = latest_orchestrator_decision(dialogue)
        user_task = dialogue[0]["text"] if dialogue and dialogue[0]["role"] == "user" else ""
        base_input = (
            f"USER REQUEST:\n{user_task}\n\n"
            f"ORCHESTRATOR INSTRUCTION:\n{decision.get('instruction', '')}\n\n"
            f"LATEST SCIENTIST PLAN:\n{latest_role_text(dialogue, 'scientist') or '(none)'}\n\n"
            f"SESSION (system constraints, session.describe()):\n"
            f"{json.dumps(self.session_description, indent=1)}\n\n"
            f"VALID TOOL ARGUMENT SCHEMA (all required):\n{json.dumps(agent_argument_schema())}\n\n"
            f"RUN LIMITS: {json.dumps(limits)}\n\n"
            f"EARLIER EXPERIMENT RESULTS:\n{experiments_table(experiments)}"
        )

        new_records: list[dict] = []
        problems: list[str] = []
        feedback = ""
        for attempt in range(1, self.max_argument_attempts + 1):
            raw = self._propose(turn_no, attempt, base_input + feedback)
            arguments, rationale, errors = self.check_proposal(raw, limits)
            if errors:
                problems.append(f"proposal {attempt}: rejected before the tool call: {'; '.join(errors)}")
                feedback = self._feedback(errors, raw)
                continue

            invocation = len(experiments) + len(new_records) + 1
            record = self._invoke_tool(invocation, turn_no, decision_no, arguments, rationale)
            new_records.append(record)
            if record["status"] == COMPLETED:
                break
            problems.append(f"proposal {attempt}: tool call {record['run_name']} {record['status']}: "
                            f"{record['error']}")
            if record["status"] == TOOL_ERROR:
                break  # not an argument problem: do not retry
            feedback = self._feedback([record["error"]], raw)

        text = self.format_executor_turn(new_records, problems, experiments + new_records)
        return {"dialogue": [{"role": "executor", "text": text}], "experiments": new_records}

    def _propose(self, turn_no: int, attempt: int, input_text: str) -> str:
        """The executor LLM's proposal; cached so a re-run of the node reuses it."""
        key = (turn_no, attempt)
        if key not in self._proposal_cache:
            self._proposal_cache[key] = self.llm.call(self.EXECUTOR_INSTRUCTIONS, input_text)
        return self._proposal_cache[key]

    @staticmethod
    def _feedback(errors: list[str], raw: str) -> str:
        return (
            f"\n\nYOUR PREVIOUS PROPOSAL WAS REJECTED; NOTHING WAS MEASURED FOR IT.\n"
            f"Problems: {'; '.join(errors)}\n"
            f"Reply again with ONLY the JSON object with all 13 settings.\nPrevious reply:\n{raw[:2000]}"
        )

    def check_proposal(self, raw: str, limits: dict) -> tuple[dict | None, str, list[str]]:
        """(arguments, rationale, errors). Strict: nothing is filled in, cast or clipped."""
        try:
            obj = parse_json_object(raw)
        except Exception as exc:
            return None, "", [f"reply is not a JSON object ({type(exc).__name__}: {exc})"]
        if not isinstance(obj, dict):
            return None, "", ["reply must be a JSON object"]
        unknown_top = sorted(set(obj) - {"arguments", "rationale"})
        if unknown_top:
            return None, "", [f"unknown top-level field(s) {unknown_top}; use 'arguments' and 'rationale'"]
        arguments = obj.get("arguments")
        rationale = str(obj.get("rationale") or "")
        if not isinstance(arguments, dict):
            return None, rationale, ["'arguments' must be a JSON object with the 13 settings"]
        if "run_name" in arguments:
            return None, rationale, ["run_name is assigned by the program; remove it"]
        try:
            ExperimentArguments.from_dict(arguments)   # the tool's own type/minimum checks
        except FocusingToolError as exc:
            return None, rationale, [str(exc)]
        errors = []
        if arguments["max_acquisitions"] > limits["run_acquisitions_remaining"]:
            errors.append(f"max_acquisitions {arguments['max_acquisitions']} exceeds the run's remaining "
                          f"{limits['run_acquisitions_remaining']} camera acquisitions")
        return dict(arguments), rationale, errors

    def _invoke_tool(self, invocation: int, turn_no: int, decision_no: int,
                     arguments: dict, rationale: str) -> dict:
        """Exactly one FocusingSession.call per new run_name; replays never call again."""
        run_name = f"{self.run_id}_call{invocation:02d}"
        if run_name in self._tool_ledger:
            recorded = self._tool_ledger[run_name]
            if recorded["arguments"] != arguments:
                raise _ReplayMismatch(f"{run_name} was recorded with different arguments")
            self.replayed_tool_calls.append(run_name)
            return recorded

        before = self._session.describe()["camera_acquisitions_so_far"]
        try:
            output = self._session.call({**arguments, "run_name": run_name})
        except Exception as exc:  # the tool raised instead of returning ok:false
            output = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "crashed": True}
        frames = self._session.describe()["camera_acquisitions_so_far"] - before

        ok = output.get("ok") is True
        status = COMPLETED if ok else (TOOL_ERROR if output.get("crashed") else REJECTED_BY_TOOL)
        record = {
            "invocation": invocation,
            "executor_turn": turn_no,
            "orchestrator_decision": decision_no,
            "run_name": run_name,
            "arguments": dict(arguments),
            "rationale": rationale,
            "status": status,
            "error": None if ok else str(output.get("error")),
            "acquisitions": frames,
            "result": output if ok else None,
            "summary": validated_summary(output) if ok else None,
        }
        self._tool_ledger[run_name] = record
        return record

    def format_executor_turn(self, records: list[dict], problems: list[str], all_experiments: list[dict]) -> str:
        done = [r for r in records if r["status"] == COMPLETED]
        if done:
            r = done[0]
            s = r["summary"]
            share_f = s["share_of_log_gain_from_focus_brightening"]
            lines = [
                f"{COMPLETED_MARK}",
                f"tool=FocusingSession.call (simulated bench) success=True",
                f"run_id={r['run_name']}",
                f"settings={json.dumps(r['arguments'], sort_keys=True)}",
                f"executor_rationale={r['rationale']}",
                f"validated_initial_enhancement={_fmt(s['validated_initial_enhancement_mean'])} "
                f"(std {_fmt(s['validated_initial_enhancement_std'])}, n={s['validation_count']})",
                f"validated_final_enhancement={_fmt(s['validated_final_enhancement_mean'])} "
                f"(std {_fmt(s['validated_final_enhancement_std'])}, n={s['validation_count']})",
                f"enhancement_gain={_fmt(s['enhancement_gain'])}  (local contrast vs. background strips)",
                f"focus_intensity_gain={_fmt(s['focus_intensity_gain'])}  (absolute brightening of the focus)",
                f"background_suppression_factor={_fmt(s['background_suppression_factor'])}",
                "log_gain_share_focus/background="
                + ("not reported" if share_f is None else
                   f"{share_f:.3f}/{s['share_of_log_gain_from_background_suppression']:.3f}"),
                f"camera_acquisitions={r['acquisitions']} (run total {run_acquisitions(all_experiments)})",
                f"transmission_fingerprint={s['transmission_fingerprint'][:16]}...",
                "artifacts:",
            ]
            lines += [f"  {k}={v}" for k, v in s["artifacts"].items()]
            lines += [f"tool_note={n}" for n in s["notes"]]
            lines.append("Search maxima are noisy and biased upward and are not reported here.")
        else:
            lines = [f"{REJECTED_MARK if records else NOT_RUN_MARK}", "success=False",
                     "no experiment completed in this turn; no measurement is reported"]
        if problems:
            lines.append("earlier problems in this turn:" if done else "problems:")
            lines += [f"  {p}" for p in problems]
        for r in records:
            if r["status"] != COMPLETED:
                lines.append(f"  tool call {r['run_name']}: {r['status']}, "
                             f"camera_acquisitions={r['acquisitions']}")
        return "\n".join(lines)

    # -----------------------------
    # Graph (HW1 topology; routing functions inherited from HW1)
    # -----------------------------

    def build(self, checkpointer=None):
        builder = StateGraph(FocusingState)
        builder.add_node("orchestrator", self.orchestrator)
        builder.add_node("scientist", self.scientist)
        builder.add_node("general", self.general)
        builder.add_node("executor", self.executor)

        builder.add_edge(START, "orchestrator")
        builder.add_conditional_edges(
            "orchestrator",
            self.route_after_orchestrator,
            {"scientist": "scientist", "general": "general", "executor": "executor", "finish": END},
        )
        for node in ("scientist", "general", "executor"):
            builder.add_conditional_edges(
                node, self.route_after_specialist, {"orchestrator": "orchestrator", END: END}
            )
        return builder.compile(checkpointer=checkpointer or InMemorySaver())


# -----------------------------
# Programmatic run interface
# -----------------------------

def thread_config(thread_id: str = "hw2-task2") -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}


def _stream(graph, graph_input, config: dict, on_turn: Callable[[dict], None] | None) -> None:
    seen = len(graph.get_state(config).values.get("dialogue", []))
    for state in graph.stream(graph_input, config=config, stream_mode="values"):
        dialogue = state.get("dialogue", [])
        for turn in dialogue[seen:]:
            if on_turn:
                on_turn(turn)
        seen = max(seen, len(dialogue))


def run_agent(system: FocusingAgentSystem, task_text: str, thread_id: str = "hw2-task2",
              on_turn: Callable[[dict], None] | None = None, graph=None, resume: bool = False) -> dict:
    """Run (or, with resume=True, continue) the thread to its end and summarise it honestly.

    status: completed (accepted finish with a program-built final report),
            decision_budget_exhausted (no finish within the eight decisions),
            error (a node raised; the state reached so far is reported).
    """
    graph = graph or system.build()
    config = thread_config(thread_id)
    error = None
    try:
        _stream(graph, None if resume else {"dialogue": [{"role": "user", "text": task_text}]},
                config, on_turn)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    values = graph.get_state(config).values
    dialogue = values.get("dialogue", [])
    experiments = values.get("experiments", [])
    final_report = values.get("final_report")
    decisions = [json.loads(t["text"])["next_role"] for t in dialogue if t["role"] == "orchestrator"]
    if error is not None:
        status, final_report = "error", None
    elif final_report is not None:
        status = "completed"
    else:
        status = "decision_budget_exhausted"
    return {
        "status": status,
        "error": error,
        "final_report": final_report,
        "partial_report": None if status == "completed" else build_partial_report(experiments, error or status),
        "dialogue": dialogue,
        "experiments": experiments,
        "orchestrator_decisions": decisions,
        "num_orchestrator_decisions": len(decisions),
        "tool_calls": len(experiments),
        "total_camera_acquisitions": run_acquisitions(experiments),
        "session_camera_acquisitions": system._session.describe()["camera_acquisitions_so_far"],
        "transmission_fingerprint": system._session.describe()["transmission_fingerprint"],
        "graph": graph,
        "config": config,
    }


def run_record(outcome: dict) -> dict:
    """The JSON-serialisable run record (everything but the graph objects)."""
    return {k: v for k, v in outcome.items() if k not in ("graph", "config")}


# -----------------------------
# Offline demonstration (MOCKED LLM, real focusing tool)
# -----------------------------

DEMO_TASK = "Focus the beam through the simulated scattering medium."
DEMO_SESSION = {
    "dmd_shape": (16, 16), "camera_shape": (32, 32), "medium_seed": 7,
    "noise": {"enabled": True, "photons_per_unit": 1000.0, "read_noise_std": 0.01},
    "acquisition_limit": 2000,
}
DEMO_RUN_ACQUISITION_LIMIT = 4000
DEMO_MAX_EXPERIMENTS = 3


class MockedDemoLLM:
    """OFFLINE / MOCKED model for the wiring demonstration only.

    It returns pre-written replies by role; it does not read the inputs. Every
    measured number in the demo comes from the real FocusingSession. The real
    model is connected in the next chunk.
    """

    ORCHESTRATOR = [
        {"next_role": "scientist", "instruction": "Propose a first focusing experiment for this session.",
         "reason": "[mocked] a plan is needed before measuring"},
        {"next_role": "executor", "instruction": "Run the scientist's first experiment.",
         "reason": "[mocked] a plan exists"},
        {"next_role": "executor", "instruction": "Try finer control: grouping factor 1 with more iterations.",
         "reason": "[mocked] compare a finer macropixel grid on the same medium"},
        {"next_role": "finish", "instruction": "Report the validated results.",
         "reason": "[mocked] two completed experiments on the same medium"},
    ]
    EXECUTOR = [
        # First proposal deliberately omits "seed" to show the bounded correction path.
        {"arguments": {"grouping_factor": 2, "focus_row": 15, "focus_col": 15, "focus_height": 2,
                       "focus_width": 2, "background_gap": 2, "background_width": 3, "iterations": 600,
                       "flips_per_proposal": 1, "averaging": 1, "validation_count": 10,
                       "max_acquisitions": 700},
         "rationale": "[mocked] 64 macropixels, central 2x2 focus"},
        {"arguments": {"grouping_factor": 2, "focus_row": 15, "focus_col": 15, "focus_height": 2,
                       "focus_width": 2, "background_gap": 2, "background_width": 3, "iterations": 600,
                       "flips_per_proposal": 1, "averaging": 1, "validation_count": 10, "seed": 1,
                       "max_acquisitions": 700},
         "rationale": "[mocked] corrected: seed added"},
        {"arguments": {"grouping_factor": 1, "focus_row": 15, "focus_col": 15, "focus_height": 2,
                       "focus_width": 2, "background_gap": 2, "background_width": 3, "iterations": 1200,
                       "flips_per_proposal": 1, "averaging": 1, "validation_count": 10, "seed": 2,
                       "max_acquisitions": 1300},
         "rationale": "[mocked] 256 macropixels, more iterations"},
    ]
    SCIENTIST = ("[mocked scientist] Start with grouping factor 2 (64 macropixels), a central 2x2 focus, "
                 "background strips 2 px away and 3 px wide, single flips, 600 iterations, 10 validation "
                 "measurements. Then compare grouping factor 1. Report enhancement gain and focus "
                 "intensity gain separately.")

    def __init__(self):
        self.orchestrator = [json.dumps(x) for x in self.ORCHESTRATOR]
        self.executor = [json.dumps(x) for x in self.EXECUTOR]
        self.calls: list[str] = []

    def call(self, instructions: str, input_text: str) -> str:
        role = instructions.split()[3].lower() if instructions.startswith("You are the") else "?"
        self.calls.append(role)
        if role == "orchestrator":
            return self.orchestrator.pop(0)
        if role == "executor":
            return self.executor.pop(0)
        if role == "scientist":
            return self.SCIENTIST
        return "[mocked general] (no answer scripted)"


def run_offline_demo(workdir: Path, out=print) -> dict:
    workdir = ensure_fresh_workdir(Path(workdir).resolve())
    workdir.mkdir(parents=True, exist_ok=True)
    session = FocusingSession(**DEMO_SESSION, artifact_root=workdir / "artifacts")
    llm = MockedDemoLLM()
    system = FocusingAgentSystem(llm, session, run_id="demo",
                                 max_experiments=DEMO_MAX_EXPERIMENTS,
                                 run_acquisition_limit=DEMO_RUN_ACQUISITION_LIMIT)

    out("=" * 78)
    out("OFFLINE WIRING DEMONSTRATION - the LLM is MOCKED (scripted replies, no API call).")
    out("The focusing tool, simulated bench and all measured values are REAL tool output.")
    out("=" * 78)
    out(f"Session: {json.dumps({k: v for k, v in session.describe().items() if k != 'argument_schema'})}")
    out(f"Run limits: max_experiments={DEMO_MAX_EXPERIMENTS} run_acquisition_limit={DEMO_RUN_ACQUISITION_LIMIT} "
        f"max_argument_proposals_per_turn={MAX_ARGUMENT_ATTEMPTS} max_decisions={MAX_DECISIONS}")
    out(f"User request: {DEMO_TASK}")

    def on_turn(turn: dict) -> None:
        out(f"\n===== {turn['role'].upper()} =====\n{turn['text']}")

    outcome = run_agent(system, DEMO_TASK, thread_id="hw2-task2-offline-demo", on_turn=on_turn)
    save_dialogue(outcome["dialogue"], workdir)
    record_path = write_new_file(workdir / "run_record.json",
                                 json.dumps({"mocked_llm": True, **run_record(outcome)}, indent=2))

    out("\n===== TOOL INVOCATIONS (FocusingSession.call) =====")
    for e in outcome["experiments"]:
        out(f"call {e['invocation']:02d}: FocusingSession.call({json.dumps({**e['arguments'], 'run_name': e['run_name']})})")
        if e["status"] == COMPLETED:
            s = e["summary"]
            out(f"   -> ok=True validated enhancement {s['validated_initial_enhancement_mean']:.4g} -> "
                f"{s['validated_final_enhancement_mean']:.4g} (gain {_fmt(s['enhancement_gain'])}, "
                f"focus intensity gain {_fmt(s['focus_intensity_gain'])}), acquisitions {e['acquisitions']}")
        else:
            out(f"   -> ok=False {e['status']}: {e['error']}")
    out("\n===== FINAL MESSAGE =====")
    out(outcome["final_report"]["message"] if outcome["final_report"] else json.dumps(outcome["partial_report"], indent=2))
    out("\n===== RUN SUMMARY =====")
    out(f"status={outcome['status']}")
    out(f"orchestrator decisions ({outcome['num_orchestrator_decisions']}): {outcome['orchestrator_decisions']}")
    out(f"mocked LLM calls by role: {llm.calls}")
    out(f"focusing tool calls: {outcome['tool_calls']}")
    out(f"total simulated camera acquisitions: {outcome['total_camera_acquisitions']} "
        f"(session counter {outcome['session_camera_acquisitions']})")
    out(f"transmission fingerprint (same medium for every call): {outcome['transmission_fingerprint']}")
    out(f"dialogue: {(workdir / 'dialogue.json').as_posix()}")
    out(f"run record: {record_path.as_posix()}")
    for e in completed(outcome["experiments"]):
        out(f"artifacts {e['run_name']}: {Path(e['summary']['artifacts']['result']).parent.as_posix()}")
    return outcome


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m hw2.task2_agent",
        description="HW2 Task 2: HW1 agent roles driving the simulated focusing tool (offline demo only).",
    )
    parser.add_argument("--offline-demo", action="store_true", required=True,
                        help="Run the mocked-LLM wiring demonstration with the real focusing tool.")
    parser.add_argument("--workdir", default=None,
                        help="New/empty output directory (default: task6_runs/hw2_task2_offline_<timestamp>).")
    args = parser.parse_args(argv)
    workdir = Path(args.workdir or REPO_ROOT / "task6_runs" / f"hw2_task2_offline_{datetime.now():%Y%m%d_%H%M%S}")
    try:
        outcome = run_offline_demo(workdir)
    except OverwriteRefused as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0 if outcome["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
