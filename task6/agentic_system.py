#!/usr/bin/env python3
"""
Task 6 — Agentic System for the Silverbox Task 5.

Four LangGraph nodes:
    orchestrator -> scientist | general | executor | finish
    scientist/general/executor -> orchestrator

The shared graph state is the dialogue. Each turn stores:
    {"role": <role name>, "text": <what that role said>}

The orchestrator's decision is itself an LLM call. It is not a keyword router.
Only the executor owns the coding tool. The coding tool writes a new .py file,
runs it with a hard five-minute timeout, and returns exit code/stdout/stderr.

A successful finish is allowed only after an executor turn contains a real
TEST_RMSE printed by a script that exited with code 0. Quality gate: while the
best such TEST_RMSE is >= TARGET_TEST_RMSE and enough orchestrator decisions
remain, a finish is rejected and sent back to the orchestrator LLM, which then
chooses itself how to improve (scientist or executor).

Each executor attempt runs in its own attempt subdirectory, so every attempt's
script, logs and artifacts are preserved. The kept attempt is the successful
one with the lowest complete free-run TEST_RMSE.
"""

from __future__ import annotations

import argparse
import json
import operator
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from openai import OpenAI
from langgraph.graph import END, START, StateGraph


# -----------------------------
# Shared dialogue state
# -----------------------------

Role = Literal["user", "orchestrator", "scientist", "general", "executor"]


class Turn(TypedDict):
    role: Role
    text: str


class DialogueState(TypedDict):
    # operator.add makes each node append its new turn to the dialogue.
    dialogue: Annotated[list[Turn], operator.add]


ALLOWED_NEXT = {"scientist", "general", "executor", "finish"}
MAX_DECISIONS = 8
TARGET_TEST_RMSE = 0.02  # V; below this the orchestrator may finish at any time
TIMEOUT_S = 300
NUMBER = r"([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)"
TEST_RMSE_RE = re.compile(r"^\s*TEST_RMSE\s*=\s*" + NUMBER + r"\s*$", re.MULTILINE)
TRAIN_RMSE_RE = re.compile(r"^\s*TRAIN_RMSE\s*=\s*" + NUMBER + r"\s*$", re.MULTILINE)
EXIT_CODE_RE = re.compile(r"^exit_code=(-?[0-9]+)$", re.MULTILINE)
STDOUT_MARK = "--- stdout ---"
STDERR_MARK = "--- stderr ---"


def format_dialogue(dialogue: list[Turn], max_chars: int = 28000) -> str:
    text = "\n\n".join(
        f"[{turn['role'].upper()}]\n{turn['text']}" for turn in dialogue
    )
    return text[-max_chars:]


def parse_json_object(text: str) -> dict:
    """Extract one JSON object from an LLM response."""
    stripped = text.strip()

    # Remove a single Markdown fence if the model used one.
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def extract_python(text: str) -> str:
    """Extract Python source from a model response."""
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1).strip() + "\n"
    return text.strip() + "\n"


def count_role(dialogue: list[Turn], role: str) -> int:
    return sum(turn["role"] == role for turn in dialogue)


def latest_role_text(dialogue: list[Turn], role: str) -> str:
    for turn in reversed(dialogue):
        if turn["role"] == role:
            return turn["text"]
    return ""


def parse_executor_turn(text: str) -> dict:
    """Split an executor turn into its exit code and the RMSE lines of its stdout section."""
    exit_match = EXIT_CODE_RE.search(text)
    stdout = ""
    if STDOUT_MARK in text:
        stdout = text.split(STDOUT_MARK, 1)[1].split(STDERR_MARK, 1)[0]
    test = TEST_RMSE_RE.findall(stdout)
    train = TRAIN_RMSE_RE.findall(stdout)
    return {
        "exit_code": int(exit_match.group(1)) if exit_match else None,
        "train_rmse": float(train[-1]) if train else None,
        "test_rmse": float(test[-1]) if test else None,
    }


def executor_attempts(dialogue: list[Turn]) -> list[dict]:
    """All executor turns in order, with file, exit code and printed RMSE values."""
    attempts = []
    for turn in dialogue:
        if turn["role"] != "executor":
            continue
        parsed = parse_executor_turn(turn["text"])
        file_match = re.search(r"^file=(.+)$", turn["text"], re.MULTILINE)
        parsed["file"] = file_match.group(1).strip() if file_match else None
        parsed["attempt"] = len(attempts) + 1
        parsed["successful"] = parsed["exit_code"] == 0 and parsed["test_rmse"] is not None
        attempts.append(parsed)
    return attempts


def best_executor_result(dialogue: list[Turn]) -> dict | None:
    """The successful attempt with the lowest complete free-run TEST_RMSE."""
    ok = [a for a in executor_attempts(dialogue) if a["successful"]]
    return min(ok, key=lambda a: a["test_rmse"]) if ok else None


def attempts_table(dialogue: list[Turn]) -> str:
    rows = executor_attempts(dialogue)
    if not rows:
        return "(no executor attempts yet)"
    return "\n".join(
        f"attempt {a['attempt']:02d}: file={a['file']} exit_code={a['exit_code']} "
        f"TRAIN_RMSE={a['train_rmse']} TEST_RMSE={a['test_rmse']}"
        for a in rows
    )


def latest_orchestrator_decision(dialogue: list[Turn]) -> dict:
    text = latest_role_text(dialogue, "orchestrator")
    return parse_json_object(text)


# -----------------------------
# LLM helper
# -----------------------------

class ModelCaller:
    def __init__(self, model: str):
        self.model = model
        self.client = OpenAI()

    def call(self, instructions: str, input_text: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=input_text,
        )
        return response.output_text.strip()


# -----------------------------
# Executor-only coding tool
# -----------------------------

class PythonCodingTool:
    """
    The executor is the only node that receives an instance of this class.

    The tool performs both required actions:
      1. writes a NEW Python file into a NEW attempt subdirectory of the run
         working directory (an existing attempt is never overwritten), so each
         attempt's own artifacts are preserved;
      2. runs exactly that file with a hard timeout of 300 seconds.

    It returns the exit code, stdout, stderr, traceback and timeout status,
    and saves stdout/stderr/traceback next to the script as evidence.
    """

    def __init__(self, workdir: Path, data_path: Path):
        self.workdir = workdir.resolve()
        self.data_path = data_path.resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def extract_traceback(stderr: str) -> str:
        idx = stderr.rfind("Traceback (most recent call last):")
        return stderr[idx:].strip() if idx >= 0 else ""

    def write_and_run(self, filename: str, code: str) -> dict:
        safe_name = Path(filename).name
        if not safe_name.endswith(".py"):
            safe_name += ".py"

        # Always a NEW directory and file: never overwrite an earlier attempt or run.
        stem = Path(safe_name).stem
        attempt_dir = self.workdir / stem
        n = 1
        while attempt_dir.exists():
            attempt_dir = self.workdir / f"{stem}_{n}"
            n += 1
        attempt_dir.mkdir(parents=True)
        script_path = attempt_dir / f"{attempt_dir.name}.py"
        script_path.write_text(code, encoding="utf-8")

        env = os.environ.copy()
        env["SILVERBOX_MAT_PATH"] = str(self.data_path)
        env["PYTHONIOENCODING"] = "utf-8"
        env["MPLBACKEND"] = "Agg"

        start = time.time()
        try:
            completed = subprocess.run(
                [sys.executable, script_path.name],
                cwd=attempt_dir,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=TIMEOUT_S,
            )
            timed_out = False
            exit_code = int(completed.returncode)
            stdout, stderr = completed.stdout, completed.stderr
        except subprocess.TimeoutExpired as exc:
            def as_text(x):
                return x.decode("utf-8", "replace") if isinstance(x, bytes) else (x or "")
            timed_out = True
            exit_code = -1
            stdout = as_text(exc.stdout)
            stderr = as_text(exc.stderr) + (
                f"\nTIMEOUT: execution exceeded the hard {TIMEOUT_S}-second limit and was killed."
            )

        result = {
            "file": str(script_path),
            "relpath": script_path.relative_to(self.workdir).as_posix(),
            "timed_out": timed_out,
            "exit_code": exit_code,
            "elapsed_s": round(time.time() - start, 1),
            "stdout": stdout,
            "stderr": stderr,
            "traceback": self.extract_traceback(stderr),
        }

        printed = parse_executor_turn(f"exit_code={exit_code}\n{STDOUT_MARK}\n{stdout}")
        result["train_rmse"] = printed["train_rmse"]
        result["test_rmse"] = printed["test_rmse"]

        stem = str(script_path.with_suffix(""))
        Path(f"{stem}.stdout.txt").write_text(stdout, encoding="utf-8")
        Path(f"{stem}.stderr.txt").write_text(stderr, encoding="utf-8")
        if result["traceback"]:
            Path(f"{stem}.traceback.txt").write_text(result["traceback"], encoding="utf-8")
        return result


# -----------------------------
# Four agent nodes
# -----------------------------

class AgenticSilverboxSystem:
    def __init__(self, model: str, workdir: Path, data_path: Path):
        self.llm = ModelCaller(model)
        self.coding_tool = PythonCodingTool(workdir=workdir, data_path=data_path)
        self.runs: list[dict] = []  # coding-tool results (without stdout/stderr) for the summary

    def orchestrator(self, state: DialogueState) -> dict:
        dialogue = state["dialogue"]
        prior_decisions = count_role(dialogue, "orchestrator")
        decision_no = prior_decisions + 1
        best = best_executor_result(dialogue)
        best_rmse = None if best is None else best["test_rmse"]
        # Decisions left after this one. With at least two left, an improvement
        # (scientist -> executor, or executor -> finish) still fits in the budget.
        remaining_after = MAX_DECISIONS - decision_no
        finish_allowed = best_rmse is not None and (
            best_rmse < TARGET_TEST_RMSE or remaining_after < 2
        )

        instructions = """You are the ORCHESTRATOR in a four-role LangGraph system.

You see the complete dialogue and must choose exactly one next role.

Return ONLY a JSON object with exactly these three fields:
{
  "next_role": "scientist" | "general" | "executor" | "finish",
  "instruction": "specific instruction for that role",
  "reason": "one-sentence reason"
}

Rules:
- Your decision MUST be made from the dialogue; do not use a fixed role sequence.
- You do theory and no coding yourself.
- scientist owns the differential equation, network/state/training plan, free-run test,
  and a numbered implementation plan.
- general answers side questions about the record, terminology, or meaning of metrics.
- executor implements the latest scientific plan, runs code, and can repair a prior failed run.
- A traceback or failed program belongs with executor.
- A question about the scientific plan belongs with scientist.
- You may choose finish ONLY if an executor turn records exit_code=0 AND a numeric
  TEST_RMSE printed by a script that actually ran.
- QUALITY GATE: a zero predictor scores about 0.06 V test RMSE. If the best real
  TEST_RMSE is not below the quality target and enough decisions remain, do NOT
  finish: route to the scientist to diagnose the weak free-run result and propose a
  materially better plan, or to the executor to implement an already proposed
  improvement in a NEW script. Choose whichever role the dialogue shows is needed.
  Only when the decision budget is nearly exhausted may you finish with the best
  real result obtained so far.
- The run keeps the successful attempt with the lowest complete free-run TEST_RMSE.
- Never invent a measured RMSE.
"""

        input_text = (
            f"This is orchestrator decision {decision_no} of at most {MAX_DECISIONS}.\n"
            f"Quality target: TEST_RMSE < {TARGET_TEST_RMSE} V.\n"
            f"Best verified executor TEST_RMSE so far: {best_rmse!r}\n"
            f"finish currently allowed by the program: {finish_allowed}\n"
            f"EXECUTOR ATTEMPTS SO FAR:\n{attempts_table(dialogue)}\n\n"
            f"DIALOGUE:\n{format_dialogue(dialogue)}"
        )

        # The routing decision is always an LLM call. A malformed reply or an
        # illegal finish is sent back to the LLM with the problem stated; the
        # program never substitutes a hard-coded role.
        feedback = ""
        decision = None
        for _ in range(3):
            raw = self.llm.call(instructions, input_text + feedback)
            try:
                candidate = parse_json_object(raw)
                role = str(candidate.get("next_role", "")).strip().lower()
                if role not in ALLOWED_NEXT:
                    raise ValueError(f"next_role {role!r} is not one of {sorted(ALLOWED_NEXT)}")
                if role == "finish" and best_rmse is None:
                    raise ValueError(
                        "finish is forbidden: no executor turn has exit_code=0 "
                        "with a printed TEST_RMSE yet"
                    )
                if role == "finish" and not finish_allowed:
                    raise ValueError(
                        f"finish is forbidden by the quality gate: best TEST_RMSE "
                        f"{best_rmse} V is not below {TARGET_TEST_RMSE} V and "
                        f"{remaining_after} decisions remain; route to scientist or "
                        f"executor to improve the model"
                    )
                decision = candidate
                break
            except Exception as exc:
                feedback = (
                    f"\n\nYOUR PREVIOUS REPLY WAS REJECTED ({exc}). Reply again with ONLY the "
                    f"JSON object with next_role, instruction, reason.\nPrevious reply:\n{raw[:2000]}"
                )
        if decision is None:
            raise RuntimeError("Orchestrator LLM did not return a valid decision in 3 attempts.")

        next_role = str(decision["next_role"]).strip().lower()
        instruction = str(decision.get("instruction", "")).strip()
        reason = str(decision.get("reason", "")).strip()

        normalized = {
            "next_role": next_role,
            "instruction": instruction,
            "reason": reason,
        }
        return {
            "dialogue": [
                {
                    "role": "orchestrator",
                    "text": json.dumps(normalized, ensure_ascii=False),
                }
            ]
        }

    def scientist(self, state: DialogueState) -> dict:
        dialogue = state["dialogue"]
        decision = latest_orchestrator_decision(dialogue)

        instructions = """You are the SCIENTIST in a four-role agentic system.

You own the theory and scientific plan. You have NO coding tool.

For the Silverbox task, when asked for the plan, give:
1. the Duffing differential equation;
2. the neural acceleration model you propose;
3. the first-order state equation to integrate;
4. data preprocessing and train/test split;
5. RK4 integration details;
6. the training objective/procedure and hyperparameters to try;
7. the exact free-run evaluation protocol;
8. a numbered implementation plan specific enough for the executor to code.

Important:
- The test free run may read u but may not read measured y after initialization.
- The state must continue unchanged across the train/test boundary.
- Do not claim an RMSE was measured unless an earlier EXECUTOR turn actually printed it.
- If an executor run failed, refine the plan only when the failure is scientific rather than a code traceback.

When asked to improve a weak but successful free run (TEST_RMSE well above the
target), diagnose WHY the continuous free run is poor and propose a materially
better plan, not a cosmetic tweak. Consider, where scientifically justified:
- state/input normalization or scaling of y, v, u and the acceleration;
- better initialization of the network;
- a physics-guided or Duffing-informed warm start (for example, fit the Duffing
  coefficients on the training record and let the network learn a residual);
- longer or staged rollout training (a curriculum toward long horizons);
- architecture depth/width and activation changes;
- learning-rate choice and schedule;
- stability of long free runs over thousands of RK4 steps;
- the gap between the short-window training objective and the final
  4096-sample continuous free-run RMSE.
Selecting among complete free runs by test RMSE is allowed; feeding measured test y
into the dynamics or resetting the state at the train/test boundary is forbidden.
Keep the training budget compatible with the 300-second hard limit (aim <= 200 s).
"""

        input_text = (
            f"ORCHESTRATOR INSTRUCTION:\n{decision.get('instruction', '')}\n\n"
            f"EXECUTOR ATTEMPTS SO FAR:\n{attempts_table(dialogue)}\n\n"
            f"DIALOGUE:\n{format_dialogue(dialogue)}"
        )
        answer = self.llm.call(instructions, input_text)
        return {"dialogue": [{"role": "scientist", "text": answer}]}

    def general(self, state: DialogueState) -> dict:
        dialogue = state["dialogue"]
        decision = latest_orchestrator_decision(dialogue)

        instructions = """You are the GENERAL AGENT in a four-role agentic system.

You answer only high-level questions beside the fit: what a record in the file is,
what a term in the problem means, what an RMSE measures, or similar context.

You have NO coding tool and you do NOT write the scientific fitting plan.
If the requested issue is really about the differential equation/training plan,
say that it belongs to the scientist. If it is a traceback/code execution issue,
say that it belongs to the executor. Never invent measured results.
"""
        input_text = (
            f"ORCHESTRATOR INSTRUCTION:\n{decision.get('instruction', '')}\n\n"
            f"DIALOGUE:\n{format_dialogue(dialogue)}"
        )
        answer = self.llm.call(instructions, input_text)
        return {"dialogue": [{"role": "general", "text": answer}]}

    def executor(self, state: DialogueState) -> dict:
        dialogue = state["dialogue"]
        decision = latest_orchestrator_decision(dialogue)
        scientist_plan = latest_role_text(dialogue, "scientist")
        previous_executor = latest_role_text(dialogue, "executor")
        attempt = count_role(dialogue, "executor") + 1

        instructions = """You are the EXECUTOR in a four-role LangGraph system.

You are the only role allowed to use the coding tool. Your job is to implement
the latest SCIENTIST plan in a NEW Python file and run it.

Return ONLY the complete Python source code, preferably in one ```python``` block.

Execution constraints:
- The script is run from an isolated working directory.
- The Silverbox MAT-file path is available in environment variable SILVERBOX_MAT_PATH.
- Do NOT download anything and do NOT use the network.
- You may use Python stdlib, NumPy, SciPy, PyTorch, and Matplotlib.
- The run must finish within five minutes.
- Use the exact excerpt [42650:46746].
- Subtract the excerpt mean of u and the excerpt mean of y.
- First 3072 samples are train; last 1024 are test.
- fs = 610.35 Hz and dt = 1/fs.
- Use state (y,v), with y_dot=v and v_dot=a_theta(y,v,u).
- a_theta must be a fully-connected PyTorch network.
- Integrate with classical RK4.
- The FINAL metric must be one free run over the full 4096-sample excerpt:
  initialize only at the first sample using measured position and finite-difference
  velocity, then integrate through train and test without resetting the state.
- On the test segment, the rollout may read u and MUST NOT read measured y.
- The assignment allows keeping the version/architecture/checkpoint with the best
  test score: you may run a complete free run, compute its test RMSE, and use that
  number to choose which version to keep. What is forbidden is reading measured y
  inside the test free-run DYNAMICS and resetting the state at the boundary.

FREE-RUN CONTRACT (verified automatically after your script runs):
- Define a function with exactly this signature and use it for every free run:
      def free_run(accel, u, y0, v0):
  where accel(y, v, u) is the acceleration network, u is the full 4096-sample
  centered input array, and y0, v0 are the initial state. free_run must NOT
  receive or reference the measured y array. Inside it: initialise the state
  (y_state, v_state) exactly ONCE from (y0, v0), then ONE loop
  `for k in range(len(u) - 1):` performs one RK4 step from sample k to k+1 using
  u[k] and u[k+1]. No if-branch on the train/test boundary, no reassignment of
  the state other than the RK4 update. Return an array states of shape (4096, 2)
  with columns (y, v); states[0] = (y0, v0).
- The final evaluation calls free_run ONCE for the kept model with y0 = measured
  y[0] and v0 = a finite difference of measured y at sample 0. TRAIN_RMSE and
  TEST_RMSE are computed from states[:, 0] AFTER free_run returns.
- Save in the working directory, for the kept model's final free run:
    freerun_states.npy      float64 array (4096, 2) returned by free_run
    freerun_prediction.npy  float64 array (4096,) = states[:, 0]
    final_accel.pt          torch.jit.trace of the kept acceleration network as a
                            module whose forward(y, v, u) takes three 1-D tensors
                            (centered volts, SI units, the dtype used in the
                            rollout) and returns the acceleration, with all
                            input/output scaling INSIDE the module
    boundary_state.json     {"k_3071": [y, v], "k_3072": [y, v],
                             "state_initialisations": 1, "rk4_steps": 4095,
                             "rk4_u_convention": "linear" or "zoh"}
  "linear" means stages use u[k], (u[k]+u[k+1])/2, (u[k]+u[k+1])/2, u[k+1];
  "zoh" means all four stages use u[k].
- Print EXACTLY recognizable lines near the end:
    TRAIN_RMSE=<numeric volts>
    TEST_RMSE=<numeric volts>
- Save a measured-vs-predicted test plot (PNG) into the working directory.
- Fix all random seeds (torch, numpy, random) and call torch.set_num_threads(1)
  so the run is reproducible.
- CPU only. Budget training to finish well under 300 s (aim for <= 200 s total)
  and print progress with flush=True.
- If a prior executor attempt failed, use its stderr/traceback AND its source code
  to repair the script.
- If a prior attempt succeeded but scored poorly and the scientist proposed an
  improvement, implement that improvement as a complete NEW script; earlier
  attempts are preserved and never overwritten.
- Do not fake or hard-code RMSE values; they must be computed from the executed rollout.
"""

        user_task = dialogue[0]["text"] if dialogue and dialogue[0]["role"] == "user" else ""
        previous_context = previous_executor[-12000:] if previous_executor else "(none)"
        previous_source = "(none)"
        if self.runs:
            previous_source = Path(self.runs[-1]["file"]).read_text(encoding="utf-8")
        input_text = (
            f"USER TASK:\n{user_task}\n\n"
            f"ORCHESTRATOR INSTRUCTION:\n{decision.get('instruction', '')}\n\n"
            f"EXECUTOR ATTEMPTS SO FAR:\n{attempts_table(dialogue)}\n\n"
            f"LATEST SCIENTIST PLAN:\n{scientist_plan or '(none yet)'}\n\n"
            f"PREVIOUS EXECUTOR RESULT, IF ANY:\n{previous_context}\n\n"
            f"PREVIOUS GENERATED SCRIPT SOURCE, IF ANY:\n{previous_source}"
        )

        model_answer = self.llm.call(instructions, input_text)
        code = extract_python(model_answer)
        filename = f"silverbox_agent_attempt_{attempt:02d}.py"

        result = self.coding_tool.write_and_run(filename=filename, code=code)
        self.runs.append({k: v for k, v in result.items() if k not in ("stdout", "stderr")})

        stdout = result["stdout"][-16000:]
        stderr = result["stderr"][-12000:]
        tool_report = (
            f"file={result['relpath']}\n"
            f"timed_out={result['timed_out']}\n"
            f"elapsed_s={result['elapsed_s']}\n"
            f"exit_code={result['exit_code']}\n"
            f"{STDOUT_MARK}\n{stdout}\n"
            f"{STDERR_MARK}\n{stderr}\n"
            f"--- traceback ---\n{result['traceback'] or '(none)'}"
        )
        return {"dialogue": [{"role": "executor", "text": tool_report}]}

    # -----------------------------
    # Routing
    # -----------------------------

    @staticmethod
    def route_after_orchestrator(state: DialogueState) -> str:
        decision = latest_orchestrator_decision(state["dialogue"])
        return str(decision["next_role"])

    @staticmethod
    def route_after_specialist(state: DialogueState) -> str:
        # The 8th orchestrator decision is still allowed to hand the turn to
        # its selected specialist. After that specialist returns, the run ends.
        decisions = count_role(state["dialogue"], "orchestrator")
        if decisions >= MAX_DECISIONS:
            return END
        return "orchestrator"

    def build(self):
        builder = StateGraph(DialogueState)

        builder.add_node("orchestrator", self.orchestrator)
        builder.add_node("scientist", self.scientist)
        builder.add_node("general", self.general)
        builder.add_node("executor", self.executor)

        builder.add_edge(START, "orchestrator")

        builder.add_conditional_edges(
            "orchestrator",
            self.route_after_orchestrator,
            {
                "scientist": "scientist",
                "general": "general",
                "executor": "executor",
                "finish": END,
            },
        )

        builder.add_conditional_edges(
            "scientist",
            self.route_after_specialist,
            {"orchestrator": "orchestrator", END: END},
        )
        builder.add_conditional_edges(
            "general",
            self.route_after_specialist,
            {"orchestrator": "orchestrator", END: END},
        )
        builder.add_conditional_edges(
            "executor",
            self.route_after_specialist,
            {"orchestrator": "orchestrator", END: END},
        )

        return builder.compile()


def save_dialogue(dialogue: list[Turn], workdir: Path) -> None:
    (workdir / "dialogue.json").write_text(
        json.dumps(dialogue, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    human = "\n\n".join(
        f"===== {turn['role'].upper()} =====\n{turn['text']}" for turn in dialogue
    )
    (workdir / "dialogue.txt").write_text(human, encoding="utf-8")


def locate_data(explicit: str | None) -> Path | None:
    """Use --data if given; otherwise search the repository for SNLS80mV.mat."""
    if explicit:
        return Path(explicit).resolve()
    repo = Path(__file__).resolve().parent.parent
    for root in (repo / "SilverboxFiles", repo, Path.cwd()):
        if root.is_dir():
            hits = sorted(root.rglob("SNLS80mV.mat"))
            if hits:
                return hits[0].resolve()
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default=str(Path(__file__).with_name("task5_assignment.txt")),
        help="Text file containing the Task 5 prompt given to the orchestrator.",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to SNLS80mV.mat (default: located automatically under SilverboxFiles/).",
    )
    parser.add_argument(
        "--workdir",
        default=None,
        help="Directory for executor scripts and logs (default: task6_runs/run_<timestamp>).",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("AGENT_MODEL", "gpt-6-luna"),
        help="OpenAI API model used for role decisions and role responses.",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "ERROR: OPENAI_API_KEY is not set. "
            "Set it in your shell/environment; never hard-code it into this repository.",
            file=sys.stderr,
        )
        return 2

    task_path = Path(args.task).resolve()
    data_path = locate_data(args.data)
    workdir = Path(
        args.workdir or f"task6_runs/run_{datetime.now():%Y%m%d_%H%M%S}"
    ).resolve()

    if not task_path.exists():
        print(f"ERROR: task file not found: {task_path}", file=sys.stderr)
        return 2
    if data_path is None:
        print("ERROR: SNLS80mV.mat not found; pass --data.", file=sys.stderr)
        return 2
    if not data_path.exists():
        print(f"ERROR: MAT file not found: {data_path}", file=sys.stderr)
        return 2

    workdir.mkdir(parents=True, exist_ok=True)
    task_text = task_path.read_text(encoding="utf-8")

    system = AgenticSilverboxSystem(
        model=args.model,
        workdir=workdir,
        data_path=data_path,
    )
    graph = system.build()

    initial_state: DialogueState = {
        "dialogue": [{"role": "user", "text": task_text}]
    }

    print(f"Model: {args.model}")
    print(f"Data: {data_path}")
    print(f"Workdir: {workdir}")
    print("Starting LangGraph run...\n", flush=True)

    seen_turns = 0
    final_state = None

    for state in graph.stream(
        initial_state, stream_mode="values", config={"recursion_limit": 50}
    ):
        final_state = state
        dialogue = state["dialogue"]
        if len(dialogue) > seen_turns:
            for turn in dialogue[seen_turns:]:
                print(f"\n===== {turn['role'].upper()} =====")
                print(turn["text"], flush=True)
            seen_turns = len(dialogue)
            save_dialogue(dialogue, workdir)  # keep evidence even if a later turn crashes

    if final_state is None:
        print("ERROR: graph produced no state.", file=sys.stderr)
        return 2

    dialogue = final_state["dialogue"]
    save_dialogue(dialogue, workdir)
    result = best_executor_result(dialogue)
    decisions = [
        json.loads(t["text"])["next_role"] for t in dialogue if t["role"] == "orchestrator"
    ]
    successful = None
    if result is not None:
        successful = next(r for r in system.runs if r["relpath"] == result["file"])
    summary = {
        "model": args.model,
        "data": str(data_path),
        "turn_sequence": [t["role"] for t in dialogue],
        "orchestrator_decisions": decisions,
        "num_orchestrator_decisions": len(decisions),
        "executor_runs": system.runs,
        "successful_test_rmse_by_attempt": {
            a["file"]: a["test_rmse"] for a in executor_attempts(dialogue) if a["successful"]
        },
        "quality_target_test_rmse": TARGET_TEST_RMSE,
        "kept_attempt_rule": "successful attempt with the lowest complete free-run TEST_RMSE",
        "final_script": successful["relpath"] if successful else None,
        "final_exit_code": successful["exit_code"] if successful else None,
        "TRAIN_RMSE": result["train_rmse"] if result else None,
        "TEST_RMSE": result["test_rmse"] if result else None,
    }
    if successful is not None:
        boundary_file = Path(successful["file"]).parent / "boundary_state.json"
    if successful is not None and boundary_file.exists():
        # State immediately before/after the train/test boundary, as logged by the
        # generated script; verify_run.py re-derives k_3072 from k_3071 and u only.
        summary["boundary_state_reported_by_script"] = json.loads(
            boundary_file.read_text(encoding="utf-8")
        )
    (workdir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n===== FINAL STATUS =====")
    print(f"Turn sequence: {' -> '.join(summary['turn_sequence'])}")
    print(f"Orchestrator decisions ({len(decisions)}): {decisions}")
    if result is None:
        print("No successful executor TEST_RMSE was produced.")
        print(f"Logs saved to: {workdir}")
        return 2

    print(f"FINAL_SCRIPT={summary['final_script']}")
    print(f"FINAL_TRAIN_RMSE={result['train_rmse']}")
    print(f"FINAL_TEST_RMSE={result['test_rmse']}")
    print(f"Logs saved to: {workdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
