"""Offline tests for HW2 Task 2 agent integration (HW1 roles + real focusing tool).

No paid LLM is called: the system receives a scripted model with the same
call(instructions, input_text) interface as task6's ModelCaller, and the OpenAI
client constructor is patched to fail if anything tries to create it. The
focusing tool is the REAL FocusingSession on a small simulated bench in a
temporary directory; no measured value is faked.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hw2 import task2_agent as t2  # noqa: E402
from hw2.focusing_tool import FocusingSession  # noqa: E402
from hw2.task2_agent import (  # noqa: E402
    AgenticSilverboxSystem,
    FocusingAgentSystem,
    run_agent,
)

TASK = "Focus the beam through the simulated scattering medium."
SECRET = "sk-test-SENTINEL-not-a-real-key-1234567890"
NOISE = {"enabled": True, "photons_per_unit": 300.0, "read_noise_std": 0.02}


def settings(**overrides) -> dict:
    base = {
        "grouping_factor": 2, "focus_row": 7, "focus_col": 7, "focus_height": 2, "focus_width": 2,
        "background_gap": 1, "background_width": 2, "iterations": 30, "flips_per_proposal": 1,
        "averaging": 1, "validation_count": 3, "seed": 0, "max_acquisitions": 100,
    }
    base.update(overrides)
    return base


def proposal(rationale: str = "mocked choice", **overrides) -> str:
    return json.dumps({"arguments": settings(**overrides), "rationale": rationale})


def decision(next_role: str, instruction: str = "") -> str:
    return json.dumps({"next_role": next_role, "instruction": instruction or f"mocked {next_role}",
                       "reason": f"mocked decision for {next_role}"})


class ScriptedLLM:
    """Mocked model: orchestrator and executor replies come from scripts."""

    def __init__(self, orchestrator_replies, executor_replies=()):
        self.orchestrator_replies = list(orchestrator_replies)
        self.executor_replies = list(executor_replies)
        self.calls: list[tuple[str, str, str]] = []

    def call(self, instructions: str, input_text: str) -> str:
        for role in ("ORCHESTRATOR", "SCIENTIST", "GENERAL", "EXECUTOR"):
            if instructions.startswith(f"You are the {role}"):
                break
        else:
            raise AssertionError("unknown role called")
        role = role.lower()
        self.calls.append((role, instructions, input_text))
        if role == "orchestrator":
            if not self.orchestrator_replies:
                raise AssertionError("orchestrator called more often than scripted")
            return self.orchestrator_replies.pop(0)
        if role == "executor":
            if not self.executor_replies:
                raise AssertionError("executor LLM called more often than scripted")
            return self.executor_replies.pop(0)
        return f"{role.upper()} ANSWER: use grouping factor 2 and a central focus."

    def inputs(self, role: str) -> list[str]:
        return [text for r, _, text in self.calls if r == role]

    def count(self, role: str) -> int:
        return sum(r == role for r, _, _ in self.calls)


class RecordingSession(FocusingSession):
    """The real FocusingSession; also records which functions were on the stack of each call."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.call_log: list[dict] = []

    def call(self, arguments):
        self.call_log.append({"arguments": dict(arguments),
                              "stack": [f.function for f in inspect.stack()[1:8]]})
        return super().call(arguments)


class AgentTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="hw2_task2_agent_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        patcher = mock.patch.object(t2.hw1, "OpenAI", side_effect=AssertionError("real OpenAI client used"))
        patcher.start()
        self.addCleanup(patcher.stop)

    def make(self, orchestrator, executor=(), **system_kwargs) -> FocusingAgentSystem:
        self.llm = ScriptedLLM(orchestrator, executor)
        self.root = self.tmp / f"artifacts_{len(list(self.tmp.iterdir()))}"   # new root per system
        self.session = RecordingSession((8, 8), (16, 16), medium_seed=3, noise=NOISE,
                                        artifact_root=self.root, acquisition_limit=500)
        self.system = FocusingAgentSystem(self.llm, self.session, run_id="t", **system_kwargs)
        return self.system

    def run_task(self, orchestrator, executor=(), **system_kwargs) -> dict:
        self.make(orchestrator, executor, **system_kwargs)
        return run_agent(self.system, TASK)

    def acquisitions(self) -> int:
        return self.session.describe()["camera_acquisitions_so_far"]


class TestArchitectureReuse(AgentTestCase):
    def test_01_four_roles_and_hw1_graph(self):
        system = self.make([])
        self.assertIsInstance(system, AgenticSilverboxSystem)
        graph = system.build()
        self.assertEqual(set(graph.get_graph().nodes) - {"__start__", "__end__"},
                         {"orchestrator", "scientist", "general", "executor"})
        # HW1 routing functions and decision limit are reused unchanged.
        self.assertIs(FocusingAgentSystem.route_after_specialist, AgenticSilverboxSystem.route_after_specialist)
        self.assertIs(FocusingAgentSystem.route_after_orchestrator, AgenticSilverboxSystem.route_after_orchestrator)
        self.assertEqual(t2.MAX_DECISIONS, 8)

    def test_02_orchestrator_selection_uses_the_llm(self):
        a = self.run_task([decision("general"), decision("scientist")] * 4)
        self.assertEqual(a["orchestrator_decisions"], ["general", "scientist"] * 4)
        self.assertEqual(self.llm.count("orchestrator"), 8)
        b = self.run_task([decision("scientist")] * 8)
        self.assertEqual(b["orchestrator_decisions"], ["scientist"] * 8)   # same task, other route
        # An invalid reply is sent back to the LLM, not replaced by a program choice.
        c = self.run_task(["not json", json.dumps({"next_role": "coder"}), decision("general")]
                          + [decision("general")] * 7)
        self.assertEqual(c["orchestrator_decisions"][0], "general")
        self.assertIn("REJECTED", self.llm.inputs("orchestrator")[1])
        self.assertIn("REJECTED", self.llm.inputs("orchestrator")[2])

    def test_03_shared_dialogue_preserved(self):
        out = self.run_task([decision("scientist"), decision("executor"), decision("finish")], [proposal()])
        roles = [t["role"] for t in out["dialogue"]]
        self.assertEqual(roles, ["user", "orchestrator", "scientist", "orchestrator", "executor", "orchestrator"])
        self.assertEqual(out["dialogue"][0]["text"], TASK)
        executor_input = self.llm.inputs("executor")[0]
        self.assertIn(TASK, executor_input)
        self.assertIn("SCIENTIST ANSWER", executor_input)                       # optional scientist plan
        self.assertIn("mocked executor", executor_input)                        # orchestrator instruction
        self.assertIn("SCIENTIST ANSWER", self.llm.inputs("orchestrator")[1])   # everyone sees the dialogue


class TestExecutorTool(AgentTestCase):
    def test_04_executor_alone_invokes_the_tool(self):
        out = self.run_task([decision("scientist"), decision("general"), decision("executor"),
                             decision("executor"), decision("finish")],
                            [proposal(seed=1), proposal(seed=2)])
        self.assertEqual(len(self.session.call_log), 2)
        for entry in self.session.call_log:
            self.assertIn("executor", entry["stack"])
            self.assertIn("_invoke_tool", entry["stack"])
        # The other nodes never touch the tool, even when called directly.
        state = {"dialogue": out["dialogue"], "experiments": out["experiments"]}
        self.llm.orchestrator_replies = [decision("general")]
        for node in (self.system.orchestrator, self.system.scientist, self.system.general):
            node(state)
        self.assertEqual(len(self.session.call_log), 2)

    def test_05_executor_llm_supplies_every_setting(self):
        out = self.run_task([decision("executor"), decision("finish")], [proposal(seed=11, iterations=25)])
        record = out["experiments"][0]
        self.assertEqual(record["arguments"], settings(seed=11, iterations=25))   # exactly the LLM's values
        sent = self.session.call_log[0]["arguments"]
        self.assertEqual({k: v for k, v in sent.items() if k != "run_name"}, settings(seed=11, iterations=25))
        self.assertEqual(sent["run_name"], "t_call01")                            # assigned by the program
        executor_input = self.llm.inputs("executor")[0]
        for key in ("dmd_shape", "camera_shape", "valid_grouping_factors", "argument_schema",
                    "acquisition_limit_per_call", "run_acquisitions_remaining"):
            self.assertIn(key, executor_input)
        for name in t2.REQUIRED_SETTINGS:
            self.assertIn(f'"{name}"', executor_input)
        self.assertEqual(len(t2.REQUIRED_SETTINGS), 13)
        # Omitting any one setting is rejected; nothing is filled in.
        limits = self.system._limits([])
        for name in t2.REQUIRED_SETTINGS:
            args = settings()
            del args[name]
            parsed, _, errors = self.system.check_proposal(json.dumps({"arguments": args}), limits)
            self.assertIsNone(parsed, name)
            self.assertTrue(any(name in e for e in errors), name)

    def test_06_different_llm_settings_give_different_records(self):
        out = self.run_task([decision("executor"), decision("executor"), decision("finish")],
                            [proposal(grouping_factor=2, seed=1), proposal(grouping_factor=1, seed=5, iterations=40)])
        first, second = out["experiments"]
        self.assertEqual(first["arguments"], settings(grouping_factor=2, seed=1))
        self.assertEqual(second["arguments"], settings(grouping_factor=1, seed=5, iterations=40))
        self.assertNotEqual(first["arguments"], second["arguments"])
        for record in (first, second):
            saved = json.loads(Path(record["result"]["artifacts"]["settings"]).read_text(encoding="utf-8"))
            self.assertEqual(saved["tool_arguments"], record["arguments"])
        self.assertEqual(second["acquisitions"], 1 + 40 + 2 * 3)

    def test_07_invalid_arguments_do_not_run_experiments(self):
        bad = [
            proposal(seed=None),                                                       # null
            json.dumps({"arguments": {**settings(), "iterations": 30.0}}),             # float, not cast
            json.dumps({"arguments": {**settings(), "run_name": "../escape"}}),        # program-owned field
        ]
        out = self.run_task([decision("executor")] + [decision("general")] * 7, bad)
        self.assertEqual(out["experiments"], [])
        self.assertEqual(self.session.call_log, [])
        self.assertEqual(self.acquisitions(), 0)
        self.assertEqual(list(self.root.iterdir()), [])
        turn = next(t["text"] for t in out["dialogue"] if t["role"] == "executor")
        self.assertTrue(turn.startswith(t2.NOT_RUN_MARK))
        self.assertIn("success=False", turn)
        self.assertEqual(self.llm.count("executor"), t2.MAX_ARGUMENT_ATTEMPTS)    # bounded correction
        self.assertIn("NOTHING WAS MEASURED", self.llm.inputs("executor")[1])

    def test_07_tool_rejection_is_recorded_and_corrected(self):
        out = self.run_task([decision("executor"), decision("finish")],
                            [proposal(focus_row=15), proposal(grouping_factor=3), proposal()])
        statuses = [e["status"] for e in out["experiments"]]
        self.assertEqual(statuses, [t2.REJECTED_BY_TOOL, t2.REJECTED_BY_TOOL, t2.COMPLETED])
        for e in out["experiments"][:2]:
            self.assertEqual(e["acquisitions"], 0)
            self.assertIsNone(e["result"])
            self.assertFalse((self.root / e["run_name"]).exists())
        self.assertIn("outside", self.llm.inputs("executor")[1])                 # tool error fed back
        self.assertEqual(out["status"], "completed")

    def test_08_invalid_outputs_handled_honestly(self):
        out = self.run_task([decision("executor"), decision("finish"), decision("finish"), decision("finish")],
                            ["I would like to focus the beam.", "{broken json", json.dumps([1, 2])])
        self.assertEqual(out["status"], "error")
        self.assertIn("did not return a valid decision", out["error"])
        self.assertIsNone(out["final_report"])
        self.assertEqual(out["partial_report"]["completed_experiments"], [])
        self.assertEqual(self.acquisitions(), 0)
        self.assertIn("finish is forbidden", self.llm.inputs("orchestrator")[2])

    def test_09_same_transmission_matrix_reused(self):
        fingerprint = self.make([]).session_description["transmission_fingerprint"]
        out = self.run_task([decision("executor")] * 3 + [decision("finish")],
                            [proposal(grouping_factor=g, seed=g) for g in (1, 2, 4)])
        self.assertEqual({e["result"]["transmission_fingerprint"] for e in out["experiments"]}, {fingerprint})
        self.assertEqual(out["transmission_fingerprint"], fingerprint)
        self.assertEqual(len(out["experiments"]), 3)

    def test_10_11_multiple_experiments_visible_to_orchestrator(self):
        out = self.run_task([decision("executor"), decision("executor"), decision("finish")],
                            [proposal(seed=1), proposal(seed=2, iterations=20)])
        self.assertEqual([e["status"] for e in out["experiments"]], [t2.COMPLETED] * 2)
        self.assertEqual(out["total_camera_acquisitions"], (1 + 30 + 6) + (1 + 20 + 6))
        self.assertEqual(out["total_camera_acquisitions"], out["session_camera_acquisitions"])
        inputs = self.llm.inputs("orchestrator")
        for i, e in enumerate(out["experiments"]):
            s = e["summary"]
            seen = inputs[i + 1]
            self.assertIn(f"run_id={e['run_name']}", seen)
            self.assertIn(f"validated_final_enhancement={s['validated_final_enhancement_mean']:.6g}", seen)
            self.assertIn(f"focus_intensity_gain={t2._fmt(s['focus_intensity_gain'])}", seen)
        self.assertNotIn("t_call02", inputs[1])


class TestHonestReporting(AgentTestCase):
    def test_12_only_validated_values_are_reported(self):
        out = self.run_task([decision("executor"), decision("finish")], [proposal(iterations=60)])
        result = out["experiments"][0]["result"]
        report = out["final_report"]
        v = result["validated"]
        selected = report["selected"]
        self.assertEqual(selected["validated_final_enhancement_mean"], v["final"]["enhancement_mean"])
        self.assertEqual(selected["validated_initial_enhancement_mean"], v["initial"]["enhancement_mean"])
        self.assertEqual(selected["enhancement_gain"], v["enhancement_gain"])
        self.assertEqual(selected["focus_intensity_gain"], v["focus_intensity_gain"])
        self.assertEqual(selected["background_suppression_factor"], v["background_suppression_factor"])
        self.assertNotIn("search", json.dumps(selected))
        best_search = result["search"]["best_observed_enhancement"]
        self.assertNotEqual(best_search, v["final"]["enhancement_mean"])
        self.assertNotIn(f"{best_search:.6g}", report["message"])
        executor_turn = next(t["text"] for t in out["dialogue"] if t["role"] == "executor")
        self.assertNotIn(f"{best_search:.6g}", executor_turn)
        # Contrast and absolute brightening stay separate, with the limitations stated.
        self.assertIn("Enhancement gain (local contrast)", report["message"])
        self.assertIn("Focus intensity gain (absolute brightening", report["message"])
        self.assertIn("brightest point", report["message"])
        self.assertNotAlmostEqual(selected["enhancement_gain"], selected["focus_intensity_gain"], places=3)

    def test_13_successful_experiment_permits_finish_without_rmse_gate(self):
        out = self.run_task([decision("executor"), decision("finish")], [proposal(iterations=1)])
        self.assertEqual(out["status"], "completed")
        self.assertEqual(out["orchestrator_decisions"], ["executor", "finish"])
        self.assertEqual(out["final_report"]["selected_run_id"], "t_call01")
        for text in self.llm.inputs("orchestrator"):
            self.assertNotIn("TEST_RMSE", text)
            self.assertNotIn("quality gate", text.lower())

    def test_14_empty_results_cannot_become_success(self):
        self.assertIsNone(t2.build_final_report([], "x"))
        out = self.run_task(
            [decision("executor"), decision("finish"), decision("general")] + [decision("general")] * 6,
            [proposal(focus_row=15)] * 3,
        )
        self.assertEqual([e["status"] for e in out["experiments"]], [t2.REJECTED_BY_TOOL] * 3)
        self.assertEqual(out["status"], "decision_budget_exhausted")
        self.assertIsNone(out["final_report"])
        self.assertNotIn("finish", out["orchestrator_decisions"])
        self.assertIn("finish is forbidden", self.llm.inputs("orchestrator")[2])
        self.assertEqual(out["partial_report"]["completed_experiments"], [])

    def test_15_eight_decision_limit(self):
        out = self.run_task([decision("scientist")] * 8)
        self.assertEqual(out["num_orchestrator_decisions"], 8)
        self.assertEqual(self.llm.count("orchestrator"), 8)
        self.assertEqual(out["status"], "decision_budget_exhausted")
        # An experiment chosen at decision 8 still runs; the run then ends honestly unfinished.
        out = self.run_task([decision("general")] * 7 + [decision("executor")], [proposal()])
        self.assertEqual(out["num_orchestrator_decisions"], 8)
        self.assertEqual(out["dialogue"][-1]["role"], "executor")
        self.assertEqual(out["status"], "decision_budget_exhausted")
        self.assertIsNone(out["final_report"])
        self.assertEqual(len(out["partial_report"]["completed_experiments"]), 1)
        self.assertIn("not finalized", out["partial_report"]["note"])


class TestSafetyAndLimits(AgentTestCase):
    def test_16_no_arbitrary_coding_tool(self):
        system = self.make([decision("executor"), decision("finish")], [proposal()])
        self.assertIsNone(system.coding_tool)
        with mock.patch.object(t2.hw1.PythonCodingTool, "write_and_run",
                               side_effect=AssertionError("coding tool used")), \
             mock.patch.object(subprocess, "run", side_effect=AssertionError("subprocess used")), \
             mock.patch.object(subprocess, "Popen", side_effect=AssertionError("subprocess used")):
            self.assertEqual(run_agent(system, TASK)["status"], "completed")
        tree = ast.parse(Path(t2.__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, {"exec", "eval", "compile", "__import__"})
        self.assertFalse(imported & {"subprocess", "os", "shutil", "openai", "importlib"})
        self.assertNotIn("PythonCodingTool", {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)})

    def test_17_no_real_llm_or_api_key_exposure(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": SECRET}):
            out = self.run_task([decision("scientist"), decision("executor"), decision("finish")], [proposal()])
        self.assertEqual(out["status"], "completed")
        self.assertIs(self.system.llm, self.llm)
        for _, instructions, text in self.llm.calls:
            self.assertNotIn(SECRET, instructions + text)
        for path in self.root.rglob("*"):
            if path.is_file():
                self.assertNotIn(SECRET.encode(), path.read_bytes(), path)
        self.assertNotIn(SECRET, json.dumps(t2.run_record(out)))

    def test_18_no_files_outside_the_working_directory(self):
        default_root = REPO / "task6_runs"
        before = sorted(p.as_posix() for p in default_root.rglob("*")) if default_root.exists() else []
        self.make([decision("executor"), decision("finish")], [proposal(), proposal()])
        root = self.root
        # A pre-existing directory with the next run name is refused, not overwritten.
        (root / "t_call01").mkdir()
        sentinel = root / "t_call01" / "keep.txt"
        sentinel.write_text("earlier evidence", encoding="utf-8")
        out = run_agent(self.system, TASK)
        self.assertEqual([e["status"] for e in out["experiments"]], [t2.REJECTED_BY_TOOL, t2.COMPLETED])
        self.assertIn("already exists", out["experiments"][0]["error"])
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "earlier evidence")
        self.assertEqual(sorted(p.name for p in (root / "t_call01").iterdir()), ["keep.txt"])
        for path in out["experiments"][1]["result"]["artifacts"].values():
            self.assertEqual(Path(path).resolve().parent.parent, root.resolve())
        after = sorted(p.as_posix() for p in default_root.rglob("*")) if default_root.exists() else []
        self.assertEqual(before, after)

    def test_run_limits_enforced(self):
        # Run-level acquisition budget: a proposal above the remaining frames is refused before the tool.
        out = self.run_task([decision("executor"), decision("executor"), decision("finish")],
                            [proposal(), proposal(max_acquisitions=100), proposal(max_acquisitions=80, iterations=40)],
                            run_acquisition_limit=120)
        self.assertEqual([e["status"] for e in out["experiments"]], [t2.COMPLETED, t2.COMPLETED])
        self.assertIn("remaining 83 camera acquisitions", self.llm.inputs("executor")[2])
        self.assertLessEqual(out["total_camera_acquisitions"], 120)
        # Experiment limit: the orchestrator may not route to the executor again.
        out = self.run_task([decision("executor"), decision("executor"), decision("finish")], [proposal()],
                            max_experiments=1)
        self.assertEqual(out["orchestrator_decisions"], ["executor", "finish"])
        self.assertIn("executor is forbidden", self.llm.inputs("orchestrator")[2])
        self.assertEqual(len(self.session.call_log), 1)
        # Fail closed if the executor node is entered anyway.
        state = {"dialogue": out["dialogue"], "experiments": out["experiments"]}
        update = self.system.executor(state)
        self.assertTrue(update["dialogue"][0]["text"].startswith(t2.NOT_RUN_MARK))
        self.assertEqual(len(self.session.call_log), 1)

    def test_no_duplicate_tool_call_on_graph_retry(self):
        system = self.make([decision("executor"), decision("finish")], [proposal()])
        graph = system.build()
        real_format = system.format_executor_turn
        failures = iter([RuntimeError("crash after the tool returned")])

        def flaky(*args, **kwargs):
            exc = next(failures, None)
            if exc:
                raise exc
            return real_format(*args, **kwargs)

        system.format_executor_turn = flaky
        first = run_agent(system, TASK, graph=graph)
        self.assertEqual(first["status"], "error")
        self.assertEqual(len(self.session.call_log), 1)
        frames = self.acquisitions()
        second = run_agent(system, TASK, graph=graph, resume=True)      # same thread, from the checkpoint
        self.assertEqual(second["status"], "completed")
        self.assertEqual(len(self.session.call_log), 1)                  # no second experiment
        self.assertEqual(self.acquisitions(), frames)
        self.assertEqual(system.replayed_tool_calls, ["t_call01"])
        self.assertEqual(self.llm.count("executor"), 1)                  # proposal not regenerated
        self.assertEqual(len(second["experiments"]), 1)


class TestOfflineDemo(AgentTestCase):
    def test_offline_demo_runs_real_tool_with_mocked_llm(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = t2.main(["--offline-demo", "--workdir", str(self.tmp / "demo")])
        text = buffer.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("MOCKED", text)
        self.assertIn("missing argument(s): ['seed']", text)           # correction path shown
        record = json.loads((self.tmp / "demo" / "run_record.json").read_text(encoding="utf-8"))
        self.assertTrue(record["mocked_llm"])
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["orchestrator_decisions"], ["scientist", "executor", "executor", "finish"])
        for e in record["experiments"]:
            saved = json.loads(Path(e["result"]["artifacts"]["result"]).read_text(encoding="utf-8"))
            self.assertEqual(saved["validated"], e["result"]["validated"])
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(t2.main(["--offline-demo", "--workdir", str(self.tmp / "demo")]), 2)  # never reused


class TestOriginalFilesUnchanged(unittest.TestCase):
    # SHA-256 of the protected HW2 files as found before chunk 2C.2, computed after
    # canonicalising CRLF to LF (fixed baselines; newline style is checkout-dependent).
    PROTECTED = {
        "hw2/task1_hitl.py": "9147f1f6758c5c7f8c6439da7929eba31cce4d5231a8932914fdf5e17a2e1ffb",
        "hw2/demo_task1.py": "13ec67b7ad5b988fe5785964efc51c0d2135877dcca29ce830179e66e8df7e5d",
        "hw2/demo_task1_hard.py": "aa4ec6b95f2aeacac14fb02514d57083f0b27fd7fabdf3c7e70378ebb06e3a8f",
        "hw2/simulated_bench.py": "cfd076b0ec4b7e92c188df25ca50dff9a5578862931d4d50f9668fbabfe2349d",
        "hw2/focusing_optimizer.py": "fd3671bb4f0e3096b2ba98ffea0955438f6198c666673434b3221e0cf4b45ed2",
        "hw2/focusing_tool.py": "f712be47caa775def9a7c7aeae9f3ab371ec2540a9340e7d0bff9586aebb7dfe",
        "hw2/tests/test_task1_hitl.py": "5482c9be482c4cdc9fe036a4c8fd95c5df4b1ff445523fa0e4e6ce4aed32576c",
        "hw2/tests/test_focusing_tool.py": "e57f10eaebce87ce55e40b403b32b68e3b1c2ddb9b05567207e2362784cc74df",
        "hw2/tests/test_focusing_optimizer.py": "b5afcb78342031f8a227b9684e93acab492bf1177c6ff160cfaffce18316254f",
        "hw2/tests/test_simulated_bench.py": "c40efd856d92d307552f33c3faa6596f21cc9427f13a5cd3e3d46887a81755c2",
    }

    def test_20_hw1_and_hw2_task1_files_unchanged(self):
        for rel, digest in self.PROTECTED.items():
            # Newline-only differences (CRLF vs LF checkouts) are not content changes:
            # hash the content with CRLF canonicalised to LF against the fixed LF baseline.
            canonical = (REPO / rel).read_bytes().replace(b"\r\n", b"\n")
            self.assertEqual(hashlib.sha256(canonical).hexdigest(), digest, rel)
        if shutil.which("git") is None:
            self.skipTest("git not available")
        out = subprocess.run(["git", "status", "--porcelain", "--", "task5", "task6", "README.md",
                              "requirements.txt"], cwd=REPO, capture_output=True, text=True, check=True).stdout
        self.assertEqual(out, "")


if __name__ == "__main__":
    unittest.main()
