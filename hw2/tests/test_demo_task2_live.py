"""Offline tests for the HW2 Task 2 live runner (hw2/demo_task2_live.py).

No external API is called. The real runtime path is exercised by replacing
the OpenAI client class used by task6's ModelCaller with a fake client whose
responses come from a script; every socket connection attempt fails the test.
The focusing tool is the REAL FocusingSession in a temporary directory.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hw2 import demo_task2_live as live  # noqa: E402
from hw2 import task2_agent as t2  # noqa: E402

SECRET = "sk-test-LIVE-SENTINEL-0123456789abcdef"


def settings(**overrides) -> dict:
    base = {
        "grouping_factor": 4, "focus_row": 15, "focus_col": 15, "focus_height": 2, "focus_width": 2,
        "background_gap": 2, "background_width": 3, "iterations": 20, "flips_per_proposal": 1,
        "averaging": 1, "validation_count": 3, "seed": 0, "max_acquisitions": 100,
    }
    base.update(overrides)
    return base


def proposal(**overrides) -> str:
    return json.dumps({"arguments": settings(**overrides), "rationale": "scripted test choice"})


def decision(next_role: str) -> str:
    return json.dumps({"next_role": next_role, "instruction": f"scripted {next_role}",
                       "reason": f"scripted decision for {next_role}"})


class Script:
    """Scripted replies by role (stands in for the model behind the fake API)."""

    def __init__(self, orchestrator, executor=(), scientist="SCIENTIST PLAN: grouping 4, small budget."):
        self.orchestrator = list(orchestrator)
        self.executor = list(executor)
        self.scientist = scientist
        self.requests: list[dict] = []

    def reply(self, instructions: str, input_text: str) -> str:
        role = instructions.split()[3].lower()
        self.requests.append({"role": role, "instructions": instructions, "input": input_text})
        if role == "orchestrator":
            return self.orchestrator.pop(0)
        if role == "executor":
            return self.executor.pop(0)
        return self.scientist if role == "scientist" else "GENERAL ANSWER"

    def call(self, instructions: str, input_text: str) -> str:   # ModelCaller interface
        return self.reply(instructions, input_text)


def fake_openai_class(script: Script):
    """A stand-in for openai.OpenAI: client.responses.create(...) -> object with output_text."""
    created: list[dict] = []

    class FakeOpenAI:
        instances = created

        def __init__(self, *args, **kwargs):
            created.append({"args": args, "kwargs": kwargs, "requests": []})
            self.responses = SimpleNamespace(create=self._create)

        def _create(self, model, instructions, input):
            created[-1]["requests"].append({"model": model})
            return SimpleNamespace(output_text=script.reply(instructions, input))

    return FakeOpenAI


def no_network(*args, **kwargs):
    raise AssertionError("network access attempted in an offline test")


class LiveTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="hw2_task2_live_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        for target, attr, effect in ((t2.hw1, "OpenAI", AssertionError("real OpenAI client used")),
                                     (socket.socket, "connect", None), (socket, "create_connection", None)):
            patcher = mock.patch.object(target, attr, side_effect=effect or no_network)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.output: list[str] = []

    def run_live(self, script: Script, name: str = "run", **kwargs) -> dict:
        kwargs.setdefault("max_api_calls", 30)
        return live.run_live(script, model="fake-model", task_text=live.DEFAULT_REQUEST,
                             workdir=self.tmp / name, thread_id=f"test-{name}",
                             backend="OFFLINE TEST: scripted model", out=self.output.append, **kwargs)

    def load(self, name: str, file: str):
        return json.loads((self.tmp / name / file).read_text(encoding="utf-8"))

    def run_main(self, script: Script, *args: str) -> tuple[int, str, type]:
        fake = fake_openai_class(script)
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(t2.hw1, "OpenAI", fake), \
             mock.patch.dict(os.environ, {"OPENAI_API_KEY": SECRET}), \
             redirect_stdout(out), redirect_stderr(err):
            code = live.main(list(args))
        return code, out.getvalue() + err.getvalue(), fake


TWO_EXPERIMENTS = ([decision("scientist"), decision("executor"), decision("executor"), decision("finish")],
                   [proposal(seed=1), proposal(grouping_factor=2, seed=2, iterations=30)])


class TestRealRuntimePath(LiveTestCase):
    def test_02_03_main_constructs_model_caller_without_network(self):
        script = Script(*TWO_EXPERIMENTS)
        workdir = self.tmp / "main_run"
        code, text, fake = self.run_main(script, "--model", "model-x", "--workdir", str(workdir),
                                         "--max-api-calls", "20")
        self.assertEqual(code, 0, text)
        self.assertEqual(len(fake.instances), 1)                                  # one real ModelCaller client
        requests = fake.instances[0]["requests"]
        self.assertEqual({r["model"] for r in requests}, {"model-x"})
        evidence = json.loads((workdir / "demo_evidence.json").read_text(encoding="utf-8"))
        self.assertEqual(evidence["api_call_count"], len(requests))
        self.assertEqual(evidence["llm_backend"], live.BACKEND_REAL)
        self.assertEqual(evidence["model"], "model-x")
        self.assertIn("Model: model-x", text)
        self.assertIn(f"Workdir: {workdir.resolve()}", text)
        self.assertIn("OPENAI_API_KEY: set (value not shown)", text)
        self.assertNotIn(SECRET, text)

    def test_missing_key_refuses_without_fallback(self):
        with mock.patch.dict(os.environ, {}, clear=True), redirect_stderr(io.StringIO()) as err:
            code = live.main(["--workdir", str(self.tmp / "nokey")])
        self.assertEqual(code, 2)
        self.assertIn("no offline fallback", err.getvalue())
        self.assertFalse((self.tmp / "nokey").exists())

    def test_help(self):
        with redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit) as ctx:
            live.main(["--help"])
        self.assertEqual(ctx.exception.code, 0)
        self.assertIn("--max-api-calls", out.getvalue())


class TestAgentUse(LiveTestCase):
    def test_01_uses_existing_agent_and_graph(self):
        with mock.patch.object(t2, "run_agent", wraps=t2.run_agent) as spy:
            evidence = self.run_live(Script(*TWO_EXPERIMENTS))
        system = spy.call_args.args[0]
        self.assertIs(type(system), t2.FocusingAgentSystem)
        self.assertEqual(system.max_experiments, 3)
        self.assertEqual(system.run_acquisition_limit, 4000)
        self.assertEqual(system.max_argument_attempts, 3)
        self.assertEqual(system._session.acquisition_limit, 2000)
        c = evidence["session_constraints"]
        self.assertEqual((c["dmd_shape"], c["camera_shape"], c["medium_seed"]), ([16, 16], [32, 32], 7))
        self.assertEqual(c["noise"], {"enabled": True, "photons_per_unit": 1000.0, "read_noise_std": 0.01})
        self.assertEqual(evidence["completion_status"], "completed")

    def test_04_no_forced_routing_or_hard_coded_parameters(self):
        a = self.run_live(Script(*TWO_EXPERIMENTS), "a")
        b = self.run_live(Script([decision("executor"), decision("finish")], [proposal(seed=9, iterations=12)]), "b")
        self.assertEqual(a["orchestrator_decisions"], ["scientist", "executor", "executor", "finish"])
        self.assertEqual(b["orchestrator_decisions"], ["executor", "finish"])
        self.assertEqual([p["arguments"] for p in a["llm_selected_parameter_sets"]],
                         [settings(seed=1), settings(grouping_factor=2, seed=2, iterations=30)])
        self.assertEqual([p["arguments"] for p in b["llm_selected_parameter_sets"]],
                         [settings(seed=9, iterations=12)])
        source = Path(live.__file__).read_text(encoding="utf-8")
        for forbidden in ('"next_role":', "next_role=", '"iterations":', '"grouping_factor":', '"seed":'):
            self.assertNotIn(forbidden, source)
        self.assertEqual(live.SESSION_CONFIG.keys() & set(t2.REQUIRED_SETTINGS), set())

    def test_05_same_T_reused(self):
        evidence = self.run_live(Script(*TWO_EXPERIMENTS))
        rows = self.load("run", "run_record.json")["experiments"]
        self.assertEqual(len(rows), 2)
        self.assertEqual({e["result"]["transmission_fingerprint"] for e in rows},
                         {evidence["transmission_fingerprint"]})

    def test_09_no_executor_coding_tool(self):
        with mock.patch.object(t2, "run_agent", wraps=t2.run_agent) as spy, \
             mock.patch.object(subprocess, "run", side_effect=AssertionError("subprocess")), \
             mock.patch.object(subprocess, "Popen", side_effect=AssertionError("subprocess")):
            self.run_live(Script(*TWO_EXPERIMENTS))
        self.assertIsNone(spy.call_args.args[0].coding_tool)
        tree = ast.parse(Path(live.__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, {"exec", "eval", "compile", "__import__"})
        self.assertFalse(imported & {"subprocess", "shutil", "openai", "importlib", "ctypes"})


class TestEvidence(LiveTestCase):
    FILES = {"dialogue.json", "dialogue.txt", "run_record.json", "demo_evidence.json",
             "experiments.json", "experiments.csv"}

    def test_06_11_12_successful_run_saves_matching_evidence(self):
        evidence = self.run_live(Script(*TWO_EXPERIMENTS))
        names = {p.name for p in (self.tmp / "run").iterdir()}
        self.assertEqual(names, self.FILES | {"final_report.txt", "artifacts"})
        self.assertEqual(self.load("run", "demo_evidence.json"), json.loads(json.dumps(evidence)))
        rows = self.load("run", "experiments.json")
        with open(self.tmp / "run" / "experiments.csv", encoding="utf-8", newline="") as fh:
            csv_rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(csv_rows), 2)
        for row, csv_row in zip(rows, csv_rows):
            saved = json.loads((Path(row["artifact_dir"]) / "result.json").read_text(encoding="utf-8"))
            v = saved["validated"]
            self.assertEqual(row["validated_final_enhancement_mean"], v["final"]["enhancement_mean"])
            self.assertEqual(row["validated_initial_enhancement_mean"], v["initial"]["enhancement_mean"])
            self.assertEqual(row["focus_intensity_gain"], v["focus_intensity_gain"])
            self.assertEqual(row["enhancement_gain"], v["enhancement_gain"])
            self.assertEqual(row["search_best_observed_enhancement_BIASED_not_a_result"],
                             saved["search"]["best_observed_enhancement"])
            self.assertNotEqual(row["search_best_observed_enhancement_BIASED_not_a_result"],
                                row["validated_final_enhancement_mean"])
            self.assertEqual(row["camera_acquisitions"], saved["acquisitions"]["counted_by_bench"])
            self.assertEqual({k: row[k] for k in t2.REQUIRED_SETTINGS}, saved["arguments"])
            self.assertEqual(float(csv_row["validated_final_enhancement_mean"]), row["validated_final_enhancement_mean"])
            self.assertTrue(row["committed_to_graph_state"])
        self.assertEqual(evidence["total_camera_acquisitions"], (1 + 20 + 6) + (1 + 30 + 6))
        self.assertEqual(evidence["total_camera_acquisitions"], evidence["session_camera_acquisitions"])
        report = (self.tmp / "run" / "final_report.txt").read_text(encoding="utf-8")
        self.assertIn("Enhancement gain (local contrast)", report)
        self.assertIn("brightest point", report)
        for key in ("model", "user_request", "thread_id", "run_id", "orchestrator_decisions",
                    "llm_selected_parameter_sets", "tool_calls", "validated_results",
                    "transmission_fingerprint", "api_call_count", "runtime_s", "completion_status"):
            self.assertIn(key, evidence)
        self.assertEqual(evidence["api_call_count"], 7)    # 4 orchestrator + 1 scientist + 2 executor
        text = "\n".join(self.output)
        for marker in ("decision 1 of 8", "LLM-selected destination: scientist", "SCIENTIST plan",
                       "executor LLM proposed tool arguments", "[SIMULATED MEASUREMENT] FocusingSession.call",
                       "tool result: SUCCESS", "[VALIDATED RESULT] initial enhancement",
                       "focus intensity gain", "camera_acquisitions=", "Another experiment requested: YES",
                       "Another experiment requested: NO", "FINAL REPORT", "OFFLINE TEST: scripted model"):
            self.assertIn(marker, text)

    def test_07_08_api_cap_stops_safely_and_keeps_measurements(self):
        inner = Script(*TWO_EXPERIMENTS)
        counter = mock.Mock(side_effect=inner.call)
        model = SimpleNamespace(call=counter)
        evidence = self.run_live(model, max_api_calls=4)   # orchestrator, scientist, orchestrator, executor
        self.assertEqual(counter.call_count, 4)              # the cap is never exceeded
        self.assertEqual(evidence["api_call_count"], 4)
        self.assertEqual(evidence["api_calls_refused_by_cap"], 1)
        self.assertEqual(evidence["completion_status"], "incomplete_api_call_budget_exhausted")
        self.assertIsNone(evidence["final_report"])
        self.assertEqual(len(evidence["validated_results"]), 1)                    # measurement preserved
        self.assertEqual(len(evidence["partial_report"]["completed_experiments"]), 1)
        names = {p.name for p in (self.tmp / "run").iterdir()}
        self.assertEqual(names, self.FILES | {"artifacts"})                       # no final_report.txt
        self.assertIn("PARTIAL REPORT", "\n".join(self.output))

    def test_07_tool_calls_of_an_interrupted_executor_turn_are_kept(self):
        # The cap hits during an executor correction: the tool-rejected call was never
        # committed to graph state, but it is still in the evidence.
        script = Script([decision("executor")], [proposal(focus_row=31)])
        evidence = self.run_live(script, max_api_calls=2)
        self.assertEqual(evidence["completion_status"], "incomplete_api_call_budget_exhausted")
        rows = self.load("run", "experiments.json")
        self.assertEqual([r["status"] for r in rows], [t2.REJECTED_BY_TOOL])
        self.assertFalse(rows[0]["committed_to_graph_state"])
        self.assertEqual(evidence["tool_calls"]["not_committed_to_graph_state"], [rows[0]["run_id"]])
        self.assertEqual(evidence["total_camera_acquisitions"], 0)

    def test_07_error_and_decision_budget_runs_preserve_evidence(self):
        bad = self.run_live(Script(["?", "?", "?"]), "bad")
        self.assertEqual(bad["completion_status"], "error")
        self.assertIn("did not return a valid decision", bad["error"])
        self.assertTrue(self.FILES <= {p.name for p in (self.tmp / "bad").iterdir()})
        long = self.run_live(Script([decision("general")] * 7 + [decision("executor")], [proposal()]), "long")
        self.assertEqual(long["completion_status"], "incomplete_decision_budget_exhausted")
        self.assertEqual(len(long["validated_results"]), 1)
        self.assertIsNone(long["final_report"])

    def test_10_workdir_collisions_rejected(self):
        existing = self.tmp / "taken"
        existing.mkdir()
        (existing / "dialogue.json").write_text("earlier evidence", encoding="utf-8")
        with self.assertRaises(live.OverwriteRefused):
            self.run_live(Script(*TWO_EXPERIMENTS), "taken")
        code, text, fake = self.run_main(Script(*TWO_EXPERIMENTS), "--workdir", str(existing))
        self.assertEqual(code, 2)
        self.assertEqual(fake.instances, [])                       # refused before any model client
        self.assertEqual((existing / "dialogue.json").read_text(encoding="utf-8"), "earlier evidence")
        self.assertEqual([p.name for p in existing.iterdir()], ["dialogue.json"])

    def test_13_secrets_not_persisted(self):
        leaky = Script(*TWO_EXPERIMENTS, scientist=f"PLAN (echoing {SECRET} by mistake)")
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": SECRET, "OTHER_TOKEN": SECRET[::-1]}):
            self.run_live(leaky)
        for path in (self.tmp / "run").rglob("*"):
            if path.is_file():
                data = path.read_bytes()
                self.assertNotIn(SECRET.encode(), data, path)
                self.assertNotIn(SECRET[::-1].encode(), data, path)
        self.assertIn("[REDACTED]", (self.tmp / "run" / "dialogue.txt").read_text(encoding="utf-8"))
        self.assertNotIn(SECRET, "\n".join(self.output))
        for request in leaky.requests:                             # the key never enters a prompt
            self.assertNotIn(SECRET, request["instructions"] + request["input"].replace(
                f"echoing {SECRET}", ""))


class TestProtectedFiles(unittest.TestCase):
    # Fixed SHA-256 baselines of the protected files with CRLF canonicalised to LF.
    PROTECTED = {
        "hw2/task2_agent.py": "a54beb9b1b508c50274b5e4a7a8ee859b7d6a803e7dfcf0d0dc9dd4ad1eb46c2",
        "hw2/tests/test_task2_agent.py": "441c2c4a2c45cfb5fb04fcf66a1a344160082fb50dfaefa293f24e5b6f34cfa7",
        "hw2/task1_hitl.py": "9147f1f6758c5c7f8c6439da7929eba31cce4d5231a8932914fdf5e17a2e1ffb",
        "hw2/demo_task1.py": "13ec67b7ad5b988fe5785964efc51c0d2135877dcca29ce830179e66e8df7e5d",
        "hw2/demo_task1_hard.py": "aa4ec6b95f2aeacac14fb02514d57083f0b27fd7fabdf3c7e70378ebb06e3a8f",
        "hw2/simulated_bench.py": "cfd076b0ec4b7e92c188df25ca50dff9a5578862931d4d50f9668fbabfe2349d",
        "hw2/focusing_optimizer.py": "fd3671bb4f0e3096b2ba98ffea0955438f6198c666673434b3221e0cf4b45ed2",
        "hw2/focusing_tool.py": "f712be47caa775def9a7c7aeae9f3ab371ec2540a9340e7d0bff9586aebb7dfe",
    }

    def test_15_previous_files_unchanged(self):
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
