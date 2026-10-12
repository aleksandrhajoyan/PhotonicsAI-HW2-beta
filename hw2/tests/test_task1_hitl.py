"""Offline tests for HW2 Task 1 (human routing + interrupt/resume).

No paid LLM is called: the system receives a scripted model with the same
call(instructions, input_text) interface as task6's ModelCaller, and the OpenAI
client constructor is patched to fail if anything tries to create it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hw2 import task1_hitl as hitl  # noqa: E402
from hw2.task1_hitl import (  # noqa: E402
    AgenticSilverboxSystem,
    HumanInTheLoopSystem,
    pending_question,
    resume,
    start,
    thread_config,
)

TASK = "Report the RMSE from the Silverbox work."
QUESTION = "Do you mean the Train RMSE or the Test RMSE, and from which task?"
REPLY = "Use the Task 6 Test RMSE."


def decision(next_role: str, question: str = "", instruction: str = "") -> str:
    return json.dumps({
        "next_role": next_role,
        "instruction": instruction,
        "question": question,
        "reason": f"mocked decision for {next_role}",
    })


def approval_request(question: str = "May the executor write and run the Silverbox script?",
                     instruction: str = "Implement the Silverbox plan in a new script and run it.") -> str:
    return json.dumps({
        "next_role": "human",
        "request": "approval",
        "instruction": instruction,
        "question": question,
        "reason": "mocked decision: running code needs human approval",
    })


def final_request(question: str = "May I end the dialogue and report the Task 6 Test RMSE?",
                  reason: str = "mocked decision: a real result exists and the run should end") -> str:
    return json.dumps({
        "next_role": "human",
        "request": "final_approval",
        "instruction": "",
        "question": question,
        "reason": reason,
    })


def generated_code(n: int) -> str:
    return f"# generated script {n}\nprint('TRAIN_RMSE=0.012')\nprint('TEST_RMSE=0.015')\n"


def count_orchestrator(dialogue: list[dict]) -> int:
    return hitl.count_role(dialogue, "orchestrator")


class ScriptedLLM:
    """Mocked model. Orchestrator and executor replies come from scripts; other roles get fixed text."""

    def __init__(self, orchestrator_replies: list[str], executor_replies: list[str] | None = None):
        self.orchestrator_replies = list(orchestrator_replies)
        self.executor_replies = list(executor_replies or [])
        self.calls: list[tuple[str, str, str]] = []  # (role, instructions, input_text)

    def call(self, instructions: str, input_text: str) -> str:
        if instructions.startswith("You are the ORCHESTRATOR"):
            role = "orchestrator"
        elif instructions.startswith("You are the GENERAL"):
            role = "general"
        elif instructions.startswith("You are the SCIENTIST"):
            role = "scientist"
        elif instructions.startswith("You are the EXECUTOR"):
            role = "executor"
        else:
            raise AssertionError("unknown role called")
        self.calls.append((role, instructions, input_text))
        if role == "orchestrator":
            if not self.orchestrator_replies:
                raise AssertionError("orchestrator called more often than scripted")
            return self.orchestrator_replies.pop(0)
        if role == "executor":
            if not self.executor_replies:
                raise AssertionError("executor LLM called more often than scripted")
            return f"```python\n{self.executor_replies.pop(0)}```"
        return f"{role.upper()} ANSWER: the Task 6 test RMSE is reported in run_summary.json."

    def orchestrator_inputs(self) -> list[str]:
        return [text for role, _, text in self.calls if role == "orchestrator"]

    def count(self, role: str) -> int:
        return sum(r == role for r, _, _ in self.calls)


class FakeCodingTool:
    """Records write_and_run calls and returns a canned result. Never runs a subprocess."""

    def __init__(self, workdir: Path, test_rmse: float = 0.015):
        self.workdir = workdir
        self.test_rmse = test_rmse
        self.calls: list[tuple[str, str]] = []

    def write_and_run(self, filename: str, code: str) -> dict:
        self.calls.append((filename, code))
        stem = Path(filename).stem
        path = self.workdir / f"{stem}_{len(self.calls)}" / filename
        path.parent.mkdir(parents=True)
        path.write_text(code, encoding="utf-8")  # the HW1 executor reads the previous source
        return {
            "file": str(path),
            "relpath": path.relative_to(self.workdir).as_posix(),
            "timed_out": False,
            "exit_code": 0,
            "elapsed_s": 0.1,
            "stdout": f"fake run\nTRAIN_RMSE=0.012\nTEST_RMSE={self.test_rmse}\n",
            "stderr": "",
            "traceback": "",
            "train_rmse": 0.012,
            "test_rmse": self.test_rmse,
        }


class HitlTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="hw2_task1_test_"))
        # Fail loudly if anything constructs the real OpenAI client.
        patcher = mock.patch.object(hitl.hw1, "OpenAI", side_effect=AssertionError("real OpenAI client used"))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def make_graph(self, replies: list[str], thread_id: str = "test-thread",
                   executor_replies: list[str] | None = None, history: dict | None = None):
        self.llm = ScriptedLLM(replies, executor_replies)
        self.tool = FakeCodingTool(self.tmp)
        self.system = HumanInTheLoopSystem(
            self.llm, workdir=self.tmp / "work", data_path=self.tmp / "SNLS80mV.mat",
            coding_tool=self.tool, history=history,
        )
        self.graph = self.system.build()
        self.config = thread_config(thread_id)
        return self.graph

    def dialogue(self) -> list[dict]:
        return self.graph.get_state(self.config).values["dialogue"]

    def action(self) -> dict:
        return self.graph.get_state(self.config).values["pending_action"]


class TestHumanRouting(HitlTestCase):
    def test_a_b_c_orchestrator_selects_human_and_graph_pauses(self):
        self.make_graph([decision("human", question=QUESTION)])
        state = start(self.graph, TASK, self.config)

        # A. the orchestrator LLM chose human, with the question recorded.
        orch = json.loads(state["dialogue"][-1]["text"])
        self.assertEqual(orch["next_role"], "human")
        self.assertEqual(orch["question"], QUESTION)
        self.assertTrue(orch["reason"])

        # B. a real LangGraph interrupt surfaces the exact question.
        snapshot = self.graph.get_state(self.config)
        self.assertEqual(len(snapshot.interrupts), 1)
        self.assertEqual(snapshot.interrupts[0].value["question"], QUESTION)
        self.assertEqual(pending_question(self.graph, self.config), QUESTION)

        # C. paused at the human node: nothing ran after the orchestrator.
        self.assertEqual(snapshot.next, ("human",))
        self.assertEqual([t["role"] for t in state["dialogue"]], ["user", "orchestrator"])
        self.assertEqual([c[0] for c in self.llm.calls], ["orchestrator"])

    def test_d_e_f_resume_same_thread_and_reply_visible(self):
        self.make_graph(
            [decision("human", question=QUESTION)]
            + [decision("general", instruction="Explain the Task 6 Test RMSE.")] * 7
        )
        start(self.graph, TASK, self.config)

        # Another thread has no state: the pause belongs to this thread_id only.
        self.assertEqual(self.graph.get_state(thread_config("other-thread")).values, {})

        # D. resume the SAME thread.
        state = resume(self.graph, REPLY, self.config)
        self.assertIsNone(pending_question(self.graph, self.config))
        self.assertEqual(self.graph.get_state(self.config).next, ())

        # E. the reply is in the dialogue exactly once, in the required form.
        dialogue = state["dialogue"]
        human_turns = [t for t in dialogue if t["role"] == "human"]
        self.assertEqual(human_turns, [{"role": "human", "text": REPLY}])
        self.assertEqual(dialogue[2], {"role": "human", "text": REPLY})
        self.assertEqual(sum(t["text"] == REPLY for t in dialogue), 1)

        # F. the next orchestrator decision sees the question and the reply.
        second_input = self.llm.orchestrator_inputs()[1]
        self.assertIn(f"[HUMAN]\n{REPLY}", second_input)
        self.assertIn(QUESTION, second_input)
        self.assertIn(TASK, second_input)
        # ...and so does every later decision.
        for later in self.llm.orchestrator_inputs()[1:]:
            self.assertIn(f"[HUMAN]\n{REPLY}", later)

    def test_g_non_human_route_continues_without_interrupt(self):
        self.make_graph([decision("general", instruction="Explain the RMSE.")] * 8)
        state = start(self.graph, TASK, self.config)

        self.assertIsNone(pending_question(self.graph, self.config))
        self.assertEqual(self.graph.get_state(self.config).next, ())
        roles = [t["role"] for t in state["dialogue"]]
        self.assertEqual(roles, ["user"] + ["orchestrator", "general"] * 8)
        self.assertNotIn("human", roles)

    def test_h_eight_decision_budget_with_human_as_last_decision(self):
        self.make_graph(
            [decision("general", instruction="Explain the RMSE.")] * 7
            + [decision("human", question=QUESTION)]
        )
        start(self.graph, TASK, self.config)
        self.assertEqual(pending_question(self.graph, self.config), QUESTION)
        self.assertEqual(hitl.count_role(self.dialogue(), "orchestrator"), 8)

        state = resume(self.graph, REPLY, self.config)
        roles = [t["role"] for t in state["dialogue"]]
        # The human decision counted as decision 8; after the reply the run ends.
        self.assertEqual(roles.count("orchestrator"), hitl.MAX_DECISIONS)
        self.assertEqual(roles[-1], "human")
        self.assertEqual(self.graph.get_state(self.config).next, ())
        self.assertEqual(len(self.llm.orchestrator_inputs()), 8)

    def test_h_human_decisions_count_toward_budget(self):
        self.make_graph([decision("human", question=f"{QUESTION} ({i})") for i in range(8)])
        start(self.graph, TASK, self.config)
        answered = 0
        while pending_question(self.graph, self.config) is not None:
            resume(self.graph, f"{REPLY} ({answered})", self.config)
            answered += 1
        roles = [t["role"] for t in self.dialogue()]
        self.assertEqual(answered, 8)
        self.assertEqual(roles, ["user"] + ["orchestrator", "human"] * 8)


class TestPreservedHw1Behaviour(HitlTestCase):
    def test_human_without_question_is_rejected_and_reasked(self):
        self.make_graph([decision("human", question=""), decision("human", question=QUESTION)])
        start(self.graph, TASK, self.config)
        self.assertEqual(pending_question(self.graph, self.config), QUESTION)
        self.assertIn("question is empty", self.llm.orchestrator_inputs()[1])

    def test_finish_without_executor_result_is_rejected(self):
        self.make_graph([decision("finish"), decision("human", question=QUESTION)])
        start(self.graph, TASK, self.config)
        self.assertIn("finish is forbidden", self.llm.orchestrator_inputs()[1])
        self.assertEqual(pending_question(self.graph, self.config), QUESTION)

    def test_four_original_roles_and_executor_unchanged(self):
        self.make_graph([])
        nodes = set(self.graph.get_graph().nodes) - {"__start__", "__end__"}
        self.assertEqual(nodes, {"orchestrator", "scientist", "general", "executor", "human"})
        for name in ("scientist", "general"):
            self.assertIs(getattr(HumanInTheLoopSystem, name), getattr(AgenticSilverboxSystem, name))
        # Chunk 1B: the executor is gated by human approval, but code generation is
        # still the HW1 executor and execution is still the HW1 coding tool.
        self.assertIsNot(HumanInTheLoopSystem.executor, AgenticSilverboxSystem.executor)
        self.assertEqual(hitl.ALLOWED_NEXT, hitl.hw1.ALLOWED_NEXT | {"human"})

    def test_initialisation_matches_hw1_attributes(self):
        workdir, data = self.tmp / "work", self.tmp / "SNLS80mV.mat"
        with mock.patch.object(hitl.hw1, "OpenAI"):  # offline stand-in for the real client
            original = AgenticSilverboxSystem(model="offline", workdir=workdir, data_path=data)
        default = HumanInTheLoopSystem(ScriptedLLM([]), workdir=workdir, data_path=data)
        # Every HW1 attribute is set; the only addition is the optional history context.
        self.assertEqual(set(vars(default)) - set(vars(original)), {"history_context"})
        self.assertTrue(set(vars(original)) <= set(vars(default)))
        self.assertIsNone(default.history_context)
        self.assertIsInstance(default.coding_tool, hitl.PythonCodingTool)
        self.assertEqual(default.coding_tool.workdir, original.coding_tool.workdir)
        self.assertEqual(default.runs, [])
        fake = FakeCodingTool(self.tmp)
        injected = HumanInTheLoopSystem(ScriptedLLM([]), workdir=workdir, data_path=data, coding_tool=fake)
        self.assertIs(injected.coding_tool, fake)


class TestApprovalGate(HitlTestCase):
    """Chunk 1B: the coding tool runs only for an action the human approved."""

    def test_1_coding_action_requires_approval(self):
        self.make_graph([approval_request()], executor_replies=[generated_code(1)])
        state = start(self.graph, "Run the Silverbox fitting script.", self.config)

        payload = hitl.pending_interrupt(self.graph, self.config)
        self.assertEqual(payload["kind"], "approval")
        action = payload["action"]
        self.assertEqual(action["action_type"], "python_write_and_run")
        self.assertEqual(action["target_role"], "executor")
        self.assertEqual(action["status"], "awaiting_approval")
        self.assertEqual(action["filename"], "silverbox_agent_attempt_01.py")
        self.assertIn("silverbox_agent_attempt_01.py", action["operation"])
        self.assertEqual(action["code"], generated_code(1))
        self.assertEqual(action["code_sha256"], hitl.code_sha256(generated_code(1)))
        self.assertTrue(action["action_id"].startswith("action-01-"))

        self.assertEqual(self.tool.calls, [])
        self.assertEqual(self.graph.get_state(self.config).next, ("human",))
        roles = [t["role"] for t in state["dialogue"]]
        self.assertEqual(roles, ["user", "orchestrator", "executor"])
        self.assertTrue(state["dialogue"][-1]["text"].startswith("ACTION_PREPARED (NOT EXECUTED"))

    def test_2_to_5_refusal_then_fresh_approval_executes_exactly_once(self):
        self.make_graph(
            [approval_request(), approval_request("The plan is unchanged. May the executor run it now?"),
             final_request()],
            executor_replies=[generated_code(1), generated_code(2)],
        )
        start(self.graph, "Run the Silverbox fitting script.", self.config)
        first_id = hitl.pending_interrupt(self.graph, self.config)["action"]["action_id"]

        # 2. refusal prevents execution.
        resume(self.graph, hitl.approval(first_id, False, "No, do not run it."), self.config)
        self.assertEqual(self.tool.calls, [])
        human_turns = [t for t in self.dialogue() if t["role"] == "human"]
        self.assertEqual(human_turns, [{"role": "human", "text": "No, do not run it."}])

        # 3. the orchestrator saw the refusal, and the refused action is recorded.
        second_input = self.llm.orchestrator_inputs()[1]
        self.assertIn("[HUMAN]\nNo, do not run it.", second_input)
        self.assertIn(f"action_id={first_id} status=refused", second_input)

        # It asked again: a NEW action, paused for a fresh approval.
        payload = hitl.pending_interrupt(self.graph, self.config)
        second_id = payload["action"]["action_id"]
        self.assertNotEqual(second_id, first_id)
        self.assertTrue(second_id.startswith("action-02-"))
        self.assertEqual(self.tool.calls, [])

        # 4./5. the approval runs exactly the approved code, exactly once.
        state = resume(self.graph, hitl.approval(second_id, True, "Yes, proceed."), self.config)
        self.assertEqual(self.tool.calls, [("silverbox_agent_attempt_01.py", generated_code(2))])
        dialogue = state["dialogue"]
        self.assertEqual(
            [t["text"] for t in dialogue if t["role"] == "human"],
            ["No, do not run it.", "Yes, proceed."],
        )
        executed = [t for t in dialogue if t["role"] == "executor" and t["text"].startswith("action_id=")]
        self.assertEqual(len(executed), 1)
        self.assertTrue(executed[0]["text"].startswith(f"action_id={second_id}"))
        self.assertIn("TEST_RMSE=0.015", executed[0]["text"])
        self.assertEqual(state["pending_action"]["status"], "executed")
        self.assertEqual(state["pending_action"]["outcome"]["test_rmse"], 0.015)

        # The graph continued: decision 3 saw the result and asked to finish
        # (Chunk 1C: finishing now goes through final-measurement approval).
        self.assertIn(f"action_id={second_id}", self.llm.orchestrator_inputs()[2])
        self.assertIn("Best verified executor TEST_RMSE in this run so far: 0.015",
                      self.llm.orchestrator_inputs()[2])
        self.assertEqual(json.loads(dialogue[-1]["text"])["request"], "final_approval")
        self.assertEqual(hitl.pending_interrupt(self.graph, self.config)["kind"], "final_approval")
        self.assertEqual(self.system.runs[0]["test_rmse"], 0.015)

    def test_6_resume_does_not_repeat_llm_generated_action(self):
        self.make_graph([approval_request()] + [decision("general")] * 7, executor_replies=[generated_code(1)])
        start(self.graph, "Run the Silverbox fitting script.", self.config)
        self.assertEqual(self.llm.count("executor"), 1)
        self.assertEqual(self.llm.count("orchestrator"), 1)
        action_id = self.action()["action_id"]

        resume(self.graph, hitl.approval(action_id, True, "Yes, proceed."), self.config)
        # The code was generated once, before the interrupt, and the stored code ran.
        self.assertEqual(self.llm.count("executor"), 1)
        self.assertEqual(self.tool.calls, [("silverbox_agent_attempt_01.py", generated_code(1))])
        prepared = [t for t in self.dialogue() if t["text"].startswith("ACTION_PREPARED")]
        self.assertEqual(len(prepared), 1)

    def test_7_approval_cannot_be_reused(self):
        self.make_graph(
            [approval_request(), approval_request("May the executor run a second script?")]
            + [decision("general")] * 6,
            executor_replies=[generated_code(1), generated_code(2)],
        )
        start(self.graph, "Run the Silverbox fitting script.", self.config)
        first_id = self.action()["action_id"]
        resume(self.graph, hitl.approval(first_id, True, "Yes, proceed."), self.config)
        self.assertEqual(len(self.tool.calls), 1)
        # The orchestrator saw the first action as executed (consumed) before asking again.
        self.assertIn(f"action_id={first_id} status=executed", self.llm.orchestrator_inputs()[1])

        # The next run needs a new action and a new approval; the graph is paused again.
        payload = hitl.pending_interrupt(self.graph, self.config)
        self.assertNotEqual(payload["action"]["action_id"], first_id)
        self.assertEqual(len(self.tool.calls), 1)

        # Replaying the old approval for the new action is a refusal, not an approval.
        resume(self.graph, hitl.approval(first_id, True, "Yes, proceed."), self.config)
        self.assertEqual(len(self.tool.calls), 1)
        self.assertEqual(self.action()["status"], "refused")
        self.assertIn("not the pending", self.action()["refusal_reason"])

        self.assertEqual(self.graph.get_state(self.config).next, ())  # remaining decisions ran

        # Re-entering the executor with the consumed first approval does not run the tool.
        first_executed = {**self.action(), "action_id": first_id, "status": "executed"}
        update = self.system.executor({
            "dialogue": self.dialogue()[:4],  # user, orchestrator, prepared, human "Yes, proceed."
            "pending_action": first_executed,
        })
        self.assertTrue(update["dialogue"][0]["text"].startswith("ACTION_REJECTED"))
        self.assertEqual(len(self.tool.calls), 1)

    def test_8_unapproved_executor_route_cannot_invoke_tool(self):
        direct = json.dumps({"next_role": "executor", "instruction": "Run the script now.",
                             "question": "", "reason": "mocked unsafe decision"})
        self.make_graph([direct, decision("human", question=QUESTION)], executor_replies=[generated_code(1)])
        start(self.graph, "Run the Silverbox fitting script.", self.config)
        # The orchestrator transition was rejected with an explanation sent back to the LLM.
        self.assertIn("executor is forbidden without human approval", self.llm.orchestrator_inputs()[1])
        self.assertEqual(self.llm.count("executor"), 0)
        self.assertEqual(self.tool.calls, [])

        # Defence in depth: the executor node itself refuses an unapproved entry.
        unsafe_state = {
            "dialogue": [{"role": "user", "text": "Run it."}, {"role": "orchestrator", "text": direct}],
        }
        update = self.system.executor(unsafe_state)
        self.assertTrue(update["dialogue"][0]["text"].startswith("ACTION_REJECTED (NOT EXECUTED)"))
        self.assertIn("no such approval exists", update["dialogue"][0]["text"])
        self.assertEqual(self.tool.calls, [])

        # A refused action entering the executor is rejected too.
        refused = {"action_id": "action-01-x", "status": "refused", "operation": "op", "code": "", "code_sha256": ""}
        update = self.system.executor({
            "dialogue": unsafe_state["dialogue"] + [{"role": "human", "text": "No."}],
            "pending_action": refused,
        })
        self.assertTrue(update["dialogue"][0]["text"].startswith("ACTION_REJECTED"))
        self.assertEqual(self.tool.calls, [])

    def test_8_free_text_reply_is_not_approval(self):
        self.make_graph([approval_request()] + [decision("general")] * 7, executor_replies=[generated_code(1)])
        start(self.graph, "Run the Silverbox fitting script.", self.config)
        resume(self.graph, "Yes, proceed.", self.config)  # not a structured approval
        self.assertEqual(self.tool.calls, [])
        self.assertEqual(self.action()["status"], "refused")
        self.assertEqual([t["text"] for t in self.dialogue() if t["role"] == "human"], ["Yes, proceed."])

    def test_9_approval_at_eighth_decision_runs_before_end(self):
        self.make_graph([decision("general")] * 7 + [approval_request()], executor_replies=[generated_code(1)])
        start(self.graph, "Run the Silverbox fitting script.", self.config)
        self.assertEqual(count_orchestrator(self.dialogue()), 8)
        action_id = hitl.pending_interrupt(self.graph, self.config)["action"]["action_id"]

        state = resume(self.graph, hitl.approval(action_id, True, "Yes, proceed."), self.config)
        self.assertEqual(len(self.tool.calls), 1)
        roles = [t["role"] for t in state["dialogue"]]
        self.assertEqual(roles[-4:], ["orchestrator", "executor", "human", "executor"])
        self.assertTrue(state["dialogue"][-1]["text"].startswith(f"action_id={action_id}"))
        self.assertEqual(count_orchestrator(state["dialogue"]), 8)
        self.assertEqual(self.llm.count("orchestrator"), 8)
        self.assertEqual(self.graph.get_state(self.config).next, ())

    def test_9_refusal_at_eighth_decision_ends_without_execution(self):
        self.make_graph([decision("general")] * 7 + [approval_request()], executor_replies=[generated_code(1)])
        start(self.graph, "Run the Silverbox fitting script.", self.config)
        action_id = self.action()["action_id"]
        state = resume(self.graph, hitl.approval(action_id, False, "No, do not run it."), self.config)
        self.assertEqual(self.tool.calls, [])
        self.assertEqual(state["dialogue"][-1], {"role": "human", "text": "No, do not run it."})
        self.assertEqual(self.graph.get_state(self.config).next, ())
        self.assertEqual(self.llm.count("orchestrator"), 8)


TASK5_RMSE = "0.0015988781"  # the hand-written Task 5 result; never a Task 6 agent measurement


class TestFinalApproval(HitlTestCase):
    """Chunk 1C: ending the dialogue with a measurement needs human approval."""

    def run_to_final_request(self, after_execution: list[str], executor_replies=None):
        """Decision 1 requests code approval; the code runs; later decisions follow the script."""
        self.make_graph([approval_request()] + after_execution,
                        executor_replies=executor_replies or [generated_code(1)])
        start(self.graph, "Fit the Silverbox model and report the Task 6 Test RMSE.", self.config)
        self.code_action_id = self.action()["action_id"]
        resume(self.graph, hitl.approval(self.code_action_id, True, "Yes, proceed."), self.config)
        self.assertEqual(len(self.tool.calls), 1)
        return hitl.pending_interrupt(self.graph, self.config)

    def final(self) -> dict | None:
        return self.graph.get_state(self.config).values.get("pending_final")

    def report(self) -> dict | None:
        return self.graph.get_state(self.config).values.get("final_report")

    def test_1_finish_with_measurement_requires_approval(self):
        payload = self.run_to_final_request([final_request()])
        self.assertEqual(payload["kind"], "final_approval")
        action = payload["action"]
        self.assertEqual(action["action_type"], "report_final_measurement")
        self.assertEqual(action["target_role"], "finish")
        self.assertEqual(action["status"], "awaiting_approval")
        self.assertTrue(action["action_id"].startswith("final-01-"))
        # The question names the measurement, value and units, the run and the final action.
        question = payload["question"]
        self.assertIn("May I end the dialogue and report the Task 6 Test RMSE?", question)
        self.assertIn("TEST_RMSE = 0.015 V", question)
        self.assertIn("silverbox_agent_attempt_01", question)
        self.assertIn(self.code_action_id, question)
        self.assertIn("end the dialogue and report exactly this measurement", question)
        self.assertIsNone(self.report())
        self.assertEqual(self.graph.get_state(self.config).next, ("human",))

    def test_1_direct_finish_fails_closed(self):
        self.run_to_final_request([decision("finish"), final_request()])
        self.assertIn("finish is forbidden without human approval", self.llm.orchestrator_inputs()[2])
        self.assertIsNone(self.report())

    def test_2_9_approval_finalizes_exact_result_without_further_calls(self):
        payload = self.run_to_final_request([final_request()])
        calls_before = (self.llm.count("orchestrator"), self.llm.count("executor"), len(self.tool.calls))
        final_id = payload["action"]["action_id"]

        state = resume(self.graph, hitl.approval(final_id, True, "Approved, report it."), self.config)
        report = state["final_report"]
        self.assertEqual(report["quantity"], "TEST_RMSE")
        self.assertEqual(report["value"], 0.015)
        self.assertEqual(report["units"], "V")
        self.assertEqual(report["train_rmse"], 0.012)
        self.assertEqual(report["run_file"], self.system.runs[0]["relpath"])
        self.assertEqual(report["execution_action_id"], self.code_action_id)
        self.assertEqual(report["code_sha256"], hitl.code_sha256(generated_code(1)))
        self.assertEqual(report["approved_action_id"], final_id)
        self.assertEqual(state["pending_final"]["status"], "finalized")
        self.assertEqual(state["dialogue"][-1], {"role": "human", "text": "Approved, report it."})
        self.assertEqual(self.graph.get_state(self.config).next, ())
        # 9. no LLM call, code generation or tool execution after the approval.
        self.assertEqual(
            (self.llm.count("orchestrator"), self.llm.count("executor"), len(self.tool.calls)),
            calls_before,
        )
        # The approved report is written once, with exclusive creation.
        path = hitl.write_final_report(self.tmp, report)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), report)

    def test_3_refusal_prevents_final_report(self):
        payload = self.run_to_final_request([final_request()] + [decision("general")] * 6)
        final_id = payload["action"]["action_id"]
        resume(self.graph, hitl.approval(final_id, False, "No, do not report that yet."), self.config)
        self.assertIsNone(self.report())
        self.assertEqual(self.final()["status"], "refused")
        # Back to the orchestrator, which sees the refusal.
        later = self.llm.orchestrator_inputs()[2]
        self.assertIn("[HUMAN]\nNo, do not report that yet.", later)
        self.assertIn(f"action_id={final_id} status=refused", later)
        self.assertEqual(self.graph.get_state(self.config).next, ())  # budget used up, ended
        self.assertIsNone(self.report())

    def test_4_wrong_action_id_is_rejected(self):
        payload = self.run_to_final_request([final_request()] + [decision("general")] * 6)
        resume(self.graph, hitl.approval("final-99-000000000000", True, "Yes."), self.config)
        self.assertIsNone(self.report())
        self.assertEqual(self.final()["status"], "refused")
        self.assertIn("not the pending", self.final()["refusal_reason"])
        self.assertNotEqual(payload["action"]["action_id"], "final-99-000000000000")

    def test_4_coding_approval_cannot_finalize(self):
        self.run_to_final_request([final_request()] + [decision("general")] * 6)
        resume(self.graph, hitl.approval(self.code_action_id, True, "Yes."), self.config)
        self.assertIsNone(self.report())
        self.assertEqual(self.final()["status"], "refused")

    def test_5_old_approval_cannot_be_reused(self):
        payload = self.run_to_final_request(
            [final_request(), final_request("Report it now?")] + [decision("general")] * 5
        )
        first_id = payload["action"]["action_id"]
        resume(self.graph, hitl.approval(first_id, False, "Not yet."), self.config)
        second = hitl.pending_interrupt(self.graph, self.config)["action"]
        self.assertTrue(second["action_id"].startswith("final-02-"))
        self.assertNotEqual(second["action_id"], first_id)

        # The first action's id does not approve the second action.
        resume(self.graph, hitl.approval(first_id, True, "Yes."), self.config)
        self.assertIsNone(self.report())
        self.assertEqual(self.final()["status"], "refused")

    def test_5_finalized_approval_is_consumed(self):
        payload = self.run_to_final_request([final_request()])
        final_id = payload["action"]["action_id"]
        resume(self.graph, hitl.approval(final_id, True, "Yes."), self.config)
        self.assertEqual(self.final()["status"], "finalized")
        # Re-entering the human node with the consumed action does not finalize again.
        state = self.graph.get_state(self.config).values
        update = self.system.human(state)
        self.assertNotIn("final_report", update)
        self.assertEqual(update["dialogue"][0]["text"], "(no final measurement awaiting approval)")

    def test_6_decision_8_approval_finalizes(self):
        payload = self.run_to_final_request([decision("general")] * 6 + [final_request()])
        self.assertEqual(count_orchestrator(self.dialogue()), 8)
        state = resume(self.graph, hitl.approval(payload["action"]["action_id"], True, "Yes."), self.config)
        self.assertEqual(state["final_report"]["value"], 0.015)
        self.assertEqual(self.graph.get_state(self.config).next, ())
        self.assertEqual(self.llm.count("orchestrator"), 8)

    def test_6_decision_8_refusal_ends_without_report(self):
        payload = self.run_to_final_request([decision("general")] * 6 + [final_request()])
        resume(self.graph, hitl.approval(payload["action"]["action_id"], False, "No."), self.config)
        self.assertIsNone(self.report())
        self.assertEqual(self.graph.get_state(self.config).next, ())
        self.assertEqual(self.llm.count("orchestrator"), 8)

    def test_7_no_fabricated_rmse_in_final_report(self):
        fabricated = json.dumps({
            "next_role": "human", "request": "final_approval", "instruction": "",
            "question": f"Report TEST_RMSE = {TASK5_RMSE} V?",
            "reason": f"the best result is {TASK5_RMSE} V",
            "measurement": {"value": float(TASK5_RMSE)},  # ignored: not part of the decision schema
        })
        payload = self.run_to_final_request([fabricated])
        self.assertEqual(payload["action"]["measurement"]["value"], 0.015)
        state = resume(self.graph, hitl.approval(payload["action"]["action_id"], True, "Yes."), self.config)
        self.assertEqual(state["final_report"]["value"], 0.015)
        self.assertNotIn(TASK5_RMSE, json.dumps(state["final_report"]))

    def test_7_final_approval_needs_a_real_result(self):
        self.make_graph([final_request(), decision("human", question=QUESTION)])
        start(self.graph, TASK, self.config)
        self.assertIn("no executor turn has exit_code=0", self.llm.orchestrator_inputs()[1])
        self.assertIsNone(self.final())

    def test_10_quality_gate_still_applies(self):
        self.make_graph([approval_request(), final_request(), decision("human", question=QUESTION)],
                        executor_replies=[generated_code(1)])
        self.tool.test_rmse = 0.06
        start(self.graph, TASK, self.config)
        resume(self.graph, hitl.approval(self.action()["action_id"], True, "Yes."), self.config)
        self.assertIn("forbidden by the quality gate", self.llm.orchestrator_inputs()[2])
        self.assertIsNone(self.final())


class TestOverwriteSafety(HitlTestCase):
    def test_8_existing_run_directory_is_refused(self):
        for evidence in ("validated_run", "validated_run_v2"):
            with self.assertRaises(hitl.OverwriteRefused):
                hitl.ensure_fresh_workdir(REPO / "task6" / evidence)
        occupied = self.tmp / "previous_run"
        occupied.mkdir()
        (occupied / "dialogue.json").write_text("[]", encoding="utf-8")
        with self.assertRaises(hitl.OverwriteRefused):
            hitl.ensure_fresh_workdir(occupied)
        self.assertEqual((occupied / "dialogue.json").read_text(encoding="utf-8"), "[]")
        # A new or empty directory is fine.
        hitl.ensure_fresh_workdir(self.tmp / "new_run")
        (self.tmp / "empty").mkdir()
        hitl.ensure_fresh_workdir(self.tmp / "empty")

    def test_8_final_report_is_never_overwritten(self):
        existing = self.tmp / hitl.FINAL_REPORT_FILE
        existing.write_text('{"value": 0.0595}', encoding="utf-8")
        with self.assertRaises(hitl.OverwriteRefused):
            hitl.write_final_report(self.tmp, {"value": 0.015})
        self.assertEqual(existing.read_text(encoding="utf-8"), '{"value": 0.0595}')

    def test_8_coding_tool_never_overwrites_an_attempt(self):
        workdir = self.tmp / "run"
        old = workdir / "silverbox_agent_attempt_01"
        old.mkdir(parents=True)
        (old / "silverbox_agent_attempt_01.py").write_text("# earlier attempt\n", encoding="utf-8")
        tool = hitl.PythonCodingTool(workdir=workdir, data_path=self.tmp / "SNLS80mV.mat")
        done = subprocess.CompletedProcess(args=[], returncode=0, stdout="TEST_RMSE=0.1\n", stderr="")
        with mock.patch.object(hitl.hw1.subprocess, "run", return_value=done) as fake_run:
            result = tool.write_and_run("silverbox_agent_attempt_01.py", "# new attempt\n")
        fake_run.assert_called_once()  # no real subprocess
        self.assertEqual(result["relpath"], "silverbox_agent_attempt_01_1/silverbox_agent_attempt_01_1.py")
        self.assertEqual((old / "silverbox_agent_attempt_01.py").read_text(encoding="utf-8"), "# earlier attempt\n")


class TestDemoRunner(HitlTestCase):
    """Chunk 1D-A: the bounded ambiguity demonstration, with a mocked model."""

    def run_demo(self, replies: list[str], workdir_name: str = "demo"):
        from hw2 import demo_task1
        self.llm = ScriptedLLM(replies)
        self.asked: list[dict] = []
        workdir = self.tmp / workdir_name

        def reply(payload: dict) -> str:
            self.asked.append(payload)
            return REPLY

        evidence = demo_task1.run_demo(self.llm, TASK, workdir, "demo-thread", "mock-model", reply)
        return evidence, workdir

    def test_clarification_round_trip_stops_after_next_decision(self):
        evidence, workdir = self.run_demo([
            decision("human", question=QUESTION),
            decision("general", instruction="Explain where the Task 6 Test RMSE is recorded."),
        ])
        self.assertEqual(evidence["outcome"], "human_clarification_resumed")
        self.assertEqual(evidence["thread_id"], "demo-thread")
        self.assertEqual(evidence["first_decision"]["next_role"], "human")
        self.assertEqual(evidence["interrupt"]["payload"]["question"], QUESTION)
        self.assertEqual(evidence["interrupt"]["graph_next_while_paused"], ["human"])
        self.assertEqual([p["question"] for p in self.asked], [QUESTION])
        self.assertEqual(evidence["human_reply"], REPLY)
        self.assertEqual(evidence["human_turns_with_reply"], 1)
        self.assertTrue(evidence["clarification_visible_to_resumed_orchestrator"])
        self.assertEqual(evidence["post_resume_message"]["role"], "orchestrator")
        self.assertEqual(evidence["post_resume_decision"]["next_role"], "general")
        # Stopped before the chosen node: the general agent never ran, no code ran.
        self.assertEqual(evidence["stopped_before_node_not_executed"], ["general"])
        self.assertEqual(self.llm.count("general"), 0)
        self.assertEqual(evidence["api_call_count"], 2)
        self.assertEqual(evidence["coding_tool_calls"], 0)
        roles = [t["role"] for t in evidence["dialogue_final"]]
        self.assertEqual(roles, ["user", "orchestrator", "human", "orchestrator"])
        for name in ("dialogue.json", "dialogue.txt", "demo_evidence.json"):
            self.assertTrue((workdir / name).exists(), name)
        saved = json.loads((workdir / "demo_evidence.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["post_resume_decision"], evidence["post_resume_decision"])

    def test_approval_request_after_resume_does_not_prepare_code(self):
        evidence, _ = self.run_demo([decision("human", question=QUESTION), approval_request()])
        self.assertEqual(evidence["post_resume_decision"]["request"], "approval")
        self.assertEqual(evidence["stopped_before_node_not_executed"], ["executor"])
        self.assertEqual(self.llm.count("executor"), 0)  # no code generation
        self.assertEqual(evidence["coding_tool_calls"], 0)

    def test_non_human_decision_is_recorded_honestly(self):
        evidence, _ = self.run_demo([decision("general", instruction="Answer from the record.")])
        self.assertEqual(evidence["outcome"], "llm_did_not_choose_human_clarification")
        self.assertEqual(evidence["first_decision"]["next_role"], "general")
        self.assertNotIn("interrupt", evidence)
        self.assertNotIn("human_reply", evidence)
        self.assertEqual(self.asked, [])
        self.assertEqual(evidence["stopped_before_node_not_executed"], ["general"])
        self.assertEqual(self.llm.count("general"), 0)
        self.assertEqual(evidence["api_call_count"], 1)

    def test_demo_refuses_existing_directory(self):
        from hw2 import demo_task1
        occupied = self.tmp / "old_demo"
        occupied.mkdir()
        (occupied / "demo_evidence.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(hitl.OverwriteRefused):
            demo_task1.run_demo(ScriptedLLM([]), TASK, occupied, "t", "m", lambda p: REPLY)
        self.assertEqual((occupied / "demo_evidence.json").read_text(encoding="utf-8"), "{}")


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def fake_history_tree(root: Path, t5=0.111, run1=0.222, run2=0.333) -> Path:
    """A repository-shaped tree with the same JSON structure as the real evidence files."""
    write_json(root / "task5/task5_results.json",
               {"runs": [{}, {}], "best": {"name": "cfg_x", "train_rmse_V": 0.101, "test_rmse_V": t5}})
    write_json(root / "task6/validated_run/run_summary.json",
               {"model": "m1", "final_script": "a.py", "final_exit_code": 0,
                "TRAIN_RMSE": 0.201, "TEST_RMSE": run1, "verification": {"passed": True}})
    write_json(root / "task6/validated_run/verification.json", {"passed": True})
    write_json(root / "task6/validated_run_v2/run_summary.json",
               {"model": "m2", "final_script": "b.py", "final_exit_code": 0,
                "TRAIN_RMSE": 0.301, "TEST_RMSE": run2, "verification": {"passed": False}})
    return root


class TestHistoricalContext(HitlTestCase):
    """Chunk 1D-A fix: read-only recorded results as orchestrator context."""

    def test_1_values_are_loaded_from_files(self):
        root = fake_history_tree(self.tmp / "repo")
        history = hitl.load_historical_results(root)
        self.assertEqual([r["test_rmse"] for r in history["records"]], [0.111, 0.222, 0.333])
        self.assertEqual([r["train_rmse"] for r in history["records"]], [0.101, 0.201, 0.301])
        # Changing a file changes the loaded value: nothing is hard-coded.
        fake_history_tree(root, run1=0.444)
        self.assertEqual(hitl.load_historical_results(root)["records"][1]["test_rmse"], 0.444)

        # The real repository files, compared with the files themselves.
        real = {(r["task"], r["run_id"]): r for r in hitl.load_historical_results()["records"]}
        t5 = json.loads((REPO / "task5/task5_results.json").read_text(encoding="utf-8"))["best"]
        self.assertEqual(real[("task5", "hand_written_solution")]["test_rmse"], t5["test_rmse_V"])
        self.assertEqual(real[("task5", "hand_written_solution")]["train_rmse"], t5["train_rmse_V"])
        for run_id, folder in (("agent_run_1", "validated_run"), ("agent_run_2", "validated_run_v2")):
            summary = json.loads((REPO / "task6" / folder / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(real[("task6", run_id)]["test_rmse"], summary["TEST_RMSE"])
            self.assertEqual(real[("task6", run_id)]["train_rmse"], summary["TRAIN_RMSE"])

    def test_2_task5_and_task6_stay_distinguishable(self):
        history = hitl.load_historical_results()
        self.assertEqual(
            [(r["task"], r["run_id"]) for r in history["records"]],
            [("task5", "hand_written_solution"), ("task6", "agent_run_1"), ("task6", "agent_run_2")],
        )
        task5, run1, run2 = history["records"]
        self.assertIn("hand-written", task5["provenance"])
        self.assertIn("NOT produced by the Task 6 agent system", task5["provenance"])
        self.assertEqual(task5["verification"], "no independent verifier record")
        for record, n in ((run1, 1), (run2, 2)):
            self.assertIn(f"agent-generated result, run {n}", record["provenance"])
            self.assertIn("PASSED", record["verification"])
        self.assertEqual({r["units"] for r in history["records"]}, {"V"})
        self.assertEqual(
            history["sources_read"],
            ["task5/task5_results.json", "task6/validated_run/run_summary.json",
             "task6/validated_run_v2/run_summary.json"],
        )
        self.assertEqual(history["problems"], [])
        text = hitl.format_history(history)
        for tag in ("[task5 / hand_written_solution]", "[task6 / agent_run_1]", "[task6 / agent_run_2]"):
            self.assertIn(tag, text)

    def test_3_missing_and_malformed_files_do_not_crash(self):
        empty = hitl.load_historical_results(self.tmp / "nothing_here")
        self.assertEqual(empty["records"], [])
        self.assertEqual(len(empty["problems"]), 3)
        self.assertTrue(all(p.endswith("missing") for p in empty["problems"]))
        self.assertIn("(no recorded results could be read)", hitl.format_history(empty))

        root = fake_history_tree(self.tmp / "repo")
        (root / "task5/task5_results.json").write_text("{not json", encoding="utf-8")
        write_json(root / "task6/validated_run_v2/run_summary.json", {"TRAIN_RMSE": 0.3})  # no TEST_RMSE
        broken = hitl.load_historical_results(root)
        self.assertEqual([r["run_id"] for r in broken["records"]], ["agent_run_1"])
        self.assertEqual(len(broken["problems"]), 2)
        self.assertTrue(all("unusable" in p for p in broken["problems"]))

        # The graph runs with an empty history.
        self.make_graph([decision("human", question=QUESTION)], history=empty)
        start(self.graph, TASK, self.config)
        self.assertEqual(pending_question(self.graph, self.config), QUESTION)
        self.assertIn("(no recorded results could be read)", self.llm.orchestrator_inputs()[0])

    def test_4_orchestrator_receives_history_only_when_enabled(self):
        history = hitl.load_historical_results()
        self.make_graph([decision("human", question=QUESTION)], history=history)
        start(self.graph, TASK, self.config)
        received = self.llm.orchestrator_inputs()[0]
        self.assertIn("RECORDED RESULTS FROM EARLIER WORK", received)
        self.assertIn("(Read-only historical evidence, not produced during the current run.", received)
        self.assertIn(hitl.format_history(history), received)
        for record in history["records"]:
            self.assertIn(f"TEST_RMSE = {record['test_rmse']} V", received)
        # Both facts are present and labelled: no attempts in this run, history on disk.
        self.assertIn("EXECUTOR ATTEMPTS IN THIS RUN:\n(no executor attempts yet)", received)
        self.assertNotIn("RECORDED RESULTS", hitl.format_dialogue(self.dialogue()))  # not in the dialogue

        self.make_graph([decision("human", question=QUESTION)], thread_id="no-history")
        start(self.graph, TASK, self.config)
        self.assertNotIn("RECORDED RESULTS FROM EARLIER WORK", self.llm.orchestrator_inputs()[0])

    def test_5_quality_target_applies_to_new_fitting_only(self):
        self.make_graph([decision("human", question=QUESTION)], history=hitl.load_historical_results())
        start(self.graph, TASK, self.config)
        _, instructions, received = self.llm.calls[0]
        self.assertIn("Quality target (applies only to new Silverbox model fitting or optimization "
                      "in this run): TEST_RMSE < 0.02 V.", received)
        self.assertIn("QUALITY GATE (new fitting work in this run only)", instructions)
        self.assertIn("the quality target and the QUALITY GATE below apply\n  ONLY to (a)", instructions)
        self.assertIn("FIRST, UNDERSTAND THE REQUEST", instructions)
        self.assertIn("AMBIGUITY:", instructions)
        self.assertIn("do not ask merely to confirm", instructions)

    def test_6_history_never_satisfies_finish_gate(self):
        history = hitl.load_historical_results()
        self.make_graph([final_request(), decision("finish"), decision("human", question=QUESTION)],
                        history=history)
        start(self.graph, TASK, self.config)
        inputs = self.llm.orchestrator_inputs()
        self.assertIn("finish is forbidden: no executor turn has exit_code=0", inputs[1])
        self.assertIn("finish is forbidden: no executor turn has exit_code=0", inputs[2])
        self.assertIn("Best verified executor TEST_RMSE in this run so far: None", inputs[0])
        self.assertIn("finish currently allowed by the program: False", inputs[0])
        state = self.graph.get_state(self.config).values
        self.assertIsNone(state.get("pending_final"))
        self.assertIsNone(state.get("final_report"))
        self.assertEqual(hitl.executed_results(state["dialogue"]), [])

    def test_7_approval_safeguards_unchanged_with_history(self):
        self.make_graph([approval_request()] + [decision("general")] * 7,
                        executor_replies=[generated_code(1)], history=hitl.load_historical_results())
        start(self.graph, "Run the Silverbox fitting script.", self.config)
        payload = hitl.pending_interrupt(self.graph, self.config)
        self.assertEqual(payload["kind"], "approval")
        self.assertEqual(self.tool.calls, [])
        resume(self.graph, payload["action"]["action_id"], self.config)  # not a structured approval
        self.assertEqual(self.tool.calls, [])
        self.assertEqual(self.action()["status"], "refused")

    def test_8_demo_evidence_records_history_sources(self):
        from hw2 import demo_task1
        history = hitl.load_historical_results()
        llm = ScriptedLLM([decision("human", question=QUESTION), decision("general")])
        evidence = demo_task1.run_demo(llm, TASK, self.tmp / "demo", "t", "mock-model",
                                       lambda payload: REPLY, history)
        ctx = evidence["historical_context"]
        self.assertTrue(ctx["supplied"])
        self.assertEqual(ctx["sources_read"], history["sources_read"])
        self.assertEqual(ctx["records"], history["records"])
        self.assertEqual(ctx["context_text_passed_to_orchestrator"], hitl.format_history(history))
        for call in evidence["llm_calls"]:
            self.assertIn(ctx["context_text_passed_to_orchestrator"], call["input_text"])
        saved = json.loads((self.tmp / "demo" / "demo_evidence.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["historical_context"]["sources_read"], history["sources_read"])

        plain = demo_task1.run_demo(ScriptedLLM([decision("general")]), TASK, self.tmp / "demo2",
                                    "t2", "mock-model", lambda payload: REPLY)
        self.assertFalse(plain["historical_context"]["supplied"])
        self.assertIsNone(plain["historical_context"]["context_text_passed_to_orchestrator"])


def refuse(payload: dict) -> dict:
    return hitl.approval(payload["action"]["action_id"], False, "No, do not run it.")


def approve(payload: dict) -> dict:
    return hitl.approval(payload["action"]["action_id"], True, "Yes, proceed.")


class TestHardDecisionDemo(HitlTestCase):
    """Chunk 1D-B: bounded refusal/approval demonstration, mocked LLM and fake tool."""

    def run_hard(self, orchestrator_replies, executor_replies, answers, workdir_name="hard", **kwargs):
        from hw2 import demo_task1_hard
        self.llm = ScriptedLLM(orchestrator_replies, executor_replies)
        self.tool = FakeCodingTool(self.tmp / "tool")
        self.payloads: list[dict] = []
        pending = list(answers)

        def decide(payload: dict):
            self.payloads.append(payload)
            return pending.pop(0)(payload)

        self.workdir = self.tmp / workdir_name
        return demo_task1_hard.run_hard_demo(
            self.llm, "Compute the zero-predictor Test RMSE.", self.workdir, "hard-thread",
            "mock-model", self.tool, decide, **kwargs,
        )

    def standard_run(self):
        return self.run_hard(
            [approval_request(), approval_request("I prepared a new script. May the executor run it?")],
            [generated_code(1), generated_code(2)],
            [refuse, approve],
        )

    def test_refusal_then_approval_sequence(self):
        evidence = self.standard_run()
        first, second = (p["action"] for p in self.payloads)

        # 1. the first action is prepared, with its exact code shown before approval.
        self.assertEqual(first["code"], generated_code(1))
        self.assertTrue(first["action_id"].startswith("action-01-"))
        self.assertEqual(evidence["actions"][first["action_id"]]["status_history"],
                         ["awaiting_approval", "refused"])

        # 2. refusal prevents execution.
        proof = evidence["refusal_proof"]
        self.assertEqual(proof["refused_action_ids"], [first["action_id"]])
        self.assertEqual(proof["tool_calls_before_first_approval"], 0)
        self.assertEqual(proof["tool_calls_running_refused_code"], 0)

        # 3. the refusal is in the dialogue, exactly once.
        humans = [t["text"] for t in evidence["dialogue_final"] if t["role"] == "human"]
        self.assertEqual(humans, ["No, do not run it.", "Yes, proceed."])

        # 4. the orchestrator received the refusal.
        self.assertEqual(proof["orchestrator_saw_refusal"], [True])
        self.assertIn("[HUMAN]\nNo, do not run it.", self.llm.orchestrator_inputs()[1])

        # 5. a fresh approval request for a new action.
        self.assertTrue(second["action_id"].startswith("action-02-"))
        self.assertNotEqual(second["action_id"], first["action_id"])
        self.assertEqual(second["code"], generated_code(2))

        # 6./7. the approval applies to the new action, which runs exactly once.
        approval_proof = evidence["approval_proof"]
        self.assertEqual(approval_proof["approved_action_ids"], [second["action_id"]])
        self.assertEqual(approval_proof["executed_action_ids"], [second["action_id"]])
        self.assertEqual(self.tool.calls, [("silverbox_agent_attempt_01.py", generated_code(2))])
        self.assertTrue(approval_proof["executed_code_matches_approved_hash"])
        self.assertEqual(evidence["actions"][second["action_id"]]["status_history"],
                         ["awaiting_approval", "approved", "executed"])

    def test_demo_stops_after_execution(self):
        evidence = self.standard_run()
        self.assertEqual(evidence["outcome"], "approved_action_executed")
        self.assertEqual(evidence["approval_proof"]["llm_calls_after_execution"], 0)
        # Only the two scripted orchestrator decisions ran; a third would raise in ScriptedLLM.
        self.assertEqual(self.llm.count("orchestrator"), 2)
        self.assertEqual(self.llm.count("executor"), 2)
        self.assertEqual(evidence["api_call_count"], 4)
        self.assertEqual(evidence["dialogue_final"][-1]["role"], "executor")
        self.assertTrue(evidence["dialogue_final"][-1]["text"].startswith("action_id="))

    def test_evidence_records_sequence(self):
        evidence = self.standard_run()
        sequence = [e["event"] for e in evidence["events"] if e["event"] not in ("human_reply", "user_request")]
        self.assertEqual(sequence, [
            "orchestrator_decision", "action_prepared", "interrupt", "action_refused",
            "orchestrator_decision", "action_prepared", "interrupt", "action_approved",
            "action_executed",
        ])
        self.assertEqual(len(evidence["tool_calls"]), 1)
        self.assertEqual(evidence["tool_calls"][0]["test_rmse"], 0.015)
        saved = json.loads((self.workdir / "demo_evidence.json").read_text(encoding="utf-8"))
        self.assertEqual([e["event"] for e in saved["events"]], [e["event"] for e in evidence["events"]])
        for name in ("dialogue.json", "dialogue.txt"):
            self.assertTrue((self.workdir / name).exists())

    def test_other_llm_behaviour_is_recorded_honestly(self):
        evidence = self.run_hard(
            [approval_request()] + [decision("general")] * 7, [generated_code(1)], [refuse],
        )
        self.assertEqual(evidence["outcome"], "llm_call_cap_reached")
        self.assertEqual(self.tool.calls, [])
        self.assertEqual(evidence["approval_proof"]["executed_action_ids"], [])
        self.assertEqual(evidence["api_call_count"], 12)

    def test_refuses_existing_directory(self):
        occupied = self.tmp / "hard"
        occupied.mkdir()
        (occupied / "demo_evidence.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(hitl.OverwriteRefused):
            self.standard_run()
        self.assertEqual(self.tool.calls, [])
        self.assertEqual((occupied / "demo_evidence.json").read_text(encoding="utf-8"), "{}")


ZERO_REQUEST = (
    "Create and run a Python script to calculate the zero-predictor Test RMSE using the "
    "Silverbox 1024-sample test split. Do not train any model. Use the existing dataset "
    "and print the actually computed TEST_RMSE."
)
# A mocked executor reply. It is only stored and hashed in these tests, never run.
ZERO_CODE = (
    "import os\nimport numpy as np\nfrom scipy.io import loadmat\n"
    "m = loadmat(os.environ['SILVERBOX_MAT_PATH'])\n"
    "y = np.asarray(m['V2'], dtype=float).reshape(-1)[42650:46746]\n"
    "y = y - y.mean()\n"
    "print(f'TEST_RMSE={np.sqrt(np.mean(y[3072:] ** 2))}')\n"
)
FAKE_KEY = "sk-test-not-a-real-key"


class EnvSnoopTool(FakeCodingTool):
    """Fake tool that records the environment a child process would inherit."""

    def __init__(self, workdir: Path):
        super().__init__(workdir)
        self.child_env: list[dict] = []

    def write_and_run(self, filename: str, code: str) -> dict:
        self.child_env.append(dict(os.environ))  # what PythonCodingTool copies for the child
        return super().write_and_run(filename, code)


class TestExecutorAlignmentAndIsolation(HitlTestCase):
    """Chunk 1D-B.1: the executor follows the user's request; generated code gets no secrets."""

    def prepare_zero_predictor(self, tool=None):
        self.make_graph(
            [approval_request(instruction="Compute the zero-predictor Test RMSE; do not train any model.")]
            + [decision("general")] * 7,
            executor_replies=[ZERO_CODE],
        )
        if tool is not None:
            self.system.coding_tool = self.tool = tool
        start(self.graph, ZERO_REQUEST, self.config)
        return hitl.pending_interrupt(self.graph, self.config)

    def executor_call(self) -> tuple[str, str]:
        return next((ins, inp) for role, ins, inp in self.llm.calls if role == "executor")

    def test_1_executor_instructions_follow_user_request(self):
        self.prepare_zero_predictor()
        instructions, input_text = self.executor_call()
        self.assertTrue(instructions.startswith(hitl.EXECUTOR_TASK_ALIGNMENT))
        self.assertIn("This section takes precedence over the HW1 instructions below", instructions)
        self.assertLess(instructions.index("TASK ALIGNMENT"),
                        instructions.index("You are the EXECUTOR in a four-role LangGraph system"))
        self.assertIn(f"USER TASK:\n{ZERO_REQUEST}", input_text)
        self.assertIn("ORCHESTRATOR INSTRUCTION:\nCompute the zero-predictor Test RMSE", input_text)
        # The wrapper is used only while preparing.
        self.assertIs(self.system.llm, self.llm)

    def test_2_zero_predictor_request_does_not_require_training(self):
        payload = self.prepare_zero_predictor()
        instructions, _ = self.executor_call()
        self.assertIn("For any other computation, do NOT define or train a model", instructions)
        self.assertIn("ONLY when the user asks for that\n  modeling, fitting or training work", instructions)
        self.assertIn("subtract the excerpt mean of u and of y", instructions)
        self.assertIn("never hard-code a result", instructions)
        # The prepared action is exactly the model's code (here a mocked non-training script).
        self.assertEqual(payload["action"]["code"], ZERO_CODE)
        self.assertNotIn("torch", payload["action"]["code"])

    def test_3_hw1_modeling_contract_still_available(self):
        self.prepare_zero_predictor()
        instructions, _ = self.executor_call()
        hw1_part = instructions[len(hitl.EXECUTOR_TASK_ALIGNMENT):]
        self.assertTrue(hw1_part.startswith("You are the EXECUTOR in a four-role LangGraph system"))
        for text in ("FREE-RUN CONTRACT", "def free_run(accel, u, y0, v0):",
                     "a_theta must be a fully-connected PyTorch network", "TEST_RMSE=<numeric volts>"):
            self.assertIn(text, hw1_part)

    def test_4_to_6_prepared_code_runs_unchanged_once_after_approval(self):
        payload = self.prepare_zero_predictor()
        action = payload["action"]
        self.assertEqual(self.tool.calls, [])
        resume(self.graph, hitl.approval(action["action_id"], True, "Yes, proceed."), self.config)
        self.assertEqual(self.tool.calls, [(action["filename"], ZERO_CODE)])
        self.assertEqual(hitl.code_sha256(self.tool.calls[0][1]), action["code_sha256"])

    def test_5_6_demo_refusal_then_approval_with_zero_predictor(self):
        from hw2 import demo_task1_hard
        self.assertEqual(demo_task1_hard.SCENARIO_REQUEST, ZERO_REQUEST)
        self.llm = ScriptedLLM([approval_request(), approval_request("New script prepared. Run it?")],
                               [ZERO_CODE, ZERO_CODE + "# second\n"])
        tool = FakeCodingTool(self.tmp / "tool")
        answers = [refuse, approve]
        evidence = demo_task1_hard.run_hard_demo(
            self.llm, ZERO_REQUEST, self.tmp / "hard", "t", "mock", tool,
            lambda payload: answers.pop(0)(payload),
        )
        self.assertEqual(evidence["refusal_proof"]["tool_calls_before_first_approval"], 0)
        self.assertEqual(tool.calls, [("silverbox_agent_attempt_01.py", ZERO_CODE + "# second\n")])
        self.assertTrue(evidence["approval_proof"]["executed_code_matches_approved_hash"])

    def test_7_child_execution_gets_no_api_key(self):
        secrets = {"OPENAI_API_KEY": FAKE_KEY, "SOME_SERVICE_TOKEN": "t0k", "MY_SECRET": "s3"}
        with mock.patch.dict(os.environ, secrets):
            payload = self.prepare_zero_predictor(tool=EnvSnoopTool(self.tmp / "snoop"))
            resume(self.graph, hitl.approval(payload["action"]["action_id"], True, "Yes."), self.config)
            child = self.tool.child_env[0]
            for name in secrets:
                self.assertNotIn(name, child)
            self.assertNotIn(FAKE_KEY, json.dumps(child))
            # The parent's environment is restored after the call.
            for name, value in secrets.items():
                self.assertEqual(os.environ[name], value)
            withheld = self.action()["outcome"]["env_vars_withheld_from_child"]
            self.assertTrue(set(secrets) <= set(withheld))
            # Names are recorded, values never.
            self.assertNotIn(FAKE_KEY, json.dumps(self.graph.get_state(self.config).values))

    def test_7_real_coding_tool_subprocess_env_has_no_api_key(self):
        workdir = self.tmp / "real_tool"
        tool = hitl.PythonCodingTool(workdir=workdir, data_path=self.tmp / "SNLS80mV.mat")
        done = subprocess.CompletedProcess(args=[], returncode=0, stdout="TEST_RMSE=0.06\n", stderr="")
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": FAKE_KEY}), \
                mock.patch.object(hitl.hw1.subprocess, "run", return_value=done) as fake_run:
            payload = self.prepare_zero_predictor(tool=tool)
            resume(self.graph, hitl.approval(payload["action"]["action_id"], True, "Yes."), self.config)
            self.assertEqual(os.environ["OPENAI_API_KEY"], FAKE_KEY)
        fake_run.assert_called_once()  # mocked: nothing was actually executed
        env = fake_run.call_args.kwargs["env"]
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertNotIn(FAKE_KEY, json.dumps(env))
        self.assertIn("SILVERBOX_MAT_PATH", env)
        self.assertIn("PATH", {k.upper() for k in env})
        self.assertEqual(self.action()["outcome"]["test_rmse"], 0.06)

    def test_secret_name_matching(self):
        for name in ("OPENAI_API_KEY", "OPENAI_ORG_ID", "GITHUB_TOKEN", "AWS_SECRET_ACCESS_KEY", "DB_PASSWORD"):
            self.assertTrue(hitl.SECRET_ENV_NAME.search(name), name)
        for name in ("PATH", "SYSTEMROOT", "PYTHONIOENCODING", "SILVERBOX_MAT_PATH", "TEMP", "MPLBACKEND"):
            self.assertFalse(hitl.SECRET_ENV_NAME.search(name), name)


class TestOriginalFilesUnchanged(unittest.TestCase):
    def test_i_original_hw1_files_unchanged(self):
        if shutil.which("git") is None:
            self.skipTest("git not available")
        protected = ["task5", "task6", "README.md", "requirements.txt"]
        out = subprocess.run(
            ["git", "status", "--porcelain", "--", *protected],
            cwd=REPO, capture_output=True, text=True, check=True,
        ).stdout
        self.assertEqual(out, "", f"original HW1 files changed:\n{out}")


if __name__ == "__main__":
    unittest.main()
