"""Offline tests for the HW2 Task 2 focusing experiment tool (small dimensions, temp directories)."""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hw2 import focusing_tool as ft  # noqa: E402
from hw2.focusing_tool import (  # noqa: E402
    ExperimentArguments,
    FocusingSession,
    FocusingToolError,
    validated_intensity_metrics,
)
from hw2.simulated_bench import Region, region_metrics  # noqa: E402

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
NOISY = {"enabled": True, "photons_per_unit": 300.0, "read_noise_std": 0.02}
NO_NOISE = {"enabled": False}


def arguments(**overrides) -> dict:
    base = {
        "grouping_factor": 2, "focus_row": 7, "focus_col": 7, "focus_height": 2, "focus_width": 2,
        "background_gap": 1, "background_width": 2, "iterations": 30, "flips_per_proposal": 1,
        "averaging": 1, "validation_count": 3, "seed": 0, "max_acquisitions": 100,
    }
    base.update(overrides)
    return base


class ToolTestCase(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="hw2_focusing_tool_"))
        self.addCleanup(shutil.rmtree, self.root, True)

    def session(self, noise=NOISY, medium_seed=1, root=None, **kwargs) -> FocusingSession:
        return FocusingSession((8, 8), (16, 16), medium_seed=medium_seed, noise=noise,
                               artifact_root=root or self.root, **kwargs)

    @staticmethod
    def history(result) -> dict:
        with np.load(result["artifacts"]["history_npz"]) as data:
            return {k: data[k] for k in data.files}


class TestInterface(ToolTestCase):
    def test_1_json_input_and_output(self):
        s = self.session()
        result = s.call(json.loads(json.dumps(arguments())))
        self.assertTrue(result["ok"])
        self.assertEqual(json.loads(json.dumps(result)), result)            # fully JSON-compatible
        saved = json.loads(Path(result["artifacts"]["result"]).read_text(encoding="utf-8"))
        self.assertEqual(saved, result)
        settings = json.loads(Path(result["artifacts"]["settings"]).read_text(encoding="utf-8"))
        self.assertEqual(settings["tool_arguments"], arguments())
        schema = s.describe()["argument_schema"]
        json.dumps(s.describe())
        self.assertEqual(sorted(schema["required"]), sorted(arguments()))
        self.assertFalse(schema["additionalProperties"])
        # The tool chooses nothing: every core setting is required.
        for name in arguments():
            incomplete = arguments()
            del incomplete[name]
            self.assertFalse(s.call(incomplete)["ok"], name)

    def test_2_parameter_validation(self):
        s = self.session()
        bad = [
            {"unexpected": 1}, dict(grouping_factor=3), dict(grouping_factor=0), dict(grouping_factor=True),
            dict(iterations=0), dict(iterations=-5), dict(iterations=2.0), dict(seed=-1), dict(seed="1"),
            dict(averaging=0), dict(validation_count=0), dict(flips_per_proposal=17),
            dict(focus_row=15), dict(focus_col=-1), dict(focus_height=0),
            dict(focus_row=0, focus_col=0, focus_height=16, focus_width=16),        # no room for background
            dict(seed=float("nan")), dict(max_acquisitions=0),
        ]
        for overrides in bad:
            result = s.call(arguments(**overrides))
            self.assertFalse(result["ok"], overrides)
            self.assertIn("error", result)
        self.assertEqual(s.describe()["camera_acquisitions_so_far"], 0)       # nothing was measured
        self.assertEqual(list(self.root.iterdir()), [])                         # no run directory created
        with self.assertRaises(FocusingToolError):
            s.run_experiment(arguments(iterations=0))
        self.assertFalse(s.call("not a dict")["ok"])
        for bad_session in (dict(noise={"enabled": 1}), dict(noise={"bogus": 1}), dict(medium_seed=-1)):
            with self.assertRaises(FocusingToolError):
                self.session(**bad_session)
        with self.assertRaises(FocusingToolError):
            FocusingSession((64, 64), (128, 128), artifact_root=self.root)     # medium too large

    def test_12_budget_violations_rejected_before_measuring(self):
        s = self.session(acquisition_limit=500)
        over_call = s.call(arguments(iterations=200, validation_count=3, max_acquisitions=150))
        self.assertFalse(over_call["ok"])
        self.assertIn("need 207 camera acquisitions", over_call["error"])
        over_session = s.call(arguments(max_acquisitions=501))
        self.assertFalse(over_session["ok"])
        self.assertIn("session limit 500", over_session["error"])
        self.assertEqual(s.describe()["camera_acquisitions_so_far"], 0)
        exact = s.call(arguments(iterations=93, validation_count=3, max_acquisitions=100))  # 1 + 93 + 6
        self.assertTrue(exact["ok"])
        self.assertEqual(exact["acquisitions"]["counted_by_bench"], 100)


class TestFixedMediumAndNoise(ToolTestCase):
    def test_3_same_T_across_calls(self):
        s = self.session()
        fingerprint = s.describe()["transmission_fingerprint"]
        results = [s.call(arguments(grouping_factor=g)) for g in (1, 2, 4)]
        self.assertTrue(all(r["ok"] for r in results))
        self.assertEqual({r["transmission_fingerprint"] for r in results}, {fingerprint})
        self.assertEqual(self.session(medium_seed=1).describe()["transmission_fingerprint"], fingerprint)
        self.assertNotEqual(self.session(medium_seed=2).describe()["transmission_fingerprint"], fingerprint)

    def test_4_independent_noise_across_runs(self):
        s = self.session()
        a, b = s.call(arguments()), s.call(arguments())
        ha, hb = self.history(a), self.history(b)
        np.testing.assert_array_equal(ha["initial_pattern"], hb["initial_pattern"])   # same seed, same p0
        self.assertFalse(np.array_equal(ha["initial_validation"], hb["initial_validation"]))  # fresh noise
        self.assertEqual(a["transmission_fingerprint"], b["transmission_fingerprint"])

    def test_14_reproducible_from_fresh_sessions(self):
        def sequence(root):
            s = self.session(root=root)
            return [s.call(arguments(seed=3)), s.call(arguments(grouping_factor=1, seed=4))]

        first = sequence(self.root / "a")
        second = sequence(self.root / "b")
        for x, y in zip(first, second):
            for key in ("validated", "search", "acquisitions", "transmission_fingerprint", "objective"):
                self.assertEqual(x[key], y[key], key)


class TestScience(ToolTestCase):
    def test_5_real_optimizer_execution(self):
        result = self.session().call(arguments(iterations=40))
        h = self.history(result)
        self.assertEqual(len(h["enhancement_history"]), 40)
        self.assertEqual(int(h["accepted_history"].sum()), result["search"]["accepted_proposals"])
        self.assertEqual(result["acquisitions"]["counted_by_bench"], result["acquisitions"]["planned"])
        self.assertEqual(result["acquisitions"]["total"], 1 + 40 + 2 * 3)
        self.assertEqual(h["final_pattern_expanded"].shape, (8, 8))

    def test_6_validated_enhancement_matches_validation_measurements(self):
        noisy = self.session().call(arguments())
        h = self.history(noisy)
        self.assertAlmostEqual(noisy["validated"]["final"]["enhancement_mean"], float(h["final_validation"].mean()))
        self.assertAlmostEqual(noisy["validated"]["initial"]["enhancement_mean"], float(h["initial_validation"].mean()))
        self.assertEqual(noisy["validated"]["final"]["enhancement_values"], h["final_validation"].tolist())
        # Noise-free: every validation measurement equals the enhancement of the mean image.
        clean = self.session(noise=NO_NOISE).call(arguments())
        hc = self.history(clean)
        focus = Region(**clean["regions"]["focus"])
        backgrounds = [Region(**b) for b in clean["regions"]["backgrounds"]]
        floor = clean["objective"]["background_floor"]
        expected = region_metrics(hc["final_image"], focus, backgrounds, floor).enhancement
        self.assertAlmostEqual(clean["validated"]["final"]["enhancement_mean"], expected, places=12)
        self.assertEqual(clean["validated"]["final"]["enhancement_std"], 0.0)

    def test_7_focus_intensity_gain_and_decomposition(self):
        result = self.session().call(arguments(iterations=80, max_acquisitions=200))
        h = self.history(result)
        focus = Region(**result["regions"]["focus"])
        backgrounds = [Region(**b) for b in result["regions"]["backgrounds"]]
        start = region_metrics(h["initial_image"], focus, backgrounds)
        end = region_metrics(h["final_image"], focus, backgrounds)
        v = result["validated"]
        self.assertAlmostEqual(v["initial"]["focus_mean_intensity"], start.focus_mean)
        self.assertAlmostEqual(v["final"]["focus_mean_intensity"], end.focus_mean)
        self.assertAlmostEqual(v["focus_intensity_gain"], end.focus_mean / start.focus_mean)
        self.assertAlmostEqual(v["background_suppression_factor"], start.background_mean / end.background_mean)
        self.assertAlmostEqual(v["enhancement_gain"],
                               v["final"]["enhancement_mean"] / v["initial"]["enhancement_mean"])
        # Enhancement gain and focus gain are different quantities.
        self.assertNotAlmostEqual(v["enhancement_gain"], v["focus_intensity_gain"], places=3)
        d = v["decomposition"]
        self.assertIsNotNone(d)
        image_gain = (end.focus_mean / end.background_mean) / (start.focus_mean / start.background_mean)
        self.assertAlmostEqual(d["image_enhancement_gain"], image_gain)
        self.assertAlmostEqual(d["share_of_log_gain_from_focus_brightening"]
                               + d["share_of_log_gain_from_background_suppression"], 1.0)
        # Region means of the mean image equal the means over the individual frames (linearity).
        self.assertAlmostEqual(v["final"]["focus_mean_intensity"],
                               float(h["final_image"][focus.slices].mean()))

    def test_8_near_zero_intensity_handled_explicitly(self):
        focus, backgrounds = Region(0, 2, 0, 2), [Region(4, 6, 4, 6)]
        initial = np.ones((6, 6))
        initial[focus.slices] = 0.0
        final = np.ones((6, 6))
        final[focus.slices] = 3.0
        m = validated_intensity_metrics(initial, final, focus, backgrounds, 1e-3, [0.0, 0.0], [3.0, 3.0])
        self.assertIsNone(m["focus_intensity_gain"])
        self.assertIsNone(m["enhancement_gain"])
        self.assertIsNone(m["decomposition"])
        self.assertTrue(any(n.startswith("focus_intensity_gain not reported") for n in m["notes"]))
        self.assertTrue(any(n.startswith("enhancement_gain not reported") for n in m["notes"]))
        self.assertAlmostEqual(m["background_suppression_factor"], 1.0)
        json.dumps(m)
        # Final background near zero: suppression factor is not reported either.
        final[backgrounds[0].slices] = 0.0
        m2 = validated_intensity_metrics(np.ones((6, 6)), final, focus, backgrounds, 1e-3, [1.0], [3000.0])
        self.assertIsNone(m2["background_suppression_factor"])
        self.assertTrue(m2["final"]["background_floor_active"])


class TestArtifacts(ToolTestCase):
    def test_9_unique_run_directories(self):
        s = self.session()
        dirs = [Path(s.call(arguments())["run_dir"]) for _ in range(3)]
        self.assertEqual(len(set(dirs)), 3)
        for d in dirs:
            self.assertEqual(d.parent, self.root.resolve())
            self.assertEqual(sorted(p.name for p in d.iterdir()), sorted(ft.ARTIFACT_FILES.values()))
        self.assertEqual(s.runs, [d.name for d in dirs])

    def test_10_png_and_json_artifacts_are_valid(self):
        from matplotlib.image import imread
        result = self.session().call(arguments())
        for key in ("initial_camera_png", "final_camera_png", "enhancement_history_png", "final_dmd_pattern_png"):
            path = Path(result["artifacts"][key])
            self.assertEqual(path.read_bytes()[:8], PNG_SIGNATURE, key)
            image = imread(path)
            self.assertEqual(image.ndim, 3)
            self.assertGreater(image.shape[0], 100)
        for key in ("settings", "result"):
            json.loads(Path(result["artifacts"][key]).read_text(encoding="utf-8"))
        h = self.history(result)
        self.assertEqual(h["initial_image"].shape, (16, 16))
        self.assertTrue(np.all(h["final_image"] >= 0))

    def test_11_never_overwrites(self):
        s = self.session()
        first = s.call(dict(arguments(), run_name="fixed_name"))
        self.assertTrue(first["ok"])
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(first["run_dir"]).iterdir()}
        frames = s.describe()["camera_acquisitions_so_far"]
        again = s.call(dict(arguments(), run_name="fixed_name"))
        self.assertFalse(again["ok"])
        self.assertIn("already exists", again["error"])
        after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(first["run_dir"]).iterdir()}
        self.assertEqual(before, after)
        self.assertEqual(s.describe()["camera_acquisitions_so_far"], frames)   # rejected before measuring
        with self.assertRaises(FocusingToolError):
            ft._create(Path(first["run_dir"]) / "result.json")

    def test_paths_outside_artifact_root_rejected(self):
        s = self.session()
        outside = self.root.parent / "escaped_run"
        for name in ("../escaped_run", "..", "a/b", "a\\b", "", "C:/temp/x", ".hidden", "x" * 65, 7):
            result = s.call(dict(arguments(), run_name=name))
            self.assertFalse(result["ok"], name)
        self.assertFalse(outside.exists())
        self.assertEqual(list(self.root.iterdir()), [])


class TestIsolation(ToolTestCase):
    def test_13_transmission_matrix_never_exposed(self):
        s = self.session()
        result = s.call(arguments())
        n_out, n_in = 16 * 16, 8 * 8
        public = {k: v for k, v in vars(s).items() if not k.startswith("_")}
        for value in public.values():
            self.assertFalse(isinstance(value, np.ndarray) and value.size >= n_out * n_in)
        for text in (json.dumps(result), json.dumps(s.describe())):
            self.assertNotIn("transmission_matrix", text)
            self.assertLess(len(text), 50_000)
        with np.load(result["artifacts"]["history_npz"]) as data:
            for key in data.files:
                self.assertNotIn(data[key].size, (n_out * n_in,), key)
                self.assertFalse(np.iscomplexobj(data[key]), key)
        source = Path(ft.__file__).read_text(encoding="utf-8")
        for forbidden in ("_transmission", "transmission_matrix", "_ideal_intensity"):
            self.assertNotIn(forbidden, source)

    def test_16_imports_and_original_files(self):
        tree = ast.parse(Path(ft.__file__).read_text(encoding="utf-8"))
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                modules.add(node.module)
            elif isinstance(node, ast.Import):
                modules |= {a.name for a in node.names}
        self.assertFalse({m for m in modules if m and ("task1" in m or "agentic" in m or "langgraph" in m
                                                       or "openai" in m)})
        if shutil.which("git") is None:
            self.skipTest("git not available")
        out = subprocess.run(["git", "status", "--porcelain", "--", "task5", "task6", "README.md",
                              "requirements.txt"], cwd=REPO, capture_output=True, text=True, check=True).stdout
        self.assertEqual(out, "")


if __name__ == "__main__":
    unittest.main()
