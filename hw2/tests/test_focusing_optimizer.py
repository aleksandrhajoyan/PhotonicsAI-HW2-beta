"""Offline tests for the HW2 Task 2 black-box focusing optimizer (small dimensions)."""

from __future__ import annotations

import ast
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hw2 import focusing_optimizer as fo  # noqa: E402
from hw2.focusing_optimizer import (  # noqa: E402
    OptimizerSettings,
    expand_pattern,
    grouped_shape,
    optimize_focus,
    planned_acquisitions,
)
from hw2.simulated_bench import (  # noqa: E402
    NoiseModel,
    Region,
    SimulatedBench,
    region_metrics,
    surrounding_background_regions,
)

DMD_SHAPE = (8, 8)
CAMERA_SHAPE = (12, 12)
FOCUS = Region(5, 7, 5, 7, name="focus")
BACKGROUNDS = surrounding_background_regions(FOCUS, CAMERA_SHAPE, gap=1, width=2)
NO_NOISE = NoiseModel(enabled=False)
NOISY = NoiseModel(photons_per_unit=300.0, read_noise_std=0.02)


def make_bench(seed=4, noise=NO_NOISE, **kwargs) -> SimulatedBench:
    return SimulatedBench(DMD_SHAPE, CAMERA_SHAPE, FOCUS, BACKGROUNDS, seed=seed, noise=noise, **kwargs)


def settings(**overrides) -> OptimizerSettings:
    base = dict(grouping_factor=2, iterations=40, flips_per_proposal=1, focus=FOCUS,
                backgrounds=BACKGROUNDS, seed=1, averaging=1, validation_count=3)
    base.update(overrides)
    return OptimizerSettings(**base)


# ---------------------------------------------------------------- fixtures

def known_medium_bench():
    """A 4x4 DMD whose best achievable enhancement is known analytically.

    The focus pixel's row of T is t_j = +1 on the set S (every other macropixel)
    and -1 elsewhere; every background pixel's row is the constant c. Then
        eta = |sum_on t_j|^2 / (c * n_on)^2 = (mean of t over the on set)^2 / c^2 <= 1 / c^2,
    with equality exactly when the on set lies inside S or inside its complement.
    The all-on start has sum t = 0, i.e. eta = 0.
    """
    dmd, cam, c = (4, 4), (8, 8), 0.25
    focus = Region(3, 4, 3, 4, name="focus")
    backgrounds = surrounding_background_regions(focus, cam, gap=1, width=2)
    s = np.zeros(16, dtype=bool)
    s[::2] = True
    t = np.zeros((64, 16), dtype=complex)
    for b in backgrounds:
        for r in range(b.row_start, b.row_stop):
            for col in range(b.col_start, b.col_stop):
                t[r * cam[1] + col, :] = c
    t[3 * cam[1] + 3, :] = np.where(s, 1.0, -1.0)
    bench = SimulatedBench(dmd, cam, focus, backgrounds, noise=NO_NOISE, transmission_matrix=t)
    return bench, focus, backgrounds, s, 1.0 / c ** 2


class ScriptedBench:
    """Returns camera frames whose focus/background values follow a script; no medium at all."""

    def __init__(self, script):
        self.dmd = SimpleNamespace(shape=(2, 2))
        self.camera_shape = (4, 4)
        self.focus = Region(0, 1, 0, 1)
        self.background = Region(2, 4, 2, 4)
        self.script = list(script)
        self.patterns = []
        self.num_acquisitions = 0

    def acquire(self, pattern):
        self.patterns.append(np.array(pattern))
        focus_value, background_value = self.script.pop(0)
        image = np.zeros(self.camera_shape)
        image[self.focus.slices] = focus_value
        image[self.background.slices] = background_value
        self.num_acquisitions += 1
        return image

    def transmission_fingerprint(self):
        return "scripted"


class AttributeSpy:
    """Records every attribute the optimizer reads from the bench."""

    def __init__(self, target):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "names", [])

    def __getattr__(self, name):
        self.names.append(name)
        return getattr(self._target, name)


# ---------------------------------------------------------------- tests

class TestGrouping(unittest.TestCase):
    def test_1_macropixel_expansion(self):
        grouped = np.array([[1, 0], [0, 1]], dtype=bool)
        expected = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]], dtype=bool)
        np.testing.assert_array_equal(expand_pattern(grouped, 2), expected)
        np.testing.assert_array_equal(expand_pattern(grouped, 1), grouped)
        self.assertEqual(expand_pattern(np.ones((2, 2), bool), 4).shape, (8, 8))
        self.assertEqual([grouped_shape(DMD_SHAPE, g) for g in (1, 2, 4, 8)], [(8, 8), (4, 4), (2, 2), (1, 1)])

    def test_2_patterns_stay_binary(self):
        result = optimize_focus(make_bench(), settings())
        for pattern in (result.initial_pattern, result.final_pattern, result.final_pattern_expanded):
            self.assertEqual(pattern.dtype, bool)
        self.assertEqual(result.final_pattern_expanded.shape, DMD_SHAPE)
        # Block structure: every 2x2 block of the expanded pattern is constant.
        blocks = result.final_pattern_expanded.reshape(4, 2, 4, 2)
        self.assertTrue(np.all(blocks == blocks[:, :1, :, :1]))
        scripted = ScriptedBench([(1, 1)] * 20)
        optimize_focus(scripted, OptimizerSettings(iterations=5, focus=scripted.focus,
                                                    backgrounds=(scripted.background,), validation_count=2))
        for sent in scripted.patterns:
            self.assertEqual(sent.dtype, bool)

    def test_3_same_T_for_all_grouping_factors(self):
        bench = make_bench()
        fingerprint = bench.transmission_fingerprint()
        shapes = []
        for g in (1, 2, 4):
            result = optimize_focus(bench, settings(grouping_factor=g, iterations=10))
            self.assertEqual(result.transmission_fingerprint, fingerprint)
            shapes.append(result.grouped_shape)
        self.assertEqual(shapes, [(8, 8), (4, 4), (2, 2)])
        self.assertEqual(bench.transmission_fingerprint(), fingerprint)


class TestSearchMechanics(unittest.TestCase):
    def test_4_objective_calculation(self):
        bench = make_bench()
        result = optimize_focus(bench, settings(iterations=0))
        reference = make_bench()  # same seed, same T, no noise
        image = reference.acquire(expand_pattern(result.initial_pattern, 2))
        background = region_metrics(image, FOCUS, BACKGROUNDS).background_mean
        floor = 0.05 * background
        self.assertAlmostEqual(result.objective_background_floor, floor, places=12)
        expected = region_metrics(image, FOCUS, BACKGROUNDS, floor).enhancement
        self.assertAlmostEqual(result.initial_enhancement, expected, places=12)
        self.assertAlmostEqual(result.initial_enhancement,
                               region_metrics(image, FOCUS, BACKGROUNDS).focus_mean / background, places=12)

    def test_5_accept_reject_behaviour(self):
        # Objective = focus / background with background 1: 1 (initial), then candidates
        # 2 (accept), 1.5 (reject), 3 (accept), 3 (equal: reject), then 2 validation frames.
        script = [(1, 1), (2, 1), (1.5, 1), (3, 1), (3, 1), (1, 1), (1, 1)]
        bench = ScriptedBench(script)
        p0 = np.array([[1, 0], [0, 1]], dtype=bool)
        result = optimize_focus(bench, OptimizerSettings(
            iterations=4, focus=bench.focus, backgrounds=(bench.background,), seed=3,
            initial_pattern=p0, validation_count=1))
        self.assertEqual(result.enhancement_history, [2.0, 1.5, 3.0, 3.0])
        self.assertEqual(result.accepted_history, [True, False, True, False])
        self.assertEqual(result.accepted_objective_history, [1.0, 2.0, 2.0, 3.0, 3.0])
        expected = p0.ravel().copy()
        for k in (0, 2):                              # only accepted proposals change the pattern
            expected[result.proposal_history[k]] ^= True
        np.testing.assert_array_equal(result.final_pattern.ravel(), expected)
        # Each candidate was the accepted pattern with exactly the proposed macropixel flipped.
        accepted = p0.ravel().copy()
        for k, flips in enumerate(result.proposal_history):
            candidate = accepted.copy()
            candidate[flips] ^= True
            np.testing.assert_array_equal(bench.patterns[1 + k].ravel(), candidate)
            if result.accepted_history[k]:
                accepted = candidate

    def test_5_dark_pattern_not_rewarded(self):
        # Candidate: focus 0.01 over background 1e-6 is a raw ratio of 1e4, but the
        # objective floor is 0.05 * initial background (= 0.05), giving 0.2 < 1.
        bench = ScriptedBench([(1, 1), (0.01, 1e-6), (1, 1), (1, 1)])
        result = optimize_focus(bench, OptimizerSettings(
            iterations=1, focus=bench.focus, backgrounds=(bench.background,), validation_count=1))
        self.assertAlmostEqual(result.enhancement_history[0], 0.01 / 0.05)
        self.assertEqual(result.accepted_history, [False])

    def test_6_7_images_and_history_lengths(self):
        result = optimize_focus(make_bench(noise=NOISY), settings(iterations=25, flips_per_proposal=3))
        for image in (result.initial_image, result.final_image):
            self.assertEqual(image.shape, CAMERA_SHAPE)
            self.assertTrue(np.all(np.isfinite(image)) and np.all(image >= 0))
        self.assertEqual(len(result.enhancement_history), 25)
        self.assertEqual(len(result.accepted_history), 25)
        self.assertEqual(len(result.accepted_objective_history), 26)
        self.assertEqual(len(result.proposal_history), 25)
        self.assertTrue(all(len(set(p)) == 3 for p in result.proposal_history))
        self.assertEqual(result.best_observed_enhancement, max(result.accepted_objective_history))
        summary = result.summary()
        self.assertEqual(summary["iterations"], 25)
        self.assertEqual(result.settings["iterations"], 25)
        self.assertEqual(result.settings["focus"]["row_start"], 5)


class TestBudgetReproducibilityValidation(unittest.TestCase):
    def test_8_exact_acquisition_accounting(self):
        bench = make_bench(noise=NOISY)
        s = settings(averaging=3, iterations=7, validation_count=2)
        self.assertEqual(planned_acquisitions(s), 3 * (1 + 7 + 2 * 2))
        result = optimize_focus(bench, s)
        self.assertEqual(bench.num_acquisitions, 36)
        self.assertEqual(result.acquisitions,
                         {"search": 24, "validation": 12, "total": 36, "planned": 36, "frames_per_measurement": 3})
        # A budget that is too small is rejected before any acquisition.
        fresh = make_bench(noise=NOISY)
        with self.assertRaises(ValueError):
            optimize_focus(fresh, settings(averaging=3, iterations=7, validation_count=2, max_acquisitions=35))
        self.assertEqual(fresh.num_acquisitions, 0)
        optimize_focus(fresh, settings(averaging=3, iterations=7, validation_count=2, max_acquisitions=36))

    def test_9_reproducible_with_fixed_seeds(self):
        a = optimize_focus(make_bench(seed=9, noise=NOISY), settings(seed=5, averaging=2))
        b = optimize_focus(make_bench(seed=9, noise=NOISY), settings(seed=5, averaging=2))
        self.assertEqual(a.enhancement_history, b.enhancement_history)
        self.assertEqual(a.accepted_history, b.accepted_history)
        self.assertEqual(a.final_validation, b.final_validation)
        np.testing.assert_array_equal(a.final_pattern, b.final_pattern)
        np.testing.assert_array_equal(a.final_image, b.final_image)
        c = optimize_focus(make_bench(seed=9, noise=NOISY), settings(seed=6, averaging=2))
        self.assertNotEqual(a.proposal_history, c.proposal_history)

    def test_9_proposals_independent_of_camera_noise(self):
        t = np.random.default_rng(0).standard_normal((144, 64)) + 0j
        a = optimize_focus(make_bench(seed=1, noise=NOISY, transmission_matrix=t), settings(seed=5))
        b = optimize_focus(make_bench(seed=2, noise=NOISY, transmission_matrix=t), settings(seed=5))
        self.assertEqual(a.transmission_fingerprint, b.transmission_fingerprint)
        self.assertEqual(a.proposal_history, b.proposal_history)     # same proposal stream
        self.assertNotEqual(a.enhancement_history, b.enhancement_history)  # different noise

    def test_10_independent_validation(self):
        noisy = optimize_focus(make_bench(noise=NOISY), settings(validation_count=6))
        self.assertEqual(len(noisy.final_validation), 6)
        self.assertEqual(len(noisy.initial_validation), 6)
        self.assertGreater(len(set(noisy.final_validation)), 1)       # fresh noise per measurement
        clean = optimize_focus(make_bench(), settings(validation_count=4))
        self.assertEqual(len(set(clean.final_validation)), 1)
        self.assertAlmostEqual(clean.final_validation[0], clean.final_search_enhancement, places=12)
        self.assertAlmostEqual(clean.initial_validation[0], clean.initial_enhancement, places=12)


class TestValidationOfSettings(unittest.TestCase):
    def test_11_invalid_parameters_rejected(self):
        bad = [
            dict(grouping_factor=3), dict(grouping_factor=0), dict(grouping_factor=True),
            dict(grouping_factor=16), dict(iterations=-1), dict(iterations=2.5),
            dict(flips_per_proposal=0), dict(flips_per_proposal=17), dict(averaging=0),
            dict(validation_count=0), dict(seed=-1), dict(background_floor_fraction=0),
            dict(background_floor_fraction=1), dict(focus=None),
            dict(backgrounds=()), dict(backgrounds=(Region(5, 6, 5, 6),)),           # none / overlaps focus
            dict(backgrounds=(Region(0, 2, 11, 14),)),                                # outside the image
            dict(initial_pattern=np.ones((8, 8), bool)),                             # wrong grouped shape
            dict(initial_pattern=np.full((4, 4), 0.5)),                              # not binary
            dict(max_acquisitions=0),
        ]
        for overrides in bad:
            bench = make_bench()
            with self.assertRaises(ValueError, msg=str(overrides)):
                optimize_focus(bench, settings(**overrides))
            self.assertEqual(bench.num_acquisitions, 0, msg=str(overrides))
        with self.assertRaises(ValueError):
            expand_pattern(np.ones((2, 2)), 2)                                       # not boolean

    def test_12_optimizer_never_reads_T(self):
        spy = AttributeSpy(make_bench(noise=NOISY))
        optimize_focus(spy, settings(iterations=10))
        self.assertTrue(set(spy.names) <= {"dmd", "camera_shape", "acquire", "transmission_fingerprint",
                                           "num_acquisitions"}, spy.names)
        source = Path(fo.__file__).read_text(encoding="utf-8")
        for forbidden in ("_transmission", "transmission_matrix", "_ideal_intensity", "_noise_rng"):
            self.assertNotIn(forbidden, source)
        # The scripted bench has no medium at all and the optimizer still works.
        scripted = ScriptedBench([(1, 1)] * 10)
        optimize_focus(scripted, OptimizerSettings(iterations=3, focus=scripted.focus,
                                                    backgrounds=(scripted.background,), validation_count=2))


class TestImprovement(unittest.TestCase):
    def test_13_reaches_known_optimum_on_deterministic_medium(self):
        for seed in range(4):
            bench, focus, backgrounds, s, eta_max = known_medium_bench()
            result = optimize_focus(bench, OptimizerSettings(
                grouping_factor=1, iterations=200, focus=focus, backgrounds=backgrounds, seed=seed,
                initial_pattern=np.ones((4, 4), bool), validation_count=2))
            self.assertEqual(result.validated_initial_enhancement, 0.0)
            self.assertAlmostEqual(result.validated_final_enhancement, eta_max, places=9)
            final = result.final_pattern.ravel()
            self.assertTrue(np.array_equal(final, s) or np.array_equal(final, ~s), final)
            # The search never exceeds the proven upper bound.
            self.assertLessEqual(max(result.enhancement_history), eta_max + 1e-9)

    def test_13_noisy_random_medium_improves_and_is_validated(self):
        bench = SimulatedBench((16, 16), (24, 24), Region(11, 13, 11, 13),
                               surrounding_background_regions(Region(11, 13, 11, 13), (24, 24), gap=2, width=3),
                               seed=2, noise=NOISY)
        focus = Region(11, 13, 11, 13)
        result = optimize_focus(bench, OptimizerSettings(
            grouping_factor=2, iterations=300, focus=focus,
            backgrounds=surrounding_background_regions(focus, (24, 24), gap=2, width=3),
            seed=0, validation_count=8))
        # Independent validation, not the biased search value, shows the improvement.
        self.assertGreater(result.validated_final_enhancement, 2.0 * result.validated_initial_enhancement)
        self.assertGreater(min(result.final_validation), max(result.initial_validation))


class TestIsolation(unittest.TestCase):
    def test_15_original_files_unchanged(self):
        if shutil.which("git") is None:
            self.skipTest("git not available")
        out = subprocess.run(
            ["git", "status", "--porcelain", "--", "task5", "task6", "README.md", "requirements.txt"],
            cwd=REPO, capture_output=True, text=True, check=True,
        ).stdout
        self.assertEqual(out, "")

    def test_optimizer_imports(self):
        tree = ast.parse(Path(fo.__file__).read_text(encoding="utf-8"))
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                modules.add(node.module)
        self.assertEqual(modules, {"__future__", "argparse", "json", "time", "dataclasses", "numpy",
                                   "hw2.simulated_bench"})


if __name__ == "__main__":
    unittest.main()
