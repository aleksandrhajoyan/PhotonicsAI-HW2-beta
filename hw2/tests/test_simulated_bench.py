"""Offline tests for the HW2 Task 2 simulated optical bench (small dimensions, NumPy only)."""

from __future__ import annotations

import ast
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hw2 import simulated_bench as sb  # noqa: E402
from hw2.simulated_bench import (  # noqa: E402
    DMD,
    NoiseModel,
    Region,
    SimulatedBench,
    region_metrics,
    surrounding_background_regions,
)

DMD_SHAPE = (4, 5)          # 20 macropixels
CAMERA_SHAPE = (12, 12)     # 144 pixels
FOCUS = Region(5, 7, 5, 7, name="focus")
BACKGROUNDS = surrounding_background_regions(FOCUS, CAMERA_SHAPE, gap=1, width=2)
NO_NOISE = NoiseModel(enabled=False)


def bench(seed=3, noise=NO_NOISE, **kwargs) -> SimulatedBench:
    return SimulatedBench(DMD_SHAPE, CAMERA_SHAPE, FOCUS, BACKGROUNDS, seed=seed, noise=noise, **kwargs)


def random_pattern(seed=0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 2, size=DMD_SHAPE)


class TestDimensionsAndPhysics(unittest.TestCase):
    def test_1_dimensions(self):
        b = bench()
        self.assertEqual(b.num_macropixels, 20)
        self.assertEqual(b.dmd.input_field(random_pattern()).shape, (20,))
        image = b.acquire(random_pattern())
        self.assertEqual(image.shape, CAMERA_SHAPE)
        self.assertEqual(image.dtype, np.float64)
        # A flat pattern of the right length is accepted too.
        self.assertEqual(b.acquire(random_pattern().ravel()).shape, CAMERA_SHAPE)

    def test_2_binary_amplitude_modulation_only(self):
        dmd = DMD(DMD_SHAPE, illumination_amplitude=2.5)
        p = random_pattern()
        e_in = dmd.input_field(p)
        self.assertTrue(np.isrealobj(e_in))                      # no phase at all
        self.assertTrue(set(np.unique(e_in)) <= {0.0, 2.5})      # amplitude a or 0
        np.testing.assert_array_equal(e_in, 2.5 * p.ravel())
        np.testing.assert_array_equal(dmd.input_field(p.astype(bool)), e_in)
        for bad in (np.full(DMD_SHAPE, 0.5), np.full(DMD_SHAPE, 2), -np.ones(DMD_SHAPE),
                    np.full(DMD_SHAPE, np.nan), np.ones(DMD_SHAPE) * 1j, np.ones((3, 5)),
                    np.full(DMD_SHAPE, "1")):
            with self.assertRaises(ValueError):
                dmd.input_field(bad)

    def test_3_intensity_equals_abs_T_E_squared(self):
        rng = np.random.default_rng(11)
        t = rng.standard_normal((144, 20)) + 1j * rng.standard_normal((144, 20))
        b = bench(transmission_matrix=t, illumination_amplitude=0.7)
        p = random_pattern(5)
        expected = np.abs(t @ (0.7 * p.ravel().astype(float))) ** 2
        np.testing.assert_allclose(b.acquire(p), expected.reshape(CAMERA_SHAPE), rtol=1e-12)
        # Row-major mapping: macropixel (r, c) -> column r*cols + c of T.
        single = np.zeros(DMD_SHAPE, dtype=int)
        single[2, 3] = 1
        np.testing.assert_allclose(b.acquire(single).ravel(), np.abs(0.7 * t[:, 2 * 5 + 3]) ** 2)
        # The bench keeps its own copy: changing the caller's array changes nothing.
        before = b.acquire(p)
        t[:] = 0
        np.testing.assert_array_equal(b.acquire(p), before)

    def test_default_T_statistics(self):
        b = SimulatedBench((10, 10), (40, 40), Region(18, 22, 18, 22),
                           surrounding_background_regions(Region(18, 22, 18, 22), (40, 40)),
                           seed=1, noise=NO_NOISE)
        mean_intensity = b.acquire(np.ones((10, 10))).mean()
        self.assertAlmostEqual(mean_intensity, 1.0, delta=0.1)    # E|T|^2 = 1/n_in, all on
        np.testing.assert_array_equal(b.acquire(np.zeros((10, 10))), 0.0)


class TestFixedTransmissionAndNoise(unittest.TestCase):
    def test_4_T_fixed_within_experiment(self):
        b = bench(noise=NoiseModel())
        fingerprint = b.transmission_fingerprint()
        for k in range(25):
            b.measure(random_pattern(k))
        self.assertEqual(b.transmission_fingerprint(), fingerprint)
        self.assertEqual(b.num_acquisitions, 25)
        self.assertFalse(b._transmission.flags.writeable)
        with self.assertRaises(ValueError):
            b._transmission[0, 0] = 0

    def test_5_noise_disabled_identical_measurements(self):
        b = bench(noise=NO_NOISE)
        p = random_pattern()
        frames = [b.acquire(p) for _ in range(5)]
        for frame in frames[1:]:
            np.testing.assert_array_equal(frame, frames[0])

    def test_6_noise_enabled_independent_measurements(self):
        b = bench(noise=NoiseModel(photons_per_unit=200.0, read_noise_std=0.02))
        p = np.ones(DMD_SHAPE)
        frames = np.array([b.acquire(p) for _ in range(400)])
        self.assertFalse(np.array_equal(frames[0], frames[1]))
        ideal = bench(noise=NO_NOISE).acquire(p)            # same seed -> same T
        residuals = (frames - ideal).reshape(400, -1)
        # Unbiased on average where the signal is well above zero (no clipping).
        bright = ideal.ravel() > 0.5
        self.assertLess(abs(residuals[:, bright].mean()), 0.01)
        # Successive frames are uncorrelated.
        corr = np.corrcoef(residuals[:-1].ravel(), residuals[1:].ravel())[0, 1]
        self.assertLess(abs(corr), 0.05)
        # Variance follows the model: I / photons_per_unit + read_noise_std^2.
        pixel = int(np.argmax(ideal))
        expected_var = ideal.ravel()[pixel] / 200.0 + 0.02 ** 2
        self.assertAlmostEqual(residuals[:, pixel].var() / expected_var, 1.0, delta=0.25)

    def test_6_noise_components_separately(self):
        p = np.ones(DMD_SHAPE)
        read_only = bench(noise=NoiseModel(photons_per_unit=None, read_noise_std=0.05))
        shot_only = bench(noise=NoiseModel(photons_per_unit=50.0, read_noise_std=0.0))
        self.assertFalse(np.array_equal(read_only.acquire(p), read_only.acquire(p)))
        frame = shot_only.acquire(p)
        np.testing.assert_allclose(frame * 50.0, np.round(frame * 50.0), atol=1e-9)  # photon counts

    def test_7_seeded_reproducibility(self):
        noisy = NoiseModel(photons_per_unit=100.0, read_noise_std=0.05)
        a, b, c = bench(seed=7, noise=noisy), bench(seed=7, noise=noisy), bench(seed=8, noise=noisy)
        self.assertEqual(a.transmission_fingerprint(), b.transmission_fingerprint())
        self.assertNotEqual(a.transmission_fingerprint(), c.transmission_fingerprint())
        for k in range(5):
            np.testing.assert_array_equal(a.acquire(random_pattern(k)), b.acquire(random_pattern(k)))
        # Noise on/off does not change T for the same seed.
        self.assertEqual(bench(seed=7, noise=NO_NOISE).transmission_fingerprint(), a.transmission_fingerprint())

    def test_measurements_real_finite_nonnegative(self):
        b = bench(noise=NoiseModel(photons_per_unit=5.0, read_noise_std=0.5))  # heavy noise -> clipping
        for k in range(20):
            image = b.acquire(random_pattern(k))
            self.assertTrue(np.isrealobj(image))
            self.assertTrue(np.all(np.isfinite(image)))
            self.assertTrue(np.all(image >= 0))
        self.assertTrue(np.all(b.acquire(np.zeros(DMD_SHAPE)) >= 0))


class TestRegionsAndEnhancement(unittest.TestCase):
    def test_8_valid_regions(self):
        self.assertEqual([r.name for r in BACKGROUNDS], ["top", "bottom", "left", "right"])
        self.assertEqual(BACKGROUNDS[0], Region(2, 4, 5, 7, name="top"))
        self.assertEqual(BACKGROUNDS[3], Region(5, 7, 8, 10, name="right"))
        for i, a in enumerate(BACKGROUNDS):
            self.assertFalse(a.overlaps(FOCUS))
            for other in BACKGROUNDS[i + 1:]:
                self.assertFalse(a.overlaps(other))
        # Clipped at the image border; empty sides dropped.
        corner = surrounding_background_regions(Region(0, 2, 0, 2), CAMERA_SHAPE)
        self.assertEqual([r.name for r in corner], ["bottom", "right"])

    def test_8_invalid_regions_rejected(self):
        cases = [
            (Region(5, 7, 5, 7), [Region(6, 8, 6, 8)]),                      # overlaps focus
            (FOCUS, [Region(0, 2, 0, 2), Region(1, 3, 1, 3)]),               # backgrounds overlap
            (Region(10, 13, 0, 2), [Region(0, 2, 0, 2)]),                    # focus outside image
            (FOCUS, [Region(0, 2, 11, 13)]),                                 # background outside
            (FOCUS, []),                                                     # no background
            (FOCUS, [(0, 2, 0, 2)]),                                         # not a Region
        ]
        for focus, backgrounds in cases:
            with self.assertRaises(ValueError, msg=str((focus, backgrounds))):
                SimulatedBench(DMD_SHAPE, CAMERA_SHAPE, focus, backgrounds)
        for bad in [(2, 2, 0, 1), (0, 1, 3, 3), (-1, 1, 0, 1), (0, 1.5, 0, 1), (True, 2, 0, 1)]:
            with self.assertRaises(ValueError, msg=str(bad)):
                Region(*bad)

    def test_9_enhancement_definition(self):
        image = np.zeros((6, 6))
        focus = Region(0, 2, 0, 2)
        b1, b2 = Region(4, 5, 0, 2), Region(3, 6, 4, 6)       # 2 and 6 pixels
        image[focus.slices] = 10.0
        image[b1.slices] = 1.0
        image[b2.slices] = 3.0
        m = region_metrics(image, focus, [b1, b2])
        self.assertEqual(m.focus_mean, 10.0)
        self.assertEqual(m.background_region_means, (1.0, 3.0))
        self.assertEqual(m.background_mean, (2 * 1.0 + 6 * 3.0) / 8)   # pixel-weighted = 2.5
        self.assertEqual(m.enhancement, 10.0 / 2.5)
        self.assertFalse(m.background_floored)

    def test_9_near_zero_background_uses_floor(self):
        image = np.zeros((6, 6))
        focus = Region(0, 2, 0, 2)
        image[focus.slices] = 4.0
        m = region_metrics(image, focus, [Region(4, 6, 4, 6)], background_floor=1e-3)
        self.assertTrue(m.background_floored)
        self.assertEqual(m.enhancement, 4.0 / 1e-3)
        self.assertTrue(np.isfinite(m.enhancement))

    def test_9_bench_measure_consistent_with_region_metrics(self):
        b = bench(noise=NoiseModel())
        m = b.measure(random_pattern(2))
        again = region_metrics(m.image, FOCUS, BACKGROUNDS, b.background_floor)
        self.assertEqual(m.focus_mean, again.focus_mean)
        self.assertEqual(m.background_mean, again.background_mean)
        self.assertEqual(m.enhancement, again.enhancement)
        self.assertEqual(m.acquisition_index, 0)
        self.assertEqual(b.measure(random_pattern(2)).acquisition_index, 1)
        self.assertEqual(len(m.background_region_means), 4)


class TestValidation(unittest.TestCase):
    def test_10_invalid_parameters_rejected(self):
        bad_constructions = [
            lambda: SimulatedBench((0, 4), CAMERA_SHAPE, FOCUS, BACKGROUNDS),
            lambda: SimulatedBench((4,), CAMERA_SHAPE, FOCUS, BACKGROUNDS),
            lambda: SimulatedBench((4, 2.5), CAMERA_SHAPE, FOCUS, BACKGROUNDS),
            lambda: SimulatedBench(DMD_SHAPE, (12, -1), FOCUS, BACKGROUNDS),
            lambda: bench(seed=-1),
            lambda: bench(seed=1.5),
            lambda: bench(illumination_amplitude=0),
            lambda: bench(illumination_amplitude=float("inf")),
            lambda: bench(background_floor=0),
            lambda: bench(noise="none"),
            lambda: bench(transmission_matrix=np.ones((144, 19))),
            lambda: bench(transmission_matrix=np.full((144, 20), np.nan)),
            lambda: NoiseModel(photons_per_unit=0),
            lambda: NoiseModel(photons_per_unit=float("nan")),
            lambda: NoiseModel(read_noise_std=-0.1),
            lambda: NoiseModel(enabled=1),
            lambda: surrounding_background_regions(FOCUS, CAMERA_SHAPE, gap=-1),
            lambda: surrounding_background_regions(Region(0, 12, 0, 12), CAMERA_SHAPE),
        ]
        for i, construct in enumerate(bad_constructions):
            with self.assertRaises(ValueError, msg=f"case {i}"):
                construct()
        b = bench()
        with self.assertRaises(ValueError):
            b.measure(np.ones((5, 4)))
        with self.assertRaises(ValueError):
            region_metrics(-np.ones(CAMERA_SHAPE), FOCUS, BACKGROUNDS)
        with self.assertRaises(ValueError):
            region_metrics(np.full(CAMERA_SHAPE, np.inf), FOCUS, BACKGROUNDS)


class TestIsolation(unittest.TestCase):
    def test_11_simulator_imports_only_numpy_and_stdlib(self):
        tree = ast.parse(Path(sb.__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
        self.assertEqual(imported, {"__future__", "hashlib", "math", "dataclasses", "numpy"})

    def test_11_original_files_unchanged(self):
        if shutil.which("git") is None:
            self.skipTest("git not available")
        out = subprocess.run(
            ["git", "status", "--porcelain", "--", "task5", "task6", "README.md", "requirements.txt"],
            cwd=REPO, capture_output=True, text=True, check=True,
        ).stdout
        self.assertEqual(out, "")


if __name__ == "__main__":
    unittest.main()
