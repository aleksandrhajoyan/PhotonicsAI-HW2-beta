"""
HW2 Task 2 — black-box DMD focusing optimizer (no agent integration).

The optimizer improves the focus behind the simulated scattering medium by
iterative binary amplitude flipping, using ONLY camera feedback. It talks to
the optical system exclusively through BenchAccess, which exposes:

    dmd_shape, camera_shape, acquire(pattern), num_acquisitions,
    transmission_fingerprint()

It never reads the transmission matrix T.

Macropixel grouping
-------------------
The bench has a fixed underlying DMD grid of R x C pixels. With grouping
factor g (g must divide R and C) the optimizer controls an (R/g) x (C/g) grid of
macropixels; macropixel (i, j) drives the g x g block of underlying pixels
rows i*g..i*g+g-1, cols j*g..j*g+g-1 (expand_pattern). The bench and its T are
the same for every g, so different grouping factors probe the SAME medium.

Measurement
-----------
One "measurement" of a pattern = `averaging` independent camera acquisitions;
their pixel-wise mean image is evaluated with simulated_bench.region_metrics on
the optimizer's focus region and background regions.

Objective (the enhancement eta of simulated_bench, with a guarded floor):
    eta = <I>_F / max(<I>_B, floor)
    floor = max(background_floor_fraction * <I>_B(initial measurement), 1e-12)
The relative floor stops a nearly dark pattern from scoring well through a tiny
background denominator: no pattern can gain by pushing the background below
background_floor_fraction of the initial speckle background.

Algorithm
---------
 1. p0 = initial_pattern, or a random binary pattern from rng(seed)  (P(1) = 1/2)
 2. measure p0 -> eta_0; accepted = p0, eta_acc = eta_0
 3. for k = 1..iterations:
      choose flips_per_proposal distinct macropixels uniformly at random (rng),
      candidate = accepted with those macropixels flipped 0 <-> 1,
      measure candidate -> eta_k,
      accept iff eta_k > eta_acc (strict); then accepted = candidate, eta_acc = eta_k
 4. validation: validation_count NEW measurements of p0 and of the final
    accepted pattern (independent noise), reported as mean / std / all values.

eta_acc is the value of a single noisy measurement that won a comparison, so
it is biased upward ("best observed"). The validated values are the
unbiased figures of merit.

Acquisition budget (every camera frame counts)
----------------------------------------------
    total = averaging * (1 + iterations + 2 * validation_count)
An optional max_acquisitions is checked BEFORE any acquisition.

Randomness
----------
The optimizer's proposals come from np.random.default_rng(seed); the camera
noise comes from the bench's own seeded generator. They are independent;
freshly built identical benches + identical settings reproduce the full result.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from hw2.simulated_bench import (
    NoiseModel,
    Region,
    SimulatedBench,
    region_metrics,
    surrounding_background_regions,
)

__all__ = [
    "BenchAccess",
    "OptimizerSettings",
    "OptimizationResult",
    "expand_pattern",
    "grouped_shape",
    "optimize_focus",
    "planned_acquisitions",
]

ABSOLUTE_FLOOR = 1e-12


def _int_at_least(name: str, value, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


class BenchAccess:
    """The only view of the optical system the optimizer uses (no access to T)."""

    def __init__(self, bench):
        self._bench = bench
        self.dmd_shape = tuple(int(n) for n in bench.dmd.shape)
        self.camera_shape = tuple(int(n) for n in bench.camera_shape)
        self.frames = 0

    def acquire(self, pattern) -> np.ndarray:
        self.frames += 1
        return self._bench.acquire(pattern)

    def transmission_fingerprint(self) -> str:
        return self._bench.transmission_fingerprint()

    @property
    def num_acquisitions(self) -> int:
        return self._bench.num_acquisitions


def grouped_shape(dmd_shape, grouping_factor: int) -> tuple[int, int]:
    g = _int_at_least("grouping_factor", grouping_factor, 1)
    rows, cols = dmd_shape
    if rows % g or cols % g:
        raise ValueError(f"grouping_factor {g} does not divide the DMD grid {tuple(dmd_shape)}")
    return rows // g, cols // g


def expand_pattern(grouped, grouping_factor: int) -> np.ndarray:
    """Each macropixel value fills its g x g block of underlying DMD pixels."""
    g = _int_at_least("grouping_factor", grouping_factor, 1)
    grouped = np.asarray(grouped)
    if grouped.ndim != 2 or grouped.dtype != bool:
        raise ValueError("grouped pattern must be a 2-D boolean array")
    return np.repeat(np.repeat(grouped, g, axis=0), g, axis=1)


@dataclass(frozen=True)
class OptimizerSettings:
    grouping_factor: int = 1
    iterations: int = 500
    flips_per_proposal: int = 1
    focus: Region | None = None
    backgrounds: tuple = ()
    seed: int = 0
    initial_pattern: np.ndarray | None = field(default=None, compare=False)
    averaging: int = 1
    validation_count: int = 5
    background_floor_fraction: float = 0.05
    max_acquisitions: int | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["focus"] = asdict(self.focus) if self.focus else None
        d["backgrounds"] = [asdict(b) for b in self.backgrounds]
        d["initial_pattern"] = None if self.initial_pattern is None else np.asarray(self.initial_pattern).astype(int).tolist()
        return d


def planned_acquisitions(settings: OptimizerSettings) -> int:
    return settings.averaging * (1 + settings.iterations + 2 * settings.validation_count)


def _validate(settings: OptimizerSettings, access: BenchAccess) -> tuple[tuple[int, int], np.ndarray | None]:
    shape = grouped_shape(access.dmd_shape, settings.grouping_factor)
    n = shape[0] * shape[1]
    _int_at_least("iterations", settings.iterations, 0)
    _int_at_least("averaging", settings.averaging, 1)
    _int_at_least("validation_count", settings.validation_count, 1)
    _int_at_least("seed", settings.seed, 0)
    flips = _int_at_least("flips_per_proposal", settings.flips_per_proposal, 1)
    if flips > n:
        raise ValueError(f"flips_per_proposal {flips} exceeds the {n} macropixels")
    fraction = settings.background_floor_fraction
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction < 1:
        raise ValueError(f"background_floor_fraction must be in (0, 1), got {fraction!r}")
    if settings.focus is None:
        raise ValueError("a focus region is required")
    # Raises ValueError for bad, overlapping or out-of-image regions.
    region_metrics(np.zeros(access.camera_shape), settings.focus, settings.backgrounds)
    if settings.max_acquisitions is not None:
        budget = _int_at_least("max_acquisitions", settings.max_acquisitions, 1)
        if planned_acquisitions(settings) > budget:
            raise ValueError(f"the settings need {planned_acquisitions(settings)} acquisitions, "
                             f"more than max_acquisitions={budget}")
    initial = None
    if settings.initial_pattern is not None:
        p = np.asarray(settings.initial_pattern)
        if p.shape != shape:
            raise ValueError(f"initial_pattern must have the grouped shape {shape}, got {p.shape}")
        if p.dtype != bool and not np.all((p == 0) | (p == 1)):
            raise ValueError("initial_pattern must be binary")
        initial = p.astype(bool)
    return shape, initial


@dataclass
class OptimizationResult:
    settings: dict
    grouped_shape: tuple
    initial_pattern: np.ndarray            # grouped, bool
    final_pattern: np.ndarray              # grouped, bool
    final_pattern_expanded: np.ndarray     # underlying DMD grid, bool
    objective_background_floor: float
    initial_enhancement: float             # search measurement of p0
    best_observed_enhancement: float       # max accepted (single noisy measurement, biased up)
    final_search_enhancement: float        # accepted value at the end of the search
    initial_validation: list               # validation_count enhancements of p0
    final_validation: list                 # validation_count enhancements of the final pattern
    initial_image: np.ndarray              # mean of the p0 validation frames
    final_image: np.ndarray                # mean of the final-pattern validation frames
    enhancement_history: list              # candidate objective per iteration
    accepted_history: list                 # bool per iteration
    accepted_objective_history: list       # eta_acc after step 0..iterations
    proposal_history: list                 # flipped macropixel indices per iteration
    acquisitions: dict
    transmission_fingerprint: str
    runtime_s: float

    @property
    def validated_initial_enhancement(self) -> float:
        return float(np.mean(self.initial_validation))

    @property
    def validated_final_enhancement(self) -> float:
        return float(np.mean(self.final_validation))

    @property
    def accepted_count(self) -> int:
        return int(sum(self.accepted_history))

    def summary(self) -> dict:
        return {
            "grouped_shape": list(self.grouped_shape),
            "initial_enhancement": self.initial_enhancement,
            "best_observed_enhancement": self.best_observed_enhancement,
            "final_search_enhancement": self.final_search_enhancement,
            "validated_initial_enhancement": self.validated_initial_enhancement,
            "validated_initial_enhancement_std": float(np.std(self.initial_validation)),
            "validated_final_enhancement": self.validated_final_enhancement,
            "validated_final_enhancement_std": float(np.std(self.final_validation)),
            "accepted_count": self.accepted_count,
            "iterations": len(self.accepted_history),
            "acquisitions": self.acquisitions,
            "objective_background_floor": self.objective_background_floor,
            "transmission_fingerprint": self.transmission_fingerprint,
            "runtime_s": self.runtime_s,
        }


def optimize_focus(bench, settings: OptimizerSettings) -> OptimizationResult:
    """Run the black-box binary flipping search on `bench` (see module docs)."""
    started = time.perf_counter()
    access = BenchAccess(bench)
    shape, initial = _validate(settings, access)
    g = settings.grouping_factor
    rng = np.random.default_rng(settings.seed)
    fingerprint = access.transmission_fingerprint()

    def frames_of(pattern: np.ndarray) -> np.ndarray:
        expanded = expand_pattern(pattern, g)
        return np.mean([access.acquire(expanded) for _ in range(settings.averaging)], axis=0)

    def enhancement(image: np.ndarray, floor: float) -> float:
        return region_metrics(image, settings.focus, settings.backgrounds, floor).enhancement

    # 1-2. initial pattern and its measurement; the floor is fixed from it.
    p0 = initial if initial is not None else rng.random(shape) < 0.5
    image0 = frames_of(p0)
    first = region_metrics(image0, settings.focus, settings.backgrounds, ABSOLUTE_FLOOR)
    floor = max(settings.background_floor_fraction * first.background_mean, ABSOLUTE_FLOOR)
    eta0 = enhancement(image0, floor)

    accepted, eta_acc = p0.copy(), eta0
    history, accepted_flags, accepted_values, proposals = [], [], [eta0], []
    n = shape[0] * shape[1]

    # 3. iterative flipping
    for _ in range(settings.iterations):
        idx = rng.choice(n, size=settings.flips_per_proposal, replace=False)
        candidate = accepted.copy().ravel()
        candidate[idx] = ~candidate[idx]
        candidate = candidate.reshape(shape)
        eta = enhancement(frames_of(candidate), floor)
        take = eta > eta_acc
        if take:
            accepted, eta_acc = candidate, eta
        history.append(eta)
        accepted_flags.append(bool(take))
        accepted_values.append(eta_acc)
        proposals.append(sorted(int(i) for i in idx))
    search_frames = access.frames

    # 4. independent validation of the initial and final patterns
    def validate(pattern: np.ndarray) -> tuple[list, np.ndarray]:
        images = [frames_of(pattern) for _ in range(settings.validation_count)]
        return [enhancement(im, floor) for im in images], np.mean(images, axis=0)

    initial_validation, initial_image = validate(p0)
    final_validation, final_image = validate(accepted)

    if access.transmission_fingerprint() != fingerprint:
        raise RuntimeError("the transmission matrix changed during the experiment")
    return OptimizationResult(
        settings=settings.to_dict(),
        grouped_shape=shape,
        initial_pattern=p0,
        final_pattern=accepted,
        final_pattern_expanded=expand_pattern(accepted, g),
        objective_background_floor=floor,
        initial_enhancement=eta0,
        best_observed_enhancement=float(max(accepted_values)),
        final_search_enhancement=eta_acc,
        initial_validation=initial_validation,
        final_validation=final_validation,
        initial_image=initial_image,
        final_image=final_image,
        enhancement_history=history,
        accepted_history=accepted_flags,
        accepted_objective_history=accepted_values,
        proposal_history=proposals,
        acquisitions={
            "search": search_frames,
            "validation": access.frames - search_frames,
            "total": access.frames,
            "planned": planned_acquisitions(settings),
            "frames_per_measurement": settings.averaging,
        },
        transmission_fingerprint=fingerprint,
        runtime_s=time.perf_counter() - started,
    )


# -----------------------------
# Small offline demonstration
# -----------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hw2.focusing_optimizer",
                                     description="Small offline simulated focusing experiment.")
    parser.add_argument("--dmd", type=int, default=32, help="Underlying DMD grid size (square).")
    parser.add_argument("--camera", type=int, default=64, help="Camera size (square).")
    parser.add_argument("--grouping", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--flips", type=int, default=1)
    parser.add_argument("--averaging", type=int, default=1)
    parser.add_argument("--validation", type=int, default=10)
    parser.add_argument("--bench-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1, help="Optimizer seed.")
    parser.add_argument("--no-noise", action="store_true")
    args = parser.parse_args(argv)

    c = args.camera // 2
    focus = Region(c - 1, c + 1, c - 1, c + 1, name="focus")
    backgrounds = surrounding_background_regions(focus, (args.camera, args.camera), gap=2, width=4)
    noise = NoiseModel(enabled=not args.no_noise)
    bench = SimulatedBench((args.dmd, args.dmd), (args.camera, args.camera), focus, backgrounds,
                           seed=args.bench_seed, noise=noise)
    settings = OptimizerSettings(
        grouping_factor=args.grouping, iterations=args.iterations, flips_per_proposal=args.flips,
        focus=focus, backgrounds=backgrounds, seed=args.seed, averaging=args.averaging,
        validation_count=args.validation,
    )
    result = optimize_focus(bench, settings)
    print(json.dumps({"bench": {"dmd": [args.dmd] * 2, "camera": [args.camera] * 2,
                                "bench_seed": args.bench_seed, "noise": asdict(noise)},
                      "settings": settings.to_dict(), "result": result.summary()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
