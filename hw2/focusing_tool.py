"""
HW2 Task 2 — focusing experiment tool (session + artifact contract).

A FocusingSession owns ONE simulated bench (laser -> binary DMD -> fixed
transmission matrix T -> noisy camera) for its whole lifetime:

  * DMD/camera size, medium seed and noise model are fixed at construction;
  * T is drawn once and reused by every experiment of the session;
  * the camera noise generator keeps running, so every new acquisition has
    fresh, independent noise;
  * T is never returned or saved; only its SHA-256 fingerprint is reported.

Each call of run_experiment(arguments) runs hw2.focusing_optimizer.optimize_focus
with settings that are ALL supplied by the caller (the future LLM executor):
grouping factor, focus rectangle, background gap/width, iterations, flips per
proposal, averaging, validation count, seed and acquisition budget. Nothing is
chosen or silently reduced here; invalid or over-budget arguments raise
FocusingToolError (call() returns it as {"ok": false, "error": ...}).

Metrics (all from the optimizer's independent validation measurements)
--------------------------------------------------------------------
enhancement eta = <I>_F / max(<I>_B, floor), as in hw2.simulated_bench /
hw2.focusing_optimizer (unchanged objective). For the initial and the final
pattern the tool reports:
  * validated enhancement: mean / std / values over validation_count measurements;
  * focus mean intensity <I>_F and background mean intensity <I>_B of the
    mean validation image. Region means are linear in the image, so these
    equal the means over the individual validation frames.
Derived:
  * enhancement_gain             = validated final eta / validated initial eta
  * focus_intensity_gain         = final <I>_F / initial <I>_F   (absolute brightening)
  * background_suppression_factor = initial <I>_B / final <I>_B
For the mean validation images (no floor active) the image-level enhancement
gain factorises exactly:  gain = focus_intensity_gain * background_suppression_factor,
and the tool reports which share of log(gain) comes from each factor. A large
enhancement can come mostly from background suppression; the focus itself is
then much less brighter than the enhancement suggests.
A ratio whose denominator is near zero (below NEAR_ZERO_RELATIVE times the
initial mean image intensity) is reported as null with an explicit note.

Search values (single noisy measurements that won comparisons, biased upward)
are reported separately under "search" and are never used as final results.

Artifacts (one new directory per run under the session's artifact root; files
are created exclusively and never overwritten):
  settings.json, result.json, initial_camera.png, final_camera.png,
  enhancement_history.png, final_dmd_pattern.png, history.npz
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import uuid
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from hw2.focusing_optimizer import OptimizerSettings, optimize_focus, planned_acquisitions
from hw2.simulated_bench import (
    NoiseModel,
    Region,
    SimulatedBench,
    region_metrics,
    surrounding_background_regions,
)

__all__ = [
    "ExperimentArguments",
    "FocusingSession",
    "FocusingToolError",
    "validated_intensity_metrics",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "task6_runs" / "hw2_focusing"   # git-ignored working data
DEFAULT_SESSION_ACQUISITION_LIMIT = 20_000
MAX_T_ELEMENTS = 1 << 23            # 8.4 M complex entries (~134 MB): laptop-safe medium size
NEAR_ZERO_RELATIVE = 1e-6
RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
ARTIFACT_FILES = {
    "settings": "settings.json",
    "result": "result.json",
    "initial_camera_png": "initial_camera.png",
    "final_camera_png": "final_camera.png",
    "enhancement_history_png": "enhancement_history.png",
    "final_dmd_pattern_png": "final_dmd_pattern.png",
    "history_npz": "history.npz",
}


class FocusingToolError(ValueError):
    """A rejected tool call (invalid argument, budget violation, path problem)."""


def _int(name: str, value, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise FocusingToolError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise FocusingToolError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


# -----------------------------
# Tool arguments
# -----------------------------

@dataclass(frozen=True)
class ExperimentArguments:
    """Every experiment setting is explicit; there are no defaults."""

    grouping_factor: int        # DMD pixels per macropixel side
    focus_row: int              # top-left pixel of the focus rectangle
    focus_col: int
    focus_height: int
    focus_width: int
    background_gap: int         # pixels between focus and the proximal background strips
    background_width: int       # thickness of each background strip
    iterations: int             # proposals (>= 1)
    flips_per_proposal: int
    averaging: int              # camera frames per measurement
    validation_count: int       # independent validation measurements per pattern
    seed: int                   # optimizer proposal seed
    max_acquisitions: int       # camera-frame budget for this call

    MINIMUMS = {"focus_row": 0, "focus_col": 0, "background_gap": 0, "seed": 0}

    @classmethod
    def from_dict(cls, data) -> "ExperimentArguments":
        if not isinstance(data, dict):
            raise FocusingToolError(f"arguments must be a JSON object, got {type(data).__name__}")
        names = [f.name for f in fields(cls)]
        unknown = sorted(set(data) - set(names) - {"run_name"})
        missing = [n for n in names if n not in data]
        if unknown:
            raise FocusingToolError(f"unknown argument(s): {unknown}")
        if missing:
            raise FocusingToolError(f"missing argument(s): {missing}")
        return cls(**{n: _int(n, data[n], cls.MINIMUMS.get(n, 1)) for n in names})

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def json_schema(cls) -> dict:
        props = {f.name: {"type": "integer", "minimum": cls.MINIMUMS.get(f.name, 1)} for f in fields(cls)}
        props["run_name"] = {"type": "string", "pattern": RUN_NAME.pattern,
                             "description": "optional new run directory name under the artifact root"}
        return {"type": "object", "properties": props, "required": [f.name for f in fields(cls)],
                "additionalProperties": False}


# -----------------------------
# Metrics
# -----------------------------

def _ratio(numerator: float, denominator: float, scale: float, what: str, notes: list) -> float | None:
    if denominator <= NEAR_ZERO_RELATIVE * scale or denominator <= 0:
        notes.append(f"{what} not reported: its denominator {denominator:.3e} is near zero "
                     f"(<= {NEAR_ZERO_RELATIVE:g} x initial mean image intensity {scale:.3e})")
        return None
    return numerator / denominator


def validated_intensity_metrics(initial_image, final_image, focus: Region, backgrounds,
                                floor: float, initial_values, final_values) -> dict:
    """Validated metrics from the mean validation images and validation enhancements."""
    notes: list[str] = []
    start = region_metrics(initial_image, focus, backgrounds, floor)
    end = region_metrics(final_image, focus, backgrounds, floor)
    scale = max(float(np.mean(initial_image)), 1e-300)

    def block(m, values):
        return {
            "enhancement_mean": float(np.mean(values)),
            "enhancement_std": float(np.std(values)),
            "enhancement_values": [float(v) for v in values],
            "focus_mean_intensity": m.focus_mean,
            "background_mean_intensity": m.background_mean,
            "background_region_mean_intensities": list(m.background_region_means),
            "background_floor_active": m.background_floored,
        }

    initial, final = block(start, initial_values), block(end, final_values)
    enhancement_gain = _ratio(final["enhancement_mean"], initial["enhancement_mean"], 1.0,
                              "enhancement_gain", notes)
    focus_gain = _ratio(end.focus_mean, start.focus_mean, scale, "focus_intensity_gain", notes)
    suppression = _ratio(start.background_mean, end.background_mean, scale,
                         "background_suppression_factor", notes)

    decomposition = None
    if focus_gain and suppression and not (start.background_floored or end.background_floored):
        image_gain = focus_gain * suppression
        if image_gain > 1 and not math.isclose(image_gain, 1.0):
            log_gain = math.log(image_gain)
            decomposition = {
                "basis": "mean validation images: gain = focus_intensity_gain * background_suppression_factor",
                "image_enhancement_gain": image_gain,
                "share_of_log_gain_from_focus_brightening": math.log(focus_gain) / log_gain,
                "share_of_log_gain_from_background_suppression": math.log(suppression) / log_gain,
            }
    if decomposition is None:
        notes.append("decomposition not reported (no image-level gain above 1, a ratio is "
                     "undefined, or the background floor was active)")
    return {
        "initial": initial,
        "final": final,
        "enhancement_gain": enhancement_gain,
        "focus_intensity_gain": focus_gain,
        "background_suppression_factor": suppression,
        "decomposition": decomposition,
        "notes": notes,
    }


# -----------------------------
# Artifacts
# -----------------------------

def _create(path: Path, mode: str = "x"):
    try:
        return open(path, mode, **({"encoding": "utf-8"} if "b" not in mode else {}))
    except FileExistsError:
        raise FocusingToolError(f"refusing to overwrite existing artifact {path}") from None


def _write_json(path: Path, data: dict) -> None:
    with _create(path) as fh:
        json.dump(data, fh, indent=2)


def _save_figure(fig: Figure, path: Path) -> None:
    FigureCanvasAgg(fig)
    with _create(path, "xb") as fh:
        fig.savefig(fh, format="png", dpi=110)


def _camera_png(path: Path, image, title: str, vmax: float, focus: Region, backgrounds) -> None:
    fig = Figure(figsize=(5, 4.3))
    ax = fig.add_subplot()
    shown = ax.imshow(image, cmap="inferno", vmin=0.0, vmax=vmax, interpolation="nearest")
    fig.colorbar(shown, ax=ax, label="intensity (simulated units, shared scale)")
    for region, colour in [(focus, "cyan")] + [(b, "white") for b in backgrounds]:
        ax.add_patch(Rectangle((region.col_start - 0.5, region.row_start - 0.5),
                               region.col_stop - region.col_start, region.row_stop - region.row_start,
                               fill=False, edgecolor=colour, linewidth=0.8))
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("camera column")
    ax.set_ylabel("camera row")
    _save_figure(fig, path)


def _history_png(path: Path, result) -> None:
    fig = Figure(figsize=(6.5, 4))
    ax = fig.add_subplot()
    steps = np.arange(1, len(result.enhancement_history) + 1)
    ax.plot(steps, result.enhancement_history, ".", markersize=2, color="0.6",
            label="candidate (single noisy measurement)")
    ax.plot(np.arange(len(result.accepted_objective_history)), result.accepted_objective_history,
            color="tab:blue", linewidth=1.2, label="accepted (search value, biased up)")
    for values, colour, name in ((result.initial_validation, "tab:orange", "validated initial"),
                                 (result.final_validation, "tab:green", "validated final")):
        ax.axhline(float(np.mean(values)), color=colour, linestyle="--", linewidth=1,
                   label=f"{name} mean = {np.mean(values):.3g}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("enhancement  <I>_F / max(<I>_B, floor)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    _save_figure(fig, path)


def _pattern_png(path: Path, expanded, grouped_shape, grouping: int) -> None:
    fig = Figure(figsize=(4.5, 4.5))
    ax = fig.add_subplot()
    ax.imshow(expanded, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(f"final DMD pattern ({expanded.shape[0]}x{expanded.shape[1]} pixels, "
                 f"{grouped_shape[0]}x{grouped_shape[1]} macropixels of {grouping}x{grouping}; white = on)",
                 fontsize=7)
    ax.set_xlabel("DMD column")
    ax.set_ylabel("DMD row")
    _save_figure(fig, path)


# -----------------------------
# Session
# -----------------------------

class FocusingSession:
    """One fixed simulated medium; many experiments. T is never exposed."""

    def __init__(self, dmd_shape=(32, 32), camera_shape=(64, 64), medium_seed: int = 0,
                 noise: dict | None = None, artifact_root=DEFAULT_ARTIFACT_ROOT,
                 acquisition_limit: int = DEFAULT_SESSION_ACQUISITION_LIMIT):
        try:
            dmd = tuple(_int("dmd_shape", n, 1) for n in dmd_shape)
            cam = tuple(_int("camera_shape", n, 1) for n in camera_shape)
        except TypeError:
            raise FocusingToolError("dmd_shape and camera_shape must be (rows, cols) pairs") from None
        if len(dmd) != 2 or len(cam) != 2:
            raise FocusingToolError("dmd_shape and camera_shape must be (rows, cols) pairs")
        if dmd[0] * dmd[1] * cam[0] * cam[1] > MAX_T_ELEMENTS:
            raise FocusingToolError(f"medium too large: {dmd} DMD x {cam} camera exceeds "
                                    f"{MAX_T_ELEMENTS} transmission-matrix entries")
        self.medium_seed = _int("medium_seed", medium_seed, 0)
        self.acquisition_limit = _int("acquisition_limit", acquisition_limit, 1)
        noise = dict(noise or {})
        try:
            self.noise = NoiseModel(**noise)
        except (TypeError, ValueError) as exc:
            raise FocusingToolError(f"invalid noise model {noise!r}: {exc}") from None
        self.dmd_shape, self.camera_shape = dmd, cam
        self.artifact_root = Path(artifact_root).resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)

        # The bench needs reference regions at construction; the tool never uses
        # them: every experiment evaluates the regions given in its arguments.
        ref = Region(cam[0] // 2, cam[0] // 2 + 1, cam[1] // 2, cam[1] // 2 + 1)
        try:
            ref_backgrounds = surrounding_background_regions(ref, cam, gap=0, width=1)
            self._bench = SimulatedBench(dmd, cam, ref, ref_backgrounds,
                                         seed=self.medium_seed, noise=self.noise)
        except ValueError as exc:
            raise FocusingToolError(f"cannot build the simulated bench: {exc}") from None
        self._fingerprint = self._bench.transmission_fingerprint()
        self.runs: list[str] = []

    # ---- read-only description (safe for the agent)

    def describe(self) -> dict:
        return {
            "dmd_shape": list(self.dmd_shape),
            "camera_shape": list(self.camera_shape),
            "medium_seed": self.medium_seed,
            "noise": asdict(self.noise),
            "transmission_fingerprint": self._fingerprint,
            "acquisition_limit_per_call": self.acquisition_limit,
            "camera_acquisitions_so_far": self._bench.num_acquisitions,
            "artifact_root": self.artifact_root.as_posix(),
            "valid_grouping_factors": [g for g in range(1, min(self.dmd_shape) + 1)
                                       if self.dmd_shape[0] % g == 0 and self.dmd_shape[1] % g == 0],
            "argument_schema": ExperimentArguments.json_schema(),
        }

    # ---- validation

    def _regions(self, a: ExperimentArguments) -> tuple[Region, tuple[Region, ...]]:
        rows, cols = self.camera_shape
        if a.focus_row + a.focus_height > rows or a.focus_col + a.focus_width > cols:
            raise FocusingToolError(f"focus rectangle rows {a.focus_row}..{a.focus_row + a.focus_height - 1}, "
                                    f"cols {a.focus_col}..{a.focus_col + a.focus_width - 1} is outside the "
                                    f"{rows}x{cols} camera")
        focus = Region(a.focus_row, a.focus_row + a.focus_height, a.focus_col,
                       a.focus_col + a.focus_width, name="focus")
        try:
            backgrounds = surrounding_background_regions(focus, self.camera_shape,
                                                         gap=a.background_gap, width=a.background_width)
        except ValueError as exc:
            raise FocusingToolError(str(exc)) from None
        return focus, backgrounds

    def _settings(self, a: ExperimentArguments, focus, backgrounds) -> OptimizerSettings:
        if self.dmd_shape[0] % a.grouping_factor or self.dmd_shape[1] % a.grouping_factor:
            raise FocusingToolError(f"grouping_factor {a.grouping_factor} does not divide the DMD "
                                    f"{self.dmd_shape}; valid: {self.describe()['valid_grouping_factors']}")
        if a.max_acquisitions > self.acquisition_limit:
            raise FocusingToolError(f"max_acquisitions {a.max_acquisitions} exceeds the session limit "
                                    f"{self.acquisition_limit}")
        settings = OptimizerSettings(
            grouping_factor=a.grouping_factor, iterations=a.iterations,
            flips_per_proposal=a.flips_per_proposal, focus=focus, backgrounds=backgrounds,
            seed=a.seed, averaging=a.averaging, validation_count=a.validation_count,
            max_acquisitions=a.max_acquisitions,
        )
        needed = planned_acquisitions(settings)
        if needed > a.max_acquisitions:
            raise FocusingToolError(
                f"these settings need {needed} camera acquisitions = averaging {a.averaging} x "
                f"(1 + iterations {a.iterations} + 2 x validation_count {a.validation_count}), "
                f"more than max_acquisitions {a.max_acquisitions}")
        n_macro = (self.dmd_shape[0] // a.grouping_factor) * (self.dmd_shape[1] // a.grouping_factor)
        if a.flips_per_proposal > n_macro:
            raise FocusingToolError(f"flips_per_proposal {a.flips_per_proposal} exceeds the {n_macro} macropixels")
        return settings

    def _run_dir(self, run_name) -> Path:
        if run_name is None:
            run_name = f"focus_run_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
        if not isinstance(run_name, str) or not RUN_NAME.match(run_name):
            raise FocusingToolError(f"run_name must match {RUN_NAME.pattern}, got {run_name!r}")
        run_dir = (self.artifact_root / run_name).resolve()
        if run_dir.parent != self.artifact_root:
            raise FocusingToolError(f"run directory {run_dir} is outside {self.artifact_root}")
        try:
            run_dir.mkdir()
        except FileExistsError:
            raise FocusingToolError(f"run directory {run_dir} already exists; results are never overwritten") from None
        return run_dir

    # ---- the tool

    def run_experiment(self, arguments) -> dict:
        """Validate, run one focusing experiment, save artifacts, return JSON-compatible results."""
        started = time.perf_counter()
        if isinstance(arguments, ExperimentArguments):
            args, run_name = arguments, None
        else:
            args = ExperimentArguments.from_dict(arguments)
            run_name = arguments.get("run_name")
        focus, backgrounds = self._regions(args)
        settings = self._settings(args, focus, backgrounds)
        try:
            region_metrics(np.zeros(self.camera_shape), focus, backgrounds)
        except ValueError as exc:
            raise FocusingToolError(str(exc)) from None

        run_dir = self._run_dir(run_name)          # only after every check passed
        before = self._bench.num_acquisitions
        result = optimize_focus(self._bench, settings)
        frames = self._bench.num_acquisitions - before
        if result.transmission_fingerprint != self._fingerprint:
            raise RuntimeError("the simulated medium changed within the session")

        metrics = validated_intensity_metrics(
            result.initial_image, result.final_image, focus, backgrounds,
            result.objective_background_floor, result.initial_validation, result.final_validation)
        paths = {key: (run_dir / name).as_posix() for key, name in ARTIFACT_FILES.items()}
        regions = {"focus": asdict(focus), "backgrounds": [asdict(b) for b in backgrounds]}
        session = {k: v for k, v in self.describe().items() if k not in ("argument_schema",)}

        _write_json(run_dir / ARTIFACT_FILES["settings"], {
            "tool_arguments": args.to_dict(), "run_name": run_dir.name, "regions": regions,
            "optimizer_settings": settings.to_dict(), "session": session,
        })
        vmax = float(max(result.initial_image.max(), result.final_image.max(), 1e-12))
        _camera_png(run_dir / ARTIFACT_FILES["initial_camera_png"], result.initial_image,
                    f"initial pattern: mean of {args.validation_count} validation measurements",
                    vmax, focus, backgrounds)
        _camera_png(run_dir / ARTIFACT_FILES["final_camera_png"], result.final_image,
                    f"final pattern: mean of {args.validation_count} validation measurements",
                    vmax, focus, backgrounds)
        _history_png(run_dir / ARTIFACT_FILES["enhancement_history_png"], result)
        _pattern_png(run_dir / ARTIFACT_FILES["final_dmd_pattern_png"], result.final_pattern_expanded,
                     result.grouped_shape, args.grouping_factor)
        with _create(run_dir / ARTIFACT_FILES["history_npz"], "xb") as fh:
            np.savez(fh,
                     enhancement_history=np.asarray(result.enhancement_history),
                     accepted_history=np.asarray(result.accepted_history, dtype=bool),
                     accepted_objective_history=np.asarray(result.accepted_objective_history),
                     proposal_history=np.asarray(result.proposal_history, dtype=np.int64),
                     initial_validation=np.asarray(result.initial_validation),
                     final_validation=np.asarray(result.final_validation),
                     initial_pattern=result.initial_pattern, final_pattern=result.final_pattern,
                     final_pattern_expanded=result.final_pattern_expanded,
                     initial_image=result.initial_image, final_image=result.final_image)

        output = {
            "ok": True,
            "run_name": run_dir.name,
            "run_dir": run_dir.as_posix(),
            "session": session,
            "arguments": args.to_dict(),
            "regions": regions,
            "objective": {
                "definition": "enhancement = <I>_F / max(<I>_B, floor); <I>_B pixel-weighted over all "
                              "background strips; floor = 0.05 x <I>_B of the initial search measurement",
                "background_floor": result.objective_background_floor,
            },
            "validated": metrics,
            "search": {
                "note": "single noisy measurements that won comparisons; biased upward; not final results",
                "initial_enhancement_single_measurement": result.initial_enhancement,
                "best_observed_enhancement": result.best_observed_enhancement,
                "final_accepted_enhancement": result.final_search_enhancement,
                "accepted_proposals": result.accepted_count,
                "iterations": len(result.accepted_history),
            },
            "acquisitions": {**result.acquisitions, "counted_by_bench": frames},
            "transmission_fingerprint": result.transmission_fingerprint,
            "runtime_s": time.perf_counter() - started,
            "artifacts": paths,
        }
        _write_json(run_dir / ARTIFACT_FILES["result"], output)
        self.runs.append(run_dir.name)
        return output

    def call(self, arguments) -> dict:
        """Agent-facing wrapper: never raises for a rejected call."""
        try:
            return self.run_experiment(arguments)
        except FocusingToolError as exc:
            return {"ok": False, "error": str(exc)}


# -----------------------------
# Small offline demonstration
# -----------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hw2.focusing_tool",
                                     description="One bounded simulated focusing experiment through the tool.")
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--arguments", default=None,
                        help="JSON object with all tool arguments (default: the demo values below).")
    args = parser.parse_args(argv)
    session = FocusingSession(dmd_shape=(32, 32), camera_shape=(64, 64), medium_seed=0,
                              noise={"enabled": True, "photons_per_unit": 1000.0, "read_noise_std": 0.01},
                              artifact_root=args.artifact_root)
    arguments = json.loads(args.arguments) if args.arguments else {
        "grouping_factor": 2, "focus_row": 31, "focus_col": 31, "focus_height": 2, "focus_width": 2,
        "background_gap": 2, "background_width": 4, "iterations": 1000, "flips_per_proposal": 1,
        "averaging": 1, "validation_count": 10, "seed": 1, "max_acquisitions": 1100,
    }
    print(json.dumps(session.call(arguments), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
