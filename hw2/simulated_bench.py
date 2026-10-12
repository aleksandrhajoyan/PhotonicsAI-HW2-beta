"""
HW2 Task 2 — simulated optical bench (core simulator only).

Optical path modelled:   laser -> DMD -> random scattering medium (RSM) -> camera

The real setup has further optomechanical elements (lenses, mirrors, relay
optics, polarisation optics). This simulator does NOT model them individually:
the whole image-forming path from the DMD plane to the camera pixels is
lumped into ONE complex transmission matrix T, as in the standard
transmission-matrix description of light propagation through a scattering
medium.

Model
-----
DMD:    a grid of R x C binary macropixels. A pattern p in {0, 1}^(R*C)
        switches macropixels on/off; the DMD modulates AMPLITUDE ONLY:
            E_in = a * p            (real, nonnegative; a = illumination amplitude)
        The incident laser field is taken as uniform, real amplitude a and
        zero phase over the DMD. No phase modulation is applied anywhere.
Medium: T is a complex (H*W) x (R*C) matrix. By default its entries are i.i.d.
        circularly-symmetric complex Gaussian with E|T_mn|^2 = 1/(R*C), so an
        all-on pattern gives a mean ideal intensity of about a^2 per camera
        pixel (fully developed speckle). T is drawn ONCE per bench (per
        simulated experiment) from the seeded generator and is never redrawn;
        the array is read-only. A user-supplied T can be given for validation.
Camera: ideal intensity of H x W pixels
            I = |T @ E_in|^2        (reshaped to H x W)
        followed by the noise model below.

Noise model (explicit assumptions)
----------------------------------
Each acquisition draws NEW, independent noise from the bench's seeded noise
generator (separate from the generator that drew T):
  * shot noise: photon counts ~ Poisson(I * photons_per_unit), converted back
    to intensity units by dividing by photons_per_unit (disabled if None);
  * read noise: additive zero-mean Gaussian, standard deviation read_noise_std
    (intensity units), independent per pixel and per frame;
  * the result is clipped at 0, because a camera reports nonnegative values
    (this introduces a small positive bias when the signal is near zero).
No dark current, saturation, quantisation, pixel crosstalk, laser power drift
or medium decorrelation is modelled. With noise disabled the camera returns the
ideal intensity exactly.

Regions and enhancement
-----------------------
Regions are half-open pixel rectangles [row_start, row_stop) x [col_start, col_stop).
One focus region F and one or more background regions B_1..B_K; all must lie
inside the image and must not overlap each other or F.

    focus mean            <I>_F   = mean of I over the pixels of F
    background region k   <I>_Bk  = mean of I over the pixels of B_k
    aggregate background  <I>_B   = mean of I over ALL pixels of B_1 U ... U B_K
                                    (pixel-weighted, not the mean of the <I>_Bk)
    focus enhancement     eta     = <I>_F / max(<I>_B, background_floor)

background_floor (> 0) prevents division by a near-zero background; when it is
used, the measurement reports background_floored = True.

Access rule
-----------
Optimisation code should use only the measurement interface (SimulatedBench.measure /
acquire and the shape properties). T is kept in a private, read-only attribute;
transmission_fingerprint() allows checking that T stayed fixed without exposing it.

This module depends on NumPy and the standard library only (no LangGraph, LLM,
MCP or device drivers).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace

import numpy as np

__all__ = [
    "Region",
    "NoiseModel",
    "DMD",
    "Measurement",
    "SimulatedBench",
    "region_metrics",
    "surrounding_background_regions",
]


def _positive_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _shape2(name: str, shape) -> tuple[int, int]:
    try:
        rows, cols = shape
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a (rows, cols) pair, got {shape!r}") from None
    return _positive_int(f"{name}[0]", rows), _positive_int(f"{name}[1]", cols)


def _finite_number(name: str, value, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if positive and value <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return value


# -----------------------------
# Regions
# -----------------------------

@dataclass(frozen=True)
class Region:
    """Half-open pixel rectangle [row_start, row_stop) x [col_start, col_stop)."""

    row_start: int
    row_stop: int
    col_start: int
    col_stop: int
    name: str = ""

    def __post_init__(self):
        for attr in ("row_start", "row_stop", "col_start", "col_stop"):
            value = getattr(self, attr)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"Region.{attr} must be an integer, got {value!r}")
        if self.row_start < 0 or self.col_start < 0:
            raise ValueError(f"Region {self.name!r} has a negative start: {self}")
        if self.row_stop <= self.row_start or self.col_stop <= self.col_start:
            raise ValueError(f"Region {self.name!r} is empty: {self}")

    @property
    def num_pixels(self) -> int:
        return (self.row_stop - self.row_start) * (self.col_stop - self.col_start)

    @property
    def slices(self) -> tuple[slice, slice]:
        return slice(self.row_start, self.row_stop), slice(self.col_start, self.col_stop)

    def overlaps(self, other: "Region") -> bool:
        return (self.row_start < other.row_stop and other.row_start < self.row_stop
                and self.col_start < other.col_stop and other.col_start < self.col_stop)

    def fits(self, shape: tuple[int, int]) -> bool:
        return self.row_stop <= shape[0] and self.col_stop <= shape[1]


def _validate_regions(shape: tuple[int, int], focus: Region, backgrounds) -> tuple[Region, ...]:
    if not isinstance(focus, Region):
        raise ValueError(f"focus must be a Region, got {focus!r}")
    backgrounds = tuple(backgrounds)
    if not backgrounds:
        raise ValueError("at least one background region is required")
    for region in (focus, *backgrounds):
        if not isinstance(region, Region):
            raise ValueError(f"background regions must be Region objects, got {region!r}")
        if not region.fits(shape):
            raise ValueError(f"region {region} does not fit inside the {shape} camera image")
    for i, a in enumerate(backgrounds):
        if a.overlaps(focus):
            raise ValueError(f"background region {a} overlaps the focus region {focus}")
        for b in backgrounds[i + 1:]:
            if a.overlaps(b):
                raise ValueError(f"background regions {a} and {b} overlap")
    return backgrounds


def surrounding_background_regions(focus: Region, camera_shape, gap: int = 1, width: int = 2) -> tuple[Region, ...]:
    """Proximal background rectangles above, below, left and right of the focus.

    Each lies `gap` pixels from the focus and is `width` pixels thick; the
    top/bottom ones span the focus columns and the left/right ones the focus
    rows, so they never overlap each other or the focus. Parts outside the
    image are clipped; empty ones are dropped.
    """
    rows, cols = _shape2("camera_shape", camera_shape)
    if isinstance(gap, bool) or not isinstance(gap, (int, np.integer)) or gap < 0:
        raise ValueError(f"gap must be a nonnegative integer, got {gap!r}")
    width = _positive_int("width", width)
    f = focus
    candidates = [
        ("top", f.row_start - gap - width, f.row_start - gap, f.col_start, f.col_stop),
        ("bottom", f.row_stop + gap, f.row_stop + gap + width, f.col_start, f.col_stop),
        ("left", f.row_start, f.row_stop, f.col_start - gap - width, f.col_start - gap),
        ("right", f.row_start, f.row_stop, f.col_stop + gap, f.col_stop + gap + width),
    ]
    regions = []
    for name, r0, r1, c0, c1 in candidates:
        r0, r1 = max(r0, 0), min(r1, rows)
        c0, c1 = max(c0, 0), min(c1, cols)
        if r1 > r0 and c1 > c0:
            regions.append(Region(int(r0), int(r1), int(c0), int(c1), name=name))
    if not regions:
        raise ValueError("no background region fits inside the image around this focus")
    return tuple(regions)


# -----------------------------
# Metrics
# -----------------------------

@dataclass(frozen=True)
class Measurement:
    image: np.ndarray                      # camera frame, H x W, real, finite, >= 0
    focus_mean: float                      # <I>_F
    background_region_means: tuple        # (<I>_B1, ..., <I>_BK)
    background_mean: float                 # <I>_B, pixel-weighted over all background pixels
    enhancement: float                     # <I>_F / max(<I>_B, background_floor)
    background_floored: bool               # True if background_floor replaced <I>_B
    acquisition_index: int = -1            # 0-based frame counter of the bench (-1: not from a bench)


def region_metrics(image, focus: Region, backgrounds, background_floor: float = 1e-12) -> Measurement:
    """Focus/background statistics and enhancement of one camera image (see module docs)."""
    image = np.asarray(image)
    if image.ndim != 2 or not np.isrealobj(image) or not np.all(np.isfinite(image)):
        raise ValueError("image must be a finite, real 2-D array")
    if np.any(image < 0):
        raise ValueError("image must be nonnegative")
    background_floor = _finite_number("background_floor", background_floor, positive=True)
    backgrounds = _validate_regions(image.shape, focus, backgrounds)

    focus_mean = float(image[focus.slices].mean())
    region_means = tuple(float(image[b.slices].mean()) for b in backgrounds)
    total = sum(float(image[b.slices].sum()) for b in backgrounds)
    background_mean = total / sum(b.num_pixels for b in backgrounds)
    floored = background_mean < background_floor
    enhancement = focus_mean / max(background_mean, background_floor)
    return Measurement(
        image=image,
        focus_mean=focus_mean,
        background_region_means=region_means,
        background_mean=background_mean,
        enhancement=enhancement,
        background_floored=floored,
    )


# -----------------------------
# Devices
# -----------------------------

@dataclass(frozen=True)
class NoiseModel:
    """Camera noise per acquisition (see module docs). Defaults: shot + small read noise."""

    enabled: bool = True
    photons_per_unit: float | None = 1000.0   # shot noise scale; None disables shot noise
    read_noise_std: float = 0.01              # intensity units

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError(f"NoiseModel.enabled must be bool, got {self.enabled!r}")
        if self.photons_per_unit is not None:
            _finite_number("photons_per_unit", self.photons_per_unit, positive=True)
        _finite_number("read_noise_std", self.read_noise_std, nonnegative=True)


class DMD:
    """Binary-amplitude DMD with a rows x cols grid of macropixels."""

    def __init__(self, shape, illumination_amplitude: float = 1.0):
        self.shape = _shape2("dmd_shape", shape)
        self.illumination_amplitude = _finite_number(
            "illumination_amplitude", illumination_amplitude, positive=True
        )

    @property
    def num_macropixels(self) -> int:
        return self.shape[0] * self.shape[1]

    def validate_pattern(self, pattern) -> np.ndarray:
        """Return the pattern as a boolean (rows, cols) array; reject anything not binary."""
        p = np.asarray(pattern)
        if p.shape not in (self.shape, (self.num_macropixels,)):
            raise ValueError(f"pattern shape {p.shape} does not match the DMD {self.shape} "
                             f"or ({self.num_macropixels},)")
        if p.dtype == bool:
            return p.reshape(self.shape).copy()
        if not np.issubdtype(p.dtype, np.integer) and not np.issubdtype(p.dtype, np.floating):
            raise ValueError(f"pattern must be boolean or 0/1 numbers, got dtype {p.dtype}")
        if not np.all((p == 0) | (p == 1)):
            raise ValueError("pattern must be binary: every macropixel must be 0 or 1")
        return p.reshape(self.shape).astype(bool)

    def input_field(self, pattern) -> np.ndarray:
        """E_in = a * p: real, nonnegative amplitude modulation, flattened row-major."""
        p = self.validate_pattern(pattern)
        return self.illumination_amplitude * p.ravel().astype(np.float64)


# -----------------------------
# Bench
# -----------------------------

class SimulatedBench:
    """Laser -> binary DMD -> fixed transmission matrix -> noisy camera."""

    def __init__(self, dmd_shape, camera_shape, focus: Region, backgrounds, *, seed: int = 0,
                 noise: NoiseModel | None = None, illumination_amplitude: float = 1.0,
                 background_floor: float = 1e-12, transmission_matrix=None):
        self.dmd = DMD(dmd_shape, illumination_amplitude)
        self.camera_shape = _shape2("camera_shape", camera_shape)
        self.focus = focus
        self.backgrounds = _validate_regions(self.camera_shape, focus, backgrounds)
        self.noise = noise if noise is not None else NoiseModel()
        if not isinstance(self.noise, NoiseModel):
            raise ValueError(f"noise must be a NoiseModel, got {self.noise!r}")
        self.background_floor = _finite_number("background_floor", background_floor, positive=True)
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError(f"seed must be a nonnegative integer, got {seed!r}")
        self.seed = int(seed)

        # Separate, reproducible streams: one draws T once, the other the camera noise.
        t_seq, noise_seq = np.random.SeedSequence(self.seed).spawn(2)
        n_out = self.camera_shape[0] * self.camera_shape[1]
        n_in = self.dmd.num_macropixels
        if transmission_matrix is None:
            rng = np.random.default_rng(t_seq)
            t = (rng.standard_normal((n_out, n_in)) + 1j * rng.standard_normal((n_out, n_in)))
            t *= 1.0 / math.sqrt(2.0 * n_in)   # E|T_mn|^2 = 1 / n_in
        else:
            t = np.array(transmission_matrix, dtype=np.complex128, copy=True)
            if t.shape != (n_out, n_in):
                raise ValueError(f"transmission_matrix must have shape {(n_out, n_in)}, got {t.shape}")
            if not np.all(np.isfinite(t)):
                raise ValueError("transmission_matrix must be finite")
        t.flags.writeable = False
        self._transmission = t                 # private: fixed for this experiment
        self._noise_rng = np.random.default_rng(noise_seq)
        self._acquisitions = 0

    # ---- shapes / bookkeeping (safe for optimisation code)

    @property
    def num_macropixels(self) -> int:
        return self.dmd.num_macropixels

    @property
    def num_acquisitions(self) -> int:
        return self._acquisitions

    def transmission_fingerprint(self) -> str:
        """SHA-256 of T, to check that T stays fixed without exposing it."""
        return hashlib.sha256(self._transmission.tobytes()).hexdigest()

    # ---- physics

    def _ideal_intensity(self, pattern) -> np.ndarray:
        e_out = self._transmission @ self.dmd.input_field(pattern)
        return (np.abs(e_out) ** 2).reshape(self.camera_shape)

    def acquire(self, pattern) -> np.ndarray:
        """One camera frame for this DMD pattern, with fresh noise if enabled."""
        image = self._ideal_intensity(pattern)
        if self.noise.enabled:
            if self.noise.photons_per_unit is not None:
                scale = self.noise.photons_per_unit
                image = self._noise_rng.poisson(image * scale) / scale
            if self.noise.read_noise_std > 0:
                image = image + self._noise_rng.normal(0.0, self.noise.read_noise_std, image.shape)
            image = np.clip(image, 0.0, None)
        self._acquisitions += 1
        return np.asarray(image, dtype=np.float64)

    def measure(self, pattern) -> Measurement:
        """Acquire one frame and evaluate focus/background intensities and enhancement."""
        index = self._acquisitions
        m = region_metrics(self.acquire(pattern), self.focus, self.backgrounds, self.background_floor)
        return replace(m, acquisition_index=index)
