"""Boundary adapter between `basecoat v0.3` and `openimpala >= 5.0, < 6.0`.

Responsibilities:
1. Enforce physical unit calibration: convert `basecoat.Volume` spacing to SI metres
   via `vol.spacing_in("m")`, raising `openimpala.UnitCalibrationError` (on `v5.1+`)
   or a compatible calibration exception (on `v5.0.x`) when `unit='voxel'`.
2. Pass `basecoat.Volume` directly when `openimpala >= 5.1` supports `VolumeProtocol`,
   and transparently unpack `(arr, voxel_size_m)` on `openimpala == 5.0.x`.
3. Enforce the public web grid-size cap (`HUB_MAX_SOLVE_SIDE`, default `128`) using
   either centre ROI cropping or discrete majority-mode coarsening while keeping
   `AxisSpec` cell-centre coordinates exact (`coarsen_axes`).
"""

from __future__ import annotations

import os
from typing import Literal

import numpy as np
from basecoat import AxisSpec, UnitConversionError, Volume, VolumeSet, coarsen_axes
import openimpala as oi

# Resolve UnitCalibrationError across openimpala v5.1+ and v5.0.x
if hasattr(oi, "UnitCalibrationError"):
    HubUnitCalibrationError = oi.UnitCalibrationError
else:

    class HubUnitCalibrationError(ValueError):  # type: ignore[no-redef]
        """Raised when an uncalibrated (`unit='voxel'`) volume reaches a physics solver."""


_SUPPORTS_VOLUME_PROTOCOL = hasattr(oi, "VolumeProtocol")


def default_max_solve_side() -> int:
    """Return configured maximum per-axis voxel count for interactive solves."""
    raw = os.environ.get("HUB_MAX_SOLVE_SIDE", "128").strip()
    if raw.lower() in ("0", "none", "unlimited", "off"):
        return 0
    val = int(raw)
    return max(0, val)


def clamp_effective_max_side(requested: int | None) -> int:
    """Clamp caller-supplied `max_solve_side` so `0` or large values cannot bypass `HUB_MAX_SOLVE_SIDE`."""
    env_cap = default_max_solve_side()
    if env_cap <= 0:
        return 0 if requested is None else max(0, int(requested))
    if requested is None or int(requested) <= 0:
        return env_cap
    return min(int(requested), env_cap)


def extract_phase_volume(source: Volume | VolumeSet, *, label_key: str = "phase") -> Volume:
    """Extract the integer `(z, y, x)` phase `Volume` from a `Volume` or `VolumeSet`."""
    if isinstance(source, VolumeSet):
        labels = source.labels
        if label_key in labels:
            return labels[label_key]
        if labels:
            first_key = sorted(labels.keys())[0]
            return labels[first_key]
        img = source.image
    else:
        img = source

    if img.ndim != 3 or img.axis_names != ("z", "y", "x"):
        raise ValueError(
            f"Expected a 3D ('z', 'y', 'x') Volume, got shape={img.shape}, axes={img.axis_names}."
        )
    if not np.issubdtype(img.dtype, np.integer):
        raise ValueError(
            f"Solver requires an integer phase-labelled Volume (or VolumeSet with labels['{label_key}']), "
            f"got floating dtype={img.dtype}. Apply Otsu or manual thresholding first."
        )
    return img


def resolve_voxel_size_metres(vol: Volume) -> tuple[float, float, float]:
    """Convert calibrated `Volume` `(z, y, x)` spacing to SI metres, raising on `'voxel'`."""
    try:
        dz, dy, dx = vol.spacing_in("m")
    except UnitConversionError as exc:
        raise HubUnitCalibrationError(
            f"{exc} Supply a physical voxel size in micrometres before running OpenImpala."
        ) from exc
    return (float(dz), float(dy), float(dx))


def solver_input_for_openimpala(
    vol: Volume,
) -> tuple[object, dict[str, object]]:
    """Prepare `(data_arg, voxel_kwargs)` compatible with both `openimpala 5.0` and `5.1+`."""
    dz_m, dy_m, dx_m = resolve_voxel_size_metres(vol)
    if _SUPPORTS_VOLUME_PROTOCOL:
        return vol, {}
    arr = np.ascontiguousarray(vol.data, dtype=np.int32)
    return arr, {"voxel_size": (dz_m, dy_m, dx_m)}


def _downsample_majority_2x_step(
    arr: np.ndarray,
    axes: tuple[AxisSpec, ...],
) -> tuple[np.ndarray, tuple[AxisSpec, ...]]:
    """Coarsen spatial dimensions > 1 by up to 2x using discrete mode (smallest label on ties)."""
    factors: list[int] = []
    slices: list[slice] = []
    for length in arr.shape:
        if length >= 4:
            even_len = (length // 2) * 2
            factors.append(2)
            slices.append(slice(0, even_len))
        else:
            factors.append(1)
            slices.append(slice(0, length))

    if all(f == 1 for f in factors):
        return arr, axes

    trimmed = arr[tuple(slices)]
    trimmed_axes = tuple(
        ax.sliced(slc, orig_len)
        for ax, slc, orig_len in zip(axes, slices, arr.shape, strict=True)
    )
    fz, fy, fx = factors
    nz, ny, nx = trimmed.shape
    blocks = (
        trimmed.reshape(nz // fz, fz, ny // fy, fy, nx // fx, fx)
        .transpose(0, 2, 4, 1, 3, 5)
        .reshape(nz // fz, ny // fy, nx // fx, fz * fy * fx)
    )

    unique_vals = np.unique(trimmed)
    best_labels = np.full(blocks.shape[:-1], int(unique_vals[0]), dtype=np.int32)
    best_counts = np.full(blocks.shape[:-1], -1, dtype=np.int16)
    for u in sorted(int(v) for v in unique_vals):
        counts = np.sum(blocks == u, axis=-1, dtype=np.int16)
        mask = counts > best_counts
        best_labels[mask] = u
        best_counts[mask] = counts[mask]

    new_axes = coarsen_axes(trimmed_axes, factors)
    return best_labels, new_axes


def enforce_solve_grid_cap(
    vol: Volume,
    *,
    max_side: int | None = None,
    policy: Literal["crop_center", "mode_downsample", "error"] = "crop_center",
) -> tuple[Volume, str | None]:
    """Bound a 3D `Volume` to `max_side` per axis if configured, preserving cell-centre coordinates."""
    cap = clamp_effective_max_side(max_side)
    if cap <= 0 or all(s <= cap for s in vol.shape):
        return vol, None

    orig_shape = vol.shape
    if policy == "error":
        raise ValueError(
            f"Volume shape {orig_shape} exceeds solve cap ({cap}^3). "
            f"Crop a sub-ROI, enable mode_downsample, or raise HUB_MAX_SOLVE_SIDE locally."
        )

    if policy == "crop_center":
        slices = []
        for s in orig_shape:
            if s > cap:
                start = (s - cap) // 2
                slices.append(slice(start, start + cap))
            else:
                slices.append(slice(0, s))
        cropped = vol[tuple(slices)]
        note = (
            f"Centre-cropped from {orig_shape} to {cropped.shape} "
            f"(public web cap {cap}^3; physical coordinates preserved)."
        )
        return cropped, note

    cur_arr = np.ascontiguousarray(vol.data, dtype=np.int32)
    cur_axes = vol.axes
    while any(s > cap for s in cur_arr.shape):
        next_arr, next_axes = _downsample_majority_2x_step(cur_arr, cur_axes)
        if next_arr.shape == cur_arr.shape:
            break
        cur_arr, cur_axes = next_arr, next_axes

    coarsened = Volume(
        cur_arr,
        cur_axes,
        space=vol.space,
        centering=vol.centering,
        label_names=vol.label_names,
        provenance={**vol.provenance, "coarsened_from": list(orig_shape)},
    )
    note = (
        f"Majority-mode downsampled from {orig_shape} to {coarsened.shape} "
        f"(public web cap {cap}^3; voxel spacing updated to "
        f"{tuple(round(ax.scale, 4) for ax in coarsened.axes)} {coarsened.axes[0].unit})."
    )
    return coarsened, note
