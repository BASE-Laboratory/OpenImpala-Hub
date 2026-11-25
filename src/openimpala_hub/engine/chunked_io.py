"""Out-of-core lazy slice streaming, Dask-parallel 3D slab pooling, and 4D operando metrics.

Incorporates and extends the volumetric I/O and degradation analysis architecture developed by
Ali Humza Butt in `BASE-Laboratory/BatteryExplorer` (`app/data_io.py`, `app/processing.py`,
`app/figures.py`), unified with `basecoat` OME-Zarr region-lazy readers and native-resolution
(`step=1`) ROI subvolume extraction for `OpenImpala` and `SMS` physics simulations:

- Per-path thread locks (`_lock_for`) serialise lazy `tifffile` IFD directory walks and page
  seeks so concurrent Dash thread-pool callbacks never corrupt page counts.
- `load_lazy_slice` reads a single XY page on demand (`@lru_cache(maxsize=24)`), stretches
  `1st–99th` percentiles, applies cached 256-entry colormaps (`grayscale`, `magma`, `inferno`,
  `viridis`), and encodes a `900 px` PNG `data:` URI (`html.Img`).
- `load_chunked_volume` uses `dask.delayed` with per-thread `TiffFile` handles (`_thread_tiff`)
  and anti-aliased `block_mean_2d` pooling so thin cracks survive as partial-volume dips, while
  stretching percentiles over non-zero material voxels (`vol > 0`) to avoid registration-padding
  air dead zones and preserving original pixel/page coordinates.
- `extract_native_roi_volume` completes the `(z_range, y_range, x_range)` sub-box hook at full
  `1x1x1` resolution, segmenting voids/pores (`0`) vs solid (`1`) via fixed pristine
  air/material-midpoint, Otsu, or manual thresholding into a calibrated `basecoat.Volume`.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from functools import lru_cache
import io
from pathlib import Path
import threading
import time
from typing import Any

from basecoat import AxisSpec, Volume, open_volume
import dask
from dask import delayed
import matplotlib
import numpy as np
import openimpala as oi
import pandas as pd
from PIL import Image, ImageDraw
from skimage import measure
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
import tifffile as tiff

COLORMAPS = ("grayscale", "magma", "inferno", "viridis")
DISPLAY_MAX_DIM = 900
VOLUME_Z_STEP = 4
VOLUME_XY_STEP = 16
HOUGH_DS = 4

METRIC_INTERIOR_R = 0.90
METRIC_CENTER_R = 0.35
METRIC_EDGE_R = 0.85
METRIC_LA_XY_STEP = 4
METRIC_LA_Z_STEP = 2
METRIC_LA_HALF_BAND = 8
METRIC_AIR_RING = (1.05, 1.20)


@dataclass(frozen=True)
class Timepoint:
    """Resolved tomography timepoint paired with its cycling hour cursor."""

    label: str
    path: Path
    t_h: float


@dataclass(frozen=True)
class OperandoDataset:
    """Multi-timepoint 4D tomography dataset paired with an electrochemical cycling CSV."""

    key: str
    label: str
    voxel_size_um: float
    echem_csv: Path
    timepoints: tuple[Timepoint, ...]


# ── Per-path thread-safe TIFF handle locks & per-thread Dask handles ───────

_tiff_locks: dict[str, threading.Lock] = {}
_tiff_locks_guard = threading.Lock()


def _lock_for(path_str: str) -> threading.Lock:
    with _tiff_locks_guard:
        return _tiff_locks.setdefault(path_str, threading.Lock())


@lru_cache(maxsize=8)
def _open_tiff(path_str: str) -> tiff.TiffFile:
    return tiff.TiffFile(path_str)


_volume_handles = threading.local()
_volume_handle_registry: list[tiff.TiffFile] = []
_volume_handle_epoch = 0
_volume_registry_lock = threading.Lock()


def _thread_tiff(path_str: str) -> tiff.TiffFile:
    with _volume_registry_lock:
        epoch = _volume_handle_epoch
    if getattr(_volume_handles, "epoch", None) != epoch:
        _volume_handles.cache = {}
        _volume_handles.epoch = epoch
    tf = _volume_handles.cache.get(path_str)
    if tf is None:
        tf = _volume_handles.cache[path_str] = tiff.TiffFile(path_str)
        with _volume_registry_lock:
            _volume_handle_registry.append(tf)
    return tf


def close_volume_handles() -> None:
    """Close all pooled per-thread TIFF handles and clear LRU caches."""
    global _volume_handle_epoch
    with _volume_registry_lock:
        _volume_handle_epoch += 1
        for tf in _volume_handle_registry:
            try:
                tf.close()
            except Exception:
                pass
        _volume_handle_registry.clear()
    load_lazy_slice.cache_clear()
    dataset_intensity_range.cache_clear()
    dataset_structural_metrics.cache_clear()
    load_chunked_volume.cache_clear()
    load_echem_csv.cache_clear()
    _open_tiff.cache_clear()


def _is_zarr_path(path_str: str) -> bool:
    p = Path(path_str)
    return p.suffix == ".zarr" or p.name.endswith(".ome.zarr") or p.is_dir()


def stack_dimensions(path_str: str) -> tuple[int, int, int]:
    """Return `(n_z, height, width)` without reading pixel arrays into memory."""
    if _is_zarr_path(path_str):
        with open_volume(path_str) as lv:
            nz, ny, nx = lv.shape
            return int(nz), int(ny), int(nx)
    with _lock_for(path_str):
        tf = _open_tiff(path_str)
        nz = len(tf.pages)
        h, w = tf.pages[0].shape[:2]
        return int(nz), int(h), int(w)


def n_pages(path_str: str) -> int:
    return stack_dimensions(path_str)[0]


def page_shape(path_str: str) -> tuple[int, int]:
    _, h, w = stack_dimensions(path_str)
    return h, w


# ── Image & Colormap Helpers ───────────────────────────────────────────────


@lru_cache(maxsize=None)
def _colormap_lut(name: str) -> np.ndarray:
    cmap = matplotlib.colormaps[name]
    return (cmap(np.linspace(0.0, 1.0, 256))[:, :3] * 255.0).astype(np.uint8)


def apply_colormap(img8: np.ndarray, cmap: str) -> np.ndarray:
    if cmap not in COLORMAPS or cmap == "grayscale":
        return img8
    return _colormap_lut(cmap)[img8]


def block_mean_2d(page: np.ndarray, step: int) -> np.ndarray:
    """Mean-pool an `(H, W)` slice by `step` in both axes so thin cracks survive as dips."""
    if step <= 1:
        return page.astype(np.float32, copy=False)
    h = (page.shape[0] // step) * step
    w = (page.shape[1] // step) * step
    if h == 0 or w == 0:
        return page.astype(np.float32, copy=False)
    return (
        page[:h, :w]
        .reshape(h // step, step, w // step, step)
        .mean(axis=(1, 3), dtype=np.float32)
    )


def normalize_to_uint8(
    img: np.ndarray, p_low: float = 1.0, p_high: float = 99.0
) -> np.ndarray:
    arr = img.astype(np.float32, copy=False)
    nonzero = arr[arr > 0]
    ref = nonzero if nonzero.size > 16 else arr
    lo, hi = np.percentile(ref, [p_low, p_high])
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0) * 255.0
    if nonzero.size > 16:
        out = np.where(arr > 0, out, 0.0)
    return out.astype(np.uint8)


def array_to_png_src(
    img8: np.ndarray,
    max_dim: int = DISPLAY_MAX_DIM,
    roi_box_xy: tuple[int, int, int, int] | None = None,
    orig_shape_hw: tuple[int, int] | None = None,
) -> tuple[str, int]:
    """Encode a 2D `uint8` grayscale or RGB slice as a resized PNG `data:` URI + byte length."""
    im = Image.fromarray(img8)
    orig_h, orig_w = orig_shape_hw if orig_shape_hw is not None else (im.height, im.width)
    scale_x = 1.0
    scale_y = 1.0
    if max(im.size) > max_dim:
        sc = max_dim / float(max(im.size))
        new_w = max(1, round(im.width * sc))
        new_h = max(1, round(im.height * sc))
        im = im.resize((new_w, new_h), Image.BILINEAR)
    if orig_w > 0 and orig_h > 0:
        scale_x = im.width / float(orig_w)
        scale_y = im.height / float(orig_h)

    if roi_box_xy is not None:
        if im.mode != "RGB":
            im = im.convert("RGB")
        x0, x1, y0, y1 = roi_box_xy
        px0 = int(round(x0 * scale_x))
        px1 = int(round(x1 * scale_x))
        py0 = int(round(y0 * scale_y))
        py1 = int(round(y1 * scale_y))
        draw = ImageDraw.Draw(im)
        draw.rectangle([px0, py0, px1, py1], outline=(56, 189, 248), width=2)

    buf = io.BytesIO()
    im.save(buf, format="PNG")
    raw_bytes = buf.getvalue()
    encoded = base64.b64encode(raw_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}", len(raw_bytes)


# ── Lazy Single-Slice Streaming ────────────────────────────────────────────


def _read_single_page(path_str: str, z: int) -> np.ndarray:
    if _is_zarr_path(path_str):
        with open_volume(path_str) as lv:
            nz = lv.shape[0]
            zc = int(np.clip(z, 0, nz - 1))
            sub = lv.read_region((slice(zc, zc + 1), slice(None), slice(None)))
            return np.asarray(sub.data)[0]
    tf = _open_tiff(path_str)
    with _lock_for(path_str):
        zc = int(np.clip(z, 0, len(tf.pages) - 1))
        return tf.pages[zc].asarray()


@lru_cache(maxsize=24)
def load_lazy_slice(
    path_str: str,
    z: int,
    hist_range: tuple[float, float] | None = None,
    cmap: str = "grayscale",
) -> dict[str, Any]:
    """Read a single XY slice lazily from disk/Zarr and return PNG + 64-bin histogram."""
    t0 = time.perf_counter()
    raw = _read_single_page(path_str, int(z))
    if raw.ndim > 2:
        raw = raw[..., 0]
    img8 = normalize_to_uint8(raw)
    colored = apply_colormap(img8, cmap)
    hist, edges = np.histogram(raw, bins=64, range=hist_range)
    png_src, n_bytes = array_to_png_src(colored, DISPLAY_MAX_DIM)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "img8": colored,
        "orig_hw": (int(raw.shape[0]), int(raw.shape[1])),
        "hist": hist,
        "edges": edges,
        "png_src": png_src,
        "png_bytes": n_bytes,
        "read_ms": round(elapsed_ms, 2),
        "raw_min": float(raw.min()),
        "raw_max": float(raw.max()),
    }


def render_slice_with_roi_overlay(
    slice_dict: dict[str, Any],
    roi_box_xy: tuple[int, int, int, int] | None = None,
) -> str:
    """Overlay a crisp cyan selection rectangle onto an already-loaded slice without disk I/O."""
    if roi_box_xy is None:
        return str(slice_dict["png_src"])
    src, _ = array_to_png_src(
        slice_dict["img8"],
        DISPLAY_MAX_DIM,
        roi_box_xy=roi_box_xy,
        orig_shape_hw=slice_dict["orig_hw"],
    )
    return src


@lru_cache(maxsize=6)
def dataset_intensity_range(paths: tuple[str, ...]) -> tuple[float, float]:
    """Sample mid-stack pages (4x strided) across timepoints for shared Compare histogram axes."""
    samples = []
    for p in paths:
        nz = n_pages(p)
        raw = _read_single_page(p, nz // 2)
        sub = raw[::4, ::4].ravel()
        nz_vox = sub[sub > 0]
        samples.append(nz_vox if nz_vox.size else sub)
    lo, hi = np.percentile(np.concatenate(samples), [1.0, 99.0])
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


# ── Dask-Parallel Block-Mean 3D Slab Loader & Native ROI Drill-Down ────────


def _read_slab(
    path_str: str,
    z_lo: int,
    z_hi: int,
    y_lo: int,
    y_hi: int,
    x_lo: int,
    x_hi: int,
    xy_step: int,
) -> np.ndarray:
    """Average pages `[z_lo, z_hi)` into one block-mean-pooled 2D output plane."""
    if _is_zarr_path(path_str):
        with open_volume(path_str) as lv:
            slab_3d = np.asarray(
                lv.read_region((slice(z_lo, z_hi), slice(y_lo, y_hi), slice(x_lo, x_hi))).data,
                dtype=np.float32,
            )
        acc = None
        for k in range(slab_3d.shape[0]):
            pooled = block_mean_2d(slab_3d[k], xy_step)
            acc = pooled if acc is None else acc + pooled
        return acc / float(max(1, slab_3d.shape[0]))

    tf = _thread_tiff(path_str)
    acc = None
    for z in range(z_lo, z_hi):
        page = tf.pages[z].asarray()
        if page.ndim > 2:
            page = page[..., 0]
        pooled = block_mean_2d(page[y_lo:y_hi, x_lo:x_hi], xy_step)
        acc = pooled if acc is None else acc + pooled
    return acc / float(max(1, z_hi - z_lo))


@lru_cache(maxsize=8)
def load_chunked_volume(
    path_str: str,
    z_range: tuple[int, int] | None = None,
    y_range: tuple[int, int] | None = None,
    x_range: tuple[int, int] | None = None,
    z_step: int = VOLUME_Z_STEP,
    xy_step: int = VOLUME_XY_STEP,
    scheduler: str = "threads",
    num_workers: int | None = None,
) -> dict[str, Any]:
    """Load a Dask-parallel block-mean-downsampled 3D volume (or native-step ROI sub-box)."""
    t0 = time.perf_counter()
    n, h, w = stack_dimensions(path_str)

    z_lo, z_hi = z_range if z_range is not None else (0, n)
    y_lo, y_hi = y_range if y_range is not None else (0, h)
    x_lo, x_hi = x_range if x_range is not None else (0, w)
    z_lo, z_hi = max(0, int(z_lo)), min(n, int(z_hi))
    y_lo, y_hi = max(0, int(y_lo)), min(h, int(y_hi))
    x_lo, x_hi = max(0, int(x_lo)), min(w, int(x_hi))
    if z_hi <= z_lo:
        z_hi = min(n, z_lo + 1)
    if y_hi <= y_lo:
        y_hi = min(h, y_lo + 1)
    if x_hi <= x_lo:
        x_hi = min(w, x_lo + 1)

    z_step = max(1, int(z_step))
    xy_step = max(1, int(xy_step))
    z_starts = list(range(z_lo, z_hi, z_step))
    tasks = [
        delayed(_read_slab)(
            path_str, z0, min(z0 + z_step, z_hi), y_lo, y_hi, x_lo, x_hi, xy_step
        )
        for z0 in z_starts
    ]
    compute_kwargs: dict[str, Any] = {"scheduler": scheduler}
    if num_workers is not None:
        compute_kwargs["num_workers"] = num_workers
    vol = np.stack(dask.compute(*tasks, **compute_kwargs))

    nz, ny, nx = vol.shape
    material = vol[vol > 0]
    lo, hi = np.percentile(material, [1.0, 99.0]) if material.size else (0.0, 0.0)
    if hi <= lo:
        lo, hi = np.percentile(vol, [1.0, 99.0])
    if hi > lo:
        value = (np.clip((vol - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.uint8)
        if material.size:
            value = np.where(vol > 0, value, 0).astype(np.uint8)
    else:
        value = np.zeros(vol.shape, dtype=np.uint8)

    zc = np.asarray(z_starts, dtype=np.uint16)
    yc = np.uint16(y_lo) + np.arange(ny, dtype=np.uint16) * np.uint16(xy_step)
    xc = np.uint16(x_lo) + np.arange(nx, dtype=np.uint16) * np.uint16(xy_step)
    zg, yg, xg = np.meshgrid(zc, yc, xc, indexing="ij")

    hist, edges = np.histogram(value, bins=64, range=(0, 255))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "x": xg.ravel(),
        "y": yg.ravel(),
        "z": zg.ravel(),
        "value": value.ravel(),
        "raw_float": vol,
        "shape": (int(nz), int(ny), int(nx)),
        "lo": float(lo),
        "hi": float(hi),
        "hist": hist,
        "edges": edges,
        "load_ms": round(elapsed_ms, 1),
    }


def extract_isosurface_mesh(
    vol: dict[str, Any],
    threshold: float,
    *,
    rim_cutoff: float = 1.0,
    circle: tuple[float, float, float] | None = None,
    clip_box: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract marching-cubes `(verts_xyz, faces)` in original pixel/page coordinates."""
    values = np.asarray(vol["value"]).reshape(vol["shape"]).astype(np.float32)
    nz, ny, nx = vol["shape"]
    if min(nz, ny, nx) < 2:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)

    zc = np.asarray(vol["z"], dtype=np.float64).reshape(vol["shape"])[:, 0, 0]
    yc = np.asarray(vol["y"], dtype=np.float64).reshape(vol["shape"])[0, :, 0]
    xc = np.asarray(vol["x"], dtype=np.float64).reshape(vol["shape"])[0, 0, :]
    spacing = (
        float(zc[1] - zc[0]) if nz > 1 else 1.0,
        float(yc[1] - yc[0]) if ny > 1 else 1.0,
        float(xc[1] - xc[0]) if nx > 1 else 1.0,
    )

    if circle is not None and rim_cutoff < 1.0:
        cx, cy, r_disc = circle
        xg, yg = np.meshgrid(xc, yc, indexing="xy")
        values[:, np.hypot(xg - cx, yg - cy) > float(rim_cutoff) * r_disc] = 0.0

    if clip_box is not None:
        (x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi) = clip_box
        values[(zc < x_lo * 0 + z_lo) | (zc > z_hi)] = 0.0
        values[:, (yc < y_lo) | (yc > y_hi)] = 0.0
        values[:, :, (xc < x_lo) | (xc > x_hi)] = 0.0

    try:
        verts_zyx, faces, _, _ = measure.marching_cubes(
            values, level=float(threshold), spacing=spacing
        )
    except (ValueError, RuntimeError):
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)

    verts_xyz = np.column_stack(
        [
            verts_zyx[:, 2] + xc[0],
            verts_zyx[:, 1] + yc[0],
            verts_zyx[:, 0] + zc[0],
        ]
    )
    return verts_xyz, faces


# ── Hough Circle Fit, Structural Degradation Metrics & Native ROI Volume ───


def fit_reference_circle(path_str: str) -> tuple[float, float, float]:
    """Fit cylindrical electrode disc `(cx, cy, R)` in original pixel coordinates."""
    nz, h_full, w_full = stack_dimensions(path_str)
    ds = max(1, min(HOUGH_DS, min(h_full, w_full) // 48))
    fits = []
    for frac in (0.12, 0.30, 0.50, 0.70, 0.88):
        z = int(frac * max(0, nz - 1))
        a = _read_single_page(path_str, z)[::ds, ::ds].astype(np.float32)
        if float(a.max()) <= 0.0:
            continue
        a = (a - float(a.min())) / max(float(a.max() - a.min()), 1e-9)
        edges = canny(a, sigma=1.6)
        s = min(a.shape)
        r_lo = max(4, int(0.28 * s))
        r_hi = max(r_lo + 3, int(0.49 * s))
        radii = np.arange(r_lo, r_hi, max(1, s // 64))
        h_acc = hough_circle(edges, radii)
        accums, cxs, cys, rs = hough_circle_peaks(h_acc, radii, total_num_peaks=1)
        if len(accums):
            fits.append((float(accums[0]), float(cxs[0]), float(cys[0]), float(rs[0])))
    if not fits:
        return (w_full * 0.5, h_full * 0.5, min(w_full, h_full) * 0.42)
    best = max(fits)
    good = [f for f in fits if f[0] >= 0.6 * best[0] and abs(f[3] - best[3]) <= 10]
    cx = float(np.mean([f[1] for f in good])) * ds
    cy = float(np.mean([f[2] for f in good])) * ds
    rad = float(np.mean([f[3] for f in good])) * ds
    return cx, cy, rad


def _low_atten_threshold(path_str: str, circle: tuple[float, float, float]) -> float:
    """Compute fixed pristine air/material-midpoint threshold in raw intensity units."""
    cx, cy, radius = circle
    nz, h_full, _ = stack_dimensions(path_str)
    ds = max(1, min(4, h_full // 64))
    page = _read_single_page(path_str, nz // 2)[::ds, ::ds]
    yy, xx = np.mgrid[0 : page.shape[0], 0 : page.shape[1]]
    r = np.hypot(xx * ds - cx, yy * ds - cy)
    ring = page[
        (r > METRIC_AIR_RING[0] * radius)
        & (r < METRIC_AIR_RING[1] * radius)
        & (page > 0)
    ]
    interior = page[(r < METRIC_INTERIOR_R * radius) & (page > 0)]
    air_med = float(np.median(ring)) if ring.size else 0.0
    mat_med = float(np.median(interior)) if interior.size else float(np.median(page))
    return 0.5 * (air_med + mat_med)


@lru_cache(maxsize=4)
def dataset_structural_metrics(
    paths: tuple[str, ...],
    z_step: int = 2,
    xy_step: int = 4,
) -> dict[str, Any]:
    """Compute timepoint-synced interface intensity, static rim ratio, and void+Li fraction."""
    cx, cy, radius = fit_reference_circle(paths[0])
    curves = []
    rim_ratios = []
    z_pages = None

    raw_stacks = []
    plane_masks = []
    for p in paths:
        vol = load_chunked_volume(p, z_step=z_step, xy_step=xy_step)
        nz, ny, nx = vol["shape"]
        raw = vol["raw_float"]
        v = vol["value"].reshape(nz, ny, nx)
        xs = vol["x"].reshape(nz, ny, nx)[0, 0, :].astype(np.float64) + 0.5 * float(xy_step)
        ys = vol["y"].reshape(nz, ny, nx)[0, :, 0].astype(np.float64) + 0.5 * float(xy_step)
        z_pages = vol["z"].reshape(nz, ny, nx)[:, 0, 0].astype(np.int64)
        xg, yg = np.meshgrid(xs, ys, indexing="xy")
        r = np.hypot(xg - cx, yg - cy)
        interior = r < METRIC_INTERIOR_R * radius
        center = r < METRIC_CENTER_R * radius
        # Stay inside 0.90*R so block-mean pooling never straddles the outer disc->air step
        edge = (r > 0.70 * radius) & (r <= METRIC_INTERIOR_R * radius)

        stretched_mean = np.array(
            [float(v[k][interior].mean()) if interior.any() else 0.0 for k in range(nz)]
        )
        curve = np.array(
            [float(raw[k][interior].mean()) if interior.any() else np.nan for k in range(nz)]
        )
        curve[stretched_mean == 0.0] = np.nan
        curves.append(curve)
        raw_stacks.append(raw)
        plane_masks.append((curve, center, edge))

    m = np.vstack(curves)
    valid = ~np.isnan(m).any(axis=0)
    if m.shape[0] >= 2 and valid.any():
        drop = np.where(valid, m[1] - m[0], np.inf)
    elif valid.any():
        drop = np.where(valid, m[0], np.inf)
    else:
        drop = np.zeros(m.shape[1])
    k_best = int(np.argmin(drop))
    assert z_pages is not None
    interface_z = int(z_pages[k_best])

    for raw, (curve, center, edge) in zip(raw_stacks, plane_masks, strict=True):
        bulk_planes = [
            k
            for k in range(len(curve))
            if not np.isnan(curve[k]) and abs(int(z_pages[k]) - interface_z) > 8
        ]
        if not bulk_planes:
            bulk_planes = [k for k in range(len(curve)) if not np.isnan(curve[k])]
        if bulk_planes and edge.any() and center.any():
            edge_m = float(np.mean([raw[k][edge].mean() for k in bulk_planes]))
            cent_m = max(1e-6, float(np.mean([raw[k][center].mean() for k in bulk_planes])))
            rim_ratios.append(edge_m / cent_m)
        else:
            rim_ratios.append(1.0)

    la_thr = _low_atten_threshold(paths[0], (cx, cy, radius))
    la_fracs = []
    la_voxels = []
    band = None
    for p in paths:
        tot_z = n_pages(p)
        half_b = min(METRIC_LA_HALF_BAND, max(2, tot_z // 8))
        z_lo = max(0, interface_z - half_b)
        z_hi = min(tot_z, interface_z + half_b + 1)
        if band is None:
            band = (z_lo, z_hi)
        sub = load_chunked_volume(
            p, z_range=(z_lo, z_hi), z_step=max(1, z_step // 2), xy_step=max(1, xy_step // 2)
        )
        slab = sub["raw_float"]
        ny_p, nx_p = slab.shape[1:]
        step_xy = max(1, xy_step // 2)
        step_z = max(1, z_step // 2)
        yy, xx = np.meshgrid(np.arange(ny_p), np.arange(nx_p), indexing="ij")
        r = np.hypot((xx + 0.5) * step_xy - cx, (yy + 0.5) * step_xy - cy)
        valid_vox = (r < METRIC_INTERIOR_R * radius)[None, :, :] & (slab > 0)
        low = valid_vox & (slab < la_thr)
        n_valid = int(valid_vox.sum())
        la_fracs.append(float(low.sum() / n_valid) if n_valid else 0.0)
        la_voxels.append(int(low.sum()) * (step_xy**2) * step_z)

    return {
        "interface_z": interface_z,
        "interface_intensity": tuple(float(m[i, k_best]) for i in range(m.shape[0])),
        "rim_ratio": tuple(rim_ratios),
        "circle": (cx, cy, radius),
        "low_atten_fraction": tuple(la_fracs),
        "low_atten_voxels": tuple(la_voxels),
        "low_atten_thr": float(la_thr),
        "low_atten_band": band or (0, 1),
    }


@lru_cache(maxsize=6)
def load_echem_csv(csv_path_str: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path_str)
    for col in ("Time/h", "Potential/V"):
        if col not in df.columns:
            raise ValueError(f"Missing column {col!r}; found {list(df.columns)}")
    out = pd.DataFrame(
        {
            "time_h": pd.to_numeric(df["Time/h"], errors="coerce"),
            "V": pd.to_numeric(df["Potential/V"], errors="coerce"),
            "I": (
                pd.to_numeric(df["Current/A"], errors="coerce")
                if "Current/A" in df.columns
                else 0.0
            ),
        }
    )
    return out.dropna(subset=["time_h", "V"])


def extract_native_roi_volume(
    path_str: str,
    *,
    x_range: tuple[int, int],
    y_range: tuple[int, int],
    z_range: tuple[int, int],
    voxel_size_um: float | tuple[float, float, float] = 1.6,
    threshold_mode: str = "midpoint",
    manual_threshold: float | None = None,
    dark_is_pore: bool = True,
    pristine_ref_path: str | None = None,
) -> tuple[Volume, float]:
    """Extract a native `1x1x1` (`step=1`) subvolume and segment into a calibrated `Volume`.

    Completes the `(z_range, y_range, x_range)` drill-down hook in `BatteryExplorer`: reads
    only the requested sub-box via `load_chunked_volume(..., z_step=1, xy_step=1)`, segments
    low-attenuation voids/pores (`0`) vs solid matrix (`1`), and attaches physical `um` axes.
    """
    sub = load_chunked_volume(
        path_str,
        z_range=z_range,
        y_range=y_range,
        x_range=x_range,
        z_step=1,
        xy_step=1,
    )
    raw = np.asarray(sub["raw_float"], dtype=np.float32)

    mode = str(threshold_mode).lower()
    if mode == "manual" and manual_threshold is not None:
        thr = float(manual_threshold)
        seg = np.where(raw <= thr, 0, 1) if dark_is_pore else np.where(raw > thr, 0, 1)
    elif mode == "otsu":
        seg, thr_val = oi.binarize_otsu(
            raw, pore_value=0, solid_value=1, dark_is_pore=dark_is_pore
        )
        thr = float(thr_val)
    else:
        ref = pristine_ref_path or path_str
        circ = fit_reference_circle(ref)
        thr = _low_atten_threshold(ref, circ)
        seg = np.where(raw <= thr, 0, 1) if dark_is_pore else np.where(raw > thr, 0, 1)

    if isinstance(voxel_size_um, (int, float)):
        vz = vy = vx = float(voxel_size_um)
    else:
        vz, vy, vx = (float(voxel_size_um[0]), float(voxel_size_um[1]), float(voxel_size_um[2]))

    axes = (
        AxisSpec(name="z", kind="space", unit="um", scale=vz),
        AxisSpec(name="y", kind="space", unit="um", scale=vy),
        AxisSpec(name="x", kind="space", unit="um", scale=vx),
    )
    vol = Volume(
        np.ascontiguousarray(seg, dtype=np.int32),
        axes,
        label_names={0: "pore", 1: "solid"},
        provenance={
            "data_origin": "benchmark_xct",
            "source_file": Path(path_str).name,
            "roi_zyx": [list(z_range), list(y_range), list(x_range)],
            "threshold_raw": round(thr, 2),
        },
    )
    return vol, thr
