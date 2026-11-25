"""UUID-isolated upload staging, `.ome.zarr.zip` zip-slip defence, and pre-solve volume inspection.

Security invariants:
- Every browser or REST upload is written inside `<upload_root>/<uuid4>/` so concurrent users
  uploading `sample.tif` never collide or overwrite each other's tomograms.
- Old upload directories beyond `MAX_RETAINED_UPLOADS` (default 24) are automatically pruned.
- `.ome.zarr.zip` / `.zarr.zip` / `.zip` archives are extracted entry-by-entry with symlink
  rejection, `is_relative_to` path-containment (zip-slip defence), entry-count bounds, and
  streamed byte counting (zip-bomb defence).
- `is_staged_upload_path(path)` ensures `POST /api/v1/solve` never reads arbitrary server files.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any
import uuid
import zipfile

import numpy as np
from basecoat import Volume, VolumeSet, open_volume, read_ome_zarr
import openimpala as oi

MAX_UPLOAD_ARCHIVE_BYTES = 64 * 1024 * 1024  # 64 MiB compressed/raw upload limit
MAX_ZARR_UNCOMPRESSED_BYTES = 128 * 1024 * 1024  # 128 MiB uncompressed limit
MAX_ZARR_ENTRIES = 2000
MAX_RETAINED_UPLOADS = 24

SUPPORTED_UPLOAD_SUFFIXES: tuple[str, ...] = (
    ".ome.zarr.zip",
    ".zarr.zip",
    ".zip",
    ".tif",
    ".tiff",
    ".npy",
    ".h5",
    ".hdf5",
)

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_UUID_HEX_RE = re.compile(r"^[0-9a-f]{32}$")


def get_upload_stage_root() -> Path:
    """Return the resolved root directory used for isolated upload staging."""
    env_override = os.environ.get("OPENIMPALA_HUB_UPLOAD_DIR")
    base = (
        Path(env_override).expanduser()
        if env_override
        else Path(tempfile.gettempdir()) / "openimpala_hub_uploads"
    )
    base.mkdir(parents=True, exist_ok=True)
    return base.resolve()


def _sanitize_filename(filename: str) -> str:
    raw = Path(filename).name.strip()
    cleaned = _SAFE_FILENAME_RE.sub("_", raw).lstrip(".")
    if not cleaned:
        raise ValueError("Invalid empty upload filename.")
    low = cleaned.lower()
    if not low.endswith(SUPPORTED_UPLOAD_SUFFIXES):
        raise ValueError(
            f"Unsupported upload '{raw}'. Supported formats: "
            ".tif/.tiff, .npy, .h5/.hdf5, and .ome.zarr.zip."
        )
    return cleaned


def prune_staged_uploads(*, max_keep: int = MAX_RETAINED_UPLOADS) -> int:
    """Delete oldest UUID upload directories beyond `max_keep`. Returns number removed."""
    root = get_upload_stage_root()
    dirs = [
        d
        for d in root.iterdir()
        if d.is_dir() and _UUID_HEX_RE.match(d.name) is not None
    ]
    if len(dirs) <= max_keep:
        return 0
    dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    removed = 0
    for stale in dirs[max_keep:]:
        shutil.rmtree(stale, ignore_errors=True)
        removed += 1
    return removed


def is_staged_upload_path(candidate: str | Path | None) -> bool:
    """Return True iff `candidate` resolves inside a valid UUID subdirectory of `get_upload_stage_root()`."""
    if not candidate:
        return False
    try:
        root = get_upload_stage_root()
        raw_p = Path(str(candidate))
        if raw_p.is_symlink():
            return False
        resolved = raw_p.resolve(strict=True)
        if not resolved.is_relative_to(root):
            return False
        rel = resolved.relative_to(root)
        if len(rel.parts) < 2:
            return False
        return _UUID_HEX_RE.match(rel.parts[0]) is not None
    except (OSError, ValueError, RuntimeError):
        return False


def _extract_safe_zarr_zip(raw_bytes: bytes, dest_dir: Path) -> Path:
    """Safely extract a `.ome.zarr.zip` archive into `dest_dir` with zip-slip & zip-bomb guards."""
    archive_path = dest_dir / "_archive.zip"
    archive_path.write_bytes(raw_bytes)
    extract_root = (dest_dir / "extracted.ome.zarr").resolve()
    extract_root.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            infos = zf.infolist()
            if len(infos) == 0:
                raise ValueError("Uploaded .zip archive is empty.")
            if len(infos) > MAX_ZARR_ENTRIES:
                raise ValueError(
                    f"Archive contains {len(infos)} entries (limit {MAX_ZARR_ENTRIES})."
                )

            total_written = 0
            for info in infos:
                rel_name = info.filename.replace("\\", "/")
                if rel_name.startswith("/") or ".." in Path(rel_name).parts:
                    raise ValueError(f"Unsafe archive entry path rejected: {info.filename!r}.")
                # Reject Unix symlinks encoded in zip external attributes
                unix_mode = (info.external_attr >> 16) & 0o170000
                if unix_mode == 0o120000:
                    raise ValueError(f"Symlink archive entry rejected: {info.filename!r}.")

                target = (extract_root / rel_name).resolve()
                if not target.is_relative_to(extract_root):
                    raise ValueError(f"Archive path traversal rejected: {info.filename!r}.")

                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue

                if info.file_size > MAX_ZARR_UNCOMPRESSED_BYTES:
                    raise ValueError("Archive exceeds 128 MiB uncompressed safety limit.")

                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as src, target.open("wb") as dst:
                    while True:
                        chunk = src.read(65536)
                        if not chunk:
                            break
                        total_written += len(chunk)
                        if total_written > MAX_ZARR_UNCOMPRESSED_BYTES:
                            raise ValueError(
                                "Decompressed archive exceeds 128 MiB safety limit."
                            )
                        dst.write(chunk)
    except zipfile.BadZipFile as exc:
        raise ValueError("Corrupt or invalid .zip archive.") from exc
    finally:
        archive_path.unlink(missing_ok=True)

    # Locate the OME-Zarr root (either extract_root itself or a nested *.ome.zarr / *.zarr directory)
    if (extract_root / ".zgroup").exists() or (extract_root / ".zarray").exists():
        return extract_root

    candidates: list[Path] = []
    for p in sorted(extract_root.rglob("*")):
        if p.is_dir() and (
            p.name.lower().endswith((".ome.zarr", ".zarr"))
            or (p / ".zgroup").exists()
            or (p / ".zarray").exists()
        ):
            candidates.append(p.resolve())
    if not candidates:
        raise ValueError(
            "Uploaded .zip archive does not contain an OME-Zarr v0.4 (.ome.zarr) store."
        )
    # Prefer top-most .ome.zarr store (shortest relative path)
    candidates.sort(key=lambda c: len(c.relative_to(extract_root).parts))
    return candidates[0]


def stage_upload_bytes(
    raw_bytes: bytes,
    filename: str,
    *,
    max_bytes: int = MAX_UPLOAD_ARCHIVE_BYTES,
) -> Path:
    """Stage uploaded file bytes into a fresh UUID directory and return the resolved target `Path`."""
    if len(raw_bytes) == 0:
        raise ValueError("Uploaded file is empty (0 bytes).")
    if len(raw_bytes) > max_bytes:
        raise ValueError(
            f"File exceeds {max_bytes // (1024 * 1024)} MiB public upload limit "
            f"({len(raw_bytes) / (1024 * 1024):.1f} MiB)."
        )

    safe_name = _sanitize_filename(filename)
    root = get_upload_stage_root()
    slot_dir = (root / uuid.uuid4().hex).resolve()
    slot_dir.mkdir(parents=True, exist_ok=False)

    try:
        if safe_name.lower().endswith(".zip"):
            staged_path = _extract_safe_zarr_zip(raw_bytes, slot_dir)
        else:
            staged_path = (slot_dir / safe_name).resolve()
            staged_path.write_bytes(raw_bytes)
    except Exception:
        shutil.rmtree(slot_dir, ignore_errors=True)
        raise

    prune_staged_uploads()
    return staged_path


def _is_zarr_dir(path: Path) -> bool:
    name_low = path.name.lower()
    return path.is_dir() and (
        name_low.endswith((".zarr", ".ome.zarr"))
        or (path / ".zgroup").exists()
        or (path / ".zarray").exists()
    )


def inspect_staged_volume(
    path: str | Path,
    *,
    display_name: str | None = None,
    max_preview_plane_side: int = 96,
) -> dict[str, Any]:
    """Inspect a staged volume out-of-core for pre-solve histogram, calibration, and slice preview."""
    p = Path(path).resolve()
    has_embedded_labels = False

    if _is_zarr_dir(p):
        loaded = read_ome_zarr(p, level=0, as_set=True)
        if isinstance(loaded, VolumeSet):
            has_embedded_labels = bool(loaded.labels)
            preview_vol = loaded.image
        else:
            preview_vol = loaded
        orig_shape = tuple(int(s) for s in preview_vol.shape)
    else:
        with open_volume(p) as lv:
            orig_shape = tuple(int(s) for s in lv.shape)
            max_preload = 192
            if any(s > max_preload for s in orig_shape):
                slices = tuple(
                    slice((s - max_preload) // 2, (s - max_preload) // 2 + max_preload)
                    if s > max_preload
                    else slice(0, s)
                    for s in orig_shape
                )
                preview_vol = lv.read_region(slices)
            else:
                preview_vol = lv.to_volume(max_bytes=256 * 1024**2)

    arr = np.asarray(preview_vol.data)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D volume (z, y, x), got {arr.ndim}D shape={arr.shape}.")

    finite_vals = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.floating) else arr.ravel()
    if finite_vals.size == 0:
        raise ValueError("Uploaded volume contains no finite voxel values.")

    vmin = float(np.min(finite_vals))
    vmax = float(np.max(finite_vals))
    unique_sample = np.unique(finite_vals[: min(finite_vals.size, 262144)])
    is_integer_labels = bool(
        np.issubdtype(arr.dtype, np.integer) and int(unique_sample.size) <= 8
    )

    if vmax > vmin:
        try:
            _, otsu_val = oi.binarize_otsu(arr)
            otsu_threshold = float(otsu_val)
        except Exception:
            otsu_threshold = 0.5 * (vmin + vmax)
        counts, edges = np.histogram(finite_vals, bins=64, range=(vmin, vmax))
    else:
        otsu_threshold = vmin
        counts = np.array([int(finite_vals.size)], dtype=np.int64)
        edges = np.array([vmin - 0.5, vmax + 0.5], dtype=np.float64)

    # Extract middle XZ slice (and downsample stride if > max_preview_plane_side to keep browser Store tiny)
    pz, py, px = arr.shape
    raw_xz = np.asarray(arr[:, py // 2, :], dtype=np.float32)
    stride_z = max(1, int(np.ceil(pz / float(max_preview_plane_side))))
    stride_x = max(1, int(np.ceil(px / float(max_preview_plane_side))))
    preview_xz = np.round(raw_xz[::stride_z, ::stride_x], 5).tolist()

    calibrated = bool(preview_vol.is_calibrated)
    cal_um = (
        [round(float(v), 5) for v in preview_vol.spacing_in("um")]
        if calibrated
        else None
    )

    return {
        "path": str(p),
        "name": display_name or p.name,
        "shape_zyx": list(orig_shape),
        "preview_shape_zyx": [int(pz), int(py), int(px)],
        "dtype": str(arr.dtype),
        "is_calibrated": calibrated,
        "calibrated_voxel_um_zyx": cal_um,
        "is_integer_labels": is_integer_labels,
        "has_embedded_labels": has_embedded_labels,
        "vmin": round(vmin, 6),
        "vmax": round(vmax, 6),
        "otsu_threshold": round(otsu_threshold, 6),
        "hist_bin_edges": [round(float(x), 6) for x in edges],
        "hist_counts": [int(c) for c in counts],
        "mid_slice_xz": preview_xz,
    }
