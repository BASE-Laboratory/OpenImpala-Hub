"""Microstructure synthesis, benchmark loading, out-of-core uploads, and PorousDiff gallery loader.

Supports four honest `data_origin` provenance categories:
- `'synthetic'`: Procedural 3D generators (`carbon_felt`, `gaussian_blobs`, `overlapping_spheres`,
  `multiphase_electrode`, `open_cube`).
- `'porousdiff'`: Pre-computed Isambard-AI GH200 conditional 3D diffusion `.ome.zarr` stores
  (or live checkpoint inference when `PorousDiff` + `.pt` weights are mounted).
- `'benchmark_xct'`: Curated SHA-256 verified volumes from `openimpala.datasets`.
- `'user_upload'`: User-supplied `.ome.zarr`, `.tif`/`.tiff`, `.npy`, or `.h5`/`.hdf5` files
  ingested via `basecoat` out-of-core `open_volume` / `read_ome_zarr`.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage
from basecoat import Volume, VolumeSet, open_volume, read_ome_zarr
import openimpala as oi

from ..contracts import DataOrigin, MicrostructureSpec


def _repo_porousdiff_dir() -> Path:
    env_dir = os.environ.get("OPENIMPALA_HUB_POROUSDIFF_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return Path(__file__).resolve().parents[3] / "data" / "porousdiff"


def list_porousdiff_gallery(gallery_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Discover drop-in `PorousDiff` `.ome.zarr`, `.tif`, or `.npy` realisations on disk."""
    root = Path(gallery_dir) if gallery_dir is not None else _repo_porousdiff_dir()
    if not root.is_dir():
        return []

    entries: list[dict[str, Any]] = []
    manifest_file = root / "gallery_manifest.json"
    if manifest_file.is_file():
        try:
            raw = json.loads(manifest_file.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                for item in raw:
                    rel = Path(str(item.get("path", "")))
                    full = (root / rel).resolve() if not rel.is_absolute() else rel
                    if full.exists():
                        entries.append({**item, "resolved_path": str(full)})
                if entries:
                    return entries
        except (OSError, ValueError):
            pass

    for candidate in sorted(root.iterdir()):
        name_low = candidate.name.lower()
        if candidate.is_dir() and (name_low.endswith(".ome.zarr") or name_low.endswith(".zarr")):
            kpa_match = re.search(r"(\d+(?:\.\d+)?)\s*kpa", name_low)
            entries.append(
                {
                    "id": candidate.stem.removesuffix(".ome"),
                    "label": candidate.name,
                    "clamping_kpa": float(kpa_match.group(1)) if kpa_match else 150.0,
                    "resolved_path": str(candidate.resolve()),
                }
            )
        elif candidate.is_file() and name_low.endswith((".tif", ".tiff", ".npy")):
            kpa_match = re.search(r"(\d+(?:\.\d+)?)\s*kpa", name_low)
            entries.append(
                {
                    "id": candidate.stem,
                    "label": candidate.name,
                    "clamping_kpa": float(kpa_match.group(1)) if kpa_match else 150.0,
                    "resolved_path": str(candidate.resolve()),
                }
            )
    return entries


def strain_from_clamping_kpa(clamping_kpa: float) -> float:
    """Map physical clamping load `sigma` [kPa] (`0..600 kPa`) to compressive strain `eps_z` (`0..0.74`).

    Follows the non-linear stiffening law of fibrous carbon-felt electrodes: rapid initial
    bedding-in below 180 kPa followed by high-load pore throat closure above 320 kPa.
    """
    p = max(0.0, float(clamping_kpa))
    return float(0.80 * (1.0 - math.exp(-p / 215.0)))


def generate_carbon_felt(
    shape: tuple[int, int, int] = (64, 64, 64),
    *,
    base_porosity: float = 0.84,
    compression_strain: float = 0.18,
    fiber_radius_vox: float = 1.9,
    in_plane_bias: float = 0.22,
    coating_fraction: float = 0.08,
    seed: int = 42,
) -> np.ndarray:
    """Synthesise a 3D anisotropic non-woven fibrous carbon-felt microstructure `(Nz, Ny, Nx)`.

    Physics & geometry:
    - Cylindrical carbon fibres (`phase = 1`) are oriented predominantly in the `XY` bedding
      plane (`|dz/dl| ~ in_plane_bias`), with a minor through-plane Z-bridging population so
      mechanical Z-compression brings adjacent fibre mats into electrical contact.
    - Applying `compression_strain` (`eps_z in [0, 0.76]`) compresses fibre Z-centres and
      vertical pitch by `1 / (1 - eps_z)`, increasing solid fraction from `1 - base_porosity`
      at zero load up to dense jammed mats at high load while preserving deterministic seed
      correspondence across a compression sweep.
    - When `coating_fraction > 0`, a hierarchical surface-supported catalyst/binder shell
      (`phase = 2`) is deposited exclusively onto pore-adjacent fibre surfaces (zero floating
      catalyst islands).
    """
    nz, ny, nx = (int(shape[0]), int(shape[1]), int(shape[2]))
    eps_z = min(max(float(compression_strain), 0.0), 0.76)
    r_fib = max(1.0, float(fiber_radius_vox))
    rng = np.random.default_rng(int(seed))

    base_solid = min(max(1.0 - float(base_porosity), 0.06), 0.55)
    compaction_factor = 1.0 / max(1.0 - eps_z, 0.22)
    target_backbone_solid = min(base_solid * compaction_factor, 0.82)

    domain_vol = float(nz * ny * nx)
    mean_len = 1.35 * float(max(ny, nx))
    single_fiber_vol = math.pi * (r_fib**2) * mean_len
    n_fibers = max(14, int(round(1.38 * target_backbone_solid * domain_vol / single_fiber_vol)))

    # Reference Z-thickness prior to compression so the SAME seed tracks physical compression
    nz_uncompressed = float(nz) * compaction_factor
    z_scale = 1.0 / compaction_factor

    vol = np.zeros((nz, ny, nx), dtype=np.int32)
    r2 = r_fib * r_fib
    pad = int(math.ceil(r_fib + 1.0))

    for i in range(n_fibers):
        cy = float(rng.uniform(-0.1 * ny, 1.1 * ny))
        cx = float(rng.uniform(-0.1 * nx, 1.1 * nx))
        cz_ref = float(rng.uniform(0.0, nz_uncompressed))
        cz = cz_ref * z_scale
        if cz < -r_fib or cz > nz + r_fib:
            continue

        phi = float(rng.uniform(0.0, math.pi))
        # ~16% inclined Z-tie fibres + ~84% XY-bedding fibres flattened further under compression
        if (i % 6) == 0:
            tilt = float(rng.normal(0.0, 0.62 * (1.0 - 0.45 * eps_z)))
        else:
            tilt = float(rng.normal(0.0, float(in_plane_bias) * (1.0 - 0.55 * eps_z)))

        ux = math.cos(tilt) * math.cos(phi)
        uy = math.cos(tilt) * math.sin(phi)
        uz = math.sin(tilt)

        half_len = 0.68 * float(max(ny, nx, nz))
        p0z, p0y, p0x = cz - half_len * uz, cy - half_len * uy, cx - half_len * ux
        p1z, p1y, p1x = cz + half_len * uz, cy + half_len * uy, cx + half_len * ux

        z0 = max(0, int(math.floor(min(p0z, p1z))) - pad)
        z1 = min(nz, int(math.ceil(max(p0z, p1z))) + pad + 1)
        y0 = max(0, int(math.floor(min(p0y, p1y))) - pad)
        y1 = min(ny, int(math.ceil(max(p0y, p1y))) + pad + 1)
        x0 = max(0, int(math.floor(min(p0x, p1x))) - pad)
        x1 = min(nx, int(math.ceil(max(p0x, p1x))) + pad + 1)
        if z0 >= z1 or y0 >= y1 or x0 >= x1:
            continue

        zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
        dz = zz - p0z
        dy = yy - p0y
        dx = xx - p0x
        vz, vy, vx = (p1z - p0z), (p1y - p0y), (p1x - p0x)
        seg_len2 = max(vz * vz + vy * vy + vx * vx, 1e-9)
        t = np.clip((dz * vz + dy * vy + dx * vx) / seg_len2, 0.0, 1.0)
        dist2 = (dz - t * vz) ** 2 + (dy - t * vy) ** 2 + (dx - t * vx) ** 2
        sub = vol[z0:z1, y0:y1, x0:x1]
        sub[dist2 <= r2] = 1

    # Deposit hierarchical surface-supported active coating (`phase = 2`) into adjacent pore shell
    coat_target = max(0.0, float(coating_fraction)) * compaction_factor
    if coat_target > 1e-4:
        dilated = ndimage.binary_dilation(vol == 1, iterations=1)
        shell_mask = dilated & (vol == 0)
        shell_count = int(np.count_nonzero(shell_mask))
        if shell_count > 0:
            # Correlated patchy catalyst/CBD deposition along fibre surfaces
            noise = rng.standard_normal((nz, ny, nx)).astype(np.float32)
            smooth = ndimage.gaussian_filter(noise, sigma=1.6, mode="nearest")
            desired_vox = min(shell_count, int(round(coat_target * domain_vol)))
            if desired_vox > 0:
                shell_scores = smooth[shell_mask]
                q = max(0.0, 1.0 - float(desired_vox) / float(shell_count))
                thresh = float(np.quantile(shell_scores, q))
                coat_vox = shell_mask & (smooth >= thresh)
                vol[coat_vox] = 2

    return np.ascontiguousarray(vol, dtype=np.int32)


def _binarize_uploaded_volume(
    vol: Volume,
    *,
    threshold: float | str | None,
    dark_is_pore: bool,
) -> tuple[Volume, str]:
    """Convert an uploaded Volume into an integer phase Volume while preserving axes."""
    arr = vol.data
    thr_mode = threshold.strip().lower() if isinstance(threshold, str) else None

    if thr_mode == "labels":
        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(
                f"Cannot treat floating-point volume ({arr.dtype.name}) directly as integer phase labels; "
                "choose Otsu or Manual threshold."
            )
        int_arr = np.ascontiguousarray(arr, dtype=np.int32)
        return (
            Volume(
                int_arr,
                vol.axes,
                space=vol.space,
                centering=vol.centering,
                label_names=vol.label_names or {0: "pore", 1: "solid"},
                provenance=vol.provenance,
            ),
            f"integer phase labels ({arr.dtype.name})",
        )

    if np.issubdtype(arr.dtype, np.integer) and threshold is None:
        unique_vals = np.unique(arr)
        if unique_vals.size <= 8:
            int_arr = np.ascontiguousarray(arr, dtype=np.int32)
            return (
                Volume(
                    int_arr,
                    vol.axes,
                    space=vol.space,
                    centering=vol.centering,
                    label_names=vol.label_names or {0: "pore", 1: "solid"},
                    provenance=vol.provenance,
                ),
                f"integer phase labels ({arr.dtype.name})",
            )

    if threshold is None or thr_mode == "otsu":
        seg, thr_val = oi.binarize_otsu(
            arr,
            pore_value=0,
            solid_value=1,
            dark_is_pore=dark_is_pore,
        )
        precision = f"{arr.dtype.name} -> Otsu threshold ({thr_val:.4g})"
    else:
        thr_val = float(threshold)
        if dark_is_pore:
            seg = np.where(arr <= thr_val, 0, 1).astype(np.int32)
        else:
            seg = np.where(arr > thr_val, 0, 1).astype(np.int32)
        precision = f"{arr.dtype.name} -> manual threshold ({thr_val:.4g})"

    return (
        Volume(
            np.ascontiguousarray(seg, dtype=np.int32),
            vol.axes,
            space=vol.space,
            centering=vol.centering,
            label_names={0: "pore", 1: "solid"},
            provenance={**vol.provenance, "segmentation": precision},
        ),
        precision,
    )


def load_or_generate_microstructure(
    spec: MicrostructureSpec,
) -> tuple[Volume, DataOrigin, str, tuple[int, int, int]]:
    """Build or load a calibrated (or explicitly uncalibrated) `basecoat.Volume` from `spec`.

    Returns `(phase_volume, data_origin, precision_tag, original_shape_zyx)`.
    """
    kind = spec.source_kind

    if kind == "open_cube":
        arr = np.full(spec.shape, int(0), dtype=np.int32)
        vol = Volume.from_array(
            arr,
            voxel_size_um=spec.voxel_size_um,
            label_names={0: "pore"},
            provenance={"generator": "open_cube", "seed": spec.seed, "data_origin": "synthetic"},
        )
        return vol, "synthetic", "int32 analytical uniform open domain", arr.shape

    if kind == "carbon_felt":
        strain = (
            spec.compression_strain
            if spec.compression_strain >= 0.0
            else strain_from_clamping_kpa(spec.clamping_kpa)
        )
        arr = generate_carbon_felt(
            spec.shape,
            base_porosity=max(spec.porosity, 0.78),
            compression_strain=strain,
            fiber_radius_vox=spec.fiber_radius_vox,
            in_plane_bias=max(0.08, float(spec.anisotropy_zyx[0]) * 0.45),
            coating_fraction=spec.coating_fraction,
            seed=spec.seed,
        )
        labels = {0: "pore", 1: "carbon_fiber"}
        if spec.coating_fraction > 0.0:
            labels[2] = "active_coating"
        vol = Volume.from_array(
            arr,
            voxel_size_um=spec.voxel_size_um,
            label_names=labels,
            provenance={
                "generator": "carbon_felt",
                "seed": spec.seed,
                "compression_strain": strain,
                "clamping_kpa": spec.clamping_kpa,
                "data_origin": "synthetic",
            },
        )
        return vol, "synthetic", "int32 procedural 3D fibrous mat", arr.shape

    if kind == "gaussian_blobs":
        arr = oi.synthetic.gaussian_blobs(
            shape=spec.shape,
            porosity=spec.porosity,
            blobiness=spec.blobiness,
            anisotropy=spec.anisotropy_zyx,
            pore_value=0,
            solid_value=1,
            seed=spec.seed,
        )
        vol = Volume.from_array(
            np.ascontiguousarray(arr, dtype=np.int32),
            voxel_size_um=spec.voxel_size_um,
            label_names={0: "pore", 1: "solid"},
            provenance={"generator": "openimpala.synthetic.gaussian_blobs", "seed": spec.seed},
        )
        return vol, "synthetic", "int32 synthetic Gaussian random field", arr.shape

    if kind == "overlapping_spheres":
        arr = oi.synthetic.overlapping_spheres(
            shape=spec.shape,
            porosity=spec.porosity,
            radius=spec.sphere_radius_vox,
            pore_value=0,
            solid_value=1,
            seed=spec.seed,
        )
        vol = Volume.from_array(
            np.ascontiguousarray(arr, dtype=np.int32),
            voxel_size_um=spec.voxel_size_um,
            label_names={0: "pore", 1: "solid_particles"},
            provenance={"generator": "openimpala.synthetic.overlapping_spheres", "seed": spec.seed},
        )
        return vol, "synthetic", "int32 synthetic Boolean sphere pack", arr.shape

    if kind == "multiphase_electrode":
        pore_f = max(0.08, min(0.85, float(spec.porosity)))
        rem = 1.0 - pore_f
        binder_share = min(float(spec.binder_fraction), rem * 0.45)
        active_share = max(0.05, rem - binder_share)
        arr = oi.synthetic.multiphase_electrode(
            shape=spec.shape,
            pore_fraction=pore_f,
            active_fraction=active_share,
            binder_fraction=binder_share,
            blobiness=spec.blobiness,
            seed=spec.seed,
        )
        vol = Volume.from_array(
            np.ascontiguousarray(arr, dtype=np.int32),
            voxel_size_um=spec.voxel_size_um,
            label_names={0: "pore_electrolyte", 1: "active_material", 2: "carbon_binder_domain"},
            provenance={"generator": "openimpala.synthetic.multiphase_electrode", "seed": spec.seed},
        )
        return vol, "synthetic", "int32 synthetic 3-phase battery cathode", arr.shape

    if kind == "benchmark_dataset":
        ds_name = spec.dataset_name or "two_phase_cubic"
        raw_arr = oi.datasets.fetch_example(ds_name)
        orig_shape = tuple(int(s) for s in raw_arr.shape)
        # Note: openimpala.datasets uses phase 1 = pore, phase 0 = solid; remap to Hub convention 0=pore, 1=solid
        remapped = np.where(raw_arr == 1, 0, 1).astype(np.int32)
        if spec.roi_zyx is not None:
            (z0, z1), (y0, y1), (x0, x1) = spec.roi_zyx
            remapped = remapped[z0:z1, y0:y1, x0:x1]
        vol = Volume.from_array(
            np.ascontiguousarray(remapped, dtype=np.int32),
            voxel_size_um=spec.voxel_size_um,
            label_names={0: "pore", 1: "solid"},
            provenance={
                "dataset": f"openimpala.datasets.{ds_name}",
                "remapped_conducting_phase": "1->0",
            },
        )
        return vol, "benchmark_xct", f"int32 benchmark ({ds_name})", orig_shape

    if kind == "porousdiff_gallery":
        entries = list_porousdiff_gallery()
        if not entries:
            raise FileNotFoundError(
                "No PorousDiff `.ome.zarr` realisations found in `data/porousdiff/`. "
                "Export calibrated `.ome.zarr` compression samples from Isambard-AI into "
                "`data/porousdiff/` (or set $OPENIMPALA_HUB_POROUSDIFF_DIR), or select "
                "the procedural 'Anisotropic Carbon Felt' source."
            )
        chosen = min(
            entries,
            key=lambda e: abs(float(e.get("clamping_kpa", 150.0)) - float(spec.clamping_kpa)),
        )
        p = Path(str(chosen["resolved_path"]))
        return _load_path_microstructure(
            p,
            spec=spec,
            origin="porousdiff",
            default_precision="PorousDiff conditional 3D diffusion (.ome.zarr)",
        )

    if kind == "user_upload":
        if not spec.upload_path:
            raise ValueError("`upload_path` is required when `source_kind='user_upload'`.")
        return _load_path_microstructure(
            Path(spec.upload_path),
            spec=spec,
            origin="user_upload",
            default_precision="user uploaded volume",
        )

    raise ValueError(f"Unsupported source_kind={kind!r}.")


def _load_path_microstructure(
    path: Path,
    *,
    spec: MicrostructureSpec,
    origin: DataOrigin,
    default_precision: str,
) -> tuple[Volume, DataOrigin, str, tuple[int, int, int]]:
    """Read a volume or bounded ROI out-of-core via `basecoat` without full-RAM spikes."""
    name_low = path.name.lower()
    is_zarr = path.is_dir() and (
        name_low.endswith((".zarr", ".ome.zarr"))
        or (path / ".zgroup").exists()
        or (path / ".zarray").exists()
    )

    if is_zarr:
        loaded = read_ome_zarr(path, level=0, as_set=True)
        if isinstance(loaded, VolumeSet) and loaded.labels:
            lbl_key = "phase" if "phase" in loaded.labels else sorted(loaded.labels.keys())[0]
            base_vol = loaded.labels[lbl_key]
            precision = f"{default_precision} [labels/{lbl_key}]"
        else:
            img_vol = loaded.image if isinstance(loaded, VolumeSet) else loaded
            base_vol, seg_desc = _binarize_uploaded_volume(
                img_vol, threshold=spec.threshold, dark_is_pore=spec.dark_is_pore
            )
            precision = f"{default_precision} ({seg_desc})"
        orig_shape = tuple(int(s) for s in base_vol.shape)
        if spec.roi_zyx is not None:
            (z0, z1), (y0, y1), (x0, x1) = spec.roi_zyx
            base_vol = base_vol[z0:z1, y0:y1, x0:x1]
        if (spec.override_calibrated_spacing or not base_vol.is_calibrated) and (
            spec.voxel_size_um is not None
        ):
            base_vol = base_vol.with_voxel_size_um(spec.voxel_size_um)
        return base_vol, origin, precision, orig_shape

    init_vox_um = spec.voxel_size_um if spec.override_calibrated_spacing else None
    with open_volume(path, voxel_size_um=init_vox_um) as lv:
        orig_shape = tuple(int(s) for s in lv.shape)
        if spec.roi_zyx is not None:
            (z0, z1), (y0, y1), (x0, x1) = spec.roi_zyx
            region = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
            raw_vol = lv.read_region(region)
        else:
            # Protect RAM on multi-GB uploads by reading up to 256^3 centre slab if uncapped
            max_preload = 256
            if any(s > max_preload for s in orig_shape):
                slices = tuple(
                    slice((s - max_preload) // 2, (s - max_preload) // 2 + max_preload)
                    if s > max_preload
                    else slice(0, s)
                    for s in orig_shape
                )
                raw_vol = lv.read_region(slices)
            else:
                raw_vol = lv.to_volume(max_bytes=512 * 1024**2)

    seg_vol, seg_desc = _binarize_uploaded_volume(
        raw_vol, threshold=spec.threshold, dark_is_pore=spec.dark_is_pore
    )
    if (spec.override_calibrated_spacing or not seg_vol.is_calibrated) and (
        spec.voxel_size_um is not None
    ):
        seg_vol = seg_vol.with_voxel_size_um(spec.voxel_size_um)
    return seg_vol, origin, f"{default_precision} ({seg_desc})", orig_shape
