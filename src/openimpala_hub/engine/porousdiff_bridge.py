"""Virtual Z-compression & optional `PorousDiff` (`feature/paper-ready-conditional`) bridge (`Pillar A`).

Provides two honest, provenance-tagged virtual-compression pathways:
1. **Kinematic Solid-Mass-Conserving Virtual Compression (`kinematic_compress_volume`)**:
   Zero-PyTorch CPU level-set resampling along `Z` by `(1 - strain_z)` that conserves solid
   backbone & active-coating mass as Z-thickness decreases. Enables virtual compression
   sweeps directly on uploaded experimental XCT / FIB-SEM scans or benchmark volumes.
2. **Learned `PorousDiff` Conditional 3D UNet Hook (`has_porousdiff_runtime`)**:
   Detects when optional `porousdiff` (`v0.5+`) and a `.pt`/`.pth` checkpoint
   (`$OPENIMPALA_HUB_POROUSDIFF_CKPT`) are installed (e.g. on Modal L4 GPU or Isambard-AI).
"""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path

import numpy as np
from scipy import ndimage
from basecoat import AxisSpec, Volume


def has_porousdiff_runtime() -> bool:
    """Return True if the `porousdiff` package (`feature/paper-ready-conditional`) is importable."""
    return importlib.util.find_spec("porousdiff") is not None


def porousdiff_checkpoint_path() -> Path | None:
    """Return a resolved PorousDiff `.pt`/`.pth` checkpoint path if configured on disk."""
    raw = os.environ.get("OPENIMPALA_HUB_POROUSDIFF_CKPT")
    if not raw:
        return None
    p = Path(raw).expanduser().resolve()
    return p if p.is_file() else None


def kinematic_compress_volume(
    vol: Volume,
    strain_z: float,
    *,
    conducting_phase: int = 0,
) -> Volume:
    """Compress a 3D phase `Volume` along `Z` by `strain_z` while conserving solid voxel mass.

    Uses signed-distance level-set interpolation (`EDT(solid) - EDT(pore)`) mapped onto the
    compressed slab thickness `Nz_new = max(8, round(Nz * (1 - strain_z)))`, selecting the
    exact iso-contour quantile that preserves `N_solid` (up to a physical jamming ceiling of
    `92%` solid fraction). If a secondary coating phase (`phase == 2`) exists, its mass is
    re-allocated onto the compressed pore-adjacent solid shell.
    """
    eps = min(max(float(strain_z), 0.0), 0.65)
    if eps <= 1e-4:
        return vol

    arr = np.asarray(vol.data, dtype=np.int32)
    nz, ny, nx = arr.shape
    nz_new = max(8, int(round(nz * (1.0 - eps))))
    if nz_new >= nz:
        return vol

    solid_mask = arr != int(conducting_phase)
    orig_solid_vox = int(np.count_nonzero(solid_mask))
    if orig_solid_vox == 0:
        # Uniform open domain stays open under slab thinning
        out_arr = np.full((nz_new, ny, nx), int(conducting_phase), dtype=np.int32)
    elif orig_solid_vox == arr.size:
        out_arr = np.full((nz_new, ny, nx), int(arr.flat[0]), dtype=np.int32)
    else:
        coat_vox_orig = int(np.count_nonzero(arr == 2))
        backbone_vox_orig = max(0, orig_solid_vox - coat_vox_orig)

        # Signed distance field (>0 inside solid, <0 inside pore) for smooth boundary kinematics
        sdf = ndimage.distance_transform_edt(solid_mask) - ndimage.distance_transform_edt(
            ~solid_mask
        )
        # Resample SDF along Z from nz -> nz_new via linear spline (order=1)
        zoom_factors = (float(nz_new) / float(nz), 1.0, 1.0)
        sdf_comp = ndimage.zoom(sdf.astype(np.float32), zoom=zoom_factors, order=1)

        new_total = nz_new * ny * nx
        target_solid_vox = min(int(math.floor(0.92 * new_total)), orig_solid_vox)
        comp_solid = np.zeros(sdf_comp.shape, dtype=bool)
        if target_solid_vox > 0:
            flat_sdf = sdf_comp.ravel()
            if target_solid_vox >= flat_sdf.size:
                comp_solid[...] = True
            else:
                top_idx = np.argpartition(-flat_sdf, target_solid_vox - 1)[:target_solid_vox]
                comp_solid.ravel()[top_idx] = True

        primary_solid_id = 1 if int(conducting_phase) == 0 else 0
        out_arr = np.where(comp_solid, primary_solid_id, int(conducting_phase)).astype(np.int32)

        # Preserve catalyst/binder shell (`phase == 2`) mass on compressed pore-solid interface
        if coat_vox_orig > 0 and primary_solid_id == 1:
            target_coat = min(
                coat_vox_orig,
                max(0, target_solid_vox - max(1, int(0.65 * backbone_vox_orig))),
            )
            eroded = ndimage.binary_erosion(comp_solid, iterations=1)
            surface_solid = comp_solid & (~eroded)
            surf_indices = np.flatnonzero(surface_solid)
            if surf_indices.size > 0 and target_coat > 0:
                chosen_count = min(int(surf_indices.size), int(target_coat))
                # Select boundary voxels closest to pore interface (lowest SDF among solid)
                order_idx = np.argsort(sdf_comp.ravel()[surf_indices])[:chosen_count]
                out_arr.ravel()[surf_indices[order_idx]] = 2

    ax_z, ax_y, ax_x = vol.axes
    new_axes = (
        AxisSpec(
            name=ax_z.name,
            kind=ax_z.kind,
            unit=ax_z.unit,
            scale=ax_z.scale,
            offset=ax_z.offset,
        ),
        ax_y,
        ax_x,
    )
    return Volume(
        np.ascontiguousarray(out_arr, dtype=np.int32),
        new_axes,
        space=vol.space,
        centering=vol.centering,
        label_names=vol.label_names,
        provenance={
            **vol.provenance,
            "virtual_compression": {
                "method": "kinematic_levelset_mass_conserving",
                "strain_z": round(eps, 4),
                "nz_before": int(nz),
                "nz_after": int(nz_new),
            },
        },
    )
