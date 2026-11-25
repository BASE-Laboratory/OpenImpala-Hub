"""Operando 4D dataset discovery, deterministic 5-timepoint demo builder, and PorousDiff staging.

Provides zero-setup access to:
1. Full experimental `Ag05` / `Cu04` multi-GB TIFF + CSV folders when mounted under
   `$OPENIMPALA_HUB_OPERANDO_DIR` or `<repo>/data/operando/`.
2. A deterministic, physically structured 5-timepoint `uint16` operando plating/stripping
   stack (`0h`, `7.5h`, `15h`, `20h`, `20.25h` multi-page `.tif` files + `cycling.csv`)
   staged once on disk so `load_lazy_slice`, `load_chunked_volume`, `fit_reference_circle`,
   `dataset_structural_metrics`, and native `step=1` ROI simulations work out of the box.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np
import tifffile as tiff

from .chunked_io import OperandoDataset, Timepoint

AG_REAL_FILES: tuple[tuple[float, str, str], ...] = (
    (0.0, "0h pristine", "Ag05_registered_0h_pristine.tif"),
    (7.5, "7.5h after 1st plating", "Ag05_registered_7.5h_after_1st_plating.tif"),
    (15.0, "15h after 1st stripping", "Ag05_registered_15h_after_1st_stripping.tif"),
    (20.0, "20h after 2nd plating", "Ag05_registered_20h_after_2nd_plating.tif"),
    (20.25, "20.25h after 2nd stripping", "Ag05_registered_20.25h_after_2nd_stripping.tif"),
)

CU_REAL_FILES: tuple[tuple[float, str, str], ...] = (
    (0.0, "0h pristine", "Cu04_registered_0h_pristine.tif"),
    (7.5, "7.5h after 1st plating", "Cu04_registered_7.5h_after_1st_plating.tif"),
    (15.0, "15h after 1st stripping", "Cu04_registered_15h_after_1st_stripping.tif"),
    (16.25, "16.25h after 2nd plating", "Cu04_registered_16.25h_after_2nd_plating.tif"),
)


def _operando_search_roots() -> list[Path]:
    roots: list[Path] = []
    env_dir = os.environ.get("OPENIMPALA_HUB_OPERANDO_DIR")
    if env_dir:
        roots.append(Path(env_dir).expanduser().resolve())
    repo_root = Path(__file__).resolve().parents[3]
    roots.append(repo_root / "data" / "operando")
    roots.append(repo_root.parent / "BatteryExplorer" / "data")
    return roots


def _demo_cache_dir() -> Path:
    env_cache = os.environ.get("OPENIMPALA_HUB_DEMO_CACHE")
    if env_cache:
        return Path(env_cache).expanduser().resolve()
    return Path(tempfile.gettempdir()) / "openimpala_hub_operando_v04"


def ensure_operando_demo_staged(cache_dir: Path | None = None) -> OperandoDataset:
    """Build (if not already present) and return the 5-timepoint `uint16` Ag05 demo dataset."""
    root = (cache_dir or _demo_cache_dir()) / "ag_demo"
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "Ag_cell_operando_cycling.csv"

    nz, ny, nx = 64, 160, 160
    cx, cy, rad = 80.0, 80.0, 66.0

    # Check if all 5 TIFFs + CSV already exist
    expected_tifs = [root / fname for _, _, fname in AG_REAL_FILES]
    if not (csv_path.is_file() and all(p.is_file() for p in expected_tifs)):
        rng = np.random.default_rng(2026)
        yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
        rr = np.hypot(xx - cx, yy - cy)
        in_disc = rr <= rad
        in_air_ring = (rr > rad) & (rr <= 1.25 * rad)
        rim_boost = np.where(
            (rr > 0.65 * rad) & in_disc,
            3400.0 * np.power(np.clip((rr - 0.65 * rad) / (0.35 * rad), 0.0, 1.0), 1.15),
            0.0,
        )

        # Static granular electrode texture + intrinsic ~26% inter-particle pore network
        coarse = rng.normal(0.0, 1.0, size=(nz, ny // 2, nx // 2)).astype(np.float32)
        tex = np.repeat(np.repeat(coarse, 2, axis=1), 2, axis=2)
        fine = rng.normal(0.0, 420.0, size=(nz, ny, nx)).astype(np.float32)
        intrinsic_pores = tex < -0.68

        base_stack = np.zeros((nz, ny, nx), dtype=np.float32)
        for z in range(3, nz):  # z=0..2 are zero registration-padding planes
            plane = np.where(in_disc, 25000.0 + 1350.0 * tex[z] + fine[z] + rim_boost, 0.0)
            plane = np.where(in_disc & intrinsic_pores[z], plane * 0.44, plane)
            # Low nonzero air scatter outside sample so air-ring median is ~800 counts
            plane = np.where(in_air_ring, 820.0 + np.abs(fine[z]) * 0.25, plane)
            base_stack[z] = plane

        # Timepoint-dependent plating/void evolution around interface z = 50
        # Tuple: (interface_drop_counts, crack_severity)
        states = [
            (0.0, 0.0),      # 0.0h pristine
            (3600.0, 1.0),   # 7.5h after 1st plating (strong darkening + crack/Li channel)
            (1100.0, 0.38),  # 15.0h after 1st stripping (partial recovery, residual void)
            (4300.0, 1.25),  # 20.0h after 2nd plating (deeper plating + wider crack)
            (1650.0, 0.55),  # 20.25h after 2nd stripping (accumulated dead Li + void)
        ]

        # Precompute branching defect geometry inside ROI [88:136, 56:104, 38:62]
        zz = np.arange(nz, dtype=np.float32)[:, None, None]
        z_prof = np.exp(-0.5 * ((zz - 50.0) / 4.2) ** 2)
        crack_curve_1 = np.abs((yy[None] - 80.0) - 0.34 * (xx[None] - 112.0) - 4.0 * np.sin(xx[None] * 0.14))
        crack_curve_2 = np.abs((yy[None] - 83.0) + 0.48 * (xx[None] - 115.0) - 3.0 * np.cos(yy[None] * 0.16))
        roi_gate = (xx[None] > 86) & (xx[None] < 138) & (yy[None] > 54) & (yy[None] < 106) & (np.abs(zz - 50.0) <= 10.0)

        for (t_h, _lbl, fname), (drop_c, sev) in zip(AG_REAL_FILES, states, strict=True):
            st = base_stack.copy()
            if drop_c > 0.0:
                interior_3d = (rr < 0.90 * rad)[None, :, :] & (zz >= 3)
                st = np.where(interior_3d, st - drop_c * z_prof, st)
            if sev > 0.0:
                w1 = 2.1 * sev
                w2 = 1.5 * sev
                in_crack = roi_gate & ((crack_curve_1 < w1) | (crack_curve_2 < w2))
                st = np.where(in_crack, 1900.0 + 300.0 * np.abs(tex), st)
            st_u16 = np.clip(np.round(st), 0, 65535).astype(np.uint16)
            tiff.imwrite(root / fname, st_u16, photometric="minisblack")

        # Write realistic galvanostatic cycling CSV (0 -> 20.5 h)
        t_vec = np.linspace(0.0, 20.5, 410)
        v_vec = np.zeros_like(t_vec)
        i_vec = np.zeros_like(t_vec)
        for idx, t in enumerate(t_vec):
            if t < 7.5:
                v_vec[idx] = -0.041 - 0.014 * np.exp(-t * 1.4)
                i_vec[idx] = -0.0015
            elif t < 15.0:
                v_vec[idx] = 0.043 + 0.019 * (((t - 7.5) / 7.5) ** 2)
                i_vec[idx] = 0.0015
            elif t < 20.0:
                v_vec[idx] = -0.047 - 0.009 * ((t - 15.0) / 5.0)
                i_vec[idx] = -0.0015
            else:
                v_vec[idx] = 0.058 + 0.025 * ((t - 20.0) / 0.5)
                i_vec[idx] = 0.0015
        lines = ["Time/h,Potential/V,Current/A"] + [
            f"{t:.4f},{v:.5f},{i:.6f}" for t, v, i in zip(t_vec, v_vec, i_vec)
        ]
        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    tps = tuple(
        Timepoint(label=lbl, path=root / fname, t_h=t_h)
        for t_h, lbl, fname in AG_REAL_FILES
    )
    return OperandoDataset(
        key="ag_demo",
        label="Silver interlayer cell (Ag05 · 5 timepoints)",
        voxel_size_um=1.6,
        echem_csv=csv_path,
        timepoints=tps,
    )


def discover_operando_datasets() -> dict[str, OperandoDataset]:
    """Return available 4D operando datasets (real mounted stacks + staged Ag05 demo)."""
    found: dict[str, OperandoDataset] = {}

    for root in _operando_search_roots():
        ag_dir = root / "ag_cell"
        if ag_dir.is_dir():
            tps = [
                Timepoint(label=lbl, path=ag_dir / fn, t_h=th)
                for th, lbl, fn in AG_REAL_FILES
                if (ag_dir / fn).is_file()
            ]
            csv_p = ag_dir / "Ag_cell_operando_cycling.csv"
            if len(tps) >= 2 and csv_p.is_file():
                found["ag_full"] = OperandoDataset(
                    key="ag_full",
                    label="Silver interlayer cell (Ag05 full stack)",
                    voxel_size_um=1.6,
                    echem_csv=csv_p,
                    timepoints=tuple(tps),
                )
        cu_dir = root / "cu_cell"
        if cu_dir.is_dir():
            tps_cu = [
                Timepoint(label=lbl, path=cu_dir / fn, t_h=th)
                for th, lbl, fn in CU_REAL_FILES
                if (cu_dir / fn).is_file()
            ]
            csv_cu = cu_dir / "Cu_cell_operando_cycling.csv"
            if len(tps_cu) >= 2 and csv_cu.is_file():
                found["cu_full"] = OperandoDataset(
                    key="cu_full",
                    label="Copper interlayer cell (Cu04 full stack)",
                    voxel_size_um=1.6,
                    echem_csv=csv_cu,
                    timepoints=tuple(tps_cu),
                )

    demo = ensure_operando_demo_staged()
    found[demo.key] = demo
    return found
