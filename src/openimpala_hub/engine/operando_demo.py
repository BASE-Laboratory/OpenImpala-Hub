"""Discovery of real synchrotron & micro-CT 4D/3D datasets (plus synthetic unit-test fixture).

Priority order in `discover_operando_datasets()`:
1. Externally mounted full-size multi-GB `Ag05` / `Cu04` operando TIFF stacks (`$OPENIMPALA_HUB_OPERANDO_DIR`).
2. Real committed OME-Zarr v0.4 datasets in `<repo>/docs_site/data/`:
   - `dls_i12_felt`: Diamond Light Source `I12` Virgin Carbon Felt in-situ compression series
     (`68`, `238`, `451 kPa`, `128 x 192 x 192` at `1.6 um/px`).
   - `dls_i13_zn13x`: Diamond Light Source `I13-2` Zeolite 13X capillary + 500 um ruby fiducial
     (`48 x 400 x 400` at `3.25 um/px`, Green et al., Scientific Data 12, 329 (2025)).
   - `nlm25_xct`: Segmented dendritic / porous micro-CT (`128 x 256 x 256` at `2.0 um/px`).
3. Canonical `OpenImpala` `100^3` two-phase porous medium (`SampleData_2Phase_stack_3d_1bit.tif`).
4. Synthetic `ensure_operando_demo_staged()` fallback ONLY when none of the real stores exist on disk.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np
import openimpala as oi
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


def _bundled_zarr_dir() -> Path:
    env_zarr = os.environ.get("OPENIMPALA_HUB_ZARR_DATA_DIR")
    if env_zarr:
        return Path(env_zarr).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "docs_site" / "data"


def _demo_cache_dir() -> Path:
    env_cache = os.environ.get("OPENIMPALA_HUB_DEMO_CACHE")
    if env_cache:
        return Path(env_cache).expanduser().resolve()
    return Path(tempfile.gettempdir()) / "openimpala_hub_operando_v04"


def ensure_operando_demo_staged(cache_dir: Path | None = None) -> OperandoDataset:
    """Build (if not already present) and return the deterministic 5-timepoint `uint16` test fixture."""
    root = (cache_dir or _demo_cache_dir()) / "ag_demo"
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "Ag_cell_operando_cycling.csv"

    nz, ny, nx = 64, 160, 160
    cx, cy, rad = 80.0, 80.0, 66.0

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

        coarse = rng.normal(0.0, 1.0, size=(nz, ny // 2, nx // 2)).astype(np.float32)
        tex = np.repeat(np.repeat(coarse, 2, axis=1), 2, axis=2)
        fine = rng.normal(0.0, 420.0, size=(nz, ny, nx)).astype(np.float32)
        intrinsic_pores = tex < -0.68

        base_stack = np.zeros((nz, ny, nx), dtype=np.float32)
        for z in range(3, nz):
            plane = np.where(in_disc, 25000.0 + 1350.0 * tex[z] + fine[z] + rim_boost, 0.0)
            plane = np.where(in_disc & intrinsic_pores[z], plane * 0.44, plane)
            plane = np.where(in_air_ring, 820.0 + np.abs(fine[z]) * 0.25, plane)
            base_stack[z] = plane

        states = [
            (0.0, 0.0),
            (3600.0, 1.0),
            (1100.0, 0.38),
            (4300.0, 1.25),
            (1650.0, 0.55),
        ]

        zz = np.arange(nz, dtype=np.float32)[:, None, None]
        z_prof = np.exp(-0.5 * ((zz - 50.0) / 4.2) ** 2)
        crack_curve_1 = np.abs(
            (yy[None] - 80.0) - 0.34 * (xx[None] - 112.0) - 4.0 * np.sin(xx[None] * 0.14)
        )
        crack_curve_2 = np.abs(
            (yy[None] - 83.0) + 0.48 * (xx[None] - 115.0) - 3.0 * np.cos(xx[None] * 0.16)
        )
        roi_gate = (
            (xx[None] > 86)
            & (xx[None] < 138)
            & (yy[None] > 54)
            & (yy[None] < 106)
            & (np.abs(zz - 50.0) <= 10.0)
        )

        for (_t_h, _lbl, fname), (drop_c, sev) in zip(AG_REAL_FILES, states, strict=True):
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
        Timepoint(label=lbl, path=root / fname, t_h=t_h) for t_h, lbl, fname in AG_REAL_FILES
    )
    return OperandoDataset(
        key="ag_demo",
        label="Synthetic Ag05 verification phantom",
        voxel_size_um=1.6,
        echem_csv=csv_path,
        timepoints=tps,
    )


def discover_operando_datasets() -> dict[str, OperandoDataset]:
    """Return available real 4D/3D synchrotron & micro-CT datasets (mounted + bundled OME-Zarr)."""
    found: dict[str, OperandoDataset] = {}

    # 1. Mounted full-size Ag05 / Cu04 multi-GB operando TIFF folders (if present)
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

    # 2. Real bundled synchrotron & micro-CT OME-Zarr v0.4 datasets in docs_site/data/
    zdir = _bundled_zarr_dir()
    i12_stores = [
        (0.5, "68 kPa (0.008 mV/V)", zdir / "dls_i12_068kpa.ome.zarr"),
        (1.5, "238 kPa (0.028 mV/V)", zdir / "dls_i12_238kpa.ome.zarr"),
        (2.5, "451 kPa (0.053 mV/V)", zdir / "dls_i12_451kpa.ome.zarr"),
    ]
    i12_csv = zdir / "dls_i12_compression.csv"
    if i12_csv.is_file() and all(p.is_dir() for _, _, p in i12_stores):
        found["dls_i12_felt"] = OperandoDataset(
            key="dls_i12_felt",
            label="DLS I12 Virgin Carbon Felt · Compression (68–451 kPa, 1.6 µm)",
            voxel_size_um=1.6,
            echem_csv=i12_csv,
            timepoints=tuple(Timepoint(label=lbl, path=p, t_h=th) for th, lbl, p in i12_stores),
        )

    i13_store = zdir / "dls_i13_zn13x_capillary.ome.zarr"
    i13_csv = zdir / "dls_i13_zn13x.csv"
    if i13_store.is_dir() and i13_csv.is_file():
        found["dls_i13_zn13x"] = OperandoDataset(
            key="dls_i13_zn13x",
            label="DLS I13-2 Zeolite 13X Capillary + Ruby Sphere (3.25 µm)",
            voxel_size_um=3.25,
            echem_csv=i13_csv,
            timepoints=(Timepoint(label="Scan 169067 (1.625->3.25 µm)", path=i13_store, t_h=0.5),),
        )

    nlm_store = zdir / "nlm25_dendritic_xct.ome.zarr"
    nlm_csv = zdir / "nlm25_trace.csv"
    if nlm_store.is_dir() and nlm_csv.is_file():
        found["nlm25_xct"] = OperandoDataset(
            key="nlm25_xct",
            label="Dendritic / Porous Electrode Micro-CT (nlm25, 2.0 µm)",
            voxel_size_um=2.0,
            echem_csv=nlm_csv,
            timepoints=(Timepoint(label="Slices 850–977 (256x256 crop)", path=nlm_store, t_h=0.5),),
        )

    # 3. Canonical OpenImpala 100^3 two-phase stack if cached locally
    try:
        oi_tif = oi.datasets.fetch_path("SampleData_2Phase_stack_3d_1bit.tif")
        if oi_tif.is_file() and nlm_csv.is_file():
            found["oi_2phase"] = OperandoDataset(
                key="oi_2phase",
                label="OpenImpala Two-Phase Porous Medium (100³, 1.0 µm)",
                voxel_size_um=1.0,
                echem_csv=nlm_csv,
                timepoints=(Timepoint(label="100x100x100 binary stack", path=oi_tif, t_h=0.5),),
            )
    except Exception:
        pass

    if not found:
        demo = ensure_operando_demo_staged()
        found[demo.key] = demo
    return found
