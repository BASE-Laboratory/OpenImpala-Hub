"""Unit and integration tests for `v0.4.0` (`BatteryExplorer` chunked I/O, 4D metrics, native ROI solve, and PorousDiff)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from basecoat import Volume, write_ome_zarr
import numpy as np
import pytest

from openimpala_hub.contracts import MicrostructureSpec, SolveSpec
from openimpala_hub.engine.chunked_io import (
    close_volume_handles,
    dataset_intensity_range,
    dataset_structural_metrics,
    extract_isosurface_mesh,
    extract_native_roi_volume,
    fit_reference_circle,
    load_chunked_volume,
    load_echem_csv,
    load_lazy_slice,
    n_pages,
    render_slice_with_roi_overlay,
)
from openimpala_hub.engine.operando_demo import ensure_operando_demo_staged
from openimpala_hub.engine.solver import run_transport_pipeline


def test_concurrent_lazy_tiff_slice_reads_no_ifd_truncation(tmp_path: Path) -> None:
    """8-thread concurrent `load_lazy_slice` + `n_pages` on a fresh multi-page TIFF must never race."""
    demo = ensure_operando_demo_staged(tmp_path)
    close_volume_handles()
    tif_path = str(demo.timepoints[1].path)

    def _worker(i: int) -> tuple[int, tuple[int, int], bool]:
        tot = n_pages(tif_path)
        s = load_lazy_slice(tif_path, (i * 7) % tot, cmap="magma" if i % 2 else "grayscale")
        png = render_slice_with_roi_overlay(s, roi_box_xy=(86, 138, 54, 106))
        return tot, s["orig_hw"], png.startswith("data:image/png;base64,")

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_worker, range(24)))

    for tot, hw, ok_png in results:
        assert tot == 64
        assert hw == (160, 160)
        assert ok_png


def test_dask_threaded_vs_synchronous_slab_pooling_and_ome_zarr(tmp_path: Path) -> None:
    """Dask threaded slab pooling matches synchronous execution bit-for-bit on both `.tif` and `.ome.zarr`."""
    demo = ensure_operando_demo_staged(tmp_path)
    tif_path = str(demo.timepoints[1].path)

    close_volume_handles()
    v_sync = load_chunked_volume(tif_path, z_step=4, xy_step=8, scheduler="synchronous")
    close_volume_handles()
    v_thr = load_chunked_volume(tif_path, z_step=4, xy_step=8, scheduler="threads", num_workers=4)

    assert v_sync["shape"] == v_thr["shape"] == (16, 20, 20)
    assert np.array_equal(v_sync["x"], v_thr["x"])
    assert np.array_equal(v_sync["y"], v_thr["y"])
    assert np.array_equal(v_sync["z"], v_thr["z"])
    assert np.array_equal(v_sync["value"], v_thr["value"])
    assert v_sync["lo"] == pytest.approx(v_thr["lo"])
    assert v_sync["hi"] == pytest.approx(v_thr["hi"])

    # Also verify OME-Zarr path through chunked_io
    zarr_dir = tmp_path / "chunked_check.ome.zarr"
    vol_bc = Volume.from_array(
        v_sync["value"].reshape(v_sync["shape"]),
        voxel_size_um=(6.4, 12.8, 12.8),
    )
    write_ome_zarr(vol_bc, zarr_dir, multiscale_levels=1, chunks=(4, 20, 20), overwrite=True)
    s_zarr = load_lazy_slice(str(zarr_dir), 8)
    assert s_zarr["orig_hw"] == (20, 20)
    v_zarr = load_chunked_volume(str(zarr_dir), z_step=2, xy_step=2)
    assert v_zarr["shape"] == (8, 10, 10)


def test_operando_structural_metrics_circle_rim_and_isosurface_cutaway(tmp_path: Path) -> None:
    """Verify interface Z detection skips zero padding, rim check stays flat, and plating void grows."""
    demo = ensure_operando_demo_staged(tmp_path)
    paths = tuple(str(tp.path) for tp in demo.timepoints)

    lo, hi = dataset_intensity_range(paths)
    assert hi > lo > 0.0

    df = load_echem_csv(str(demo.echem_csv))
    assert len(df) > 100
    assert {"time_h", "V", "I"}.issubset(df.columns)

    metrics = dataset_structural_metrics(paths, z_step=2, xy_step=4)
    # Must locate active plating plane near z=50 (and never pick zero padding z=0..2)
    assert abs(metrics["interface_z"] - 50) <= 4
    # Outer Ag rim is brighter than centre (>1.03) and static within +-2% across all 5 scans
    rims = np.asarray(metrics["rim_ratio"])
    assert float(rims.mean()) > 1.03
    assert float(rims.max() - rims.min()) < 0.025
    # 7.5h plating void+Li fraction exceeds 0h pristine
    assert metrics["low_atten_fraction"][1] > metrics["low_atten_fraction"][0]

    # Verify marching-cubes rim masking & cutaway reduce triangle count cleanly
    circ = fit_reference_circle(paths[1])
    vol = load_chunked_volume(paths[1], z_step=2, xy_step=4)
    _, faces_full = extract_isosurface_mesh(vol, 120.0, rim_cutoff=1.0, circle=circ)
    _, faces_masked = extract_isosurface_mesh(
        vol,
        120.0,
        rim_cutoff=0.78,
        circle=circ,
        clip_box=((20.0, 140.0), (20.0, 140.0), (8.0, 58.0)),
    )
    assert len(faces_full) > len(faces_masked) > 0


def test_native_roi_drilldown_and_porousdiff_solve(tmp_path: Path) -> None:
    """Verify 1x1x1 native ROI extraction on Ag05 crack and PorousDiff generation run through OpenImpala."""
    demo = ensure_operando_demo_staged(tmp_path)
    roi_vol, thr = extract_native_roi_volume(
        str(demo.timepoints[1].path),
        x_range=(86, 126),
        y_range=(60, 100),
        z_range=(40, 60),
        voxel_size_um=demo.voxel_size_um,
        pristine_ref_path=str(demo.timepoints[0].path),
    )
    assert roi_vol.shape == (20, 40, 40)
    assert thr > 1000.0

    ms_roi = MicrostructureSpec(
        source_kind="benchmark_dataset",
        label="Ag05 Crack ROI",
        shape=roi_vol.shape,
        voxel_size_um=(1.6, 1.6, 1.6),
    )
    run_roi = run_transport_pipeline(
        ms_roi,
        SolveSpec(directions=("z",), compute_mechanics=False),
        existing_volume=roi_vol,
    )
    assert 0.02 < run_roi.summary.porosity < 0.95
    assert run_roi.summary.data_origin == "benchmark_xct"

    # Verify PorousDiff generator works out-of-the-box without requiring staged files
    ms_pd = MicrostructureSpec(
        source_kind="porousdiff_gallery",
        label="PorousDiff Test",
        shape=(28, 28, 28),
        voxel_size_um=(1.6, 1.6, 1.6),
        clamping_kpa=180.0,
        porosity=0.76,
        seed=42,
    )
    run_pd = run_transport_pipeline(ms_pd, SolveSpec(directions=("z",), compute_mechanics=False))
    assert 0.25 < run_pd.summary.porosity < 0.92
    assert run_pd.summary.directional["z"].tortuosity is not None
