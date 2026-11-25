"""Unit and integration tests for steering, PorousDiff gallery loading, and .zip report bundles."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest
from basecoat import Volume, write_ome_zarr

from openimpala_hub.contracts import MicrostructureSpec, SolveSpec
from openimpala_hub.engine.bundle import write_report_bundle
from openimpala_hub.engine.generators import list_porousdiff_gallery
from openimpala_hub.engine.solver import run_transport_pipeline
from openimpala_hub.engine.steering import suggest_next_setpoint_gp
from openimpala_hub.presets import PRESETS


def test_bottleneck_roi_locates_constricted_throat() -> None:
    """A slab barrier with a narrow central pinhole must concentrate flux and place ROI #1 on the pinhole."""
    arr = np.zeros((32, 32, 32), dtype=np.int32)
    # Solid wall at z=14..18 except a 4x4 pinhole at y=14..18, x=14..18
    arr[14:18, :, :] = 1
    arr[14:18, 14:18, 14:18] = 0
    vol = Volume.from_array(arr, voxel_size_um=1.0, label_names={0: "pore", 1: "barrier"})

    mspec = MicrostructureSpec(source_kind="user_upload", label="Pinhole Barrier", shape=(32, 32, 32))
    run = run_transport_pipeline(
        mspec,
        SolveSpec(directions=("z",), solver="mlmg", compute_solid_wiring=False, return_fields=True),
        existing_volume=vol,
    )
    assert len(run.summary.zoom_rois) >= 1
    top_roi = run.summary.zoom_rois[0]
    (z0, z1), (y0, y1), (x0, x1) = top_roi.bounds_zyx
    # Pinhole centre (16, 16, 16) must fall inside ROI #1
    assert z0 <= 16 <= z1
    assert y0 <= 16 <= y1
    assert x0 <= 16 <= x1
    assert top_roi.peak_flux_magnitude > 4.0


def test_bayesian_gp_proposes_interior_sweet_spot() -> None:
    """Given U-shaped cell resistance observations, Matérn-5/2 GP proposes a new interior setpoint."""
    obs_x = [30.0, 160.0, 440.0]
    obs_r = [3.65, 1.48, 3.10]
    sug = suggest_next_setpoint_gp(obs_x, obs_r, bounds=(20.0, 500.0))

    assert 20.0 <= sug.suggested_value <= 500.0
    assert sug.current_best_parameter == pytest.approx(160.0)
    # Must not re-propose an already-evaluated point
    assert all(abs(sug.suggested_value - x) > 5.0 for x in obs_x)
    assert len(sug.grid_x) == len(sug.gp_mean) == len(sug.ei_curve)


def test_porousdiff_drop_in_gallery_loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop-in calibrated `.ome.zarr` stores in `OPENIMPALA_HUB_POROUSDIFF_DIR` load with `'porousdiff'` origin."""
    monkeypatch.setenv("OPENIMPALA_HUB_POROUSDIFF_DIR", str(tmp_path))
    arr = np.zeros((24, 24, 24), dtype=np.int32)
    arr[6:18, 8:16, 8:16] = 1
    vol = Volume.from_array(arr, voxel_size_um=1.6, label_names={0: "pore", 1: "carbon_fiber"})
    write_ome_zarr(vol, tmp_path / "carbon_felt_200kpa.ome.zarr", multiscale_levels=1, overwrite=True)

    entries = list_porousdiff_gallery()
    assert len(entries) == 1
    assert entries[0]["clamping_kpa"] == pytest.approx(200.0)

    mspec = MicrostructureSpec(source_kind="porousdiff_gallery", clamping_kpa=190.0)
    run = run_transport_pipeline(mspec, SolveSpec(directions=("z",), compute_solid_wiring=False))
    assert run.summary.data_origin == "porousdiff"
    assert run.summary.shape_zyx == (24, 24, 24)


def test_multiphase_bundle_export_and_reproduce_script(tmp_path: Path) -> None:
    """Composite 3-phase preset exports a valid .zip bundle whose `reproduce.py` runs cleanly."""
    preset = PRESETS["composite_cathode_3phase"]
    # Use 32^3 for fast test turnaround
    fast_micro = MicrostructureSpec(**{**preset.microstructure.to_dict(), "shape": (32, 32, 32)})
    run = run_transport_pipeline(fast_micro, preset.solve)
    assert run.summary.homogenised_tensor_xyz is not None
    assert len(run.summary.homogenised_tensor_xyz) == 3

    sug = suggest_next_setpoint_gp([40.0, 180.0, 380.0], [2.8, 1.5, 2.4])
    bundle_zip = tmp_path / "nia_bundle.zip"
    write_report_bundle(run, bundle_zip, bayesian_suggestion=sug, include_ome_zarr=True)
    assert bundle_zip.is_file()

    extract_dir = tmp_path / "unzipped"
    with zipfile.ZipFile(bundle_zip, "r") as zf:
        names = set(zf.namelist())
        assert "manifest.json" in names
        assert "metrics.json" in names
        assert "bpx_parameters.json" in names
        assert "pybamm_parameters.json" in names
        assert "reproduce.py" in names
        assert "figures/slices_and_flux.png" in names
        assert "figures/transport_and_profile.png" in names
        assert "figures/bayesian_steering.png" in names
        zf.extractall(extract_dir)

    manifest = json.loads((extract_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "openimpala_hub.bundle.v1"
    assert manifest["contains_ome_zarr"] is True

    proc = subprocess.run(
        [sys.executable, str(extract_dir / "reproduce.py")],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "Porosity:" in proc.stdout
    assert "D_eff=" in proc.stdout
