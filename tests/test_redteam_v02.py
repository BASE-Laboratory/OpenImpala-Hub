"""Red-Team security, experimental upload, anisotropic spacing, BPX role, and REV regression suite (`v0.2`)."""

from __future__ import annotations

import base64
import io
from pathlib import Path
import zipfile

import numpy as np
import pytest
from basecoat import Volume, write_ome_zarr, write_volume

from openimpala_hub.app.callbacks import (
    compute_gp_for_campaign,
    plane_slider_bounds_for_run,
)
from openimpala_hub.app.figures import (
    build_directional_and_profile_figure,
    build_upload_segmentation_preview_figure,
)
from openimpala_hub.app.layout import history_entry_from_summary
from openimpala_hub.app.server import create_dash_app
from openimpala_hub.contracts import MicrostructureSpec, SolveSpec
from openimpala_hub.engine.solver import run_transport_pipeline
from openimpala_hub.engine.uploads import (
    inspect_staged_volume,
    is_staged_upload_path,
    prune_staged_uploads,
    stage_upload_bytes,
)
from openimpala_hub.presets import PRESETS


@pytest.fixture(scope="module")
def hub_app():
    return create_dash_app(warm_start=True, warm_shape=(28, 28, 28))


def test_api_solve_blocks_arbitrary_server_upload_path(hub_app, tmp_path: Path) -> None:
    """P0 security check: arbitrary host paths on `POST /api/v1/solve` must return HTTP 403."""
    client = hub_app.server.test_client()

    for forbidden in ("/etc/passwd", str(Path(__file__).resolve())):
        resp = client.post(
            "/api/v1/solve",
            json={
                "microstructure": {
                    "source_kind": "user_upload",
                    "upload_path": forbidden,
                    "voxel_size_um": [1.0, 1.0, 1.0],
                }
            },
        )
        assert resp.status_code == 403
        body = resp.get_json()
        assert body["error_type"] == "PermissionError"
        assert not is_staged_upload_path(forbidden)


def test_api_solve_clamps_max_solve_side_zero_and_huge_shape(
    hub_app, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P0 DoS check: `max_solve_side=0` and `shape=[256, 256, 256]` must obey `HUB_MAX_SOLVE_SIDE`."""
    monkeypatch.setenv("HUB_MAX_SOLVE_SIDE", "32")
    client = hub_app.server.test_client()

    resp = client.post(
        "/api/v1/solve",
        json={
            "microstructure": {
                "source_kind": "open_cube",
                "shape": [256, 256, 256],
                "voxel_size_um": [1.5, 1.5, 1.5],
            },
            "solve": {
                "directions": ["z"],
                "max_solve_side": 0,
            },
        },
    )
    assert resp.status_code == 200
    summary = resp.get_json()
    assert summary["shape_zyx"] == [32, 32, 32]
    assert summary["porosity"] == pytest.approx(1.0)


def test_ome_zarr_zip_roundtrip_anisotropic_calibration_and_zip_slip_guard(
    hub_app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`.ome.zarr.zip` uploads preserve calibrated `(dz, dy, dx)` spacing and reject zip-slip."""
    stage_root = tmp_path / "isolated_uploads"
    monkeypatch.setenv("OPENIMPALA_HUB_UPLOAD_DIR", str(stage_root))

    # 1. Reject zip-slip traversal entry
    evil_buf = io.BytesIO()
    with zipfile.ZipFile(evil_buf, "w") as zf:
        zf.writestr("../escape.tif", b"malicious")
    with pytest.raises(ValueError, match="Unsafe archive entry"):
        stage_upload_bytes(evil_buf.getvalue(), "evil.ome.zarr.zip")

    # 2. Build calibrated anisotropic OME-Zarr (dz=2.4, dy=1.2, dx=0.8 um) and zip it
    arr = np.zeros((20, 24, 28), dtype=np.int32)
    arr[6:14, 8:16, 10:18] = 1
    cal_vol = Volume.from_array(
        arr,
        voxel_size_um=(2.4, 1.2, 0.8),
        label_names={0: "pore", 1: "solid"},
    )
    zarr_src = tmp_path / "aniso_sample.ome.zarr"
    write_ome_zarr(cal_vol, zarr_src, multiscale_levels=1, overwrite=True)

    valid_buf = io.BytesIO()
    with zipfile.ZipFile(valid_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(zarr_src.rglob("*")):
            if fp.is_file():
                arc = f"aniso_sample.ome.zarr/{fp.relative_to(zarr_src).as_posix()}"
                zf.write(fp, arcname=arc)

    client = hub_app.server.test_client()
    b64 = base64.b64encode(valid_buf.getvalue()).decode("ascii")
    r_up = client.post(
        "/api/v1/upload",
        json={"filename": "aniso_sample.ome.zarr.zip", "content_base64": b64},
    )
    assert r_up.status_code == 200
    up_body = r_up.get_json()
    staged_path = up_body["upload_path"]
    assert is_staged_upload_path(staged_path)
    assert up_body["inspection"]["is_calibrated"] is True
    assert up_body["inspection"]["calibrated_voxel_um_zyx"] == pytest.approx([2.4, 1.2, 0.8])

    # Solve via REST without overriding native calibration -> must retain (2.4, 1.2, 0.8) um
    r_solve = client.post(
        "/api/v1/solve",
        json={
            "microstructure": {
                "source_kind": "user_upload",
                "upload_path": staged_path,
                "label": "Anisotropic FIB-SEM Upload",
            },
            "solve": {"directions": ["z"], "compute_solid_wiring": False},
        },
    )
    assert r_solve.status_code == 200
    s_json = r_solve.get_json()
    assert s_json["voxel_size_um_zyx"] == pytest.approx([2.4, 1.2, 0.8])
    assert s_json["shape_zyx"] == [20, 24, 28]

    # 3. Verify concurrent identical filenames get distinct UUID directories & prune cleanly
    npy_buf = io.BytesIO()
    np.save(npy_buf, arr)
    raw_npy = npy_buf.getvalue()
    p_a = stage_upload_bytes(raw_npy, "sample.npy")
    p_b = stage_upload_bytes(raw_npy, "sample.npy")
    assert p_a != p_b
    assert p_a.exists() and p_b.exists()
    removed = prune_staged_uploads(max_keep=1)
    assert removed >= 2


def test_inverted_contrast_greyscale_manual_threshold_and_preview_figure(
    tmp_path: Path,
) -> None:
    """Bright-is-pore (`dark_is_pore=False`) + manual threshold segments inverted XCT stacks accurately."""
    rng = np.random.default_rng(7)
    # 75% bright pores (~210), 25% dark solid (~40)
    mask_pore = rng.random((24, 24, 24)) < 0.75
    expected_pore_frac = float(np.mean(mask_pore))
    grey = np.where(mask_pore, 210.0, 40.0) + rng.normal(0.0, 4.0, size=(24, 24, 24))
    grey_u8 = np.clip(grey, 0, 255).astype(np.uint8)

    tif_path = tmp_path / "inverted_xct.tif"
    write_volume(Volume.from_array(grey_u8, voxel_size_um=None), tif_path)

    info = inspect_staged_volume(tif_path)
    fig_prev = build_upload_segmentation_preview_figure(
        info,
        seg_mode="manual",
        manual_threshold=125.0,
        dark_is_pore=False,
    )
    assert len(fig_prev.data) == 3

    mspec = MicrostructureSpec(
        source_kind="user_upload",
        upload_path=str(tif_path),
        voxel_size_um=(1.1, 1.1, 1.1),
        threshold=125.0,
        dark_is_pore=False,
    )
    executed = run_transport_pipeline(
        mspec,
        SolveSpec(directions=("z",), compute_solid_wiring=False),
    )
    assert executed.summary.porosity == pytest.approx(expected_pore_frac, abs=1e-5)


def test_positive_cathode_and_separator_bpx_pybamm_roles() -> None:
    """Preset #2 emits 'Positive electrode ...' keys and separator role omits active-material keys."""
    cathode = PRESETS["composite_cathode_3phase"]
    fast_cath = MicrostructureSpec(
        **{**cathode.microstructure.to_dict(), "shape": (24, 24, 24)}
    )
    res_pos = run_transport_pipeline(
        fast_cath,
        SolveSpec(
            directions=("z",),
            phase_diffusivities={0: 1.0, 1: 0.0, 2: 0.18},
            compute_homogenisation=False,
            compute_solid_wiring=False,
        ),
    )
    assert res_pos.summary.electrode_role == "positive"
    assert "Positive electrode porosity" in res_pos.summary.pybamm_parameters
    assert "Negative electrode porosity" not in res_pos.summary.pybamm_parameters
    bpx_p = res_pos.summary.bpx_parameters["bpx"]["Parameter"]
    assert "Positive electrode porosity" in bpx_p
    assert "Positive electrode active material volume fraction" in bpx_p

    sep_spec = MicrostructureSpec(
        source_kind="overlapping_spheres",
        shape=(24, 24, 24),
        porosity=0.55,
        voxel_size_um=(0.5, 0.5, 0.5),
        electrode_role="separator",
    )
    res_sep = run_transport_pipeline(
        sep_spec,
        SolveSpec(directions=("z",), compute_solid_wiring=False),
    )
    assert "Separator porosity" in res_sep.summary.pybamm_parameters
    assert "Separator active material volume fraction" not in res_sep.summary.pybamm_parameters


def test_rectangular_plane_bounds_rev_study_and_live_regime_gp() -> None:
    """Non-cubic `(24, 36, 48)` volume updates XY/XZ/YZ bounds; REV study & regime GP work cleanly."""
    rect_spec = MicrostructureSpec(
        source_kind="overlapping_spheres",
        shape=(24, 36, 48),
        porosity=0.72,
        sphere_radius_vox=3.5,
        voxel_size_um=(1.5, 1.0, 1.0),
        seed=11,
    )
    run = run_transport_pipeline(
        rect_spec,
        SolveSpec(
            directions=("z",),
            compute_solid_wiring=False,
            compute_rev_study=True,
        ),
    )
    assert plane_slider_bounds_for_run(run, "xy")[0] == 23
    assert plane_slider_bounds_for_run(run, "xz")[0] == 35
    assert plane_slider_bounds_for_run(run, "yz")[0] == 47

    assert run.summary.rev_sizes_vox is not None
    assert run.summary.rev_sizes_vox == sorted(run.summary.rev_sizes_vox)
    assert run.summary.rev_sizes_vox[-1] == 24
    fig_rev = build_directional_and_profile_figure(run)
    assert len(fig_rev.data) == 4

    # Verify GP campaign filtering ignores open_cube and shifts optimum under ionic vs electronic weights
    felt_runs = [
        run_transport_pipeline(
            MicrostructureSpec(
                source_kind="carbon_felt",
                shape=(28, 28, 28),
                clamping_kpa=kpa,
                compression_strain=-1.0,
                seed=42,
            ),
            SolveSpec(directions=("z",)),
        )
        for kpa in (35.0, 160.0, 420.0)
    ]
    cube_run = run_transport_pipeline(
        MicrostructureSpec(source_kind="open_cube", shape=(16, 16, 16)),
        SolveSpec(directions=("z",), compute_solid_wiring=False),
    )
    mixed_hist = [history_entry_from_summary(r.summary) for r in [*felt_runs, cube_run]]

    sug_ion_heavy = compute_gp_for_campaign(
        mixed_hist, active_source_kind="carbon_felt", w_ion=1.15, w_elec=0.05, w_kin=0.05
    )
    sug_elec_heavy = compute_gp_for_campaign(
        mixed_hist, active_source_kind="carbon_felt", w_ion=0.10, w_elec=0.95, w_kin=0.05
    )
    assert sug_ion_heavy is not None and sug_elec_heavy is not None
    # Cube run must NOT pollute observed_x (only the 3 felt loads)
    assert len(sug_ion_heavy.observed_x) == 3
    # Heavy electronic weight favours higher clamping pressure than heavy ionic mass-transport weight
    assert sug_elec_heavy.current_best_parameter >= sug_ion_heavy.current_best_parameter
