"""Flask REST API, Dash layout, and figure builder verification tests."""

from __future__ import annotations

import io
import json
import zipfile

import pytest

from openimpala_hub.app.figures import (
    build_3d_isosurface_figure,
    build_directional_and_profile_figure,
    build_gp_steering_figure,
    build_slice_figure,
)
from openimpala_hub.app.server import create_dash_app
from openimpala_hub.engine.steering import suggest_next_setpoint_gp


@pytest.fixture(scope="module")
def hub_app():
    return create_dash_app(warm_start=True, warm_shape=(32, 32, 32))


def test_healthz_and_presets_endpoints(hub_app) -> None:
    client = hub_app.server.test_client()

    r_health = client.get("/healthz")
    assert r_health.status_code == 200
    payload = r_health.get_json()
    assert payload["status"] == "ok"
    assert payload["cached_runs"] >= 3

    r_presets = client.get("/api/v1/presets")
    assert r_presets.status_code == 200
    presets = r_presets.get_json()
    keys = {p["key"] for p in presets}
    assert {
        "carbon_felt_compression",
        "composite_cathode_3phase",
        "teaching_analytical_cube",
        "teaching_sphere_pack",
    }.issubset(keys)


def test_api_solve_and_zip_bundle_download(hub_app) -> None:
    client = hub_app.server.test_client()

    r_solve = client.post(
        "/api/v1/solve",
        json={"preset": "teaching_analytical_cube", "shape": [24, 24, 24]},
    )
    assert r_solve.status_code == 200
    summary = r_solve.get_json()
    assert summary["porosity"] == pytest.approx(1.0)
    assert summary["directional"]["z"]["tortuosity"] == pytest.approx(1.0, abs=1e-4)
    run_id = summary["run_id"]

    r_zip = client.get(f"/api/v1/bundle/{run_id}.zip")
    assert r_zip.status_code == 200
    assert r_zip.mimetype == "application/zip"
    with zipfile.ZipFile(io.BytesIO(r_zip.data), "r") as zf:
        names = set(zf.namelist())
        assert {"manifest.json", "metrics.json", "bpx_parameters.json", "reproduce.py"}.issubset(names)
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        assert manifest["run_id"] == run_id


def test_api_solve_rejects_uncalibrated_volume(hub_app) -> None:
    client = hub_app.server.test_client()
    r_bad = client.post(
        "/api/v1/solve",
        json={
            "microstructure": {
                "source_kind": "open_cube",
                "shape": [16, 16, 16],
                "voxel_size_um": None,
            }
        },
    )
    assert r_bad.status_code == 400
    body = r_bad.get_json()
    assert "uncalibrated" in body["error"].lower() or "voxel" in body["error"].lower()


def test_dash_root_and_plotly_figure_builders(hub_app) -> None:
    client = hub_app.server.test_client()
    r_index = client.get("/")
    assert r_index.status_code == 200
    assert b"The OpenImpala Hub" in r_index.data

    registry = hub_app._hub_registry
    run = registry.get(registry.default_run_id)
    assert run is not None

    for dark in (False, True):
        f_slice = build_slice_figure(run, plane="xz", slice_index=16, layer="flux", dark_mode=dark)
        assert len(f_slice.data) == 1
        f_prof = build_directional_and_profile_figure(run, dark_mode=dark)
        assert len(f_prof.data) == 3
        f_3d = build_3d_isosurface_figure(run, dark_mode=dark)
        assert len(f_3d.data) >= 1

    sug = suggest_next_setpoint_gp([35.0, 150.0, 410.0], [3.1, 1.5, 2.7])
    f_gp = build_gp_steering_figure(sug, dark_mode=True)
    assert len(f_gp.data) == 4
