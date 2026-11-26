"""Analytical and physical invariance tests for the OpenImpala-Hub engine."""

from __future__ import annotations

import numpy as np
import pytest
from basecoat import Volume

from openimpala_hub.contracts import MicrostructureSpec, SolveSpec
from openimpala_hub.engine.bridge import HubUnitCalibrationError, enforce_solve_grid_cap
from openimpala_hub.engine.solver import run_transport_pipeline
from openimpala_hub.presets import PRESETS, make_carbon_felt_spec


def test_analytical_open_cube_exact_unity() -> None:
    """An unobstructed uniform domain must yield porosity=1.0, tau=1.0, and |J/J0|=1.0 everywhere."""
    preset = PRESETS["teaching_analytical_cube"]
    run = run_transport_pipeline(preset.microstructure, preset.solve)
    s = run.summary
    assert s.porosity == pytest.approx(1.0, abs=1e-9)
    for axis in ("x", "y", "z"):
        assert s.directional[axis].percolates is True
        assert s.directional[axis].tortuosity == pytest.approx(1.0, abs=1e-4)
        assert s.directional[axis].effective_diffusivity == pytest.approx(1.0, abs=1e-4)

    assert run.potential_field is not None
    assert run.flux_field is not None
    # Exterior-face Dirichlet BCs produce linear cell-centre potential along Z
    mean_z = np.mean(run.potential_field, axis=(1, 2))
    diffs = np.diff(mean_z)
    assert np.allclose(diffs, diffs[0], atol=1e-4)
    assert np.allclose(run.flux_field, 1.0, atol=1e-3)


def test_uncalibrated_volume_raises_unit_calibration_error() -> None:
    """Passing an uncalibrated `unit='voxel'` volume must fail loudly before solving."""
    uncal = Volume.uncalibrated(np.zeros((16, 16, 16), dtype=np.int32))
    mspec = MicrostructureSpec(source_kind="open_cube", shape=(16, 16, 16), voxel_size_um=None)
    with pytest.raises(HubUnitCalibrationError):
        run_transport_pipeline(mspec, SolveSpec(), existing_volume=uncal)


def test_carbon_felt_anisotropy_and_monotonic_compression() -> None:
    """XY-bedded carbon felt must exhibit tau_z > tau_xy and monotonically decreasing porosity under load."""
    low_load = make_carbon_felt_spec(30.0, shape=(48, 48, 48), seed=21)
    mid_load = make_carbon_felt_spec(180.0, shape=(48, 48, 48), seed=21)
    high_load = make_carbon_felt_spec(480.0, shape=(48, 48, 48), seed=21)

    sspec = SolveSpec(directions=("x", "y", "z"), solver="mlmg", return_fields=True)
    r_low = run_transport_pipeline(low_load, sspec)
    r_mid = run_transport_pipeline(mid_load, sspec)
    r_high = run_transport_pipeline(high_load, sspec)

    assert r_low.summary.porosity > r_mid.summary.porosity > r_high.summary.porosity

    tz = r_mid.summary.directional["z"].tortuosity
    tx = r_mid.summary.directional["x"].tortuosity
    ty = r_mid.summary.directional["y"].tortuosity
    assert tz is not None and tx is not None and ty is not None
    assert tz > 0.5 * (tx + ty)
    assert r_mid.summary.anisotropy_ratio_z_over_xy is not None
    assert r_mid.summary.anisotropy_ratio_z_over_xy > 1.05


def test_grid_cap_mode_downsample_preserves_outer_domain_extent() -> None:
    """Mode coarsening from 64^3 to 32^3 doubles voxel scale and shifts index-0 cell centre by +0.5*scale."""
    arr = np.zeros((64, 64, 64), dtype=np.int32)
    arr[16:48, 16:48, 16:48] = 1
    vol = Volume.from_array(arr, voxel_size_um=2.0, origin_um=(1.0, 1.0, 1.0))
    coarse, note = enforce_solve_grid_cap(vol, max_side=32, policy="mode_downsample")

    assert coarse.shape == (32, 32, 32)
    assert note is not None
    assert coarse.spacing_in("um") == pytest.approx((4.0, 4.0, 4.0))
    # Outer left face: origin_old - 0.5*2.0 = 0.0 -> origin_new = 1.0 + 0.5*(2-1)*2.0 = 2.0
    assert coarse.origin_in("um") == pytest.approx((2.0, 2.0, 2.0))


def test_cpp_and_scipy_solver_parity() -> None:
    """C++ AMReX MLMG and pure-Python SciPy CG backends must agree closely on identical porous media."""
    mspec = MicrostructureSpec(
        source_kind="gaussian_blobs",
        shape=(28, 28, 28),
        voxel_size_um=(1.5, 1.5, 1.5),
        porosity=0.58,
        seed=99,
    )
    r_cpp = run_transport_pipeline(
        mspec,
        SolveSpec(directions=("z",), solver="mlmg", compute_solid_wiring=False, return_fields=True),
    )
    r_py = run_transport_pipeline(
        mspec,
        SolveSpec(directions=("z",), solver="scipy", compute_solid_wiring=False, return_fields=True),
    )
    assert r_py.summary.solver_backend == "scipy-pure"
    assert r_cpp.summary.directional["z"].tortuosity == pytest.approx(
        r_py.summary.directional["z"].tortuosity, rel=1e-3
    )


def test_v52_active_ssa_and_native_flux_pocket_zeroing() -> None:
    """OpenImpala v5.2 native active_only SSA and return_flux must exclude isolated pockets."""
    arr = np.ones((16, 16, 16), dtype=np.int32)
    # Z-spanning pore channel (phase 0) + isolated 2^3 pore pocket
    arr[:, 2:6, 2:6] = 0
    arr[10:12, 10:12, 10:12] = 0
    vol = Volume.from_array(arr, voxel_size_um=1.0)

    run = run_transport_pipeline(
        MicrostructureSpec(source_kind="open_cube", shape=(16, 16, 16), voxel_size_um=(1.0, 1.0, 1.0)),
        SolveSpec(
            directions=("z",),
            conducting_phase=0,
            solid_phases=(1,),
            field_direction="z",
            solver="mlmg",
            return_fields=True,
            compute_solid_wiring=False,
        ),
        existing_volume=vol,
    )
    assert run.summary.interfacial_area_active_m_inv < run.summary.specific_surface_area_m_inv
    assert run.flux_field is not None
    assert np.all(run.flux_field[10:12, 10:12, 10:12] == 0.0)
    assert np.allclose(run.flux_field[:, 2:6, 2:6], 1.0, atol=1e-3)


