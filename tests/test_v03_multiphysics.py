"""Analytical, stereological, electrochemical, and 3D SMS mechanics tests for `v0.3.0`."""

from __future__ import annotations

import io
import json
import zipfile

from basecoat import AxisSpec, Volume
import numpy as np
import pytest

from openimpala_hub.app.callbacks import (
    evaluate_teaching_lab_progress,
    parse_campaign_json,
    serialize_campaign_json,
)
from openimpala_hub.app.layout import history_entry_from_summary
from openimpala_hub.contracts import MicrostructureSpec, SolveSpec
from openimpala_hub.engine.bundle import build_report_bundle_bytes
from openimpala_hub.engine.electrochem import CHEMISTRIES, evaluate_cell_electrochemistry
from openimpala_hub.engine.generators import load_or_generate_microstructure
from openimpala_hub.engine.mechanics import has_sms_backend, run_optional_mechanics
from openimpala_hub.engine.morphology import compute_morphology_report
from openimpala_hub.engine.porousdiff_bridge import kinematic_compress_volume
from openimpala_hub.engine.solver import run_transport_pipeline


def _isotropic_vol(arr: np.ndarray, vox_um: float = 2.0) -> Volume:
    axes = tuple(
        AxisSpec(name=ax, kind="space", unit="um", scale=float(vox_um))
        for ax in ("z", "y", "x")
    )
    return Volume(np.ascontiguousarray(arr, dtype=np.int32), axes)


def test_morphology_open_cube_and_exact_pore_partition() -> None:
    """Verify exact 6-connected Percolating / Dead-End / Trapped partition and S2(0) == porosity."""
    open_arr = np.zeros((16, 16, 16), dtype=np.int32)
    rep_open = compute_morphology_report(_isotropic_vol(open_arr, 1.5))
    assert rep_open.percolating_fraction == pytest.approx(1.0, abs=1e-6)
    assert rep_open.dead_end_fraction == pytest.approx(0.0, abs=1e-6)
    assert rep_open.trapped_fraction == pytest.approx(0.0, abs=1e-6)
    assert rep_open.num_trapped_clusters == 0
    assert rep_open.s2_radial[0] == pytest.approx(1.0, abs=1e-5)

    # Construct block (all solid = 1) with three disjoint 6-connected pore regions:
    # 1) Z-percolating channel: z=0..19, y=2..4, x=2..4 -> 20 * 3 * 3 = 180 voxels
    # 2) Dead-end pocket on z=0 only: z=0..5, y=10..12, x=10..12 -> 6 * 3 * 3 = 54 voxels
    # 3) Sealed interior cavity: z=10..13, y=10..12, x=2..4 -> 4 * 3 * 3 = 36 voxels
    n = 20
    blk = np.ones((n, n, n), dtype=np.int32)
    blk[:, 2:5, 2:5] = 0
    blk[0:6, 10:13, 10:13] = 0
    blk[10:14, 10:13, 2:5] = 0

    total_vox = float(n**3)
    exp_perc = 180.0 / total_vox
    exp_dead = 54.0 / total_vox
    exp_trap = 36.0 / total_vox
    exp_eps = exp_perc + exp_dead + exp_trap

    rep = compute_morphology_report(_isotropic_vol(blk, 2.0), conducting_phase=0, flow_direction="z")
    assert rep.percolating_fraction == pytest.approx(exp_perc, abs=1e-5)
    assert rep.dead_end_fraction == pytest.approx(exp_dead, abs=1e-5)
    assert rep.trapped_fraction == pytest.approx(exp_trap, abs=1e-5)
    assert (
        rep.percolating_fraction + rep.dead_end_fraction + rep.trapped_fraction
        == pytest.approx(exp_eps, abs=1e-5)
    )
    assert rep.num_percolating_clusters == 1
    assert rep.num_dead_end_clusters == 1
    assert rep.num_trapped_clusters == 1
    assert rep.s2_radial[0] == pytest.approx(exp_eps, abs=1e-5)
    assert rep.s2_z[0] == pytest.approx(exp_eps, abs=1e-5)
    assert rep.d50_um > 0.0


def test_kinematic_virtual_compression_conserves_solid_mass() -> None:
    """Verify level-set Z-compression reduces slab thickness while conserving solid voxel mass."""
    spec = MicrostructureSpec(
        source_kind="carbon_felt",
        shape=(32, 32, 32),
        voxel_size_um=2.0,
        porosity=0.80,
        clamping_kpa=60.0,
        coating_fraction=0.08,
        seed=17,
    )
    vol, _, _, _ = load_or_generate_microstructure(spec)
    orig_solid = int(np.count_nonzero(np.asarray(vol.data) != 0))
    orig_coat = int(np.count_nonzero(np.asarray(vol.data) == 2))
    assert orig_solid > 0 and orig_coat > 0

    comp = kinematic_compress_volume(vol, strain_z=0.25, conducting_phase=0)
    assert comp.shape == (24, 32, 32)
    comp_solid = int(np.count_nonzero(np.asarray(comp.data) != 0))
    comp_coat = int(np.count_nonzero(np.asarray(comp.data) == 2))
    assert comp_solid == orig_solid
    assert comp_coat > 0


def test_electrochemistry_monotonic_tortuosity_and_chemistries() -> None:
    """Verify doubling tortuosity halves J_lim & C_lim and degrades 4C accessible capacity."""
    low_tau = evaluate_cell_electrochemistry(
        porosity=0.28,
        d_eff_z=0.28 / 2.0,  # tau_Z = 2.0
        specific_surface_area_m_inv=2.5e5,
        solid_sigma_eff_z=0.25,
        thickness_m=140.0e-6,
        chemistry_key="nmc811_li_ion",
        preserve_exact_thickness=True,
    )
    high_tau = evaluate_cell_electrochemistry(
        porosity=0.28,
        d_eff_z=0.28 / 4.0,  # tau_Z = 4.0 (doubled)
        specific_surface_area_m_inv=2.5e5,
        solid_sigma_eff_z=0.25,
        thickness_m=140.0e-6,
        chemistry_key="nmc811_li_ion",
        preserve_exact_thickness=True,
    )

    assert low_tau.limiting_current_ma_cm2 == pytest.approx(
        2.0 * high_tau.limiting_current_ma_cm2, rel=1e-4
    )
    assert low_tau.limiting_c_rate == pytest.approx(2.0 * high_tau.limiting_c_rate, rel=1e-4)

    c4_low = next(c for c in low_tau.curves if c.c_rate == 4.0)
    c4_high = next(c for c in high_tau.curves if c.c_rate == 4.0)
    assert c4_high.eta_ionic_mv == pytest.approx(2.0 * c4_low.eta_ionic_mv, rel=1e-4)
    assert c4_high.accessible_capacity_pct < c4_low.accessible_capacity_pct
    assert c4_high.specific_energy_pct < c4_low.specific_energy_pct

    for key in CHEMISTRIES:
        rep = evaluate_cell_electrochemistry(
            porosity=0.55,
            d_eff_z=0.30,
            specific_surface_area_m_inv=1.8e5,
            solid_sigma_eff_z=0.20,
            thickness_m=90.0e-6,
            chemistry_key=key,
        )
        assert len(rep.curves) == 4
        assert rep.limiting_c_rate > 0.0


@pytest.mark.skipif(not has_sms_backend(), reason="SMS Fortran extension not installed")
def test_sms_mechanics_solid_cube_porous_felt_and_anisotropy_guard() -> None:
    """Verify SMS recovers exact bulk Young's modulus on a solid cube and stress concentration on felt."""
    solid = _isotropic_vol(np.ones((16, 16, 16), dtype=np.int32), vox_um=2.0)
    m_solid, vm_solid = run_optional_mechanics(
        solid,
        solid_phases=(1,),
        direction="z",
        bulk_youngs_modulus_gpa=10.0,
        poisson_ratio=0.25,
        rtol=1e-5,
    )
    assert m_solid is not None and vm_solid is not None
    assert m_solid.effective_modulus_gpa == pytest.approx(10.0, rel=2e-3)
    assert m_solid.relative_stiffness == pytest.approx(1.0, rel=2e-3)
    assert vm_solid.shape == solid.shape

    felt_spec = MicrostructureSpec(
        source_kind="carbon_felt",
        shape=(24, 24, 24),
        voxel_size_um=2.0,
        porosity=0.72,
        clamping_kpa=180.0,
        seed=42,
    )
    felt_vol, _, _, _ = load_or_generate_microstructure(felt_spec)
    m_felt, vm_felt = run_optional_mechanics(
        felt_vol,
        solid_phases=(1, 2),
        direction="z",
        bulk_youngs_modulus_gpa=12.0,
        max_mechanics_side=24,
    )
    assert m_felt is not None and vm_felt is not None
    assert 0.0 < m_felt.effective_modulus_gpa < 12.0
    assert m_felt.peak_stress_concentration > 1.2
    assert vm_felt.shape == felt_vol.shape

    # Anisotropic spacing must degrade cleanly without raising ValueError
    aniso_axes = (
        AxisSpec(name="z", kind="space", unit="um", scale=3.5),
        AxisSpec(name="y", kind="space", unit="um", scale=1.5),
        AxisSpec(name="x", kind="space", unit="um", scale=1.5),
    )
    aniso_vol = Volume(np.ones((12, 12, 12), dtype=np.int32), aniso_axes)
    m_aniso, vm_aniso = run_optional_mechanics(aniso_vol)
    assert m_aniso is not None
    assert m_aniso.backend == "sms-skipped"
    assert vm_aniso is None


def test_full_v03_pipeline_bundle_campaign_json_and_teaching_lab() -> None:
    """Verify end-to-end v0.3 multiphysics run, report zip bundle, campaign JSON, and Teaching Lab."""
    ms = MicrostructureSpec(
        source_kind="multiphase_electrode",
        shape=(24, 24, 24),
        voxel_size_um=2.0,
        active_fraction=0.48,
        binder_fraction=0.12,
        virtual_compression_strain=0.20,
        electrode_role="positive",
        seed=42,
    )
    ss = SolveSpec(
        compute_mechanics=has_sms_backend(),
        chemistry_preset="lfp_high_rate",
    )
    run = run_transport_pipeline(ms, ss)
    assert run.summary.morphology is not None
    assert run.summary.electrochem is not None
    assert run.summary.electrochem["chemistry_key"] == "lfp_high_rate"
    assert run.phase_volume.shape == (19, 24, 24)  # virtually calendered by 20% along Z
    if has_sms_backend():
        assert run.summary.mechanics is not None
        assert run.vm_stress_field is not None

    bundle_bytes = build_report_bundle_bytes(run)
    with zipfile.ZipFile(io.BytesIO(bundle_bytes), "r") as zf:
        names = set(zf.namelist())
        assert "figures/morphology_and_electrochem.png" in names
        metrics_doc = json.loads(zf.read("metrics.json").decode("utf-8"))
        assert "morphology" in metrics_doc and metrics_doc["morphology"] is not None
        assert "electrochem" in metrics_doc and metrics_doc["electrochem"] is not None

    # Campaign JSON round-trip & Teaching Lab progress evaluation
    entry = history_entry_from_summary(run.summary)
    cube_entry = dict(entry, run_id="cube1", source_kind="open_cube", porosity=1.0, tau_z=1.0)
    sph_entry = dict(
        entry,
        run_id="sph1",
        source_kind="benchmark_dataset",
        porosity=0.64,
        tau_z=1.48,  # > 0.64**(-0.5) == 1.25
        limiting_c_rate=2.4,
    )
    felt_entries = [
        dict(entry, run_id=f"f{k}", source_kind="carbon_felt", clamping_kpa=float(k))
        for k in (40, 120, 220, 360)
    ]
    full_hist = [cube_entry, sph_entry, *felt_entries]
    raw_json = serialize_campaign_json(full_hist)
    restored = parse_campaign_json(raw_json)
    assert len(restored) == len(full_hist)

    steps = evaluate_teaching_lab_progress(restored)
    assert len(steps) == 4
    assert all(s["passed"] for s in steps)
