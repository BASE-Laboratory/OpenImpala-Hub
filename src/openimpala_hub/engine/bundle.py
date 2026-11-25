"""Deterministic `.zip` NIA Report Bundle generator (`M2.1`).

Produces a self-contained reproducible archive containing:
- `manifest.json`: Provenance, software versions, git commit (if available), timing, and data origin.
- `metrics.json`: Full `SolveRunSummary` + optional `BayesianSetpointSuggestion`.
- `bpx_parameters.json` & `pybamm_parameters.json`: Continuum battery model parameters.
- `reproduce.py`: Standalone Python script that reproduces the volume & `OpenImpala` solve.
- `figures/slices_and_flux.png`: High-DPI orthogonal phase, potential `phi`, and flux `|J/J0|` panels.
- `figures/transport_and_profile.png`: Directional tortuosity comparison + through-thickness porosity.
- `figures/bayesian_steering.png`: GP posterior `R_cell(sigma)` + Expected Improvement (when available).
- `microstructure.ome.zarr/`: Calibrated OME-Zarr v0.4 multiscale store written via `basecoat` (optional).
"""

from __future__ import annotations

from dataclasses import asdict
import io
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from basecoat import write_ome_zarr

from ..contracts import BayesianSetpointSuggestion
from .solver import ExecutedRun


def _detect_git_sha() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short=10", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return proc.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _render_slices_and_flux_png(run: ExecutedRun) -> bytes:
    vol_arr = run.phase_volume.data
    nz, ny, nx = vol_arr.shape
    mid_z = nz // 2
    mid_y = ny // 2
    if run.summary.zoom_rois:
        (z0, z1), (y0, y1), (x0, x1) = run.summary.zoom_rois[0].bounds_zyx
        mid_z = (z0 + z1) // 2
        mid_y = (y0 + y1) // 2
    else:
        z0 = z1 = y0 = y1 = x0 = x1 = 0

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), dpi=160)
    fig.suptitle(
        f"{run.summary.label}  |  origin={run.summary.data_origin}  |  shape={run.summary.shape_zyx}",
        fontsize=11,
        fontweight="bold",
    )

    im0 = axes[0].imshow(vol_arr[mid_z, :, :], cmap="viridis", origin="lower", interpolation="nearest")
    axes[0].set_title(f"XY Phase Slice (z = {mid_z})")
    axes[0].set_xlabel("x [vox]")
    axes[0].set_ylabel("y [vox]")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    if run.potential_field is not None:
        im1 = axes[1].imshow(
            run.potential_field[:, mid_y, :],
            cmap="coolwarm",
            origin="lower",
            vmin=0.0,
            vmax=1.0,
        )
        axes[1].set_title(f"XZ Potential phi (y = {mid_y})")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    else:
        axes[1].imshow(vol_arr[:, mid_y, :], cmap="viridis", origin="lower", interpolation="nearest")
        axes[1].set_title(f"XZ Phase Slice (y = {mid_y})")
    axes[1].set_xlabel("x [vox]")
    axes[1].set_ylabel("z [vox]")

    if run.flux_field is not None:
        vmax_j = max(float(np.percentile(run.flux_field, 99.2)), 1.2)
        im2 = axes[2].imshow(
            run.flux_field[:, mid_y, :],
            cmap="inferno",
            origin="lower",
            vmin=0.0,
            vmax=vmax_j,
        )
        axes[2].set_title(f"XZ Flux Crowding |J/J0| (y = {mid_y})")
        if run.summary.zoom_rois:
            rect = mpatches.Rectangle(
                (x0 - 0.5, z0 - 0.5),
                x1 - x0,
                z1 - z0,
                linewidth=1.8,
                edgecolor="#00f5d4",
                facecolor="none",
                linestyle="--",
                label="Zoom ROI #1",
            )
            axes[2].add_patch(rect)
            axes[2].legend(loc="upper right", fontsize=8)
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    else:
        axes[2].text(0.5, 0.5, "No 3D flux field", ha="center", va="center")
        axes[2].set_title("Flux Field")
    axes[2].set_xlabel("x [vox]")
    axes[2].set_ylabel("z [vox]")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _render_transport_and_profile_png(run: ExecutedRun) -> bytes:
    fig, (ax_bar, ax_prof) = plt.subplots(1, 2, figsize=(10.5, 4.0), dpi=160)

    dirs = list(run.summary.directional.keys())
    taus = [
        run.summary.directional[d].tortuosity
        if run.summary.directional[d].tortuosity is not None
        else 0.0
        for d in dirs
    ]
    deffs = [run.summary.directional[d].effective_diffusivity for d in dirs]

    x_pos = np.arange(len(dirs))
    width = 0.36
    ax_bar.bar(x_pos - width / 2, taus, width, label="Tortuosity tau", color="#2563eb")
    ax_bar.bar(x_pos + width / 2, deffs, width, label="D_eff / D_0", color="#0d9488")
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels([d.upper() for d in dirs])
    ax_bar.set_title(f"Directional Transport (porosity = {run.summary.porosity:.3f})")
    ax_bar.grid(axis="y", alpha=0.25)
    ax_bar.legend()

    ax_prof.plot(
        run.summary.through_thickness_z_um,
        run.summary.through_thickness_porosity,
        color="#1e293b",
        linewidth=1.8,
    )
    ax_prof.axhline(
        run.summary.porosity,
        color="#dc2626",
        linestyle="--",
        linewidth=1.2,
        label=f"Mean = {run.summary.porosity:.3f}",
    )
    ax_prof.set_xlabel("Position along flow axis [um]")
    ax_prof.set_ylabel("Slice porosity")
    ax_prof.set_title("Through-Thickness Porosity Profile")
    ax_prof.grid(alpha=0.25)
    ax_prof.legend()

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _render_bayesian_steering_png(suggestion: BayesianSetpointSuggestion) -> bytes:
    fig, (ax_gp, ax_ei) = plt.subplots(
        2, 1, figsize=(8.5, 5.6), dpi=160, sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]}
    )

    gx = np.asarray(suggestion.grid_x)
    mu = np.asarray(suggestion.gp_mean)
    lo = np.asarray(suggestion.gp_lower_95)
    hi = np.asarray(suggestion.gp_upper_95)
    ei = np.asarray(suggestion.ei_curve)

    ax_gp.fill_between(gx, lo, hi, color="#93c5fd", alpha=0.35, label="GP 95% credible band")
    ax_gp.plot(gx, mu, color="#1d4ed8", linewidth=2.0, label="GP posterior mean R_cell")
    ax_gp.scatter(
        suggestion.observed_x,
        suggestion.observed_r_cell,
        color="#0f172a",
        s=48,
        zorder=5,
        label="Evaluated OpenImpala runs",
    )
    ax_gp.axvline(
        suggestion.suggested_value,
        color="#059669",
        linestyle="--",
        linewidth=1.6,
        label=f"Suggested next: {suggestion.suggested_value:.1f} {suggestion.parameter_unit}",
    )
    ax_gp.set_ylabel("Series cell resistance R_cell [-]")
    ax_gp.set_title("Autonomous Experiment Steering (Gaussian Process Digital Twin)")
    ax_gp.grid(alpha=0.25)
    ax_gp.legend(fontsize=8)

    ax_ei.plot(gx, ei, color="#059669", linewidth=1.8)
    ax_ei.fill_between(gx, 0.0, ei, color="#6ee7b7", alpha=0.35)
    ax_ei.axvline(suggestion.suggested_value, color="#059669", linestyle="--", linewidth=1.4)
    ax_ei.set_xlabel(f"{suggestion.parameter_name} [{suggestion.parameter_unit}]")
    ax_ei.set_ylabel("Expected Improvement")
    ax_ei.grid(alpha=0.25)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _build_reproduce_script(run: ExecutedRun, *, has_zarr: bool) -> str:
    s = run.summary
    vz_m = tuple(round(v * 1e-6, 10) for v in s.voxel_size_um_zyx)
    dirs_repr = repr(tuple(s.directional.keys()))
    phase_diff = s.solve_spec.get("phase_diffusivities")
    if has_zarr:
        load_snippet = (
            'from pathlib import Path\n'
            'from basecoat import read_ome_zarr\n\n'
            'HERE = Path(__file__).resolve().parent\n'
            'vol = read_ome_zarr(HERE / "microstructure.ome.zarr", level=0)\n'
            'arr = vol.data\n'
        )
    else:
        load_snippet = (
            'from openimpala_hub.contracts import MicrostructureSpec\n'
            'from openimpala_hub.engine.generators import load_or_generate_microstructure\n\n'
            f'spec = MicrostructureSpec.from_dict({repr(s.microstructure_spec)})\n'
            'vol, _, _, _ = load_or_generate_microstructure(spec)\n'
            'arr = vol.data\n'
        )

    if phase_diff:
        pdiff_int = {int(k): float(v) for k, v in phase_diff.items()}
        solve_call = (
            f"    for d in {dirs_repr}:\n"
            f"        res = oi.tortuosity(arr, phase_diffusivities={repr(pdiff_int)}, "
            f"direction=d, voxel_size={repr(vz_m)})\n"
            "        print(d, 'D_eff=', round(res.effective_diffusivity or 0.0, 6), "
            "'tau=', round(res.tortuosity, 6))\n"
        )
    else:
        solve_call = (
            f"    res = oi.tortuosity_all(arr, phase={s.conducting_phase}, "
            f"directions={dirs_repr}, voxel_size={repr(vz_m)})\n"
            "    for d, r in res.items():\n"
            "        print(d, 'tau=', round(r.tortuosity, 6), "
            "'D_eff=', round(r.active_volume_fraction / r.tortuosity, 6))\n"
        )

    return (
        '#!/usr/bin/env python3\n'
        '"""Standalone reproduction script exported by The OpenImpala Hub."""\n\n'
        'import openimpala as oi\n'
        f'{load_snippet}\n'
        'with oi.Session():\n'
        f'    vf = oi.volume_fraction(arr, phase={s.conducting_phase})\n'
        '    print("Porosity:", round(vf.fraction, 6))\n'
        f'{solve_call}'
    )


def build_report_bundle_bytes(
    run: ExecutedRun,
    *,
    bayesian_suggestion: BayesianSetpointSuggestion | None = None,
    include_ome_zarr: bool = True,
) -> bytes:
    """Assemble a complete reproducible `.zip` NIA report bundle in memory."""
    manifest = {
        "schema": "openimpala_hub.bundle.v1",
        "run_id": run.summary.run_id,
        "label": run.summary.label,
        "timestamp_utc": run.summary.timestamp_utc,
        "data_origin": run.summary.data_origin,
        "precision_tag": run.summary.precision_tag,
        "hub_version": run.summary.hub_version,
        "openimpala_version": run.summary.openimpala_version,
        "basecoat_version": run.summary.basecoat_version,
        "solver_backend": run.summary.solver_backend,
        "hub_git_sha": _detect_git_sha(),
        "generation_time_s": run.summary.generation_time_s,
        "solve_time_s": run.summary.solve_time_s,
        "contains_ome_zarr": bool(include_ome_zarr),
    }

    metrics_payload = run.summary.to_dict()
    if bayesian_suggestion is not None:
        metrics_payload["bayesian_steering"] = asdict(bayesian_suggestion)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        zf.writestr("metrics.json", json.dumps(metrics_payload, indent=2))
        zf.writestr("bpx_parameters.json", json.dumps(run.summary.bpx_parameters, indent=2))
        zf.writestr("pybamm_parameters.json", json.dumps(run.summary.pybamm_parameters, indent=2))
        zf.writestr(
            "reproduce.py",
            _build_reproduce_script(run, has_zarr=include_ome_zarr),
        )
        zf.writestr("figures/slices_and_flux.png", _render_slices_and_flux_png(run))
        zf.writestr("figures/transport_and_profile.png", _render_transport_and_profile_png(run))
        morph_echem_png = _render_morphology_and_electrochem_png(run)
        if morph_echem_png is not None:
            zf.writestr("figures/morphology_and_electrochem.png", morph_echem_png)
        if bayesian_suggestion is not None:
            zf.writestr(
                "figures/bayesian_steering.png",
                _render_bayesian_steering_png(bayesian_suggestion),
            )

        if include_ome_zarr:
            with tempfile.TemporaryDirectory() as tmpdir:
                zarr_dir = Path(tmpdir) / "microstructure.ome.zarr"
                write_ome_zarr(run.phase_volume, zarr_dir, multiscale_levels=2, overwrite=True)
                for fpath in sorted(zarr_dir.rglob("*")):
                    if fpath.is_file():
                        arcname = f"microstructure.ome.zarr/{fpath.relative_to(zarr_dir).as_posix()}"
                        zf.write(fpath, arcname=arcname)

    return buf.getvalue()


def _render_morphology_and_electrochem_png(run: ExecutedRun) -> bytes | None:
    morph = run.summary.morphology
    echem = run.summary.electrochem
    if not morph and not echem:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), dpi=150)
    fig.suptitle(
        f"{run.summary.label} — 3D Morphology & Continuum Cell Electrochemistry",
        fontsize=11,
        fontweight="bold",
    )

    # (0, 0) Two-point correlation S2(r)
    ax00 = axes[0, 0]
    if morph and morph.get("s2_r_um"):
        ax00.plot(morph["s2_r_um"], morph["s2_radial"], label="Radial S2(r)", color="#2563eb", lw=2)
        ax00.plot(morph["s2_z_um"], morph["s2_z"], "--", label="Through-plane Z", color="#ef4444")
        ax00.plot(morph["s2_xy_um"], morph["s2_xy"], ":", label="In-plane XY", color="#0d9488")
        ax00.legend(fontsize=8)
    ax00.set_title("Two-Point Correlation S2(r)")
    ax00.set_xlabel("Lag r [um]")
    ax00.set_ylabel("S2(r)")
    ax00.grid(alpha=0.25)

    # (0, 1) Pore network connectivity partition
    ax01 = axes[0, 1]
    if morph:
        cats = ["Percolating", "Dead-End", "Trapped"]
        vals = [
            float(morph.get("percolating_fraction", 0.0)),
            float(morph.get("dead_end_fraction", 0.0)),
            float(morph.get("trapped_fraction", 0.0)),
        ]
        ax01.bar(cats, vals, color=["#10b981", "#f59e0b", "#ef4444"])
        ax01.set_title(f"Pore Network Partition (D50={morph.get('d50_um', 0):.1f} um)")
        ax01.set_ylabel("Volume Fraction [-]")
        ax01.grid(axis="y", alpha=0.25)

    # (1, 0) Galvanostatic discharge curves
    ax10 = axes[1, 0]
    if echem and echem.get("curves"):
        palette = ["#0d9488", "#2563eb", "#f59e0b", "#ef4444"]
        for idx, c in enumerate(echem["curves"]):
            ax10.plot(
                c["capacity_pct_grid"],
                c["voltage_v_grid"],
                lw=1.9,
                color=palette[idx % len(palette)],
                label=f"{c['c_rate']}C ({c['accessible_capacity_pct']:.0f}%)",
            )
        ax10.legend(fontsize=8)
        ax10.set_title(f"Discharge Curves (C_lim = {echem.get('limiting_c_rate', 0):.2f}C)")
    ax10.set_xlabel("Utilised Capacity [%]")
    ax10.set_ylabel("Cell Voltage [V]")
    ax10.grid(alpha=0.25)

    # (1, 1) Overpotential breakdown vs C-rate
    ax11 = axes[1, 1]
    if echem and echem.get("curves"):
        crates = [f"{c['c_rate']}C" for c in echem["curves"]]
        ion = np.array([c["eta_ionic_mv"] for c in echem["curves"]])
        elec = np.array([c["eta_electronic_mv"] for c in echem["curves"]])
        kin = np.array([c["eta_kinetic_mv"] for c in echem["curves"]])
        conc = np.array([c["eta_concentration_mv"] for c in echem["curves"]])
        ax11.bar(crates, ion, label="Ionic", color="#3b82f6")
        ax11.bar(crates, elec, bottom=ion, label="Electronic", color="#64748b")
        ax11.bar(crates, kin, bottom=ion + elec, label="Kinetic", color="#10b981")
        ax11.bar(crates, conc, bottom=ion + elec + kin, label="Concentration", color="#ef4444")
        ax11.legend(fontsize=8)
    ax11.set_title("Mid-SOC Overpotential Breakdown")
    ax11.set_ylabel("Overpotential [mV]")
    ax11.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def write_report_bundle(
    run: ExecutedRun,
    output_path: str | Path,
    *,
    bayesian_suggestion: BayesianSetpointSuggestion | None = None,
    include_ome_zarr: bool = True,
) -> Path:
    """Write a `.zip` NIA report bundle to disk and return its resolved `Path`."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(
        build_report_bundle_bytes(
            run,
            bayesian_suggestion=bayesian_suggestion,
            include_ome_zarr=include_ome_zarr,
        )
    )
    return out
