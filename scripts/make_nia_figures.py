#!/usr/bin/env python3
"""Generate the 3-figure EPSRC NIA feasibility & CSEG Seed benchmark suite from real OpenImpala runs.

Honesty contract (`ECOSYSTEM.md`):
- Every number plotted is produced live by executing `openimpala_hub`'s pipeline on this host.
- Hardware specs, software versions, solver backends, data origins (`'synthetic'`), and exact
  numerical arrays are saved alongside the figures in `docs/figures/benchmark_provenance.json`.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import resource
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import basecoat
import openimpala as oi

from openimpala_hub import __version__ as hub_version
from openimpala_hub.contracts import MicrostructureSpec, SolveSpec
from openimpala_hub.engine.solver import run_transport_pipeline
from openimpala_hub.engine.steering import suggest_next_setpoint_gp
from openimpala_hub.presets import make_carbon_felt_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "docs" / "figures"


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def run_fig1_scaling() -> dict:
    """Measure real 3-axis solve + 3D flux-field wall-clock times across C++ MLMG vs Pure-Python SciPy."""
    cpp_sides = [32, 48, 64, 80, 96, 128]
    scipy_sides = [32, 48, 64, 80]

    cpp_times: list[float] = []
    cpp_gen_times: list[float] = []
    scipy_times: list[float] = []

    for n in cpp_sides:
        mspec = MicrostructureSpec(
            source_kind="gaussian_blobs",
            label=f"Scaling {n}^3",
            shape=(n, n, n),
            voxel_size_um=(1.6, 1.6, 1.6),
            porosity=0.55,
            anisotropy_zyx=(1.0, 1.8, 1.8),
            seed=42,
        )
        run_cpp = run_transport_pipeline(
            mspec,
            SolveSpec(directions=("x", "y", "z"), solver="mlmg", compute_solid_wiring=False, return_fields=True),
        )
        cpp_times.append(run_cpp.summary.solve_time_s)
        cpp_gen_times.append(run_cpp.summary.generation_time_s)

        if n in scipy_sides:
            run_py = run_transport_pipeline(
                mspec,
                SolveSpec(directions=("x", "y", "z"), solver="scipy", compute_solid_wiring=False, return_fields=True),
            )
            scipy_times.append(run_py.summary.solve_time_s)

    peak_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.2, 4.3), dpi=200)
    cpp_vox = [n**3 / 1e6 for n in cpp_sides]
    scipy_vox = [n**3 / 1e6 for n in scipy_sides]

    ax_a.plot(cpp_vox, cpp_times, "o-", color="#1d4ed8", linewidth=2.2, markersize=6, label="C++ AMReX MLMG (pybind11)")
    ax_a.plot(scipy_vox, scipy_times, "s--", color="#dc2626", linewidth=2.0, markersize=6, label="Pure-Python SciPy CG")
    for n, vx, tc in zip(cpp_sides, cpp_vox, cpp_times):
        if n >= 64:
            offset = (0, 8) if n < 128 else (-14, 8)
            ax_a.annotate(f"{n}³ ({tc:.2f}s)", (vx, tc), textcoords="offset points", xytext=offset, ha="center", fontsize=8)
    ax_a.set_xlabel("Grid Size [Million Voxels]")
    ax_a.set_ylabel("3-Axis Solve + 3D Flux Field Time [s]")
    ax_a.set_title("(a) Solver Latency Scaling (2-vCPU Xeon Container)")
    ax_a.grid(alpha=0.28)
    ax_a.legend(fontsize=9)

    speedups = [st / max(ct, 1e-6) for st, ct in zip(scipy_times, cpp_times[: len(scipy_sides)])]
    x_idx = np.arange(len(scipy_sides))
    bars = ax_b.bar(x_idx, speedups, width=0.52, color="#0d9488", edgecolor="#0f172a", linewidth=0.8)
    for b, sp in zip(bars, speedups):
        ax_b.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15, f"{sp:.1f}×", ha="center", fontsize=9, fontweight="bold")
    ax_b.set_xticks(x_idx)
    ax_b.set_xticklabels([f"{n}³" for n in scipy_sides])
    ax_b.set_xlabel("Cubic Domain Size")
    ax_b.set_ylabel("C++ MLMG Speedup over SciPy CG [×]")
    ax_b.set_ylim(0, max(speedups) * 1.25)
    ax_b.set_title("(b) Interactive Web Speedup via WP1 pybind11 Engine")
    ax_b.grid(axis="y", alpha=0.28)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_solver_scaling.png", bbox_inches="tight")
    plt.close(fig)

    return {
        "cpp_sides": cpp_sides,
        "cpp_3axis_solve_time_s": cpp_times,
        "cpp_generation_time_s": cpp_gen_times,
        "scipy_sides": scipy_sides,
        "scipy_3axis_solve_time_s": scipy_times,
        "speedup_cpp_vs_scipy": [round(s, 3) for s in speedups],
        "process_peak_rss_mib": round(peak_rss_mib, 1),
    }


def run_fig2_compression_tradeoff() -> dict:
    """Evaluate anisotropic carbon-felt compression across 7 loads x 3 stochastic seeds (`56^3`)."""
    loads_kpa = [25.0, 75.0, 130.0, 190.0, 270.0, 370.0, 480.0]
    seeds = [42, 43, 44]
    sspec = SolveSpec(
        conducting_phase=0,
        solid_phases=(1, 2),
        directions=("x", "y", "z"),
        solver="mlmg",
        compute_solid_wiring=True,
        return_fields=False,
    )

    tau_z_mat = np.zeros((len(seeds), len(loads_kpa)))
    tau_xy_mat = np.zeros((len(seeds), len(loads_kpa)))
    deff_z_mat = np.zeros((len(seeds), len(loads_kpa)))
    sigma_z_mat = np.zeros((len(seeds), len(loads_kpa)))
    rcell_mat = np.zeros((len(seeds), len(loads_kpa)))
    eps_mat = np.zeros((len(seeds), len(loads_kpa)))

    for si, sd in enumerate(seeds):
        for li, kpa in enumerate(loads_kpa):
            mspec = make_carbon_felt_spec(kpa, shape=(56, 56, 56), seed=sd)
            res = run_transport_pipeline(mspec, sspec).summary
            tz = res.directional["z"].tortuosity or np.nan
            tx = res.directional["x"].tortuosity or np.nan
            ty = res.directional["y"].tortuosity or np.nan
            tau_z_mat[si, li] = tz
            tau_xy_mat[si, li] = 0.5 * (tx + ty)
            deff_z_mat[si, li] = res.directional["z"].effective_diffusivity
            sigma_z_mat[si, li] = res.solid_effective_conductivity_z
            rcell_mat[si, li] = res.cell_resistance_fom
            eps_mat[si, li] = res.porosity

    tz_mean, tz_std = np.nanmean(tau_z_mat, axis=0), np.nanstd(tau_z_mat, axis=0)
    txy_mean, txy_std = np.nanmean(tau_xy_mat, axis=0), np.nanstd(tau_xy_mat, axis=0)
    deff_mean = np.mean(deff_z_mat, axis=0)
    sig_mean = np.mean(sigma_z_mat, axis=0)
    rc_mean, rc_std = np.mean(rcell_mat, axis=0), np.std(rcell_mat, axis=0)

    opt_idx = int(np.argmin(rc_mean))
    opt_kpa = loads_kpa[opt_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.4), dpi=200)
    x = np.asarray(loads_kpa)

    ax1.plot(x, tz_mean, "o-", color="#dc2626", linewidth=2.2, label="Through-plane τ_Z")
    ax1.fill_between(x, tz_mean - tz_std, tz_mean + tz_std, color="#fca5a5", alpha=0.35)
    ax1.plot(x, txy_mean, "s--", color="#2563eb", linewidth=2.0, label="In-plane τ_XY = (τ_X + τ_Y)/2")
    ax1.fill_between(x, txy_mean - txy_std, txy_mean + txy_std, color="#93c5fd", alpha=0.35)
    ax1.set_xlabel("Clamping Pressure σ [kPa]")
    ax1.set_ylabel("Tortuosity Factor τ [-]")
    ax1.set_title("(a) Directional Tortuosity Anisotropy (56³, n=3 seeds)")
    ax1.grid(alpha=0.28)
    ax1.legend(fontsize=9)

    ax2_twin = ax2.twinx()
    l1 = ax2.plot(x, rc_mean, "o-", color="#0f172a", linewidth=2.4, label="Cell Resistance R_cell (left)")
    ax2.fill_between(x, rc_mean - rc_std, rc_mean + rc_std, color="#cbd5e1", alpha=0.45)
    ax2.axvline(opt_kpa, color="#059669", linestyle=":", linewidth=1.8)
    ax2.annotate(
        f"Sweet spot σ* ≈ {opt_kpa:.0f} kPa",
        (opt_kpa, rc_mean[opt_idx]),
        textcoords="offset points",
        xytext=(12, 14),
        fontsize=8.5,
        fontweight="bold",
        color="#059669",
    )

    l2 = ax2_twin.plot(x, deff_mean, "^--", color="#0284c7", linewidth=1.7, label="Ionic D_eff,Z / D_0 (right)")
    l3 = ax2_twin.plot(x, sig_mean, "v--", color="#d97706", linewidth=1.7, label="Solid σ_eff,Z / σ_0 (right)")

    ax2.set_xlabel("Clamping Pressure σ [kPa]")
    ax2.set_ylabel("Series Cell Resistance R_cell [-]")
    ax2_twin.set_ylabel("Normalised Transport Coefficient [-]")
    ax2.set_title("(b) Ionic Pore vs Electronic Fibre Percolation Trade-Off")
    ax2.grid(alpha=0.25)

    lines = l1 + l2 + l3
    labels = [ln.get_label() for ln in lines]
    ax2.legend(lines, labels, loc="upper center", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_anisotropic_compression_tradeoff.png", bbox_inches="tight")
    plt.close(fig)

    return {
        "loads_kpa": loads_kpa,
        "seeds": seeds,
        "shape_zyx": [56, 56, 56],
        "porosity_mean": [round(float(v), 5) for v in np.mean(eps_mat, axis=0)],
        "tau_z_mean": [round(float(v), 4) for v in tz_mean],
        "tau_z_std": [round(float(v), 4) for v in tz_std],
        "tau_xy_mean": [round(float(v), 4) for v in txy_mean],
        "d_eff_z_mean": [round(float(v), 5) for v in deff_mean],
        "sigma_eff_z_mean": [round(float(v), 5) for v in sig_mean],
        "r_cell_mean": [round(float(v), 5) for v in rc_mean],
        "r_cell_std": [round(float(v), 5) for v in rc_std],
        "optimal_clamping_kpa": opt_kpa,
    }


def run_fig3_steering_efficiency() -> dict:
    """Compare Active Bayesian GP steering vs Passive Uniform Grid on a 17-point OpenImpala ground truth."""
    dense_loads = [float(x) for x in np.linspace(20.0, 500.0, 17)]
    sspec = SolveSpec(
        conducting_phase=0,
        solid_phases=(1, 2),
        directions=("z",),
        solver="mlmg",
        compute_solid_wiring=True,
        return_fields=False,
    )

    gt_rcell: dict[float, float] = {}
    for kpa in dense_loads:
        mspec = make_carbon_felt_spec(kpa, shape=(48, 48, 48), seed=42)
        gt_rcell[kpa] = run_transport_pipeline(mspec, sspec).summary.cell_resistance_fom

    true_min = min(gt_rcell.values())
    true_opt_kpa = min(gt_rcell, key=gt_rcell.get)  # type: ignore[arg-type]

    # Active Bayesian campaign: start with endpoints [20, 500] and iteratively query nearest grid point to x_next
    active_x = [dense_loads[0], dense_loads[-1]]
    active_y = [gt_rcell[active_x[0]], gt_rcell[active_x[1]]]
    budgets = list(range(2, 10))
    active_regret = [min(active_y) - true_min]

    for _step in range(3, 10):
        sug = suggest_next_setpoint_gp(active_x, active_y, bounds=(20.0, 500.0))
        remaining = [x for x in dense_loads if x not in active_x]
        chosen = min(remaining, key=lambda x: abs(x - sug.suggested_value))
        active_x.append(chosen)
        active_y.append(gt_rcell[chosen])
        active_regret.append(min(active_y) - true_min)

    # Passive uniform campaign: for each scan budget N in 2..9, evaluate N evenly-spaced grid indices
    passive_regret: list[float] = []
    for n_scans in budgets:
        idxs = np.round(np.linspace(0, len(dense_loads) - 1, n_scans)).astype(int)
        sampled_y = [gt_rcell[dense_loads[int(i)]] for i in idxs]
        passive_regret.append(min(sampled_y) - true_min)

    final_gp = suggest_next_setpoint_gp(active_x[:5], active_y[:5], bounds=(20.0, 500.0))

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.2, 4.3), dpi=200)

    ax_l.plot(dense_loads, [gt_rcell[k] for k in dense_loads], "k:", linewidth=1.6, label="17-scan OpenImpala ground truth")
    ax_l.fill_between(final_gp.grid_x, final_gp.gp_lower_95, final_gp.gp_upper_95, color="#93c5fd", alpha=0.35, label="GP 95% band (after 5 scans)")
    ax_l.plot(final_gp.grid_x, final_gp.gp_mean, color="#2563eb", linewidth=2.1, label="GP mean (5 scans)")
    ax_l.scatter(active_x[:5], active_y[:5], color="#059669", s=55, zorder=5, label="5 Active Beamline Scans")
    ax_l.set_xlabel("Clamping Pressure σ [kPa]")
    ax_l.set_ylabel("Series Cell Resistance R_cell [-]")
    ax_l.set_title("(a) 5-Scan Active GP Fit vs 17-Scan Ground Truth")
    ax_l.grid(alpha=0.28)
    ax_l.legend(fontsize=8)

    ax_r.plot(budgets, passive_regret, "s--", color="#64748b", linewidth=2.0, markersize=6, label="Passive Uniform Scan Schedule")
    ax_r.plot(budgets, active_regret, "o-", color="#059669", linewidth=2.3, markersize=6, label="Active Hub GP Steering (EI)")
    ax_r.set_xlabel("Number of Operando Tomography Scans Acquired")
    ax_r.set_ylabel("Optimality Regret  R_cell(σ̂*) − R_cell(σ*)  [-]")
    ax_r.set_title("(b) Beamtime Reduction ('Insight per Photon')")
    ax_r.grid(alpha=0.28)
    ax_r.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_autonomous_steering_efficiency.png", bbox_inches="tight")
    plt.close(fig)

    return {
        "dense_loads_kpa": dense_loads,
        "ground_truth_r_cell": [round(gt_rcell[k], 5) for k in dense_loads],
        "true_optimal_kpa": true_opt_kpa,
        "true_min_r_cell": round(true_min, 5),
        "scan_budgets": budgets,
        "active_sequence_kpa": active_x,
        "active_regret": [round(v, 5) for v in active_regret],
        "passive_regret": [round(v, 5) for v in passive_regret],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print("[1/3] Running Fig 1 solver latency & speedup benchmark...")
    f1 = run_fig1_scaling()
    print("[2/3] Running Fig 2 anisotropic carbon-felt compression sweep (7 loads x 3 seeds)...")
    f2 = run_fig2_compression_tradeoff()
    print("[3/3] Running Fig 3 autonomous Bayesian steering vs passive grid benchmark...")
    f3 = run_fig3_steering_efficiency()
    elapsed = time.perf_counter() - t0

    with oi.Session():
        binfo = oi.build_info()

    provenance = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_origin": "synthetic (procedural 3D Gaussian random fields & fibrous carbon-felt mats)",
        "precision": "int32 phase grids + float64 PDE solves + float32 flux fields",
        "host_cpu": _cpu_model(),
        "python_version": platform.python_version(),
        "hub_version": hub_version,
        "openimpala_version": oi.__version__,
        "openimpala_backend": binfo.get("backend"),
        "basecoat_version": basecoat.__version__,
        "total_benchmark_walltime_s": round(elapsed, 2),
        "fig1_scaling": f1,
        "fig2_compression_tradeoff": f2,
        "fig3_autonomous_steering": f3,
    }
    prov_path = OUT_DIR / "benchmark_provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Done in {elapsed:.1f}s -> wrote PNGs & {prov_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
