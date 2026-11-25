"""Command-line interface (`openimpala-hub`) for serving the web app and batch runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from . import __version__
from .contracts import MicrostructureSpec, SolveSpec
from .engine.bundle import write_report_bundle
from .engine.solver import run_transport_pipeline
from .presets import DEFAULT_PRESET_KEY, PRESETS, make_carbon_felt_spec


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openimpala-hub",
        description="The OpenImpala Hub — 3D microstructure synthesis, transport simulation, and steering.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- presets ---
    p_presets = sub.add_parser("presets", help="List built-in Hub presets")
    p_presets.add_argument("--json", action="store_true", help="Emit JSON array")

    # --- run ---
    p_run = sub.add_parser("run", help="Execute a preset or uploaded volume headlessly")
    p_run.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default=DEFAULT_PRESET_KEY,
        help=f"Preset scenario to run (default: {DEFAULT_PRESET_KEY})",
    )
    p_run.add_argument(
        "--input",
        dest="input_path",
        default=None,
        help="Optional path to a .ome.zarr, .tif, .npy, or .h5 volume (overrides --preset)",
    )
    p_run.add_argument(
        "--voxel-size-um",
        type=float,
        default=1.6,
        help="Isotropic physical voxel spacing in micrometres when using --input (default: 1.6)",
    )
    p_run.add_argument(
        "--clamping-kpa",
        type=float,
        default=None,
        help="Override clamping pressure [kPa] for the carbon_felt_compression preset",
    )
    p_run.add_argument(
        "--solver",
        choices=("mlmg", "hypre", "scipy"),
        default=None,
        help="Override linear solver backend",
    )
    p_run.add_argument(
        "--output-bundle",
        default=None,
        help="Optional output .zip path for the reproducible NIA report bundle",
    )
    p_run.add_argument(
        "--no-zarr",
        action="store_true",
        help="Omit microstructure.ome.zarr from --output-bundle to reduce archive size",
    )
    p_run.add_argument(
        "--json",
        action="store_true",
        help="Print SolveRunSummary as formatted JSON",
    )

    # --- verify ---
    sub.add_parser(
        "verify",
        help="Run analytical verification checks (uniform cube tau=1.000 & felt anisotropy)",
    )

    # --- serve ---
    p_serve = sub.add_parser("serve", help="Launch the interactive Plotly Dash web server")
    p_serve.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    p_serve.add_argument("--port", type=int, default=7860, help="Bind port (default: 7860)")
    p_serve.add_argument("--debug", action="store_true", help="Enable Dash debug reloader")

    return parser


def _cmd_presets(args: argparse.Namespace) -> int:
    rows = [
        {
            "key": p.key,
            "title": p.title,
            "badge": p.badge,
            "summary": p.summary,
            "source_kind": p.microstructure.source_kind,
            "shape": list(p.microstructure.shape),
        }
        for p in PRESETS.values()
    ]
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for r in rows:
            print(f"{r['key']:28s} | {r['badge']:24s} | {r['title']}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    if args.input_path:
        p = Path(args.input_path).resolve()
        mspec = MicrostructureSpec(
            source_kind="user_upload",
            label=p.name,
            upload_path=str(p),
            voxel_size_um=(args.voxel_size_um, args.voxel_size_um, args.voxel_size_um),
            threshold="otsu",
        )
        sspec = SolveSpec(solver=args.solver or "mlmg")
    else:
        preset = PRESETS[args.preset]
        mspec = preset.microstructure
        if args.clamping_kpa is not None and args.preset == "carbon_felt_compression":
            mspec = make_carbon_felt_spec(
                args.clamping_kpa,
                shape=mspec.shape,
                seed=mspec.seed,
            )
        sspec = preset.solve
        if args.solver is not None:
            sspec = SolveSpec(**{**sspec.to_dict(), "solver": args.solver})  # type: ignore[arg-type]

    executed = run_transport_pipeline(mspec, sspec)
    if args.output_bundle:
        out_path = write_report_bundle(
            executed,
            args.output_bundle,
            include_ome_zarr=not args.no_zarr,
        )
        if not args.json:
            print(f"Wrote report bundle: {out_path}")

    if args.json:
        print(json.dumps(executed.summary.to_dict(), indent=2))
    else:
        s = executed.summary
        print(f"Run [{s.run_id}] {s.label} ({s.data_origin}, backend={s.solver_backend})")
        print(f"  Shape: {s.shape_zyx} @ {s.voxel_size_um_zyx} um | Porosity: {s.porosity:.4f}")
        for d, dt in s.directional.items():
            print(
                f"  Axis {d.upper()}: percolates={dt.percolates}, "
                f"tau={dt.tortuosity}, D_eff={dt.effective_diffusivity:.4f}"
            )
        print(
            f"  SSA (Cauchy-Crofton): {s.specific_surface_area_m_inv:.1f} m^-1 | "
            f"R_cell FoM: {s.cell_resistance_fom:.4f} | Solve: {s.solve_time_s:.3f}s"
        )
    return 0


def _cmd_verify() -> int:
    cube = PRESETS["teaching_analytical_cube"]
    res_cube = run_transport_pipeline(cube.microstructure, cube.solve)
    s_cube = res_cube.summary
    assert abs(s_cube.porosity - 1.0) < 1e-9, f"Expected porosity=1.0, got {s_cube.porosity}"
    for d in ("x", "y", "z"):
        tau = s_cube.directional[d].tortuosity
        assert tau is not None and abs(tau - 1.0) < 1e-4, f"Expected tau_{d}=1.0, got {tau}"
    assert res_cube.flux_field is not None
    max_flux_err = float(np.max(np.abs(res_cube.flux_field - 1.0)))
    assert max_flux_err < 1e-3, f"Expected uniform |J/J0|=1.0, max error={max_flux_err}"

    felt = PRESETS["carbon_felt_compression"]
    res_felt = run_transport_pipeline(felt.microstructure, felt.solve)
    tz = res_felt.summary.directional["z"].tortuosity
    tx = res_felt.summary.directional["x"].tortuosity
    assert tz is not None and tx is not None and tz > tx, (
        f"Expected through-plane anisotropy tau_z > tau_x in carbon felt, got tau_z={tz}, tau_x={tx}"
    )
    print(
        f"VERIFIED: open_cube tau=1.0000 (max |J-1|={max_flux_err:.2e}); "
        f"carbon_felt anisotropy tau_z={tz:.3f} > tau_x={tx:.3f} (backend={res_felt.summary.solver_backend})."
    )
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from .app.server import create_dash_app

    app = create_dash_app(warm_start=True)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "presets":
        return _cmd_presets(args)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "verify":
        return _cmd_verify()
    if args.command == "serve":
        return _cmd_serve(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
