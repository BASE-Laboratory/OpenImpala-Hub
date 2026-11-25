"""Serverless deployment entrypoint for Modal.com (£0 Starter tier + per-second L4 GPU burst).

Usage (once authenticated via `modal setup`):
    modal serve deploy/modal_app.py     # ephemeral live dev URL
    modal deploy deploy/modal_app.py    # persistent https://<workspace>--openimpala-hub-web.modal.run

Cost profile (verified Nov 2026):
- Modal Starter plan has $0/mo subscription and includes $30/mo in free compute credits.
- `web()` scales to zero when idle ($0.00/s) and wakes in ~4-8s on visitor requests.
- `gpu_burst_solve()` attaches an NVIDIA L4 24 GB GPU (~$0.80/hr = $0.00022/s, <0.2p per 10s run)
  exclusively when invoked for large `256^3` CUDA solves or live `PorousDiff` 3D diffusion sampling.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_modal_objects() -> tuple[Any, Any, Any]:
    """Construct the Modal `App`, CPU WSGI endpoint, and optional L4 GPU burst worker."""
    import modal

    cpu_image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git", "libgomp1", "gfortran", "libopenmpi-dev", "cmake", "gcc", "make")
        .pip_install(
            "basecoat @ git+https://github.com/BASE-Laboratory/basecoat.git@v0.3.0",
            "openimpala[openmpi]>=5.0,<6.0",
            "sms @ git+https://github.com/BASE-Laboratory/structural_mechanics_solver.git",
            "dash>=2.17,<4.0",
            "plotly>=5.22,<7.0",
            "numpy>=1.26",
            "scipy>=1.11",
            "scikit-image>=0.22",
            "matplotlib>=3.8",
            "pillow>=10.0",
        )
        .env(
            {
                "OMP_NUM_THREADS": "2",
                "OMPI_MCA_btl_vader_single_copy_mechanism": "none",
                "HUB_MAX_SOLVE_SIDE": "128",
            }
        )
        .add_local_python_source("openimpala_hub")
    )

    gpu_image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git", "libgomp1", "gfortran", "libopenmpi-dev", "cmake", "gcc", "make")
        .pip_install(
            "basecoat @ git+https://github.com/BASE-Laboratory/basecoat.git@v0.3.0",
            "openimpala-cuda>=5.0,<6.0",
            "sms @ git+https://github.com/BASE-Laboratory/structural_mechanics_solver.git",
            "porousdiff @ git+https://github.com/BASE-Laboratory/PorousDiff.git@feature/paper-ready-conditional",
            "openmpi",
            "numpy>=1.26",
            "scipy>=1.11",
            "scikit-image>=0.22",
            "matplotlib>=3.8",
            "pillow>=10.0",
        )
        .env(
            {
                "OMPI_MCA_btl_vader_single_copy_mechanism": "none",
                "HUB_MAX_SOLVE_SIDE": "256",
            }
        )
        .add_local_python_source("openimpala_hub")
    )

    weights_vol = modal.Volume.from_name("openimpala-hub-porousdiff", create_if_missing=True)
    app = modal.App("openimpala-hub")

    @app.function(
        image=cpu_image,
        cpu=2.0,
        memory=4096,
        scaledown_window=180,
        volumes={"/data/porousdiff": weights_vol},
    )
    @modal.wsgi_app()
    def web() -> Any:
        import os

        os.environ.setdefault("OPENIMPALA_HUB_POROUSDIFF_DIR", "/data/porousdiff")
        from openimpala_hub.app.server import get_wsgi_app

        return get_wsgi_app()

    @app.function(
        image=gpu_image,
        gpu="L4",
        memory=16384,
        timeout=300,
        scaledown_window=60,
        volumes={"/data/porousdiff": weights_vol},
    )
    def gpu_burst_solve(micro_dict: dict[str, Any], solve_dict: dict[str, Any]) -> dict[str, Any]:
        """Execute a high-resolution (`<= 256^3`) OpenImpala CUDA solve on an on-demand L4 GPU."""
        import os

        os.environ.setdefault("OPENIMPALA_HUB_POROUSDIFF_DIR", "/data/porousdiff")
        from openimpala_hub.contracts import MicrostructureSpec, SolveSpec
        from openimpala_hub.engine.solver import run_transport_pipeline

        mspec = MicrostructureSpec.from_dict(micro_dict)
        sspec = SolveSpec.from_dict({**solve_dict, "max_solve_side": 256})
        executed = run_transport_pipeline(mspec, sspec)
        return executed.summary.to_dict()

    return app, web, gpu_burst_solve


try:
    app, web, gpu_burst_solve = build_modal_objects()
except ImportError:
    # Allows importing or syntax-checking deploy/modal_app.py when `modal` extra is not installed
    app = web = gpu_burst_solve = None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate or inspect the OpenImpala-Hub serverless Modal deployment spec."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify syntax and Modal availability without deploying",
    )
    args = parser.parse_args()
    if args.check:
        status = "installed" if app is not None else "not installed (install via `pip install .[modal]`)"
        print(f"Modal spec valid (repo={REPO_ROOT.name}, modal SDK: {status}).")
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
