#!/usr/bin/env python3
"""Automated headless Playwright walkthrough & video recorder (`M2.3`) for OpenImpala-Hub.

Starts a local Gunicorn/Werkzeug server on an ephemeral port, walks a real Chromium browser
through all four Hub tabs (2D Flux, 3D Cutaway, Compare States, Autonomous GP Planner),
executes the recommended next scan live, downloads & verifies the `.zip` NIA Report Bundle,
saves high-DPI UI screenshots into `docs/figures/`, and optionally records a `.webm` walkthrough.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import socket
import threading
import time
import zipfile

from werkzeug.serving import make_server

from openimpala_hub.app.server import create_dash_app

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "docs" / "figures"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def run_walkthrough(*, record_video_dir: Path | None = None) -> dict[str, str]:
    from playwright.sync_api import sync_playwright

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    dash_app = create_dash_app(warm_start=True, warm_shape=(48, 48, 48))
    port = _free_port()
    srv = make_server("127.0.0.1", port, dash_app.server, threaded=True)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()

    url = f"http://127.0.0.1:{port}/"
    outputs: dict[str, str] = {}

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            ctx_kwargs: dict = {
                "viewport": {"width": 1440, "height": 920},
                "device_scale_factor": 2,
                "accept_downloads": True,
            }
            if record_video_dir is not None:
                record_video_dir.mkdir(parents=True, exist_ok=True)
                ctx_kwargs["record_video_dir"] = str(record_video_dir)
                ctx_kwargs["record_video_size"] = {"width": 1280, "height": 800}

            context = browser.new_context(**ctx_kwargs)
            page = context.new_page()

            # 1. Landing view (2D Slices & Diffusive Flux Crowding)
            t0 = time.perf_counter()
            page.goto(url, wait_until="networkidle")
            page.wait_for_selector("#kpi-cards-row .kpi-card", timeout=20000)
            page.wait_for_timeout(1200)
            t_first_paint = time.perf_counter() - t0

            p1 = FIG_DIR / "ui_01_2d_flux_and_metrics.png"
            page.screenshot(path=str(p1), full_page=False)
            outputs["ui_2d_flux"] = str(p1)

            # 2. Switch to 3D Microstructure & Cutaway tab
            page.get_by_text("3D Microstructure & Cutaway").click()
            page.wait_for_timeout(1800)
            p2 = FIG_DIR / "ui_02_3d_isosurface_and_roi.png"
            page.screenshot(path=str(p2), full_page=False)
            outputs["ui_3d_isosurface"] = str(p2)

            # 3. Switch to Compare States tab
            page.get_by_text("Compare States").click()
            page.wait_for_timeout(1200)
            p3 = FIG_DIR / "ui_03_compare_compression_states.png"
            page.screenshot(path=str(p3), full_page=False)
            outputs["ui_compare"] = str(p3)

            # 4. Switch to Autonomous Planner (Digital Twin) & execute recommended scan live!
            page.get_by_text("Autonomous Planner (Digital Twin)").click()
            page.wait_for_timeout(1200)
            page.locator("#run-suggested-setpoint-btn").click()
            page.wait_for_timeout(3000)
            p4 = FIG_DIR / "ui_04_autonomous_gp_planner.png"
            page.screenshot(path=str(p4), full_page=False)
            outputs["ui_autonomous_twin"] = str(p4)

            # 5. Click Export NIA Bundle (.zip) and verify archive integrity
            with page.expect_download(timeout=30000) as dl_info:
                page.locator("#export-bundle-btn").click()
            download = dl_info.value
            tmp_zip = Path("/tmp") / download.suggested_filename
            download.save_as(str(tmp_zip))
            with zipfile.ZipFile(tmp_zip, "r") as zf:
                assert "manifest.json" in zf.namelist()
                assert "metrics.json" in zf.namelist()
            tmp_zip.unlink(missing_ok=True)

            video_obj = page.video
            context.close()
            if video_obj is not None:
                outputs["walkthrough_video"] = str(video_obj.path())
            browser.close()

        outputs["landing_load_time_s"] = f"{t_first_paint:.3f}"
    finally:
        srv.shutdown()

    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture UI screenshots and optional walkthrough video.")
    parser.add_argument(
        "--video-dir",
        default=None,
        help="Optional directory to write Playwright .webm walkthrough video",
    )
    args = parser.parse_args()
    vdir = Path(args.video_dir).resolve() if args.video_dir else None
    out = run_walkthrough(record_video_dir=vdir)
    for k, v in out.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
