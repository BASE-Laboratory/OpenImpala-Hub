"""End-to-end headless Chromium browser test via Playwright."""

from __future__ import annotations

from pathlib import Path
import socket
import threading
import zipfile

import pytest
from werkzeug.serving import make_server

from openimpala_hub.app.server import create_dash_app


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_browser_happy_path_preset_and_zip_download(tmp_path: Path) -> None:
    playwright = pytest.importorskip("playwright.sync_api")
    import numpy as np
    from basecoat import Volume, write_volume

    dash_app = create_dash_app(warm_start=True, warm_shape=(28, 28, 28))
    port = _free_port()
    srv = make_server("127.0.0.1", port, dash_app.server, threaded=True)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()

    try:
        with playwright.sync_playwright() as p:
            try:
                browser = p.chromium.launch(headless=True)
            except Exception as exc:
                pytest.skip(f"Headless Chromium binary unavailable on runner: {exc}")

            context = browser.new_context(accept_downloads=True)
            page = context.new_page()
            js_errors: list[str] = []
            page.on("pageerror", lambda err: js_errors.append(str(err)))

            page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")
            page.wait_for_selector("#kpi-cards-row .kpi-card", timeout=15000)

            # 1. Switch to Preset #2 (3-Phase Cathode) and verify Positive electrode role in readout
            page.locator("#preset-btn-cathode").click()
            page.wait_for_function(
                "() => document.querySelector('#pybamm-bpx-preview')?.textContent.includes('Positive electrode porosity')",
                timeout=30000,
            )
            assert "role: positive" in page.locator("#preset-context-banner").inner_text()

            # 2. Switch to analytical cube preset and assert exact 1.0000 porosity & 1.000 tortuosity in DOM
            page.locator("#preset-btn-cube").click()
            page.wait_for_function(
                "() => document.querySelector('#kpi-cards-row')?.innerText.includes('1.0000')",
                timeout=15000,
            )
            kpi_text = page.locator("#kpi-cards-row").inner_text()
            assert "1.0000" in kpi_text
            assert "1.000" in kpi_text

            # 2b. Toggle Teaching Lab drawer and verify Lab 1 badge turns verified after analytical cube solve
            page.locator("#teaching-lab-btn").click()
            page.wait_for_selector("#teaching-lab-drawer", state="visible", timeout=10000)
            drawer_text = page.locator("#teaching-lab-drawer").inner_text()
            assert "Exercises Verified" in drawer_text
            assert "✅ Lab 1" in drawer_text

            # 2c. Open Morphology & Pore Network and Cell Electrochemistry & Mechanics tabs
            page.get_by_text("Morphology & Pore Network").click()
            page.wait_for_selector("#morphology-graph .main-svg", state="visible", timeout=15000)
            page.get_by_text("Cell Electrochemistry & Mechanics").click()
            page.wait_for_selector("#electrochem-graph .main-svg", state="visible", timeout=15000)

            # 3. Open Digital Twin tab and test 1-click Jump to ROI #1
            page.get_by_text("Autonomous Planner (Digital Twin)").click()
            page.wait_for_selector("#jump-roi-btn", timeout=10000)
            page.locator("#jump-roi-btn").click()
            page.wait_for_selector("#slice-heatmap-graph", state="visible", timeout=10000)

            # 4. Upload a real greyscale .tif stack through the browser file input & solve
            rng = np.random.default_rng(3)
            grey = np.where(rng.random((20, 20, 20)) < 0.70, 30, 210).astype(np.uint8)
            sample_tif = tmp_path / "browser_xct.tif"
            write_volume(Volume.from_array(grey, voxel_size_um=None), sample_tif)

            page.locator("#source-kind-dropdown").click()
            page.get_by_text("Upload Experimental Volume").click()
            page.wait_for_selector("#upload-volume-box input[type='file']", state="attached", timeout=10000)
            page.locator("#upload-volume-box input[type='file']").set_input_files(str(sample_tif))
            page.wait_for_function(
                "() => document.querySelector('#upload-status-text')?.innerText.includes('browser_xct.tif')",
                timeout=15000,
            )
            page.locator("#run-pipeline-btn").click()
            page.wait_for_function(
                "() => document.querySelector('#preset-context-banner')?.innerText.includes('origin: user_upload')",
                timeout=20000,
            )

            # 5. Trigger real browser download of the NIA report bundle
            with page.expect_download(timeout=25000) as dl_info:
                page.locator("#export-bundle-btn").click()
            dl_file = tmp_path / dl_info.value.suggested_filename
            dl_info.value.save_as(str(dl_file))
            with zipfile.ZipFile(dl_file, "r") as zf:
                assert {
                    "manifest.json",
                    "metrics.json",
                    "bpx_parameters.json",
                    "reproduce.py",
                    "figures/morphology_and_electrochem.png",
                }.issubset(set(zf.namelist()))

            assert js_errors == [], f"Unexpected browser JS errors: {js_errors}"
            context.close()
            browser.close()
    finally:
        srv.shutdown()
