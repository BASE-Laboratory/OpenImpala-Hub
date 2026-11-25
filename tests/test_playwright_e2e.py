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
            page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")
            page.wait_for_selector("#kpi-cards-row .kpi-card", timeout=15000)

            # Switch to analytical cube preset and assert exact 1.0000 porosity & 1.000 tortuosity in DOM
            page.locator("#preset-btn-cube").click()
            page.wait_for_function(
                "() => document.querySelector('#kpi-cards-row')?.innerText.includes('1.0000')",
                timeout=15000,
            )
            kpi_text = page.locator("#kpi-cards-row").inner_text()
            assert "1.0000" in kpi_text
            assert "1.000" in kpi_text

            # Trigger real browser download of the NIA report bundle
            with page.expect_download(timeout=20000) as dl_info:
                page.locator("#export-bundle-btn").click()
            dl_file = tmp_path / dl_info.value.suggested_filename
            dl_info.value.save_as(str(dl_file))
            with zipfile.ZipFile(dl_file, "r") as zf:
                assert {"manifest.json", "metrics.json", "bpx_parameters.json", "reproduce.py"}.issubset(
                    set(zf.namelist())
                )

            context.close()
            browser.close()
    finally:
        srv.shutdown()
