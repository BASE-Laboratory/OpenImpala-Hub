"""End-to-end headless Chromium browser tests (`v0.4.0` Dash app + static GitHub Pages OME-Zarr)."""

from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import socket
import threading
import zipfile

import pytest
from werkzeug.serving import make_server

from openimpala_hub.app.server import create_dash_app

# Reject decorative pictographic emojis while allowing standard math/scientific symbols (µ, τ, ε, ·, –)
_EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_browser_v04_baselab_signature_porousdiff_xray_and_downloads(tmp_path: Path) -> None:
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

            context = browser.new_context(accept_downloads=True, viewport={"width": 1440, "height": 920})
            page = context.new_page()
            js_errors: list[str] = []
            page.on("pageerror", lambda err: js_errors.append(str(err)))

            page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")
            page.wait_for_selector("#results-summary-table tr", timeout=15000)

            # 1. Verify BASE Lab Signature light theme (#FCD0B1 -> rgb(252, 208, 177)) & zero emojis
            bg_light = page.evaluate("() => getComputedStyle(document.body).backgroundColor")
            assert bg_light == "rgb(252, 208, 177)"
            body_txt = page.evaluate("() => document.body.innerText")
            assert _EMOJI_RE.search(body_txt) is None

            # 2. Default Generate panel: run PorousDiff AI · Carbon felt + save 3D volume (.ome.zarr.zip)
            page.locator("#run-generate-btn").click()
            page.wait_for_function(
                "() => document.querySelector('#active-sample-subtitle')?.innerText.includes('PorousDiff Carbon Felt')",
                timeout=25000,
            )
            with page.expect_download(timeout=25000) as vol_dl:
                page.locator("#save-volume-btn").click()
            vol_zip = tmp_path / vol_dl.value.suggested_filename
            vol_dl.value.save_as(str(vol_zip))
            with zipfile.ZipFile(vol_zip, "r") as zf:
                assert any(name.endswith(".zattrs") for name in zf.namelist())

            # 3. Toggle Warm Darkroom (#222222 -> rgb(34, 34, 34) after 150ms CSS transition) & back
            page.locator("#theme-toggle-btn").click()
            page.wait_for_function(
                "() => getComputedStyle(document.body).backgroundColor === 'rgb(34, 34, 34)'",
                timeout=5000,
            )
            page.locator("#theme-toggle-btn").click()
            page.wait_for_function(
                "() => getComputedStyle(document.body).backgroundColor === 'rgb(252, 208, 177)'",
                timeout=5000,
            )

            # 4. Switch to X-ray Scans: verify lazy PNG slice + native 1x1x1 region simulation
            page.locator("#sample-mode-radio").get_by_text("X-ray Scans").click()
            page.wait_for_function(
                "() => document.querySelector('#primary-caption-left')?.innerText.includes('XY slice')",
                timeout=15000,
            )
            img_src = page.locator("#primary-slice-img").get_attribute("src") or ""
            assert img_src.startswith("data:image/png;base64,")

            page.locator("#simulate-roi-btn").click()
            page.wait_for_function(
                "() => document.querySelector('#active-sample-subtitle')?.innerText.includes('X-ray region')",
                timeout=25000,
            )

            # 5. Navigate across Compare, Discharge & Pores, Plan Scans, and Guide
            page.locator("#main-view-tabs").get_by_text("Compare").click()
            page.wait_for_function(
                "() => (document.querySelector('#cmp-left-img')?.getAttribute('src') || '').startsWith('data:image/png;base64,')",
                timeout=15000,
            )

            page.locator("#main-view-tabs").get_by_text("Discharge & Pores").click()
            page.wait_for_selector("#electrochem-graph .main-svg", state="visible", timeout=15000)

            page.locator("#main-view-tabs").get_by_text("Plan Scans").click()
            page.wait_for_selector("#gp-steering-graph .main-svg", state="visible", timeout=15000)
            assert "Suggested next load:" in page.locator("#gp-recommendation-banner").inner_text()

            page.locator("#main-view-tabs").get_by_text("Guide").click()
            page.locator("#preset-btn-cube").click()
            page.wait_for_function(
                "() => document.querySelector('#guide-cards-container')?.innerText.includes('DONE')",
                timeout=20000,
            )
            assert _EMOJI_RE.search(page.locator("#guide-cards-container").inner_text()) is None

            # 6. Upload greyscale .tif & download report bundle (.zip)
            page.locator("#main-view-tabs").get_by_text("Slices").click()
            page.locator("#sample-mode-radio").get_by_text("Upload").click()
            rng = np.random.default_rng(3)
            grey = np.where(rng.random((20, 20, 20)) < 0.70, 30, 210).astype(np.uint8)
            sample_tif = tmp_path / "browser_xct.tif"
            write_volume(Volume.from_array(grey, voxel_size_um=None), sample_tif)

            page.locator("#upload-volume-box input[type='file']").set_input_files(str(sample_tif))
            page.wait_for_function(
                "() => document.querySelector('#upload-status-text')?.innerText.includes('browser_xct.tif')",
                timeout=15000,
            )
            page.locator("#run-upload-btn").click()
            page.wait_for_function(
                "() => document.querySelector('#active-sample-subtitle')?.innerText.includes('browser_xct.tif')",
                timeout=20000,
            )

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
                }.issubset(set(zf.namelist()))

            assert js_errors == [], f"Unexpected browser JS errors: {js_errors}"
            context.close()
            browser.close()
    finally:
        srv.shutdown()


def test_static_github_pages_ome_zarr_http_chunk_streaming() -> None:
    """Serve `docs_site/` over HTTP and verify `DecompressionStream('deflate')` real synchrotron OME-Zarr chunk reads."""
    playwright = pytest.importorskip("playwright.sync_api")
    docs_dir = Path(__file__).resolve().parents[1] / "docs_site"
    assert (docs_dir / "data" / "dls_i12_451kpa.ome.zarr" / "0" / ".zarray").is_file()
    assert (docs_dir / "data" / "dls_i13_zn13x_capillary.ome.zarr" / "0" / ".zarray").is_file()

    port = _free_port()
    handler = partial(SimpleHTTPRequestHandler, directory=str(docs_dir))
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    try:
        with playwright.sync_playwright() as p:
            try:
                browser = p.chromium.launch(headless=True)
            except Exception as exc:
                pytest.skip(f"Headless Chromium binary unavailable on runner: {exc}")

            page = browser.new_page(viewport={"width": 1440, "height": 900})
            js_errors: list[str] = []
            page.on("pageerror", lambda err: js_errors.append(str(err)))

            page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")
            page.wait_for_function(
                "() => document.querySelector('#results-table')?.innerText.includes('Open pore space')",
                timeout=10000,
            )

            # Switch to X-ray Scans -> triggers real HTTP fetch('./data/dls_i12_451kpa.ome.zarr/0/4/0/0')
            page.locator("#source-bar").get_by_text("X-ray Scans").click()
            page.wait_for_function(
                "() => (document.querySelector('#tech-readout')?.textContent || '').includes('OME-Zarr chunk ./data/dls_i12_451kpa.ome.zarr/0/')",
                timeout=10000,
            )

            # Simulate selected 1x1x1 region in-browser
            page.locator("#btn-sim-roi").click()
            page.wait_for_function(
                "() => document.querySelector('#sample-subtitle')?.innerText.includes('X-ray Region')",
                timeout=10000,
            )
            assert _EMOJI_RE.search(page.evaluate("() => document.body.innerText")) is None
            assert js_errors == [], f"Static GitHub Pages JS errors: {js_errors}"
            browser.close()
    finally:
        httpd.shutdown()
