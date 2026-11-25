"""Dash + Flask application factory and JSON REST endpoints (`/healthz`, `/api/v1/*`)."""

from __future__ import annotations

import io
from typing import Any

import basecoat
from dash import Dash
from flask import Flask, Response, jsonify, request
import openimpala as oi

from .. import __version__ as _HUB_VERSION
from ..contracts import MicrostructureSpec, SolveSpec
from ..engine.bundle import build_report_bundle_bytes
from ..engine.solver import SOLVER_LOCK, run_transport_pipeline
from ..presets import DEFAULT_PRESET_KEY, PRESETS
from .callbacks import register_callbacks
from .layout import build_hub_layout
from .store import RunRegistry


def _register_api_routes(server: Flask, registry: RunRegistry) -> None:
    """Attach stable healthcheck and headless JSON API routes onto `app.server`."""

    @server.get("/healthz")
    def healthz() -> Response:
        with SOLVER_LOCK, oi.Session():
            binfo = oi.build_info()
        return jsonify(
            {
                "status": "ok",
                "service": "openimpala-hub",
                "hub_version": _HUB_VERSION,
                "openimpala_version": oi.__version__,
                "basecoat_version": basecoat.__version__,
                "solver_backend": binfo.get("backend", "unknown"),
                "cached_runs": len(registry.list_summaries()),
            }
        )

    @server.get("/api/v1/presets")
    def api_presets() -> Response:
        return jsonify(
            [
                {
                    "key": p.key,
                    "title": p.title,
                    "badge": p.badge,
                    "summary": p.summary,
                    "microstructure": p.microstructure.to_dict(),
                    "solve": p.solve.to_dict(),
                }
                for p in PRESETS.values()
            ]
        )

    @server.post("/api/v1/solve")
    def api_solve() -> tuple[Response, int] | Response:
        body: dict[str, Any] = request.get_json(silent=True) or {}
        try:
            if "preset" in body and body["preset"]:
                pkey = str(body["preset"])
                if pkey not in PRESETS:
                    return jsonify({"error": f"Unknown preset {pkey!r}."}), 400
                preset = PRESETS[pkey]
                mspec = preset.microstructure
                sspec = preset.solve
                if "shape" in body:
                    shp = tuple(int(x) for x in body["shape"])
                    mspec = MicrostructureSpec(**{**mspec.to_dict(), "shape": shp})
            elif "microstructure" in body:
                mspec = MicrostructureSpec.from_dict(body["microstructure"])
                sspec = SolveSpec.from_dict(body.get("solve", {}))
            else:
                preset = PRESETS[DEFAULT_PRESET_KEY]
                mspec = preset.microstructure
                sspec = preset.solve

            executed = run_transport_pipeline(mspec, sspec)
            registry.put(executed)
            return jsonify(executed.summary.to_dict())
        except Exception as exc:
            return jsonify({"error": str(exc), "error_type": type(exc).__name__}), 400

    @server.get("/api/v1/bundle/<run_id>.zip")
    def api_bundle(run_id: str) -> tuple[Response, int] | Response:
        run = registry.get(run_id)
        if run is None:
            return jsonify({"error": f"Run {run_id!r} not found in server cache."}), 404
        raw_zip = build_report_bundle_bytes(run, include_ome_zarr=True)
        return Response(
            io.BytesIO(raw_zip).getvalue(),
            mimetype="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="openimpala_hub_bundle_{run_id}.zip"'
            },
        )


def create_dash_app(
    *,
    warm_start: bool = True,
    warm_shape: tuple[int, int, int] = (48, 48, 48),
) -> Dash:
    """Create and configure the Plotly Dash application and its underlying Flask server."""
    registry = RunRegistry(maxsize=48)
    if warm_start:
        registry.warm_populate_defaults(shape=warm_shape)

    app = Dash(
        __name__,
        title="The OpenImpala Hub | BASE Laboratory",
        update_title="Solving in OpenImpala...",
        suppress_callback_exceptions=False,
    )
    app.layout = build_hub_layout(registry)
    register_callbacks(app, registry)
    _register_api_routes(app.server, registry)
    app._hub_registry = registry  # type: ignore[attr-defined]
    return app


_WSGI_SINGLETON: Flask | None = None


def get_wsgi_app() -> Flask:
    """WSGI factory entrypoint for Gunicorn and Modal (`@modal.wsgi_app()`)."""
    global _WSGI_SINGLETON
    if _WSGI_SINGLETON is None:
        dash_app = create_dash_app(warm_start=True)
        _WSGI_SINGLETON = dash_app.server
    return _WSGI_SINGLETON
