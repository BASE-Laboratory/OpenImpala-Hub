"""Interactive Plotly Dash web application and REST API for The OpenImpala Hub."""

from __future__ import annotations

from .server import create_dash_app, get_wsgi_app

__all__ = ["create_dash_app", "get_wsgi_app"]
