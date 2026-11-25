"""Ecosystem layer import boundary test (`[tool.basecoat.layers]`, Layer 4)."""

from __future__ import annotations

from pathlib import Path

from basecoat import check_layer_imports


def test_layer_boundaries() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    violations = check_layer_imports(repo_root)
    assert violations == [], f"Layer import violations: {[v.format(repo_root) for v in violations]}"
