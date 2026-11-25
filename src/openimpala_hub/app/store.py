"""Thread-safe bounded server-side registry for solved 3D runs (`RunRegistry`).

Keeps multi-megabyte 3D arrays (`phase_volume`, `potential_field`, `flux_field`)
in server memory keyed by `run_id`, so browser `dcc.Store` state stays tiny (< 4 KB).
"""

from __future__ import annotations

from collections import OrderedDict
import threading

from ..engine.solver import ExecutedRun, run_transport_pipeline
from ..presets import DEFAULT_SWEEP_LOADS_KPA, PRESETS, make_carbon_felt_spec


class RunRegistry:
    """Bounded LRU store of `ExecutedRun` instances safe across Gunicorn threads."""

    def __init__(self, maxsize: int = 48) -> None:
        self._maxsize = max(8, int(maxsize))
        self._lock = threading.Lock()
        self._runs: OrderedDict[str, ExecutedRun] = OrderedDict()
        self.boot_run_ids: list[str] = []
        self.default_run_id: str | None = None

    def put(self, run: ExecutedRun) -> str:
        rid = run.summary.run_id
        with self._lock:
            self._runs[rid] = run
            self._runs.move_to_end(rid)
            while len(self._runs) > self._maxsize:
                oldest_key, _ = next(iter(self._runs.items()))
                if oldest_key in self.boot_run_ids and len(self._runs) > len(self.boot_run_ids):
                    # Keep boot runs pinned when evicting user runs
                    self._runs.move_to_end(oldest_key)
                    oldest_key, _ = next(iter(self._runs.items()))
                self._runs.pop(oldest_key, None)
        return rid

    def get(self, run_id: str | None) -> ExecutedRun | None:
        if not run_id:
            return None
        with self._lock:
            run = self._runs.get(run_id)
            if run is not None:
                self._runs.move_to_end(run_id)
            return run

    def get_or_rehydrate(
        self,
        run_id: str | None,
        history: list[dict] | None = None,
    ) -> ExecutedRun | None:
        """Fetch `run_id` from memory, or re-execute from `history` specs if evicted from LRU."""
        existing = self.get(run_id)
        if existing is not None or not run_id or not history:
            return existing
        from ..contracts import MicrostructureSpec, SolveSpec

        for item in history:
            if item.get("run_id") == run_id and item.get("microstructure_spec"):
                mspec = MicrostructureSpec.from_dict(item["microstructure_spec"])
                sspec = SolveSpec.from_dict(item.get("solve_spec") or {})
                rebuilt = run_transport_pipeline(mspec, sspec)
                self.put(rebuilt)
                return rebuilt
        return None

    def list_summaries(self) -> list[dict]:
        with self._lock:
            return [r.summary.to_dict() for r in self._runs.values()]

    def warm_populate_defaults(self, *, shape: tuple[int, int, int] = (48, 48, 48)) -> None:
        """Pre-solve the initial 3-point Carbon Felt compression sweep so page load is instant."""
        if self.boot_run_ids:
            return
        base_solve = PRESETS["carbon_felt_compression"].solve
        ids: list[str] = []
        default_id: str | None = None
        for kpa in DEFAULT_SWEEP_LOADS_KPA:
            mspec = make_carbon_felt_spec(kpa, shape=shape, seed=42)
            executed = run_transport_pipeline(mspec, base_solve)
            rid = self.put(executed)
            ids.append(rid)
            if abs(kpa - 150.0) < 1.0:
                default_id = rid
        self.boot_run_ids = ids
        self.default_run_id = default_id or ids[len(ids) // 2]
