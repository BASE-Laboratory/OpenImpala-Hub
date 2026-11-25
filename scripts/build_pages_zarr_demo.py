"""Build compact OME-Zarr v0.4 4D X-ray demo stores + cycling CSV for GitHub Pages (`docs_site/data/`).

Uses `basecoat.write_ome_zarr` to generate standard OME-NGFF v0.4 multiscale metadata (`.zattrs`,
`.zgroup`, `0/`, `1/`), then re-encodes the chunk payloads with standard RFC 1950 `zlib` (`{"id": "zlib"}`)
so both Python (`basecoat.read_ome_zarr`, `openimpala.read_zarr_volume`) and zero-dependency browsers
(`DecompressionStream("deflate")`) can stream individual chunks over HTTP Range / `fetch()`.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import zlib

from basecoat import Volume, open_volume, read_ome_zarr, write_ome_zarr
import numpy as np

from openimpala_hub.engine.chunked_io import load_chunked_volume
from openimpala_hub.engine.operando_demo import ensure_operando_demo_staged


def _reencode_level_zlib(level_dir: Path, arr: np.ndarray, chunks: tuple[int, int, int]) -> None:
    """Write standard Zarr v2 `zlib`-compressed `/` chunks matching `level_dir/.zarray`."""
    zarray_path = level_dir / ".zarray"
    meta = json.loads(zarray_path.read_text(encoding="utf-8"))
    # Remove existing blosc chunk directories
    for child in list(level_dir.iterdir()):
        if child.name != ".zarray" and child.name != ".zattrs":
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    cz, cy, cx = chunks
    nz, ny, nx = arr.shape
    meta["chunks"] = [cz, cy, cx]
    meta["compressor"] = {"id": "zlib", "level": 6}
    meta["dimension_separator"] = "/"
    meta["order"] = "C"
    meta["dtype"] = "|u1"
    meta["shape"] = [nz, ny, nx]
    zarray_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    n_cz = (nz + cz - 1) // cz
    n_cy = (ny + cy - 1) // cy
    n_cx = (nx + cx - 1) // cx
    for iz in range(n_cz):
        for iy in range(n_cy):
            for ix in range(n_cx):
                sub = arr[
                    iz * cz : min(nz, (iz + 1) * cz),
                    iy * cy : min(ny, (iy + 1) * cy),
                    ix * cx : min(nx, (ix + 1) * cx),
                ]
                buf = np.zeros((cz, cy, cx), dtype=np.uint8, order="C")
                buf[: sub.shape[0], : sub.shape[1], : sub.shape[2]] = sub
                chunk_file = level_dir / str(iz) / str(iy) / str(ix)
                chunk_file.parent.mkdir(parents=True, exist_ok=True)
                chunk_file.write_bytes(zlib.compress(buf.tobytes(order="C"), level=6))


def build_static_pages_data(out_root: Path | None = None) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = out_root or (repo_root / "docs_site" / "data")
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    demo = ensure_operando_demo_staged()
    shutil.copy2(demo.echem_csv, data_dir / "ag05_cycling.csv")

    selected = [
        ("t0_0h.ome.zarr", demo.timepoints[0]),
        ("t1_7p5h.ome.zarr", demo.timepoints[1]),
        ("t2_15h.ome.zarr", demo.timepoints[2]),
        ("t3_20h.ome.zarr", demo.timepoints[3]),
    ]
    manifest_entries = []
    for store_name, tp in selected:
        # Pool 64x160x160 -> 32x80x80 (3.2 um isotropic) for crisp static chunk streaming
        pooled = load_chunked_volume(str(tp.path), z_step=2, xy_step=2, scheduler="synchronous")
        nz, ny, nx = pooled["shape"]
        arr_u8 = np.asarray(pooled["value"], dtype=np.uint8).reshape(nz, ny, nx)
        vol = Volume.from_array(
            arr_u8,
            voxel_size_um=(3.2, 3.2, 3.2),
            provenance={"time_h": tp.t_h, "label": tp.label},
        )
        store_path = data_dir / store_name
        write_ome_zarr(vol, store_path, multiscale_levels=2, chunks=(8, 80, 80), overwrite=True)

        # Read back level 0 and level 1 via basecoat before re-encoding as RFC 1950 zlib
        with open_volume(store_path) as lv0:
            arr0 = np.asarray(lv0.read_region((slice(None), slice(None), slice(None))).data, dtype=np.uint8)
        arr1 = arr0[::2, ::2, ::2]
        _reencode_level_zlib(store_path / "0", arr0, (8, 80, 80))
        _reencode_level_zlib(store_path / "1", arr1, (8, 40, 40))

        # Verify basecoat.read_ome_zarr round-trips the zlib OME-Zarr store cleanly!
        rt = read_ome_zarr(store_path)
        assert rt.shape == arr0.shape
        assert np.array_equal(np.asarray(rt.data), arr0)

        manifest_entries.append(
            {
                "store": store_name,
                "label": tp.label,
                "time_h": tp.t_h,
                "shape_zyx": list(arr0.shape),
                "chunks_zyx": [8, 80, 80],
                "voxel_um": 3.2,
            }
        )

    (data_dir / "manifest.json").write_text(
        json.dumps({"dataset": demo.label, "timepoints": manifest_entries}, indent=2) + "\n",
        encoding="utf-8",
    )
    return data_dir


if __name__ == "__main__":
    out = build_static_pages_data()
    total_kb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1024.0
    print(f"Built static OME-Zarr v0.4 stores in {out} ({total_kb:.1f} KB total)")
