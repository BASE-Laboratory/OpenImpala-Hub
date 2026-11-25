# `PorousDiff` & Diamond Light Source `I12` Gallery (`data/porousdiff/`)

Bundled by default (`90 KB` total, RFC 1950 `zlib` OME-Zarr v0.4):
- `dls_i12_felt_068kpa.ome.zarr` — Segmented virgin carbon felt at `68 kPa` (`0.008 mV/V`, `64 × 96 × 96` @ `3.2 µm/px`, topology-preserving `2×` max-pooled from `1.6 µm/px` Diamond Light Source `I12` synchrotron tomography).
- `dls_i12_felt_238kpa.ome.zarr` — Segmented virgin carbon felt at `238 kPa` (`0.028 mV/V`, `64 × 96 × 96` @ `3.2 µm/px`).
- `dls_i12_felt_451kpa.ome.zarr` — Segmented virgin carbon felt at `451 kPa` (`0.053 mV/V`, `64 × 96 × 96` @ `3.2 µm/px`).

You can drop additional calibrated OME-Zarr v0.4 stores (`.ome.zarr`) exported from `BASE-Laboratory/PorousDiff` (`feature/paper-ready-conditional` on Isambard-AI GH200) into this directory (or point `$OPENIMPALA_HUB_POROUSDIFF_DIR` at an external volume).

## Automatic Discovery

1. **Filename convention:** Any directory named `*_150kpa.ome.zarr` (or `.tif` / `.npy`) is discovered automatically by `openimpala_hub.engine.generators.list_porousdiff_gallery()`, parsing the clamping pressure in `kPa` and recording `data_origin="porousdiff"` in exported manifests.
2. **Optional `gallery_manifest.json`:** For explicit multi-material metadata, add a `gallery_manifest.json` list:

```json
[
  {
    "id": "cuhcf_50kpa_seed0",
    "label": "PorousDiff CuHCF Felt (50 kPa)",
    "clamping_kpa": 50.0,
    "path": "cuhcf_50kpa_seed0.ome.zarr"
  }
]
```
