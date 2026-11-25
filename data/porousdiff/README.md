# Drop-In `PorousDiff` Gallery Slot (`data/porousdiff/`)

Place calibrated OME-Zarr v0.4 stores (`.ome.zarr`) exported from `BASE-Laboratory/PorousDiff` (`feature/paper-ready-conditional` on Isambard-AI GH200) in this directory (or point `$OPENIMPALA_HUB_POROUSDIFF_DIR` at an external volume).

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
