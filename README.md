# pyclem3d

Volumetric confocal ↔ volume-EM correlation (successor to pyCLEM 0.6): take a confocal
z-stack and a FIB-SEM / SBF-SEM stack, optionally align the EM stack slice-to-slice, register
the two volumes with endogenous landmarks, verify the registration honestly, and write the
**registered multimodal dataset** in formats other tools open.

The design is in [`pyclem3d_plan_v2.md`](pyclem3d_plan_v2.md); this README documents what the
code does today. Python ≥ 3.11.

## Install

```bash
uv sync --extra dev --extra export          # headless core + tests + BDV/h5 export
uv sync --extra gui                          # + napari (PyQt6) plugin
uv sync --extra readers                      # + .lif / .czi / .nd2 via bioio
uv sync --extra gpu                          # + cupy (GPU phase correlation)
```

or `pip install -e ".[dev,export,gui]"`.

## Status against the plan

| Phase | Deliverable | State |
|---|---|---|
| 1 Data layer | `Volume` model, TIFF / TIFF-directory / MRC / OME-Zarr readers (lazy, no conversion), bioio readers (optional), OME-Zarr / OME-TIFF / MRC writers, anisotropy-aware cached pyramid, per-volume memory strategy, `doctor` | done |
| 2 EM alignment | phase correlation with confidence, multi-neighbour / chain / running-mean trajectories, drift keep/remove, bad-slice detection, lazy apply or bake, xz/yz reslice, trajectory plot, CLI | done |
| 3 Registration core | weighted rigid / similarity / affine / regularized TPS in 3D, LOO, per-axis RMS, round trip, systematic z check, pre-align remap, session v2, `register` / `report` CLI, phantom suite | done |
| 4 Viewer | napari plugin: reader + workflow dock widget (Load, Align EM, Pre-align, Locate, Landmarks with midpoint / centroid / paired / snap refinement, Fit, Verify with EM-driven and LM-driven z modes, blend / swipe / checker / flicker, z readout, z profile, Export) | done, tested with hidden windows |
| 5 Z tools + deformable + outputs | slab projections, PSF-aware sampling, displacement-grid inverse, deformable overlay, fused OME-Zarr, transform exports (JSON / ITK / BigWarp / BDV XML+H5 / NRRD), EM-in-LM, report + figures | done |
| 6 Automation | nucleus-centroid matching, intensity refinement, extra readers | not started (optional) |

Everything in phases 1–5 is exercised by `pytest` on synthetic phantoms with known truth
(`tests/`), including a 67 GB virtual stack that must never be loaded in full.

## Quick start (headless)

```bash
pyclem3d doctor --data /path/to/data          # RAM, cores, GPU, throughput -> ~/.pyclem3d/config.json
pyclem3d info em.mrc --voxel-size 8 8 8       # describe without loading (MRC headers are checked, not trusted)
pyclem3d align em.mrc --out em_aligned.zarr --drift keep --plot align.png
pyclem3d session init --em em_aligned.zarr --lm confocal.lif --out session.json --kind affine
pyclem3d landmarks session.json --add EMZ EMY EMX LMZ LMY LMX --feature nucleus   # world nm
pyclem3d register session.json                # fits, writes session.report.json
pyclem3d export session.json --fused fused.zarr --transforms tf/ --em-in-lm em_in_lm.ome.tif --figures figs/
```

`pyclem3d phantom --out demo --misalign` writes a synthetic EM + confocal pair with known
truth and a ready session, which is the fastest way to try the whole pipeline.

## napari

```bash
napari                                        # Plugins -> pyCLEM-3D
```

Drop any supported file on napari to open it lazily in nanometres. The dock widget walks the
plan's workflow: the main viewer is the EM (reference grid, never resampled) with the LM
overlaid through its layer affine; a second viewer shows the raw LM for picking landmarks,
with a synced cursor and z. Deformable fits show a lazily computed overlay.

## Mitochondria segmentation (empanada) in the same napari

`uv sync --extra gui --extra seg` installs [empanada-napari](https://github.com/volume-em/empanada-napari)
(MitoNet) next to the pyCLEM-3D panel, with PyTorch from the CUDA 12.8 wheel index so the GPU is used.
MitoNet weights are cached in `~/.empanada` on first use (about 220 MB from Zenodo; pre-download with
`torch.hub.download_url_to_file` if Zenodo is slow). On 8 nm FIB-SEM, "3D Inference" with inference
scale 2 (16 nm) gives the most contiguous mitochondria at a quarter of the cost; one 1462 × 1142 slice
takes about a second on an RTX 4070.

Segmentation-driven registration (plan Phase 6) is implemented in `pyclem3d.seg`:

```bash
pyclem3d segment FIBVOLUME.tif --out seg/mito.zarr --model mito          # MitoNet, ~0.05 s/slice on a GPU
pyclem3d register-seg session.json --mask seg/mito.zarr --channel 1       # channel 1 = MitoTracker
pyclem3d export session.json --fused correlated.zarr
```

`register-seg` blurs the mask with the confocal PSF into a synthetic fluorescence volume, tries both
z directions, scans the depth offset and z scale (the printed NCC curve shows whether depth is
determined at all), then optimises a 3D affine with SimpleITK using the synthetic volume as the
fixed image. It needs a starting transform in the session (a landmark fit, even a single-plane one,
or `locate`). On the example FIB-SEM / Airyscan pair it reproduced the manual single-slice overlay
and held across the whole confocal slab; the phantom test recovers a 540 nm perturbation to 40 nm.
A B-spline stage (`register_bspline`) exists for local deformation but is not wired into the CLI yet.

## Conventions

* Units are **nanometres**; arrays are `(C, Z, Y, X)`; every volume has a 4×4
  voxel→world affine that carries voxel size, origin and the FIB-SEM y tilt factor.
* Transforms map **LM world → EM world**. Exports give both directions and the xyz
  convention where a tool needs it (ITK, BigWarp, BDV).
* The EM is never resampled by registration. The alignment module moves EM slices by
  integer pixels by default (lossless); sub-pixel and per-slice rigid/affine are opt-in.
* Alignment defaults to `--drift keep`: slow motion is real structure passing through z
  and is left alone; only jitter is corrected. `--drift remove` straightens everything.
* TPS regularization: `reg_i = λ σ_i² / σ̄` per point, λ chosen by leave-one-out.

## Layout

```
src/pyclem3d/
  io/        Volume, readers, writers, pyramid, memory strategy, metadata
  align/     pairwise phase correlation, trajectories, bad slices, apply/bake, pipeline
  register/  transforms, estimators, landmarks, pre-align, QC, fit
  refine/    midpoint, local 3D centroid, paired centroids, xy snap
  resample/  grids, lazy per-chunk resampling, displacement grid, slab projections, z views
  export/    fused OME-Zarr, transform files, EM-in-LM, report, figures
  phantom/   synthetic data with known truth
  napari/    plugin manifest, reader, workflow widget
  session.py, cli.py, doctor.py, config.py
```

## Tests

```bash
pytest                       # core (~2 min)
pytest tests/test_napari_plugin.py   # GUI, hidden windows (Windows) or QT_QPA_PLATFORM=offscreen (Linux)
```

## Still to confirm on real data (plan §13)

FIB-SEM y-scaling state (sign/value of the tilt factor), MRC header voxel sizes, which
endogenous features are visible in both modalities, and whether BDV-XML or OME-Zarr should
be first in the export list.
