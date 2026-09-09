# pyCLEM-3D — Plan v2 (volumetric confocal ↔ volume-EM correlation)

Working name: `pyclem3d`, successor to pyCLEM 0.6. Python ≥ 3.11, no legacy pins.

**Goal (as clarified):** take a confocal z-stack (any confocal; Stellaris or Zeiss
980 most likely; tif most likely) and a volume-EM stack (FIB-SEM or SBF-SEM; tif or
mrc), optionally align the EM stack slice-to-slice, register the two volumes using
endogenous landmarks, verify the registration honestly, and **output the registered
multimodal data together** in formats other tools can open. No lamella targeting.
Hardware ranges from a laptop to a workstation with GPU and NVMe, so every
heavy path adapts to what is available.

---

## 1. Decisions fixed by your answers

| Question | Decision |
|---|---|
| vEM formats | **TIFF (stack of files or one multi-page/BigTIFF) and MRC** are first-class. Others (HDF5/BDV, N5, Zarr, dm3/dm4) via optional readers later. |
| EM stack alignment | **In scope.** Assume stacks are usually aligned; ship an alignment module (§4) that can align a raw stack or *check* and touch up an aligned one. |
| Confocal formats | tifffile handles plain TIFF stacks, OME-TIFF, ImageJ hyperstacks. `.lif` (Stellaris), `.czi` (Zeiss 980), `.nd2` via `bioio` plugins / `readlif` / `pylibczirw`, all optional. |
| PSF defaults | Not hard-coded to one objective. Derive lateral/axial FWHM from metadata (NA, emission λ, immersion RI) when present; otherwise user-entered; fallback σ = voxel size. Used only as landmark-uncertainty defaults and for the PSF-aware z view. |
| Fiducials | **Endogenous landmarks only** (nuclei, nucleoli, mitochondria, lipid droplets, cell boundaries, vessels). Refinement tools are centroid-of-segmentation, not Gaussian bead fits. Automation (Phase 6) becomes nucleus-centroid matching, not bead detection. |
| Downstream | **Co-registered multimodal output**: one OME-Zarr with EM + warped LM channels in a shared frame at a chosen resolution, plus transform files so tools that accept affines (BigDataViewer, napari) can show the *unresampled* data registered. |
| Hardware | Flexible: memory strategy auto-detected per volume, OME-Zarr conversion optional, pyramid cached on demand, chunk sizes tuned by storage, CPU by default, GPU/cluster opt-in. |

---

## 2. What carries over from pyCLEM

Keep: headless core + thin GUI split (core never imports Qt); one importer that
returns a normalized model so the core never sees a file format; **the EM is the
reference grid and is not resampled by registration**; the landmark workflow
(coarse pre-align → pick pairs → fit → QC → overlay); rigid/similarity/affine/TPS;
RMS + leave-one-out RMS + residual arrows; blend/swipe/checker/flicker compare
modes; the per-channel model (true intensity vs display window); session JSON that
reproduces headlessly through a CLI; remapping landmarks through a pre-align change.

Replace: whole-image `skimage.warp` → lazy per-region resampling; MIP at import →
the confocal stays a volume; 2D skimage estimators → own numpy 3D estimators;
bespoke Qt canvases → napari plugin; Python 3.9 pins → ≥ 3.11.

One deliberate exception to "EM never resampled": the alignment module (§4) does
move EM slices, but defaults to **integer-pixel shifts** (lossless) and keeps
sub-pixel / rigid / affine per-slice corrections opt-in.

---

## 3. Physical realities that drive the design

Typical scales (all confirmable from metadata at load time):

| Modality | xy | z | Example volume | Voxels | Size |
|---|---|---|---|---|---|
| Confocal | 80–200 nm | 300–500 nm | 200×200×50 µm, 3 ch | ~2×10⁹ | ~4 GB (16-bit) |
| FIB-SEM | 5–10 nm | 5–10 nm | 50×50×30 µm | ~1.5×10¹¹ | ~150 GB |
| SBF-SEM | 5–15 nm | 40–100 nm | 200×200×100 µm | 10¹¹–10¹² | 0.1–1 TB |

1. **Asymmetry.** The confocal often fits in RAM; the EM rarely does. Design for
   "EM always lazy, confocal lazy-or-loaded depending on the machine" (§5).
2. **Z ratio.** One confocal slice spans ~30–60 FIB-SEM or ~5–10 SBF-SEM slices,
   and the confocal axial PSF (~0.6–1 µm FWHM) makes each slice a weighted slab.
   This is a first-class concept (§7).
3. **The block face is an arbitrary plane** through the confocal volume, so the
   registration must be 3D and the correlation an oblique resample, never a
   slice-index lookup.
4. **Deformation is anisotropic**: dehydration/embedding shrinkage (10–30 %),
   confocal refractive-index z-scaling, SBF-SEM knife compression in z. → **3D
   affine is the default model**; regularized 3D TPS for local nonlinearity.
5. **Endogenous landmarks are big and fuzzy.** A nucleus is 5–10 µm; its "position"
   is a centroid, not a point. Accuracy comes from consistent centroiding on both
   sides and from using enough landmarks (aim for 10–30 well spread in 3D), not
   from sub-voxel picking.

---

## 4. EM stack alignment module (`pyclem3d.align`)

Purpose: turn a raw or roughly aligned tif/mrc stack into a well-aligned one, or
verify an already-aligned one, without leaving the tool.

**Method (coarse-to-fine, translation first):**

1. Work at a pyramid level where a slice is ~1–2k px; estimate pairwise
   (i → i+1) translation with phase cross-correlation (upsampled for sub-pixel
   estimate; record the correlation peak as a confidence score).
2. Optional per-pair rigid / affine (SBF-SEM sometimes shows small rotation or
   scale changes from charging) via feature matching (skimage ORB/SIFT + RANSAC)
   or by fitting to phase-correlation on tiles.
3. **Drift handling**, the part naïve chaining gets wrong: cumulative pairwise
   shifts random-walk and also "straighten" real structure. Options exposed to
   the user: (a) register to a running mean of the previous *k* slices; (b) chain
   the pairwise shifts, then high-pass the cumulative trajectory so slow drift is
   kept as real geometry and only jitter is corrected; (c) multi-neighbour
   consensus (each slice matched to i±1, i±2, i±4; least-squares solve for the
   trajectory) — robust to a single bad slice.
4. **Bad-slice detection**: low correlation or outlier shift (knife chatter,
   charging, focus loss) → flag, interpolate the transform across it, and list
   in the report; optional slice exclusion or replacement by neighbour average.
5. Refine at a finer level only within the tiles/regions the coarse pass says
   need it (cheap on FIB-SEM, meaningful on big SBF-SEM fields).

**Application:** the result is a per-slice 2D transform stored as a sidecar
(`align.json`) next to the source. Two ways to use it:

- *Lazy*: the `Volume` applies the per-slice shift while reading (integer shifts
  are pure slicing/padding, no interpolation), so nothing is written.
- *Bake*: write an aligned OME-Zarr (or tif/mrc) cropped to the common area, with
  the pyramid built at the same time. Recommended when the stack is used often.

**QC views:** shift-trajectory plot (x/y vs slice, before/after detrending),
per-pair correlation plot with flagged slices, and **xz / yz reslice views** in
the viewer — misalignment is obvious as jagged edges in an orthogonal slice, and
these are also the views you use to sanity-check a "probably aligned" stack.

FIB-SEM specifics: an input flag for whether the stack is already y-scaled for the
imaging tilt; if not, the known factor goes into the EM `world_affine`, and its
sign/value is verified against a real capture, never derived (pyCLEM's rule).

This module is independent of the registration core and can be used alone from
the CLI (`pyclem3d align stack.mrc --out aligned.zarr`).

---

## 5. Data layer, coordinates, and hardware flexibility

### Volume model

```python
@dataclass
class Volume:
    data: dask.array              # (C, Z, Y, X) or (Z, Y, X); lazy, chunked
    pyramid: list[dask.array]     # level 0 = full res; built or cached on demand
    voxel_size_nm: (dz, dy, dx)
    world_affine: 4x4             # voxel (z,y,x,1) -> world nm; carries tilt/flip corrections
    per_slice_transforms: ...     # optional, from the align module
    channels: list[Channel]       # pyCLEM's Channel model
    kind: "em" | "lm"
    source: str; metadata: dict; psf_nm: (fwhm_z, fwhm_xy) | None
```

- Canonical unit **nanometres**, canonical axis order **(z, y, x)** internally;
  (x, y, z) only at import/export boundaries, converted explicitly.
- **Import without conversion is always possible.** tifffile (`aszarr`) gives
  chunked lazy access to TIFF stacks and BigTIFF; `mrcfile` memory-maps MRC; a
  directory of single-slice TIFFs is wrapped as a lazy dask stack. OME-Zarr
  conversion is a *recommendation* the tool makes when it detects (network share,
  compressed/striped TIFF with poor random access, repeated use), not a
  prerequisite.
- **Pyramid on demand.** If the source has no pyramid, levels are computed
  lazily at first use and cached in a user-configurable cache dir (or built
  eagerly with `pyclem3d pyramid`). Downsampling is anisotropy-aware: xy-only
  until voxels are ~isotropic, then all axes — so "the EM level that matches the
  confocal xy resolution" is a lookup.
- **Memory strategy is decided per volume at load**, not globally: if a
  volume (or the subset needed) is under a configurable fraction of free RAM
  (default ~30 %), it is loaded into memory for speed; otherwise it stays lazy.
  The confocal typically loads; the EM typically doesn't. The choice is shown
  in the UI and overridable.
- **Compute only what is viewed/exported.** Resample, projection and composite
  are dask graphs evaluated on the requested slice/ROI/level. Never materialize
  the warped confocal at EM resolution for a whole volume (that is ~10¹¹ voxels).
- **`pyclem3d doctor`**: measures RAM, cores, GPU presence, and read throughput
  of the data path, then writes a config (chunk sizes, cache location, dask
  scheduler: threads / processes / `distributed` LocalCluster, GPU on/off).
  Everything has a CPU path; `cupy` resampling and GPU-accelerated
  phase-correlation are drop-ins when a GPU exists.

### Landmarks and session

```python
@dataclass
class Landmark3D:
    em_world_nm: (z, y, x); lm_world_nm: (z, y, x)     # LM side is post-pre-align
    em_voxel, lm_voxel                                  # provenance/display
    sigma_nm: (sz, sy, sx)                              # per-point uncertainty
    method: {"em": "click"|"midpoint"|"centroid", "lm": ...}
    feature: str            # "nucleus", "mito", ... (free text, for the report)
    enabled: bool
```

Session JSON v2: volume paths + voxel sizes + axes + alignment sidecar,
`PreAlign3D` (axis permutation, flips, coarse rotation), landmarks in world nm,
transform kind, TPS λ, fitted 4×4 / TPS coefficients, display settings. A session
reproduces headlessly through the CLI, bit-for-bit, as in pyCLEM.

---

## 6. Registration core

All transforms map **LM world nm → EM world nm** (moving → fixed).

| kind | dof | min pairs | estimator |
|---|---|---|---|
| rigid | 6 | 3 | weighted Kabsch (SVD) |
| similarity | 7 | 3 | weighted Umeyama |
| affine | 12 | 4 non-coplanar | weighted least squares |
| tps 3D | ∞ | 5 (use ≥ 10) | kernel U(r)=r, smoothing λ |

- **Weighted fit.** Each landmark's `sigma_nm` (defaults: ~half the LM xy
  voxel for xy, ~half the axial PSF FWHM for z; centroid-refined points get the
  centroid's std error) weights the residuals per axis. Confocal z is treated as
  what it is: the least certain coordinate.
- **Regularized TPS** (λ·diag(σ²) added to the kernel block) so the deformable
  model doesn't chase z noise or nuclear-centroid jitter. LOO-RMS picks λ.
- **Inverse mapping** (EM → LM) is exact for linear models; for TPS, fit the
  reverse spline and report the round-trip error, or invert numerically on the
  coarse displacement grid (§7).
- **Coarse "locate" stage** before landmarks: the EM block is usually a small
  sub-region of the confocal field. Show the EM low-res xy projection beside
  the confocal overview; 2–3 rough pairs give a similarity that puts the viewer
  in the right neighbourhood.
- **Pre-align 3D**: axis permutation + flips + coarse rotation about z; existing
  landmarks are remapped, as pyCLEM's `prealign_map` did.

### Landmark refinement for endogenous features (`pyclem3d.refine`)

Point-click on a 5 µm nucleus is a poor landmark; these tools make it a good one:

- **Top/bottom midpoint**: mark the first and last slice where the feature is
  visible (in either modality) → z centre, with σ_z from the extent.
- **Local 3D centroid**: draw a box (or click and grow a small crop); threshold
  (Otsu on the crop, EM contrast inverted as needed) + largest connected
  component → intensity-weighted centroid in world nm. Works for nuclei,
  nucleoli, lipid droplets, mitochondria on both sides. Cheap because the crop
  is small even at full EM resolution.
- **Paired centroids**: run the same segmentation-centroid on both sides of a
  pair in one action so the same definition of "centre" is used in EM and LM
  (this consistency matters more than sub-voxel precision).
- **Local xy snap** (after a coarse fit): normalized cross-correlation of a
  small LM crop against the EM slab projection to nudge xy.
- Every refinement shows what it did (crop, mask, centroid) and can be rejected.

### Automation (Phase 6, optional)

Nucleus-centroid matching: segment nuclei in the LM (DAPI/Hoechst; classical
threshold or a pretrained model) and in a low-res EM level (threshold-based or a
small model), then match centroid clouds under the affine model with RANSAC,
seeded by the coarse locate transform. Intensity-based refinement (mutual
information via SimpleITK / itk-elastix) initialized from the landmark fit. Both
sit behind a "refine" interface; the manual path never depends on them.

---

## 7. The Z problem: many EM slices per confocal slice

Handled in four places:

**a) Transform.** Voxel sizes live in the world affines, so a 300 nm / 8 nm ratio
is just scale; registration is in nm.

**b) Landmarks.** Sub-slice z on both sides via midpoint/centroid tools (§6), with
per-axis σ feeding the weighted fit.

**c) Views.** Two explicit modes, always with a readout such as
`LM z 12/48 ↔ EM z 480–521 (42 slices, centre 500)`:

- *EM-driven* (default): scrolling EM z shows the confocal resampled at that exact
  world plane — trilinear in z, so it varies smoothly across the ~40 EM slices
  instead of jumping. Free for linear transforms (napari layer affine). Optional
  *PSF-aware* weighting of LM slices by the axial PSF.
- *LM-driven*: for confocal slice k, an **EM slab projection** — the EM slices
  whose world z falls in the confocal slab (slice thickness or PSF FWHM,
  user-chosen), projected by mean (default), min, or Gaussian-weighted mean,
  resampled into the LM grid. Computed from the EM pyramid level matching the LM
  xy resolution, so a 40-slice slab over a 6000² field is ~10⁷ voxels, not 10⁹.
  Cached per slice.

**d) QC.** RMS split into xy and z (nm); 3D residual vectors so a systematic z
error (wrong z scale, wrong tilt factor) shows as a pattern; a **z-profile check**
that plots LM intensity along z at any mapped EM feature.

### Lazy resampling mechanics

Linear: no resampling for display; exports use `scipy.ndimage.map_coordinates`
per dask chunk on the output grid. Deformable: evaluate the displacement field on
a coarse world grid (0.5–1 µm, cached in the session), trilinearly interpolate for
any chunk, then `map_coordinates`. Coverage masks (pyCLEM's alpha) are computed
the same way.

---

## 8. Viewer: napari plugin

napari already provides multiscale lazy rendering of dask/zarr, per-layer affines
applied on the GPU, nD slicing + 3D rendering, Points/Vectors/Shapes layers and a
dock-widget system; reimplementing that in raw Qt would dominate the project. The
core stays viewer-agnostic.

Workflow panels: **Load** (memory strategy shown; per-channel controls) →
**Align EM** (optional; trajectory/correlation plots; xz/yz reslice) →
**Pre-align** → **Locate** → **Landmarks** (two linked viewers with synced world
cursor + overlay; refinement tools; per-point residual table) → **Fit** (kind, λ,
live RMS xy/z, LOO, residual vectors) → **Verify** (blend/swipe/checker/flicker on
the current slice; EM-driven / LM-driven z mode; z-profile check) → **Export**.

Check at kickoff: maturity of napari multi-canvas in the pinned version (fallback:
two `napari.Viewer` windows with a cursor/camera sync), and refresh performance of
the computed deformable-overlay layer on slice change.

---

## 9. Outputs: the registered multimodal dataset

Ordered by priority, since this is the point of the tool:

1. **Fused OME-Zarr** at a user-chosen voxel size (default: the EM pyramid level
   nearest the LM xy resolution; full EM resolution available for cropped ROIs):
   channels = EM + each LM channel warped into EM world space + coverage mask,
   one shared coordinate frame, multiscale pyramid, OME metadata with channel
   names/colours and physical sizes. Written chunk-by-chunk, so it works on any
   hardware. Opens in napari, Fiji (via MoBIE/ome-zarr), neuroglancer, webKnossos.
2. **Transform-only export** (no resampling, no data loss): the 4×4 (LM world →
   EM world) as JSON, ITK `.tfm`, BigWarp landmark CSV, and a **BigDataViewer
   XML+H5 / XML+N5** where the LM source carries the affine — BDV then displays
   the original LM registered to the EM. For TPS, the coarse displacement field
   as a Zarr/NRRD so ITK/elastix tools can apply it.
3. **EM-in-LM-space**: the EM slab-projected and resampled into the confocal
   grid (small, useful for light-microscopy-side analysis and for figures),
   as OME-TIFF or OME-Zarr.
4. **Cropped full-res ROI fusions** as OME-TIFF for Fiji/Imaris users.
5. `session.json`, `report.json` (n, RMS xy/z, LOO, per-point residuals, λ,
   round-trip error, alignment stats, warnings), and figure slices (PNG/TIFF
   overlays for chosen EM z or oblique planes, with scale bars).

Targets/handoff (pyCLEM's elymaker path) become an optional plugin point, not a
core feature.

---

## 10. Stack

- Python ≥ 3.11 (pin the minor to what napari's current stable supports at
  kickoff — verify then rather than assume).
- numpy 2.x, scipy, scikit-image (phase correlation, ORB/SIFT, morphology),
  dask[array], zarr + ome-zarr, tifffile, mrcfile; optional readers: bioio +
  bioio-lif / bioio-czi / bioio-nd2 / bioio-ome-tiff, readlif, pylibczirw, h5py.
- napari + magicgui (npe2 plugin), qtpy (PyQt6 or PySide6).
- Optional: SimpleITK / itk-elastix, cupy, dask.distributed, psutil (doctor).
- `pyproject.toml`, `uv`, pytest, ruff, pre-commit; CI runs core tests on
  synthetic data and GUI tests offscreen.

---

## 11. Testing (offline, synthetic, as pyCLEM did)

- **Phantom generator**: a synthetic block with nuclei-like blobs and smaller
  organelle-like features at EM resolution; the "confocal" is derived by a known
  affine (anisotropic scale + tilt), anisotropic PSF blur, downsampling, noise.
  A second generator produces a *misaligned* EM stack (per-slice jitter + slow
  drift + a few bad slices) with known truth for the align module.
- Tests: estimators recover truth to sub-voxel; regularized TPS beats λ=0 under z
  noise (LOO); centroid refinement beats integer clicks; slab projection equals
  brute force on a small case; chunked resample equals whole-array resample;
  alignment recovers jitter while preserving injected slow drift when asked to;
  bad slices are flagged; pre-align remap pins points; session round-trips; CLI
  reproduces GUI; fused OME-Zarr has correct sizes/metadata and opens in napari.
- **Large-data smoke test**: a chunked zarr larger than RAM; assert that view,
  slab projection and export never trigger a full load (track peak memory), on
  both the "load into RAM" and "lazy" strategies.

---

## 12. Roadmap

**Phase 0 (1 week).** Grab one FIB-SEM and one SBF-SEM tif/mrc stack plus their
confocal stacks; record voxel sizes, tilt-correction state, which endogenous
features are visible in both. Run `doctor` on the target machines.

**Phase 1 — Data layer.** `Volume`, tif/mrc/lif/czi/OME importers, lazy access
without conversion, optional OME-Zarr conversion, on-demand cached pyramid,
memory strategy, `doctor`. Deliverable: any stack opens lazily in the plugin
on a laptop and a workstation.

**Phase 2 — EM alignment.** Phase-correlation translation, drift options,
bad-slice detection, lazy/bake application, trajectory plots, reslice QC, CLI.
Deliverable: align a raw stack and verify an "already aligned" one.

**Phase 3 — Registration core (headless).** 3D weighted estimators, regularized
TPS, LOO, session v2, CLI `register` / `report`, phantom suite.

**Phase 4 — Viewer MVP.** napari plugin: load, pre-align, locate, landmarks with
midpoint/centroid refinement, fit, overlay via layer affine, EM-driven z view,
residual vectors, transform export. Deliverable: full manual correlation on real
data with an affine model.

**Phase 5 — Z tools + deformable + outputs.** LM-driven slab view, PSF-aware
interpolation, displacement-field cache, deformable overlay, compare modes,
z-profile QC, **fused OME-Zarr / BDV / EM-in-LM exports**.

**Phase 6 — Automation (optional).** Nucleus-centroid matching, intensity-based
refinement, GPU paths, extra readers (BDV/N5, dm3/dm4).

Phases 1–4 give a usable tool; 5 delivers the actual product (the registered
multimodal dataset) with the z-handling that makes it accurate.

---

## 13. Remaining assumptions to confirm on real data

- Whether FIB-SEM stacks arrive y-scaled for tilt (sign/value verified, not
  derived).
- Whether MRC headers carry correct voxel sizes (often they don't; the UI must
  let the user override and the report must record what was used).
- Which endogenous features are reliably visible in *both* modalities for your
  samples — this sets which refinement tool is the default.
- Whether downstream users want BDV-XML (Fiji-centric labs) or OME-Zarr
  (napari/web-centric) first; both are planned, order can flip.
