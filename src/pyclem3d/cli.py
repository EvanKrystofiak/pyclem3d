"""Command-line interface. Every GUI action has a headless twin here (plan §5, §8, §12).

pyclem3d doctor | info | pyramid | convert | align | phantom | session | landmarks |
         register | report | export
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("pyclem3d")


def _vs(arg: list[float] | None) -> tuple[float, float, float] | None:
    if arg is None:
        return None
    if len(arg) == 1:
        return (float(arg[0]),) * 3
    if len(arg) != 3:
        raise SystemExit("--voxel-size takes 1 or 3 values (dz dy dx, nm)")
    return (float(arg[0]), float(arg[1]), float(arg[2]))


def _open(
    path: str,
    kind: str,
    args: argparse.Namespace,
    memory: str | None = "skip",
    pyramid: str = "lazy",
):
    from .io.readers import open_volume

    return open_volume(
        path,
        kind=kind,
        voxel_size_nm=_vs(getattr(args, "voxel_size", None)),
        y_scale=float(getattr(args, "y_scale", 1.0) or 1.0),
        memory=memory,
        pyramid=pyramid,
        cache_dir=getattr(args, "cache_dir", None),
    )


# ------------------------------------------------------------------ commands
def cmd_doctor(args: argparse.Namespace) -> int:
    from .doctor import format_doctor, run_doctor

    rep = run_doctor(args.data, write=not args.no_write, config_path=args.config, gpu=args.gpu)
    print(format_doctor(rep))
    if args.json:
        print(json.dumps(rep, indent=1, default=str))
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    vol = _open(args.path, args.kind, args, memory="skip", pyramid="lazy")
    from .io.memory import decide

    d = decide(vol.nbytes)
    info: dict[str, Any] = {
        "source": vol.source,
        "shape_czyx": [int(s) for s in vol.data.shape],
        "dtype": str(vol.dtype),
        "size_gb": vol.nbytes / 1e9,
        "voxel_size_nm": list(vol.voxel_size_nm),
        "voxel_size_source": vol.metadata.get("voxel_size_source"),
        "channels": [c.name for c in vol.channels],
        "pyramid_levels": vol.n_levels(),
        "level_voxel_sizes_nm": [list(vol.level_voxel_size_nm(i)) for i in range(vol.n_levels())],
        "memory_strategy_would_be": d.strategy,
        "memory_reason": d.reason,
        "psf_nm": vol.psf_nm,
        "format": vol.metadata.get("format"),
        "warnings": vol.metadata.get("warnings", []),
        "recommend_zarr": vol.metadata.get("recommend_zarr"),
    }
    if args.json:
        print(json.dumps(info, indent=1, default=str))
    else:
        print(vol.describe())
        for k in (
            "voxel_size_source",
            "format",
            "channels",
            "pyramid_levels",
            "memory_strategy_would_be",
            "memory_reason",
            "psf_nm",
        ):
            print(f"  {k}: {info[k]}")
        for w in info["warnings"]:
            print(f"  warning: {w}")
        if info["recommend_zarr"]:
            print(
                "  recommendation: convert to OME-Zarr (pyclem3d convert) because: "
                + "; ".join(info["recommend_zarr"])
            )
    return 0


def cmd_pyramid(args: argparse.Namespace) -> int:
    from .io.pyramid import ensure_pyramid

    vol = _open(args.path, args.kind, args, memory="skip", pyramid="none")
    vol = ensure_pyramid(vol, cache_dir=args.cache_dir, eager=True, min_size=args.min_size)
    print(
        f"{vol.n_levels()} levels: "
        + ", ".join(
            f"L{i} {list(vol.level_data(i).shape[1:])} @ {[round(v, 1) for v in vol.level_voxel_size_nm(i)]} nm"
            for i in range(vol.n_levels())
        )
    )
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    from .io.writers import write_ome_zarr_from_level0

    vol = _open(args.path, args.kind, args, memory="skip", pyramid="none")
    origin = tuple(float(v) for v in vol.world_affine[:3, 3])
    out = write_ome_zarr_from_level0(
        args.out,
        vol.data,
        vol.voxel_size_nm,
        vol.channels,
        name=Path(args.path).name,
        translation_nm=origin,
        min_size=args.min_size,
    )  # type: ignore[arg-type]
    print(f"wrote {out}")
    return 0


def cmd_align(args: argparse.Namespace) -> int:
    from .align.apply import bake
    from .align.pipeline import align_stack, plot_alignment, verify_stack
    from .align.sidecar import sidecar_path

    vol = _open(args.stack, "em", args, memory="skip", pyramid="lazy")
    offsets = tuple(int(o) for o in args.offsets.split(","))
    kw = dict(
        method=args.method,
        offsets=offsets,
        drift=args.drift,
        highpass_sigma=args.highpass_sigma,
        running_k=args.running_k,
        upsample=args.upsample,
        level=args.level,
        subpixel=args.subpixel,
        gpu=args.gpu,
        per_pair=args.per_pair,
        bad=args.bad,
        channel=args.channel,
        refine_fine=args.refine_fine,
        margin=args.margin,
    )
    if args.verify:
        res = verify_stack(vol, **{k: v for k, v in kw.items() if k not in ("method", "drift")})
    else:
        res = align_stack(vol, **kw)
    print(res.summary())
    if args.verify:
        print(f"verdict: {res.stats['verdict']}")
    sc = Path(args.sidecar) if args.sidecar else sidecar_path(args.stack)
    if not args.verify or args.sidecar:
        res.transforms.save(sc)
        print(f"sidecar: {sc}")
    if args.plot:
        p = plot_alignment(res, args.plot)
        print(f"plot: {p}" if p else "plot skipped (matplotlib not installed)")
    if args.out and not args.verify:
        fmt = args.format or (
            "zarr"
            if str(args.out).endswith(".zarr")
            else "ome-tiff"
            if str(args.out).lower().endswith((".tif", ".tiff"))
            else "mrc"
            if str(args.out).lower().endswith(".mrc")
            else "zarr"
        )
        out = bake(
            vol,
            res.transforms,
            args.out,
            fmt=fmt,
            crop=args.crop,
            subpixel=args.subpixel,
            bad_slices=args.bad_slices,
            min_size=args.min_size,
        )
        print(f"baked: {out}")
    return 0


def cmd_phantom(args: argparse.Namespace) -> int:
    import dask.array as da

    from .io.writers import write_ome_tiff
    from .phantom.generate import make_misaligned_stack, make_phantom
    from .register.landmarks import LandmarkSet, default_sigma_nm
    from .session import Session, VolumeSpec

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    em, lm, truth = make_phantom(
        em_shape=tuple(args.em_shape),
        em_voxel_nm=_vs(args.em_voxel) or (20.0, 20.0, 20.0),
        lm_voxel_nm=_vs(args.lm_voxel) or (300.0, 100.0, 100.0),
        n_nuclei=args.n_nuclei,
        n_organelles=args.n_organelles,
        seed=args.seed,
        noise=args.noise,
    )
    em_path = out / "em.ome.tif"
    lm_path = out / "lm.ome.tif"
    write_ome_tiff(em_path, em.data, em.voxel_size_nm, em.channels)
    write_ome_tiff(lm_path, lm.data, lm.voxel_size_nm, lm.channels)
    rng = np.random.default_rng(args.seed + 1)
    ls = LandmarkSet()
    sig = default_sigma_nm(lm, em)
    for c_em, c_lm in zip(truth.nuclei_em_world, truth.nuclei_lm_world):
        noise = rng.normal(0, 1, 3) * np.asarray(sig) * args.landmark_noise
        ls.add(
            tuple(c_em),
            tuple(c_lm + noise),
            sig,
            feature="nucleus",
            method={"em": "truth", "lm": "truth"},
        )
    truth_d = {
        "lm_to_em": truth.lm_to_em.tolist(),
        "nuclei_em_world_nm": truth.nuclei_em_world.tolist(),
        "nuclei_lm_world_nm": truth.nuclei_lm_world.tolist(),
        "organelles_em_world_nm": truth.organelles_em_world.tolist(),
        "psf_fwhm_nm": list(truth.psf_fwhm_nm),
        "em_voxel_nm": list(truth.em_voxel_nm),
        "lm_voxel_nm": list(truth.lm_voxel_nm),
    }
    if args.misalign:
        stack = np.asarray(em.data[0].compute())
        mis, mt = make_misaligned_stack(
            stack, jitter_px=args.jitter, drift_px=args.drift, n_bad=args.n_bad, seed=args.seed
        )
        write_ome_tiff(
            out / "em_misaligned.ome.tif",
            da.from_array(mis[None], chunks=(1, 4, *mis.shape[1:])),
            em.voxel_size_nm,
            em.channels,
        )
        truth_d["misalignment"] = mt.to_dict()
    (out / "truth.json").write_text(json.dumps(truth_d, indent=1), encoding="utf-8")
    sess = Session(
        em=VolumeSpec("em.ome.tif", "em", voxel_size_nm=em.voxel_size_nm),
        lm=VolumeSpec("lm.ome.tif", "lm", voxel_size_nm=lm.voxel_size_nm, psf_nm=lm.psf_nm),
        landmarks=ls,
        kind=args.kind,
        notes="synthetic phantom with known truth (truth.json)",
    )
    sp = sess.save(out / "session.json")
    print(
        f"phantom written to {out}: em.ome.tif {em.shape_zyx}, lm.ome.tif {lm.shape_zyx}, {len(ls)} landmarks, session {sp.name}"
    )
    return 0


def cmd_session(args: argparse.Namespace) -> int:
    from .session import Session, VolumeSpec

    if args.sub == "init":
        out = Path(args.out)
        base = out.parent

        def rel(p: str) -> str:
            try:
                return str(Path(p).resolve().relative_to(base.resolve()))
            except ValueError:
                return str(Path(p).resolve())

        sess = Session(
            em=VolumeSpec(
                rel(args.em),
                "em",
                voxel_size_nm=_vs(args.em_voxel_size),
                y_scale=args.y_scale,
                align_sidecar=(rel(args.em_align_sidecar) if args.em_align_sidecar else None),
            ),
            lm=VolumeSpec(
                rel(args.lm),
                "lm",
                voxel_size_nm=_vs(args.lm_voxel_size),
                psf_nm=(tuple(args.psf) if args.psf else None),
            ),
            kind=args.kind,
            lam=args.lam,
        )
        sess.save(out)
        print(f"session written: {out}")
        return 0
    if args.sub == "show":
        sess = Session.load(args.session)
        d = sess.to_dict()
        d["landmarks"] = {"n": len(sess.landmarks), "n_enabled": sess.landmarks.n_enabled}
        if d.get("transform") and d["transform"].get("type") == "tps":
            d["transform"] = {
                "type": "tps",
                "n_control": len(d["transform"]["control_pts"]),
                "lam": d["transform"]["lam"],
            }
        print(json.dumps(d, indent=1, default=str))
        return 0
    raise SystemExit("session: use 'init' or 'show'")


def cmd_landmarks(args: argparse.Namespace) -> int:
    from .export.transforms import export_bigwarp_csv, import_bigwarp_csv
    from .session import Session

    sess = Session.load(args.session)
    ls = sess.landmarks
    changed = False
    if args.add:
        v = [float(x) for x in args.add]
        sig = tuple(args.sigma) if args.sigma else _default_sigma(sess)
        lm = ls.add((v[0], v[1], v[2]), (v[3], v[4], v[5]), sig, feature=args.feature or "")
        print(f"added landmark {lm.id}")
        changed = True
    if args.remove is not None:
        ls.remove(args.remove)
        changed = True
    if args.enable is not None:
        ls.get(args.enable).enabled = True
        changed = True
    if args.disable is not None:
        ls.get(args.disable).enabled = False
        changed = True
    if args.import_bigwarp:
        imported = import_bigwarp_csv(
            args.import_bigwarp, unit=args.unit, sigma_nm=_default_sigma(sess)
        )
        for lm in imported.landmarks:
            ls.add(
                lm.em_world_nm,
                lm.lm_world_nm,
                lm.sigma_nm,
                feature=lm.feature,
                enabled=lm.enabled,
                method=lm.method,
            )
        print(f"imported {len(imported)} landmarks from {args.import_bigwarp}")
        changed = True
    if args.export_bigwarp:
        export_bigwarp_csv(ls, args.export_bigwarp, unit=args.unit)
        print(f"exported {len(ls)} landmarks to {args.export_bigwarp}")
    if changed:
        sess.transform = None
        sess.fit = None
        sess.save()
    if args.list or not (changed or args.export_bigwarp):
        print(f"{len(ls)} landmarks ({ls.n_enabled} enabled); spread: {ls.spread_report()}")
        for lm in ls.landmarks:
            flag = "" if lm.enabled else " (disabled)"
            print(
                f"  #{lm.id:3d} EM {np.round(lm.em_world_nm, 1).tolist()} <- LM {np.round(lm.lm_world_nm, 1).tolist()} sigma {np.round(lm.sigma_nm, 1).tolist()} {lm.feature}{flag}"
            )
    return 0


def _default_sigma(sess) -> tuple[float, float, float]:
    from .register.landmarks import default_sigma_nm

    try:
        em, lm = sess.open_volumes(memory="skip", pyramid="none")
        return default_sigma_nm(lm, em)
    except Exception:  # pragma: no cover - volumes missing
        return (300.0, 50.0, 50.0)


def cmd_register(args: argparse.Namespace) -> int:
    from .export.report import build_report, write_report
    from .session import Session

    sess = Session.load(args.session)
    if args.kind:
        sess.kind = args.kind
    if args.lam is not None:
        sess.lam = "auto" if args.lam == "auto" else float(args.lam)
    em, lm = sess.open_volumes(cache_dir=args.cache_dir, memory=args.memory)
    fr = sess.do_fit(with_loo=not args.no_loo)
    print(fr.summary())
    for w in fr.warnings:
        print(f"warning: {w}")
    if sess.kind == "tps":
        dg = sess.ensure_displacement(em)
        if dg is not None:
            print(f"displacement grid: {sess.displacement}")
    rep = build_report(
        em,
        lm,
        sess.landmarks,
        fr,
        session={
            "path": sess.path,
            "kind": sess.kind,
            "lambda": sess.lam,
            "prealign": sess.prealign.to_dict(),
        },
        outputs=sess.outputs,
    )
    rp = args.report or (
        Path(sess.path).with_suffix(".report.json") if sess.path else Path("report.json")
    )
    write_report(rep, rp)
    sess.outputs["report"] = str(rp)
    if args.command == "register":
        sess.save()
        print(f"session updated: {sess.path}")
    print(f"report: {rp}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from .export.emlm import export_em_in_lm
    from .export.fused import export_fused_ome_zarr
    from .export.report import figure_em_slice, figure_lm_slice
    from .export.transforms import export_all_transforms, export_bdv_xml_h5
    from .resample.slab import SlabParams
    from .session import Session

    sess = Session.load(args.session)
    t = sess.transform_obj()
    if t is None:
        raise SystemExit("session has no fitted transform: run 'pyclem3d register' first")
    em, lm = sess.open_volumes(cache_dir=args.cache_dir, memory=args.memory)
    dg = sess.ensure_displacement(em) if sess.kind == "tps" or t.kind == "tps" else None
    done: dict[str, Any] = {}
    if args.fused:
        roi = None
        if args.roi:
            r = [float(v) for v in args.roi]
            roi = np.array([r[:3], r[3:]])
        out = export_fused_ome_zarr(
            em,
            lm,
            t,
            args.fused,
            voxel_size_nm=args.voxel_size_out,
            em_level=args.em_level,
            roi_world=roi,
            displacement=dg,
            min_size=args.min_size,
        )
        done["fused_ome_zarr"] = str(out)
        print(f"fused OME-Zarr: {out}")
        if args.imagej_tif:
            from .io.readers import open_volume as _open_volume
            from .io.writers import write_imagej_tiff

            fused_vol = _open_volume(out, kind="em", memory="skip", pyramid="none")
            ij = write_imagej_tiff(
                args.imagej_tif, fused_vol.data, fused_vol.voxel_size_nm, fused_vol.channels
            )
            done["fused_imagej_tif"] = str(ij)
            print(f"ImageJ hyperstack (drag into Fiji): {ij}")
    if args.transforms:
        w = export_all_transforms(
            t,
            sess.landmarks,
            args.transforms,
            displacement=dg,
            unit=args.unit,
            meta={"session": sess.path, "kind": sess.kind},
        )
        done["transforms"] = w
        print("transforms: " + ", ".join(f"{k}={v}" for k, v in w.items()))
    if args.bdv:
        out = export_bdv_xml_h5(em, lm, t, args.bdv, em_level=args.em_level, unit=args.unit)
        done["bdv_xml"] = str(out)
        print(f"BDV: {out}")
    if args.em_in_lm:
        thickness: str | float = args.thickness
        try:
            thickness = float(args.thickness)
        except ValueError:
            pass
        p = SlabParams(thickness=thickness, projection=args.projection)
        fmt = "ome-zarr" if str(args.em_in_lm).endswith(".zarr") else "ome-tiff"
        out = export_em_in_lm(em, lm, t, args.em_in_lm, fmt=fmt, slab=p)
        done["em_in_lm"] = str(out)
        print(f"EM-in-LM: {out}")
    if args.figures:
        figs = []
        fig_dir = Path(args.figures)
        em_z = [int(v) for v in args.em_z.split(",")] if args.em_z else []
        lm_z = [int(v) for v in args.lm_z.split(",")] if args.lm_z else []
        if not em_z and not lm_z:
            em_z = [em.shape_zyx[0] // 2]
        for j in em_z:
            figs.append(
                figure_em_slice(
                    em,
                    lm,
                    t,
                    j,
                    fig_dir / f"em_z{j:05d}_{args.mode}.png",
                    mode=args.mode,
                    displacement=dg,
                )
            )
        for k in lm_z:
            figs.append(
                figure_lm_slice(
                    em, lm, t, k, fig_dir / f"lm_z{k:04d}_{args.mode}.png", mode=args.mode
                )
            )
        done["figures"] = figs
        for f in figs:
            print(f"figure: {f['path']}  {f['readout']}")
    sess.outputs.update(done)
    sess.save()
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    """Open napari with the pyCLEM-3D panel docked (and a session loaded)."""
    try:
        import napari
    except ImportError as e:
        raise ImportError(
            "the viewer needs napari: uv sync --extra gui  (or pip install 'pyclem3d[gui]')"
        ) from e
    from .napari._widget import PyCLEM3DWidget

    viewer = napari.Viewer(title="pyCLEM-3D")
    widget = PyCLEM3DWidget(viewer)
    viewer.window.add_dock_widget(widget, name="pyCLEM-3D", area="right")
    if args.session:
        widget.session_path.setText(str(Path(args.session).resolve()))
        widget._open_session()
    napari.run()
    return 0


def cmd_segment(args: argparse.Namespace) -> int:
    """Segment an EM stack into a zarr mask with empanada (MitoNet/NucleoNet) or QuantEM."""
    from .seg.quantem import is_quantem_model, segment_volume_quantem

    vol = _open(args.stack, "em", args, memory="ram" if not args.lazy else "lazy", pyramid="none")
    zr = tuple(args.z_range) if args.z_range else None
    backend = args.backend
    if backend == "auto":
        backend = "quantem" if is_quantem_model(args.model) else "empanada"

    def prog(i, n):
        if i % 50 == 0 or i == n:
            print(f"  {i}/{n} slices", flush=True)

    if backend == "quantem":
        out = segment_volume_quantem(
            vol,
            args.out,
            model=args.model,
            semantic=not args.instances,
            channel=args.channel,
            z_range=zr,
            device="cpu" if args.cpu else "auto",
            threshold=args.threshold,
            save_probability=not args.no_probability,
            progress=prog,
        )
    else:
        from .seg.mitonet import segment_volume

        out = segment_volume(
            vol,
            args.out,
            model=args.model,
            inference_scale=args.scale,
            semantic=not args.instances,
            channel=args.channel,
            z_range=zr,
            use_gpu=not args.cpu,
            progress=prog,
        )
    print(f"mask written ({backend}): {out}")
    return 0


def cmd_register_seg(args: argparse.Namespace) -> int:
    """Segmentation-to-image registration: mask -> synthetic fluorescence -> z scan -> affine."""
    from .register.fit import fit_landmarks
    from .register.transforms import AffineTransform
    from .seg.intensity import ncc_on_synthetic, register_affine, z_scan
    from .seg.mitonet import open_mask
    from .seg.quantem import open_probability
    from .seg.synthetic import synthetic_fluorescence
    from .session import Session

    sess = Session.load(args.session)
    em, lm = sess.open_volumes(cache_dir=args.cache_dir, memory="ram")
    mask, _ = open_mask(args.mask)
    soft = None if args.hard else open_probability(args.mask)
    if soft is not None:
        mask = soft  # probability-weighted synthetic fluorescence (QuantEM 'prob' array)
        print("using the segmentation probability map as the synthetic source")
    psf = tuple(args.psf) if args.psf else (lm.psf_nm if lm.psf_nm else (500.0, 200.0))
    zr = tuple(args.z_range) if args.z_range else None
    syn = synthetic_fluorescence(mask, em, target_voxel_nm=args.voxel, psf_fwhm_nm=psf, z_range=zr)  # type: ignore[arg-type]
    coarse = synthetic_fluorescence(
        mask, em, target_voxel_nm=2 * args.voxel, psf_fwhm_nm=psf, z_range=zr
    )  # type: ignore[arg-type]
    t0 = sess.transform_obj()
    if t0 is None:
        if sess.landmarks.n_enabled >= 3:
            t0 = fit_landmarks(sess.landmarks, "similarity", lam=None, with_loo=False).transform
            print("initial transform: similarity from the session landmarks")
        else:
            raise SystemExit(
                "the session needs a transform (register / locate) or >= 3 landmarks to start from"
            )
    else:
        print(f"initial transform: session ({t0.kind})")
    if not t0.is_linear:
        t0 = t0.affine_part()  # type: ignore[attr-defined]
    M = t0.matrix.copy()
    # z sign: try both when asked (coplanar landmarks leave it undetermined)
    cands = []
    if args.z_sign == "auto":
        s_xy = float(np.sqrt(abs(np.linalg.det(M[1:3, 1:3]))))
        for sign in (+1.0, -1.0):
            Mi = M.copy()
            Mi[0, :3] = 0.0
            Mi[0, 0] = sign * s_xy
            cands.append(AffineTransform(Mi))
    else:
        cands.append(AffineTransform(M))
    scored = [(ncc_on_synthetic(coarse, lm, args.channel, t), t) for t in cands]
    for n, t in scored:
        print(f"  candidate z row {np.round(t.matrix[0], 3).tolist()}: NCC {n:.4f}")
    ncc0, t_init = max(scored, key=lambda x: x[0])
    offsets = np.arange(-args.z_range_nm, args.z_range_nm + 1, args.z_step)
    scales = tuple(float(v) for v in args.z_scales.split(","))
    scan = z_scan(coarse, lm, args.channel, t_init, offsets, z_scales=scales)
    best = max(scan, key=lambda r: r["ncc"] if np.isfinite(r["ncc"]) else -1)
    print(
        "z scan at z scale 1.0: "
        + " ".join(f"{r['offset_nm']:+.0f}:{r['ncc']:.3f}" for r in scan if r["z_scale"] == 1.0)
    )
    print(
        f"z scan best: offset {best['offset_nm']:+.0f} nm, z scale {best['z_scale']}, NCC {best['ncc']:.4f}"
    )
    Mb = t_init.matrix.copy()
    Mb[0, :3] *= best["z_scale"]
    Mb[0, 3] += best["offset_nm"]
    res = register_affine(
        syn,
        lm,
        args.channel,
        AffineTransform(Mb),
        metric=args.metric,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        sampling=args.sampling,
    )
    print(res.summary())
    ncc1 = ncc_on_synthetic(syn, lm, args.channel, res.lm_to_em)
    print(f"NCC on synthetic: initial {ncc0:.4f} -> seed {best['ncc']:.4f} -> affine {ncc1:.4f}")
    sess.transform = res.lm_to_em.to_dict()
    sess.kind = "affine"
    sess.fit = None
    sess.displacement = None
    sess.outputs["register_seg"] = {
        "mask": str(args.mask),
        "soft": soft is not None,
        "channel": int(args.channel),
        "ncc": {"initial": ncc0, "seed": best["ncc"], "affine": ncc1},
        "z_scan": scan,
        "metric": [res.metric_before, res.metric_after],
    }
    sess.save()
    print(f"session updated: {sess.path}")
    return 0


# --------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pyclem3d", description="volumetric confocal <-> volume-EM correlation"
    )
    p.add_argument("-v", "--verbose", action="count", default=0)
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="measure the machine and write a config")
    d.add_argument("--data", help="data path to measure read throughput on")
    d.add_argument("--no-write", action="store_true")
    d.add_argument("--config")
    d.add_argument("--gpu", choices=["auto", "on", "off"], default="auto")
    d.add_argument("--json", action="store_true")
    d.set_defaults(func=cmd_doctor)

    def add_open_args(sp, kind_default="em"):
        sp.add_argument("--kind", choices=["em", "lm"], default=kind_default)
        sp.add_argument(
            "--voxel-size",
            nargs="+",
            type=float,
            metavar="NM",
            help="dz dy dx in nm (overrides metadata)",
        )
        sp.add_argument(
            "--y-scale",
            type=float,
            default=1.0,
            help="FIB-SEM tilt factor on y (verify sign/value on real data)",
        )
        sp.add_argument("--cache-dir")

    i = sub.add_parser("info", help="describe a volume without loading it")
    i.add_argument("path")
    add_open_args(i)
    i.add_argument("--json", action="store_true")
    i.set_defaults(func=cmd_info)

    py = sub.add_parser("pyramid", help="build the cached pyramid eagerly")
    py.add_argument("path")
    add_open_args(py)
    py.add_argument("--min-size", type=int, default=64)
    py.set_defaults(func=cmd_pyramid)

    cv = sub.add_parser("convert", help="convert any input to OME-Zarr (with pyramid)")
    cv.add_argument("path")
    cv.add_argument("--out", required=True)
    add_open_args(cv)
    cv.add_argument("--min-size", type=int, default=64)
    cv.set_defaults(func=cmd_convert)

    al = sub.add_parser("align", help="align (or verify) an EM stack slice-to-slice")
    al.add_argument("stack")
    add_open_args(al)
    al.add_argument("--out", help="bake the aligned stack here (.zarr / .ome.tif / .mrc)")
    al.add_argument("--format", choices=["zarr", "ome-tiff", "mrc"])
    al.add_argument("--sidecar", help="where to write align.json (default: next to the stack)")
    al.add_argument("--method", choices=["multi", "chain", "running-mean"], default="multi")
    al.add_argument("--offsets", default="1,2,4")
    al.add_argument("--drift", choices=["keep", "remove"], default="keep")
    al.add_argument(
        "--highpass-sigma", type=float, default=5.0, help="slices; drift slower than this is kept"
    )
    al.add_argument("--running-k", type=int, default=5)
    al.add_argument("--upsample", type=int, default=10)
    al.add_argument("--level", type=int)
    al.add_argument("--subpixel", action="store_true")
    al.add_argument("--gpu", action="store_true")
    al.add_argument("--per-pair", choices=["translation", "rigid", "affine"], default="translation")
    al.add_argument("--bad", choices=["flag", "exclude"], default="flag")
    al.add_argument(
        "--bad-slices",
        choices=["keep", "interpolate", "drop"],
        default="keep",
        help="what to do with excluded slices when baking",
    )
    al.add_argument("--crop", choices=["common", "union", "same"], default="common")
    al.add_argument("--channel", type=int, default=0)
    al.add_argument("--refine-fine", action="store_true")
    al.add_argument("--margin", type=int, default=0)
    al.add_argument("--min-size", type=int, default=64)
    al.add_argument("--plot", help="write trajectory/confidence plot (PNG; needs matplotlib)")
    al.add_argument("--verify", action="store_true", help="only check whether the stack is aligned")
    al.set_defaults(func=cmd_align)

    ph = sub.add_parser(
        "phantom", help="write a synthetic EM+LM pair with known truth and a ready session"
    )
    ph.add_argument("--out", required=True)
    ph.add_argument("--em-shape", nargs=3, type=int, default=[48, 192, 192])
    ph.add_argument("--em-voxel", nargs="+", type=float)
    ph.add_argument("--lm-voxel", nargs="+", type=float)
    ph.add_argument("--n-nuclei", type=int, default=10)
    ph.add_argument("--n-organelles", type=int, default=40)
    ph.add_argument("--seed", type=int, default=0)
    ph.add_argument("--noise", type=float, default=0.05)
    ph.add_argument(
        "--landmark-noise", type=float, default=0.0, help="in units of the default sigma"
    )
    ph.add_argument("--kind", default="affine")
    ph.add_argument("--misalign", action="store_true")
    ph.add_argument("--jitter", type=float, default=2.0)
    ph.add_argument("--drift", type=float, default=10.0)
    ph.add_argument("--n-bad", type=int, default=2)
    ph.set_defaults(func=cmd_phantom)

    se = sub.add_parser("session", help="create or show a session")
    ses = se.add_subparsers(dest="sub", required=True)
    si = ses.add_parser("init")
    si.add_argument("--em", required=True)
    si.add_argument("--lm", required=True)
    si.add_argument("--out", required=True)
    si.add_argument("--em-voxel-size", nargs="+", type=float)
    si.add_argument("--lm-voxel-size", nargs="+", type=float)
    si.add_argument("--y-scale", type=float, default=1.0)
    si.add_argument("--psf", nargs=2, type=float, metavar=("FWHM_Z", "FWHM_XY"))
    si.add_argument("--em-align-sidecar")
    si.add_argument("--kind", default="affine")
    si.add_argument("--lam", default="auto")
    ss = ses.add_parser("show")
    ss.add_argument("session")
    se.set_defaults(func=cmd_session)

    lm = sub.add_parser("landmarks", help="list / add / remove / import / export landmarks")
    lm.add_argument("session")
    lm.add_argument("--list", action="store_true")
    lm.add_argument(
        "--add", nargs=6, metavar=("EMZ", "EMY", "EMX", "LMZ", "LMY", "LMX"), help="world nm"
    )
    lm.add_argument("--sigma", nargs=3, type=float)
    lm.add_argument("--feature")
    lm.add_argument("--remove", type=int)
    lm.add_argument("--enable", type=int)
    lm.add_argument("--disable", type=int)
    lm.add_argument("--import-bigwarp")
    lm.add_argument("--export-bigwarp")
    lm.add_argument("--unit", default="um")
    lm.set_defaults(func=cmd_landmarks)

    for name, help_ in (
        ("register", "fit the transform and save it into the session"),
        ("report", "reproduce the fit headlessly and write report.json"),
    ):
        r = sub.add_parser(name, help=help_)
        r.add_argument("session")
        r.add_argument("--kind", choices=["rigid", "similarity", "affine", "tps"])
        r.add_argument("--lambda", dest="lam")
        r.add_argument("--report")
        r.add_argument("--no-loo", action="store_true")
        r.add_argument("--cache-dir")
        r.add_argument("--memory", choices=["ram", "lazy"])
        r.set_defaults(func=cmd_register)

    ex = sub.add_parser("export", help="write the registered multimodal outputs")
    ex.add_argument("session")
    ex.add_argument("--fused", help="fused OME-Zarr path")
    ex.add_argument("--voxel-size-out", type=float, help="nm; picks the EM level nearest to it")
    ex.add_argument("--em-level", type=int)
    ex.add_argument(
        "--roi", nargs=6, type=float, metavar="NM", help="z0 y0 x0 z1 y1 x1 in EM world nm"
    )
    ex.add_argument(
        "--imagej-tif",
        help="also write the fused stack as an ImageJ hyperstack (.tif, < 4 GB) for plain Fiji File > Open",
    )
    ex.add_argument("--transforms", help="directory for JSON / ITK / BigWarp / NRRD exports")
    ex.add_argument("--bdv", help="BigDataViewer XML path (H5 next to it)")
    ex.add_argument("--em-in-lm", help="EM-in-LM-space OME-TIFF (.ome.tif) or OME-Zarr (.zarr)")
    ex.add_argument("--thickness", default="slice", help="slice | psf | nm")
    ex.add_argument("--projection", choices=["mean", "min", "max", "gaussian"], default="mean")
    ex.add_argument("--figures", help="directory for PNG overlays")
    ex.add_argument("--em-z", help="comma-separated EM slice indices")
    ex.add_argument("--lm-z", help="comma-separated LM slice indices")
    ex.add_argument("--mode", choices=["blend", "checker", "swipe", "em", "lm"], default="blend")
    ex.add_argument("--unit", default="um")
    ex.add_argument("--min-size", type=int, default=64)
    ex.add_argument("--cache-dir")
    ex.add_argument("--memory", choices=["ram", "lazy"])
    ex.set_defaults(func=cmd_export)

    sg = sub.add_parser(
        "segment",
        help="segment an EM stack with empanada (mito | nucleus | MitoNet_v1) or QuantEM (quantem/mito, omniem/nucleus, ...)",
    )
    sg.add_argument("stack")
    add_open_args(sg)
    sg.add_argument("--out", required=True, help="zarr mask path")
    sg.add_argument(
        "--model",
        default="quantem/mito",
        help="quantem/mito (default), omniem/nucleus|er|ld, or an empanada name (mito, nucleus, MitoNet_v1)",
    )
    sg.add_argument(
        "--scale", type=int, default=2, help="inference scale (2 = 16 nm for 8 nm data)"
    )
    sg.add_argument(
        "--instances", action="store_true", help="2D instance labels instead of a semantic mask"
    )
    sg.add_argument("--channel", type=int, default=0)
    sg.add_argument("--z-range", nargs=2, type=int, metavar=("Z0", "Z1"))
    sg.add_argument("--lazy", action="store_true", help="do not load the stack into RAM")
    sg.add_argument("--cpu", action="store_true")
    sg.add_argument(
        "--backend",
        choices=["auto", "empanada", "quantem"],
        default="auto",
        help="auto: QuantEM for 'quantem/...' or 'omniem/...' model ids, empanada otherwise",
    )
    sg.add_argument("--threshold", type=float, help="QuantEM foreground probability threshold")
    sg.add_argument(
        "--save-probability", action="store_true", help="QuantEM: also store the probability map"
    )
    sg.set_defaults(func=cmd_segment)

    rs = sub.add_parser(
        "register-seg",
        help="register a segmentation-derived synthetic volume to an LM channel (affine)",
    )
    rs.add_argument("session")
    rs.add_argument("--mask", required=True, help="zarr mask from 'segment'")
    rs.add_argument(
        "--channel",
        type=int,
        required=True,
        help="LM channel index that labels the segmented organelle",
    )
    rs.add_argument("--voxel", type=float, default=32.0, help="synthetic voxel size (nm)")
    rs.add_argument("--psf", nargs=2, type=float, metavar=("FWHM_Z", "FWHM_XY"))
    rs.add_argument(
        "--z-range", nargs=2, type=int, metavar=("Z0", "Z1"), help="EM slices to consider"
    )
    rs.add_argument("--z-sign", choices=["auto", "keep"], default="auto")
    rs.add_argument("--z-range-nm", type=float, default=2400.0)
    rs.add_argument("--z-step", type=float, default=150.0)
    rs.add_argument("--z-scales", default="0.8,1.0,1.2")
    rs.add_argument("--metric", choices=["correlation", "mi"], default="correlation")
    rs.add_argument("--iterations", type=int, default=250)
    rs.add_argument("--learning-rate", type=float, default=0.5)
    rs.add_argument("--sampling", type=float, default=0.3)
    rs.add_argument(
        "--hard",
        action="store_true",
        help="use the binary mask even if a probability map is stored",
    )
    rs.add_argument("--cache-dir")
    rs.set_defaults(func=cmd_register_seg)

    g = sub.add_parser(
        "gui", help="open napari with the pyCLEM-3D panel (optionally with a session)"
    )
    g.add_argument("session", nargs="?")
    g.set_defaults(func=cmd_gui)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    try:
        return int(args.func(args) or 0)
    except (ValueError, FileNotFoundError, ImportError) as e:
        print(f"error: {e}", file=sys.stderr)
        if args.verbose:
            raise
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
