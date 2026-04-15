"""
Command-line interface for the NERVE dataset toolkit.

Usage::

    nerve list [--split SPLIT] [--min-duration N] [--min-persons N]
               [--categories C ...] [--sensors S ...] [--max-size SIZE]
               [--format {table,json}] [--export PATH]

    nerve download [SESSION ...] [--split SPLIT] [--from-file PATH]
                   [--data-root PATH] [--min-persons N] ...

    nerve generate [--split SPLIT] [--from-file PATH] [--data-root PATH]
                   [--template NAME] [--dest PATH]

    nerve train [--config PATH] [...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

def _add_filter_args(parser: argparse.ArgumentParser) -> None:
    """Add shared session filter flags to a subcommand parser."""
    grp = parser.add_argument_group("session filters")
    grp.add_argument("--split", choices=["train", "val", "test"])
    grp.add_argument("--from-file", metavar="PATH",
                     help="Session list file (one name per line)")
    grp.add_argument("--min-duration", type=float, metavar="SEC")
    grp.add_argument("--max-duration", type=float, metavar="SEC")
    grp.add_argument("--min-persons", type=int, metavar="N")
    grp.add_argument("--categories", nargs="+", metavar="CAT")
    grp.add_argument("--sensors", nargs="+", metavar="SENSOR")
    grp.add_argument("--groups", nargs="+", metavar="GROUP")
    grp.add_argument("--max-size", metavar="SIZE",
                     help="Max archive size, e.g. 2G, 500M")


def _parse_size(size_str: str | None) -> float | None:
    """Parse human-readable size like '2G' or '500M' to GB."""
    if size_str is None:
        return None
    size_str = size_str.strip().upper()
    if size_str.endswith("G"):
        return float(size_str[:-1])
    if size_str.endswith("GB"):
        return float(size_str[:-2])
    if size_str.endswith("M"):
        return float(size_str[:-1]) / 1000
    if size_str.endswith("MB"):
        return float(size_str[:-2]) / 1000
    if size_str.endswith("T"):
        return float(size_str[:-1]) * 1000
    return float(size_str)


def _resolve_sessions(args) -> list[str] | None:
    """Resolve session names from CLI args. Returns None to use filter_kwargs."""
    if hasattr(args, "sessions") and args.sessions:
        return args.sessions

    if args.from_file:
        from nerve.session_list import read_session_list
        return read_session_list(args.from_file)

    return None


def _build_filter_kwargs(args) -> dict:
    """Build kwargs dict for registry.filter_sessions from CLI args."""
    kwargs: dict = {}
    if args.split:
        kwargs["split"] = args.split
    if args.min_duration is not None:
        kwargs["min_duration"] = args.min_duration
    if args.max_duration is not None:
        kwargs["max_duration"] = args.max_duration
    if args.min_persons is not None:
        kwargs["min_persons"] = args.min_persons
    if args.categories:
        kwargs["categories"] = args.categories
    if args.sensors:
        kwargs["sensors"] = args.sensors
    if args.groups:
        kwargs["groups"] = args.groups

    max_gb = _parse_size(args.max_size)
    if max_gb is not None:
        kwargs["max_size_gb"] = max_gb

    names = _resolve_sessions(args)
    if names is not None:
        kwargs["names"] = names

    return kwargs


def cmd_list(args) -> None:
    from nerve.registry import filter_sessions, total_size

    kwargs = _build_filter_kwargs(args)
    sessions = filter_sessions(data_root=args.data_root, **kwargs)

    if args.format == "json":
        data = [
            {
                "name": s.name,
                "split": s.split,
                "group": s.group,
                "size_gb": round(s.size_gb, 2),
                "duration_s": s.duration_seconds,
            }
            for s in sessions
        ]
        print(json.dumps(data, indent=2))
        return

    gb = total_size(sessions)
    print(f"\n{'Name':<28} {'Split':<7} {'Group':<14} {'Size':>8} {'Duration':>10}")
    print("-" * 72)
    for s in sessions:
        dur = f"{s.duration_seconds:.1f}s" if s.duration_seconds else "?"
        size = f"{s.size_gb:.2f}GB"
        print(f"{s.name:<28} {s.split:<7} {s.group:<14} {size:>8} {dur:>10}")
    print("-" * 72)
    print(f"Total: {len(sessions)} sessions, {gb:.1f} GB\n")

    if args.export:
        from nerve.session_list import write_session_list
        names = [s.name for s in sessions]
        write_session_list(args.export, names,
                          header=f"NERVE query result — {len(names)} sessions")
        print(f"Exported to {args.export}")


def cmd_download(args) -> None:
    from nerve import remote

    data_root = args.data_root

    if args.utils:
        remote.download_utils(data_root=data_root)
        return

    names = _resolve_sessions(args)
    if names is not None:
        remote.download_sessions(names, data_root=data_root)
    else:
        kwargs = _build_filter_kwargs(args)
        remote.download_filtered(
            data_root=data_root, dry_run=args.dry_run, **kwargs
        )


def _resolve_template(template_name: str) -> str:
    """Resolve a template name to the absolute path of its .template.json file."""
    from importlib.resources import files

    if os.path.isfile(template_name):
        return os.path.abspath(template_name)

    base = files("nerve.generation.templates")
    candidate = base.joinpath(f"{template_name}.template.json")
    if candidate.is_file():
        return str(candidate)

    available = [
        p.name.replace(".template.json", "")
        for p in base.iterdir()
        if hasattr(p, "name") and p.name.endswith(".template.json")
    ]
    raise FileNotFoundError(
        f"Template '{template_name}' not found.\n"
        f"Available templates: {', '.join(sorted(available))}\n"
        f"Or pass a direct path to a .json file."
    )


def _resolve_template_settings(template_path: str) -> list:
    """Load a template JSON and resolve the $NERVE_MAPPINGS sentinel."""
    import json as _json
    from importlib.resources import files

    mappings_res = files("nerve.data.mappings")
    # importlib may return a MultiplexedPath; resolve via a known child file
    sample = mappings_res.joinpath("rgb_to_davis.json")
    mappings_dir = str(sample).rsplit(os.sep, 1)[0]

    with open(template_path) as f:
        settings = _json.load(f)

    for entry in settings:
        for key, value in entry.items():
            if isinstance(value, str) and "$NERVE_MAPPINGS" in value:
                entry[key] = value.replace("$NERVE_MAPPINGS", mappings_dir)

    return settings


def cmd_generate(args) -> None:
    import json as _json
    import os
    import gc
    from pathlib import Path

    from nerve.config import get_data_root

    data_root = get_data_root(args.data_root)
    template_path = _resolve_template(args.template)
    settings = _resolve_template_settings(template_path)
    dest = os.path.abspath(args.dest)
    split_label = args.split_label or (args.split if args.split else "")
    verbose = getattr(args, "verbose", False)
    clean = getattr(args, "clean", False)
    add = getattr(args, "add", False)

    names = _resolve_sessions(args)
    if names is None:
        from nerve.registry import filter_sessions
        kwargs = _build_filter_kwargs(args)
        sessions = filter_sessions(data_root=args.data_root, **kwargs)
        names = [s.name for s in sessions]

    session_paths = [str(data_root / n) for n in names]

    missing = [p for p in session_paths if not os.path.isdir(p)]
    if missing:
        print(f"Warning: {len(missing)} session(s) not found locally:")
        for m in missing[:5]:
            print(f"  {m}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")
        session_paths = [p for p in session_paths if os.path.isdir(p)]
        if not session_paths:
            print("No sessions available. Download them first.")
            sys.exit(1)

    print(f"Template:  {args.template} ({template_path})")
    print(f"Data root: {data_root}")
    print(f"Output:    {dest}")
    if split_label:
        print(f"Split:     {split_label}")
    print(f"Sessions:  {len(session_paths)}")

    from nerve.generation.creator import (
        extract_from_single_session,
        generate_data_yaml,
        create_view_samples,
        get_collected_samples,
        reset_collected_samples,
        LabelWriter,
    )
    from nerve.extraction.utils.dataset_utils import ResolveClassNamesToIds

    reset_collected_samples()

    if split_label:
        result_dir = os.path.join(dest, split_label)
    else:
        result_dir = dest

    filter_class_ids = None
    for s in settings:
        if "filter_classes" in s:
            filter_class_ids = ResolveClassNamesToIds(s["filter_classes"])
            break
        if "only_classes" in s:
            filter_class_ids = s["only_classes"]
            break

    clip_mode = settings[0].get("clip_mode", "single_frame")
    clip_length = settings[0].get("clip_length", 1)
    clip_stride = settings[0].get("clip_stride", 1)
    output_format = settings[0].get("output_format", "reyolov8")
    is_rvt = output_format == "rvt"

    if clean and os.path.isdir(result_dir):
        import shutil
        shutil.rmtree(result_dir)

    if os.path.isdir(result_dir) and not add:
        print(f"Output directory exists: {result_dir}")
        print("Use --clean to override or --add to append.")
        sys.exit(1)

    label_writers: dict = {}
    if is_rvt:
        annotations_dir = os.path.join(result_dir, "_rvt_annotations_tmp")
        for s in settings:
            name = s["data"]
            out_path = os.path.join(annotations_dir, name + ".json")
            label_writers[name] = LabelWriter(out_path, filter_class_ids=filter_class_ids)
        os.makedirs(result_dir, exist_ok=True)
        data_path = result_dir
        if add:
            existing = [d for d in os.listdir(result_dir) if d.startswith("sequence_")]
            current_index = len(existing)
        else:
            current_index = 0
    else:
        annotations_dir = os.path.join(result_dir, "annotations")
        data_path = os.path.join(result_dir, "data")
        for s in settings:
            name = s["data"]
            out_path = os.path.join(annotations_dir, name + ".json")
            label_writers[name] = LabelWriter(out_path, filter_class_ids=filter_class_ids)

        fused_modality_name = None
        for s in settings:
            if s.get("fuse_dvs_radar_png", False):
                dvs_name = s["data"]
                fused_modality_name = s.get("fused_modality_name", f"{dvs_name}_radar")
                fused_path = os.path.join(annotations_dir, fused_modality_name + ".json")
                label_writers[fused_modality_name] = LabelWriter(fused_path, filter_class_ids=filter_class_ids)
                break

        if not add:
            os.makedirs(result_dir, exist_ok=True)
            os.makedirs(annotations_dir, exist_ok=True)
            os.makedirs(data_path, exist_ok=True)
            current_index = 0
        else:
            current_index = list(label_writers.values())[0].get_last_image_index() + 1

    from tqdm import tqdm

    for idx, session_path in enumerate(tqdm(session_paths, desc="Sessions")):
        current_index = extract_from_single_session(
            session_path, settings, label_writers, data_path,
            current_index, clip_mode=clip_mode,
            clip_length=clip_length, clip_stride=clip_stride,
            verbose=verbose, filter_class_ids=filter_class_ids,
        )
        if not is_rvt:
            for w in label_writers.values():
                w.write_file()
        gc.collect()

    if not is_rvt:
        for w in label_writers.values():
            w.write_file()
    else:
        import shutil
        if os.path.exists(annotations_dir):
            shutil.rmtree(annotations_dir, ignore_errors=True)

    if split_label:
        generate_data_yaml(dest, settings, verbose=verbose)

    collected = get_collected_samples()
    if collected:
        create_view_samples(result_dir, collected, output_format, verbose=verbose)

    print(f"\nDataset generation complete: {result_dir}")


def cmd_enrich(args) -> None:
    from nerve.metadata import enrich_from_local

    data_root = args.data_root
    count = enrich_from_local(data_root=data_root, verbose=True)
    print(f"\nCached metadata for {count} sessions.")
    if count == 0:
        print("No locally extracted sessions found. Download some sessions first.")


def cmd_precompute_radar_cache(args) -> None:
    from pathlib import Path

    from nerve.radar import available_backends
    from nerve.radar.cache import build_cache
    from nerve.radar.cached_backend import CACHE_FILENAME

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    radar_subdir = args.radar_subdir

    if args.path:
        targets = [Path(p).expanduser().resolve() for p in args.path]
    else:
        from nerve.config import get_data_root

        names = _resolve_sessions(args)
        if names is None:
            from nerve.registry import filter_sessions
            kwargs = _build_filter_kwargs(args)
            sessions = filter_sessions(data_root=args.data_root, **kwargs)
            names = [s.name for s in sessions]

        if not names:
            print(
                "No sessions selected. Pass session names as positional "
                "arguments, --from-file, or use --split / --sensors filters.",
                file=sys.stderr,
            )
            sys.exit(1)

        data_root = get_data_root(args.data_root)
        targets = [Path(data_root) / n / radar_subdir for n in names]

    backends = available_backends()
    print(f"Available radar backends: {backends}")
    if args.backend:
        print(f"Using backend: {args.backend}")
    else:
        from nerve.radar.cached_backend import CachedBackend
        from nerve.radar import _REGISTRY

        sources = [n for n, c in _REGISTRY.items() if c is not CachedBackend]
        if not sources:
            print(
                "Error: no source backend (e.g. pycore) is available; "
                "the cached backend alone cannot generate a cache. "
                "Install the proprietary DSP library or implement a "
                "custom RadarBackend, then re-run.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Using backend (auto): {sources[0]}")

    successes: list[Path] = []
    skipped: list[Path] = []
    failures: list[tuple[Path, str]] = []

    for target in targets:
        if not target.is_dir():
            print(f"  ! Skip {target}: not a directory")
            skipped.append(target)
            continue

        cache_path = target / CACHE_FILENAME
        if cache_path.is_file() and not args.force:
            print(f"  - Skip {target}: cache already exists "
                  f"(pass --force to overwrite)")
            skipped.append(target)
            continue

        print(f"\n{'=' * 70}\nPrecomputing cache for {target}\n{'=' * 70}")

        bar = None
        last_total = [0]

        def progress(current, total):
            if tqdm is not None:
                if bar is None or last_total[0] != total:
                    pass
            else:
                if current == total or current == 1 or current % 50 == 0:
                    print(f"  frame {current}/{total}")

        if tqdm is not None:
            with tqdm(total=0, unit="frame",
                      desc=str(target.name)) as bar:
                def progress(current, total):
                    if bar.total != total:
                        bar.total = total
                        bar.refresh()
                    bar.n = current
                    bar.refresh()

                try:
                    out = build_cache(
                        target,
                        backend_name=args.backend,
                        include_range_doppler=not args.no_fft,
                        force=args.force,
                        progress_callback=progress,
                    )
                    successes.append(out)
                    bar.close()
                    print(f"  -> {out} "
                          f"({out.stat().st_size / (1024 * 1024):.1f} MB)")
                except Exception as exc:  # noqa: BLE001
                    bar.close()
                    print(f"  ! Failed: {exc}")
                    failures.append((target, str(exc)))
        else:
            try:
                out = build_cache(
                    target,
                    backend_name=args.backend,
                    include_range_doppler=not args.no_fft,
                    force=args.force,
                    progress_callback=progress,
                )
                successes.append(out)
                print(f"  -> {out} "
                      f"({out.stat().st_size / (1024 * 1024):.1f} MB)")
            except Exception as exc:  # noqa: BLE001
                print(f"  ! Failed: {exc}")
                failures.append((target, str(exc)))

    print(f"\n{'=' * 70}")
    print(f"Done. {len(successes)} cache(s) written, "
          f"{len(skipped)} skipped, {len(failures)} failed.")
    if failures:
        print("Failures:")
        for path, msg in failures:
            print(f"  - {path}: {msg}")
        sys.exit(1)


def cmd_reconstruct(args) -> None:
    import subprocess
    from pathlib import Path
    from nerve.config import get_data_root

    recon_dir = Path(__file__).parent / "extraction" / "reconstruction"
    script = recon_dir / "run_reconstruction.py"
    default_weights = recon_dir / "weights" / "E2VID_lightweight.pth.tar"

    weights = args.weights or str(default_weights)
    if not Path(weights).exists():
        print(f"Error: model weights not found at {weights}", file=sys.stderr)
        print("Download E2VID_lightweight.pth.tar and place it in "
              f"{recon_dir / 'weights' / ''}", file=sys.stderr)
        sys.exit(1)

    if args.input:
        inputs = [args.input]
        outputs = [args.output_dir or str(Path(args.input).parent)]
    elif args.session:
        data_root = get_data_root(args.data_root)
        session_dir = data_root / args.session
        if not session_dir.is_dir():
            for sub in ("train", "val", "test"):
                candidate = data_root / sub / args.session
                if candidate.is_dir():
                    session_dir = candidate
                    break
            else:
                print(f"Session not found: {args.session}", file=sys.stderr)
                sys.exit(1)

        sensors = args.sensors or ["davis", "prophesee"]
        inputs = []
        outputs = []
        for sensor in sensors:
            events = session_dir / sensor / "events.hdf5"
            if events.exists():
                inputs.append(str(events))
                outputs.append(str(session_dir / sensor))
            else:
                print(f"  Skipping {sensor}/ (no events.hdf5)")
    else:
        print("Error: provide --session or --input.", file=sys.stderr)
        sys.exit(1)

    for inp, out in zip(inputs, outputs):
        print(f"  Reconstructing {inp} -> {out}/ ...")
        cmd = [
            sys.executable, str(script),
            "-i", inp,
            "-o", out,
            "-c", weights,
            "-t", args.output_type,
            "--fps", str(args.fps),
        ]
        if args.use_gpu:
            cmd.append("--use_gpu")
        subprocess.run(cmd, cwd=str(recon_dir), check=True)
        print(f"    -> {out}/reconstruction.mp4")
        print(f"    -> {out}/events.mp4")


def cmd_visualize(args) -> None:
    from pathlib import Path
    from nerve.config import get_data_root
    from nerve.extraction.plot.visualize_annotations import (
        render_annotations,
        visualize_session_sensor,
    )

    if args.input:
        if args.output is None:
            print("Error: at least one -o OUTPUT_PATH MODE pair is required "
                  "when using --input.", file=sys.stderr)
            sys.exit(1)
        render_annotations(
            args.input,
            [(p, m) for p, m in args.output],
            background=args.background or "",
            background_delay_ms=args.background_delay,
            from_ms=args.from_ms,
            to_ms=args.to_ms,
        )
        return

    if not args.session:
        print("Error: provide --session (or --input for raw mode).",
              file=sys.stderr)
        sys.exit(1)

    data_root = get_data_root(args.data_root)
    session_dir = data_root / args.session
    if not session_dir.is_dir():
        for sub in ("train", "val", "test"):
            candidate = data_root / sub / args.session
            if candidate.is_dir():
                session_dir = candidate
                break
        else:
            print(f"Session not found: {args.session}", file=sys.stderr)
            sys.exit(1)

    sensors = args.sensors or ["rgb", "davis", "prophesee"]
    modes = args.modes or ["only_base", "only_human_pose", "only_human_seg", "all"]

    for sensor in sensors:
        sensor_dir = session_dir / sensor
        ann_file = sensor_dir / "annotations" / "annotations.json"
        if not ann_file.exists():
            print(f"  Skipping {sensor}/ (no annotations)")
            continue
        print(f"  Rendering {sensor}/ ...")
        created = visualize_session_sensor(
            sensor_dir,
            modes=modes,
            background=args.background or "",
            background_delay_ms=args.background_delay,
            from_ms=args.from_ms,
            to_ms=args.to_ms,
        )
        for p in created:
            print(f"    -> {p}")


def cmd_train(args) -> None:
    from nerve.training.train import (
        import_exp_from_file,
        train_yolox,
        train_yolov8,
        train_reyolov8,
        train_rvt,
    )

    if args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    elif args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    print("=" * 70)
    print("NERVE Training Runner")
    print("=" * 70)
    print(f"\nLoading experiment from: {args.config}")

    Exp = import_exp_from_file(args.config)
    exp = Exp()
    print("\n" + str(exp))

    config = exp.to_dict()
    model_type = config.get("model_type", "unknown")

    # Forward CLI overrides into the config dict
    if args.batch_size is not None:
        config["batch"] = args.batch_size
        config["batch_size"] = args.batch_size
    if args.epochs is not None:
        config["epochs"] = args.epochs
        config["max_epochs"] = args.epochs
    if args.device:
        config["device"] = args.device
    if args.resume:
        config["resume"] = args.resume
    if args.experiment_name:
        config["name"] = args.experiment_name
    if args.fp16:
        config["fp16"] = True
        config["amp"] = True
    if args.verbose:
        config["verbose"] = True

    # Build a lightweight namespace that the trainer functions expect as `args`
    train_args = argparse.Namespace(
        batch_size=args.batch_size,
        epochs=args.epochs,
        imgsz=None,
        workers=None,
        lr=None,
        device=args.device or "",
        resume=args.resume,
        experiment_name=args.experiment_name,
        cache=False,
        verbose=args.verbose,
        fp16=args.fp16,
        wandb_project=args.wandb_project,
        no_wandb=args.no_wandb,
    )

    print(f"\nModel Type: {model_type.upper()}")
    print("=" * 70)

    try:
        if model_type == "yolox":
            result_dir = train_yolox(config, train_args)
        elif model_type == "yolov8":
            result_dir = train_yolov8(config, train_args)
        elif model_type == "reyolov8":
            result_dir = train_reyolov8(config, train_args)
        elif model_type == "rvt":
            result_dir = train_rvt(config, train_args)
        else:
            print(f"Unknown model type: {model_type}")
            sys.exit(1)

        print("\n" + "=" * 70)
        print("Training completed successfully!")
        if result_dir:
            print(f"Results saved to: {result_dir}")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n" + "=" * 70)
        print("Training interrupted by user")
        print("=" * 70)


def main(argv: list[str] | None = None) -> None:
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--data-root", metavar="PATH",
        help="Override data root directory (default: $NERVE_DATA_ROOT or ~/.nerve/data/)",
    )

    parser = argparse.ArgumentParser(
        prog="nerve",
        description="NERVE dataset toolkit — download, explore, process, train",
        parents=[shared],
    )

    subs = parser.add_subparsers(dest="command")

    # --- list ---
    p_list = subs.add_parser("list", help="List and filter sessions",
                             parents=[shared])
    _add_filter_args(p_list)
    p_list.add_argument("--format", choices=["table", "json"], default="table")
    p_list.add_argument("--export", metavar="PATH",
                        help="Export matching session names to a .txt file")

    # --- download ---
    p_dl = subs.add_parser("download", help="Download sessions from 4TU",
                           parents=[shared])
    p_dl.add_argument("sessions", nargs="*", metavar="SESSION",
                      help="Specific session names to download")
    _add_filter_args(p_dl)
    p_dl.add_argument("--utils", action="store_true",
                      help="Download the utils.tar.gz archive")
    p_dl.add_argument("--dry-run", action="store_true",
                      help="Show what would be downloaded without downloading")

    # --- generate ---
    p_gen = subs.add_parser("generate", help="Generate training dataset",
                            parents=[shared])
    _add_filter_args(p_gen)
    p_gen.add_argument("--template", required=True,
                       help="Template name (e.g. reyolov8_distance) or path to a .json file")
    p_gen.add_argument("--dest", required=True,
                       help="Output dataset directory")
    p_gen.add_argument("--split-label",
                       help="Split name for output directory (train/val/test)")
    p_gen.add_argument("--clean", action="store_true",
                       help="Remove existing output directory before generating")
    p_gen.add_argument("--add", action="store_true",
                       help="Append to an existing dataset instead of failing")
    p_gen.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")

    # --- enrich ---
    subs.add_parser(
        "enrich",
        help="Scan local sessions and cache their metadata (duration, sensors, etc.)",
        parents=[shared],
    )

    # --- precompute-radar-cache ---
    p_prc = subs.add_parser(
        "precompute-radar-cache",
        help=("Pre-extract per-frame radar point clouds and Range-Doppler "
              "maps to a portable HDF5 file, so that downstream users "
              "without the radar DSP library can still generate datasets."),
        parents=[shared],
    )
    p_prc.add_argument(
        "sessions", nargs="*", metavar="SESSION",
        help="Session names (resolved against --data-root). Mutually "
             "exclusive with --path.",
    )
    _add_filter_args(p_prc)
    p_prc.add_argument(
        "--path", nargs="+", metavar="DIR",
        help="One or more radar recording directories (e.g. "
             "/data/.../session/ti_radar). Mutually exclusive with "
             "session-name resolution.",
    )
    p_prc.add_argument(
        "--radar-subdir", default="ti_radar",
        help="Subdirectory inside each session that holds the radar "
             "recording (default: ti_radar).",
    )
    p_prc.add_argument(
        "--backend", default=None,
        help="Source radar backend identifier (e.g. pycore). Defaults "
             "to the first registered non-cached backend.",
    )
    p_prc.add_argument(
        "--no-fft", action="store_true",
        help="Skip Range-Doppler maps to produce a much smaller cache "
             "(point cloud only). Use only if your dataset settings "
             "have store_fft=false.",
    )
    p_prc.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing radar_cache.h5.",
    )

    # --- visualize ---
    p_vis = subs.add_parser(
        "visualize",
        help="Render annotation overlay videos for a session",
        parents=[shared],
    )
    p_vis.add_argument(
        "--session", type=str, default=None,
        help="Session name (e.g. 2023-10-26_15-34-07). "
             "Renders all requested sensors.",
    )
    p_vis.add_argument(
        "--sensors", nargs="+", default=None,
        metavar="SENSOR",
        help="Sensor directories to visualize (default: rgb davis prophesee)",
    )
    p_vis.add_argument(
        "--modes", nargs="+", default=None,
        metavar="MODE",
        choices=["all", "only_base", "only_human_pose", "only_human_seg"],
        help="Visualization modes (default: all four)",
    )
    p_vis.add_argument(
        "-i", "--input", type=str, default=None,
        help="Raw mode: path to a specific annotations.json",
    )
    p_vis.add_argument(
        "-o", "--output", action="append", nargs=2,
        metavar=("PATH", "MODE"),
        help="Raw mode: output path and mode (repeatable)",
    )
    p_vis.add_argument(
        "-b", "--background", type=str, default=None,
        help="Background video (.mp4) for compositing",
    )
    p_vis.add_argument(
        "--background-delay", type=int, default=0,
        help="Temporal offset for background video (ms)",
    )
    p_vis.add_argument(
        "--from-ms", type=int, default=-1,
        help="Start rendering from this timestamp (ms)",
    )
    p_vis.add_argument(
        "--to-ms", type=int, default=-1,
        help="Stop rendering at this timestamp (ms)",
    )

    # --- reconstruct ---
    p_rec = subs.add_parser(
        "reconstruct",
        help="Reconstruct intensity frames from event camera data (E2VID)",
        parents=[shared],
    )
    p_rec.add_argument(
        "--session", type=str, default=None,
        help="Session name. Reconstructs events for the requested sensor directories.",
    )
    p_rec.add_argument(
        "--sensors", nargs="+", default=None,
        metavar="SENSOR",
        help="Sensor directories to reconstruct (default: davis prophesee)",
    )
    p_rec.add_argument(
        "-i", "--input", type=str, default=None,
        help="Raw mode: path to a specific events.hdf5 file",
    )
    p_rec.add_argument(
        "--output-dir", type=str, default=None,
        help="Raw mode: output directory (default: same directory as input)",
    )
    p_rec.add_argument(
        "--weights", type=str, default=None,
        help="Path to E2VID model weights (.pth.tar)",
    )
    p_rec.add_argument(
        "--fps", type=float, default=60.0,
        help="Output video frame rate (default: 60)",
    )
    p_rec.add_argument(
        "--output-type", type=str, default="lossy-video",
        choices=["img", "lossy-video", "lossless-video"],
        help="Output format (default: lossy-video)",
    )
    p_rec.add_argument(
        "--use-gpu", action="store_true",
        help="Use GPU for reconstruction",
    )

    # --- train ---
    p_train = subs.add_parser("train", help="Train a model",
                              parents=[shared])
    p_train.add_argument("--config", required=True,
                         help="Experiment config file (.py with Exp class)")
    p_train.add_argument("-b", "--batch-size", type=int, default=None,
                         help="Batch size (overrides config)")
    p_train.add_argument("--epochs", type=int, default=None,
                         help="Number of epochs (overrides config)")
    p_train.add_argument("--device", type=str, default="",
                         help="Device to use (e.g. 0, 1, cpu)")
    p_train.add_argument("--resume", type=str, default=None,
                         help="Path to checkpoint to resume training from")
    p_train.add_argument("-expn", "--experiment-name", type=str, default=None,
                         help="Experiment name (overrides config)")
    p_train.add_argument("--verbose", "-v", action="store_true",
                         help="Verbose output")
    p_train.add_argument("--fp16", action="store_true",
                         help="Use FP16 mixed precision training")
    p_train.add_argument("--no-wandb", action="store_true",
                         help="Disable Weights & Biases logging")
    p_train.add_argument("--wandb-project", type=str, default=None,
                         help="W&B project name")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "list":
        cmd_list(args)
    elif args.command == "download":
        cmd_download(args)
    elif args.command == "enrich":
        cmd_enrich(args)
    elif args.command == "precompute-radar-cache":
        cmd_precompute_radar_cache(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "visualize":
        cmd_visualize(args)
    elif args.command == "reconstruct":
        cmd_reconstruct(args)
    elif args.command == "train":
        cmd_train(args)


if __name__ == "__main__":
    main()
