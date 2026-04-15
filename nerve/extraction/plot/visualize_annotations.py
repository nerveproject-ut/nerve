"""
Render annotation overlay videos for NERVE session sensor directories.

Produces up to four visualisation videos per sensor directory:

- ``base.mp4``  — bounding boxes, track IDs, class labels, distances
- ``pose.mp4``  — human pose skeletons
- ``seg.mp4``   — human body-part segmentation
- ``all.mp4``   — composite of all three (base + thumbnails)

Can be invoked standalone::

    python -m nerve.extraction.plot.visualize_annotations \\
        -i session/rgb/annotations/annotations.json \\
        -o session/rgb/base.mp4 only_base

Or through the CLI::

    nerve visualize --session 2023-10-26_15-34-07 --sensor rgb
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from nerve.extraction.custom_coco import VisualCOCO
from nerve.extraction.utils.ffmpegWriters import VideoWriter_x264

MODES = {"all", "only_base", "only_human_pose", "only_human_seg"}


class _BackgroundVideo:
    def __init__(self, video_path: str) -> None:
        cap = cv2.VideoCapture(video_path)
        self._cap = cap
        self._width = int(cap.get(3))
        self._height = int(cap.get(4))
        self._fps = cap.get(cv2.CAP_PROP_FPS)
        self._total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._period_ms = 1000.0 / self._fps
        self._current_frame = 0

    @property
    def dimensions(self):
        return (self._height, self._width)

    def move_to_ms(self, time_ms: float):
        target = round(time_ms / self._period_ms)
        if target < 0 or target >= self._total:
            return None
        if target != self._current_frame + 1:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = self._cap.read()
        self._current_frame = target
        return frame if ok else None


def _overlap(fg: torch.Tensor, alpha: torch.Tensor, bg: torch.Tensor):
    a = (alpha.to(torch.float32) / 255.0).unsqueeze(-1)
    return (bg * (1.0 - a) + fg * a).to(torch.uint8).cpu().numpy()


def render_annotations(
    annotation_json: str,
    outputs: list[tuple[str, str]],
    *,
    background: str = "",
    background_delay_ms: int = 0,
    from_ms: int = -1,
    to_ms: int = -1,
    device: torch.device | None = None,
) -> None:
    """Render annotation overlay video(s).

    Args:
        annotation_json: Path to a COCO-format ``annotations.json``.
        outputs: List of ``(output_path, mode)`` pairs.  *mode* is one of
            ``all``, ``only_base``, ``only_human_pose``, ``only_human_seg``.
        background: Optional background video path.
        background_delay_ms: Temporal offset for background video (ms).
        from_ms: Start rendering at this time (ms), or -1 for the beginning.
        to_ms: Stop rendering at this time (ms), or -1 for the end.
        device: Torch device for compositing.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for path, mode in outputs:
        if mode not in MODES:
            raise ValueError(f"Unknown mode '{mode}'. Choose from {sorted(MODES)}")

    dataset = VisualCOCO(annotation_json)
    h, w = dataset.frame_height, dataset.frame_width
    rh, rw = h // 2, w // 2

    need_base = any(m in ("all", "only_base") for _, m in outputs)
    need_pose = any(m in ("all", "only_human_pose") for _, m in outputs)
    need_seg = any(m in ("all", "only_human_seg") for _, m in outputs)

    use_bg = background != "" and os.path.isfile(background)
    bg_vid = _BackgroundVideo(background) if use_bg else None
    empty_bg = np.zeros((h, w, 3), dtype=np.uint8) if use_bg else None

    writers: list[dict] = []
    for path, mode in outputs:
        entry: dict = {"path": path, "mode": mode}
        if mode == "all":
            entry["final_w"] = w + rw
        else:
            entry["final_w"] = w
        entry["final_h"] = h
        if path.endswith(".mp4"):
            entry["writer"] = VideoWriter_x264(path, dataset.fps)
        writers.append(entry)

    for idx in tqdm(list(dataset.imgs.keys()), desc="Rendering"):
        img = dataset.loadImgs(idx)[0]
        ann_time = img["time_ms"]

        if ann_time < from_ms:
            continue
        if to_ms >= 0 and ann_time > to_ms:
            continue

        bg_frame_t = None
        if use_bg:
            raw = bg_vid.move_to_ms(ann_time - background_delay_ms)
            bg_frame_t = torch.from_numpy(raw if raw is not None else empty_bg.copy()).to(device)

        base_img = seg_img = pose_img = None

        if need_base:
            raw = dataset.getAnnotatedFrame(
                idx, show_entity_seg=True, show_ID=True, show_class=True,
                show_skeleton=False, show_conf=False, show_distance=True,
            )
            if use_bg:
                base_img = _overlap(
                    torch.from_numpy(raw[:, :, :3]).to(device),
                    torch.from_numpy(raw[:, :, -1]).to(device),
                    bg_frame_t,
                )
            else:
                base_img = raw[:, :, :3]

        if need_seg:
            raw = dataset.getAnnotatedFrame(
                idx, show_bb=False, show_body_parts_seg=True, show_skeleton=False,
            )
            if use_bg:
                seg_img = _overlap(
                    torch.from_numpy(raw[:, :, :3]).to(device),
                    torch.from_numpy(raw[:, :, -1]).to(device),
                    bg_frame_t,
                )
            else:
                seg_img = raw[:, :, :3]

        if need_pose:
            raw = dataset.getAnnotatedFrame(
                idx, show_bb=False, show_ID=False, show_skeleton=True,
            )
            if use_bg:
                pose_img = _overlap(
                    torch.from_numpy(raw[:, :, :3]).to(device),
                    torch.from_numpy(raw[:, :, -1]).to(device),
                    bg_frame_t,
                )
            else:
                pose_img = raw[:, :, :3]

        for entry in writers:
            mode = entry["mode"]
            if mode == "only_base":
                frame = base_img
            elif mode == "only_human_pose":
                frame = pose_img
            elif mode == "only_human_seg":
                frame = seg_img
            elif mode == "all":
                fw = entry["final_w"]
                out_w = fw + 1 if fw % 2 else fw
                frame = np.zeros((h, out_w, 3), dtype=np.uint8)
                frame[0:h, 0:w] = base_img
                frame[0:rh, w:fw] = cv2.resize(seg_img, dsize=(rw, rh))
                frame[rh:h, w:fw] = cv2.resize(pose_img, dsize=(rw, rh))
            else:
                continue

            if entry["path"].endswith(".png"):
                cv2.imwrite(entry["path"], frame)
            elif "writer" in entry:
                entry["writer"].write(frame)

    for entry in writers:
        if "writer" in entry:
            entry["writer"].release()


def visualize_session_sensor(
    sensor_dir: str | Path,
    modes: list[str] | None = None,
    *,
    background: str = "",
    background_delay_ms: int = 0,
    from_ms: int = -1,
    to_ms: int = -1,
) -> list[str]:
    """Render standard visualization videos for a sensor directory.

    Args:
        sensor_dir: Path to e.g. ``session/rgb/`` or ``session/davis/``.
        modes: Which videos to generate. Defaults to all four.
        background: Optional background video.
        background_delay_ms: Temporal offset for background video (ms).

    Returns:
        List of output file paths created.
    """
    sensor_dir = Path(sensor_dir)
    ann_file = sensor_dir / "annotations" / "annotations.json"
    if not ann_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")

    mode_to_filename = {
        "all": "all.mp4",
        "only_base": "base.mp4",
        "only_human_pose": "pose.mp4",
        "only_human_seg": "seg.mp4",
    }

    if modes is None:
        modes = list(mode_to_filename.keys())

    outputs = []
    for m in modes:
        if m not in mode_to_filename:
            raise ValueError(f"Unknown mode '{m}'. Choose from {sorted(mode_to_filename)}")
        outputs.append((str(sensor_dir / mode_to_filename[m]), m))

    render_annotations(
        str(ann_file),
        outputs,
        background=background,
        background_delay_ms=background_delay_ms,
        from_ms=from_ms,
        to_ms=to_ms,
    )

    return [p for p, _ in outputs]


def main():
    parser = argparse.ArgumentParser(
        description="Render annotation overlay videos.",
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Path to annotations.json",
    )
    parser.add_argument(
        "-o", "--output", action="append", nargs=2,
        metavar=("OUTPUT_PATH", "MODE"),
        help="Output path and mode (all, only_base, only_human_pose, only_human_seg). Repeatable.",
    )
    parser.add_argument(
        "-b", "--background", default="",
        help="Background video (.mp4) for compositing",
    )
    parser.add_argument(
        "-d", "--background-delay", type=int, default=0,
        help="Temporal offset for background video (ms)",
    )
    parser.add_argument(
        "-f", "--from-ms", type=int, default=-1,
        help="Start rendering from this timestamp (ms)",
    )
    parser.add_argument(
        "-t", "--to-ms", type=int, default=-1,
        help="Stop rendering at this timestamp (ms)",
    )
    args = parser.parse_args()

    if args.output is None:
        print("Error: at least one -o OUTPUT_PATH MODE pair is required.", file=sys.stderr)
        sys.exit(1)

    render_annotations(
        args.input,
        [(p, m) for p, m in args.output],
        background=args.background,
        background_delay_ms=args.background_delay,
        from_ms=args.from_ms,
        to_ms=args.to_ms,
    )


if __name__ == "__main__":
    main()
