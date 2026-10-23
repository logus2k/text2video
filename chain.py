"""Chain several Wan2.2 clips into one longer video: each clip after the first starts from the previous clip's last frame
(image-to-video), then the clips are joined.

    python chain.py --prompt "A red fox trotting through snow at sunrise" \\
                    --prompt "The fox stops, looks back at the camera" \\
                    --prompt "The fox runs off between the pine trees" \\
                    --lora --out outputs/fox-story.mp4

Each --prompt is one segment; the first is rendered from text, the rest from the previous segment's last frame.
Any other arguments are passed through to generate.py (e.g. --model, --quant, --lora, --frames, --width, --height,
--seed, --steps). Segments are written next to the output as <out>-01.mp4, <out>-02.mp4, ... so a bad one can be
re-rendered alone (see --start). The first frame of every continuation equals the seed frame, so it is dropped when
joining; --crossfade N additionally blends N frames across each seam.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from diffusers.utils import export_to_video
from PIL import Image

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], epilog="Unknown arguments are passed to generate.py.")
    p.add_argument("--prompt", action="append", required=True, help="one per segment, in order")
    p.add_argument("--out", type=Path, required=True, help="joined video (.mp4)")
    p.add_argument("--crossfade", type=int, default=0, help="frames to blend across each seam (default 0: hard cut)")
    p.add_argument("--fps", type=int, help="output fps (default: the segments' fps)")
    p.add_argument("--start", type=int, default=1, help="resume: first segment to (re)render; earlier ones are reused")
    p.add_argument("--image", type=Path, help="start the first segment from this image instead of from text")
    return p.parse_known_args()


def read_video(path):
    """All frames of an mp4 as a (frames, height, width, 3) uint8 array, plus its fps (imageio-ffmpeg backend)."""
    with imageio.get_reader(str(path), "ffmpeg") as reader:
        fps = reader.get_meta_data().get("fps", 16)
        return np.stack([frame for frame in reader]), fps


def last_frame(video, path):
    frames, _ = read_video(video)
    Image.fromarray(frames[-1]).save(path)
    return frames.shape[1:3]


def join(segments, out, crossfade, fps):
    clips, fps_list = zip(*(read_video(seg) for seg in segments))
    fps = fps or round(fps_list[0])
    video = clips[0]
    for clip in clips[1:]:
        clip = clip[1:]  # frame 0 is the seed frame = last frame of the previous clip
        if crossfade and len(video) > crossfade and len(clip) > crossfade:
            w = np.linspace(0, 1, crossfade + 2)[1:-1, None, None, None]
            blend = (video[-crossfade:] * (1 - w) + clip[:crossfade] * w).astype(np.uint8)
            video = np.concatenate([video[:-crossfade], blend, clip[crossfade:]])
        else:
            video = np.concatenate([video, clip])
    # PIL frames: export_to_video would scale ndarray frames by 255, which overflows uint8 input.
    export_to_video([Image.fromarray(frame) for frame in video], str(out), fps=fps)
    return len(video), fps


def main():
    args, passthrough = parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = out.with_suffix("")
    segments = []
    seed_image = args.image
    for i, prompt in enumerate(args.prompt, 1):
        seg = Path(f"{stem}-{i:02d}.mp4")
        if i >= args.start:
            cmd = [sys.executable, str(ROOT / "generate.py"), "--prompt", prompt, "--out", str(seg), *passthrough]
            if seed_image:
                cmd += ["--image", str(seed_image)]
            print(f"=== segment {i}/{len(args.prompt)}: {prompt}", flush=True)
            subprocess.run(cmd, check=True)
        elif not seg.is_file():
            raise SystemExit(f"--start {args.start} but {seg} does not exist")
        segments.append(seg)
        seed_image = Path(f"{stem}-{i:02d}-last.png")
        size = last_frame(seg, seed_image)
        print(f"    last frame {size[1]}x{size[0]} -> {seed_image.name}", flush=True)
    n, fps = join(segments, out, args.crossfade, args.fps)
    (stem.parent / f"{stem.name}.json").write_text(json.dumps(
        {"prompts": args.prompt, "segments": [s.name for s in segments], "generate_args": passthrough,
         "crossfade": args.crossfade, "frames": n, "fps": fps}, indent=2))
    print(f"Saved {out}: {n} frames at {fps} fps = {n / fps:.1f} s from {len(segments)} segments", flush=True)


if __name__ == "__main__":
    main()
