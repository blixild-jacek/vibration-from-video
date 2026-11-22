"""Make a test video of a part ringing down, with a known natural frequency and damping ratio.

The clip checks free_decay.py's arithmetic against values set in advance; it is not a real part. A textured patch
(the part's tip) swings up and down as a damped oscillation; a second patch (a point that does not move) stays
still. Sensor noise is added to every frame and the video is compressed, as a real camera's would be. It leaves out
bending, glare, motion blur, flickering light and lens distortion.

Usage: python3 make_test_clip.py [OUT.mp4] [--fn 2.0] [--zeta 0.03] [--fps 60] [--seconds 15]
       then run the free_decay.py command it prints.

SPDX-License-Identifier: MIT
Copyright (c) 2026 Jacek Garbuliński Blixild, blixild.com
"""
import argparse

import cv2
import numpy as np

TIP = (424, 240)    # the middle of the swinging patch at rest, px; the first frame shows it a full swing lower
FIXED = (104, 84)   # the middle of the patch that does not move, px


def first_tip(amplitude):
    """The middle of the swinging patch in the first frame, where the swing starts, px."""
    return TIP[0], TIP[1] + int(round(amplitude))


def patch(rng, size=48):
    """A square of random texture that a feature tracker can lock on to."""
    texture = rng.integers(0, 255, (size // 4, size // 4), dtype=np.uint8)
    return cv2.resize(texture, (size, size), interpolation=cv2.INTER_NEAREST)


def make(out, fn=2.0, zeta=0.03, amplitude=40.0, fps=60.0, seconds=15.0, noise=2.0):
    """Write the clip to out."""
    if not (fn > 0 and 0 < zeta < 1 and 0 < amplitude <= 150 and fps > 0 and seconds > 0):
        raise SystemExit("needs a natural frequency above 0, a damping ratio between 0 and 1, a first swing of up to "
                         "150 px, and a frame rate and length above 0")
    rng = np.random.default_rng(1)
    tip, fixed = patch(rng), patch(rng)
    w, h = 640, 480
    wn = 2 * np.pi * fn
    wd = wn * np.sqrt(1 - zeta ** 2)
    video = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h), isColor=False)
    if not video.isOpened():
        raise SystemExit(f"cannot write {out}")
    for i in range(int(seconds * fps)):
        t = i / fps
        y = amplitude * np.exp(-zeta * wn * t) * np.cos(wd * t)
        frame = np.full((h, w), 128, np.float32)
        frame[FIXED[1] - 24:FIXED[1] + 24, FIXED[0] - 24:FIXED[0] + 24] = fixed  # the point that does not move
        # the tip, shifted by a fraction of a pixel where needed
        canvas = np.full((h, w), 128, np.float32)
        canvas[TIP[1] - 24:TIP[1] + 24, TIP[0] - 24:TIP[0] + 24] = tip
        shift = np.float32([[1, 0, 0], [0, 1, y]])
        moved = cv2.warpAffine(canvas, shift, (w, h), flags=cv2.INTER_LINEAR, borderValue=128)
        top, bottom = max(TIP[1] - 24 - int(amplitude) - 4, 0), min(TIP[1] + 24 + int(amplitude) + 4, h)
        frame[top:bottom, 360:490] = moved[top:bottom, 360:490]  # the strip the tip swings in
        frame += rng.normal(0, noise, frame.shape)
        video.write(np.clip(frame, 0, 255).astype(np.uint8))
    video.release()


def main():
    ap = argparse.ArgumentParser(description="A test video with a known natural frequency and damping ratio.")
    ap.add_argument("out", nargs="?", default="test_clip.mp4")
    ap.add_argument("--fn", type=float, default=2.0, help="natural frequency, Hz")
    ap.add_argument("--zeta", type=float, default=0.03, help="damping ratio")
    ap.add_argument("--amplitude", type=float, default=40.0, help="first swing, pixels")
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--noise", type=float, default=2.0, help="sensor noise, grey levels")
    args = ap.parse_args()
    make(args.out, args.fn, args.zeta, args.amplitude, args.fps, args.seconds, args.noise)
    print(f"{args.out}: natural frequency {args.fn} Hz, damping ratio {args.zeta}, {args.fps:g} fps, "
          f"{args.seconds:g} s. Read it back with:")
    x, y = first_tip(args.amplitude)
    print(f"  python3 free_decay.py {args.out} --part {x} {y} --fixed {FIXED[0]} {FIXED[1]} --fps {args.fps:g}")


if __name__ == "__main__":
    main()
