"""Make a test video of a part with a known natural frequency and damping ratio: ringing down after a tap, or
shaken by a machine coasting down through the resonance.

The clip checks free_decay.py's and speed_test.py's arithmetic against values set in advance; it is not a real part.
A textured patch (the part's tip) swings up and down; a second patch (a point that does not move) stays still. After
a tap the swing is a damped oscillation. Driven by the machine, it is a single resonance driven by a rotating
unbalance, whose force grows with the square of the speed, while the speed falls exponentially from one frequency to
another (a coast-down) or steps through a list of speeds.
Sensor noise is added to every frame and the video is compressed, as a real camera's would be. It leaves out
bending, glare, motion blur, flickering light and lens distortion.

Usage: python3 make_test_clip.py [OUT.mp4] [--fn 2.0] [--zeta 0.03] [--fps 60] [--seconds 15]
       python3 make_test_clip.py OUT.mp4 --fn 12 --zeta 0.02 --fps 240 --seconds 12 --coast-down 18 7
       python3 make_test_clip.py OUT.mp4 --fn 12 --zeta 0.02 --fps 240 --seconds 18 --steps 9 10 11 12 13 14 15
       then run the command it prints.

SPDX-License-Identifier: MIT
Copyright (c) 2026 Jacek Garbuliński Blixild, blixild.com
"""
import argparse

import cv2
import numpy as np

TIP = (424, 240)    # the middle of the swinging patch at rest, px; the first frame shows it a full swing lower
FIXED = (104, 84)   # the middle of the patch that does not move, px


def first_tip(amplitude, scale=1):
    """The middle of the swinging patch in the first frame of a ring-down, where the swing starts, px."""
    return TIP[0] * scale, TIP[1] * scale + int(round(amplitude))


def patch(rng, size=48):
    """A square of random texture that a feature tracker can lock on to."""
    texture = rng.integers(0, 255, (size // 4, size // 4), dtype=np.uint8)
    return cv2.resize(texture, (size, size), interpolation=cv2.INTER_NEAREST)


def ring_down(fn, zeta, amplitude):
    """The tip's swing after a tap (px) at times t (s): a damped oscillation starting a full swing low."""
    wn = 2 * np.pi * fn
    wd = wn * np.sqrt(1 - zeta ** 2)
    return lambda t: amplitude * np.exp(-zeta * wn * t) * np.cos(wd * t)


def running_speed(start, end, seconds):
    """A coast-down's running frequency (Hz) at times t (s): falling exponentially from start to end."""
    return lambda t: start * (end / start) ** (np.asarray(t) / seconds)


def steps(speeds, seconds):
    """A drive stepped through speeds (Hz), each held for an equal share of seconds, at times t (s)."""
    speeds = np.asarray(speeds, float)
    return lambda t: speeds[np.minimum((np.asarray(t) / seconds * len(speeds)).astype(int), len(speeds) - 1)]


def driven(fn, zeta, amplitude, speed, seconds):
    """The tip's swing (px) at times t (s) while the machine runs at speed(t) Hz: one resonance driven by a rotating
    unbalance, x'' + 2 zeta wn x' + wn^2 x = W^2 cos(theta), W = 2 pi speed(t) and theta' = W, integrated from a steady
    run at the first speed and scaled so the largest swing is amplitude."""
    from scipy.integrate import solve_ivp
    wn = 2 * np.pi * fn
    lead = 8 / (zeta * wn)  # a steady run at the first speed first, so the start-up transient has died away
    f0 = float(speed(0.0))

    def rhs(t, s):
        w = 2 * np.pi * (f0 if t < 0 else float(speed(t)))
        return [s[1], w ** 2 * np.cos(s[2]) - 2 * zeta * wn * s[1] - wn ** 2 * s[0], w]

    step = 1 / (40 * max(f0, fn, float(np.max(speed(np.linspace(0, seconds, 200))))))
    sol = solve_ivp(rhs, (-lead, seconds), [0.0, 0.0, 0.0], max_step=step, dense_output=True, rtol=1e-8, atol=1e-10)
    grid = np.arange(0, seconds, step)
    scale = amplitude / np.abs(sol.sol(grid)[0]).max()
    return lambda t: scale * sol.sol(np.asarray(t))[0]


def coast_down(fn, zeta, amplitude, start, end, seconds):
    """The tip's swing (px) at times t (s) while the machine coasts down from start to end Hz (driven())."""
    return driven(fn, zeta, amplitude, running_speed(start, end, seconds), seconds)


def make(out, fn=2.0, zeta=0.03, amplitude=40.0, fps=60.0, seconds=15.0, noise=2.0, coast=None, speeds=None,
         scale=1):
    """Write the clip to out: a ring-down after a tap; with coast=(start, end) a coast-down from start to end Hz over
    the whole clip; with speeds=[...] the drive stepped through those speeds (Hz), each held for an equal share. With
    scale=3 the frame, the patches and their places are three times larger (1920 x 1440), the texture as fine and the
    swing as given."""
    if not (fn > 0 and 0 < zeta < 1 and 0 < amplitude <= 150 and fps > 0 and seconds > 0):
        raise SystemExit("needs a natural frequency above 0, a damping ratio between 0 and 1, a first swing of up to "
                         "150 px, and a frame rate and length above 0")
    if coast and not (coast[0] > coast[1] > 0):
        raise SystemExit("a coast-down needs a start frequency above its end frequency, both above 0")
    if speeds is not None and not (len(speeds) and min(speeds) > 0):
        raise SystemExit("speed steps need at least one speed, all above 0")
    swing = (coast_down(fn, zeta, amplitude, coast[0], coast[1], seconds) if coast else
             driven(fn, zeta, amplitude, steps(speeds, seconds), seconds) if speeds is not None else
             ring_down(fn, zeta, amplitude))
    rng = np.random.default_rng(1)
    s, half = int(scale), 24 * int(scale)
    tip, fixed = patch(rng, 2 * half), patch(rng, 2 * half)
    w, h = 640 * s, 480 * s
    (tx, ty), (fx, fy) = (TIP[0] * s, TIP[1] * s), (FIXED[0] * s, FIXED[1] * s)
    video = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h), isColor=False)
    if not video.isOpened():
        raise SystemExit(f"cannot write {out}")
    for i in range(int(seconds * fps)):
        t = i / fps
        y = float(swing(t))
        frame = np.full((h, w), 128, np.float32)
        frame[fy - half:fy + half, fx - half:fx + half] = fixed  # the point that does not move
        # the tip, shifted by a fraction of a pixel where needed
        canvas = np.full((h, w), 128, np.float32)
        canvas[ty - half:ty + half, tx - half:tx + half] = tip
        shift = np.float32([[1, 0, 0], [0, 1, y]])
        moved = cv2.warpAffine(canvas, shift, (w, h), flags=cv2.INTER_LINEAR, borderValue=128)
        top, bottom = max(ty - half - int(amplitude) - 4, 0), min(ty + half + int(amplitude) + 4, h)
        frame[top:bottom, 360 * s:490 * s] = moved[top:bottom, 360 * s:490 * s]  # the strip the tip swings in
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
    ap.add_argument("--coast-down", type=float, nargs=2, metavar=("START", "END"),
                    help="a machine coasting down from START to END Hz drives the part, instead of a tap")
    ap.add_argument("--steps", type=float, nargs="+", metavar="HZ",
                    help="the drive stepped through these speeds, Hz, each held for an equal share of the clip")
    args = ap.parse_args()
    make(args.out, args.fn, args.zeta, args.amplitude, args.fps, args.seconds, args.noise, args.coast_down, args.steps)
    print(f"{args.out}: natural frequency {args.fn} Hz, damping ratio {args.zeta}, {args.fps:g} fps, "
          f"{args.seconds:g} s" + (f", coast-down from {args.coast_down[0]:g} to {args.coast_down[1]:g} Hz"
                                  if args.coast_down else "") +
          (f", speed steps {', '.join(f'{s:g}' for s in args.steps)} Hz" if args.steps else "") +
          ". Read it back with:")
    if args.coast_down or args.steps:
        print(f"  python3 speed_test.py {args.out} --part {TIP[0]} {TIP[1]} --fixed {FIXED[0]} {FIXED[1]} "
              f"--fps {args.fps:g}")
    else:
        x, y = first_tip(args.amplitude)
        print(f"  python3 free_decay.py {args.out} --part {x} {y} --fixed {FIXED[0]} {FIXED[1]} --fps {args.fps:g}")


if __name__ == "__main__":
    main()
