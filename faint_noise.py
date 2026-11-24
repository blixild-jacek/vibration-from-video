"""How free_decay.faint() behaves on simulated swings: on noise alone, how high its ratio gets (which sets
free_decay.FAINT_RATIO), and how small a ring it finds and how close its frequency comes. With --speed, the same for
speed_test.faint_response(), the speed test below a pixel (which sets speed_test.FAINT_LOCAL): noise alone, and run-ups
and coast-downs through a resonance driven by a rotating unbalance.

The swing is white tracker jitter (standard deviation 1) plus a slow drift, a random walk scaled to a given multiple of
the jitter. A ring is a tap: a damped sine starting at 5 s, its first swing a given multiple of the jitter. Nothing in
it comes from a real video, so the threshold is fixed before any real video is read with it.

Usage:
  python3 faint_noise.py              the tables: 1,000 noise-only swings a case, 200 unsteady ones, 100 rings
  python3 faint_noise.py --quick      a tenth of that
  python3 faint_noise.py --speed      the speed test's tables: 200 noise-only swings a case, 40 unsteady, 40 sweeps

SPDX-License-Identifier: MIT
Copyright (c) 2026 Jacek Garbuliński Blixild, blixild.com
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # free_decay beside this file, also under python3 -I
import free_decay  # noqa: E402
import speed_test  # noqa: E402


def swing(rng, fps, seconds, drift, ring=0.0, hz=0.0, zeta=0.02, tap=5.0):
    """A simulated swing in units of the jitter: white jitter, a random-walk drift `drift` times its size, and a ring
    of first swing `ring` at `hz` from `tap` s."""
    n = int(seconds * fps)
    x = rng.normal(0, 1, n)
    if drift:
        walk = np.cumsum(rng.normal(0, 1, n))
        x += drift * (walk - walk.mean()) / walk.std()
    if ring:
        t = np.arange(n) / fps - tap
        on = t >= 0
        x[on] += ring * np.exp(-zeta * 2 * np.pi * hz * t[on]) * np.sin(2 * np.pi * hz * np.sqrt(1 - zeta ** 2) * t[on])
    return x


def ratio(x, fps):
    found = free_decay.faint(x, fps)
    return found[1] if found else 0.0


UNSTEADY = {  # noise that is not steady, as an independent audit of faint() built it (24 Nov 2026)
    "frozen": "the last 3 s one repeated frame",
    "quieter": "the last 6 s ten times quieter",
    "burst": "noise five times larger for 4 s after an event at 5 s",
    "blip": "a 3-frame jump of 30 times the jitter at 5 s",
    "bump": "a 12 Hz wobble of 30 times the jitter, damping ratio 0.05, 2.5 s before the end",
}


def unsteady(rng, fps, kind, seconds=20.0, drift=5):
    """A noise-only swing (units of the jitter) whose noise is not steady: one of UNSTEADY."""
    n = int(seconds * fps)
    t = np.arange(n) / fps
    white = rng.normal(0, 1, n)
    walk = np.cumsum(rng.normal(0, 1, n))
    walk = drift * (walk - walk.mean()) / walk.std()
    if kind == "quieter":
        white[t >= seconds - 6] *= 0.1
    if kind == "burst":
        white[(t >= 5) & (t < 9)] *= 5
    x = white + walk
    if kind == "frozen":
        x[t >= seconds - 3] = x[np.argmax(t >= seconds - 3)]
    if kind == "blip":
        x[int(5 * fps):int(5 * fps) + 3] += 30
    if kind == "bump":
        on = t >= seconds - 2.5
        tt = t[on] - (seconds - 2.5)
        x[on] += 30 * np.exp(-0.05 * 2 * np.pi * 12 * tt) * np.sin(2 * np.pi * 12 * tt)
    return x


def unsteady_table(rng, count):
    print("Noise that is not steady: the share of noise-only swings that faint() passes (both bars)")
    print("case       25 fps  60 fps  240 fps")
    for kind, about in UNSTEADY.items():
        shares = []
        for fps in (25.0, 60.0, 240.0):
            shares.append(np.mean([free_decay.passes(free_decay.faint(unsteady(rng, fps, kind), fps), fps)
                                   for _ in range(count)]))
        print(f"{kind:9s} " + "  ".join(f"{v:6.0%}" for v in shares) + f"   ({about})", flush=True)
    print()


def noise_table(rng, count):
    print("Noise only: faint()'s ratio (the threshold is %g)" % free_decay.FAINT_RATIO)
    print("seconds  fps  drift  99 %   99.9 %  largest")
    largest = 0.0
    for seconds in (10.0, 20.0):
        for fps in (25.0, 60.0, 240.0):
            for drift in (0, 1, 5, 10, 30):
                r = np.array([ratio(swing(rng, fps, seconds, drift), fps) for _ in range(count)])
                largest = max(largest, r.max())
                print(f"{seconds:7g} {fps:4g} {drift:5d}x {np.quantile(r, .99):5.2f} {np.quantile(r, .999):7.2f} "
                      f"{r.max():7.2f}", flush=True)
    print(f"largest of all {30 * count} swings: {largest:.2f}\n")


def ring_table(rng, count, drift=5):
    print(f"Rings (damping ratio 0.02, drift {drift} times the jitter, 20 s): found at both bars, and the frequency's"
          " error")
    print("  Hz   fps  ring/jitter  found  median error %  90 % within %")
    for hz, fps in ((3.0, 25.0), (3.7, 25.0), (12.3, 60.0), (47.0, 240.0), (90.0, 240.0)):
        damped = hz * np.sqrt(1 - 0.02 ** 2)
        for ring in (5, 10, 20, 50):
            errors = []
            for _ in range(count):
                found = free_decay.faint(swing(rng, fps, 20.0, drift, ring, hz), fps)
                if free_decay.passes(found, fps):
                    errors.append(abs(found[0] / damped - 1) * 100)
            share = len(errors) / count
            err = f"{np.median(errors):13.2f}  {np.quantile(errors, .9):13.2f}" if errors else f"{'-':>13}  {'-':>13}"
            print(f"{hz:5g} {fps:5g} {ring:11d}  {share:5.0%}  {err}", flush=True)


def speed_ratio(x, fps):
    """The largest ratio speed_test.faint_response() would compare with FAINT_LOCAL on this swing (a window's peak over
    the noise around it), over the windows that pass its other window checks (its neighbours' frequency, the cycles in
    the window, below fps / NEAR_HALF; the swing's rise and fall is left out, so the ratio errs high); 0 when no
    window passes those, or the windows do not fit twice in the video (the speed test's "short" refusal)."""
    try:
        found = speed_test.faint_response(x, fps, bar=0.0)
    except ValueError:
        return 0.0
    return float(np.nanmax(found[8], initial=0.0)) if found else 0.0


def speed_noise_table(rng, count):
    print("Speed test, noise only: the ratio faint_response() would compare with the bar (%g)" % speed_test.FAINT_LOCAL)
    print("seconds  fps  drift  99 %   largest")
    largest = 0.0
    for seconds in (5.0, 10.0, 20.0):
        for fps in (25.0, 60.0, 240.0, 500.0):
            for drift in (0, 1, 5, 10, 30):
                r = np.array([speed_ratio(swing(rng, fps, seconds, drift), fps) for _ in range(count)])
                largest = max(largest, r.max())
                print(f"{seconds:7g} {fps:4g} {drift:5d}x {np.quantile(r, .99):5.2f} {r.max():7.2f}", flush=True)
    print(f"largest of all {60 * count} swings: {largest:.2f}\n")


def stopped(rng, fps, part, seconds=20.0, drift=5, quieter=0.3):
    """Noise only, the jitter `quieter` times as large over the first `part` of the video (a machine stopped while the
    camera films, then running): the median window comes from the quiet part when it is the larger."""
    x = swing(rng, fps, seconds, drift)
    x[:int(part * len(x))] *= quieter
    return x


def speed_unsteady_table(rng, count):
    print("Speed test, noise that is not steady: the share of noise-only swings read as a resonance, and the largest "
          "ratio")
    print("case        25 fps        60 fps        240 fps       500 fps")
    cases = dict(UNSTEADY)
    cases.update({"stopped 30 %": "the first 30 % of the video at 0.3 times the jitter",
                  "stopped 60 %": "the first 60 % at 0.3 times the jitter"})
    for kind, about in cases.items():
        row = []
        for fps in (25.0, 60.0, 240.0, 500.0):
            if kind.startswith("stopped"):
                xs = [stopped(rng, fps, float(kind.split()[1]) / 100) for _ in range(count)]
            else:
                xs = [unsteady(rng, fps, kind) for _ in range(count)]
            r = np.array([speed_ratio(x, fps) for x in xs])
            row.append(f"{np.mean(r >= speed_test.FAINT_LOCAL):4.0%} {r.max():6.2f}")
        print(f"{kind:12s} " + "   ".join(row) + f"   ({about})", flush=True)
    print()


def sweep(rng, fps, fn, zeta, f0, f1, ramp, hold, ring, drift=5, before=0.5):
    """A simulated speed test in units of the jitter: still for `before` s, the speed swept from f0 to f1 Hz over
    `ramp` s, then held for `hold` s, the part driven by a rotating unbalance (force growing with the square of the
    speed) through a resonance at fn with damping ratio zeta, solved in time at ten times the frame rate (so a fast
    sweep lags as it would); its largest swing `ring` times the jitter, plus white jitter and a drift."""
    from scipy.signal import lsim
    up = 10
    n = int((before + ramp + hold) * fps)
    t = np.arange(n * up) / (fps * up)
    speed = np.clip((t - before) / ramp, 0, 1) * (f1 - f0) + f0
    speed[t < before] = 0.0
    phase = 2 * np.pi * np.cumsum(speed) / (fps * up)
    wn = 2 * np.pi * fn
    force = (2 * np.pi * speed) ** 2 * np.sin(phase)
    _, y, _ = lsim(([1.0], [1.0, 2 * zeta * wn, wn ** 2]), force, t)
    y = y[::up]
    return swing(rng, fps, n / fps, drift) + ring * y / np.abs(y).max()


SWEEPS = (  # (fps, natural frequency Hz, from Hz, to Hz, ramp s, hold s): a rotor's run-ups as filmed at 500 fps
    (500.0, 35.5, 0.0, 50.0, 4.0, 2.9),        # (the dataset of 24 Nov 2026), a slower machine at 60 fps and a
    (500.0, 35.5, 0.0, 50.0, 6.0, 0.9),        # coast-down at 240 fps
    (60.0, 12.0, 0.0, 20.0, 10.0, 2.0),
    (240.0, 12.0, 20.0, 0.0, 20.0, 2.0),
)


def speed_ring_table(rng, count):
    print("Speed test, sweeps through a resonance (drift 5 times the jitter): found, and the error against the natural"
          " frequency")
    print(" fps     fn  sweep            zeta  swing/jitter  found  median error %  90 % within %")
    for fps, fn, f0, f1, ramp, hold in SWEEPS:
        for zeta in (0.02, 0.05):
            for ring in (3, 5, 10, 20, 50):
                errors = []
                for _ in range(count):
                    x = sweep(rng, fps, fn, zeta, f0, f1, ramp, hold, ring)
                    try:
                        found = speed_test.faint_response(x, fps)
                    except ValueError:  # the windows do not fit twice ("short"): not found
                        continue
                    if speed_test.faint_passes(found):
                        errors.append(abs(found[0] / fn - 1) * 100)
                err = (f"{np.median(errors):13.2f}  {np.quantile(errors, .9):13.2f}" if errors
                       else f"{'-':>13}  {'-':>13}")
                print(f"{fps:4g} {fn:6g}  {f0:g}-{f1:g} Hz in {ramp:g} s  {zeta:5.2f} {ring:12d}  "
                      f"{len(errors) / count:5.0%}  {err}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="free_decay.faint() on simulated noise and rings.")
    ap.add_argument("--quick", action="store_true", help="a tenth of the videos")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--speed", action="store_true", help="the speed test's tables instead")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    if args.speed:
        speed_noise_table(rng, 20 if args.quick else 200)
        speed_unsteady_table(rng, 10 if args.quick else 40)
        speed_ring_table(rng, 10 if args.quick else 40)
        return
    noise_table(rng, 100 if args.quick else 1000)
    unsteady_table(rng, 20 if args.quick else 200)
    ring_table(rng, 10 if args.quick else 100)


if __name__ == "__main__":
    main()
