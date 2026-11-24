"""How free_decay.faint() behaves on simulated swings: on noise alone, how high its ratio gets (which sets
free_decay.FAINT_RATIO), and how small a ring it finds and how close its frequency comes; and how close
free_decay.faint_damping() reads the damping ratio below the floor (which sets free_decay.FIT_D).

The swing is white tracker jitter (standard deviation 1) plus a slow drift, a random walk scaled to a given multiple of
the jitter. A ring is a tap: a damped sine starting at 5 s, its first swing a given multiple of the jitter. Nothing in
it comes from a real video, so the threshold is fixed before any real video is read with it.

Usage:
  python3 faint_noise.py              the tables: 1,000 noise-only swings a case, 200 unsteady ones, 100 rings
  python3 faint_noise.py --quick      a tenth of that
  python3 faint_noise.py --damping    free_decay.faint_damping() on simulated rings of a minute: its accuracy by the
                                      ring-down's d (which sets free_decay.FIT_D) and the halves' false alarms

SPDX-License-Identifier: MIT
Copyright (c) 2026 Jacek Garbuliński Blixild, blixild.com
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # free_decay beside this file, also under python3 -I
import free_decay  # noqa: E402


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


DAMPING_CASES = ((3.0, 25.0), (3.7, 25.0), (12.3, 60.0), (47.0, 240.0), (90.0, 240.0))  # Hz, fps: as ring_table


def damping_rows(rng, count, zetas=(0.005, 0.01, 0.02, 0.05), rings=(10, 20, 50, 100, 200), drifts=(5, 30)):
    """free_decay.faint_damping() on simulated rings that faint() finds: per ring, (Hz, fps, zeta, ring, drift, d, the
    fit's own relative standard error of the decay, the relative error of the damping ratio read, whether the halves
    differ). Each swing lasts a minute, the strike at 5 s, so a lightly damped ring has room to die away and a quiet end
    after it, as real recordings do. Nothing in it comes from a real video."""
    rows = []
    for hz, fps in DAMPING_CASES:
        for zeta in zetas:
            for ring in rings:
                for drift in drifts:
                    for _ in range(count):
                        x = swing(rng, fps, 60.0, drift, ring, hz, zeta)  # a minute: a ring and a quiet end
                        found = free_decay.faint(x, fps)
                        if not free_decay.passes(found, fps):
                            continue
                        f, _, seconds, a, b, _ = found
                        noise = free_decay.noise_in(free_decay.drift_off(x, fps)[len(x) - free_decay.FAINT_QUIET * (b - a):])
                        got = free_decay.faint_damping(x, fps, f, a, b, noise)
                        if got is None:
                            rows.append((hz, fps, zeta, ring, drift, 0.0, np.inf, np.inf, False, 0.0, 0.0))
                            continue
                        z, d, halves, rel = got[0], got[1], got[2], got[3]
                        rows.append((hz, fps, zeta, ring, drift, d, rel, z / zeta - 1, bool(halves and halves[2]),
                                     halves[3] if halves else 0.0, halves[4] if halves else 0.0))
    return np.array(rows, dtype=float)


D_BANDS = (5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 1000, np.inf)  # bands of d for damping_table()


def damping_table(rng, count, within=0.15, share=0.95):
    """The gate for faint_damping() (free_decay.FIT_D): for each band of d (the ring-down's signal-to-noise figure),
    how close `share` of its readings lie to the damping ratio the ring was made with, and the gate: the lowest band
    from which every band reads `share` of its rings within `within`, so a reading just past the gate is as good as the
    bar says, not only the readings past it taken together (which the cumulative table shows, and which pass from far
    lower d); and the halves' false alarms on these clean rings."""
    rows = damping_rows(rng, count)
    d, rel, err, halves = rows[:, 5], rows[:, 6], np.abs(rows[:, 7]), rows[:, 8].astype(bool)
    print(f"Damping below the floor: {len(rows)} simulated rings found by faint(), {np.isfinite(err).sum()} fitted")
    print("band of d          rings   95 % within")
    band_q = {}
    for lo, hi in zip(D_BANDS, D_BANDS[1:]):
        m = (d >= lo) & (d < hi)
        band_q[lo] = np.quantile(err[m], share) if m.any() else np.nan
        print(f"  {lo:5g} to {hi:<6g}  {m.sum():6d}  {100 * band_q[lo]:6.1f} %")
    best_d = next((lo for lo in D_BANDS[:-1] if not any(q > within for b, q in band_q.items() if b >= lo)), None)
    print("cumulative: bar on d    kept   95 % within")
    for bar in D_BANDS[:-1]:
        keep = d >= bar
        q = np.quantile(err[keep], share) if keep.any() else np.nan
        print(f"  d >= {bar:5g}  {keep.sum():12d}  {100 * q:6.1f} %")
    print("cumulative: bar on the fit's own relative error (for comparison)   kept   95 % within")
    for bar in (0.30, 0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02, 0.01):
        keep = rel <= bar
        q = np.quantile(err[keep], share) if keep.any() else np.nan
        print(f"  rel <= {bar:4.2f}  {keep.sum():6d}  {100 * q:6.1f} %")
    keep = d >= (best_d if best_d is not None else np.inf)
    print(f"gate on d: {best_d}, from which every band reads {share:.0%} within {within:.0%}; keeps {keep.sum()} of "
          f"{len(rows)}; halves differ on {halves[keep].mean():.1%} of the kept" if keep.any() else "gate on d: none")
    print("halves: false alarms on these clean rings, by the bars (combined standard errors, share of the mean)")
    for sig in (3.0, 4.0, 5.0):
        print(f"  {sig:g} sigma: " + "  ".join(f"share {sh:.1f}: {np.mean((rows[:, 9] > sig) & (rows[:, 10] > sh)):5.1%}"
                                             for sh in (0.2, 0.3, 0.4, 0.5)))
    print("by case at the d gate (kept / found, 95 % within):")
    for hz, fps in DAMPING_CASES:
        for zeta in (0.005, 0.01, 0.02, 0.05):
            m = (rows[:, 0] == hz) & (rows[:, 2] == zeta)
            k = m & keep
            q = f"{100 * np.quantile(err[k], share):5.1f} %" if k.any() else "    -"
            print(f"  {hz:5g} Hz {fps:4g} fps zeta {zeta:5.3f}: {k.sum():4d} / {m.sum():4d}  {q}")
    return best_d


def main():
    ap = argparse.ArgumentParser(description="free_decay.faint() on simulated noise and rings.")
    ap.add_argument("--quick", action="store_true", help="a tenth of the videos")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--damping", action="store_true", help="only the damping ratio below the floor (faint_damping())")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    if args.damping:
        damping_table(rng, 5 if args.quick else 30)
        return
    noise_table(rng, 100 if args.quick else 1000)
    unsteady_table(rng, 20 if args.quick else 200)
    ring_table(rng, 10 if args.quick else 100)


if __name__ == "__main__":
    main()
