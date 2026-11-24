"""A part's response to its machine's own running speeds, from a slow-motion video: the machine stepped through its
speeds, holding each for a few seconds, or coasting down after it is switched off.

The part and a still point are followed as free_decay.py follows them (its Tracker, swing and floor). The swing is
cut into windows six cycles long, a quarter of a window apart. In each, the strongest frequency (free_decay.tone) is
what the part moves at, the running speed when the machine shakes it once a revolution, and its amplitude is the swing
at that frequency. The windows whose swing rises above the floor make the response curve, and its largest swing gives
the resonance the machine excites. For light damping, the frequency of the largest steady response lies within about
zeta squared of the natural frequency: below it for a force of constant size (Downey and Micheli, Vibration
Mechanics, eq. 3.93), above it for a rotating unbalance, whose force grows with the square of the speed. A coast-down
passes the peak before the part has fully answered, so it reads low: by 0.1 to 1.8 % on generated coast-downs from
20 to 6 Hz lasting 10 to 80 s, through a 12 Hz resonance with damping ratios of 0.01 to 0.05 (tests). It gives no
damping ratio: on the same coast-downs the width of the peak read it 3 to 29 % high or not at all, and a few speed
steps cannot resolve the peak. Tap the part for that (free_decay.py).

Usage: python3 speed_test.py FILM --part X Y --fixed X Y [--fps 240] [--plot response.png]
       (without --part and --fixed, it saves the first frame with a pixel grid to read them from)

SPDX-License-Identifier: MIT
Copyright (c) 2026 Jacek Garbuliński Blixild, blixild.com
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # free_decay.py beside it, also under python3 -I
import free_decay as fd  # noqa: E402
from free_decay import Note  # noqa: E402

CYCLES = 6  # a window's length, in cycles of the loudest frequency
HOPS = 4    # windows per window length: each starts a quarter of a window after the last
# below this many frames per swing the speed test warns: a resonance above half the frame rate folds back and reads as
# a slower one. Down to 2.4 frames per swing (100 Hz at 240 frames a second) it read the frequency within 0.6 % and the
# swing in full on generated videos (tests); the tap test's damping ratio needs four (free_decay.FEW_FRAMES)
NEAR_HALF = 2.5
# a resonance counts as found inside the speeds filmed when the swing falls to 1/sqrt(2) of its largest on both sides
# of it, the half-power points that bound a resonance's peak; otherwise it may lie beyond them, or the speeds were
# stepped too coarsely around it
HALF_POWER = 1 / np.sqrt(2)


def window_length(x, fps):
    """Frames in a window: CYCLES cycles of the strongest frequency in the two seconds around the largest swing."""
    k = int(np.argmax(np.abs(x)))
    f, _ = fd.tone(x[max(0, k - int(fps)):k + int(fps)], fps)
    if not np.isfinite(f):  # no frequency there (tone()): a window a second long instead
        return max(24, int(round(fps)))
    return max(24, int(round(CYCLES * fps / f)))


def response(x, fps, floor):
    """Each window's middle (s), frequency (Hz) and amplitude (px), and whether its swing rises above the floor."""
    n = window_length(x, fps)
    if len(x) < 2 * n:
        raise ValueError(Note("short", "the video is too short for a speed test: film at least a few seconds at each "
                              "speed, or the whole coast-down"))
    hop = max(1, n // HOPS)
    t, f, a = [], [], []
    for s in range(0, len(x) - n + 1, hop):
        fi, ai = fd.tone(x[s:s + n], fps)
        if not np.isfinite(fi):  # no frequency in this window (tone()): left out
            continue
        t.append((s + n / 2) / fps)
        f.append(fi)
        a.append(ai)
    t, f, a = np.array(t), np.array(f), np.array(a)
    return t, f, a, a > floor


def resonance(f, a, above, fps):
    """The frequency (Hz) and size (px) of the largest swing above the floor, and warnings."""
    if not above.any():
        raise ValueError(Note("below-floor", "the swing never rises above the floor: film closer, or film speeds "
                              "where the part visibly shakes"))
    F, A = f[above], a[above]
    k = int(np.argmax(A))
    peak, size = float(F[k]), float(A[k])
    notes = []
    for side, rest in (("below", a[above & (f < peak)]), ("above", a[above & (f > peak)])):  # windows below the floor
        # read random frequencies (a stopped machine) and would bracket any peak
        if not (rest <= HALF_POWER * size).any():
            notes.append(Note("edge-peak", f"the swing does not fall to 71 % of its largest {side} {peak:.2f} Hz "
                              "within the speeds filmed: the resonance may lie beyond them, or the speeds were stepped "
                              "too coarsely around it", side=side, f=peak))
    if peak > fps / NEAR_HALF:
        notes.append(Note("near-half", f"the largest swing is at {peak:.1f} Hz, close to half the frame rate "
                          f"({fps / 2:g} Hz): a faster resonance would read as a slower one there; film at a higher "
                          "frame rate to be sure", f=peak, half=fps / 2))
    return peak, size, notes


def resolution(f, a, above):
    """How finely the windows place the resonance (Hz): the gap from the largest swing's frequency to the nearest
    frequencies read on either side, the median of the two. The windows above the floor are grouped by frequency, within
    0.2 % of each other; when most windows sit in groups of two or more (speeds held in steps), only those groups count,
    so a window that straddles a step does not shrink the gap. A coast-down's windows lie about (rate of slowing) x 1.5 /
    f^2 apart in relative frequency; stepped speeds, a step apart. NaN when no other frequency was read."""
    peak = float(f[above][np.argmax(a[above])])
    freqs = np.sort(f[above])
    groups = np.split(freqs, np.flatnonzero(np.diff(freqs) > 0.002 * peak) + 1)
    if sum(len(g) for g in groups if len(g) > 1) > len(freqs) / 2:
        groups = [g for g in groups if len(g) > 1] or groups
    centers = np.array([np.median(g) for g in groups])
    k = int(np.argmin(np.abs(centers - peak)))
    gaps = [abs(centers[j] - centers[k]) for j in (k - 1, k + 1) if 0 <= j < len(centers)]
    return float(np.median(gaps)) if gaps else float("nan")


def read(part, fixed, fps, min_peak=None):
    """The swing, the jitter, the floor, the windows (middles, frequencies, amplitudes, above the floor) and the
    resonance (frequency, size, warnings), from the two points' shifts: what the script prints, for the command line
    and for a page that runs it."""
    x = fd.swing(part, fixed)
    noise, floor = fd.floor_of(x, fixed, min_peak)
    windows = response(x, fps, floor)
    peak, size, notes = resonance(windows[1], windows[2], windows[3], fps)
    note = fd.lost_note(part, fixed)
    if note:
        notes.append(note)
    return x, noise, floor, windows, (peak, size, notes)


def main():
    args = fd.arguments("A part's response to its machine's running speeds, from a video of the machine stepped "
                        "through its speeds or coasting down.",
                        "save the response curve and the swing to this image").parse_args()
    fps, stored, frames = fd.film(args)
    part, fixed = fd.track(frames, [args.part, args.fixed], args.box, report=True)
    timing = fd.uneven(args.frames.times)
    if timing:
        print(f"warning: {timing}")
    try:
        x, noise, floor, (t, f, a, above), (peak, size, notes) = read(part, fixed, fps, args.min_peak)
    except ValueError as e:
        raise SystemExit(str(e))
    print(f"frames {len(x)} at {fps:g} fps" + ("" if args.fps is None else f" (the file stores {stored:g})") +
          f"; jitter {noise:.3f} px; floor {floor:.2f} px")
    print(f"windows above the floor: {above.sum()} of {len(above)}, from {f[above].min():.2f} to "
          f"{f[above].max():.2f} Hz")
    spacing = resolution(f, a, above)
    print(f"largest swing {size:.1f} px at {peak:.3f} Hz ({60 * peak:.0f} rpm if the part moves once a revolution)"
          + (f"; the nearest other frequencies read lie {spacing:.3f} Hz away, the resolution" if spacing == spacing
             else ""))
    for w in notes:
        print(f"warning: {w}")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 5.5))
        top.plot(f[above], a[above], "o", ms=3)
        top.plot(f[~above], a[~above], "o", ms=2, color="0.7")
        top.axhline(floor, color="0.6", lw=0.6, ls=":")
        top.axvline(peak, color="0.4", lw=0.6)
        top.set_xlabel("frequency, Hz")
        top.set_ylabel("swing, px")
        bottom.plot(np.arange(len(x)) / fps, x, lw=0.5)
        bottom.set_xlabel("time, s")
        bottom.set_ylabel("swing, px")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)


if __name__ == "__main__":
    main()
