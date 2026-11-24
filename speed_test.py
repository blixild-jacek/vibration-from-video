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
# Below a pixel (the frequency alone, as the tap test's free_decay.faint()), the rule fixed from simulated noise before
# any real video was read with it (faint_noise.py --speed). The swing's drift is taken off and cut into windows; in each,
# the peak of its spectrum is set against the noise in the bins around it, LOCAL_SPAN bins either side of the peak with
# LOCAL_GUARD bins kept clear for the peak's own width (a constant-false-alarm-rate test, as radar sets a target against
# the cells around it): a reference that follows the noise where the peak is, so a burst of noise or a camera's slow
# drift raises it too, and that a strong response elsewhere in the video does not raise (against the median window,
# a strong run-up held at its top speed hid its own resonance in simulation). A window stands out when its peak reaches
# FAINT_LOCAL times that noise, it holds at least FAINT_CYCLES cycles in the window (the bottom of the search, three
# cycles, plus the guard: lower, the drift's leftover rises steeply and the peak would stand over noise on one side only;
# every high ratio of noise alone in simulation sat there), it lies below fps / NEAR_HALF, and the two windows either
# side of it, which share three
# quarters of its frames, read the same frequency within one bin (a response to the running speed changes smoothly,
# while noise jumps from window to window); the largest such window is the resonance when the swing falls to HALF_POWER
# of it both before and after it in time, within the video: a run-up or a coast-down passes a resonance and leaves it,
# while a speed held to the video's end keeps its swing (a real rotor's top speed read so with the floor lowered,
# worklog of 24 Nov 2026).
LOCAL_GUARD = 2   # bins of the unpadded spectrum, either side of the peak: a Hann window's main lobe
LOCAL_SPAN = 12   # bins either side of the peak whose median is the noise around it
LOCAL_LEAST = 8   # the fewest bins around the peak that make a noise reference (a 24-frame window holds 12 in all: noise
                  # alone stood 15 times over a handful of them in simulation)
FAINT_WINDOW = 2 * CYCLES  # a window's length below a pixel, in cycles: twice as long, so its spectrum holds
                           # bins enough around the peak to measure the noise there
FAINT_CYCLES = fd.LOWEST // 8 + LOCAL_GUARD  # 5
FAINT_LOCAL = 30.0  # about twice the largest ratio of noise alone in simulation (16.1 in 12,000 steady videos, 11.9 in
                   # 1,120 unsteady ones), as the tap test's free_decay.FAINT_RATIO is about twice its noise's largest


def window_length(x, fps, cycles=CYCLES):
    """Frames in a window: `cycles` cycles of the strongest frequency in the two seconds around the largest swing."""
    k = int(np.argmax(np.abs(x)))
    f, _ = fd.tone(x[max(0, k - int(fps)):k + int(fps)], fps)
    if not np.isfinite(f):  # no frequency there (tone()): a window a second long instead
        return max(24, int(round(fps)))
    return max(24, int(round(cycles * fps / f)))


def windows(x, fps, cycles=CYCLES):
    """The windows' first frames, middles (s), frequencies (Hz) and amplitudes (px), and their length in frames:
    `cycles` cycles of the frequency at the largest swing (window_length())."""
    n = window_length(x, fps, cycles)
    if len(x) < 2 * n:
        raise ValueError(Note("short", "the video is too short for a speed test: film at least a few seconds at each "
                              "speed, or the whole coast-down"))
    hop = max(1, n // HOPS)
    first, t, f, a = [], [], [], []
    for s in range(0, len(x) - n + 1, hop):
        fi, ai = fd.tone(x[s:s + n], fps)
        if not np.isfinite(fi):  # no frequency in this window (tone()): left out
            continue
        first.append(s)
        t.append((s + n / 2) / fps)
        f.append(fi)
        a.append(ai)
    return np.array(first, int), np.array(t), np.array(f), np.array(a), n


def response(x, fps, floor):
    """Each window's middle (s), frequency (Hz) and amplitude (px), and whether its swing rises above the floor."""
    _, t, f, a, _ = windows(x, fps)
    return t, f, a, a > floor


def local(x):
    """A record's spectral peak (spectrum(), above three cycles in the record, as tone() searches) over the median of
    the bins around it: LOCAL_SPAN bins of the unpadded spectrum either side, without the LOCAL_GUARD bins next to the
    peak; NaN when fewer than LOCAL_LEAST bins lie there. In white noise its size does not depend on the noise's
    level."""
    s = fd.spectrum(x)[0][fd.LOWEST:-1]
    k = int(np.argmax(s))
    pad = 8  # spectrum() pads to 8 times the record
    idx = np.arange(len(s))
    around = (np.abs(idx - k) > LOCAL_GUARD * pad) & (np.abs(idx - k) <= LOCAL_SPAN * pad)
    return float(s[k] / np.median(s[around])) if around.sum() >= LOCAL_LEAST * pad else float("nan")


def faint_response(x, fps, bar=None):
    """A resonance whose swing stays under the floor: the frequency alone. The swing's drift is taken off
    (free_decay.drift_off) and cut into windows() FAINT_WINDOW cycles long; windows holding a repeated frame (a duplicated frame tracks exactly
    still) are left out. A window stands out when its spectral peak is `bar` (FAINT_LOCAL) times the noise around it
    (local()), it holds FAINT_CYCLES cycles in the window, it lies below fps / NEAR_HALF, and the windows either side of
    it read its frequency within one bin (fps / window length). Returns (frequency Hz, swing px, its local() ratio, whether the swing falls to HALF_POWER of it
    both before and after in time, the windows' middles, frequencies, amplitudes, which stand out, and every window's
    local() ratio, NaN where a check other than the ratio fails) for the largest window that stands out; else for the
    largest window, with the mask empty; None when there are too few windows."""
    bar = FAINT_LOCAL if bar is None else bar
    y = fd.drift_off(x, fps)
    first, t, f, a, n = windows(y, fps, FAINT_WINDOW)
    still = np.r_[False, np.diff(x) == 0]
    ok = np.array([still[s + 1:s + n].sum() <= 1 for s in first], bool)
    if ok.sum() < 3:
        return None
    ratio = np.array([local(y[s:s + n]) for s in first])
    near = np.abs(np.diff(f)) <= fps / n  # each window and the next read the same frequency within a bin
    steady = np.r_[False, near] & np.r_[near, False] & np.r_[False, ok[:-1]] & np.r_[ok[1:], False]
    fits = (f * n / fps >= FAINT_CYCLES) & (f < fps / NEAR_HALF) & np.isfinite(ratio)
    stands = ok & steady & fits & (ratio >= bar)
    k = int(np.flatnonzero(stands)[np.argmax(a[stands])]) if stands.any() else int(np.flatnonzero(ok)[np.argmax(a[ok])])
    low = HALF_POWER * a[k]
    rises_and_falls = bool((ok[:k] & (a[:k] <= low)).any() and (ok[k + 1:] & (a[k + 1:] <= low)).any())
    checked = np.where(ok & steady & fits, ratio, np.nan)
    return float(f[k]), float(a[k]), float(ratio[k]), rises_and_falls, t, f, a, stands, checked


def faint_passes(found):
    """Whether faint_response() found a resonance: a window stands out, and its swing rises and falls around it."""
    return bool(found) and bool(found[7].any()) and found[3]


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
    notes += near_half(peak, fps)
    return peak, size, notes


def near_half(peak, fps):
    """A warning, in a list, when the largest swing lies close to half the frame rate; else an empty list."""
    if peak <= fps / NEAR_HALF:
        return []
    return [Note("near-half", f"the largest swing is at {peak:.1f} Hz, close to half the frame rate ({fps / 2:g} Hz): "
                 "a faster resonance would read as a slower one there; film at a higher frame rate to be sure",
                 f=peak, half=fps / 2)]


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
    and for a page that runs it. When no window rises above the floor, the frequency alone from faint_response(), its
    windows that stand out in place of those above the floor, with a "frequency-only" warning; when nothing stands out,
    or the largest swing does not rise and fall within the video, the refusal says which."""
    x = fd.swing(part, fixed)
    noise, floor = fd.floor_of(x, fixed, min_peak)
    windows = response(x, fps, floor)
    try:
        peak, size, notes = resonance(windows[1], windows[2], windows[3], fps)
    except ValueError as e:
        if getattr(e.args[0], "code", None) != "below-floor":
            raise
        found = faint_response(x, fps)
        if not found or not found[7].any():
            raise
        peak, size, ratio, both, t, f, a, stands, _ = found
        if not both:
            raise ValueError(Note("held-speed", f"the swing never rises above the floor of {floor:.2f} px, and its "
                                  f"largest, at {peak:.2f} Hz, lasts to the video's start or end: a speed held there, "
                                  "not a resonance passed. Film the whole run-up or coast-down through the speeds, or "
                                  "film closer", floor=floor, f=peak))
        windows = (t, f, a, stands)
        notes = [Note("frequency-only", f"only the frequency is read: the swing never rises above the floor of "
                      f"{floor:.2f} px, but at {peak:.2f} Hz it stands {ratio:.1f} times above the noise around it, "
                      f"where {FAINT_LOCAL:g} is needed, and it rises and falls around that speed. Its size is too "
                      "small to trust: film closer for it", floor=floor, f=peak, ratio=ratio, needed=FAINT_LOCAL)]
        notes += near_half(peak, fps)
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
    only = any(getattr(w, "code", "") == "frequency-only" for w in notes)
    print(f"windows {'standing out of the noise' if only else 'above the floor'}: {above.sum()} of {len(above)}, "
          f"from {f[above].min():.2f} to {f[above].max():.2f} Hz")
    spacing = resolution(f, a, above)
    print(f"largest swing {size:.{2 if only else 1}f} px at {peak:.3f} Hz ({60 * peak:.0f} rpm if the part moves once a revolution)"
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
