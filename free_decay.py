"""Natural frequency and damping ratio of a part, from a video of it ringing down after a tap or a release.

Pick two points in the first frame: one on the part that swings, one on something that does not move. The script
follows the texture around each point with a feature tracker (OpenCV: Shi-Tomasi corners, Lucas-Kanade optical
flow, checked forwards and backwards), measures the swing in pixels, and reads:
  - the damped natural frequency from the spectrum of the swing, while it rings above the floor,
  - the damping ratio from how fast the upper peaks shrink (the logarithmic decrement),
  - the noise floor from the frame-to-frame jitter of the point that does not move.
Neither result needs millimetres: frequency comes from time, damping from a ratio of peaks.

Usage:
  python3 free_decay.py VIDEO                                 saves VIDEO_first_frame.png with a pixel grid to read
                                                              the two points from
  python3 free_decay.py VIDEO --pick                          click the two points in a window (needs a desktop)
  python3 free_decay.py VIDEO --part X Y --fixed X Y [--fps 240] [--plot swing.png]

SPDX-License-Identifier: MIT
Copyright (c) 2026 Jacek Garbuliński Blixild, blixild.com
"""
import argparse
import itertools
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks

LK = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
BOX = 41  # px: the square around each point whose corners are followed; it must lie on the part (or the still object)
# The strongest corner in a square, as cv2.cornerMinEigenVal gives it for an 8-bit frame: printed or painted texture
# measured 0.18-0.23, plain grey with sensor noise 0.0001 (noise of 2 grey levels) to 0.002 (8 grey levels)
TEXTURE_STOP, TEXTURE_WARN = 0.005, 0.02


def read_video(video):
    """The frame rate a video file stores and its frames, one greyscale image at a time."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot read {video}: check the path, or that OpenCV can open this kind of file")

    def frames():
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    return
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        finally:
            cap.release()
    return cap.get(cv2.CAP_PROP_FPS), frames()


def grid_image(first, path, step=50):
    """Save the first frame with a labelled pixel grid, to read the two points' coordinates from."""
    img = cv2.cvtColor(first, cv2.COLOR_GRAY2BGR)
    h, w = first.shape
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h - 1), (0, 140, 255), 1)
        if x % (2 * step) == 0:
            cv2.putText(img, str(x), (x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 140, 255), 1)
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w - 1, y), (0, 140, 255), 1)
        if y % (2 * step) == 0 and y:
            cv2.putText(img, str(y), (2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 140, 255), 1)
    ok, png = cv2.imencode(".png", img)  # written by Python, so any file name works (cv2.imwrite fails on some)
    try:
        if not ok:
            raise OSError("cannot encode the image")
        Path(path).write_bytes(png.tobytes())
    except OSError as e:
        raise SystemExit(f"cannot write {path}: {e}")


def pick_points(first):
    """Two clicks in a window on the first frame: the part, then something that does not move."""
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise SystemExit("no screen here: run without --pick to get the first frame with a grid and read the points")
    title = "click the part, then something that does not move (Esc stops)"
    view, points = cv2.cvtColor(first, cv2.COLOR_GRAY2BGR), []

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))
            cv2.circle(view, (x, y), 8, (0, 140, 255), 2)
            cv2.imshow(title, view)
    try:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.imshow(title, view)
        cv2.setMouseCallback(title, click)
        while len(points) < 2 and cv2.waitKey(50) != 27 and cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) >= 1:
            pass
        cv2.destroyAllWindows()
    except cv2.error:
        raise SystemExit("no window can open here: run without --pick to get the first frame with a grid and read "
                         "the points")
    if len(points) < 2:
        raise SystemExit("two points are needed: the part, then something that does not move")
    (px, py), (fx, fy) = points
    print(f"points: --part {px} {py} --fixed {fx} {fy}")
    return (px, py), (fx, fy)


def track(frames, points, box=BOX, report=False):
    """Follow the corners inside a box around each point through greyscale frames; return each point's shift (px).

    Each point is followed from the last frame in which it was found, so a frame where it is lost (a blink, glare)
    leaves a gap, not an offset in every frame after it."""
    frames = iter(frames)
    first = next(frames)
    h, w = first.shape
    half = box // 2
    eigen = cv2.cornerMinEigenVal(first, 3)  # how well a patch at each pixel can be followed in both directions
    start, now, ref, shifts, reach, ids = [], [], [], [], [], []
    for x, y in points:
        if not (0 <= x < w and 0 <= y < h):
            raise SystemExit(f"the point ({x}, {y}) lies outside the {w} x {h} frame")
        mask = np.zeros_like(first)
        mask[max(y - half, 0):y + half + 1, max(x - half, 0):x + half + 1] = 255
        corners = cv2.goodFeaturesToTrack(first, maxCorners=30, qualityLevel=0.01, minDistance=3, mask=mask)
        strength = float(eigen[max(y - half, 0):y + half + 1, max(x - half, 0):x + half + 1].max())
        if corners is None or len(corners) < 3 or strength < TEXTURE_STOP:
            raise SystemExit(f"too little texture to track around ({x}, {y}): pick another point or add a sticker")
        if report:
            print(f"point ({x}, {y}): {len(corners)} corners in its {box} px square")
            if strength < TEXTURE_WARN:
                print(f"warning: the texture around ({x}, {y}) is faint; the tracker may follow noise: pick a point on "
                      "sharper detail, or add a sticker")
        start.append(corners.reshape(-1, 2))
        now.append(corners)
        ref.append(first)
        shifts.append([np.zeros(2)])
        reach.append(np.zeros(len(corners)))  # each corner's largest shift while it was followed, px
        ids.append(np.arange(len(corners)))   # which of the first frame's corners are still followed
    for gray in frames:
        for i in range(len(points)):
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(ref[i], gray, now[i], None, **LK)
            back, st_back, _ = cv2.calcOpticalFlowPyrLK(gray, ref[i], nxt, None, **LK)
            # keep a corner only if following it back lands within half a pixel of where it was
            good = (st.ravel() == 1) & (st_back.ravel() == 1) & \
                   (np.linalg.norm((back - now[i]).reshape(-1, 2), axis=1) < 0.5)
            if good.sum() < 3:
                shifts[i].append(np.full(2, np.nan))  # lost in this frame: the next one is followed from ref[i]
                continue
            start[i], now[i], ref[i], ids[i] = start[i][good], nxt[good], gray, ids[i][good]
            moved = now[i].reshape(-1, 2) - start[i]
            reach[i][ids[i]] = np.maximum(reach[i][ids[i]], np.linalg.norm(moved, axis=1))
            shifts[i].append(np.median(moved, axis=0))
    if report:
        for (x, y), r in zip(points, reach):
            # a square that spans the part and something still holds corners that moved and corners that did not
            top = np.percentile(r, 90) if len(r) else 0  # robust to a corner that slipped onto other texture
            if top > 2 and (r < 0.2 * top).sum() >= 3 and (r > 0.5 * top).sum() >= 3:
                print(f"warning: the square around ({x}, {y}) holds corners that moved and corners that stayed still; "
                      "it may cover more than the part: pick a point further inside it, or use a smaller --box")
    return [np.array(s) for s in shifts]


def positive_first(x):
    """The signal turned so that its largest excursion is positive."""
    return x if x[np.argmax(np.abs(x))] > 0 else -x


def swing(part, fixed):
    """The part's motion relative to the fixed point, along the direction it swings most (px)."""
    rel = part - fixed  # also removes small shifts of the camera
    ok = ~np.isnan(rel).any(axis=1)
    if ok.sum() < len(rel) // 2:
        raise SystemExit("a point was lost in more than half the frames: pick points on clear texture")
    rel[~ok] = np.nan
    centred = rel - np.nanmean(rel[ok][-len(rel) // 10:], axis=0)  # zero = where the part comes to rest
    _, _, axes = np.linalg.svd(centred[ok], full_matrices=False)
    x = centred @ axes[0]
    idx = np.arange(len(x))
    x[~ok] = np.interp(idx[~ok], idx[ok], x[ok])  # bridge frames where a point was lost
    return positive_first(x)


def tops(x, peaks):
    """The top of each peak, from a parabola through it and the frames either side."""
    a, b, c = x[peaks - 1], x[peaks], x[peaks + 1]
    return b - (a - c) ** 2 / (8 * (a - 2 * b + c))


def frequency(x, fps):
    """Damped natural frequency (Hz): the highest peak of the spectrum above three cycles in the record, refined
    between bins. A straight-line trend is taken off first, so a slow drift or a bridged gap is not read as ringing."""
    if len(x) < 12 or not np.all(np.isfinite(x)):
        raise ValueError("too few swings to find a frequency: the largest swing should come at the knock, not later")
    t = np.arange(len(x))
    flat = x - np.polyval(np.polyfit(t, x, 1), t)
    n = 8 * len(x)
    spectrum = np.abs(np.fft.rfft(flat * np.hanning(len(x)), n))
    lowest = 3 * 8  # three cycles in the record, in the padded spectrum's bins
    k = np.argmax(spectrum[lowest:-1]) + lowest
    a, b, c = np.log(spectrum[k - 1:k + 2])
    return (k + 0.5 * (a - c) / (a - 2 * b + c)) * fps / n


def damping(x, fps, fd, floor):
    """Damping ratio, logarithmic decrement, the peaks used and how straight their logarithms lie (R squared).

    Each swing's amplitude is half the drop from an upper peak to the troughs either side, so a rest position that is
    off by a constant, or drifts slowly, does not change it; its logarithm falls by the decrement every cycle."""
    spacing = max(1, 0.7 * fps / fd)
    peaks, _ = find_peaks(x, distance=spacing)
    troughs, _ = find_peaks(-x, distance=spacing)
    peaks = peaks[(peaks > 0) & (peaks < len(x) - 1)]
    troughs = troughs[(troughs > 0) & (troughs < len(x) - 1)]
    after = np.searchsorted(troughs, peaks)
    between = (after > 0) & (after < len(troughs))  # a trough on each side
    peaks, after = peaks[between], after[between]
    bottom = -tops(-x, troughs)
    height = (tops(x, peaks) - (bottom[after - 1] + bottom[after]) / 2) / 2
    keep = height > floor  # small peaks carry the tracker's error and bend the fit
    peaks, height = peaks[keep], height[keep]
    if len(peaks) < 3:
        raise ValueError("fewer than three peaks above the floor: film closer, tap harder, check --fps, or lower the "
                         "floor with --min-peak")
    t, logs = peaks / fps, np.log(height)
    slope, lift = np.polyfit(t, logs, 1)  # 1/s
    residual = logs - (slope * t + lift)
    straight = 1 - residual.var() / logs.var() if logs.var() > 0 else 1.0
    decrement = -slope / fd  # logarithmic decrement, per cycle
    return decrement / np.sqrt(4 * np.pi ** 2 + decrement ** 2), decrement, peaks, straight


def noise_of(fixed):
    """How much a point that does not move seems to move from one frame to the next (px); a slow wobble of the
    camera is left out, since the part's swing is taken relative to that point anyway."""
    return np.nanstd(np.diff(fixed, axis=0), axis=0).max() / np.sqrt(2)


def ringing_part(x, fps, floor):
    """Where the part rings above the floor: from its largest swing (the knock) to the end of the first unbroken run
    of swings above the floor. A quiet start, a quiet tail and a later glitch are left out, so the spectrum's window
    falls on the ringing."""
    above = np.flatnonzero(np.abs(x) > floor)
    if len(above) == 0:
        raise ValueError("the swing never rises above the floor: film closer or tap harder")
    start = int(np.argmax(np.abs(x)))
    run = above[above >= start]
    period = fps / frequency(x[start:run[-1] + 1], fps)  # a first estimate, to find gaps of several swings
    gaps = np.flatnonzero(np.diff(run) > 3 * period)
    end = run[gaps[0]] if len(gaps) else run[-1]
    return start, end + 1


def measure(x, fps, floor):
    """Damped natural frequency, natural frequency, damping ratio, decrement, the peaks used (indices into x) and any
    warnings, read where the part rings above the floor."""
    start, end = ringing_part(x, fps, floor)
    ringing = x[start:end]
    fd = frequency(ringing, fps)
    if not 0 < fd < fps / 2:
        raise ValueError(f"no clear frequency in the swing ({fd:.3f} Hz at {fps:g} fps): check --fps and the points")
    zeta, decrement, peaks, straight = damping(ringing, fps, fd, floor)
    if zeta <= 0:
        raise ValueError("the peaks do not shrink: two resonances may be beating, or the film holds no ring-down")
    warnings = []
    if straight < 0.95:
        warnings.append(f"the peaks' logarithms do not lie on a straight line (R squared {straight:.2f}): two "
                        "resonances may share the swing, and the damping ratio is doubtful")
    if np.abs(x[-len(x) // 10:]).max() > floor:
        warnings.append("the part still swings in the last tenth of the film, which the script takes as rest: film "
                        "until it stops, or the damping ratio may read low")
    if fd > fps / 6:
        warnings.append(f"fewer than six frames per swing ({fps / fd:.1f}): the damping ratio may be off")
    return fd, fd / np.sqrt(1 - zeta ** 2), zeta, decrement, peaks + start, warnings


def main():
    ap = argparse.ArgumentParser(description="Natural frequency and damping ratio from a video of a part ringing down.")
    ap.add_argument("video")
    ap.add_argument("--part", type=int, nargs=2, metavar=("X", "Y"), help="a point on the part that swings, px")
    ap.add_argument("--fixed", type=int, nargs=2, metavar=("X", "Y"), help="a point on something that does not move")
    ap.add_argument("--pick", action="store_true", help="click the two points in a window instead")
    ap.add_argument("--fps", type=float, help="the rate the video was filmed at, frames a second, when the file "
                    "stores another (a slow-motion file may store the rate it plays back at)")
    ap.add_argument("--box", type=int, default=BOX, help="size of the square tracked around each point, px; the whole "
                    "square must lie on the part (or on the still object), or still corners outvote moving ones")
    ap.add_argument("--min-peak", type=float, help="smallest peak used, px; by default the larger of 1 px and ten "
                    "times the frame-to-frame jitter (of the still point, or of the swing once it has settled)")
    ap.add_argument("--start", type=float, default=0, help="seconds into the file, on its own clock, to start at")
    ap.add_argument("--end", type=float, help="seconds into the file, on its own clock, to stop at")
    ap.add_argument("--plot", help="save the swing and the peaks used to this image")
    args = ap.parse_args()
    if args.fps is not None and args.fps <= 0:
        raise SystemExit("--fps must be above zero")
    if args.box < 7:
        raise SystemExit("--box must be at least 7 px")
    if args.min_peak is not None and args.min_peak <= 0:
        raise SystemExit("--min-peak must be above zero")
    if args.end is not None and args.end <= args.start:
        raise SystemExit("--end must come after --start")
    if args.plot and not Path(args.plot).resolve().parent.is_dir():
        raise SystemExit(f"no folder for {args.plot}")

    stored, frames = read_video(args.video)
    if args.start or args.end is not None:  # keep the frames between --start and --end, by the file's own clock
        if not stored or stored <= 0:
            raise SystemExit("the file stores no frame rate, so --start and --end cannot be placed")
        frames = itertools.islice(frames, int(round(args.start * stored)),
                                  None if args.end is None else int(round(args.end * stored)))
    first = next(frames, None)
    if first is None:
        raise SystemExit(f"no frames in {args.video}")
    if args.pick:
        args.part, args.fixed = pick_points(first)
    elif not (args.part and args.fixed):
        grid = Path(args.video).with_name(Path(args.video).stem + "_first_frame.png")
        grid_image(first, str(grid))
        raise SystemExit(f"saved {grid} with a pixel grid: read a point on the part and a point on something that "
                         "does not move, then run again with --part X Y --fixed X Y (or use --pick)")
    fps = args.fps or stored
    if not fps or fps <= 0:
        raise SystemExit("the file stores no frame rate: give the rate it was filmed at with --fps")

    part, fixed = track(itertools.chain([first], frames), [args.part, args.fixed], args.box, report=True)
    lost = np.isnan(part).any(axis=1) | np.isnan(fixed).any(axis=1)
    x = swing(part, fixed)
    # the tracker's jitter: the still point's, unless a shaking camera inflates it, then the settled swing's
    settled = x[-len(x) // 10:]
    noise = min(noise_of(fixed), np.std(np.diff(settled)) / np.sqrt(2)) if len(settled) > 2 else noise_of(fixed)
    floor = args.min_peak if args.min_peak is not None else max(10 * noise, 1.0)
    try:
        fd, fn, zeta, decrement, peaks, warnings = measure(x, fps, floor)
    except ValueError as e:
        raise SystemExit(str(e))
    if lost.mean() > 0.02:
        warnings.append(f"a point was lost in {lost.sum()} of {len(x)} frames, bridged by a straight line: check them")
    print(f"frames {len(x)} at {fps:g} fps" + ("" if args.fps is None else f" (the file stores {stored:g})") +
          f"; jitter {noise:.3f} px; floor {floor:.2f} px; largest swing {np.abs(x).max():.1f} px")
    print(f"damped natural frequency {fd:.3f} Hz; natural frequency {fn:.3f} Hz")
    print(f"logarithmic decrement {decrement:.4f}; damping ratio {zeta:.4f} (from {len(peaks)} peaks)")
    for w in warnings:
        print(f"warning: {w}")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = np.arange(len(x)) / fps
        plt.figure(figsize=(8, 3))
        plt.plot(t, x, lw=0.8)
        plt.plot(peaks / fps, x[peaks], "o", ms=3)
        plt.axhline(floor, color="0.6", lw=0.6, ls=":")
        plt.xlabel("time, s")
        plt.ylabel("swing, px")
        plt.tight_layout()
        plt.savefig(args.plot, dpi=150)


if __name__ == "__main__":
    main()
