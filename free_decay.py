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
# px kept around each point: the tracker follows each point in this window of the frame, not the whole frame. It holds
# the point's square, the tracker's coarsest level (21 px at an eighth of the size, about 80 px) and a swing of up to
# REACH px, beyond which the tracker warns
MARGIN = 256
REACH = MARGIN - 100


class Note(str):
    """A warning or error for the reader: the sentence the script prints, with a code and its values, so that a page
    running these functions can word it in its own language."""

    def __new__(cls, code, text, **values):
        note = super().__new__(cls, text)
        note.code, note.values = code, values
        return note


class Frames:
    """A video's frames, one greyscale image at a time, and the time of each frame read so far (ms, as OpenCV gives
    it), to tell a film saved at a variable frame rate (uneven())."""

    def __init__(self, cap):
        self.cap, self.times = cap, []

    def __iter__(self):
        return self

    def __next__(self):
        ok, frame = self.cap.read() if self.cap.isOpened() else (False, None)
        if not ok:
            self.cap.release()
            raise StopIteration
        self.times.append(self.cap.get(cv2.CAP_PROP_POS_MSEC))
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame


def read_video(video):
    """The frame rate a video file stores and its frames (Frames), one greyscale image at a time."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot read {video}: check the path, or that OpenCV can open this kind of file")
    return cap.get(cv2.CAP_PROP_FPS), Frames(cap)


UNEVEN = 0.01  # frame times further than this from their usual spacing: a variable frame rate


def uneven(times):
    """A warning when the frames' times are unevenly spaced (a film saved at a variable frame rate, which the frame
    rate the file stores misrepresents: the frequency comes out wrong), or None."""
    steps = np.diff(np.asarray(times, float))
    if len(steps) < 3 or not np.all(np.isfinite(steps)):
        return None
    usual = np.median(steps)
    spread = float(np.abs(steps - usual).max() / usual) if usual > 0 else 0.0
    if spread > UNEVEN:
        return Note("uneven", f"the film's frames are unevenly spaced in time (by up to {100 * spread:.0f} %): it was "
                    "saved at a variable frame rate, so the frequency may be off; save it at a constant frame rate "
                    "first", spread=round(100 * spread))
    return None


def draw_grid(gray, step=50):
    """A greyscale frame as a colour image with a pixel grid every step pixels, labelled every second line."""
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    h, w = gray.shape
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h - 1), (0, 140, 255), 1)
        if x % (2 * step) == 0:
            cv2.putText(img, str(x), (x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 140, 255), 1)
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w - 1, y), (0, 140, 255), 1)
        if y % (2 * step) == 0 and y:
            cv2.putText(img, str(y), (2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 140, 255), 1)
    return img


def grid_image(first, path, step=50):
    """Save the first frame with a labelled pixel grid, to read the two points' coordinates from."""
    # written by Python, so any file name works (cv2.imwrite fails on some)
    ok, png = cv2.imencode(".png", draw_grid(first, step))
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


class Tracker:
    """Follows the corners inside a box around each point, one greyscale frame at a time: track() runs it over a
    film, and a page that decodes a film itself can hand it each frame as it comes.

    Each point is followed from the last frame in which it was found, so a frame where it is lost (a blink, glare)
    leaves a gap, not an offset in every frame after it. Each point is followed in its own window of the frame
    (windows: x0, y0, x1, y1), so a large frame costs little more than a small one; step_crops() takes those windows
    alone, cut from the frame by whoever decodes it."""

    def __init__(self, first, points, box=BOX, margin=MARGIN):
        h, w = first.shape
        half = box // 2
        self.points, self.box = [tuple(p) for p in points], box
        self.windows = []
        self.start, self.now, self.ref, self.moves, self.reach, self.ids = [], [], [], [], [], []
        self.corners, self.faint = [], []  # corners found in each square; points whose texture is faint
        for x, y in points:
            if not (0 <= x < w and 0 <= y < h):
                raise SystemExit(Note("outside", f"the point ({x}, {y}) lies outside the {w} x {h} frame",
                                      x=x, y=y, w=w, h=h))
            x0, y0 = max(x - margin, 0), max(y - margin, 0)
            self.windows.append((x0, y0, min(x + margin + 1, w), min(y + margin + 1, h)))
            crop = first[y0:self.windows[-1][3], x0:self.windows[-1][2]]
            cx, cy = x - x0, y - y0  # the point in its window
            eigen = cv2.cornerMinEigenVal(crop, 3)  # how well a patch at each pixel can be followed both ways
            mask = np.zeros_like(crop)
            mask[max(cy - half, 0):cy + half + 1, max(cx - half, 0):cx + half + 1] = 255
            corners = cv2.goodFeaturesToTrack(crop, maxCorners=30, qualityLevel=0.01, minDistance=3, mask=mask)
            strength = float(eigen[max(cy - half, 0):cy + half + 1, max(cx - half, 0):cx + half + 1].max())
            if corners is None or len(corners) < 3 or strength < TEXTURE_STOP:
                raise SystemExit(Note("texture", f"too little texture to track around ({x}, {y}): pick another point "
                                      "or add a sticker", x=x, y=y))
            self.corners.append(len(corners))
            if strength < TEXTURE_WARN:
                self.faint.append((x, y))
            self.start.append(corners.reshape(-1, 2))
            self.now.append(corners)
            self.ref.append(crop)
            self.moves.append([np.zeros(2)])
            self.reach.append(np.zeros(len(corners)))  # each corner's largest shift while it was followed, px
            self.ids.append(np.arange(len(corners)))   # which of the first frame's corners are still followed

    def step(self, gray):
        """Follow every point into the next frame."""
        self.step_crops([gray[y0:y1, x0:x1] for x0, y0, x1, y1 in self.windows])

    def step_crops(self, crops):
        """Follow every point into the next frame, given as each point's window of it."""
        for i, gray in enumerate(crops):
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.ref[i], gray, self.now[i], None, **LK)
            back, st_back, _ = cv2.calcOpticalFlowPyrLK(gray, self.ref[i], nxt, None, **LK)
            # keep a corner only if following it back lands within half a pixel of where it was
            good = (st.ravel() == 1) & (st_back.ravel() == 1) & \
                   (np.linalg.norm((back - self.now[i]).reshape(-1, 2), axis=1) < 0.5)
            if good.sum() < 3:
                self.moves[i].append(np.full(2, np.nan))  # lost in this frame: the next one is followed from ref[i]
                continue
            self.start[i], self.now[i], self.ref[i], self.ids[i] = \
                self.start[i][good], nxt[good], gray, self.ids[i][good]
            moved = self.now[i].reshape(-1, 2) - self.start[i]
            self.reach[i][self.ids[i]] = np.maximum(self.reach[i][self.ids[i]], np.linalg.norm(moved, axis=1))
            self.moves[i].append(np.median(moved, axis=0))

    def shifts(self):
        """Each point's shift in every frame so far (px), NaN where it was lost."""
        return [np.array(s) for s in self.moves]

    def far(self):
        """The points that moved more than REACH px from where they started, near the edge of their window, where
        the tracker may lose them."""
        return [p for p, r in zip(self.points, self.reach) if len(r) and r.max() > REACH]

    def mixed(self):
        """The points whose square holds corners that moved and corners that stayed still, which suggests it covers
        more than the part."""
        out = []
        for p, r in zip(self.points, self.reach):
            top = np.percentile(r, 90) if len(r) else 0  # robust to a corner that slipped onto other texture
            if top > 2 and (r < 0.2 * top).sum() >= 3 and (r > 0.5 * top).sum() >= 3:
                out.append(p)
        return out


def faint_warning(p):
    return Note("faint", f"the texture around ({p[0]}, {p[1]}) is faint; the tracker may follow noise: pick a point on "
                "sharper detail, or add a sticker", x=p[0], y=p[1])


def far_warning(p):
    return Note("far", f"the part around ({p[0]}, {p[1]}) moved more than {REACH} px, near the edge of the window the "
                "tracker follows it in: film from further away", x=p[0], y=p[1], reach=REACH)


def mixed_warning(p):
    return Note("mixed", f"the square around ({p[0]}, {p[1]}) holds corners that moved and corners that stayed "
                "still; it may cover more than the part: pick a point further inside it, or use a smaller --box",
                x=p[0], y=p[1])


def track(frames, points, box=BOX, report=False):
    """Follow the corners inside a box around each point through greyscale frames; return each point's shift (px)."""
    frames = iter(frames)
    tracker = Tracker(next(frames), points, box)
    if report:
        for (x, y), n in zip(tracker.points, tracker.corners):
            print(f"point ({x}, {y}): {n} corners in its {box} px square")
            if (x, y) in tracker.faint:
                print(f"warning: {faint_warning((x, y))}")
    for gray in frames:
        tracker.step(gray)
    if report:
        for p in tracker.far():
            print(f"warning: {far_warning(p)}")
        for p in tracker.mixed():
            print(f"warning: {mixed_warning(p)}")
    return tracker.shifts()


def positive_first(x):
    """The signal turned so that its largest excursion is positive."""
    return x if x[np.argmax(np.abs(x))] > 0 else -x


def swing(part, fixed):
    """The part's motion relative to the fixed point, along the direction it swings most (px)."""
    rel = part - fixed  # also removes small shifts of the camera
    ok = ~np.isnan(rel).any(axis=1)
    if ok.sum() < len(rel) // 2:
        raise SystemExit(Note("lost-half", "a point was lost in more than half the frames: pick points on clear "
                              "texture"))
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


def tone(x, fps):
    """The strongest frequency in a record (Hz), the highest peak of its spectrum above three cycles in the record,
    refined between bins, and that tone's amplitude (px). A straight-line trend is taken off first, so a slow drift or
    a bridged gap is not read as ringing. NaN for both when the highest point lies at the bottom of the searched
    range, or the parabola's top more than half a bin beyond the three bins it is drawn through: no peak to place."""
    if len(x) < 12 or not np.all(np.isfinite(x)):
        raise ValueError(Note("few-swings", "too few swings to find a frequency: the largest swing should come at "
                              "the knock, not later"))
    t = np.arange(len(x))
    flat = x - np.polyval(np.polyfit(t, x, 1), t)
    n = 8 * len(x)
    window = np.hanning(len(x))
    spectrum = np.abs(np.fft.rfft(flat * window, n))
    lowest = 3 * 8  # three cycles in the record, in the padded spectrum's bins
    k = np.argmax(spectrum[lowest:-1]) + lowest
    if k == lowest:  # the highest point at the bottom of the searched range: the spectrum still rises below it, so
        return float("nan"), float("nan")  # the peak lies below three cycles in the record (no frequency)
    a, b, c = np.log(spectrum[k - 1:k + 2])
    between = 0.5 * (a - c) / (a - 2 * b + c)  # the parabola's top, in bins from k
    if not abs(between) <= 1.5:  # a top beyond the three bins it was drawn through, by more than the half bin a peak
        # on the outer bin can give (at half the frame rate, 1.002 bins), would be extrapolated: the audit's case, -6.4
        return float("nan"), float("nan")
    top = np.exp(b - 0.25 * (a - c) * between)  # and its height
    return (k + between) * fps / n, 2 * top / window.sum()


def frequency(x, fps):
    """Damped natural frequency (Hz): the strongest frequency in the record (tone())."""
    return tone(x, fps)[0]


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
        raise ValueError(Note("few-peaks", "fewer than three peaks above the floor: film closer, tap harder, check "
                              "--fps, or lower the floor with --min-peak"))
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
        raise ValueError(Note("below-floor", "the swing never rises above the floor: film closer or tap harder"))
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
        raise ValueError(Note("no-frequency", f"no clear frequency in the swing ({fd:.3f} Hz at {fps:g} fps): check "
                              "--fps and the points", fd=fd, fps=fps))
    zeta, decrement, peaks, straight = damping(ringing, fps, fd, floor)
    if zeta <= 0:
        raise ValueError(Note("no-decay", "the peaks do not shrink: two resonances may be beating, or the film "
                              "holds no ring-down"))
    warnings = []
    if straight < 0.95:
        warnings.append(Note("not-straight", f"the peaks' logarithms do not lie on a straight line (R squared "
                             f"{straight:.2f}): two resonances may share the swing, and the damping ratio is doubtful",
                             r2=straight))
    if np.abs(x[-len(x) // 10:]).max() > floor:
        warnings.append(Note("still-swinging", "the part still swings in the last tenth of the film, which the "
                             "script takes as rest: film until it stops, or the damping ratio may read low"))
    if fd > fps / 6:
        warnings.append(Note("few-frames", f"fewer than six frames per swing ({fps / fd:.1f}): the damping ratio may "
                             "be off", per_swing=fps / fd))
    return fd, fd / np.sqrt(1 - zeta ** 2), zeta, decrement, peaks + start, warnings


def floor_of(x, fixed, min_peak=None):
    """The tracker's jitter (px) and the floor below which a swing is not trusted: the larger of 1 px and ten times
    the jitter, unless min_peak is given. The jitter is the still point's, unless a shaking camera inflates it, then
    the settled swing's."""
    settled = x[-len(x) // 10:]
    noise = min(noise_of(fixed), np.std(np.diff(settled)) / np.sqrt(2)) if len(settled) > 2 else noise_of(fixed)
    return noise, (min_peak if min_peak is not None else max(10 * noise, 1.0))


def lost_note(part, fixed, span=None):
    """A warning when a point was lost in any frame of span (the frames read, a slice), whose peaks a bridged gap
    can bend, or in more than 2 % of all the frames; or None."""
    lost = np.isnan(part).any(axis=1) | np.isnan(fixed).any(axis=1)
    if span is not None and lost[span].any():
        inside = lost[span]
        return Note("lost-some", f"a point was lost in {inside.sum()} of the {len(inside)} frames read, bridged by a "
                    "straight line: the readings may be off; check them", lost=int(inside.sum()), frames=len(inside))
    if lost.mean() > 0.02:
        return Note("lost-some", f"a point was lost in {lost.sum()} of {len(lost)} frames, bridged by a straight line: "
                    "check them", lost=int(lost.sum()), frames=len(lost))
    return None


def read(part, fixed, fps, min_peak=None):
    """The swing, the tracker's jitter, the floor and measure()'s readings, from the two points' shifts: what the
    script prints, for the command line and for a page that runs it."""
    if len(part) < 12:
        raise ValueError(Note("few-swings", "too few frames to find a frequency: film longer"))
    x = swing(part, fixed)
    noise, floor = floor_of(x, fixed, min_peak)
    readings = measure(x, fps, floor)
    peaks, period = readings[4], fps / readings[0]
    span = slice(max(0, int(peaks[0] - period)), int(peaks[-1] + period) + 1)  # the peaks read, a swing either side
    note = lost_note(part, fixed, span)
    if note:
        readings[-1].append(note)
    return x, noise, floor, readings


def arguments(description, plot_help):
    """The command line both scripts share: the film, the two points, the frame rate, the square, the floor, the part
    of the film to read and the plot."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("video")
    ap.add_argument("--part", type=int, nargs=2, metavar=("X", "Y"), help="a point on the part that swings, px")
    ap.add_argument("--fixed", type=int, nargs=2, metavar=("X", "Y"), help="a point on something that does not move")
    ap.add_argument("--pick", action="store_true", help="click the two points in a window instead")
    ap.add_argument("--fps", type=float, help="the rate the video was filmed at, frames a second, when the file "
                    "stores another (a slow-motion file may store the rate it plays back at)")
    ap.add_argument("--box", type=int, default=BOX, help="size of the square tracked around each point, px; the whole "
                    "square must lie on the part (or on the still object), or still corners outvote moving ones")
    ap.add_argument("--min-peak", type=float, help="smallest swing used, px; by default the larger of 1 px and ten "
                    "times the frame-to-frame jitter (of the still point, or of the swing once it has settled)")
    ap.add_argument("--start", type=float, default=0, help="seconds into the file, on its own clock, to start at")
    ap.add_argument("--end", type=float, help="seconds into the file, on its own clock, to stop at")
    ap.add_argument("--plot", help=plot_help)
    return ap


def film(args):
    """The film's frame rate (filmed, and as the file stores it) and its frames, the first included, after checking
    the arguments and getting the two points (args.part, args.fixed): clicked, given, or read from a saved grid."""
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
    args.frames = frames  # its frame times, for uneven() once they have been read
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
    return fps, stored, itertools.chain([first], frames)


def main():
    args = arguments("Natural frequency and damping ratio from a video of a part ringing down.",
                     "save the swing and the peaks used to this image").parse_args()
    fps, stored, frames = film(args)
    part, fixed = track(frames, [args.part, args.fixed], args.box, report=True)
    timing = uneven(args.frames.times)
    if timing:
        print(f"warning: {timing}")
    try:
        x, noise, floor, (fd, fn, zeta, decrement, peaks, warnings) = read(part, fixed, fps, args.min_peak)
    except ValueError as e:
        raise SystemExit(str(e))
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
