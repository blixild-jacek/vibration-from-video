"""Natural frequency and damping ratio of a part, from a video of it ringing down after a tap or a release.

Pick two points in the first frame: one on the part that swings, one on something that does not move. The script
follows the texture around each point with a feature tracker (OpenCV: Shi-Tomasi corners, Lucas-Kanade optical
flow, checked forwards and backwards), measures the swing in pixels, and reads:
  - the damped natural frequency from the spectrum of the swing, while it rings above the floor,
  - the damping ratio from how fast the upper peaks shrink (the logarithmic decrement),
  - the noise floor from the frame-to-frame jitter of the point that does not move;
  - for a swing too small for the peaks (fewer than three rise above the floor), the frequency from the spectrum of
    the ring after the strike, when it stands out of the video's own noise at its quiet end, and the damping ratio
    from a decaying wave fitted to every frame of that ring, when it stands far enough out of the noise (its d
    reaches FIT_D); else the frequency alone (and always for an outline point, --outline).
Neither result needs millimeters: frequency comes from time, damping from a ratio of peaks or a fitted decay.

Usage:
  python3 free_decay.py VIDEO                                 saves VIDEO_first_frame.png with a pixel grid to read
                                                              the two points from
  python3 free_decay.py VIDEO --pick                          click the two points in a window (needs a desktop)
  python3 free_decay.py VIDEO --part X Y --fixed X Y [--fps 240] [--plot swing.png]
  python3 free_decay.py VIDEO --part X Y --fixed X Y --outline   the part's point on its outline (a turning shaft)

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
# measured 0.18-0.23, plain gray with sensor noise 0.0001 (noise of 2 gray levels) to 0.002 (8 gray levels)
TEXTURE_STOP, TEXTURE_WARN = 0.005, 0.02
# px kept around each point: the tracker follows each point in this window of the frame, not the whole frame. It holds
# the point's square, the tracker's coarsest level (21 px at an eighth of the size, about 80 px) and a swing of up to
# REACH px, beyond which the tracker warns
MARGIN = 256
# An outline point: a point on the part's outline against what lies behind it, for a part whose surface turns (a
# shaft, a disk, a coupling), where the corners in a square turn away with the surface or are lost, while the outline
# stays put in the picture and moves only as the part does. The edge through the point is found in the first frame
# (its normal from the gradients in the square, its strongest step along that normal), and in every frame its
# sub-pixel position along the normal, from the brightness profile across it averaged over the square's width. Only
# motion across the edge is seen: motion along it is not. On make_test_clip.py's turning bar, whose surface texture
# slides past just inside the outline, the edge was placed to about a tenth of a pixel RMS (an independent audit
# measured 0.11 px with the edge sharp or blurred by 1 px, 0.34 px blurred by 2 px), against the corner tracker's
# hundredth of a pixel on texture, and its error depends on where the edge falls within a pixel. So an outline point
# suits swings of a pixel or more, as the speed test reads; below the floor it gives the frequency alone (read()).
OUTLINE_REACH = 12  # px either side of the point along the edge's normal that are searched for the edge
OUTLINE_FAR = 2  # px: an edge this close to either end of the stretch searched, or of the frame, is warned about
OUTLINE_STOP = 8.0  # gray levels: the edge's whole step (Outline.step()) below which there is no outline to follow
OUTLINE_STRAIGHT = 0.5  # the gradients' coherence in the square, 0 (no one direction) to 1 (a straight edge)
REACH = MARGIN - 100
# px a point may move from one frame to the next before the tracker is likely to lose it: half the reach of a 21 px
# Lucas-Kanade window over three pyramid levels (10 px x 2^3 = 80 px). A drawn video tracked at 42 px a frame and lost
# frames at 67 px (an independent audit, 24 Nov 2026)
STEP = 40
# frames per swing below which the damping ratio is warned about. Without noise, the peak arithmetic reads it within
# 1.3 % from four frames per swing at damping ratios of 0.01 and 0.02, in any sampling phase; at 0.05 up to about 5 %
# near four frames (1.9 % at 4.25), and 3-5 % off at 3 to 3.7 frames; fewer frames also mean larger steps for the
# tracker (STEP)
FEW_FRAMES = 4


class Note(str):
    """A warning or error for the reader: the sentence the script prints, with a code and its values, so that a page
    running these functions can word it in its own language."""

    def __new__(cls, code, text, **values):
        note = super().__new__(cls, text)
        note.code, note.values = code, values
        return note


class Frames:
    """A video's frames, one grayscale image at a time, and the time of each frame read so far (ms, as OpenCV gives
    it), to tell a video saved at a variable frame rate (uneven())."""

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
    """The frame rate a video file stores and its frames (Frames), one grayscale image at a time."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot read {video}: check the path, or that OpenCV can open this kind of file")
    return cap.get(cv2.CAP_PROP_FPS), Frames(cap)


UNEVEN = 0.01  # frame times further than this from their usual spacing: a variable frame rate


def uneven(times):
    """A warning when the frames' times are unevenly spaced (a video saved at a variable frame rate, which the frame
    rate the file stores misrepresents: the frequency comes out wrong), or None."""
    steps = np.diff(np.asarray(times, float))
    if len(steps) < 3 or not np.all(np.isfinite(steps)):
        return None
    usual = np.median(steps)
    spread = float(np.abs(steps - usual).max() / usual) if usual > 0 else 0.0
    if spread > UNEVEN:
        return Note("uneven", f"the video's frames are unevenly spaced in time (by up to {100 * spread:.0f} %): it was "
                    "saved at a variable frame rate, so the frequency may be off; save it at a constant frame rate "
                    "first", spread=round(100 * spread))
    return None


def draw_grid(gray, step=50):
    """A grayscale frame as a color image with a pixel grid every step pixels, labeled every second line."""
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
    """Save the first frame with a labeled pixel grid, to read the two points' coordinates from."""
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
    """Follows the corners inside a box around each point, one grayscale frame at a time: track() runs it over a
    video, and a page that decodes a video itself can hand it each frame as it comes.

    Each point is followed from the last frame in which it was found, so a frame where it is lost (a blink, glare)
    leaves a gap, not an offset in every frame after it. Each point is followed in its own window of the frame
    (windows: x0, y0, x1, y1), so a large frame costs little more than a small one; step_crops() takes those windows
    alone, cut from the frame by whoever decodes it."""

    def __init__(self, first, points, box=BOX, margin=MARGIN, outline=()):
        h, w = first.shape
        half = box // 2
        self.points, self.box = [tuple(p) for p in points], box
        self.windows, self.outline = [], {}  # outline: index -> the edge followed for an outline point (Outline)
        self.start, self.now, self.ref, self.moves, self.reach, self.ids = [], [], [], [], [], []
        self.corners, self.faint = [], []  # corners found in each square; points whose texture is faint
        for i, (x, y) in enumerate(points):
            if not (0 <= x < w and 0 <= y < h):
                raise SystemExit(Note("outside", f"the point ({x}, {y}) lies outside the {w} x {h} frame",
                                      x=x, y=y, w=w, h=h))
            x0, y0 = max(x - margin, 0), max(y - margin, 0)
            self.windows.append((x0, y0, min(x + margin + 1, w), min(y + margin + 1, h)))
            crop = first[y0:self.windows[-1][3], x0:self.windows[-1][2]]
            cx, cy = x - x0, y - y0  # the point in its window
            if i in outline:
                self.outline[i] = Outline(crop, (cx, cy), half, (x, y))
                self.corners.append(0)
                self.start.append(np.zeros((0, 2)))
                self.now.append(None)
                self.ref.append(None)
                self.moves.append([np.zeros(2)])
                self.reach.append(np.zeros(1))
                self.ids.append(np.arange(0))
                continue
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
            if i in self.outline:
                shift = self.outline[i].shift(gray)
                self.moves[i].append(shift)
                if np.isfinite(shift).all():
                    self.reach[i][0] = max(self.reach[i][0], float(np.hypot(*shift)))
                continue
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
        the tracker may lose them; an outline point whose edge came within OUTLINE_FAR px of either end of the stretch
        searched for it (Outline.near_end())."""
        return [p for i, (p, r) in enumerate(zip(self.points, self.reach))
                if (self.outline[i].near_end() if i in self.outline else len(r) and r.max() > REACH)]

    def far_limit(self, p):
        """How far (px) point p could move before far() warns: REACH, or for an outline point the room its edge had,
        from where it started to the end of the stretch it came near (Outline.room())."""
        i = self.points.index(tuple(p))
        return self.outline[i].room() if i in self.outline else REACH

    def mixed(self):
        """The points whose square holds corners that moved and corners that stayed still, which suggests it covers
        more than the part."""
        out = []
        for p, r in zip(self.points, self.reach):
            top = np.percentile(r, 90) if len(r) else 0  # robust to a corner that slipped onto other texture
            if top > 2 and (r < 0.2 * top).sum() >= 3 and (r > 0.5 * top).sum() >= 3:
                out.append(p)
        return out


class Outline:
    """The edge through an outline point (OUTLINE_REACH and the constants above it): found in the first frame's window
    around the point, then its shift along its normal in each later window. The edge is searched for between lo and
    hi (indices of the profile's steps): the stretch of OUTLINE_REACH px either side of the point, cut short where it
    leaves the frame, whose last row would otherwise be repeated and hold the edge there."""

    def __init__(self, crop, center, half, point):
        cx, cy = center
        square = crop[max(cy - half, 0):cy + half + 1, max(cx - half, 0):cx + half + 1].astype(np.float32)
        gx, gy = cv2.Sobel(square, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(square, cv2.CV_32F, 0, 1, ksize=3)
        tensor = np.array([[(gx * gx).sum(), (gx * gy).sum()], [(gx * gy).sum(), (gy * gy).sum()]])
        values, vectors = np.linalg.eigh(tensor)
        straight = float((values[1] - values[0]) / (values[1] + values[0])) if values.sum() > 0 else 0.0
        self.normal = vectors[:, 1] / np.linalg.norm(vectors[:, 1])  # across the edge
        along = np.array([-self.normal[1], self.normal[0]])
        steps = np.arange(-OUTLINE_REACH, OUTLINE_REACH + 1, dtype=np.float32)
        width = np.arange(-half, half + 1, dtype=np.float32)
        xs = cx + steps[:, None] * self.normal[0] + width[None, :] * along[0]
        ys = cy + steps[:, None] * self.normal[1] + width[None, :] * along[1]
        self.maps = (xs.astype(np.float32), ys.astype(np.float32))
        h, w = crop.shape
        inside = (xs[:, half] >= 0) & (xs[:, half] <= w - 1) & (ys[:, half] >= 0) & (ys[:, half] <= h - 1)
        bad = np.flatnonzero(~(inside[:-1] & inside[1:]))  # steps with a sample outside the frame
        self.lo = int(bad[bad < OUTLINE_REACH].max()) + 1 if (bad < OUTLINE_REACH).any() else 0
        self.hi = int(bad[bad >= OUTLINE_REACH].min()) - 1 if (bad >= OUTLINE_REACH).any() else 2 * OUTLINE_REACH - 1
        self.ended = None  # "lo" or "hi" once the edge was found at that end, where it may lie beyond the stretch
        d = self.steps(crop) if self.hi > self.lo else np.zeros(1)
        k = int(np.nanargmax(np.abs(d)))
        self.sign = 1.0 if d[k] > 0 else -1.0
        self.strength = self.step(self.sign * d, k)
        self.start = self.position(crop) if self.strength >= OUTLINE_STOP else float("nan")
        if not np.isfinite(self.start) or straight < OUTLINE_STRAIGHT:
            raise SystemExit(Note("outline", f"no clear outline through ({point[0]}, {point[1]}) to follow: pick a "
                                  "point on a straight stretch of the part's edge against a plain background",
                                  x=point[0], y=point[1]))
        self.low = self.high = self.start  # the edge's lowest and highest positions found

    def profile(self, crop):
        """The brightness across the edge, averaged along it (one value a px along the normal)."""
        return cv2.remap(crop.astype(np.float32), *self.maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).mean(1)

    def steps(self, crop):
        """The profile's step from each px to the next (gray levels), NaN outside lo to hi."""
        d = np.diff(self.profile(crop))
        d[:self.lo], d[self.hi + 1:] = np.nan, np.nan
        return d

    @staticmethod
    def step(d, k):
        """The edge's whole step in brightness around its steepest sample k: the rise over two samples either side,
        which does not depend on where within a pixel the edge falls (a sharp edge's step can sit in one sample or
        be shared by two)."""
        return float(np.nansum(np.clip(d[max(k - 2, 0):k + 3], 0, None)))

    def position(self, crop):
        """The edge's position along the normal (px), sub-pixel: its strongest step of the first frame's sign,
        refined by a parabola; NaN when the whole step is under half the first frame's (the edge hidden or gone, a
        hand passing, a dark frame), or when a step that strong lies at lo or hi, where the edge may be beyond the
        stretch searched."""
        d = self.sign * self.steps(crop)
        k = int(np.nanargmax(d))
        if self.step(d, k) < 0.5 * self.strength:
            return float("nan")
        if k <= self.lo or k >= self.hi:
            self.ended = "lo" if k <= self.lo else "hi"
            return float("nan")
        a, b, c = d[k - 1], d[k], d[k + 1]
        return float(k + 0.5 * (a - c) / (a - 2 * b + c))

    def shift(self, crop):
        """The point's shift since the first frame (px, x and y): along the normal by the edge's move."""
        position = self.position(crop)
        if np.isfinite(position):
            self.low, self.high = min(self.low, position), max(self.high, position)
        return (position - self.start) * self.normal

    def near_end(self):
        """Whether the edge came within OUTLINE_FAR px of lo or hi, or was found at one of them."""
        return bool(self.ended) or self.low < self.lo + OUTLINE_FAR or self.high > self.hi - OUTLINE_FAR

    def room(self):
        """How far (px) the edge could move from where it was first found toward the end it came nearer (or was found
        at), before near_end(); 0 when it was first found that close to an end (the point picked off the edge)."""
        to_hi, to_lo = (self.hi - OUTLINE_FAR) - self.high, self.low - (self.lo + OUTLINE_FAR)
        toward_hi = self.ended == "hi" if self.ended else to_hi < to_lo
        room = (self.hi - OUTLINE_FAR) - self.start if toward_hi else self.start - (self.lo + OUTLINE_FAR)
        return round(max(0.0, float(room)), 1)


def faint_warning(p):
    return Note("faint", f"the texture around ({p[0]}, {p[1]}) is faint; the tracker may follow noise: pick a point on "
                "sharper detail, or add a sticker", x=p[0], y=p[1])


def far_warning(p, reach=REACH, outline=False):
    if outline:  # Outline.near_end(): the edge near the end of the stretch searched, reach px from where it was found
        return Note("far", f"the edge through ({p[0]}, {p[1]}) came within {OUTLINE_FAR} px of the end of the "
                    f"{OUTLINE_REACH} px searched either side of the point, {reach} px from where it was first found: "
                    "pick the point on the edge itself, and if the part swings that far, film from further away",
                    x=p[0], y=p[1], reach=reach, outline=True)
    return Note("far", f"the part around ({p[0]}, {p[1]}) moved more than {reach} px, near the edge of the window the "
                "tracker follows it in: film from further away", x=p[0], y=p[1], reach=reach)


def mixed_warning(p):
    return Note("mixed", f"the square around ({p[0]}, {p[1]}) holds corners that moved and corners that stayed "
                "still; it may cover more than the part: pick a point further inside it, or use a smaller --box",
                x=p[0], y=p[1])


def track(frames, points, box=BOX, report=False, outline=()):
    """Follow the corners inside a box around each point through grayscale frames (or, for the points whose index is
    in outline, the edge through them: Outline); return each point's shift (px)."""
    frames = iter(frames)
    tracker = Tracker(next(frames), points, box, outline=outline)
    if report:
        for i, ((x, y), n) in enumerate(zip(tracker.points, tracker.corners)):
            if i in tracker.outline:
                nx, ny = tracker.outline[i].normal
                print(f"point ({x}, {y}): an outline, followed across it (normal {nx:+.2f}, {ny:+.2f}) over its {box} "
                      "px width; motion along it is not seen")
                continue
            print(f"point ({x}, {y}): {n} corners in its {box} px square")
            if (x, y) in tracker.faint:
                print(f"warning: {faint_warning((x, y))}")
    for gray in frames:
        tracker.step(gray)
    if report:
        for p in tracker.far():
            edge = tracker.points.index(tuple(p)) in tracker.outline
            print(f"warning: {far_warning(p, tracker.far_limit(p), edge)}")
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
    centered = rel - np.nanmean(rel[ok][-len(rel) // 10:], axis=0)  # zero = where the part comes to rest
    _, _, axes = np.linalg.svd(centered[ok], full_matrices=False)
    x = centered @ axes[0]
    idx = np.arange(len(x))
    x[~ok] = np.interp(idx[~ok], idx[ok], x[ok])  # bridge frames where a point was lost
    return positive_first(x)


def tops(x, peaks):
    """The top of each peak, from a parabola through it and the frames either side."""
    a, b, c = x[peaks - 1], x[peaks], x[peaks + 1]
    return b - (a - c) ** 2 / (8 * (a - 2 * b + c))


LOWEST = 3 * 8  # the spectrum is searched above three cycles in the record: this many bins of the 8x padded one


def spectrum(x, falling=False):
    """The record's magnitude spectrum, padded to 8 times its length, with a straight-line trend taken off first, so a
    slow drift or a bridged gap is not read as ringing, and a Hann window; or, with falling, the falling half of one,
    full weight at the record's start, for a record that starts at a strike, where a ring is largest. Returns the
    spectrum and the window's sum."""
    t = np.arange(len(x))
    flat = x - np.polyval(np.polyfit(t, x, 1), t)
    window = np.hanning(2 * len(x))[len(x):] if falling else np.hanning(len(x))
    return np.abs(np.fft.rfft(flat * window, 8 * len(x))), window.sum()


def tone(x, fps, falling=False):
    """The strongest frequency in a record (Hz), the highest peak of its spectrum() above three cycles in the record
    (with the falling window if falling), refined between bins, and that tone's amplitude (px). NaN for both when the
    record is flat, the highest point lies at the bottom of the searched range, or the parabola's top lies more than
    half a bin beyond the three bins it is drawn through: no peak to place."""
    if len(x) < 12 or not np.all(np.isfinite(x)):
        raise ValueError(Note("few-swings", "too few swings to find a frequency: the largest swing should come at "
                              "the knock, not later"))
    spectrum_, total = spectrum(x, falling)
    n, lowest = 8 * len(x), LOWEST
    size = float(np.abs(x - x.mean()).max())
    if not size > 0 or not spectrum_[lowest:-1].max() > 1e-9 * total * size:  # a flat record, such as a gap bridged
        return float("nan"), float("nan")  # by a straight line: only rounding noise, no frequency
    k = np.argmax(spectrum_[lowest:-1]) + lowest
    if k == lowest:  # the highest point at the bottom of the searched range: the spectrum still rises below it, so
        return float("nan"), float("nan")  # the peak lies below three cycles in the record (no frequency)
    a, b, c = np.log(spectrum_[k - 1:k + 2])
    between = 0.5 * (a - c) / (a - 2 * b + c)  # the parabola's top, in bins from k
    if not abs(between) <= 1.5:  # a top beyond the three bins it was drawn through, by more than the half bin a peak
        # on the outer bin can give (at half the frame rate, 1.002 bins), would be extrapolated: the audit's case, -6.4
        return float("nan"), float("nan")
    top = np.exp(b - 0.25 * (a - c) * between)  # and its height
    return (k + between) * fps / n, 2 * top / total


def frequency(x, fps):
    """Damped natural frequency (Hz): the strongest frequency in the record (tone())."""
    return tone(x, fps)[0]


def slope(t, y):
    """A straight line's slope fitted by least squares, and its standard error."""
    b, a = np.polyfit(t, y, 1)
    spread = ((t - t.mean()) ** 2).sum()
    return b, np.sqrt(((y - (b * t + a)) ** 2).sum() / max(1, len(t) - 2) / spread)


def damping(x, fps, fd, floor):
    """Damping ratio, logarithmic decrement, the peaks used, and with eight peaks or more the decrements of their first
    and second halves and whether they differ by more than three times their combined standard error and by more than
    a fifth, as they do when two resonances share the swing or the part is not one linear resonance (None with fewer
    peaks). Neighboring peaks share troughs and their small errors are correlated, so the standard errors alone come
    out too small: a clean ring was called bent in most sampling phases without the fifth.

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
    fall = np.polyfit(t, logs, 1)[0]  # 1/s
    decrement = -fall / fd  # logarithmic decrement, per cycle
    halves = None
    if len(peaks) >= 8:
        h = len(peaks) // 2
        (s1, e1), (s2, e2) = slope(t[:h], logs[:h]), slope(t[h:], logs[h:])
        d1, d2 = -s1 / fd, -s2 / fd
        halves = (d1, d2, bool(abs(s1 - s2) > 3 * np.hypot(e1, e2) and abs(d1 - d2) > 0.2 * abs(d1 + d2) / 2))
    return decrement / np.sqrt(4 * np.pi ** 2 + decrement ** 2), decrement, peaks, halves


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
    if not np.isfinite(fd):  # no peak at all: the largest swing was not a tap, such as someone moving the camera
        raise ValueError(Note("no-peak", "no clear frequency in the swing: check the points, and read only the stretch "
                              "from just before the tap until the part is still (--start, --end)"))
    if not 0 < fd < fps / 2:
        raise ValueError(Note("no-frequency", f"no clear frequency in the swing ({fd:.3f} Hz at {fps:g} fps): check "
                              "--fps and the points", fd=fd, fps=fps))
    zeta, decrement, peaks, halves = damping(ringing, fps, fd, floor)
    if zeta <= 0:
        raise ValueError(Note("no-decay", "the peaks do not shrink: two resonances may be beating, or the video "
                              "holds no ring-down"))
    warnings = []
    if halves and halves[2]:
        z1, z2 = (d / np.sqrt(4 * np.pi ** 2 + d ** 2) for d in halves[:2])
        warnings.append(Note("not-straight", f"the first and second halves of the peaks give damping ratios of "
                             f"{z1:.4f} and {z2:.4f}, further apart than their scatter allows: two resonances may "
                             "share the swing, or the part is not one linear resonance, and the damping ratio is "
                             "doubtful", z1=z1, z2=z2))
    if np.abs(x[-len(x) // 10:]).max() > floor:
        warnings.append(Note("still-swinging", "the part still swings in the last tenth of the video, which the "
                             "script takes as rest: film until it stops, or the damping ratio may read low"))
    if fd > fps / FEW_FRAMES:
        warnings.append(Note("few-frames", f"fewer than {FEW_FRAMES} frames per swing ({fps / fd:.1f}): the damping "
                             "ratio may be off", per_swing=fps / fd, few=FEW_FRAMES))
    return fd, fd / np.sqrt(1 - zeta ** 2), zeta, decrement, peaks + start, warnings


# Below the floor, the frequency alone. The damping ratio needs swings above the floor, since smaller ones carry the
# tracker's error, but a ring far smaller still shows in the spectrum, as long as it stands out of the video's own
# noise. So the swing's slow drift is taken off, and the stretch right after the strike is compared with stretches of
# the same length at the video's end, where the part has stopped, and before the strike: each video's own jitter and
# drift set the bar, not an assumed level. And the stretch's own peak must stand out of its own spectrum, which a burst
# of noise or a blip does not. faint_noise.py measures both bars on simulated swings: steady noise (white tracker
# jitter plus a random-walk drift of up to 30 times its size, 25 to 240 fps, 10 and 20 s), noise that is not steady
# (a frozen or quieter end, a burst after an event, a blip), and rings.
FAINT_CUT = 0.5  # Hz: the drift's high-pass, below the lowest frequency read (three cycles in 4 s, 0.75 Hz)
FAINT_LENGTHS = (0.5, 1.0, 2.0, 4.0)  # s after the strike: a short stretch for a fast ring that dies quickly
FAINT_QUIET = 3  # stretches of the same length at the video's end, against which the ring is measured: when short
                 # videos used one or two, noise alone reached 5.67 in 30,000 simulated swings, so a length counts only
                 # when all three fit; up to as many before the jump are measured against too
FAINT_SKIP = 5  # frames after the largest frame-to-frame jump before the stretch starts: a jump or a blip a few frames
                # long, such as the hammer passing a point, is not read as ringing
FAINT_OWN = 5.0  # the stretch's own highest spectral peak over its median: a ring gathers in one peak, while a burst of
                 # noise or a blip spreads over every frequency. Of the simulated rings that passed FAINT_RATIO, 95 %
                 # stood 5.7 or more; a blip about 1, white noise 2-4
FAINT_RATIO = 5.0
SMALL = {"below-floor": "the swing never rises above the floor of {floor:.2f} px",
         "few-peaks": "fewer than three swings rise above the floor of {floor:.2f} px",
         "few-swings": "too few swings rise above the floor of {floor:.2f} px"}  # the refusals of a swing too small


def drift_off(x, fps, cut=FAINT_CUT):
    """The swing with its slow drift taken off: a second-order Butterworth high-pass at cut Hz, run forwards and
    backwards so it shifts nothing in time."""
    from scipy.signal import butter, sosfiltfilt
    return sosfiltfilt(butter(2, cut, "high", fs=fps, output="sos"), x)


def faint(x, fps, lengths=FAINT_LENGTHS, quiet=FAINT_QUIET):
    """A ring too small for the damping ratio. The swing's drift is taken off, the strike is its largest frame-to-frame
    jump, and for each length the stretch from FAINT_SKIP frames after it is measured two ways (spectrum() with the
    falling window, which finds a short ring best): its highest peak over the highest of `quiet` stretches of the same
    length at the video's end and of up to as many before the jump (the ratio), and over its own median (own). Lengths
    where any of these stretches holds a repeated frame (a duplicated frame tracks exactly still) are left out, as are
    lengths whose frequency tone() cannot place. The frequency is tone()'s from the same spectrum, so it is the ring
    that was found: a full window would weight a fast ring's start, where it all is, near zero. The falling window reads
    a clean ring up to about 2.7 % off at three cycles in the stretch, under 1 % from about eight. Returns
    (frequency Hz, ratio, seconds, start, end, own) for the longest length that passes() (a short stretch cannot hold
    three cycles of a slow ring, and the strike's own jolt stands out in it: a real video's 3 Hz ring read 7.95 Hz
    from 0.5 s); else the length with the highest ratio; None when no length fits (no quiet end after the stretch)."""
    if not FAINT_CUT < fps / 2 or len(x) < 24:
        return None
    y = drift_off(x, fps)
    jump = int(np.argmax(np.abs(np.diff(y))))
    start = jump + FAINT_SKIP
    still = np.r_[False, np.diff(x) == 0]  # a frame the same as the one before: a duplicated frame
    found = []
    for seconds in lengths:
        n = int(round(seconds * fps))
        ends = [len(y) - i * n for i in range(quiet) if len(y) - (i + 1) * n >= start + n]
        if n < 12 or len(ends) < quiet:
            continue
        spans = [(start, start + n)] + [(e - n, e) for e in ends] + \
                [(jump - (i + 1) * n, jump - i * n) for i in range(quiet) if jump - (i + 1) * n >= 0]
        if any(still[a + 1:b].sum() > 1 for a, b in spans):
            continue
        hz = tone(y[start:start + n], fps, True)[0]
        ring = spectrum(y[start:start + n], True)[0][LOWEST:-1]
        top = max(spectrum(y[a:b], True)[0][LOWEST:-1].max() for a, b in spans[1:])
        if not np.isfinite(hz) or not top > 0:  # no peak to place, or a quiet stretch with no noise at all
            continue
        found.append((hz, ring.max() / top, seconds, start, start + n, ring.max() / np.median(ring)))
    passing = [f for f in found if passes(f, fps)]
    return max(passing, key=lambda f: f[2]) if passing else max(found, key=lambda f: f[1], default=None)


# Below the floor, the damping ratio too, from a fit (24 Nov 2026, after a real struck frame that swung 0.74 px read
# "damping ratio: not read" while its chart showed the ring dying away). Peak by peak, a swing under the floor carries
# the tracker's error in every peak; a decaying wave fitted to every frame of the ring averages that error, as the
# frequency's spectrum does. The stretch is faint()'s, band-passed around its frequency (FIT_BAND times it); the reading
# stands only when the ring-down's d (signal_to_noise()) reaches FIT_D, and the stretch's two halves are fitted apart
# and compared as the peaks' halves are, since a structure's damping can change as its swing dies away. FIT_D and the
# halves' bar come from simulated rings alone (faint_noise.py --damping), fixed before any unseen video was read.
# d counts white noise only. It does not see a tracker's error that grows as the swing shrinks: a compressed video's
# (on make_test_clip.py's compressed clips the tracker read less of a swing the smaller it was, about a third of a
# 0.04 px one by two independent audits' probes, so the fit reads the damping high) or an outline point's (read() keeps
# outline points out of the fit). And it takes the noise from the
# video's quiet end, so an end quieter than the ring's surroundings raises d.
FIT_BAND = (0.6, 1.5)
FIT_ABOVE = 3.0  # the fit runs until the fitted envelope falls to this many times the band's noise
FIT_CYCLES = 8  # the fewest cycles fitted
FIT_HALVES = (3.0, 0.2)  # the halves differ: by more than this many combined standard errors and this share
FIT_D = 30.0  # the ring-down's d below which the damping ratio is not read: on simulated rings (faint_noise.py
              # --damping, its table by band of d), 95 % of fits with d of 30 to 50 lay within 13.4 % of the damping
              # ratio set, of 20 to 30 within 17.3 %: from 30 up, every band of its table reads 95 % within 15 %


def wave(t, a, decay, hz, phase, offset):
    """A decaying wave: amplitude a, decay (1/s), frequency hz, phase and offset, at times t (s)."""
    return a * np.exp(-decay * t) * np.sin(2 * np.pi * hz * t + phase) + offset


def fit_ring(y, fps, hz, zeta0=0.02):
    """wave() fitted to y by least squares, from a start whose amplitude, phase and offset are the linear fit for a
    decay of zeta0 at hz: (a, decay, hz, phase, offset) and their covariance, or None when it does not converge."""
    from scipy.optimize import curve_fit
    t = np.arange(len(y)) / fps
    for z in (zeta0, 0.005, 0.05):
        decay = z * 2 * np.pi * hz
        e = np.exp(-decay * t)
        basis = np.stack([e * np.sin(2 * np.pi * hz * t), e * np.cos(2 * np.pi * hz * t), np.ones_like(t)], 1)
        (s_, c_, o_), *_ = np.linalg.lstsq(basis, y, rcond=None)
        try:
            fit, cov = curve_fit(wave, t, y, p0=[np.hypot(s_, c_), decay, hz, np.arctan2(c_, s_), o_], maxfev=5000)
        except (RuntimeError, ValueError):
            continue
        if np.isfinite(cov).all() and fit[1] > 0 and 0 < fit[2] < fps / 2:
            if fit[0] < 0:  # the same wave with its phase turned half a cycle
                fit[0], fit[3] = -fit[0], fit[3] + np.pi
            return fit, cov
    return None


def faint_damping(x, fps, hz, start, end, noise):
    """The damping ratio of a ring below the floor, from wave() fitted to every frame of the ring, band-passed around
    hz (FIT_BAND): first over faint()'s stretch x[start:end], then from start until that fit's envelope falls to
    FIT_ABOVE times the band's noise (at least FIT_CYCLES cycles), so that all of the ring that stands out is used, as
    the peaks above the floor are.
    Returns (zeta, its ring-down's d, halves, the fit's relative standard error of the decay, the seconds fitted, and
    the fitted envelope's (amplitude px, decay 1/s) from start, for a chart), where
    halves is (zeta of the first half of those seconds, of the second, whether they differ by more than FIT_HALVES
    times their combined standard error and by more than its share of their mean, and those two figures), or None
    when a half does not fit; None when the fit fails
    or does not decay. noise: the swing's noise (px a frame), as signal_to_noise() takes it. Band-passed noise is
    correlated from frame to frame, so the fits' standard errors are widened by sqrt(fps / (2 x the band's width)).
    The gate (FIT_D) is read()'s."""
    from scipy.signal import butter, sosfiltfilt
    lo, hi = FIT_BAND[0] * hz, min(FIT_BAND[1] * hz, 0.45 * fps)
    if not 0 < lo < hi or not noise > 0:
        return None
    y = sosfiltfilt(butter(2, [lo, hi], "bandpass", fs=fps, output="sos"), x)
    band_noise = noise * np.sqrt(2 * (hi - lo) / fps)  # white noise's share in the band
    widen = np.sqrt(fps / (2 * (hi - lo)))
    got = fit_ring(y[start:end], fps, hz)
    if got is None:
        return None
    (a, decay, f, _, _), _ = got
    last = len(y) - 2 * int(np.ceil(fps / f))  # the filter's own edge left out
    span = np.log(a / (FIT_ABOVE * band_noise)) / decay if a > FIT_ABOVE * band_noise else 0.0
    stop = int(min(last, start + max(span, FIT_CYCLES / f) * fps))  # where the ring meets the noise, not the
    # stretch faint() compared with the quiet end: a well-damped ring dies within it, and a half of noise fits nothing
    got = fit_ring(y[start:stop], fps, f)
    if got is None:
        return None
    (a, decay, f, _, _), cov = got
    to_zeta = lambda d_, f_: d_ / np.hypot(d_, 2 * np.pi * f_)
    zeta = to_zeta(decay, f)
    d = a / noise * np.sqrt(fps / (8 * np.pi * zeta * f / np.sqrt(1 - zeta ** 2)))
    half = (stop - start) // 2
    parts = [fit_ring(y[start:start + half], fps, f), fit_ring(y[start + half:stop], fps, f)]
    halves = None
    if all(parts):
        (s1, e1), (s2, e2) = ((q[0][1], widen * np.sqrt(q[1][1, 1])) for q in parts)
        z1, z2 = to_zeta(s1, parts[0][0][2]), to_zeta(s2, parts[1][0][2])
        sigmas, share = abs(s1 - s2) / np.hypot(e1, e2), abs(z1 - z2) / ((z1 + z2) / 2)
        halves = (z1, z2, bool(sigmas > FIT_HALVES[0] and share > FIT_HALVES[1]), sigmas, share)
    return zeta, d, halves, float(widen * np.sqrt(cov[1, 1]) / decay), (stop - start) / fps, (float(a), float(decay))


def passes(found, fps):
    """Whether faint()'s reading stands out of the noise: its ratio to the quiet stretches reaches FAINT_RATIO, its own
    peak over its median FAINT_OWN, and its frequency lies below half the frame rate."""
    return bool(found) and found[1] >= FAINT_RATIO and found[5] >= FAINT_OWN and 0 < found[0] < fps / 2


def noise_in(x):
    """The white noise in a stretch of swing (px RMS a frame), from the median of its spectrum: a ring fills a few
    bins, so the median bin holds noise. Taken on the moving point while it moves, it includes what the tracker and the
    video's compression add there, which a still point's jitter can miss: on a compressed test video an independent
    audit measured 0.021 px on the moving point against 0.0006 px on the still one."""
    if len(x) < 8:
        return float("nan")
    t = np.arange(len(x))
    flat = x - np.polyval(np.polyfit(t, x, 1), t)
    window = np.hanning(len(x))
    power = np.abs(np.fft.rfft(flat * window))[1:-1] ** 2
    return float(np.sqrt(np.median(power) / (np.log(2) * (window ** 2).sum())))  # an exponential's median is ln 2
    # times its mean


def first_and_noise(y, fps, hz):
    """A ring's first swing (px, the largest within one period of the stretch's start) and the stretch's noise
    (noise_in()): the two sides of its signal-to-noise ratio."""
    return float(np.abs(y[:max(2, int(round(fps / hz)))]).max()), noise_in(y)


def signal_to_noise(x, fps, floor, readings):
    """The ring's signal-to-noise ratio: its first swing (px) over the swing's noise (noise_in(): what is left of the
    stretch read once the ring is taken off, or the quiet end a frequency alone was measured against), and, when the
    damping ratio was read, the ring-down's d = (first / noise) x sqrt(fps / (8 pi zeta fn)), with the best precision
    noise alone allows: the Cramer-Rao bound for a decaying sine in white noise, which d gives to within 1-4 % in the
    five cases checked, the frequency to 2 zeta / d and the damping ratio to 2 / d, as fractions. It is a bound: this
    script's own readings, from the peaks and a windowed spectrum, scattered several times more in simulation."""
    fd, fn, zeta, decrement, peaks, warnings = readings
    only = [w for w in warnings if getattr(w, "code", "") in ("frequency-only", "damping-fit")]
    if only:  # the ring's first swing, against the noise of the quiet end it was measured against
        a, b = only[0].values["start"], only[0].values["end"]
        y = drift_off(x, fps)
        first, noise = first_and_noise(y[a:b], fps, fd)[0], noise_in(y[len(y) - FAINT_QUIET * (b - a):])
    else:  # the noise left once the ring read is taken off: the ring's own leakage would pass for noise
        a, b = ringing_part(x, fps, floor)
        t = np.arange(b - a) / fps
        decay = np.exp(-zeta * 2 * np.pi * fn * t)
        phase = 2 * np.pi * fd * t
        basis = np.stack([decay * np.cos(phase), decay * np.sin(phase), np.ones_like(t), t], 1)
        rest = x[a:b] - basis @ np.linalg.lstsq(basis, x[a:b], rcond=None)[0]
        first, noise = float(abs(x[a])), noise_in(rest)
    out = {"first": first, "noise": noise, "ratio": first / noise}
    if np.isfinite(zeta) and zeta > 0 and np.isfinite(noise):
        d = first / noise * np.sqrt(fps / (8 * np.pi * zeta * fn))
        out.update(d=d, frequency=2 * zeta / d, damping=2 / d)
    return out


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


def read(part, fixed, fps, min_peak=None, outline=False):
    """The swing, the tracker's jitter, the floor and measure()'s readings, from the two points' shifts: what the
    script prints, for the command line and for a page that runs it. When the swing is too small for the damping ratio,
    the frequency from faint(), and the damping ratio from faint_damping() with a "damping-fit" note when its d reaches
    FIT_D, or else NaN with a "frequency-only" note (no peaks either way); when no ring stands out, measure()'s
    refusal. outline: the part's point is an outline point (Outline), whose edge is placed too coarsely for a fit below
    the floor (a 0.6 px ring on make_test_clip.py's turning bar read 0.054 for 0.02, with no warning): the frequency
    alone there, its note's values saying why (outline True)."""
    if len(part) < 12:
        raise ValueError(Note("few-swings", "too few frames to find a frequency: film longer"))
    x = swing(part, fixed)
    noise, floor = floor_of(x, fixed, min_peak)
    try:
        readings = measure(x, fps, floor)
    except ValueError as e:  # too small for the damping ratio: the frequency alone, if a ring stands out (faint())
        why = getattr(e.args[0], "code", None)
        found = faint(x, fps) if why in SMALL else None
        if not passes(found, fps):
            raise
        fd, ratio, seconds, start, end, _ = found
        lost = np.isnan(part).any(axis=1) | np.isnan(fixed).any(axis=1)
        if lost[start:end].any() or lost[len(lost) - FAINT_QUIET * (end - start):].any():
            raise  # a gap bridged by a straight line can make a quiet end look flat, or a ring
        quiet = noise_in(drift_off(x, fps)[len(x) - FAINT_QUIET * (end - start):])
        fit = None if outline else faint_damping(x, fps, fd, start, end, quiet)
        if fit and fit[1] >= FIT_D:  # the damping ratio from a decaying wave fitted to the ring
            zeta, d, halves, _, fitted, (amp, rate) = fit
            decrement = 2 * np.pi * zeta / np.sqrt(1 - zeta ** 2)
            from_s, to_s = start / fps, start / fps + fitted  # seconds into the frames read, as a chart's axis
            notes = [Note("damping-fit", f"{SMALL[why].format(floor=floor)}, so the damping ratio is read not peak by "
                          f"peak but from a decaying wave fitted to every frame from {from_s:.1f} to {to_s:.1f} s of the "
                          f"frames read, the {fitted:.1f} s after the strike where the ring stands out of the noise (its "
                          f"ring-down's d is {d:.0f}, where {FIT_D:g} is needed)", why=why, floor=floor,
                          seconds=fitted, d=d, needed=FIT_D, start=start, end=end, from_s=from_s, to_s=to_s, amp=amp,
                          decay=rate)]
            if halves and halves[2]:
                notes.append(Note("damping-changes", f"the first and second halves of those {fitted:.1f} s give damping "
                                  f"ratios of {halves[0]:.4f} and {halves[1]:.4f}, further apart than their scatter "
                                  "allows: the damping may change as the swing dies away, as a bolted structure's "
                                  "does, two resonances may beat, or the tracker read the smaller swings short (a "
                                  "compressed video can); give the damping ratio with the stretch it was read over",
                                  z1=halves[0], z2=halves[1], seconds=fitted))
            readings = (fd, fd / np.sqrt(1 - zeta ** 2), zeta, decrement, np.array([], int), notes)
            span = slice(start, start + int(round(fitted * fps)))
        else:
            readings = (fd, fd, float("nan"), float("nan"), np.array([], int), [Note(
                "frequency-only", f"only the frequency is read: {SMALL[why].format(floor=floor)}, too few for the "
                "damping ratio" + (", and an outline point's edge is placed too coarsely for one fitted below the floor"
                                   if outline else "") +
                f", but the {seconds:g} s after the strike stand {ratio:.1f} times above the video's quiet "
                f"end, where {FAINT_RATIO:g} is needed. For the damping ratio, film closer or tap harder",
                why=why, floor=floor, seconds=seconds, ratio=ratio, start=start, end=end, needed=FAINT_RATIO,
                outline=bool(outline))])
            span = slice(start, end)
    else:
        peaks, period = readings[4], fps / readings[0]
        span = slice(max(0, int(peaks[0] - period)), int(peaks[-1] + period) + 1)  # the peaks read, a swing each side
    note = lost_note(part, fixed, span)
    if note:
        readings[-1].append(note)
    step = np.nanmax(np.hypot(*np.diff(part, axis=0).T))
    if step > STEP:
        readings[-1].append(Note("fast-step", f"the part moves up to {step:.0f} px from one frame to the next, more "
                                 f"than the {STEP} px the tracker reliably follows: film from further away or at a "
                                 "higher frame rate", step=float(step), limit=STEP))
    return x, noise, floor, readings


def arguments(description, plot_help):
    """The command line both scripts share: the video, the two points, the frame rate, the square, the floor, the part
    of the video to read and the plot."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("video")
    ap.add_argument("--part", type=int, nargs=2, metavar=("X", "Y"), help="a point on the part that swings, px")
    ap.add_argument("--fixed", type=int, nargs=2, metavar=("X", "Y"), help="a point on something that does not move")
    ap.add_argument("--pick", action="store_true", help="click the two points in a window instead")
    ap.add_argument("--outline", action="store_true", help="the part's point is on its outline against the background "
                    "(a shaft, a disk or another part whose surface turns): the edge is followed, not texture")
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
    """The video's frame rate (filmed, and as the file stores it) and its frames, the first included, after checking
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
    part, fixed = track(frames, [args.part, args.fixed], args.box, report=True, outline=(0,) if args.outline else ())
    timing = uneven(args.frames.times)
    if timing:
        print(f"warning: {timing}")
    try:
        x, noise, floor, (fd, fn, zeta, decrement, peaks, warnings) = read(part, fixed, fps, args.min_peak,
                                                                           outline=args.outline)
    except ValueError as e:
        raise SystemExit(str(e))
    print(f"frames {len(x)} at {fps:g} fps" + ("" if args.fps is None else f" (the file stores {stored:g})") +
          f"; jitter {noise:.3f} px; floor {floor:.2f} px; largest swing {np.abs(x).max():.1f} px")
    if np.isnan(zeta):
        print(f"damped natural frequency {fd:.5f} Hz; the damping ratio is not read")
    else:
        print(f"damped natural frequency {fd:.5f} Hz; natural frequency {fn:.5f} Hz")  # five decimals: the two
        # differ by about fn zeta^2 / 2, 0.0009 Hz at 2 Hz and 3 %, which three decimals would hide
        print(f"logarithmic decrement {decrement:.4f}; damping ratio {zeta:.4f} " +
              (f"(from {len(peaks)} peaks)" if len(peaks) else "(from a decaying wave fitted below the floor)"))
    snr = signal_to_noise(x, fps, floor, (fd, fn, zeta, decrement, peaks, warnings))
    print(f"signal-to-noise: the first swing, {snr['first']:.2f} px, is {snr['ratio']:.0f} times the swing's noise, "
          f"{snr['noise']:.3f} px" + (f"; noise alone allows at best +-{100 * snr['frequency']:.2g} % in the "
                                      f"frequency and +-{100 * snr['damping']:.2g} % in the damping ratio"
                                      if "d" in snr else ""))
    for w in warnings:  # how the damping ratio was read below the floor is a reading's note, not a warning
        print(f"{w}" if getattr(w, "code", "") == "damping-fit" else f"warning: {w}")
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
