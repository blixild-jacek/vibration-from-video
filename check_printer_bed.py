"""Natural frequency and damping from a high-speed film of a shaken plate, checked against an accelerometer.

Data: the printer-bed dataset, version 1 (Gorjup, Zaletelj, Slavič 2026, Zenodo, doi:10.5281/zenodo.21476609,
CC BY 4.0): a 3D printer's bed hung on cords and shaken by a shaker with a sine sweep, filmed at 1,000 frames a
second (384 x 384 px) from two sides in turn, with the force and the acceleration measured where the shaker meets
the plate (an impedance head) recorded with each film. The record describes the sweep as 60-300 Hz; in the force
signal it rises from 60 Hz at about 0.1 s past 300 Hz at about 1.45 s to about 500 Hz near the end of the 2 s record.
This script reads 60-300 Hz (SWEEP). --download fetches and unpacks the two archives it needs (about 450 MB, and about
1 GB on disk once unpacked), checked against the checksums the Zenodo record publishes.

To use it on your own recording, change what only fits this dataset: SWEEP, BANDS (one around each resonance), GRIDS
(points whose squares lie on your part), and read_mraw() and read_signals() for your file formats.

The plate is shaken, not tapped, so resonances are read from frequency response functions (displacement divided by
force, both over the same 2 s; the accelerometer's acceleration is first divided by -w^2), from the camera and from
the accelerometer, in two ways:
  - fit: one resonance plus a constant and a 1/w^2 term (the resonances above and below the band) fitted to the
    complex response in a band around it;
  - ring-down: the band turned back into an impulse response (how the plate would ring after an ideal knock), then
    read by free_decay.py's frequency() and damping(), as the tap test reads a film.

The camera's motion comes from one of two readings (--reading):
  - tracker: free_decay.py's feature tracker on a 6 x 6 grid of points on the plate, x and y;
  - brightness: no tracking; each pixel's brightness over time, on the steepest tenth of the plate's pixels. For
    motion well under a pixel a pixel's brightness changes by (brightness slope) x (shift), so it carries the same
    frequencies and decay; scale and direction are lost.

Usage:
  python3 check_printer_bed.py --download                     fetch the data into printer-bed-v1, then check view 0
  python3 check_printer_bed.py [DATASET_DIR] [--view 1] [--reading brightness] [--plot frf.png]

SPDX-License-Identifier: MIT
Copyright (c) 2026 Jacek Garbuliński Blixild, blixild.com
"""
import argparse
import hashlib
import http.client
import shutil
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

import sys

import cv2
import h5py
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))  # free_decay.py beside it, also under python3 -I
import free_decay  # noqa: E402

# The two archives of the Zenodo record this script reads, with the MD5 checksums the record publishes
RECORD = "https://zenodo.org/records/21476609/files/{}?download=1"
ARCHIVES = {"mraw.zip": "c1e3fd2e2aa1b59ea686f136411e2041", "measured_signals.zip": "bb93d8be17476b1ea13d8f148cde9fb9"}
NEEDED = ["mraw/view_0_01/view_0_01.cih", "mraw/view_0_01/view_0_01.mraw", "mraw/view_1_01/view_1_01.cih",
          "mraw/view_1_01/view_1_01.mraw", "measured_signals/view_0.hdf5", "measured_signals/view_1.hdf5"]
# The part of the sweep read, and bands around the resonances the accelerometer's response shows there, Hz
SWEEP = (60, 300)
BANDS = [(70, 85), (100, 118), (118, 140), (195, 240), (250, 285)]
# A grid of points on the plate in each view's first frame (x0, x1, y0, y1), px: every 41 px box around a point lies
# on the plate (view 0's bottom edge runs from y 312 at x 80 to y 372 at x 345; view 1's right edge is at x 348)
GRIDS = {0: (80, 310, 60, 280), 1: (70, 315, 70, 295)}


def md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def complete(folder, needed=NEEDED):
    """Whether every file needed is there, each film whole (an interrupted unpack leaves a short .mraw)."""
    folder = Path(folder)
    if not all((folder / name).exists() for name in needed):
        return False
    for name in needed:
        if name.endswith(".mraw"):
            try:
                read_mraw((folder / name).parent)
            except SystemExit:
                return False
    return True


def fetch(url, path):
    """Download url to path through a .part file, so an interrupted download never looks finished."""
    part = path.with_name(path.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "vibration-from-video"})
    try:
        with urllib.request.urlopen(request, timeout=60) as reply, open(part, "wb") as out:
            total, done, shown = int(reply.headers.get("Content-Length") or 0), 0, 0
            for block in iter(lambda: reply.read(1 << 20), b""):
                out.write(block)
                done += len(block)
                if total and done * 100 // total >= shown + 10:
                    shown = done * 100 // total
                    print(f"  {shown} % of {total / 1e6:.0f} MB")
        if total and done != total:
            raise OSError(f"the connection closed after {done} of {total} bytes")
    except (OSError, http.client.HTTPException) as e:  # no network, refused, HTTP error, certificates, cut short
        part.unlink(missing_ok=True)
        raise SystemExit(f"cannot download {url}: {e}")
    part.replace(path)


def download(folder, archives=ARCHIVES, record=RECORD, needed=NEEDED):
    """Fetch and unpack the archives into folder, unless the files that are needed are there and whole."""
    folder = Path(folder)
    if complete(folder, needed):
        print(f"{folder}: the data are there already")
        return
    folder.mkdir(parents=True, exist_ok=True)
    for name, checksum in archives.items():
        path = folder / name
        if path.exists() and md5(path) == checksum:
            print(f"{name}: already downloaded")
        else:
            print(f"downloading {name} from Zenodo into {folder} ...")
            fetch(record.format(name), path)
            if md5(path) != checksum:
                path.unlink()
                raise SystemExit(f"{name}: the checksum does not match the Zenodo record; run --download again")
        with tempfile.TemporaryDirectory(dir=folder) as tmp, zipfile.ZipFile(path) as z:
            z.extractall(tmp)  # zipfile keeps every file inside tmp; moved into place only once whole
            for item in Path(tmp).iterdir():
                target = folder / item.name
                if target.is_dir():
                    shutil.rmtree(target)
                elif target.exists():
                    target.unlink()
                item.replace(target)
        print(f"unpacked {name}")


def read_mraw(folder):
    """Frame rate and frames of a Photron recording (.cih header + 8-bit .mraw)."""
    folder = Path(folder)
    header = folder / f"{folder.name}.cih"
    cih = dict(line.split(" : ", 1) for line in header.read_text(encoding="latin-1").splitlines() if " : " in line)
    keys = ("Record Rate(fps)", "Image Width", "Image Height", "Total Frame", "Color Bit")
    if any(k not in cih for k in keys):
        raise SystemExit(f"{header} lacks one of {', '.join(keys)}")
    if cih["Color Bit"].strip() != "8":
        raise SystemExit("only 8-bit .mraw is read here")
    w, h, n = int(cih["Image Width"]), int(cih["Image Height"]), int(cih["Total Frame"])
    film = folder / f"{folder.name}.mraw"
    if film.stat().st_size != n * h * w:
        raise SystemExit(f"{film} holds {film.stat().st_size} bytes, not the {n * h * w} its header gives: "
                         "delete the folder and run --download again")
    frames = np.memmap(film, dtype=np.uint8, mode="r").reshape(n, h, w)
    return float(cih["Record Rate(fps)"]), frames


def read_signals(path):
    """Force (N), acceleration (m/s^2) and their sample rate (Hz)."""
    with h5py.File(path, "r") as f:
        g = f["InputTask"]
        names = [x.decode() if isinstance(x, bytes) else x for x in g["channel_names"][()]]
        data = g["data"][()]
        return data[:, names.index("force")], data[:, names.index("acc")], int(g["sample_rate"][()])


def one_curve(H, band):
    """Many camera channels -> one response: weighted by the first singular vector in the band."""
    u, _, _ = np.linalg.svd(H[:, band], full_matrices=False)
    return u[:, 0].conj() @ H


def resonance_model(p, w):
    """mode_fit's model at angular frequencies w (rad/s): one resonance plus a constant and a 1/w^2 term, from its
    parameters (natural frequency in rad/s, damping ratio, then the real and imaginary parts of A, C and D)."""
    wn, z, ar, ai, cr, ci, dr, di = p
    return (ar + 1j * ai) / (wn ** 2 - w ** 2 + 2j * z * wn * w) + (cr + 1j * ci) + (dr + 1j * di) / w ** 2


def mode_fit(f, H, band, params=False):
    """Natural frequency (Hz), damping ratio, relative misfit of one resonance fitted in a band, and whether the fit
    stopped at a limit (a damping ratio of 0.0001 or 0.2, or a frequency at the band's edge), which makes it doubtful.
    H is displacement per force: the constant stands for the resonances above the band and the 1/w^2 term for those
    below it, which holds for displacement and not for acceleration (where the two change places with w^2).
    With params=True, the fitted parameters for resonance_model() come last."""
    w, y = 2 * np.pi * f[band], H[band]
    k = np.argmax(np.abs(y))
    scale = np.abs(y).max()

    def misfits(p):
        r = (resonance_model(p, w) - y) / scale
        return np.concatenate([r.real, r.imag])

    a0 = y[k] * 2j * 0.01 * w[k] ** 2
    sol = least_squares(misfits, [w[k], 0.01, a0.real, a0.imag, 0, 0, 0, 0],
                        bounds=([w[0], 1e-4] + [-np.inf] * 6, [w[-1], 0.2] + [np.inf] * 6))
    misfit = np.sqrt(np.mean(np.abs(resonance_model(sol.x, w) - y) ** 2)) / scale
    wn, z = sol.x[:2]
    edge = 0.01 * (w[-1] - w[0])  # the solver stops just inside a limit, not on it
    at_limit = z > 0.98 * 0.2 or z < 1.5e-4 or wn - w[0] < edge or w[-1] - wn < edge
    result = (wn / 2 / np.pi, z, misfit, bool(at_limit))
    return result + (sol.x,) if params else result


def impulse_response(f, H, lo, hi, fps, n=None):
    """The band as an impulse response (how the part would ring after an ideal knock): the damped natural frequency
    (Hz) from its spectrum, the response from the end of the taper's smear on, and that start (frames). n is the
    record's length in frames (an even length is assumed when it is not given)."""
    n = n or 2 * (len(f) - 1)
    band = (f >= lo) & (f <= hi)
    taper = np.zeros(len(f))
    taper[band] = np.hanning(band.sum())
    h = np.fft.irfft(H * taper, n)[: n // 2]
    fd = free_decay.frequency(h, fps)
    x = free_decay.positive_first(h)
    start = int(2 * fps / (hi - lo))  # skip the taper's smear at the start
    return fd, x[start:], start


def ring_down(f, H, lo, hi, fps, n=None):
    """The band's impulse response read as a tap test reads a film: (frequency Hz, damping ratio, peaks)."""
    fd, x, _ = impulse_response(f, H, lo, hi, fps, n)
    try:
        zeta, _, peaks, _ = free_decay.damping(x, fps, fd, floor=0.05 * np.abs(x).max())
    except ValueError:
        return fd, np.nan, 0
    if zeta <= 0:  # peaks that do not shrink: no ring-down to read
        return fd, np.nan, len(peaks)
    return fd, zeta, len(peaks)


def grid_points(view):
    """The 6 x 6 grid of points on the plate in a view's first frame (x, y), px."""
    x0, x1, y0, y1 = GRIDS[view]
    return [(x, y) for y in np.linspace(y0, y1, 6).astype(int) for x in np.linspace(x0, x1, 6).astype(int)]


def tracker_channels(frames, view):
    """The tracker's x and y shift of each grid point (px), one channel per row."""
    points = grid_points(view)
    shifts = np.stack(free_decay.track(iter(frames), points))  # points x frames x 2, px
    if np.isnan(shifts).any():
        raise SystemExit("a point was lost: move the grid")
    return np.concatenate([shifts[:, :, 0], shifts[:, :, 1]]), f"camera, tracker ({len(points)} points)"


def brightness_channels(frames, view):
    """The brightness of the steepest tenth of the plate's pixels (grey levels), one channel per row."""
    x0, x1, y0, y1 = GRIDS[view]
    first = frames[0].astype(np.float32)
    slope = np.hypot(cv2.Sobel(first, cv2.CV_32F, 1, 0), cv2.Sobel(first, cv2.CV_32F, 0, 1)) / 8
    plate = np.zeros(first.shape, bool)
    plate[y0:y1, x0:x1] = True
    pick = plate & (slope >= np.percentile(slope[plate], 90))
    return frames[:, pick].astype(np.float32).T, f"camera, brightness ({pick.sum()} pixels)"


def responses(signals, force, acc, fps, fs):
    """Frequencies (Hz); the camera channels' spectra; the accelerometer's and the camera's motion per newton.
    The accelerometer's acceleration is turned into displacement (divided by -(2 pi f)^2), so that both responses
    are displacement per force, the form mode_fit's model and its two residual terms are written for."""
    n = signals.shape[1]
    if abs(n / fps - len(force) / fs) > 1 / fs:  # one signal sample of slack
        raise SystemExit("film and signals differ in length")
    f = np.fft.rfftfreq(n, 1 / fps)  # same bins as the signals' over the same time
    F = np.fft.rfft(force)[: len(f)]
    X = np.fft.rfft(signals - signals.mean(axis=1, keepdims=True), axis=1) * (fs / fps)  # same scale as F
    w2 = (2 * np.pi * f) ** 2
    head = -np.fft.rfft(acc)[: len(f)] / F / np.where(w2 > 0, w2, np.inf)  # m / N, 0 at 0 Hz
    return f, X, head, X / F  # m / N; px / N or grey levels / N


def main():
    ap = argparse.ArgumentParser(description="Resonances of the printer-bed plate from its films, against an "
                                             "accelerometer.")
    ap.add_argument("dataset", nargs="?", default="printer-bed-v1", help="folder with mraw/ and measured_signals/")
    ap.add_argument("--download", action="store_true", help="fetch and unpack the data into that folder first")
    ap.add_argument("--view", type=int, choices=(0, 1), default=0, help="which of the two films")
    ap.add_argument("--reading", choices=("tracker", "brightness"), default="tracker")
    ap.add_argument("--plot", help="save both responses to this image")
    args = ap.parse_args()

    root = Path(args.dataset)
    if args.download:
        download(root)
    missing = [name for name in NEEDED if not (root / name).exists()]
    if missing:
        raise SystemExit(f"{root} lacks {missing[0]}: run with --download, or give the folder the data are in")
    lo_sweep, hi_sweep = SWEEP
    fps, frames = read_mraw(root / "mraw" / f"view_{args.view}_01")
    force, acc, fs = read_signals(root / "measured_signals" / f"view_{args.view}.hdf5")
    t0 = time.time()
    read = tracker_channels if args.reading == "tracker" else brightness_channels
    signals, label = read(frames, args.view)  # channels x frames
    n = signals.shape[1]
    f, X, H_head, H_cam = responses(signals, force, acc, fps, fs)
    shaken = (f >= lo_sweep) & (f <= hi_sweep)
    rms = np.sqrt(2 * (np.abs(X[:, shaken] * (fps / fs)) ** 2).sum(axis=1)) / n  # each channel, in the sweep read
    if args.reading == "tracker":
        rms = np.sqrt(rms[: len(rms) // 2] ** 2 + rms[len(rms) // 2:] ** 2)  # x and y of each point together
    what, unit = ("motion", "px") if args.reading == "tracker" else ("brightness change", "grey levels")
    print(f"view {args.view}, {label}: {n} frames at {fps:g} fps, read in {time.time() - t0:.1f} s; "
          f"{what} in the shaken band ({lo_sweep}-{hi_sweep} Hz) {rms.min():.3f}-{rms.max():.3f} {unit} rms")

    print(f"{'band, Hz':>9} | {'accelerometer':^30} | {'camera':^30}")
    print(f"{'':>9} | {'fit Hz':>7} {'zeta':>6} {'misfit':>6} {'ring zeta':>9} | "
          f"{'fit Hz':>7} {'zeta':>6} {'misfit':>6} {'ring zeta':>9}")
    marks = []
    for lo, hi in BANDS:
        band = (f >= lo) & (f <= hi)
        cam = one_curve(H_cam, band)
        fa, za, ma, ba = mode_fit(f, H_head, band)
        fc, zc, mc, bc = mode_fit(f, cam, band)
        ra, rc = ring_down(f, H_head, lo - 5, hi + 5, fps, n), ring_down(f, cam, lo - 5, hi + 5, fps, n)
        flag = lambda at_limit: "*" if at_limit else " "  # noqa: E731
        print(f"{lo:>4}-{hi:<4} | {fa:7.2f} {za:6.4f}{flag(ba)}{ma:6.2f} {ra[1]:9.4f} | "
              f"{fc:7.2f} {zc:6.4f}{flag(bc)}{mc:6.2f} {rc[1]:9.4f}")
        marks.append(fc)
    print("misfit: the fit's error against the response's largest value; * marks a fit stopped at a limit; "
          "ring zeta: the ring-down's damping ratio (nan when too few peaks)")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        band = (f >= lo_sweep) & (f <= hi_sweep)
        head = np.abs(H_head[band])
        cam = np.sqrt((np.abs(H_cam[:, band]) ** 2).sum(axis=0))
        plt.figure(figsize=(8, 3.6))
        plt.semilogy(f[band], head / head.max(), lw=0.9, label="accelerometer (at the shaker)")
        plt.semilogy(f[band], cam / cam.max(), lw=0.9, label=label)
        for m in marks:
            plt.axvline(m, color="0.6", lw=0.6, ls=":")
        plt.xlabel("frequency, Hz")
        plt.ylabel("response / force, scaled to 1")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(args.plot, dpi=150)


if __name__ == "__main__":
    main()
