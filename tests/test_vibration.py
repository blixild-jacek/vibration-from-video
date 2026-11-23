"""Tests: the readings on signals, films and files with known answers, and the download against a local server.

Run from the repository folder: python3 -m unittest discover tests

SPDX-License-Identifier: MIT
Copyright (c) 2026 Jacek Garbuliński Blixild, blixild.com
"""
import http.server
import io
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import check_printer_bed  # noqa: E402
import free_decay  # noqa: E402
import make_test_clip  # noqa: E402
import speed_test  # noqa: E402


def ring(fn, zeta, fps, seconds, amplitude=10.0, noise=0.0, seed=0):
    """A damped swing sampled at fps, with optional noise."""
    t = np.arange(int(seconds * fps)) / fps
    wn = 2 * np.pi * fn
    x = amplitude * np.exp(-zeta * wn * t) * np.cos(wn * np.sqrt(1 - zeta ** 2) * t)
    return x + np.random.default_rng(seed).normal(0, noise, len(x)) if noise else x


def resonance(f, fn, zeta):
    """One resonance's response to force at frequencies f (Hz)."""
    w, wn = 2 * np.pi * f, 2 * np.pi * fn
    return 1 / (wn ** 2 - w ** 2 + 2j * zeta * wn * w)


def run(script, *args, cwd=None):
    return subprocess.run([sys.executable, str(HERE / script), *args], capture_output=True, text=True, cwd=cwd,
                          timeout=300)


class Readings(unittest.TestCase):
    def test_frequency_between_bins_is_refined(self):
        # 3 s at 60 fps: 8x padded bins are 0.042 Hz apart, so an unrefined peak can be 0.02 Hz off
        for fn in (2.013, 7.37, 11.11):
            x = ring(fn, 0.002, 60, 3)
            self.assertAlmostEqual(free_decay.frequency(x, 60), fn * np.sqrt(1 - 0.002 ** 2), delta=0.003)

    def test_damping_ratio_uses_the_exact_formula(self):
        # at a damping ratio of 0.2 the shortcut decrement / 2 pi reads 2 % high; the exact formula does not
        x = ring(5.0, 0.2, 500, 3, amplitude=100.0)
        fd = 5.0 * np.sqrt(1 - 0.2 ** 2)
        zeta, decrement, _, _ = free_decay.damping(x, 500, fd, floor=0.5)
        self.assertAlmostEqual(zeta, 0.2, delta=0.001)
        self.assertAlmostEqual(decrement, 2 * np.pi * 0.2 / np.sqrt(1 - 0.04), delta=0.005)

    def test_known_ring_downs(self):
        for fn, zeta, fps in [(2.0, 0.03, 60), (45.0, 0.005, 500), (10.0, 0.02, 60)]:
            x = ring(fn, zeta, fps, 8)
            fd = free_decay.frequency(x, fps)
            self.assertAlmostEqual(fd, fn * np.sqrt(1 - zeta ** 2), delta=fn * 1e-3)
            found, _, _, _ = free_decay.damping(x, fps, fd, floor=0.5)
            self.assertAlmostEqual(found, zeta, delta=zeta * 0.02)

    def test_too_few_peaks_is_a_value_error(self):
        with self.assertRaises(ValueError) as caught:
            free_decay.damping(ring(2.0, 0.3, 60, 5, amplitude=2.0), 60, 2.0, floor=1.0)
        self.assertEqual(caught.exception.args[0].code, "few-peaks")  # a page words it by its code

    def test_a_warning_carries_its_code_and_values(self):
        *_, warnings = free_decay.measure(ring(2.0, 0.02, 60, 3), 60, floor=1.0)  # cut before rest
        note = next(w for w in warnings if w.code == "still-swinging")
        self.assertTrue(note.startswith("the part still swings"))  # printed as before
        *_, warnings = free_decay.measure(ring(12.0, 0.02, 60, 8), 60, floor=1.0)
        note = next(w for w in warnings if w.code == "few-frames")
        self.assertAlmostEqual(note.values["per_swing"], 5, delta=0.01)

    def test_a_long_quiet_tail_does_not_hide_the_frequency(self):
        # 40 Hz at 240 fps, filmed for 5 s: the ringing is over in 0.3 s; the rest is noise
        x = ring(40.0, 0.05, 240, 5, amplitude=10.0, noise=0.03)
        fd, fn, zeta, _, _, _ = free_decay.measure(x, 240, floor=0.3)
        self.assertAlmostEqual(fn, 40.0, delta=0.4)
        self.assertAlmostEqual(zeta, 0.05, delta=0.005)

    def test_a_film_cut_before_rest_is_warned_about(self):
        _, _, _, _, _, warnings = free_decay.measure(ring(2.0, 0.03, 60, 3, amplitude=40.0), 60, floor=1.0)
        self.assertTrue(any("last tenth" in w for w in warnings))

    def test_noise_reads_white_jitter(self):
        jitter = np.random.default_rng(5).normal(0, 0.1, (2000, 2))
        self.assertAlmostEqual(free_decay.noise_of(jitter), 0.1, delta=0.01)

    def test_a_film_that_starts_before_the_tap_is_read_from_the_tap(self):
        x = np.concatenate([np.random.default_rng(6).normal(0, 0.05, 180), ring(20.0, 0.02, 60, 5, amplitude=5.0)])
        fd, fn, zeta, _, peaks, _ = free_decay.measure(x, 60, floor=0.5)
        self.assertAlmostEqual(fn, 20.0, delta=0.02)
        self.assertGreater(peaks.min(), 180)

    def test_a_slow_drift_does_not_win_the_frequency(self):
        t = np.arange(600) / 60
        x = ring(30.0, 0.01, 60, 10, amplitude=3.0) + np.sin(2 * np.pi * 0.07 * t)
        self.assertAlmostEqual(free_decay.frequency(x, 60), 30.0, delta=0.05)

    def test_an_offset_rest_position_does_not_bias_the_damping(self):
        x = ring(2.0, 0.03, 60, 15, amplitude=10.0) + 0.5
        zeta, _, _, _ = free_decay.damping(x, 60, 2.0, floor=1.0)
        self.assertAlmostEqual(zeta, 0.03, delta=0.0006)

    def test_growing_peaks_stop_with_a_message(self):
        with self.assertRaises(ValueError):
            free_decay.measure(ring(2.0, -0.01, 60, 8, amplitude=5.0), 60, floor=1.0)

    def test_noise_leaves_out_a_slow_camera_wobble(self):
        t = np.arange(600) / 60
        fixed = np.stack([np.sin(2 * np.pi * 0.5 * t), np.zeros_like(t)], axis=1)  # a 1 px, 0.5 Hz wobble
        self.assertLess(free_decay.noise_of(fixed), 0.05)

    def test_one_resonance_fit(self):
        f = np.arange(0.5, 500.5, 0.5)
        band = (f >= 118) & (f <= 140)
        fn, zeta, misfit, at_limit = check_printer_bed.mode_fit(f, resonance(f, 126.0, 0.013), band)
        self.assertAlmostEqual(fn, 126.0, delta=0.05)
        self.assertAlmostEqual(zeta, 0.013, delta=0.013 * 0.02)
        self.assertLess(misfit, 0.01)
        self.assertFalse(at_limit)

    def test_the_fit_returns_its_model(self):
        f = np.arange(0.5, 500.5, 0.5)
        band = (f >= 60) & (f <= 95)
        H = resonance(f, 76.0, 0.02)
        fn, zeta, misfit, _, p = check_printer_bed.mode_fit(f, H, band, params=True)
        model = check_printer_bed.resonance_model(p, 2 * np.pi * f[band])
        self.assertLess(np.abs(model - H[band]).max() / np.abs(H[band]).max(), 0.01)
        self.assertAlmostEqual(p[0] / (2 * np.pi), fn)

    def test_a_fit_at_its_limit_is_flagged(self):
        f = np.arange(0.5, 500.5, 0.5)
        band = (f >= 118) & (f <= 140)
        self.assertTrue(check_printer_bed.mode_fit(f, resonance(f, 126.0, 0.5), band)[3])

    def test_ring_down_of_one_resonance(self):
        for n in (2000, 1999):  # an odd length too
            f = np.fft.rfftfreq(n, 1 / 1000)
            fd, zeta, peaks = check_printer_bed.ring_down(f, resonance(f, 76.0, 0.02), 60, 95, 1000.0, n)
            self.assertAlmostEqual(fd, 76.0 * np.sqrt(1 - 0.02 ** 2), delta=0.005)
            self.assertAlmostEqual(zeta, 0.02, delta=0.02 * 0.02)
            self.assertGreater(peaks, 10)

    def test_response_is_motion_per_newton(self):
        fps, fs, seconds = 1000, 25600, 2
        t_cam, t_sig = np.arange(fps * seconds) / fps, np.arange(fs * seconds) / fs
        force = 2.0 * np.sin(2 * np.pi * 100 * t_sig)
        motion = 0.05 * np.sin(2 * np.pi * 100 * t_cam - 0.3)[None, :]  # px, 0.3 rad behind the force
        f, _, _, H = check_printer_bed.responses(motion, force, force, fps, fs)
        k = np.argmin(abs(f - 100))
        self.assertAlmostEqual(abs(H[0, k]), 0.025, delta=1e-4)
        self.assertAlmostEqual(np.angle(H[0, k]), -0.3, delta=0.01)

    def test_accelerometer_response_is_displacement_per_newton(self):
        fps, fs, seconds = 1000, 25600, 2
        t_sig = np.arange(fs * seconds) / fs
        w = 2 * np.pi * 100
        force = 2.0 * np.sin(w * t_sig)
        acc = -w ** 2 * 1e-6 * np.sin(w * t_sig - 0.3)  # the acceleration of 1 micrometre, 0.3 rad behind the force
        f, _, H, _ = check_printer_bed.responses(np.zeros((1, fps * seconds)), force, acc, fps, fs)
        k = np.argmin(abs(f - 100))
        self.assertAlmostEqual(abs(H[k]), 0.5e-6, delta=1e-9)
        self.assertAlmostEqual(np.angle(H[k]), -0.3, delta=0.01)
        self.assertEqual(H[0], 0)


class SpeedTest(unittest.TestCase):
    FPS = 240

    def read(self, swing, seconds, floor=1.0):
        t = np.arange(int(seconds * self.FPS)) / self.FPS
        x = swing(t) + np.random.default_rng(0).normal(0, 0.05, len(t))
        return speed_test.resonance(*speed_test.response(x, self.FPS, floor)[1:], self.FPS)

    def test_a_coast_down_finds_the_resonance_within_two_percent(self):
        for zeta in (0.01, 0.02, 0.05):
            for seconds in (10, 40):
                peak, size, notes = self.read(make_test_clip.coast_down(12.0, zeta, 40.0, 20.0, 6.0, seconds), seconds)
                self.assertLess(abs(peak - 12.0) / 12.0, 0.02, (zeta, seconds, peak))
                self.assertLess(peak, 12.0)  # a coast-down passes the peak before the part has fully answered
                self.assertEqual(notes, [])

    def test_speed_steps_find_the_step_nearest_the_resonance(self):
        speeds = [9, 10, 11, 11.5, 12, 12.5, 13, 14, 15]
        peak, size, notes = self.read(make_test_clip.driven(12.0, 0.02, 40.0, make_test_clip.steps(speeds, 18), 18), 18)
        self.assertLess(abs(peak - 12.0), 0.25)

    def test_a_resonance_at_few_frames_per_swing_reads_in_full(self):
        peak, size, notes = self.read(make_test_clip.coast_down(80.0, 0.02, 40.0, 128.0, 40.0, 20), 20)
        self.assertLess(abs(peak - 80.0) / 80.0, 0.02)  # 3 frames per swing at 240 frames a second
        self.assertAlmostEqual(size, 40.0, delta=0.5)
        self.assertEqual(notes, [])

    def test_a_resonance_near_half_the_frame_rate_is_warned_about(self):
        peak, size, notes = self.read(make_test_clip.coast_down(105.0, 0.02, 40.0, 115.0, 60.0, 20), 20)
        self.assertEqual([n.code for n in notes], ["near-half"])

    def test_a_resonance_beyond_the_speeds_filmed_is_warned_about(self):
        peak, size, notes = self.read(make_test_clip.coast_down(25.0, 0.02, 40.0, 20.0, 6.0, 20), 20)
        self.assertEqual([n.code for n in notes], ["edge-peak"])
        self.assertEqual(notes[0].values["side"], "highest")

    def test_a_swing_below_the_floor_is_not_read(self):
        with self.assertRaises(ValueError) as caught:
            self.read(make_test_clip.coast_down(12.0, 0.02, 0.5, 20.0, 6.0, 10), 10)
        self.assertEqual(caught.exception.args[0].code, "below-floor")


class Films(unittest.TestCase):
    def test_tracker_reads_a_generated_film(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "clip.mp4")
            make_test_clip.make(out, fn=2.0, zeta=0.03, fps=60, seconds=8)
            stored, frames = free_decay.read_video(out)
            self.assertAlmostEqual(stored, 60, delta=0.5)
            part, fixed = free_decay.track(frames, [make_test_clip.first_tip(40.0), make_test_clip.FIXED])
            x = free_decay.swing(part, fixed)
            fd, fn, zeta, _, _, _ = free_decay.measure(x, 60, floor=1.0)
            self.assertAlmostEqual(fn, 2.0, delta=0.01)
            self.assertAlmostEqual(zeta, 0.03, delta=0.002)

    def test_frames_handed_one_at_a_time_read_as_the_whole_film(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "clip.mp4")
            make_test_clip.make(out, fn=2.0, zeta=0.03, fps=60, seconds=8)
            frames = list(free_decay.read_video(out)[1])
            points = [make_test_clip.first_tip(40.0), make_test_clip.FIXED]
            whole = free_decay.track(frames, points)
            tracker = free_decay.Tracker(frames[0], points)
            for gray in frames[1:]:
                tracker.step(gray)
            for a, b in zip(whole, tracker.shifts()):
                np.testing.assert_array_equal(a, b)
            x, noise, floor, (fd, fn, zeta, _, peaks, warnings) = free_decay.read(*whole, 60)
            self.assertEqual(floor, 1.0)
            self.assertAlmostEqual(fn, 2.0, delta=0.01)
            self.assertAlmostEqual(zeta, 0.03, delta=0.002)

    def test_the_speed_test_reads_a_coast_down_film(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "coast.mp4")
            make_test_clip.make(out, fn=12.0, zeta=0.02, fps=240, seconds=10, coast=(20.0, 6.0))
            fps, frames = free_decay.read_video(out)
            part, fixed = free_decay.track(frames, [make_test_clip.TIP, make_test_clip.FIXED])
            x, noise, floor, windows, (peak, size, notes) = speed_test.read(part, fixed, fps)
            self.assertLess(abs(peak - 12.0) / 12.0, 0.02)
            self.assertGreater(size, 30)

    def test_a_blank_frame_leaves_a_gap_not_an_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "clip.mp4")
            make_test_clip.make(out, fn=2.0, zeta=0.03, fps=60, seconds=4)
            frames = list(free_decay.read_video(out)[1])
            clean = free_decay.track(frames, [make_test_clip.first_tip(40.0)])[0]
            frames[7] = np.full_like(frames[7], 128)
            gapped = free_decay.track(frames, [make_test_clip.first_tip(40.0)])[0]
            self.assertTrue(np.isnan(gapped[7]).all())
            self.assertLess(np.nanmax(np.abs(gapped[8:] - clean[8:])), 0.2)

    def test_the_command_line_reads_a_film(self):
        with tempfile.TemporaryDirectory() as tmp:
            made = run("make_test_clip.py", "clip.mp4", cwd=tmp)  # 15 s: the part settles, so no warning is due
            self.assertEqual(made.returncode, 0, made.stderr)
            command = made.stdout.splitlines()[-1].split()[2:]  # the printed "python3 free_decay.py ..." line
            read = run("free_decay.py", *command, cwd=tmp)
            self.assertEqual(read.returncode, 0, read.stderr)
            self.assertIn("natural frequency 2.00", read.stdout)
            self.assertIn("damping ratio 0.03", read.stdout)
            self.assertNotIn("warning", read.stdout)

    def test_no_points_saves_a_grid_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            run("make_test_clip.py", "clip.mp4", "--seconds", "1", cwd=tmp)
            read = run("free_decay.py", "clip.mp4", cwd=tmp)
            self.assertNotEqual(read.returncode, 0)
            self.assertIn("clip_first_frame.png", read.stderr)
            self.assertGreater((Path(tmp) / "clip_first_frame.png").stat().st_size, 1000)

    def test_a_square_over_the_part_and_its_background_is_warned_about(self):
        # a 12 px strip swinging 20 px, with still texture beside it: a 41 px square covers both
        rng = np.random.default_rng(3)
        background = np.repeat(np.repeat(rng.integers(0, 255, (30, 40)), 4, 0), 4, 1).astype(np.uint8)
        strip = np.repeat(np.repeat(rng.integers(0, 255, (20, 3)), 4, 0), 4, 1).astype(np.uint8)
        frames = []
        for k in range(120):
            frame = background.copy()
            x = 74 + int(round(20 * np.exp(-0.05 * k) * np.cos(2 * np.pi * k / 20)))
            frame[20:100, x:x + 12] = strip
            frames.append(frame)
        out = io.StringIO()
        with redirect_stdout(out):
            free_decay.track(frames, [(80, 60), (20, 20)], report=True)
        self.assertIn("warning: the square around (80, 60)", out.getvalue())

    def test_a_square_wholly_on_the_part_is_not_warned_about(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "clip.mp4")
            make_test_clip.make(out, seconds=3)
            printed = io.StringIO()
            with redirect_stdout(printed):
                free_decay.track(free_decay.read_video(out)[1], [make_test_clip.first_tip(40.0)], report=True)
            self.assertNotIn("warning", printed.getvalue())

    def test_plain_grey_has_too_little_texture(self):
        grey = [np.clip(128 + np.random.default_rng(k).normal(0, 2, (120, 160)), 0, 255).astype(np.uint8)
                for k in range(3)]
        with self.assertRaises(SystemExit) as caught:
            free_decay.track(grey, [(80, 60)])
        self.assertEqual(caught.exception.args[0].code, "texture")

    def test_a_ring_down_that_grows_is_not_read(self):
        f = np.fft.rfftfreq(2000, 1 / 1000)
        _, zeta, _ = check_printer_bed.ring_down(f, resonance(f, 76.0, -0.01), 60, 95, 1000.0, 2000)
        self.assertTrue(np.isnan(zeta))

    def test_brightness_reading_picks_the_steepest_pixels_on_the_plate(self):
        rng = np.random.default_rng(4)
        frames = np.repeat(rng.integers(0, 255, (1, 384, 384)).astype(np.uint8), 5, axis=0)
        signals, label = check_printer_bed.brightness_channels(frames, 0)
        x0, x1, y0, y1 = check_printer_bed.GRIDS[0]
        self.assertEqual(signals.shape[1], 5)
        self.assertAlmostEqual(signals.shape[0] / ((x1 - x0) * (y1 - y0)), 0.1, delta=0.02)
        self.assertIn("brightness", label)


class Files(unittest.TestCase):
    def write_film(self, folder, n=5, h=4, w=6, short=0):
        folder.mkdir(parents=True)
        (folder / f"{folder.name}.cih").write_text(
            f"Record Rate(fps) : 1000.0\nImage Width : {w}\nImage Height : {h}\nTotal Frame : {n}\nColor Bit : 8\n",
            encoding="latin-1")
        (folder / f"{folder.name}.mraw").write_bytes(bytes(range(n * h * w - short)))

    def test_read_mraw_and_its_size_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            good, short = Path(tmp) / "view_0_01", Path(tmp) / "view_1_01"
            self.write_film(good)
            fps, frames = check_printer_bed.read_mraw(good)
            self.assertEqual((fps, frames.shape, int(frames[1, 0, 0])), (1000.0, (5, 4, 6), 24))
            self.write_film(short, short=7)
            with self.assertRaises(SystemExit):
                check_printer_bed.read_mraw(short)

    def test_read_signals(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.hdf5"
            with h5py.File(path, "w") as f:
                g = f.create_group("InputTask")
                g["channel_names"] = np.array([b"force", b"acc", b"trigger"])
                g["data"] = np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]])
                g["sample_rate"] = 25600
            force, acc, fs = check_printer_bed.read_signals(path)
            self.assertEqual((list(force), list(acc), fs), ([1.0, 3.0], [2.0, 4.0], 25600))

    def test_missing_data_is_a_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = run("check_printer_bed.py", str(Path(tmp) / "nowhere"))
            self.assertNotEqual(out.returncode, 0)
            self.assertIn("--download", out.stderr)
            self.assertNotIn("Traceback", out.stderr)


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


class Download(unittest.TestCase):
    """The download against a local server: the checksum is checked, a bad one stops it, no network is a message."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        served = Path(self.tmp.name) / "served"
        served.mkdir()
        with zipfile.ZipFile(served / "small.zip", "w") as z:
            z.writestr("small/data.txt", "data")
        self.checksum = check_printer_bed.md5(served / "small.zip")
        self.server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), lambda *a, **k: Quiet(*a, directory=str(served), **k))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.record = f"http://127.0.0.1:{self.server.server_address[1]}/{{}}"
        self.dest = Path(self.tmp.name) / "dest"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def test_downloads_checks_and_unpacks(self):
        with redirect_stdout(io.StringIO()):
            check_printer_bed.download(self.dest, {"small.zip": self.checksum}, self.record, ["small/data.txt"])
        self.assertEqual((self.dest / "small" / "data.txt").read_text(), "data")

    def test_a_wrong_checksum_stops_it(self):
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            check_printer_bed.download(self.dest, {"small.zip": "0" * 32}, self.record, ["small/data.txt"])
        self.assertFalse((self.dest / "small").exists())
        self.assertFalse((self.dest / "small.zip").exists())

    def test_no_server_is_a_message_and_leaves_no_file(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]  # a port nothing listens on once closed
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as stop:
            check_printer_bed.download(self.dest, {"small.zip": self.checksum}, f"http://127.0.0.1:{port}/{{}}",
                                       ["small/data.txt"])
        self.assertIn("cannot download", str(stop.exception))
        self.assertEqual([p.name for p in self.dest.iterdir()], [])


if __name__ == "__main__":
    unittest.main()
