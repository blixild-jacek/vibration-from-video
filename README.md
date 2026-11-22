# vibration-from-video

Reads a part's natural frequency and damping ratio from a video of it, with no sensor on the part. Tap the part and
film it ringing down, or shake it with a shaker and a force sensor while a fast camera films it, and these scripts
follow the motion in the film and read both numbers.

It is an example to adapt, written up on the Blixild blog in
[How to measure damping ratio and natural frequency on video](https://blixild.com/measure-damping-ratio-natural-frequency-video.html).

## What is in it

- `free_decay.py`: the tap test. It reads a film of a part ringing after a tap.
- `check_printer_bed.py`: the shaker test, on open high-speed films of a shaken plate with the force and the
  acceleration measured where the shaker pushes ([printer-bed dataset](https://doi.org/10.5281/zenodo.21476609),
  CC BY 4.0). It downloads the data itself.
- `make_test_clip.py`: makes a test film with a natural frequency and damping ratio you choose, to check the
  arithmetic.
- `tests/`: unit tests.

## How it works

**The tap test.** Two points are picked in the first frame, one on the part and one on something that does not move.
A feature tracker follows the texture around each (Shi-Tomasi corners and Lucas-Kanade optical flow from OpenCV,
each corner checked by following it back), and the part's swing is taken relative to the fixed point, which cancels
small shifts of the camera. The highest peak of the swing's spectrum, while it rings above the noise, gives the
damped natural frequency. The upper
peaks of the swing shrink by the same ratio every swing; the logarithm of that ratio is the logarithmic decrement δ,
and the damping ratio follows:

    δ = ln(one upper peak ÷ the next)
    damping ratio = δ ÷ √(4π² + δ²)

**The shaker test.** When the part moves less than a pixel, the swing cannot be read peak by peak. The shaker's
force is recorded with the film, and the filmed motion divided by the force, frequency by frequency, gives the
part's response to each newton. It peaks at each natural frequency. One resonance is fitted to each peak, which gives
its natural frequency and damping ratio; the same peak is also turned back into a ring-down and read as the tap test
reads a film. The script does both for the camera and for the accelerometer, whose acceleration it first turns into
displacement (divided by −ω², with ω = 2πf), since the fitted model's two extra terms for the neighbouring
resonances hold for displacement per force.

Neither needs millimetres: the frequency comes from time and the damping from a ratio of swings.

## Try it

You need Python 3.11 or newer. On Windows, type `python` or `py` where these steps say `python3`.

1. Install the libraries:

   ```bash
   python3 -m pip install -r requirements.txt
   ```

2. Make a test film and read it back. The first command prints the second:

   ```bash
   python3 make_test_clip.py
   python3 free_decay.py test_clip.mp4 --part 424 280 --fixed 104 84 --fps 60
   ```

   It should report a natural frequency of 2.000 Hz and a damping ratio of about 0.030 (0.0301 on our run), against
   the 2 Hz and 0.03 the film was made with.

3. Run the shaker test on the printer-bed films. The first run downloads about 450 MB from Zenodo and needs about
   1 GB of disk once unpacked:

   ```bash
   python3 check_printer_bed.py --download
   python3 check_printer_bed.py --view 1
   python3 check_printer_bed.py --reading brightness
   ```

Tests: `python3 -m unittest discover tests`.

## Use it on your own part

1. Decide how fast to film. The part's natural frequency is what you are after, so it is not known yet; the machine's
   running speeds (rpm ÷ 60 gives hertz) are the frequencies to compare it with afterwards. A part that rings faster
   than half the frame rate shows up at a false, lower frequency with no warning: film it at two frame rates, and a
   true frequency stays put while a false one moves.
2. Film the part from the side, so that it swings across the picture, with something that does not move in the same
   view. Film close enough that the swing after a tap spans a pixel or more, and fast enough for at least six frames
   per swing: 240 frames a second covers up to about 40 Hz.
3. Tap the part and keep filming until it stops. Use the original slow-motion recording, not an edited or shared
   copy: an exported slow-motion file often plays its start and end at normal speed, so one frame rate cannot time
   it all. If the ring-down lies in one stretch of the file, `--start` and `--end` (seconds on the file's own clock)
   keep only that stretch.
4. Run `python3 free_decay.py tap.mp4`. It saves `tap_first_frame.png` beside the film, with a pixel grid; read a
   point on the part and a point on something still, then run it again with them:

   ```bash
   python3 free_decay.py tap.mp4 --part 640 410 --fixed 120 90 --fps 240 --plot swing.png
   ```

   On a computer with a screen, `--pick` lets you click the two points instead. The script refuses a square with too
   little texture and warns when it is faint. Each point's square (`--box`, 41 px)
   must lie wholly on the part, or wholly on the still object: still corners in the part's square outvote the moving
   ones. For a thin strip, use a smaller `--box`; the script prints how many corners it found in each square and
   warns when a square holds corners that moved and corners that did not.
5. Check the frame rate it prints. A slow-motion file may store the rate it plays back at; `--fps` gives the rate it
   was filmed at. A wrong rate scales the frequency and leaves the damping ratio as it is. A file with a varying
   frame rate, as some phones record, is timed by its average rate.

It worked when two taps give the same frequency, the script prints no warning, and the peaks in `swing.png` shrink
evenly.

## Settings

| Option | Default | What it does |
| --- | --- | --- |
| `--fps` | the file's own | The rate the film was taken at. |
| `--box` | 41 px | The square around each point whose corners are followed. It must lie on the part (or on the still object). |
| `--min-peak` | 1 px, or ten times the still point's jitter if that is larger | The smallest peak the frequency and the damping are read from. On test films, peaks under about a pixel came out up to 17 % low and bent the reading. Given, it replaces the default. |
| `--plot` | none | Saves the swing, the peaks used and the floor. |

The jitter is how much a point seems to move from one frame to the next, the tracker's noise: the still point's, or
the settled swing's when a shaking camera makes the still point's larger. The swing's amplitude at each peak is half
the drop to the troughs either side, so a rest position that is a little off does not bias the damping. The
frequency and the damping are read from the knock (the largest swing) to the end of the first unbroken run of swings
above the floor. The script warns when the part still swings in the last tenth of
the film (it takes that as rest, and the damping ratio would read low) and when there are fewer than six frames per
swing.

What the scripts gave, on an Apple M3 laptop, 21-22 November 2026, in two installs: Python 3.13 with OpenCV 5.0,
NumPy 1.26 and SciPy 1.17 from conda, and Python 3.15 with OpenCV 5.1, NumPy 2.5 and SciPy 1.18 from
requirements.txt:

- **Test films** made by `make_test_clip.py`, 1 to 200 Hz: the frequency within 0.05 % and the damping ratio within
  2 % of the values set, with six or more frames per swing. At 3.75 frames per swing the tap test stopped, and at 2.5
  it read the damping 10 % low.
- **The printer-bed films**: the camera's resonances at 76, 126, 215 and 264 Hz within about 0.3 % of the
  accelerometer's, and their damping ratios within 7 % when both are read by the same fit. At about 110 Hz the camera
  reads 109 Hz and the accelerometer 112 Hz: there the accelerometer, where the shaker pushes, shows only a faint
  peak on the rising flank of the 126 Hz resonance, and its fit is doubtful. A `*` beside a fit's
  damping ratio marks a fit stopped at a limit of its range, to be doubted. The plate moved 0.01 to
  0.06 pixel in the shaken band. Tracking one film took about a minute; the brightness reading about a second.

## What it does not do

- **Running machines.** A part on a running machine moves micrometres: at 50 Hz, 1 mm/s is a swing of about
  3 micrometres, a few hundredths of a pixel, with no force recorded to divide by. That stays with an accelerometer.
- **Motion towards the camera.** One camera sees only motion across the picture. Millimetres, and motion along the
  line of sight, need a second camera.
- **Two resonances close together.** The decrement assumes one resonance. Where two share a band, it follows the one
  that dies away slowest: in one of the printer-bed films the ring-down read almost no damping at 126 and 215 Hz,
  where the fit read 0.013 and 0.021. The script warns when the peaks' logarithms do not lie on a straight line.
- **Every surface.** The tracker needs texture, such as corners, bolts or a sticker. Plain or shiny surfaces, glare,
  motion blur and flickering light are not covered by the tests, and the tap test has not yet read a filmed tap of a
  real part.
- **The brightness reading** is a quick first look: on the printer bed it read 221 Hz for the resonance at 215 Hz,
  and at 264 Hz it found the frequency but read the damping 30 to 40 % low.

## Using it on your own recording

`check_printer_bed.py` fits the printer-bed films only. For your own shaker test, change what fits that dataset: `SWEEP`
(the frequencies read), `BANDS` (one around each resonance), `GRIDS` (points whose squares lie on your part), and
`read_mraw()` and `read_signals()` for your files. The force and the film must be recorded over the same time.

## If something goes wrong

- **"cannot read" a video:** your OpenCV may lack the codec. The `opencv-python` package from pip reads MP4 and MOV.
- **`--pick` on a server:** there is no window; run without it and read the points from the saved first frame. On
  a server without a screen, `opencv-python-headless` installs without the window libraries.
- **A file name with accents or other letters outside English, on Windows:** OpenCV may not open the film; rename
  it or copy it to a plain path.
- **"cannot download":** check the connection. On macOS, a Python from python.org needs its certificates installed
  once (the "Install Certificates" command in its folder).

## Privacy

Everything runs on your computer. The only connection is `--download`, which fetches the dataset from Zenodo.

## Status

An example to adapt, not a maintained product. Issues and pull requests are read, without a promise of support.

## Licence

MIT, see [LICENSE](LICENSE). Copyright (c) 2026 Jacek Garbuliński Blixild, [blixild.com](https://blixild.com/).

The data: D. Gorjup, K. Zaletelj, J. Slavič (2026), "Printer-bed dataset: high-speed-camera and accelerometer modal
measurements of a suspended 3D-printer bed", Zenodo record 21476609 (the dataset's first version), [doi:10.5281/zenodo.21476609](https://doi.org/10.5281/zenodo.21476609),
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Their own processing code is at
[ladisk/mssp-image-tutorial](https://github.com/ladisk/mssp-image-tutorial). This repository does not include the
data; `--download` fetches it from Zenodo.
