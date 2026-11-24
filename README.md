# vibration-from-video

Reads a part's natural frequency and damping ratio from a video of it, with no sensor on the part. Tap the part and
film it ringing down, or shake it with a shaker and a force sensor while a fast camera films it, and these scripts
follow the motion in the video and read both numbers. Or film the part while its machine runs through its speeds, and
read the speed at which it shakes most.

It is an example to adapt, written up on the Blixild blog in
[How to measure damping ratio and natural frequency on video](https://blixild.com/measure-damping-ratio-natural-frequency-video.html).

## What is in it

- `free_decay.py`: the tap test. It reads a video of a part ringing after a tap.
- `speed_test.py`: the speed test. It reads a video of a part while its machine is stepped through its speeds or
  coasts down, and finds the speed at which the part shakes most.
- `check_printer_bed.py`: the shaker test, on open high-speed videos of a shaken plate with the force and the
  acceleration measured where the shaker pushes ([printer-bed dataset](https://doi.org/10.5281/zenodo.21476609),
  CC BY 4.0). It downloads the data itself.
- `make_test_clip.py`: makes a test video with a natural frequency and damping ratio you choose, ringing after a tap
  or shaken by a machine coasting down or stepped through speeds, to check the arithmetic.
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

**Below a pixel, the frequency alone.** When the swing is too small for the damping ratio, the script takes its slow
drift off (a high-pass at 0.5 Hz) and compares the spectrum of the stretch just after the tap with three stretches of
the same length at the video's end, where the part has stopped, and up to three before the tap. If the ring's peak
stands at least 5 times above the highest of theirs, and 5 times above its own spectrum's median, it gives the
frequency and no damping ratio. `faint_noise.py` checks both bars on simulated swings in half a minute:
- steady tracker noise with a slow drift of up to 30 times its size never passed the first bar (2.64 at most in 30,000
  swings);
- noise that is not steady never passed both (200 swings each at 25, 60 and 240 frames a second): the last 3 s one
  repeated frame, the last 6 s ten times quieter, noise five times larger for 4 s, a 3-frame blip;
- a ring whose first swing is ten times the noise was found in 66-79 % of cases (3 to 47 Hz), twenty times in
  90-100 %, its frequency within 1.5 % in nine cases of ten.

Anything that rings after the largest jolt is read, the camera included: a bump of the tripod near the end was taken
for the ring (42 % of cases at 60 frames a second, all at 240). So keep the camera untouched until the reading's end,
and stop the reading (`--end`) before anyone touches it.

**The shaker test.** When the part moves less than a pixel, the swing cannot be read peak by peak. The shaker's
force is recorded with the video, and the filmed motion divided by the force, frequency by frequency, gives the
part's response to each newton. It peaks at each natural frequency. One resonance is fitted to each peak, which gives
its natural frequency and damping ratio; the same peak is also turned back into a ring-down and read as the tap test
reads a video. The script does both for the camera and for the accelerometer, whose acceleration it first turns into
displacement (divided by −ω², with ω = 2πf), since the fitted model's two extra terms for the neighbouring
resonances hold for displacement per force.

**The speed test.** The machine shakes the part at the speed it runs. The part and a still point are followed as in
the tap test, and the swing is cut into windows six cycles long, a quarter of a window apart. In each window the
strongest frequency is what the part moves at, the running speed when it is shaken once a revolution, and its
amplitude is how far the part swings at that frequency. The window with the largest swing above the noise gives the
resonance the machine excites. A coast-down passes that peak before the part has fully answered, so it reads a little
low: by 0.1 to 1.8 % on generated coast-downs from 20 to 6 Hz lasting 10 to 80 s, through a 12 Hz resonance with
damping ratios of 0.01 to 0.05. The speed test gives no damping ratio: on the same videos the width of the peak read
it 3 to 29 % high or not at all. Tap the part for the damping ratio. The speed test warns when the swing does not
fall to 71 % of its largest on both sides of the peak within the speeds filmed (the half-power points that bound a
resonance): the resonance may then lie beyond them, or the speeds were stepped too coarsely. It prints how far apart
the windows near the peak lie, which is how finely it places the resonance.

None of them needs millimeters: the frequency comes from time and the damping from a ratio of swings.

## Try it

You need Python 3.11 or newer. On Windows, type `python` or `py` where these steps say `python3`.

1. Install the libraries:

   ```bash
   python3 -m pip install -r requirements.txt
   ```

2. Make a test video and read it back. The first command prints the second:

   ```bash
   python3 make_test_clip.py
   python3 free_decay.py test_clip.mp4 --part 424 280 --fixed 104 84 --fps 60
   ```

   It should report a natural frequency of 2.000 Hz and a damping ratio of about 0.030 (0.0301 on our run), against
   the 2 Hz and 0.03 the video was made with.

   A coast-down from 18 to 7 Hz through a 12 Hz resonance, read by the speed test:

   ```bash
   python3 make_test_clip.py coast_clip.mp4 --fn 12 --zeta 0.02 --fps 240 --seconds 12 --coast-down 18 7
   python3 speed_test.py coast_clip.mp4 --part 424 240 --fixed 104 84 --fps 240 --plot response.png
   ```

   It should report the largest swing at about 11.8 Hz (11.836 Hz on our run, an Apple M3, 23 November 2026), against
   the 12 Hz resonance the video was made with.

3. Run the shaker test on the printer-bed videos. The first run downloads about 450 MB from Zenodo and needs about
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
   view. For the damping ratio, film close enough that the swing after a tap spans a pixel or more; below that, the
   script gives the frequency alone, when the ring stands out of the video's noise. For the damping ratio, film fast
   enough for at least four frames per swing: 240 frames a second covers up to about 60 Hz, as long as the part moves
   under about 40 px from one frame to the next (the script warns). The natural frequency alone, and the speed test,
   need a little over two frames per swing: up to about 100 Hz at 240 frames a second.
3. Tap the part and keep filming until it stops, then a few seconds more without touching the camera: the script
   measures the noise there. Use the original slow-motion recording, not an edited or shared
   copy: an exported slow-motion file often plays its start and end at normal speed, so one frame rate cannot time
   it all. If the ring-down lies in one stretch of the file, `--start` and `--end` (seconds on the file's own clock)
   keep only that stretch: end it before anyone moves the camera or walks past.
4. Run `python3 free_decay.py tap.mp4`. It saves `tap_first_frame.png` beside the video, with a pixel grid; read a
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
   was filmed at. A wrong rate scales the frequency and leaves the damping ratio as it is. A file saved at a varying
   frame rate, as some phones record, would be timed by its average rate, which gives a wrong frequency: the script
   warns when the frames' times are unevenly spaced, and such a video is best saved again at a constant rate first
   (for example `ffmpeg -i in.mp4 -vf fps=240 out.mp4`, at the rate it was filmed).

It worked when two taps give the same frequency, the script prints no warning, and the peaks in `swing.png` shrink
evenly.

## Settings

| Option | Default | What it does |
| --- | --- | --- |
| `--fps` | the file's own | The rate the video was taken at. |
| `--box` | 41 px | The square around each point whose corners are followed. It must lie on the part (or on the still object). |
| `--min-peak` | 1 px, or ten times the jitter if that is larger | The smallest peak the damping is read from. A pixel is a safe target from test videos, where peaks under about a pixel came out up to 17 % low and bent the reading, not a physical limit; below the floor the script reads the frequency alone. Given, it replaces the default. |
| `--plot` | none | Saves the swing, the peaks used and the floor. |
| `--outline` | off | The part's point is on its outline against a plain background (a shaft, or another part whose surface turns and carries its texture away): the edge through it is followed across its 41 px width, not texture. Pick a stretch of edge with one clean step and nothing else within 12 px, such as a stripe of light, a key or a bolt head. |

The jitter is how much a point seems to move from one frame to the next, the tracker's noise: the still point's, or
the settled swing's when a shaking camera makes the still point's larger. A video's compression can leave a still point
almost untouched while it blurs a moving one, so the signal-to-noise ratio below takes its noise from the moving point.
The swing's amplitude at each peak is half
the drop to the troughs either side, so a rest position that is a little off does not bias the damping. The
frequency and the damping are read from the knock (the largest swing) to the end of the first unbroken run of swings
above the floor. The script warns when the part still swings in the last tenth of
the video (it takes that as rest, and the damping ratio would read low), when there are fewer than four frames per
swing, when the part moves more than 40 px from one frame to the next (half the reach of the tracker's 21 px window
over three pyramid levels), and when the first and second halves of the peaks give damping ratios more than a fifth
apart and further than three times their combined standard error (two resonances, or a part that is not one linear
resonance; neighbouring peaks share troughs, so the standard errors alone come out too small).

It also prints the signal-to-noise ratio: the first swing over the swing's noise, what is left of the stretch read
once the ring read is taken off (or, for the frequency alone, the noise of the quiet end), and from it the best
precision noise alone allows, the Cramér-Rao bound for a decaying sine in white noise. With d = (first swing ÷ noise)
× √(frame rate ÷ (8π × damping ratio × natural frequency)), that bound is about 2 × damping ratio ÷ d in the frequency
and 2 ÷ d in the damping ratio, as fractions (checked numerically against the bound to within 1-4 %). It is a bound,
not this script's precision: the script reads the peaks and a windowed spectrum, which in three simulated cases
scattered about 2 times more than the bound in frequency and 2.5-4 times more in damping, and real readings add the
clock, the tracker's bias below a pixel, and a part whose frequency shifts as it rings down.

What the scripts gave, on an Apple M3 laptop, 21-22 November 2026, in two installs: Python 3.13 with OpenCV 5.0,
NumPy 1.26 and SciPy 1.17 from conda, and Python 3.15 with OpenCV 5.1, NumPy 2.5 and SciPy 1.18 from
requirements.txt:

- **Test videos** made by `make_test_clip.py`, 1 to 200 Hz: the frequency within 0.05 % and the damping ratio within
  2 % of the values set where the tracker followed every frame (a frame it loses is bridged with a straight line,
  which can bend the peaks: the script warns about any lost frame among those it reads). The peak arithmetic itself
  holds to four frames per swing: on swings computed without a video, within 1.3 % at damping ratios of 0.01 and 0.02
  in any sampling phase, and up to about 5 % at 0.05 near four frames (1.9 % at 4.25). The frequency held within 0.014 % down to 2.4 frames per swing (100 Hz at
  240 frames a second). On two videos at 30 frames a second the damping ratio read 4 % and 12.5 % low; in the second,
  at 3.75 frames per swing, a 40 px swing moved 67 px a frame and the tracker lost frames. The speed test read 50 and 80 Hz at 240 frames a second within 0.6 %, the swing in full.
- **The printer-bed videos**: the camera's resonances at 76, 126, 215 and 264 Hz within about 0.3 % of the
  accelerometer's, and their damping ratios within 7 % when both are read by the same fit. At about 110 Hz the camera
  reads 109 Hz and the accelerometer 112 Hz: there the accelerometer, where the shaker pushes, shows only a faint
  peak on the rising flank of the 126 Hz resonance, and its fit is doubtful. A `*` beside a fit's
  damping ratio marks a fit stopped at a limit of its range, to be doubted. The plate moved 0.01 to
  0.06 pixel in the shaken band. Tracking one video took about a minute; the brightness reading about a second.

## What it does not do

- **Running machines at normal levels.** A part on a running machine moves micrometers: at 50 Hz, 1 mm/s is a swing
  of about 3 micrometers, a few hundredths of a pixel, with no force recorded to divide by. That stays with an
  accelerometer. The speed test reads a running machine only where the part shakes a pixel or more at some speed,
  which is near a resonance.
- **Motion toward the camera.** One camera sees only motion across the picture. Millimeters, and motion along the
  line of sight, need a second camera.
- **Two resonances close together.** The decrement assumes one resonance. Where two share a band, it follows the one
  that dies away slowest: in one of the printer-bed videos the ring-down read almost no damping at 126 and 215 Hz,
  where the fit read 0.013 and 0.021. The script warns when the first and second halves of the peaks give damping
  ratios further apart than their scatter allows.
- **Every surface.** The tracker needs texture, such as corners, bolts or a sticker; a turning part is followed by
  its outline instead (`--outline`). Plain or shiny surfaces, glare, motion blur and flickering light are not covered
  by the tests. The tap test has read six real videos, of a frame
  struck with a hammer in the CAMELOT benchmark (Politecnico di Torino, https://doi.org/10.5281/zenodo.15119038,
  CC BY 4.0), with accelerometers on the frame. It moved under a pixel, so no damping ratio. Read up to 32 s, before
  someone walks past the still point, four videos gave the frequency alone within 0.4-1.4 % of the accelerometer over
  the same seconds, their ring 37-70 times the noise; the other two, at 3.8 and 2.0 times, were declined. Four more
  videos of the same frame, unopened until the rule was frozen and the predictions written, all gave the frequency
  alone within 1.2 % of the accelerometer over the same seconds, compared as displacement. The speed test read
  six run-ups of a real rotor (Kang et al. 2024, Mendeley Data, https://doi.org/10.17632/47y7rwc2yg.1, CC BY 4.0)
  filmed from about 2 m, whose critical speed the shaft's eddy-current probe puts at 35.4-35.8 Hz. Points chosen on
  texture swung 0.1-0.2 px and were declined: their squares held the disk's face and the coupling's key, which turn
  with the rotor, and a still probe's stand. Measured on its edges, the rotor swings 1-2 px at the critical speed.
  With `--outline` on the shaft's top edge at two places, 11 of 12 readings fell within 2.1 % of the probe, most
  with the warning that the swing, barely over the floor, is not seen falling on either side; the twelfth read 10 %
  low, as a second edge 8 px away took the point's place once a revolution. On the disk's top, 2 of 6 read right:
  a stripe of reflected light 7-9 px under its edge gave a second edge of the same kind. These readings are not blind
  (the probe's answers were known); the outline's settings were fixed on generated videos first. Unlike the tap
  test, the speed test has no reading below a pixel.
- **The brightness reading** is a quick first look: on the printer bed it read 221 Hz for the resonance at 215 Hz,
  and at 264 Hz it found the frequency but read the damping 30 to 40 % low.

## Using it on your own recording

`check_printer_bed.py` fits the printer-bed videos only. For your own shaker test, change what fits that dataset: `SWEEP`
(the frequencies read), `BANDS` (one around each resonance), `GRIDS` (points whose squares lie on your part), and
`read_mraw()` and `read_signals()` for your files. The force and the video must be recorded over the same time.

## If something goes wrong

- **"cannot read" a video:** your OpenCV may lack the codec. The `opencv-python` package from pip reads MP4 and MOV.
- **`--pick` on a server:** there is no window; run without it and read the points from the saved first frame. On
  a server without a screen, `opencv-python-headless` installs without the window libraries.
- **A file name with accents or other letters outside English, on Windows:** OpenCV may not open the video; rename
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
data; `--download` fetches it from Zenodo, and stops rather than replace a folder of the same name it did not
unpack itself.
