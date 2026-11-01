"""
generate_ev_sound.py
--------------------
Builds a pure electric-vehicle drive sound (electric motor + inverter only,
NO tyre/road/rumble/air noise) that is perfectly LEVEL (no ups and downs),
so the AVAS algorithm alone controls volume and pitch according to speed.

How the level is kept flat:
  * Every layer is synthesised in the frequency domain on exact FFT bins,
    so the whole file is perfectly periodic -> seamless looping, no click.
  * All tonal components are integer multiples of f0/2 and spaced >= 130 Hz
    apart, so they can never "beat" against each other (beating was the cause
    of the 11-14 Hz wobble in emulated_sound_flat_equalized.wav).
  * Tonal phases use the Schroeder formula -> low crest factor, smooth peaks.
  * A final circular (loop-safe) RMS leveller guarantees a constant level.

Layers (gain in dB relative to the motor layer):
  motor    : f0 + harmonics, timbre extracted from the original recording
  sub      : f0/2 body / weight
  whine    : electric-motor magnetic whine (mid-high orders)
  inverter : thin, high PWM inverter whine

Usage:
  python generate_ev_sound.py
  python generate_ev_sound.py --whine -10 --inverter -18 --out my_ev.wav
  python generate_ev_sound.py --header      (also export a C wavetable header)
"""

import argparse
import os
import wave

import numpy as np

SR = 16000
# Timbre source (first file that exists is used).
SOURCE_WAVS = [
    r"d:\AVAS\Sound_Anlyze\emulated_sound_perfectly_flat.wav",
    r"d:\AVAS\Sound_Anlyze\emulated_sound_flat_equalized.wav",
]
DEFAULT_OUT = r"d:\AVAS\Sound_Anlyze\emulated_sound_ev_pure.wav"
HEADER_OUT = r"d:\AVAS\emulated_sound_ev_pure.h"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def load_wav(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return data.astype(np.float64), sr


def save_wav(path, signal, sr):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.clip(np.round(signal), -32768, 32767).astype(np.int16).tobytes())


def rms(x):
    return np.sqrt(np.mean(x ** 2))


def db(g):
    return 10.0 ** (g / 20.0)


def extract_harmonic_profile(audio, sr, f0, max_harm):
    """Relative amplitude of each harmonic of f0 in the source recording."""
    spec = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / sr)
    search = int(10 * len(audio) / sr)  # +-10 Hz search window
    amps = []
    for h in range(1, max_harm + 1):
        idx = int(np.argmin(np.abs(freqs - h * f0)))
        lo, hi = max(0, idx - search), min(len(spec), idx + search)
        amps.append(np.max(spec[lo:hi]))
    amps = np.array(amps)
    return amps / amps[0]


class PeriodicSynth:
    """Synthesises exactly periodic signals of length N on FFT bins."""

    def __init__(self, n, sr):
        self.n = n
        self.sr = sr
        self.df = sr / n
        self.nbins = n // 2 + 1

    def snap(self, f):
        return round(f / self.df) * self.df

    def tones(self, freqs, amps):
        """Sum of sines on exact bins with Schroeder phases (low crest factor)."""
        spec = np.zeros(self.nbins, dtype=complex)
        count = len(freqs)
        for i, (f, a) in enumerate(zip(freqs, amps)):
            k = int(round(f / self.df))
            if 0 < k < self.nbins - 1:
                phase = -np.pi * i * (i + 1) / max(count, 1)
                spec[k] += a * (self.n / 2) * np.exp(1j * phase)
        return np.fft.irfft(spec, n=self.n)


def circular_level(signal, sr, window_s=0.03):
    """Loop-safe RMS leveller: divides the signal by its moving RMS envelope."""
    n = len(signal)
    win = max(1, int(sr * window_s))
    kernel = np.zeros(n)
    kernel[:win] = np.hanning(win)
    kernel /= kernel.sum()
    kernel = np.roll(kernel, -win // 2)  # centre the window (zero phase)
    power = np.fft.irfft(np.fft.rfft(signal ** 2) * np.fft.rfft(kernel), n=n)
    env = np.sqrt(np.maximum(power, 1e-9))
    return signal / env * rms(signal)


def soft_rolloff(freqs, cutoff, slope_db_oct):
    """Smooth high-frequency roll-off (warmth). Flat below cutoff, then
    falls by slope_db_oct per octave. Tames the harsh 2-4 kHz region."""
    freqs = np.asarray(freqs, dtype=float)
    octaves_above = np.log2(np.maximum(freqs, cutoff) / cutoff)
    return 10.0 ** (-slope_db_oct * octaves_above / 20.0)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def build(args):
    n = int(SR * args.duration)
    synth = PeriodicSynth(n, SR)
    f0 = synth.snap(args.f0)
    nyq = SR / 2 - 200

    src_path = next((p for p in SOURCE_WAVS if os.path.exists(p)), None)
    if src_path is None:
        raise FileNotFoundError("No timbre source wav found: " + ", ".join(SOURCE_WAVS))
    source, src_sr = load_wav(src_path)
    max_harm = int(nyq // f0)
    profile = extract_harmonic_profile(source, src_sr, args.f0, max_harm)

    # 1) Motor tone: harmonics of f0 with the recorded timbre, rolled off
    #    above the cutoff so the upper harmonics don't sound buzzy/harsh.
    harm_f = np.array([h * f0 for h in range(1, max_harm + 1)])
    motor = synth.tones(harm_f, profile * soft_rolloff(harm_f, args.cutoff, args.slope))

    # 2) Sub / body: f0/2 plus a little of 1.5*f0 (integer multiples of f0/2).
    sub = synth.tones([0.5 * f0, 1.5 * f0], [1.0, 0.35])

    # 3) Electric-motor whine: a soft, low-mid hum placed exactly between
    #    motor harmonics (>= f0/2 away -> no beating).
    whine_f = np.array([2.5 * f0, 4.5 * f0, 6.5 * f0])
    whine = synth.tones(whine_f, np.array([1.0, 0.6, 0.3]) * soft_rolloff(whine_f, args.cutoff, args.slope))

    # 4) Inverter (PWM) whine: thin, high-pitched tones (off by default -
    #    this is the most piercing layer).
    inv_f = np.array([16.5 * f0, 20.5 * f0, 24.5 * f0])
    inverter = synth.tones(inv_f, [1.0, 0.7, 0.4])

    layers = {
        "motor": (motor, 0.0),
        "sub": (sub, args.sub),
        "whine": (whine, args.whine),
        "inverter": (inverter, args.inverter),
    }

    mix = np.zeros(n)
    for name, (sig, gain_db) in layers.items():
        if gain_db <= -90:
            continue
        mix += sig / rms(sig) * db(gain_db)

    # Guarantee a constant level (loop-safe).
    mix = circular_level(mix, SR)

    # Normalise to target RMS, but never clip.
    mix *= args.rms / rms(mix)
    peak = np.max(np.abs(mix))
    if peak > 32000:
        mix *= 32000 / peak
    return mix, f0


def report(signal, sr):
    win = int(sr * 0.05)
    blocks = [signal[i:i + win] for i in range(0, len(signal) - win + 1, win)]
    r = np.array([rms(b) for b in blocks])
    p = np.array([np.max(np.abs(b)) for b in blocks])
    swing_db = 20 * np.log10(r.max() / r.min())
    print(f"  50 ms RMS  : min {r.min():7.1f}  max {r.max():7.1f}  swing {swing_db:.2f} dB")
    print(f"  50 ms peak : min {p.min():7.1f}  max {p.max():7.1f}")
    print(f"  loop seam  : last->first sample jump {abs(signal[-1] - signal[0]):.1f}"
          f" (typical step {np.mean(np.abs(np.diff(signal))):.1f})")


def export_header(path, signal, f0):
    data = np.clip(np.round(signal), -32768, 32767).astype(np.int16)
    lines = [
        "#ifndef EMULATED_SOUND_EV_PURE_H",
        "#define EMULATED_SOUND_EV_PURE_H",
        "",
        "#include <stdint.h>",
        "#include <stddef.h>",
        "",
        "// Pure EV motor/inverter sound (no road noise), perfectly level, seamless loop.",
        f"// {len(data)} samples @ {SR} Hz, base pitch {f0:.2f} Hz",
        f"const size_t emulated_sound_ev_pure_length = {len(data)};",
        f"const float emulated_sound_ev_pure_base_freq = {f0:.2f}f;",
        "const int16_t emulated_sound_ev_pure_pcm[] = {",
    ]
    for i in range(0, len(data), 12):
        lines.append("    " + ", ".join(f"{v:6d}" for v in data[i:i + 12]) + ",")
    lines += ["};", "", "#endif // EMULATED_SOUND_EV_PURE_H", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Exported C header: {path}")


def main():
    ap = argparse.ArgumentParser(description="Generate a level, pure EV motor sound (no road noise).")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--f0", type=float, default=273.2, help="motor base pitch (Hz)")
    ap.add_argument("--duration", type=float, default=3.0, help="loop length (s)")
    ap.add_argument("--rms", type=float, default=4000.0, help="output RMS level")
    ap.add_argument("--sub", type=float, default=-5.0, help="gain dB (<= -90 disables)")
    ap.add_argument("--whine", type=float, default=-16.0)
    ap.add_argument("--inverter", type=float, default=-90.0, help="off by default (harsh)")
    ap.add_argument("--cutoff", type=float, default=700.0,
                    help="roll-off start (Hz); lower = warmer/softer")
    ap.add_argument("--slope", type=float, default=12.0,
                    help="roll-off steepness (dB/octave); higher = softer")
    ap.add_argument("--header", action="store_true", help="also export a C header")
    args = ap.parse_args()

    mix, f0 = build(args)
    save_wav(args.out, mix, SR)
    print(f"Created {args.out}  ({args.duration:.1f} s, {SR} Hz, f0 = {f0:.2f} Hz)")
    report(mix, SR)

    if args.header:
        export_header(HEADER_OUT, mix, f0)


if __name__ == "__main__":
    main()
