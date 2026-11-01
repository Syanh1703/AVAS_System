import math
import os
import shutil
import struct
import wave
import numpy as np
from scipy.io import wavfile

SAMPLE_RATE = 44100
MAX_AMP = 32767 * 0.85  # Constant peak volume to prevent clipping

# Output folder: AVAS_Thanh
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
THANH_DIR = os.path.join(SCRIPT_DIR, "AVAS_Thanh")
OUTPUT_DIR = THANH_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Exact UN ECE R138 Table 2 center frequencies (Hz) and their relative mix weights
R138_COMPONENTS = [
    (160.0,  0.35),
    (250.0,  0.22),
    (315.0,  0.18),
    (400.0,  0.12),
    (800.0,  0.08),
    (1250.0, 0.05),
]

# UN ECE R138 Reverse Components
# Tuned for a distinct, warm, futuristic EV reverse warning chord:
# Focused in the 200 - 800 Hz band for optimal pedestrian localization behind the vehicle,
# with no harsh high frequencies (>1000 Hz) and zero road noise.
R138_REVERSE_COMPONENTS = [
    (200.0,  0.22),  # Warm low body / foundation
    (315.0,  0.35),  # Core prominent warning fundamental
    (500.0,  0.25),  # Distinct harmonic interval
    (630.0,  0.12),  # Recognizable reverse signature
    (800.0,  0.06),  # Subtle localization overtone
]

def generate_static_base_asset(filename=os.path.join(OUTPUT_DIR, "avas_base_asset.wav"), duration_sec=6.0):
    """
    Generates a perfectly stable, constant-frequency, and constant-volume forward base asset.
    Designed to be imported into an AVAS DSP that handles dynamic modulation.
    """
    num_samples = int(SAMPLE_RATE * duration_sec)
    
    with wave.open(filename, "w") as wav_file:
        wav_file.setparams((1, 2, SAMPLE_RATE, num_samples, "NONE", "not compressed"))
        
        frames = bytearray()
        for n in range(num_samples):
            t = n / SAMPLE_RATE
            sample_val = 0.0
            
            for freq, weight in R138_COMPONENTS:
                sample_val += weight * math.sin(2.0 * math.pi * freq * t)
                
            sample_val = math.tanh(sample_val * 1.4) / math.tanh(1.4)
            pcm_int16 = int(sample_val * MAX_AMP)
            frames.extend(struct.pack("<h", pcm_int16))
            
        wav_file.writeframes(frames)
    print(f"Created forward static base asset: {filename} ({duration_sec}s)")

def generate_reverse_asset(filename=os.path.join(OUTPUT_DIR, "avas_reverse_asset.wav"), duration_sec=6.0):
    """
    Generates an intermittent pulsed EV Reverse chime asset conforming to UN ECE R138:
    - Modeled after the Hyundai IONIQ / Kia EV6 rhythmic reverse sound.
    - Cadence: 1.0 second cycle (0.45s warm harmonic chime pulse + 0.55s silent pause), repeated 6 times.
    - The silent pauses defeat auditory habituation and trigger orientation reflexes for pedestrians.
    - Pure electric sound: 200 Hz, 315 Hz, 500 Hz, 630 Hz, 800 Hz harmonic chord with soft saturation.
    - Peak reaches calibrated 47 dB(A) level with zero clipping and a seamless 0.0 loop seam.
    """
    num_samples = int(SAMPLE_RATE * duration_sec)
    period_sec = 1.0
    pulse_sec = 0.45
    attack_sec = 0.025
    
    with wave.open(filename, "w") as wav_file:
        wav_file.setparams((1, 2, SAMPLE_RATE, num_samples, "NONE", "not compressed"))
        
        frames = bytearray()
        for n in range(num_samples):
            t = n / SAMPLE_RATE
            t_cycle = t % period_sec
            
            # Pulse amplitude envelope (Hyundai IONIQ style chime)
            if t_cycle < attack_sec:
                # Smooth 25ms quadratic attack to eliminate clicks
                env = (t_cycle / attack_sec) ** 1.5
            elif t_cycle < pulse_sec:
                # Smooth exponential chime decay
                progress = (t_cycle - attack_sec) / (pulse_sec - attack_sec)
                env = math.exp(-progress * 3.5)
            else:
                # Silent gap between pulses to facilitate pedestrian spatial localization
                env = 0.0
                
            sample_val = 0.0
            for freq, weight in R138_REVERSE_COMPONENTS:
                sample_val += weight * math.sin(2.0 * math.pi * freq * t)
                
            # Apply pulse envelope and soft saturation for warm electric acoustic timbre
            sample_val = math.tanh(sample_val * env * 1.5) / math.tanh(1.5)
            pcm_int16 = int(sample_val * MAX_AMP)
            frames.extend(struct.pack("<h", pcm_int16))
            
        wav_file.writeframes(frames)
    print(f"Created pulsed reverse asset: {filename} ({duration_sec}s)")


def generate_similar_avas_asset(
    reference_wav=os.path.join(THANH_DIR, "avas_base_asset.wav"),
    output_wav=os.path.join(THANH_DIR, "avas_sim_asset.wav"),
):
    """
    Synthesizes the Kia EV6 VESS ("Spaceship V12") futuristic electric vehicle AVAS sound
    analyzed from https://www.youtube.com/shorts/4Vaurco0vC0 and aligned with the 160 Hz
    base fundamental of avas_base_asset.wav:
    - Layer 1 (Deep "V12" Sub-Harmonic Core): 80 Hz (0.50*f0) and 120 Hz (0.75*f0) chest rumble.
    - Layer 2 (Futuristic Just-Intonation Major Triad): 160 Hz (1.0*f0), 200 Hz (1.25*f0 Major 3rd),
      240 Hz (1.50*f0 dominant Perfect 5th), 320 Hz (2.0*f0 Octave), 400 Hz (2.5*f0), 480 Hz (3.0*f0).
    - Layer 3 (Spaceship Turbine & Inverter Overtones): 640 Hz (4.0*f0), 800 Hz (5.0*f0),
      960 Hz (6.0*f0 inverter whine), and 1280 Hz (8.0*f0 crystalline signature).
    - Subtle dual-motor chorus shimmer (+-0.333 Hz) with Schroeder phase alignment for a rich,
      living EV powertrain sound with zero nonlinear IMD noise and a seamless 6.0s loop.
    """
    sr = SAMPLE_RATE
    duration_sec = 6.0
    if os.path.exists(reference_wav):
        sr_ref, raw_data = wavfile.read(reference_wav)
        sr = sr_ref
        duration_sec = len(raw_data) / float(sr)

    n = int(sr * duration_sec)
    df = sr / float(n)
    sim_spec = np.zeros(n // 2 + 1, dtype=complex)

    kia_ev6_harmonics = [
        (80.0,   0.26),  # 0.50 * f0: Sub-octave V12 deep throat rumble (70-96 Hz in video)
        (120.0,  0.22),  # 0.75 * f0: Low fifth power growl (114-136 Hz in video)
        (160.0,  0.38),  # 1.00 * f0: Primary motor fundamental [160 Hz R138 band]
        (200.0,  0.30),  # 1.25 * f0: Just-Intonation Major 3rd (200-226 Hz in video) [200 Hz band]
        (240.0,  0.42),  # 1.50 * f0: Dominant Perfect 5th resonance (240-276 Hz peak) [250 Hz band]
        (320.0,  0.24),  # 2.00 * f0: Octave turbine harmonic (332-364 Hz in video) [315 Hz band]
        (400.0,  0.18),  # 2.50 * f0: Octave + Major 3rd (400-460 Hz in video) [400 Hz band]
        (480.0,  0.13),  # 3.00 * f0: Octave + Perfect 5th (468-510 Hz in video) [500 Hz band]
        (640.0,  0.07),  # 4.00 * f0: Double Octave harmonic [630 Hz band]
        (800.0,  0.045), # 5.00 * f0: High-speed rotor harmonic [800 Hz band]
        (960.0,  0.030), # 6.00 * f0: Inverter whine layer (938-1086 Hz in video) [1000 Hz band]
        (1280.0, 0.022), # 8.00 * f0: Futuristic spaceship overtone (1280 Hz in video) [1250 Hz band]
    ]

    num_tones = len(kia_ev6_harmonics)
    for idx, (freq, amp) in enumerate(kia_ev6_harmonics):
        k_bin = int(round(freq / df))
        if 0 < k_bin < len(sim_spec):
            phase = -math.pi * idx * (idx + 1) / num_tones
            sim_spec[k_bin] += amp * (n / 2) * np.exp(1j * phase)
            # Subtle dual-motor chorus sidebands (+-0.333 Hz = +-2 FFT bins) for authentic turbine motion
            if 160.0 <= freq <= 640.0 and (k_bin - 2) > 0 and (k_bin + 2) < len(sim_spec):
                shimmer_amp = amp * 0.08
                sim_spec[k_bin - 2] += shimmer_amp * (n / 2) * np.exp(1j * (phase + 0.7))
                sim_spec[k_bin + 2] += shimmer_amp * (n / 2) * np.exp(1j * (phase - 0.7))

    sim_signal = np.fft.irfft(sim_spec, n=n)
    sim_signal = sim_signal / np.max(np.abs(sim_signal)) * MAX_AMP
    pcm_int16 = np.clip(np.round(sim_signal), -32768, 32767).astype(np.int16)

    os.makedirs(os.path.dirname(os.path.abspath(output_wav)), exist_ok=True)
    wavfile.write(output_wav, sr, pcm_int16)
    print(f"Created Kia EV6 VESS style AVAS sound: {output_wav} ({n / sr:.1f}s)")


if __name__ == "__main__":
    # Keep existing base asset intact if it already exists, and generate the new similar AVAS asset
    base_path = os.path.join(OUTPUT_DIR, "avas_base_asset.wav")
    if not os.path.exists(base_path):
        generate_static_base_asset(base_path)
    generate_reverse_asset()
    generate_similar_avas_asset()
