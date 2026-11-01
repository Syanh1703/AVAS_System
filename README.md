# Acoustic Vehicle Alerting System (AVAS) - ECE R138 Simulator

This repository contains a signal processing algorithm designed to synthesize an Electric Vehicle (EV) engine sound according to **UN Regulation No. 138 (ECE R138)**. 

## 📁 Repository Contents

### 1. Interactive Real-Time UI (`avas_simulator.html`)
A fully interactive web dashboard that runs the AVAS algorithm in real-time using your browser's Web Audio API. 
* **How to use:** Simply double-click `avas_simulator.html` to open it in Chrome, Edge, or Firefox.
* **Features:** 
  * Real-time engine acceleration slider (0 to 20 km/h).
  * **Dual Sound Decks (Forward & Reverse):** Dedicated slots to load both `Sound_Anlyze/avas_base_asset.wav` and `Sound_Anlyze/avas_reverse_asset.wav` simultaneously for instant A/B testing and seamless direction toggling.
  * **ECE R138 Compliant Logic:** Mutes forward sound at 0 km/h, dynamically shifts pitch (+1.0%/km/h above 5 km/h), and keeps reverse sound audibly active at 47 dB(A) even at standstill (Annex 3 §3.3.3.3).
  * **Quick Audition Buttons:** Instant one-click auditioning of Forward vs. Reverse sounds.
  * Optional Tire Noise toggle and dedicated Dual-Tone Horn button.

### 2. Core C DSP Engine (`avas_algorithm.c` & `avas_algorithm.h`)
The mathematical engine that generates the EV sounds natively using wavetable additive synthesis.
* Mixes 4 harmonic oscillators (0.5x, 1.0x, 1.5x, 2.0x) to ensure at least two 1/3 octave bands are filled.
* Automatically scales the output pitch by 1.0% per km/h (satisfying the > 0.8% requirement).
* Implements the exact logarithmic volume curves required for 10 km/h (50 dB), 20 km/h (56 dB), and Reversing (47 dB).
* Embeds a fast pseudo-random number generator for tire noise and dual-tone square wave generation for the car horn.

### 3. C Execution Wrapper (`main_custom.c`)
A wrapper script that initializes the AVAS algorithm and generates a mathematically stable drone test file. 
* **Sample Rate:** 33,100 Hz.
* **Output:** Generates `avas_output_c_custom.wav` (A stable 5 km/h base pitch, statically normalized down to ~ -24.9 LUFS).

### 4. Static Base Asset Generators (`generate_wav.py`)
Generates mathematically stable, constant-level, 16-bit 44.1 kHz base assets conforming to UN ECE R138:
* **Forward Base Asset:** `Sound_Anlyze/avas_base_asset.wav` (160 Hz, 250 Hz, 315 Hz, 400 Hz, 800 Hz, 1250 Hz with 160-250 Hz low propulsion focus).
* **Reverse Warning Asset:** `Sound_Anlyze/avas_reverse_asset.wav` (Hyundai IONIQ style intermittent pulsed chime: 1.0s cadence [0.45s warm harmonic bell + 0.55s silent pause], 200–800 Hz, with no harsh high frequencies or road noise).
* **Run:** `python generate_wav.py` to regenerate both assets.

### 5. Pure EV Sound Generator (`generate_ev_sound.py`)
Builds a constant-level, multi-harmonic electric-vehicle sound that loops seamlessly.
* **Run:** `python generate_ev_sound.py` → `Sound_Anlyze/emulated_sound_ev_pure.wav`
* **C wavetable:** add `--header` to export `emulated_sound_ev_pure.h`.

---

## 🚀 How to Compile and Run the C Engine

If you make modifications to the `avas_algorithm.c` or `main_custom.c` files, you can recompile the system locally using GCC (via MSYS2/MinGW):

```bash
# 1. Compile the source code
gcc main_custom.c avas_algorithm.c -o avas_test_custom.exe

# 2. Run the executable to generate the audio file
.\avas_test_custom.exe
```

*This will output `avas_output_c_custom.wav`. You can then upload this wav file directly into the `avas_simulator.html` UI to test it interactively!*

---

## 📜 Regulatory Compliance Summary
- **6.2.1.2(b) Frequency Bands:** Output utilizes 250Hz, 500Hz, 750Hz, and 1000Hz bands.
- **6.2.3.2 Frequency Shift:** Shifts pitch dynamically at 1.0% / km/h.
- **6.2.4 Stationary Sound:** Engine output automatically mutes at 0 km/h to prevent noise pollution.
- **6.2.8 Minimum SPL:** Implements exact relative dB amplitude scaling corresponding to target test speeds.
