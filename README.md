# Acoustic Vehicle Alerting System (AVAS) - UN ECE R138 Simulator & Analyzer

This repository contains the official Python DSP testing environment, vehicle acoustic profile libraries, WAV asset generators, and interactive web simulator for synthesizing and analyzing Electric Vehicle (EV) sounds according to **UN Regulation No. 138 (ECE R138)**.

---

## 📁 Workspace Structure

```text
AVAS/
├── AVAS_Thanh/
│   ├── AVAS_S.py                         # Official PyQt5 Real-Time AVAS DSP & ECE R138 Compliance Analyzer
│   ├── Profile_Audi/
│   │   ├── audi_etron_forward.wav        # Flattened Audi e-tron Forward Hover Hum (44.1 kHz, 16-bit, 6.0s)
│   │   └── audi_etron_backward.wav       # Audi e-tron Reverse Safety Spaceship Hum & Turbine (44.1 kHz, 16-bit, 6.0s)
│   └── Profile_Skoda/
│       ├── avas_base_asset.wav           # Reference UN ECE R138 Base Asset (160 Hz fundamental)
│       └── avas_sim_asset.wav            # Synthesized Futuristic Multi-Harmonic AVAS Asset
├── generate_wav.py                       # Standalone WAV Asset Synthesis Utility
├── avas_simulator.html                   # Browser-Based Web Audio API Dual-Deck Simulator
├── R138e.pdf                             # Official UN ECE Regulation No. 138 Specification
└── README.md                             # Project Documentation
```

---

## 🚀 Key Components

### 1. Official Python Desktop Analyzer (`AVAS_Thanh/AVAS_S.py`)
A real-time `PyQt5` + `pyqtgraph` + `sounddevice` + `scipy` AVAS synthesis, DSP conditioning, and **UN ECE R138** compliance verification suite:

* **Interactive `.WAV` Import & Profile Selector:**
  * Prompts the user to select a `.wav` file on startup or switch/import any `.wav` file on the fly via the **"📂 Import File .WAV..."** button and auto-discovered profile dropdown (`Profile_Audi`, `Profile_Skoda`, or external directories).
* **Continuous Idle Background Hum + Speed-Tuned Acceleration Architecture:**
  * **Background Hum Layer (`< 190 Hz`, 8th-order crossover):** Plays continuously at `1.0x` pitch—even when the vehicle is stationary at **Idle (`0 km/h`)**—keeping the deep EV hover hum (`70–175 Hz`) permanently anchored in the background with zero frequency overlap against the upper acceleration layer.
  * **Speed-Tuned Spaceship/Acceleration Layers (`>= 195 Hz`, `1–20 km/h`):** Splits the upper propulsion harmonics into a **Mid-Body Layer (`195–560 Hz`)** and a linear **High-Turbine Layer (`480–2400 Hz`)** interpolated via 4-point **Catmull-Rom cubic interpolation**. As vehicle speed increases from `1 km/h` to `20 km/h`, the software smoothly tunes pitch (`160 Hz -> 256 Hz`, `+60%`) and opens up high-frequency brightness, including a transient **Torque Load Boost ($dv/dt > 0$)** during acceleration.
* **Real-Time DSP Conditioning Pipeline:**
  * **Automatic IMD De-Spiker & Schroeder Phase Disperser:** Automatically strips nonlinear `tanh` Intermodulation Distortion (IMD) spikes on file load, locks beating inharmonic partials (`250 Hz -> 240 Hz`, `315 Hz -> 320 Hz`) onto the `80 Hz` harmonic grid, and applies Schroeder phase dispersion ($\phi_k = -\pi k(k+1)/N$) for a flat waveform envelope.
  * **Equal-Power Crossfade Loop Guard & DC-Offset Removal:** Automatically applies a `60 ms` $\cos^2/\sin^2$ seam guard when needed so any user-imported `.wav` loops without clicks.
  * **Fixed Body Cavity Formant Preservation (`215 Hz` & `485 Hz`):** Dual persistent biquad resonators preserve the physical vehicle body resonance while motor harmonics shift upward in pitch.
  * **Soft-Knee Hyperbolic Tangent Limiter:** Protects output peaks (`threshold = 0.72`, `ceiling = 0.95`) without hard-clipping distortion.
* **Full UN ECE R138 Compliance Evaluator:**
  * **8192-Point FFT & IEC 61672 Fast (`125 ms`) Exponential Time-Weighting** with Hann window RMS energy compensation.
  * **Speed-Adaptive Table 3 Limit Curve:** Dynamically interpolates the 16 1/3-octave band minimums (`160 Hz` – `5000 Hz`) between the `10 km/h` and `20 km/h` (`+5 dB(A)`) Table 3 columns.
  * **Live Compliance Verdict Badges & Color-Coded RTA Bars:**
    * **Overall SPL Check (Para. 6.2.1 & 6.2.8):** Verifies `50–75 dB(A)` at `10 km/h` forward, `56–75 dB(A)` at `20 km/h` forward, and `47–75 dB(A)` in reverse.
    * **Two-Band Rule (Para. 6.2.1.3):** Highlights passing 1/3-octave RTA bars in **Green** and verifies that at least **2 bands** meet Table 3 minimums with at least **1 band $\le 1600\text{ Hz}$**.
    * **Frequency Shift Check (Para. 6.2.3):** Verifies average pitch shift rate (`3.16% / km/h >= 0.8% / km/h` minimum requirement).

#### Run the Desktop Analyzer:
```bash
python AVAS_Thanh/AVAS_S.py
```

---

### 2. Vehicle Acoustic Profiles (`AVAS_Thanh/Profile_*`)
* **`AVAS_Thanh/Profile_Audi/`** *(Analyzed from Audi e-tron exterior/interior drive & reverse tests)*:
  * `audi_etron_forward.wav` — Flattened, zero-ripple (`0.000000 dB` envelope variation) Audi e-tron forward hover hum designed as a clean foundation for real-time software pitch & acceleration tuning.
  * `audi_etron_backward.wav` — Audi e-tron reverse safety profile featuring the prominent `157.5 Hz` and `313.6 Hz` (`+15 dB`) safety boost, `484.5 Hz` cavity tone, and `958.2 Hz` high-tech whistle with $\pm 12.1\text{ Hz}$ chorus twins.
* **`AVAS_Thanh/Profile_Skoda/`**:
  * `avas_base_asset.wav` — Original `44.1 kHz`, `16-bit PCM`, `6.0 s` reference base asset (`160 Hz` fundamental).
  * `avas_sim_asset.wav` — Multi-harmonic futuristic EV propulsion asset (`80 Hz` – `1280 Hz` harmonic series).

---

### 3. Standalone WAV Generator (`generate_wav.py`)
Utility script for synthesizing static forward, pulsed reverse, and multi-harmonic reference `.wav` assets.
```bash
python generate_wav.py
```

---

### 4. Interactive Web Simulator (`avas_simulator.html`)
Browser-based Web Audio API simulator with dual Forward & Reverse sound decks for quick browser auditioning.

---

### 5. Regulatory Standard (`R138e.pdf`)
Official **UN ECE Regulation No. 138** specification document (*Uniform provisions concerning the approval of Quiet Road Transport Vehicles with regard to their reduced audibility*).
