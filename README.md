# Acoustic Vehicle Alerting System (AVAS) - ECE R138 Simulator

This repository contains the official Python DSP testing environment, WAV asset generators, and interactive web simulator for synthesizing and analyzing Electric Vehicle (EV) sounds according to **UN Regulation No. 138 (ECE R138)**.

## 📁 Workspace Structure

### 1. Official Python Desktop Analyzer (`AVAS_Thanh/AVAS_S.py`)
Real-time PyQt5 + `pyqtgraph` + `sounddevice` AVAS testing and analysis environment:
* **Speed Control (0–20 km/h):** Mutes smoothly at `0 km/h` and shifts pitch dynamically from `160 Hz` (`1 km/h`) up to `256 Hz` (`20 km/h`, `+60%` pitch shift).
* **Live Spectrum & 1/3-Octave RTA:** Displays real-time A-weighted FFT spectrum and the 16 UN ECE R138 Table 3 1/3-octave bands (`160 Hz` – `5000 Hz`).
* **3-Mode Live Sound Selector:**
  1. `AVAS_Thanh/avas_sim_asset.wav` (Kia EV6 VESS *"Spaceship V12"* style synthesized AVAS sound).
  2. `AVAS_Thanh/avas_base_asset.wav` (Original base WAV with real-time in-memory DSP spectral denoising & harmonic phase-locking).
  3. `AVAS_Thanh/avas_base_asset.wav` (Raw original WAV).
* **Run:**
  ```bash
  python AVAS_Thanh/AVAS_S.py
  ```

### 2. Audio Assets & Generator (`generate_wav.py`)
* `AVAS_Thanh/avas_base_asset.wav` — Original `44.1 kHz`, `16-bit PCM`, `6.0 s` reference base asset (`160 Hz` fundamental).
* `AVAS_Thanh/avas_sim_asset.wav` — Smooth Kia EV6 VESS style harmonic AVAS sound synthesized from the reference profile.
* **Run:**
  ```bash
  python generate_wav.py
  ```

### 3. Interactive Web Simulator (`avas_simulator.html`)
Browser-based Web Audio API simulator with dual Forward & Reverse sound decks for quick auditioning.

### 4. Regulatory Standard (`R138e.pdf`)
Official **UN ECE Regulation No. 138** specification document.
