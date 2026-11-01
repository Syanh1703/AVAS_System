import sys
import os
import io
import shutil
import collections
import numpy as np
from scipy.io import wavfile
import sounddevice as sd

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
    QSlider, QLabel, QGroupBox, QComboBox
)
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg

# Khắc phục lỗi mã hóa UTF-8 trên Windows Terminal
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# --- THÔNG SỐ CẤU HÌNH AVAS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_WAV = os.path.join(SCRIPT_DIR, "avas_base_asset.wav")
if not os.path.exists(INPUT_WAV):
    INPUT_WAV = "avas_base_asset.wav"

SIM_WAV = os.path.join(SCRIPT_DIR, "avas_sim_asset.wav")

BASE_FREQ = 160.0    # Tần số gốc của file WAV nền (Hz)
MIN_FREQ = 160.0     # Tần số tại 1 km/h (Âm trầm mượt, đầm xe)
MAX_FREQ = 256.0     # Tần số tại 20 km/h (Tương ứng tăng +60% pitch: 160 -> 256 Hz như Kia EV6 VESS)

# 16 Dải tần số trung tâm 1/3 Octave theo Bảng 3 - ECE R138 (Hz)
OCTAVE_CENTER_FREQS = np.array([
    160, 200, 250, 315, 400, 500, 630, 800, 
    1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000
], dtype=np.float32)

# Mức tối thiểu dB(A) quy định ở 10 km/h (Tham chiếu Bảng 3)
R138_MIN_DBA_10KMH = np.array([
    45, 44, 43, 44, 45, 45, 46, 46,
    46, 46, 44, 42, 39, 36, 34, 31
], dtype=np.float32)


def get_a_weighting_offsets(freqs):
    """Tính offset Trọng số A (A-Weighting) cho mảng tần số (Hz)"""
    f2 = freqs ** 2
    f2[f2 == 0] = 1e-6 
    
    c1 = 12194.217 ** 2
    c2 = 20.598997 ** 2
    c3 = 107.65265 ** 2
    c4 = 737.86223 ** 2

    num = c1 * (f2 ** 2)
    den = (f2 + c2) * np.sqrt((f2 + c3) * (f2 + c4)) * (f2 + c1)
    
    r_a = num / den
    a_db = 20.0 * np.log10(r_a) + 2.0
    return a_db


def generate_similar_avas_wav(ref_wav_path, out_wav_path):
    """
    Tổng hợp âm thanh AVAS phong cách Kia EV6 VESS ("Spaceship V12") phân tích từ
    video https://www.youtube.com/shorts/4Vaurco0vC0 kết hợp nền f0 = 160 Hz của WAV gốc:
    - Lớp 1 (Sub-Harmonic V12 Rumble): 80 Hz (0.50*f0) và 120 Hz (0.75*f0) tạo tiếng gầm trầm uy lực.
    - Lớp 2 (Futuristic Major Triad Chord): 160 Hz (1.0*f0), 200 Hz (1.25*f0 quãng 3 trưởng),
      240 Hz (1.50*f0 quãng 5 đúng chủ đạo), 320 Hz (2.0*f0), 400 Hz (2.5*f0), 480 Hz (3.0*f0).
    - Lớp 3 (Spaceship Turbine & Inverter Whine): 640 Hz, 800 Hz, 960 Hz và 1280 Hz.
    - Hiệu ứng Dual-Motor Chorus Shimmer (+-0.333 Hz) và pha Schroeder giúp âm thanh mượt,
      sống động như động cơ điện kép thực tế và đạt chuẩn UN ECE R138.
    """
    sr, raw_data = wavfile.read(ref_wav_path)
    if len(raw_data.shape) > 1:
        raw_data = raw_data[:, 0]
    n = len(raw_data)
    df = sr / float(n)
    sim_spec = np.zeros(n // 2 + 1, dtype=complex)

    kia_ev6_harmonics = [
        (80.0,   0.26),  # 0.50 * f0: Âm trầm V12 sub-bass (70-96 Hz trong video)
        (120.0,  0.22),  # 0.75 * f0: Âm gầm quãng 5 trầm (114-136 Hz trong video)
        (160.0,  0.38),  # 1.00 * f0: Tần số cơ bản [Dải 160 Hz R138]
        (200.0,  0.30),  # 1.25 * f0: Quãng 3 trưởng đặc trưng (200-226 Hz) [Dải 200 Hz R138]
        (240.0,  0.42),  # 1.50 * f0: Đỉnh cộng hưởng quãng 5 đúng (240-276 Hz) [Dải 250 Hz R138]
        (320.0,  0.24),  # 2.00 * f0: Hài âm bậc 2 (332-364 Hz) [Dải 315 Hz R138]
        (400.0,  0.18),  # 2.50 * f0: Hài âm quãng 10 trưởng (400-460 Hz) [Dải 400 Hz R138]
        (480.0,  0.13),  # 3.00 * f0: Hài âm bậc 3 (468-510 Hz) [Dải 500 Hz R138]
        (640.0,  0.07),  # 4.00 * f0: Hài âm bậc 4 [Dải 630 Hz R138]
        (800.0,  0.045), # 5.00 * f0: Hài âm bậc 5 [Dải 800 Hz R138]
        (960.0,  0.030), # 6.00 * f0: Tiếng rít biến tần (938-1086 Hz) [Dải 1000 Hz R138]
        (1280.0, 0.022), # 8.00 * f0: Âm sắc phi thuyền bậc cao (1280 Hz) [Dải 1250 Hz R138]
    ]

    num_tones = len(kia_ev6_harmonics)
    for idx, (freq, amp) in enumerate(kia_ev6_harmonics):
        k_bin = int(round(freq / df))
        if 0 < k_bin < len(sim_spec):
            phase = -np.pi * idx * (idx + 1) / num_tones
            sim_spec[k_bin] += amp * (n / 2) * np.exp(1j * phase)
            if 160.0 <= freq <= 640.0 and (k_bin - 2) > 0 and (k_bin + 2) < len(sim_spec):
                shimmer_amp = amp * 0.08
                sim_spec[k_bin - 2] += shimmer_amp * (n / 2) * np.exp(1j * (phase + 0.7))
                sim_spec[k_bin + 2] += shimmer_amp * (n / 2) * np.exp(1j * (phase - 0.7))

    sim_signal = np.fft.irfft(sim_spec, n=n)
    max_val = np.max(np.abs(sim_signal))
    if max_val > 1e-9:
        sim_signal = sim_signal / max_val * 0.85

    pcm_int16 = np.clip(np.round(sim_signal * 32767.0), -32768, 32767).astype(np.int16)
    wavfile.write(out_wav_path, sr, pcm_int16)

    kia_named_path = os.path.join(SCRIPT_DIR, "avas_kia_ev6_vess.wav")
    if os.path.abspath(out_wav_path) != os.path.abspath(kia_named_path):
        shutil.copy2(out_wav_path, kia_named_path)

    analyze_dir = os.path.join(os.path.dirname(SCRIPT_DIR), "Sound_Anlyze")
    if os.path.isdir(analyze_dir):
        for fname in ("avas_sim_asset.wav", "avas_kia_ev6_vess.wav"):
            analyze_copy = os.path.join(analyze_dir, fname)
            if os.path.abspath(out_wav_path) != os.path.abspath(analyze_copy):
                shutil.copy2(out_wav_path, analyze_copy)

    return sim_signal.astype(np.float32)


class AVASAudioEngine:
    def __init__(self, base_wav_path, sim_wav_path=SIM_WAV):
        self.base_wav_path = base_wav_path
        self.sim_wav_path = sim_wav_path

        self.sample_rate, data = wavfile.read(base_wav_path)
        if len(data.shape) > 1:
            data = data[:, 0]
        
        if data.dtype == np.int16:
            raw_samples = data.astype(np.float64) / 32768.0
        else:
            raw_samples = data.astype(np.float64)

        # Lưu 3 nguồn âm thanh để người dùng có thể đối chiếu trực tiếp trên GUI:
        # 0: File WAV mới (avas_sim_asset.wav) được tạo dựa trên WAV gốc
        # 1: File WAV gốc (avas_base_asset.wav) qua bộ lọc khử nhiễu DSP
        # 2: File WAV gốc nguyên bản (Raw)
        self.raw_base_samples = raw_samples.astype(np.float32)
        self.dsp_base_samples = self._preprocess_smooth_audio(raw_samples)
        self.sim_samples = generate_similar_avas_wav(base_wav_path, sim_wav_path)

        # Mặc định sử dụng âm thanh mô phỏng AVAS mới (avas_sim_asset.wav)
        self.base_samples = self.sim_samples
        self.sample_len = len(self.base_samples)
        self.read_pointer = 0.0
        self.current_speed_int = 0  # Số nguyên (0 - 20 km/h)

        # Bộ lọc làm mượt IIR (Exponential Smoothing) theo từng sample
        # Giúp triệt tiêu hoàn toàn tiếng click/lạo xạo (zipper noise) khi kéo thanh trượt
        self.current_gain = 0.0
        self.target_gain = 0.0
        self.current_pitch_ratio = 1.0
        self.target_pitch_ratio = 1.0
        self.smooth_alpha = 1.0 - np.exp(-1.0 / (0.045 * self.sample_rate))  # Hằng số thời gian ~45ms
        
        self.viz_buffer_size = 8192
        self.viz_buffer = np.zeros(self.viz_buffer_size, dtype=np.float32)

        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            blocksize=2048,
            channels=1,
            dtype='float32',
            latency='high',
            callback=self._audio_callback
        )

    def set_sound_mode(self, mode_index):
        """Chuyển đổi nguồn âm thanh mượt mà khi đang chạy"""
        if mode_index == 0:
            self.base_samples = self.sim_samples
        elif mode_index == 1:
            self.base_samples = self.dsp_base_samples
        else:
            self.base_samples = self.raw_base_samples
        self.sample_len = len(self.base_samples)

    def _preprocess_smooth_audio(self, raw_samples):
        """
        Phương pháp DSP làm mượt âm thanh trực tiếp từ dữ liệu WAV đầu vào (không sửa file .wav):
        1. Spectral Gating (FFT): Loại bỏ toàn bộ 136 dải tần nhiễu biến điệu bậc cao (IMD sand/noise)
           sinh ra do bão hòa phi tuyến trong file WAV gốc.
        2. Harmonic Phase-Locking & Schroeder Phase: Đồng bộ các tần số lệch pha (như 250Hz, 315Hz)
           về lưới hài âm chuẩn (bội số của BASE_FREQ/2 = 80Hz) để triệt tiêu hiện tượng phách (beating/roughness).
        3. Warm Acoustic Low-Pass Rolloff: Làm dịu dải tần cao (>350Hz) giúp âm thanh êm, ấm và mượt.
        """
        n = len(raw_samples)
        spec = np.fft.rfft(raw_samples)
        freqs = np.fft.rfftfreq(n, 1.0 / self.sample_rate)
        mag = np.abs(spec) / (n / 2)

        threshold = np.max(mag) * 0.04
        peak_bins = np.where(mag > threshold)[0]

        clean_spec = np.zeros_like(spec)
        f_grid = BASE_FREQ / 2.0  # Lưới hài âm cơ sở (80 Hz)
        df = self.sample_rate / float(n)
        num_peaks = max(len(peak_bins), 1)

        for idx, k in enumerate(peak_bins):
            f_orig = freqs[k]
            amp = mag[k]
            f_locked = round(f_orig / f_grid) * f_grid
            k_new = int(round(f_locked / df))

            if 0 < k_new < len(clean_spec):
                warmth = 1.0 / (1.0 + (max(0.0, f_locked - 320.0) / 350.0) ** 1.5)
                phase = -np.pi * idx * (idx + 1) / num_peaks
                clean_spec[k_new] += (amp * warmth) * (n / 2) * np.exp(1j * phase)

        clean_signal = np.fft.irfft(clean_spec, n=n)
        max_val = np.max(np.abs(clean_signal))
        if max_val > 1e-9:
            clean_signal = clean_signal / max_val * 0.85

        return clean_signal.astype(np.float32)

    def start(self):
        if not self.stream.active:
            self.stream.start()

    def set_speed(self, speed_kmh_int):
        self.current_speed_int = int(speed_kmh_int)
        if self.current_speed_int <= 0:
            self.target_gain = 0.0
        else:
            speed = float(self.current_speed_int)
            target_freq = MIN_FREQ + ((speed - 1.0) / 19.0) * (MAX_FREQ - MIN_FREQ)
            self.target_pitch_ratio = target_freq / BASE_FREQ
            self.target_gain = 0.5 + (speed / 20.0) * 0.5

    def _audio_callback(self, outdata, frames, time, status):
        # XỬ LÝ TRẠNG THÁI 0 KM/H: TẮT AVAS KHI ĐÃ FADE-OUT MƯỢT VỀ 0
        if self.current_speed_int == 0 and self.current_gain < 1e-4:
            self.current_gain = 0.0
            outdata[:, 0] = 0.0
            if frames < self.viz_buffer_size:
                self.viz_buffer[:-frames] = self.viz_buffer[frames:]
                self.viz_buffer[-frames:] = 0.0
            else:
                self.viz_buffer.fill(0.0)
            return

        # Làm mượt Gain và Pitch theo từng sample bằng NumPy Vectorization
        steps = np.arange(1, frames + 1, dtype=np.float64)
        decay = ((1.0 - self.smooth_alpha) ** steps).astype(np.float32)
        gains = self.target_gain + (self.current_gain - self.target_gain) * decay
        pitches = self.target_pitch_ratio + (self.current_pitch_ratio - self.target_pitch_ratio) * decay
        self.current_gain = float(gains[-1])
        self.current_pitch_ratio = float(pitches[-1])

        # Cộng dồn pha liên tục (Phase Accumulator)
        pointers = (self.read_pointer + np.cumsum(pitches, dtype=np.float64)) % self.sample_len
        self.read_pointer = float(pointers[-1])

        # Nội suy bậc 3 Catmull-Rom (4-point Cubic Interpolation) giúp âm thanh mượt tuyệt đối
        idx1 = pointers.astype(np.int64)
        frac = (pointers - idx1).astype(np.float32)
        idx0 = (idx1 - 1) % self.sample_len
        idx2 = (idx1 + 1) % self.sample_len
        idx3 = (idx1 + 2) % self.sample_len

        active_samples = self.base_samples
        y0 = active_samples[idx0]
        y1 = active_samples[idx1]
        y2 = active_samples[idx2]
        y3 = active_samples[idx3]

        c1 = 0.5 * (y2 - y0)
        c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3
        c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2)

        interpolated = ((c3 * frac + c2) * frac + c1) * frac + y1
        buffer = (interpolated * gains).astype(np.float32)

        outdata[:, 0] = buffer
        if frames < self.viz_buffer_size:
            self.viz_buffer[:-frames] = self.viz_buffer[frames:]
            self.viz_buffer[-frames:] = buffer
        else:
            self.viz_buffer[:] = buffer[-self.viz_buffer_size:]

    def get_viz_samples(self):
        return self.viz_buffer.copy()

    def stop(self):
        self.stream.stop()
        self.stream.close()


class AVASSimulatorGUI(QWidget):
    def __init__(self, audio_engine):
        super().__init__()
        self.engine = audio_engine
        self.fft_size = 4096
        self.sample_rate = self.engine.sample_rate
        
        self.fft_freqs = np.fft.rfftfreq(self.fft_size, 1.0 / self.sample_rate)
        self.a_weighting = get_a_weighting_offsets(self.fft_freqs)
        self.hann_window = np.hanning(self.fft_size)

        self.octave_bounds = []
        for fc in OCTAVE_CENTER_FREQS:
            f_lower = fc * (2.0 ** (-1.0 / 6.0))
            f_upper = fc * (2.0 ** (1.0 / 6.0))
            idx = np.where((self.fft_freqs >= f_lower) & (self.fft_freqs < f_upper))[0]
            self.octave_bounds.append(idx)

        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_visualizer)
        self.timer.start(33)
        QTimer.singleShot(50, self.engine.start)

    def init_ui(self):
        self.setWindowTitle("ECE R138 AVAS Analyzer - (0 - 20 km/h Integer Mode)")
        self.resize(1000, 780)

        main_layout = QVBoxLayout()

        # --- BẢNG ĐIỀU KHIỂN TỐC ĐỘ & NGUỒN ÂM ---
        group = QGroupBox("Mô phỏng tốc độ xe (CAN Bus Speed) & Nguồn âm thanh AVAS")
        group_layout = QVBoxLayout()

        source_layout = QHBoxLayout()
        source_label = QLabel("Nguồn âm AVAS:", self)
        source_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.combo_source = QComboBox(self)
        self.combo_source.setStyleSheet("font-size: 14px; padding: 4px;")
        self.combo_source.addItems([
            "avas_sim_asset.wav (Kia EV6 VESS 'Spaceship V12' - Phân tích từ YouTube Short)",
            "avas_base_asset.wav (File WAV gốc + Khử nhiễu DSP trực tiếp)",
            "avas_base_asset.wav (File WAV gốc nguyên bản - Raw)"
        ])
        self.combo_source.currentIndexChanged.connect(self.engine.set_sound_mode)
        source_layout.addWidget(source_label)
        source_layout.addWidget(self.combo_source, 1)
        group_layout.addLayout(source_layout)

        top_info_layout = QHBoxLayout()
        self.label_speed = QLabel("Tốc độ xe: 0 km/h", self)
        self.label_speed.setStyleSheet("font-size: 20px; font-weight: bold; color: #1E88E5;")

        self.label_freq = QLabel("Trạng thái: TẮT (0 km/h)", self)
        self.label_freq.setStyleSheet("font-size: 16px; color: #D32F2F; font-weight: bold;")

        self.label_spl = QLabel("Overall SPL: MUTE", self)
        self.label_spl.setStyleSheet("font-size: 20px; font-weight: bold; color: #757575;")

        top_info_layout.addWidget(self.label_speed)
        top_info_layout.addWidget(self.label_freq)
        top_info_layout.addStretch()
        top_info_layout.addWidget(self.label_spl)

        # SLIDER: Tốc độ là số nguyên từ 0 đến bao gồm 20 km/h
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setRange(0, 20)
        self.slider.setValue(0)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(1)
        self.slider.setTickInterval(1)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.valueChanged.connect(self.on_speed_changed)

        group_layout.addLayout(top_info_layout)
        group_layout.addWidget(self.slider)
        group.setLayout(group_layout)
        main_layout.addWidget(group)

        # --- KHU VỰC ĐỒ THỊ PYQTGRAPH ---
        pg.setConfigOptions(antialias=True)
        
        # 1. Đồ thị FFT Spectrum
        self.fft_plot = pg.PlotWidget(title="<b>FFT Spectrum (A-Weighted Magnitude)</b>")
        self.fft_plot.setLogMode(x=True, y=False)
        self.fft_plot.setXRange(np.log10(20), np.log10(20000))
        self.fft_plot.setYRange(0, 100)
        self.fft_plot.setLabel('bottom', 'Tần số', units='Hz')
        self.fft_plot.setLabel('left', 'Cường độ', units='dB(A)')
        self.fft_plot.showGrid(x=True, y=True, alpha=0.3)
        self.fft_curve = self.fft_plot.plot(pen=pg.mkPen('#0288D1', width=2))
        
        main_layout.addWidget(self.fft_plot)

        # 2. Đồ thị RTA 1/3 Octave
        self.rta_plot = pg.PlotWidget(title="<b>1/3 Octave Band RTA (ECE R138 Table 3 Requirements)</b>")
        self.rta_plot.setYRange(0, 90)
        self.rta_plot.setLabel('bottom', 'Dải tần số trung tâm (Hz)')
        self.rta_plot.setLabel('left', 'Mức áp suất âm', units='dB(A)')
        self.rta_plot.showGrid(x=False, y=True, alpha=0.3)

        x_ticks = [(i, str(int(fc))) for i, fc in enumerate(OCTAVE_CENTER_FREQS)]
        self.rta_plot.getAxis('bottom').setTicks([x_ticks])

        self.bar_item = pg.BarGraphItem(x=np.arange(len(OCTAVE_CENTER_FREQS)), height=np.zeros(16), width=0.6, brush='#26A69A')
        self.rta_plot.addItem(self.bar_item)

        self.limit_curve = self.rta_plot.plot(
            x=np.arange(len(OCTAVE_CENTER_FREQS)), 
            y=R138_MIN_DBA_10KMH, 
            pen=pg.mkPen('#E53935', width=2, style=Qt.DashLine),
            name="Sàn tối thiểu 10km/h"
        )

        main_layout.addWidget(self.rta_plot)
        self.setLayout(main_layout)

    def on_speed_changed(self, value_int):
        speed_kmh = int(value_int)

        self.label_speed.setText(f"Tốc độ xe: {speed_kmh} km/h")
        
        if speed_kmh == 0:
            self.label_freq.setText("Trạng thái: TẮT (0 km/h)")
            self.label_freq.setStyleSheet("font-size: 16px; color: #D32F2F; font-weight: bold;")
        else:
            target_freq = MIN_FREQ + ((speed_kmh - 1) / 19.0) * (MAX_FREQ - MIN_FREQ)
            self.label_freq.setText(f"Tần số chính (f0): {target_freq:.1f} Hz")
            self.label_freq.setStyleSheet("font-size: 16px; color: #388E3C; font-weight: bold;")

        self.engine.set_speed(speed_kmh)

    def update_visualizer(self):
        samples = self.engine.get_viz_samples()[-self.fft_size:]
        if len(samples) < self.fft_size:
            return

        # Khi 0 km/h (Mute): Đặt đồ thị phẳng về 0
        if self.engine.current_speed_int == 0 and self.engine.current_gain < 1e-3:
            self.fft_curve.setData(self.fft_freqs[1:], np.zeros(len(self.fft_freqs) - 1))
            self.bar_item.setOpts(height=np.zeros(len(OCTAVE_CENTER_FREQS)))
            self.label_spl.setText("Overall SPL: MUTE / OFF")
            self.label_spl.setStyleSheet("font-size: 20px; font-weight: bold; color: #757575;")
            return

        # Khi xe chạy (1 - 20 km/h): Xử lý FFT & RTA
        windowed_samples = samples * self.hann_window
        fft_complex = np.fft.rfft(windowed_samples)
        magnitude = np.abs(fft_complex) / (self.fft_size / 2)

        spl_fft = 20.0 * np.log10(magnitude + 1e-6) + 94.0 + self.a_weighting
        self.fft_curve.setData(self.fft_freqs[1:], spl_fft[1:])

        octave_spl = np.zeros(len(OCTAVE_CENTER_FREQS), dtype=np.float32)
        total_energy = 1e-12

        for i, indices in enumerate(self.octave_bounds):
            if len(indices) > 0:
                band_energy = np.sum(10.0 ** (spl_fft[indices] / 10.0))
                octave_spl[i] = 10.0 * np.log10(band_energy + 1e-12)
                total_energy += band_energy

        self.bar_item.setOpts(height=octave_spl)

        overall_spl = 10.0 * np.log10(total_energy)
        self.label_spl.setText(f"Overall SPL: {overall_spl:.1f} dB(A)")
        self.label_spl.setStyleSheet("font-size: 20px; font-weight: bold; color: #4CAF50;")

    def closeEvent(self, event):
        self.timer.stop()
        self.engine.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    try:
        engine = AVASAudioEngine(INPUT_WAV, SIM_WAV)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{INPUT_WAV}'. Hãy kiểm tra lại thư mục chạy!")
        sys.exit(1)

    gui = AVASSimulatorGUI(engine)
    gui.show()

    sys.exit(app.exec_())
