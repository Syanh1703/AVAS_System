import sys
import os
import io
import collections
import numpy as np
from scipy.io import wavfile
from scipy import signal
import sounddevice as sd

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
    QSlider, QLabel, QGroupBox, QComboBox, QPushButton,
    QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg

# Khắc phục lỗi mã hóa UTF-8 trên Windows Terminal
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# --- THÔNG SỐ CẤU HÌNH AVAS & TIÊU CHUẨN UN ECE R138 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_FREQ = 160.0    # Tần số gốc của file WAV nền (Hz)
MIN_FREQ = 160.0     # Tần số tại 1 km/h (Âm trầm mượt, đầm xe)
MAX_FREQ = 256.0     # Tần số tại 20 km/h (Tương ứng tăng +60% pitch: 160 -> 256 Hz khi tăng tốc)

# Hệ số độ dịch tần số trung bình (% / km/h) theo Mục 6.2.3 - UN ECE R138 (Yêu cầu >= 0.8% / km/h)
AVG_PITCH_SHIFT_PCT_PER_KMH = (((MAX_FREQ - MIN_FREQ) / BASE_FREQ) * 100.0) / 19.0  # ~3.16% / km/h
R138_MIN_SHIFT_PCT_PER_KMH = 0.8

# Mức áp suất âm tổng thể (Overall SPL) theo Mục 6.2.1 & 6.2.8 - UN ECE R138 (Đo tại khoảng cách 2m)
R138_MIN_OVERALL_10KMH = 50.0  # Tối thiểu 50 dB(A) tại 10 km/h (Tiến)
R138_MIN_OVERALL_20KMH = 56.0  # Tối thiểu 56 dB(A) tại 20 km/h (Tiến)
R138_MIN_OVERALL_REV   = 47.0  # Tối thiểu 47 dB(A) khi Lùi
R138_MAX_OVERALL_SPL   = 75.0  # Tối đa không vượt quá 75 dB(A) ở mọi tốc độ (Mục 6.2.8)

# Chuẩn hiệu chuẩn SPL tại khoảng cách micro 2.0m (Annex 3 - ECE R138)
SPL_CALIBRATION_DB = 85.0

# 16 Dải tần số trung tâm 1/3 Octave theo Bảng 3 - ECE R138 (Hz)
OCTAVE_CENTER_FREQS = np.array([
    160, 200, 250, 315, 400, 500, 630, 800, 
    1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000
], dtype=np.float32)

# Mức tối thiểu dB(A) từng dải 1/3 Octave quy định ở 10 km/h (Bảng 3 - Cột 10 km/h)
R138_MIN_DBA_10KMH = np.array([
    45, 44, 43, 44, 45, 45, 46, 46,
    46, 46, 44, 42, 39, 36, 34, 31
], dtype=np.float32)

# Mức tối thiểu dB(A) từng dải 1/3 Octave quy định ở 20 km/h (Bảng 3 - Cột 20 km/h: +5 dB so với 10 km/h)
R138_MIN_DBA_20KMH = np.array([
    50, 49, 48, 49, 50, 50, 51, 51,
    51, 51, 49, 47, 44, 41, 39, 36
], dtype=np.float32)


def get_r138_band_limits_for_speed(speed_kmh):
    """
    Nội suy đường giới hạn sàn Bảng 3 - UN ECE R138 theo tốc độ xe:
    - Từ 0 đến 10 km/h: Tham chiếu cột tiêu chuẩn 10 km/h (R138_MIN_DBA_10KMH).
    - Từ 10 đến 20 km/h: Nội suy tuyến tính từ cột 10 km/h lên cột 20 km/h (+5 dB(A)).
    """
    if speed_kmh <= 10:
        return R138_MIN_DBA_10KMH.copy()
    ratio = min(max((float(speed_kmh) - 10.0) / 10.0, 0.0), 1.0)
    return R138_MIN_DBA_10KMH + ratio * (R138_MIN_DBA_20KMH - R138_MIN_DBA_10KMH)


def get_r138_min_overall_spl(speed_kmh, is_reverse=False):
    """Tính ngưỡng Overall SPL tối thiểu theo UN ECE R138 (Mục 6.2.1.1 & 6.2.1.2)"""
    if is_reverse:
        return R138_MIN_OVERALL_REV
    if speed_kmh <= 10:
        return R138_MIN_OVERALL_10KMH
    ratio = min(max((float(speed_kmh) - 10.0) / 10.0, 0.0), 1.0)
    return R138_MIN_OVERALL_10KMH + ratio * (R138_MIN_OVERALL_20KMH - R138_MIN_OVERALL_10KMH)


def evaluate_r138_compliance(octave_spl, overall_spl, speed_kmh, is_reverse=False):
    """
    Đánh giá toàn diện các tiêu chuẩn UN ECE R138 (Mục 6.2 & Bảng 3):
    1. Two-Band Rule (Mục 6.2.1.3): Tối thiểu 2 dải 1/3 Octave đạt ngưỡng Bảng 3,
       trong đó có ít nhất 1 dải có tần số trung tâm <= 1600 Hz.
    2. Overall SPL: Nằm trong khoảng [Min Overall SPL .. 75.0 dB(A)].
    3. Frequency Shift (Mục 6.2.3): Độ dịch tần số >= 0.8% / km/h.
    """
    band_limits = get_r138_band_limits_for_speed(speed_kmh)
    passed_mask = octave_spl >= band_limits
    num_passed_bands = int(np.sum(passed_mask))

    # Các dải từ 160 Hz đến 1600 Hz nằm ở chỉ số 0..10
    low_mid_passed = int(np.sum(passed_mask[:11]))
    two_band_ok = (num_passed_bands >= 2) and (low_mid_passed >= 1)

    min_overall = get_r138_min_overall_spl(speed_kmh, is_reverse=is_reverse)
    overall_min_ok = overall_spl >= min_overall
    overall_max_ok = overall_spl <= R138_MAX_OVERALL_SPL
    overall_ok = overall_min_ok and overall_max_ok

    shift_ok = AVG_PITCH_SHIFT_PCT_PER_KMH >= R138_MIN_SHIFT_PCT_PER_KMH
    all_pass = two_band_ok and overall_ok and shift_ok

    return {
        "band_limits": band_limits,
        "passed_mask": passed_mask,
        "num_passed_bands": num_passed_bands,
        "low_mid_passed": low_mid_passed,
        "two_band_ok": two_band_ok,
        "min_overall": min_overall,
        "overall_ok": overall_ok,
        "overall_max_ok": overall_max_ok,
        "shift_ok": shift_ok,
        "all_pass": all_pass,
    }


def get_a_weighting_offsets(freqs):
    """Tính offset Trọng số A (A-Weighting IEC 61672) cho mảng tần số (Hz)"""
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


def discover_wav_files(root_dir):
    """Quét toàn bộ các file .wav hiện có trong thư mục gốc và các thư mục Profile con"""
    wav_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in sorted(filenames):
            if fname.lower().endswith(".wav"):
                wav_files.append(os.path.join(dirpath, fname))
    return wav_files


def soft_knee_limit(samples, threshold=0.72, ceiling=0.95):
    """
    Bộ giới hạn biên độ mềm (Soft-Knee Hyperbolic Tangent Limiter):
    - Giữ tuyến tính 100% với tín hiệu dưới ngưỡng threshold (0.72).
    - Nén mềm theo hàm tanh với các đỉnh cao hơn, loại bỏ hoàn toàn méo hài bậc lẻ do cắt ngọn (hard clipping).
    """
    abs_x = np.abs(samples)
    over = np.maximum(abs_x - threshold, 0.0)
    headroom = ceiling - threshold
    compressed = threshold + headroom * np.tanh(over / max(headroom, 1e-6))
    return np.where(abs_x > threshold, np.sign(samples) * compressed, samples).astype(np.float32)


class AVASAudioEngine:
    def __init__(self, initial_wav_path=None):
        self.current_wav_path = None
        self.sample_rate = 44100
        self.sound_mode = 0  # 0: Original WAV (Auto De-Spiked), 1: Smooth DSP Filter

        # Khởi tạo buffer mặc định (im lặng) nếu chưa nạp file WAV
        default_len = self.sample_rate * 6
        self.raw_base_samples = np.zeros(default_len, dtype=np.float32)
        self.dsp_base_samples = np.zeros(default_len, dtype=np.float32)
        self.bg_hum_samples = np.zeros(default_len, dtype=np.float32)
        self.acc_mid_samples = np.zeros(default_len, dtype=np.float32)
        self.acc_high_samples = np.zeros(default_len, dtype=np.float32)
        self.base_samples = self.raw_base_samples
        self.sample_len = len(self.base_samples)

        # Con trỏ đọc cho 2 lớp âm thanh:
        # 1. bg_read_pointer : Lớp nền (Background Hum) luôn hoạt động kể cả khi xe đứng yên (Idle 0 km/h)
        # 2. read_pointer    : Lớp tăng tốc (Acceleration Layer) được điều chỉnh pitch & âm sắc theo vận tốc
        self.bg_read_pointer = 0.0
        self.read_pointer = 0.0
        self.current_speed_int = 0  # Số nguyên (0 - 20 km/h)

        # Bộ lọc làm mượt IIR (Exponential Smoothing) theo từng sample
        # Giúp triệt tiêu hoàn toàn tiếng click/lạo xạo (zipper noise) khi kéo thanh trượt
        self.current_bg_gain = 0.50
        self.target_bg_gain = 0.50
        self.current_acc_gain = 0.16
        self.target_acc_gain = 0.16
        self.current_bright_gain = 0.05
        self.target_bright_gain = 0.05
        self.accel_torque_boost = 0.0  # Tăng cường hài âm tức thời khi xe đang gia tốc (dv/dt > 0)
        self.current_pitch_ratio = 1.0
        self.target_pitch_ratio = 1.0

        self._update_dsp_coefficients()
        
        # Bộ đệm hiển thị 16384 mẫu phục vụ phân tích FFT độ phân giải cao (8192-point FFT)
        self.viz_buffer_size = 16384
        self.viz_buffer = np.zeros(self.viz_buffer_size, dtype=np.float32)
        self.stream = None

        if initial_wav_path and os.path.exists(initial_wav_path):
            self.load_wav_file(initial_wav_path)
        else:
            self._init_stream()

    def _update_dsp_coefficients(self):
        """Khởi tạo các bộ lọc Formant cộng hưởng thân xe cố định (Fixed Body Formant Filter)"""
        self.smooth_alpha = 1.0 - np.exp(-1.0 / (0.045 * self.sample_rate))  # Hằng số thời gian ~45ms
        self.torque_decay = float(np.exp(-1.0 / (0.22 * self.sample_rate)))  # Độ rã tải gia tốc ~220ms

        # Bộ lọc cộng hưởng Formant cố định (215 Hz và 485 Hz - đặc trưng khoang cộng hưởng Audi e-tron)
        # Giữ nguyên màu âm trầm ấm của thân xe ngay cả khi hài âm động cơ quét pitch lên cao
        self.b_f1, self.a_f1 = signal.iirpeak(215.0, Q=3.0, fs=self.sample_rate)
        self.zi_f1 = np.zeros(max(len(self.a_f1), len(self.b_f1)) - 1, dtype=np.float64)

        self.b_f2, self.a_f2 = signal.iirpeak(485.0, Q=3.8, fs=self.sample_rate)
        self.zi_f2 = np.zeros(max(len(self.a_f2), len(self.b_f2)) - 1, dtype=np.float64)

    def _init_stream(self):
        was_active = False
        if self.stream is not None:
            was_active = self.stream.active
            self.stream.stop()
            self.stream.close()

        self._update_dsp_coefficients()
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            blocksize=2048,
            channels=1,
            dtype='float32',
            latency='high',
            callback=self._audio_callback
        )
        if was_active:
            self.stream.start()

    def _ensure_seamless_loop(self, samples):
        """
        1. Khử thành phần một chiều (DC Offset Removal).
        2. Bảo vệ điểm nối vòng lặp (Equal-Power Crossfade Loop Guard):
           Nếu người dùng import một file .wav bất kỳ có độ lệch biên độ/pha ở điểm cuối và điểm đầu,
           tự động áp dụng cửa sổ hòa trộn Cosine (60 ms) để triệt tiêu hoàn toàn tiếng 'tách' (seam click).
        """
        n = len(samples)
        if n < 2048:
            return samples

        samples = samples - np.mean(samples)
        seam_step = abs(float(samples[-1]) - float(samples[0]))

        # Nếu file WAV đã có chu kỳ nguyên hoàn hảo (seam_step rất nhỏ), giữ nguyên 100%
        if seam_step <= 0.012:
            return samples

        fade_len = min(int(0.060 * self.sample_rate), n // 8)
        if fade_len < 64:
            return samples

        out = samples.copy()
        t = np.linspace(0.0, np.pi / 2.0, fade_len, dtype=np.float64)
        fade_out = np.cos(t) ** 2
        fade_in = np.sin(t) ** 2

        # Hòa trộn mượt đoạn cuối (tail) hướng về đoạn đầu (head) để điểm nối tròn pha
        out[-fade_len:] = out[-fade_len:] * fade_out + out[:fade_len] * fade_in
        return out

    def _auto_despike_and_disperse_phase(self, raw_samples):
        """
        Solution 1: Tự động khử gai nhiễu biến điệu (IMD De-Spiker), chống phách bất hài âm (Beating Guard)
        và phân tán pha Schroeder (Schroeder Phase Dispersion) ngay khi nạp file .wav (kể cả ở chế độ Original WAV):
        1. Spectral IMD Gate (5%): Cắt bỏ toàn bộ ~136 gai nhiễu biến điệu bậc cao (IMD spikes) sinh ra
           bởi hàm bão hòa tanh(x * 1.4) trong avas_base_asset.wav, giữ lại các đỉnh tần số chính.
        2. Inharmonic Beating Guard: Phát hiện các tần số lệch lưới gây phách 5 Hz (như 250 Hz và 315 Hz
           đánh nhau với 160 Hz / 320 Hz trong avas_base_asset.wav) và căn chỉnh nhẹ về lưới hài âm chuẩn.
        3. Schroeder Phase Dispersion: Phân tán góc pha của các đỉnh tần số theo công thức Schroeder
           phi_k = -pi * k * (k + 1) / N để ngăn các sóng sin đồng pha (phi = 0) cộng hưởng thành gai nhọn.
        """
        n = len(raw_samples)
        if n == 0:
            return np.zeros(1, dtype=np.float64)

        spec = np.fft.rfft(raw_samples)
        freqs = np.fft.rfftfreq(n, 1.0 / self.sample_rate)
        mag = np.abs(spec) / (n / 2)

        max_mag = np.max(mag)
        if max_mag <= 1e-9:
            return raw_samples

        # Kiểm tra xem file có bị nhiễu biến điệu bậc cao (IMD spikes như avas_base_asset.wav có > 300 vạch phổ) không.
        # Nếu file đã sạch và phẳng sẵn (như audi_etron_forward.wav có <= 45 vạch hài âm chuẩn), giữ nguyên 100%
        # để bảo toàn trọn vẹn cả tiếng Hum trầm lẫn hợp âm phi thuyền (Spaceship-like sound).
        active_bins_count = int(np.sum(mag > max_mag * 1e-3))
        if active_bins_count <= 45:
            return raw_samples

        # 1. Cổng lọc phổ (Spectral Gate @ 5% max peak): Loại bỏ toàn bộ gai nhiễu nền IMD
        threshold = max_mag * 0.05
        peak_bins = np.where(mag > threshold)[0]

        despiked_spec = np.zeros_like(spec)
        df = self.sample_rate / float(n)
        f_grid = BASE_FREQ / 2.0  # Lưới hài âm 80 Hz cho file bị nhiễu IMD
        num_peaks = max(len(peak_bins), 1)

        for idx, k in enumerate(peak_bins):
            f_orig = freqs[k]
            amp = mag[k]

            # 2. Chống phách (Inharmonic Beating Guard):
            nearest_harmonic = round(f_orig / f_grid) * f_grid
            if abs(f_orig - nearest_harmonic) <= 10.5 and nearest_harmonic > 0:
                f_target = nearest_harmonic
            else:
                f_target = f_orig

            k_new = int(round(f_target / df))
            if 0 < k_new < len(despiked_spec):
                # 3. Phân tán pha Schroeder (Schroeder Phase Dispersion) giúp đường bao phẳng tuyệt đối
                phase = -np.pi * idx * (idx + 1) / num_peaks
                despiked_spec[k_new] += amp * (n / 2) * np.exp(1j * phase)

        clean_signal = np.fft.irfft(despiked_spec, n=n)
        peak_val = np.max(np.abs(clean_signal))
        orig_peak = min(np.max(np.abs(raw_samples)), 0.85)
        if peak_val > 1e-9:
            clean_signal = clean_signal / peak_val * orig_peak

        return clean_signal

    def load_wav_file(self, wav_path):
        """Nạp file .wav do người dùng chọn/import, tự động khử gai IMD (Solution 1) và tách lớp âm thanh"""
        new_sr, data = wavfile.read(wav_path)
        if len(data.shape) > 1:
            data = data[:, 0]

        if data.dtype == np.int16:
            raw_samples = data.astype(np.float64) / 32768.0
        elif data.dtype == np.int32:
            raw_samples = data.astype(np.float64) / 2147483648.0
        elif data.dtype == np.uint8:
            raw_samples = (data.astype(np.float64) - 128.0) / 128.0
        else:
            raw_samples = data.astype(np.float64)

        sr_changed = (new_sr != self.sample_rate) or (self.stream is None)
        self.sample_rate = int(new_sr)
        self.current_wav_path = os.path.abspath(wav_path)

        # Khử DC offset và đảm bảo vòng lặp mượt không bị click
        raw_samples = self._ensure_seamless_loop(raw_samples)

        # Áp dụng Solution 1 ngay khi nạp file: Tự động khử gai IMD + chống phách + phân tán pha Schroeder
        despiked_samples = self._auto_despike_and_disperse_phase(raw_samples)

        raw_f32 = despiked_samples.astype(np.float32)
        dsp_f32 = self._preprocess_smooth_audio(despiked_samples)

        self.raw_base_samples = raw_f32
        self.dsp_base_samples = dsp_f32
        self.bg_read_pointer = 0.0
        self.read_pointer = 0.0
        self.set_sound_mode(self.sound_mode)

        if sr_changed:
            self._init_stream()

    def _build_layers(self, active_source):
        """
        Tách tín hiệu gốc thành 3 dải thành phần tuyến tính có khử răng cưa (Band-Limited Anti-Aliasing)
        để phần mềm điều khiển động lực học khi xe tăng tốc (không dùng tanh phi tuyến tránh sinh gai IMD):
        1. bg_hum_samples   : Lớp âm trầm nền (< 260 Hz) luôn phát liên tục kể cả ở 0 km/h (Idle).
        2. acc_mid_samples  : Lớp hài âm động cơ trung tâm (150 - 520 Hz) quét pitch mượt theo tốc độ.
        3. acc_high_samples : Lớp hài âm cao / tiếng vút Turbine (380 - 2400 Hz, lọc tuyến tính sạch 100%)
                              mở rộng dần khi xe tăng tốc nhanh (dv/dt > 0) và ở vận tốc cao.
        """
        n = len(active_source)
        if n == 0:
            z = np.zeros(1, dtype=np.float32)
            return z, z, z

        spec = np.fft.rfft(active_source)
        freqs = np.fft.rfftfreq(n, 1.0 / self.sample_rate)

        # 1. Lớp Background Hum (< 190 Hz): bộ lọc thông thấp bậc 8 cắt dứt khoát tại 190 Hz
        #    Giữ nguyên 100% tiếng Hum trầm ấm (70, 105, 140, 175 Hz) ở nền và tuyệt đối không
        #    chứa các tần số >= 210 Hz để triệt tiêu hoàn toàn giao thoa (spinning/beating) với lớp tăng tốc!
        bg_LP = 1.0 / (1.0 + (freqs / 190.0) ** 8.0)
        bg_sig = np.fft.irfft(spec * bg_LP, n=n)

        # 2. Lớp Acceleration / Spaceship Mid-Body (195 Hz - 560 Hz): chỉ chứa hợp âm Spaceship (>= 210 Hz)
        acc_HP = 1.0 / (1.0 + (195.0 / np.maximum(freqs, 1.0)) ** 8.0)
        acc_mid_LP = 1.0 / (1.0 + (freqs / 560.0) ** 4.0)
        acc_mid_sig = np.fft.irfft(spec * acc_HP * acc_mid_LP, n=n)

        # 3. Lớp Acceleration High-Turbine (480 Hz - 2400 Hz) tuyến tính hoàn toàn (không sinh gai IMD)
        high_HP = 1.0 / (1.0 + (480.0 / np.maximum(freqs, 1.0)) ** 4.0)
        anti_alias_LP = 1.0 / (1.0 + (freqs / 2400.0) ** 6.0)
        high_spec = spec * high_HP * anti_alias_LP
        acc_high_sig = np.fft.irfft(high_spec, n=n)

        return (
            bg_sig.astype(np.float32),
            acc_mid_sig.astype(np.float32),
            acc_high_sig.astype(np.float32)
        )

    def set_sound_mode(self, mode_index):
        """Chuyển đổi chế độ xử lý âm thanh mượt mà khi đang chạy (0: WAV gốc đã khử gai, 1: Làm ấm DSP)"""
        self.sound_mode = int(mode_index)
        if self.sound_mode == 1:
            self.base_samples = self.dsp_base_samples
        else:
            self.base_samples = self.raw_base_samples
        self.sample_len = max(len(self.base_samples), 1)
        self.bg_hum_samples, self.acc_mid_samples, self.acc_high_samples = self._build_layers(self.base_samples)
        self.bg_read_pointer = self.bg_read_pointer % self.sample_len
        self.read_pointer = self.read_pointer % self.sample_len

    def _preprocess_smooth_audio(self, raw_samples):
        """
        Phương pháp DSP làm mượt & làm ấm âm thanh nâng cao (Chế độ Smooth DSP):
        - Áp dụng thêm bộ lọc Warm Acoustic Low-Pass Rolloff (>320 Hz) giúp âm thanh trầm ấm hơn.
        """
        n = len(raw_samples)
        if n == 0:
            return np.zeros(1, dtype=np.float32)

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
        if self.stream is not None and not self.stream.active:
            self.stream.start()

    def set_speed(self, speed_kmh_int):
        new_speed = int(speed_kmh_int)
        delta_speed = new_speed - self.current_speed_int
        self.current_speed_int = new_speed

        # Tiếng Hum nền (0.50) luôn giữ nguyên đầy đặn trong suốt mọi trạng thái, kể cả khi đứng yên (Idle 0 km/h)
        self.target_bg_gain = 0.50

        if new_speed <= 0:
            # Tại 0 km/h (Idle): Phát tiếng Hum nền đầy đặn + một lớp duy nhất hợp âm Spaceship nhẹ ở pitch 1.0x
            self.target_acc_gain = 0.16
            self.target_bright_gain = 0.05
            self.target_pitch_ratio = 1.0
        else:
            speed = float(new_speed)
            norm_speed = (speed - 1.0) / 19.0  # 0.0 tại 1 km/h -> 1.0 tại 20 km/h
            target_freq = MIN_FREQ + norm_speed * (MAX_FREQ - MIN_FREQ)
            self.target_pitch_ratio = target_freq / BASE_FREQ

            # Điều chỉnh độ mở của dải trung và dải cao (Dynamic Low-Pass / Brightness) theo tốc độ
            self.target_acc_gain = 0.24 + 0.50 * norm_speed
            self.target_bright_gain = 0.08 + 0.52 * (norm_speed ** 1.35)

            # Nếu xe đang đạp ga tăng tốc (dv/dt > 0), kích hoạt xung tải động cơ (Torque Load Boost)
            if delta_speed > 0:
                boost_step = min(0.35, 0.06 * float(delta_speed))
                self.accel_torque_boost = min(0.40, self.accel_torque_boost + boost_step)

    def _audio_callback(self, outdata, frames, time, status):
        # Làm mượt Gain (Background + Acceleration + Dynamic Brightness) và Pitch theo từng sample
        steps = np.arange(1, frames + 1, dtype=np.float64)
        decay = ((1.0 - self.smooth_alpha) ** steps).astype(np.float32)

        bg_gains = self.target_bg_gain + (self.current_bg_gain - self.target_bg_gain) * decay
        acc_gains = self.target_acc_gain + (self.current_acc_gain - self.target_acc_gain) * decay
        bright_gains = self.target_bright_gain + (self.current_bright_gain - self.target_bright_gain) * decay
        pitches = self.target_pitch_ratio + (self.current_pitch_ratio - self.target_pitch_ratio) * decay

        # Đường bao tải gia tốc tức thời (dv/dt Torque Load) giảm dần mượt mà khi giữ đều ga
        torque_env = (self.accel_torque_boost * (self.torque_decay ** steps)).astype(np.float32)
        self.accel_torque_boost = float(torque_env[-1])

        self.current_bg_gain = float(bg_gains[-1])
        self.current_acc_gain = float(acc_gains[-1])
        self.current_bright_gain = float(bright_gains[-1])
        self.current_pitch_ratio = float(pitches[-1])

        bg_source = self.bg_hum_samples
        mid_source = self.acc_mid_samples
        high_source = self.acc_high_samples
        n_len = min(len(bg_source), len(mid_source), len(high_source))

        # 1. LỚP NỀN (BACKGROUND HUM): Luôn giữ nguyên tần số gốc (bước nhảy = 1.0 sample) kể cả ở 0 km/h
        bg_idx = (int(self.bg_read_pointer) + np.arange(frames, dtype=np.int64)) % n_len
        self.bg_read_pointer = float((int(self.bg_read_pointer) + frames) % n_len)
        bg_layer = bg_source[bg_idx] * bg_gains

        # 2. LỚP TĂNG TỐC (ACCELERATION LAYER): Điều chỉnh tần số (pitch) & độ mở tần số cao theo tốc độ + tải ga
        pointers = (self.read_pointer + np.cumsum(pitches, dtype=np.float64)) % n_len
        self.read_pointer = float(pointers[-1])

        # Nội suy bậc 3 Catmull-Rom (4-point Cubic Interpolation) cho cả dải Mid và High
        idx1 = pointers.astype(np.int64)
        frac = (pointers - idx1).astype(np.float32)
        idx0 = (idx1 - 1) % n_len
        idx2 = (idx1 + 1) % n_len
        idx3 = (idx1 + 2) % n_len

        # Nội suy lớp Mid-Body
        m0, m1, m2, m3 = mid_source[idx0], mid_source[idx1], mid_source[idx2], mid_source[idx3]
        mc1 = 0.5 * (m2 - m0)
        mc2 = m0 - 2.5 * m1 + 2.0 * m2 - 0.5 * m3
        mc3 = 0.5 * (m3 - m0) + 1.5 * (m1 - m2)
        mid_interp = ((mc3 * frac + mc2) * frac + mc1) * frac + m1

        # Nội suy lớp High-Turbine
        h0, h1, h2, h3 = high_source[idx0], high_source[idx1], high_source[idx2], high_source[idx3]
        hc1 = 0.5 * (h2 - h0)
        hc2 = h0 - 2.5 * h1 + 2.0 * h2 - 0.5 * h3
        hc3 = 0.5 * (h3 - h0) + 1.5 * (h1 - h2)
        high_interp = ((hc3 * frac + hc2) * frac + hc1) * frac + h1

        # Kết hợp động lực học: Dải cao mở rộng theo tốc độ (bright_gains) + xung gia tốc (torque_env)
        acc_raw = mid_interp * acc_gains + high_interp * (bright_gains + torque_env)

        # 3. BỘ LỌC BẢO TỒN FORMANT THÂN XE CỐ ĐỊNH (Fixed Body Cavity Formant @ 215 Hz & 485 Hz)
        # Giữ cộng hưởng vật lý của khoang xe ở tần số cố định trong khi các hài âm bên trong tăng pitch
        f1_out, self.zi_f1 = signal.lfilter(self.b_f1, self.a_f1, acc_raw, zi=self.zi_f1)
        f2_out, self.zi_f2 = signal.lfilter(self.b_f2, self.a_f2, acc_raw, zi=self.zi_f2)
        acc_layer = (0.72 * acc_raw + 0.18 * f1_out + 0.10 * f2_out).astype(np.float32)

        # 4. BỘ GIỚI HẠN MỀM (SOFT-KNEE LIMITER): Tránh méo cắt ngọn (hard clipping)
        buffer = soft_knee_limit(bg_layer + acc_layer)

        outdata[:, 0] = buffer
        if frames < self.viz_buffer_size:
            self.viz_buffer[:-frames] = self.viz_buffer[frames:]
            self.viz_buffer[-frames:] = buffer
        else:
            self.viz_buffer[:] = buffer[-self.viz_buffer_size:]

    def get_viz_samples(self):
        return self.viz_buffer.copy()

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()


class AVASSimulatorGUI(QWidget):
    def __init__(self, audio_engine, available_wavs=None):
        super().__init__()
        self.engine = audio_engine
        self.available_wavs = list(available_wavs) if available_wavs else []

        # Nâng độ phân giải FFT lên 8192 điểm (~5.38 Hz/bin ở 44.1 kHz) theo chuẩn phân tích 1/3 Octave
        self.fft_size = 8192
        self.hann_window = np.hanning(self.fft_size)
        # Hệ số bù năng lượng cửa sổ Hann (Energy Correction = 1 / sqrt(3/8))
        self.window_rms_correction = np.sqrt(8.0 / 3.0)

        # Hằng số làm mượt thời gian IEC 61672 Fast (tau = 125 ms ở chu kỳ quét GUI 33 ms)
        self.fast_time_alpha = 1.0 - np.exp(-0.033 / 0.125)
        self.smoothed_fft_power = None
        self.smoothed_band_energy = np.zeros(len(OCTAVE_CENTER_FREQS), dtype=np.float64)

        self._update_fft_grid()
        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_visualizer)
        self.timer.start(33)
        QTimer.singleShot(50, self.engine.start)

    def _update_fft_grid(self):
        self.sample_rate = self.engine.sample_rate
        self.fft_freqs = np.fft.rfftfreq(self.fft_size, 1.0 / self.sample_rate)
        self.a_weighting = get_a_weighting_offsets(self.fft_freqs)
        self.smoothed_fft_power = np.zeros(len(self.fft_freqs), dtype=np.float64)
        self.smoothed_band_energy = np.zeros(len(OCTAVE_CENTER_FREQS), dtype=np.float64)

        self.octave_bounds = []
        for fc in OCTAVE_CENTER_FREQS:
            f_lower = fc * (2.0 ** (-1.0 / 6.0))
            f_upper = fc * (2.0 ** (1.0 / 6.0))
            idx = np.where((self.fft_freqs >= f_lower) & (self.fft_freqs < f_upper))[0]
            self.octave_bounds.append(idx)

    def _is_reverse_file(self):
        if not self.engine.current_wav_path:
            return False
        fname = os.path.basename(self.engine.current_wav_path).lower()
        return ("backward" in fname) or ("reverse" in fname)

    def _format_wav_label(self, wav_path):
        try:
            rel = os.path.relpath(wav_path, SCRIPT_DIR)
            if not rel.startswith(".."):
                return rel.replace("\\", "/")
        except ValueError:
            pass
        return wav_path

    def init_ui(self):
        self.setWindowTitle("UN ECE R138 AVAS Compliance Analyzer - (0 - 20 km/h)")
        self.resize(1060, 840)

        main_layout = QVBoxLayout()

        # --- BẢNG ĐIỀU KHIỂN TỐC ĐỘ & IMPORT FILE ÂM THANH ---
        group = QGroupBox("Mô phỏng tốc độ xe (CAN Bus Speed) & Nguồn âm thanh AVAS (.WAV)")
        group_layout = QVBoxLayout()

        # Hàng chọn / Import file WAV
        source_layout = QHBoxLayout()
        source_label = QLabel("File âm thanh (.wav):", self)
        source_label.setStyleSheet("font-size: 14px; font-weight: bold;")

        self.combo_wav = QComboBox(self)
        self.combo_wav.setStyleSheet("font-size: 14px; padding: 4px;")
        for path in self.available_wavs:
            self.combo_wav.addItem(self._format_wav_label(path), path)

        if self.engine.current_wav_path:
            idx = self.combo_wav.findData(self.engine.current_wav_path)
            if idx >= 0:
                self.combo_wav.setCurrentIndex(idx)

        self.combo_wav.currentIndexChanged.connect(self.on_wav_combo_changed)

        self.btn_import = QPushButton("📂 Import File .WAV...", self)
        self.btn_import.setStyleSheet(
            "font-size: 14px; font-weight: bold; padding: 5px 12px; "
            "background-color: #1E88E5; color: white; border-radius: 4px;"
        )
        self.btn_import.clicked.connect(self.on_import_wav_clicked)

        mode_label = QLabel("Chế độ DSP:", self)
        mode_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.combo_mode = QComboBox(self)
        self.combo_mode.setStyleSheet("font-size: 14px; padding: 4px;")
        self.combo_mode.addItems([
            "Âm thanh gốc (Auto IMD De-Spiked)",
            "Làm mượt & Khử nhiễu DSP (Smooth DSP)"
        ])
        self.combo_mode.currentIndexChanged.connect(self.engine.set_sound_mode)

        source_layout.addWidget(source_label)
        source_layout.addWidget(self.combo_wav, 1)
        source_layout.addWidget(self.btn_import)
        source_layout.addSpacing(10)
        source_layout.addWidget(mode_label)
        source_layout.addWidget(self.combo_mode)
        group_layout.addLayout(source_layout)

        top_info_layout = QHBoxLayout()
        self.label_speed = QLabel("Tốc độ xe: 0 km/h", self)
        self.label_speed.setStyleSheet("font-size: 20px; font-weight: bold; color: #1E88E5;")

        self.label_freq = QLabel(f"Trạng thái: IDLE (0 km/h) | Hum nền: {BASE_FREQ:.0f} Hz", self)
        self.label_freq.setStyleSheet("font-size: 16px; color: #FB8C00; font-weight: bold;")

        self.label_spl = QLabel("Overall SPL: -- dB(A)", self)
        self.label_spl.setStyleSheet("font-size: 20px; font-weight: bold; color: #4CAF50;")

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

        # Thanh trạng thái đánh giá chuẩn UN ECE R138 trực tiếp (Table 3 + Two-Band Rule + Pitch Shift + Max 75 dB(A))
        r138_status_layout = QHBoxLayout()
        self.label_r138_verdict = QLabel("UN ECE R138: ĐANG ĐO...", self)
        self.label_r138_verdict.setStyleSheet(
            "font-size: 13px; font-weight: bold; padding: 4px 8px; "
            "background-color: #263238; color: #FFFFFF; border-radius: 4px;"
        )
        self.label_r138_bands = QLabel("Two-Band Rule (Bảng 3): --/16 dải đạt", self)
        self.label_r138_bands.setStyleSheet("font-size: 13px; font-weight: bold; color: #37474F;")

        self.label_r138_shift = QLabel(
            f"Độ dịch tần (Mục 6.2.3): {AVG_PITCH_SHIFT_PCT_PER_KMH:.2f}%/km/h (Yêu cầu >= 0.8%/km/h) [PASS]",
            self
        )
        self.label_r138_shift.setStyleSheet("font-size: 13px; font-weight: bold; color: #2E7D32;")

        r138_status_layout.addWidget(self.label_r138_verdict)
        r138_status_layout.addSpacing(8)
        r138_status_layout.addWidget(self.label_r138_bands)
        r138_status_layout.addStretch()
        r138_status_layout.addWidget(self.label_r138_shift)

        group_layout.addLayout(top_info_layout)
        group_layout.addWidget(self.slider)
        group_layout.addLayout(r138_status_layout)
        group.setLayout(group_layout)
        main_layout.addWidget(group)

        # --- KHU VỰC ĐỒ THỊ PYQTGRAPH ---
        pg.setConfigOptions(antialias=True)
        
        # 1. Đồ thị FFT Spectrum (IEC 61672 Fast Time-Weighted, 8192-point FFT)
        self.fft_plot = pg.PlotWidget(title="<b>FFT Spectrum (A-Weighted, IEC 61672 Fast 125ms, 8192-pt)</b>")
        self.fft_plot.setLogMode(x=True, y=False)
        self.fft_plot.setXRange(np.log10(20), np.log10(20000))
        self.fft_plot.setYRange(0, 90)
        self.fft_plot.setLabel('bottom', 'Tần số', units='Hz')
        self.fft_plot.setLabel('left', 'Cường độ', units='dB(A)')
        self.fft_plot.showGrid(x=True, y=True, alpha=0.3)
        self.fft_curve = self.fft_plot.plot(pen=pg.mkPen('#0288D1', width=2))
        
        main_layout.addWidget(self.fft_plot)

        # 2. Đồ thị RTA 1/3 Octave
        self.rta_plot = pg.PlotWidget(
            title="<b>1/3 Octave Band RTA (UN ECE R138 Table 3: Xanh lá = Đạt ngưỡng dải tần | Đỏ đứt = Sàn tối thiểu)</b>"
        )
        self.rta_plot.setYRange(0, 85)
        self.rta_plot.setLabel('bottom', 'Dải tần số trung tâm 1/3 Octave (Hz)')
        self.rta_plot.setLabel('left', 'Mức áp suất âm', units='dB(A)')
        self.rta_plot.showGrid(x=False, y=True, alpha=0.3)

        x_ticks = [(i, str(int(fc))) for i, fc in enumerate(OCTAVE_CENTER_FREQS)]
        self.rta_plot.getAxis('bottom').setTicks([x_ticks])

        self.bar_item = pg.BarGraphItem(
            x=np.arange(len(OCTAVE_CENTER_FREQS)),
            height=np.zeros(16),
            width=0.6,
            brushes=[pg.mkBrush('#26A69A') for _ in range(16)]
        )
        self.rta_plot.addItem(self.bar_item)

        self.limit_curve = self.rta_plot.plot(
            x=np.arange(len(OCTAVE_CENTER_FREQS)), 
            y=R138_MIN_DBA_10KMH, 
            pen=pg.mkPen('#E53935', width=2, style=Qt.DashLine),
            name="Ngưỡng tối thiểu Bảng 3 ECE R138"
        )

        main_layout.addWidget(self.rta_plot)
        self.setLayout(main_layout)

    def on_wav_combo_changed(self, index):
        if index < 0:
            return
        wav_path = self.combo_wav.itemData(index)
        if wav_path and os.path.exists(wav_path):
            try:
                self.engine.load_wav_file(wav_path)
                self._update_fft_grid()
            except Exception as exc:
                QMessageBox.critical(self, "Lỗi đọc file WAV", f"Không thể nạp file:\n{wav_path}\n\nChi tiết: {exc}")

    def on_import_wav_clicked(self):
        start_dir = os.path.dirname(self.engine.current_wav_path) if self.engine.current_wav_path else SCRIPT_DIR
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn file âm thanh AVAS (.wav)",
            start_dir,
            "WAV Audio Files (*.wav);;All Files (*)"
        )
        if not file_path:
            return

        abs_path = os.path.abspath(file_path)
        try:
            self.engine.load_wav_file(abs_path)
            self._update_fft_grid()
        except Exception as exc:
            QMessageBox.critical(self, "Lỗi nạp file WAV", f"Không thể nạp file:\n{abs_path}\n\nChi tiết: {exc}")
            return

        existing_idx = self.combo_wav.findData(abs_path)
        self.combo_wav.blockSignals(True)
        if existing_idx < 0:
            self.combo_wav.addItem(self._format_wav_label(abs_path), abs_path)
            self.combo_wav.setCurrentIndex(self.combo_wav.count() - 1)
        else:
            self.combo_wav.setCurrentIndex(existing_idx)
        self.combo_wav.blockSignals(False)

    def on_speed_changed(self, value_int):
        speed_kmh = int(value_int)

        self.label_speed.setText(f"Tốc độ xe: {speed_kmh} km/h")
        
        if speed_kmh == 0:
            self.label_freq.setText(f"Trạng thái: IDLE (0 km/h) | Hum nền: {BASE_FREQ:.0f} Hz")
            self.label_freq.setStyleSheet("font-size: 16px; color: #FB8C00; font-weight: bold;")
        else:
            target_freq = MIN_FREQ + ((speed_kmh - 1) / 19.0) * (MAX_FREQ - MIN_FREQ)
            self.label_freq.setText(f"Hum nền: {BASE_FREQ:.0f} Hz | Tăng tốc (f0): {target_freq:.1f} Hz")
            self.label_freq.setStyleSheet("font-size: 16px; color: #388E3C; font-weight: bold;")

        # Cập nhật đường giới hạn Bảng 3 theo tốc độ xe (10 km/h -> 20 km/h)
        self.limit_curve.setData(
            np.arange(len(OCTAVE_CENTER_FREQS)),
            get_r138_band_limits_for_speed(speed_kmh)
        )
        self.engine.set_speed(speed_kmh)

    def update_visualizer(self):
        samples = self.engine.get_viz_samples()[-self.fft_size:]
        if len(samples) < self.fft_size:
            return

        # 1. Tính FFT 8192 điểm với bù năng lượng cửa sổ Hann (Hann Window Energy Correction)
        windowed_samples = samples * self.hann_window
        fft_complex = np.fft.rfft(windowed_samples)
        inst_mag = (np.abs(fft_complex) / (self.fft_size / 2)) * self.window_rms_correction
        inst_power = inst_mag ** 2

        # 2. Tích phân thời gian hàm mũ chuẩn IEC 61672 Fast (tau = 125 ms)
        alpha = self.fast_time_alpha
        if np.all(self.smoothed_fft_power == 0):
            self.smoothed_fft_power = inst_power.copy()
        else:
            self.smoothed_fft_power = alpha * inst_power + (1.0 - alpha) * self.smoothed_fft_power

        smoothed_mag = np.sqrt(np.maximum(self.smoothed_fft_power, 1e-16))
        spl_fft = 20.0 * np.log10(smoothed_mag + 1e-8) + SPL_CALIBRATION_DB + self.a_weighting
        self.fft_curve.setData(self.fft_freqs[1:], spl_fft[1:])

        # 3. Tính mức áp suất âm từng dải 1/3 Octave (IEC 61260 / UN ECE R138 Table 3)
        octave_spl = np.zeros(len(OCTAVE_CENTER_FREQS), dtype=np.float32)
        total_energy = 1e-12

        for i, indices in enumerate(self.octave_bounds):
            if len(indices) > 0:
                inst_band_energy = float(np.sum(10.0 ** (spl_fft[indices] / 10.0)))
                if self.smoothed_band_energy[i] <= 0.0:
                    self.smoothed_band_energy[i] = inst_band_energy
                else:
                    self.smoothed_band_energy[i] = (
                        alpha * inst_band_energy + (1.0 - alpha) * self.smoothed_band_energy[i]
                    )
                octave_spl[i] = 10.0 * np.log10(self.smoothed_band_energy[i] + 1e-12)
                total_energy += self.smoothed_band_energy[i]

        overall_spl = float(10.0 * np.log10(total_energy))

        # 4. Đánh giá tiêu chuẩn UN ECE R138 trực tiếp và tô màu các cột RTA đạt chuẩn
        speed_kmh = self.engine.current_speed_int
        is_rev = self._is_reverse_file()
        comp = evaluate_r138_compliance(octave_spl, overall_spl, speed_kmh, is_reverse=is_rev)

        brushes = [
            pg.mkBrush('#2ECC71') if passed else pg.mkBrush('#546E7A')
            for passed in comp["passed_mask"]
        ]
        self.bar_item.setOpts(height=octave_spl, brushes=brushes)

        self.label_spl.setText(
            f"Overall SPL: {overall_spl:.1f} dB(A) "
            f"[{comp['min_overall']:.0f}-{R138_MAX_OVERALL_SPL:.0f} dB(A)]"
        )
        if comp["overall_ok"]:
            self.label_spl.setStyleSheet("font-size: 19px; font-weight: bold; color: #2E7D32;")
        else:
            self.label_spl.setStyleSheet("font-size: 19px; font-weight: bold; color: #D32F2F;")

        self.label_r138_bands.setText(
            f"Two-Band Rule (Bảng 3): {comp['num_passed_bands']}/16 dải đạt "
            f"({comp['low_mid_passed']} dải <= 1600 Hz) "
            f"[{'PASS' if comp['two_band_ok'] else 'FAIL'}]"
        )
        self.label_r138_bands.setStyleSheet(
            f"font-size: 13px; font-weight: bold; color: {'#2E7D32' if comp['two_band_ok'] else '#D32F2F'};"
        )

        if speed_kmh == 0:
            self.label_r138_verdict.setText(
                f"UN ECE R138 (IDLE 0 km/h): HUM NỀN HOẠT ĐỘNG ({overall_spl:.1f} dB(A) <= 75 dB(A))"
            )
            self.label_r138_verdict.setStyleSheet(
                "font-size: 13px; font-weight: bold; padding: 4px 8px; "
                "background-color: #EF6C00; color: #FFFFFF; border-radius: 4px;"
            )
        elif comp["all_pass"]:
            mode_str = "LÙI (REVERSE)" if is_rev else f"TIẾN ({speed_kmh} km/h)"
            self.label_r138_verdict.setText(f"UN ECE R138 [{mode_str}]: ĐẠT CHUẨN (PASS)")
            self.label_r138_verdict.setStyleSheet(
                "font-size: 13px; font-weight: bold; padding: 4px 8px; "
                "background-color: #2E7D32; color: #FFFFFF; border-radius: 4px;"
            )
        else:
            self.label_r138_verdict.setText(f"UN ECE R138 ({speed_kmh} km/h): CHƯA ĐẠT (FAIL)")
            self.label_r138_verdict.setStyleSheet(
                "font-size: 13px; font-weight: bold; padding: 4px 8px; "
                "background-color: #C62828; color: #FFFFFF; border-radius: 4px;"
            )

    def closeEvent(self, event):
        self.timer.stop()
        self.engine.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Quét các file .wav có sẵn trong thư mục (Profile_Audi, Profile_Skoda, ...)
    discovered_wavs = discover_wav_files(SCRIPT_DIR)

    # Cho phép người dùng chọn file .wav ngay khi khởi động (hoặc bấm Cancel để chọn sau trong GUI)
    selected_wav, _ = QFileDialog.getOpenFileName(
        None,
        "Chọn file âm thanh AVAS (.wav) để mô phỏng",
        SCRIPT_DIR,
        "WAV Audio Files (*.wav);;All Files (*)"
    )

    if selected_wav:
        selected_wav = os.path.abspath(selected_wav)
        if selected_wav not in discovered_wavs:
            discovered_wavs.insert(0, selected_wav)
    elif discovered_wavs:
        selected_wav = discovered_wavs[0]
    else:
        selected_wav = None

    engine = AVASAudioEngine(selected_wav)
    gui = AVASSimulatorGUI(engine, discovered_wavs)
    gui.show()

    sys.exit(app.exec_())
