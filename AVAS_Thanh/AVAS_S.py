import sys
import os
import io
import collections
import numpy as np
from scipy.io import wavfile
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

# --- THÔNG SỐ CẤU HÌNH AVAS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_FREQ = 160.0    # Tần số gốc của file WAV nền (Hz)
MIN_FREQ = 160.0     # Tần số tại 1 km/h (Âm trầm mượt, đầm xe)
MAX_FREQ = 256.0     # Tần số tại 20 km/h (Tương ứng tăng +60% pitch: 160 -> 256 Hz khi tăng tốc)

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


def discover_wav_files(root_dir):
    """Quét toàn bộ các file .wav hiện có trong thư mục gốc và các thư mục Profile con"""
    wav_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in sorted(filenames):
            if fname.lower().endswith(".wav"):
                wav_files.append(os.path.join(dirpath, fname))
    return wav_files


class AVASAudioEngine:
    def __init__(self, initial_wav_path=None):
        self.current_wav_path = None
        self.sample_rate = 44100
        self.sound_mode = 0  # 0: Original WAV, 1: Smooth DSP Filter

        # Khởi tạo buffer mặc định (im lặng) nếu chưa nạp file WAV
        default_len = self.sample_rate * 6
        self.raw_base_samples = np.zeros(default_len, dtype=np.float32)
        self.dsp_base_samples = np.zeros(default_len, dtype=np.float32)
        self.bg_hum_samples = np.zeros(default_len, dtype=np.float32)
        self.acc_turbine_samples = np.zeros(default_len, dtype=np.float32)
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
        # Tiếng Hum nền (bg_gain = 0.45) luôn duy trì ngay cả ở trạng thái Idle 0 km/h
        self.current_bg_gain = 0.45
        self.target_bg_gain = 0.45
        self.current_acc_gain = 0.0
        self.target_acc_gain = 0.0
        self.current_pitch_ratio = 1.0
        self.target_pitch_ratio = 1.0
        self.smooth_alpha = 1.0 - np.exp(-1.0 / (0.045 * self.sample_rate))  # Hằng số thời gian ~45ms
        
        self.viz_buffer_size = 8192
        self.viz_buffer = np.zeros(self.viz_buffer_size, dtype=np.float32)
        self.stream = None

        if initial_wav_path and os.path.exists(initial_wav_path):
            self.load_wav_file(initial_wav_path)
        else:
            self._init_stream()

    def _init_stream(self):
        was_active = False
        if self.stream is not None:
            was_active = self.stream.active
            self.stream.stop()
            self.stream.close()

        self.smooth_alpha = 1.0 - np.exp(-1.0 / (0.045 * self.sample_rate))
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

    def load_wav_file(self, wav_path):
        """Nạp file .wav do người dùng chọn/import và tách lớp Background Hum + Acceleration Layer"""
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

        raw_f32 = raw_samples.astype(np.float32)
        dsp_f32 = self._preprocess_smooth_audio(raw_samples)

        self.raw_base_samples = raw_f32
        self.dsp_base_samples = dsp_f32
        self.bg_read_pointer = 0.0
        self.read_pointer = 0.0
        self.set_sound_mode(self.sound_mode)

        if sr_changed:
            self._init_stream()

    def _build_layers(self, active_source):
        """
        Tách tín hiệu gốc thành 2 thành phần để phần mềm điều khiển:
        - bg_hum_samples      : Lớp âm trầm nền (Background Hum, < 280 Hz) luôn phát liên tục kể cả ở 0 km/h (Idle).
        - acc_turbine_samples : Lớp hài âm tăng tốc (Acceleration/Turbine Layer) sẽ được phần mềm
                                dịch tần số (pitch-shift) và tăng cường độ khi xe tăng tốc (1 -> 20 km/h).
        """
        n = len(active_source)
        if n == 0:
            z = np.zeros(1, dtype=np.float32)
            return z, z

        spec = np.fft.rfft(active_source)
        freqs = np.fft.rfftfreq(n, 1.0 / self.sample_rate)

        # Lớp Background Hum: giữ vững dải trầm - trung trầm (< 280 Hz) mượt và phẳng ở nền (kể cả khi Idle 0 km/h)
        bg_LP = 1.0 / (1.0 + (freqs / 260.0) ** 4.0)
        bg_sig = np.fft.irfft(spec * bg_LP, n=n)

        # Lớp Acceleration: tập trung vào dải hài âm động cơ/turbine (>= 150 Hz) để quét pitch khi tăng tốc
        acc_HP = 1.0 / (1.0 + (150.0 / np.maximum(freqs, 1.0)) ** 3.0)
        acc_sig = np.fft.irfft(spec * acc_HP, n=n)

        return bg_sig.astype(np.float32), acc_sig.astype(np.float32)

    def set_sound_mode(self, mode_index):
        """Chuyển đổi chế độ xử lý âm thanh mượt mà khi đang chạy (0: WAV gốc, 1: Khử nhiễu DSP)"""
        self.sound_mode = int(mode_index)
        if self.sound_mode == 1:
            self.base_samples = self.dsp_base_samples
        else:
            self.base_samples = self.raw_base_samples
        self.sample_len = max(len(self.base_samples), 1)
        self.bg_hum_samples, self.acc_turbine_samples = self._build_layers(self.base_samples)
        self.bg_read_pointer = self.bg_read_pointer % self.sample_len
        self.read_pointer = self.read_pointer % self.sample_len

    def _preprocess_smooth_audio(self, raw_samples):
        """
        Phương pháp DSP làm mượt âm thanh trực tiếp từ dữ liệu WAV đầu vào (không sửa file .wav):
        1. Spectral Gating (FFT): Loại bỏ toàn bộ dải tần nhiễu biến điệu bậc cao (IMD sand/noise)
           sinh ra do bão hòa phi tuyến trong file WAV gốc.
        2. Harmonic Phase-Locking & Schroeder Phase: Đồng bộ các tần số lệch pha (như 250Hz, 315Hz)
           về lưới hài âm chuẩn (bội số của BASE_FREQ/2 = 80Hz) để triệt tiêu hiện tượng phách (beating/roughness).
        3. Warm Acoustic Low-Pass Rolloff: Làm dịu dải tần cao (>350Hz) giúp âm thanh êm, ấm và mượt.
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
        self.current_speed_int = int(speed_kmh_int)
        # Tiếng Hum nền (0.45) luôn giữ nguyên trong suốt mọi trạng thái, kể cả khi đứng yên (Idle 0 km/h)
        self.target_bg_gain = 0.45

        if self.current_speed_int <= 0:
            # Tại 0 km/h (Idle): Chỉ phát tiếng Hum nền ổn định, tắt dần lớp âm tăng tốc
            self.target_acc_gain = 0.0
            self.target_pitch_ratio = 1.0
        else:
            speed = float(self.current_speed_int)
            norm_speed = (speed - 1.0) / 19.0  # 0.0 tại 1 km/h -> 1.0 tại 20 km/h
            target_freq = MIN_FREQ + norm_speed * (MAX_FREQ - MIN_FREQ)
            self.target_pitch_ratio = target_freq / BASE_FREQ

            # Khi xe tăng tốc (1 -> 20 km/h), phần mềm đẩy lớp tăng tốc (Acceleration Layer) lên trên nền Hum
            self.target_acc_gain = 0.25 + 0.55 * norm_speed

    def _audio_callback(self, outdata, frames, time, status):
        # Làm mượt Gain (Background + Acceleration) và Pitch theo từng sample
        steps = np.arange(1, frames + 1, dtype=np.float64)
        decay = ((1.0 - self.smooth_alpha) ** steps).astype(np.float32)

        bg_gains = self.target_bg_gain + (self.current_bg_gain - self.target_bg_gain) * decay
        acc_gains = self.target_acc_gain + (self.current_acc_gain - self.target_acc_gain) * decay
        pitches = self.target_pitch_ratio + (self.current_pitch_ratio - self.target_pitch_ratio) * decay

        self.current_bg_gain = float(bg_gains[-1])
        self.current_acc_gain = float(acc_gains[-1])
        self.current_pitch_ratio = float(pitches[-1])

        bg_source = self.bg_hum_samples
        acc_source = self.acc_turbine_samples
        n_len = len(acc_source)

        # 1. LỚP NỀN (BACKGROUND HUM): Luôn giữ nguyên tần số gốc (bước nhảy = 1.0 sample) kể cả ở 0 km/h
        bg_idx = (int(self.bg_read_pointer) + np.arange(frames, dtype=np.int64)) % n_len
        self.bg_read_pointer = float((int(self.bg_read_pointer) + frames) % n_len)
        bg_layer = bg_source[bg_idx] * bg_gains

        # 2. LỚP TĂNG TỐC (ACCELERATION LAYER): Điều chỉnh tần số (pitch) & biên độ theo tốc độ xe
        pointers = (self.read_pointer + np.cumsum(pitches, dtype=np.float64)) % n_len
        self.read_pointer = float(pointers[-1])

        # Nội suy bậc 3 Catmull-Rom (4-point Cubic Interpolation) giúp âm thanh mượt tuyệt đối
        idx1 = pointers.astype(np.int64)
        frac = (pointers - idx1).astype(np.float32)
        idx0 = (idx1 - 1) % n_len
        idx2 = (idx1 + 1) % n_len
        idx3 = (idx1 + 2) % n_len

        y0 = acc_source[idx0]
        y1 = acc_source[idx1]
        y2 = acc_source[idx2]
        y3 = acc_source[idx3]

        c1 = 0.5 * (y2 - y0)
        c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3
        c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2)

        acc_interpolated = ((c3 * frac + c2) * frac + c1) * frac + y1
        acc_layer = acc_interpolated * acc_gains

        # Tổng hợp 2 lớp: Tiếng Hum trầm luôn giữ ở nền (cả lúc Idle 0 km/h) + Tiếng vút tăng tốc
        buffer = np.clip(bg_layer + acc_layer, -0.98, 0.98).astype(np.float32)

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
        self.fft_size = 4096
        self.hann_window = np.hanning(self.fft_size)
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

        self.octave_bounds = []
        for fc in OCTAVE_CENTER_FREQS:
            f_lower = fc * (2.0 ** (-1.0 / 6.0))
            f_upper = fc * (2.0 ** (1.0 / 6.0))
            idx = np.where((self.fft_freqs >= f_lower) & (self.fft_freqs < f_upper))[0]
            self.octave_bounds.append(idx)

    def _format_wav_label(self, wav_path):
        try:
            rel = os.path.relpath(wav_path, SCRIPT_DIR)
            if not rel.startswith(".."):
                return rel.replace("\\", "/")
        except ValueError:
            pass
        return wav_path

    def init_ui(self):
        self.setWindowTitle("ECE R138 AVAS Analyzer - (0 - 20 km/h Integer Mode)")
        self.resize(1020, 800)

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
            "Âm thanh gốc (Original WAV)",
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

        self.engine.set_speed(speed_kmh)

    def update_visualizer(self):
        samples = self.engine.get_viz_samples()[-self.fft_size:]
        if len(samples) < self.fft_size:
            return

        # Xử lý FFT & RTA liên tục (kể cả khi Idle 0 km/h để hiển thị phổ của tiếng Hum nền)
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
