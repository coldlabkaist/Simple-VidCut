import sys, os, shutil, subprocess, math
from typing import Optional, List
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QEvent, QTimer
from PyQt5.QtGui import QImage, QPixmap, QIntValidator, QIcon, QColor, QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QListWidget, QPushButton, QLabel, QSlider, QFileDialog, QGroupBox, QLineEdit,
    QDoubleSpinBox, QSpinBox, QComboBox, QMessageBox, QSizePolicy, QCheckBox, QProgressBar,
    QRadioButton, QStyle, QDialog, QDialogButtonBox, QAbstractItemView, QShortcut
)
import cv2
import numpy as np


def _resource_base_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _find_app_icon_path() -> str:
    cand = os.path.join(_resource_base_dir(), "CoLD_icon.png")
    return cand if os.path.isfile(cand) else ""


# ---------------------------- Video worker thread ----------------------------
class VideoThread(QThread):
    frameReady = pyqtSignal(QImage, int)   # image, frame_index
    finished   = pyqtSignal()

    def __init__(self, path: str):
        super().__init__()
        self.path = path
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps = 30.0
        self.total = 0
        self.width = 0
        self.height = 0
        self.playing = False
        self.speed = 1.0
        self.contrast = 1.0
        self.brightness = 0.0
        self.saturation = 1.0
        self._seek_to: Optional[int] = None
        self.current_idx = 0
        self._stop = False

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap or not self.cap.isOpened():
            return False
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 0
        self.fps = float(fps) if fps > 1e-3 else 30.0
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        self.current_idx = 0
        return True

    def run(self):
        if not self.cap:
            ok = self.open()
            if not ok:
                self.finished.emit()
                return
        frame_interval_ms = lambda: max(1, int(1000.0 / self.fps / max(0.1, self.speed)))
        while not self._stop:
            if self._seek_to is not None:
                idx = max(0, min(self._seek_to, max(0, self.total - 1)))
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                self.current_idx = idx
                self._seek_to = None
                if not self.playing:
                    ret, frame = self.cap.read()
                    if ret:
                        self._emit_frame(frame)
                    self.msleep(1)
            if self.playing:
                ret, frame = self.cap.read()
                if not ret:
                    self.playing = False
                    self.finished.emit()
                    self.msleep(20)
                    continue
                self._emit_frame(frame)
                self.current_idx += 1
                self.msleep(frame_interval_ms())
            else:
                # idle wait
                self.msleep(15)

    def _apply_adjustments(self, frame):
        c = float(self.contrast)
        b = float(self.brightness)
        s = float(self.saturation)
        if abs(c - 1.0) <= 1e-6 and abs(b) <= 1e-6 and abs(s - 1.0) <= 1e-6:
            return frame

        # Match ffmpeg eq behavior more closely: operate in luminance/chroma space.
        ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb).astype(np.float32) / 255.0
        y = ycc[..., 0]
        cr = ycc[..., 1]
        cb = ycc[..., 2]

        y = (y - 0.5) * c + 0.5 + b
        cr = (cr - 0.5) * s + 0.5
        cb = (cb - 0.5) * s + 0.5

        ycc[..., 0] = np.clip(y, 0.0, 1.0)
        ycc[..., 1] = np.clip(cr, 0.0, 1.0)
        ycc[..., 2] = np.clip(cb, 0.0, 1.0)

        return cv2.cvtColor((ycc * 255.0).astype(np.uint8), cv2.COLOR_YCrCb2BGR)

    def _emit_frame(self, frame):
        adj = self._apply_adjustments(frame)
        h, w = adj.shape[:2]
        img = cv2.cvtColor(adj, cv2.COLOR_BGR2RGB)
        qimg = QImage(img.data, w, h, w * 3, QImage.Format_RGB888)
        self.frameReady.emit(qimg.copy(), self.current_idx)

    @pyqtSlot()
    def play(self):
        self.playing = True

    @pyqtSlot()
    def pause(self):
        self.playing = False

    @pyqtSlot(float)
    def set_speed(self, s: float):
        self.speed = max(0.1, float(s))

    @pyqtSlot(float, float, float)
    def set_adjustments(self, contrast: float, brightness: float, saturation: float):
        self.contrast = max(0.0, float(contrast))
        self.brightness = max(-1.0, min(1.0, float(brightness)))
        self.saturation = max(0.0, float(saturation))

    @pyqtSlot(int)
    def seek(self, frame_idx: int):
        self._seek_to = int(frame_idx)

    @pyqtSlot()
    def stop(self):
        self._stop = True
        self.playing = False


class ClickJumpSlider(QSlider):
    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            if self.orientation() == Qt.Horizontal:
                pos = ev.x()
                span = max(1, self.width() - 1)
                val = QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), pos, span, upsideDown=False)
            else:
                pos = max(0, self.height() - 1 - ev.y())
                span = max(1, self.height() - 1)
                val = QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), pos, span, upsideDown=False)
            self.setValue(val)
            ev.accept()
        super().mousePressEvent(ev)


class ExportThread(QThread):
    progressChanged = pyqtSignal(int)
    finishedOk = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, cmd: List[str], out_path: str, duration_sec: float):
        super().__init__()
        self.cmd = list(cmd)
        self.out_path = out_path
        self.duration_us = max(1, int(max(0.001, float(duration_sec)) * 1_000_000.0))

    def run(self):
        err_tail: List[str] = []
        try:
            proc = subprocess.Popen(
                self.cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                universal_newlines=True,
                bufsize=1,
            )
            self.progressChanged.emit(0)
            if proc.stderr:
                for raw in proc.stderr:
                    line = (raw or "").strip()
                    if not line:
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        if k == "out_time_ms":
                            try:
                                cur = max(0, int(v))
                                pct = max(0, min(99, int(cur * 100 / self.duration_us)))
                                self.progressChanged.emit(pct)
                            except Exception:
                                pass
                        elif k == "progress" and v == "end":
                            self.progressChanged.emit(100)
                    else:
                        err_tail.append(line)
                        if len(err_tail) > 120:
                            err_tail = err_tail[-120:]
            rc = proc.wait()
            if rc != 0:
                msg = "\n".join(err_tail[-60:]) if err_tail else f"ffmpeg exited with code {rc}"
                self.failed.emit(msg)
                return
            self.progressChanged.emit(100)
            self.finishedOk.emit(self.out_path)
        except Exception as e:
            self.failed.emit(f"Failed to run ffmpeg: {e}")


class BatchExportThread(QThread):
    progressChanged = pyqtSignal(int)
    itemChanged = pyqtSignal(int, int, str)  # current_index(1-based), total, label
    done = pyqtSignal(str, bool)  # summary, has_errors

    def __init__(self, tasks: List[dict]):
        super().__init__()
        self.tasks = list(tasks)

    def _run_one(self, cmd: List[str], duration_us: int):
        err_tail: List[str] = []
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                universal_newlines=True,
                bufsize=1,
            )
            self.progressChanged.emit(0)
            if proc.stderr:
                for raw in proc.stderr:
                    line = (raw or "").strip()
                    if not line:
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        if k == "out_time_ms":
                            try:
                                cur = max(0, int(v))
                                pct = max(0, min(99, int(cur * 100 / max(1, duration_us))))
                                self.progressChanged.emit(pct)
                            except Exception:
                                pass
                        elif k == "progress" and v == "end":
                            self.progressChanged.emit(100)
                    else:
                        err_tail.append(line)
                        if len(err_tail) > 120:
                            err_tail = err_tail[-120:]
            rc = proc.wait()
            if rc != 0:
                msg = "\n".join(err_tail[-60:]) if err_tail else f"ffmpeg exited with code {rc}"
                return False, msg
            self.progressChanged.emit(100)
            return True, ""
        except Exception as e:
            return False, f"Failed to run ffmpeg: {e}"

    def run(self):
        total = len(self.tasks)
        if total <= 0:
            self.done.emit("No batch tasks to run.", True)
            return

        failures: List[str] = []
        for i, task in enumerate(self.tasks, start=1):
            label = task.get("label", f"item {i}")
            self.itemChanged.emit(i, total, label)
            ok, err = self._run_one(task["cmd"], int(task["duration_us"]))
            if not ok:
                failures.append(f"[{label}] {err}")

        if failures:
            summary = (
                f"Batch export completed with errors: {total - len(failures)}/{total} succeeded.\n\n"
                + "\n\n".join(failures[:8])
            )
            self.done.emit(summary, True)
            return

        self.done.emit(f"Batch export completed: {total}/{total} succeeded.", False)


# ------------------------------ Main Window ------------------------------
class Cutter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simple VidCut")
        icon_path = _find_app_icon_path()
        if icon_path:
            self.setWindowIcon(QIcon(icon_path))
        self.setMinimumSize(1100, 720)
        self.video_folder = ""
        self.video_path: Optional[str] = None
        self.fps = 30.0
        self.total_frames = 0
        self.video_width = 0
        self.video_height = 0
        self.current_frame = 0
        self.thread: Optional[VideoThread] = None
        self.is_playing = False
        self.scrub_was_playing = False
        self.duration_warning_text = "Warning: The requested duration may not be obtained, resulting in shorter image length"
        self.default_contrast_ui = 100
        self.default_brightness_ui = 0
        self.default_saturation_ui = 100
        self.loaded_video_name: Optional[str] = None
        self.export_thread: Optional[ExportThread] = None
        self.batch_export_thread: Optional[BatchExportThread] = None

        # ---------- UI ----------
        root = QWidget(self); self.setCentralWidget(root)
        G = QGridLayout(root); G.setContentsMargins(8,8,8,8); G.setSpacing(8)

        self.video_label = QLabel("No video"); self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setStyleSheet("background:#111; color:#bbb;")
        # (row=0, col=0) 2사분면
        G.addWidget(self.video_label, 0, 0)

        # playback panel
        play_group = QGroupBox("Playback")
        gp = QGridLayout(play_group)

        self.btn_play = QPushButton("Play"); self.btn_play.setEnabled(False)
        self.slider = ClickJumpSlider(Qt.Horizontal); self.slider.setEnabled(False)
        self.lbl_frame = QLabel("Frame: 0 / 0")
        self.lbl_time  = QLabel("Time: 00:00.000 / 00:00.000")
        self.speed = QDoubleSpinBox(); self.speed.setRange(0.25, 3.0); self.speed.setSingleStep(0.25); self.speed.setValue(1.0)
        self.speed.setSuffix("×")

        gp.addWidget(self.slider, 0, 0, 1, 6)
        gp.addWidget(QLabel("Speed:"), 1, 0)
        gp.addWidget(self.speed, 1, 1)
        gp.addWidget(self.btn_play, 1, 2, 1, 2)
        gp.addWidget(self.lbl_frame, 1, 4)
        gp.addWidget(self.lbl_time,  1, 5)
        gp.setColumnStretch(2, 1)
        gp.setColumnStretch(3, 1)

        # bookmarks panel
        bm_group = QGroupBox("Bookmarks")
        bml = QHBoxLayout(bm_group)
        bml.setContentsMargins(6, 6, 6, 6)
        self.btn_bm_add = QPushButton("Add")
        self.btn_bm_go  = QPushButton("Go")
        self.btn_bm_del = QPushButton("Delete")
        self.bm_list = QListWidget()
        bml.addWidget(self.bm_list, 4)
        vb = QVBoxLayout(); bml.addLayout(vb, 1)
        vb.addWidget(self.btn_bm_add); vb.addWidget(self.btn_bm_go); vb.addWidget(self.btn_bm_del); vb.addStretch(1)

        # image adjustments panel
        adj_group = QGroupBox("Image Adjustments")
        bmr = QGridLayout(adj_group)
        bmr.setContentsMargins(6, 6, 6, 6)
        bmr.setHorizontalSpacing(8)
        bmr.setVerticalSpacing(6)

        self.sld_contrast = ClickJumpSlider(Qt.Horizontal); self.sld_contrast.setRange(0, 200); self.sld_contrast.setValue(self.default_contrast_ui)
        self.spn_contrast = QSpinBox(); self.spn_contrast.setRange(0, 200); self.spn_contrast.setValue(self.default_contrast_ui); self.spn_contrast.setSuffix("%")
        self.sld_brightness = ClickJumpSlider(Qt.Horizontal); self.sld_brightness.setRange(-100, 100); self.sld_brightness.setValue(self.default_brightness_ui)
        self.spn_brightness = QSpinBox(); self.spn_brightness.setRange(-100, 100); self.spn_brightness.setValue(self.default_brightness_ui)
        self.sld_saturation = ClickJumpSlider(Qt.Horizontal); self.sld_saturation.setRange(0, 200); self.sld_saturation.setValue(self.default_saturation_ui)
        self.spn_saturation = QSpinBox(); self.spn_saturation.setRange(0, 200); self.spn_saturation.setValue(self.default_saturation_ui); self.spn_saturation.setSuffix("%")
        self.btn_adjust_reset = QPushButton("Reset")

        bmr.addWidget(QLabel("Contrast"),   0, 0)
        bmr.addWidget(self.sld_contrast,    0, 1)
        bmr.addWidget(self.spn_contrast,    0, 2)
        bmr.addWidget(QLabel("Brightness"), 1, 0)
        bmr.addWidget(self.sld_brightness,  1, 1)
        bmr.addWidget(self.spn_brightness,  1, 2)
        bmr.addWidget(QLabel("Saturation"), 2, 0)
        bmr.addWidget(self.sld_saturation,  2, 1)
        bmr.addWidget(self.spn_saturation,  2, 2)
        bmr.addWidget(self.btn_adjust_reset, 3, 0, 1, 3)

        bmr.setColumnStretch(1, 1)

        # right: file list + cut params + run
        file_group = QGroupBox("Video Files")
        file_group.setMinimumWidth(330)
        gf = QVBoxLayout(file_group)
        self.btn_open = QPushButton("Open Folder")
        self.list_videos = QListWidget()
        self.btn_load = QPushButton("Load Video")
        gf.addWidget(self.btn_open); gf.addWidget(self.list_videos); gf.addWidget(self.btn_load)
        G.addWidget(file_group, 0, 1)

        cut_group = QGroupBox("Cut Parameters")
        cut_group.setMinimumWidth(330)
        gc = QGridLayout(cut_group)

        # checkboxes
        self.chk_start = QCheckBox("Start")
        self.chk_dur   = QCheckBox("Duration")
        self.chk_end   = QCheckBox("End")
        self.chk_start.setChecked(True); self.chk_end.setChecked(False); self.chk_dur.setChecked(True)
        # 숨김 + 비활성
        for _w in (self.chk_start, self.chk_dur, self.chk_end):
            _w.setVisible(False)
            _w.setEnabled(False)

        # mode combo box (Start+Duration / Duration+End / Start+End)
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Start + Duration", "Duration + End", "Start + End"])
        self.combo_mode.setCurrentIndex(2)  # default: state3 (Start + End)
        gc.addWidget(QLabel("Mode:"), 0, 0)
        gc.addWidget(self.combo_mode, 0, 1, 1, 4)

        # start widgets
        self.ed_start = QLineEdit(); self.ed_start.setValidator(QIntValidator(0, 10**9, self))
        self.btn_start_from_cur = QPushButton("Use current")
        self.lbl_start_time = QLabel("(00:00.000)")
        # duration widgets
        self.ed_dur = QSpinBox(); self.ed_dur.setRange(1, 10**6); self.ed_dur.setValue(5)
        self.unit_dur = QComboBox(); self.unit_dur.addItems(["seconds", "frames", "minutes"]); self.unit_dur.setCurrentText("minutes")
        # end widgets
        self.ed_end = QLineEdit(); self.ed_end.setValidator(QIntValidator(0, 10**9, self))
        self.btn_end_from_cur = QPushButton("Use current")
        self.lbl_end_time = QLabel("(00:00.000)")

        # Start
        gc.addWidget(self.ed_start,            1, 2)
        self.ed_start.setPlaceholderText("(frame)")
        gc.addWidget(self.btn_start_from_cur,  1, 3)
        gc.addWidget(self.lbl_start_time,      1, 4)
        # Duration
        gc.addWidget(self.ed_dur,              2, 2)
        gc.addWidget(self.unit_dur,            2, 3)
        gc.addWidget(QWidget(),                2, 4)
        # End
        gc.addWidget(self.ed_end,              3, 2)
        self.ed_end.setPlaceholderText("(frame)") 
        gc.addWidget(self.btn_end_from_cur,    3, 3)
        gc.addWidget(self.lbl_end_time,        3, 4)

        # suffix + run
        run_group = QGroupBox("Export")
        run_group.setMinimumWidth(330)
        ge = QGridLayout(run_group)
        self.ed_prefix = QLineEdit("")
        self.ed_suffix = QLineEdit("cut")
        self.btn_cut = QPushButton("Save Current Video")
        self.btn_cut_multi = QPushButton("Save Videos...")
        self.rad_accurate = QRadioButton("accurate (re-encode)")
        self.rad_fast = QRadioButton("fast (stream copy)")
        self.rad_accurate.setChecked(True)
        mode_row = QWidget()
        mode_row_l = QHBoxLayout(mode_row)
        mode_row_l.setContentsMargins(0, 0, 0, 0)
        mode_row_l.setSpacing(12)
        mode_row_l.addWidget(self.rad_accurate)
        mode_row_l.addWidget(self.rad_fast)
        mode_row_l.addStretch(1)
        self._apply_cut_mode_tooltips()
        ge.addWidget(QLabel("Prefix:"), 0, 0)
        ge.addWidget(self.ed_prefix,    0, 1)
        ge.addWidget(QLabel("Suffix:"), 0, 2)
        ge.addWidget(self.ed_suffix,    0, 3)
        ge.addWidget(mode_row,          1, 0, 1, 4)
        export_btn_row = QWidget()
        export_btn_row_l = QHBoxLayout(export_btn_row)
        export_btn_row_l.setContentsMargins(0, 0, 0, 0)
        export_btn_row_l.setSpacing(8)
        export_btn_row_l.addWidget(self.btn_cut)
        export_btn_row_l.addWidget(self.btn_cut_multi)
        ge.addWidget(export_btn_row, 2, 0, 1, 4)

        # 3사분면: Playback + Bookmarks
        bottom_left = QWidget(); bl = QVBoxLayout(bottom_left); bl.setContentsMargins(0,0,0,0); bl.setSpacing(8)
        bl.addWidget(play_group)
        tools_row = QWidget()
        tr = QHBoxLayout(tools_row); tr.setContentsMargins(0, 0, 0, 0); tr.setSpacing(8)
        tr.addWidget(adj_group, 1)
        tr.addWidget(cut_group, 1)
        bl.addWidget(tools_row)
        bm_group.setMaximumHeight(140)
        adj_group.setMaximumHeight(220)
        G.addWidget(bottom_left, 1, 0)

        # 4사분면
        bottom_right = QWidget(); br = QVBoxLayout(bottom_right); br.setContentsMargins(0,0,0,0); br.setSpacing(8)
        br.addWidget(bm_group)
        br.addWidget(run_group)
        G.addWidget(bottom_right, 1, 1)

        # 스트레치: (행) 위쪽 크게, (열) 오른쪽(비디오) 크게
        G.setRowStretch(0, 1)   # 상단(파일+비디오) 비중
        G.setRowStretch(1, 0)   # 하단(플레이/북마크 + 컷) 비중
        G.setColumnStretch(0, 5)  # 좌(비디오/플레이)
        G.setColumnStretch(1, 1)  # 우(리스트/컷)
        self._apply_panel_styles(play_group, bm_group, adj_group, file_group, cut_group, run_group)
        self._apply_slider_styles()
        self._apply_video_list_styles()

        # ---------- connections ----------
        self.btn_open.clicked.connect(self.open_folder)
        self.btn_load.clicked.connect(self.load_video)
        self.list_videos.itemDoubleClicked.connect(self.load_video)
        self.list_videos.itemSelectionChanged.connect(self._refresh_loaded_video_highlight)
        self.btn_play.clicked.connect(self.toggle_play)

        self.slider.sliderPressed.connect(self.on_slider_pressed)
        self.slider.sliderMoved.connect(self.on_slider_moved)
        self.slider.sliderReleased.connect(self.on_slider_released)

        self.speed.valueChanged.connect(self.change_speed)

        self.btn_bm_add.clicked.connect(self.add_bookmark)
        self.btn_bm_go.clicked.connect(self.goto_bookmark)
        self.btn_bm_del.clicked.connect(self.del_bookmark)
        self.sld_contrast.valueChanged.connect(self.spn_contrast.setValue)
        self.spn_contrast.valueChanged.connect(self.sld_contrast.setValue)
        self.sld_brightness.valueChanged.connect(self.spn_brightness.setValue)
        self.spn_brightness.valueChanged.connect(self.sld_brightness.setValue)
        self.sld_saturation.valueChanged.connect(self.spn_saturation.setValue)
        self.spn_saturation.valueChanged.connect(self.sld_saturation.setValue)
        self.sld_contrast.valueChanged.connect(lambda _: self._on_adjustment_changed())
        self.sld_brightness.valueChanged.connect(lambda _: self._on_adjustment_changed())
        self.sld_saturation.valueChanged.connect(lambda _: self._on_adjustment_changed())
        self.btn_adjust_reset.clicked.connect(self.reset_adjustments)

        # param toggles
        self._param_state = 3  # 1: start+duration, 2: duration+end, 3: start+end
        # 초기 상태 강제(혹시 UI 초기값이 어긋나 있어도 맞춤)
        QTimer.singleShot(0, lambda: self._apply_state(self._param_state))
        # 클릭(토글) 이벤트를 상태머신 입력으로 사용
        self._param_state = 3  # 1: Start+Duration, 2: Duration+End, 3: Start+End
        QTimer.singleShot(0, lambda: self._apply_state(self._param_state))
        self.combo_mode.currentIndexChanged.connect(self.on_mode_changed)
        self.ed_start.textChanged.connect(lambda _: self._on_cut_param_changed())
        self.ed_end.textChanged.connect(lambda _: self._on_cut_param_changed())
        self.ed_dur.valueChanged.connect(lambda _: self._on_cut_param_changed())
        self.unit_dur.currentIndexChanged.connect(lambda _: self._on_cut_param_changed())

        # copy from current
        self.btn_start_from_cur.clicked.connect(lambda: self.set_from_current('start'))
        self.btn_end_from_cur.clicked.connect(lambda: self.set_from_current('end'))

        # cut
        self.btn_cut.clicked.connect(self.cut_video)
        self.btn_cut_multi.clicked.connect(self.open_batch_export_dialog)
        self.rad_accurate.toggled.connect(lambda _: self._update_reencode_eta_status())
        self.rad_accurate.toggled.connect(lambda _: self._update_cut_mode_tooltip())
        self.rad_fast.toggled.connect(lambda _: self._update_cut_mode_tooltip())
        # initial state
        self.update_enable_state(folder_loaded=False, video_loaded=False)

        # keyboard focus & no-focus
        self.setFocusPolicy(Qt.StrongFocus)
        self._apply_no_focus()
        self._apply_adjustment_focus_rules()
        self._sync_export_mode_for_adjustments()

        self.statusBar().showMessage("")
        self.status_progress = QProgressBar()
        self.status_progress.setRange(0, 100)
        self.status_progress.setValue(0)
        self.status_progress.setFixedWidth(180)
        self.status_progress.setFixedHeight(10)
        self.status_progress.setTextVisible(False)
        self.status_progress.setStyleSheet(
            "QProgressBar {"
            "background: #e5e9f0;"
            "border: 1px solid #c7cfdb;"
            "border-radius: 5px;"
            "}"
            "QProgressBar::chunk {"
            "background: #4a90ff;"
            "border-radius: 5px;"
            "}"
        )
        self.status_progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.status_progress)
        self.status_progress_info = QLabel("")
        self.status_progress_info.setStyleSheet("color: #2f3a4a;")
        self.status_progress_info.setVisible(False)
        self.statusBar().addPermanentWidget(self.status_progress_info)

    # 비차단 상태 메시지 표시 유틸 (하단 status bar + Export 라벨 동시 갱신)
    def _set_export_status(self, text: str, tooltip: str = None, auto_clear_ms: int = 0):
        self.statusBar().showMessage(text or "")
        if auto_clear_ms and auto_clear_ms > 0:
            QTimer.singleShot(auto_clear_ms, lambda: self.statusBar().showMessage(""))

    def _apply_no_focus(self):
        # 대부분 위젯은 NoFocus, 텍스트 입력/리스트는 ClickFocus 유지
        keep_click = {QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox, QListWidget}
        for w in self.findChildren(QWidget):
            if any(isinstance(w, k) for k in keep_click):
                w.setFocusPolicy(Qt.ClickFocus)
            else:
                w.setFocusPolicy(Qt.NoFocus)

    # ----------------------------- helpers -----------------------------
    def fmt_time(self, seconds: float) -> str:
        if seconds < 0: seconds = 0
        m, s = divmod(seconds, 60.0)
        return f"{int(m):02d}:{s:06.3f}"

    def _cut_mode_tooltip_text(self, index: int) -> str:
        if index == 0:
            return (
                "Accurate mode: re-encodes video/audio for precise start and end points. "
                "Best when exact timing matters, but export is slower and output bitrate/quality settings may change."
            )
        if index == 1:
            return (
                "Fast mode: uses stream copy without re-encoding. Much faster and keeps original streams, "
                "but cut points can shift to nearby keyframes."
            )
        return ""

    def _apply_cut_mode_tooltips(self):
        self.rad_accurate.setToolTip(self._cut_mode_tooltip_text(0))
        self.rad_fast.setToolTip(self._cut_mode_tooltip_text(1))
        self._update_cut_mode_tooltip()

    def _update_cut_mode_tooltip(self):
        tip = self._cut_mode_tooltip_text(0) if self.rad_accurate.isChecked() else self._cut_mode_tooltip_text(1)
        self.rad_accurate.setToolTip(tip if self.rad_accurate.isChecked() else self._cut_mode_tooltip_text(0))
        if self.rad_fast.isEnabled():
            self.rad_fast.setToolTip(tip if self.rad_fast.isChecked() else self._cut_mode_tooltip_text(1))

    def _apply_panel_styles(self, *groups):
        for g in groups:
            bg = "#fdfdfd"
            style = (
                "QGroupBox {"
                f"background: {bg};"
                "border: 1px solid #d9d9d9;"
                "border-radius: 6px;"
                "margin-top: 8px;"
                "}"
                "QGroupBox::title {"
                "subcontrol-origin: margin;"
                "left: 8px;"
                "padding: 0 3px;"
                "}"
            )
            g.setStyleSheet(style)

    def _apply_slider_styles(self):
        style = (
            "QSlider::groove:horizontal {"
            "height: 6px;"
            "background: #d7d7d7;"
            "border-radius: 3px;"
            "}"
            "QSlider::sub-page:horizontal {"
            "background: #4a90ff;"
            "border-radius: 3px;"
            "}"
            "QSlider::handle:horizontal {"
            "background: #4a90ff;"
            "border: 1px solid #2f74dd;"
            "width: 14px;"
            "height: 14px;"
            "margin: -5px 0;"
            "border-radius: 7px;"
            "}"
        )
        for s in (self.slider, self.sld_contrast, self.sld_brightness, self.sld_saturation):
            s.setStyleSheet(style)

    def _apply_video_list_styles(self):
        self.list_videos.setStyleSheet("QListWidget::item { padding: 2px 4px; }")

    def _refresh_loaded_video_highlight(self):
        selected_rows = {idx.row() for idx in self.list_videos.selectedIndexes()}
        for i in range(self.list_videos.count()):
            it = self.list_videos.item(i)
            is_loaded = bool(self.loaded_video_name and it.text() == self.loaded_video_name)
            is_selected = i in selected_rows
            if is_loaded and is_selected:
                it.setBackground(QColor("#356bb3"))  # darker blue-gray overlay state
                it.setForeground(QColor("white"))
            elif is_loaded:
                it.setBackground(QColor("#4a90ff"))
                it.setForeground(QColor("white"))
            elif is_selected:
                it.setBackground(QColor("#d9dee6"))
                it.setForeground(QColor("#111111"))
            else:
                it.setBackground(QColor("white"))
                it.setForeground(QColor("black"))

    def _apply_adjustment_focus_rules(self):
        for sp in (self.spn_contrast, self.spn_brightness, self.spn_saturation):
            sp.setFocusPolicy(Qt.NoFocus)
            sp.lineEdit().setFocusPolicy(Qt.ClickFocus)
        self.btn_adjust_reset.setFocusPolicy(Qt.NoFocus)

    def _current_adjustments(self):
        contrast = self.sld_contrast.value() / 100.0
        brightness = self.sld_brightness.value() / 100.0
        saturation = self.sld_saturation.value() / 100.0
        return contrast, brightness, saturation

    def _adjustments_active(self) -> bool:
        contrast, brightness, saturation = self._current_adjustments()
        return abs(contrast - 1.0) > 1e-6 or abs(brightness) > 1e-6 or abs(saturation - 1.0) > 1e-6

    def _ffmpeg_eq_filter(self) -> str:
        contrast, brightness, saturation = self._current_adjustments()
        return f"eq=contrast={contrast:.3f}:brightness={brightness:.3f}:saturation={saturation:.3f}"

    def _sync_export_mode_for_adjustments(self):
        if not self.video_path:
            self.rad_accurate.setEnabled(False)
            self.rad_fast.setEnabled(False)
            return
        if self._adjustments_active():
            if not self.rad_accurate.isChecked():
                self.rad_accurate.setChecked(True)
            self.rad_fast.setEnabled(False)
            self.rad_fast.setToolTip("When image adjustments are enabled, only Accurate mode is available.")
        else:
            self.rad_accurate.setEnabled(True)
            self.rad_fast.setEnabled(True)
            self.rad_fast.setToolTip(self._cut_mode_tooltip_text(1))
            self._update_cut_mode_tooltip()

    def _on_adjustment_changed(self, _=None):
        self._sync_export_mode_for_adjustments()
        if not self.thread:
            return
        self.thread.set_adjustments(*self._current_adjustments())
        if not self.is_playing:
            self.thread.seek(self.current_frame)

    def reset_adjustments(self):
        self.sld_contrast.setValue(self.default_contrast_ui)
        self.sld_brightness.setValue(self.default_brightness_ui)
        self.sld_saturation.setValue(self.default_saturation_ui)
        self._on_adjustment_changed()

    def update_labels(self):
        self.lbl_frame.setText(f"Frame: {self.current_frame} / {max(0,self.total_frames-1)}")
        cur_sec = self.current_frame / self.fps if self.fps else 0.0
        tot_sec = (self.total_frames-1) / self.fps if (self.fps and self.total_frames>0) else 0.0
        self.lbl_time.setText(f"Time: {self.fmt_time(cur_sec)} / {self.fmt_time(tot_sec)}")

        # reflect into start/end time hint labels
        try:
            sf = int(self.ed_start.text()) if self.ed_start.text() else 0
        except ValueError:
            sf = 0
        try:
            ef = int(self.ed_end.text()) if self.ed_end.text() else 0
        except ValueError:
            ef = 0
        self.lbl_start_time.setText(f"({self.fmt_time(sf / self.fps if self.fps else 0.0)})")
        self.lbl_end_time.setText(f"({self.fmt_time(ef / self.fps if self.fps else 0.0)})")

    def video_duration_sec(self) -> float:
        if self.fps <= 1e-6 or self.total_frames <= 0:
            return 0.0
        return self.total_frames / self.fps

    def _duration_input_seconds_for_fps(self, fps: float) -> Optional[float]:
        dur_val = float(self.ed_dur.value())
        unit = self.unit_dur.currentText()
        if unit == "seconds":
            return dur_val
        if unit == "minutes":
            return dur_val * 60.0
        if fps <= 1e-6:
            return None
        return dur_val / fps

    def _duration_input_seconds(self) -> Optional[float]:
        return self._duration_input_seconds_for_fps(self.fps)

    def _set_duration_warning(self, enabled: bool):
        try:
            self.ed_dur.lineEdit().setStyleSheet("color: #cc0000;" if enabled else "")
        except Exception:
            self.ed_dur.setStyleSheet("color: #cc0000;" if enabled else "")
        tooltip = self.duration_warning_text if enabled else ""
        self.ed_dur.setToolTip(tooltip)
        self.unit_dur.setToolTip(tooltip)

    def _duration_may_truncate(self) -> bool:
        state = getattr(self, "_param_state", 1)
        if state not in (1, 2):
            return False
        total_sec = self.video_duration_sec()
        if total_sec <= 0 or self.fps <= 1e-6:
            return False

        requested_dur_sec = self._duration_input_seconds()
        if requested_dur_sec is None or requested_dur_sec <= 0:
            return False

        if state == 1:
            try:
                sf = int(self.ed_start.text()) if self.ed_start.text() else 0
            except ValueError:
                return False
            start_sec = max(0.0, min(sf / self.fps, total_sec))
            available_sec = max(0.0, total_sec - start_sec)
            return requested_dur_sec - available_sec > 1e-6

        # state == 2 (Duration + End)
        try:
            ef = int(self.ed_end.text()) if self.ed_end.text() else 0
        except ValueError:
            return False
        end_sec = max(0.0, min(ef / self.fps, total_sec))
        available_sec = end_sec
        return requested_dur_sec - available_sec > 1e-6

    def _update_duration_warning(self):
        self._set_duration_warning(self._duration_may_truncate())

    def _fmt_eta(self, seconds: float) -> str:
        sec = max(0.0, float(seconds))
        if sec < 60:
            return f"{sec:.1f}s"
        whole = int(round(sec))
        m, s = divmod(whole, 60)
        if m < 60:
            return f"{m}m {s}s"
        h, m = divmod(m, 60)
        return f"{h}h {m}m {s}s"

    def _estimate_cut_walltime(self, clip_seconds: float):
        clip = max(0.0, float(clip_seconds))
        if clip <= 1e-6:
            return 0.0, 0.0, 1.0
        width = max(1, int(self.video_width) if self.video_width else 1920)
        height = max(1, int(self.video_height) if self.video_height else 1080)
        fps = self.fps if self.fps > 1e-6 else 30.0

        complexity = (width * height) / (1920.0 * 1080.0)
        complexity *= max(0.5, fps / 30.0)
        encode_speed_rt = max(0.25, 1.8 / max(0.2, complexity))
        copy_speed_rt = max(8.0, 35.0 / math.sqrt(max(1.0, complexity)))

        encode_time = clip / encode_speed_rt
        copy_time = clip / copy_speed_rt
        slowdown = encode_time / max(copy_time, 1e-6)
        return copy_time, encode_time, slowdown

    def _update_reencode_eta_status(self):
        if not self.rad_accurate.isChecked():
            return
        res, err = self._resolve_cut_params()
        if err:
            return
        _, encode_time, slowdown = self._estimate_cut_walltime(res["dur_sec"])
        self._set_export_status(
            f"Estimated re-encode time: {self._fmt_eta(encode_time)} (about {slowdown:.0f}x slower than fast copy)."
        )

    def _on_cut_param_changed(self):
        self.update_labels()
        self._update_duration_warning()
        self._update_reencode_eta_status()

    def update_enable_state(self, folder_loaded: bool, video_loaded: bool):
        # defaults
        self.btn_open.setEnabled(True)
        self.list_videos.setEnabled(folder_loaded)
        self.btn_load.setEnabled(folder_loaded)

        # right side panels
        enable_right = video_loaded
        for w in (self.ed_start, self.btn_start_from_cur, self.ed_dur, self.unit_dur,
                  self.ed_end, self.btn_end_from_cur, self.ed_prefix, self.ed_suffix, self.btn_cut, self.btn_cut_multi,
                  self.rad_accurate, self.rad_fast,
                  self.combo_mode):
            w.setEnabled(enable_right)

        # playback + bookmarks
        for w in (self.btn_play, self.slider, self.speed,
                  self.btn_bm_add, self.btn_bm_go, self.btn_bm_del, self.bm_list,
                  self.sld_contrast, self.sld_brightness, self.sld_saturation,
                  self.spn_contrast, self.spn_brightness, self.spn_saturation,
                  self.btn_adjust_reset):
            w.setEnabled(video_loaded)

        # enforce the 2-of-3 rule even when enabling
        self._apply_state(getattr(self, "_param_state", 1))
        self._on_cut_param_changed()
        if video_loaded:
            self._sync_export_mode_for_adjustments()
        else:
            self.rad_accurate.setEnabled(False)
            self.rad_fast.setEnabled(False)

    # ----------------------------- file ops -----------------------------
    def open_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Open Video Folder", self.video_folder or os.getcwd())
        if not path:
            return
        self.video_folder = path
        self.list_videos.clear()
        # list video files
        exts = (".mp4",".mkv",".avi",".mov",".m4v",".webm")
        files = [f for f in os.listdir(path) if f.lower().endswith(exts)]
        files.sort()
        self.list_videos.addItems(files)
        self._refresh_loaded_video_highlight()
        self.update_enable_state(folder_loaded=True, video_loaded=False)

    def _apply_mode_values_on_video_load(self):
        last_frame = max(0, self.total_frames - 1)
        state = getattr(self, "_param_state", 1)

        # In Start+End mode, always reset to the full span of the newly loaded video.
        if state == 3:
            self.ed_start.setText("0")
            self.ed_end.setText(str(last_frame))
            return

        # In Duration+End mode, clamp existing End to the last frame of the new video.
        if state == 2:
            try:
                end_frame = int(self.ed_end.text()) if self.ed_end.text() else last_frame
            except ValueError:
                end_frame = last_frame
            if end_frame > last_frame:
                self.ed_end.setText(str(last_frame))

    def load_video(self):
        item = self.list_videos.currentItem()
        if not item:
            return
        self.video_path = os.path.join(self.video_folder, item.text())
        # stop existing thread
        if self.thread:
            self.thread.stop()
            self.thread.wait()
            self.thread = None

        self.thread = VideoThread(self.video_path)
        if not self.thread.open():
            QMessageBox.warning(self, "Error", "Failed to open video.")
            self.thread = None
            self.video_path = None
            return

        self.fps = self.thread.fps
        self.total_frames = self.thread.total
        self.video_width = self.thread.width
        self.video_height = self.thread.height
        self.slider.setRange(0, max(0, self.total_frames-1))
        self.slider.setValue(0)
        self.current_frame = 0
        self.is_playing = False
        self.btn_play.setText("Play")
        self._apply_mode_values_on_video_load()

        # connect signals
        self.thread.frameReady.connect(self.on_frame)
        self.thread.finished.connect(self.on_video_finished)
        self.thread.start()
        self.thread.set_adjustments(*self._current_adjustments())

        # auto show first frame
        self.thread.seek(0)
        self.thread.pause()

        self.update_labels()
        self.loaded_video_name = os.path.basename(self.video_path)
        self._refresh_loaded_video_highlight()
        self.update_enable_state(folder_loaded=True, video_loaded=True)
        self._on_cut_param_changed()

    # --------------------------- playback handlers ---------------------------
    @pyqtSlot(QImage, int)
    def on_frame(self, qimg: QImage, idx: int):
        self.current_frame = idx
        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)
        self.video_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.update_labels()

    def resizeEvent(self, e):
        # keep aspect ratio on label resize; request current frame again
        if self.thread:
            self.thread.seek(self.current_frame)
        super().resizeEvent(e)

    def on_video_finished(self):
        self.is_playing = False
        self.btn_play.setText("Play")

    def toggle_play(self):
        if not self.thread:
            return
        if self.is_playing:
            self.thread.pause()
            self.is_playing = False
            self.btn_play.setText("Play")
        else:
            # at end → rewind to last value if needed
            if self.current_frame >= max(0, self.total_frames-1):
                self.thread.seek(0)
            self.thread.play()
            self.is_playing = True
            self.btn_play.setText("Pause")

    def change_speed(self, s: float):
        if self.thread:
            self.thread.set_speed(float(s))

    def on_slider_pressed(self):
        if not self.thread: return
        self.scrub_was_playing = self.is_playing
        if self.is_playing:
            self.thread.pause()
            self.is_playing = False
            self.btn_play.setText("Play")

    def on_slider_moved(self, pos: int):
        if not self.thread: return
        self.thread.seek(int(pos))

    def on_slider_released(self):
        if not self.thread: return
        self.thread.seek(int(self.slider.value()))
        if self.scrub_was_playing:
            self.thread.play()
            self.is_playing = True
            self.btn_play.setText("Pause")
        self.scrub_was_playing = False

    # ---------------------------- keyboard nav ----------------------------
    def keyPressEvent(self, ev):
        if not self.thread:
            return super().keyPressEvent(ev)
        key = ev.key()
        if key == Qt.Key_Space:
            self.toggle_play(); return
        if key == Qt.Key_Right:
            # step +1
            self.thread.pause(); self.is_playing = False; self.btn_play.setText("Play")
            self.thread.seek(min(self.current_frame + 1, max(0, self.total_frames-1))); return
        if key == Qt.Key_Left:
            self.thread.pause(); self.is_playing = False; self.btn_play.setText("Play")
            self.thread.seek(max(self.current_frame - 1, 0)); return
        return super().keyPressEvent(ev)

    # ------------------------------ bookmarks ------------------------------
    def add_bookmark(self):
        if self.total_frames <= 0: return
        f = self.current_frame
        t = f / self.fps if self.fps else 0.0
        self.bm_list.addItem(f"Frame {f}  ({self.fmt_time(t)})")

    def current_bm_frame(self) -> Optional[int]:
        it = self.bm_list.currentItem()
        if not it: return None
        # parse "Frame X  (..)"
        try:
            prefix = "Frame "
            s = it.text()
            pos = s.find(prefix)
            if pos >= 0:
                rest = s[pos+len(prefix):].strip().split()[0]
                return int(rest)
        except Exception:
            return None
        return None

    def goto_bookmark(self):
        f = self.current_bm_frame()
        if f is None: return
        self.thread.pause(); self.is_playing = False; self.btn_play.setText("Play")
        self.thread.seek(max(0, min(f, max(0, self.total_frames-1))))

    def del_bookmark(self):
        row = self.bm_list.currentRow()
        if row >= 0:
            self.bm_list.takeItem(row)

    # --------------------------- cut parameters ---------------------------
    def _apply_state(self, state: int):
        """state: 1=start+duration, 2=duration+end, 3=start+end"""
        def _set(box, val):
            box.blockSignals(True); box.setChecked(val); box.blockSignals(False)
        if state == 1:   # start+duration
            _set(self.chk_start, True)
            _set(self.chk_dur,   True)
            _set(self.chk_end,   False)
        elif state == 2: # duration+end
            _set(self.chk_start, False)
            _set(self.chk_dur,   True)
            _set(self.chk_end,   True)
        elif state == 3: # start+end
            _set(self.chk_start, True)
            _set(self.chk_dur,   False)
            _set(self.chk_end,   True)
        else:
            # 안전 기본값
            _set(self.chk_start, True)
            _set(self.chk_dur,   True)
            _set(self.chk_end,   False)
        # 행 활성화: 체크박스 표시 대신 입력행만 모드에 맞게 on/off
        self._enable_param_row('start', state in (1, 3))
        self._enable_param_row('dur',   state in (1, 2))
        self._enable_param_row('end',   state in (2, 3))
        # 콤보 표시와 내부 상태 동기화
        combo_idx = {1:0, 2:1, 3:2}.get(state, 0)
        if self.combo_mode.currentIndex() != combo_idx:
            self.combo_mode.blockSignals(True)
            self.combo_mode.setCurrentIndex(combo_idx)
            self.combo_mode.blockSignals(False)

    def on_mode_changed(self, idx: int):
        mapping = {0: 1, 1: 2, 2: 3}
        self._param_state = mapping.get(idx, 1)
        self._apply_state(self._param_state)
        self._on_cut_param_changed()

    def _enable_param_row(self, which: str, enabled: bool):
        if which == 'start':
            for w in (self.ed_start, self.btn_start_from_cur):
                w.setEnabled(enabled)
        elif which == 'dur':
            for w in (self.ed_dur, self.unit_dur):
                w.setEnabled(enabled)
        elif which == 'end':
            for w in (self.ed_end, self.btn_end_from_cur):
                w.setEnabled(enabled)

    def set_from_current(self, which: str):
        if which == 'start' and self.chk_start.isChecked():
            self.ed_start.setText(str(self.current_frame))
        elif which == 'end' and self.chk_end.isChecked():
            self.ed_end.setText(str(self.current_frame))
        self._on_cut_param_changed()

    # ------------------------------- cutter -------------------------------
    def _resolve_cut_params_for_video(self, fps: float, total_frames: int):
        """Return ({start_sec, dur_sec, duration_truncated}, None) or (None, error_text)."""
        state = getattr(self, "_param_state", 1)  # 1=S+D, 2=D+E, 3=S+E
        have_s = state in (1, 3)
        have_d = state in (1, 2)
        have_e = state in (2, 3)
        if fps <= 1e-6 or total_frames <= 0:
            return None, "Video duration is not available."
        total_sec = total_frames / fps
        if total_sec <= 0:
            return None, "Video duration is not available."

        # read inputs
        try:
            sf = int(self.ed_start.text()) if (self.ed_start.text() and have_s) else None
            ef = int(self.ed_end.text()) if (self.ed_end.text() and have_e) else None
        except ValueError:
            return None, "Start/End must be valid frame numbers."

        requested_dur_sec = None
        if have_d:
            requested_dur_sec = self._duration_input_seconds_for_fps(fps)
            if requested_dur_sec is None:
                return None, "FPS information is missing for frame-based duration."
            if requested_dur_sec <= 0:
                return None, "Duration must be larger than 0."

        # compute missing
        try:
            if have_s and have_e:
                if sf is None or ef is None:
                    return None, "Start/End must be provided."
                if ef <= sf:
                    return None, "End must be larger than Start."
                start_sec = max(0.0, min(sf / fps, total_sec))
                end_sec = max(0.0, min(ef / fps, total_sec))
                dur_sec = end_sec - start_sec
                if dur_sec <= 0:
                    return None, "Invalid range after clamp."
                return {
                    "start_sec": start_sec,
                    "dur_sec": dur_sec,
                    "requested_dur_sec": dur_sec,
                    "duration_truncated": False,
                }, None
            elif have_s and have_d:
                if sf is None or requested_dur_sec is None:
                    return None, "Start/Duration must be provided."
                start_sec = max(0.0, min(sf / fps, total_sec))
                available = max(0.0, total_sec - start_sec)
                dur_sec = min(requested_dur_sec, available)
                if dur_sec <= 0:
                    return None, "Requested range starts at or beyond the video end."
                return {
                    "start_sec": start_sec,
                    "dur_sec": dur_sec,
                    "requested_dur_sec": requested_dur_sec,
                    "duration_truncated": requested_dur_sec - dur_sec > 1e-6,
                }, None
            elif have_d and have_e:
                if requested_dur_sec is None or ef is None:
                    return None, "Duration/End must be provided."
                end_sec = max(0.0, min(ef / fps, total_sec))
                start_sec = max(0.0, end_sec - requested_dur_sec)
                dur_sec = end_sec - start_sec
                if dur_sec <= 0:
                    return None, "Requested range ends at or before the video start."
                return {
                    "start_sec": start_sec,
                    "dur_sec": dur_sec,
                    "requested_dur_sec": requested_dur_sec,
                    "duration_truncated": requested_dur_sec - dur_sec > 1e-6,
                }, None
            else:
                return None, "Exactly two parameters must be selected."
        except Exception as e:
            return None, f"Invalid input: {e}"

    def _resolve_cut_params(self):
        return self._resolve_cut_params_for_video(self.fps, self.total_frames)

    # ------------------------------ ffmpeg check ------------------------------

    def _find_ffmpeg(self) -> str:
        """
        1) PyInstaller onedir 배포물(dist/<app>/) 루트에 동봉된 ffmpeg(.exe) 우선
        2) PATH에서 ffmpeg 검색
        찾지 못하면 빈 문자열 반환
        """
        try:
            base = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(__file__)
            cand = os.path.join(base, "ffmpeg.exe" if os.name == "nt" else "ffmpeg")
            if os.path.isfile(cand):
                return cand
        except Exception:
            pass
        found = shutil.which("ffmpeg")
        return found or ""

    def _ffmpeg_exists(self) -> bool:
        return bool(self._find_ffmpeg())

    def _is_export_running(self) -> bool:
        return bool(
            (self.export_thread and self.export_thread.isRunning())
            or (self.batch_export_thread and self.batch_export_thread.isRunning())
        )

    def _make_output_path(self, video_path: str) -> str:
        folder = os.path.dirname(video_path)
        base, ext = os.path.splitext(os.path.basename(video_path))
        prefix = self.ed_prefix.text().strip().strip("_")
        suffix = self.ed_suffix.text().strip().strip("_")
        prefix_part = f"{prefix}_" if prefix else ""
        suffix_part = f"_{suffix}" if suffix else ""
        out_name = f"{prefix_part}{base}{suffix_part}{ext}"
        return os.path.join(folder, out_name)

    def _build_export_command(self, ffmpeg: str, video_path: str, out_path: str, start_sec: float, dur_sec: float):
        adjustments_active = self._adjustments_active()
        progress_args = ["-progress", "pipe:2", "-nostats"]

        fast_cmd = [
            ffmpeg,
            "-y",
            "-ss", f"{start_sec:.6f}",
            "-i", video_path,
            "-t", f"{dur_sec:.6f}",
            "-c", "copy",
            *progress_args,
            out_path
        ]

        accurate_cmd = [
            ffmpeg,
            "-y",
            "-ss", f"{start_sec:.6f}",
            "-i", video_path,
            "-t", f"{dur_sec:.6f}",
        ]
        if adjustments_active:
            accurate_cmd.extend(["-vf", self._ffmpeg_eq_filter()])
        accurate_cmd.extend([
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            *progress_args,
            out_path
        ])

        mode = "accurate" if self.rad_accurate.isChecked() else "fast"
        if adjustments_active and mode.startswith("fast"):
            mode = "accurate"
        return (accurate_cmd if mode.startswith("accurate") else fast_cmd), mode

    def _read_video_meta(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap or not cap.isOpened():
            return None
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            fps = float(fps) if fps > 1e-3 else 30.0
            return fps, total
        finally:
            cap.release()

    def _set_progress_context(self, text: str):
        if text:
            self.status_progress_info.setText(text)
            self.status_progress_info.setVisible(True)
            return
        self.status_progress_info.setText("")
        self.status_progress_info.setVisible(False)

    def _set_export_running(self, running: bool):
        self.btn_cut.setEnabled((self.video_path is not None) and (not running))
        self.btn_cut_multi.setEnabled((self.video_path is not None) and (not running))
        self.status_progress.setVisible(running)
        if not running:
            self.status_progress.setValue(0)
            self._set_progress_context("")

    def _on_export_progress(self, pct: int):
        self.status_progress.setVisible(True)
        self.status_progress.setValue(max(0, min(100, int(pct))))

    def _on_export_finished(self, out_path: str):
        self._set_export_running(False)
        self._set_export_status(f"Saved: {out_path}", tooltip=out_path, auto_clear_ms=6000)
        self.export_thread = None

    def _on_export_failed(self, err_text: str):
        self._set_export_running(False)
        QMessageBox.critical(self, "ffmpeg error", (err_text or "")[-8000:])
        self._set_export_status("Export failed. See error dialog.", auto_clear_ms=6000)
        self.export_thread = None

    def _start_export_thread(self, cmd: List[str], out_path: str, dur_sec: float):
        self._set_export_running(True)
        self.status_progress.setValue(0)
        self._set_export_status("Processing video...")
        self._set_progress_context(f"1/1  {os.path.basename(self.video_path or out_path)}")
        self.export_thread = ExportThread(cmd, out_path, dur_sec)
        self.export_thread.progressChanged.connect(self._on_export_progress)
        self.export_thread.finishedOk.connect(self._on_export_finished)
        self.export_thread.failed.connect(self._on_export_failed)
        self.export_thread.start()

    def _on_batch_item_changed(self, idx: int, total: int, label: str):
        self._set_progress_context(f"{idx}/{total}  {label}")

    def _on_batch_done(self, summary: str, has_errors: bool):
        self._set_export_running(False)
        self.batch_export_thread = None
        if has_errors:
            QMessageBox.warning(self, "Batch export", summary[-8000:])
            self._set_export_status("Batch export completed with errors.", auto_clear_ms=8000)
        else:
            self._set_export_status(summary, auto_clear_ms=8000)

    def _start_batch_export_thread(self, tasks: List[dict]):
        self._set_export_running(True)
        self.status_progress.setValue(0)
        self._set_export_status("Processing selected videos...")
        self.batch_export_thread = BatchExportThread(tasks)
        self.batch_export_thread.itemChanged.connect(self._on_batch_item_changed)
        self.batch_export_thread.progressChanged.connect(self._on_export_progress)
        self.batch_export_thread.done.connect(self._on_batch_done)
        self.batch_export_thread.start()

    def open_batch_export_dialog(self):
        if self._is_export_running():
            QMessageBox.information(self, "Export in progress", "Another export is already running.")
            return
        if self.list_videos.count() <= 0:
            QMessageBox.information(self, "Save Videos", "No videos in list.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Save Videos")
        dlg.setMinimumSize(420, 380)
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel("Select videos to export (Shift/Ctrl multi-select, Ctrl+A select all)."))

        lw = QListWidget(dlg)
        lw.setSelectionMode(QAbstractItemView.ExtendedSelection)
        for i in range(self.list_videos.count()):
            lw.addItem(self.list_videos.item(i).text())
        v.addWidget(lw)

        QShortcut(QKeySequence.SelectAll, lw, activated=lw.selectAll)

        box = QDialogButtonBox(dlg)
        run_btn = box.addButton("Run", QDialogButtonBox.AcceptRole)
        box.addButton("Cancel", QDialogButtonBox.RejectRole)
        v.addWidget(box)

        selected_names: List[str] = []

        def on_run():
            names = [it.text() for it in lw.selectedItems()]
            if not names:
                QMessageBox.information(dlg, "Save Videos", "Select at least one video.")
                return
            selected_names.clear()
            selected_names.extend(names)
            dlg.accept()

        run_btn.clicked.connect(on_run)
        box.rejected.connect(dlg.reject)

        if dlg.exec_() != QDialog.Accepted:
            return
        if not selected_names:
            return
        self.cut_videos_batch(selected_names)

    def cut_videos_batch(self, selected_names: List[str]):
        if self._is_export_running():
            self._set_export_status("Another export is already running.")
            return
        ffmpeg = self._find_ffmpeg()
        if not ffmpeg:
            QMessageBox.warning(
                self, "ffmpeg not found",
                "ffmpeg is required to cut videos.\n\n"
                "If you ran it as an exe file: Check if the ffmpeg.exe file exists.\n\n"
                "If you ran it as a python file: Add ffmpeg to your system PATH."
            )
            return

        tasks: List[dict] = []
        prep_errors: List[str] = []
        existing_out_paths: List[str] = []
        any_truncated = False

        for name in selected_names:
            video_path = os.path.join(self.video_folder, name)
            if not os.path.isfile(video_path):
                prep_errors.append(f"{name}: file not found.")
                continue
            meta = self._read_video_meta(video_path)
            if not meta:
                prep_errors.append(f"{name}: failed to read video metadata.")
                continue
            fps, total_frames = meta
            res, err = self._resolve_cut_params_for_video(fps, total_frames)
            if err:
                prep_errors.append(f"{name}: {err}")
                continue
            start_sec = res["start_sec"]
            dur_sec = res["dur_sec"]
            any_truncated = any_truncated or bool(res.get("duration_truncated", False))

            out_path = self._make_output_path(video_path)
            cmd, _ = self._build_export_command(ffmpeg, video_path, out_path, start_sec, dur_sec)
            tasks.append({
                "cmd": cmd,
                "out_path": out_path,
                "duration_us": max(1, int(max(0.001, float(dur_sec)) * 1_000_000.0)),
                "label": os.path.basename(video_path),
            })
            if os.path.exists(out_path):
                existing_out_paths.append(out_path)

        if not tasks:
            details = "\n".join(prep_errors[:10])
            QMessageBox.warning(self, "Batch export", f"No valid videos to process.\n\n{details}")
            return

        if prep_errors:
            details = "\n".join(prep_errors[:8])
            r = QMessageBox.question(
                self, "Batch export",
                f"{len(prep_errors)} video(s) cannot be processed.\n\n{details}\n\nContinue with {len(tasks)} valid video(s)?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            if r != QMessageBox.Yes:
                self._set_export_status("Batch export canceled.", auto_clear_ms=5000)
                return

        if existing_out_paths:
            r = QMessageBox.question(
                self, "Overwrite files?",
                f"{len(existing_out_paths)} output file(s) already exist.\n\nOverwrite all?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if r != QMessageBox.Yes:
                self._set_export_status("Batch export canceled (file exists).", auto_clear_ms=5000)
                return

        if any_truncated:
            self._set_export_status(self.duration_warning_text)
        self._start_batch_export_thread(tasks)

    # ------------------------------ cutting ------------------------------
    def cut_video(self):
        if self._is_export_running():
            self._set_export_status("Another export is already running.")
            QMessageBox.information(
                self, "Export in progress",
                "Only one export can run at a time to avoid slowdowns and resource issues."
            )
            return
        if not self.video_path:
            QMessageBox.information(self, "Cut", "No video loaded.")
            return
        ffmpeg = self._find_ffmpeg()
        if not ffmpeg:
            QMessageBox.warning(
                self, "ffmpeg not found",
                "ffmpeg is required to cut videos.\n\n"
                "• If you ran it as an exe file: Check if the ffmpeg.exe file exists. \n\n"
                "• If you ran it as a python file: Add ffmpeg to your system PATH."
            )
            return

        res, err = self._resolve_cut_params()
        if err:
            QMessageBox.warning(self, "Invalid parameters", err)
            return
        start_sec = res["start_sec"]
        dur_sec = res["dur_sec"]
        if res.get("duration_truncated", False):
            self._set_export_status(self.duration_warning_text)

        out_path = self._make_output_path(self.video_path)

        if os.path.exists(out_path):
            r = QMessageBox.question(
                self, "Overwrite?",
                f"File already exists:\n{out_path}\n\nOverwrite?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if r != QMessageBox.Yes:
                self._set_export_status("Export canceled (file exists).", auto_clear_ms=6000)
                return

        QApplication.processEvents()

        cmd, mode = self._build_export_command(ffmpeg, self.video_path, out_path, start_sec, dur_sec)
        if mode.startswith("accurate"):
            _, encode_time, slowdown = self._estimate_cut_walltime(dur_sec)
            self._set_export_status(
                f"Estimated re-encode time: {self._fmt_eta(encode_time)} (about {slowdown:.0f}x slower than fast copy)."
            )
        self._start_export_thread(cmd, out_path, dur_sec)

    # ------------------------------ close ------------------------------
    def closeEvent(self, e):
        if self.export_thread and self.export_thread.isRunning():
            self.export_thread.wait(300)
        if self.batch_export_thread and self.batch_export_thread.isRunning():
            self.batch_export_thread.wait(300)
        if self.thread:
            self.thread.stop()
            self.thread.wait(300)
        super().closeEvent(e)


def main():
    app = QApplication(sys.argv)
    icon_path = _find_app_icon_path()
    if icon_path:
        app.setWindowIcon(QIcon(icon_path))
    w = Cutter()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
