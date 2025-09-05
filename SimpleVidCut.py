import sys, os, shutil, subprocess, math
from typing import Optional, List
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QEvent, QTimer
from PyQt5.QtGui import QImage, QPixmap, QIntValidator
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QListWidget, QPushButton, QLabel, QSlider, QFileDialog, QGroupBox, QLineEdit,
    QDoubleSpinBox, QSpinBox, QComboBox, QMessageBox, QSizePolicy, QCheckBox, QProgressBar
)
import cv2


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
        self.playing = False
        self.speed = 1.0
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
                        h, w = frame.shape[:2]
                        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        qimg = QImage(img.data, w, h, w * 3, QImage.Format_RGB888)
                        self.frameReady.emit(qimg.copy(), self.current_idx)
                    self.msleep(1)
            if self.playing:
                ret, frame = self.cap.read()
                if not ret:
                    self.playing = False
                    self.finished.emit()
                    self.msleep(20)
                    continue
                h, w = frame.shape[:2]
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                qimg = QImage(img.data, w, h, w * 3, QImage.Format_RGB888)
                self.frameReady.emit(qimg.copy(), self.current_idx)
                self.current_idx += 1
                self.msleep(frame_interval_ms())
            else:
                # idle wait
                self.msleep(15)

    @pyqtSlot()
    def play(self):
        self.playing = True

    @pyqtSlot()
    def pause(self):
        self.playing = False

    @pyqtSlot(float)
    def set_speed(self, s: float):
        self.speed = max(0.1, float(s))

    @pyqtSlot(int)
    def seek(self, frame_idx: int):
        self._seek_to = int(frame_idx)

    @pyqtSlot()
    def stop(self):
        self._stop = True
        self.playing = False


# ------------------------------ Main Window ------------------------------
class Cutter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simple VidCut")
        self.setMinimumSize(1100, 720)
        self.video_folder = ""
        self.video_path: Optional[str] = None
        self.fps = 30.0
        self.total_frames = 0
        self.current_frame = 0
        self.thread: Optional[VideoThread] = None
        self.is_playing = False
        self.scrub_was_playing = False

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
        self.slider = QSlider(Qt.Horizontal); self.slider.setEnabled(False)
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

        # bookmarks
        bm_group = QGroupBox("Bookmarks")
        gb = QHBoxLayout(bm_group)
        self.btn_bm_add = QPushButton("Add")
        self.btn_bm_go  = QPushButton("Go")
        self.btn_bm_del = QPushButton("Delete")
        self.bm_list = QListWidget()
        gb.addWidget(self.bm_list, 4)
        vb = QVBoxLayout(); gb.addLayout(vb, 1)
        vb.addWidget(self.btn_bm_add); vb.addWidget(self.btn_bm_go); vb.addWidget(self.btn_bm_del); vb.addStretch(1)

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

        # 모드 콤보박스 (Start+Duration / Duration+End / Start+End)
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Start + Duration", "Duration + End", "Start + End"])
        self.combo_mode.setCurrentIndex(0)  # 기본: state1
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
        self.ed_suffix = QLineEdit("_cut")
        self.btn_cut = QPushButton("Cut Video")
        self.comb_cut = QComboBox()
        self.comb_cut.addItems([
            "slow (preserve bitrate/accurate cut)",
            "fast (seek & cut; may vary bitrate)"
        ])
        self.comb_cut.setCurrentIndex(0)  # 기본 slow
        ge.addWidget(QLabel("Suffix:"), 0, 0)
        ge.addWidget(self.ed_suffix,    0, 1, 1, 3)
        ge.addWidget(self.comb_cut,     1, 0, 1, 3)
        ge.addWidget(self.btn_cut,      1, 3, 1, 1)

        # 3사분면: Playback + Bookmarks
        bottom_left = QWidget(); bl = QVBoxLayout(bottom_left); bl.setContentsMargins(0,0,0,0); bl.setSpacing(8)
        bl.addWidget(play_group)
        bl.addWidget(bm_group)
        bm_group.setMaximumHeight(160)
        G.addWidget(bottom_left, 1, 0)

        # 4사분면
        bottom_right = QWidget(); br = QVBoxLayout(bottom_right); br.setContentsMargins(0,0,0,0); br.setSpacing(8)
        br.addWidget(cut_group)
        br.addWidget(run_group)
        G.addWidget(bottom_right, 1, 1)

        # 스트레치: (행) 위쪽 크게, (열) 오른쪽(비디오) 크게
        G.setRowStretch(0, 1)   # 상단(파일+비디오) 비중
        G.setRowStretch(1, 0)   # 하단(플레이/북마크 + 컷) 비중
        G.setColumnStretch(0, 5)  # 좌(비디오/플레이)
        G.setColumnStretch(1, 1)  # 우(리스트/컷)

        # ---------- connections ----------
        self.btn_open.clicked.connect(self.open_folder)
        self.btn_load.clicked.connect(self.load_video)
        self.list_videos.itemDoubleClicked.connect(self.load_video)
        self.btn_play.clicked.connect(self.toggle_play)

        self.slider.sliderPressed.connect(self.on_slider_pressed)
        self.slider.sliderMoved.connect(self.on_slider_moved)
        self.slider.sliderReleased.connect(self.on_slider_released)

        self.speed.valueChanged.connect(self.change_speed)

        self.btn_bm_add.clicked.connect(self.add_bookmark)
        self.btn_bm_go.clicked.connect(self.goto_bookmark)
        self.btn_bm_del.clicked.connect(self.del_bookmark)

        # param toggles
        self._param_state = 1  # 1: start+duration, 2: duration+end, 3: start+end
        # 초기 상태 강제(혹시 UI 초기값이 어긋나 있어도 맞춤)
        QTimer.singleShot(0, lambda: self._apply_state(self._param_state))
        # 클릭(토글) 이벤트를 상태머신 입력으로 사용
        self._param_state = 1  # 1: Start+Duration, 2: Duration+End, 3: Start+End
        QTimer.singleShot(0, lambda: self._apply_state(self._param_state))
        self.combo_mode.currentIndexChanged.connect(self.on_mode_changed)

        # copy from current
        self.btn_start_from_cur.clicked.connect(lambda: self.set_from_current('start'))
        self.btn_end_from_cur.clicked.connect(lambda: self.set_from_current('end'))

        # cut
        self.btn_cut.clicked.connect(self.cut_video)
        # initial state
        self.update_enable_state(folder_loaded=False, video_loaded=False)

        # keyboard focus & no-focus
        self.setFocusPolicy(Qt.StrongFocus)
        self._apply_no_focus()

        self.statusBar().showMessage("")  # 빈 메시지로 바 생성

    # 비차단 상태 메시지 표시 유틸 (하단 status bar + Export 라벨 동시 갱신)
    def _set_export_status(self, text: str, tooltip: str = None, auto_clear_ms: int = 0):
        # 하단 전역 상태바
        self.statusBar().showMessage(text or "")
        # auto clear 요청 시 일정 시간 후 지우기
        if auto_clear_ms and auto_clear_ms > 0:
            QTimer.singleShot(auto_clear_ms, lambda: (self.statusBar().showMessage("")))

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

    def update_enable_state(self, folder_loaded: bool, video_loaded: bool):
        # defaults
        self.btn_open.setEnabled(True)
        self.list_videos.setEnabled(folder_loaded)
        self.btn_load.setEnabled(folder_loaded)

        # right side panels
        enable_right = video_loaded
        for w in (self.ed_start, self.btn_start_from_cur, self.ed_dur, self.unit_dur,
                  self.ed_end, self.btn_end_from_cur, self.ed_suffix, self.btn_cut, self.comb_cut,
                  self.combo_mode):
            w.setEnabled(enable_right)

        # playback + bookmarks
        for w in (self.btn_play, self.slider, self.speed,
                  self.btn_bm_add, self.btn_bm_go, self.btn_bm_del, self.bm_list):
            w.setEnabled(video_loaded)

        # enforce the 2-of-3 rule even when enabling
        self._apply_state(getattr(self, "_param_state", 1))

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
        self.update_enable_state(folder_loaded=True, video_loaded=False)

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
            return

        self.fps = self.thread.fps
        self.total_frames = self.thread.total
        self.slider.setRange(0, max(0, self.total_frames-1))
        self.slider.setValue(0)
        self.current_frame = 0
        self.is_playing = False
        self.btn_play.setText("Play")

        # connect signals
        self.thread.frameReady.connect(self.on_frame)
        self.thread.finished.connect(self.on_video_finished)
        self.thread.start()

        # auto show first frame
        self.thread.seek(0)
        self.thread.pause()

        self.update_labels()
        self.update_enable_state(folder_loaded=True, video_loaded=True)

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
        self.update_labels()

    # ------------------------------- cutter -------------------------------
    def _resolve_cut_params(self):
        """Return (start_frame, duration_frames) or (None, error_text)"""
        state = getattr(self, "_param_state", 1)  # 1=S+D, 2=D+E, 3=S+E
        have_s = state in (1, 3)
        have_d = state in (1, 2)
        have_e = state in (2, 3)
        # read inputs
        sf = int(self.ed_start.text()) if (self.ed_start.text() and have_s) else None
        ef = int(self.ed_end.text())   if (self.ed_end.text() and have_e) else None

        dur_val = float(self.ed_dur.value()) if have_d else None
        if dur_val is not None:
            unit = self.unit_dur.currentText()
            if unit == "seconds":
                df = int(round(dur_val * self.fps))
            elif unit == "minutes":
                df = int(round(dur_val * 60 * self.fps))
            else:  # frames
                df = int(round(dur_val))
        else:
            df = None

        # compute missing
        try:
            if have_s and have_e:
                if sf is None or ef is None:
                    return None, "Start/End must be provided."
                df = ef - sf
                if df <= 0: return None, "End must be larger than Start."
            elif have_s and have_d:
                if sf is None or df is None:
                    return None, "Start/Duration must be provided."
                ef = sf + df
            elif have_d and have_e:
                if df is None or ef is None:
                    return None, "Duration/End must be provided."
                sf = ef - df
            else:
                return None, "Exactly two parameters must be selected."
        except Exception as e:
            return None, f"Invalid input: {e}"

        # clamp
        sf = max(0, min(sf, max(0, self.total_frames-1)))
        ef = max(0, min(ef, max(0, self.total_frames-1)))
        if ef <= sf:
            return None, "Invalid range after clamp."
        df = ef - sf
        return (sf, df), None

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
        return bool(_find_ffmpeg())

    # ------------------------------ cutting ------------------------------
    def cut_video(self):
        if not self.video_path:
            QMessageBox.information(self, "Cut", "No video loaded.")
            return
        ffmpeg = self._find_ffmpeg()
        if not ffmpeg:
            QMessageBox.warning(
                self, "ffmpeg not found",
                "ffmpeg is required to cut without re-encoding.\n\n"
                "• If you ran it as an exe file: Check if the ffmpeg.exe file exists. \n\n"
                "• If you ran it as a python file: Add ffmpeg to your system PATH."
            )

        res, err = self._resolve_cut_params()
        if err:
            QMessageBox.warning(self, "Invalid parameters", err)
            return
        start_f, dur_f = res
        start_sec = start_f / self.fps if self.fps else 0.0
        dur_sec   = dur_f / self.fps if self.fps else 0.0

        folder = os.path.dirname(self.video_path)
        base, ext = os.path.splitext(os.path.basename(self.video_path))
        suffix = self.ed_suffix.text().strip() or "_cut"
        out_path = os.path.join(folder, f"{base}{suffix}{ext}")

        if os.path.exists(out_path):
            r = QMessageBox.question(
                self, "Overwrite?",
                f"File already exists:\n{out_path}\n\nOverwrite?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if r != QMessageBox.Yes:
                self._set_export_status("Export canceled (file exists).", auto_clear_ms=6000)
                return

        self.btn_cut.setEnabled(False)
        QApplication.processEvents()

        # ffmpeg command (fast cut: stream copy)
        fast_cmd = [
            ffmpeg,
            "-y",
            "-ss", f"{start_sec:.3f}",
            "-i", self.video_path,
            "-t", f"{dur_sec:.3f}",
            "-c", "copy",
            out_path
        ]
        # ffmpeg command 
        slow_cmd = [
            ffmpeg,
            "-y",
            "-i", self.video_path,
            "-ss", f"{start_sec:.3f}",
            "-t", f"{dur_sec:.3f}",
            "-c", "copy",
            out_path
        ]
        mode = self.comb_cut.currentText().lower()
        if mode.startswith("slow"):
            cmd = slow_cmd
        else:
            cmd = fast_cmd
        try:
            # run and show minimal feedback
            self.btn_cut.setEnabled(False)
            completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if completed.returncode != 0:
                self.btn_cut.setEnabled(True)
                QMessageBox.critical(self, "ffmpeg error", completed.stderr[-8000:])
                self.btn_cut.setEnabled(True)
                self._set_export_status("Export failed. See error dialog.", auto_clear_ms=6000)
                return
        except Exception as e:
            self.btn_cut.setEnabled(True)
            QMessageBox.critical(self, "Error", f"Failed to run ffmpeg:\n{e}")
            self.btn_cut.setEnabled(True)
            self._set_export_status("Export failed. See error dialog.", auto_clear_ms=6000)
            return

        self.btn_cut.setEnabled(True)
        # 비차단 완료 알림
        self._set_export_status(f"Saved: {out_path}", tooltip=out_path, auto_clear_ms=6000)
        self.btn_cut.setEnabled(True)

    # ------------------------------ close ------------------------------
    def closeEvent(self, e):
        if self.thread:
            self.thread.stop()
            self.thread.wait(300)
        super().closeEvent(e)


def main():
    app = QApplication(sys.argv)
    w = Cutter()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
