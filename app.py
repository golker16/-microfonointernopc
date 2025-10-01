# app.py
import sys, time, logging, math
from logging.handlers import RotatingFileHandler
from pathlib import Path
import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets
import qdarkstyle

# Loopback por altavoz (WASAPI) -> soundcard
import soundcard as sc
# Reproducción hacia el "mic virtual" (CABLE Input) -> sounddevice
import sounddevice as sd

APP_NAME = "PC Loopback → Virtual Mic"
LOG_DIR = Path("logs"); LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"

SUGGESTED_SR = 48000        # 48 kHz suele ser el estándar para CABLE/Voicemeeter
BLOCK = 960                  # ~20 ms a 48 kHz
GAIN = 1.0                   # ganancia global
FORCE_STEREO = True
LIMITER = True
FADE_MS = 120

# ---------- Logging ----------
logger = logging.getLogger(APP_NAME)
logger.setLevel(logging.DEBUG)
fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(threadName)s | %(name)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
fh = RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
fh.setFormatter(fmt); fh.setLevel(logging.DEBUG); logger.addHandler(fh)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(fmt); ch.setLevel(logging.INFO); logger.addHandler(ch)

logger.info(f"soundcard {getattr(sc,'__version__','?')}  sounddevice {sd.__version__}")

# ---------- Utilidades audio ----------
def sanitize(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(x, -1.0, 1.0)

def to_mono(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1 or arr.shape[1] == 1: return arr
    return arr.mean(axis=1, keepdims=True)

def match_channels(x: np.ndarray, ch: int) -> np.ndarray:
    if x.ndim == 1: x = x[:, None]
    if x.shape[1] == ch: return x
    if x.shape[1] == 1 and ch == 2: return np.repeat(x, 2, axis=1)
    if x.shape[1] == 2 and ch == 1: return to_mono(x)
    return x[:, :ch]

def rms_dbfs(x: np.ndarray, eps: float = 1e-12) -> float:
    if x.size == 0: return -120.0
    r = float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))
    return 20.0 * math.log10(max(r, eps))

def find_output_device(name_substr: str):
    name_sub = name_substr.lower()
    for dev in sd.query_devices():
        if dev.get("max_output_channels", 0) > 0 and name_sub in dev.get("name","").lower():
            return dev
    return None

# ---------- Worker ----------
class LoopbackWorker(QtCore.QThread):
    log   = QtCore.Signal(str)
    level = QtCore.Signal(float)

    def __init__(self, speaker_hint: str = ""):
        super().__init__()
        self.speaker_hint = speaker_hint.strip()
        self._stop = False

    def stop(self): self._stop = True

    def run(self):
        try:
            # 1) Elegir altavoz (BT/HDMI/lo que uses)
            spk = None
            if self.speaker_hint:
                for s in sc.all_speakers():
                    if self.speaker_hint.lower() in s.name.lower():
                        spk = s; break
            if spk is None:
                spk = sc.default_speaker()
            if spk is None:
                self.log.emit("[ERROR] No encontré altavoz por defecto.")
                return

            mic_loop = sc.get_microphone(spk.name, include_loopback=True)
            if mic_loop is None:
                self.log.emit(f"[ERROR] Ese altavoz no expone loopback: {spk.name}")
                return

            # 2) Elegir salida → CABLE Input
            out_dev = find_output_device("cable input")
            if out_dev is None:
                self.log.emit("[ERROR] No encontré 'CABLE Input (VB-Audio Virtual Cable)'. ¿Instalado/activado?")
                return

            ch_out = 2 if FORCE_STEREO else max(1, min(out_dev.get("max_output_channels",2), 2))
            sr = SUGGESTED_SR
            fade_left = int(sr * (FADE_MS/1000.0))

            # abrimos salida con callback pull
            def out_cb(outdata, frames, time_info, status):
                if self._stop: raise sd.CallbackStop()
                # el buffer lo llena el bucle principal; si está vacío, silencio
                if not hasattr(self, "_buf") or self._buf.shape[0] < frames:
                    outdata[:] = 0
                    return
                b = self._buf[:frames]
                self._buf = self._buf[frames:]
                if b.shape[0] < frames:
                    pad = np.zeros((frames - b.shape[0], b.shape[1]), dtype=np.float32)
                    b = np.vstack([b, pad])
                outdata[:] = b

            out_stream = sd.OutputStream(device=out_dev["name"], samplerate=sr, dtype="float32",
                                         channels=ch_out, blocksize=BLOCK, callback=out_cb)
            out_stream.__enter__()
            self.log.emit(f"Loopback de: {spk.name}  →  Salida: {out_dev['name']} (SR={sr}, block={BLOCK})")

            # 3) Bucle de captura (pull) con soundcard
            block_secs = max(BLOCK / float(sr), 0.01)
            self._buf = np.zeros((0, ch_out), dtype=np.float32)
            warm = 6  # descartar primeros bloques para evitar clicks
            with mic_loop.recorder(samplerate=sr) as rec:
                while not self._stop:
                    data = rec.record(numframes=int(block_secs*sr))
                    x = np.asarray(data, dtype=np.float32)
                    if x.ndim == 1: x = x[:, None]
                    if x.shape[1] != ch_out: x = match_channels(x, ch_out)
                    if warm > 0:
                        warm -= 1
                        x[:] = 0.0
                    # fade-in + ganancia + limiter
                    n = min(fade_left, x.shape[0])
                    if n > 0:
                        fade = np.linspace(0.0, 1.0, n, endpoint=True, dtype=np.float32)
                        x[:n, :] *= fade[:, None]
                        fade_left -= n
                    x = sanitize(x) * GAIN
                    if LIMITER: x = np.clip(x, -0.98, 0.98)

                    try: self.level.emit(rms_dbfs(x))
                    except Exception: pass

                    # append para que lo consuma el callback
                    self._buf = np.vstack([self._buf, x])
                    time.sleep(0.001)

            out_stream.__exit__(None, None, None)

        except Exception as e:
            self.log.emit(f"[ERROR] {repr(e)}")

# ---------- UI mínima (bandeja + medidor + logs) ----------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, speaker_hint: str = ""):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(560, 220)
        self.setWindowIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))

        self.worker = LoopbackWorker(speaker_hint=speaker_hint)

        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        v = QtWidgets.QVBoxLayout(central)

        self.info = QtWidgets.QLabel("Capturando audio del PC (loopback) → CABLE Input")
        self.info.setAlignment(QtCore.Qt.AlignCenter)
        self.level_lbl = QtWidgets.QLabel("Nivel: — dBFS")
        self.level_bar = QtWidgets.QProgressBar(); self.level_bar.setRange(0, 100); self.level_bar.setTextVisible(False)

        self.open_logs_btn = QtWidgets.QPushButton("Abrir logs")
        self.quit_btn = QtWidgets.QPushButton("Salir")

        grid = QtWidgets.QGridLayout()
        grid.addWidget(self.info, 0, 0, 1, 2)
        grid.addWidget(self.level_lbl, 1, 0)
        grid.addWidget(self.level_bar, 1, 1)
        v.addLayout(grid)

        h = QtWidgets.QHBoxLayout()
        h.addStretch(1); h.addWidget(self.open_logs_btn); h.addWidget(self.quit_btn)
        v.addLayout(h)

        self.footer = QtWidgets.QLabel("© 2025 Gabriel Golker")
        self.footer.setAlignment(QtCore.Qt.AlignCenter)
        v.addWidget(self.footer)

        # señales
        self.open_logs_btn.clicked.connect(self.open_logs)
        self.quit_btn.clicked.connect(self._quit)

        # worker
        self.worker.log.connect(self._append_log)
        self.worker.level.connect(self._update_level)

        # bandeja
        self._init_tray()

        # arranque automático
        QtCore.QTimer.singleShot(200, self._start)

    def _start(self):
        self.worker.start()
        self.tray.showMessage(APP_NAME, "Capturando en segundo plano.\nSelecciona 'CABLE Output' como micrófono en tu navegador.",
                              QtWidgets.QSystemTrayIcon.Information, 2500)

    def _append_log(self, msg: str):
        logging.getLogger(APP_NAME).info(msg)

    def _update_level(self, dbfs: float):
        clipped = max(min(dbfs, 0.0), -60.0)
        val = int((clipped + 60.0) * (100.0 / 60.0))
        self.level_bar.setValue(val)
        self.level_lbl.setText(f"Nivel: {dbfs:.1f} dBFS")

    def open_logs(self):
        try:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(LOG_FILE)))
        except Exception as e:
            logger.exception(f"No se pudieron abrir los logs: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"No se pudieron abrir los logs:\n{e}")

    # bandeja
    def _init_tray(self):
        self.tray = QtWidgets.QSystemTrayIcon(self)
        self.tray.setIcon(self.windowIcon()); self.tray.setVisible(True)
        menu = QtWidgets.QMenu()
        act_show = menu.addAction("Mostrar ventana")
        act_logs = menu.addAction("Abrir logs")
        act_quit = menu.addAction("Salir")
        act_show.triggered.connect(self._show_window)
        act_logs.triggered.connect(self.open_logs)
        act_quit.triggered.connect(self._quit)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(lambda r: self._show_window() if r==QtWidgets.QSystemTrayIcon.Trigger else None)
        self.tray.setToolTip(APP_NAME)

    def _show_window(self):
        self.showNormal(); self.activateWindow(); self.raise_()

    def _quit(self):
        try:
            if self.worker and self.worker.isRunning():
                self.worker.stop()
                self.worker.wait(1500)
        finally:
            QtWidgets.QApplication.quit()

    # cerrar → minimizar a bandeja
    def closeEvent(self, e: QtGui.QCloseEvent):
        e.ignore()
        self.hide()
        self.tray.showMessage(APP_NAME, "Sigo ejecutándome en segundo plano.", QtWidgets.QSystemTrayIcon.Information, 1600)

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyside6'))

    # OPCIONAL: pasar nombre parcial del altavoz por argv, ej:
    #   app.exe "LE-HD 458BT Stereo"
    speaker_hint = sys.argv[1] if len(sys.argv) > 1 else ""
    w = MainWindow(speaker_hint=speaker_hint)
    w.show()
    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())


