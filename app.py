# app.py
import os
import sys
import time
import threading
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# --- Parche compat NumPy 2.x (soundcard usa fromstring binario) ----------------
import numpy as np
_original_fromstring = np.fromstring
def _compat_fromstring(s, dtype=float, count=-1, sep=''):
    if isinstance(s, (bytes, bytearray, memoryview)) and (sep == '' or sep is None):
        return np.frombuffer(s, dtype=dtype, count=count if count != -1 else -1)
    return _original_fromstring(s, dtype=dtype, count=count, sep=sep)
np.fromstring = _compat_fromstring
# ------------------------------------------------------------------------------

# Backends de audio
try:
    import soundcard as sc
except Exception:
    sc = None

try:
    import sounddevice as sd
except Exception:
    sd = None

from PySide6 import QtCore, QtGui, QtWidgets
import qdarkstyle

APP_NAME = "VirtualMicRelay"
# === Defaults alineados con tu otro proyecto ===
DEFAULT_SR = 32000         # 32 kHz como CallCenter Helper
DEFAULT_BLOCK = 640        # ~20 ms a 32 kHz
WARMUP_BLOCKS = 6
FADE_MS = 120

LOG_DIR = Path("logs"); LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"

# ---------- Logging ----------
logger = logging.getLogger(APP_NAME)
logger.setLevel(logging.DEBUG)
fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(threadName)s | %(name)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
fh = RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
fh.setFormatter(fmt); fh.setLevel(logging.DEBUG); logger.addHandler(fh)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(fmt); ch.setLevel(logging.INFO); logger.addHandler(ch)

logger.info(f"NumPy={np.__version__}  soundcard={getattr(sc, '__version__', 'N/A')}  sounddevice={'OK' if sd else 'N/A'}")

# ---------- Utils ----------
def sanitize(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(x, -1.0, 1.0)

def make_fade_in(num_frames: int, channels: int) -> np.ndarray:
    if num_frames <= 0: return None
    fade = np.linspace(0.0, 1.0, num_frames, endpoint=True, dtype=np.float32)
    return fade[:, None].repeat(channels, axis=1)

def to_mono(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1 or arr.shape[1] == 1: return arr
    return arr.mean(axis=1, keepdims=True)

# ---------- Audio Worker ----------
class AudioWorker(threading.Thread):
    """
    Captura loopback del ALTAVOZ POR DEFECTO del sistema (igual que tu proyecto)
    y lo envía en tiempo real al altavoz virtual 'CABLE Input'.
    Si soundcard falla, intenta fallback con sounddevice (entrada por defecto).
    """
    def __init__(self, *, out_name, mix_mic_name, sr, block, mono, sys_gain, mic_gain, limiter, monitor_local=False):
        super().__init__(daemon=True, name="AudioWorker")
        self.stop_event = threading.Event()
        self.out_name = out_name
        self.mix_mic_name = mix_mic_name
        self.sr = int(sr)
        self.block = int(block)
        self.mono = bool(mono)
        self.sys_gain = float(sys_gain)
        self.mic_gain = float(mic_gain)
        self.limiter = bool(limiter)
        self.monitor_local = bool(monitor_local)
        self._use_fallback = False

    # ---- Helpers soundcard ----
    def _find_speaker(self, name_substring):
        if sc is None:
            raise RuntimeError("soundcard no está disponible.")
        spks = sc.all_speakers()
        logger.debug(f"Speakers: {[s.name for s in spks]}")
        for s in spks:
            if name_substring.lower() in s.name.lower():
                return s
        raise RuntimeError(f"No se encontró salida con nombre que contenga: '{name_substring}'")

    def _get_default_loopback(self):
        """Loopback del ALTAVOZ POR DEFECTO del sistema, como en tu proyecto."""
        if sc is None:
            raise RuntimeError("soundcard no disponible para loopback.")
        spk = sc.default_speaker()
        mic = sc.get_microphone(spk.name, include_loopback=True)
        if mic is None:
            raise RuntimeError(f"No pude obtener loopback de: {spk.name}")
        return mic, spk.name

    def _get_physical_mic(self, name_substring):
        if not name_substring or sc is None: return None
        mics = sc.all_microphones(include_loopback=True)
        logger.debug(f"Microphones: {[m.name for m in mics]}")
        for m in mics:
            if name_substring.lower() in m.name.lower():
                return m
        raise RuntimeError(f"No se encontró micrófono que contenga: '{name_substring}'")

    def _apply_limiter(self, x: np.ndarray, ceiling: float = 0.98) -> np.ndarray:
        return np.clip(x, -ceiling, ceiling)

    # ---- Fallback con sounddevice (si no hay loopback posible) ----
    def _fallback_stream(self):
        """Captura desde el dispositivo de entrada por defecto con sounddevice."""
        if sd is None:
            raise RuntimeError("Fallback requerido pero sounddevice no está disponible.")
        logger.warning("Usando fallback con sounddevice (entrada por defecto). No es loopback del sistema.")

        # Intentamos reproducir al CABLE Input con soundcard; si no, salida por sounddevice.
        player_ctx = None
        player = None
        try:
            out_spk = self._find_speaker(self.out_name)
            player_ctx = out_spk.player(samplerate=self.sr, blocksize=self.block)
            player_ctx.__enter__()
            player = player_ctx
        except Exception as e:
            logger.error(f"No pude abrir salida con soundcard: {e}. Reproduciré con sounddevice.")
            player_ctx = None
            player = None

        def callback(indata, frames, time_info, status):
            if self.stop_event.is_set():
                raise sd.CallbackStop()
            x = np.asarray(indata, dtype=np.float32)
            if self.mono:
                x = to_mono(x)
            x *= self.sys_gain
            if self.limiter:
                x = self._apply_limiter(x)
            if player is not None:
                player.play(x)
            else:
                sd.play(x, samplerate=self.sr, blocking=False)

        with sd.InputStream(samplerate=self.sr, channels=1 if self.mono else 2,
                            dtype="float32", blocksize=self.block, callback=callback):
            while not self.stop_event.is_set():
                time.sleep(0.05)
        if player_ctx:
            player_ctx.__exit__(None, None, None)

    # ---- Loop principal ----
    def run(self):
        try:
            logger.info("Inicializando captura y reproducción…")

            # Intenta abrir loopback del altavoz por defecto + salida con soundcard
            try:
                sys_mic, src_spk_name = self._get_default_loopback()
                out_spk = self._find_speaker(self.out_name)
                phys_mic = self._get_physical_mic(self.mix_mic_name)
                logger.info(f"Loopback de (por defecto): {src_spk_name}")
                logger.info(f"Salida destino  : {out_spk.name}")
                logger.info(f"Mic físico mix  : {phys_mic.name if phys_mic else '(ninguno)'}")
                logger.info(f"SR={self.sr} block={self.block} mono={self.mono} "
                            f"sys_gain={self.sys_gain} mic_gain={self.mic_gain} limiter={self.limiter}")
            except Exception as e:
                logger.error(f"No se pudo preparar loopback con soundcard: {e}")
                self._use_fallback = True

            if self._use_fallback:
                self._fallback_stream()
                logger.info("Audio detenido (fallback).")
                return

            mon_spk = sc.default_speaker() if self.monitor_local else None

            with sys_mic.recorder(samplerate=self.sr, blocksize=self.block) as sys_rec, \
                 (phys_mic.recorder(samplerate=self.sr, blocksize=self.block) if phys_mic else _NullCtx()) as mic_rec, \
                 out_spk.player(samplerate=self.sr, blocksize=self.block) as player, \
                 (mon_spk.player(samplerate=self.sr, blocksize=self.block) if mon_spk else _NullCtx()) as mon_player:

                # Warm-up
                for _ in range(WARMUP_BLOCKS):
                    _ = sys_rec.record(self.block)

                fade_frames = int(self.sr * (FADE_MS / 1000.0))
                fade_left = fade_frames

                while not self.stop_event.is_set():
                    sys_frames = sys_rec.record(self.block)  # (N, ch) float32
                    sys_frames = sanitize(sys_frames)
                    if self.mono:
                        sys_frames = to_mono(sys_frames)
                    sys_frames *= self.sys_gain

                    # Mezcla mic físico si aplica
                    if mic_rec:
                        mic_frames = mic_rec.record(self.block)
                        mic_frames = sanitize(mic_frames)
                        if self.mono:
                            mic_frames = to_mono(mic_frames)
                        if sys_frames.shape[1] != mic_frames.shape[1]:
                            if sys_frames.shape[1] == 1 and mic_frames.shape[1] == 2:
                                sys_frames = np.repeat(sys_frames, 2, axis=1)
                            elif sys_frames.shape[1] == 2 and mic_frames.shape[1] == 1:
                                mic_frames = np.repeat(mic_frames, 2, axis=1)
                        mix = sys_frames + mic_frames * self.mic_gain
                    else:
                        mix = sys_frames

                    # Fade-in inicial
                    if fade_left > 0:
                        n = min(fade_left, mix.shape[0])
                        fade = make_fade_in(n, mix.shape[1])
                        mix[:n, :] *= fade
                        fade_left -= n

                    # Limiter suave
                    mix = np.clip(mix, -0.98, 0.98) if self.limiter else mix

                    player.play(mix)
                    if mon_player:
                        mon_player.play(mix)

            logger.info("Audio detenido.")
        except Exception as e:
            logger.exception(f"ERROR en audio loop: {e}")

    def stop(self):
        self.stop_event.set()

class _NullCtx:
    def __enter__(self): return None
    def __exit__(self, exc_type, exc, tb): return False

# ---------- GUI ----------
class MainWindow(QtWidgets.QMainWindow):
    startRequested = QtCore.Signal(dict)
    stopRequested  = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(840, 560)
        self.setWindowIcon(QtGui.QIcon("assets/app.ico") if Path("assets/app.ico").exists()
                           else self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))

        self.worker: AudioWorker | None = None
        self.tray = None
        self.is_running = False

        central = QtWidgets.QWidget(); self.setCentralWidget(central)

        # Controles (simplificados: sin seleccionar fuente, siempre default)
        self.out_combo   = QtWidgets.QComboBox()   # destino → CABLE Input
        self.mic_combo   = QtWidgets.QComboBox()   # mic físico (opcional)
        self.refresh_btn = QtWidgets.QPushButton("Actualizar dispositivos")

        self.sr_spin    = QtWidgets.QSpinBox(); self.sr_spin.setRange(8000, 192000); self.sr_spin.setValue(DEFAULT_SR)
        self.block_spin = QtWidgets.QSpinBox(); self.block_spin.setRange(120, 4096); self.block_spin.setValue(DEFAULT_BLOCK)
        self.mono_chk   = QtWidgets.QCheckBox("Forzar mono"); self.mono_chk.setChecked(True)   # como tu proyecto
        self.limiter_chk= QtWidgets.QCheckBox("Limitador"); self.limiter_chk.setChecked(True)

        self.sys_gain_d = QtWidgets.QDoubleSpinBox(); self.sys_gain_d.setRange(0.0, 5.0); self.sys_gain_d.setSingleStep(0.1); self.sys_gain_d.setValue(1.0)
        self.mic_gain_d = QtWidgets.QDoubleSpinBox(); self.mic_gain_d.setRange(0.0, 5.0); self.mic_gain_d.setSingleStep(0.1); self.mic_gain_d.setValue(1.0)

        self.start_btn   = QtWidgets.QPushButton("Iniciar")
        self.stop_btn    = QtWidgets.QPushButton("Detener"); self.stop_btn.setEnabled(False)
        self.open_logs_btn = QtWidgets.QPushButton("Abrir logs")

        self.footer = QtWidgets.QLabel("© 2025 Gabriel Golker"); self.footer.setAlignment(QtCore.Qt.AlignCenter)

        # Layout
        form = QtWidgets.QFormLayout()
        form.addRow("Salida (→ CABLE Input):", self.out_combo)
        form.addRow("Mic físico (opcional):", self.mic_combo)

        h1 = QtWidgets.QHBoxLayout()
        h1.addWidget(self.refresh_btn); h1.addStretch()
        h1.addWidget(QtWidgets.QLabel("Sample Rate:")); h1.addWidget(self.sr_spin)
        h1.addWidget(QtWidgets.QLabel("Block:")); h1.addWidget(self.block_spin)

        h2 = QtWidgets.QHBoxLayout()
        h2.addWidget(self.mono_chk); h2.addWidget(self.limiter_chk); h2.addStretch()
        h2.addWidget(QtWidgets.QLabel("Ganancia sistema:")); h2.addWidget(self.sys_gain_d)
        h2.addWidget(QtWidgets.QLabel("Ganancia mic:")); h2.addWidget(self.mic_gain_d)

        h3 = QtWidgets.QHBoxLayout()
        h3.addWidget(self.start_btn); h3.addWidget(self.stop_btn); h3.addStretch(); h3.addWidget(self.open_logs_btn)

        v = QtWidgets.QVBoxLayout(central)
        v.addLayout(form); v.addSpacing(8)
        v.addLayout(h1); v.addLayout(h2); v.addSpacing(12)
        v.addLayout(h3); v.addStretch(); v.addWidget(self.footer)

        # Señales
        self.refresh_btn.clicked.connect(self.populate_devices)
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)
        self.open_logs_btn.clicked.connect(self.open_logs)
        self.startRequested.connect(self.start_worker)
        self.stopRequested.connect(self.stop_worker)

        # Bandeja
        self._init_tray()

        # Carga inicial + autostart
        self.populate_devices()
        QtCore.QTimer.singleShot(500, self.autostart)

    def populate_devices(self):
        try:
            # Salidas (destino → CABLE Input)
            self.out_combo.clear()
            outs = [s.name for s in sc.all_speakers()] if sc else []
            outs_sorted = sorted(outs, key=lambda n: 0 if "cable input" in n.lower() else 1)
            self.out_combo.addItems(outs_sorted)

            # Mics físicos
            self.mic_combo.clear(); self.mic_combo.addItem("(ninguno)")
            if sc:
                mics = [m.name for m in sc.all_microphones(include_loopback=True)]
                self.mic_combo.addItems(mics)

            logger.info("Dispositivos actualizados.")
        except Exception as e:
            logger.exception(f"No se pudieron listar dispositivos: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Error al listar dispositivos:\n{e}")

    def autostart(self):
        chosen_out = self.out_combo.currentText()
        if "cable input" not in (chosen_out or "").lower():
            QtWidgets.QMessageBox.critical(self, "VB-CABLE no detectado",
                "No encontré 'CABLE Input (VB-Audio Virtual Cable)' como salida.\n"
                "Instálalo y reinicia Windows, o selecciónalo en 'Salida' manualmente.")
        self._on_start()

    def _on_start(self):
        cfg = {
            "out_name": self.out_combo.currentText(),
            "mix_mic_name": None if self.mic_combo.currentIndex() == 0 else self.mic_combo.currentText(),
            "sr": self.sr_spin.value(),
            "block": self.block_spin.value(),
            "mono": self.mono_chk.isChecked(),        # True por defecto
            "sys_gain": self.sys_gain_d.value(),
            "mic_gain": self.mic_gain_d.value(),
            "limiter": self.limiter_chk.isChecked(),
            "monitor_local": False,                    # evita feedback
        }
        logger.info("Start solicitado por la app (auto/usuario).")
        self.startRequested.emit(cfg)

    def _on_stop(self):
        self.stopRequested.emit()

    @QtCore.Slot(dict)
    def start_worker(self, cfg):
        try:
            if self.worker and self.worker.is_alive():
                QtWidgets.QMessageBox.information(self, "Info", "El audio ya está en ejecución.")
                return
            self.worker = AudioWorker(**cfg)
            self.worker.start()
            time.sleep(0.25)
            self.is_running = self.worker.is_alive()
            if not self.is_running:
                logger.error("El worker no quedó en ejecución. Revisa logs.")
                QtWidgets.QMessageBox.critical(self, "Error", "No se pudo iniciar el audio. Revisa los logs.")
                self.worker = None; return
            self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True)
            self._tray_set_tooltip(running=True)
            logger.info("Audio iniciado.")
        except Exception as e:
            logger.exception(f"Fallo al lanzar worker: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"No se pudo iniciar el audio:\n{e}")

    @QtCore.Slot()
    def stop_worker(self):
        if self.worker:
            self.worker.stop()
            self.worker.join(timeout=2.0)
            self.worker = None
        self.is_running = False
        self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        self._tray_set_tooltip(running=False)
        logger.info("Audio detenido por el usuario.")

    def open_logs(self):
        try:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(LOG_FILE)))
        except Exception as e:
            logger.exception(f"No se pudieron abrir logs: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"No se pudieron abrir los logs:\n{e}")

    # --- Bandeja ---
    def _init_tray(self):
        self.tray = QtWidgets.QSystemTrayIcon(self)
        icon = QtGui.QIcon("assets/app.ico") if Path("assets/app.ico").exists() else self.windowIcon()
        self.tray.setIcon(icon); self.tray.setVisible(True); self._tray_set_tooltip(False)
        menu = QtWidgets.QMenu()
        self.action_show = menu.addAction("Mostrar ventana")
        self.action_toggle = menu.addAction("Iniciar")
        self.action_logs = menu.addAction("Abrir logs")
        menu.addSeparator()
        self.action_quit = menu.addAction("Salir")
        self.action_show.triggered.connect(self._tray_show_window)
        self.action_toggle.triggered.connect(self._tray_toggle)
        self.action_logs.triggered.connect(self.open_logs)
        self.action_quit.triggered.connect(self._tray_quit)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._tray_activated)

    def _tray_set_tooltip(self, running: bool):
        state = "En ejecución" if running else "Detenido"
        self.tray.setToolTip(f"{APP_NAME} - {state}")
        if self.tray.contextMenu():
            self.tray.contextMenu().actions()[1].setText("Detener" if running else "Iniciar")

    def _tray_show_window(self):
        self.showNormal(); self.activateWindow(); self.raise_()

    def _tray_toggle(self):
        if self.is_running: self._on_stop()
        else: self._on_start()

    def _tray_quit(self):
        try:
            if self.is_running: self.stop_worker()
        finally:
            QtWidgets.QApplication.quit()

    def _tray_activated(self, reason):
        if reason == QtWidgets.QSystemTrayIcon.Trigger:
            self._tray_show_window()

    # Cerrar → minimizar a bandeja
    def closeEvent(self, event: QtGui.QCloseEvent):
        if self.tray and self.tray.isVisible():
            event.ignore()
            self.hide()
            self.tray.showMessage(APP_NAME, "Sigo ejecutándome en segundo plano.", QtWidgets.QSystemTrayIcon.Information, 2500)
        else:
            super().closeEvent(event)

def main():
    try:
        app = QtWidgets.QApplication(sys.argv)
        app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyside6'))
        mw = MainWindow(); mw.show()
        sys.exit(app.exec())
    except Exception as e:
        logger.exception(f"Fallo crítico al iniciar la app: {e}")
        raise

if __name__ == "__main__":
    main()



