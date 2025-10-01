# app.py
import sys, time, logging, math
from logging.handlers import RotatingFileHandler
from pathlib import Path
from collections import deque
import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets
import qdarkstyle
import sounddevice as sd

APP_NAME = "VirtualMicRelay"
LOG_DIR = Path("logs"); LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"

# ====== Parámetros por defecto (el SR ahora es "sugerido") ======
SUGGESTED_SR  = 48000
DEFAULT_BLOCK = 960          # ≈20 ms @48k. Cambia en UI si hace falta.
FORCE_STEREO  = True
LIMITER       = True
FADE_MS       = 120
WARMUP_BLOCKS = 6

# ---------- Logging ----------
logger = logging.getLogger(APP_NAME)
logger.setLevel(logging.DEBUG)
fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(threadName)s | %(name)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
fh = RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
fh.setFormatter(fmt); fh.setLevel(logging.DEBUG); logger.addHandler(fh)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(fmt); ch.setLevel(logging.INFO); logger.addHandler(ch)

logger.info(f"sounddevice {sd.__version__}")

# ---------- Util ----------
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

def dump_devices_to_log():
    try:
        hostapis = sd.query_hostapis()
        logger.info(f"HostAPIs: {[h.get('name') for h in hostapis]}")
        for idx, api in enumerate(hostapis):
            logger.info(f"  [{idx}] {api.get('name')}")
        devs = sd.query_devices()
        logger.info(f"Dispositivos ({len(devs)})")
        for i, d in enumerate(devs):
            logger.info(
                f"  #{i} name='{d.get('name')}' hostapi={d.get('hostapi')} "
                f"in={d.get('max_input_channels')} out={d.get('max_output_channels')} "
                f"default_sr={d.get('default_samplerate')}"
            )
    except Exception as e:
        logger.exception(f"No pude listar dispositivos: {e}")

# ---------- Audio Worker (WASAPI loopback → CABLE Input) ----------
class AudioWorker(QtCore.QThread):
    log   = QtCore.Signal(str)
    level = QtCore.Signal(float)   # dBFS

    def __init__(self, out_device_name: str, block: int, mono: bool, sys_gain: float):
        super().__init__()
        self.out_device_name = out_device_name
        self.block = int(block)
        self.mono = bool(mono)
        self.sys_gain = float(sys_gain)
        self._stop = False

        # Diccionarios + índices reales de PortAudio
        self._in_dev = None;   self._in_idx = None
        self._out_dev = None;  self._out_idx = None

        self._sr_in_use = None
        self._fade_left = 0
        self._q = deque(maxlen=10)

    # ---- WASAPI helpers ----
    def _get_wasapi_index(self):
        for idx, api in enumerate(sd.query_hostapis()):
            if "wasapi" in api.get("name", "").lower():
                return idx
        return None

    def _find_devices(self):
        wasapi_index = self._get_wasapi_index()
        if wasapi_index is None:
            raise RuntimeError("No encontré host API WASAPI (requerido para loopback).")

        devices = sd.query_devices()

        # Salida destino (CABLE Input) por nombre → índice
        sel_out_idx = sel_out = None
        for i, dev in enumerate(devices):
            if dev.get("hostapi") == wasapi_index and dev.get("max_output_channels", 0) > 0:
                if self.out_device_name.lower() in dev.get("name", "").lower():
                    sel_out_idx, sel_out = i, dev; break
        if sel_out_idx is None:
            raise RuntimeError(f"No encontré salida WASAPI que contenga: {self.out_device_name}")
        self._out_idx, self._out_dev = sel_out_idx, sel_out

        # Fuente loopback: altavoz por defecto (salida) → índice
        default_out_idx = None
        try:
            if sd.default.device and sd.default.device[1] is not None:
                default_out_idx = sd.default.device[1]
        except Exception:
            default_out_idx = None

        sel_in_idx = sel_in = None
        if default_out_idx is not None:
            try:
                cand = sd.query_devices(default_out_idx)
                if cand.get("hostapi") == wasapi_index and cand.get("max_output_channels", 0) > 0:
                    sel_in_idx, sel_in = default_out_idx, cand
            except Exception:
                sel_in_idx = sel_in = None

        if sel_in_idx is None:
            for i, dev in enumerate(devices):
                if dev.get("hostapi") == wasapi_index and dev.get("max_output_channels", 0) > 0:
                    sel_in_idx, sel_in = i, dev; break
        if sel_in_idx is None:
            raise RuntimeError("No hay dispositivo de salida WASAPI para usar como loopback.")

        self._in_idx, self._in_dev = sel_in_idx, sel_in
        self.log.emit(f"Loopback de: {self._in_dev['name']}  →  Salida: {self._out_dev['name']}")

    # ---- Intento robusto de abrir streams con varias combinaciones ----
    def _open_streams_robusto(self):
        # Candidatos de SR: sugerido + defaults de dispositivos + fallback
        sr_candidates = []
        seen = set()
        for sr in [SUGGESTED_SR,
                   int(self._out_dev.get("default_samplerate") or 0),
                   int(self._in_dev.get("default_samplerate") or 0),
                   44100, 48000, 32000]:
            if sr and sr not in seen:
                seen.add(sr); sr_candidates.append(sr)

        # Modos: primero compartido, luego exclusivo
        mode_candidates = [
            dict(in_excl=False, out_excl=False),
            dict(in_excl=True,  out_excl=False),
            dict(in_excl=False, out_excl=True),
            dict(in_excl=True,  out_excl=True),
        ]

        # Canales
        ch_out = 2 if FORCE_STEREO else min(self._out_dev.get("max_output_channels", 2), 2)
        ch_in  = min(self._in_dev.get("max_output_channels", 2), 2)

        last_error = None
        for sr in sr_candidates:
            for mode in mode_candidates:
                try:
                    was_in  = sd.WasapiSettings(loopback=True,  exclusive=mode["in_excl"])
                    was_out = sd.WasapiSettings(exclusive=mode["out_excl"])

                    # warm-up counter y fade para cada inicio
                    warm = WARMUP_BLOCKS
                    self._fade_left = int(sr * (FADE_MS / 1000.0))
                    self._sr_in_use = sr

                    def in_cb(indata, frames, time_info, status):
                        nonlocal warm
                        if self._stop:
                            raise sd.CallbackStop()
                        if status:
                            self.log.emit(f"[IN-STATUS] {status}")
                        x = np.asarray(indata, dtype=np.float32)
                        if warm > 0:
                            warm -= 1
                            return
                        x = sanitize(x)
                        if self.mono:
                            x = to_mono(x)
                        if x.shape[1] != ch_out:
                            x = match_channels(x, ch_out)
                        x *= self.sys_gain
                        if LIMITER:
                            x = np.clip(x, -0.98, 0.98)
                        # fade-in
                        n = min(self._fade_left, x.shape[0])
                        if n > 0:
                            fade = np.linspace(0.0, 1.0, n, endpoint=True, dtype=np.float32)
                            if x.ndim == 1: x[:n] *= fade
                            else:           x[:n, :] *= fade[:, None]
                            self._fade_left -= n
                        # nivel:
                        try:
                            self.level.emit(rms_dbfs(x))
                        except Exception:
                            pass
                        self._q.append(x.copy())

                    def out_cb(outdata, frames, time_info, status):
                        if self._stop:
                            raise sd.CallbackStop()
                        if status:
                            self.log.emit(f"[OUT-STATUS] {status}")
                        if self._q:
                            b = self._q.popleft()
                            if b.shape[0] < frames:
                                pad = np.zeros((frames - b.shape[0], b.shape[1]), dtype=np.float32)
                                b = np.vstack([b, pad])
                            elif b.shape[0] > frames:
                                b = b[:frames]
                        else:
                            b = np.zeros((frames, ch_out), dtype=np.float32)
                        outdata[:] = b

                    # Abrimos ambos con contexto; si sale bien, devolvemos los context managers
                    in_stream = sd.InputStream(device=self._in_idx, samplerate=sr, dtype="float32",
                                               channels=ch_in, blocksize=self.block,
                                               callback=in_cb, extra_settings=was_in)
                    out_stream = sd.OutputStream(device=self._out_idx, samplerate=sr, dtype="float32",
                                                 channels=ch_out, blocksize=self.block,
                                                 callback=out_cb, extra_settings=was_out)
                    in_stream.__enter__()
                    out_stream.__enter__()
                    self.log.emit(f"OK con SR={sr}  in_excl={mode['in_excl']} out_excl={mode['out_excl']} ch_in={ch_in} ch_out={ch_out}")
                    return in_stream, out_stream
                except Exception as e:
                    last_error = repr(e)
                    self.log.emit(f"[INTENTO FALLÓ] SR={sr} in_excl={mode['in_excl']} out_excl={mode['out_excl']} → {last_error}")
                    # Cierra si quedó medio abierto (por seguridad)
                    try:
                        in_stream.__exit__(None, None, None)  # type: ignore
                    except Exception:
                        pass
                    try:
                        out_stream.__exit__(None, None, None) # type: ignore
                    except Exception:
                        pass

        raise RuntimeError(f"No pude abrir streams tras {len(sr_candidates)*len(mode_candidates)} intentos. Último error: {last_error}")

    def run(self):
        try:
            self._find_devices()
        except Exception as e:
            dump_devices_to_log()
            self.log.emit(f"[ERROR] Dispositivos: {e}")
            return

        try:
            in_stream, out_stream = self._open_streams_robusto()
        except Exception as e:
            dump_devices_to_log()
            self.log.emit(f"[ERROR] Inicio de audio: {e}")
            return

        try:
            with in_stream, out_stream:
                self.log.emit(f"Audio en marcha (SR={self._sr_in_use}, block={self.block}).")
                while not self._stop:
                    time.sleep(0.05)
        except Exception as e:
            dump_devices_to_log()
            self.log.emit(f"[ERROR] Audio en ejecución: {repr(e)}")

    def stop(self):
        self._stop = True

# ---------- Ventana ----------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(900, 560)
        self.setWindowIcon(QtGui.QIcon("assets/app.ico") if Path("assets/app.ico").exists()
                           else self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))

        self.worker: AudioWorker | None = None
        self.tray = None
        self.is_running = False

        central = QtWidgets.QWidget(); self.setCentralWidget(central)

        # Controles
        self.out_combo     = QtWidgets.QComboBox()     # destino → CABLE Input
        self.refresh_btn   = QtWidgets.QPushButton("Actualizar dispositivos")

        self.block_spin    = QtWidgets.QSpinBox();  self.block_spin.setRange(120, 4096); self.block_spin.setValue(DEFAULT_BLOCK)
        self.mono_chk      = QtWidgets.QCheckBox("Forzar mono"); self.mono_chk.setChecked(True)
        self.sys_gain      = QtWidgets.QDoubleSpinBox(); self.sys_gain.setRange(0.0, 5.0); self.sys_gain.setSingleStep(0.1); self.sys_gain.setValue(1.0)

        self.start_btn     = QtWidgets.QPushButton("Iniciar")
        self.stop_btn      = QtWidgets.QPushButton("Detener"); self.stop_btn.setEnabled(False)
        self.open_logs_btn = QtWidgets.QPushButton("Abrir logs")
        self.test_tone_btn = QtWidgets.QPushButton("Tono de prueba (440 Hz)")
        self.level_label   = QtWidgets.QLabel("Nivel: — dBFS")
        self.level_bar     = QtWidgets.QProgressBar(); self.level_bar.setRange(0, 100); self.level_bar.setValue(0)
        self.level_bar.setTextVisible(False)

        self.footer = QtWidgets.QLabel("© 2025 Gabriel Golker"); self.footer.setAlignment(QtCore.Qt.AlignCenter)

        # Layout
        form = QtWidgets.QFormLayout()
        form.addRow("Salida (→ CABLE Input):", self.out_combo)

        h1 = QtWidgets.QHBoxLayout()
        h1.addWidget(self.refresh_btn); h1.addStretch()
        h1.addWidget(QtWidgets.QLabel("Block:")); h1.addWidget(self.block_spin)

        h2 = QtWidgets.QHBoxLayout()
        h2.addWidget(self.mono_chk); h2.addStretch()
        h2.addWidget(QtWidgets.QLabel("Ganancia:")); h2.addWidget(self.sys_gain)

        h3 = QtWidgets.QHBoxLayout()
        h3.addWidget(self.start_btn); h3.addWidget(self.stop_btn); h3.addStretch()
        h3.addWidget(self.open_logs_btn); h3.addWidget(self.test_tone_btn)

        h4 = QtWidgets.QHBoxLayout()
        h4.addWidget(self.level_label); h4.addWidget(self.level_bar)

        v = QtWidgets.QVBoxLayout(central)
        v.addLayout(form); v.addSpacing(8)
        v.addLayout(h1); v.addLayout(h2); v.addSpacing(8)
        v.addLayout(h4); v.addSpacing(8)
        v.addLayout(h3); v.addStretch(); v.addWidget(self.footer)

        # Señales
        self.refresh_btn.clicked.connect(self.populate_devices)
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)
        self.open_logs_btn.clicked.connect(self.open_logs)
        self.test_tone_btn.clicked.connect(self._play_test_tone)

        # Bandeja
        self._init_tray()

        # Carga + autostart
        self.populate_devices()
        QtCore.QTimer.singleShot(400, self.autostart)

    def _get_wasapi_index(self):
        for idx, api in enumerate(sd.query_hostapis()):
            if "wasapi" in api.get("name", "").lower():
                return idx
        return None

    def populate_devices(self):
        try:
            self.out_combo.clear()
            names = []
            wasapi_index = self._get_wasapi_index()
            if wasapi_index is None:
                raise RuntimeError("No hay WASAPI disponible.")
            for i, dev in enumerate(sd.query_devices()):
                if dev.get("hostapi") == wasapi_index and dev.get("max_output_channels", 0) > 0:
                    names.append(f"#{i}  {dev['name']}")
            names = sorted(names, key=lambda n: 0 if "cable input" in n.lower() else 1)
            if not names:
                raise RuntimeError("No hay salidas WASAPI.")
            self.out_combo.addItems(names)
            logger.info("Dispositivos actualizados.")
        except Exception as e:
            logger.exception(f"No se pudieron listar dispositivos: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Error al listar dispositivos:\n{e}")

    def _extract_out_name(self):
        txt = self.out_combo.currentText()
        return txt.split("  ", 1)[1] if "  " in txt else txt

    def autostart(self):
        chosen_out = self._extract_out_name().lower()
        if "cable input" not in chosen_out:
            QtWidgets.QMessageBox.critical(self, "VB-CABLE no detectado",
                "Selecciona 'CABLE Input (VB-Audio Virtual Cable)' como salida.\n"
                "En el navegador/Discord elige 'CABLE Output' como micrófono.")
        self._on_start()

    def _on_start(self):
        try:
            if self.worker and self.worker.isRunning():
                QtWidgets.QMessageBox.information(self, "Info", "El audio ya está en ejecución.")
                return
            out_name = self._extract_out_name()
            self.worker = AudioWorker(
                out_device_name=out_name,
                block=self.block_spin.value(),
                mono=self.mono_chk.isChecked(),
                sys_gain=self.sys_gain.value(),
            )
            self.worker.log.connect(self._append_log)
            self.worker.level.connect(self._update_level)
            self.worker.start()
            QtCore.QTimer.singleShot(400, self._post_start_check)
        except Exception as e:
            logger.exception(f"Fallo al lanzar worker: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"No se pudo iniciar el audio:\n{e}")

    def _post_start_check(self):
        if self.worker and self.worker.isRunning():
            self.is_running = True
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self._tray_set_tooltip(True)
            logger.info("Audio iniciado (WASAPI loopback).")
        else:
            self.is_running = False
            self._tray_set_tooltip(False)
            QtWidgets.QMessageBox.critical(self, "Error", "No se pudo iniciar el audio. Revisa los logs (verás cada intento y el error exacto).")

    def _on_stop(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait(2000)
            self.worker = None
        self.is_running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._tray_set_tooltip(False)
        self._update_level(-120.0)
        logger.info("Audio detenido por el usuario.")

    def _update_level(self, dbfs: float):
        clipped = max(min(dbfs, 0.0), -60.0)
        val = int((clipped + 60.0) * (100.0 / 60.0))
        self.level_bar.setValue(val)
        self.level_label.setText(f"Nivel: {dbfs:.1f} dBFS")

    def _play_test_tone(self):
        # 440 Hz, -12 dBFS, 2 s
        sr = int(SUGGESTED_SR)
        dur = 2.0
        t = np.linspace(0, dur, int(sr*dur), endpoint=False, dtype=np.float32)
        amp = 10.0 ** (-12.0 / 20.0)
        tone = (amp * np.sin(2*np.pi*440.0*t)).astype(np.float32)
        if FORCE_STEREO: tone = np.stack([tone, tone], axis=1)
        try:
            sd.play(tone, samplerate=sr, blocking=False)  # al altavoz por defecto (para ver nivel)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"No pude reproducir tono de prueba:\n{e}")

    def open_logs(self):
        try:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(LOG_FILE)))
        except Exception as e:
            logger.exception(f"No se pudieron abrir los logs: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"No se pudieron abrir los logs:\n{e}")

    def _append_log(self, msg: str):
        logger.info(msg)

    # ---- Bandeja ----
    def _init_tray(self):
        self.tray = QtWidgets.QSystemTrayIcon(self)
        icon = QtGui.QIcon("assets/app.ico") if Path("assets/app.ico").exists() else self.windowIcon()
        self.tray.setIcon(icon); self.tray.setVisible(True); self._tray_set_tooltip(False)
        menu = QtWidgets.QMenu()
        self.action_show = menu.addAction("Mostrar ventana")
        self.action_toggle = menu.addAction("Iniciar/Detener")
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

    def _tray_show_window(self):
        self.showNormal(); self.activateWindow(); self.raise_()

    def _tray_toggle(self):
        if self.is_running: self._on_stop()
        else: self._on_start()

    def _tray_quit(self):
        try:
            if self.is_running: self._on_stop()
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
            self.tray.showMessage(APP_NAME, "Sigo ejecutándome en segundo plano.",
                                  QtWidgets.QSystemTrayIcon.Information, 2500)
        else:
            super().closeEvent(event)

def main():
    try:
        app = QtWidgets.QApplication(sys.argv)
        app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyside6'))
        w = MainWindow(); w.show()
        return app.exec()
    except Exception as e:
        logger.exception(f"Fallo crítico al iniciar la app: {e}")
        raise

if __name__ == "__main__":
    raise SystemExit(main())


