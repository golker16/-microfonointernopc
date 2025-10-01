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

SUGGESTED_SR  = 48000
DEFAULT_BLOCK = 960
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

# ---------- Utils ----------
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

def has_wasapi_loopback_support() -> bool:
    # En tu 0.5.2 lanza TypeError por 'loopback'
    try:
        _ = sd.WasapiSettings(exclusive=False)  # existe la clase
        # pero NO intentemos pasar loopback aquí, porque en tu versión no existe
        return False
    except Exception:
        return False

# ---------- Audio Worker ----------
class AudioWorker(QtCore.QThread):
    log   = QtCore.Signal(str)
    level = QtCore.Signal(float)

    def __init__(self, out_device_name: str, block: int, mono: bool, sys_gain: float):
        super().__init__()
        self.out_device_name = out_device_name
        self.block = int(block)
        self.mono = bool(mono)
        self.sys_gain = float(sys_gain)
        self._stop = False

        self._in_idx = None; self._in_dev = None
        self._out_idx = None; self._out_dev = None
        self._sr_in_use = None
        self._fade_left = 0
        self._q = deque(maxlen=10)

        self._use_wasapi_loopback = has_wasapi_loopback_support()

    def _get_wasapi_index(self):
        for idx, api in enumerate(sd.query_hostapis()):
            if "wasapi" in api.get("name","").lower():
                return idx
        return None

    def _find_cable_output_index(self):
        # Buscamos "CABLE Input" en cualquier hostapi que tenga salida
        for i, dev in enumerate(sd.query_devices()):
            if dev.get("max_output_channels", 0) > 0:
                if "cable input" in dev.get("name","").lower():
                    return i, dev
        return None, None

    def _find_stereo_mix_index(self):
        # Candidatos de mezcla física (input)
        keys = ["stereo mix", "mezcla estéreo", "what u hear", "loopback", "stereomix", "stereo-mix"]
        for i, dev in enumerate(sd.query_devices()):
            if dev.get("max_input_channels", 0) > 0:
                name = dev.get("name","").lower()
                if any(k in name for k in keys):
                    return i, dev
        return None, None

    def _find_default_output_index_any(self):
        # Por si hiciera falta (no usamos loopback en tu versión, pero lo dejo)
        try:
            if sd.default.device and sd.default.device[1] is not None:
                return sd.default.device[1], sd.query_devices(sd.default.device[1])
        except Exception:
            pass
        # fallback: primera salida que encontremos
        for i, dev in enumerate(sd.query_devices()):
            if dev.get("max_output_channels", 0) > 0:
                return i, dev
        return None, None

    def _choose_devices(self):
        # Salida: CABLE Input
        out_idx, out_dev = self._find_cable_output_index()
        if out_idx is None:
            raise RuntimeError("No encontré 'CABLE Input (VB-Audio Virtual Cable)'. Instálalo y reinicia Windows.")
        self._out_idx, self._out_dev = out_idx, out_dev

        # Entrada:
        if self._use_wasapi_loopback:
            # (no se usará en tu 0.5.2, pero dejamos el camino)
            idx, dev = self._find_default_output_index_any()
            if idx is None:
                raise RuntimeError("No encontré salida por defecto para usar como loopback.")
            self._in_idx, self._in_dev = idx, dev
            self.log.emit(f"Loopback (WASAPI) de salida idx #{idx}: {dev['name']}")
        else:
            # Usar fuente física tipo "Mezcla estéreo"
            idx, dev = self._find_stereo_mix_index()
            if idx is None:
                raise RuntimeError("No encontré 'Stereo Mix / Mezcla estéreo'. Actívalo en Panel de control → Sonido → Grabación.")
            self._in_idx, self._in_dev = idx, dev
            self.log.emit(f"Fuente: {dev['name']}  →  Salida: {out_dev['name']}")

    def _open_streams(self):
        # Negociar SR: sugerido → default de in/out → 44100 → 48000
        sr_candidates = []
        seen = set()
        for sr in [SUGGESTED_SR,
                   int(self._in_dev.get("default_samplerate") or 0),
                   int(self._out_dev.get("default_samplerate") or 0),
                   44100, 48000]:
            if sr and sr not in seen:
                seen.add(sr); sr_candidates.append(sr)

        ch_out = 2 if FORCE_STEREO else min(self._out_dev.get("max_output_channels",2), 2)
        # Si usamos Stereo Mix como **input**, sus canales son "max_input_channels"
        ch_in  = min(self._in_dev.get("max_input_channels", 2), 2) if not self._use_wasapi_loopback \
                 else min(self._in_dev.get("max_output_channels", 2), 2)

        last_error = None
        for sr in sr_candidates:
            try:
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
                    if self.mono: x = to_mono(x)
                    if x.shape[1] != ch_out: x = match_channels(x, ch_out)
                    x *= self.sys_gain
                    if LIMITER: x = np.clip(x, -0.98, 0.98)
                    n = min(self._fade_left, x.shape[0])
                    if n > 0:
                        fade = np.linspace(0.0, 1.0, n, endpoint=True, dtype=np.float32)
                        if x.ndim == 1: x[:n] *= fade
                        else: x[:n, :] *= fade[:, None]
                        self._fade_left -= n
                    try: self.level.emit(rms_dbfs(x))
                    except Exception: pass
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

                # extra_settings solo si tuviera WASAPI loopback (no en tu caso)
                extra_in  = None
                extra_out = None
                if self._use_wasapi_loopback:
                    try:
                        extra_in  = sd.WasapiSettings(exclusive=False)  # sin 'loopback', no disponible en tu build
                        extra_out = sd.WasapiSettings(exclusive=False)
                    except Exception:
                        extra_in = extra_out = None

                in_stream = sd.InputStream(device=self._in_idx, samplerate=sr, dtype="float32",
                                           channels=ch_in, blocksize=self.block,
                                           callback=in_cb, extra_settings=extra_in)
                out_stream = sd.OutputStream(device=self._out_idx, samplerate=sr, dtype="float32",
                                             channels=ch_out, blocksize=self.block,
                                             callback=out_cb, extra_settings=extra_out)
                in_stream.__enter__(); out_stream.__enter__()
                self.log.emit(f"OK SR={sr} ch_in={ch_in} ch_out={ch_out}")
                return in_stream, out_stream
            except Exception as e:
                last_error = repr(e)
                self.log.emit(f"[INTENTO FALLÓ] SR={sr} → {last_error}")
                try: in_stream.__exit__(None, None, None)  # type: ignore
                except Exception: pass
                try: out_stream.__exit__(None, None, None) # type: ignore
                except Exception: pass

        raise RuntimeError(f"No pude abrir streams. Último error: {last_error}")

    def run(self):
        try:
            self._choose_devices()
        except Exception as e:
            dump_devices_to_log()
            self.log.emit(f"[ERROR] Dispositivos: {e}")
            return

        try:
            in_stream, out_stream = self._open_streams()
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

# ---------- UI ----------
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

        self.out_combo     = QtWidgets.QComboBox()
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

        self.refresh_btn.clicked.connect(self.populate_devices)
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)
        self.open_logs_btn.clicked.connect(self.open_logs)
        self.test_tone_btn.clicked.connect(self._play_test_tone)

        self._init_tray()

        self.populate_devices()
        QtCore.QTimer.singleShot(400, self.autostart)

    def populate_devices(self):
        try:
            self.out_combo.clear()
            names = []
            for i, dev in enumerate(sd.query_devices()):
                if dev.get("max_output_channels", 0) > 0:
                    names.append(f"#{i}  {dev['name']}")
            names = sorted(names, key=lambda n: 0 if "cable input" in n.lower() else 1)
            if not names:
                raise RuntimeError("No hay salidas de audio en el sistema.")
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
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self._tray_set_tooltip(True)
            logger.info("Audio iniciado.")
        else:
            self._tray_set_tooltip(False)
            QtWidgets.QMessageBox.critical(self, "Error", "No se pudo iniciar el audio. Revisa logs.")

    def _on_stop(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait(2000)
            self.worker = None
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
        # Reproduce 440 Hz al altavoz por defecto (para ver subir el medidor si la fuente es Stereo Mix)
        sr = 44100
        dur = 2.0
        t = np.linspace(0, dur, int(sr*dur), endpoint=False, dtype=np.float32)
        amp = 10.0 ** (-12.0 / 20.0)
        tone = (amp * np.sin(2*np.pi*440.0*t)).astype(np.float32)
        if FORCE_STEREO: tone = np.stack([tone, tone], axis=1)
        try:
            sd.play(tone, samplerate=sr, blocking=False)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"No pude reproducir tono de prueba:\n{e}")

    def open_logs(self):
        try:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(LOG_FILE)))
        except Exception as e:
            logger.exception(f"No se pudieron abrir los logs: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"No se pudieron abrir los logs:\n{e}")

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
        if self.stop_btn.isEnabled(): self._on_stop()
        else: self._on_start()

    def _tray_quit(self):
        try:
            if self.stop_btn.isEnabled(): self._on_stop()
        finally:
            QtWidgets.QApplication.quit()

    def _tray_activated(self, reason):
        if reason == QtWidgets.QSystemTrayIcon.Trigger:
            self._tray_show_window()

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

