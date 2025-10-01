# app.py
import sys, time, logging, math, json
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
CONFIG_FILE = Path("config.json")

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

# ---------- Config ----------
DEFAULT_CFG = {
    "input_name": "",   # nombre visible de la FUENTE a usar (p.ej. "Mezcla estéreo ...")
    "output_name": "",  # nombre visible de la SALIDA (p.ej. "CABLE Input ...")
    "block": DEFAULT_BLOCK,
    "mono": True,
    "gain": 1.0
}

def load_cfg():
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            return {**DEFAULT_CFG, **data}
        except Exception as e:
            logger.error(f"No pude leer config.json: {e}")
    CONFIG_FILE.write_text(json.dumps(DEFAULT_CFG, indent=2), encoding="utf-8")
    return DEFAULT_CFG.copy()

def save_cfg(cfg: dict):
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

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

def find_device_by_name(name_substr: str, need_input: bool | None) -> tuple[int|None, dict|None]:
    """
    Busca por substring (case-insensitive) en nombres. Si need_input=True -> dispositivos con entrada.
    Si need_input=False -> dispositivos con salida. Si None -> cualquiera.
    Retorna (index, device_dict) o (None, None).
    """
    name_sub = (name_substr or "").strip().lower()
    if not name_sub:
        return None, None
    for i, dev in enumerate(sd.query_devices()):
        ok = True
        if need_input is True and dev.get("max_input_channels",0) <= 0: ok = False
        if need_input is False and dev.get("max_output_channels",0) <= 0: ok = False
        if not ok: continue
        if name_sub in dev.get("name","").lower():
            return i, dev
    return None, None

def find_best_stereo_mix() -> tuple[int|None, dict|None]:
    keys = ["stereo mix", "mezcla estéreo", "what u hear", "stereomix", "stereo-mix"]
    for i, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels",0) > 0:
            name = dev.get("name","").lower()
            if any(k in name for k in keys):
                return i, dev
    return None, None

# ---------- Audio Worker ----------
class AudioWorker(QtCore.QThread):
    log   = QtCore.Signal(str)
    level = QtCore.Signal(float)

    def __init__(self, in_name: str, out_name: str, block: int, mono: bool, sys_gain: float):
        super().__init__()
        self.in_name = in_name
        self.out_name = out_name
        self.block = int(block)
        self.mono = bool(mono)
        self.sys_gain = float(sys_gain)
        self._stop = False

        self._in_idx = None; self._in_dev = None
        self._out_idx = None; self._out_dev = None
        self._sr_in_use = None
        self._fade_left = 0
        self._q = deque(maxlen=10)

    def _choose_devices(self):
        # SALIDA
        out_idx, out_dev = None, None
        if self.out_name:
            out_idx, out_dev = find_device_by_name(self.out_name, need_input=False)
        if out_idx is None:
            # preferimos CABLE Input
            out_idx, out_dev = find_device_by_name("cable input", need_input=False)
        if out_idx is None:
            # lo primero que tenga salida
            for i, dev in enumerate(sd.query_devices()):
                if dev.get("max_output_channels",0) > 0:
                    out_idx, out_dev = i, dev; break
        if out_idx is None:
            raise RuntimeError("No hay dispositivo de SALIDA disponible.")
        self._out_idx, self._out_dev = out_idx, out_dev

        # ENTRADA
        in_idx, in_dev = None, None
        if self.in_name:
            in_idx, in_dev = find_device_by_name(self.in_name, need_input=True)
        if in_idx is None:
            # intentar Mezcla estéreo
            in_idx, in_dev = find_best_stereo_mix()
        if in_idx is None:
            # cualquier input
            for i, dev in enumerate(sd.query_devices()):
                if dev.get("max_input_channels",0) > 0:
                    in_idx, in_dev = i, dev; break
        if in_idx is None:
            raise RuntimeError("No hay dispositivo de ENTRADA disponible.")
        self._in_idx, self._in_dev = in_idx, in_dev

        self.log.emit(f"Fuente: {self._in_dev['name']}  →  Salida: {self._out_dev['name']}")

    def _open_streams(self):
        # SR candidatos
        sr_candidates = []
        seen = set()
        for sr in [SUGGESTED_SR,
                   int(self._in_dev.get("default_samplerate") or 0),
                   int(self._out_dev.get("default_samplerate") or 0),
                   44100, 48000]:
            if sr and sr not in seen:
                seen.add(sr); sr_candidates.append(sr)

        ch_out = 2 if FORCE_STEREO else min(self._out_dev.get("max_output_channels",2), 2)
        ch_in  = min(self._in_dev.get("max_input_channels",2), 2)

        last_error = None
        for sr in sr_candidates:
            try:
                warm = WARMUP_BLOCKS
                self._fade_left = int(sr * (FADE_MS / 1000.0))
                self._sr_in_use = sr

                def in_cb(indata, frames, time_info, status):
                    nonlocal warm
                    if self._stop: raise sd.CallbackStop()
                    if status: self.log.emit(f"[IN-STATUS] {status}")
                    x = np.asarray(indata, dtype=np.float32)
                    if warm > 0:
                        warm -= 1; return
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
                    if self._stop: raise sd.CallbackStop()
                    if status: self.log.emit(f"[OUT-STATUS] {status}")
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

                in_stream = sd.InputStream(device=self._in_idx, samplerate=sr, dtype="float32",
                                           channels=ch_in, blocksize=self.block,
                                           callback=in_cb)
                out_stream = sd.OutputStream(device=self._out_idx, samplerate=sr, dtype="float32",
                                             channels=ch_out, blocksize=self.block,
                                             callback=out_cb)
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
        self.setMinimumSize(980, 600)
        self.setWindowIcon(QtGui.QIcon("assets/app.ico") if Path("assets/app.ico").exists()
                           else self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))

        self.cfg = load_cfg()
        self.worker: AudioWorker | None = None
        self.tray = None

        central = QtWidgets.QWidget(); self.setCentralWidget(central)

        # Controles
        self.in_combo     = QtWidgets.QComboBox()      # FUENTE
        self.out_combo    = QtWidgets.QComboBox()      # SALIDA
        self.refresh_btn  = QtWidgets.QPushButton("Actualizar dispositivos")
        self.save_btn     = QtWidgets.QPushButton("Guardar preferencias")

        self.block_spin   = QtWidgets.QSpinBox();  self.block_spin.setRange(120, 4096); self.block_spin.setValue(int(self.cfg["block"]))
        self.mono_chk     = QtWidgets.QCheckBox("Forzar mono"); self.mono_chk.setChecked(bool(self.cfg["mono"]))
        self.sys_gain     = QtWidgets.QDoubleSpinBox(); self.sys_gain.setRange(0.0, 5.0); self.sys_gain.setSingleStep(0.1); self.sys_gain.setValue(float(self.cfg["gain"]))

        self.start_btn    = QtWidgets.QPushButton("Iniciar")
        self.stop_btn     = QtWidgets.QPushButton("Detener"); self.stop_btn.setEnabled(False)
        self.open_logs_btn= QtWidgets.QPushButton("Abrir logs")
        self.test_tone_btn= QtWidgets.QPushButton("Tono de prueba (440 Hz)")
        self.level_label  = QtWidgets.QLabel("Nivel: — dBFS")
        self.level_bar    = QtWidgets.QProgressBar(); self.level_bar.setRange(0, 100); self.level_bar.setValue(0)
        self.level_bar.setTextVisible(False)

        self.footer = QtWidgets.QLabel("© 2025 Gabriel Golker"); self.footer.setAlignment(QtCore.Qt.AlignCenter)

        # Layout
        form = QtWidgets.QFormLayout()
        form.addRow("Fuente (input):", self.in_combo)
        form.addRow("Salida (→ mic virtual):", self.out_combo)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.refresh_btn); top.addWidget(self.save_btn); top.addStretch()
        top.addWidget(QtWidgets.QLabel("Block:")); top.addWidget(self.block_spin)
        top.addWidget(QtWidgets.QLabel("Ganancia:")); top.addWidget(self.sys_gain)
        top.addWidget(self.mono_chk)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self.start_btn); controls.addWidget(self.stop_btn); controls.addStretch()
        controls.addWidget(self.open_logs_btn); controls.addWidget(self.test_tone_btn)

        meter = QtWidgets.QHBoxLayout()
        meter.addWidget(self.level_label); meter.addWidget(self.level_bar)

        v = QtWidgets.QVBoxLayout(central)
        v.addLayout(form); v.addSpacing(8)
        v.addLayout(top); v.addSpacing(8)
        v.addLayout(meter); v.addSpacing(8)
        v.addLayout(controls); v.addStretch(); v.addWidget(self.footer)

        # Señales
        self.refresh_btn.clicked.connect(self.populate_devices)
        self.save_btn.clicked.connect(self.save_prefs)
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)
        self.open_logs_btn.clicked.connect(self.open_logs)
        self.test_tone_btn.clicked.connect(self._play_test_tone)

        # Bandeja
        self._init_tray()

        # Cargar dispositivos + seleccionar lo guardado
        self.populate_devices(select_saved=True)
        QtCore.QTimer.singleShot(400, self.autostart)

    def populate_devices(self, select_saved: bool = False):
        try:
            self.in_combo.clear(); self.out_combo.clear()
            in_names = []; out_names = []
            for i, dev in enumerate(sd.query_devices()):
                name = dev.get("name","")
                if dev.get("max_input_channels",0) > 0:
                    in_names.append(name)
                if dev.get("max_output_channels",0) > 0:
                    out_names.append(name)
            # ordenar: priorizar Mezcla estéreo y CABLE Input
            in_names = sorted(in_names, key=lambda n: 0 if any(k in n.lower() for k in ["stereo mix","mezcla estéreo","what u hear"]) else 1)
            out_names = sorted(out_names, key=lambda n: 0 if "cable input" in n.lower() else 1)
            if not in_names: raise RuntimeError("No hay dispositivos de entrada en el sistema.")
            if not out_names: raise RuntimeError("No hay dispositivos de salida en el sistema.")
            self.in_combo.addItems(in_names)
            self.out_combo.addItems(out_names)
            logger.info("Dispositivos actualizados.")

            if select_saved:
                # intenta seleccionar el guardado
                def _select(cb: QtWidgets.QComboBox, name: str):
                    if not name: return
                    idx = cb.findText(name, QtCore.Qt.MatchFlag.MatchContains)
                    if idx >= 0: cb.setCurrentIndex(idx)
                _select(self.in_combo, self.cfg.get("input_name",""))
                _select(self.out_combo, self.cfg.get("output_name",""))
        except Exception as e:
            logger.exception(f"No se pudieron listar dispositivos: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Error al listar dispositivos:\n{e}")

    def save_prefs(self):
        self.cfg["input_name"]  = self.in_combo.currentText()
        self.cfg["output_name"] = self.out_combo.currentText()
        self.cfg["block"] = int(self.block_spin.value())
        self.cfg["mono"]  = bool(self.mono_chk.isChecked())
        self.cfg["gain"]  = float(self.sys_gain.value())
        try:
            save_cfg(self.cfg)
            QtWidgets.QMessageBox.information(self, "Guardado", "Preferencias guardadas en config.json.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"No se pudo guardar: {e}")

    def autostart(self):
        # Si hay CABLE Input y Mezcla estéreo seleccionados, intentamos arrancar
        self._on_start()

    def _on_start(self):
        try:
            if self.worker and self.worker.isRunning():
                QtWidgets.QMessageBox.information(self, "Info", "El audio ya está en ejecución.")
                return
            # Guarda lo actual para la próxima
            self.save_prefs()
            self.worker = AudioWorker(
                in_name=self.in_combo.currentText(),
                out_name=self.out_combo.currentText(),
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

    def _append_log(self, msg: str):
        logger.info(msg)

    def _play_test_tone(self):
        # 440 Hz al altavoz por defecto (para validar que tu fuente “ve” el sistema)
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

