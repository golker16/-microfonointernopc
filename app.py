# app.py
import sys, time, logging, math, json
from logging.handlers import RotatingFileHandler
from pathlib import Path
from collections import deque
import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets
import qdarkstyle
import sounddevice as sd

# loopback por altavoz (opcional)
try:
    import soundcard as sc
except Exception:
    sc = None

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
    "mode": "stereo_mix",      # "stereo_mix" | "soundcard_loopback"
    "input_name": "",          # usado en stereo_mix
    "output_name": "",         # destino, normalmente "CABLE Input ..."
    "speaker_name": "",        # usado en soundcard_loopback
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

# ---------- Workers ----------
class BaseWorker(QtCore.QThread):
    log   = QtCore.Signal(str)
    level = QtCore.Signal(float)
    def __init__(self, block: int, mono: bool, sys_gain: float):
        super().__init__()
        self.block = int(block)
        self.mono = bool(mono)
        self.sys_gain = float(sys_gain)
        self._stop = False
        self._sr_in_use = None
        self._fade_left = 0
        self._q = deque(maxlen=10)
    def stop(self): self._stop = True

# --- Worker A: Stereo Mix (sounddevice → sounddevice) ---
class StereoMixWorker(BaseWorker):
    def __init__(self, in_name: str, out_name: str, block: int, mono: bool, sys_gain: float):
        super().__init__(block, mono, sys_gain)
        self.in_name = in_name
        self.out_name = out_name
        self._in_idx = None; self._in_dev = None
        self._out_idx = None; self._out_dev = None

    def _choose_devices(self):
        # salida
        out_idx, out_dev = None, None
        if self.out_name:
            out_idx, out_dev = find_device_by_name(self.out_name, need_input=False)
        if out_idx is None:
            out_idx, out_dev = find_device_by_name("cable input", need_input=False)
        if out_idx is None:
            for i, dev in enumerate(sd.query_devices()):
                if dev.get("max_output_channels",0) > 0:
                    out_idx, out_dev = i, dev; break
        if out_idx is None: raise RuntimeError("No hay dispositivo de SALIDA.")
        self._out_idx, self._out_dev = out_idx, out_dev

        # entrada
        in_idx, in_dev = None, None
        if self.in_name:
            in_idx, in_dev = find_device_by_name(self.in_name, need_input=True)
        if in_idx is None:
            in_idx, in_dev = find_best_stereo_mix()
        if in_idx is None:
            for i, dev in enumerate(sd.query_devices()):
                if dev.get("max_input_channels",0) > 0:
                    in_idx, in_dev = i, dev; break
        if in_idx is None: raise RuntimeError("No hay dispositivo de ENTRADA.")
        self._in_idx, self._in_dev = in_idx, in_dev

        self.log.emit(f"[StereoMix] Fuente: {self._in_dev['name']}  →  Salida: {self._out_dev['name']}")

    def _open_streams(self):
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
                    if warm > 0: warm -= 1; return
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
                self.log.emit(f"[StereoMix] OK SR={sr} ch_in={ch_in} ch_out={ch_out}")
                return in_stream, out_stream
            except Exception as e:
                last_error = repr(e)
                self.log.emit(f"[StereoMix] Intento falló SR={sr} → {last_error}")
                try: in_stream.__exit__(None, None, None)  # type: ignore
                except Exception: pass
                try: out_stream.__exit__(None, None, None) # type: ignore
                except Exception: pass

        raise RuntimeError(f"No pude abrir streams (StereoMix). Último error: {last_error}")

    def run(self):
        try:
            self._choose_devices()
            in_stream, out_stream = self._open_streams()
        except Exception as e:
            dump_devices_to_log()
            self.log.emit(f"[ERROR StereoMix] {e}")
            return

        try:
            with in_stream, out_stream:
                self.log.emit(f"[StereoMix] Audio en marcha (SR={self._sr_in_use}, block={self.block}).")
                while not self._stop:
                    time.sleep(0.05)
        except Exception as e:
            dump_devices_to_log()
            self.log.emit(f"[ERROR StereoMix] {repr(e)}")

# --- Worker B: Loopback por altavoz (soundcard → sounddevice) ---
class SoundcardLoopbackWorker(BaseWorker):
    def __init__(self, speaker_name: str, out_name: str, block: int, mono: bool, sys_gain: float):
        super().__init__(block, mono, sys_gain)
        self.speaker_name = speaker_name
        self.out_name = out_name
        self._speaker = None
        self._mic_loop = None
        self._out_idx = None; self._out_dev = None

    def _choose_endpoints(self):
        if sc is None:
            raise RuntimeError("Falta 'soundcard'. Instala con: pip install soundcard==0.4.3")
        # Altavoz
        spk = None
        if self.speaker_name:
            for s in sc.all_speakers():
                if self.speaker_name.lower() in s.name.lower():
                    spk = s; break
        if spk is None:
            spk = sc.default_speaker()
        if spk is None:
            raise RuntimeError("No encontré altavoz para loopback.")
        self._speaker = spk
        # Micro 'loopback' del altavoz
        self._mic_loop = sc.get_microphone(spk.name, include_loopback=True)
        if self._mic_loop is None:
            raise RuntimeError(f"No hay loopback para el altavoz: {spk.name}")

        # Salida (CABLE Input) con sounddevice
        out_idx, out_dev = None, None
        if self.out_name:
            out_idx, out_dev = find_device_by_name(self.out_name, need_input=False)
        if out_idx is None:
            out_idx, out_dev = find_device_by_name("cable input", need_input=False)
        if out_idx is None:
            for i, dev in enumerate(sd.query_devices()):
                if dev.get("max_output_channels",0) > 0:
                    out_idx, out_dev = i, dev; break
        if out_idx is None:
            raise RuntimeError("No hay dispositivo de SALIDA.")
        self._out_idx, self._out_dev = out_idx, out_dev

        self.log.emit(f"[Loopback] Altavoz: {self._speaker.name}  →  Salida: {self._out_dev['name']}")

    def run(self):
        try:
            self._choose_endpoints()
        except Exception as e:
            dump_devices_to_log()
            self.log.emit(f"[ERROR Loopback] {e}")
            return

        # SR: probamos 48k → 44.1k
        for sr in [SUGGESTED_SR, 44100]:
            try:
                ch_out = 2 if FORCE_STEREO else min(self._out_dev.get("max_output_channels",2), 2)
                self._sr_in_use = sr
                self._fade_left = int(sr * (FADE_MS / 1000.0))
                warm = WARMUP_BLOCKS

                def out_cb(outdata, frames, time_info, status):
                    if self._stop: raise sd.CallbackStop()
                    if status: self.log.emit(f"[OUT-STATUS] {status}")
                    if hasattr(self, "_buf") and self._buf is not None and self._buf.shape[0] >= frames:
                        b = self._buf[:frames]
                        self._buf = self._buf[frames:]
                    else:
                        b = np.zeros((frames, ch_out), dtype=np.float32)
                    outdata[:] = b

                # Abrimos salida
                out_stream = sd.OutputStream(device=self._out_idx, samplerate=sr, dtype="float32",
                                             channels=ch_out, blocksize=self.block,
                                             callback=out_cb)
                out_stream.__enter__()

                self.log.emit(f"[Loopback] OK salida SR={sr} ch_out={ch_out}")

                # Bucle de captura con soundcard (bloques pull)
                block_secs = max(self.block / float(sr), 0.01)
                with self._mic_loop.recorder(samplerate=sr) as rec:
                    self.log.emit(f"[Loopback] Audio en marcha (SR={sr}, block={self.block}).")
                    self._buf = np.zeros((0, ch_out), dtype=np.float32)
                    while not self._stop:
                        data = rec.record(numframes=int(block_secs * sr))
                        x = np.asarray(data, dtype=np.float32)
                        if self.mono: x = to_mono(x)
                        if x.ndim == 1: x = x[:, None]
                        if x.shape[1] != ch_out: x = match_channels(x, ch_out)
                        x = sanitize(x) * self.sys_gain
                        if LIMITER: x = np.clip(x, -0.98, 0.98)
                        # fade-in
                        n = min(self._fade_left, x.shape[0])
                        if n > 0:
                            fade = np.linspace(0.0, 1.0, n, endpoint=True, dtype=np.float32)
                            x[:n, :] *= fade[:, None]
                            self._fade_left -= n
                        try: self.level.emit(rms_dbfs(x))
                        except Exception: pass
                        # append al buffer que consume el callback
                        self._buf = np.vstack([self._buf, x])
                        time.sleep(0.001)

                out_stream.__exit__(None, None, None)
                return
            except Exception as e:
                try: out_stream.__exit__(None, None, None)  # type: ignore
                except Exception: pass
                self.log.emit(f"[Loopback] Intento falló SR={sr} → {repr(e)}")
                continue

        self.log.emit("[ERROR Loopback] No pude iniciar con 48k ni 44.1k.")

# ---------- UI ----------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(1040, 640)
        self.setWindowIcon(QtGui.QIcon("assets/app.ico") if Path("assets/app.ico").exists()
                           else self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))

        self.cfg = load_cfg()
        self.worker: BaseWorker | None = None
        self.tray = None

        central = QtWidgets.QWidget(); self.setCentralWidget(central)

        # Controles
        self.mode_cb     = QtWidgets.QComboBox()
        self.mode_cb.addItems(["Stereo Mix (input físico)", "Loopback por altavoz (soundcard)"])
        self.mode_cb.setCurrentIndex(0 if self.cfg["mode"]=="stereo_mix" else 1)

        self.in_combo    = QtWidgets.QComboBox()    # FUENTE (para stereo_mix)
        self.spk_combo   = QtWidgets.QComboBox()    # ALTAVOZ (para loopback soundcard)
        self.out_combo   = QtWidgets.QComboBox()    # SALIDA (CABLE Input)
        self.refresh_btn = QtWidgets.QPushButton("Actualizar dispositivos")
        self.save_btn    = QtWidgets.QPushButton("Guardar preferencias")

        self.block_spin  = QtWidgets.QSpinBox();  self.block_spin.setRange(120, 4096); self.block_spin.setValue(int(self.cfg["block"]))
        self.mono_chk    = QtWidgets.QCheckBox("Forzar mono"); self.mono_chk.setChecked(bool(self.cfg["mono"]))
        self.sys_gain    = QtWidgets.QDoubleSpinBox(); self.sys_gain.setRange(0.0, 5.0); self.sys_gain.setSingleStep(0.1); self.sys_gain.setValue(float(self.cfg["gain"]))

        self.start_btn   = QtWidgets.QPushButton("Iniciar")
        self.stop_btn    = QtWidgets.QPushButton("Detener"); self.stop_btn.setEnabled(False)
        self.open_logs_btn=QtWidgets.QPushButton("Abrir logs")
        self.test_tone_btn=QtWidgets.QPushButton("Tono de prueba (440 Hz)")
        self.level_label = QtWidgets.QLabel("Nivel: — dBFS")
        self.level_bar   = QtWidgets.QProgressBar(); self.level_bar.setRange(0, 100); self.level_bar.setValue(0)
        self.level_bar.setTextVisible(False)

        self.footer = QtWidgets.QLabel("© 2025 Gabriel Golker"); self.footer.setAlignment(QtCore.Qt.AlignCenter)

        # Layout
        form = QtWidgets.QFormLayout()
        form.addRow("Modo de captura:", self.mode_cb)
        form.addRow("Fuente (Stereo Mix):", self.in_combo)
        form.addRow("Altavoz (loopback):", self.spk_combo)
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
        self.mode_cb.currentIndexChanged.connect(self._toggle_mode_rows)

        # Bandeja
        self._init_tray()

        # Cargar dispositivos + seleccionar lo guardado
        self.populate_devices(select_saved=True)
        self._toggle_mode_rows()
        QtCore.QTimer.singleShot(400, self.autostart)

    def _toggle_mode_rows(self):
        is_loop = (self.mode_cb.currentIndex() == 1)
        self.in_combo.setEnabled(not is_loop)
        self.spk_combo.setEnabled(is_loop)

    def populate_devices(self, select_saved: bool = False):
        try:
            # INPUTS (para stereo_mix)
            self.in_combo.clear()
            in_names = []
            for dev in sd.query_devices():
                if dev.get("max_input_channels",0) > 0:
                    in_names.append(dev.get("name",""))
            in_names = sorted(in_names, key=lambda n: 0 if any(k in n.lower() for k in ["stereo mix","mezcla estéreo","what u hear"]) else 1)
            if in_names:
                self.in_combo.addItems(in_names)

            # SPEAKERS (para loopback soundcard)
            self.spk_combo.clear()
            if sc is not None:
                spk_names = [s.name for s in sc.all_speakers()]
            else:
                spk_names = []
            if spk_names:
                # prioriza el por defecto
                def_name = (sc.default_speaker().name if sc and sc.default_speaker() else "")
                spk_names = sorted(spk_names, key=lambda n: 0 if def_name and def_name in n else 1)
                self.spk_combo.addItems(spk_names)

            # OUTPUTS
            self.out_combo.clear()
            out_names = []
            for dev in sd.query_devices():
                if dev.get("max_output_channels",0) > 0:
                    out_names.append(dev.get("name",""))
            out_names = sorted(out_names, key=lambda n: 0 if "cable input" in n.lower() else 1)
            if out_names:
                self.out_combo.addItems(out_names)

            logger.info("Dispositivos actualizados.")

            if select_saved:
                cfg = load_cfg()
                # modo
                self.mode_cb.setCurrentIndex(0 if cfg.get("mode","stereo_mix")=="stereo_mix" else 1)
                # selects
                def _select(cb: QtWidgets.QComboBox, name: str):
                    if not name: return
                    idx = cb.findText(name, QtCore.Qt.MatchFlag.MatchContains)
                    if idx >= 0: cb.setCurrentIndex(idx)
                _select(self.in_combo, cfg.get("input_name",""))
                _select(self.spk_combo, cfg.get("speaker_name",""))
                _select(self.out_combo, cfg.get("output_name",""))
                self.block_spin.setValue(int(cfg.get("block", DEFAULT_BLOCK)))
                self.mono_chk.setChecked(bool(cfg.get("mono", True)))
                self.sys_gain.setValue(float(cfg.get("gain", 1.0)))
        except Exception as e:
            logger.exception(f"No se pudieron listar dispositivos: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Error al listar dispositivos:\n{e}")

    def save_prefs(self):
        cfg = {
            "mode": "stereo_mix" if self.mode_cb.currentIndex()==0 else "soundcard_loopback",
            "input_name":  self.in_combo.currentText(),
            "speaker_name":self.spk_combo.currentText(),
            "output_name": self.out_combo.currentText(),
            "block": int(self.block_spin.value()),
            "mono":  bool(self.mono_chk.isChecked()),
            "gain":  float(self.sys_gain.value()),
        }
        try:
            save_cfg(cfg)
            QtWidgets.QMessageBox.information(self, "Guardado", "Preferencias guardadas en config.json.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"No se pudo guardar: {e}")

    def autostart(self):
        self._on_start()

    def _on_start(self):
        try:
            if self.worker and self.worker.isRunning():
                QtWidgets.QMessageBox.information(self, "Info", "El audio ya está en ejecución.")
                return

            self.save_prefs()
            mode = "stereo_mix" if self.mode_cb.currentIndex()==0 else "soundcard_loopback"
            block = self.block_spin.value()
            mono  = self.mono_chk.isChecked()
            gain  = self.sys_gain.value()

            if mode == "stereo_mix":
                self.worker = StereoMixWorker(
                    in_name=self.in_combo.currentText(),
                    out_name=self.out_combo.currentText(),
                    block=block, mono=mono, sys_gain=gain
                )
            else:
                self.worker = SoundcardLoopbackWorker(
                    speaker_name=self.spk_combo.currentText(),
                    out_name=self.out_combo.currentText(),
                    block=block, mono=mono, sys_gain=gain
                )

            self.worker.log.connect(self._append_log)
            self.worker.level.connect(self._update_level)
            self.worker.start()
            QtCore.QTimer.singleShot(500, self._post_start_check)
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

    def open_logs(self):
        try:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(LOG_FILE)))
        except Exception as e:
            logger.exception(f"No se pudieron abrir los logs: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"No se pudieron abrir los logs:\n{e}")

    def _play_test_tone(self):
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


