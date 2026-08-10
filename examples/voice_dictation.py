"""
Voice Dictation — Push-to-Talk диктовка вместо клавиатуры.

Заменяет Superwhisper / Wispr Flow / Aqua Voice — но локально и бесплатно.

Использование:
    python -m examples.voice_dictation                         # с дефолтным конфигом
    python -m examples.voice_dictation --config my-config.json # свой конфиг
    python -m examples.voice_dictation --setup                 # сгенерить шаблон конфига

Как работает:
    1. Скрипт висит в фоне, слушает глобальный хоткей.
    2. Жмёшь хоткей (по дефолту Ctrl+Shift+Space) → начинается запись.
    3. Говоришь, держа хоткей.
    4. Отпускаешь → Whisper транскрибирует → текст вставляется в активное поле через clipboard.

Зависимости (поставит wizard, или вручную):
    pip install sounddevice soundfile pynput pyperclip pystray Pillow numpy

Пермишены:
    macOS — нужно дать разрешение на Accessibility и Microphone:
        Системные настройки → Конфиденциальность → Универсальный доступ → добавить Terminal/iTerm
        Системные настройки → Конфиденциальность → Микрофон → добавить Terminal/iTerm
    Linux — на Wayland могут быть проблемы с глобальным хоткеем (X11 ок).
    Windows — обычно работает out-of-box.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import re
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


# ─── Config ─────────────────────────────────────────────────────────────────


DEFAULT_CONFIG = {
    "hotkey": "<ctrl>+<shift>+<space>",  # формат pynput Listener
    "mode": "ptt",                       # "ptt" (push-to-talk) или "toggle"
    "language": None,                    # null = auto. Лучше указать ("ru", "en")
    "model": "large-v3-turbo",
    "backend": None,                     # "openvino" | "faster" | "mlx" | "cpp" | null = auto
    "ov_device": "GPU",                  # OpenVINO: GPU | NPU | CPU | AUTO
    "sample_rate": 16000,
    "channels": 1,
    "auto_paste": True,                  # вставить через Cmd+V/Ctrl+V после копирования
    "play_sound": True,                  # бипы на старт/стоп
    "show_tray": True,                   # значок в трее (если установлен pystray)
    "show_cursor_indicator": True,       # мигающая красная точка у курсора во время записи
    "cursor_indicator_color": "#ef4444", # цвет точки (CSS hex)
    "show_hud": True,                    # macOS: панель внизу экрана с эквалайзером
    "keep_in_clipboard": True,           # оставить надиктованное в буфере (Cmd+V куда угодно)
    "log_file": None,                    # путь к файлу лога или null = stdout
    "trim_silence_ms": 200,              # обрезать тишину в начале/конце записи
    "min_duration_ms": 300,              # игнорировать слишком короткие записи (промахи кнопкой)
    "unload_after_idle_min": 15,         # выгрузить модель из памяти после N минут простоя (0 = держать всегда)
    # macOS-специфика: pystray/Tk известно жрут CPU в фоне на macOS
    # (NSRunLoop в non-main thread + Tk thread-safety). Этот флаг автоматически
    # отключает show_tray и show_cursor_indicator на macOS, оставляя CLI-вывод
    # как единственный feedback. Если хочешь tray на Mac на свой страх и риск —
    # поставь false (тогда show_tray/show_cursor_indicator будут уважаться).
    "mac_low_cpu_mode": True,
}


def default_config_path() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "whisper-skill" / "voice_dictation.json"


def load_config(path: Optional[Path] = None) -> dict:
    path = path or default_config_path()
    if not path.exists():
        return dict(DEFAULT_CONFIG)
    # utf-8-sig: терпим BOM, если конфиг создан старой версией install.ps1
    # (PS 5.1 Set-Content -Encoding UTF8 пишет UTF-8 с BOM).
    user_cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(user_cfg)
    return cfg


def write_config(path: Path, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


# ─── Available models (OpenVINO cache) ──────────────────────────────────────


_MODEL_QUALITY_ORDER = [
    # Best → worst. Turbo (distilled-decoder) — лучший компромисс скорость/качество,
    # но чуть слабее int8/int8-sym на сложных кейсах (шум, акцент).
    # int4 теряет в качестве заметнее всех.
    "large-v3",
    "large-v3-int8",
    "large-v3-int8-sym",
    "large-v3-turbo",
    "large-v3-int4",
    "medium", "small", "base", "tiny",
]


def list_available_ov_models() -> list:
    """Папки `whisper-*-ov` в ~/.cache/openvino-whisper/ — те, что openvino-
    backend умеет грузить (см. _transcribe_openvino в common.py).
    Возвращает model-name'ы в порядке убывания качества (best первый);
    модели вне whitelist'а уходят в конец alphabetically.
    """
    base = Path.home() / ".cache" / "openvino-whisper"
    if not base.exists():
        return []
    found = set()
    for p in base.iterdir():
        if p.is_dir() and p.name.startswith("whisper-") and p.name.endswith("-ov"):
            found.add(p.name[len("whisper-"):-len("-ov")])
    ordered = [m for m in _MODEL_QUALITY_ORDER if m in found]
    extras = sorted(found - set(ordered))
    return ordered + extras


# ─── Single-instance lock ───────────────────────────────────────────────────


_single_instance_handle = None  # держим ссылку чтобы lock не сборщик мусора убил


def acquire_single_instance_lock(timeout_seconds: float = 2.0) -> bool:
    """Захватить named mutex (Windows) / file lock (Unix). True — захватили.
    False — другая копия уже работает.

    timeout_seconds покрывает self-restart: старая копия только что вызвала
    os._exit, новая стартует, ОС ещё не успела освободить lock — повторяем.
    """
    global _single_instance_handle
    deadline = time.monotonic() + timeout_seconds

    if platform.system() == "Windows":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        ERROR_ALREADY_EXISTS = 183
        while True:
            handle = kernel32.CreateMutexW(None, True, "WhisperVoiceDictation_SingleInstance")
            if kernel32.GetLastError() != ERROR_ALREADY_EXISTS:
                _single_instance_handle = handle
                return True
            kernel32.CloseHandle(handle)
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.15)
    else:
        try:
            import fcntl
        except ImportError:
            return True  # нет fcntl — пропускаем lock (Win-вариант покрыт выше)
        lock_path = Path.home() / ".config" / "whisper-skill" / "voice_dictation.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            fh = open(lock_path, "a+")
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fh.seek(0); fh.truncate()
                fh.write(str(os.getpid())); fh.flush()
                _single_instance_handle = fh
                return True
            except (BlockingIOError, OSError):
                fh.close()
                if time.monotonic() >= deadline:
                    return False
                time.sleep(0.15)


def restart_self() -> None:
    """Завершить текущий процесс и запустить новую копию через VBS launcher.
    Используется при смене модели через tray-меню."""
    repo_root = Path(__file__).resolve().parents[1]
    if platform.system() == "Windows":
        vbs = repo_root / "launcher" / "voice_dictation_silent.vbs"
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        if vbs.exists():
            subprocess.Popen(
                ["wscript.exe", str(vbs)],
                creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
                close_fds=True,
            )
        else:
            subprocess.Popen(
                [sys.executable, "-m", "examples.voice_dictation"],
                cwd=str(repo_root),
                creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
                close_fds=True,
            )
    elif getattr(sys, "frozen", None) == "macosx_app" and Path(
            "~/Library/LaunchAgents/com.qspeak.dictation.plist").expanduser().exists():
        # Под py2app sys.executable — сам бандл QSpeak: запустить его с
        # "-m examples.voice_dictation" нельзя, argparse упадёт на этих
        # аргументах и приложение не поднимется. Перезапуск отдаём launchd —
        # он же вернёт правильный PATH с homebrew (иначе ffmpeg not found).
        subprocess.Popen(
            ["/bin/sh", "-c",
             f"sleep 1; launchctl kickstart -k gui/{os.getuid()}/com.qspeak.dictation"],
            start_new_session=True, close_fds=True,
        )
    else:
        subprocess.Popen(
            [sys.executable, "-m", "examples.voice_dictation"],
            cwd=str(repo_root),
            start_new_session=True,
            close_fds=True,
        )
    # os._exit — мгновенный hard exit без atexit/finally; ОС освободит mutex/lock,
    # новая копия подхватит после retry в acquire_single_instance_lock.
    os._exit(0)


# ─── Setup helper ───────────────────────────────────────────────────────────


def setup_wizard():
    """Создать дефолтный конфиг и подсказать что делать дальше."""
    path = default_config_path()
    if path.exists():
        print(f"Конфиг уже есть: {path}")
        print("Хочешь перезаписать? [y/N] ", end="", flush=True)
        if input().strip().lower() != "y":
            return
    write_config(path, DEFAULT_CONFIG)
    print(f"\n✓ Создал конфиг: {path}")
    print(f"\nДефолтный хоткей: {DEFAULT_CONFIG['hotkey']}")
    print(f"Дефолтная модель: {DEFAULT_CONFIG['model']}")
    print(f"\nЗапусти диктовку:")
    print(f"  python -m examples.voice_dictation\n")


# ─── Audio recording ────────────────────────────────────────────────────────


def _close_stream(stream) -> None:
    """Закрыть аудиопоток. Вызывается в отдельном потоке — на мёртвом
    устройстве оба вызова могут не вернуться никогда."""
    try:
        stream.stop()
        stream.close()
    except Exception as e:
        logging.warning(f"stream close failed: {e}")


class AudioRecorder:
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self._frames: list = []
        self._stream = None
        self._recording = False
        self.level = 0.0  # текущая громкость 0..1, читает HUD

    def start(self) -> None:
        import sounddevice as sd
        import numpy as np

        self._frames = []
        self._recording = True

        def callback(indata, frames, time_info, status):
            if status:
                logging.warning(f"audio status: {status}")
            self._frames.append(indata.copy())
            # Громкость для HUD. sqrt растягивает тихую часть шкалы — на линейной
            # RMS обычная речь еле шевелит полоски.
            self.level = min(1.0, float(np.sqrt(np.abs(indata).mean())) * 3.0)

        def _open():
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                callback=callback,
            )
            self._stream.start()

        try:
            _open()
        except Exception as e:
            # PortAudio кэширует список устройств с момента импорта. Сменили
            # вход (воткнули вебкамеру, ушли в наушники) — открытие падает
            # с -9986 навсегда, пока процесс не перезапустят. Переинициализация
            # даёт актуальный список без перезапуска приложения.
            print(f"⚠️  Микрофон не открылся ({e}) — переинициализирую PortAudio")
            sd._terminate()
            sd._initialize()
            _open()
        try:
            dev = sd.query_devices(kind="input")
            print(f"🎙  Вход: {dev['name']} @ {self.sample_rate} Гц")
        except Exception:
            pass

    def stop(self) -> Optional[str]:
        """Остановить запись и сохранить в WAV. Вернуть путь к файлу."""
        import numpy as np
        import soundfile as sf

        if not self._recording or not self._stream:
            return None
        self._recording = False
        stream, self._stream = self._stream, None

        # Отвалившийся вход (USB-микрофон вебкамеры, смена устройства) оставляет
        # CoreAudio висеть в stop()/close() бесконечно, а зовут нас из потока
        # хоткея — вместе с ним умирает приём клавиш (инцидент 04.08.2026).
        # Ждём 2 с и уходим с уже накопленными кадрами: следующий start()
        # поймает ошибку открытия и переинициализирует PortAudio.
        closer = threading.Thread(target=_close_stream, args=(stream,), daemon=True)
        closer.start()
        closer.join(timeout=2.0)
        if closer.is_alive():
            print("⚠️  Микрофон не закрылся за 2 с — продолжаю без него")

        if not self._frames:
            return None
        audio = np.concatenate(self._frames, axis=0)

        tmp = tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False, prefix="voice_dictation_"
        )
        sf.write(tmp.name, audio, self.sample_rate, subtype="PCM_16")
        return tmp.name

    def snapshot(self) -> Optional[str]:
        """Сохранить ТЕКУЩИЙ накопленный звук в WAV, НЕ останавливая запись.
        Нужно для потоковой диктовки — промежуточного распознавания на лету."""
        import numpy as np
        import soundfile as sf

        frames = list(self._frames)  # копия: callback пишет параллельно
        if not frames:
            return None
        audio = np.concatenate(frames, axis=0)
        tmp = tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False, prefix="vd_stream_"
        )
        sf.write(tmp.name, audio, self.sample_rate, subtype="PCM_16")
        return tmp.name

    @property
    def duration_sec(self) -> float:
        if not self._frames:
            return 0.0
        import numpy as np
        total_samples = sum(f.shape[0] for f in self._frames)
        return total_samples / self.sample_rate


def _common_prefix_words(a: list, b: list) -> list:
    """LocalAgreement: слова считаем «устоявшимися» только если два подряд
    распознавания дали одинаковый префикс. Это защищает от того, что модель
    переобдумывает concу фразы по мере поступления звука."""
    out = []
    for x, y in zip(a, b):
        if x != y:
            break
        out.append(x)
    return out


CLAUDE_BIN_CANDIDATES = [
    # PATH-имя первым: на Windows это claude.cmd, на Unix — бинарь из PATH.
    "claude",
    str(Path.home() / ".local" / "bin" / "claude"),
]


def ai_cleanup(text: str, timeout_sec: float = 30.0) -> str:
    """Умная правка надиктованного через Claude CLI (как «AI mode» у Wispr Flow).
    Работает на текущей подписке — отдельный API-ключ не нужен.
    При любой ошибке/таймауте возвращает исходный текст (диктовка важнее правки)."""
    prompt = (
        "Ниже — сырой текст голосовой диктовки. Расставь пунктуацию и заглавные буквы, "
        "убери слова-паразиты, оговорки и повторы, исправь очевидные ошибки распознавания. "
        "НЕ меняй смысл, НЕ добавляй ничего от себя, НЕ переводи. "
        "Верни ТОЛЬКО итоговый текст, без пояснений и кавычек.\n\n" + text
    )
    for binary in CLAUDE_BIN_CANDIDATES:
        try:
            r = subprocess.run(
                [binary, "-p", prompt],
                capture_output=True, text=True, timeout=timeout_sec,
            )
            out = (r.stdout or "").strip()
            if out:
                return out
        except FileNotFoundError:
            continue
        except Exception as e:
            logging.warning(f"ai_cleanup failed: {e}")
            break
    return text


# ─── Text insertion ─────────────────────────────────────────────────────────


def _windows_set_clipboard_text(text: str) -> bool:
    """Надёжная запись CF_UNICODETEXT через Win32. Возвращает True при успехе.

    pyperclip на Windows периодически не выдерживает rapid-fire вызовы и
    может отвалиться без исключения. Эта реализация делает retry на
    OpenClipboard (буфер мог быть занят другим процессом) и явно владеет
    памятью до момента, когда система её забирает.
    """
    import ctypes
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    user32.OpenClipboard.argtypes = [ctypes.c_void_p]
    user32.OpenClipboard.restype = ctypes.c_int
    user32.EmptyClipboard.restype = ctypes.c_int
    user32.CloseClipboard.restype = ctypes.c_int
    user32.SetClipboardData.argtypes = [ctypes.c_uint, ctypes.c_void_p]
    user32.SetClipboardData.restype = ctypes.c_void_p
    kernel32.GlobalAlloc.argtypes = [ctypes.c_uint, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = ctypes.c_void_p
    kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalUnlock.restype = ctypes.c_int
    kernel32.GlobalFree.argtypes = [ctypes.c_void_p]
    kernel32.GlobalFree.restype = ctypes.c_void_p

    GMEM_MOVEABLE = 0x0002
    CF_UNICODETEXT = 13

    data = text.encode("utf-16-le") + b"\x00\x00"
    h_mem = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data))
    if not h_mem:
        return False
    p_mem = kernel32.GlobalLock(h_mem)
    if not p_mem:
        kernel32.GlobalFree(h_mem)
        return False
    ctypes.memmove(p_mem, data, len(data))
    kernel32.GlobalUnlock(h_mem)

    opened = False
    for _ in range(10):
        if user32.OpenClipboard(None):
            opened = True
            break
        time.sleep(0.01)
    if not opened:
        kernel32.GlobalFree(h_mem)
        return False

    try:
        user32.EmptyClipboard()
        if not user32.SetClipboardData(CF_UNICODETEXT, h_mem):
            kernel32.GlobalFree(h_mem)
            return False
        return True
    finally:
        user32.CloseClipboard()


def _windows_get_clipboard_text() -> Optional[str]:
    """Чтение CF_UNICODETEXT через Win32. None если буфер пуст / не текст /
    OpenClipboard не удался."""
    import ctypes
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    user32.OpenClipboard.argtypes = [ctypes.c_void_p]
    user32.OpenClipboard.restype = ctypes.c_int
    user32.CloseClipboard.restype = ctypes.c_int
    user32.GetClipboardData.argtypes = [ctypes.c_uint]
    user32.GetClipboardData.restype = ctypes.c_void_p
    kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalUnlock.restype = ctypes.c_int

    CF_UNICODETEXT = 13

    opened = False
    for _ in range(10):
        if user32.OpenClipboard(None):
            opened = True
            break
        time.sleep(0.01)
    if not opened:
        return None

    try:
        h = user32.GetClipboardData(CF_UNICODETEXT)
        if not h:
            return None
        p = kernel32.GlobalLock(h)
        if not p:
            return None
        try:
            return ctypes.wstring_at(p)
        finally:
            kernel32.GlobalUnlock(h)
    finally:
        user32.CloseClipboard()


def _get_clipboard_text() -> Optional[str]:
    if platform.system() == "Windows":
        return _windows_get_clipboard_text()
    try:
        import pyperclip
        return pyperclip.paste() or None
    except Exception:
        return None


def copy_to_clipboard(text: str) -> None:
    if platform.system() == "Windows":
        if _windows_set_clipboard_text(text):
            return
        logging.warning("win32 clipboard set failed, falling back to pyperclip")
    try:
        import pyperclip
        pyperclip.copy(text)
    except Exception as e:
        logging.error(f"clipboard copy failed: {e}")


def load_vocabulary() -> list:
    """Правила из vocabulary.txt: [(правильное написание, [как слышится, ...]), ...].

    Читаем на каждую диктовку: файл крошечный, зато правки подхватываются без
    перезапуска. initial_prompt для этого не годится — проверено 2026-08-03:
    Whisper трактует его как контекст, термины не чинит и сбивает капитализацию.
    """
    try:
        p = default_config_path().parent / "vocabulary.txt"
        if not p.exists():
            return []
        rules = []
        for ln in p.read_text(encoding="utf-8-sig").splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#") or "=" not in ln:
                continue
            target, variants = ln.split("=", 1)
            forms = [v.strip() for v in variants.split(",") if v.strip()]
            if target.strip() and forms:
                # длинные варианты первыми: «сейф гриппи» должен сработать
                # раньше, чем «сейф грип» съест его начало
                rules.append((target.strip(), sorted(forms, key=len, reverse=True)))
        return rules
    except Exception as e:
        logging.error(f"vocabulary read failed: {e}")
        return []


def apply_vocabulary(text: str, rules: list) -> str:
    """Причесать термины после распознавания. Детерминированно, в отличие от
    подсказок модели."""
    for target, forms in rules:
        for form in forms:
            text = re.sub(rf"(?<!\w){re.escape(form)}(?!\w)", target, text,
                          flags=re.IGNORECASE)
    return text


def add_vocabulary_rule(target: str, form: str) -> bool:
    """Дописать правило «правильно = как слышится» в vocabulary.txt.

    Термин в файле уже есть — кривой вариант уходит в его строку, иначе
    заводится новая. False, если такой вариант там уже был.
    """
    p = default_config_path().parent / "vocabulary.txt"
    lines = p.read_text(encoding="utf-8-sig").splitlines() if p.exists() else []
    for i, ln in enumerate(lines):
        if ln.strip().startswith("#") or "=" not in ln:
            continue
        t, variants = ln.split("=", 1)
        if t.strip().lower() != target.strip().lower():
            continue
        if form.lower() in [v.strip().lower() for v in variants.split(",")]:
            return False
        lines[i] = f"{ln.rstrip()}, {form}"
        break
    else:
        lines.append(f"{target} = {form}")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def _last_asr_terms() -> list:
    """Слова последней диктовки — то, что Whisper *услышал*, до правок руками.
    Биграммы тоже: термин часто разваливается надвое («сейф грип»)."""
    try:
        d = default_config_path().parent
        asr = json.loads((d / "last.json").read_text(encoding="utf-8-sig"))["asr"]
    except Exception:
        return []
    w = re.findall(r"[^\W\d_]+", asr, flags=re.UNICODE)
    return w + [f"{a} {b}" for a, b in zip(w, w[1:])]


def _osa_dialog(body: str):
    """Показать диалог от лица активного приложения.

    Через System Events нельзя: это фоновый процесс, его окна наверх не
    выходят — диалоги просто висели невидимыми (поймано 07.08.2026).
    """
    return subprocess.run(
        ["osascript",
         "-e", "set fa to path to frontmost application as text",
         "-e", f"tell application fa to {body}"],
        capture_output=True, text=True, timeout=120)


def _ask(question: str, default: str = "") -> str:
    """Однострочный диалог с полем ввода."""
    if platform.system() != "Darwin":
        return ""
    # ensure_ascii=False обязателен: \uXXXX AppleScript не разбирает и падает
    q, d = json.dumps(question, ensure_ascii=False), json.dumps(default, ensure_ascii=False)
    r = _osa_dialog(f'display dialog {q} default answer {d} '
                    'with title "QSpeak — словарь" '
                    'buttons {"Отмена", "Добавить"} default button "Добавить"')
    if r.returncode != 0:
        return ""
    return r.stdout.split("text returned:", 1)[-1].strip()


def _choose(items: list, prompt: str) -> str:
    """Выбор из списка вместо ввода: правка часто на другом алфавите
    («синкани» → «Xingyu»), похожесть по буквам там не находит ничего, а
    печатать кривой вариант руками — ровно та работа, от которой уходим."""
    if platform.system() != "Darwin" or not items:
        return ""
    lst = ", ".join(json.dumps(i, ensure_ascii=False) for i in items[:60])
    r = _osa_dialog(f'choose from list {{{lst}}} '
                    f'with prompt {json.dumps(prompt, ensure_ascii=False)} '
                    'with title "QSpeak — словарь"')
    out = r.stdout.strip()
    return "" if r.returncode != 0 or out == "false" else out


def copy_selection() -> str:
    """Cmd+C по текущему выделению → текст. Буфер возвращаем как был."""
    saved = save_clipboard()
    copy_to_clipboard("")          # пустой буфер = «ничего не выделено», а не старый текст
    try:
        subprocess.run(                     # key code 8 = физическая C, раскладка не важна
            ["osascript", "-e",
             'tell application "System Events" to key code 8 using command down'],
            check=True, capture_output=True, timeout=2)
    except Exception as e:
        logging.warning(f"copy selection failed: {e}")
    time.sleep(0.2)
    text = (_get_clipboard_text() or "").strip()
    restore_clipboard(saved)
    return text


def learn_selected_word() -> None:
    """Выделенное слово → правило словаря. Направление определяем сами.

    Слово нашлось в последней диктовке — значит выделено то, что Whisper
    услышал, и спросить надо правильное написание. Не нашлось — значит это
    уже исправленная руками форма, а кривую берём из той же диктовки по
    похожести, и диалог не нужен вовсе.
    """
    sel = " ".join(copy_selection().split())
    if not sel or len(sel) > 60:
        notify("Сначала выдели слово, потом двойной тап правого ⌘")
        return
    terms = _last_asr_terms()
    if any(t.lower() == sel.lower() for t in terms):
        target, form = _ask("Как это писать правильно?", sel), sel
        if not target or target.lower() == sel.lower():
            return
    else:
        import difflib
        near = difflib.get_close_matches(sel.lower(), [t.lower() for t in terms], n=1, cutoff=0.55)
        target = sel
        form = near[0] if near else _choose(terms, f"Что диктовка услышала вместо «{sel}»?")
        if not form:
            return
    msg = (f"{target} ← {form}" if add_vocabulary_rule(target, form)
           else f"«{form}» уже в словаре")
    print(f"📚 Словарь: {msg}")
    notify(msg)


def learn_from_last_pair() -> str:
    """Последняя пара из «Поправить последнее» → правила словаря.
    Слова, которые ты в диалоге изменил, и есть криво слышимые термины."""
    import difflib
    p = default_config_path().parent / "dataset" / "pairs.jsonl"
    try:
        last = json.loads(p.read_text(encoding="utf-8-sig").strip().splitlines()[-1])
    except Exception:
        return "эталонов нет"
    a = re.findall(r"[^\W\d_]+", last["asr"], flags=re.UNICODE)
    b = re.findall(r"[^\W\d_]+", last["truth"], flags=re.UNICODE)
    added = []
    ops = difflib.SequenceMatcher(a=[w.lower() for w in a], b=[w.lower() for w in b]).get_opcodes()
    for op, i1, i2, j1, j2 in ops:
        # длинные куски — это переписанная фраза, а не термин; в словарь им нельзя
        if op != "replace" or i2 - i1 > 3 or j2 - j1 > 3:
            continue
        form, target = " ".join(a[i1:i2]), " ".join(b[j1:j2])
        if add_vocabulary_rule(target, form):
            added.append(f"{target} ← {form}")
    if added:
        notify(", ".join(added), title="📚 Словарь")
    return ", ".join(added) if added else "новых правил нет"


def _split_on_silence(audio, sr: int, chunk_sec: float = 28.0, search_from: float = 0.78):
    """Нарезать длинное аудио на куски ~chunk_sec, разрезая в самом тихом месте.

    Нужно для смены языка внутри записи: Whisper определяет язык по первому
    30-секундному окну и применяет ко всей записи, поэтому английский хвост
    длинного монолога декодируется как русский и превращается в мусор.
    Каждый кусок распознаётся отдельно и получает свой язык.
    """
    import numpy as np

    parts, pos, n = [], 0, len(audio)
    step = int(chunk_sec * sr)
    while pos < n:
        end = pos + step
        if end >= n:
            parts.append(audio[pos:])
            break
        # Ищем паузу на всей второй половине куска, а не у самой границы:
        # смена языка обычно совпадает с настоящей паузой, и разрез должен
        # попасть в неё, иначе хвост чужого языка уедет в предыдущий кусок
        # и там потеряется.
        lo = pos + int(chunk_sec * search_from * sr)
        hi = min(n, end)
        win = int(0.2 * sr)
        seg = np.abs(audio[lo:hi])
        if len(seg) > win * 2:
            hops = np.array([seg[i:i + win].mean()
                             for i in range(0, len(seg) - win, win // 2)])
            # Берём ПЕРВУЮ настоящую паузу, а не самую тихую точку: смена языка
            # идёт сразу после неё, а глобальный минимум может оказаться паузой
            # между фразами уже нового языка — тогда его начало съест этот кусок.
            quiet = np.where(hops < max(hops.mean() * 0.25, 1e-4))[0]
            k = int(quiet[0]) if len(quiet) else int(np.argmin(hops))
            end = lo + k * (win // 2) + win // 2
        parts.append(audio[pos:end])
        pos = end
    return parts


# ─── Память: выгрузка модели по простою ─────────────────────────────────────

_unload_timer = None


def _clear_mlx_cache() -> None:
    """Отдать системе буферный кэш MLX. Веса остаются — это только то, что
    MLX держит про запас между вызовами и что копится от диктовки к диктовке."""
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass


def _release_model() -> None:
    """Выкинуть веса Whisper из памяти. Модель (~3 ГБ) висит между диктовками;
    на забитой машине её страницы уезжают в swap, и следующая расшифровка ждёт
    подкачку с диска (05.08.2026: 2865 → 212 кадров/с, ответ 1,1с → 30,8с)."""
    try:  # mlx: своя одиночная ячейка под модель
        from mlx_whisper.transcribe import ModelHolder
        ModelHolder.model = None
    except Exception:
        pass
    try:  # faster-whisper / whisperx: кэш в common
        from examples.common import _loaded_models
        _loaded_models.clear()
    except Exception:
        pass
    import gc
    gc.collect()
    _clear_mlx_cache()
    print("♻️  Модель выгружена из памяти (простой)", flush=True)


def _after_dictation(cfg: dict) -> None:
    """После каждой расшифровки: отдать кэш и перевести таймер выгрузки.
    `unload_after_idle_min: 0` в конфиге — держать модель всегда."""
    global _unload_timer
    _clear_mlx_cache()
    if _unload_timer:
        _unload_timer.cancel()
        _unload_timer = None
    mins = cfg.get("unload_after_idle_min", 15)
    if not mins:
        return
    _unload_timer = threading.Timer(mins * 60, _release_model)
    _unload_timer.daemon = True
    _unload_timer.start()


def transcribe_long(wav_path: str, cfg: dict) -> str:
    """Распознать запись целиком. Длинную — по кускам, чтобы каждый получил
    свой язык. Короткую (<30с) Whisper и так тянет с переключением языка."""
    import numpy as np
    import soundfile as sf

    lang = cfg.get("language")
    model = cfg.get("model")

    def _one(path):
        # импорт ленивый: бэкенд выбирается из конфига до первого импорта common
        from examples.common import transcribe
        return transcribe(path, language=lang, model_name=model,
                          word_timestamps=False, verbose=False).text.strip()

    audio, sr = sf.read(wav_path)
    # порог 30с — размер окна Whisper; ниже него делить нечего
    if lang is not None or len(audio) / sr <= 30.0:
        return _one(wav_path)

    texts = []
    for i, part in enumerate(_split_on_silence(audio, sr)):
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix=f"vd_part{i}_")
        sf.write(tmp.name, part, sr, subtype="PCM_16")
        try:
            t = _one(tmp.name)
            if t:
                texts.append(t)
        finally:
            try: os.unlink(tmp.name)
            except Exception: pass
    return " ".join(texts)


def history_path() -> Path:
    return default_config_path().parent / "history.md"


def save_to_history(text: str) -> None:
    """Дописать надиктованное в историю. Страховка от потери текста:
    вставка может уйти не в то окно, буфер — быть перетёрт."""
    try:
        p = history_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(f"\n## {time.strftime('%Y-%m-%d %H:%M:%S')}\n{text}\n")
    except Exception as e:
        logging.error(f"history write failed: {e}")


def save_last_take(wav_path: str, text: str) -> None:
    """Придержать последнюю запись рядом с расшифровкой.

    Без этого эталон брать неоткуда: wav удаляется сразу после расшифровки, а
    в истории лежит только то, что Whisper *услышал*. Пункт меню «Поправить
    последнее» превращает эту пару в обучающий пример.
    """
    try:
        import shutil
        d = default_config_path().parent
        shutil.copy(wav_path, d / "last.wav")
        (d / "last.json").write_text(
            json.dumps({"wav": str(d / "last.wav"), "asr": text}, ensure_ascii=False),
            encoding="utf-8")
    except Exception as e:
        logging.error(f"save_last_take failed: {e}")


def last_history_entry() -> str:
    """Текст последней диктовки — для пункта меню «Вставить последнее»."""
    try:
        blocks = history_path().read_text(encoding="utf-8-sig").split("\n## ")
    except OSError:
        return ""
    if len(blocks) < 2:
        return ""
    return "\n".join(blocks[-1].splitlines()[1:]).strip()


def save_clipboard() -> Optional[str]:
    """Снимок текстового содержимого буфера для последующего восстановления.

    None если буфер пуст / содержит не-текст (картинку, файл) / не удалось
    прочитать. В этих случаях restore тоже no-op — мы не пытаемся
    восстановить то, что не сохранили.
    """
    text = _get_clipboard_text()
    if text is None:
        logging.info("clipboard save: empty or non-text, restore will be skipped")
    return text


def restore_clipboard(saved: Optional[str]) -> None:
    """Положить сохранённое содержимое обратно с verify-and-retry.

    После записи читаем буфер и сравниваем; если не совпало — повторяем
    до 3 попыток. Защита от того, что в момент нашей записи буфер был
    занят другим процессом (Win+V history listener, clipboard manager).
    """
    if not saved:
        return
    for attempt in range(3):
        copy_to_clipboard(saved)
        current = _get_clipboard_text()
        if current == saved:
            return
        time.sleep(0.05)
    logging.warning(
        f"clipboard restore not verified after 3 attempts "
        f"(expected len={len(saved)}, got len={len(current) if current else 0})"
    )


def restore_clipboard_deferred(saved: Optional[str], delay_sec: float = 1.0) -> None:
    """Восстановить буфер с задержкой в отдельном потоке.

    SendInput возвращается синхронно, но таргет-приложение обрабатывает
    Ctrl+V асинхронно: сначала помещает WM_PASTE в очередь, обработчик
    читает буфер в свою очередь. На медленных таргетах (Chrome,
    Electron, web-приложения вроде ChatGPT/Claude) между нашим SendInput
    и реальным чтением буфера может пройти 200–800 мс. Если восстановить
    буфер слишком быстро — приложение прочитает уже восстановленный
    оригинал, а не диктованный текст. delay_sec=1.0 покрывает медленные
    таргеты с запасом.
    """
    if not saved:
        return

    def _do():
        time.sleep(delay_sec)
        restore_clipboard(saved)

    threading.Thread(target=_do, daemon=True).start()


def _windows_paste() -> None:
    """Надёжная симуляция Ctrl+V на Windows через Win32 SendInput.

    Принудительно отпускает все возможные "залипшие" модификаторы
    (после хоткея типа Ctrl+Alt пользователь может ещё их удерживать),
    затем выполняет чистый Ctrl+V.
    """
    import ctypes
    import time as _t
    user32 = ctypes.windll.user32

    KEYEVENTF_KEYUP = 0x0002
    VK = {
        "ctrl": 0x11, "lctrl": 0xA2, "rctrl": 0xA3,
        "alt": 0x12, "lalt": 0xA4, "ralt": 0xA5,
        "shift": 0x10, "lshift": 0xA0, "rshift": 0xA1,
        "lwin": 0x5B, "rwin": 0x5C,
        "v": 0x56,
    }
    # 1) Release any held modifiers (idempotent — release of unpressed key is no-op)
    for name in ("lctrl", "rctrl", "ctrl", "lalt", "ralt", "alt",
                 "lshift", "rshift", "shift", "lwin", "rwin"):
        user32.keybd_event(VK[name], 0, KEYEVENTF_KEYUP, 0)
    _t.sleep(0.03)
    # 2) Clean Ctrl+V
    user32.keybd_event(VK["ctrl"], 0, 0, 0)
    user32.keybd_event(VK["v"], 0, 0, 0)
    _t.sleep(0.02)
    user32.keybd_event(VK["v"], 0, KEYEVENTF_KEYUP, 0)
    user32.keybd_event(VK["ctrl"], 0, KEYEVENTF_KEYUP, 0)


def notify(text: str, title: str = "🎙 Диктовка") -> None:
    """Нативный баннер macOS — визуальный фидбэк без терминала.
    No-op на других ОС и при ошибке (уведомление не критично)."""
    if platform.system() != "Darwin":
        return
    try:
        safe = text.replace('"', "'").replace("\\", "")[:180]
        subprocess.run(
            ["osascript", "-e",
             f'display notification "{safe}" with title "{title}"'],
            check=False, capture_output=True, timeout=2,
        )
    except Exception:
        pass


def erase_chars(n: int) -> None:
    """Стереть N символов назад — нужно, чтобы заменить черновик потоковой
    диктовки на финальный (AI-выправленный) текст."""
    if n <= 0:
        return
    try:
        from pynput.keyboard import Controller, Key
        kb = Controller()
        for _ in range(n):
            kb.press(Key.backspace)
            kb.release(Key.backspace)
    except Exception as e:
        logging.error(f"erase_chars failed: {e}")


def paste_from_clipboard() -> None:
    """Симулировать Cmd+V (Mac) или Ctrl+V (Linux/Win).

    На macOS приоритет — osascript: он работает через System Events, для которого
    разрешение Accessibility даётся один раз на Terminal/iTerm, и срабатывает
    надёжно. pynput-путь оставлен как fallback (на случай отсутствия osascript
    или сломанного System Events).

    На Linux/Windows используется pynput напрямую.
    """
    # macOS: предпочитаем osascript (надёжнее с защитой Accessibility)
    if platform.system() == "Darwin":
        try:
            # key code 9 = физическая клавиша V. НЕ зависит от раскладки
            # (keystroke "v" при русской раскладке не находит клавишу → Cmd+V не срабатывает).
            subprocess.run(
                ["osascript", "-e",
                 'tell application "System Events" to key code 9 using command down'],
                check=True, capture_output=True, timeout=2,
            )
            return
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or b"").decode(errors="replace")
            logging.warning(f"osascript paste failed ({stderr.strip()}), trying pynput fallback")
        except Exception as e:
            logging.warning(f"osascript paste failed: {e}, trying pynput fallback")

    # Fallback и основной путь для Linux/Windows
    try:
        if platform.system() == "Windows":
            _windows_paste()
        elif platform.system() == "Darwin":
            from pynput.keyboard import Controller, Key, KeyCode
            kb = Controller()
            v = KeyCode.from_vk(9)  # физическая V, независимо от раскладки
            with kb.pressed(Key.cmd):
                kb.press(v)
                kb.release(v)
        else:
            from pynput.keyboard import Controller, Key
            kb = Controller()
            with kb.pressed(Key.ctrl):
                kb.press("v")
                kb.release("v")
    except Exception as e:
        logging.error(
            f"paste simulation failed: {e}\n"
            f"Текст в clipboard — вставь вручную через Cmd+V/Ctrl+V."
        )


def play_beep(frequency: int = 800, duration_ms: int = 80) -> None:
    """Короткий синтезированный бип через sounddevice. Сохранён для
    обратной совместимости и для платформ без winsound (Linux/macOS).

    На Windows предпочитай play_beep_system — он громче, гарантированно
    слышен и не конфликтует с активным sd.InputStream (запись микрофона).
    """
    try:
        import numpy as np
        import sounddevice as sd
        sample_rate = 44100
        t = np.linspace(0, duration_ms / 1000, int(sample_rate * duration_ms / 1000), False)
        tone = 0.15 * np.sin(2 * np.pi * frequency * t)
        fade = int(sample_rate * 0.005)
        envelope = np.ones_like(tone)
        envelope[:fade] = np.linspace(0, 1, fade)
        envelope[-fade:] = np.linspace(1, 0, fade)
        tone = tone * envelope
        sd.play(tone.astype(np.float32), sample_rate, blocking=True)
    except Exception:
        pass


def _make_dual_beep_wav(
    f1: int, f2: int, dur_ms: int = 60, gap_ms: int = 40,
    sample_rate: int = 22050, vol: float = 0.01,
    tail_silence_ms: int = 80,
) -> bytes:
    """Сгенерировать in-memory WAV с двумя тонами через паузу.

    Возвращает байты PCM-WAV пригодные для winsound.PlaySound(SND_MEMORY).

    tail_silence_ms — хвост тишины после второго тона. Нужен потому, что
    Windows audio mixer иногда обрезает последние ~30-50ms короткого WAV
    (артефакт буферизации). Просто добавляем «зазор» из нулей.
    """
    import math as _m
    import struct

    def _tone_samples(freq: int, dur_ms: int) -> list:
        n = int(sample_rate * dur_ms / 1000)
        fade_n = max(1, int(sample_rate * 0.015))  # 15ms fade — длинный ramp убирает крякание BT-кодеков на attack
        out = []
        two_pi_f = 2.0 * _m.pi * freq
        for i in range(n):
            env = 1.0
            if i < fade_n:
                env = i / fade_n
            elif i > n - fade_n:
                env = max(0.0, (n - i) / fade_n)
            sample = int(32767 * vol * env * _m.sin(two_pi_f * (i / sample_rate)))
            out.append(struct.pack("<h", sample))
        return out

    silence_samples = [b"\x00\x00"] * int(sample_rate * gap_ms / 1000)
    tail_samples = [b"\x00\x00"] * int(sample_rate * tail_silence_ms / 1000)
    samples = (
        _tone_samples(f1, dur_ms) + silence_samples
        + _tone_samples(f2, dur_ms) + tail_samples
    )
    data = b"".join(samples)

    # 16-bit mono PCM WAV header
    fmt_chunk = struct.pack(
        "<4sIHHIIHH",
        b"fmt ", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
    )
    data_chunk = struct.pack("<4sI", b"data", len(data)) + data
    riff = struct.pack("<4sI4s", b"RIFF", 4 + len(fmt_chunk) + len(data_chunk), b"WAVE")
    return riff + fmt_chunk + data_chunk


def _make_single_beep_wav(
    freq: int = 600, dur_ms: int = 100,
    sample_rate: int = 22050, vol: float = 0.01,
    fade_ms: int = 10, tail_silence_ms: int = 80,
) -> bytes:
    """Однотоновый WAV. Мягкий, не сливается с речью — для стоп-сигнала."""
    import math as _m
    import struct

    n = int(sample_rate * dur_ms / 1000)
    fade_n = max(1, int(sample_rate * fade_ms / 1000))
    two_pi_f = 2.0 * _m.pi * freq
    tone = []
    for i in range(n):
        env = 1.0
        if i < fade_n:
            env = i / fade_n
        elif i > n - fade_n:
            env = max(0.0, (n - i) / fade_n)
        sample = int(32767 * vol * env * _m.sin(two_pi_f * (i / sample_rate)))
        tone.append(struct.pack("<h", sample))
    tail = [b"\x00\x00"] * int(sample_rate * tail_silence_ms / 1000)
    data = b"".join(tone + tail)

    fmt_chunk = struct.pack(
        "<4sIHHIIHH",
        b"fmt ", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
    )
    data_chunk = struct.pack("<4sI", b"data", len(data)) + data
    riff = struct.pack("<4sI4s", b"RIFF", 4 + len(fmt_chunk) + len(data_chunk), b"WAVE")
    return riff + fmt_chunk + data_chunk


# Pre-render the two beep WAVs once at import time — playing them later
# is then just a fire-and-forget winsound call.
_BEEP_WAV_START: Optional[bytes] = None
_BEEP_WAV_STOP: Optional[bytes] = None
try:
    _BEEP_WAV_START = _make_dual_beep_wav(700, 900)  # rising
    _BEEP_WAV_STOP = _make_single_beep_wav(600, 100)
except Exception:
    pass


def _play_wav_bytes(wav: bytes) -> None:
    """Проиграть готовые WAV-байты через основную звуковую карту."""
    if platform.system() == "Windows":
        try:
            import winsound
            winsound.PlaySound(wav, winsound.SND_MEMORY | winsound.SND_NODEFAULT)
            return
        except Exception as e:
            logging.error(f"winsound play failed: {e}")


# macOS: системные звуки через afplay. Синтез через sounddevice здесь не годится —
# он конфликтует с уже открытым InputStream микрофона.
# Purr на старт (мягкий, тянется) + Morse на стоп (резкий обрыв): пара читается
# как «поехали» / «всё». Submarine на стопе звучал как начало, Glass занят под
# уведомления VS Code. Переопределяется в конфиге:
# "sounds": {"start": "/System/Library/Sounds/Morse.aiff", ...},
# громкость — "sound_volume": 0..1. Список системных: ls /System/Library/Sounds
MAC_SOUND_START = "/System/Library/Sounds/Purr.aiff"
MAC_SOUND_STOP = "/System/Library/Sounds/Morse.aiff"
MAC_SOUND_DONE = "/System/Library/Sounds/Pop.aiff"
_SOUND_CFG: dict = {}


def configure_sounds(cfg: dict) -> None:
    """Забрать звуки и громкость из конфига — play_*_beep зовутся без cfg."""
    _SOUND_CFG.update(cfg.get("sounds") or {})
    _SOUND_CFG["volume"] = str(cfg.get("sound_volume", 1.0))


def _play_mac_sound(path: str, volume: str = "") -> None:
    volume = volume or _SOUND_CFG.get("volume", "1.0")
    try:
        subprocess.Popen(["afplay", "-v", volume, path],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        logging.error(f"afplay failed: {e}")


def play_start_beep() -> None:
    if platform.system() == "Darwin":
        _play_mac_sound(_SOUND_CFG.get("start", MAC_SOUND_START))
    elif _BEEP_WAV_START is not None:
        _play_wav_bytes(_BEEP_WAV_START)


def play_stop_beep() -> None:
    if platform.system() == "Darwin":
        _play_mac_sound(_SOUND_CFG.get("stop", MAC_SOUND_STOP))
    elif _BEEP_WAV_STOP is not None:
        _play_wav_bytes(_BEEP_WAV_STOP)


def play_done_beep() -> None:
    """Третий сигнал — текст вставлен, можно продолжать. Тише прочих: звучит
    в момент, когда пользователь уже смотрит на вставленный текст."""
    if platform.system() == "Darwin":
        quiet = float(_SOUND_CFG.get("volume", 1.0)) * 0.6
        _play_mac_sound(_SOUND_CFG.get("done", MAC_SOUND_DONE), volume=str(quiet))


def play_dual_beep(f1: int, f2: int, dur_ms: int = 60, gap_ms: int = 40) -> None:
    """Двутоновый бип на произвольных частотах. Синтезирует WAV каждый раз —
    использовать только для редких/нестандартных тонов; для обычных
    старт/стоп есть play_start_beep / play_stop_beep с предрендеренным WAV.
    """
    if platform.system() == "Windows":
        try:
            _play_wav_bytes(_make_dual_beep_wav(f1, f2, dur_ms, gap_ms))
            return
        except Exception as e:
            logging.error(f"winsound dual beep failed: {e}")
    try:
        play_beep(f1, dur_ms)
        if gap_ms > 0:
            time.sleep(gap_ms / 1000.0)
        play_beep(f2, dur_ms)
    except Exception:
        pass


# ─── Tray icon ──────────────────────────────────────────────────────────────


class TrayIcon:
    """Иконка в трее. Показывает текущее состояние цветом."""

    def __init__(self):
        self.icon = None
        self._ready = False

    def start(self, current_model: Optional[str] = None,
              available_models: Optional[list] = None,
              on_select_model=None):
        """current_model / available_models / on_select_model — для подменю
        "Модель". on_select_model(name) вызывается при клике; обычно делает
        write_config + restart_self()."""
        try:
            import pystray
            from PIL import Image, ImageDraw

            self._images = self._build_images()

            menu_entries = []
            if available_models:
                def _make_handler(name):
                    return lambda icon, item: on_select_model and on_select_model(name)

                def _make_check(name):
                    return lambda item: current_model == name

                model_items = [
                    pystray.MenuItem(
                        m, _make_handler(m),
                        checked=_make_check(m), radio=True,
                    )
                    for m in available_models
                ]
                menu_entries.append(
                    pystray.MenuItem("Model", pystray.Menu(*model_items))
                )

            menu_entries.append(pystray.MenuItem("Quit", lambda: self.icon.stop()))

            self.icon = pystray.Icon(
                "voice_dictation",
                self._images["idle"],
                "Whisper Voice Dictation",
                menu=pystray.Menu(*menu_entries),
            )
            threading.Thread(target=self.icon.run, daemon=True).start()
            self._ready = True
        except Exception as e:
            logging.warning(f"Tray icon disabled: {e}", exc_info=True)

    @staticmethod
    def _build_images():
        from PIL import Image, ImageDraw

        size = 64
        # Базовая иконка — assets/icon.png рядом с репо. Если её нет
        # (минимальная установка) — fallback на серый круг.
        repo_root = Path(__file__).resolve().parents[1]
        icon_path = repo_root / "assets" / "icon.png"
        if icon_path.exists():
            base = Image.open(icon_path).convert("RGBA").resize((size, size), Image.LANCZOS)
        else:
            base = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            d = ImageDraw.Draw(base)
            d.ellipse((8, 8, 56, 56), fill="#666666")

        def _with_dot(color: Optional[str]):
            img = base.copy()
            if color is not None:
                d = ImageDraw.Draw(img)
                # Точка-индикатор поверх встроенной красной точки логотипа
                # (правый нижний угол) — повторяет позицию dot'а возле курсора.
                d.ellipse((size - 26, size - 26, size - 4, size - 4),
                          fill=color, outline="white", width=2)
            return img

        return {
            "idle": _with_dot(None),
            "recording": _with_dot("#e63946"),
            "transcribing": _with_dot("#f4a261"),
        }

    def set_state(self, state: str):
        if not self._ready or not self.icon:
            return
        img = self._images.get(state)
        if img:
            self.icon.icon = img


# ─── Hotkey-driven main loop ────────────────────────────────────────────────


def _warmup(transcribe_fn, cfg: dict, tray) -> None:
    """Прогрев модели в фоне — компилирует OV-граф / прогружает веса.

    Без warmup'а первый Ctrl+Alt тратит 5–30 сек на cold start (особенно
    на OpenVINO + iGPU при первом compile=True). Запись на короткий буфер
    тишины, результат игнорируем.
    """
    try:
        import wave
        # 0.5 сек тишины 16k mono int16 — минимум, который не отлетает по VAD
        sample_rate = 16000
        silence = b"\x00\x00" * (sample_rate // 2)
        tmp = Path(tempfile.gettempdir()) / "whisper_skill_warmup.wav"
        with wave.open(str(tmp), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(silence)

        t0 = time.time()
        transcribe_fn(
            str(tmp),
            language=cfg.get("language"),
            model_name=cfg.get("model"),
            backend=cfg.get("backend"),
            word_timestamps=False,
            verbose=False,
        )
        logging.info(f"warmup done in {time.time() - t0:.1f}s")
    except Exception as e:
        # Прогрев — best-effort. Если упал, обычная диктовка продолжит работать
        # как раньше: просто первый запрос будет холодным.
        logging.info(f"warmup failed (non-fatal): {e}")


@dataclass
class State:
    is_recording: bool = False
    is_transcribing: bool = False
    last_dictation_at: float = 0.0  # time.time() последней успешной вставки


def main_loop(cfg: dict, cfg_path: Path):
    from pynput import keyboard

    # Lazy-import чтобы не падать на импортов при ошибке отсутствия пакетов
    try:
        from examples.common import transcribe
    except Exception as e:
        print(f"❌ Не могу загрузить examples.common: {e}", file=sys.stderr)
        print("Запусти из корня whisper-skill: cd whisper-skill && python -m examples.voice_dictation", file=sys.stderr)
        return 1

    state = State()
    state_lock = threading.Lock()
    recorder = AudioRecorder(cfg["sample_rate"], cfg["channels"])

    # macOS low-CPU mode: pystray в фоне и Tk у нас вызывают серьёзный
    # idle-CPU на маке (наблюдалось ~90% на M-чипе). Дефолтно отключаем оба
    # GUI-feedback'а на Mac. Юзер видит CLI-stdout (Recording/Transcribing/✓).
    is_mac_low_cpu = (
        platform.system() == "Darwin"
        and cfg.get("mac_low_cpu_mode", True)
    )
    if is_mac_low_cpu:
        if cfg.get("show_tray") or cfg.get("show_cursor_indicator"):
            logging.info(
                "macOS: tray и cursor_indicator отключены ради экономии CPU. "
                "Чтобы включить — поставь mac_low_cpu_mode: false в конфиге."
            )

    def _on_select_model(new_model: str):
        if new_model == cfg.get("model"):
            return
        try:
            cur = load_config(cfg_path)
            cur["model"] = new_model
            write_config(cfg_path, cur)
        except Exception as e:
            logging.error(f"failed to write new model to config: {e}")
            return
        print(f"🔁 Переключаю модель → {new_model}, перезапуск...")
        # restart_self спавнит новую копию через VBS launcher и os._exit'ит текущую.
        # Новая копия дождётся освобождения mutex (retry в acquire_single_instance_lock).
        restart_self()

    tray = TrayIcon()
    if cfg.get("show_tray") and not is_mac_low_cpu:
        tray.start(
            current_model=cfg.get("model"),
            available_models=list_available_ov_models(),
            on_select_model=_on_select_model,
        )

    # Прогрев модели в фоне: первый hotkey-press не должен ждать
    # компиляцию OpenVINO-графа / загрузку весов faster-whisper.
    # Event сигнализирует завершение warmup'а — work() ждёт его перед
    # первым transcribe. Это решает две проблемы одним механизмом:
    #   1) Cold start первой диктовки: без ожидания первый hotkey ловил
    #      холодную модель + компиляцию OV-графа (5–30с задержка).
    #   2) Race на module import: warmup-thread и work-thread параллельно
    #      делают `from optimum.intel import OVModelForSpeechSeq2Seq`.
    #      transformers._LazyModule под Python 3.12 даёт partially-initialized
    #      module второму thread'у → AttributeError → Python преобразует в
    #      ImportError на первой диктовке.
    warmup_done = threading.Event()
    if cfg.get("warmup", True):
        def _warmup_then_signal():
            try:
                _warmup(transcribe, cfg, tray)
            finally:
                # set даже на failure — иначе work() заблокируется навсегда.
                # Если warmup упал, первый transcribe сам потерпит cold start
                # — это лучше чем deadlock.
                warmup_done.set()

        threading.Thread(target=_warmup_then_signal, daemon=True).start()
    else:
        warmup_done.set()

    # Cursor indicator (small blinking dot near the mouse cursor while recording).
    # Optional — silently disables if Tk unavailable. На macOS всегда no-op
    # (см. scripts/cursor_indicator.py — Tk thread-safety issue).
    cursor_ind = None
    # Меню в статус-баре живёт в HUD-процессе и шлёт команды сюда. enabled —
    # только в памяти: тумблер на время, а не настройка на всю жизнь.
    menu_state = {"enabled": True}
    if platform.system() == "Darwin" and cfg.get("show_hud", True):
        # На macOS Tk-индикатор не работает (main-thread), поэтому отдельный
        # процесс с Cocoa-панелью внизу экрана.
        try:
            from scripts.hud_mac import MacHUD
            # lambda, а не сама функция: _on_menu_command определён ниже, а
            # команды приходят уже после сборки main_loop.
            cursor_ind = MacHUD(on_command=lambda c: _on_menu_command(c))
            cursor_ind.start()
        except Exception as e:
            logging.error(f"macOS HUD init failed: {e}")
            cursor_ind = None
    elif cfg.get("show_cursor_indicator", True) and not is_mac_low_cpu:
        try:
            from scripts.cursor_indicator import CursorIndicator
            cursor_ind = CursorIndicator(color=cfg.get("cursor_indicator_color", "#ef4444"))
            cursor_ind.start()
        except Exception as e:
            logging.error(f"cursor indicator init failed: {e}")
            cursor_ind = None

    # Состояние потоковой диктовки: что уже вставлено в поле по ходу речи.
    stream_state = {"emitted": "", "prev_words": []}

    def _stream_worker():
        """Пока идёт запись — периодически распознаём накопленный звук быстрой
        моделью и дописываем в поле только «устоявшиеся» слова (LocalAgreement)."""
        interval = float(cfg.get("stream_interval_sec", 1.8))
        fast_model = cfg.get("stream_model", "mlx-community/whisper-large-v3-turbo")
        while True:
            time.sleep(interval)
            if not state.is_recording:
                return
            wav = recorder.snapshot()
            if not wav:
                print("[stream] нет аудио в снапшоте", flush=True)
                continue
            try:
                res = transcribe(
                    wav, language=cfg.get("language"), model_name=fast_model,
                    word_timestamps=False, verbose=False,
                )
                words = apply_vocabulary(res.text.strip(), load_vocabulary()).split()
                print(f"[stream] снапшот → {len(words)} слов: {' '.join(words)[:80]}", flush=True)
            except Exception as e:
                print(f"[stream] transcribe FAILED: {e}", flush=True)
                continue
            finally:
                try: os.unlink(wav)
                except Exception: pass

            if not state.is_recording:
                return
            stable = _common_prefix_words(stream_state["prev_words"], words)
            stream_state["prev_words"] = words
            already = stream_state["emitted"].split()
            if len(stable) > len(already):
                delta = " ".join(stable[len(already):])
                # Расшифровка снапшота небыстрая: за это время запись могли
                # отменить. Вставлять черновик после отмены — мусор в поле.
                if delta and state.is_recording:
                    chunk = (" " if stream_state["emitted"] else "") + delta
                    copy_to_clipboard(chunk)
                    paste_from_clipboard()
                    stream_state["emitted"] = (
                        stream_state["emitted"] + chunk if stream_state["emitted"] else delta
                    )

    auto_stop = {"t": None}   # сторожевой таймер записи, max_duration_sec
    cancelled = {"v": False}  # отмена, пойманная уже во время расшифровки

    def start_recording():
        if not menu_state["enabled"]:   # тумблер в меню — глушим на входе, а не в каждом хоткее
            return
        with state_lock:
            if state.is_recording or state.is_transcribing:
                return
            state.is_recording = True
        tray.set_state("recording")
        if cursor_ind:
            cursor_ind.show()
        if cfg.get("play_sound"):
            threading.Thread(target=play_start_beep, daemon=True).start()
        try:
            recorder.start()
            # Ключ max_duration_sec до сих пор был мёртвым: в конфиге есть,
            # в коде не использовался. Страховка от «нажал и ушёл».
            limit = float(cfg.get("max_duration_sec") or 0)
            if limit > 0:
                auto_stop["t"] = threading.Timer(limit, stop_and_transcribe)
                auto_stop["t"].daemon = True
                auto_stop["t"].start()
            if cursor_ind and hasattr(cursor_ind, "set_level"):
                def _feed_level():
                    while state.is_recording:
                        cursor_ind.set_level(recorder.level)
                        time.sleep(1 / 25)
                threading.Thread(target=_feed_level, daemon=True).start()
            print("🎙  Recording... (release hotkey to transcribe)")
            if cfg.get("show_notification", True):
                threading.Thread(target=notify, args=("Слушаю… (тап ⌥ чтобы закончить)",), daemon=True).start()
            if cfg.get("streaming", False):
                stream_state["emitted"] = ""
                stream_state["prev_words"] = []
                threading.Thread(target=_stream_worker, daemon=True).start()
        except Exception as e:
            print(f"❌ Recording failed: {e}")
            state.is_recording = False
            tray.set_state("idle")
            if cursor_ind:
                cursor_ind.hide()

    def cancel_recording():
        """Передумал: выбросить запись, ничего не расшифровывать, черновик
        стриминга стереть из поля. Esc во время записи или крестик на HUD."""
        with state_lock:
            if not state.is_recording:
                # расшифровка уже идёт — прервать mlx нельзя, но результат
                # можно выбросить: он останется только в истории
                if state.is_transcribing:
                    cancelled["v"] = True
                    print("✖ Отменяю расшифровку")
                return
            state.is_recording = False
        if auto_stop["t"]:
            auto_stop["t"].cancel()
            auto_stop["t"] = None
        wav_path = recorder.stop()
        streamed = stream_state.get("emitted", "")
        if streamed and cfg.get("auto_paste"):
            erase_chars(len(streamed))
        stream_state["emitted"] = ""
        stream_state["prev_words"] = []
        if wav_path:
            try: os.unlink(wav_path)
            except OSError: pass
        tray.set_state("idle")
        if cursor_ind:
            cursor_ind.hide()
        print("✖ Запись отменена")

    def stop_and_transcribe():
        with state_lock:
            if not state.is_recording:
                return
            state.is_recording = False
        if auto_stop["t"]:
            auto_stop["t"].cancel()
            auto_stop["t"] = None
        tray.set_state("transcribing")
        # Точка → катушка ровно в той же позиции возле курсора. Скрываем
        # индикатор только когда текст уже вставлен (в work() finally) или
        # на ранних выходах ниже.
        if cursor_ind:
            cursor_ind.show_transcribing()

        # Сначала закрываем микрофон, потом играем бип. Параллельный запуск
        # winsound во время stream.close() PortAudio даёт повторное звучание
        # (наблюдалось 2026-05-01: один вызов PlaySound → два слышимых тона).
        wav_path = recorder.stop()
        if cfg.get("play_sound"):
            threading.Thread(target=play_stop_beep, daemon=True).start()
        if not wav_path:
            tray.set_state("idle")
            if cursor_ind:
                cursor_ind.hide()
            return
        duration_ms = recorder.duration_sec * 1000

        if duration_ms < cfg.get("min_duration_ms", 300):
            print(f"⏭  Skipped (too short: {duration_ms:.0f}ms)")
            try: os.unlink(wav_path)
            except: pass
            tray.set_state("idle")
            if cursor_ind:
                cursor_ind.hide()
            return

        print(f"⏳ Transcribing {duration_ms:.0f}ms of audio...")
        state.is_transcribing = True

        def work():
            try:
                if not warmup_done.is_set():
                    print("⏳ Waiting for model warmup to finish...")
                    warmup_done.wait()
                t0 = time.time()
                text = apply_vocabulary(transcribe_long(wav_path, cfg), load_vocabulary())
                elapsed = time.time() - t0

                if not text:
                    print("⏭  Empty transcription")
                else:
                    print(f"✓ ({elapsed:.1f}s) → {text}")
                    # Пишем в историю ДО вставки: если фокус ушёл не туда или
                    # вставка сорвалась — текст всё равно не потерян.
                    save_to_history(text)
                    save_last_take(wav_path, text)

                    if cancelled["v"]:
                        cancelled["v"] = False
                        draft = stream_state.get("emitted", "")
                        if draft and cfg.get("auto_paste"):
                            erase_chars(len(draft))
                        stream_state["emitted"] = ""
                        stream_state["prev_words"] = []
                        print("✖ Отменено — текст остался только в истории")
                        return

                    # AI-правка (аналог «AI mode» у Wispr Flow): пунктуация,
                    # заглавные, чистка оговорок. Смысл не меняется.
                    if cfg.get("ai_cleanup", False):
                        polished = ai_cleanup(text, cfg.get("ai_timeout_sec", 30.0)).strip()
                        if polished:
                            if polished != text:
                                print(f"✎ AI → {polished}")
                            text = polished

                    if cfg.get("show_notification", True):
                        threading.Thread(target=notify, args=(text, "✓ Вставлено"), daemon=True).start()

                    streamed = stream_state.get("emitted", "")
                    if streamed:
                        # Потоковый режим: черновик уже в поле — стираем его и
                        # кладём финальный текст одним куском.
                        saved_clipboard = save_clipboard() if cfg.get("auto_paste") else None
                        if cfg.get("auto_paste"):
                            erase_chars(len(streamed))
                            copy_to_clipboard(text)
                            time.sleep(0.15)
                            paste_from_clipboard()
                            if not cfg.get("keep_in_clipboard", True):
                                restore_clipboard_deferred(saved_clipboard, delay_sec=1.0)
                        else:
                            copy_to_clipboard(text)
                        stream_state["emitted"] = ""
                        stream_state["prev_words"] = []
                        state.last_dictation_at = time.time()
                        return

                    # Если предыдущая диктовка была недавно — начинаем
                    # новую с переноса строки. Порог 30s — «продолжаем
                    # в то же место»; после большой паузы вставка чистая.
                    newline_threshold_s = cfg.get("newline_after_dictation_within_sec", 30.0)
                    now = time.time()
                    if state.last_dictation_at and (now - state.last_dictation_at) < newline_threshold_s:
                        text_to_paste = "\n" + text
                    else:
                        text_to_paste = text

                    saved_clipboard = save_clipboard() if cfg.get("auto_paste") else None
                    copy_to_clipboard(text_to_paste)
                    if cfg.get("auto_paste"):
                        time.sleep(0.25)  # дать целевому полю стать активным
                        paste_from_clipboard()
                        # keep_in_clipboard: диктованный текст остаётся в буфере,
                        # чтобы вставить его Cmd+V куда угодно, если фокус ушёл
                        # не туда. Иначе — возвращаем прежнее содержимое буфера
                        # (асинхронно через 1с, чтобы таргет успел обработать вставку).
                        if not cfg.get("keep_in_clipboard", True):
                            restore_clipboard_deferred(saved_clipboard, delay_sec=1.0)
                    if cfg.get("play_sound"):
                        threading.Thread(target=play_done_beep, daemon=True).start()
                    state.last_dictation_at = now
            except Exception as e:
                print(f"❌ Transcription failed: {e}")
            finally:
                state.is_transcribing = False
                tray.set_state("idle")
                if cursor_ind:
                    cursor_ind.hide()
                try: os.unlink(wav_path)
                except: pass
                _after_dictation(cfg)

        threading.Thread(target=work, daemon=True).start()

    def toggle():
        if state.is_recording:
            stop_and_transcribe()
        else:
            start_recording()

    def _on_menu_command(cmd: str) -> None:
        """Клик в меню статус-бара (протокол — в scripts/hud_mac.py)."""
        parts = cmd.split(maxsplit=2)
        if parts[0] == "enabled" and len(parts) == 2:
            menu_state["enabled"] = parts[1] == "1"
            if not menu_state["enabled"] and state.is_recording:
                stop_and_transcribe()
            print(f"🎛 Диктовка {'включена' if menu_state['enabled'] else 'выключена'}")
        elif parts[0] == "set" and len(parts) == 3:
            key, raw = parts[1], parts[2]
            val = {"auto": None, "0": False, "1": True}.get(raw, raw)
            cfg[key] = val
            try:
                cur = load_config(cfg_path)
                cur[key] = val
                write_config(cfg_path, cur)
            except Exception as e:
                logging.error(f"config write failed: {e}")
            print(f"🎛 {key} → {val}")
            if key in ("model", "hotkey"):
                # листенер клавиш собран на старте под конкретный хоткей,
                # модель держится в памяти — и то и другое меняется перезапуском
                print(f"🔁 Перезапуск: {key} изменён...")
                restart_self()
        elif cmd == "cancel":
            cancel_recording()
        elif cmd == "paste_last":
            text = last_history_entry()
            if text:
                copy_to_clipboard(text)
                time.sleep(0.15)
                paste_from_clipboard()
                print(f"📋 Вставлено из истории: {text[:60]}")
        elif cmd == "vocab_learn":
            print(f"📚 Словарь: {learn_from_last_pair()}")
        elif parts[0] == "log":
            print(f"🎛 {cmd[4:]}")
        elif cmd == "quit_app":
            print("👋 Выход из меню")
            os._exit(0)

    hotkey_str = cfg["hotkey"]
    print(f"🎤 Whisper Voice Dictation активна")
    print(f"   Хоткей: {hotkey_str} ({cfg['mode']})")
    print(f"   Модель: {cfg['model']}")
    print(f"   Язык:   {cfg.get('language') or 'auto'}")
    print(f"\nНажми {hotkey_str} чтобы говорить. Ctrl+C чтобы выйти.\n")

    # Оба режима — через один Listener. GlobalHotKeys в pynput не ловит одиночный
    # модификатор (напр. правый Option), поэтому toggle тоже отслеживаем сами.
    keys_needed = _parse_hotkey(hotkey_str)
    currently_pressed = set()

    if cfg["mode"] == "ptt":
        # Push-to-talk: нажал → запись, отпустил → транскрибировать
        def on_press(key):
            currently_pressed.update(_canonical_keys(key))
            if keys_needed.issubset(currently_pressed):
                start_recording()

        def on_release(key):
            names = _canonical_keys(key)
            if (names & keys_needed) and state.is_recording:
                stop_and_transcribe()
            currently_pressed.difference_update(names)
    else:
        # Toggle: тап → старт, тап ещё раз → стоп. Срабатываем на ФРОНТЕ нажатия
        # (armed), чтобы автоповтор/удержание не переключали запись многократно.
        armed = {"v": True}

        # Хоткей из одного модификатора, который живёт и в комбинациях (⌃, ⌥ —
        # Ctrl+C, ⌥+клик, Ctrl+стрелки): фронт нажатия ловить нельзя, иначе запись
        # стартует от обычной работы. Для них срабатываем по ТАПУ — нажал и отпустил,
        # не тронув по дороге ни другой клавиши, ни мыши. Правые модификаторы ни на
        # что не назначены, поэтому им тап не нужен и остаётся мгновенный фронт.
        tap_only = keys_needed <= {"ctrl", "alt", "cmd", "shift"}
        tap = {"clean": False, "t0": 0.0}
        TAP_MAX_SEC = 0.6

        def on_press(key):
            if key == keyboard.Key.esc and (state.is_recording or state.is_transcribing):
                cancel_recording()
                return
            names = _canonical_keys(key)
            was = keys_needed.issubset(currently_pressed)
            currently_pressed.update(names)
            now = keys_needed.issubset(currently_pressed)
            if tap_only:
                if now and not was:
                    tap["clean"], tap["t0"] = True, time.time()
                elif not (names & keys_needed):
                    tap["clean"] = False      # пошла комбинация — это не тап
                return
            if now and not was and armed["v"]:
                armed["v"] = False
                toggle()

        def on_release(key):
            names = _canonical_keys(key)
            held = keys_needed.issubset(currently_pressed)
            currently_pressed.difference_update(names)
            if tap_only:
                if held and (names & keys_needed):
                    if tap["clean"] and time.time() - tap["t0"] <= TAP_MAX_SEC:
                        toggle()
                    tap["clean"] = False
                return
            if not keys_needed.issubset(currently_pressed):
                armed["v"] = True

        def _mouse_dirties_tap(*_args):
            # Ctrl+клик (контекстное меню) — тоже комбинация, не тап
            tap["clean"] = False

        if tap_only:
            try:
                from pynput import mouse
                mouse.Listener(on_click=_mouse_dirties_tap,
                               on_scroll=_mouse_dirties_tap).start()
            except Exception as e:
                logging.warning(f"mouse listener failed: {e}")

    # Двойной тап правого ⌘ — выделенное слово в словарь. Правый ⌥ не годится:
    # его двойной тап забрал Claude Desktop под Quick Entry (поймано 07.08.2026).
    # Если правый ⌘ занят под саму диктовку — не мешаем.
    if "cmd_r" not in keys_needed:
        _dbl = {"t": 0.0}
        # базовый обработчик — через переменную, а НЕ дефолтным аргументом:
        # pynput смотрит на сигнатуру и во второй параметр кладёт свой injected
        _base_press = on_press

        def on_press(key):                      # noqa: F811
            if "cmd_r" in _canonical_keys(key):
                now = time.time()
                if now - _dbl["t"] <= 0.8:
                    _dbl["t"] = 0.0
                    threading.Thread(target=learn_selected_word, daemon=True).start()
                else:
                    _dbl["t"] = now
            else:
                # ⌘+буква (Cmd+C, Cmd+V) — это комбинация, а не тап по словарю
                _dbl["t"] = 0.0
            return _base_press(key)

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        try:
            listener.join()
        except KeyboardInterrupt:
            pass

    print("\n👋 Bye")
    return 0


def _parse_hotkey(s: str) -> set:
    """'<ctrl>+<shift>+<space>' → set of canonical key names"""
    parts = [p.strip().lower() for p in s.replace(" ", "").split("+")]
    keys = set()
    for p in parts:
        if p.startswith("<") and p.endswith(">"):
            keys.add(p[1:-1])
        else:
            keys.add(p)
    return keys


def _canonical_key(key) -> str:
    """Канонизирует key из pynput в строку, совпадающую с _parse_hotkey."""
    from pynput.keyboard import Key, KeyCode
    if isinstance(key, Key):
        # Key.ctrl_l, Key.shift_r → "ctrl", "shift"
        name = key.name
        # Убрать суффиксы _l/_r
        for suffix in ("_l", "_r"):
            if name.endswith(suffix):
                name = name[:-2]
        return name
    if isinstance(key, KeyCode):
        if key.char:
            return key.char.lower()
        return str(key)
    return str(key).lower()


def _canonical_keys(key) -> set:
    """Как _canonical_key, но возвращает МНОЖЕСТВО имён-кандидатов.
    Для side-specific модификаторов даёт и точное, и общее имя:
    Key.alt_r → {"alt_r", "alt"}. Тогда хоткей '<alt_r>' (только правый)
    и '<alt>' (любой) оба матчатся корректно."""
    from pynput.keyboard import Key, KeyCode
    if isinstance(key, Key):
        name = key.name          # напр. 'alt_r'
        names = {name}
        for suffix in ("_l", "_r"):
            if name.endswith(suffix):
                names.add(name[:-2])   # общее 'alt'
        return names
    if isinstance(key, KeyCode):
        return {key.char.lower()} if key.char else {str(key)}
    return {str(key).lower()}


# ─── Entry point ────────────────────────────────────────────────────────────


_LOG_ROTATE_BYTES = 5 * 1024 * 1024


def _attach_log_file(log_path: str, verbose: bool) -> None:
    """Перенаправить stdout/stderr/logging в файл.

    Нужен и для отладки (видно что транскрибируется), и чтобы под pythonw.exe
    (autostart) print() не падал молча — там sys.stdout/sys.stderr = None.
    """
    expanded = os.path.expandvars(os.path.expanduser(log_path))
    Path(expanded).parent.mkdir(parents=True, exist_ok=True)
    try:
        if os.path.getsize(expanded) > _LOG_ROTATE_BYTES:
            backup = expanded + ".old"
            try: os.replace(expanded, backup)
            except OSError: pass
    except OSError:
        pass
    fh = open(expanded, "a", buffering=1, encoding="utf-8", errors="replace")
    sys.stdout = fh
    sys.stderr = fh
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(fh)],
        force=True,
    )
    from datetime import datetime as _dt
    fh.write(f"\n--- voice_dictation started {_dt.now().isoformat(timespec='seconds')} ---\n")


def main():
    p = argparse.ArgumentParser(description="Push-to-talk голосовая диктовка через Whisper")
    p.add_argument("--config", default=None, help="Путь к JSON-конфигу")
    p.add_argument("--setup", action="store_true", help="Создать дефолтный конфиг и выйти")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    if args.setup:
        logging.basicConfig(
            level=logging.INFO if args.verbose else logging.WARNING,
            format="%(asctime)s %(levelname)s %(message)s",
        )
        setup_wizard()
        return 0

    cfg_path = Path(args.config) if args.config else default_config_path()
    cfg = load_config(cfg_path)

    if cfg.get("log_file"):
        _attach_log_file(cfg["log_file"], args.verbose)
    else:
        logging.basicConfig(
            level=logging.INFO if args.verbose else logging.WARNING,
            format="%(asctime)s %(levelname)s %(message)s",
        )

    # Single-instance: вторая копия (autostart + ярлык, или ручной запуск
    # поверх работающей) выходит тихо. Retry — на случай self-restart при
    # переключении модели через tray-меню.
    if not acquire_single_instance_lock(timeout_seconds=2.0):
        logging.info("Another voice_dictation instance is already running — exiting silently.")
        return 0

    # Fast mode for dictation: greedy decoding, no temperature fallback
    os.environ.setdefault("WHISPER_BEAM_SIZE", "1")
    os.environ.setdefault("WHISPER_BEST_OF", "1")
    os.environ.setdefault("WHISPER_CONDITION_ON_PREV", "0")

    # Apply backend selection from config (must happen before transcribe is imported)
    if cfg.get("backend"):
        os.environ["WHISPER_BACKEND"] = cfg["backend"]
    if cfg.get("ov_device"):
        os.environ["WHISPER_OV_DEVICE"] = cfg["ov_device"]

    # Проверка зависимостей
    missing = []
    for mod in ["sounddevice", "soundfile", "pynput", "pyperclip", "numpy"]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    # tkinter нужен только если включён cursor_indicator на не-Mac
    if cfg.get("show_cursor_indicator", True) and platform.system() != "Darwin":
        try:
            __import__("tkinter")
        except ImportError:
            print("⚠ tkinter не найден — cursor_indicator не будет работать.", file=sys.stderr)
            print("  Mac:    brew install python-tk@3.12", file=sys.stderr)
            print("  Linux:  sudo apt install python3-tk", file=sys.stderr)
            print("  Windows: переустанови Python и отметь 'tcl/tk and IDLE'", file=sys.stderr)
            print("  Или просто отключи в конфиге: show_cursor_indicator: false\n", file=sys.stderr)
    if missing:
        print(f"❌ Не установлены пакеты: {missing}", file=sys.stderr)
        print(f"\nПоставь:")
        print(f"  pip install {' '.join(missing)} pystray Pillow")
        return 1

    # macOS Accessibility check — без него глобальный hotkey не сработает,
    # но pynput даёт только WARNING в stderr и не падает. Пользователь
    # думает что всё сломано. Явно проверяем + открываем системные настройки.
    if platform.system() == "Darwin":
        if not _check_macos_accessibility():
            return 1

    configure_sounds(cfg)
    return main_loop(cfg, cfg_path)


def _check_macos_accessibility() -> bool:
    """Проверить что процессу выдан Accessibility-permission на macOS.

    Использует CoreFoundation/ApplicationServices через ctypes. Если
    permission не выдан — печатает чёткую инструкцию и автоматически
    открывает соответствующий раздел System Settings. Возвращает False
    если permission не выдан (caller должен exit'нуть с этим кодом).
    """
    try:
        import ctypes
        from ctypes import c_void_p, c_bool

        # AXIsProcessTrustedWithOptions из ApplicationServices framework
        ApplicationServices = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"
        )
        ApplicationServices.AXIsProcessTrusted.restype = c_bool
        trusted = ApplicationServices.AXIsProcessTrusted()
        if not trusted:
            # Просим систему показать свой запрос: она сама заведёт запись с
            # правильной подписью бандла. Ручное добавление в список после
            # правки бандла (смена иконки) не срабатывает — запись остаётся
            # привязанной к прежней версии.
            try:
                import HIServices
                HIServices.AXIsProcessTrustedWithOptions(
                    {"AXTrustedCheckOptionPrompt": True})
            except Exception as e:
                logging.debug(f"AX prompt failed: {e}")
    except Exception as e:
        # Если не удалось проверить — не блокируем запуск (пусть pynput сам разберётся)
        logging.debug(f"AX trust check failed: {e}")
        return True

    if trusted:
        return True

    print("\n" + "─" * 60, file=sys.stderr)
    print("❌ macOS Accessibility permission не выдан", file=sys.stderr)
    print("─" * 60, file=sys.stderr)
    print(
        f"\nЭтому Python-бинарю нужен Accessibility доступ для глобального hotkey:\n"
        f"  {sys.executable}\n",
        file=sys.stderr,
    )
    print("Что делать:", file=sys.stderr)
    print("  1. Сейчас откроется System Settings → Privacy → Accessibility", file=sys.stderr)
    print("  2. Нажми + → Cmd+Shift+G → вставь путь выше → выбери python3", file=sys.stderr)
    print("  3. Включи галочку напротив добавленного python3", file=sys.stderr)
    print("  4. Запусти voice_dictation заново\n", file=sys.stderr)

    try:
        subprocess.Popen([
            "open",
            "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
        ])
    except Exception:
        pass

    return False


if __name__ == "__main__":
    sys.exit(main())
