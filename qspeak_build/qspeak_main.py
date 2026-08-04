import sys, os, multiprocessing

# Alias-бандл: Contents/Resources/qspeak_main.py — симлинк на этот файл внутри
# скилла, поэтому корень скилла берётся от него, а не хардкодится.
SKILL = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# mlx_whisper зовёт ffmpeg через PATH, а PATH зависит от способа запуска:
# launchd берёт его из plist, а вот запуск из Finder/Spotlight/Dock даёт
# урезанный GUI-PATH без homebrew — и расшифровка падает на «ffmpeg not found».
# Дописываем сами, чтобы приложение не зависело от того, как его подняли.
for _bin in ("/opt/homebrew/bin", "/usr/local/bin"):
    if os.path.isdir(_bin) and _bin not in os.environ.get("PATH", "").split(":"):
        os.environ["PATH"] = _bin + ":" + os.environ.get("PATH", "")

# КРИТИЧНО для py2app: под бандлом sys.executable = сам QSpeak.app. Любой
# multiprocessing/hf-download через spawn иначе перезапускает приложение,
# ловит single-instance lock и роняет главный процесс. Направляем воркеров
# на настоящий python.
sys.path.insert(0, SKILL)
from scripts.hud_mac import real_python
try:
    multiprocessing.set_executable(real_python())
except Exception:
    pass
multiprocessing.freeze_support()

os.chdir(SKILL)

# Под py2app нет терминала — перенаправляем весь вывод в файл, чтобы видеть
# работу диктовки/потока при отладке.
try:
    _log = open("/tmp/qspeak.log", "a", buffering=1, encoding="utf-8", errors="replace")
    sys.stdout = _log
    sys.stderr = _log
    print("\n===== QSpeak start =====", flush=True)
except Exception:
    pass

# Быстрый режим декодирования (как в самом voice_dictation)
os.environ.setdefault("WHISPER_BEAM_SIZE", "1")
os.environ.setdefault("WHISPER_BEST_OF", "1")


def _preauth_microphone():
    """Явно запросить доступ к микрофону через AVFoundation до старта диктовки.
    Теперь процесс имеет identity бандла QSpeak (Info.plist с NSMicrophoneUsageDescription),
    поэтому системный диалог показывается корректно и решение атрибутируется QSpeak."""
    try:
        import AVFoundation
        from Foundation import NSRunLoop, NSDate
        AT = AVFoundation.AVMediaTypeAudio
        st = AVFoundation.AVCaptureDevice.authorizationStatusForMediaType_(AT)
        # 0 не спрашивали · 1 запрещено политикой · 2 отказано · 3 разрешено
        print(f"[mic preauth] статус доступа к микрофону: {st}", flush=True)
        if st == 3:  # уже разрешено
            return
        done = {"v": False}
        def cb(granted):
            done["v"] = True
        AVFoundation.AVCaptureDevice.requestAccessForMediaType_completionHandler_(AT, cb)
        n = 0
        while not done["v"] and n < 150:  # ждать ответа до 30 сек
            NSRunLoop.currentRunLoop().runMode_beforeDate_(
                "kCFRunLoopDefaultMode", NSDate.dateWithTimeIntervalSinceNow_(0.2))
            n += 1
    except Exception as e:
        print(f"[mic preauth] пропущено: {e}")


_preauth_microphone()

from examples.voice_dictation import main
sys.exit(main())
