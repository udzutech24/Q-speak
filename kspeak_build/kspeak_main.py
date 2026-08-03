import sys, os, multiprocessing

# КРИТИЧНО для py2app: под бандлом sys.executable = сам K-speak.app. Любой
# multiprocessing/hf-download через spawn иначе перезапускает приложение,
# ловит single-instance lock и роняет главный процесс. Направляем воркеров
# на настоящий python.
_REAL_PY = "/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"
try:
    multiprocessing.set_executable(_REAL_PY)
except Exception:
    pass
multiprocessing.freeze_support()

SKILL = "/Users/alekseya/.claude/skills/whisper-skill"
os.chdir(SKILL)
if SKILL not in sys.path:
    sys.path.insert(0, SKILL)

# Под py2app нет терминала — перенаправляем весь вывод в файл, чтобы видеть
# работу диктовки/потока при отладке.
try:
    _log = open("/tmp/kspeak.log", "a", buffering=1, encoding="utf-8", errors="replace")
    sys.stdout = _log
    sys.stderr = _log
    print("\n===== K-speak start =====", flush=True)
except Exception:
    pass

# Быстрый режим декодирования (как в самом voice_dictation)
os.environ.setdefault("WHISPER_BEAM_SIZE", "1")
os.environ.setdefault("WHISPER_BEST_OF", "1")


def _preauth_microphone():
    """Явно запросить доступ к микрофону через AVFoundation до старта диктовки.
    Теперь процесс имеет identity бандла K-speak (Info.plist с NSMicrophoneUsageDescription),
    поэтому системный диалог показывается корректно и решение атрибутируется K-speak."""
    try:
        import AVFoundation
        from Foundation import NSRunLoop, NSDate
        AT = AVFoundation.AVMediaTypeAudio
        st = AVFoundation.AVCaptureDevice.authorizationStatusForMediaType_(AT)
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
