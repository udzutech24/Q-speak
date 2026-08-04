#!/usr/bin/env python3
"""Проверка: stop() не виснет на мёртвом аудиоустройстве.

Инцидент 04.08.2026: отвалился USB-микрофон → CoreAudio не вернулся из
stream.close() → вместе с ним встал поток хоткея, диктовка умерла насмерть.
Запуск: python3 scripts/test_stop_timeout.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from examples.voice_dictation import AudioRecorder  # noqa: E402


class DeadStream:
    """Устройство, из закрытия которого управление не возвращается."""

    def stop(self):
        time.sleep(30)

    def close(self):
        time.sleep(30)


def main():
    import numpy as np

    rec = AudioRecorder()
    rec._stream = DeadStream()
    rec._recording = True
    rec._frames = [np.zeros((16000, 1), dtype="float32")]

    t0 = time.time()
    wav = rec.stop()
    elapsed = time.time() - t0

    assert elapsed < 4.0, f"stop() висел {elapsed:.1f} с — таймаут не сработал"
    assert wav and Path(wav).exists(), "запись потеряна: stop() не вернул wav"
    assert rec._stream is None, "ссылка на мёртвый поток осталась"
    Path(wav).unlink()
    print(f"✓ stop() вернулся за {elapsed:.1f} с, аудио сохранено")


if __name__ == "__main__":
    main()
