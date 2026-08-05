"""Выгрузка модели по простою: таймер взводится, сбрасывается и стреляет.

Запуск: python3 scripts/test_unload_idle.py   (из корня скилла)
"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import examples.voice_dictation as vd

vd._after_dictation({"unload_after_idle_min": 15})
t1 = vd._unload_timer
assert t1 and t1.is_alive(), "таймер не взведён"

vd._after_dictation({"unload_after_idle_min": 15})
assert vd._unload_timer is not t1 and not t1.is_alive(), "старый таймер не отменён"

vd._after_dictation({"unload_after_idle_min": 0})
assert vd._unload_timer is None, "при 0 модель должна оставаться в памяти"

vd._release_model()  # без загруженной модели тоже не падает

vd._after_dictation({"unload_after_idle_min": 1 / 120})  # 0.5 сек
time.sleep(1.2)
assert not vd._unload_timer.is_alive(), "таймер не сработал"

print("OK: выгрузка по простою работает")
