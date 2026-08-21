"""macOS HUD диктовки — плавающая пилюля внизу по центру экрана.

Работает ОТДЕЛЬНЫМ процессом: Cocoa требует главный поток, а в
voice_dictation он занят pynput-листенером. Родитель шлёт команды в stdin.

Протокол (построчно):
    show          → показать, режим записи (эквалайзер по уровню звука)
    transcribing  → режим расшифровки (вращающаяся дуга)
    hide          → скрыть
    L<0..1>       → текущий уровень звука
    quit          → выйти

Обратно в stdout (клики по меню в статус-баре):
    enabled 0|1        · set language auto|ru|en · set model <имя>
    set play_sound 0|1 · paste_last · quit_app

Проверка руками:  python3 scripts/hud_mac.py --demo
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import threading
import time

PANEL_W, PANEL_H = 156.0, 40.0
BOTTOM_MARGIN = 96.0
RISE = 16.0                       # на сколько панель выезжает снизу при показе
N_BARS = 9
BAR_W, BAR_GAP = 3.5, 4.0
BAR_MIN, BAR_MAX = 3.0, 25.0
CANCEL_ZONE = 30.0                # правый край пилюли — крестик «отменить»
ANIM_SEC = 0.22                   # выезд/уход пилюли, секунды
ACCENT_A = (0.10, 1.00, 0.55)     # сочный зелёный — левый край градиента
ACCENT_B = (0.25, 0.95, 0.85)     # бирюза — правый край
ACCENT_WORK = (0.45, 0.80, 1.00)  # голубой — расшифровываем

CFG_DIR = os.path.expanduser("~/.config/whisper-skill")
CFG_FILE = os.path.join(CFG_DIR, "voice_dictation.json")
HISTORY_FILE = os.path.join(CFG_DIR, "history.md")
VOCAB_FILE = os.path.join(CFG_DIR, "vocabulary.txt")

LANGS = [("Авто", "auto"), ("Русский", "ru"), ("English", "en")]
# Только ПРАВЫЕ модификаторы и F-клавиши. Формат — как в _parse_hotkey родителя.
# ⚠️ Два класса вариантов выброшены намеренно, оба ломают работу:
#   • левый/любой модификатор («<alt>», «<alt_l>») — левый ⌥ живёт в ⌥+клик,
#     ⌥+стрелки, ⌥+буква, поэтому запись стартовала бы от обычного набора
#     (проверено 04.08.2026: текст влетал в поле сам); плюс pynput на macOS
#     отдаёт левый модификатор как общий, так что «<alt_l>» ещё и не ловится;
#   • ⌃+Пробел — системная смена источника ввода в macOS.
# «(тап)» = срабатывает по одиночному нажатию-отпусканию: Ctrl+C, ⌃+стрелки и
# ⌃+клик остаются комбинациями и запись не запускают. Левый и правый Control
# pynput на macOS не различает, поэтому вариант один и общий.
HOTKEYS = [("Правый ⌥ Option", "<alt_r>"),
           ("Правый ⌘ Command", "<cmd_r>"),
           ("Правый ⌃ Control", "<ctrl_r>"),
           ("⌃ Control — любой (тап)", "<ctrl>"),
           ("F13", "<f13>")]
MODELS = [("large-v3 (точная)", "mlx-community/whisper-large-v3-mlx"),
          ("large-v3-turbo (быстрая)", "mlx-community/whisper-large-v3-turbo")]
# Средняя скорость: слепой набор ~40 слов/мин, речь ~130. Разница = экономия.
TYPING_WPM, SPEAK_WPM = 40.0, 130.0


def _read_cfg() -> dict:
    try:
        with open(CFG_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


DATASET_DIR = os.path.join(CFG_DIR, "dataset")
PAIRS_FILE = os.path.join(DATASET_DIR, "pairs.jsonl")


def save_correction() -> str:
    """Показать распознанное, дать поправить и сложить пару «аудио + правда».

    Единственный источник эталона: сам Whisper не знает, где ошибся. Диалог —
    через osascript, чтобы не городить Cocoa-окно ради одного текстового поля.
    """
    try:
        with open(os.path.join(CFG_DIR, "last.json"), encoding="utf-8") as f:
            last = json.load(f)
    except Exception:
        return "нет последней записи"
    if not os.path.exists(last.get("wav", "")):
        return "аудио последней записи не сохранилось"

    # ensure_ascii=False обязателен: \uXXXX AppleScript не понимает и падает на
    # разборе. Переводы строк тоже недопустимы внутри его строкового литерала.
    shown = json.dumps(last["asr"].replace("\n", " "), ensure_ascii=False)
    script = ('display dialog "Что было сказано на самом деле?" '
              f'default answer {shown} with title "QSpeak" '
              'buttons {"Отмена", "Сохранить"} default button "Сохранить"')
    r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    if r.returncode != 0:                       # Отмена
        return "отменено"
    truth = r.stdout.split("text returned:", 1)[-1].strip()
    if not truth or truth == last["asr"]:
        return "правок нет"

    os.makedirs(DATASET_DIR, exist_ok=True)
    name = time.strftime("%Y%m%d-%H%M%S") + ".wav"
    subprocess.run(["cp", last["wav"], os.path.join(DATASET_DIR, name)], check=True)
    with open(PAIRS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"wav": name, "truth": truth, "asr": last["asr"]},
                           ensure_ascii=False) + "\n")
    return "сохранено"


def stats_today(path: str = HISTORY_FILE, today: str = "") -> str:
    """Слова за сегодня из history.md + сэкономленное против набора руками."""
    today = today or time.strftime("%Y-%m-%d")
    words, counting = 0, False
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("## "):
                    counting = line[3:].startswith(today)
                elif counting:
                    words += len(line.split())
    except OSError:
        return "Сегодня: истории нет"
    saved = words * (1 / TYPING_WPM - 1 / SPEAK_WPM)
    return f"Сегодня: {words} слов · ~{saved:.0f} мин сэкономлено"


# ─── дочерний процесс: само окно ────────────────────────────────────────────

def _run_child() -> int:
    from AppKit import (
        NSApplication, NSApplicationActivationPolicyAccessory, NSBackingStoreBuffered,
        NSBezierPath, NSColor, NSMenu, NSMenuItem, NSPanel, NSScreen, NSStatusBar,
        NSEvent, NSTimer, NSVariableStatusItemLength, NSView,
        NSWindowCollectionBehaviorCanJoinAllSpaces,
        NSWindowCollectionBehaviorFullScreenAuxiliary,
        NSWindowCollectionBehaviorStationary,
        NSWindowStyleMaskBorderless, NSWindowStyleMaskNonactivatingPanel,
    )
    from Foundation import (
        NSMakeRect, NSMakePoint, NSObject, NSRunLoop, NSRunLoopCommonModes)
    import objc

    state = {"mode": "hidden", "level": 0.0, "smooth": 0.0, "p": 0.0, "t": 0.0,
             "enabled": True, "title": "", "bars": [0.0] * N_BARS,
             "cancel_click": False, "peak": 0.05, "last_tick": time.time()}

    class HUDView(NSView):
        def drawRect_(self, rect):
            w, h = self.bounds().size.width, self.bounds().size.height
            t, mode = state["t"], state["mode"]

            lvl_now = max(0.0, min(1.0, state["smooth"]))
            ar, ag, ab = (ACCENT_A if mode == "recording" else ACCENT_WORK)

            # подложка-пилюля
            body = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                NSMakeRect(1, 1, w - 2, h - 2), (h - 2) / 2, (h - 2) / 2)
            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.07, 0.07, 0.09, 0.92).setFill()
            body.fill()
            # кант в цвет акцента, разгорается от громкости — плашка «дышит» целиком
            glow = 0.18 + 0.42 * (lvl_now if mode == "recording" else
                                  0.5 + 0.5 * math.sin(t * 3.0))
            NSColor.colorWithCalibratedRed_green_blue_alpha_(ar, ag, ab, glow).setStroke()
            body.setLineWidth_(1.6)
            body.stroke()

            # эквалайзер живёт слева от крестика, поэтому центрируем не по всей
            # ширине, а по свободной части
            field_w = w - CANCEL_ZONE

            if mode == "recording":
                # высоты столбиков считает Ticker (быстрый рост, ленивый спад) —
                # здесь только рисуем: пики держатся, спад плавный, как у железных
                # эквалайзеров
                lvl = max(0.0, min(1.0, state["smooth"]))
                total = N_BARS * BAR_W + (N_BARS - 1) * BAR_GAP
                x = (field_w - total) / 2
                for i in range(N_BARS):
                    k = i / (N_BARS - 1)
                    r = ACCENT_A[0] + (ACCENT_B[0] - ACCENT_A[0]) * k
                    g = ACCENT_A[1] + (ACCENT_B[1] - ACCENT_A[1]) * k
                    b = ACCENT_A[2] + (ACCENT_B[2] - ACCENT_A[2]) * k
                    bh = BAR_MIN + (BAR_MAX - BAR_MIN) * state["bars"][i]
                    bar = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                        NSMakeRect(x, (h - bh) / 2, BAR_W, bh), BAR_W / 2, BAR_W / 2)
                    NSColor.colorWithCalibratedRed_green_blue_alpha_(
                        r, g, b, 0.70 + 0.30 * state["bars"][i]).setFill()
                    bar.fill()
                    x += BAR_W + BAR_GAP
            else:
                # расшифровка: те же столбики, но по ним переливается блик —
                # видно, что работа идёт, а форма плашки не скачет
                total = N_BARS * BAR_W + (N_BARS - 1) * BAR_GAP
                x = (field_w - total) / 2
                for i in range(N_BARS):
                    k = i / (N_BARS - 1)
                    # гребень бежит по кругу; чем ближе столбик к нему, тем ярче и выше
                    d = abs(((t * 0.9 - k) % 1.0) - 0.0)
                    d = min(d, 1.0 - d)             # расстояние по кольцу
                    pulse = max(0.0, 1.0 - d * 3.2)
                    bh = BAR_MIN + 10.0 * pulse
                    bar = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                        NSMakeRect(x, (h - bh) / 2, BAR_W, bh), BAR_W / 2, BAR_W / 2)
                    NSColor.colorWithCalibratedRed_green_blue_alpha_(
                        ar, ag, ab, 0.30 + 0.70 * pulse).setFill()
                    bar.fill()
                    x += BAR_W + BAR_GAP

            # крестик «отменить»: и во время записи, и во время расшифровки —
            # чтобы не ждать результат, который уже не нужен
            cx, cy, rr = w - CANCEL_ZONE / 2 - 4, h / 2, 5.0
            ring = NSBezierPath.bezierPathWithOvalInRect_(
                NSMakeRect(cx - rr - 4, cy - rr - 4, (rr + 4) * 2, (rr + 4) * 2))
            NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.09).setFill()
            ring.fill()
            cross = NSBezierPath.bezierPath()
            cross.moveToPoint_(NSMakePoint(cx - rr / 1.6, cy - rr / 1.6))
            cross.lineToPoint_(NSMakePoint(cx + rr / 1.6, cy + rr / 1.6))
            cross.moveToPoint_(NSMakePoint(cx - rr / 1.6, cy + rr / 1.6))
            cross.lineToPoint_(NSMakePoint(cx + rr / 1.6, cy - rr / 1.6))
            NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.55, 0.5, 0.8).setStroke()
            cross.setLineWidth_(1.8)
            cross.setLineCapStyle_(1)
            cross.stroke()

        def mouseDown_(self, event):
            p = self.convertPoint_fromView_(event.locationInWindow(), None)
            if p.x >= self.bounds().size.width - CANCEL_ZONE:
                state["cancel_click"] = True

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

    def panel_origin():
        """Левый нижний угол пилюли по ЭКРАНУ, где сейчас курсор.

        Считался один раз на старте и по main-дисплею: стоило отключить
        внешний монитор (или сменить масштаб), как посчитанный по широкой
        ширине x оставлял правый край пилюли за границей узкого экрана —
        вместе с крестиком «отменить», и отменить запись было нечем."""
        pt = NSEvent.mouseLocation()
        scr = NSScreen.mainScreen()
        for s in NSScreen.screens():
            f = s.frame()
            if (f.origin.x <= pt.x < f.origin.x + f.size.width
                    and f.origin.y <= pt.y < f.origin.y + f.size.height):
                scr = s
                break
        f = scr.frame()
        return (f.origin.x + (f.size.width - PANEL_W) / 2, f.origin.y + BOTTOM_MARGIN)

    ox, oy = panel_origin()
    frame = NSMakeRect(ox, oy, PANEL_W, PANEL_H)
    panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
        frame, NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
        NSBackingStoreBuffered, False)
    panel.setOpaque_(False)
    panel.setBackgroundColor_(NSColor.clearColor())
    panel.setLevel_(25)  # NSStatusWindowLevel — поверх обычных окон
    panel.setIgnoresMouseEvents_(True)
    panel.setHidesOnDeactivate_(False)
    panel.setHasShadow_(True)
    panel.setCollectionBehavior_(
        NSWindowCollectionBehaviorCanJoinAllSpaces
        | NSWindowCollectionBehaviorStationary
        | NSWindowCollectionBehaviorFullScreenAuxiliary)
    panel.setAlphaValue_(0.0)
    view = HUDView.alloc().initWithFrame_(panel.contentView().bounds())
    panel.setContentView_(view)

    # ─── меню в статус-баре ─────────────────────────────────────────────────
    # Живёт здесь же: этот процесс уже Cocoa и уже на главном потоке, а в
    # родителе главный поток занят pynput-листенером. Клики уходят в stdout.

    def _emit(cmd: str) -> None:
        try:
            sys.stdout.write(cmd + "\n")
            sys.stdout.flush()
        except Exception:
            pass

    class MenuTarget(NSObject):
        def toggleEnabled_(self, sender):
            state["enabled"] = not state["enabled"]
            _emit(f"enabled {1 if state['enabled'] else 0}")

        def pickLang_(self, sender):
            _emit(f"set language {sender.representedObject()}")

        def pickModel_(self, sender):
            _emit(f"set model {sender.representedObject()}")

        def pickHotkey_(self, sender):
            _emit(f"set hotkey {sender.representedObject()}")

        def toggleSounds_(self, sender):
            _emit(f"set play_sound {0 if _read_cfg().get('play_sound', True) else 1}")

        def openVocab_(self, sender):
            subprocess.Popen(["open", "-t", VOCAB_FILE])

        def openHistory_(self, sender):
            subprocess.Popen(["open", "-t", HISTORY_FILE])

        def pasteLast_(self, sender):
            _emit("paste_last")

        def correctLast_(self, sender):
            # stderr ребёнка уходит в DEVNULL, поэтому результат — в общий лог
            # через родителя.
            res = save_correction()
            _emit(f"log правка эталона: {res}")
            if res == "сохранено":
                # правку в словарь превращает родитель — разбор правил живёт там
                _emit("vocab_learn")

        def quitApp_(self, sender):
            _emit("quit_app")

        def menuWillOpen_(self, menu):
            cfg = _read_cfg()
            mi["stats"].setTitle_(stats_today())
            try:
                with open(PAIRS_FILE, encoding="utf-8") as f:
                    n = sum(1 for _ in f)
            except OSError:
                n = 0
            mi["correct"].setTitle_(f"Поправить последнее… (эталонов: {n})")
            mi["onoff"].setTitle_("Диктовка включена" if state["enabled"]
                                  else "Диктовка выключена")
            mi["onoff"].setState_(1 if state["enabled"] else 0)
            mi["sounds"].setState_(1 if cfg.get("play_sound", True) else 0)
            for it in mi["langs"]:
                it.setState_(1 if (cfg.get("language") or "auto") == it.representedObject() else 0)
            for it in mi["models"]:
                it.setState_(1 if cfg.get("model") == it.representedObject() else 0)
            for it in mi["hotkeys"]:
                it.setState_(1 if cfg.get("hotkey") == it.representedObject() else 0)

    target = MenuTarget.alloc().init()

    def _item(title, action, obj=None):
        it = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, action, "")
        it.setTarget_(target)
        if obj is not None:
            it.setRepresentedObject_(obj)
        return it

    def _submenu(parent_menu, title, pairs, action):
        head = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, None, "")
        sub = NSMenu.alloc().init()
        items = []
        for label, value in pairs:
            it = _item(label, action, value)
            sub.addItem_(it)
            items.append(it)
        head.setSubmenu_(sub)
        parent_menu.addItem_(head)
        return items

    menu = NSMenu.alloc().init()
    menu.setDelegate_(target)
    mi = {"stats": _item(stats_today(), None),
          "onoff": _item("Диктовка включена", "toggleEnabled:"),
          "sounds": _item("Звуки", "toggleSounds:"),
          "correct": _item("Поправить последнее…", "correctLast:")}
    menu.addItem_(mi["stats"])
    menu.addItem_(NSMenuItem.separatorItem())
    menu.addItem_(mi["onoff"])
    mi["langs"] = _submenu(menu, "Язык", LANGS, "pickLang:")
    mi["models"] = _submenu(menu, "Модель", MODELS, "pickModel:")
    mi["hotkeys"] = _submenu(menu, "Кнопка диктовки", HOTKEYS, "pickHotkey:")
    menu.addItem_(mi["sounds"])
    menu.addItem_(NSMenuItem.separatorItem())
    menu.addItem_(_item("Вставить последнее", "pasteLast:"))
    menu.addItem_(mi["correct"])
    menu.addItem_(_item("Открыть словарь", "openVocab:"))
    menu.addItem_(_item("Открыть историю", "openHistory:"))
    menu.addItem_(NSMenuItem.separatorItem())
    menu.addItem_(_item("Выход", "quitApp:"))

    status_item = NSStatusBar.systemStatusBar().statusItemWithLength_(
        NSVariableStatusItemLength)
    status_item.setMenu_(menu)

    def _stdin_reader():
        for line in sys.stdin:
            cmd = line.strip()
            if not cmd:
                continue
            if cmd.startswith("L"):
                try:
                    state["level"] = float(cmd[1:])
                except ValueError:
                    pass
            elif cmd == "show":
                state["mode"] = "recording"
            elif cmd == "transcribing":
                state["mode"] = "transcribing"
            elif cmd == "hide":
                state["mode"] = "hidden"
            elif cmd == "quit":
                state["mode"] = "quit"
                return
        state["mode"] = "quit"

    threading.Thread(target=_stdin_reader, daemon=True).start()

    class Ticker(NSObject):
        def tick_(self, timer):
            if state["mode"] == "quit":
                app.terminate_(None)
                return
            # шаг по реальному времени: таймер иногда пропускает кадры, и на
            # фиксированном 1/30 анимация дёргалась ровно в эти моменты
            now = time.time()
            dt = min(0.1, now - state["last_tick"])
            state["last_tick"] = now
            state["t"] += dt
            # значок в баре меняется только на главном потоке — отсюда, не из stdin
            want_title = {"recording": "🔴", "transcribing": "⏳"}.get(
                state["mode"], "🎙" if state["enabled"] else "⏸")
            if want_title != state["title"]:
                state["title"] = want_title
                status_item.button().setTitle_(want_title)
            if state["cancel_click"]:
                state["cancel_click"] = False
                _emit("cancel")
            # сглаживание громкости: сырой RMS дёргается и полоски мельтешат
            state["smooth"] += (state["level"] - state["smooth"]) * 0.45
            lvl = max(0.0, min(1.0, state["smooth"]))
            # авто-усиление: тихая речь давала размах в пару пикселей. Нормируем
            # на недавний максимум (он сам медленно оседает) — полный размах на
            # любом голосе, а не только на крике в микрофон.
            state["peak"] = max(lvl, state["peak"] * 0.94)
            norm = (min(1.0, lvl / max(state["peak"], 0.04))) ** 0.65
            drive = 0.18 + 0.82 * norm      # 0.18 — столбики дышат и в тишине
            # каждый столбик живёт сам: вверх прыгает почти мгновенно, вниз
            # оседает лениво — от этого движение «дышит», а не мерцает
            for i in range(N_BARS):
                k = i / (N_BARS - 1)
                wave = 0.45 + 0.55 * (0.5 + 0.5 * math.sin(state["t"] * 9.0 - k * 4.6))
                shape = 0.55 + 0.45 * math.sin(math.pi * k)   # холм: центр выше краёв
                target = min(1.0, drive * wave * shape * 1.35)
                cur = state["bars"][i]
                state["bars"][i] = cur + (target - cur) * (0.7 if target > cur else 0.14)

            # Появление/исчезновение по времени, не «процент за кадр»: экспонента
            # давала первый скачок сразу на четверть пути — он и читался как рывок.
            want = 0.0 if state["mode"] == "hidden" else 1.0
            step = dt / ANIM_SEC
            state["p"] = max(0.0, min(1.0, state["p"] + (step if want else -step)))
            p = state["p"]
            e = p * p * (3.0 - 2.0 * p)      # smoothstep: мягкий старт и мягкий конец
            panel.setAlphaValue_(e)
            # мышь ловим только пока пилюля на экране — иначе она бы съедала
            # клики по пустому месту у дока
            panel.setIgnoresMouseEvents_(state["mode"] == "hidden")
            # окно не прячем через orderOut: первый кадр после показа стоил
            # заметного подтормаживания. Невидимая панель с alpha 0 не мешает.
            if state["mode"] != "hidden" or p > 0.0:
                # позицию берём заново на каждом показе — экран мог смениться,
                # пока пилюля была скрыта
                ox, oy = panel_origin()
                panel.setFrameOrigin_(NSMakePoint(ox, oy - RISE * (1.0 - e)))
            if p > 0.0:
                view.setNeedsDisplay_(True)

    ticker = Ticker.alloc().init()
    timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
        1 / 60, ticker, objc.selector(ticker.tick_, signature=b"v@:@"), None, True)
    # Без common modes таймер замирает, пока открыто меню или идёт скролл —
    # это и был «затык» на выезде пилюли. И один раз показываем окно, дальше
    # видимостью управляет alpha.
    NSRunLoop.currentRunLoop().addTimer_forMode_(timer, NSRunLoopCommonModes)
    panel.orderFrontRegardless()
    app.run()
    return 0


# ─── родитель: обёртка с API как у CursorIndicator ──────────────────────────

def real_python() -> str:
    """Под py2app sys.executable — сам бандл QSpeak, дочерний процесс так не поднять.

    Порядок: QSPEAK_PYTHON из окружения (его пишет install.sh в LaunchAgent) →
    интерпретатор, которым собран бандл (sys.base_prefix) → просто python3.
    """
    exe = sys.executable or ""
    if os.path.basename(exe).lower().startswith("python"):
        return exe
    for cand in (os.environ.get("QSPEAK_PYTHON"),
                 os.path.join(sys.base_prefix, "bin", "python3")):
        if cand and os.path.exists(cand):
            return cand
    return "python3"


_real_python = real_python   # старое имя — на случай внешних вызовов


class MacHUD:
    """Тот же интерфейс, что у CursorIndicator, плюс set_level().

    on_command(str) — клики по меню в статус-баре (см. протокол вверху файла).
    """

    def __init__(self, on_command=None) -> None:
        self._proc: subprocess.Popen | None = None
        self._on_command = on_command

    def start(self) -> None:
        if self._proc:
            return
        try:
            self._proc = subprocess.Popen(
                [_real_python(), os.path.abspath(__file__), "--child"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, bufsize=1,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            )
            threading.Thread(target=self._read_commands, daemon=True).start()
        except Exception:
            self._proc = None

    def _read_commands(self) -> None:
        for line in self._proc.stdout:
            cmd = line.strip()
            if not cmd or not self._on_command:
                continue
            try:
                self._on_command(cmd)
            except Exception as e:
                print(f"[hud] команда {cmd!r} упала: {e}", flush=True)

    def _send(self, cmd: str) -> None:
        p = self._proc
        if not p or not p.stdin or p.poll() is not None:
            return
        try:
            p.stdin.write(cmd + "\n")
            p.stdin.flush()
        except Exception:
            self._proc = None  # HUD умер — диктовка продолжает работать без него

    def show(self) -> None:
        self._send("show")

    def show_transcribing(self) -> None:
        self._send("transcribing")

    def hide(self) -> None:
        self._send("hide")

    def set_level(self, level: float) -> None:
        self._send(f"L{level:.3f}")

    def stop(self) -> None:
        self._send("quit")
        self._proc = None


def _demo() -> None:
    """Ручная проверка: пилюля пишет 4 сек по синусоиде, потом 2 сек расшифровка."""
    hud = MacHUD()
    hud.start()
    time.sleep(1.0)
    hud.show()
    t0 = time.time()
    while time.time() - t0 < 4.0:
        hud.set_level(abs(math.sin((time.time() - t0) * 2.0)))
        time.sleep(1 / 30)
    hud.show_transcribing()
    time.sleep(2.0)
    hud.hide()
    time.sleep(1.0)
    hud.stop()


def _selftest() -> None:
    """Проверка разбора истории: считаем только сегодняшние блоки."""
    import tempfile
    hist = ("\n## 2026-08-02 10:00:00\nвчера три слова\n"
            "\n## 2026-08-03 09:00:00\nсегодня ровно четыре слова\n"
            "\n## 2026-08-03 11:00:00\nещё два\n")
    with tempfile.NamedTemporaryFile("w", suffix=".md", encoding="utf-8", delete=False) as f:
        f.write(hist)
    out = stats_today(f.name, today="2026-08-03")
    assert "6 слов" in out, out                       # 4 + 2, вчерашние не в счёт
    assert stats_today("/нет/такого", today="x").startswith("Сегодня: истории нет")
    os.unlink(f.name)
    print("ok:", out)

    # Каждый пункт «Кнопка диктовки» обязан реально ловиться слушателем: выбрать
    # в меню хоткей, который pynput не отдаёт (напр. левый ⌥), = молча остаться
    # без диктовки.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from examples.voice_dictation import _parse_hotkey, _canonical_keys
    from pynput.keyboard import Key
    for label, value in HOTKEYS:
        need = _parse_hotkey(value)
        for part in need:
            key = getattr(Key, part, None)
            if key is not None:      # модификаторы и F-клавиши, не буквы
                assert part in _canonical_keys(key), f"{label}: {part} не ловится"
    print(f"ok: все {len(HOTKEYS)} вариантов хоткея распознаются")


if __name__ == "__main__":
    if "--child" in sys.argv:
        sys.exit(_run_child())
    if "--selftest" in sys.argv:
        _selftest()
    else:
        _demo()
