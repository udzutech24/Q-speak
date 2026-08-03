"""macOS HUD диктовки — плавающая пилюля внизу по центру экрана.

Работает ОТДЕЛЬНЫМ процессом: Cocoa требует главный поток, а в
voice_dictation он занят pynput-листенером. Родитель шлёт команды в stdin.

Протокол (построчно):
    show          → показать, режим записи (эквалайзер по уровню звука)
    transcribing  → режим расшифровки (вращающаяся дуга)
    hide          → скрыть
    L<0..1>       → текущий уровень звука
    quit          → выйти

Проверка руками:  python3 scripts/hud_mac.py --demo
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import threading
import time

PANEL_W, PANEL_H = 190.0, 52.0
BOTTOM_MARGIN = 96.0
RISE = 16.0                       # на сколько панель выезжает снизу при показе
N_BARS = 7
BAR_W, BAR_GAP = 5.0, 6.0
BAR_MIN, BAR_MAX = 5.0, 28.0
ACCENT_A = (0.10, 1.00, 0.55)     # сочный зелёный — левый край градиента
ACCENT_B = (0.25, 0.95, 0.85)     # бирюза — правый край
ACCENT_WORK = (0.45, 0.80, 1.00)  # голубой — расшифровываем


# ─── дочерний процесс: само окно ────────────────────────────────────────────

def _run_child() -> int:
    from AppKit import (
        NSApplication, NSApplicationActivationPolicyAccessory, NSBackingStoreBuffered,
        NSBezierPath, NSColor, NSPanel, NSScreen, NSTimer, NSView,
        NSWindowCollectionBehaviorCanJoinAllSpaces,
        NSWindowCollectionBehaviorFullScreenAuxiliary,
        NSWindowCollectionBehaviorStationary,
        NSWindowStyleMaskBorderless, NSWindowStyleMaskNonactivatingPanel,
    )
    from Foundation import NSMakeRect, NSMakePoint, NSObject
    import objc

    state = {"mode": "hidden", "level": 0.0, "smooth": 0.0, "shown": 0.0, "t": 0.0}

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

            if mode == "recording":
                # эквалайзер: сглаженная громкость × собственная фаза каждого
                # столбика — синхронные полоски выглядят мёртво
                lvl = max(0.0, min(1.0, state["smooth"]))
                total = N_BARS * BAR_W + (N_BARS - 1) * BAR_GAP
                x = (w - total) / 2
                for i in range(N_BARS):
                    k = i / (N_BARS - 1)
                    r = ACCENT_A[0] + (ACCENT_B[0] - ACCENT_A[0]) * k
                    g = ACCENT_A[1] + (ACCENT_B[1] - ACCENT_A[1]) * k
                    b = ACCENT_A[2] + (ACCENT_B[2] - ACCENT_A[2]) * k
                    # бегущая волна: фаза сдвинута по позиции, поэтому гребень
                    # идёт слева направо, а не все столбики прыгают разом
                    wave = 0.35 + 0.65 * (0.5 + 0.5 * math.sin(t * 7.0 - k * 4.2))
                    # центральные столбики выше крайних — форма «холма»
                    shape = 0.55 + 0.45 * math.sin(math.pi * k)
                    bh = BAR_MIN + (BAR_MAX - BAR_MIN) * lvl * wave * shape
                    bar = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                        NSMakeRect(x, (h - bh) / 2, BAR_W, bh), BAR_W / 2, BAR_W / 2)
                    NSColor.colorWithCalibratedRed_green_blue_alpha_(
                        r, g, b, 0.75 + 0.25 * lvl).setFill()
                    bar.fill()
                    x += BAR_W + BAR_GAP
            else:
                # расшифровка: те же столбики, но по ним переливается блик —
                # видно, что работа идёт, а форма плашки не скачет
                total = N_BARS * BAR_W + (N_BARS - 1) * BAR_GAP
                x = (w - total) / 2
                for i in range(N_BARS):
                    k = i / (N_BARS - 1)
                    # гребень бежит по кругу; чем ближе столбик к нему, тем ярче и выше
                    d = abs(((t * 0.9 - k) % 1.0) - 0.0)
                    d = min(d, 1.0 - d)             # расстояние по кольцу
                    pulse = max(0.0, 1.0 - d * 3.2)
                    bh = BAR_MIN + 12.0 * pulse
                    bar = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                        NSMakeRect(x, (h - bh) / 2, BAR_W, bh), BAR_W / 2, BAR_W / 2)
                    NSColor.colorWithCalibratedRed_green_blue_alpha_(
                        ar, ag, ab, 0.30 + 0.70 * pulse).setFill()
                    bar.fill()
                    x += BAR_W + BAR_GAP

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

    scr = NSScreen.mainScreen().frame()
    frame = NSMakeRect((scr.size.width - PANEL_W) / 2, BOTTOM_MARGIN, PANEL_W, PANEL_H)
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
            state["t"] += 1 / 30
            # сглаживание громкости: сырой RMS дёргается и полоски мельтешат
            state["smooth"] += (state["level"] - state["smooth"]) * 0.35
            want = 0.0 if state["mode"] == "hidden" else 1.0
            # плавное появление/угасание — резкий показ выглядит дёшево
            state["shown"] += (want - state["shown"]) * 0.25
            if abs(state["shown"] - want) < 0.01:
                state["shown"] = want
            panel.setAlphaValue_(state["shown"])
            if state["shown"] > 0.01:
                if not panel.isVisible():
                    panel.orderFrontRegardless()
                # выезд снизу вместе с проявлением
                panel.setFrameOrigin_(NSMakePoint(
                    frame.origin.x, BOTTOM_MARGIN - RISE * (1.0 - state["shown"])))
                view.setNeedsDisplay_(True)
            elif panel.isVisible():
                panel.orderOut_(None)

    ticker = Ticker.alloc().init()
    NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
        1 / 30, ticker, objc.selector(ticker.tick_, signature=b"v@:@"), None, True)
    app.run()
    return 0


# ─── родитель: обёртка с API как у CursorIndicator ──────────────────────────

def _real_python() -> str:
    """Под py2app sys.executable — сам бандл K-speak, дочерний процесс так не поднять."""
    exe = sys.executable or ""
    if os.path.basename(exe).lower().startswith("python"):
        return exe
    return "/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"


class MacHUD:
    """Тот же интерфейс, что у CursorIndicator, плюс set_level()."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None

    def start(self) -> None:
        if self._proc:
            return
        try:
            self._proc = subprocess.Popen(
                [_real_python(), os.path.abspath(__file__), "--child"],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, text=True, bufsize=1,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            )
        except Exception:
            self._proc = None

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


if __name__ == "__main__":
    if "--child" in sys.argv:
        sys.exit(_run_child())
    _demo()
