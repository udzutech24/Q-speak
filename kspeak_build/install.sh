#!/bin/bash
# K-speak — установка на чистый Mac (Apple Silicon).
#
#   ./kspeak_build/install.sh          # поставить
#   ./kspeak_build/install.sh --check  # проверить установленное
#
# Модели (~4.4 ГБ) качаются сами при первой диктовке — в установку не входят,
# поэтому standalone-бандл смысла не имеет: тяжёлое всё равно тянется с сети.
set -euo pipefail

SKILL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${KSPEAK_PYTHON:-$(command -v python3 || true)}"
APP="/Applications/K-speak.app"
CFG="$HOME/.config/whisper-skill"
PLIST="$HOME/Library/LaunchAgents/com.kspeak.dictation.plist"

check() {
    echo "Приложение: $([ -d "$APP" ] && echo "$APP" || echo "НЕ УСТАНОВЛЕНО")"
    echo "Подпись:    $(codesign --verify "$APP" 2>&1 || echo "битая → codesign --force --deep --sign - $APP")"
    echo "Автозапуск: $(launchctl list | grep com.kspeak.dictation || echo "не загружен")"
    echo "Процессы:   $(pgrep -fl "K-speak|hud_mac" | tr '\n' ' ' || echo нет)"
    echo "Лог:";       tail -5 /tmp/kspeak.log 2>/dev/null || echo "  пуст"
}

[ "${1:-}" = "--check" ] && { check; exit 0; }

[ "$(uname -m)" = "arm64" ] || { echo "❌ Нужен Mac на Apple Silicon: mlx на Intel не работает"; exit 1; }
[ -n "$PY" ] || { echo "❌ Нет python3. Поставь: brew install python@3.12"; exit 1; }
command -v ffmpeg >/dev/null || { echo "→ ffmpeg через brew"; brew install ffmpeg; }

echo "→ Python: $PY ($("$PY" -V))"
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet mlx-whisper sounddevice soundfile pynput pyperclip numpy \
    pyobjc-framework-Cocoa pyobjc-framework-AVFoundation py2app

# Конфиг и словарь заводим только если их ещё нет — переустановка не затирает настройки.
mkdir -p "$CFG"
[ -f "$CFG/voice_dictation.json" ] || cp "$SKILL/kspeak_build/default_config.json" "$CFG/voice_dictation.json"
[ -f "$CFG/vocabulary.txt" ] || printf '# Формат: Правильно = как слышится, ещё вариант\n' > "$CFG/vocabulary.txt"

# Бандл в alias-режиме: ссылается на этот каталог, но даёт свой Info.plist —
# TCC тогда спрашивает микрофон от имени K-speak, а не голого python.
echo "→ Собираю K-speak.app"
cd "$SKILL/kspeak_build"
rm -rf build dist
"$PY" setup.py py2app -A >/dev/null
rm -rf "$APP"
cp -R dist/K-speak.app "$APP"
# Любая правка содержимого бандла ломает подпись, а macOS после этого молча не
# выдаёт Accessibility — галочка в настройках при этом выглядит включённой.
codesign --force --deep --sign - "$APP"

# launchd стартует с урезанным PATH без homebrew → mlx_whisper не найдёт ffmpeg.
echo "→ Автозапуск"
cat > "$PLIST" <<PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.kspeak.dictation</string>
    <key>ProgramArguments</key>
    <array>
        <string>$APP/Contents/MacOS/K-speak</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
        <key>KSPEAK_PYTHON</key>
        <string>$PY</string>
    </dict>
</dict>
</plist>
PLIST_EOF
launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"
sleep 5

cat <<TXT

✅ Установлено. Осталось руками — macOS иначе не пустит:
   1. Системные настройки → Конфиденциальность → Микрофон → включить K-speak
   2. Там же → Универсальный доступ (Accessibility) → добавить и включить K-speak
      (без него не ловится хоткей и не работает вставка)
   3. Нажать правый Alt, сказать фразу, нажать ещё раз. Первая диктовка тянет
      модель ~4.4 ГБ — это одна минута ожидания, дальше локально и мгновенно.

   Хоткей и язык — в меню 🎙 в статус-баре. Словарь терминов: $CFG/vocabulary.txt
   Лог: /tmp/kspeak.log · проверка установки: $0 --check

TXT
check
