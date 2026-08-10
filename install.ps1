# QSpeak / whisper-skill — установка голосовой диктовки на Windows.
#
#   ПКМ по файлу → «Выполнить с помощью PowerShell»
#   либо:  powershell -ExecutionPolicy Bypass -File .\install.ps1
#          powershell -ExecutionPolicy Bypass -File .\install.ps1 -Check   # проверить установку
#
# Модель (~1.5 ГБ) качается сама при первой диктовке — в установку не входит.
param([switch]$Check)

$ErrorActionPreference = 'Stop'
$repo   = $PSScriptRoot
# Путь venv зашит в launcher\voice_dictation_silent.vbs — менять только вместе с ним.
$venv   = Join-Path $env:USERPROFILE '.venvs\whisper'
$vpy    = Join-Path $venv 'Scripts\python.exe'
$cfgDir = Join-Path $env:USERPROFILE '.config\whisper-skill'
$cfg    = Join-Path $cfgDir 'voice_dictation.json'

# PS 5.1: 'Set-Content -Encoding UTF8' пишет UTF-8 С BOM, а python читает
# конфиг как чистый UTF-8 и падает на первом же байте. Пишем без BOM руками.
function Set-Utf8NoBom {
    param([Parameter(ValueFromPipeline)][string]$InputObject, [Parameter(Mandatory)][string]$Path)
    process {
        [System.IO.File]::WriteAllText($Path, $InputObject, (New-Object System.Text.UTF8Encoding $false))
    }
}

function Show-Status {
    Write-Host "Python venv:  $(if (Test-Path $vpy) { $vpy } else { 'НЕ УСТАНОВЛЕН' })"
    Write-Host "ffmpeg:       $(if (Get-Command ffmpeg -EA SilentlyContinue) { 'есть' } else { 'НЕТ' })"
    Write-Host "Конфиг:       $(if (Test-Path $cfg) { $cfg } else { 'нет' })"
    $run = Get-ItemProperty -Path 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Run' -Name WhisperVoice -EA SilentlyContinue
    Write-Host "Автозапуск:   $(if ($run) { $run.WhisperVoice } else { 'не включён' })"
    Write-Host "Процессы:     $((Get-Process pythonw -EA SilentlyContinue | Measure-Object).Count) шт."
}

if ($Check) { Show-Status; exit 0 }

# 1. Python 3.10-3.13 (3.14 ещё без колёс под faster-whisper)
$py = Get-Command python -EA SilentlyContinue
$pyOk = $false
if ($py) {
    # В свежей Windows «python» — это заглушка-алиас на Store: вернёт не '1'.
    try { $pyOk = (& $py.Source -c "import sys; print(1 if (3,10)<=sys.version_info<(3,14) else 0)" 2>$null) -eq '1' } catch { $pyOk = $false }
}
if (-not $pyOk) {
    Write-Host '-> Ставлю Python 3.12 (winget)'
    winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
    $py = Get-Command python -EA SilentlyContinue
    if (-not $py) { throw 'Python установлен, но не виден в PATH. Закрой это окно, открой заново и запусти install.ps1 ещё раз.' }
}
Write-Host "-> Python: $($py.Source)"

# 2. ffmpeg
if (-not (Get-Command ffmpeg -EA SilentlyContinue)) {
    Write-Host '-> Ставлю ffmpeg (winget)'
    winget install -e --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements
}

# 3. venv + зависимости
if (-not (Test-Path $vpy)) {
    Write-Host "-> Создаю окружение: $venv"
    & $py.Source -m venv $venv
}
Write-Host '-> Ставлю зависимости (2-5 минут)'
& $vpy -m pip install --quiet --upgrade pip
& $vpy -m pip install --quiet faster-whisper sounddevice soundfile pynput pyperclip pystray Pillow numpy

# 4. Конфиг и словарь — только если их ещё нет, переустановка не затирает настройки.
New-Item -ItemType Directory -Force -Path $cfgDir | Out-Null
if (-not (Test-Path $cfg)) {
    @'
{
  "hotkey": "<ctrl>+<shift>+<space>",
  "mode": "toggle",
  "language": "ru",
  "model": "large-v3-turbo",
  "backend": "faster",
  "auto_paste": true,
  "play_sound": true,
  "show_tray": true,
  "show_cursor_indicator": true,
  "keep_in_clipboard": true,
  "trim_silence_ms": 200,
  "min_duration_ms": 300
}
'@ | Set-Utf8NoBom -Path $cfg
}
$vocab = Join-Path $cfgDir 'vocabulary.txt'
if (-not (Test-Path $vocab)) {
    '# Формат: Правильно = как слышится, ещё вариант' | Set-Utf8NoBom -Path $vocab
}

# 5. Автозапуск при входе в систему + ярлык
& (Join-Path $repo 'tools\install_autostart.ps1')

# 6. Запустить сейчас
Start-Process wscript.exe -ArgumentList "`"$(Join-Path $repo 'launcher\voice_dictation_silent.vbs')`""

Write-Host ''
Write-Host '✅ Готово.'
Write-Host '   Хоткей: Ctrl+Shift+Space — нажать, сказать фразу, нажать ещё раз.'
Write-Host '   Первая диктовка качает модель ~1.5 ГБ (одна-две минуты), дальше всё локально.'
Write-Host '   Текст сам вставится в активное поле и останется в буфере обмена.'
Write-Host ''
Write-Host "   Настройки: $cfg"
Write-Host "   Словарь терминов: $vocab"
Write-Host "   Проверка: powershell -ExecutionPolicy Bypass -File .\install.ps1 -Check"
Write-Host ''
Show-Status
