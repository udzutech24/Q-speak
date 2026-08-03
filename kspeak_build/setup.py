"""py2app build for K-speak (alias mode).

Сборка:
    python setup.py py2app -A

Alias-режим: бандл ссылается на локальные файлы (не копирует mlx/numpy),
но даёт СВОЙ бинарь-загрузчик с Info.plist → TCC атрибутирует микрофон
приложению K-speak, а не голому python.
"""
from setuptools import setup

APP = ["kspeak_main.py"]

OPTIONS = {
    "argv_emulation": False,
    "plist": {
        "CFBundleName": "K-speak",
        "CFBundleDisplayName": "K-speak",
        "CFBundleIdentifier": "com.kspeak.dictation",
        "CFBundleShortVersionString": "1.0",
        "CFBundleVersion": "1",
        "LSUIElement": True,
        "LSArchitecturePriority": ["arm64"],
        "NSMicrophoneUsageDescription": "Для голосовой диктовки через Whisper",
    },
}

setup(
    app=APP,
    name="K-speak",
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
