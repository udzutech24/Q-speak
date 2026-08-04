"""py2app build for QSpeak (alias mode).

Сборка:
    python setup.py py2app -A

Alias-режим: бандл ссылается на локальные файлы (не копирует mlx/numpy),
но даёт СВОЙ бинарь-загрузчик с Info.plist → TCC атрибутирует микрофон
приложению QSpeak, а не голому python.
"""
from setuptools import setup

APP = ["qspeak_main.py"]

OPTIONS = {
    "argv_emulation": False,
    "plist": {
        "CFBundleName": "QSpeak",
        "CFBundleDisplayName": "QSpeak",
        "CFBundleIdentifier": "com.qspeak.dictation",
        "CFBundleShortVersionString": "1.0",
        "CFBundleVersion": "1",
        "LSUIElement": True,
        "LSArchitecturePriority": ["arm64"],
        "NSMicrophoneUsageDescription": "Для голосовой диктовки через Whisper",
    },
}

setup(
    app=APP,
    name="QSpeak",
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
