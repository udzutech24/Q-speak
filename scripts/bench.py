"""Замер качества диктовки на собственном наборе эталонов.

Набор копится через меню QSpeak → «Поправить последнее…»:
    ~/.config/whisper-skill/dataset/pairs.jsonl  +  wav-файлы рядом.

    python3 scripts/bench.py                          # текущая модель из конфига
    python3 scripts/bench.py --model mlx-community/whisper-large-v3-turbo
    python3 scripts/bench.py --selftest               # проверка счётчика WER

Печатает WER (доля неверных слов), точность терминов из словаря и задержку.
Цифра сама по себе ничего не значит — смысл в сравнении «до / после».
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from difflib import SequenceMatcher

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CFG_DIR = os.path.expanduser("~/.config/whisper-skill")
DATASET_DIR = os.path.join(CFG_DIR, "dataset")
PAIRS_FILE = os.path.join(DATASET_DIR, "pairs.jsonl")


def _words(s: str) -> list:
    return re.findall(r"\w+", s.lower())


def wer(truth: str, hyp: str) -> float:
    """Доля неверных слов: (замены + пропуски + вставки) / слов в эталоне.

    Считаем через SequenceMatcher — стандартная библиотека вместо jiwer ради
    одной формулы. На коротких диктовках расхождение с классическим
    Левенштейном непринципиально.
    """
    a, b = _words(truth), _words(hyp)
    if not a:
        return 0.0 if not b else 1.0
    matched = sum(bl.size for bl in SequenceMatcher(None, a, b).get_matching_blocks())
    return (max(len(a), len(b)) - matched) / len(a)


def load_pairs() -> list:
    try:
        with open(PAIRS_FILE, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except OSError:
        return []


def terms_from_vocabulary() -> list:
    """Левые части словаря — термины, которые обязаны попадать в текст."""
    out = []
    try:
        with open(os.path.join(CFG_DIR, "vocabulary.txt"), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    out.append(line.split("=", 1)[0].strip())
    except OSError:
        pass
    return out


def run(model: str | None) -> int:
    pairs = load_pairs()
    if not pairs:
        print(f"Набор пуст: {PAIRS_FILE}\n"
              "Копи эталоны через меню QSpeak → «Поправить последнее…»")
        return 1

    from examples.common import transcribe
    from examples.voice_dictation import load_config, apply_vocabulary, load_vocabulary

    cfg = load_config()
    model = model or cfg["model"]
    vocab, terms = load_vocabulary(), terms_from_vocabulary()

    total_wer, latencies, term_hits, term_total, worst = 0.0, [], 0, 0, []
    for p in pairs:
        wav = os.path.join(DATASET_DIR, p["wav"])
        if not os.path.exists(wav):
            continue
        t0 = time.time()
        text = apply_vocabulary(
            transcribe(wav, language=cfg.get("language"), model_name=model,
                       word_timestamps=False, verbose=False).text.strip(), vocab)
        latencies.append(time.time() - t0)
        e = wer(p["truth"], text)
        total_wer += e
        worst.append((e, p["truth"], text))
        for t in terms:                      # термин ждём только там, где он есть в эталоне
            if t.lower() in p["truth"].lower():
                term_total += 1
                term_hits += t.lower() in text.lower()

    n = len(latencies)
    if not n:
        print("Ни одного wav из набора не нашлось на диске")
        return 1

    print(f"\nМодель:    {model}")
    print(f"Записей:   {n}")
    print(f"WER:       {total_wer / n * 100:.1f}%   (меньше — лучше)")
    print(f"Термины:   {term_hits}/{term_total}" if term_total else "Термины:   нет в эталонах")
    print(f"Задержка:  {sum(latencies) / n:.2f} с в среднем, "
          f"{max(latencies):.2f} с худшая")

    worst.sort(reverse=True)
    print("\nХудшие записи:")
    for e, truth, got in worst[:3]:
        if e == 0:
            break
        print(f"  WER {e * 100:.0f}%\n    сказано:   {truth[:100]}\n    услышано:  {got[:100]}")
    return 0


def _selftest() -> None:
    assert wer("привет как дела", "привет как дела") == 0.0
    assert wer("привет как дела", "привет как жизнь") - 1 / 3 < 1e-9   # одна замена из трёх
    assert wer("один два три четыре", "один два") == 0.5               # два пропуска
    assert wer("", "") == 0.0 and wer("", "лишнее") == 1.0
    print("ok: счётчик WER считает замены, пропуски и пустые случаи")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="проверить другую модель, не трогая конфиг")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    else:
        sys.exit(run(a.model))
