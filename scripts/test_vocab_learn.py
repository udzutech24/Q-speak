"""Пополнение словаря: дозапись правил и разбор правки эталона.

Запуск: python3 scripts/test_vocab_learn.py   (из корня скилла)
"""
import json, os, sys, tempfile
from pathlib import Path

tmp = tempfile.mkdtemp()
os.environ["XDG_CONFIG_HOME"] = tmp          # читается на каждый вызов, до импорта не обязателен

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import examples.voice_dictation as vd

vd.notify = lambda *a, **k: None             # без баннеров в тесте

d = Path(tmp) / "whisper-skill"
d.mkdir(parents=True)
vocab = d / "vocabulary.txt"
vocab.write_text("# коммент = не правило\nSafeGrip = сейфгрип\n", encoding="utf-8")

assert vd.add_vocabulary_rule("Xingyu", "синкани"), "новый термин не добавлен"
assert vd.add_vocabulary_rule("safegrip", "сейф грип"), "вариант к существующему не добавлен"
assert not vd.add_vocabulary_rule("SafeGrip", "СЕЙФГРИП"), "дубль варианта не отсечён"

rules = dict(vd.load_vocabulary())
assert rules["Xingyu"] == ["синкани"], rules
assert sorted(rules["SafeGrip"]) == ["сейф грип", "сейфгрип"], rules
assert "# коммент" in vocab.read_text(encoding="utf-8"), "комментарии затёрты"
assert vd.apply_vocabulary("звонил в синкани", vd.load_vocabulary()) == "звонил в Xingyu"

# «Поправить последнее»: изменённые слова становятся правилами, переписанная
# фраза целиком — нет
(d / "dataset").mkdir()
(d / "dataset" / "pairs.jsonl").write_text(json.dumps(
    {"wav": "x.wav", "asr": "заказ у синкони готов", "truth": "заказ у Xingyu готов"},
    ensure_ascii=False) + "\n", encoding="utf-8")
res = vd.learn_from_last_pair()
assert res == "Xingyu ← синкони", res
assert vd.learn_from_last_pair() == "новых правил нет", "правило добавилось дважды"

(d / "dataset" / "pairs.jsonl").write_text(json.dumps(
    {"wav": "x.wav", "asr": "а б в г д", "truth": "совсем другая длинная фраза тут"},
    ensure_ascii=False) + "\n", encoding="utf-8")
assert vd.learn_from_last_pair() == "новых правил нет", "длинный кусок ушёл в словарь"

# Направление: слово из последней диктовки → спрашиваем правильное написание;
# незнакомое → это уже правка руками, кривой вариант берём из той же диктовки
(d / "last.json").write_text(json.dumps(
    {"wav": "x.wav", "asr": "поставка от синкани"}, ensure_ascii=False), encoding="utf-8")
terms = vd._last_asr_terms()
assert "синкани" in terms and "от синкани" in terms, terms

boom = lambda *a, **k: (_ for _ in ()).throw(AssertionError("лишний диалог"))

vocab.write_text("", encoding="utf-8")
vd.copy_selection, vd._choose = (lambda: "синкани"), boom
vd._ask = lambda q, default="": "Xingyu"
vd.learn_selected_word()
assert dict(vd.load_vocabulary())["Xingyu"] == ["синкани"], vocab.read_text(encoding="utf-8")

# похожее написание — вопросов не задаём вовсе
vocab.write_text("", encoding="utf-8")
vd.copy_selection, vd._ask, vd._choose = (lambda: "Синькони"), boom, boom
vd.learn_selected_word()
assert dict(vd.load_vocabulary())["Синькони"] == ["синкани"], vocab.read_text(encoding="utf-8")

# другой алфавит — похожести нет, показываем список слов диктовки
vocab.write_text("", encoding="utf-8")
picked = {}
vd.copy_selection, vd._ask = (lambda: "Xingyu"), boom
vd._choose = lambda items, prompt: picked.setdefault("items", items) and "" or "синкани"
vd.learn_selected_word()
assert "синкани" in picked["items"], picked["items"]
assert dict(vd.load_vocabulary())["Xingyu"] == ["синкани"], vocab.read_text(encoding="utf-8")

print("✅ словарь: дозапись, дубли, разбор эталона и все три направления выделения")
