"""Разделяне на изречения и думи за български (и латиница)."""
import re

WORD_RE = re.compile(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*")
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
SENT_END_RE = re.compile(r"[.!?…]+[»“\")]*\s+")

# Съкращения, след които точката обикновено не е край на изречение.
ABBREVIATIONS = {
    "г", "гг", "в", "вв", "др", "напр", "т.е", "т.н", "вкл", "гр", "с", "р", "о", "ул", "бул",
    "проф", "д-р", "акад", "св", "хр", "пр", "сл", "н.е", "пр.н.е", "млн", "млрд", "хил", "км",
    "м", "см", "кв", "бел", "стр", "изд", "вж", "ср", "англ", "лат", "гр", "нем", "фр",
}


def words(text):
    return WORD_RE.findall(text.lower())


def split_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences, start = [], 0
    for m in SENT_END_RE.finditer(text):
        before = text[start:m.start() + 1]
        last = before.rstrip(".").split(" ")[-1].lower().strip("(„\"")
        nxt = text[m.end():m.end() + 1]
        # Не режем след съкращение/инициал или ако следва малка буква.
        if last in ABBREVIATIONS or (len(last) == 1 and last.isalpha()) or (nxt and nxt.islower()):
            continue
        sentences.append(text[start:m.end()].strip())
        start = m.end()
    if start < len(text):
        sentences.append(text[start:].strip())
    return [s for s in sentences if words(s)]
