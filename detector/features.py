"""Ръчно конструирани стилометрични признаци (Стъпка 2).

Всички признаци са нормализирани спрямо дължината (на 100 думи, средни
стойности, съотношения), за да не може моделът да „хване“ просто дължината.
"""
import re
from collections import Counter, defaultdict

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .ngram_lm import CharNgramLM
from .text_utils import NUMBER_RE, split_sentences, words

# Най-честите български служебни думи.
FUNCTION_WORDS = set("""
и в на за с от да е се че не по като към при до но или а ще са би бе беше бяха съм си сме сте
то той тя те ние вие аз ти го я ги му ѝ им ми ни ви който която което които това този тази тези
там тук така също още вече само много така как кога където защото ако докато след преди между
през над под чрез без всеки всички някои друг други един една едно само
""".split())

# Изрази, които езиковите модели прекаляват да използват (ръчно подбран списък —
# евристика, не доказателство).
AI_PHRASES = [
    "също така", "освен това", "в заключение", "като цяло", "важно е да", "ключова роля",
    "важна роля", "съществена роля", "неразделна част", "благодарение на", "не само",
    "богато културно", "богата история", "символ на", "се отличава с", "съчетава",
    "впечатляващ", "уникален", "уникална", "уникално", "значително", "по този начин",
    "в съвременния", "в днешно време", "допринася за", "дълбоко", "разнообразие от",
]

FEATURES = {
    # име: (етикет, описание при високо, описание при ниско)
    "lexical_diversity": ("Лексикално разнообразие (MATTR)",
                          "думите се повтарят рядко — богат речник",
                          "думите често се повтарят — беден речник"),
    "hapax_ratio": ("Дял на думите, срещнати веднъж",
                    "много уникални думи",
                    "малко уникални думи"),
    "sent_len_mean": ("Средна дължина на изречение (думи)",
                      "изреченията са дълги",
                      "изреченията са къси"),
    "sent_len_std": ("Разсейване на дължината на изреченията (σ, думи)",
                     "дължината на изреченията силно варира",
                     "изреченията са необичайно равномерни по дължина"),
    "sent_burstiness": ("Burstiness на изреченията",
                        "текстът е „на пристъпи“ — редува къси и дълги изречения",
                        "ритъмът на изреченията е равномерен, почти механичен"),
    "word_burstiness": ("Burstiness на думите",
                        "думите идват на групи (типично за човешко писане)",
                        "повтарящите се думи са разпръснати равномерно"),
    "word_len_mean": ("Средна дължина на дума (букви)",
                      "думите са дълги",
                      "думите са къси"),
    "word_len_std": ("Разсейване на дължината на думите",
                     "смесват се къси и много дълги думи",
                     "думите са сходни по дължина"),
    "comma_rate": ("Запетаи на 100 думи",
                   "много запетаи",
                   "малко запетаи"),
    "colon_semicolon_rate": ("Двоеточия и точки и запетаи на 100 думи",
                             "често се ползват „:“ и „;“",
                             "рядко се ползват „:“ и „;“"),
    "dash_rate": ("Тирета на 100 думи",
                  "много тирета",
                  "малко тирета"),
    "paren_rate": ("Скоби на 100 думи",
                   "много пояснения в скоби",
                   "почти няма скоби"),
    "quote_rate": ("Кавички на 100 думи",
                   "много кавички",
                   "почти няма кавички"),
    "excl_quest_rate": ("Удивителни и въпросителни на 100 думи",
                        "емоционален/въпросителен тон",
                        "неутрален тон без възклицания"),
    "number_rate": ("Числа на 100 думи",
                    "много числа и дати — конкретика",
                    "почти няма числа и дати — общи формулировки"),
    "function_word_ratio": ("Дял служебни думи",
                            "много служебни думи (и, в, на, че…)",
                            "малко служебни думи — сбит, „номинален“ стил"),
    "ai_phrase_rate": ("Типични за ИИ изрази на 100 думи",
                       "има много клиширани изрази („също така“, „ключова роля“…)",
                       "почти няма клиширани изрази"),
    "char_entropy": ("Перплексити (бита/символ, n-грамен модел)",
                     "текстът е по-малко предсказуем",
                     "текстът е много предсказуем"),
    "char_entropy_std": ("Разсейване на предсказуемостта по изречения",
                         "предсказуемостта варира силно между изреченията",
                         "всички изречения са еднакво предсказуеми"),
}
FEATURE_NAMES = list(FEATURES)

# Признаци от невронните езикови модели (detector/binoculars.py). Изчисляват се
# отделно, защото са бавни, и се долепват след стиловите: ALL_FEATURE_NAMES.
NEURAL_FEATURES = {
    "bino_log_ppl": ("Невронна перплексити (Qwen, log)",
                     "текстът изненадва езиковия модел",
                     "текстът е лесно предсказуем за езиковия модел"),
    "bino_x_ppl": ("Очаквана изненада (крос-перплексити)",
                   "в тези позиции изборът на думи е широк",
                   "в тези позиции изборът на думи е тесен"),
    "binoculars": ("Binoculars (перплексити / крос-перплексити)",
                   "текстът е по-непредсказуем, отколкото би бил текст на модел — типично за човек",
                   "текстът е точно толкова предсказуем, колкото би го написал модел"),
}
ALL_FEATURES = {**FEATURES, **NEURAL_FEATURES}
ALL_FEATURE_NAMES = list(ALL_FEATURES)

# Групи за аблационния анализ. „Жанр“ са признаците, които отразяват по-скоро
# конвенциите на Уикипедия (произношение в скоби, години, цитирани заглавия),
# отколкото разликата човек/ИИ.
FEATURE_GROUPS = {
    "Лексика": ["lexical_diversity", "hapax_ratio"],
    "Изречения": ["sent_len_mean", "sent_len_std"],
    "Burstiness": ["sent_burstiness", "word_burstiness"],
    "Думи": ["word_len_mean", "word_len_std"],
    "Пунктуация": ["comma_rate", "colon_semicolon_rate", "dash_rate", "excl_quest_rate"],
    "Жанр (скоби, числа, кавички)": ["paren_rate", "number_rate", "quote_rate"],
    "Служебни думи и ИИ изрази": ["function_word_ratio", "ai_phrase_rate"],
    "Перплексити": ["char_entropy", "char_entropy_std"],
}
GENRE_FEATURES = FEATURE_GROUPS["Жанр (скоби, числа, кавички)"]


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Оставя само дадени признаци от пълния вектор (за модели върху подмножество)."""

    def __init__(self, keep):
        self.keep = keep

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X)[:, [ALL_FEATURE_NAMES.index(n) for n in self.keep]]


def _per100(count, n_words):
    return 100.0 * count / max(n_words, 1)


def _burstiness(values):
    """Goh–Barabási: B = (σ − μ) / (σ + μ); −1 равномерно, 0 случайно, → 1 на пристъпи."""
    if len(values) < 2:
        return 0.0
    mu, sigma = float(np.mean(values)), float(np.std(values))
    return (sigma - mu) / (sigma + mu) if sigma + mu else 0.0


def _mattr(tokens, window=50):
    if len(tokens) <= window:
        return len(set(tokens)) / max(len(tokens), 1)
    counts = Counter(tokens[:window])
    ratios = [len(counts) / window]
    for i in range(window, len(tokens)):
        counts[tokens[i]] += 1
        old = tokens[i - window]
        counts[old] -= 1
        if not counts[old]:
            del counts[old]
        ratios.append(len(counts) / window)
    return float(np.mean(ratios))


def _word_burstiness(tokens):
    positions = defaultdict(list)
    for i, w in enumerate(tokens):
        if w not in FUNCTION_WORDS and len(w) > 3:
            positions[w].append(i)
    scores = [_burstiness(np.diff(p)) for p in positions.values() if len(p) >= 3]
    return float(np.mean(scores)) if scores else 0.0


class FeatureExtractor:
    def __init__(self, lm: CharNgramLM):
        self.lm = lm

    def sentence_stats(self, text):
        return [(s, len(words(s))) for s in split_sentences(text)]

    def extract(self, text):
        tokens = words(text)
        n = len(tokens)
        sentences = split_sentences(text)
        sent_lens = [len(words(s)) for s in sentences] or [n]
        word_lens = [len(w) for w in tokens] or [0]
        lower = " " + re.sub(r"\s+", " ", text.lower()) + " "
        ai_hits = sum(lower.count(p) for p in AI_PHRASES)
        sent_bits = [self.lm.bits_per_char(s) for s in sentences if len(s) > 15]

        return {
            "lexical_diversity": _mattr(tokens),
            "hapax_ratio": sum(1 for c in Counter(tokens).values() if c == 1) / max(n, 1),
            "sent_len_mean": float(np.mean(sent_lens)),
            "sent_len_std": float(np.std(sent_lens)),
            "sent_burstiness": _burstiness(sent_lens),
            "word_burstiness": _word_burstiness(tokens),
            "word_len_mean": float(np.mean(word_lens)),
            "word_len_std": float(np.std(word_lens)),
            "comma_rate": _per100(text.count(","), n),
            "colon_semicolon_rate": _per100(text.count(":") + text.count(";"), n),
            "dash_rate": _per100(len(re.findall(r"\s[-–—]\s|—|–", text)), n),
            "paren_rate": _per100(text.count("("), n),
            "quote_rate": _per100(len(re.findall(r"[„“”\"«»]", text)) / 2, n),
            "excl_quest_rate": _per100(len(re.findall(r"[!?]", text)), n),
            "number_rate": _per100(len(NUMBER_RE.findall(text)), n),
            "function_word_ratio": sum(1 for w in tokens if w in FUNCTION_WORDS) / max(n, 1),
            "ai_phrase_rate": _per100(ai_hits, n),
            "char_entropy": self.lm.bits_per_char(text),
            "char_entropy_std": float(np.std(sent_bits)) if len(sent_bits) > 1 else 0.0,
        }

    def transform(self, texts):
        return np.array([[f[name] for name in FEATURE_NAMES] for f in map(self.extract, texts)],
                        dtype=float)

    # Позволява на извличането да е част от sklearn Pipeline-а.
    def fit(self, X, y=None):
        return self

