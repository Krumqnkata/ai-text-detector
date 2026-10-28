"""Зареждане на обучения модел, предсказване и обяснение (Стъпки 5–6)."""
from pathlib import Path

import joblib
import numpy as np

from . import binoculars
from .features import ALL_FEATURE_NAMES, ALL_FEATURES, FEATURE_NAMES
from .text_utils import split_sentences, words

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "detector.joblib"
MIN_RELIABLE_WORDS = 80
FALLBACK_MODEL = "features_lr"  # само стилови признаци — бърз и без езикови модели


class Detector:
    """Обвивка около артефакта, записан от scripts/train.py.

    Моделите са (sklearn модел, вход), където входът е "features" (стиловите
    признаци), "neural" (стиловите + Binoculars) или "text" (суров текст).
    """

    def __init__(self, bundle):
        self.extractor = bundle["extractor"]
        self.models = dict(bundle["models"])
        self.main = bundle["main_model"]
        self.stats = bundle["feature_stats"]    # признак -> {human_mean, ai_mean, human_std}
        self.metrics = bundle.get("metrics", {})
        # (a, b) по модел: P = σ(a·логит + b), Platt scaling.
        self.calibrators = bundle.get("calibrators") or {self.main: bundle.get("calibrator")}
        self.bino = bundle.get("binoculars")
        if self.bino is None or not binoculars.available():
            # Без езиковите модели падаме обратно на стиловия модел.
            self.bino = None
            self.models = {k: v for k, v in self.models.items() if v[1] != "neural"}
            if self.main not in self.models:
                self.main = FALLBACK_MODEL

    @classmethod
    def load(cls, path=MODEL_PATH):
        return cls(joblib.load(path))

    @property
    def neural(self):
        return self.bino is not None

    def vectors(self, texts):
        """{"features": стилови признаци, "neural": стилови + Binoculars (ако е наличен)}."""
        base = self.extractor.transform(texts)
        out = {"features": base}
        if self.neural:
            out["neural"] = np.hstack([base, self.bino.transform(texts)])
        return out

    def proba(self, name, X, texts=None):
        """Калибрирана (ако има калибратор) вероятност за ИИ от даден модел."""
        model, kind = self.models[name]
        x = texts if kind == "text" else X[kind]
        cal = self.calibrators.get(name)
        if cal is None:
            return model.predict_proba(x)[:, 1]
        a, b = cal
        return 1 / (1 + np.exp(-(a * model.decision_function(x) + b)))

    def feature_names(self, name=None):
        return ALL_FEATURE_NAMES if self.models[name or self.main][1] == "neural" else FEATURE_NAMES

    def contributions(self, vec):
        """Принос на всеки признак към логита на основния (линеен) модел: coef · z."""
        pipe = self.models[self.main][0]
        scaler, clf = pipe.named_steps["scale"], pipe.named_steps["clf"]
        z = (vec - scaler.mean_) / scaler.scale_
        return clf.coef_[0] * z, z

    def explain_feature(self, name, value, contrib):
        label, high, low = ALL_FEATURES[name]
        st = self.stats[name]
        is_high = value > st["human_mean"]
        return {
            "name": name,
            "label": label,
            "value": round(float(value), 3),
            "human_mean": round(st["human_mean"], 3),
            "ai_mean": round(st["ai_mean"], 3),
            "contribution": round(float(contrib), 3),
            "toward": "ai" if contrib > 0 else "human",
            "text": (high if is_high else low)
                    + f" ({value:.2f} при средно {st['human_mean']:.2f} за човешки"
                    + f" и {st['ai_mean']:.2f} за ИИ текстове)",
        }

    def segments(self, text, window=3):
        """„Човешкост“ по дължината на текста: плъзгащ се прозорец от изречения.

        Ползва стиловия модел: езиковите модели биха отнели секунди на всеки прозорец.
        """
        sentences = split_sentences(text)
        if len(sentences) < window + 1:
            return []
        chunks = [" ".join(sentences[i:i + window]) for i in range(len(sentences) - window + 1)]
        probs = self.proba(FALLBACK_MODEL, {"features": self.extractor.transform(chunks)})
        # Вероятност за всяко изречение = средното от прозорците, които го покриват.
        per_sentence = []
        for i, sent in enumerate(sentences):
            covering = probs[max(0, i - window + 1):min(i + 1, len(probs))]
            per_sentence.append({"sentence": sent, "ai_probability": round(float(np.mean(covering)), 3)})
        return per_sentence

    def predict(self, text, top_k=5):
        n_words = len(words(text))
        X = self.vectors([text])
        proba = {name: round(float(self.proba(name, X, [text])[0]), 4) for name in self.models}
        vec = X[self.models[self.main][1]][0]
        contrib, _ = self.contributions(vec)
        explained = [self.explain_feature(n, v, c) for n, v, c in zip(self.feature_names(), vec, contrib)]
        top = sorted(explained, key=lambda e: abs(e["contribution"]), reverse=True)[:top_k]

        p = proba[self.main]
        warnings = []
        if n_words < MIN_RELIABLE_WORDS:
            warnings.append(f"Текстът е кратък ({n_words} думи). Под ~{MIN_RELIABLE_WORDS} думи "
                            "признаците са шумни и резултатът е ненадежден.")
        if max(proba.values()) - min(proba.values()) > 0.5:
            warnings.append("Моделите силно се разминават — резултатът е несигурен.")
        return {
            "ai_probability": p,
            "calibrated": self.calibrators.get(self.main) is not None,
            "neural": self.neural,
            "verdict": verdict(p) if n_words >= MIN_RELIABLE_WORDS / 2 else "Текстът е твърде кратък за извод",
            "word_count": n_words,
            "model": self.main,
            "model_probabilities": proba,
            "top_features": top,
            "features": explained,
            "segments": self.segments(text),
            "warnings": warnings,
        }


def verdict(p):
    if p >= 0.8:
        return "Вероятно е писан от ИИ"
    if p >= 0.6:
        return "По-скоро ИИ"
    if p > 0.4:
        return "Неясно — признаците са смесени"
    if p > 0.2:
        return "По-скоро човек"
    return "Вероятно е писан от човек"
