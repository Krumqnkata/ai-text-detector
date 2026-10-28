"""Зареждане на обучения модел, предсказване и обяснение (Стъпки 5–6)."""
from pathlib import Path

import joblib
import numpy as np

from .features import FEATURE_NAMES, FEATURES
from .text_utils import split_sentences, words

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "detector.joblib"
MIN_RELIABLE_WORDS = 80


class Detector:
    """Обвивка около артефакта, записан от scripts/train.py."""

    def __init__(self, bundle):
        self.extractor = bundle["extractor"]
        self.models = bundle["models"]          # име -> (sklearn модел, вход: "features"|"text")
        self.main = bundle["main_model"]
        self.stats = bundle["feature_stats"]    # признак -> {human_mean, ai_mean, human_std}
        self.metrics = bundle.get("metrics", {})
        self.calibrator = bundle.get("calibrator")  # (a, b): P = σ(a·логит + b), Platt scaling

    @classmethod
    def load(cls, path=MODEL_PATH):
        return cls(joblib.load(path))

    def _proba(self, name, text, feats):
        model, kind = self.models[name]
        x = feats.reshape(1, -1) if kind == "features" else [text]
        return float(model.predict_proba(x)[0, 1])

    def main_proba(self, X):
        """Калибрирана вероятност от основния модел за матрица от признаци."""
        model = self.models[self.main][0]
        if self.calibrator is None:
            return model.predict_proba(X)[:, 1]
        a, b = self.calibrator
        return 1 / (1 + np.exp(-(a * model.decision_function(X) + b)))

    def contributions(self, feats):
        """Принос на всеки признак към логита на линейния модел: coef · z."""
        pipe = self.models[self.main][0]
        scaler, clf = pipe.named_steps["scale"], pipe.named_steps["clf"]
        z = (feats - scaler.mean_) / scaler.scale_
        return clf.coef_[0] * z, z

    def explain_feature(self, name, value, contrib):
        label, high, low = FEATURES[name]
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
        """„Човешкост“ по дължината на текста: плъзгащ се прозорец от изречения."""
        sentences = split_sentences(text)
        if len(sentences) < window + 1:
            return []
        chunks = [" ".join(sentences[i:i + window]) for i in range(len(sentences) - window + 1)]
        probs = self.main_proba(self.extractor.transform(chunks))
        # Вероятност за всяко изречение = средното от прозорците, които го покриват.
        per_sentence = []
        for i, sent in enumerate(sentences):
            covering = probs[max(0, i - window + 1):min(i + 1, len(probs))]
            per_sentence.append({"sentence": sent, "ai_probability": round(float(np.mean(covering)), 3)})
        return per_sentence

    def predict(self, text, top_k=5):
        n_words = len(words(text))
        feats = self.extractor.transform([text])[0]
        proba = {name: round(self._proba(name, text, feats), 4) for name in self.models}
        contrib, _ = self.contributions(feats)
        explained = [self.explain_feature(n, v, c) for n, v, c in zip(FEATURE_NAMES, feats, contrib)]
        top = sorted(explained, key=lambda e: abs(e["contribution"]), reverse=True)[:top_k]

        p = round(float(self.main_proba(feats.reshape(1, -1))[0]), 4)
        warnings = []
        if n_words < MIN_RELIABLE_WORDS:
            warnings.append(f"Текстът е кратък ({n_words} думи). Под ~{MIN_RELIABLE_WORDS} думи "
                            "признаците са шумни и резултатът е ненадежден.")
        if max(proba.values()) - min(proba.values()) > 0.5:
            warnings.append("Моделите силно се разминават — резултатът е несигурен.")
        return {
            "ai_probability": p,
            "calibrated": self.calibrator is not None,
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
