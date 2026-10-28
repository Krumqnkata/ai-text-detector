import pickle

import pytest

from detector import binoculars
from detector.binoculars import Binoculars
from detector.model import MODEL_PATH, Detector


def test_pickle_keeps_config_but_not_models():
    b = Binoculars(max_tokens=128)
    b._models, b._cache = "тежки модели", {"k": [1, 2, 3]}
    restored = pickle.loads(pickle.dumps(b))
    assert restored._models is None and restored._cache is None
    assert (restored.observer, restored.performer, restored.max_tokens) == (b.observer, b.performer, 128)


def test_cache_key_depends_on_text_and_models():
    a, b = Binoculars(), Binoculars(performer="друг/модел")
    assert a._key("текст") != a._key("друг текст")
    assert a._key("текст") != b._key("текст")


def test_available_respects_env(monkeypatch):
    monkeypatch.setenv("DETECTOR_NO_LM", "1")
    assert binoculars.available() is False


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="няма обучен модел (scripts/train.py)")
def test_detector_falls_back_without_lm(monkeypatch):
    monkeypatch.setenv("DETECTOR_NO_LM", "1")
    det = Detector.load()
    assert not det.neural
    assert all(kind != "neural" for _, kind in det.models.values())
    text = ("Това е обикновен текст за проверка. Той има няколко изречения с различна дължина. "
            "Някои са кратки. Други са малко по-дълги и съдържат повече думи, за да има какво да се мери.")
    out = det.predict(text)
    assert out["neural"] is False and 0 <= out["ai_probability"] <= 1
