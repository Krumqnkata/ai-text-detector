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


def test_disabled_unless_opted_in(monkeypatch):
    monkeypatch.delenv("DETECTOR_LM", raising=False)
    assert binoculars.enabled() is False
    monkeypatch.setenv("DETECTOR_LM", "1")
    assert binoculars.enabled() is binoculars.installed()


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="няма обучен модел (scripts/train.py)")
def test_detector_uses_style_model_by_default(monkeypatch):
    monkeypatch.delenv("DETECTOR_LM", raising=False)
    det = Detector.load()
    assert not det.neural and det.main == "features_lr"
    assert all(kind != "neural" for _, kind in det.models.values())
    text = ("Това е обикновен текст за проверка. Той има няколко изречения с различна дължина. "
            "Някои са кратки. Други са малко по-дълги и съдържат повече думи, за да има какво да се мери.")
    out = det.predict(text)
    assert out["neural"] is False and 0 <= out["ai_probability"] <= 1


@pytest.mark.skipif(not (MODEL_PATH.exists() and binoculars.installed()), reason="няма модел или torch")
def test_detector_uses_binoculars_when_opted_in(monkeypatch):
    from tests.test_api import TEXT  # резултатът му е в кеша, така че езиковите модели не се зареждат

    monkeypatch.setenv("DETECTOR_LM", "1")
    det = Detector.load()
    assert det.neural and det.main == "features_bino_lr"
    out = det.predict(TEXT)
    assert out["neural"] is True
    assert {f["name"] for f in out["features"]} >= {"binoculars", "bino_log_ppl", "bino_x_ppl"}
