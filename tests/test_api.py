import pytest
from fastapi.testclient import TestClient

from app import main
from detector.model import MODEL_PATH

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="няма обучен модел (scripts/train.py)")

TEXT = ("Фотосинтезата е процес, при който растенията превръщат светлинната енергия в химична. "
        "Тя протича в хлоропластите. С помощта на светлината въглеродният диоксид и водата се "
        "превръщат в глюкоза, а като страничен продукт се отделя кислород. Процесът е в основата "
        "на почти всички хранителни вериги. Освен това той поддържа нивата на кислород в атмосферата "
        "и играе ключова роля в кръговрата на въглерода. Без фотосинтезата животът на Земята, какъвто "
        "го познаваме, не би бил възможен. Учените я изучават вече повече от два века.")


@pytest.fixture(scope="module")
def client():
    return TestClient(main.app)


def test_index(client):
    assert client.get("/").status_code == 200


def test_predict_shape(client):
    data = client.post("/predict", json={"text": TEXT}).json()
    assert 0 <= data["ai_probability"] <= 1
    assert data["calibrated"] is True
    assert set(data["model_probabilities"]) >= {"features_lr", "features_rf", "tfidf_lr"}
    assert len(data["top_features"]) == 5
    assert {"label", "value", "human_mean", "ai_mean", "toward", "text"} <= set(data["top_features"][0])
    assert data["segments"] and all(0 <= s["ai_probability"] <= 1 for s in data["segments"])


def test_contributions_explain_the_logit(client):
    """Сумата от приносите + свободния член трябва да дава точно логита на модела."""
    det = main.detector
    vec = det.vectors([TEXT])[det.models[det.main][1]]
    contrib, _ = det.contributions(vec[0])
    pipe = det.models[det.main][0]
    assert contrib.sum() + pipe.named_steps["clf"].intercept_[0] == pytest.approx(
        pipe.decision_function(vec)[0])


def test_short_text_warning(client):
    data = client.post("/predict", json={"text": "Това е съвсем кратък текст за проба."}).json()
    assert data["verdict"] == "Текстът е твърде кратък за извод"
    assert any("кратък" in w for w in data["warnings"])


def test_rejects_tiny_and_empty(client):
    assert client.post("/predict", json={"text": "две думи"}).status_code == 422
    assert client.post("/predict", json={"text": ""}).status_code == 422
