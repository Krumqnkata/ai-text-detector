import math

import numpy as np
import pytest

from detector.features import FEATURE_NAMES, ColumnSelector, FeatureExtractor, _burstiness, _mattr
from detector.ngram_lm import CharNgramLM

REFERENCE = ["Българският език е южнославянски език.", "Столицата на България е София.",
             "Реките в България се вливат в Черно море или в Бяло море."]

UNIFORM = " ".join(["Котката спи на топлия диван в стаята."] * 6)
VARIED = ("Да. Котката, която живееше у баба ми на село, спеше по цял ден на топлия диван "
          "до печката, докато навън валеше сняг. Странно? Не, напълно нормално за котка.")


@pytest.fixture(scope="module")
def extractor():
    return FeatureExtractor(CharNgramLM(order=3).fit(REFERENCE))


def test_extract_returns_all_features(extractor):
    feats = extractor.extract(VARIED)
    assert list(feats) == FEATURE_NAMES
    assert all(math.isfinite(v) for v in feats.values())


def test_uniform_sentences_have_low_spread(extractor):
    uniform, varied = extractor.extract(UNIFORM), extractor.extract(VARIED)
    assert uniform["sent_len_std"] == 0
    assert varied["sent_len_std"] > uniform["sent_len_std"]
    assert varied["sent_burstiness"] > uniform["sent_burstiness"]


def test_lexical_diversity_drops_with_repetition(extractor):
    assert extractor.extract(UNIFORM)["lexical_diversity"] < extractor.extract(VARIED)["lexical_diversity"]


def test_rates_are_per_100_words(extractor):
    text = "Едно, две, три, четири, пет, шест, седем, осем, девет, десет"
    assert extractor.extract(text)["comma_rate"] == pytest.approx(90.0)


def test_ai_phrases_counted(extractor):
    feats = extractor.extract("Също така, това играе ключова роля. Освен това е важно.")
    assert feats["ai_phrase_rate"] > 0


def test_burstiness_bounds():
    assert _burstiness([5, 5, 5, 5]) == -1.0
    assert -1 <= _burstiness([1, 30, 2, 25]) <= 1


def test_mattr_short_text_is_plain_ttr():
    assert _mattr(["а", "б", "а"]) == pytest.approx(2 / 3)


def test_lm_prefers_seen_text():
    lm = CharNgramLM(order=3).fit(REFERENCE * 5)
    assert lm.bits_per_char("Столицата на България") < lm.bits_per_char("Xqzw jjkv pptt")


def test_column_selector_keeps_order():
    X = np.arange(len(FEATURE_NAMES) * 2, dtype=float).reshape(2, -1)
    out = ColumnSelector(["comma_rate", "lexical_diversity"]).transform(X)
    assert out.tolist() == X[:, [FEATURE_NAMES.index("comma_rate"), 0]].tolist()
