from detector.text_utils import split_sentences, words


def test_words_lowercase_and_hyphenated():
    assert words("Стара планина е Балкан-ска верига!") == ["стара", "планина", "е", "балкан-ска", "верига"]


def test_words_skip_numbers_and_punctuation():
    assert words("През 1876 г. (юни) — 1-ви") == ["през", "г", "юни", "ви"]


def test_split_basic():
    assert split_sentences("Първо изречение. Второ изречение! Трето?") == [
        "Първо изречение.", "Второ изречение!", "Трето?"]


def test_split_keeps_abbreviations():
    text = "Ботев загива през 1876 г. Във Врачанския Балкан. Това е т.е. пример."
    sents = split_sentences(text)
    assert sents[0] == "Ботев загива през 1876 г. Във Врачанския Балкан."
    assert len(sents) == 2


def test_split_keeps_initials():
    assert len(split_sentences("Авторът е Хр. Ботев. Той е поет.")) == 2


def test_split_ignores_period_before_lowercase():
    assert len(split_sentences("Около 5 км. от града има река. Тя е дълга.")) == 2


def test_split_empty():
    assert split_sentences("   ") == []
