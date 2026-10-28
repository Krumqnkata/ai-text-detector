"""Обединява човешките и ИИ текстовете, балансира класовете и прави train/test.

- Човешките текстове се режат до --max-words думи (по граница на изречение),
  за да са със сходна дължина с ИИ текстовете.
- Разделянето е по тема: една тема е или само в train, или само в test,
  иначе TF-IDF моделът би „запомнил“ темата вместо стила.
- Балансът и разделянето се правят поотделно за всеки жанр (колона `genre`;
  файловете без нея се броят за "encyclopedia"), за да не стане жанрът подсказка.
- --holdout-model изважда текстовете на даден модел в test_unseen.csv —
  те не участват в обучението и показват устойчивостта към нов модел.

    python scripts/build_dataset.py
    python scripts/build_dataset.py --holdout-model claude-haiku-5-5
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from detector.text_utils import split_sentences, words  # noqa: E402


def truncate(text, max_words):
    out, count = [], 0
    for sent in split_sentences(text):
        n = len(words(sent))
        if count and count + n > max_words:
            break
        out.append(sent)
        count += n
    return " ".join(out)


def split_genre(df, test_size, rng):
    """Балансира класовете в един жанр и го разделя на train/test по теми."""
    # Махаме излишъка от по-големия клас, като първо пазим темите,
    # които имат и двата класа (най-добре съпоставими двойки).
    counts = df["label"].value_counts()
    major = counts.idxmax()
    paired_topics = set(df[df.label == 0]["topic"]) & set(df[df.label == 1]["topic"])
    major_rows = df[df.label == major]
    priority = major_rows["topic"].isin(paired_topics).astype(int) + rng.random(len(major_rows))
    keep = major_rows.loc[priority.sort_values(ascending=False).index[:counts.min()]]
    df = pd.concat([df[df.label != major], keep])

    topics = np.array(df["topic"].unique(), dtype=object)
    rng.shuffle(topics)
    target = test_size * len(df)
    test_topics, n = set(), 0
    for topic in topics:
        if n >= target:
            break
        test_topics.add(topic)
        n += (df["topic"] == topic).sum()
    return df[~df["topic"].isin(test_topics)], df[df["topic"].isin(test_topics)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--human", nargs="+", type=Path,
                    default=sorted((ROOT / "data").glob("human*.csv")))
    ap.add_argument("--ai", nargs="+", type=Path,
                    default=sorted((ROOT / "data").glob("ai_*.csv")))
    ap.add_argument("--holdout-model", nargs="*", default=[])
    ap.add_argument("--max-words", type=int, default=130,
                    help="горна граница на дължината (близо до дължината на ИИ текстовете)")
    ap.add_argument("--min-words", type=int, default=60)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    human = pd.concat([pd.read_csv(p) for p in args.human], ignore_index=True)
    ai = pd.concat([pd.read_csv(p) for p in args.ai], ignore_index=True)
    human["label"], ai["label"] = 0, 1
    human["text"] = human["text"].map(lambda t: truncate(t, args.max_words))
    ai["text"] = ai["text"].map(lambda t: truncate(t, args.max_words))
    df = pd.concat([human, ai], ignore_index=True)
    df["genre"] = df["genre"].fillna("encyclopedia") if "genre" in df else "encyclopedia"
    df = df[df["text"].map(lambda t: len(words(t))) >= args.min_words].reset_index(drop=True)

    unseen = df[df["model"].isin(args.holdout_model)]
    df = df[~df["model"].isin(args.holdout_model)]

    rng = np.random.default_rng(args.seed)
    if df.groupby("genre")["label"].nunique().min() < 2:
        sys.exit("Някой жанр няма и двата класа — провери входните файлове.")
    parts = [split_genre(g, args.test_size, rng) for _, g in df.groupby("genre")]
    train = pd.concat([tr for tr, _ in parts]).sample(frac=1, random_state=args.seed)
    test = pd.concat([te for _, te in parts]).sample(frac=1, random_state=args.seed)

    cols = ["text", "label", "topic", "source", "model", "genre"]
    out = ROOT / "data"
    train[cols].to_csv(out / "train.csv", index=False)
    test[cols].to_csv(out / "test.csv", index=False)
    if len(unseen):
        unseen[cols].to_csv(out / "test_unseen.csv", index=False)
    elif (out / "test_unseen.csv").exists():
        (out / "test_unseen.csv").unlink()

    for name, part in [("train", train), ("test", test), ("test_unseen", unseen)]:
        if len(part):
            print(f"{name:12s} {len(part):4d} текста  (човек: {(part.label == 0).sum()}, ИИ: {(part.label == 1).sum()})  "
                  + ", ".join(f"{g}: {n}" for g, n in part["genre"].value_counts().items()))


if __name__ == "__main__":
    main()
