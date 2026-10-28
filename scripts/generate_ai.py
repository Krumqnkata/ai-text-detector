"""Генерира ИИ текстове на същите теми като човешките (data/human.csv).

Нужен е ANTHROPIC_API_KEY. Всеки модел пише по един текст на тема с дължина,
близка до тази на човешкия текст, за да не стане дължината издайнически признак.
Скриптът е възобновяем: пропуска двойките (тема, модел), които вече ги има.

    python scripts/generate_ai.py
    python scripts/generate_ai.py --models claude-haiku-5-5 --limit 20

Резервни модели (server-side fallbacks) нарочно НЕ са включени: ако един модел
откаже и друг довърши, етикетът `model` в данните щеше да е грешен.
"""
import argparse
import csv
import random
import sys
from pathlib import Path

import anthropic
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
FIELDS = ["text", "label", "topic", "source", "model"]

PROMPTS = [
    "Напиши енциклопедичен текст на български език на тема „{topic}“, около {words} думи. "
    "Без заглавие, без списъци и без форматиране — само обикновени абзаци.",
    "Обясни накратко на български какво трябва да знае човек за „{topic}“, около {words} думи, "
    "в стила на статия от енциклопедия. Само текст в абзаци, без заглавие и списъци.",
]


def generate(client, model, topic, words):
    prompt = random.choice(PROMPTS).format(topic=topic, words=words)
    response = client.messages.create(
        model=model,
        max_tokens=4000,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": prompt}],
    )
    if response.stop_reason == "refusal":
        return None
    return "\n".join(b.text for b in response.content if b.type == "text").strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--human", default=ROOT / "data" / "human.csv", type=Path)
    ap.add_argument("--out", default=ROOT / "data" / "ai_generated.csv", type=Path)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--limit", type=int, default=None, help="макс. брой теми на модел")
    ap.add_argument("--max-words", type=int, default=220, help="горна граница за поисканата дължина")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    human = pd.read_csv(args.human)
    topics = human[["topic", "text"]].drop_duplicates("topic")
    if args.limit:
        topics = topics.sample(min(args.limit, len(topics)), random_state=args.seed)

    done = set()
    if args.out.exists():
        prev = pd.read_csv(args.out)
        done = set(zip(prev["topic"], prev["model"]))
    new_file = not args.out.exists()

    client = anthropic.Anthropic()
    with args.out.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for model in args.models:
            for topic, human_text in topics.itertuples(index=False):
                if (topic, model) in done:
                    continue
                words = min(args.max_words, max(80, len(human_text.split())))
                words = int(round(words * random.uniform(0.8, 1.1), -1))
                try:
                    text = generate(client, model, topic, words)
                except anthropic.NotFoundError:
                    sys.exit(f"Моделът {model} не е намерен.")
                except anthropic.APIStatusError as exc:
                    print(f"  ! {model} / {topic}: {exc.status_code} {exc.message}", file=sys.stderr)
                    continue
                if not text:
                    print(f"  - {model} / {topic}: отказ или празен отговор")
                    continue
                writer.writerow({"text": text, "label": 1, "topic": topic,
                                 "source": "anthropic-api", "model": model})
                f.flush()
                print(f"  + {model} / {topic} ({len(text.split())} думи)")


if __name__ == "__main__":
    main()
