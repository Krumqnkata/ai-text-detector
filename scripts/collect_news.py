"""Събира човешки новинарски текстове от българските Уикиновини (жанр "news").

Почти всички статии там са от 2005–2019 г., т.е. отпреди масовите езикови модели.
Заглавието на статията става тема — генераторът на ИИ текстове пише новина
със същото заглавие. Вземат се случайни статии на пакети от по 20.

    python scripts/collect_news.py
    python scripts/collect_news.py --n 120 --contact you@example.com
"""
import argparse
import csv
import re
import time
from pathlib import Path

from collect_human import BATCH, DELAY, extracts

ROOT = Path(__file__).resolve().parent.parent
API = "https://bg.wikinews.org/w/api.php"
SKIP_TITLE = re.compile(r"преглед на печата|кратки новини|новини накратко", re.I)
# Дата-ред в началото („27 декември 2008 г.“) — етикет на жанра, не част от текста.
DATELINE = re.compile(r"^\s*\d{1,2}\s+[а-я]+\s+\d{4}(\s*г\.?)?\s*[-–—.,]?\s*", re.I)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=ROOT / "data" / "human_news.csv", type=Path)
    ap.add_argument("--n", type=int, default=80, help="колко статии да се съберат")
    ap.add_argument("--min-words", type=int, default=80)
    ap.add_argument("--contact", default="")
    args = ap.parse_args()
    user_agent = "ai-text-detector/0.1 (educational project" + (f"; {args.contact})" if args.contact else ")")

    rows, seen, requests = [], set(), 0
    while len(rows) < args.n and requests < 4 * args.n // BATCH + 10:
        pages, _ = extracts(user_agent, api=API, generator="random", grnnamespace=0, grnlimit=BATCH)
        requests += 1
        for title, text in pages.items():
            text = DATELINE.sub("", text)
            if title in seen or SKIP_TITLE.search(title) or len(text.split()) < args.min_words:
                continue
            seen.add(title)
            rows.append({"text": text, "label": 0, "topic": title, "source": f"bg.wikinews: {title}",
                         "model": "human", "genre": "news"})
        print(f"  новини: {len(rows)}", flush=True)
        time.sleep(DELAY)

    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "topic", "source", "model", "genre"])
        writer.writeheader()
        writer.writerows(rows[:args.n])
    print(f"Записани {min(len(rows), args.n)} новини в {args.out}")


if __name__ == "__main__":
    main()
