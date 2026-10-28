"""Събира човешки текстове от българската Уикипедия.

За всяка тема от data/topics_bg.txt взима въведението на статията (plain text
през TextExtracts). Заявките са пакетни — до 20 статии на заявка — защото
Уикипедия бързо ограничава (HTTP 429) клиенти, които пращат заявка на статия.
Допълнително сваля въведенията на случайни статии като референтен корпус за
езиковия модел (перплексити признака).

    python scripts/collect_human.py
    python scripts/collect_human.py --reference 300 --contact you@example.com
"""
import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = "https://bg.wikipedia.org/w/api.php"
BATCH = 20   # максимумът на TextExtracts за exintro заявки
DELAY = 2.0  # секунди между заявките


def api_get(user_agent, retries=5, **params):
    params = {"format": "json", "formatversion": "2", "maxlag": "5", **params}
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 503) or attempt == retries - 1:
                raise
            wait = int(exc.headers.get("Retry-After") or 0) or 5 * 2 ** attempt
            print(f"  … HTTP {exc.code}, изчакване {wait} s", file=sys.stderr, flush=True)
            time.sleep(wait)


def clean(text):
    text = re.sub(r"\[[^\]]{0,20}\]", "", text)   # остатъци от бележки
    text = re.sub(r"\(\s*[,;]?\s*\)", "", text)     # празни скоби след махнато произношение
    paras = [re.sub(r"[ \t]+", " ", p).strip() for p in text.split("\n")]
    return "\n".join(p for p in paras if len(p.split()) >= 8)


def extracts(user_agent, **params):
    """Една пакетна заявка → {каноничното заглавие: текст}; плюс картата на пренасочванията."""
    data = api_get(user_agent, action="query", prop="extracts", exintro=1, explaintext=1,
                   exlimit=BATCH, redirects=1, **params)
    query = data.get("query", {})
    redirects = {r["from"]: r["to"] for r in query.get("redirects", []) + query.get("normalized", [])}
    pages = {p["title"]: clean(p.get("extract", "")) for p in query.get("pages", []) if "missing" not in p}
    return pages, redirects


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topics", default=ROOT / "data" / "topics_bg.txt", type=Path)
    ap.add_argument("--out", default=ROOT / "data" / "human.csv", type=Path)
    ap.add_argument("--reference", type=int, default=200, help="брой случайни статии за референтния корпус")
    ap.add_argument("--reference-out", default=ROOT / "data" / "reference.txt", type=Path)
    ap.add_argument("--min-words", type=int, default=60)
    ap.add_argument("--contact", default="",
                    help="имейл или URL за User-Agent (политиката на Уикимедия го препоръчва)")
    args = ap.parse_args()
    user_agent = "ai-text-detector/0.1 (educational project" + (f"; {args.contact})" if args.contact else ")")

    topics = [line.strip() for line in args.topics.read_text(encoding="utf-8").splitlines()
              if line.strip() and not line.startswith("#")]

    rows = []
    for i in range(0, len(topics), BATCH):
        chunk = topics[i:i + BATCH]
        pages, redirects = extracts(user_agent, titles="|".join(chunk))
        for topic in chunk:
            title = topic
            while title in redirects:
                title = redirects[title]
            text = pages.get(title, "")
            if len(text.split()) < args.min_words:
                print(f"  - {topic}: твърде кратко въведение ({len(text.split())} думи)")
                continue
            rows.append({"text": text, "label": 0, "topic": topic,
                         "source": f"bg.wikipedia: {title}", "model": "human"})
            print(f"  + {topic} ({len(text.split())} думи)")
        time.sleep(DELAY)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "topic", "source", "model"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Записани {len(rows)} човешки текста в {args.out}")

    if args.reference:
        texts, seen, requests = [], set(topics), 0
        while len(texts) < args.reference and requests < args.reference:  # стопер срещу безкраен цикъл
            pages, _ = extracts(user_agent, generator="random", grnnamespace=0, grnlimit=BATCH)
            requests += 1
            for title, text in pages.items():
                if title not in seen and len(text.split()) >= 40:
                    seen.add(title)
                    texts.append(text)
            print(f"  референтни: {len(texts)}", flush=True)
            time.sleep(DELAY)
        args.reference_out.write_text("\n\n".join(texts[:args.reference]), encoding="utf-8")
        print(f"Записани {min(len(texts), args.reference)} референтни текста в {args.reference_out}")


if __name__ == "__main__":
    main()
