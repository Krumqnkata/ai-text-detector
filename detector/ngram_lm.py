"""Малък символен n-грамен езиков модел (Witten–Bell) за „перплексити“.

Истински невронен езиков модел би бил по-точен, но изисква PyTorch и голям
модел. Символният n-грамен модел, обучен върху референтен корпус от човешки
текстове, дава грубо, но обяснимо мерило колко „предсказуем“ е даден текст.
"""
import math
import re
from collections import Counter, defaultdict


def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


class CharNgramLM:
    def __init__(self, order=4):
        self.order = order

    def fit(self, texts):
        self.counts = [Counter() for _ in range(self.order + 1)]      # n-грам -> брой
        self.ctx = [Counter() for _ in range(self.order + 1)]         # контекст -> брой
        followers = [defaultdict(set) for _ in range(self.order + 1)]
        for text in texts:
            s = " " * (self.order - 1) + normalize(text)
            for i in range(self.order - 1, len(s)):
                for n in range(1, self.order + 1):
                    gram = s[i - n + 1:i + 1]
                    self.counts[n][gram] += 1
                    self.ctx[n][gram[:-1]] += 1
                    followers[n][gram[:-1]].add(gram[-1])
        self.types = [{h: len(f) for h, f in fol.items()} for fol in followers]
        self.vocab = len(self.counts[1]) + 1  # +1 за непознат символ
        self.total = sum(self.counts[1].values())
        return self

    def prob(self, history, char):
        p = (self.counts[1][char] + 1) / (self.total + self.vocab)
        for n in range(2, self.order + 1):
            h = history[-(n - 1):]
            c_h = self.ctx[n].get(h, 0)
            if not c_h:
                break
            t_h = self.types[n][h]
            p = (self.counts[n].get(h + char, 0) + t_h * p) / (c_h + t_h)
        return p

    def bits_per_char(self, text):
        s = " " * (self.order - 1) + normalize(text)
        n = len(s) - (self.order - 1)
        if n <= 0:
            return float("nan")
        total = sum(-math.log2(self.prob(s[i - self.order + 1:i], s[i]))
                    for i in range(self.order - 1, len(s)))
        return total / n
