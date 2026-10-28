"""Binoculars (Hans et al., 2024): zero-shot детекция с два сродни езикови модела.

Идеята: колко изненадващ е текстът за един езиков модел (log-перплексити),
разделено на колко изненадващ *би бил* типичният текст на втори, сроден модел в
същите позиции (крос-перплексити). ИИ текстът е предсказуем, но „предсказуем по
очаквания начин“, затова съотношението при него е ниско; при човешкия текст е
по-високо. Не изисква обучение — само два модела с общ токенизатор.

Тук: observer = Qwen2.5-1.5B (base), performer = Qwen2.5-1.5B-Instruct.
Моделите се зареждат при първата нужда; резултатите се кешират на диска,
защото на процесор един текст отнема няколко секунди.
"""
import hashlib
import json
import os
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HF_CACHE = ROOT / "models" / "hf"
SCORE_CACHE = ROOT / "data" / "cache" / "binoculars.json"
OBSERVER = "Qwen/Qwen2.5-1.5B"
PERFORMER = "Qwen/Qwen2.5-1.5B-Instruct"
BINO_FEATURES = ["bino_log_ppl", "bino_x_ppl", "binoculars"]
_LOCK = threading.RLock()


def available():
    """Може ли да се ползва: има torch/transformers и не е изключено с DETECTOR_NO_LM=1."""
    if os.environ.get("DETECTOR_NO_LM") == "1":
        return False
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


class Binoculars:
    def __init__(self, observer=OBSERVER, performer=PERFORMER, max_tokens=512):
        self.observer = observer
        self.performer = performer
        self.max_tokens = max_tokens
        self._models = None
        self._cache = None

    # Моделите и кешът не се записват в joblib артефакта — само имената им.
    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k not in ("_models", "_cache")}

    def __setstate__(self, state):
        self.__dict__.update(state, _models=None, _cache=None)

    def _load(self):
        with _LOCK:
            return self._load_locked()

    def _load_locked(self):
        if self._models is None:
            import torch
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer

            transformers.utils.logging.set_verbosity_error()
            transformers.utils.logging.disable_progress_bar()
            tok = AutoTokenizer.from_pretrained(self.observer, cache_dir=HF_CACHE)
            # float32: на процесори без bf16 инструкции bfloat16 е по-бавен (мерено: ~14 s срещу ~8.5 s).
            load = lambda name: AutoModelForCausalLM.from_pretrained(  # noqa: E731
                name, cache_dir=HF_CACHE, dtype=torch.float32).eval()
            self._models = (tok, load(self.observer), load(self.performer))
        return self._models

    def _key(self, text):
        raw = f"{self.observer}|{self.performer}|{self.max_tokens}|{text}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _load_cache(self):
        if self._cache is None:
            self._cache = json.loads(SCORE_CACHE.read_text(encoding="utf-8")) if SCORE_CACHE.exists() else {}
        return self._cache

    def _save_cache(self):
        SCORE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        SCORE_CACHE.write_text(json.dumps(self._cache), encoding="utf-8")

    def score(self, text):
        """[log-перплексити под performer, крос-перплексити observer→performer, съотношение]."""
        import torch
        import torch.nn.functional as F

        tok, observer, performer = self._load()
        ids = tok(text, return_tensors="pt", truncation=True, max_length=self.max_tokens).input_ids
        if ids.shape[1] < 2:
            return [float("nan")] * 3
        # По една заявка наведнъж: паралелни прогони на процесора се бавят взаимно многократно.
        # inference_mode тук, а не глобално: grad режимът в torch е за всяка нишка поотделно,
        # а сървърът обработва заявките в различни нишки.
        with _LOCK, torch.inference_mode():
            obs_logits = observer(ids).logits[0, :-1].float()
            perf_logits = performer(ids).logits[0, :-1].float()
            targets = ids[0, 1:]
            perf_logp = F.log_softmax(perf_logits, dim=-1)
            # Колко изненадващ е реалният текст за performer (средна крос-ентропия на токен).
            log_ppl = float(-perf_logp.gather(1, targets[:, None]).mean())
            # Колко изненадващо би било за performer това, което observer очаква на всяка позиция.
            x_ppl = float(-(F.softmax(obs_logits, dim=-1) * perf_logp).sum(-1).mean())
        return [log_ppl, x_ppl, log_ppl / x_ppl]

    def transform(self, texts, verbose=False):
        cache = self._load_cache()
        out, dirty = [], False
        texts = list(texts)
        for i, text in enumerate(texts):
            key = self._key(text)
            if key not in cache:
                cache[key] = self.score(text)
                dirty = True
                if verbose:
                    print(f"  Binoculars {i + 1}/{len(texts)}", flush=True)
                if dirty and (i + 1) % 10 == 0:
                    self._save_cache()
            out.append(cache[key])
        if dirty:
            self._save_cache()
        return np.array(out, dtype=float)
