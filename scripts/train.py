"""Обучава и оценява моделите (Стъпки 3–5) и записва models/detector.joblib.

Модели:
  features_lr           — Logistic Regression върху стилометричните признаци (основен, обясним)
  features_lr_no_genre  — същото без „жанровите“ признаци (скоби, числа, кавички)
  features_rf           — Random Forest върху признаците
  tfidf_lr              — TF-IDF (символни n-грами) + Logistic Regression върху суровия текст (baseline)

Освен метриките прави оценка по жанрове и между жанрове (обучение на единия,
тест на другия), аблация по групи признаци, калибриране на вероятностите
(Platt scaling върху out-of-fold логитите) и крива на ученето. Резултатите отиват
в reports/report.md, reports/metrics.json и reports/learning_curve.svg.

    python scripts/train.py
"""
import json
import sys
from pathlib import Path

from itertools import permutations

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (StratifiedGroupKFold, cross_val_predict, cross_validate,
                                     learning_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from detector.features import (FEATURE_GROUPS, FEATURE_NAMES, FEATURES, GENRE_FEATURES,  # noqa: E402
                               ColumnSelector, FeatureExtractor)
from detector.model import Detector  # noqa: E402
from detector.ngram_lm import CharNgramLM  # noqa: E402

DATA, REPORTS, MODELS = ROOT / "data", ROOT / "reports", ROOT / "models"
MODEL_LABELS = {
    "features_lr": "Признаци + LR",
    "features_lr_no_genre": "Признаци без жанр + LR",
    "features_rf": "Признаци + RF",
    "tfidf_lr": "TF-IDF + LR",
}


def feature_lr(keep=None):
    steps = [("select", ColumnSelector(keep))] if keep else []
    return Pipeline(steps + [("scale", StandardScaler()),
                             ("clf", LogisticRegression(C=0.5, max_iter=2000))])


def build_models(seed=42):
    return {
        "features_lr": (feature_lr(), "features"),
        "features_lr_no_genre": (feature_lr([n for n in FEATURE_NAMES if n not in GENRE_FEATURES]),
                                 "features"),
        "features_rf": (RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                               random_state=seed), "features"),
        "tfidf_lr": (Pipeline([("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                                         sublinear_tf=True, min_df=2)),
                               ("clf", LogisticRegression(C=4.0, max_iter=2000))]), "text"),
    }


def scores(y, proba):
    pred = (proba >= 0.5).astype(int)
    out = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }
    if len(set(y)) == 2:
        out["roc_auc"] = roc_auc_score(y, proba)
    return out


def cross_genre(models, X_all, df_all):
    """Обучение само върху жанр A, тест върху целия жанр B — пренася ли се наученото?"""
    out = {}
    for src, dst in permutations(sorted(df_all.genre.unique()), 2):
        a, b = (df_all.genre == src).values, (df_all.genre == dst).values
        for name in ("features_lr", "features_lr_no_genre", "tfidf_lr"):
            model, kind = models[name]
            data = X_all if kind == "features" else np.array(df_all.text.tolist(), dtype=object)
            fitted = clone(model).fit(data[a], df_all.label.values[a])
            res = scores(df_all.label.values[b], fitted.predict_proba(data[b])[:, 1])
            out.setdefault(f"{src} → {dst}", {})[name] = res
    return out


def ablation(X, train, cv):
    """CV на LR без всяка група признаци и само с нея."""
    def cv_scores(keep):
        res = cross_validate(feature_lr(keep), X, train.label, groups=train.topic, cv=cv,
                             scoring=["f1", "roc_auc"])
        return {"f1": float(res["test_f1"].mean()), "roc_auc": float(res["test_roc_auc"].mean())}

    out = {"Всички признаци": {"without": cv_scores(None), "only": None}}
    for group, names in FEATURE_GROUPS.items():
        out[group] = {"without": cv_scores([n for n in FEATURE_NAMES if n not in names]),
                      "only": cv_scores(names)}
    return out


def ece(y, proba, bins=5):
    """Expected calibration error: средната разлика между увереност и реална честота по кошчета."""
    idx = np.clip((proba * bins).astype(int), 0, bins - 1)
    return float(sum(abs(proba[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def calibrate(model, X, train, test, cv):
    """Platt scaling: P = σ(a·логит + b), напаснато върху out-of-fold логитите на train.

    Out-of-fold, защото върху собствените си обучаващи данни моделът винаги
    изглежда по-уверен, отколкото е. Линейният модел остава непроменен, така че
    обяснението (приносите към логита) важи и за калибрираната вероятност.
    """
    oof = cross_val_predict(model, X["train"], train.label, groups=train.topic, cv=cv,
                            method="decision_function")
    platt = LogisticRegression(C=1e4).fit(oof.reshape(-1, 1), train.label)
    a, b = float(platt.coef_[0, 0]), float(platt.intercept_[0])
    logits = model.decision_function(X["test"])
    y = test.label.values
    report = {"a": a, "b": b}
    for name, p in [("raw", 1 / (1 + np.exp(-logits))), ("calibrated", 1 / (1 + np.exp(-(a * logits + b))))]:
        report[name] = {"brier": float(brier_score_loss(y, p)), "ece": ece(y, p),
                        "mean_confidence": float(np.mean(np.maximum(p, 1 - p))),
                        "accuracy": float(accuracy_score(y, (p >= 0.5).astype(int)))}
    return (a, b), report


def curve(models, X, train, cv):
    out = {}
    for name in ("features_lr", "features_rf", "tfidf_lr"):
        model, kind = models[name]
        data = X["train"] if kind == "features" else np.array(train.text.tolist(), dtype=object)
        sizes, _, test_scores = learning_curve(model, data, train.label, groups=train.topic, cv=cv,
                                               train_sizes=np.linspace(0.25, 1.0, 5), scoring="f1",
                                               shuffle=True, random_state=42)
        out[name] = {"sizes": sizes.tolist(), "f1": test_scores.mean(axis=1).tolist(),
                     "f1_std": test_scores.std(axis=1).tolist()}
    return out


def write_learning_curve_svg(lc, path):
    """Линейна графика на F1 спрямо броя обучаващи текстове (без външни библиотеки)."""
    W, H = 680, 340
    L, R, T, B = 52, 190, 52, 46
    series = [("features_lr", "--s1"), ("features_rf", "--s2"), ("tfidf_lr", "--s3")]
    sizes = lc["features_lr"]["sizes"]
    lowest = min(v for name, _ in series for v in lc[name]["f1"])
    x0, x1, y0, y1 = min(sizes), max(sizes), min(0.5, np.floor(lowest * 10) / 10), 1.0
    ticks = np.round(np.arange(y0, y1 + 0.001, 0.1), 1)

    def x(v):
        return L + (v - x0) / ((x1 - x0) or 1) * (W - L - R)

    def y(v):
        return T + (1 - (min(max(v, y0), y1) - y0) / (y1 - y0)) * (H - T - B)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
           'font-family="system-ui, Segoe UI, sans-serif" role="img" '
           'aria-label="Крива на ученето: F1 при крос-валидация спрямо броя обучаващи текстове">',
           "<style>svg{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--grid:#e3e2de;"
           "--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a}"
           "@media (prefers-color-scheme: dark){svg{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;"
           "--grid:#3a3a37;--s1:#3987e5;--s2:#d95926;--s3:#199e70}}</style>",
           f'<rect width="{W}" height="{H}" fill="var(--bg)"/>',
           f'<text x="{L}" y="24" font-size="15" font-weight="600" fill="var(--ink)">Крива на ученето</text>',
           f'<text x="{L}" y="40" font-size="12" fill="var(--ink2)">F1 при 5-кратна крос-валидация по теми</text>']
    for v in ticks:
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="var(--grid)"/>')
        out.append(f'<text x="{L - 8}" y="{y(v) + 4:.1f}" font-size="11" text-anchor="end" '
                   f'fill="var(--ink2)">{v:.1f}</text>')
    for v in sizes:
        out.append(f'<text x="{x(v):.1f}" y="{H - B + 18}" font-size="11" text-anchor="middle" '
                   f'fill="var(--ink2)">{v}</text>')
    out.append(f'<text x="{(L + W - R) / 2:.1f}" y="{H - 8}" font-size="12" text-anchor="middle" '
               'fill="var(--ink2)">брой обучаващи текстове</text>')
    label_ys = []
    for name, color in series:
        pts = list(zip(lc[name]["sizes"], lc[name]["f1"]))
        line = " ".join(f"{x(a):.1f},{y(b):.1f}" for a, b in pts)
        out.append(f'<polyline points="{line}" fill="none" stroke="var({color})" stroke-width="2" '
                   'stroke-linejoin="round"/>')
        for a, b in pts:
            out.append(f'<circle cx="{x(a):.1f}" cy="{y(b):.1f}" r="4.5" fill="var({color})" '
                       f'stroke="var(--bg)" stroke-width="2"><title>{MODEL_LABELS[name]}: {a} текста, '
                       f'F1 = {b:.3f}</title></circle>')
        # Директен етикет в края на линията; разминаваме етикетите, ако се застъпват.
        ly = y(pts[-1][1])
        while any(abs(ly - other) < 15 for other in label_ys):
            ly += 15
        label_ys.append(ly)
        out.append(f'<circle cx="{W - R + 16}" cy="{ly:.1f}" r="4.5" fill="var({color})"/>')
        out.append(f'<text x="{W - R + 26}" y="{ly + 4:.1f}" font-size="12" fill="var(--ink)">'
                   f'{MODEL_LABELS[name]} · {pts[-1][1]:.2f}</text>')
    out.append("</svg>")
    path.write_text("\n".join(out), encoding="utf-8")


def main():
    REPORTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    train, test = pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv")
    unseen_path = DATA / "test_unseen.csv"
    unseen = pd.read_csv(unseen_path) if unseen_path.exists() else None

    print("Обучение на n-грамния езиков модел върху референтния корпус…")
    reference = (DATA / "reference.txt").read_text(encoding="utf-8").split("\n\n")
    extractor = FeatureExtractor(CharNgramLM(order=4).fit(reference))

    print("Извличане на признаци…")
    X = {"train": extractor.transform(train.text), "test": extractor.transform(test.text)}
    if unseen is not None:
        X["unseen"] = extractor.transform(unseen.text)
    inputs = lambda kind, split, df: X[split] if kind == "features" else df.text.tolist()  # noqa: E731

    models = build_models()
    metrics = {"data": {"train": len(train), "test": len(test),
                        "unseen": 0 if unseen is None else len(unseen),
                        "genres": sorted(train.genre.unique().tolist()),
                        "train_models": sorted(train[train.label == 1].model.unique().tolist()),
                        "unseen_models": [] if unseen is None else sorted(unseen.model.unique().tolist())}}
    test_proba = {}
    n_splits = min(5, train.topic.nunique())
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for name, (model, kind) in models.items():
        cv_res = cross_validate(model, inputs(kind, "train", train), train.label, groups=train.topic,
                                cv=cv, scoring=["accuracy", "f1", "roc_auc"])
        model.fit(inputs(kind, "train", train), train.label)
        test_proba[name] = model.predict_proba(inputs(kind, "test", test))[:, 1]
        m = {"cv": {k[5:]: float(np.mean(v)) for k, v in cv_res.items() if k.startswith("test_")},
             "test": scores(test.label.values, test_proba[name]),
             "test_by_genre": {g: scores(test.label.values[mask], test_proba[name][mask])
                               for g in sorted(test.genre.unique())
                               for mask in [(test.genre == g).values]}}
        if unseen is not None:
            p = model.predict_proba(inputs(kind, "unseen", unseen))[:, 1]
            m["unseen"] = scores(unseen.label.values, p)
        metrics[name] = m
        print(f"  {name:21s} CV F1={m['cv']['f1']:.3f}  test F1={m['test']['f1']:.3f}  "
              f"test acc={m['test']['accuracy']:.3f}")

    if train.genre.nunique() > 1:
        print("Оценка между жанрове…")
        both = pd.concat([train, test], ignore_index=True)
        metrics["cross_genre"] = cross_genre(models, np.vstack([X["train"], X["test"]]), both)

    print("Аблация по групи признаци…")
    metrics["ablation"] = ablation(X["train"], train, cv)
    print("Калибриране…")
    calibrator, metrics["calibration"] = calibrate(models["features_lr"][0], X, train, test, cv)
    print("Крива на ученето…")
    metrics["learning_curve"] = curve(models, X, train, cv)
    write_learning_curve_svg(metrics["learning_curve"], REPORTS / "learning_curve.svg")

    stats = {}
    for i, name in enumerate(FEATURE_NAMES):
        col = X["train"][:, i]
        human, ai = col[train.label.values == 0], col[train.label.values == 1]
        stats[name] = {"human_mean": float(human.mean()), "ai_mean": float(ai.mean()),
                       "human_std": float(human.std())}

    bundle = {"extractor": extractor, "models": models, "main_model": "features_lr",
              "feature_stats": stats, "metrics": metrics, "calibrator": calibrator}
    joblib.dump(bundle, MODELS / "detector.joblib", compress=3)
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(Detector(bundle), metrics, test, test_proba, stats)
    print(f"Моделът е записан в {MODELS / 'detector.joblib'}; отчетът — в {REPORTS / 'report.md'}")


def write_report(detector, metrics, test, test_proba, stats):
    lines = ["# Отчет от обучението", ""]
    d = metrics["data"]
    lines += [f"Train: {d['train']} текста, test: {d['test']}, unseen: {d['unseen']}.  ",
              f"Жанрове: {', '.join(d['genres'])}.  ",
              f"ИИ модели в обучението: {', '.join(d['train_models'])}.  ",
              f"Невиждани модели: {', '.join(d['unseen_models']) or '—'}.", ""]

    lines += ["## Сравнение на моделите", "",
              "| Модел | CV F1 | Test accuracy | Precision | Recall | F1 | ROC AUC | Unseen recall |",
              "|---|---|---|---|---|---|---|---|"]
    for name in MODEL_LABELS:
        m, t = metrics[name], metrics[name]["test"]
        unseen = f"{m['unseen']['recall']:.3f}" if "unseen" in m else "—"
        lines.append(f"| {MODEL_LABELS[name]} | {m['cv']['f1']:.3f} | {t['accuracy']:.3f} | "
                     f"{t['precision']:.3f} | {t['recall']:.3f} | {t['f1']:.3f} | "
                     f"{t.get('roc_auc', float('nan')):.3f} | {unseen} |")
    lines += ["", "Unseen recall = какъв дял от текстовете на невидян модел са разпознати като ИИ.", ""]

    lines += ["## По жанрове (test)", "", "| Модел | Жанр | Текстове | Accuracy | F1 | ROC AUC |",
              "|---|---|---|---|---|---|"]
    for name in MODEL_LABELS:
        for g, t in metrics[name]["test_by_genre"].items():
            n = sum(sum(row) for row in t["confusion_matrix"])
            lines.append(f"| {MODEL_LABELS[name]} | {g} | {n} | {t['accuracy']:.3f} | {t['f1']:.3f} | "
                         f"{t.get('roc_auc', float('nan')):.3f} |")
    lines.append("")

    if "cross_genre" in metrics:
        lines += ["## Между жанрове", "",
                  "Обучение само върху единия жанр (train + test), тест върху целия друг жанр. "
                  "Показва дали моделът е научил стил на ИИ, или особеностите на един жанр.", "",
                  "| Посока | Модел | Accuracy | Precision | Recall | F1 | ROC AUC |", "|---|---|---|---|---|---|---|"]
        for direction, res in metrics["cross_genre"].items():
            for name, t in res.items():
                lines.append(f"| {direction} | {MODEL_LABELS[name]} | {t['accuracy']:.3f} | {t['precision']:.3f} | "
                             f"{t['recall']:.3f} | {t['f1']:.3f} | {t.get('roc_auc', float('nan')):.3f} |")
        lines.append("")

    lines += ["## Аблация по групи признаци", "",
              "Logistic Regression, 5-кратна крос-валидация по теми върху train. "
              "„Без групата“ показва колко губим, ако я махнем; „само групата“ — колко носи сама.", "",
              "| Група | CV F1 без групата | CV AUC без групата | CV F1 само с групата | CV AUC само с групата |",
              "|---|---|---|---|---|"]
    for group, r in metrics["ablation"].items():
        only = r["only"]
        lines.append(f"| {group} | {r['without']['f1']:.3f} | {r['without']['roc_auc']:.3f} | "
                     f"{only['f1'] if only else float('nan'):.3f} | "
                     f"{only['roc_auc'] if only else float('nan'):.3f} |".replace("nan", "—"))
    lines.append("")

    c = metrics["calibration"]
    lines += ["## Калибриране на вероятностите (features_lr, test)", "",
              f"Platt scaling върху out-of-fold логитите: P = σ({c['a']:.3f} · логит {c['b']:+.3f}). "
              "По-нисък Brier/ECE = вероятностите по-добре отговарят на реалната честота на грешките.", "",
              "| | Brier | ECE | Средна увереност | Accuracy |", "|---|---|---|---|---|"]
    for name, label in [("raw", "Некалибрирана"), ("calibrated", "Калибрирана")]:
        r = c[name]
        lines.append(f"| {label} | {r['brier']:.3f} | {r['ece']:.3f} | {r['mean_confidence']:.3f} | "
                     f"{r['accuracy']:.3f} |")
    lines.append("")

    lines += ["## Крива на ученето", "", "![Крива на ученето](learning_curve.svg)", "",
              "| Обучаващи текстове | " + " | ".join(MODEL_LABELS[n] for n in metrics["learning_curve"]) + " |",
              "|---" * (1 + len(metrics["learning_curve"])) + "|"]
    lc = metrics["learning_curve"]
    first = next(iter(lc.values()))
    for i, size in enumerate(first["sizes"]):
        lines.append(f"| {size} | " + " | ".join(f"{lc[n]['f1'][i]:.3f} ± {lc[n]['f1_std'][i]:.3f}"
                                                for n in lc) + " |")
    lines.append("")

    lines += ["## Матрици на объркване (test)", ""]
    for name in test_proba:
        (tn, fp), (fn, tp) = metrics[name]["test"]["confusion_matrix"]
        lines += [f"**{MODEL_LABELS[name]}**", "", "| | предсказан човек | предсказан ИИ |", "|---|---|---|",
                  f"| истински човек | {tn} | {fp} |", f"| истински ИИ | {fn} | {tp} |", ""]

    clf = detector.models["features_lr"][0].named_steps["clf"]
    rf = detector.models["features_rf"][0]
    order = np.argsort(-np.abs(clf.coef_[0]))
    lines += ["## Тежести на признаците", "",
              "Положителна тежест → натежава към ИИ, отрицателна → към човек "
              "(признаците са стандартизирани, така че тежестите са сравними).", "",
              "| Признак | Тежест (LR) | Важност (RF) | Средно човек | Средно ИИ |", "|---|---|---|---|---|"]
    for i in order:
        name = FEATURE_NAMES[i]
        lines.append(f"| {FEATURES[name][0]} | {clf.coef_[0][i]:+.3f} | {rf.feature_importances_[i]:.3f} | "
                     f"{stats[name]['human_mean']:.3f} | {stats[name]['ai_mean']:.3f} |")

    lines += ["", "## Анализ на грешките (features_lr, test)", ""]
    proba = test_proba["features_lr"]
    wrong_mask = (proba >= 0.5).astype(int) != test.label.values
    wrong = test[wrong_mask].assign(p=proba[wrong_mask])
    if wrong.empty:
        lines.append("Няма грешно класифицирани текстове в test.")
    for row in wrong.itertuples():
        truth = "ИИ" if row.label else "човек"
        result = detector.predict(row.text, top_k=3)
        lines += [f"### {row.topic} — {row.genre}, истински: {truth} ({row.model}), P(ИИ) = {row.p:.2f}", "",
                  "> " + row.text[:400].replace("\n", " ") + ("…" if len(row.text) > 400 else ""), "",
                  "Най-силни признаци:", ""]
        lines += [f"- {'→ ИИ' if f['toward'] == 'ai' else '→ човек'}: {f['label']} — {f['text']}"
                  for f in result["top_features"]]
        lines.append("")
    (REPORTS / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
