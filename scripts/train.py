"""Treina o classificador (TF-IDF + Regressão Logística) e exporta o modelo para o site rodar no navegador."""
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from common import STOP_WORDS, tokens

ROOT = Path(__file__).resolve().parent.parent
MIN_PER_TOPIC = 20


def load():
    with (ROOT / "data" / "headlines.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    counts = Counter(r["topic"] for r in rows)
    rows = [r for r in rows if counts[r["topic"]] >= MIN_PER_TOPIC]
    return [r["title"] for r in rows], [r["topic"] for r in rows]


def vectorizer():
    return TfidfVectorizer(analyzer=tokens, min_df=2, sublinear_tf=True, max_features=6000)


def main():
    X, y = load()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    vec = vectorizer()
    clf = LogisticRegression(max_iter=2000, C=8)
    clf.fit(vec.fit_transform(X_tr), y_tr)
    pred = clf.predict(vec.transform(X_te))
    labels = list(clf.classes_)
    report = classification_report(y_te, pred, labels=labels, output_dict=True, zero_division=0)

    # modelo final usa todos os dados
    vec = vectorizer()
    clf = LogisticRegression(max_iter=2000, C=8)
    clf.fit(vec.fit_transform(X), y)

    vocab = vec.get_feature_names_out().tolist()
    coef = clf.coef_
    top_words = {}
    for i, c in enumerate(clf.classes_):
        idx = np.argsort(coef[i])[::-1]
        top_words[c] = [vocab[j] for j in idx if " " not in vocab[j]][:8]

    model = {
        "classes": clf.classes_.tolist(),
        "vocab": vocab,
        "idf": [round(float(v), 4) for v in vec.idf_],
        "coef": [[round(float(v), 4) for v in row] for row in coef],
        "intercept": [round(float(v), 4) for v in clf.intercept_],
        "stop_words": STOP_WORDS,
    }
    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_headlines": len(X),
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "vocab_size": len(vocab),
        "accuracy": round(accuracy_score(y_te, pred), 4),
        "per_topic": {c: {"f1": round(report[c]["f1-score"], 3), "n": int(report[c]["support"])} for c in labels},
        "per_topic_total": dict(Counter(y)),
        "labels": labels,
        "confusion": confusion_matrix(y_te, pred, labels=labels).tolist(),
        "top_words": top_words,
    }
    (ROOT / "model.json").write_text(json.dumps(model, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (ROOT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(X)} manchetes · acurácia {metrics['accuracy']:.1%} · vocabulário {len(vocab)}")


if __name__ == "__main__":
    main()
