"""Sparse lexical/stylometric member: char + word TF-IDF -> linear model."""
from __future__ import annotations

import re

import numpy as np
from scipy.sparse import hstack
from scipy.special import log_softmax
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

_WORD = re.compile(r"\w+|[^\w\s]")


def _shape(s: str) -> str:
    """Topic-free surface form: letters -> case class, digits -> 9, punctuation kept."""
    return re.sub(r"[a-z]", "a", re.sub(r"[A-Z]", "A", re.sub(r"\d", "9", s)))


def _funcwords(s: str) -> str:
    """Function words and punctuation kept, content words -> length bucket."""
    return " ".join(w.lower() if w.lower() in ENGLISH_STOP_WORDS or not w[0].isalnum()
                    else f"W{min(len(w), 12)}" for w in _WORD.findall(s))


def _vecs(max_char=300_000, views: str = "base"):
    v = [
        TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), min_df=2, sublinear_tf=True,
                        lowercase=False, max_features=max_char),
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                        lowercase=False, token_pattern=r"(?u)\b\w+\b|[^\w\s]", max_features=300_000),
    ]
    if views == "4view":  # + two topic-free style views (cf. public dhruv nlm-v21 4-view SVC)
        v += [TfidfVectorizer(analyzer="char", ngram_range=(2, 6), min_df=3, sublinear_tf=True,
                              lowercase=False, preprocessor=_shape, max_features=200_000),
              TfidfVectorizer(analyzer="word", ngram_range=(1, 4), min_df=3, sublinear_tf=True,
                              lowercase=False, preprocessor=_funcwords, token_pattern=r"\S+",
                              max_features=200_000)]
    return v


def fit_predict(train_text, y, test_text, C: float = 10.0, seed: int = 0, model: str = "lr",
                sample_weight=None, T: float = 1.0, views: str = "base") -> np.ndarray:
    """Return class probabilities for test_text. ``model='svc'`` uses LinearSVC with
    softmax(decision / T) as a probability proxy (T tuned on OOF log-loss)."""
    vecs = _vecs(views=views)
    Xtr = hstack([v.fit_transform(train_text) for v in vecs]).tocsr()
    Xte = hstack([v.transform(test_text) for v in vecs]).tocsr()
    if model == "svc":
        clf = LinearSVC(C=C, random_state=seed)
        clf.fit(Xtr, y, sample_weight=sample_weight)
        d = clf.decision_function(Xte)
        if d.ndim == 1:  # binary: sklearn returns the positive-class margin only
            d = np.stack([-d, d], 1)
        return np.exp(log_softmax(d / T, axis=1))
    clf = LogisticRegression(C=C, max_iter=3000, random_state=seed)
    clf.fit(Xtr, y, sample_weight=sample_weight)
    return clf.predict_proba(Xte)
