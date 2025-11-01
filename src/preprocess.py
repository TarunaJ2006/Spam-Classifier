"""Preprocessing utilities for Spam-Classifier.

Provides simple text cleaning, heuristic feature extraction and TF-IDF vectorization
for both word and character n-grams. Returns sparse feature matrices and fitted
vectorizers so they can be persisted for inference.
"""
from typing import Tuple, Optional
import re
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


URL_RE = re.compile(r"https?://\S+|www\.\S+")


def simple_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # basic normalization
    text = text.strip()
    text = text.replace('\n', ' ').replace('\r', ' ')
    # collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    return text


def heuristic_features(texts: pd.Series) -> np.ndarray:
    """Compute simple heuristic features from raw text.

    Features: word_count, char_count, url_count, uppercase_ratio, digit_ratio,
    punctuation_density
    """
    wc = texts.fillna("").map(lambda t: len(str(t).split()))
    cc = texts.fillna("").map(lambda t: len(str(t)))
    urlc = texts.fillna("").map(lambda t: len(URL_RE.findall(str(t))))
    def upper_ratio(s):
        s = str(s)
        letters = re.findall(r"[A-Za-z]", s)
        if not letters:
            return 0.0
        up = sum(1 for c in letters if c.isupper())
        return up / len(letters)
    ur = texts.fillna("").map(upper_ratio)
    digit_ratio = texts.fillna("").map(lambda s: sum(c.isdigit() for c in str(s)) / max(1, len(str(s))))
    punct_density = texts.fillna("").map(lambda s: sum(1 for c in str(s) if c in "!?,.;:") / max(1, len(str(s))))
    arr = np.vstack([wc, cc, urlc, ur, digit_ratio, punct_density]).T.astype(float)
    return arr


def build_vectorizers(
    max_word_features: int = 15000,
    max_char_features: int = 3000,
    word_ngram: tuple = (1, 2),
    char_ngram: tuple = (3, 5),
) -> Tuple[TfidfVectorizer, TfidfVectorizer]:
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=word_ngram,
        max_features=max_word_features,
        strip_accents="unicode",
        lowercase=True,
    )
    char_vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=char_ngram,
        max_features=max_char_features,
        lowercase=True,
    )
    return word_vec, char_vec


def extract_features(
    df: pd.DataFrame,
    text_col: str = "text",
    word_vec: Optional[TfidfVectorizer] = None,
    char_vec: Optional[TfidfVectorizer] = None,
    fit: bool = True,
) -> Tuple[sparse.spmatrix, Optional[TfidfVectorizer], Optional[TfidfVectorizer]]:
    """Given a dataframe and the text column, return a feature matrix and
    the fitted vectorizers (if fit=True). If vectorizers are provided and
    fit=False they are used to transform.
    """
    texts = df[text_col].fillna("").map(simple_clean)

    if word_vec is None or char_vec is None:
        word_vec, char_vec = build_vectorizers()

    if fit:
        Xw = word_vec.fit_transform(texts)
        Xc = char_vec.fit_transform(texts)
    else:
        Xw = word_vec.transform(texts)
        Xc = char_vec.transform(texts)

    heur = heuristic_features(texts)

    # combine sparse and dense
    X = sparse.hstack([Xw, Xc, sparse.csr_matrix(heur)], format="csr")

    return X, word_vec, char_vec


if __name__ == "__main__":
    print("Preprocess module: functions available -> simple_clean, extract_features")
