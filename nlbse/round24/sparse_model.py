"""Independent token and character experts for the two exact rule outcomes."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import re

import numpy as np
from scipy.sparse import hstack, vstack
from sklearn.feature_extraction.text import HashingVectorizer

HERE = Path(__file__).resolve().parent
TOKEN = r"[A-Za-z_][A-Za-z_0-9]*|[0-9]+|[^\s\w]"
# Character fragments stay within identifier tokens and retain naming and casing cues.
IDENT = re.compile(r"[A-Za-z_][A-Za-z_0-9]{1,31}")
MODES = ("token", "both")
WORDS = HashingVectorizer(n_features=2**17, token_pattern=TOKEN, lowercase=False,
                          ngram_range=(1, 2), norm="l2", dtype=np.float32)
CHARS = HashingVectorizer(n_features=2**17, analyzer="char_wb", lowercase=False,
                          ngram_range=(3, 4), norm="l2", dtype=np.float32)


def fragments(text):
    return " ".join(IDENT.findall(text))


def vectorize(texts, mode):
    tokens = WORDS.transform(texts)
    if mode == "token":
        return tokens
    chars = CHARS.transform(map(fragments, texts))
    return hstack((tokens, chars), format="csr")


def block(texts):
    return vectorize(texts, "both")


def features(texts, workers=1):
    if workers == 1:
        return block(texts)
    parts = [texts[i:i+1024] for i in range(0, len(texts), 1024)]
    with ProcessPoolExecutor(max_workers=workers) as jobs:
        return vstack(list(jobs.map(block, parts, chunksize=2)), format="csr")


def aligned(models, texts, mode):
    matrix = vectorize(texts, mode)
    return predict(models, matrix), matrix.nnz


def predict(models, x):
    return np.column_stack([model.decision_function(x) for model in models]).astype(np.float32)
