"""Recover task markers with punctuation-normalized candidate words."""
import re

import numpy as np

from comment_spans import LEXEME

LETTERS = re.compile(r"[^a-zA-Z\s]")
LICENSE = re.compile(r"license|copyright|all rights reserved", re.I)
TAGS = ("todo", "fixme", "xxx", "hack")


def upstream_tag(comment):
    if LICENSE.search(comment):
        return False
    for word in LETTERS.sub("", comment).lower().split():
        if 2 < len(word) < 20 and (word.startswith(TAGS) or word.endswith(TAGS)):
            return "xxx" not in word or word == "xxx"
    return False


def supplemental_marker(text):
    if "//" not in text and "/*" not in text:
        return False
    return any(upstream_tag(span[0]) for span in LEXEME.finditer(text)
               if span[0].startswith(("//", "/*")))


def recover(texts, base):
    return np.fromiter((bool(tagged) or supplemental_marker(text)
                        for text, tagged in zip(texts, base, strict=True)),
                       dtype=bool, count=len(texts))
