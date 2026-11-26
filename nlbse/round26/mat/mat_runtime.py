"""Read the canonical MAT marker from the supplied function text."""
import re

import numpy as np

from marker import upstream_tag


SOURCE_COMMENT = re.compile(r"/\*.*?\*/|(?<!:)//[^\n]*", re.S)


def canonical_mat(texts):
    return np.fromiter(
        (any(upstream_tag(comment[0]) for comment in SOURCE_COMMENT.finditer(text)) for text in texts),
        dtype=bool, count=len(texts),
    )
