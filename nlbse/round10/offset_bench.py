"""Check lazy tokenizer offsets against materializing every token span."""
from statistics import median
import time

import numpy as np
from transformers import AutoTokenizer

from student_context import CAP, FAMILY, FAMILIES, HERE, PUBLIC, ROOT, feature_block, load, write_json


def main():
    texts, _, _ = load(PUBLIC)
    output = {}
    for family in (FAMILY, "unixcoder9"):
        tk = AutoTokenizer.from_pretrained(ROOT / "models" / FAMILIES[family], local_files_only=True)
        reference = []
        samples = {"all": [], "deferred": []}
        for attempt in range(2):
            for mode in samples if attempt == 0 else reversed(samples):
                start = time.perf_counter()
                for i in range(0, len(texts), 2048):
                    result = feature_block(texts[i:i+2048], tk, family=family, deferred_offsets=(mode == "deferred"))
                    if attempt == 0 and mode == "all":
                        reference.append(result)
                    else:
                        ids, tags, extra = reference[i//2048]
                        if ids != result[0] or not np.array_equal(tags, result[1]) or not np.array_equal(extra, result[2]):
                            raise ValueError(f"Changing how offsets are read changed inputs for {family}")
                samples[mode].append(time.perf_counter() - start)
        output[family] = dict(rows=len(texts), token_cap=CAP,
                              seconds={k: median(v) for k, v in samples.items()}, per_run=samples)
    write_json(HERE / "offset_bench.json", output)
    print(output, flush=True)


if __name__ == "__main__":
    main()
