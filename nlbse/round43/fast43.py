"""Compute the same numeric first-hit inputs with less text and scanner work."""
from pathlib import Path

import numpy as np

from features43 import HERE, PROGRAM, WINDOW, f, without_literals
from context39 import CONTROL, SKIP, whole_tokens, opening

SCANNER = HERE/'scan_numeric.mjs'


def numeric(text,values):
    first,last,distinct,gap,noticed,total = map(int,values[3:])
    if not first or not total:
        return np.zeros(17,np.float32)
    lines = text.split('\n')
    first,last = min(first,len(lines))-1,min(last,len(lines))-1
    brace = text.count('\n',0,opening(text))
    marked = without_literals(lines[first])
    lengths = [len(row.lstrip()) for row in lines[max(0,first-WINDOW):first+WINDOW+1]]
    indent = len(lines[first])-len(lines[first].lstrip(' \t'))
    tokens = list(whole_tokens(marked))
    shape = (
        first/total,last/total,(last-first)/total,distinct/total,np.log1p(gap),noticed/max(1,values[2]),
        (first-brace)/total,float(first>=brace),np.log1p(total),np.log1p(len(marked)),np.log1p(indent),
        np.log1p(len(tokens)),len(set(tokens))/max(1,len(tokens)),
        len(CONTROL.findall(marked))/max(1,len(marked)),float(bool(SKIP.search(lines[first]))),
        np.log1p(max(lengths,default=0)),np.log1p(sum(n==0 for n in lengths)),
    )
    return np.asarray(shape,np.float32)


def script():
    old = PROGRAM.read_text()
    find = '        const valid = lines.filter'
    assert old.count(find)==1
    new = old.replace(find,'''        if (j !== 1) return [row, 1, j + 1, lines.length, 0, 0, 0, 0, 0, body.length];
        if (lines.length === 1) {
          const line = lines[0];
          const good = Number.isSafeInteger(line) && line >= 1 && line <= body.length;
          const k = good ? line : 0;
          const seen = Number(good);
          return [row, 1, j + 1, 1, k, k, seen, 0, seen, body.length];
        }
'''+find)
    if SCANNER.exists():
        assert SCANNER.read_text()==new
    else:
        SCANNER.write_text(new)


def scan(texts):
    from location39 import COLUMNS
    names,r = f.scanner(texts,program=SCANNER)
    assert tuple(names)==COLUMNS
    return r
