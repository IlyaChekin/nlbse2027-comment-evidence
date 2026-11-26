"""Extract numeric context and matched-length text around a scanner location."""
import re

import numpy as np

from comment_spans import LEXEME
from section_runtime import opening

WINDOW = 2
CAP = 320
CONTROL = re.compile(r'[^\w\s]',re.ASCII)
WORDS = re.compile(r'[A-Za-z_][A-Za-z_0-9]*|[0-9]+|[^\w\s]',re.ASCII)
SKIP = re.compile(r'//|/\*')


def without_literals(text):
    return LEXEME.sub(lambda match: '\n'*match[0].count('\n')+' ',text)


def whole_tokens(part):
    for m in WORDS.finditer(part):
        word=m[0]
        yield 'num' if word[0].isdigit() else word.lower()


def profile(text, values):
    first,last,distinct,gap,noticed,total = map(int,values[3:])
    if not first or not total:
        return np.zeros(17,np.float32),'',''
    original=text.split('\n')
    first=min(first,len(original))-1
    last=min(last,len(original))-1
    split_at=opening(text)
    brace=text.count('\n',0,split_at)
    start=max(0,first-WINDOW)
    stop=min(len(original),first+WINDOW+1)
    cleaned=without_literals('\n'.join(original[start:stop]))
    local=cleaned[:CAP]
    pivot=(len(original)-1)//2
    cut=min(len(original),max(stop-start,pivot+WINDOW+1))
    begin=cut-(stop-start)
    middle=without_literals('\n'.join(original[begin:cut]))[:CAP]
    marked=without_literals(original[first])
    lengths=[len(x.lstrip()) for x in original[start:stop]]
    indent=len(original[first])-len(original[first].lstrip(' \t'))
    words=list(whole_tokens(marked))
    shape=(
        first/total,last/total,(last-first)/total,distinct/total,np.log1p(gap),noticed/max(1,values[2]),
        (first-brace)/total,float(first>=brace),np.log1p(total),np.log1p(len(marked)),np.log1p(indent),
        np.log1p(len(words)),len(set(words))/max(1,len(words)),
        len(CONTROL.findall(marked))/max(1,len(marked)),float(bool(SKIP.search(original[first]))),
        np.log1p(max(lengths,default=0)),np.log1p(np.sum(np.asarray(lengths)==0)),
    )
    return np.asarray(shape,np.float32),' '.join(whole_tokens(local)),' '.join(whole_tokens(middle))


def ordinary(texts, values):
    measurements=[profile(t,result) for t,result in zip(texts,values,strict=True)]
    shape,near,decoy=zip(*measurements,strict=True)
    return np.asarray(shape,np.float32),near,decoy
