"""An identifier-role view for classification, preserving comments and API spellings."""
from concurrent.futures import ProcessPoolExecutor
import re

from project_split import TOKEN

KEYWORDS=frozenset("alignas alignof asm auto bool break case char class const constexpr continue default delete do double else enum explicit extern false float for friend goto if inline int long mutable namespace new noexcept nullptr operator private protected public register restrict return short signed sizeof static struct switch template this thread_local throw true try typedef typename union unsigned using virtual void volatile while NULL".split())
IDENTIFIER=re.compile(r"^[A-Za-z_][A-Za-z_0-9]*$")


def normalize(text):
    tokens=list(TOKEN.finditer(text))
    names={}
    spans=[]
    previous=""
    for i,match in enumerate(tokens):
        token=match[0]
        if match["comment"] is not None:
            continue
        nxt=tokens[i+1][0] if i+1<len(tokens) else ""
        protected=(previous in {".",":",">","struct","class","enum","union","::"} or nxt in {"(",":"}
                   or token in KEYWORDS or token.isupper() or token.startswith("__"))
        if not protected and IDENTIFIER.fullmatch(token):
            identifier=names.setdefault(token,f"local_{len(names)+1}")
            spans.append((match.start(),match.end(),identifier))
        previous=token
    out=[]
    last=0
    for start,end,name in spans:
        out.extend((text[last:start],name))
        last=end
    out.append(text[last:])
    return "".join(out)


def batch(texts):
    return list(map(normalize,texts))


def parallel(texts,workers=3):
    chunks=[texts[i:i+1024] for i in range(0,len(texts),1024)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return [text for result in pool.map(batch,chunks) for text in result]
