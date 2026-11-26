"""Extract name-free syntax paths around a recorded first scanner match."""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
import re
import sys
from time import perf_counter

import numpy as np
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import HashingVectorizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
R39 = ROOT/'round39'
sys.path[:0] = [str(HERE/'deps'),str(R39)]
from location39 import OUTPUT, PROGRAM, f, checksum, json_at, write_json
from context39 import CAP, WINDOW, WORDS, profile, without_literals
from study39 import vectorizer as lexical_vectorizer
from tree_sitter import Language, Parser
import tree_sitter_c
import tree_sitter_cpp

DEST = HERE/'features'
GRAMMARS = ('tree-sitter','tree-sitter-c','tree-sitter-cpp')
KEYWORDS = frozenset("""
alignas alignof asm auto bool break case catch char class const consteval constexpr constinit
continue default delete do double else enum explicit export extern false float for friend goto
if inline int long mutable namespace new noexcept nullptr operator private protected public
register reinterpret_cast restrict return short signed sizeof static static_assert static_cast
struct switch template this throw true try typedef typename union unsigned using virtual void
volatile while define defined endif ifdef ifndef elif include pragma
""".split())
MAX_LEAVES = 96
MAX_ANCHORS = 32
LITERAL = {'string_literal','char_literal','raw_string_literal','number_literal','system_lib_string'}


@lru_cache(maxsize=1)
def parser_pair():
    return Parser(Language(tree_sitter_cpp.language())),Parser(Language(tree_sitter_c.language()))


def syntax(text):
    a,b = parser_pair()
    raw = text.encode('utf-8',errors='replace')
    tree = a.parse(raw)
    if tree.root_node.has_error:
        other = b.parse(raw)
        if not other.root_node.has_error:
            return other,'c'
    return tree,'cpp'


def intersection(node,top,bottom):
    y,x = node.end_point
    return node.start_point.row<bottom and (y>top or (y==top and x>0))


def leaves(node,start,stop):
    todo = [node]
    while todo:
        current = todo.pop()
        if not intersection(current,start,stop):
            continue
        if not current.child_count or current.type in LITERAL or current.type=='comment':
            yield current
        else:
            todo.extend(reversed(current.children))


def kind(node):
    value = node.type
    if node.is_missing:
        return 'missing:'+value
    return value.replace(' ','_')


def labels_for(tree,pivot,start,stop):
    local = []
    paths = set()
    anchored = 0
    for leaf in leaves(tree.root_node,start,stop):
        if len(local)>=MAX_LEAVES:
            break
        if leaf.type=='comment':
            continue
        value = kind(leaf)
        local.append(value)
        if anchored>=MAX_ANCHORS or not intersection(leaf,pivot,pivot+1):
            continue
        anchored += 1
        walk = [value]
        parent = leaf.parent
        while parent is not None and len(walk)<8:
            walk.append(kind(parent))
            parent = parent.parent
        for length in (2,3,4,6,8):
            if len(walk)>=length:
                paths.add('path:'+'>'.join(walk[:length]))
        paths.update('anc:'+a+'>'+b for a,b in zip(walk[1:],walk[2:]))
    for n in (1,2,3):
        paths.update('leaf:'+'>'.join(local[i:i+n]) for i in range(len(local)-n+1))
    return ' '.join(sorted(paths))


def letters(text):
    for token in WORDS.finditer(text):
        word = token[0]
        if word in KEYWORDS or not word[0].isalnum() and word[0]!='_':
            yield word
        elif word[0].isdigit():
            yield 'num'
        else:
            yield 'identifier'


def contexts(text,location):
    geometry,lexical,_ = profile(text,location)
    rows = text.split('\n')
    line = int(location[3])-1
    length = len(rows)
    assert 0<=line<length
    begin,end = max(0,line-WINDOW),min(length,line+WINDOW+1)
    center = (length-1)//2
    far_end = min(length,max(end-begin,center+WINDOW+1))
    far_begin = far_end-(end-begin)
    masked = without_literals('\n'.join(rows[begin:end]))[:CAP]
    tree,language = syntax(text)
    structural = labels_for(tree,line,begin,end)
    decoy = labels_for(tree,center,far_begin,far_end)
    return geometry,lexical,' '.join(letters(masked)),structural,decoy,language,tree.root_node.has_error


def context_job(job):
    text,location = job
    return contexts(text,location)


def vectorizer(view):
    if view in ('ast','center'):
        return HashingVectorizer(analyzer='word',tokenizer=str.split,token_pattern=None,n_features=2**17,
                                norm='l2',alternate_sign=True,lowercase=False,dtype=np.float32)
    return lexical_vectorizer()


def main():
    DEST.mkdir(exist_ok=True)
    ref = json_at(OUTPUT/'receipt.json')
    assert ref['extended_scanner_sha256']==checksum(PROGRAM)
    path = OUTPUT/'training.npz'
    assert ref['predictions_sha256']==checksum(path)
    with np.load(path) as raw:
        mask = (raw['values'][:,0]==1)&(raw['values'][:,3]>0)
        ids,geo = raw['ids'][mask],raw['values'][mask]
    corpus = f.inputs()[0]
    jobs = ((corpus[i],p) for i,p in zip(ids,geo,strict=True))
    started = perf_counter()
    with ProcessPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(context_job,jobs,chunksize=96))
    elapsed = perf_counter()-started
    geo,text,masked,ast,center,lang,error = zip(*results,strict=True)
    geo = np.asarray(geo,np.float32)
    lang = np.array(lang)
    error = np.array(error)
    file = DEST/'rows.npz'
    np.savez_compressed(file,ids=ids,shape=geo,language=lang,parse_error=error)
    manifests = {}
    for view,terms in (('text',text),('masked',masked),('ast',ast),('center',center)):
        features = vectorizer(view).transform(terms)
        p = DEST/(view+'.npz')
        save_npz(p,features)
        manifests[view] = dict(sha256=checksum(p),nonzero=features.nnz,columns=features.shape[1])
    print(len(ids),dict(c=int((lang=='c').sum()),parse_errors=int(error.sum())),elapsed,flush=True)
    parser_manifest = HERE/'parser.sha256'
    write_json(DEST/'receipt.json',dict(rows=len(ids),scanner_receipt_sha256=checksum(OUTPUT/'receipt.json'),
               rows_sha256=checksum(file),features=manifests,
               parser={v:version(v) for v in GRAMMARS},parser_wheels_manifest_sha256=checksum(parser_manifest),
               details=dict(c=int((lang=='c').sum()),parse_errors=int(error.sum()),seconds=elapsed),
               context_sha256=checksum(Path(__file__)),previous_context_sha256=checksum(R39/'context39.py')))


if __name__=='__main__':
    main()
