"""Train a scanner-independent sparse expert on the established file folds."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import re
import sys
from time import perf_counter

import joblib
import numpy as np
from scipy.sparse import load_npz, save_npz, vstack
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import normalize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT/'round38'))
from data38 import PARTS, checksum, f, json_at, write_json
from points33 import OUT
from experiment33 import read_phase
from project_split import fingerprint

CACHE = HERE/'sparse_features'
DEST = HERE/'sparse'
DIMENSIONS = 2**18
KEEP = 620
EDGE = 100
TOKEN = re.compile(r'[A-Za-z_][A-Za-z_0-9]*|[0-9]+|[^\s\w]',re.ASCII)
WORD = re.compile(r'[A-Z]{2,}(?=[A-Z][a-z]|\b)|[A-Z]?[a-z]+|[A-Z]+|[0-9]+')
TERMS = frozenset("""alignas auto bool break case char class const constexpr continue default do double
elif else enum false float for goto if inline int long namespace new nullptr private public register
return short signed sizeof static struct switch template true typedef union unsigned using void volatile while""".split())
HASHER = FeatureHasher(n_features=DIMENSIONS,input_type='string',alternate_sign=False,dtype=np.float32)


def emitted(text):
    tokens = TOKEN.findall(text)
    if len(tokens)>KEEP:
        stride = (len(tokens)-2*EDGE)/(KEEP-2*EDGE)
        core = [tokens[EDGE+int(q*stride)] for q in range(KEEP-2*EDGE)]
        parts = (tokens[:EDGE],core,tokens[-EDGE:])
    else:
        parts = (tokens,)
    for group in parts:
        before = 'start'
        previous = 'start'
        previous_word = ''
        for word in group:
            if word[0].isdigit():
                symbol = 'number'
                value = 'number'
            elif not (word[0].isalpha() or word[0]=='_'):
                symbol = value = word
            else:
                value = word.lower()[:48]
                if value in TERMS:
                    symbol = value
                else:
                    symbol = 'constant' if word.isupper() else ('snake' if '_' in word else 'identifier')
                    for part in WORD.findall(word)[:8]:
                        if len(part)>=2 and not part[0].isdigit():
                            yield 'sub:'+part.lower()
            yield 'word:'+value
            yield 'shape:'+before+'/'+previous+'/'+symbol
            if previous_word:
                yield 'pair:'+previous_word+'/'+value
            before,previous = previous,symbol
            previous_word = value


def vector(texts):
    raw=HASHER.transform(emitted(x) for x in texts)
    raw.data.fill(1.)
    return normalize(raw,norm='l2',copy=False)


def collect():
    corpus,truth,_,_,_=f.inputs()
    start=perf_counter()
    CACHE.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=5) as pool:
        chunks=[corpus[i:i+1024] for i in range(0,len(corpus),1024)]
        pieces=list(pool.map(vector,chunks,chunksize=2))
    matrix=vstack(pieces,format='csr')
    file=CACHE/'matrix.npz'
    save_npz(file,matrix)
    write_json(CACHE/'receipt.json',dict(rows=len(truth),nonzero=int(matrix.nnz),dimensions=DIMENSIONS,
                   data_sha256=json_at(f.R19/'partition.json')['train_sha256'],matrix_sha256=checksum(file),
                   seconds=perf_counter()-start,code_sha256=checksum(Path(__file__))))
    print(len(truth),matrix.nnz,perf_counter()-start,flush=True)


def fit(x,y):
    z=x.copy()
    ratio=[]
    n=x.shape[1]
    for label in (False,True):
        count=np.asarray(z[y==label].sum(0)).ravel()+.3
        ratio.append(count/(count.sum()+n*.001))
    nb=np.log(ratio[1]/ratio[0]).clip(-3.,3.).astype(np.float32)
    z.data *= nb[z.indices]
    model=SGDClassifier(loss='log_loss',alpha=.000055,max_iter=34,tol=.00025,
                        average=True,random_state=4001)
    model.fit(z,y)
    return dict(weight=(model.coef_[0]*nb).astype(np.float32),bias=float(model.intercept_[0]),
                iterations=model.n_iter_)


def evaluate(x,trained):
    return (x@trained['weight']+trained['bias']).astype(np.float32)


def train(rotation):
    corpus,truth,project,path,_=f.inputs()
    origin=OUT/rotation
    old=origin/'receipt.json'
    proof=json_at(old)
    data={name:read_phase(name,origin,proof) for name in proof['phases']}
    catalog=json_at(CACHE/'receipt.json')
    file=CACHE/'matrix.npz'
    assert catalog['matrix_sha256']==checksum(file)
    assert catalog['code_sha256']==checksum(Path(__file__))
    assert catalog['data_sha256']==json_at(f.R19/'partition.json')['train_sha256']
    full=load_npz(file)
    oof=data['oof']
    training=oof['ids']
    assert np.array_equal(truth[training,0],oof['y'])
    assert len(training)==len(np.unique(training))
    for row in np.unique(oof['fold']):
        owners=set(zip(project[training[oof['fold']==row]],path[training[oof['fold']==row]],strict=True))
        remaining=set(zip(project[training[oof['fold']!=row]],path[training[oof['fold']!=row]],strict=True))
        assert not owners&remaining
    for side,item in data.items():
        if side!='oof':
            assert not np.intersect1d(training,item['ids']).size
    if 'test' in data:
        assert not set(project[training])&set(project[data['test']['ids']])
    if rotation=='official':
        held=np.unique(data['dev']['ids'])
    else:
        held=np.unique(np.r_[data['dev']['ids'],data['test']['ids']])
    assert not np.intersect1d(training,held).size
    keys={fingerprint(corpus[i]) for i in held}
    # The final head also excludes token copies of any validation function.
    distinct=np.fromiter((fingerprint(corpus[i]) not in keys for i in training),dtype=bool)
    matrix=full[training]
    values=np.empty(len(training),np.float32)
    start=perf_counter()
    counts={}
    for fold in np.unique(oof['fold']):
        held=oof['fold']==fold
        active=(oof['duplicates']&(1<<int(fold)))==0
        assert not (active&held).any()
        model=fit(matrix[active],oof['y'][active])
        values[held]=evaluate(matrix[held],model)
        counts[int(fold)]=dict(fit=int(active.sum()),held=int(held.sum()),
                           removed_copies=int((~(active|held)).sum()),iters=model['iterations'])
        print(rotation,int(fold),counts[int(fold)],round(perf_counter()-start,2),flush=True)
    model=fit(matrix[distinct],oof['y'][distinct])
    scores=dict(oof_ids=training,fold=oof['fold'],oof=values)
    for side,row in data.items():
        if side!='oof':
            scores[side+'_ids']=row['ids']
            scores[side]=evaluate(full[row['ids']],model)
    dest=DEST/rotation
    dest.mkdir(exist_ok=True,parents=True)
    table=dest/'scores.npz'
    weights=dest/'weights.joblib'
    np.savez_compressed(table,**scores)
    joblib.dump(model,weights,compress=3)
    write_json(dest/'receipt.json',dict(rotation=rotation,rows=len(training),folds=counts,
                   model_rows=int(distinct.sum()),excluded_validation_copies=int((~distinct).sum()),
                   base_sha256=checksum(old),weights_sha256=checksum(weights),predictions_sha256=checksum(table),
                   code_sha256=checksum(Path(__file__)),features_sha256=checksum(CACHE/'receipt.json')))
    print(rotation,'done',round(perf_counter()-start,2),flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('stage',choices=('collect',*PARTS))
    step=cli.parse_args().stage
    collect() if step=='collect' else train(step)
