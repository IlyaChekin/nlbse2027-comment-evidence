"""Extract name-free C/C++ token shapes before the scanner runs."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import re
import sys
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT/'round41'),str(ROOT/'round12')]
from source41 import PARTS, f, checksum, json_at, load, write_json
from project_split import TOKEN

DEST = HERE/'layout'
TERMS = ('if','else','switch','case','while','for','return','continue','break','goto',
         'try','throw','catch','sizeof','new','delete','nullptr','NULL','void',
         '(',')','{','}','[',']',',',';','*','&','=','<','>','!','#','?','+','-',
         'comment','identifier','literal','number','other')
LOOKUP = {k:i for i,k in enumerate(TERMS)}
QUOTED = ('"', "'", 'R"')
LIKE_NAME = re.compile(r'^[A-Za-z_]\w*$')


def summarize(counter, depth, maximum, count):
    shape = np.asarray([counter[k] for k in TERMS],dtype=np.float32)
    return np.r_[np.log1p(shape),np.log1p([max(0,depth),maximum,count])].astype(np.float32)


def layout(text):
    head,body,tail = Counter(),Counter(),Counter()
    dep=maximum=0
    body_tokens=0
    n=0
    end=max(0,len(text)*3//4)
    for m in TOKEN.finditer(text):
        n+=1
        q=m[0]
        if m['comment'] is not None:
            key='comment'
        elif q in LOOKUP:
            key=q
        elif q.startswith(QUOTED):
            key='literal'
        elif q[0].isdigit():
            key='number'
        else:
            key='identifier' if LIKE_NAME.fullmatch(q) else 'other'
        dep += (key=='{')-(key=='}')
        maximum=max(maximum,dep)
        target=body if dep>0 or key=='}' else head
        body_tokens += target is body
        target[key]+=1
        if m.start()>=end:
            tail[key]+=1
    return np.r_[summarize(head,0,0,n-body_tokens),summarize(body,dep,maximum,body_tokens),
                 summarize(tail,0,0,sum(tail.values())),np.log1p([text.count('\n')+1,len(text)])].astype(np.float32)


def main():
    DEST.mkdir(exist_ok=True)
    stamps,groups = {},[]
    for part in PARTS:
        data,_,stamps[part] = load(part)
        groups.extend(r['ids'] for r in data.values())
    ids=np.unique(np.concatenate(groups))
    source=f.inputs()
    texts=source[0]
    start=perf_counter()
    with ProcessPoolExecutor(max_workers=6) as pool:
        x=np.stack(list(pool.map(layout,(texts[i] for i in ids),chunksize=128)))
    file=DEST/'features.npz'
    np.savez_compressed(file,ids=ids,layout=x)
    receipt=dict(rows=len(ids),fields=TERMS,dimensions=x.shape[1],matrix_sources=stamps,
                 seconds=perf_counter()-start,dataset_sha256=json_at(ROOT/'data/manifest.json')['files'][0]['sha256'],
                 code_sha256=checksum(Path(__file__)),predictions_sha256=checksum(file))
    write_json(DEST/'receipt.json',receipt)
    print(receipt,flush=True)


if __name__=='__main__':
    main()
