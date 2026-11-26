"""Compare token-preserving formatting styles at the same scanner call count."""
from concurrent.futures import ThreadPoolExecutor
import csv
from pathlib import Path
from subprocess import PIPE, DEVNULL, run, TimeoutExpired
from time import perf_counter

import numpy as np

from components49 import DEST as LABELS, signature
from route49 import HERE, PARTS, f, checksum, json_at, load, write_json
from source41 import parts
from first_hit32 import scan, SCRIPT
from fetch_format49 import HERE as BIN

DEST=HERE/'styles'
STYLES=('LLVM','GNU','Chromium','WebKit','Microsoft')
WIDTH=3.8


def one(text, stamp, style):
    policy = '{BasedOnStyle: '+style+', SortIncludes: Never, ReflowComments: false, BreakStringLiterals: false}'
    exe = BIN/'clang-format'
    try:
        p=run([exe,'--style='+policy,'--assume-filename=function.cpp'],input=text,encoding='utf-8',
              stdout=PIPE,stderr=DEVNULL,check=False,timeout=3)
    except TimeoutExpired:
        return text,False
    return (p.stdout,True) if p.returncode==0 and signature(p.stdout)==stamp else (text,False)


def main():
    DEST.mkdir(exist_ok=True)
    proof=json_at(BIN/'receipt.json')
    exe=BIN/'clang-format'
    assert proof['program_sha256']==checksum(exe)
    ids,receipts,frames={}, {}, {}
    for phase in PARTS:
        source,_,receipts[phase]=load(phase)
        for role,role_data in zip(('cal','held'),parts(source),strict=True):
            scope=role_data['old'] & (role_data['flag']>=0) & (abs(role_data['unqueried'])<WIDTH)
            ids[phase,role]=role_data['ids'][scope]
            frames[phase,role]=(role_data['flag'][scope],role_data['y'][scope])
    file = LABELS/'labels.npz'
    assert json_at(file.with_name('receipt.json'))['predictions_sha256']==checksum(file)
    with np.load(file) as z:
        gold,matched,truth = (z[k] for k in ('component','selected','truth'))
    for k,i in ids.items():
        assert np.array_equal(truth[i],frames[k][1])
    roster = np.unique(np.concatenate(list(ids.values())))
    data = f.inputs()[0]
    texts = [data[r] for r in roster]
    tokens = [signature(t) for t in texts]
    baseline,base_elapsed=scan(texts,True)
    for k,i in ids.items():
        assert np.array_equal(baseline[np.searchsorted(roster,i),0],frames[k][0])
    records,profiles={},{}
    payload=dict(ids=roster,baseline=baseline,source_w=(gold[roster]&1)>0,matched=matched[roster])
    with ThreadPoolExecutor(max_workers=6) as pool:
        for style in STYLES:
            t=perf_counter()
            pairs=list(pool.map(one,texts,tokens,[style]*len(texts)))
            converted,valid=zip(*pairs,strict=True)
            tokenize=perf_counter()-t
            evaluated,elapsed=scan(converted,True)
            okay=evaluated[:,0]>=0
            selected=np.where(okay[:,None],evaluated,baseline)
            matches=baseline[:,0]!=selected[:,0]
            spans=np.fromiter(map(len,converted),np.int32)
            profiles[style]=dict(format_seconds=tokenize,scanner_seconds=elapsed,rows=len(texts),
                                  valid_format_count=int(sum(valid)),changed_scanner_bits=int(matches.sum()),
                                  chars=int(spans.sum()))
            payload[style]=selected
            print(style,profiles[style],flush=True)
            for k,i in ids.items():
                exact=i[matched[i]]
                locate=np.searchsorted(roster,exact)
                y=(gold[exact]&1)>0
                assert np.isin(selected[locate,0],(0,1)).all()
                comparison=f.comparison(y,baseline[locate,0]==1,selected[locate,0]==1)
                label='/'.join((*k,style))
                records[label]=comparison|dict(rows=len(exact),available=len(i))
    file_out=DEST/'predictions.npz'
    np.savez_compressed(file_out,**payload)
    table=DEST/'comparison.csv'
    with table.open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(('rotation','part','style','matched','available','delta_w_tp','delta_w_fp','f1_before','f1_after'))
        for name,stats in records.items():
            writer.writerow((*name.split('/'),*(stats[k] for k in ('rows','available','delta_tp','delta_fp','f1_before','f1_after'))))
    write_json(DEST/'receipt.json',dict(records=records,measurements=profiles,scanner_sha256=checksum(SCRIPT),
               formatter_sha256=checksum(BIN/'receipt.json'),annotation_sha256=checksum(file),sources=receipts,
               predictions_sha256=checksum(file_out),comparison_sha256=checksum(table),
               code_sha256=checksum(Path(__file__)),threshold=WIDTH,baseline_seconds=base_elapsed,
               total_calls_per_style=len(roster),original_characters=sum(map(len,texts))))
    fav=[(name,x['delta_tp'],x['delta_fp']) for name,x in records.items() if x['f1_after']>x['f1_before']]
    print('positive changes on the archived W label:',fav,flush=True)


if __name__=='__main__':
    main()
