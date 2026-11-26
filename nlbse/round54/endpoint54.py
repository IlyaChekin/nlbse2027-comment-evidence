"""Measure how archive-only formatting oracles affect the complete classifier."""
from collections import defaultdict
import csv
from pathlib import Path

import numpy as np

from causal54 import DEST as CAUSAL, HERE, ROOT, checksum, f, json_at, load, parts, scan, write_json
from live54 import execute, pipeline
from space54 import DEST as SPACES

DEST = HERE / 'endpoint'


def reports():
    baseline = CAUSAL / 'oracle.npz'
    assert json_at(CAUSAL/'oracle.json')['oracle_sha256'] == checksum(baseline)
    expanded = SPACES / 'predictions.npz'
    assert json_at(SPACES/'results.json')['predictions_sha256'] == checksum(expanded)
    controls = {baseline: ('lines', 'gaps', 'original'), expanded: ('exact_break', 'indent')}
    out = {}
    for file, modes in controls.items():
        with np.load(file) as z:
            for k in modes:
                out[k] = {int(i): (a,b) for i,a,b in zip(z['ids'],z['baseline'],z[k],strict=True) if b[0]>=0 and
                          not np.array_equal(a,b)}
    return out,{p.parent.name:checksum(p) for p in controls}


def scanner_override(ids, texts, values):
    grouped=defaultdict(list)
    for i,t in zip(ids,texts,strict=True):
        grouped[t].append(values.get(int(i)))
    mapping={}
    for t,p in grouped.items():
        if p[0] is not None and all(v is not None and np.array_equal(v,p[0]) for v in p):
            mapping[t]=p[0]
    bill = dict(edited=0, scanner_rows=0,text_seconds=0.,ml_gflops=0.,flipped=0)
    def scan_replayed(rows):
        answer,_=scan(rows,True)
        hit, flip=0, 0
        for i,q in enumerate(rows):
            if q in mapping:
                before, after=mapping[q]
                if np.array_equal(answer[i],before):
                    answer[i]=after
                    hit+=1
                    flip+=int(before[0]!=after[0])
        bill.update(edited=hit,flipped=flip,scanner_rows=len(rows))
        return answer
    return scan_replayed,bill


def main():
    DEST.mkdir(exist_ok=True)
    values, sources=reports()
    data,_,stamp=load('official')
    texts,label,project,paths,_=f.inputs()
    fit_files={(project[i],paths[i]) for i in data['oof']['ids'] if paths[i]}
    profiles, pred = {}, {}
    setup=pipeline()
    for phase,frame in zip(('cal','held'),parts(data),strict=True):
        ids=frame['ids']
        gold=frame['y']
        assert np.array_equal(gold,label[ids,0])
        rows=[texts[i] for i in ids]
        clean=np.asarray([bool(paths[i]) and (project[i],paths[i]) not in fit_files for i in ids])
        now,cost=execute(rows,setup)
        pred.update({phase+'_ids':ids,phase+'_gold':gold,phase+'_base':now,phase+'_file_held':clean})
        out={}
        for mode, matrix in values.items():
            trans, bill=scanner_override(ids,rows,matrix)
            bits,pay=execute(rows,setup,trans,bill)
            assert np.array_equal(bits[:,1],now[:,1]) and cost['scanner_calls']==pay['scanner_calls']
            pred[phase+'_'+mode]=bits
            out[mode]=dict(change=f.comparison(gold,now[:,0],bits[:,0]),new_files=f.comparison(gold[clean],now[clean,0],bits[clean,0]),
                           new_file_rows=int(clean.sum()), scanner_rows=pay['scanner_calls'], edited=pay['edited'], w_flips=pay['flipped'])
            print(phase,mode,out[mode],flush=True)
        profiles[phase]=out
    output=DEST/'predictions.npz'
    np.savez_compressed(output,**pred)
    table=DEST/'comparison.csv'
    with table.open('w',newline='') as handle:
        out=csv.writer(handle)
        out.writerow(('phase','arm','scanner_rows','scanner_flips','delta_tp','delta_fp','f1_before','f1_after','new_file_delta_tp','new_file_delta_fp'))
        for phase, fits in profiles.items():
            for k,v in fits.items():
                q,z=v['change'],v['new_files']
                out.writerow((phase,k,v['scanner_rows'],v['w_flips'],q['delta_tp'],q['delta_fp'],q['f1_before'],q['f1_after'],z['delta_tp'],z['delta_fp']))
    write_json(DEST/'receipt.json',dict(purpose='archive-only counterfactual, unavailable at inference', sources=sources,
               matrix_sha256=stamp,predictions_sha256=checksum(output),comparison_sha256=checksum(table),
               profiles=profiles,code_sha256=checksum(Path(__file__))))


if __name__=='__main__':
    main()
