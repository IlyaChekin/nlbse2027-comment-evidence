"""Learn source line boundaries from paired training strings and test scanner agreement."""
from collections import Counter, defaultdict
import csv
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np

from study50 import HERE, ROOT, PARTS, LABELS, checksum, json_at, load, parts, write_json, f
from format_audit49 import from_archive
from components49 import SOURCE, key, signature
from first_hit32 import SCRIPT, scan
from project_split import TOKEN, fingerprint

DEST=HERE/'restored'
SIZE=1100
SEED=5013
THRESHOLDS=(.90,.97)
WIDTH=3.8
KEYWORDS=frozenset(('if','else','for','while','do','switch','case','return','goto','break','struct',
                    'class','enum','union','int','void','static','const','unsigned','char','bool','new',
                    'delete','template','typename','try','catch','throw'))


def spots(text):
    tokens=[m for m in TOKEN.finditer(text) if m['comment'] is None]
    if any(m[0] in ('#','\\') for m in tokens):
        return [],[]
    gaps=[text[a.end():b.start()] for a,b in zip(tokens,tokens[1:])]
    return tokens,gaps


def token_type(token):
    s=token[0]
    if s in KEYWORDS or len(s)==1 and not s.isalnum() and s!='_':
        return s
    if s.startswith(('"', "'", 'R"')):
        return 'literal'
    return 'number' if s[0].isdigit() else 'identifier'


def shapes(tokens,gaps):
    terms=[token_type(t) for t in tokens]
    n=len(terms)-1
    for i,g in enumerate(gaps):
        if not g.isspace():
            continue
        br='\n' in g
        a,b=terms[i],terms[i+1]
        c=terms[i-1] if i else ''
        d=terms[i+2] if i+1<n else ''
        yield i,((br,c,a,b,d),(br,a,b,d),(br,c,a,b),(br,a,b))


def capture(source,formatted,counts):
    base,before=spots(formatted)
    gold,after=spots(source)
    assert len(base)==len(gold)
    reference=Counter()
    for i,keys in shapes(base,before):
        if not after[i].isspace():
            continue
        target='\n' in after[i]
        appear='\n' in before[i]
        if target!=appear:
            reference['different']+=1
        reference['eligible_gaps']+=1
        for k in keys:
            counts[k][int(target)]+=1
    return reference


def render(text,counts,threshold):
    tokens,gaps=spots(text)
    if not gaps:
        return text,0
    changes={}
    for i,keys in shapes(tokens,gaps):
        target=None
        for mark in keys:
            tally=counts.get(mark)
            if tally is None or sum(tally)<35:
                continue
            share=(tally[1]+1.)/(sum(tally)+2.)
            target=True if share>=threshold else False if share<=1.-threshold else None
            break
        current='\n' in gaps[i]
        if target is None or target==current:
            continue
        if target:
            start=text.rfind('\n',0,tokens[i].start())+1
            this_line=text[start:tokens[i].start()]
            left=this_line[:len(this_line)-len(this_line.lstrip(' \t'))]
            changes[i]='\n'+left
        else:
            changes[i]=' '
    if not changes:
        return text,0
    pieces=[]
    start=0
    for i,new in changes.items():
        lo,hi=tokens[i].end(),tokens[i+1].start()
        pieces.extend((text[start:lo],new))
        start=hi
    result=''.join((*pieces,text[start:]))
    assert signature(text)==signature(result)
    return result,len(changes)


def load_plans():
    data={}
    lookup={}
    labels=LABELS/'labels.npz'
    receipt=json_at(labels.with_name('receipt.json'))
    assert receipt['predictions_sha256']==checksum(labels) and receipt['archive_sha256']==checksum(SOURCE)
    with np.load(labels) as stored:
        active=stored['selected']
        comp=stored['component']
        target=stored['truth']
    texts,truth,p,files,_=f.inputs()
    require=defaultdict(set)
    for k,rotation in enumerate(PARTS):
        x,_,stamp=load(rotation)
        train=x['oof']
        cal,review=parts(x)
        held=np.unique(np.r_[x['dev']['ids'],x.get('test',x['dev'])['ids']])
        assert not np.intersect1d(held,train['ids']).size
        exclude={fingerprint(texts[i]) for i in held}
        ids=train['ids']
        assert np.array_equal(target[ids],train['y']) and np.array_equal(truth[ids,0],target[ids])
        available=train['old']&(train['flag']>=0)&active[ids]&(train['length']<=4096)
        old=train['flag']==1
        w=(comp[ids]&1)>0
        rng=np.random.default_rng(SEED+k)
        roster=[]
        for current,source in [(0,0),(0,1),(1,0),(1,1)]:
            where=np.flatnonzero(available&(w==source)&(old==current))
            for i in rng.choice(where,min(SIZE,len(where)),replace=False):
                idx=int(ids[i])
                if fingerprint(texts[idx]) in exclude:
                    continue
                where=key(p[idx],files[idx])
                sha=signature(texts[idx])
                require[where].add(sha)
                lookup[idx]=(where,sha)
                roster.append(idx)
        uses=[z['old']&(z['flag']>=0)&(abs(z['unqueried'])<WIDTH) for z in (cal,review)]
        data[rotation]=(roster,(cal,review),uses,stamp)
    return data,lookup,require,texts,comp,active


def study():
    DEST.mkdir(exist_ok=True)
    data,where,require,texts,archive_bits,eligible=load_plans()
    source=from_archive(require)
    allowed={}
    for row,place in where.items():
        candidates=source[place]
        if len(candidates)==1:
            original,=candidates
            assert candidates[original]==archive_bits[row]
            allowed[row]=original
    print('unique_fit_archive_rows',len(allowed),flush=True)
    report={}
    for rotation,(training,roles,uses,stamp) in data.items():
        dst=DEST/rotation
        dst.mkdir(exist_ok=True,parents=True)
        started=perf_counter()
        counts=defaultdict(lambda:[0,0])
        stat=Counter()
        fit=[]
        for i in training:
            original=allowed.get(i)
            if original is None:
                continue
            stat.update(capture(original,texts[i],counts))
            fit.append(i)
        weights=dst/'counts.joblib'
        joblib.dump(dict(counts=dict(counts),fit_ids=np.asarray(fit)),weights,compress=3)
        print(rotation,'fit',len(fit),dict(stat),flush=True)
        row={}
        outputs={}
        for role,frame,active in zip(('cal','held'),roles,uses,strict=True):
            ids=frame['ids'][active]
            labels=(archive_bits[ids]&1)>0
            mask=eligible[ids]
            before=frame['flag'][active]
            outputs[role+'_ids']=ids
            outputs[role+'_gold']=labels
            outputs[role+'_known']=mask
            outputs[role+'_scanner']=before
            row[role]={}
            for target in THRESHOLDS:
                begin=perf_counter()
                changed=[render(texts[i],counts,target) for i in ids]
                text_time=perf_counter()-begin
                rewritten,edits=zip(*changed,strict=True)
                changed=np.asarray(edits)>0
                new=before.copy()
                added=np.zeros((len(ids),3),np.int32)
                added[:,0]=before
                if changed.any():
                    cases=np.flatnonzero(changed)
                    output,secs=scan([rewritten[i] for i in cases],True)
                    valid=output[:,0]>=0
                    new[cases[valid]]=output[valid,0]
                    added[cases[valid]]=output[valid]
                else:
                    secs=0.
                values=f.comparison(labels[mask],before[mask]==1,new[mask]==1)
                n=str(target)
                row[role][n]=dict(truth_rows=int(mask.sum()),changed_rows=int(changed.sum()),
                          break_edits=int(sum(edits)),scans_if_replacing=len(ids),rescanned=int(changed.sum()),
                          extra_chars=int(sum(len(x)-len(texts[i]) for i,x in zip(ids,rewritten,strict=True))),
                          scanner_seconds=secs,transform_seconds=text_time,coverage=values)
                outputs[role+'_restored_'+n]=new
                outputs[role+'_scanned_'+n]=added
                print(rotation,role,n,row[role][n],flush=True)
        file=dst/'predictions.npz'
        np.savez_compressed(file,**outputs)
        result=dict(rotation=rotation,fit_pairs=len(fit),keys=len(counts),line_positions=stat,
                   threshold_margin=WIDTH,summary=row,predictions_sha256=checksum(file),
                   weights_sha256=checksum(weights),matrix_sha256=stamp,
                   elapsed=perf_counter()-started)
        write_json(dst/'results.json',result)
        report[rotation]=result|dict(receipt_sha256=checksum(dst/'results.json'))
    compare=DEST/'comparison.csv'
    with compare.open('w',newline='') as t:
        w=csv.writer(t)
        w.writerow(('rotation','role','threshold','changed','known','w_delta_tp','w_delta_fp','w_f1_before','w_f1_after'))
        for rotation,p in report.items():
            for fold,r in p['summary'].items():
                for limit,item in r.items():
                    result=item['coverage']
                    w.writerow((rotation,fold,limit,item['changed_rows'],item['truth_rows'],
                                *(result[a] for a in ('delta_tp','delta_fp','f1_before','f1_after'))))
    write_json(DEST/'receipt.json',dict(versions=report,archive_sha256=checksum(SOURCE),
             scanner_sha256=checksum(SCRIPT),comparison_sha256=checksum(compare),code_sha256=checksum(Path(__file__)),
             labels_sha256=checksum(LABELS/'labels.npz'),configuration=dict(thresholds=THRESHOLDS,max_per_panel=SIZE,seed=SEED)))


if __name__=='__main__':
    study()
