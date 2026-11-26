"""Construct matched hard examples from both scanner branches for the next correction."""
from collections import defaultdict
from pathlib import Path

import numpy as np

from section37 import HERE, ROOT, PARTS, f, checksum, json_at, write_json
from matched_veto33 import evaluation
from points33 import OUT
from experiment33 import read_phase

DEST = HERE/'matched'
NEIGHBORS = 3


def make_pairs(phase, texts, family, files, available, errors, observed):
    cohorts = defaultdict(list)
    chosen = phase['ids']
    by = [(family[i],files[i]) for i in chosen]
    rank = phase['replay']
    dist = np.column_stack((np.log1p(phase['length'])/.65,phase['unqueried']/.6,rank/.6))
    for k in np.flatnonzero(available & ~errors):
        cohorts[by[k]].append(k)
    anchors, controls, group = [],[],[]
    error_ids = np.flatnonzero(available & errors)
    for a in error_ids:
        pool = np.asarray(cohorts[by[a]],dtype=np.int64)
        if not len(pool):
            continue
        assert (phase['fold'][pool]==phase['fold'][a]).all()
        target = texts[chosen[a]]
        pool = np.asarray([k for k in pool if texts[chosen[k]]!=target],dtype=np.int64)
        if not len(pool):
            continue
        d = np.linalg.norm(dist[pool]-dist[a],axis=1)
        kept = pool[np.lexsort((chosen[pool],d))[:NEIGHBORS]]
        anchors.extend([a]*len(kept))
        controls.extend(kept)
        group.extend([by[a]]*len(kept))
    a,b = np.asarray(anchors,dtype=np.int64),np.asarray(controls,dtype=np.int64)
    return a,b,group,dict(observed_rows=int(observed.sum()),anchors=len(error_ids),
                           matched_anchors=len(np.unique(a)),pairs=len(a),files=len(set(group)))


def main():
    texts,labels,family,files,_ = f.inputs()
    DEST.mkdir(exist_ok=True)
    results = {}
    for rotation in PARTS:
        parents = OUT/rotation
        receipt = json_at(parents/'receipt.json')
        phase = read_phase('oof',parents,receipt)
        ids = phase['ids']
        truth = phase['y']
        assert np.array_equal(truth,labels[ids,0])
        prior = phase['unqueried']>0.
        final = phase['replay']>0.
        asked = phase['old']
        negative = asked&(phase['flag']==0)
        positive = asked&(phase['flag']==1)
        unseen = ~negative & ~positive
        table = ROOT/'round30/tuned_teacher'/rotation
        teacher = table/'oof.npz'
        assert checksum(teacher)==json_at(table/'fit.json')['predictions_sha256']
        with np.load(teacher) as saved:
            values,i,j = np.intersect1d(ids,saved['ids'],return_indices=True)
            assert len(values)>0
            assert np.array_equal(phase['fold'][i],saved['fold'][j])
            assert np.array_equal(truth[i],saved['y'][j])
            unix = np.full(len(ids),np.nan)
            stack = np.full(len(ids),np.nan)
            unix[i],stack[i] = saved['raw'][j,2],saved['calibrated'][j,4]
        options = (
            ('negative_consensus',negative & ~prior,False),
            ('negative_veto',negative & prior,False),
            ('negative_retained',negative,True),
            ('positive_retained',positive,True),
            ('unqueried_miss',unseen,False),
            ('unqueried_fp',unseen,True),
        )
        rank = dict(compact=phase['pre'][:,0],prior=phase['unqueried'],current=phase['replay'],unix=unix,stack=stack)
        dump, summary = {}, {}
        for name,route,base in options:
            a,b,group,rec = make_pairs(phase,texts,family,files,route & (final==base),truth!=base,route)
            assert np.equal(truth[a],not base).all() and np.equal(truth[b],base).all()
            show = {}
            for key,p in rank.items():
                present = np.isfinite(p[a]) & np.isfinite(p[b])
                if present.any():
                    files_in_pair = [row for row,ok in zip(group,present,strict=True) if ok]
                    show[key] = evaluation(-p if base else p,a[present],b[present],files_in_pair)
                    show[key]['scored_pairs'] = int(present.sum())
            teacher_present = np.isfinite(unix[a]) & np.isfinite(unix[b])
            if teacher_present.any():
                on_same = [row for row,ok in zip(group,teacher_present,strict=True) if ok]
                anchors,controls = a[teacher_present],b[teacher_present]
                score = phase['replay']
                show['current_on_teacher_pairs'] = evaluation(-score if base else score,anchors,controls,on_same)
                show['current_on_teacher_pairs']['scored_pairs'] = int(teacher_present.sum())
            dump.update({name+'_anchor':ids[a],name+'_control':ids[b],name+'_fold':phase['fold'][a],
                         name+'_anchor_dup_mask':phase['duplicates'][a],name+'_control_dup_mask':phase['duplicates'][b]})
            summary[name] = rec | {'error_type':'fp' if base else 'fn','pair_ranking':show}
        path = DEST/f'{rotation}.npz'
        np.savez_compressed(path,**dump)
        results[rotation] = dict(regimes=summary,predictions_sha256=checksum(path),
                         points_sha256=checksum(parents/'receipt.json'),teacher_sha256=checksum(teacher))
        print(rotation,{k:(r['anchors'],r['matched_anchors'],r['pairs'],
                            {a:round(b['pair_win'],4) for a,b in r['pair_ranking'].items()}) for k,r in summary.items()},flush=True)
    write_json(DEST/'results.json',dict(analysis=results,code_sha256=checksum(Path(__file__)),neighbors=NEIGHBORS))


if __name__=='__main__':
    main()
