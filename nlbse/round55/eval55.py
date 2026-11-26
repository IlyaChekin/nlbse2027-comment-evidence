"""Evaluate archived-evidence readouts on unchanged signed project predictions."""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from span55 import (ARMS, FIT, HERE, LABELS, PARTS, ROOT, batch, batch_plan, checksum, f, full_context,
                   json_at, lexical_features, load, masks, pooling, readout, release_start, small, tokenize, write_json)
from ground55 import DEST as GROUND
from eval51 import compare
from stack40 import per_project, diagnostic_order
from source41 import parts

DEST=HERE/'evaluation'
VIEWS=('v','w','joint','drift')


def head_views(q,native):
    v,w=q.T.astype(np.float64)
    return {'v':v,'w':w,'joint':.65*v+.35*w,'drift':w-native[:,1]}


@torch.inference_mode()
def data_scores(net, heads, text, regions, tok):
    tic=perf_counter()
    seq,marked,_,has=tokenize(text,tok,regions)
    lexical=torch.as_tensor(lexical_features(text),device='cuda')
    q={k:np.empty((len(seq),2),np.float32) for k in heads}
    loc={k:np.zeros((len(seq),4),np.float32) for k in heads if k!='pooled'}
    original=np.empty((len(seq),2),np.float32)
    debt=0.
    forward=0.
    with torch.autocast('cuda',dtype=torch.float16):
        for first in range(0,len(seq),2048):
            block=seq[first:first+2048]
            for group in batch_plan(list(range(len(block))),block,small.BATCH):
                bt,mask=batch(group,block,tok.pad_token_id)
                ids=first+np.asarray(group)
                period=perf_counter()
                hidden=net.encoder.token_states(bt,mask)
                native=torch.stack((net.encoder.readout(hidden,mask,lexical[ids])[:,0],
                                    net.rule_head(pooling(hidden,mask))[:,0]),1)
                original[ids]=native.float().cpu().numpy()
                marked_mask=masks(ids,marked,mask.size(1),'cuda')
                for k,head in heads.items():
                    score,tokens=head(hidden,mask,native)
                    q[k][ids]=score.cpu().numpy()
                    if tokens is not None:
                        ranks=tokens.masked_fill(~mask.bool(),-1.e4)
                        peak=ranks.argmax(1)
                        good=marked_mask[torch.arange(len(peak),device='cuda'),peak]
                        mass=(ranks.softmax(1)*marked_mask).sum(1)
                        proposal=ranks.topk(min(3,ranks.shape[1]),dim=1).indices
                        top=marked_mask.gather(1,proposal).any(1)
                        loc[k][ids]=torch.stack((good,mass,top,peak),dim=1).cpu().numpy()
                debt+=sum(2*hidden.shape[-1]*head.project.out_features*(len(group) if k=='pooled' else bt.numel())
                          for k,head in heads.items())
                forward+=perf_counter()-period
    return q,loc,has,original,dict(wall_seconds=perf_counter()-tic,forward_seconds=forward,project_gflops=debt/1e9)


def locate(frame,regions,computed,mapped,w):
    details={}
    for name,scores in computed.items():
        report={}
        for key,m in [('all',mapped&w),('missed',mapped&w&(frame['flag']==0))]:
            t=scores[m]
            base=regions[m]
            report[key]=dict(rows=len(t),top1_hits=int(t[:,0].sum()),top3_hits=int(t[:,2].sum()),
                  top1_share=float(t[:,0].mean()) if len(t) else None, avg_mass=float(t[:,1].mean()) if len(t) else None,
                  median_char_span=float(np.median(np.maximum(base[:,:,1]-base[:,:,0],0).sum(1))) if len(t) else None)
        details[name]=report
    return details


def metrics(frame, w, available, values, old):
    out={}
    upstream=frame['flag']
    base=upstream==1
    names={'all_archived':available,'queried':available & frame['old'] & (upstream>=0),
           'scanner_negative':available & frame['old'] & (upstream==0),
           'scanner_positive':available & frame['old'] & (upstream==1)}
    for name,pred in {'native':old,**values}.items():
        qual=dict(v_ap=float(average_precision_score(frame['y'],pred[:,0])))
        for kind,use in names.items():
            if use.any() and w[use].any():
                qual[kind]=dict(rows=int(use.sum()),positives=int(w[use].sum()),
                    ap=float(average_precision_score(w[use],pred[use,1])))
        use=names['queried']
        order,_=diagnostic_order(pred[use,1],frame['signed'][use],frame['ids'][use])
        budget,bits=per_project(w[use],frame['project'][use],base[use],np.ones(int(use.sum()),bool),order,('fp','tp'))
        qual['full_w_equal_budgets']=budget
        recovered=(~base[use]) & w[use] & bits['fp']
        lost=base[use] & w[use] & ~bits['fp']
        qual['w_held_fp_counts']=dict(recovered=int(recovered.sum()),lost=int(lost.sum()))
        out[name]=qual
    return out


def evaluate(rotation):
    proof=FIT/rotation/'fit.json'
    fitted=json_at(proof)
    assert fitted['fit_sha256']==checksum(proof.with_name('rows.npz'))
    assert len({v['steps'] for v in fitted['arms'].values()})==1
    assert fitted['train_sha256']==checksum(HERE/'span55.py')
    state,_,stamp=load(rotation)
    assert stamp==fitted['aligned']['matrix_sha256']
    frames=parts(state)
    label_receipt=json_at(LABELS.with_name('receipt.json'))
    assert label_receipt['predictions_sha256']==checksum(LABELS)
    raw=GROUND/'ground.npz'
    assert json_at(GROUND/'receipt.json')['data_sha256']==checksum(raw)
    with np.load(raw) as z:
        ground={phase:{int(i):s for i,s in zip(z['ids'][z[rotation+'_'+phase]],z['targets'][z[rotation+'_'+phase]],strict=True)}
                for phase in ('cal','held')}
    with np.load(LABELS) as z:
        arch=z['selected']
        w=(z['component']&1)>0
    text,labels,projects,paths,_=f.inputs()
    occupied={(projects[i],paths[i]) for i in state['oof']['ids'] if paths[i]}
    tok=f.small.tokenizer()
    net,h=release_start(rotation,tok)
    assert h==fitted['checkpoint_sha256']
    heads={k:readout(FIT/rotation,k,net,fitted) for k in ARMS}
    views,detail,payload={},{},{}
    for phase,frame in zip(('cal','held'),frames,strict=True):
        ids=frame['ids']
        regions=np.asarray([ground[phase].get(int(i),np.full((2,2),-1,np.int32)) for i in ids])
        scores,local,mapped,native,cost=data_scores(net,heads,[text[i] for i in ids],regions,tok)
        assert np.array_equal(frame['y'],labels[ids,0])
        detail[phase]=dict(ranking=metrics(frame,w[ids],arch[ids],scores,native),
                           localization=locate(frame,regions,local,mapped,w[ids]),cost=cost)
        for k,a in scores.items():
            for u,v in head_views(a,native).items():
                views.setdefault((k,u),[]).append(v)
            payload[phase+'_'+k]=a
        for k,a in local.items():
            payload[phase+'_token_'+k]=a
        payload.update({phase+'_ids':ids,phase+'_native':native,phase+'_has_span':mapped,
                        phase+'_known':arch[ids],phase+'_archival_w':w[ids],phase+'_region':regions})
        print(rotation,phase,{k:(x['scanner_negative']['ap'],x['v_ap']) for k,x in detail[phase]['ranking'].items()},
              detail[phase]['localization'],flush=True)
    report,labels=compare(rotation,frames,views)
    held=frames[1]
    labels['clean_file']=np.asarray([bool(paths[i]) and (projects[i],paths[i]) not in occupied for i in held['ids']])
    root=DEST/rotation
    root.mkdir(exist_ok=True,parents=True)
    predictions=root/'predictions.npz'
    score_file=root/'scores.npz'
    np.savez_compressed(predictions,**labels)
    np.savez_compressed(score_file,**payload)
    write_json(root/'results.json',dict(scores_sha256=checksum(score_file),predictions_sha256=checksum(predictions),
               fit_sha256=checksum(proof),matrix_sha256=stamp,analysis=detail,compared=report,
               gold_sha256=checksum(raw),code_sha256=checksum(Path(__file__))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('rotation',choices=PARTS)
    evaluate(p.parse_args().rotation)
