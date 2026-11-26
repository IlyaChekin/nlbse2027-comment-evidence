"""Evaluate compact continuation heads on exactly the signed held-out rows."""
import argparse
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import average_precision_score

from train51 import HERE, ROOT, PARTS, DEST as FIT, COMPONENTS, ARMS, f, checksum, json_at, load, write_json
from train51 import release_start, extra_head, added, encode, full_context, lexical_features, batch, batch_plan, small
from source41 import parts
from policy33 import logit
from stack40 import diagnostic_order, per_project
from eval35 import adjusted, cut

DEST=HERE/'evaluation'
PANELS=('all','negative','positive','unqueried')
WIDTHS=(1.25,3.8,None)
BLENDS=(.35,.7,1.)


def training_code(rotation):
    return HERE/('train51.py' if rotation=='official' else 'train51_project.py')


def configured(rotation,arm,tok,proof):
    net,base=release_start(rotation,tok)
    assert base==proof['config']['base_checkpoint_sha256']
    addon=None
    if arm is not None:
        path=FIT/rotation/(arm+'.pt')
        assert checksum(path)==proof['arms'][arm]['sha256']
        stored=torch.load(path,map_location='cpu',weights_only=True)
        populated=dict(net.named_parameters())
        assert set(stored['backbone'])<=populated.keys()
        with torch.no_grad():
            for k,v in stored['backbone'].items():
                populated[k].copy_(v)
        addon=extra_head(net.width).cuda().eval()
        addon.load_state_dict(stored['head'])
    net.eval()
    return net,addon


@torch.inference_mode()
def scores(net,head,seq,side,tok):
    p=np.empty((len(seq),24),np.float32)
    q=np.empty((len(seq),3),np.float32)
    ops=0.
    gpu=0.
    t0=perf_counter()
    with torch.autocast('cuda',dtype=torch.float16):
        for start in range(0,len(seq),2048):
            block=seq[start:start+2048]
            for ro in batch_plan(list(range(len(block))),block,small.BATCH):
                x,m=batch(ro,block,tok.pad_token_id)
                rows=start+np.asarray(ro)
                t1=perf_counter()
                base,pool=encode(net,x,m,side[rows])
                if head is None:
                    p[rows]=base.cpu().numpy()
                else:
                    overall=added(head,base,pool)
                    value=torch.cat((base,overall),dim=1).cpu().numpy()
                    p[rows],q[rows]=value[:,:24],value[:,24:]
                    ops+=2.*len(rows)*(net.width*2*64+64*3)/1e9
                gpu+=perf_counter()-t1
    return p,q,dict(readout_gflops=ops,forward_seconds=gpu,wall_seconds=perf_counter()-t0)


def geometry(piece,panel,width):
    m=np.ones(len(piece['ids']),bool)
    if panel!='all':
        key={'negative':0,'positive':1,'unqueried':2}[panel]
        m &= piece['observed_regime']==key
    if width is not None:
        m &= abs(piece['signed'])<width
    return m


def view(logits,arm):
    v,w,e=logits.T
    pW,pE=expit(w.astype(np.float64)),expit(e.astype(np.float64))
    if not arm.endswith('components'):
        return {'main':v.astype(np.float64)}
    return dict(main=v.astype(np.float64),component_union=logit(pW+(1.-pW)*pE),
                component_max=np.maximum(v,e))


def compare(rotation,frames,score):
    cal,held=frames
    y=[m['y'] for m in frames]
    p=[m['signed'] for m in frames]
    b=[r>0. for r in p]
    output=dict(ids=held['ids'],truth=y[1],base=b[1],cal_ids=cal['ids'])
    report={}
    for panel,width in product(PANELS,WIDTHS):
        masks=[geometry(r,panel,width) for r in frames]
        if not all(m.sum()>4 for m in masks):
            continue
        names=[*product(score,BLENDS),(('reference','threshold'),0.)]
        for ((arm,k),blend) in names:
            q=[(1.-blend)*z[m]+blend*u[m] for z,m,u in zip(p,masks,score.get((arm,k),p),strict=True)]
            name=f'{panel}/{width}/{arm}/{k}/{blend:g}'
            info=dict(panel=panel,width=width,arm=arm,view=k,mix=blend,
                      cal_rows=int(masks[0].sum()),held_rows=int(masks[1].sum()))
            if rotation!='official':
                v=masks[1]
                rank,_=diagnostic_order(q[1],p[1][v],held['ids'][v])
                info['matched'],bits=per_project(y[1],held['project'],b[1],v,rank,('fp','tp'))
                for mode,v in bits.items():
                    output[name+'_same_'+mode]=v
            for mode in ('fixed','fp','tp'):
                edge=cut(y[0][masks[0]],b[0][masks[0]],q[0],mode)
                a,_=adjusted(y[0],b[0],masks[0],q[0],edge)
                z,v=adjusted(y[1],b[1],masks[1],q[1],edge)
                info[mode]=dict(cut=edge,cal=a,held=z)
                output[name+'_'+mode]=v
            report[name]=info
    return report,output


def annotation_metrics(library,frames,forecasts):
    labels=library['component']
    available=library['selected']
    out={}
    for role,piece,z in zip(('cal','held'),frames,forecasts,strict=True):
        ids=piece['ids']
        use=available[ids]
        w=(labels[ids]&1)>0
        e=(labels[ids]&6)>0
        assert np.array_equal(piece['y'][use],(w|e)[use])
        out[role]=dict(y_ap=float(average_precision_score(piece['y'],z[:,0])),
                  w_ap=float(average_precision_score(w[use],z[use,1])),
                  e_ap=float(average_precision_score(e[use],z[use,2])),
                  e_without_w_ap=float(average_precision_score(e[use & ~w],z[use & ~w,2])),
                  w_positive=int((w&use).sum()),e_positive=int((e&use).sum()),
                  archived_rows=int(use.sum()),
                  missed_external_baseline=int((e&~w&use&(piece['signed']<=0.)).sum()))
    return out


def main(rotation):
    begin=perf_counter()
    src,_,stamp=load(rotation)
    frames=parts(src)
    train=FIT/rotation/'fit.json'
    proof=json_at(train)
    assert proof['config']['sources']['matrix_sha256']==stamp
    assert proof['config']['code_sha256']==checksum(training_code(rotation))
    assert proof['config']['roster_sha256']==checksum(train.with_name('rows.npz'))
    assert set(proof['arms'])==set(ARMS) and len({x['steps'] for x in proof['arms'].values()})==1
    src_text=f.inputs()[0]
    tok=f.small.tokenizer()
    seq=[full_context([src_text[i] for i in r['ids']],tok) for r in frames]
    side=[torch.as_tensor(lexical_features([src_text[i] for i in r['ids']]),device='cuda') for r in frames]
    original,_=configured(rotation,None,tok,proof)
    base=[scores(original,None,k,q,tok)[0] for k,q in zip(seq,side,strict=True)]
    del original
    label_path=COMPONENTS/'labels.npz'
    manifest=label_path.with_name('receipt.json')
    assert proof['config']['sources']['annotations_sha256']==checksum(manifest)
    assert json_at(manifest)['predictions_sha256']==checksum(label_path)
    with np.load(label_path) as f0:
        library=dict(f0)
    verdict={}
    options={}
    extras={}
    for arm in ARMS:
        model,extra=configured(rotation,arm,tok,proof)
        found=[scores(model,extra,k,q,tok) for k,q in zip(seq,side,strict=True)]
        model_drifts=[]
        seen=[z[1] for z in found]
        for j,(legacy,tail,cost) in enumerate(found):
            drift=legacy-base[j]
            if arm.startswith('frozen'):
                assert np.array_equal(legacy,base[j])
            model_drifts.append(dict(main_average=float(np.mean(abs(drift[:,0]))),
                                     routing_average=float(np.mean(abs(drift[:,[0,2,22,23]]))),
                                     max_delta=float(np.max(abs(drift))),cost=cost))
            extras[str(j)+'_'+arm]=tail
            for v,k in view(tail,arm).items():
                options.setdefault((arm,v),[]).append(k)
        verdict[arm]=dict(components=annotation_metrics(library,frames,seen),drift=model_drifts)
        print(rotation,arm,verdict[arm]['components'],flush=True)
        del model,extra
    report,pred=compare(rotation,frames,options)
    home=DEST/rotation
    home.mkdir(exist_ok=True,parents=True)
    values=home/'scores.npz'
    np.savez_compressed(values,cal_ids=frames[0]['ids'],held_ids=frames[1]['ids'],**extras)
    path=home/'predictions.npz'
    np.savez_compressed(path,**pred)
    write_json(home/'results.json',dict(fit_sha256=checksum(train),matrix_sha256=stamp,results=report,arms=verdict,
               prediction_sha256=checksum(path),scores_sha256=checksum(values),code_sha256=checksum(Path(__file__)),
               seconds=perf_counter()-begin))
    print(rotation,len(report),round(perf_counter()-begin,2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('rotation',choices=PARTS)
    main(p.parse_args().rotation)
