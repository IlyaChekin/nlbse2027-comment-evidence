"""Verify training provenance and file-bootstrap the external-label probe comparison."""
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from eval51 import HERE, ROOT, FIT, PARTS, ARMS, f, load, checksum, json_at, write_json, training_code
from support51 import DEST as SUPPORT
from probe51 import DEST as PROBE
from source41 import parts

TRIALS=1000
SEED=5119


def evaluator(score,target):
    ordered=np.argsort(-score,kind='stable')
    z=score[ordered]
    ends=np.flatnonzero(np.r_[z[1:]!=z[:-1],True])
    truth=target[ordered]
    def weighted(mass):
        h=mass[ordered]
        d=np.cumsum(h*truth)[ends]
        n=np.cumsum(h)[ends]
        if d[-1]==0:
            return np.nan
        precision=np.divide(d,n,out=np.zeros(len(d),dtype=np.float64),where=n>0)
        return float((precision@np.diff(np.r_[0,d]))/d[-1])
    return weighted


def bootstrapped(components):
    support=SUPPORT/'receipt.json'
    mask=support.with_name('mask.npz')
    stamp=json_at(support)
    assert stamp['mask_sha256']==checksum(mask)
    with np.load(mask) as z:
        available=z['external_available']
        e=(z['component']&6)>0
    _,_,project,files,_=f.inputs()
    output={}
    for r in ('test_a','test_b'):
        frame,_,_=load(r)
        _,held=parts(frame)
        active=available[held['ids']]
        ids=held['ids'][active]
        label=e[ids]
        assert np.all([bool(files[i]) for i in ids])
        file_ids=np.asarray([project[i]+'\0'+files[i] for i in ids])
        key,groups=np.unique(file_ids,return_inverse=True)
        size=len(key)
        proof=PROBE/r/'results.json'
        p=proof.with_name('predictions.npz')
        assert json_at(proof)['predictions_sha256']==checksum(p)
        forecasts={}
        with np.load(p) as z:
            assert np.array_equal(z['ids'],held['ids'])
            for name in ('all_matched','measured'):
                forecast=z['1_original_'+name][active]
                expected=components['scores'][r]['original/'+name]['held']['1']['ap_measured']
                assert abs(average_precision_score(label,forecast)-expected)<1e-12
                forecasts[name]=evaluator(forecast,label)
        weights=np.ones(len(label),np.int32)
        base,after=(forecasts[k](weights) for k in ('all_matched','measured'))
        rng=np.random.default_rng(SEED)
        copied=[]
        for _ in range(TRIALS):
            selected=rng.integers(0,size,size=size)
            counts=np.bincount(selected,minlength=size)[groups]
            a,b=(forecasts[k](counts) for k in ('all_matched','measured'))
            copied.append(b-a)
        band=np.asarray(copied)
        output[r]=dict(files=int(size),rows=len(ids),external_positives=int(label.sum()),
                       before=base,measured_only=after,delta=after-base,
                       ci95=np.nanquantile(band,[.025,.975]).tolist(),valid_resamples=int(np.isfinite(band).sum()))
    return output


def main():
    info={}
    release=ROOT/'round38/release_summary.json'
    old=json_at(release)
    reference=ROOT/'submission/NLBSE2027_round38_paired_rule_tool.zip'
    assert old['archive_sha256']==checksum(reference)
    for name,tool in [('comparison','select51.py'),('probe_comparison','pick_probe51.py')]:
        p=HERE/name/'selection.json'
        card=json_at(p)
        assert card['code_sha256']==checksum(HERE/tool)
        assert card['csv_sha256']==checksum(p.with_name('comparison.csv'))
        info[name]=card
    for part in PARTS:
        p=FIT/part/'fit.json'
        row=json_at(p)
        assert row['config']['code_sha256']==checksum(training_code(part))
        assert row['config']['roster_sha256']==checksum(p.with_name('rows.npz'))
        assert set(row['arms'])==set(ARMS)
        assert len({g['steps'] for g in row['arms'].values()})==1
    validation=HERE/'live/selection.json'
    readiness=json_at(validation)['selected'] if validation.exists() else None
    p=HERE/'live/public.json'
    if p.exists():
        measured=json_at(p)
        assert readiness==measured['selection']
        assert measured['predictions_sha256']==checksum(p.with_name('public_predictions.npz'))
    else:
        measured=None
    outcome=dict(reference_score=old['score'],gap=old['target_gap'],
                 multitask_selected=info['comparison']['selected'],
                 masked_probe_selected=info['probe_comparison']['selected'],
                 external_ap=bootstrapped(info['probe_comparison']),
                 strict_official_review=info['comparison']['clean_file_diagnostic'],
                 public_measurement=measured,
                 training_provenance={n:checksum(FIT/n/'fit.json') for n in PARTS},
                 code_sha256=checksum(Path(__file__)))
    write_json(HERE/'audit.json',outcome)
    print('multitask selection',outcome['multitask_selected'],flush=True)
    print('masked probe selection',outcome['masked_probe_selected'],flush=True)
    print('external AP',outcome['external_ap'],flush=True)
    print('official unseen-file scope', {k:v for k,v in outcome['strict_official_review'].items() if k!='predictions'},flush=True)


if __name__=='__main__':
    main()
