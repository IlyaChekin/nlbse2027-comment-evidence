"""Freeze one compact rescue candidate before full validation."""
from pairs import HERE, checksum, json_at, write_json


def select():
    folder=HERE/'recover_calibrated'
    sources=[folder/i/'results.json' for i in ('test_a','test_b','official')]
    a,b,o=[json_at(p) for p in sources]
    baseline=HERE/'threshold'
    anchored=json_at(baseline/'policy.json')['variants']['calibration_only']['setting']
    selected={}
    for name,row in o['options'].items():
        if not name.startswith('compact_'):
            continue
        gains=[x['options'][name]['matched']['delta_tp'] for x in (a,b)]
        if min(gains)<=0:
            continue
        for arm,c in row['cuts'].items():
            fixed=c['frozen']
            current=c['cal']
            if c['proxy']<=0 or current['delta_fp']>0 or current['delta_tp']<0 or fixed['f1_after']<=fixed['f1_before']:
                continue
            _,width,blend,direction=name.split('_')
            config=dict(width=float(width),mix=float(blend),direction=direction,cut=c['cut'])
            final=anchored|{'recover':config}
            selected[f'{name}:{arm}']=dict(c,setting=final,project_tp=gains,cal_calls=row['cal_calls'],
                                          source_sha256=o['weights_sha256'])
    winner=max(selected,key=lambda k:(selected[k]['proxy'],sum(selected[k]['project_tp'])),default=None)
    path=HERE/'recover_public'
    path.mkdir(exist_ok=True)
    result=dict(options=[winner] if winner else [],variants=selected,baseline_setting=anchored,
                tables_sha256={p.parent.name:checksum(p) for p in sources},
                original_policy_sha256=checksum(baseline/'policy.json'))
    write_json(path/'policy.json',result)
    print(winner,selected.get(winner),flush=True)
    return result
