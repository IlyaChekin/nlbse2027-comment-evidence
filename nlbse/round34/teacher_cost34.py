"""Measure the actual cost of the quality-selected teacher arm on official calibration rows."""
from statistics import median

from sections34 import HERE, R30, f, checksum, json_at, write_json, checkpoint, load_model
from eval34 import loaded, DEST
from direct_teacher import score


def main():
    rotation='official'
    data,_,_=loaded(rotation)
    d=data['dev']
    table=DEST/rotation/'results.json'
    artifact=json_at(table)
    known=[json_at(DEST/name/'results.json')['options'] for name in ('test_a','test_b')]
    options=[]
    for key,row in artifact['options'].items():
        if row['model']!='head':
            continue
        candidate=row['fixed']
        internal,review=[candidate[k] for k in ('cal','held')]
        score_gain=internal['f1_after']-internal['f1_before']
        review_gain=review['f1_after']-review['f1_before']
        trained=[t[key]['matched']['fp']['change']['delta_tp'] for t in known]
        if min(trained)<=0 or score_gain<=0 or review_gain<0:
            continue
        options.append((score_gain,-row['cal_calls'],review_gain,key))
    if not options:
        raise ValueError('No panel passes the quality criteria')
    key=max(options)[-1]
    row=artifact['options'][key]
    mask=d['support'] & d['cal']
    if row['width'] is not None:
        mask &= abs(d['queried'])<row['width']
    text=f.inputs()[0]
    selected=[text[i] for i in d['ids'][mask]]
    net,tok=load_model('unixcoder9')
    source=R30/'tuned_teacher'/rotation
    balance=checkpoint(net,source,json_at(source/'fit.json'),0)
    score(net,tok,selected,balance)
    runs=[score(net,tok,selected,balance)[1] for _ in range(5)]
    estimates={k:median(run[k] for run in runs) for k in runs[0]}
    # The full public projection uses the actual number of calibration rows, including skipped calls.
    projected=.2*(f.PUBLIC_N/int(d['cal'].sum()))*(estimates['seconds']/1000+estimates['gflops']/3e6)
    value=row['fixed']['cal']
    gain=.3*(value['f1_after']-value['f1_before'])
    record=dict(option=key,calibration_rows=int(d['cal'].sum()),teacher_calls=len(selected),runs=runs,median=estimates,
                projected_cost_penalty=projected,quality_gain=gain,net_projected_gain=gain-projected,
                table_sha256=checksum(table),code_sha256=checksum(HERE/'teacher_cost34.py'))
    write_json(HERE/'teacher_cost.json',record)
    print(record,flush=True)


if __name__=='__main__':
    main()
