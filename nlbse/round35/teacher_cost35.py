"""Profile the selected teacher readout on its independent calibration subset."""
from statistics import median
from time import perf_counter

from teacher35 import HERE, R30, f, checksum, json_at, write_json, checkpoint, load_model, windows, evaluate
from eval35 import load_all
from students import batch_plan, batch_gflops


def main():
    selection=json_at(HERE/'selection.json')['by_stage']['comparison']
    if selection is None:
        return
    data,_,_,_=load_all('official')
    phase=selection['mode'].split('_')[0]
    d=data['dev']
    use=d['cal'] & d['support']
    if selection['width'] is not None:
        use &= abs(d['latest'])<selection['width']
    ids=d['ids'][use]
    assert len(ids)==selection['cal_rows']
    full=f.inputs()[0]
    code=[full[i] for i in ids]
    net,tok=load_model('unixcoder9')
    folder=R30/'tuned_teacher/official'
    balance=checkpoint(net,folder,json_at(folder/'fit.json'),0)
    assert selection['mode'].endswith('_initial')
    import numpy as np
    center=np.log(balance)
    steps=np.arange(len(code))
    seq,sides=windows(code,tok)
    current=seq[phase]
    groups=list(batch_plan(steps.tolist(),current,96))
    ops=sum(batch_gflops('codebert',net,len(group),max(len(current[i]) for i in group)) for group in groups)
    evaluate(net,tok,current,sides,steps,center)
    shots=[]
    for _ in range(5):
        t=perf_counter()
        evaluate(net,tok,current,sides,steps,center)
        shots.append(perf_counter()-t)
    before=median(shots)
    result=dict(selection=selection,teacher_calls=len(code),cal_rows=int(d['cal'].sum()),
                forward_seconds=before,trials=shots,analytic_gflops=ops,
                quality_credit=.3*selection['cal_f1'],
                projected_cost=.2*f.PUBLIC_N/int(d['cal'].sum())*(before/1000+ops/3e6),
                selector_sha256=checksum(HERE/'selection.json'),code_sha256=checksum(HERE/'teacher_cost35.py'))
    write_json(HERE/'teacher_cost.json',result)
    print(result,flush=True)


if __name__=='__main__':
    main()
