"""Reproduce the public release and localize its false positives without selecting thresholds."""
import csv
import sys
from types import SimpleNamespace

import joblib
import numpy as np

from common31 import HERE, ROOT, R28, f, checksum, json_at, write_json
from evaluate_rules import infer
from length_eval import R24, recipe
from normalize_code import batch as canonicalize
from pipeline import PUBLIC, load
from positive_runtime import apply as correct_positive
from rank30 import rerank
from student_runtime import predict

sys.path[:0] = [str(ROOT/'round26/mat'),str(ROOT/'round21/mat')]
from mat_runtime import canonical_mat


def packages():
    report = R28/'recover_public'
    selected = json_at(report/'release_selection.json')['selected']
    option = json_at(report/'policy.json')['variants'][selected]
    recall_path = R28/'recover_calibrated/official/heads.joblib'
    assert checksum(recall_path)==option['source_sha256']
    recall = joblib.load(recall_path)['compact']
    r27 = ROOT/'round27/public'
    p27 = json_at(r27/'policy.json')
    correction = p27['supported'][p27['selected']]
    path = ROOT/'round27/joint/official/weights.joblib'
    assert checksum(path)==p27['weights_sha256']
    positive = joblib.load(path)[correction['weights_key']]
    r24 = R24/'official/tuned'
    p24 = json_at(r24/'policy.json')
    path = r24/'models.joblib'
    assert checksum(path)==p24['models_sha256']
    weights = joblib.load(path)['token']
    r25 = ROOT/'round25/views'
    p25 = json_at(r25/'public/policy.json')
    section = p25['support'][p25['selected']]
    path = r25/'official/calibrator.joblib'
    assert checksum(path)==p25['calibrator_sha256']
    section_head = joblib.load(path)[section['region']]
    return dict(negative=option['setting'],recall=recall,correction=correction['online'],
                positive=positive,weights=weights,variant=p24['candidates'][p24['selected']],
                section=section,section_head=section_head)


def baseline(texts, trace, config):
    tok = f.small.tokenizer()
    model = f.pipe('official',tok).encoder
    raw, aux, _, _ = predict(model,tok,texts,())
    first = raw[:,0].copy()
    policy = f.pinned()[0]
    reread = (first>policy['lower']) & (first<policy['upper'])
    view = canonicalize([s for s,use in zip(texts,reread,strict=True) if use])
    pipe = SimpleNamespace(encoder=model.encoder,tok=tok,mode='control')
    secondary, _ = infer(pipe,view,encoding=f.tokenize)
    raw[reread,0] = .5*(first[reread]+secondary[:,0])
    values = np.column_stack((raw[:,0],aux[:,0],aux[:,-2:]))
    core, gate, regression = recipe()
    estimates = f.estimate(regression,values)
    p, no, yes, asked = f.choose(values,estimates,gate,core['gate'])
    with np.load(trace) as receipt:
        if not np.array_equal(asked,receipt['gate']):
            raise ValueError('The signed scanner requests have changed')
        known, flag = np.zeros(len(texts),bool), np.zeros(len(texts),bool)
        known[asked] = receipt['flag']>=0
        flag[asked] = receipt['flag']==1
    p[known] = np.where(flag[known],yes[known],no[known])
    state = dict(p=p.astype(np.float32),w=estimates['w'],no=no,yes=yes,known=known,flag=flag)
    bits,_,_,_ = rerank(texts,values,state,core['threshold'],config['variant'],config['weights'],
                         config['section'],config['section_head'],config['negative'],config['recall'])
    bits,_,_,_ = correct_positive(texts,values,state,core['threshold'],bits,
                                   config['correction'],config['positive'])
    return np.column_stack((bits,canonical_mat(texts))),dict(first=first,reread=reread,asked=asked,
                                                           known=known,flag=flag)


def main():
    report = R28/'recover_public'
    release = json_at(report/'release_selection.json')['selected']
    log = json_at(report/'results.json')
    archive = report/log['results'][release]['csv']
    assert checksum(archive)==log['results'][release]['predictions_sha256']
    assert checksum(PUBLIC)==log['public_sha256']
    with archive.open() as handle:
        records = list(csv.DictReader(handle))
    assert [int(row['row']) for row in records]==list(range(len(records)))
    signed = np.array([[int(row[k]) for k in ('pred_vulnerability','pred_MAT')] for row in records],dtype=bool)
    texts, y, _ = load(PUBLIC)
    scan = ROOT/'round22/union_public'
    trace = scan/'scanner_predictions.npz'
    assert checksum(trace)==json_at(scan/'results.json')['scanner_predictions_sha256']
    bits, state = baseline(texts,trace,packages())
    assert np.array_equal(bits,signed)
    paths = dict(rule_negative=state['known'] & ~state['flag'],rule_positive=state['flag'],
                 invalid=state['asked'] & ~state['known'],unasked=~state['asked'])
    sizes = np.fromiter((len(text) for text in texts),int)
    truth = y[:,0].astype(bool)
    actual = bits[:,0]
    def counts(scope):
        return dict(rows=int(scope.sum()),tp=int((scope & truth & actual).sum()),
                    fn=int((scope & truth & ~actual).sum()),fp=int((scope & ~truth & actual).sum()))
    details = {}
    for label, scope in paths.items():
        details[label] = counts(scope) | {
            'one_pass':counts(scope & ~state['reread']),
            'second_pass':counts(scope & state['reread']),
            'length_ranges':{f'{lo}:{hi}':counts(scope & (sizes>=lo) & (sizes<hi))
                             for lo,hi in [(0,170),(170,512),(512,1024),(1024,int(sizes.max())+1)]},
            'first_probability_ranges':{f'{lo:g}:{hi:g}':counts(scope & (state['first']>=lo) & (state['first']<hi))
                             for lo,hi in [(0.,.08),(.08,.25),(.25,.5),(.5,.8),(.8,1.00001)]}}
    assert sum(v['fp'] for v in details.values())==460
    assert sum(v['fn'] for v in details.values())==512
    assert int(state['reread'].sum())==log['results'][release]['cost']['routed']
    path = HERE/'reference_audit.npz'
    np.savez_compressed(path,**state,truth=truth,bits=bits,length=sizes)
    result = dict(scope='public error description only',cohorts=details,rows=len(y),checked_decisions=bits.size,
                  readout_sha256=checksum(path),archive_predictions_sha256=checksum(archive),
                  public_sha256=checksum(PUBLIC),scanner_sha256=checksum(trace),code_sha256=checksum(HERE/'audit29.py'))
    write_json(HERE/'reference_audit.json',result)
    print({k:{name:v[name] for name in ('rows','tp','fn','fp')} for k,v in details.items()},flush=True)


if __name__=='__main__':
    main()
