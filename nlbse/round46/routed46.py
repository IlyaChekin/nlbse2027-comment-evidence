"""Time the only quality-qualified new encoder on its actual review scope."""
from pathlib import Path
from statistics import median
from time import perf_counter

import joblib
import numpy as np

from fit46 import HERE,ROOT,DEST as FIT,f,checksum,json_at,load,parts,prepare,point_matrix,join,crop,write_json,predict
from compare46 import DEST as COMPARISON
from views46 import DEST as WINDOWS,bookends,boundaries,tokenizers,TYPES
from extract46 import encode,weights
from study44 import scope
from eval35 import adjusted

DEST = HERE/'routed'
CAP = 384


def prepare_actual(text,toks):
    # Build only the requested long view. UniXcoder's boundary is retained so
    # the source region exactly matches the comparison experiment.
    packed = [t(text,add_special_tokens=False,return_offsets_mapping=True,verbose=False)['offset_mapping']
              for t in toks]
    view = [boundaries(row,pair,CAP,(1,4))[0] for row,pair in zip(text,zip(*packed,strict=True),strict=True)]
    raw = toks[0](view,add_special_tokens=False,verbose=False)['input_ids']
    a,b = bookends(*toks)['qwen']
    result = [a+x+b for x in raw]
    assert all(len(r)<=CAP for r in result)
    return result


def main():
    path = COMPARISON/'selection.json'
    policy = json_at(path)
    rule = policy['models']['qwen384']['quality_only']
    assert rule and rule['quality'] and rule['arm']=='qwen384'
    cohort,t,stamp = load('official')
    _,frame = parts(cohort)
    member = scope(frame,rule['panel'],rule['width'])
    assert int(member.sum())==rule['review_rows']
    local = crop(frame,member)
    model_file = FIT/'official/weights.joblib'
    trust = json_at(model_file.with_name('receipt.json'))
    assert trust['matrix_sha256']==stamp and trust['model_sha256']==checksum(model_file)
    model = joblib.load(model_file)['qwen384']
    names = f.inputs()[0]
    text = [names[i] for i in local['ids']]
    tok = tokenizers()
    encoder,pad,weight = weights('qwen')
    width = encoder.config.hidden_size*3
    target = np.empty((len(text),width),np.float16)
    encode(prepare_actual(text,tok),encoder,pad,'qwen',CAP,target)
    trials = []
    for _ in range(7):
        start = perf_counter()
        seq = prepare_actual(text,tok)
        tokenization = perf_counter()-start
        stats = encode(seq,encoder,pad,'qwen',CAP,target)
        trials.append(stats | {'tokenization_seconds':tokenization})
    src, enc, proof = prepare('official',stamp)
    ix=np.searchsorted(src,local['ids'])
    assert np.array_equal(src[ix],local['ids'])
    p = WINDOWS/'qwen_384.npz'
    with np.load(p) as z:
        ends,bank = z['ends'],z['tokens']
        assert all(np.array_equal(n,bank[ends[i]:ends[i+1]]) for i,n in zip(ix,seq,strict=True))
    raw = point_matrix([local],t)
    result = predict(join(raw,[target])[0],local['signed'].clip(-12.,12.),model)
    rank = (1.-rule['mix'])*local['signed']+rule['mix']*result
    baseline = frame['signed']>0.
    compared,decision = adjusted(frame['y'],baseline,member,rank,rule['cut'])
    prior = COMPARISON/'official/predictions.npz'
    trusted = json_at(prior.with_name('results.json'))
    assert checksum(prior)==trusted['predictions_sha256']
    with np.load(prior) as z:
        assert np.array_equal(z['ids'],frame['ids'])
        identical = np.array_equal(z[rule['key']+'_'+rule['rule']],decision)
    centered = {k:median(trial[k] for trial in trials) for k in trials[0]}
    change = .3*(compared['f1_after']-compared['f1_before'])
    # Use prepared-batch wall time and tokenizer wall time. The lightweight
    # readout's additional overhead is omitted, so a negative estimate is conservative.
    cost = .2*(centered['wall_seconds']+centered['tokenization_seconds'])/1000.+.2*centered['analytic_gflops']/3e6
    estimate = change-cost
    DEST.mkdir(exist_ok=True,parents=True)
    file = DEST/'predictions.npz'
    np.savez_compressed(file,ids=frame['ids'],scope=member,pred=decision,base=baseline,rank=rank)
    detail=dict(active=len(text),saved_embedding_rows_exact=int(np.all(target==enc['qwen384'][ix],axis=1).sum()),
                same_predictions=bool(identical),score_delta_quality=change,score_delta_cost=cost,
                net_estimate=estimate,review=compared,trials=trials,medians=centered,
                selection=rule,selection_sha256=checksum(path),encoder_sha256=weight,
                predictions_sha256=checksum(file),code_sha256=checksum(Path(__file__)))
    write_json(DEST/'receipt.json',detail)
    print(detail,flush=True)


if __name__=='__main__':
    main()
