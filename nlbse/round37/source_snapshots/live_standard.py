"""Benchmark the frozen regional correction on the signed public input."""
import argparse
from pathlib import Path
from statistics import median
from time import perf_counter

import joblib
import numpy as np
import torch

from section37 import HERE, ROOT, f, checksum, json_at, write_json, NATIVE, SECTIONS, OUTPUTS
from section37 import prepare, section_scores
from section37 import lexical_features, ordered_batches, batch_plan, batch, small, pooling
from eval37 import features
from eval35 import cut
from live35 import forward, heads, load_choice, recipe, packages, full_context
from pipeline import PUBLIC, load

DEST = HERE/'public'
BASE = ROOT/'round34'


@torch.inference_mode()
def encode(model,tok,texts,projection,use_regions):
    first = perf_counter()
    n = len(texts)
    raw = np.empty((n,2),np.float32)
    aux = np.empty((n,22),np.float32)
    values = np.empty((n,2+NATIVE+(SECTIONS*(OUTPUTS+1) if use_regions else 0)),np.float32)
    pay,ops,extra = 0.,0.,0.
    def prepare_chunk(i):
        src = texts[i:i+2048]
        if use_regions:
            return prepare(src,tok)
        return full_context(src,tok),None,lexical_features(src)
    with torch.autocast('cuda',dtype=torch.float16):
        for i,(sequence,limits,lexical) in ordered_batches(range(0,n,2048),prepare_chunk,True):
            raw[i:i+len(sequence),1] = lexical[:,:4].any(axis=1)
            side = torch.as_tensor(lexical,device='cuda')
            limits = None if limits is None else torch.as_tensor(limits,device='cuda')
            for row in batch_plan(list(range(len(sequence))),sequence,small.BATCH):
                x,mask = batch(row,sequence,tok.pad_token_id)
                t = perf_counter()
                states = model.encoder.token_states(x,mask)
                core = model.encoder.readout(states,mask,side[row])
                pooled = model.rule_head[0](pooling(states,mask))
                rules = model.rule_head[1](pooled)
                with torch.autocast('cuda',enabled=False):
                    approved = projection(pooled.half().float())
                if use_regions:
                    profile,surcharge = section_scores(states,mask,side[row],model,limits[row],core,rules)
                    extra += surcharge/1e9
                else:
                    profile = torch.cat((core[:,[0]],rules),1)
                original = torch.cat((core,rules),1).float().sigmoid()
                merged = torch.cat((original,approved,profile.float()),1).cpu().numpy()
                idx = i+np.asarray(row)
                raw[idx,0],aux[idx],values[idx] = merged[:,0],merged[:,2:24],merged[:,24:]
                pay += perf_counter()-t
                width = model.width
                ops += small.operations(len(row),x.shape[1],width)+len(row)*width*(4*22+6+8)/1e9
    return raw,aux,values,dict(forward_seconds=pay,wall_seconds=perf_counter()-first,
                               analytic_gflops=ops+extra,projection_gflops=extra)


def main(arm):
    plan = HERE/'selection.json'
    policy = json_at(plan)
    selected = policy['selected'] if arm=='primary' else policy['best'][arm]
    if not selected:
        raise ValueError('No supported new candidate')
    table = HERE/'comparison/official/results.json'
    assert checksum(table)==policy['source_table']['official']
    models = table.with_name('readouts.joblib')
    assert checksum(models)==json_at(table)['weights_sha256']
    model = joblib.load(models)[selected['source'],selected['mode']]
    regional = not selected['mode'].startswith('native')
    old_path = BASE/'selection_approx.json'
    old = json_at(old_path)['selected']
    projection,calibrator,_ = heads(old,old_path)
    report = BASE/'public_approx/results.json'
    reference = report.with_name('candidate.npz')
    assert checksum(reference)==json_at(report)['comparison']['candidate']['predictions_sha256']
    with np.load(reference) as z:
        expected, y = z['pred'],z['truth']
    texts, labels,_ = load(PUBLIC)
    assert np.array_equal(y,labels)
    rule,coef,_ = load_choice()
    encoder = f.pipe('official',f.small.tokenizer())
    common = (encoder,recipe(),packages(),rule,coef,old,projection,calibrator)
    def run(text,candidate):
        if candidate:
            return forward(text,*common,selected,regional,model,encode,features)
        return forward(text,*common)
    kept,series = {}, {'reference':[], 'candidate':[]}
    for k in (False,True):
        run(texts[:4096],k)
    for i in range(5):
        for k in ((False,True) if i%2==0 else (True,False)):
            bits,margin,eligible,cost = run(texts,k)
            role = 'candidate' if k else 'reference'
            if role in kept:
                assert np.array_equal(bits,kept[role][0])
            kept[role] = bits,margin,eligible
            if k:
                assert np.array_equal(bits[:,1],expected[:,1])
                assert np.array_equal(bits[~eligible],expected[~eligible])
            else:
                assert np.array_equal(bits,expected)
            quality = f.metric(y,bits)
            score = f.competition_formula(quality['mean']['f1'],cost['forward_seconds'],cost['analytic_gflops'])
            series[role].append(cost | {'score':score})
            print(i,role,score,cost,flush=True)
    output = DEST if arm=='primary' else HERE/f'public_{arm}'
    output.mkdir(exist_ok=True)
    result = {}
    for name,(bits,margin,mask) in kept.items():
        profile = series[name]
        cost = {k:median(t[k] for t in profile) for k in profile[0] if k!='score'}
        quality = f.metric(y,bits)
        score = f.competition_formula(quality['mean']['f1'],cost['forward_seconds'],cost['analytic_gflops'])
        path = output/f'{name}.npz'
        np.savez_compressed(path,pred=bits,truth=y,margin=margin,scope=mask)
        result[name] = dict(score=score,quality=quality,cost=cost,trials=profile,predictions_sha256=checksum(path))
    gold,old = y[:,0].astype(bool),expected[:,0]
    bits,margin,eligible = kept['candidate']
    quality = {'frozen':f.comparison(gold,old,bits[:,0])}
    for budget in ('tp','fp'):
        proposed = old.copy()
        bound = cut(gold[eligible],old[eligible],margin[eligible],budget)
        proposed[eligible] = margin[eligible]>bound
        quality[budget] = f.comparison(gold,old,proposed)
    paired = [b['score']-a['score'] for a,b in zip(series['reference'],series['candidate'],strict=True)]
    archive = ROOT/'submission/NLBSE2027_round34_pooled_projection_tool.zip'
    assert checksum(archive)==policy['base_archive_sha256']
    result = dict(arm=arm,selected=selected,comparison=result,paired=paired,quality=quality,
                  receipt=dict(selection_sha256=checksum(plan),model_sha256=checksum(models),
                      code_sha256=checksum(Path(__file__)),core_sha256=checksum(ROOT/'round35/live35.py'),
                      public_sha256=checksum(PUBLIC),reference_sha256=checksum(reference),archive_sha256=checksum(archive)))
    write_json(output/'results.json',result)
    print(quality,paired,flush=True)


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--arm',choices=('primary','native','semantic'),default='primary')
    main(cli.parse_args().arm)
