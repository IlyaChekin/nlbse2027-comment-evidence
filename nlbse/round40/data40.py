"""Assemble signed out-of-file expert scores and the exact held-out release control."""
from pathlib import Path
import sys

import joblib
import numpy as np

from sparse40 import HERE, ROOT, PARTS, DEST as SPARSE, f, checksum, json_at, write_json
sys.path.insert(0,str(ROOT/'round39'))
from data38 import phases
from data39 import archived, oof
from rank38 import installed
from points33 import OUT
from policy33 import matrix, logit

DEST=HERE/'matrices'


def attach(rotation):
    data,ref,older=phases(rotation)
    sources=installed(rotation,data,older)
    details,base=archived(rotation,data,ref,sources)
    saved,pairs=oof(rotation,data,ref['cutoff'],details)
    proof=OUT/rotation/'receipt.json'
    signed=ROOT/'round24'/rotation
    entry=json_at(signed/'fit.json')['models']['token']
    compact=ROOT/'round26/compact'/rotation
    certificate=json_at(compact/'manifest.json')
    assert certificate['sparse_oof_sha256']==entry['predictions_sha256']
    compact_scores=compact/'scores.npz'
    assert checksum(compact_scores)==certificate['sha256']
    with np.load(compact_scores) as z:
        model=data['oof']
        assert all(np.array_equal(z[k],model[m]) for k,m in (('ids','ids'),('y','y'),('fold','fold')))
        assert np.array_equal(z['values'],model['pre'][:,:4])
    shape=SPARSE/rotation
    chosen=json_at(shape/'receipt.json')
    table=shape/'scores.npz'
    assert chosen['base_sha256']==checksum(proof)
    assert chosen['code_sha256']==checksum(HERE/'sparse40.py')
    assert chosen['predictions_sha256']==checksum(table)
    assert chosen['weights_sha256']==checksum(shape/'weights.joblib')
    fingerprints={}
    with np.load(table) as z:
        for side,role in data.items():
            item=signed/f'token_{side}.npz'
            expected=entry['predictions_sha256'] if side=='oof' else entry[side]
            assert checksum(item)==expected
            fingerprints[side]=expected
            with np.load(item) as x:
                assert np.array_equal(x['ids'],role['ids'])
                if side=='oof':
                    assert np.array_equal(x['fold'],role['fold'])
                    assert np.array_equal(x['dup'],role['duplicates'])
                    assert np.array_equal(x['y'],role['y'])
                raw=x['logits']
            assert np.array_equal(z[side+'_ids'],role['ids'])
            role['sparse']=raw
            role['new']=z[side]
    return data,ref,dict(release_options_sha256=base,round38_crossfit=saved,pairs_sha256=pairs,
              first_scores_sha256=checksum(proof),compact_sha256=checksum(compact_scores),
              sparse_sources_sha256=fingerprints,new_scores_sha256=checksum(table))


def outside_scope(frame):
    known=frame['old']&(frame['flag']>=0)
    return np.where(known,np.where(frame['flag']==0,0,1),2).astype(np.int8)


def features(rows,t):
    z=matrix(rows['pre'],rows['length'],t)
    current=np.clip(rows['signed'],-10.,10.)
    side=outside_scope(rows)
    regimes=side[:,None]==np.arange(3)[None,:]
    # Only the actually observed scanner answer is exposed to this model.
    # Counterfactual queried scores remain outside the feature matrix.
    rank=np.column_stack((current,rows['unqueried'],z[:,0],z[:,7],z[:,5],z[:,6]))
    own=np.column_stack((z[:,:14],current,rank[:,:3]*regimes[:,0,None],
                         rank[:,:3]*regimes[:,1,None],rank[:,:3]*regimes[:,2,None],regimes))
    left,right=rows['sparse'].T
    best=np.where(side==0,left,np.where(side==1,right,(1.-rows['pre'][:,4])*left+rows['pre'][:,4]*right))
    old=np.column_stack((left,right,best,best-current,regimes*best[:,None]))
    x=rows['new'].clip(-12.,12.)
    review=np.column_stack((x,x-current,regimes*x[:,None]))
    sources=np.column_stack((rows['unqueried'],z[:,0],np.where(side==0,z[:,5],np.where(side==1,z[:,6],z[:,7])),best,x))
    return own,old,review,sources,side
