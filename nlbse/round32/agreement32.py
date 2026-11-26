"""Contrast the executed rule with the compact encoder's rule-specific expectations."""
import numpy as np
from scipy.special import logit

from context32 import HERE, checksum, json_at, rule_order

OPTIONS=('aux_context','observed','observed_interactions','tree_aux_context','tree_observed')
BASELINE='aux_context'


def inputs(rotation):
    folder = HERE/'aux'/rotation
    proof = json_at(folder/'receipt.json')
    source = folder/'scores.npz'
    if checksum(source)!=proof['predictions_sha256'] or checksum(HERE/'collect_aux.py')!=proof['code_sha256']:
        raise ValueError('Rule prediction fingerprint changed')
    with np.load(source) as z:
        return dict(z),proof['predictions_sha256']


def features(geometry, first, aux):
    _, order = rule_order()
    columns = np.asarray([index+1 if index<19 else 0 for index in order])
    actual = first[:,0]-1
    row = np.arange(len(aux))
    head = aux[:,columns]
    logits = logit(head.astype(np.float64).clip(1e-5,1.-1e-5))
    expected = aux[:,1:].mean(1)
    # Rule 195 was introduced after the auxiliary training labels. Keep it separate.
    known = columns[actual]>0
    observed = np.where(known,logits[row,actual],0.)
    ahead = actual[:,None]>np.arange(len(columns))
    outranked = ahead & (head>head[row,actual,None])
    prev = np.max(np.where(ahead,head,0.),axis=1)
    context = np.column_stack((geometry,expected,aux[:,1:].max(1),logits.mean(1)))
    proof = np.column_stack((observed,known,prev,outranked.sum(1),logits[row,actual]-logits.max(1),
                             np.log1p(first[:,1].clip(0,64))))
    combined = np.column_stack((context,proof))
    one_hot = actual[:,None]==np.arange(len(columns))
    expanded = np.column_stack((combined,one_hot,one_hot*observed[:,None],one_hot*context[:,1,None]))
    return dict(aux_context=context,observed=combined,observed_interactions=expanded,tree_aux_context=context,
                tree_observed=np.column_stack((combined,actual)))
