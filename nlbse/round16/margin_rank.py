"""Augment distillation with ordering loss around a fixed decision boundary."""

import json

import numpy as np
import torch
from torch.nn import functional as F


RANK_MODES=("control","rank_boundary","rank_pairs")


def training_weights(root,fit_ids,labels,verify):
    folder=root/"round15/mining"
    path=folder/"scores.npz"
    if verify(path)!=json.loads((folder/"manifest.json").read_text())["sha256"]:
        raise ValueError("Out-of-fold scores changed before training the ranking loss")
    with np.load(path) as z:
        if not np.array_equal(z["ids"],fit_ids) or not np.array_equal(z["y"],labels):
            raise ValueError("Ranking anchors do not align with the training split")
        p=z["compact"][:,0].clip(1e-5,1-1e-5)
    anchor=json.loads((root/"round13/study/policy.json").read_text())["compact_threshold"]
    logodds=np.log(p/(1-p))
    edge=np.log(anchor/(1-anchor))
    return torch.as_tensor(logodds,device="cuda",dtype=torch.float32),float(edge),verify(path)


def pair_loss(predicted,gold,cached,edge,mode):
    # Boundary examples come from out-of-fold scores. The pair labels are the gold function labels.
    certainty=torch.exp(-(cached-edge).abs()/1.25)
    weights=certainty[:,None]*certainty[None,:]*gold[:,None]*(1-gold[None,:])
    if mode=="rank_pairs":
        # Emphasise nearby opposing scores where reordering changes the operating point.
        weights=weights*torch.exp(-(cached[:,None]-cached[None,:]).abs())
    logits=predicted.float()
    margin=logits[None,:]-logits[:,None]+.15
    return (F.softplus(margin)*weights).sum()/weights.sum().clamp_min(.15)
