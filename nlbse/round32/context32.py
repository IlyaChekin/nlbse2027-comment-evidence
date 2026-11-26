"""Reuse the exact Round 29 decision margins on each signed partition."""
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
R31 = ROOT/'round31'
sys.path.insert(0,str(R31))
from common31 import PARTS, R28, f, checksum, json_at, write_json, reference, signed_source
from continue_eval import prepare, section_margin, neg_cut, replay
from project_eval import raw_values
from length_eval import REFERENCE, R24
from union_scanner import CONFIG, PROGRAM
from positive import strict_cut


def frames(rotation):
    dev,test,fit,cal,review,db,tb,m0,m1 = reference(rotation)
    specs = prepare(rotation,dev,test,fit,cal)
    contexts = (dev,) if test is dev else (dev,test)
    margins = []
    for ctx, spec in zip(contexts,specs,strict=True):
        cache = dict(available=np.zeros(len(ctx.texts),bool),value=np.empty((len(ctx.texts),4)))
        margins.append(section_margin(ctx,spec,cache))
    shift = neg_cut(dev,margins[0],specs[0],cal)
    points = [replay(ctx, margin, spec, shift) for ctx,margin,spec in zip(contexts,margins,specs,strict=True)]
    if test is dev:
        points.append(points[0])
    for ctx, before, rank, risk in zip((dev,test),(db,tb),points,(m0,m1),strict=True):
        if not np.array_equal(before,rank>0.) or not np.allclose(rank[ctx.active],risk[ctx.active],atol=1e-10):
            raise ValueError('The replay no longer matches the signed Round 29 decisions')
    review = np.ones(len(tb),bool) if review is None else review
    return dev,test,cal,review,points,specs[0]['cutoff']


def rule_order():
    cfg = json_at(CONFIG)
    if checksum(PROGRAM)!=cfg['program_sha256']:
        raise ValueError('The released scanner changed')
    labels = [r.removeprefix('rule_') for r in json_at(f.SOURCE/'policy.json')['rule_columns']] + ['195']
    if len(labels)!=len(cfg['order']) or set(labels)!=set(cfg['order']):
        raise ValueError('Full scanner and early stop have different rule sets')
    return cfg['order'],[labels.index(tag) for tag in cfg['order']]


def early_out(raw):
    order, indices = rule_order()
    active = raw[:,indices]>0
    hits = active.any(1)
    first = active.argmax(1)
    counts = raw[np.arange(len(raw)),np.asarray(indices)[first]]
    seen = np.where(hits,first+1,0)
    return np.column_stack((seen, np.where(hits,counts,0)))

