"""Train small frozen-prefix heads and choose a joint cost-aware routing policy."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import Ridge
import torch
from torch import nn
from torch.nn import functional as F

from cache_hybrid import ACTIONS, HERE
from frontier import THRESHOLDS, calibrated, metric
from pipeline import write_json

DEPTH = 4
HIDDEN = 192
EPOCHS = 10
SEED = 42
STUDENTS = {
    "unregularized": (0., 0., 0.),
    "regularized": (.18, .025, 0.),
    "distilled": (.18, .025, .45),
}
PENALTIES = (0., .10, .20, .35, .50, .75, 1., 1.5, 2., 3., 4.5, 7., 11., 18., 30.)


def features_for_router(x, scores, cutoffs):
    raw = logit(np.clip(scores, 1e-6, 1-1e-6))
    margin = raw-logit(cutoffs)
    # Keep the gate inexpensive and rely on early scores and observed length.
    return np.column_stack((raw, margin, np.abs(margin), scores*(1-scores),
                           x[:, -7:], x[:, -1:] * np.abs(margin))).astype(np.float32)


def normalized(values, shift, scale):
    return np.clip((values-shift)/scale, -10., 10.)


def soft_cross_entropy(scores, labels, teacher, mixture, positive_weights):
    hard = F.binary_cross_entropy_with_logits(scores, labels, pos_weight=positive_weights)
    if not mixture:
        return hard
    t = 2.
    soft = torch.sigmoid(torch.logit(teacher.clamp(1e-6,1-1e-6))/t)
    distillation = F.binary_cross_entropy_with_logits(scores/t, soft)*t*t
    return (1-mixture)*hard+mixture*distillation


def fit_students(train, y, t, test, test_y, cal, review, output):
    center=train.mean(axis=0)
    scale=np.maximum(train.std(axis=0),.025)
    train_x=torch.as_tensor(normalized(train,center,scale),device="cuda")
    labels=torch.as_tensor(y,dtype=torch.float32,device="cuda")
    soft=torch.as_tensor(t,dtype=torch.float32,device="cuda")
    test_x=torch.as_tensor(normalized(test,center,scale),device="cuda")
    positive=torch.tensor([1.,3.],device="cuda")
    candidates={}
    for name,(drop,decay,mix) in STUDENTS.items():
        torch.manual_seed(SEED)
        model=nn.Sequential(nn.Linear(train.shape[1],HIDDEN), nn.ReLU(), nn.Dropout(drop), nn.Linear(HIDDEN,2)).cuda()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=decay)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,EPOCHS,eta_min=.00025)
        loss_history=[]
        for epoch in range(EPOCHS):
            order=torch.randperm(len(train),generator=torch.Generator().manual_seed(SEED+epoch),device="cpu").cuda()
            loss_average=[]
            for part in order.split(768):
                optimizer.zero_grad(set_to_none=True)
                loss=soft_cross_entropy(model(train_x[part]),labels[part],soft[part],mix,positive)
                loss.backward()
                optimizer.step()
                loss_average.append(loss.detach())
            schedule.step()
            loss_history.append(float(torch.stack(loss_average).mean().cpu()))
        model.eval()
        with torch.inference_mode():
            prediction=model(test_x).sigmoid().cpu().numpy()
        thresholds=calibrated(test_y[cal],prediction[cal])
        candidates[name]=dict(probability=prediction, thresholds=thresholds,
            calibration=metric(test_y[cal],prediction[cal]>thresholds),
            review=metric(test_y[review],prediction[review]>thresholds),
            losses=loss_history)
        np.savez(output/f"{name}.npz",center=center,scale=scale,
                 weight0=model[0].weight.detach().cpu().numpy(),bias0=model[0].bias.detach().cpu().numpy(),
                 weight1=model[-1].weight.detach().cpu().numpy(),bias1=model[-1].bias.detach().cpu().numpy())
        print("early",name,candidates[name]["calibration"]["mean"]["f1"],
              candidates[name]["review"]["mean"]["f1"],flush=True)
    choice=max(candidates,key=lambda k:candidates[k]["calibration"]["mean"]["f1"])
    return choice, candidates


def costs(widths):
    width=widths.astype(np.float64)
    d,f=768,3072
    prefix=DEPTH*(2*width*(4*d*d+2*d*f)+4*width*width*d)/1e9
    late=[np.zeros(len(width))]
    for fraction in (.50,.65):
        queried=np.minimum(width,np.maximum(8.,np.ceil(width*fraction)))
        late.append((6-DEPTH)*(4*(width+queried)*d*d+4*queried*d*f+4*width*queried*d)/1e9)
    return prefix,np.column_stack(late)


def marginal_target(y, decisions):
    pred=decisions[:,0,:]
    totals=np.maximum(1.,y.sum(0)+pred.sum(0))
    per_label_f1=(2*(y*pred).sum(0))/totals
    # First-order effect of changing one decision on the two-label mean F1.
    utility=(2*y-per_label_f1)/totals
    return ((decisions.astype(np.float32)-pred[:,None,:])*utility[:,None,:]).sum(axis=2)*len(y)*.5


def decide(gains, prices, penalty, unit):
    return np.argmax(gains-penalty*prices/unit,axis=1)


def candidate_rows(y,prediction,prefix,late,estimates,cutoffs):
    unit=float(np.mean(late[:,1]))
    indic=np.arange(len(y))
    rows=[]
    for penalty in PENALTIES:
        actions=decide(estimates,late,penalty,unit)
        f1=metric(y,prediction[indic,actions])["mean"]["f1"]
        volume=(prefix+late[indic,actions]).mean()
        # Project the padded-operation ratio from the prior measured 768-token run.
        reference=float(np.mean(prefix+late[:,2]))
        points=.6*f1-.2*(volume/reference)*509236/3e6
        rows.append(dict(kind="learned",penalty=penalty,unit=unit,score=points,
                         f1=f1,mean_row_gflops=volume,stop=float((actions==0).mean())))
    for lower in (.055,.08,.12,.16,.22,.30):
        early_score=np.asarray(cutoffs[0])
        margin=np.abs(early_score-cutoffs[1])
        actions=np.where(margin[:,0]>=lower,0,2)
        f1=metric(y,prediction[indic,actions])["mean"]["f1"]
        volume=(prefix+late[indic,actions]).mean()
        reference=float(np.mean(prefix+late[:,2]))
        points=.6*f1-.2*(volume/reference)*509236/3e6
        rows.append(dict(kind="margin",vulnerability_margin=lower,score=points,
                         f1=f1,mean_row_gflops=volume,stop=float((actions==0).mean())))
    return rows


def apply_rule(config, early, thresholds, gains, price):
    if config["kind"]=="margin":
        return np.where(np.abs(early[:,0]-thresholds[0])>=config["vulnerability_margin"],0,2)
    return decide(gains,price,config["penalty"],config["unit"])


def main(output):
    cache=HERE/"cache"
    output.mkdir(parents=True,exist_ok=True)
    with np.load(cache/"fit.npz") as fitting, np.load(cache/"holdout.npz") as h, \
            np.load(HERE.parent/"round8/frozen/holdout.npz") as grouped:
        if not np.array_equal(grouped["rows"],h["ids"]) or not np.array_equal(grouped["gold"],h["y"]):
            raise ValueError("Stored role assignments do not match the official cache")
        if np.intersect1d(fitting["ids"],h["ids"]).size:
            raise ValueError("The student and gate folds overlap")
        fit,cal,review=grouped["folds"]<=2,grouped["folds"]==3,grouped["folds"]==4
        x=h["features"].astype(np.float32)
        y=h["y"]
        early_name,students=fit_students(fitting["features"].astype(np.float32),fitting["y"],fitting[ACTIONS[1]],
                                         x,y,cal,review,output)
        early=students[early_name]["probability"]
        thresholds=students[early_name]["thresholds"]
        predictions=np.stack((early>thresholds,*(h[name]>.5 for name in ACTIONS)),axis=1)
        m=features_for_router(x,early,thresholds)
        shift=m[fit].mean(axis=0)
        scale=m[fit].std(axis=0).clip(.025)
        standardized=normalized(m,shift,scale)
        utility=marginal_target(y[fit],predictions[fit])
        gate=Ridge(alpha=55.).fit(standardized[fit],utility[:,1:])
        gains=np.column_stack((np.zeros(len(m)),gate.predict(standardized)))
        prefix,late=costs(h["widths"])
        candidates=candidate_rows(y[cal],predictions[cal],prefix[cal],late[cal],gains[cal],(early[cal],thresholds))
        quality=metric(y[cal],predictions[cal,2])["mean"]["f1"]
        near=[r for r in candidates if r["f1"]>=quality-.0025 and r["stop"]>=.10]
        options={"score":max(candidates,key=lambda r:r["score"])}
        if near:
            options["quality"]=max(near,key=lambda r:r["score"])
        holdout={}
        for key,rule in options.items():
            action=apply_rule(rule,early,thresholds,gains,late)
            holdout[key]=dict(rule=rule,review=metric(y[review],predictions[np.arange(len(y)),action][review]),
                              counts=np.bincount(action[review],minlength=3).tolist())
        meta={name:{k:v for k,v in value.items() if k!="probability"} for name,value in students.items()}
        manifest=json.loads((cache/"manifest.json").read_text())
        write_json(output/"policy.json",dict(
            encoder_sha256=manifest["encoder_sha256"],token_config_sha256=manifest["token_config_sha256"],
            student=early_name,thresholds=thresholds,students=meta,
            gate=dict(shift=shift.tolist(),scale=scale.tolist(),weight=gate.coef_.tolist(),
                      bias=gate.intercept_.tolist()),options=holdout,candidates=candidates,
            partition={"gate_fit":int(fit.sum()),"calibration":int(cal.sum()),"review":int(review.sum())}))
        np.savez_compressed(output/"holdout.npz",rows=h["ids"],y=y,early=early,late=predictions[:,1:],
                            gains=gains,folds=grouped["folds"],width=h["widths"])
        print("selected student",early_name,"selected rules",
              {k:(v["rule"],v["review"]["mean"]["f1"]) for k,v in holdout.items()},flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=HERE/"frozen")
    args=parser.parse_args()
    start=time.perf_counter()
    main(args.output)
    print("fit seconds",time.perf_counter()-start,flush=True)
