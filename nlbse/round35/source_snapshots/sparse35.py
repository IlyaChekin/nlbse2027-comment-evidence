"""Compare low-cost character and token models on the same matched code examples."""
import argparse
from itertools import product

import joblib
import numpy as np
from scipy.sparse import vstack
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression

from teacher35 import HERE, PARTS, f, checksum, json_at, write_json, opening
from eval35 import loaded, reference, signatures, crop, point_classifier, design, comparisons

DEST=HERE/'sparse'
MODES=('symbol','characters','body')
REGIONS=1536
C=2.0


def vectorizer(mode):
    if mode=='symbol':
        return HashingVectorizer(token_pattern=r"\w+|[^\w\s]",ngram_range=(1,3),lowercase=False,
                                 n_features=2**18, dtype=np.float32,sublinear_tf_missing=False) if False else HashingVectorizer(
                                 token_pattern=r"\w+|[^\w\s]",ngram_range=(1,3),lowercase=False,
                                 n_features=2**18,dtype=np.float32)
    return HashingVectorizer(analyzer='char',ngram_range=(3,5),lowercase=False,n_features=2**18,dtype=np.float32)


def view(mode,text):
    if mode=='body':
        position=opening(text)
        text=text[:min(position,320)]+text[position:position+REGIONS]
    if len(text)>2*REGIONS:
        return text[:REGIONS]+'\n'+text[-REGIONS:]
    return text


def data_matrix(texts,rows,shape,mode):
    frame=[view(mode,texts[i]) for i in rows['ids'][rows['support']]]
    return shape.transform(frame)


def train(x,a,b,gold,margin,mode):
    ranker=LogisticRegression(C=C,solver='liblinear',fit_intercept=False,random_state=2026,max_iter=180)
    if mode=='point':
        weight=.25+2.*np.exp(-.8*abs(margin))
        return ranker.fit(x,gold,sample_weight=weight)
    difference=x[a]-x[b]
    feed=vstack((difference,-difference),format='csr')
    y=np.arange(2*len(a))<len(a)
    _,inverse,per=np.unique(a,return_inverse=True,return_counts=True)
    weight=1./per[inverse]
    return ranker.fit(feed,y,sample_weight=np.tile(weight,2))


def evaluate(rotation):
    data,ref,source=loaded(rotation)
    _,baseline=reference(rotation,data,ref)
    d=data['dev']
    z=data.get('test',d)
    oof=data['oof']
    a,b=signatures(rotation,oof)
    sample=np.flatnonzero(oof['support'])
    x,y=np.searchsorted(sample,a),np.searchsorted(sample,b)
    assert np.array_equal(sample[x],a) and np.array_equal(sample[y],b)
    fit,cal,held=crop(d,d['fit']),crop(d,d['cal']),crop(z,z['held'])
    stages=(oof,fit,cal,held)
    texts=f.inputs()[0]
    hashes={}
    models={}
    values={}
    for mode in MODES:
        shape=vectorizer(mode)
        matrices=[data_matrix(texts,ds,shape,mode) for ds in stages]
        for variant in ('point','pair'):
            arm=train(matrices[0],x,y,oof['y'][sample],oof['queried'][sample],variant)
            extras=[]
            for q,m in zip(stages,matrices,strict=True):
                predicted=arm.decision_function(m)
                full=np.zeros(len(q['ids']))
                full[q['support']]=predicted
                extras.append(np.column_stack((design(q,ref['cutoff']),full)))
            name=mode+'_'+variant
            for r,p in enumerate(('oof','fit')):
                part=stages[r]
                cal_index=part['support']
                readout=point_classifier(extras[r],part['y'],part['latest'],cal_index)
                models[p,name]=dict(first=arm,readout=readout)
                values[p,name]=[readout.decision_function(qq) for qq in extras[2:]]
            hashes[name]=dict(features=matrices[0].nnz,rows=matrices[0].shape[0],nonzeros=int(np.count_nonzero(arm.coef_)))
            print(rotation,name,hashes[name],flush=True)
    settings,forecasts=comparisons(rotation,cal,held,values)
    dest=DEST/rotation
    dest.mkdir(parents=True,exist_ok=True)
    file=dest/'predictions.npz'
    np.savez_compressed(file,**forecasts)
    weight=dest/'readouts.joblib'
    joblib.dump(models,weight,compress=3)
    write_json(dest/'results.json',dict(rotation=rotation,baseline_sha256=baseline,windows_sha256=source['predictions_sha256'],
              options=settings,heads=hashes,code_sha256=checksum(HERE/'sparse35.py'),
              predictions_sha256=checksum(file),weights_sha256=checksum(weight)))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('rotation',choices=PARTS)
    evaluate(cli.parse_args().rotation)
