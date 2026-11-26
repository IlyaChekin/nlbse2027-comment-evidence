"""Transfer marker supervision via comment language and signatures, using Function only."""
from collections import Counter
from pathlib import Path
import csv
import re
import sys
from time import perf_counter

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(ROOT/'round31/mat'))
from transfer import dataset, split, checksum, write_json, canonical_mat, TOKEN
from marker import LETTERS, TAGS, LICENSE, upstream_tag
from placement import cutoff, table

PARTS = ('dev','test_a','test_b')
WORD = re.compile(r'[a-zA-Z]+|\d+')
NORMALIZE = re.compile(r'([a-z])([A-Z])')
SPAN = re.compile(r'//[^\r\n]*|/\*.*?(?:\*/|\Z)',re.S)
MARKERS = ('todo','fixme','hack','xxx')
NORM_TAG = re.compile(r'(?i)\b(?:todo|fixme|hack|xxx)\b')
DEBT = re.compile(r'\b(?:unimplemented|placeholder|workaround|temporary|unsupported|'
                  r'kludge|pending|deprecated|stub|should be|need to|needs to|'
                  r'not implemented|for now|fix this|hard coded|broken)\b')
PARSE = HashingVectorizer(n_features=2**17,analyzer='word',tokenizer=str.split,token_pattern=None,
                          lowercase=False,ngram_range=(1,3),dtype=np.float32)
SHAPE = HashingVectorizer(n_features=2**17,analyzer='char',ngram_range=(3,5),dtype=np.float32)


def marker_token(raw):
    token = LETTERS.sub('',raw).lower()
    return 2<len(token)<20 and (
        (token.startswith(TAGS) or token.endswith(TAGS))
        and ('xxx' not in token or token=='xxx'))


def comment(phrase):
    # Omit the assigned tag and its separator while preserving the admission itself.
    words = [term.lower() for raw in phrase.split() if not marker_token(raw)
             for term in WORD.findall(raw)]
    return ' '.join('0' if word.isdigit() else word for word in words)


def lexical_variants(text, docs):
    known = [span for span in docs if upstream_tag(span[0])]
    def licensed_line():
        for block in docs:
            if LICENSE.search(block[0]):
                for line in block[0].splitlines():
                    if NORM_TAG.search(line) and upstream_tag(line):
                        return True
        return False
    return (bool(known),licensed_line(),any(upstream_tag(s[0]) for s in docs if s[0].endswith('\\')),
            any(DEBT.search(comment(s[0])) for s in docs))


def observe(text):
    tokens = list(TOKEN.finditer(text))
    docs = [t for t in tokens if t['comment'] is not None]
    signature, statements, admissions = [], [], []
    depth, body, tagged_preceding = 0, False, False
    for token in tokens:
        raw = token[0]
        if token['comment'] is not None:
            if not LICENSE.search(raw):
                content = comment(raw)
                if content:
                    admissions.append((content,upstream_tag(raw),not body))
            tagged_preceding |= not signature and not body and upstream_tag(raw)
            continue
        if not body and raw=='{':
            body = True
        depth += int(raw=='{')-int(raw=='}')
        if raw[0].isalpha() or raw[0]=='_':
            normalized = NORMALIZE.sub(r'\1 \2', raw)
            terms = [w.lower() for w in WORD.findall(normalized)]
        elif raw.startswith(('"',"'")):
            terms = ['string'] + [w.lower() for w in WORD.findall(raw)[:8]]
        elif raw[0].isdigit():
            terms = [raw if raw in ('0','1') else 'number']
        else:
            terms = [raw]
        if body:
            statements.extend(terms)
            if raw in ('return','goto','#','if','else','break','continue'):
                statements.append(f'flow_{raw}_{min(depth,6)}')
        else:
            signature.extend(terms)
    body_text = ' '.join(statements[:160]+statements[-80:] if len(statements)>240 else statements)
    sig_text = ' '.join(signature[:100])
    return admissions,sig_text,body_text,tagged_preceding,lexical_variants(text,docs)


def fit_matrix(texts, char=False):
    lexical = PARSE.transform(texts)
    return hstack((lexical,SHAPE.transform(texts)),format='csr') if char else lexical


def source_matrix(rows, full):
    signatures = fit_matrix([r[1] for r in rows],char=True)
    if not full:
        return signatures
    return hstack((signatures,fit_matrix([r[2] for r in rows])),format='csr')


def estimate(train, y, base, observed, receipts):
    pos = [adm for row in observed for adm,tag,_ in row[0] if tag]
    neg = [adm for row in observed for adm,tag,_ in row[0] if not tag]
    pos = sorted(set(pos)-set(neg))
    neg = sorted(set(neg)-set(pos))
    heads = {}
    heads['comment'] = LogisticRegression(C=4.,solver='liblinear',random_state=41,max_iter=250)
    heads['comment'].fit(fit_matrix(pos+neg,char=True),np.r_[np.ones(len(pos)),np.zeros(len(neg))])
    # Baseline-negative real misses plus genuine leading markers erased from positive examples.
    leading = np.array([r[3] for r in observed])
    cohort = ~base | (y & leading)
    weight = np.ones(len(train))
    weight[y & leading] = .35
    source_found = np.array([receipts.get(i)=='marker_in_adjacent_preceding_comment' for i in train])
    weight[source_found & y & ~base] = 3.
    for full in (False,True):
        mode = 'signature_body' if full else 'signature'
        head = LogisticRegression(C=4.,solver='liblinear',random_state=41,max_iter=250)
        matrix = source_matrix([r for i,r in enumerate(observed) if cohort[i]],full)
        head.fit(matrix,y[cohort],sample_weight=weight[cohort])
        heads[mode] = head
    fit = dict(comment_positive=len(pos),comment_negative=len(neg),
               source_rows=int(cohort.sum()),natural_positive=int((y & ~base).sum()),
               provenance_weighted=int((y & ~base & source_found).sum()),
               erased_leading=int((y & base & leading).sum()))
    return heads,fit


def scores(heads, obs):
    result = {mode:head.decision_function(source_matrix(obs,mode=='signature_body'))
              for mode,head in heads.items() if mode!='comment'}
    docs = [(i,doc,leading) for i,r in enumerate(obs) for doc,tag,leading in r[0] if not tag]
    score = heads['comment'].decision_function(fit_matrix([doc for _,doc,_ in docs],char=True))
    comment_only = np.full(len(obs),-35.)
    comment_ownership = comment_only.copy()
    row = np.array([i for i,_,_ in docs])
    lead = np.array([leading for _,_,leading in docs])
    np.maximum.at(comment_only,row,score)
    np.maximum.at(comment_ownership,row,score+.6*lead)
    result.update(comment=comment_only,comment_placement=comment_ownership)
    result['joint'] = result['signature_body']+.6*np.maximum(-8.,comment_only)
    literal = np.asarray([r[4] for r in obs])
    result.update(lexical=literal[:,0].astype(float),license_line=literal[:,1].astype(float),
                  line_splice=literal[:,2].astype(float),debt_lexicon=literal[:,3].astype(float))
    return result


def main():
    dest = HERE/'text'
    dest.mkdir(exist_ok=True)
    started = perf_counter()
    x,labels,projects,_ = dataset()
    gold = labels[:,1].astype(bool)
    parts = split()
    receipt = ROOT/'round31/mat/fit_provenance.json'
    import json
    audit = json.loads(receipt.read_text())
    import hashlib
    assert audit['fit_ids_sha256']==hashlib.sha256(parts['fit'].tobytes()).hexdigest()
    provenance = {i['row']:i['result'] for i in audit['observations'] if i['gold']}
    assert np.isin(list(provenance),parts['fit']).all()
    train = parts['fit']
    observed = [observe(x[i]) for i in train]
    base = canonical_mat([x[i] for i in train])
    heads,fit = estimate(train,gold[train],base,observed,provenance)
    del observed
    model_path = dest/'heads.joblib'
    joblib.dump(heads,model_path,compress=3)
    print('fit',fit,round(perf_counter()-started,2),flush=True)
    summaries, frozen = {}, {}
    for part in PARTS:
        ids = parts[part]
        local = [x[i] for i in ids]
        obs = [observe(t) for t in local]
        old = canonical_mat(local)
        has_comment = np.asarray([bool(r[0]) for r in obs])
        pred = scores(heads,obs)
        if part=='dev':
            frozen = {k:cutoff(gold[ids],old,v) for k,v in pred.items()}
            write_json(dest/'policy.json',dict(training=fit,cutoffs=frozen,
                       fit_source_sha256=checksum(receipt),weights_sha256=checksum(model_path),
                       data_manifest_sha256=checksum(ROOT/'data/manifest.json'),
                       partition_sha256=checksum(ROOT/'round13/partition.npz'),
                       code_sha256=checksum(Path(__file__))))
        summaries[part] = table(part,ids,gold[ids],projects[ids],pred,has_comment,old,frozen,dest)
        print(part,{k:[v['frozen']['recovered'],v['frozen']['delta_fp'],v['matched']['recovered']]
                    for k,v in summaries[part]['results'].items()},flush=True)
    write_json(dest/'results.json',summaries)
    with (dest/'comparison.csv').open('w',newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(('arm','split','base_fn','base_fp','matched_tp','frozen_tp','frozen_fp','frozen_f1','ap_residual'))
        for part, data in summaries.items():
            for mode, row in data['results'].items():
                writer.writerow((mode,part,data['baseline']['fn'],data['baseline']['fp'],
                                 row['matched']['recovered'],row['frozen']['recovered'],
                                 row['frozen']['delta_fp'],row['frozen']['f1'],row['average_precision']))
    print('total_seconds',perf_counter()-started,flush=True)


if __name__=='__main__':
    main()
