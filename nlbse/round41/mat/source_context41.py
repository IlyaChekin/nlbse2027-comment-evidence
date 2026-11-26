"""Analyze scoped leading-comment attachment in pinned upstream files, offline only."""
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import sys
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(ROOT/'round31/mat'))
from transfer import canonical_mat, checksum, dataset, split, write_json
from placement import cutoff, table
from marker import upstream_tag

sys.path.insert(0,str(ROOT/'round18/mat_residual'))
from source_audit import SOURCES, fetch, locate, stream

DEST = HERE/'source_context'
DOC = re.compile(r'^(?:/\*[* !]|//[/!])')
MARK = re.compile(r'(?i)(?<![a-z_])(?:todo|fixme|xxx|hack)(?![a-z_])')
EXPLICIT = re.compile(r'(?m)^\s*[*/! ]*[@\\](?:fn|function)\s+([^\r\n]+)')
NAME = re.compile(r'\b([A-Za-z_]\w*)\s*\(')
MODES = ('strict_exact','owner_exact','owner_aligned')


def qualified_comment(text, comment, name):
    line_start = text.rfind('\n',0,comment.start())+1
    # Exclude tail comments attached to a previous declaration or statement.
    if text[line_start:comment.start()].strip() or not upstream_tag(comment[0]):
        return 0.
    declared = EXPLICIT.search(comment[0])
    if declared and (not name or re.search(r'\b'+re.escape(name)+r'\b',declared[1]) is None):
        return 0.
    if declared:
        return 8.
    match = MARK.search(comment[0])
    if not match:
        return 1.
    line = comment[0][:match.start()].rsplit('\n',1)[-1].strip('/! *\t')
    return 5. if not line and DOC.match(comment[0]) else 4. if not line else 3.


def attach(text, comments, region):
    begin,end,prior = region
    header = text[begin:end].split('{',1)[0]
    names = NAME.findall(header)
    name = names[-1] if names else None
    internal, leading = 0., 0.
    if any(upstream_tag(c[0]) for c in comments if begin<=c.start()<end):
        internal = 7.
    cursor = begin
    for item in reversed(comments):
        if item.end()>cursor:
            continue
        if item.start()<prior:
            break
        between = text[item.end():cursor]
        if between.strip() or between.count('\n')>2:
            break
        rank = qualified_comment(text,item,name)
        if rank:
            leading = max(leading,rank)
        cursor = item.start()
    return internal,leading


def anchors(tokens, hay, starts, query):
    signature = tuple(v[0] for v in query)
    size = len(signature)
    if size<26:
        return []
    prefix,suffix = signature[:12],signature[-12:]
    limit = min(42,max(3,int(.14*size)))
    found = []
    for start in starts[prefix[0]]:
        if hay[start:start+12]!=prefix:
            continue
        for delta in range(-limit,limit+1):
            end = start+size+delta
            if end>len(hay) or end-start<24 or hay[end-12:end]!=suffix:
                continue
            ratio = SequenceMatcher(None,signature,hay[start:end],autojunk=False).ratio()
            if ratio>=.94:
                found.append((ratio,(tokens[start][1],tokens[end-1][2],tokens[start-1][2] if start else 0)))
                if len(found)>1:
                    return []
    return found


def process(key, rows, previous, texts):
    project, path = key
    cache = DEST/'cache'
    cache.mkdir(exist_ok=True)
    stamped = cache/(hashlib.sha256((project+'\0'+path).encode()).hexdigest()+'.source')
    started = perf_counter()
    if stamped.exists() and previous.get('sha256')==checksum(stamped):
        source = stamped.read_bytes().decode('utf-8',errors='surrogateescape')
        receipt = previous
    else:
        source,receipt = fetch(project,path)
        if source is not None:
            raw = source.encode('utf-8',errors='surrogateescape')
            if previous.get('sha256') and receipt['sha256']!=previous['sha256']:
                raise ValueError('Pinned upstream snapshot changed')
            assert hashlib.sha256(raw).hexdigest()==receipt['sha256']
            stamped.write_bytes(raw)
    out = []
    if source is not None:
        tokens, comments = stream(source)
        hay = tuple(t[0] for t in tokens)
        starts = defaultdict(list)
        for i,t in enumerate(hay):
            starts[t].append(i)
        for row in rows:
            query,_ = stream(texts[row])
            matches = locate(tokens,hay,starts,query)
            outcome, rank = 'multiple' if matches else 'unmatched',[0.]*3
            if len(matches)==1:
                inside,lead = attach(source,comments,matches[0])
                rank = [max(inside,lead),max(inside,lead) if inside or lead>=4. else 0.,max(inside,lead)]
                outcome = 'exact'
            elif not matches:
                alternative = anchors(tokens,hay,starts,query)
                if alternative:
                    similarity, span = alternative[0]
                    inside,lead = attach(source,comments,span)
                    rank[2] = (similarity*max(inside,lead)) if inside or lead>=4. else 0.
                    outcome = 'aligned'
            out.append((int(row),outcome,rank))
    return dict(project=project,file=path,seconds=perf_counter()-started,**receipt),out


def main():
    DEST.mkdir(exist_ok=True)
    texts,labels,projects,files = dataset()
    gold = labels[:,1].astype(bool)
    parts = split()
    roles = ('dev','test_a','test_b')
    old = ROOT/'round18/mat_residual/source_audit.json'
    source_cohort = json.loads(old.read_text())
    inventory = {(d['project'],d['file']):{k:v for k,v in d.items() if k not in ('project','file')}
                 for d in source_cohort['source_receipts']}
    jobs = defaultdict(list)
    for role in roles:
        for i in parts[role]:
            key = (projects[i],files[i])
            if key in inventory:
                jobs[key].append(i)
    receipts, entries = [],{}
    def one(job):
        key,rows = job
        return process(key,rows,inventory[key],texts)
    with ThreadPoolExecutor(max_workers=12) as pool:
        for receipt,observed in pool.map(one,sorted(jobs.items())):
            receipts.append(receipt)
            for row,status,ranks in observed:
                assert row not in entries
                entries[row] = (status,ranks)
    before = sum(len(a) for a in jobs.values())
    print('inventory',len(jobs),before,'located_file',len(entries),flush=True)
    frozen,summaries = {},{}
    counters = {}
    for role in roles:
        ids = parts[role]
        baseline = canonical_mat([texts[i] for i in ids])
        visible = np.array([bool(stream(texts[i])[1]) for i in ids])
        ranks = np.array([entries.get(i,('unavailable',[0.]*3))[1] for i in ids]).T
        score = dict(zip(MODES,ranks,strict=True))
        if role=='dev':
            frozen = {k:cutoff(gold[ids],baseline,r) for k,r in score.items()}
        summaries[role] = table(role,ids,gold[ids],projects[ids],score,visible,baseline,frozen,dest=DEST)
        counters[role] = {}
        for cat,mask in (('miss',gold[ids] & ~baseline),('negative',~gold[ids])):
            counters[role][cat] = dict(Counter(entries[i][0] for i in ids[mask] if i in entries))
        print(role,counters[role],{k:[v['frozen']['recovered'],v['frozen']['delta_fp'],v['matched']['recovered']]
                                   for k,v in summaries[role]['results'].items()},flush=True)
    write_json(DEST/'receipt.json',dict(inventory_origin_sha256=checksum(old),
               file_selection='Previously audited files, originally selected for presence of an eligible miss',
               prediction_inputs='original Projectname and Filepath plus pinned upstream source, research diagnostic only',
               mode=MODES,frozen=frozen,sources=SOURCES,receipts=receipts,
               partition_sha256=checksum(ROOT/'round13/partition.npz'),observations=entries,
               sample_summary=counters,code_sha256=checksum(Path(__file__))))
    write_json(DEST/'results.json',summaries)


if __name__=='__main__':
    main()
