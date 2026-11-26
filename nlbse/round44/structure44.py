"""Extract name-free region structure and identifier-use roles for classification."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.sparse import save_npz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0,str(ROOT/'round43'))
from features43 import syntax, LITERAL, kind, vectorizer, without_literals
from context39 import whole_tokens
sys.path.insert(0,str(ROOT/'round41'))
from source41 import PARTS, f, checksum, json_at, load, write_json

DEST = HERE/'features'
ARMS = ('lexical','signature','body','middle','usage')
BODY = 96
SIGNATURE = 48
USAGE = 128
IGNORED = ('comment',)
IDENTIFIERS = ('identifier','field_identifier','type_identifier','namespace_identifier')
DECLARATIONS = ('function_declarator','declaration','init_declarator','parameter_declaration')
STATEMENTS = ('if_statement','for_statement','while_statement','switch_statement','return_statement',
              'call_expression','subscript_expression','assignment_expression','goto_statement')


def span_nodes(root):
    stack = [root]
    body = None
    tokens = []
    while stack:
        node = stack.pop()
        name = node.type
        if name=='function_definition':
            block = node.child_by_field_name('body')
            if block is not None and (body is None or block.end_byte-block.start_byte>body.end_byte-body.start_byte):
                body = block
        if name in IGNORED:
            continue
        if not node.child_count or name in LITERAL:
            if not node.is_missing:
                tokens.append(node)
        else:
            stack.extend(reversed(node.children))
    return body,tokens


def path(node,limit=5):
    out = []
    while node is not None and len(out)<limit:
        out.append(kind(node))
        node = node.parent
    return out


def tree_terms(leaves):
    out = set()
    sequence = []
    for leaf in leaves:
        p = path(leaf)
        sequence.append(p[0])
        for n in (2,3,5):
            if len(p)>=n:
                out.add('path:'+'>'.join(p[:n]))
        out.update('ancestor:'+a+'>'+b for a,b in zip(p[1:],p[2:]))
    for n in (1,2,3):
        out.update('leaf:'+'>'.join(sequence[i:i+n]) for i in range(len(sequence)-n+1))
    return ' '.join(sorted(out))


def middle(pool,n):
    begin = (len(pool)-n)//2
    return pool[begin:begin+n]


def body_sample(nodes):
    if len(nodes)<=BODY:
        return nodes
    # Keep equal budgets for declaration/body controls. The body view includes its
    # boundaries and leaves nearest simple control, call and assignment constructs.
    chosen = set(range(BODY//4))
    chosen.update(range(len(nodes)-BODY//4,len(nodes)))
    eligible = [i for i,x in enumerate(nodes) if i not in chosen and x.type in IDENTIFIERS]
    important = [i for i in eligible if any(k in STATEMENTS for k in path(nodes[i],4)[1:])]
    first = important if important else eligible
    if first:
        idx = np.linspace(0,len(first)-1,min(BODY//2,len(first)),dtype=int)
        chosen.update(first[i] for i in idx)
    missing = BODY-len(chosen)
    if missing:
        other = [i for i in range(len(nodes)) if i not in chosen]
        idx = np.linspace(0,len(other)-1,missing,dtype=int)
        chosen.update(other[i] for i in idx)
    assert len(chosen)==BODY
    return [nodes[i] for i in sorted(chosen)]


def usage_terms(raw,header,body):
    # Identifier spellings link occurrences within a function and never become features.
    prior = defaultdict(list)
    channels = set()
    surrounding = header[:SIGNATURE]+body[:USAGE]
    for i,node in enumerate(surrounding):
        if node.type not in IDENTIFIERS:
            continue
        token = raw[node.start_byte:node.end_byte]
        p = path(node,5)
        typ = 'decl' if any(k in DECLARATIONS for k in p[1:]) else (p[1] if len(p)>1 else p[0])
        section = 'header' if i<len(header[:SIGNATURE]) else 'body'
        current = f'{section}:{p[0]}:{typ}'
        channels.add('role:'+current)
        if prior[token]:
            j,before = prior[token][-1]
            gap = i-j
            band = 0 if gap<=4 else 1 if gap<=12 else 2 if gap<=36 else 3
            channels.add(f'reuse:{before}>{current}:{band}')
            if len(prior[token])>=2:
                channels.add(f'repeat:{prior[token][0][1]}>{current}')
        prior[token].append((i,current))
    return ' '.join(sorted(channels))


def extract(text, requested=ARMS):
    results = {}
    if 'lexical' in requested:
        results['lexical'] = ' '.join(whole_tokens(without_literals(text)))
    if all(k=='lexical' for k in requested):
        return results,False,False
    tree,_ = syntax(text)
    body,tokens = span_nodes(tree.root_node)
    if body is None:
        signature,body_tokens = [],tokens
    else:
        signature = [x for x in tokens if x.end_byte<=body.start_byte]
        body_tokens = [x for x in tokens if body.start_byte<=x.start_byte and x.end_byte<=body.end_byte]
    if 'signature' in requested:
        results['signature'] = tree_terms(signature[:SIGNATURE])
    if 'body' in requested:
        results['body'] = tree_terms(body_sample(body_tokens))
    if 'middle' in requested:
        results['middle'] = tree_terms(middle(body_tokens,min(BODY,len(body_tokens))))
    if 'usage' in requested:
        results['usage'] = usage_terms(text.encode('utf-8',errors='replace'),signature,body_tokens)
    return results,tree.root_node.has_error,body is not None


def mapping(name):
    return vectorizer('text' if name=='lexical' else 'ast')


def main():
    DEST.mkdir(exist_ok=True)
    p = ROOT/'round43/parser.sha256'
    hashes = {}
    for need in PARTS:
        matrices,_,ticket = load(need)
        hashes[need] = ticket
    masks = []
    for need in PARTS:
        matrices,_,_ = load(need)
        for row in matrices.values():
            masks.append(row['ids'][row['observed_regime']==0])
    ids = np.unique(np.concatenate(masks))
    corpus = f.inputs()[0]
    start = perf_counter()
    with ProcessPoolExecutor(max_workers=6) as pool:
        bank = list(pool.map(extract,(corpus[i] for i in ids),chunksize=64))
    ready = perf_counter()-start
    data,error,parsed = zip(*bank,strict=True)
    np.savez_compressed(DEST/'rows.npz',ids=ids,parse_error=np.asarray(error),function_body=np.asarray(parsed))
    stats = {}
    for key in ARMS:
        x = mapping(key).transform(x[key] for x in data)
        path = DEST/(key+'.npz')
        save_npz(path,x)
        stats[key] = dict(nnz=x.nnz,sha256=checksum(path),nonempty=int(np.count_nonzero(np.diff(x.indptr))))
    write_json(DEST/'receipt.json',dict(rows=len(ids),structure_rows=int(sum(parsed)),
              partial_trees=int(sum(error)),extraction_seconds=ready,wall_seconds=perf_counter()-start,
              sources=hashes,features=stats,roster_sha256=checksum(DEST/'rows.npz'),
              parser_sha256=checksum(p),previous_extractor_sha256=checksum(ROOT/'round43/features43.py'),
              code_sha256=checksum(Path(__file__))))
    print('features',len(ids),sum(parsed),sum(error),ready,flush=True)


if __name__=='__main__':
    main()
