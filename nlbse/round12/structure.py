"""Classify joint labels using layout, syntax topology and identifier-role signatures."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import importlib.metadata
import json
from pathlib import Path
import re
import sys
import time

import joblib
import numpy as np
from scipy.sparse import save_npz, load_npz
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import normalize

from project_split import HERE, ROOT, TOKEN, checksum, read_partition
sys.path.insert(0,str(HERE/"vendor"))
from tree_sitter import Language, Parser
import tree_sitter_cpp
from pipeline import TRAIN, PUBLIC, load, write_json

OUT=HERE/"structure"
HASHER = FeatureHasher(n_features=2**17,input_type="dict",dtype=np.float32,alternate_sign=True)
LAYOUT_NAMES=("chars","lines","mean_line_width","max_line_width","indent_max","indent_mean","identifiers","unique_identifiers",
              "call_shapes","number_tokens","upper_tokens","underscores","comment_lines","comments_chars","operators",
              "punctuation","entry_braces","exit_braces","strings","repeat_ratio")
AST_TYPES=("function_definition","function_declarator","compound_statement","if_statement","else_clause","switch_statement",
           "case_statement","for_statement","while_statement","do_statement","return_statement","break_statement",
           "continue_statement","goto_statement","call_expression","assignment_expression","update_expression",
           "unary_expression","binary_expression","conditional_expression","subscript_expression","field_expression",
           "pointer_expression","cast_expression","declaration","init_declarator","parameter_declaration","identifier",
           "number_literal","string_literal","preproc_if","preproc_def","ERROR","comment")
AST_NAMES=(*AST_TYPES,"tree_depth","nodes","ancestor_conditional_mean","uses_in_header","lexical_repeat_in_branches",
           "identifiers_first_third","identifiers_reused_across_thirds","has_error","largest_statement_bytes","type_diversity")
KEYWORDS=frozenset("auto bool break case char class const continue default delete do double else enum false float for goto if inline int long namespace new nullptr return short signed sizeof static struct switch template true typedef union unsigned using void volatile while NULL".split())
IDENTIFIER=re.compile(r"^[A-Za-z_][A-Za-z_0-9]*$")
CONDITIONALS=frozenset({"if_statement","while_statement","switch_statement","for_statement","conditional_expression"})


def lexical(text, *, with_motifs=True):
    stems=tuple(TOKEN.finditer(text))
    tokens=[]
    comments=[]
    for occurrence in stems:
        if occurrence["comment"] is not None:
            comments.append(occurrence[0])
        else:
            tokens.append(occurrence[0])
    identifiers=[tok for tok in tokens if IDENTIFIER.fullmatch(tok) and tok not in KEYWORDS]
    shapes=[]
    distinct={}
    calls=0
    for i,tok in enumerate(tokens):
        if IDENTIFIER.fullmatch(tok) and tok not in KEYWORDS:
            is_call=i+1<len(tokens) and tokens[i+1]=="("
            calls+=is_call
            if is_call:
                shape="call"
            elif tok not in distinct:
                distinct[tok]=len(distinct)+1
                shape="new-id"
            else:
                shape="repeat-id"
            shapes.append(shape)
        elif tok[0].isdigit():
            shapes.append("number")
        elif tok.startswith(('"',"'",'R"')):
            shapes.append("literal")
        else:
            shapes.append(tok)
    lengths=[len(line) for line in text.splitlines()] or [0]
    indents=[len(line)-len(line.lstrip()) for line in text.splitlines()] or [0]
    values=(len(text),len(lengths),sum(lengths)/len(lengths),max(lengths),max(indents),sum(indents)/len(indents),
            len(identifiers),len(distinct),calls,sum(t[0].isdigit() for t in tokens),sum(t.isupper() for t in identifiers),
            sum("_" in t for t in identifiers),sum(c.count("\n")+1 for c in comments),sum(map(len,comments)),
            sum(t in "+-*/|^&=!<>%" for t in tokens),sum(t in ",;()[]{}" for t in tokens),
            tokens.count("{"), tokens.count("}"),shapes.count("literal"),len(distinct)/max(len(identifiers),1))
    motifs=Counter()
    for n in (2,4) if with_motifs else ():
        motifs.update(f"lex{n}:"+"/".join(shapes[i:i+n]) for i in range(len(shapes)-n+1))
    return values,motifs


def syntax(text,parser, *, with_motifs=True):
    data=text.encode()
    root=parser.parse(data).root_node
    depth_max=0
    counters=Counter()
    motifs=Counter()
    id_positions={}
    conditions=0
    latest_statement=0
    starts=[(root,0,"",False)]
    while starts:
        node,depth,parent,in_cond=starts.pop()
        kind=node.type
        counters[kind]+=1
        depth_max=max(depth_max,depth)
        if parent and with_motifs:
            motifs[f"ast:{parent}/{kind}"]+=1
            motifs[f"pos:{min(depth//4,6)}:{kind}"]+=1
        if kind=="comment":
            continue
        if kind.endswith("statement"):
            latest_statement=max(latest_statement,node.end_byte-node.start_byte)
        if kind=="identifier":
            name=data[node.start_byte:node.end_byte]
            which=min(2,3*node.start_byte//max(1,len(data)))
            id_positions[name]=id_positions.get(name,0)|(1<<which)
            conditions+=in_cond
        more=in_cond or kind in CONDITIONALS
        starts.extend((child,depth+1,kind,more) for child in node.named_children)
    n=sum(counters.values())
    uses=[v for v in id_positions.values()]
    extra=(depth_max,n,conditions/max(n,1),sum(v&1>0 for v in uses)/max(len(uses),1),
           sum(v!=0 and v&(v-1)!=0 for v in uses)/max(len(uses),1),sum(v&1>0 for v in uses),
           sum(v>1 and v&1 for v in uses),int(root.has_error),latest_statement,len(counters))
    return tuple(counters[t] for t in AST_TYPES)+extra,motifs


def extract_block(texts):
    parser=Parser(Language(tree_sitter_cpp.language()))
    dense=[]
    lexical_motifs=[]
    syntax_motifs=[]
    for text in texts:
        layout, words=lexical(text)
        ast, families=syntax(text,parser)
        dense.append((*layout,*ast))
        lexical_motifs.append(words)
        syntax_motifs.append(families)
    a=HASHER.transform(lexical_motifs)
    b=HASHER.transform(syntax_motifs)
    return np.asarray(dense,dtype=np.float32),a,b


def assemble(texts, workers=1):
    from scipy.sparse import vstack
    jobs=[texts[i:i+1024] for i in range(0,len(texts),1024)]
    if workers>1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts=list(pool.map(extract_block,jobs))
    else:
        parts=list(map(extract_block,jobs))
    dense=np.log1p(np.concatenate([d for d,_,_ in parts]))
    sparse=[]
    for idx in (1,2):
        x=vstack([v[idx] for v in parts],format="csr")
        x.data[:]=np.sign(x.data)*np.log1p(np.abs(x.data))
        sparse.append(normalize(x,copy=False))
    return dense,*sparse


def cache(part,workers):
    split=read_partition()
    texts,y,_=load(PUBLIC if part=="public" else TRAIN)
    ids=np.arange(len(texts),dtype=np.int32) if part=="public" else split[part]
    functions=[texts[i] for i in ids]
    OUT.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    dense,token,ast=assemble(functions,workers)
    elapsed=time.perf_counter()-start
    np.savez_compressed(OUT/f"{part}.npz",ids=ids,dense=dense,y=y[ids])
    save_npz(OUT/f"{part}_token.npz",token)
    save_npz(OUT/f"{part}_ast.npz",ast)
    write_json(OUT/f"{part}.json",dict(rows=len(ids),seconds=elapsed,dense_columns=[*LAYOUT_NAMES,*AST_NAMES],
               cxx_parser=importlib.metadata.version("tree-sitter-cpp"),workers=workers,
               token_nnz=token.nnz,ast_nnz=ast.nnz,parse_error_fraction=float((dense[:,len(LAYOUT_NAMES)+len(AST_TYPES)+7]>0).mean())))
    print(part,len(ids),round(elapsed,2),token.nnz,ast.nnz,flush=True)


def fit(part="fit",prefix="project"):
    with np.load(OUT/f"{part}.npz") as data:
        y=data["y"][:,0]
    all_models={}
    for role in ("token","ast"):
        matrix=load_npz(OUT/f"{part}_{role}.npz")
        model=SGDClassifier(loss="log_loss",alpha=3e-6,max_iter=40,random_state=42)
        model.fit(matrix,y)
        all_models[role]=model
    dest=OUT/f"{prefix}_linear.joblib"
    joblib.dump(all_models,dest,compress=3)
    write_json(dest.with_suffix(".json"),dict(sha256=checksum(dest),finite_dim=2**17,fit=part,rows=len(y)))
    print(dest.name,len(y),flush=True)


def probabilities(part,prefix="project"):
    dest=OUT/f"{prefix}_linear.joblib"
    if checksum(dest)!=json.loads(dest.with_suffix(".json").read_text())["sha256"]:
        raise ValueError("Structural model changed")
    models=joblib.load(dest)
    p=np.column_stack([models[role].predict_proba(load_npz(OUT/f"{part}_{role}.npz"))[:,1] for role in ("token","ast")])
    return p


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest="stage",required=True)
    cache_args=sub.add_parser("cache")
    cache_args.add_argument("part",choices=("fit","probe","dev","test","public"))
    cache_args.add_argument("--workers",type=int,default=4)
    sub.add_parser("fit")
    args=ap.parse_args()
    if args.stage=="cache":
        cache(args.part,args.workers)
    else:
        fit()
