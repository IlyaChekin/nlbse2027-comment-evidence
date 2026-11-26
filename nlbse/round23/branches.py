"""Extract identity-free local relations and matched unordered syntax counts."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import vstack
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import normalize

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"round12/vendor"))
from tree_sitter import Language, Parser
import tree_sitter_cpp

LANGUAGE = Language(tree_sitter_cpp.language())
HASHER = FeatureHasher(n_features=2**16,input_type="dict",dtype=np.float32,alternate_sign=True)
IDS = frozenset(("identifier","field_identifier","type_identifier"))
CALL = frozenset(("call_expression","new_expression","delete_expression"))
BRANCH = frozenset(("if_statement","switch_statement","while_statement","do_statement","for_statement",
                    "conditional_expression","case_statement"))
EXIT = frozenset(("return_statement","break_statement","continue_statement","goto_statement","throw_statement"))
EFFECT = frozenset(("assignment_expression","update_expression","init_declarator"))
OPERANDS = frozenset(("field_expression","subscript_expression","pointer_expression","binary_expression",
                      "unary_expression","cast_expression"))
ATOMS = IDS|frozenset(("number_literal","string_literal","char_literal","null","true","false"))
DENSE = ("chars","nodes","depth","parse_error","block_count","branch_count","call_count","max_calls_in_block",
         "branch_reads","branch_writes","branch_calls","branch_exits","condition_id_count",
         "condition_reuse","condition_reuse_write","condition_reuse_call","condition_reuse_exit",
         "repeat_access","block_steps","block_call_pair","call_arg_total","call_arg_max","branch_depth",
         "node_kinds","function_count")


def describe_call(node, source):
    fn = node.child_by_field_name("function")
    args = node.child_by_field_name("arguments")
    target = fn.type if fn is not None else node.type
    count = args.named_child_count if args is not None else 0
    return f"call:{target}:{min(5,count)}",count


def family(kind):
    if kind in CALL:
        return "call"
    if kind in EFFECT:
        return "write"
    if kind in EXIT:
        return kind.removesuffix("_statement")
    if kind in BRANCH:
        return "branch:"+kind.removesuffix("_statement")
    if kind in OPERANDS:
        return kind.removesuffix("_expression")
    return kind


def branch_fields(node):
    pairs = (("condition",node.child_by_field_name("condition")),("value",node.child_by_field_name("value")),
             ("then",node.child_by_field_name("consequence")),("else",node.child_by_field_name("alternative")),
             ("loop",node.child_by_field_name("body")))
    return [(k,c) for k,c in pairs if c is not None]


def extract(text,parser):
    source = text.encode()
    tree = parser.parse(source).root_node
    count, flat, linked = Counter(),Counter(),Counter()
    by_block = {}
    guards = {}
    used_under_guard = {}
    links = []
    depth_max = call_total = call_max = branch_depth = 0
    stack = [(tree,0,0,(),None,None)]
    while stack:
        node,depth,block,context,event,parent = stack.pop()
        kind = node.type
        if kind=="comment":
            continue
        count[kind]+=1
        depth_max = max(depth_max,depth)
        branch_depth = max(branch_depth,len(context))
        if kind=="compound_statement":
            block = node.id
            by_block[block] = []
        if kind in BRANCH:
            entries = branch_fields(node)
            if entries:
                guard = {child.id:role for role,child in entries}
                guards[node.id] = set(),kind
                used_under_guard[node.id] = set()
            else:
                guard = {}
        else:
            guard = {}
        if kind in CALL:
            event,n = describe_call(node,source)
            flat[event]+=1
            call_total+=n
            call_max = max(call_max,n)
        elif kind in EFFECT|EXIT|BRANCH|OPERANDS:
            event = family(kind)
        elif kind in ("declaration","expression_statement"):
            event = None
        if kind in CALL|EFFECT|EXIT|BRANCH and event:
            flat["event:"+event]+=1
            if block:
                by_block[block].append(event)
            if context:
                _, role, framing = context[-1]
                linked[f"under:{framing}:{role}:{event}"]+=1
        if kind in IDS:
            name = source[node.start_byte:node.end_byte]
            for mark,role,_ in context:
                if role in ("condition","value"):
                    guards[mark][0].add(name)
                else:
                    links.append((mark,role,event or "other",name))
        if kind in OPERANDS or kind in EFFECT:
            if parent:
                flat[f"kind:{family(kind)}"]+=1
                linked[f"parent:{family(parent)}/{family(kind)}"]+=1
            operator = node.child_by_field_name("operator")
            if operator is not None:
                flat[f"op:{source[operator.start_byte:operator.end_byte].decode(errors='replace')}"]+=1
        for c in reversed(node.named_children):
            mode = guard.get(c.id)
            framing = context if mode is None else (*context,(node.id,mode,kind))
            stack.append((c,depth+1,block,framing,event,kind))
    nodes = sum(count.values())
    for key,num in count.items():
        flat["node:"+key] = num
    reuse = Counter()
    for mark,side,event,name in links:
        names,kind = guards[mark]
        hit = name in names
        repeat = (side,event,name) in used_under_guard[mark]
        used_under_guard[mark].add((side,event,name))
        key = f"binding:{kind}:{side}:{event}:{int(hit)}:{int(repeat)}"
        linked[key]+=1
        reuse["any"]+=hit
        reuse["write"]+=hit and event=="write"
        reuse["call"]+=hit and event.startswith("call:")
        reuse["exit"]+=hit and event in ("return","goto","throw","break","continue")
        reuse["repeat"]+=repeat
    max_calls = steps = pairs = 0
    for events in by_block.values():
        calls = [e for e in events if e.startswith("call:")]
        max_calls = max(max_calls,len(calls))
        pairs += max(0,len(calls)-1)
        steps += len(events)
        for i in range(len(events)-1):
            linked[f"next:{events[i]}/{events[i+1]}"]+=1
            if i+2<len(events):
                linked[f"chain:{events[i]}/{events[i+1]}/{events[i+2]}"]+=1
        for a,b in zip(calls,calls[1:]):
            linked[f"calls:{a}/{b}"]+=1
    by_role = lambda kind:sum(value for name,value in linked.items() if name.startswith("under:") and kind in name)
    dense = (len(source),nodes,depth_max,int(tree.has_error),len(by_block),sum(count[k] for k in BRANCH),
             sum(count[k] for k in CALL),max_calls,len(links),by_role(":write"),by_role(":call:"),by_role(":return"),
             sum(len(names) for names,_ in guards.values()),reuse["any"],reuse["write"],reuse["call"],reuse["exit"],
             reuse["repeat"],steps,pairs,call_total,call_max,branch_depth,len(count),count["function_definition"])
    return dense,flat,linked


def batch(texts):
    parser = Parser(LANGUAGE)
    result = [extract(s,parser) for s in texts]
    dense,flat,linked = zip(*result,strict=True)
    return np.asarray(dense,dtype=np.float32),HASHER.transform(flat),HASHER.transform(linked)


def collect(texts,workers=5):
    chunks = [texts[i:i+768] for i in range(0,len(texts),768)]
    if workers>1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            result=list(pool.map(batch,chunks,chunksize=2))
    else:
        result=list(map(batch,chunks))
    dense = np.log1p(np.concatenate([x[0] for x in result]))
    mats = []
    for i in (1,2):
        m = vstack([x[i] for x in result],format="csr")
        m.data[:] = np.sign(m.data)*np.log1p(np.abs(m.data))
        mats.append(normalize(m,copy=False))
    return dense,*mats
