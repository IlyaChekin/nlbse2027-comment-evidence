"""Verify comment absence among public MAT misses with an independent C++ parser."""

from collections import Counter
import sys

from audit import HERE, ROOT, LEXEME, STEM, dataset, write_json
from mat import example

sys.path.insert(0, str(ROOT/"round12/vendor"))
from tree_sitter import Language, Parser
import tree_sitter_cpp


def comments(node):
    if node.type == "comment":
        yield node
    for child in node.named_children:
        yield from comments(child)


def main():
    texts,gold,_,_=dataset(True)
    parser=Parser(Language(tree_sitter_cpp.language()))
    totals=Counter()
    for index,text in enumerate(texts):
        if gold[index,1] != 1 or example(text,code_required=False)[0]:
            continue
        totals["misses"] += 1
        raw=text.encode()
        original=[m[0] for m in LEXEME.finditer(text) if m[0].startswith(("//","/*"))]
        root=parser.parse(raw).root_node
        alternate=[raw[n.start_byte:n.end_byte].decode(errors="replace") for n in comments(root)]
        for key,value in (("regex_comment",original),("tree_sitter_comment",alternate),
                          ("neither_parser_sees_comment",not (original or alternate)),
                          ("comment_marker_bytes_present", "//" in text or "/*" in text),
                          ("root_tag_anywhere",STEM.search(text)),
                          ("root_tag_in_tree_sitter_comment",any(STEM.search(c) for c in alternate)),
                          ("tree_sitter_has_parse_errors",root.has_error)):
            totals[key] += bool(value)
    write_json(HERE/"coverage.json",dict(totals))
    print(dict(totals),flush=True)


if __name__=="__main__":
    main()
