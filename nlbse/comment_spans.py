"""Align tokens with C-style comment spans in function text."""
import re

LEXEME = re.compile(
    r'R"(?P<delimiter>[^ ()\\\t\r\n]{0,16})\(.*?\)(?P=delimiter)"'
    r'|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\r\n]*|/\*.*?(?:\*/|\Z)', re.S,
)


def comment_bits(text, offsets):
    spans = [(m.start(), m.end()) for m in LEXEME.finditer(text) if m[0].startswith(("//", "/*"))]
    current = 0
    marked = []
    for start, end in offsets:
        while current < len(spans) and spans[current][1] <= start:
            current += 1
        marked.append(current < len(spans) and spans[current][0] < end and end > start)
    return marked

