"""Describe the closest same-file candidates for unresolved MAT provenance."""
import csv
from difflib import SequenceMatcher
import hashlib
import io
import json
from pathlib import Path
import re
from zipfile import ZipFile

from madewic48 import OUT, COLLECTIONS, TOKEN, key
from shape48 import checksum, dataset, write_json
from transfer import upstream_tag


def tokens(text):
    return tuple(span[0] for span in TOKEN.finditer(text) if span['comment'] is None)


def remove_splices(text):
    return re.sub(r"\\\r?\n",'',text)


def candidates(zip_path, cases):
    by_file = {location: [] for _, location, _, _ in cases}
    csv.field_size_limit(32*1024*1024)
    with ZipFile(zip_path) as source:
        for subset in COLLECTIONS:
            file = f'MADE-WIC/Dataset/{subset}/complete.csv'
            with io.TextIOWrapper(source.open(file),encoding='utf-8-sig',errors='surrogateescape',newline='') as handle:
                for rownum, row in enumerate(csv.DictReader(handle),start=1):
                    location = key(row['Projectname'],row['Filepath'])
                    if location in by_file:
                        words = tokens(row['Function'])
                        evidence = dict(corpus=subset,line=rownum,original_mat=row['MAT'],
                                        leading_tag=upstream_tag(row['LeadingComment']),code_tokens=len(words),
                                        leading_sha256=hashlib.sha256(
                                            row['LeadingComment'].encode('utf-8','surrogateescape')).hexdigest())
                        by_file[location].append((words,tokens(remove_splices(row['Function'])),evidence))
    result = []
    for i, location, query, unspliced in cases:
        exact = [evidence for _, normalized, evidence in by_file[location] if normalized==unspliced]
        nearest = []
        for words, normalized, evidence in by_file[location]:
            if len(query)<12 or not .65*len(query) <= len(words) <= 1.35*len(query):
                continue
            match = SequenceMatcher(None,query,words,autojunk=False)
            changes = [q for q in match.get_opcodes() if q[0]!='equal']
            symbols = frozenset(('\\',';',',','(',')','{','}'))
            candidate = evidence | dict(ratio=match.ratio(),head8=query[:8]==words[:8],tail8=query[-8:]==words[-8:],
                                        exact_after_line_splicing=unspliced==normalized,
                                        edits=[(kind,last-first,end-start) for kind,first,last,start,end in changes],
                                        edited_query=[v if v in symbols else '<token>'
                                                      for _,a,b,_,_ in changes for v in query[a:b]],
                                        edited_source=[v if v in symbols else '<token>'
                                                       for _,_,_,a,b in changes for v in words[a:b]])
            nearest.append(candidate)
        nearest.sort(key=lambda r:r['ratio'],reverse=True)
        result.append(dict(row=i,project=location[0],query_tokens=len(query),records=len(by_file[location]),
                           exact_after_splicing=exact,top=nearest[:3]))
    return result


def main():
    prior = OUT/'results.json'
    data = json.loads(prior.read_text())
    known = set(data['public_provenance']['ids'])
    missing = [row['row'] for row in data['observations'] if row['part']=='public' and row['gold']
               and not row['base'] and row['row'] not in known]
    x, _, project, file = dataset(public=True)
    items = [(i,key(project[i],file[i]),tokens(x[i]),tokens(remove_splices(x[i]))) for i in missing]
    archive = OUT/'v3.zip'
    assert checksum(archive)==data['sources']['v3']['archive_sha256']
    neighbors = candidates(archive,items)
    # Only unique equality after C line splicing qualifies. Edit similarity is descriptive.
    resolved = [r['row'] for r in neighbors if len(r['exact_after_splicing'])==1
                and r['exact_after_splicing'][0]['leading_tag']
                and r['exact_after_splicing'][0]['original_mat']=='1']
    write_json(OUT/'unmatched.json',dict(public_rows=neighbors,archive_sha256=checksum(archive),
               earlier_confirmed=len(known),additional_confirmed_rows=resolved,
               confirmed_total=len(known)+len(resolved),unresolved=len(missing)-len(resolved),
               exact_audit_sha256=checksum(prior),code_sha256=checksum(Path(__file__))))
    print(neighbors,flush=True)
    print('unique_matches_with_source_tag',resolved,flush=True)


if __name__=='__main__':
    main()
