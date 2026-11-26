"""Audit whether missing MAT tags are present in the original MADE-WIC record."""
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
from time import perf_counter
from zipfile import ZipFile

import numpy as np

from shape48 import DEST, HERE, ROOT, TOKEN, canonical_mat, checksum, dataset, write_json
from transfer import upstream_tag
from project_split import family

OUT = HERE/'madewic'
ARCHIVES = {
    'v2': ('https://zenodo.org/records/13370805', '19283a27939564d193aa630bcd780611'),
    'v3': ('https://zenodo.org/records/15126132', 'cffd65f2f956bf0f93a709c5f6e8f58f'),
}
COLLECTIONS = ('devign','big-vul','OSPR')


def key(project, path):
    normalized = str(PurePosixPath(path.replace('\\','/'))) if path else ''
    return family(project or ''), normalized


def signature(text):
    # Clang-format changes whitespace. Preserve spelling of code and literal tokens.
    tokens = (span[0] for span in TOKEN.finditer(text) if span['comment'] is None)
    return hashlib.blake2b('\0'.join(tokens).encode('utf-8','surrogateescape'),digest_size=24).hexdigest()


def tags(text):
    return tuple(hashlib.sha256(span[0].encode('utf-8','surrogateescape')).hexdigest() for span in TOKEN.finditer(text)
                 if span['comment'] is not None and upstream_tag(span[0]))


def seed():
    plan = {}
    cases = defaultdict(list)
    before_path = ROOT/'round38/release_verification/release_predictions.csv'
    before = json.loads(before_path.with_name('receipt.json').read_text())
    assert before['predictions_sha256'] == checksum(before_path)
    for side in ('test_a','test_b','public'):
        texts, label, p, file = dataset(side=='public')
        if side=='public':
            rows = np.arange(len(texts))
            y = label[:,1].astype(bool)
            with before_path.open(newline='') as src:
                base = np.array([bool(int(r['MAT'])) for r in csv.DictReader(src)])
        else:
            with np.load(DEST/f'{side}_predictions.npz') as frames:
                rows, y, base = (frames[k] for k in ('ids','y','base'))
                assert np.array_equal(label[rows,1],y)
        assert np.array_equal(base, canonical_mat([texts[i] for i in rows]))
        missing = rows[y & ~base]
        targets = {key(p[i],file[i]) for i in missing if p[i] and file[i]}
        # Report against all rows from selected files and count unresolved rows with missing paths.
        plan[side] = dict(rows=len(rows), pending=len(missing), audited_file_count=len(targets),
                          misses_with_no_path=sum(not (p[i] and file[i]) for i in missing))
        for i, pos, pred in zip(rows,y,base,strict=True):
            location = key(p[i],file[i])
            if location in targets or (pos and not pred and not all(location)):
                text = texts[i]
                case = dict(part=side,row=int(i),project=location[0],gold=bool(pos),base=bool(pred),
                            lexical_tags=len(tags(text)),visible_comment=any(
                                s['comment'] is not None for s in TOKEN.finditer(text)))
                cases[location].append((signature(text),case))
    return plan,cases


def index(version, cases):
    uri, expected_md5 = ARCHIVES[version]
    archive = OUT/f'{version}.zip'
    with archive.open('rb') as handle:
        md5 = hashlib.file_digest(handle,'md5').hexdigest()
    assert md5 == expected_md5
    eligible = {location:{q[0] for q in group} for location,group in cases.items() if all(location)}
    index = defaultdict(dict)
    collections = {}
    csv.field_size_limit(32*1024*1024)
    with ZipFile(archive) as z:
        for corpus in COLLECTIONS:
            file = f'MADE-WIC/Dataset/{corpus}/complete.csv'
            member = z.getinfo(file)
            rows = 0
            relevant = 0
            start = perf_counter()
            with io.TextIOWrapper(z.open(member),encoding='utf-8-sig',newline='',errors='surrogateescape') as handle:
                reader = csv.DictReader(handle)
                assert {'Projectname','Filepath','Function','LeadingComment','MAT','Commit-ID'}.issubset(reader.fieldnames)
                for row in reader:
                    rows += 1
                    location = key(row['Projectname'],row['Filepath'])
                    if location not in eligible:
                        continue
                    relevant += 1
                    token = signature(row['Function'])
                    if token not in eligible[location]:
                        continue
                    leading = row['LeadingComment']
                    inner = tags(row['Function'])
                    content = dict(corpus=corpus,line=rows,commit=row['Commit-ID'],
                                   leading_tag=upstream_tag(leading),leading_chars=len(leading),
                                   leading_sha256=hashlib.sha256(leading.encode('utf-8','surrogateescape')).hexdigest(),
                                   inner_tag_count=len(inner),inner_sha256=list(inner),annotation=row['MAT'])
                    index[location].setdefault(token,[]).append(content)
            collections[corpus] = dict(rows=rows, matching_files_rows=relevant, crc32=f'{member.CRC:08x}',
                                       seconds=perf_counter()-start)
            print(version,corpus,collections[corpus],flush=True)
    return index,dict(documentation=uri,archive_sha256=checksum(archive),archive_md5=md5,collections=collections)


def state(records):
    if not records:
        return 'unmatched'
    categories = {('leading' if r['leading_tag'] else 'inside' if r['inner_tag_count'] else 'no_tag')
                  for r in records}
    return next(iter(categories)) if len(categories)==1 else 'conflicting_matches'


def tabulate(plan, observations):
    output = {}
    for side,scope in plan.items():
        selected = [r for r in observations if r['part']==side]
        output[side] = dict(**scope,audited_rows=len(selected),versions={})
        for version in ARCHIVES:
            misses = Counter(r[version]['status'] for r in selected if r['gold'] and not r['base'])
            assert sum(misses.values())==scope['pending']
            extra = [r for r in selected if not r['base'] and r[version]['status'] in ('leading','inside')]
            visible = [r for r in extra if r['gold'] and r['visible_comment']]
            gain = dict(recovered=sum(r['gold'] for r in extra), extra_fp=sum(not r['gold'] for r in extra),
                        recovered_with_visible_comment=len(visible))
            output[side]['versions'][version] = dict(
                miss_outcomes=misses,context_diagnostic=gain,exact_matched=sum(
                    r[version]['status'] not in ('unmatched','conflicting_matches') for r in selected),
                extra_by_project={p:dict(tp=sum(r['gold'] for r in extra if r['project']==p),
                                         fp=sum(not r['gold'] for r in extra if r['project']==p))
                                  for p in sorted({r['project'] for r in extra})})
        output[side]['miss_transitions'] = Counter(r['v2']['status']+' → '+r['v3']['status']
                                                  for r in selected if r['gold'] and not r['base'])
    return output


def known_public(observations):
    old = ROOT/'round18/mat_residual/source_audit_public.json'
    also = ROOT/'round26/mat/visible_provenance.json'
    p = json.loads(old.read_text())['observations']
    v = json.loads(also.read_text())['observations']
    old_ids = {r['row'] for r in p if r['mat_positive'] and r['observation'].startswith('marker_')}
    old_ids |= {r['row'] for r in v if r['split']=='public' and r['gold'] and not r['whole_word_tag_visible']
                and r['observation'].startswith('marker_')}
    verified = json.loads((HERE/'audit.json').read_text())['public']
    assert len(old_ids)==verified['confirmed_hidden']
    subset = [r for r in observations if r['part']=='public' and r['gold'] and not r['base']]
    assert len(subset)==verified['fn'] and old_ids<={r['row'] for r in subset}
    matched = {r['row'] for r in subset if any(r[k]['status'] in ('leading','inside') for k in ARCHIVES)}
    release = ROOT/'round38/release_summary.json'
    m = json.loads(release.read_text())['MAT']
    zero_fp_upper = len(subset)-len(old_ids | matched)

    def project(recovered, new_fp):
        prior_f1 = 2*m['tp']/(2*m['tp']+m['fp']+m['fn'])
        diagnostic_f1 = 2*(m['tp']+recovered)/(2*m['tp']+m['fp']+m['fn']+recovered+new_fp)
        return verified['verified_score']+.3*(diagnostic_f1-prior_f1)

    extra = [r for r in observations if r['part']=='public' and not r['base']
             and r['v3']['status'] in ('leading','inside')]
    tp,fp = sum(r['gold'] for r in extra),sum(not r['gold'] for r in extra)
    return dict(previous_snapshot_confirmed=len(old_ids),confirmed_from_archives=len(matched),
                additionally_confirmed=len(matched-old_ids),joint_confirmed=len(old_ids | matched),
                unresolved=zero_fp_upper,source_inputs={x.name:checksum(x) for x in (old,also,release)},
                remaining_if_all_unresolved_fixed=project(zero_fp_upper,0),
                hypothetical_v3_context_score_without_inference_cost=project(tp,fp),
                ids=sorted(old_ids | matched))


def main():
    plan,cases = seed()
    print('input',plan,flush=True)
    indexes = {}
    headers = {}
    for version in ARCHIVES:
        indexes[version],headers[version] = index(version,cases)
    observations = []
    for location,pairs in cases.items():
        for token,case in pairs:
            report = {}
            for name, index_by_file in indexes.items():
                records = index_by_file[location].get(token,[])
                report[name] = dict(status=state(records),observations=records)
            observations.append(case | report | dict(token_digest=token))
    combined = tabulate(plan,observations)
    with (OUT/'comparison.csv').open('w',newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('partition','original_version','audited_rows','base_fn','leading','inside',
                         'no_tag','conflict','unmatched','context_diagnostic_tp','context_diagnostic_fp'))
        for role, group in combined.items():
            for version,data in group['versions'].items():
                writer.writerow((role,version,group['audited_rows'],group['pending'],
                    *[data['miss_outcomes'][name] for name in ('leading','inside','no_tag','conflicting_matches','unmatched')],
                    data['context_diagnostic']['recovered'],data['context_diagnostic']['extra_fp']))
    write_json(OUT/'results.json',dict(purpose='offline context provenance, using reference project and path',
               sources=headers,partitions=combined,public_provenance=known_public(observations),observations=observations,
               code_sha256=checksum(Path(__file__)),csv_sha256=checksum(OUT/'comparison.csv')))
    print(combined,flush=True)


if __name__=='__main__':
    main()
