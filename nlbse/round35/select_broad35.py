"""Freeze a candidate for a complete public pass before loading public data."""
import csv
import json

from select35 import HERE, ROOT, enumerate_options, read, sha, stage, supported

DEST = HERE/'broad'


def main():
    from select35 import FILES
    FILES['broad'] = 'broad35.py'
    records = stage('broad')
    for part, (profile, _) in records.items():
        assert sha(DEST/part/'cohorts.npz') == profile['cohorts_sha256']
    options = enumerate_options('broad', records)
    candidates = [r for r in options if supported(r)]
    winner = max(candidates, key=lambda r: (r['cal_f1'], r['review_f1'], -r['cal_rows']), default=None)
    table = DEST/'comparison.csv'
    with table.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=options[0].keys())
        writer.writeheader()
        writer.writerows(options)
    result = dict(selected=winner, tested=len(options), eligible=len(candidates),
                  result_sha256={'broad': {p: proof for p, (_, proof) in records.items()}},
                  code_sha256=sha(HERE/'select_broad35.py'), csv_sha256=sha(table),
                  released_archive_sha256=read(ROOT/'round34/release_summary.json')['archive_sha256'])
    (DEST/'selection.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
