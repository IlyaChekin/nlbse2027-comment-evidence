"""Verify whitespace decisions and select a policy at the existing error budget."""
import csv
from pathlib import Path

import numpy as np

from causal54 import DEST as CAUSAL, HERE, PARTS, ROOT, checksum, f, json_at, load, parts, write_json

DEST = HERE / 'comparison'


def inspect():
    DEST.mkdir(exist_ok=True)
    data = {}
    signed = {}
    for rotation in PARTS:
        source, _, stamp = load(rotation)
        signed[rotation] = parts(source)
        for kind in ('edit', 'indent'):
            path = HERE / kind / rotation / 'results.json'
            card = json_at(path)
            proof = card.get('proof', card)
            assert proof['code_sha256'] == checksum(HERE / ('effects54.py' if kind == 'edit' else 'indent54.py'))
            assert stamp == proof['matrix_sha256']
            npz = path.with_name('predictions.npz')
            assert checksum(npz) == proof['predictions_sha256']
            if kind == 'edit':
                assert checksum(path.with_name('models.joblib')) == proof['models_sha256']
                assert checksum(HERE / 'edit/design.json') == proof['design_sha256']
            else:
                assert checksum(path.with_name('model.joblib')) == proof['model_sha256']
            data[rotation, kind] = (card, npz)
    results = []
    for kind in ('edit', 'indent'):
        off, _ = data['official', kind]
        for variant in sorted(k for k,v in off['comparison']['held'].items() if isinstance(v,dict)):
            matrix, frozen = {}, {}
            for rotation in PARTS:
                card, path = data[rotation, kind]
                with np.load(path) as datafile:
                    for n, phase in enumerate(('cal', 'held')):
                        frame = signed[rotation][n]
                        use = frame['old'] & (frame['flag'] >= 0)
                        ids, old, y, proof = (datafile[phase + '_' + name] for name in ('ids', 'old', 'expected', 'known'))
                        assert np.array_equal(frame['ids'][use], ids) and np.array_equal(frame['flag'][use] == 1, old)
                        q = datafile[phase + '_' + variant]
                        touched = datafile[phase + '_' + variant + '_touched']
                        assert np.array_equal(q[~touched], old[~touched]) and np.isin(q,[-1,0,1]).all()
                        after = np.where(q < 0, old, q == 1)
                        performance = f.comparison(y[proof],old[proof],after[proof])
                        saved = card['comparison'][phase][variant]
                        assert performance == saved['W'] and int(touched.sum()) == saved['touched']
                        owner = frame['project'][use]
                        group = {p: f.comparison(y[active], old[active], after[active])
                                 for p in np.unique(owner[proof]) if (active := (owner == p) & proof).any()}
                        matrix[rotation + '/' + phase] = dict(change=performance, projects=group,
                                touched=saved['touched'], invalid=saved['unavailable'], comparison_sha256=checksum(path))
                frozen[rotation] = checksum(path.with_name('results.json'))
            cal, review = (matrix['official/' + p]['change'] for p in ('cal','held'))
            held = [matrix[r + '/held'] for r in ('test_a','test_b')]
            by_proj = [tot for v in held for tot in v['projects'].values()]
            keep = lambda q: q['delta_tp'] >= 0 and q['delta_fp'] <= 0 and (q['delta_tp'] or q['delta_fp'])
            offline = (all(v['invalid'] == 0 for v in matrix.values()) and bool(keep(cal)) and bool(keep(review)) and
                       all(q['delta_fp'] <= 0 for q in by_proj) and all(v['change']['delta_tp']>=0 for v in held) and
                       any(v['change']['delta_tp']>0 or v['change']['delta_fp']<0 for v in held))
            threshold = off['policy'][variant] if kind == 'edit' else off['threshold'][variant.split('/')[2]]
            results.append(dict(kind=kind, variant=variant, qualified=bool(offline), matrix=matrix,
                                official_threshold=threshold, source=frozen))
    results.sort(key=lambda z: (z['qualified'], min(v['change']['f1_after']-v['change']['f1_before']
                 for k,v in z['matrix'].items() if k.endswith('/held'))), reverse=True)
    path = DEST / 'comparison.csv'
    with path.open('w',newline='') as handle:
        out = csv.writer(handle)
        out.writerow(('family','arm','qualified','partition','w_delta_tp','w_delta_fp','w_f1_before','w_f1_after','edited','invalid'))
        for row in results:
            for part, item in row['matrix'].items():
                x = item['change']
                out.writerow((row['kind'],row['variant'],row['qualified'],part,
                             *(x[k] for k in ('delta_tp','delta_fp','f1_before','f1_after')),item['touched'],item['invalid']))
    chosen = [x for x in results if x['qualified']]
    write_json(DEST / 'selection.json', dict(candidates=len(results), selected=chosen, results=results,
               comparison_sha256=checksum(path), code_sha256=checksum(Path(__file__))))
    print('tested',len(results),'passed',[(x['kind'],x['variant']) for x in chosen],flush=True)


if __name__ == '__main__':
    inspect()
