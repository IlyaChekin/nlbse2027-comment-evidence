"""Compare the selected ranking with a threshold change on exactly the same rows."""
import csv

import numpy as np

from positive import HERE, ROOT, base, f, json_at, checksum, write_json, strict_cut
from positive_runtime import scope


def evaluate():
    tested = HERE/'public'
    freeze = json_at(tested/'policy.json')
    name = freeze['selected']
    candidate = freeze['supported'][name]
    width = candidate['online']['width']
    results = {}
    threshold = None
    for part in ('test_a', 'test_b', 'official'):
        d, t, _, dev, review, old, orig, thresh, _ = base(part)
        active, q, _ = scope(d.texts, d.state, thresh, width)
        ask, qt, _ = scope(t.texts, t.state, thresh, width)
        gold, yy = d.frame['y'], t.frame['y']
        cal = active & dev
        selected = strict_cut(gold[cal], old[cal], q[cal], 'tp')
        xd, xt = old.copy(), orig.copy()
        xd[active], xt[ask] = q[active] > selected, qt[ask] > selected
        described = np.ones(len(yy), dtype=bool) if review is None else review
        row = dict(cut=selected, calibration=f.comparison(gold[dev], old[dev], xd[dev]),
                   frozen=f.comparison(yy[described], orig[described], xt[described]),
                   scope_count=int(ask[described].sum()), source_cut=thresh, width=width)
        results[part] = row
        np.savez_compressed(HERE/f'threshold_{part}_predictions.npz', ids=t.frame['ids'][described],
                           scope=ask[described], reference=orig[described], prediction=xt[described],
                           score=qt[described])
        if part == 'official':
            threshold = selected
    old_folder = ROOT/'round25/views/public'
    archive = json_at(old_folder/'results.json')
    path = old_folder/f"{archive['selected']}.csv"
    if checksum(path) != archive['results'][archive['selected']]['predictions_sha256']:
        raise ValueError('Baseline probability archive changed')
    verified = json_at(tested/'results.json')
    trial = tested/'candidate.csv'
    if checksum(trial) != verified['results'][name]['predictions_sha256']:
        raise ValueError('Paired candidate rows changed')
    with path.open() as r, trial.open() as pfile:
        previous, candidate_rows = list(csv.DictReader(r)), list(csv.DictReader(pfile))
    if not all(a['row'] == b['row'] and a['label_vulnerability'] == b['label_vulnerability']
               for a, b in zip(previous, candidate_rows, strict=True)):
        raise ValueError('Baseline and candidate CSV rows cannot be aligned')
    raw = np.asarray([float(k['probability']) for k in previous], dtype=np.float32)
    touched = np.asarray([int(k['scoped']) for k in candidate_rows], dtype=bool)
    expected = np.asarray([int(k['pred_vulnerability']) for k in previous], dtype=bool)
    y = np.asarray([int(k['label_vulnerability']) for k in previous], dtype=bool)
    measured = np.asarray([int(k['pred_vulnerability']) for k in candidate_rows], dtype=bool)
    _, margin, _ = scope(None, dict(p=raw, known=touched, flag=touched), thresh, width)
    if np.any(abs(margin[touched]) >= width) or not np.array_equal((margin > 0)[touched], expected[touched]):
        raise ValueError('Public original scores do not recreate the selected scope')
    baseline_only = expected.copy()
    baseline_only[touched] = margin[touched] > threshold
    results['public'] = dict(cut=threshold, vs_reference=f.comparison(y, expected, baseline_only),
                            trained_vs_threshold=f.comparison(y, baseline_only, measured), scope_count=int(touched.sum()),
                            source_sha256=checksum(path), candidate_sha256=checksum(trial))
    save = HERE/'threshold_public_predictions.npz'
    np.savez_compressed(save, rows=np.arange(len(y)), gold=y, reference=expected, candidate=measured,
                       threshold_only=baseline_only, scope=touched, reference_margin=margin)
    results['public']['predictions_sha256'] = checksum(save)
    write_json(HERE/'threshold_control.json', results)
    print(results, flush=True)


if __name__ == '__main__':
    evaluate()
