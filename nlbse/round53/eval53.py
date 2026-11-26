"""Evaluate matched supervision masks on held projects and on file-held review."""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from train53 import (ARMS, DEST as FIT, HERE, PARTS, ROOT, SUPPORT, checksum, extra_head,
                     f, full_context, json_at, lexical_features, release_start, write_json)
from eval51 import compare, geometry, load, scores, view
from select51 import differences, f1, parts

DEST = HERE / 'evaluation'


def configured(rotation, arm, tok, proof):
    net, sha = release_start(rotation, tok)
    assert sha == proof['config']['base_checkpoint_sha256']
    addon = None
    if arm is not None:
        weight = FIT / rotation / (arm + '.pt')
        assert checksum(weight) == proof['arms'][arm]['sha256']
        saved = torch.load(weight, map_location='cpu', weights_only=True)
        by_name = dict(net.named_parameters())
        with torch.no_grad():
            for key, value in saved['backbone'].items():
                by_name[key].copy_(value)
        addon = extra_head(net.width).cuda().eval()
        addon.load_state_dict(saved['head'])
    return net.eval(), addon


def metric(z, frame, target, known):
    ids = frame['ids']
    w, e = (target[ids] & 1) > 0, (target[ids] & 6) > 0
    declared = known[ids]
    missed = declared & e & ~w & (frame['signed'] <= 0.)
    return dict(overall_ap=float(average_precision_score(frame['y'], z[:, 0])),
                external_ap=float(average_precision_score(e[declared], z[declared, 2])),
                external_without_w_ap=float(average_precision_score(e[declared & ~w], z[declared & ~w, 2])),
                external_positive=int((declared & e).sum()), explicit_rows=int(declared.sum()),
                external_only_baseline_misses=int(missed.sum()))


def evaluation(rotation):
    src, _, matrix = load(rotation)
    frames = parts(src)
    manifest = FIT / rotation / 'fit.json'
    proof = json_at(manifest)
    assert proof['config']['sources']['matrix_sha256'] == matrix
    assert proof['config']['roster_sha256'] == checksum(manifest.with_name('rows.npz'))
    assert proof['config']['code_sha256'] == checksum(HERE / 'train53.py')
    assert set(proof['arms']) == set(ARMS) and len({v['steps'] for v in proof['arms'].values()}) == 1
    inputs, _, project, path, _ = f.inputs()
    used_files = {(project[i], path[i]) for i in src['oof']['ids'] if path[i]}
    mask_path = SUPPORT / 'mask.npz'
    assert proof['config']['source_mask_sha256'] == checksum(mask_path)
    with np.load(mask_path) as z:
        target, known = z['component'], z['external_available']
    tok = f.small.tokenizer()
    lists = [[inputs[i] for i in d['ids']] for d in frames]
    seq = [full_context(t, tok) for t in lists]
    side = [torch.as_tensor(lexical_features(t), device='cuda') for t in lists]
    arm_scores, metrics, raw = {}, {}, {}
    for arm in ARMS:
        model, head = configured(rotation, arm, tok, proof)
        outputs = [scores(model, head, t, s, tok) for t, s in zip(seq, side, strict=True)]
        metrics[arm] = {}
        for label, frame, data in zip(('cal', 'review'), frames, outputs, strict=True):
            base, aux, cost = data
            raw[label + '_' + arm] = aux
            for name, v in view(aux, arm).items():
                arm_scores.setdefault((arm, name), []).append(v)
            metrics[arm][label] = dict(effects=metric(aux, frame, target, known), cost=cost)
        print(rotation, arm, metrics[arm], flush=True)
        del model, head
    comparison, predictions = compare(rotation, frames, arm_scores)
    review = frames[1]
    predictions['review_file_held'] = np.array([bool(path[i]) and (project[i], path[i]) not in used_files
                                                for i in review['ids']])
    out = DEST / rotation
    out.mkdir(parents=True, exist_ok=True)
    forecast = out / 'predictions.npz'
    np.savez_compressed(forecast, **predictions)
    raw_path = out / 'logits.npz'
    np.savez_compressed(raw_path, **raw)
    write_json(out / 'results.json', dict(results=comparison, metrics=metrics,
               fit_sha256=checksum(manifest), prediction_sha256=checksum(forecast),
               logits_sha256=checksum(raw_path), code_sha256=checksum(Path(__file__))))


def selection():
    cards = {}
    for part in PARTS:
        file = DEST / part / 'results.json'
        card = json_at(file)
        assert card['fit_sha256'] == checksum(FIT / part / 'fit.json')
        assert card['prediction_sha256'] == checksum(file.with_name('predictions.npz'))
        assert card['logits_sha256'] == checksum(file.with_name('logits.npz'))
        assert card['code_sha256'] == checksum(Path(__file__))
        src, _, _ = load(part)
        cal, held = parts(src)
        with np.load(file.with_name('predictions.npz')) as z:
            assert np.array_equal(z['ids'], held['ids']) and np.array_equal(z['cal_ids'], cal['ids'])
            assert np.array_equal(z['base'], held['signed'] > 0.) and np.array_equal(z['truth'], held['y'])
            for key, v in card['results'].items():
                scope = geometry(held, v['panel'], v['width'])
                for rule in ('fixed', 'tp', 'fp'):
                    pred = z[key + '_' + rule]
                    assert f.comparison(z['truth'], z['base'], pred) == {k: v[rule]['held'][k]
                             for k in f.comparison(z['truth'], z['base'], pred)}
                    assert np.array_equal(pred[~scope], z['base'][~scope])
                    if part != 'official' and rule != 'fixed':
                        equal = z[key + '_same_' + rule]
                        for found in np.unique(held['project']):
                            group = held['project'] == found
                            assert f.comparison(z['truth'][group], z['base'][group], equal[group])['delta_' + rule] == 0
        cards[part] = card
    A, B, O = (cards[part]['results'] for part in PARTS)
    rows = []
    path = DEST / 'official/predictions.npz'
    with np.load(path) as z:
        scope = z['review_file_held']
        truth, before = z['truth'][scope], z['base'][scope]
        for key, value in O.items():
            arm = value['arm']
            if arm == 'reference' or key not in A or key not in B:
                continue
            prefix = '/'.join(key.split('/')[:2])
            controls = [prefix + '/reference/threshold/0']
            if arm.endswith('components'):
                controls.append(f"{prefix}/union/main/{value['mix']:g}")
            if arm == 'measured_components':
                controls.append(key.replace(arm, 'all_components'))
            a, b = [differences(d, key, controls) for d in (A, B)]
            for rule in ('fixed', 'tp', 'fp'):
                gain = f1(value[rule]['cal'])
                held = f1(value[rule]['held'])
                edge = gain - max([0.] + [f1(O[k][rule]['cal']) for k in controls])
                clean = f.comparison(truth, before, z[key + '_' + rule][scope])
                better_projects = any(min(u[1], v[1]) >= 0 and u[1] + v[1] > 0
                                      for u, v in zip(a, b, strict=True))
                rows.append(dict(key=key, arm=arm, view=value['view'], panel=value['panel'], width=value['width'],
                                 mix=value['mix'], rule=rule, cut=value[rule]['cut'],
                                 a_tp=a[0][0], b_tp=b[0][0], a_over_tp=a[0][1], b_over_tp=b[0][1],
                                 a_removed_fp=a[1][0], b_removed_fp=b[1][0], a_over_fp=a[1][1], b_over_fp=b[1][1],
                                 cal_f1=gain, cal_over=edge, review_f1=held,
                                 new_file_tp=clean['delta_tp'], new_file_fp=clean['delta_fp'],
                                 qualified=bool(better_projects and edge > 0. and held > 0. and f1(clean) >= 0.)))
    rows.sort(key=lambda q: (q['qualified'], min(q['cal_f1'], q['review_f1']), q['cal_over']), reverse=True)
    choices = {arm: next((r for r in rows if r['arm'] == arm and r['qualified']), None) for arm in ARMS}
    table = DEST / 'comparison.csv'
    with table.open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(DEST / 'selection.json', dict(selected=choices, options=len(rows), qualified=sum(r['qualified'] for r in rows),
               new_file_review_rows=int(scope.sum()), examples=rows[:6], csv_sha256=checksum(table),
               evaluated_parts={n: checksum(DEST / n / 'results.json') for n in PARTS},
               code_sha256=checksum(Path(__file__))))
    print('selection', choices, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('rotation', choices=(*PARTS, 'select'))
    which = p.parse_args().rotation
    if which == 'select':
        selection()
    else:
        evaluation(which)
