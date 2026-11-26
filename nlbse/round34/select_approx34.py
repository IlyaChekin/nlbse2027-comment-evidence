"""Freeze one pooled approximation using only project and internal comparisons."""
import json

from select34 import HERE, PARTS, read, sha, evaluate, supported


def main():
    area=HERE/'approx_comparison'
    files={}
    hashes={}
    for part in PARTS:
        src=area/part
        r=read(src/'results.json')
        assert r['code_sha256']==sha(HERE/'compare34.py')
        assert r['predictions_sha256']==sha(src/'predictions.npz')
        assert r['weights_sha256']==sha(src/'calibrators.joblib')
        p=HERE/'approximation'/part
        heads=read(p/'receipt.json')
        assert r['heads_sha256']==sha(p/'receipt.json')
        assert heads['code_sha256']==sha(HERE/'approx34.py')
        assert heads['predictions_sha256']==sha(p/'scores.npz')
        assert heads['source_sha256']==read(HERE/'attention'/part/'receipt.json')['predictions_sha256']
        for k,h in heads['heads'].items():
            assert h['sha256']==sha(p/f'{k}.pt')
        files[part]=r
        hashes[part]=sha(src/'results.json')
    rows=evaluate('approx_comparison',files)
    selected=[r for r in rows if r['mode'].startswith('ridge') and supported(r)]
    chosen=max(selected,key=lambda r:(r['cal_f1'],r['review_f1'],-r['cal_calls']),default=None)
    result=dict(selected=chosen,eligible=len(selected),tested=len(rows),receipts={'approx_comparison':hashes})
    (HERE/'selection_approx.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
