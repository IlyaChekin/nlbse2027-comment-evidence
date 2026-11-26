"""Use the frozen R53 policy in the verified single-encoder profiling pipeline."""
from pathlib import Path
from unittest.mock import patch

from eval53 import DEST as EVAL, configured
from train53 import DEST as FIT, HERE, checksum, json_at, write_json
import live51

DEST = HERE / 'live'


def main():
    plan = EVAL / 'selection.json'
    card = json_at(plan)
    assert card['csv_sha256'] == checksum(plan.with_name('comparison.csv'))
    assert card['code_sha256'] == checksum(HERE / 'eval53.py')
    frozen = [('multitask', arm, choice, checksum(plan)) for arm, choice in card['selected'].items() if choice]
    with patch.multiple(live51, DEST=DEST, FIT=FIT, configured=configured, policies=lambda: frozen):
        live51.main()
    DEST.mkdir(exist_ok=True)
    paths = [p for p in (DEST / 'selection.json', DEST / 'public.json', DEST / 'public_predictions.npz') if p.exists()]
    assert bool(frozen) == (DEST / 'selection.json').exists()
    write_json(DEST / 'receipt.json', dict(eligible_offline=len(frozen), selection_sha256=checksum(plan),
               results={p.name: checksum(p) for p in paths}, runner_sha256=checksum(Path(__file__)),
               pipeline_sha256=checksum(Path(live51.__file__))))
    print('qualified for live internal recheck', len(frozen), flush=True)


if __name__ == '__main__':
    main()
