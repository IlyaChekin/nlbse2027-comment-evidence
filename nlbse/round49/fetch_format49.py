"""Fetch the binary of the pinned clang-format wheel and verify its published hash."""
import hashlib
import json
from pathlib import Path
from subprocess import check_output
from urllib.parse import urlparse
from urllib.request import urlopen
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent/'formatter'
VERSION = '18.1.8'
WHEEL_SHA256 = '2de122b8aa78ba49e326f974131caab2c79f4ae877c227cd4e3e5d82a98d21e5'
API = f'https://pypi.org/pypi/clang-format/{VERSION}/json'


def main():
    HERE.mkdir(exist_ok=True)
    with urlopen(API,timeout=15) as reply:
        source = json.load(reply)
    assert source['info']['version']==VERSION
    wheels = [r for r in source['urls'] if r['packagetype']=='bdist_wheel'
              and 'manylinux_2_17_x86_64' in r['filename']]
    wheel, = wheels
    assert wheel["digests"]["sha256"]==WHEEL_SHA256
    assert urlparse(wheel['url']).hostname=='files.pythonhosted.org'
    target = HERE/wheel['filename']
    if not target.exists():
        with urlopen(wheel['url'],timeout=25) as response:
            content = response.read(wheel['size']+1)
        assert len(content)==wheel['size']
        assert hashlib.sha256(content).hexdigest()==wheel['digests']['sha256']
        target.write_bytes(content)
    with target.open('rb') as handle:
        assert hashlib.file_digest(handle,'sha256').hexdigest()==wheel['digests']['sha256']
    with ZipFile(target) as z:
        entry, = [p for p in z.namelist() if p.endswith('/data/bin/clang-format')]
        binary = z.read(entry)
    exe = HERE/'clang-format'
    exe.write_bytes(binary)
    exe.chmod(0o755)
    reported = check_output([exe,'--version'],text=True).strip()
    assert VERSION in reported
    record = dict(api=API,version=reported,wheel=wheel['filename'],wheel_url=wheel['url'],
                  wheel_sha256=wheel['digests']['sha256'],program_sha256=hashlib.sha256(binary).hexdigest())
    (HERE/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
    print(record,flush=True)


if __name__=='__main__':
    main()
