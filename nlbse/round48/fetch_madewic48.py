"""Download the two pinned replication archives, resuming completed byte ranges."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import shutil
from time import sleep
from urllib.error import URLError
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent/'madewic'
VERSIONS = {
    'v3': ('15126132', 150864177, 'cffd65f2f956bf0f93a709c5f6e8f58f'),
    'v2': ('13370805', 150545453, '19283a27939564d193aa630bcd780611'),
}
CHUNK_BYTES = 24 * 1024 * 1024


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'md5').hexdigest()


def chunk(piece):
    version,start,end = piece
    record, whole,_ = VERSIONS[version]
    target = HERE/f'{version}.{start}-{end}.bytes'
    if target.exists() and target.stat().st_size == end-start:
        return target
    part = target.with_suffix('.part')
    for attempt in range(4):
        done = part.stat().st_size if part.exists() else 0
        if done == end-start:
            break
        offset = start+done
        address = f'https://zenodo.org/records/{record}/files/MADE-WIC.zip?audit_offset={offset}'
        request = Request(address, headers={'Range':f'bytes={offset}-{end-1}',
                                            'User-Agent':'NLBSE-MAT-reproducibility-audit'})
        try:
            with urlopen(request, timeout=25) as response:
                assert response.status == 206
                assert response.headers['Content-Range'] == f'bytes {offset}-{end-1}/{whole}'
                with part.open('ab') as output:
                    shutil.copyfileobj(response,output,256*1024)
            if part.stat().st_size == end-start:
                break
        except (URLError, OSError, TimeoutError) as err:
            print(version,start,end,attempt+1,type(err).__name__,flush=True)
            sleep(2**attempt)
    assert part.stat().st_size == end-start
    part.replace(target)
    print(version,start,end,'verified length',flush=True)
    return target


def download(version, workers):
    _,length,expected = VERSIONS[version]
    result = HERE/f'{version}.zip'
    if result.exists():
        assert digest(result) == expected
        return
    prefix = HERE/f'{version}.zip.part'
    have = prefix.stat().st_size if prefix.exists() else 0
    end = length
    suffix = HERE/f'{version}.suffix'
    if suffix.exists():
        assert suffix.stat().st_size == 131072
        end -= 131072
    assert have <= end
    pending = [(version,lo,min(end,lo+CHUNK_BYTES)) for lo in range(have,end,CHUNK_BYTES)]
    paths = list(workers.map(chunk,pending))
    copy = result.with_suffix('.assembled')
    with copy.open('wb') as target:
        for source in [*([prefix] if have else []), *paths, *([suffix] if end<length else [])]:
            with source.open('rb') as handle:
                shutil.copyfileobj(handle,target,1<<20)
    assert copy.stat().st_size == length
    assert digest(copy) == expected
    copy.replace(result)
    print(version,length,expected,flush=True)


def main():
    HERE.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as workers:
        for order in VERSIONS:
            download(order,workers)


if __name__=='__main__':
    main()
