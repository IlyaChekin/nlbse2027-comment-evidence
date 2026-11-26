"""Fetch only the source rules from the MADE-WIC replication archive."""
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
URL = "https://zenodo.org/records/12567874/files/MADE-WIC.zip"
LENGTH = 165515997
ROOT = "Replication/weaksatd-annotation/"
BLOCK = 256*1024


class RangeFile(io.RawIOBase):
    def __init__(self):
        self.pos = 0
        self.cache = {}

    def seek(self, n, whence=io.SEEK_SET):
        if whence == io.SEEK_END:
            n += LENGTH
        elif whence == io.SEEK_CUR:
            n += self.pos
        self.pos = n
        return n

    def tell(self):
        return self.pos

    def seekable(self):
        return True

    def read(self, n=-1):
        end = LENGTH if n < 0 else min(LENGTH,self.pos+n)
        parts = []
        while self.pos < end:
            start = BLOCK*(self.pos//BLOCK)
            boundary = min(LENGTH,start+BLOCK)
            if start not in self.cache:
                headers = {"Range": f"bytes={start}-{boundary-1}", "User-Agent": "NLBSE-WeakSATD-rule-study"}
                request = urllib.request.Request(URL+f"?offset={start}", headers=headers)
                with urllib.request.urlopen(request, timeout=12) as response:
                    if response.status != 206:
                        raise ValueError("Server did not support reading only the rule files")
                    chunk = response.read()
                if len(chunk) != boundary-start:
                    raise ValueError("Replication archive range size changed")
                self.cache[start] = chunk
            next_pos = min(end,boundary)
            parts.append(self.cache[start][self.pos-start:next_pos-start])
            self.pos = next_pos
        return b"".join(parts)


def main():
    files = {}
    with zipfile.ZipFile(RangeFile()) as archive:
        for entry in archive.infolist():
            name = entry.filename.removeprefix(ROOT)
            if not (name.startswith("WeakSATD/") and name.endswith((".js",".json")) or name in (
                "src/annotateCSV.js", "src/utility.js", "README.md", "package.json"
            )):
                continue
            content = archive.read(entry)
            path = HERE / "upstream" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            files[name] = dict(sha256=hashlib.sha256(content).hexdigest(), bytes=len(content))
    (HERE/"rules_manifest.json").write_text(json.dumps(dict(source=URL, files=files),indent=2)+"\n")
    print(len(files), "source files", flush=True)


if __name__ == "__main__":
    main()
