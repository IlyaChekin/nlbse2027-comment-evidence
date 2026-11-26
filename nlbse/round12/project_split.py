"""Freeze project-disjoint supervision, development and testing from official train."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from pipeline import TRAIN, load, write_json

TEST = frozenset({"chromium", "ffmpeg", "qemu"})
DEV = frozenset({"mozilla firefox", "android", "php", "tcpdump"})
ALIASES = {"chrome": "chromium", "linux kernel": "linux", "php-src": "php"}
TOKEN = re.compile(r'R"(?P<delim>[^ ()\\\t\r\n]{0,16})\(.*?\)(?P=delim)"|'
                   r'"(?:\\.|[^"\\])*"|\x27(?:\\.|[^\x27\\])*\x27|'
                   r'(?P<comment>//[^\r\n]*|/\*.*?(?:\*/|\Z))|\w+|[^\s\w]', re.S)
SPLIT = HERE/"partition.npz"
PLAN = HERE/"partition.json"


def family(name):
    key = name.strip().casefold()
    return ALIASES.get(key, key)


def checksum(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fingerprint(text):
    # Length delimiters preserve token boundaries and literal contents.
    tokens = [m[0] for m in TOKEN.finditer(text) if m["comment"] is None]
    return hashlib.sha256("".join(f"{len(t)}:{t}" for t in tokens).encode()).digest()


def sample_file(group, path):
    key = f"NLBSE-R12-42\0{group}\0{path}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big") % 100 < 10


def partition():
    texts,y,meta=load(TRAIN,audit=True)
    groups=[family(p) for p in meta["Projectname"]]
    paths=meta["Filepath"]
    manifest=json.loads((ROOT/"data/manifest.json").read_text())
    digest=checksum(TRAIN)
    if digest!=manifest["files"][0]["sha256"]:
        raise ValueError("Official train SHA-256 is different")
    copies=[fingerprint(t) for t in texts]
    reserved={name:[] for name in ("fit","probe","dev","test")}
    for i,(p,path) in enumerate(zip(groups,paths,strict=True)):
        bucket="test" if p in TEST else "dev" if p in DEV else "probe" if sample_file(p,path) else "fit"
        reserved[bucket].append(i)
    blocked=set()
    exclusions={}
    final={}
    for name in ("test","dev","probe","fit"):
        retained=np.asarray([i for i in reserved[name] if copies[i] not in blocked],dtype=np.int32)
        exclusions[name]=len(reserved[name])-len(retained)
        final[name]=retained
        blocked.update(copies[i] for i in retained)
    training=set(zip((groups[i] for i in final["fit"]),(paths[i] for i in final["fit"]),strict=True))
    probe=set(zip((groups[i] for i in final["probe"]),(paths[i] for i in final["probe"]),strict=True))
    if training & probe or any(groups[i] in TEST|DEV for i in final["fit"]):
        raise ValueError("Heldout project or file is present in primary fit")
    pending=SPLIT.with_suffix(".part.npz")
    np.savez_compressed(pending,**final)
    pending.replace(SPLIT)
    info=dict(input_sha256=digest,partition_sha256=checksum(SPLIT),aliases=ALIASES,
              normalized_copy_exclusions=exclusions,roles={name:dict(rows=len(ids),positives=y[ids].sum(0).tolist(),
              projects=dict(Counter(groups[i] for i in ids))) for name,ids in final.items()})
    write_json(PLAN,info)
    print(json.dumps(info,indent=2),flush=True)


def read_partition():
    plan=json.loads(PLAN.read_text())
    if checksum(SPLIT)!=plan["partition_sha256"] or checksum(TRAIN)!=plan["input_sha256"]:
        raise ValueError("Partition or input data has changed")
    with np.load(SPLIT) as stored:
        return {part:stored[part] for part in ("fit","probe","dev","test")}


if __name__=="__main__":
    partition()
