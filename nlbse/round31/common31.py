"""Signed sources for the compact continuation comparison."""
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
R30=ROOT/'round30'
sys.path.insert(0,str(R30))
from embed import f,checksum,json_at,write_json
from reference29 import reference, R28, match, strict_cut
from distill import load_target,DEST
from compact_oof import signed_source

PARTS=('test_a','test_b','official')
OUT=HERE/'trained'
