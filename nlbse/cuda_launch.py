"""Wait briefly for a CUDA context, then run the requested GPU entry point."""
import ctypes
import runpy
import sys
import time
from pathlib import Path

import torch

driver = ctypes.CDLL("libcuda.so.1")
for attempt in range(1, 9):
    status = driver.cuInit(0)
    if status == 0:
        try:
            torch.empty(1, device="cuda")
            break
        except RuntimeError as exc:
            print(f"CUDA runtime attempt {attempt}: {exc}", file=sys.stderr, flush=True)
    else:
        print(f"CUDA driver attempt {attempt}: {status}", file=sys.stderr, flush=True)
    time.sleep(2)
else:
    sys.exit("Unable to allocate a CUDA context after eight attempts.")

sys.argv.pop(0)
sys.path.insert(0, str(Path(sys.argv[0]).resolve().parent))
runpy.run_path(sys.argv[0], run_name="__main__")
