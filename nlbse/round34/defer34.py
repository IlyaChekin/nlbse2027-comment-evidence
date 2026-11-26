"""Retain bounded encoder states so the extra head can run after selecting rows."""
import numpy as np
import torch

from attention34 import HIDDEN, N_HEADS, SAMPLE, select_tokens


def operations(width, rows, samples=SAMPLE):
    return 2*rows*(samples*(width*HIDDEN+2*N_HEADS*HIDDEN)+(N_HEADS+1)*HIDDEN*96+96*2)/1e9


class Deferred:
    def __init__(self, head, width, capacity=None):
        free,_=torch.cuda.mem_get_info()
        self.capacity=min(6*1024**3,int(free*.62)) if capacity is None else capacity
        self.bytes_used=0
        self.head=head
        self.width=width
        self.saved=[]

    def capture(self, rows, states, mask):
        needed=states.numel()*states.element_size()+mask.numel()*mask.element_size()
        if self.bytes_used+needed>self.capacity:
            return False
        self.saved.append((rows,states,mask))
        self.bytes_used+=needed
        return True

    @torch.inference_mode()
    def finish(self, picked, result):
        selected=[]
        tokens=[]
        masks=[]
        with torch.autocast('cuda',dtype=torch.float16):
            for rows,states,mask in self.saved:
                local=np.flatnonzero(picked[rows])
                if len(local):
                    x,m=select_tokens(states[local],mask[local])
                    tokens.append(x)
                    masks.append(m)
                    selected.extend(rows[local])
            self.saved.clear()
            if not selected:
                return 0,0.
            value=self.head(torch.cat(tokens),torch.cat(masks)).float().cpu().numpy()
        result[np.asarray(selected)]=value
        return len(selected),operations(self.width,len(selected))
