"""Export compressed UniXcoder representations for training rows only."""

import numpy as np
import torch
from sklearn.decomposition import PCA

from view_train import HERE,ROOT,batch,batch_plan,checksum,inputs,json_at,write_json
from rotate import load_backbones
from students import tokenize as unix_tokens


OUT=HERE/"representation"


@torch.inference_mode()
def hidden(pipe,tok,texts):
    vectors=np.empty((len(texts),pipe.config.hidden_size),dtype=np.float16)
    pipe.eval()
    for start in range(0,len(texts),2048):
        seq=unix_tokens(texts[start:start+2048],tok,"unixcoder9")
        with torch.autocast("cuda",dtype=torch.float16):
            for ids in batch_plan(list(range(len(seq))),seq,64):
                x,mask=batch(ids,seq,tok.pad_token_id)
                h=pipe.encoder(input_ids=x,attention_mask=mask).last_hidden_state
                averaged=(h.float()*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True)
                vectors[start+np.asarray(ids)]=averaged.half().cpu().numpy()
        if start%32768==0:
            print("teacher representation",start,len(texts),flush=True)
    if not np.isfinite(vectors).all():
        raise ValueError("Nonfinite teacher representation")
    return vectors


def main():
    OUT.mkdir(exist_ok=True)
    texts,y,_,_,roles=inputs()
    rows=roles["fit"]
    source=ROOT/"round13/backbones/unix/selected.json"
    spec=json_at(source)
    teacher,tok=load_backbones().frozen("unix")
    vectors=hidden(teacher,tok,[texts[i] for i in rows])
    subset=np.random.default_rng(42).choice(len(rows),min(40000,len(rows)),replace=False)
    pca=PCA(n_components=48,whiten=True,svd_solver="randomized",iterated_power=3,random_state=42)
    pca.fit(vectors[subset].astype(np.float32))
    embeddings=np.vstack([pca.transform(vectors[i:i+4096].astype(np.float32)) for i in range(0,len(vectors),4096)])
    out=OUT/"teacher.npz"
    np.savez_compressed(out,ids=rows,y=y[rows],features=np.clip(embeddings,-6,6).astype(np.float16),
                        pca_components=pca.components_,pca_mean=pca.mean_,pca_variance=pca.explained_variance_,pca_sample_ids=rows[subset])
    write_json(OUT/"teacher.json",dict(rows=len(rows),source_checkpoint_sha256=spec["sha256"],sha256=checksum(out),
               pca_dims=48,pca_variance=float(pca.explained_variance_ratio_.sum()),fit_subset=len(subset)))
    print("saved",out,flush=True)


if __name__=="__main__":
    main()
