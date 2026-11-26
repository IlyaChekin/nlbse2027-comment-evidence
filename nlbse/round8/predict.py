"""Run the measured precision and throughput variants on an ordered Function CSV."""
import argparse
import csv
import json
from pathlib import Path

from frontier import BASE, HERE, PORTABLE, THRESHOLDS, load_checkpoint, run, run_policy, sha256
from tokenizer_speed import DeferredOffsets


def predict(source, output, policy, frozen, device, token_policy, variant):
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if "Function" not in (reader.fieldnames or ()):
            raise ValueError("Input CSV must have a Function column")
        texts = [row["Function"] for row in reader]
    if not texts or any(text is None for text in texts):
        raise ValueError("Input CSV must contain function text in every row")

    if variant is not None and policy != "tokens":
        raise ValueError("An explicit token variant requires --policy tokens")
    config = None
    if policy != "exact":
        if device != "cuda":
            raise ValueError("Learned frontier variants were implemented and measured on CUDA")
        config_path = token_policy if policy=="tokens" else frozen
        config = json.loads(config_path.read_text())
        checkpoint = config["encoder_sha256"] if policy=="tokens" else config["checkpoint_sha256"]
        if sha256(PORTABLE/"model.safetensors") != checkpoint:
            raise ValueError("Prediction head and encoder checkpoint do not match")

    model, tokenizer = load_checkpoint(PORTABLE, device)
    tokenizer = DeferredOffsets(tokenizer)
    if policy == "exact":
        prob, _ = run_policy(model, tokenizer, texts, device, BASE, prefetch=device=="cuda")
        thresholds = THRESHOLDS
    elif policy == "tokens":
        from late_tokens import predict_variant
        prob, _ = predict_variant(model, tokenizer, texts, config, variant or config["selected"])
        thresholds = (.5,.5)
    else:
        prob, _ = run(model, tokenizer, texts, config, policy, prefetch=True)
        thresholds = (.5,.5)
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("vulnerability","MAT"))
        writer.writerows((prob > thresholds).astype(int))


if __name__=="__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--policy",choices=("exact","four","adaptive","six","wide","tokens"),default="exact")
    ap.add_argument("--frozen",type=Path,default=HERE/"frozen/policy.json")
    ap.add_argument("--token-policy",type=Path,default=HERE/"tokens_frozen/policy.json")
    ap.add_argument("--variant",help="Choose a frozen token policy by name.")
    ap.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    args=ap.parse_args()
    predict(args.input,args.output,args.policy,args.frozen,args.device,args.token_policy,args.variant)
