#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
fetch() {
    local repo="$1" revision="$2" dest="$3" file
    shift 3
    mkdir -p "$dest"
    for file; do
        if ! test -s "$dest/$file"; then
            curl -fLsS --retry 3 --connect-timeout 15 --max-time 540 \
                "https://huggingface.co/$repo/resolve/$revision/$file?download=true" -o "$dest/$file.part"
            mv "$dest/$file.part" "$dest/$file"
        fi
        sha256sum "$dest/$file"
    done
}
fetch Salesforce/codet5p-220m 2b92f36e2782341a50551759fdba0dd15e821f99 nlbse/models/codet5p \
    config.json vocab.json merges.txt tokenizer_config.json special_tokens_map.json added_tokens.json pytorch_model.bin &
t5=$!
fetch microsoft/graphcodebert-base 2b0488a7bb0eefc7041f1bb2cad1ab26b0da269d nlbse/models/graphcodebert \
    config.json vocab.json merges.txt tokenizer_config.json special_tokens_map.json pytorch_model.bin &
graph=$!
result=0
wait "$t5" || result=1
wait "$graph" || result=1
if [[ "$result" == 0 && -s nlbse/portfolio/backbones.sha256 && -s nlbse/portfolio/backbone_weights.sha256 ]]; then
    sha256sum -c nlbse/portfolio/backbones.sha256 nlbse/portfolio/backbone_weights.sha256
fi
exit "$result"
