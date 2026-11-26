#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
fetch() {
    local repo=$1 revision=$2 dest=$3 item
    shift 3
    mkdir -p "$dest"
    for item in "$@"; do
        if [[ ! -s "$dest/$item" ]]; then
            curl -fLsS --retry 2 --connect-timeout 10 --max-time 200 \
                "https://huggingface.co/$repo/resolve/$revision/$item" -o "$dest/$item.part"
            mv "$dest/$item.part" "$dest/$item"
        fi
    done
    sha256sum "$dest"/*
}
fetch Salesforce/codet5-small b1ee9570c289f21b5922b9c768a1ce12957bf968 nlbse/models/codet5_small \
    config.json merges.txt vocab.json tokenizer_config.json special_tokens_map.json added_tokens.json pytorch_model.bin &
a=$!
fetch microsoft/unixcoder-base-nine 5f0dc256eb0904af496d427451da5d7f90261676 nlbse/models/unixcoder9 \
    config.json merges.txt vocab.json tokenizer_config.json special_tokens_map.json pytorch_model.bin &
b=$!
wait "$a"
wait "$b"
