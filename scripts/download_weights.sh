#!/usr/bin/env bash
# Usage: ./scripts/download_weights.sh [dest_dir]
#
# Downloads Qwen/Qwen3.5-0.8B at a pinned revision into dest_dir
# (default: weights/qwen3.5-0.8b) with curl, and checks the sha256 of the
# LFS files. Re-running skips files that are already complete. Set HF_TOKEN
# to send an auth header; set HF_ENDPOINT to use a mirror.

set -euo pipefail

repo="Qwen/Qwen3.5-0.8B"
revision="2fc06364715b967f1860aea9cf38778875588b17"
endpoint="${HF_ENDPOINT:-https://huggingface.co}"
dest="${1:-weights/qwen3.5-0.8b}"

# name  size  sha256 (LFS files only; "-" for regular git files)
files=(
    "config.json                                   2907        -"
    "model.safetensors.index.json                  50900       -"
    "model.safetensors-00001-of-00001.safetensors  1746942600  04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"
    "tokenizer.json                                12807982    5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
    "tokenizer_config.json                         16709       -"
    "vocab.json                                    6722759     -"
    "merges.txt                                    3353259     -"
    "chat_template.jinja                           7755        -"
    "preprocessor_config.json                      390         -"
    "video_preprocessor_config.json                385         -"
    "LICENSE                                       11544       -"
)

auth=()
if [[ -n "${HF_TOKEN:-}" ]]; then
    auth=(-H "Authorization: Bearer ${HF_TOKEN}")
fi

mkdir -p "$dest"
for entry in "${files[@]}"; do
    read -r name size sha <<<"$entry"
    out="$dest/$name"
    if [[ -f "$out" && "$(stat -c %s "$out")" == "$size" ]]; then
        echo "ok    $name"
        continue
    fi
    echo "fetch $name ($size bytes)"
    curl -fL --retry 5 --retry-delay 2 -C - "${auth[@]}" \
        -o "$out" "$endpoint/$repo/resolve/$revision/$name"
    actual="$(stat -c %s "$out")"
    if [[ "$actual" != "$size" ]]; then
        echo "download_weights.sh: $name is $actual bytes, expected $size" >&2
        exit 1
    fi
    if [[ "$sha" != "-" ]]; then
        echo "$sha  $out" | sha256sum -c --quiet - || {
            echo "download_weights.sh: sha256 mismatch for $name" >&2
            exit 1
        }
    fi
done
echo "done: $repo@${revision:0:7} -> $dest"
