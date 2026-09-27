#!/bin/sh
# Run from the website root. Input must be a detached, pinned library checkout.
set -eu
library_checkout=${1:?Usage: rebuild.sh LIBRARY_CHECKOUT VA_JSON [SCRATCH_DIR]}
benchmark_json=${2:?Pass the Vision Analysis verified-results.v1.json path}
scratch_dir=${3:-$(mktemp -d)}
python3 scripts/build-registry/collect.py "$library_checkout" "$scratch_dir" "$benchmark_json"
node scripts/build-registry/build-registry.mjs "$scratch_dir" "$scratch_dir/extracted.json"
node scripts/build-registry/emit-registry.mjs "$scratch_dir/extracted.json" src/data/docs/registry.json "$scratch_dir/registry.json"
node scripts/build-registry/finalize.mjs "$scratch_dir/registry.json" src/data/docs/registry.json "$scratch_dir" src/data/docs/registry.json
