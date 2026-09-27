#!/bin/sh
# Review the printed delta before merging or deploying. This script never deploys.
set -eu
library_repo=${1:?Usage: release-catch-up.sh LIBRARY_REPO VA_JSON}
benchmark_json=${2:?Pass the Vision Analysis verified-results.v1.json path}
recorded_sha=$(node --input-type=module -e 'import fs from "node:fs"; console.log(JSON.parse(fs.readFileSync("src/data/docs/registry.json")).source.library_commit)')
# Fetching only updates remote refs and tags; the owner's checkout is untouched.
git -C "$library_repo" fetch --quiet origin --tags
git -C "$library_repo" rev-parse --verify v1.6.0
# Read the owner's checkout through git; never check out or edit files there.
if git -C "$library_repo" merge-base --is-ancestor "$recorded_sha" v1.6.0; then
  git -C "$library_repo" log "$recorded_sha"..v1.6.0 --first-parent --oneline
else
  echo "WARNING: $recorded_sha is not an ancestor of v1.6.0 (dev history was rewritten)."
  echo "Find the tree-identical commit on dev and log from there instead."
fi
release_scratch=$(mktemp -d /tmp/libreyolo-docs-release.XXXXXX)
cleanup() {
  git -C "$library_repo" worktree remove "$release_scratch/library"
  rm -rf "$release_scratch"
}
git -C "$library_repo" worktree add --detach "$release_scratch/library" v1.6.0
trap cleanup EXIT HUP INT TERM
sh scripts/build-registry/rebuild.sh "$release_scratch/library" "$benchmark_json" "$release_scratch/registry"
git diff --stat -- src/data/docs/registry.json
# The frozen 1.5.0 tree must equal the website main it was copied from.
git fetch --quiet origin main
if ! git diff --quiet 19c1ca6 origin/main -- content/docs src/data/docs/registry.json src/data/docs/nav.json src/data/docs/upstream; then
  echo "WARNING: origin/main changed 1.5.0 docs after the snapshot (19c1ca6); re-take content/archive/docs/v1.5.0 and src/data/docs/archive/v1.5.0 from origin/main."
fi
grep -rnE 'TODO\(1\.6\.0\)|TODO\(owner\)|raw\.githubusercontent\.com/LibreYOLO/libreyolo/dev/' content/docs src || true
# Then: document the delta and update existing locale twins, stamp only after
# translation, validate every locale, and build a Vercel preview. Production
# deployment remains owner work.
