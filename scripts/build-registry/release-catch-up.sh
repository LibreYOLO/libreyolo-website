#!/bin/sh
# Review the printed delta before merging or deploying. This script never deploys.
set -eu
library_repo=${1:?Usage: release-catch-up.sh LIBRARY_REPO VA_JSON}
benchmark_json=${2:?Pass the Vision Analysis verified-results.v1.json path}
recorded_sha=$(node --input-type=module -e 'import fs from "node:fs"; console.log(JSON.parse(fs.readFileSync("src/data/docs/registry.json")).source.library_commit)')
git -C "$library_repo" rev-parse --verify v1.6.0
# Read the owner's checkout through git; never check out or edit files there.
git -C "$library_repo" log "$recorded_sha"..v1.6.0 --first-parent --oneline
release_scratch=$(mktemp -d /tmp/libreyolo-docs-release.XXXXXX)
cleanup() {
  git -C "$library_repo" worktree remove "$release_scratch/library"
  rm -rf "$release_scratch"
}
git -C "$library_repo" worktree add --detach "$release_scratch/library" v1.6.0
trap cleanup EXIT HUP INT TERM
sh scripts/build-registry/rebuild.sh "$release_scratch/library" "$benchmark_json" "$release_scratch/registry"
rg -n 'TODO\(1\.6\.0\)|TODO\(owner\)|raw.githubusercontent.com/LibreYOLO/libreyolo/dev/' content src PLAN_docs_v1.6.0.md || true
# Then: document the delta and update existing locale twins, stamp only after
# translation, validate every locale, build a Vercel preview, review owner
# archive decision and release date. Production deployment remains owner work.
