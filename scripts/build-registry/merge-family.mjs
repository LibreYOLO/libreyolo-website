/*
 * Add families from a scratch rebuild to the current registry, and nothing else.
 *
 * A family that ships between docs releases needs its generated record now,
 * but a full rebuild against an unreleased library commit would also rewrite
 * every other family, the library totals and the recorded source commit with
 * changes the docs tree has not caught up with. This copies only the named
 * families and the task labels they use, and stamps the release they first
 * shipped in, which finalize.mjs cannot know for a family it has never seen.
 *
 * Usage: run rebuild.sh's steps against a pinned checkout, but send the final
 * output to the scratch directory instead of src/data/docs/registry.json, then
 * merge from there (the family's lineage must be in lineages.mjs and its
 * reviewed provenance in upstream/<slug>.json):
 *   python3 scripts/build-registry/collect.py LIBRARY_CHECKOUT SCRATCH VA_JSON
 *   node scripts/build-registry/build-registry.mjs SCRATCH SCRATCH/extracted.json
 *   node scripts/build-registry/emit-registry.mjs SCRATCH/extracted.json src/data/docs/registry.json SCRATCH/registry.json
 *   node scripts/build-registry/finalize.mjs SCRATCH/registry.json src/data/docs/registry.json SCRATCH SCRATCH/final.json
 *   node scripts/build-registry/merge-family.mjs SCRATCH/final.json RELEASE KEY [KEY...]
 */
import fs from 'node:fs'

const [rebuiltPath, release, ...keys] = process.argv.slice(2)
if (!rebuiltPath || !/^\d+\.\d+\.\d+$/.test(release ?? '') || !keys.length) {
  throw new Error('Usage: merge-family.mjs REBUILT_REGISTRY RELEASE KEY [KEY...]')
}
const currentPath = 'src/data/docs/registry.json'
const rebuilt = JSON.parse(fs.readFileSync(rebuiltPath, 'utf8'))
const current = JSON.parse(fs.readFileSync(currentPath, 'utf8'))

for (const key of keys) {
  const family = rebuilt.families[key]
  if (!family) throw new Error(`${key} is not a family in ${rebuiltPath}`)
  if (!family.upstream) throw new Error(`Missing provenance: ${key}`)
  current.families[key] = {
    ...family,
    added_in: release,
    task_added_in: Object.fromEntries(family.registry_keys.map((k) => [k, release])),
  }
  for (const task of family.tasks) {
    const meta = rebuilt.tasks[task]
    if (!meta) throw new Error(`No task label for ${task}; add it to TASK_LABELS in emit-registry.mjs`)
    current.tasks[task] = meta
  }
  console.log(`${key}: tasks ${family.tasks.join(', ')}, ${family.checkpoints.length} checkpoints, added in ${release}`)
}
fs.writeFileSync(currentPath, JSON.stringify(current, null, 2) + '\n')
