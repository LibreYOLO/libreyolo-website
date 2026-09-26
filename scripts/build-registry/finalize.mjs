// Merge reviewed provenance and preserve verified historical release metadata.
import fs from 'node:fs'
import path from 'node:path'
const [generated, previous, collected, output] = process.argv.slice(2)
const next = JSON.parse(fs.readFileSync(generated))
const old = JSON.parse(fs.readFileSync(previous))
const source = JSON.parse(fs.readFileSync(path.join(collected, 'source.json')))
next._comment = 'GENERATED FILE. Rebuild with scripts/build-registry/rebuild.sh; reviewed provenance lives in upstream/*.json.'
next.libreyolo_version = '1.6.0'
next.source = { ...next.source, ...source }
for (const [key, family] of Object.entries(next.families)) {
  const prev = old.families[key]
  family.added_in = prev?.added_in ?? '1.6.0'
  family.task_added_in = prev?.task_added_in ?? Object.fromEntries(family.registry_keys.map(k => [k, '1.6.0']))
  const upstream = path.join('src/data/docs/upstream', family.slug + '.json')
  if (fs.existsSync(upstream)) family.upstream = JSON.parse(fs.readFileSync(upstream))
  else if (prev?.upstream) family.upstream = prev.upstream
  if (!family.upstream) throw new Error(`Missing provenance: ${key}`)
}
fs.writeFileSync(output, JSON.stringify(next, null, 2) + '\n')
