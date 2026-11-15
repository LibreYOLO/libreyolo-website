// Merge reviewed provenance and preserve verified historical release metadata.
import fs from 'node:fs'
import path from 'node:path'
const [generated, previous, collected, output] = process.argv.slice(2)
const next = JSON.parse(fs.readFileSync(generated))
const old = JSON.parse(fs.readFileSync(previous))
const capabilities = JSON.parse(fs.readFileSync(path.join(collected, 'capabilities.json')))
const extras = JSON.parse(fs.readFileSync(path.join(collected, 'extras.json')))
const repos = JSON.parse(fs.readFileSync(path.join(collected, 'hf_all.json')))
const source = JSON.parse(fs.readFileSync(path.join(collected, 'source.json')))
next._comment = 'GENERATED FILE. Rebuild with scripts/build-registry/rebuild.sh; reviewed provenance lives in upstream/*.json.'
next.libreyolo_version = '1.8.0'
next.source = { ...next.source, ...source }
if (!next.export_formats.some(f => f.key === 'axelera')) {
  next.export_formats.splice(next.export_formats.findIndex(f => f.key === 'ncnn'), 0, { key: 'axelera', label: 'Axelera', ext: '_axelera_model' })
}
const prefixes = Object.values(next.families).map(f => f.prefix).filter(Boolean)
for (const [key, family] of Object.entries(next.families)) {
  const prev = old.families[key]
  family.extra = extras[key] ?? null
  // A repo belongs to the family with the longest matching prefix, so
  // LibreQwen3VLEmbedding2b does not count as a Qwen3-VL weight.
  const owns = name => family.prefix && name.startsWith(family.prefix) && !prefixes.some(p => p.length > family.prefix.length && name.startsWith(p))
  family.weights_hosted = family.weights_hosted || repos.some(r => owns(r.id.split('/')[1]) && r.siblings?.some(f => /\.(pt|pth|safetensors|ckpt)$/.test(f.rfilename)))
  family.trainable_per_key = Object.fromEntries(family.registry_keys.map(k => [k, capabilities[k]?.train ?? null]))
  family.trainable = Object.values(family.trainable_per_key).some(v => v === true)
  family.capabilities = { train: family.trainable, val: family.registry_keys.some(k => capabilities[k]?.val === true) }
  family.added_in = prev?.added_in ?? '1.8.0'
  // Keys new to an existing page (sam31 on SAM 3) get the current release.
  // RF-DETR keeps its per-task map, which is not keyed by registry key.
  const prevAdded = prev?.task_added_in
  family.task_added_in = prevAdded && !family.registry_keys.some(k => k in prevAdded)
    ? prevAdded
    : Object.fromEntries(family.registry_keys.map(k => [k, prevAdded && k in prevAdded ? prevAdded[k] : '1.8.0']))
  const upstream = path.join('src/data/docs/upstream', family.slug + '.json')
  if (fs.existsSync(upstream)) family.upstream = JSON.parse(fs.readFileSync(upstream))
  else if (prev?.upstream) family.upstream = prev.upstream
  if (!family.upstream) throw new Error(`Missing provenance: ${key}`)
  // Optional `checkpoint_license` in upstream/<slug>.json replaces the license
  // read from each checkpoint's HF repo tag. Use it only when the HF tag
  // disagrees with what the upstream weights publisher declared (weights keep
  // the upstream-declared license; LibreYOLO adds no dataset-derived terms).
  // A string applies to every checkpoint; an object maps checkpoint file names
  // to licenses where they differ (WAFT ships one backbone licence per file).
  const override = family.upstream.checkpoint_license
  if (typeof override === 'string') {
    for (const c of family.checkpoints) c.license = override
  } else if (override) {
    for (const c of family.checkpoints) if (override[c.name]) c.license = override[c.name]
  }
}
// Published weights the released library cannot load. They are on the Hub, so
// the scan finds them, but listing them would send readers to a load error.
// Remove an entry once the library loads it (libreyolo#925).
const NOT_LOADABLE = {
  rfdetr: ['LibreRFDETRn-pose.pt', 'LibreRFDETRs-pose.pt', 'LibreRFDETRm-pose.pt', 'LibreRFDETRl-pose.pt'],
}
for (const [key, names] of Object.entries(NOT_LOADABLE)) {
  const family = next.families[key]
  if (family) family.checkpoints = family.checkpoints.filter(c => !names.includes(c.name))
}
fs.writeFileSync(output, JSON.stringify(next, null, 2) + '\n')
