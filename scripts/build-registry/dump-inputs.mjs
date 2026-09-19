/*
 * Build the input bundle that scripts/build-registry reads.
 *
 * The extractor deliberately never imports libreyolo: it reads a directory of
 * dumped sources so the registry can be pinned to an exact library revision.
 * At the 1.5.0 release that bundle only existed on one machine and could not
 * be reproduced, which is why domedetr had to be hand-added to registry.json.
 * This script makes the bundle mechanical:
 *
 *   node scripts/build-registry/dump-inputs.mjs <libreyolo-repo> <rev> <out-dir> \
 *        --hf <hf-org-listing.json> --va <verified-results.v1.json>
 *
 * Everything source-derived is read from the named git revision, never from
 * the working tree, so a dirty checkout cannot leak into the registry. The
 * two network-derived inputs (the Hugging Face org listing and the
 * vision-analysis results) are passed in as files, because they must be
 * captured at release time to mean anything.
 *
 * Bundle layout consumed by build-registry.mjs:
 *   registry.py          libreyolo/models/registry.py at <rev>
 *   tasks.py             libreyolo/tasks.py at <rev>
 *   export_support.md    docs/export_support.md at <rev>
 *   models/              every .py under libreyolo/models, flattened
 *   family_files.json    registry key -> flattened filename defining it
 *   optional.json        [{cls, extra}] from OPTIONAL_MODELS
 *   added_in.json        registry key -> first release tag shipping it
 *   hf.json, va.json     copied from --hf / --va
 */

import fs from 'fs'
import path from 'path'
import { execFileSync } from 'child_process'
import { LINEAGES } from './lineages.mjs'

const args = process.argv.slice(2)
const positional = args.filter((a) => !a.startsWith('--'))
const opt = (name) => {
  const i = args.indexOf(`--${name}`)
  return i === -1 ? null : args[i + 1]
}
const [repo, rev, outDir] = positional
if (!repo || !rev || !outDir || !opt('hf') || !opt('va')) {
  console.error('usage: node dump-inputs.mjs <libreyolo-repo> <rev> <out-dir> --hf <hf.json> --va <va.json>')
  process.exit(2)
}

const git = (...a) =>
  execFileSync('git', ['-C', repo, ...a], { maxBuffer: 64 * 1024 * 1024 }).toString()
const show = (p) => git('show', `${rev}:${p}`)

fs.mkdirSync(path.join(outDir, 'models'), { recursive: true })
const put = (name, text) => fs.writeFileSync(path.join(outDir, name), text)

/* Source-derived, at the pinned revision. */
put('registry.py', show('libreyolo/models/registry.py'))
put('tasks.py', show('libreyolo/tasks.py'))
put('export_support.md', show('docs/export_support.md'))

/* Every model source, flattened: libreyolo/models/vlm/florence2.py becomes
 * models/vlm__florence2.py. The extractor scans all of them for class
 * inheritance, so base classes ride along for free. */
const files = git('ls-tree', '-r', '--name-only', rev, 'libreyolo/models')
  .split('\n')
  .filter((f) => f.endsWith('.py'))
const dumped = {} // flattened name -> original repo path
for (const f of files) {
  const flat = f.slice('libreyolo/models/'.length).replaceAll('/', '__')
  dumped[flat] = f
  put(path.join('models', flat), show(f))
}

/* family_files.json: which dumped source declares each lineage key. The FAMILY
 * constant is the authoritative marker; a family package's model.py is the
 * fallback for classes that omit it. */
const keys = LINEAGES.flatMap((l) => l.keys)
const familyFiles = {}
const missing = []
for (const key of keys) {
  const marker = new RegExp(`FAMILY\\s*(?::[^=\\n]*)?=\\s*"${key}"`)
  const hits = Object.keys(dumped).filter((flat) =>
    marker.test(fs.readFileSync(path.join(outDir, 'models', flat), 'utf8'))
  )
  const preferred = hits.find((h) => h === `${key}__model.py`) ?? hits[0]
  if (preferred) familyFiles[key] = preferred
  else if (dumped[`${key}__model.py`]) familyFiles[key] = `${key}__model.py`
  else missing.push(key)
}
put('family_files.json', JSON.stringify(familyFiles, null, 2))
if (missing.length) console.error('no source found for:', missing.join(', '))

/* optional.json: the lazily registered tiers and their pip extras. */
const inventory = show('libreyolo/models/inventory.py')
const optional = [...inventory.matchAll(/\(\s*"([\w.]+)",\s*"(\w+)",\s*(?:"(\w+)"|None),/g)]
  .map((m) => ({ module: m[1], cls: m[2], extra: m[3] ?? null }))
put('optional.json', JSON.stringify(optional, null, 2))

/* added_in.json: the release each family first shipped in, from the first
 * commit touching its defining source and the earliest release tag containing
 * that commit. A family newer than the last tag maps to null. */
const semver = (t) => t.slice(1).split('.').map(Number)
const releaseTags = git('tag')
  .split('\n')
  .filter((t) => /^v\d+\.\d+\.\d+$/.test(t))
  .sort((a, b) => {
    const [x, y] = [semver(a), semver(b)]
    return x[0] - y[0] || x[1] - y[1] || x[2] - y[2]
  })
const tagCache = {}
function firstRelease(file) {
  if (file in tagCache) return tagCache[file]
  let version = null
  try {
    // No --follow: rename detection walks yolo9_e2e/model.py back into
    // yolo9's history and antedates the family by several releases. The
    // commit that created the exact path is the one that shipped the family.
    const adds = git('log', '--diff-filter=A', '--format=%H', '--', file)
      .trim()
      .split('\n')
    const first = adds[adds.length - 1]
    const containing = new Set(git('tag', '--contains', first).trim().split('\n'))
    version = releaseTags.find((t) => containing.has(t))?.slice(1) ?? null
  } catch {
    version = null
  }
  tagCache[file] = version
  return version
}
const addedIn = {}
for (const [key, flat] of Object.entries(familyFiles)) {
  addedIn[key] = firstRelease(dumped[flat])
}
put('added_in.json', JSON.stringify(addedIn, null, 2))

/* Network-derived inputs, captured by the caller. */
put('hf.json', fs.readFileSync(opt('hf'), 'utf8'))
put('va.json', fs.readFileSync(opt('va'), 'utf8'))

console.log(`bundle written to ${outDir}: ${files.length} sources, ${Object.keys(familyFiles).length}/${keys.length} families mapped`)
