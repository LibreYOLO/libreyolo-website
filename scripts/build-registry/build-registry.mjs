/*
 * Prototype of scripts/build-registry.
 *
 * Reads only generated or authoritative sources and emits the mechanical half
 * of registry.json. Nothing here is transcribed by hand:
 *
 *   docs/export_support.md      generated from libreyolo/export/support.py
 *   each family's model.py       FILENAME_PREFIX, SUPPORTED_TASKS, _SIZES
 *   libreyolo/models/registry.py MODEL_GROUPS tiers
 *   HF org listing              which checkpoints actually exist
 *   vision-analysis results     benchmark rows
 *
 * Upstream metadata (paper, org, license, BibTeX) is deliberately NOT produced
 * here: it needs per-family human verification against the authors' own
 * citation block, so it is merged from src/data/docs/upstream/<slug>.json.
 */

import fs from 'fs'
import path from 'path'

const SRC = process.argv[2]
const OUT = process.argv[3]
const read = (f) => fs.readFileSync(path.join(SRC, f), 'utf8')

/* Lineages: one page per upstream lineage, first key is primary. */
const LINEAGES = [
  { slug: 'yolov9', display: 'YOLOv9', keys: ['yolo9', 'yolo9_e2e', 'yolo9_p2'] },
  { slug: 'rf-detr', display: 'RF-DETR', keys: ['rfdetr'] },
  { slug: 'edgecrafter', display: 'EdgeCrafter', keys: ['ec'] },
  { slug: 'rt-detr', display: 'RT-DETR', keys: ['rtdetr', 'rtdetrv2', 'rtdetrv4'] },
  { slug: 'd-fine', display: 'D-FINE', keys: ['dfine'] },
  { slug: 'deim', display: 'DEIM', keys: ['deim', 'deimv2'] },
  { slug: 'yolo-nas', display: 'YOLO-NAS', keys: ['yolonas'] },
]

/* ── tiers ──────────────────────────────────────────────────────── */
function parseTiers() {
  const text = read('registry.py')
  const tiers = {}
  // MODEL_GROUPS maps each family key to its tier: {"yolo9": "g0", ...}
  const block = text.slice(text.indexOf('MODEL_GROUPS:'))
  for (const m of block.matchAll(/"([a-z0-9_]+)"\s*:\s*"(g[0-4]|s)"/g)) tiers[m[1]] = m[2]
  return tiers
}

/* ── export matrix ──────────────────────────────────────────────── */
function parseExport() {
  const lines = read('export_support.md').split('\n')
  const header = lines.find((l) => l.startsWith('| Family | Task |'))
  const formats = header.split('|').slice(3, -1).map((s) => s.trim())
  const out = {}
  for (const line of lines) {
    if (!line.startsWith('| ') || line.startsWith('| Family') || line.startsWith('| ---')) continue
    const cells = line.split('|').slice(1, -1).map((s) => s.trim())
    if (cells.length < 3 + formats.length - 1) continue
    const [family, task, ...rest] = cells
    if (!/^[a-z0-9_]+$/.test(family)) continue
    out[family] ??= {}
    out[family][task] = {}
    formats.forEach((fmt, i) => {
      const v = (rest[i] || '').trim()
      out[family][task][fmt] = v === '✓' ? 'validated' : v === 'exp' ? 'available' : 'blocked'
    })
  }
  return { formats, support: out }
}

/* ── per-family basics ──────────────────────────────────────────── */
function parseFamily(key) {
  let text
  try { text = read(`model_${key}.py`) } catch { return null }
  const prefix = text.match(/FILENAME_PREFIX[^=]*=\s*"([^"]+)"/)?.[1] ?? null
  // A family that does not declare SUPPORTED_TASKS inherits BaseModel's
  // ("detect",). Verified at libreyolo/models/base/model.py:122.
  const tasksRaw = text.match(/SUPPORTED_TASKS[^=]*=\s*\(([^)]*)\)/)?.[1]
  const tasks = tasksRaw === undefined
    ? ['detect']
    : [...tasksRaw.matchAll(/"([a-z]+)"/g)].map((m) => m[1])
  /*
   * Size tables. Every `NAME = {"code": px}` constant in the class is collected
   * first, because TASK_INPUT_SIZES usually maps a task to one of those names
   * (`"segment": SEG_INPUT_SIZES`) rather than to an inline dict. Following that
   * indirection is what stops segmentation checkpoints being labelled with the
   * detection resolution.
   */
  const tables = {}
  for (const m of text.matchAll(/^\s{4}([A-Z][A-Z0-9_]*)\s*(?::[^=\n]*)?=\s*\{([^}]*)\}/gm)) {
    if (!m[1].endsWith('INPUT_SIZES')) continue
    const table = Object.fromEntries(
      [...m[2].matchAll(/"([a-z0-9]+)"\s*:\s*(\d+)/g)].map((x) => [x[1], +x[2]])
    )
    if (Object.keys(table).length) tables[m[1]] = table
  }
  const sizes = tables.INPUT_SIZES ?? {}

  const taskBlock = text.match(/TASK_INPUT_SIZES[^=]*=\s*\{([\s\S]*?)\n    \}/)?.[1] ?? ''
  const taskSizes = {}
  for (const m of taskBlock.matchAll(/"([a-z]+)"\s*:\s*([A-Z][A-Z0-9_]*|\{[^}]*\})/g)) {
    const [, task, ref] = m
    const table = ref.startsWith('{')
      ? Object.fromEntries([...ref.matchAll(/"([a-z0-9]+)"\s*:\s*(\d+)/g)].map((x) => [x[1], +x[2]]))
      : tables[ref]
    // An unresolvable reference yields no entry, so the emitter reports an
    // empty cell rather than falling back to another task's resolution.
    if (table && Object.keys(table).length) taskSizes[task] = table
  }
  const unsupported = [...(text.match(/UNSUPPORTED_TRAIN_PARAMS[^=]*=\s*\{([\s\S]*?)\}/)?.[1] ?? '')
    .matchAll(/"([a-z_]+)"/g)].map((m) => m[1])
  return { key, prefix, tasks, sizes, task_sizes: taskSizes, unsupported_train_params: unsupported }
}

/* ── checkpoints from the HF org ────────────────────────────────── */
const TASK_SUFFIX = { seg: 'segment', pose: 'pose', obb: 'obb', cls: 'classify', sem: 'semantic', point: 'point' }

// Dataset tokens that can appear in a checkpoint name. A checkpoint trained on
// something other than the family default says so in its filename, and that
// token is the only mechanical signal we have for it.
const DATASETS = { visdrone: 'VisDrone2019-DET', imagenette: 'Imagenette', sidd: 'SIDD' }

function parseCheckpoints(prefixes) {
  const repos = JSON.parse(read('hf.json'))
  const byPrefix = {}
  // Longest prefix wins, so LibreYOLO9E2E is not swallowed by LibreYOLO9.
  const sorted = [...prefixes].sort((a, b) => b.length - a.length)
  for (const repo of repos) {
    const id = repo.id.split('/')[1]
    const prefix = sorted.find((p) => id.startsWith(p))
    if (!prefix) continue
    const rest = id.slice(prefix.length)
    const parts = rest.split('-')
    const size = parts[0]
    if (!size) continue
    const suffix = parts.slice(1).find((p) => TASK_SUFFIX[p])
    const datasetToken = parts.slice(1).find((p) => DATASETS[p])
    // The HF repo's own license tag is authoritative: it is the thing the
    // page tells readers to check, so the page must not disagree with it.
    const licenseTag = (repo.tags || []).find((t) => t.startsWith('license:'))
    byPrefix[prefix] ??= []
    byPrefix[prefix].push({
      name: `${id}.pt`,
      size,
      task: TASK_SUFFIX[suffix] || 'detect',
      dataset: datasetToken ? DATASETS[datasetToken] : null,
      license: licenseTag ? licenseTag.slice('license:'.length) : null,
    })
  }
  return byPrefix
}

/*
 * The release a family first appeared in, taken from the first commit that
 * added its package and the earliest release tag containing that commit.
 * Passed in from the caller because it needs the git repo, not a dumped file.
 */
function parseAddedIn() {
  try { return JSON.parse(read('added_in.json')) } catch { return {} }
}

/* ── benchmarks from Vision Analysis ────────────────────────────── */
// Vision Analysis keys some families differently from the library registry.
const VA_KEY = { yolo9: 'yolov9' }
const vaKeyFor = (key) => VA_KEY[key] ?? key

function parseBenchmarks(keys) {
  const results = JSON.parse(read('va.json')).results
  const wanted = new Map(keys.map((k) => [vaKeyFor(k), k]))
  const out = {}
  for (const r of results) {
    const key = wanted.get(r.model.family)
    if (!key) continue
    if (r.runtime?.format !== 'pytorch') continue
    const task = r.model.id.includes('-seg-') ? 'segment' : 'detect'
    const row = {
      id: r.model.id,
      size: r.model.variant,
      imgsz: r.model.input_size,
      params_m: r.model_stats?.params_millions || null,
      map: +(r.accuracy.mAP_50_95 * 100).toFixed(1),
      images: r.dataset.num_images,
      hardware: r.hardware.id,
      created: r.created_at,
    }
    const bucket = (out[key] ??= {})
    const list = (bucket[task] ??= [])
    // Keep the newest run per model id.
    const prev = list.findIndex((x) => x.id === row.id)
    if (prev === -1) list.push(row)
    else if (row.created > list[prev].created) list[prev] = row
  }
  return out
}

/* ── assemble ───────────────────────────────────────────────────── */
const tiers = parseTiers()
const addedIn = parseAddedIn()
const { formats, support } = parseExport()
const families = {}
for (const key of LINEAGES.flatMap((l) => l.keys)) {
  const f = parseFamily(key)
  if (f) families[key] = { ...f, tier: tiers[key] ?? null, export: support[key] ?? {} }
}
const checkpoints = parseCheckpoints(Object.values(families).map((f) => f.prefix).filter(Boolean))
const benchmarks = parseBenchmarks(Object.keys(families))

const report = { lineages: [], missing: [], discrepancies: [] }
for (const lin of LINEAGES) {
  const primary = families[lin.keys[0]]
  if (!primary) { report.missing.push(lin.slug); continue }
  const all = lin.keys.flatMap((k) => checkpoints[families[k]?.prefix] ?? [])
  const declared = new Set(lin.keys.flatMap((k) => families[k]?.tasks ?? []))
  // A published weight file for a task the registry does not declare is a
  // discrepancy, not content. It is reported, never rendered.
  const ckpts = all.filter((c) => declared.has(c.task))
  for (const c of all.filter((c) => !declared.has(c.task))) {
    report.discrepancies.push({
      slug: lin.slug, kind: 'weights-without-registry-task', file: c.name, task: c.task,
    })
  }
  report.lineages.push({
    slug: lin.slug,
    display: lin.display,
    keys: lin.keys,
    tier: primary.tier,
    tasks: [...new Set(lin.keys.flatMap((k) => families[k]?.tasks ?? []))],
    added_in_raw: null,
    sizes: Object.fromEntries(lin.keys.map((k) => [k, families[k]?.sizes ?? {}])),
    task_sizes: Object.fromEntries(lin.keys.map((k) => [k, families[k]?.task_sizes ?? {}])),
    prefixes: Object.fromEntries(lin.keys.map((k) => [k, families[k]?.prefix ?? null])),
    checkpoints: ckpts.length,
    checkpoint_rows: ckpts,
    checkpoint_names: ckpts.map((c) => c.name),
    // The lineage appeared when its earliest member did.
    added_in: lin.keys.map((k) => addedIn[k]).filter(Boolean).sort()[0] ?? null,
    task_added_in: Object.fromEntries(lin.keys.map((k) => [k, addedIn[k] ?? null])),
    export: Object.fromEntries(lin.keys.map((k) => [k, families[k]?.export ?? {}])),
    benchmarks: Object.fromEntries(lin.keys.map((k) => [k, benchmarks[k] ?? {}])),
    unsupported_train_params: Object.fromEntries(lin.keys.map((k) => [k, families[k]?.unsupported_train_params ?? []])),
  })
}
report.export_formats = formats
fs.writeFileSync(OUT, JSON.stringify(report, null, 2))

for (const l of report.lineages) {
  const benchCount = Object.values(l.benchmarks).flatMap((b) => Object.values(b).flat()).length
  console.log(
    `${l.slug.padEnd(13)} tier=${String(l.tier).padEnd(3)} tasks=${l.tasks.join(',').padEnd(28)} ckpts=${String(l.checkpoints).padStart(2)} bench=${benchCount}`
  )
}
if (report.missing.length) console.log('MISSING:', report.missing.join(', '))
