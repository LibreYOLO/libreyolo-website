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
import { LINEAGES } from './lineages.mjs'

/*
 * Where each family's class is defined. Eager families live at
 * libreyolo/models/<key>/model.py; the lazily-registered SAM, VLM and
 * open-vocabulary tiers live in shared modules, so the map is built by reading
 * the FAMILY constant out of every dumped source rather than guessing paths.
 */
const FAMILY_FILES = (() => {
  try { return JSON.parse(read('family_files.json')) } catch { return {} }
})()

/* Pip extras, from OPTIONAL_MODELS for the lazy tiers plus the eager ones. */
const EAGER_EXTRAS = { rfdetr: 'rfdetr', dinov2: 'rfdetr', clip: 'clip', siglip2: 'siglip2', midas: 'midas', eomt: 'eomt', sensenovavision: 'sensenova', libremodus: 'modus' }
const OPTIONAL_EXTRAS = (() => {
  try {
    const rows = JSON.parse(read('optional.json'))
    return Object.fromEntries(rows.map((r) => [r.cls, r.extra]))
  } catch { return {} }
})()

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
      // The generated table renamed its middle tier from "exp" to "available"
      // when ADR 0011's vocabulary changed. Accept both: matching only the old
      // spelling would silently reclassify every supported-but-unvalidated
      // combination as blocked, which understates support across the library.
      out[family][task][fmt] =
        v === '✓' ? 'validated'
          : v === 'exp' || v === 'available' ? 'available'
            : v === '' ? 'blocked'
              : (() => { throw new Error(`Unknown export cell value ${JSON.stringify(v)} for ${family}/${task}/${fmt}`) })()
    })
  }
  return { formats, support: out }
}

/*
 * SUPPORTED_TASKS declared by each tier base class, read from the dumped base
 * modules rather than assumed. A class that declares nothing inherits from its
 * own tier, not from BaseModel.
 */
/*
 * SUPPORTED_TASKS by class name, across EVERY dumped source, not just the tier
 * base modules. Families also subclass each other: LibreFeyNobg extends
 * LibreBiRefNet and declares no tasks of its own, so looking only at the tier
 * bases resolved it to BaseModel's ("detect",) and labelled a matting model as
 * a detector. Inheritance is followed transitively.
 */
const CLASS_TASKS = {}
const CLASS_BASE = {}
for (const file of (() => { try { return fs.readdirSync(path.join(SRC, 'models')) } catch { return [] } })()) {
  let text
  try { text = read(path.join('models', file)) } catch { continue }
  for (const m of text.matchAll(/^class\s+(\w+)\(([^)]*)\)/gm)) {
    const [, cls, bases] = m
    CLASS_BASE[cls] ??= (bases.split(',')[0] || '').trim().split('.').pop()
    const after = text.slice(m.index, text.indexOf('\nclass ', m.index + 1) === -1 ? undefined : text.indexOf('\nclass ', m.index + 1))
    const t = after.match(/SUPPORTED_TASKS[^=]*=\s*\(([^)]*)\)/)?.[1]
    if (t) CLASS_TASKS[cls] ??= [...t.matchAll(/"([a-z]+)"/g)].map((x) => x[1])
  }
}

function inheritedTasks(baseName, seen = new Set()) {
  const key = String(baseName).split('.').pop()
  if (!key || seen.has(key)) return ['detect']
  seen.add(key)
  if (CLASS_TASKS[key]) return CLASS_TASKS[key]
  if (CLASS_BASE[key]) return inheritedTasks(CLASS_BASE[key], seen)
  return ['detect']
}

/* ── per-family basics ──────────────────────────────────────────── */
function parseFamily(key) {
  let text
  const file = FAMILY_FILES[key]
  try { text = read(file ? path.join('models', file) : `model_${key}.py`) } catch { return null }
  // A shared module defines several families, so narrow to this one's class.
  if (file && /FAMILY[^=]*=\s*"/.test(text)) {
    const marker = new RegExp(`FAMILY[^=]*=\\s*"${key}"`)
    const at = text.search(marker)
    if (at !== -1) {
      const classStart = text.lastIndexOf('\nclass ', at)
      const nextClass = text.indexOf('\nclass ', at)
      text = text.slice(classStart === -1 ? 0 : classStart, nextClass === -1 ? text.length : nextClass)
    }
  }
  const prefix = text.match(/FILENAME_PREFIX[^=]*=\s*"([^"]+)"/)?.[1] ?? null
  /*
   * SUPPORTED_TASKS is inherited when a class does not declare it, and the
   * tier base classes disagree with BaseModel: the SAM tier defaults to
   * ("segment",), not ("detect",). Falling back to BaseModel for everything
   * would label all six promptable-segmentation pages as detectors.
   */
  const tasksRaw = text.match(/SUPPORTED_TASKS[^=]*=\s*\(([^)]*)\)/)?.[1]
  let tasks
  if (tasksRaw !== undefined) {
    tasks = [...tasksRaw.matchAll(/"([a-z]+)"/g)].map((m) => m[1])
  } else {
    const base = text.match(/^class\s+\w+\(([\w.]+)/m)?.[1] ?? ''
    tasks = inheritedTasks(base)
  }
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

  /*
   * Trainability, read rather than assumed. A family whose train() raises
   * NotImplementedError is inference-only, and hardcoding trainable:true put a
   * false capability on every museum and specialist family in the registry.
   * Only an unconditional raise counts: a raise guarded by an `if` is a
   * validation error on bad input, not a missing capability.
   */
  const trainDef = text.match(/\n    def train\(([\s\S]*?)(?=\n    def |\n\nclass |$)/)?.[0] ?? ''
  const body = trainDef.replace(/"""[\s\S]*?"""/g, '')
  const trainable = trainDef === ''
    ? null
    : !/^\s{8}raise NotImplementedError/m.test(body)
  return { key, prefix, tasks, sizes, task_sizes: taskSizes, unsupported_train_params: unsupported, trainable }
}

/* ── checkpoints from the HF org ────────────────────────────────── */
/*
 * Every task suffix in the naming scheme (docs/nomenclature.md). Detect is
 * implicit, so a file with no suffix takes its family's default task, which is
 * NOT always detect: a SAM checkpoint is segmentation, a Depth Anything
 * checkpoint is depth. Treating an unsuffixed file as detection discards it as
 * a discrepancy against a family that never declared detection at all.
 */
const TASK_SUFFIX = {
  seg: 'segment', sem: 'semantic', panoptic: 'panoptic', pose: 'pose',
  cls: 'classify', gaze: 'gaze', obb: 'obb', point: 'point', depth: 'depth',
  edge: 'edge', normal: 'normal', restore: 'restore', matte: 'matte',
  ocr: 'ocr', embed: 'embed', mesh: 'mesh', gazetarget: 'gaze',
}

// Dataset tokens that can appear in a checkpoint name. A checkpoint trained on
// something other than the family default says so in its filename, and that
// token is the only mechanical signal we have for it.
const DATASETS = { visdrone: 'VisDrone2019-DET', imagenette: 'Imagenette', sidd: 'SIDD' }

function parseCheckpoints(prefixToDefaultTask) {
  const repos = JSON.parse(read('hf.json'))
  const byPrefix = {}
  const prefixes = Object.keys(prefixToDefaultTask)
  // Longest prefix wins, so LibreYOLO9E2E is not swallowed by LibreYOLO9.
  const sorted = [...prefixes].sort((a, b) => b.length - a.length)
  for (const repo of repos) {
    const id = repo.id.split('/')[1]
    const prefix = sorted.find((p) => id.startsWith(p))
    if (!prefix) continue
    const rest = id.slice(prefix.length)
    const parts = rest.split('-')
    // A single-variant family publishes one repo named exactly for its prefix
    // (LibreMobileSAM, LibreEdgeTAM), so an empty size is a real checkpoint,
    // not a parse failure.
    const size = parts[0] ?? ''
    const suffix = parts.slice(1).find((p) => TASK_SUFFIX[p])
    const datasetToken = parts.slice(1).find((p) => DATASETS[p])
    // The HF repo's own license tag is authoritative: it is the thing the
    // page tells readers to check, so the page must not disagree with it.
    const licenseTag = (repo.tags || []).find((t) => t.startsWith('license:'))
    byPrefix[prefix] ??= []
    byPrefix[prefix].push({
      name: `${id}.pt`,
      size,
      task: TASK_SUFFIX[suffix] || prefixToDefaultTask[prefix] || 'detect',
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
const checkpoints = parseCheckpoints(Object.fromEntries(Object.values(families).filter((f) => f.prefix).map((f) => [f.prefix, f.tasks[0]])))
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
    trainable: lin.keys.some((k) => families[k]?.trainable === true) ? true : lin.keys.every((k) => families[k]?.trainable === false) ? false : null,
    trainable_per_key: Object.fromEntries(lin.keys.map((k) => [k, families[k]?.trainable ?? null])),
  })
}
report.export_formats = formats

/*
 * Library-wide totals, for the docs landing page. Counted, never typed: the
 * landing page's credibility rests on these being true, and a hand-typed "86
 * families" goes stale the first time a family lands.
 */
const tasksText = (() => { try { return read('tasks.py') } catch { return '' } })()
const taskList = [...(tasksText.match(/^TASKS[^=]*=\s*\(([\s\S]*?)\)/m)?.[1] ?? '')
  .matchAll(/"([a-z]+)"/g)].map((m) => m[1])
report.library = {
  families: Object.keys(tiers).length,
  tasks: taskList.length,
  task_names: taskList,
  export_formats: formats.length,
}
fs.writeFileSync(OUT, JSON.stringify(report, null, 2))

for (const l of report.lineages) {
  const benchCount = Object.values(l.benchmarks).flatMap((b) => Object.values(b).flat()).length
  console.log(
    `${l.slug.padEnd(13)} tier=${String(l.tier).padEnd(3)} tasks=${l.tasks.join(',').padEnd(28)} ckpts=${String(l.checkpoints).padStart(2)} bench=${benchCount}`
  )
}
if (report.missing.length) console.log('MISSING:', report.missing.join(', '))
