/* Turn extracted.json into the registry shape the docs render from. */
import fs from 'fs'

const [, , extractedPath, currentPath, outPath] = process.argv
const ex = JSON.parse(fs.readFileSync(extractedPath, 'utf8'))
const current = JSON.parse(fs.readFileSync(currentPath, 'utf8'))

const TASK_LABELS = {
  detect: { label: 'Detection', slug: 'object-detection' },
  segment: { label: 'Instance segmentation', slug: 'instance-segmentation' },
  pose: { label: 'Pose', slug: 'pose-estimation' },
  obb: { label: 'Oriented boxes', slug: 'oriented-detection' },
}
const EXTRAS = { rfdetr: 'rfdetr', rtdetr: 'rtdetr' } // families needing a pip extra
const DATASET = { detect: 'COCO', segment: 'COCO', pose: 'COCO keypoints', obb: 'DOTA' }

const families = {}
for (const lin of ex.lineages) {
  const primary = lin.keys[0]
  // Merge the per-key export maps: a lineage page shows the union, and where
  // sibling keys disagree the weakest state wins, so nothing is oversold.
  const RANK = { blocked: 0, available: 1, validated: 2 }
  const merged = {}
  for (const task of lin.tasks) {
    for (const key of lin.keys) {
      const row = lin.export[key]?.[task]
      if (!row) continue
      merged[task] ??= {}
      for (const [fmt, state] of Object.entries(row)) {
        const prev = merged[task][fmt]
        merged[task][fmt] = prev === undefined || RANK[state] < RANK[prev] ? state : prev
      }
    }
  }

  const sizeOf = (name) => {
    for (const key of lin.keys) {
      const table = ex.lineages.find((l) => l.slug === lin.slug).sizes[key] || {}
      for (const [code, px] of Object.entries(table)) {
        if (name.toLowerCase().includes(code.toLowerCase())) return px
      }
    }
    return null
  }

  const checkpoints = lin.checkpoint_names.map((name) => {
    const bare = name.replace(/\.pt$/, '')
    const task = /-seg$/.test(bare) ? 'segment'
      : /-pose$/.test(bare) ? 'pose'
      : /-obb$/.test(bare) ? 'obb' : 'detect'
    return { name, task, imgsz: sizeOf(bare), data: DATASET[task] || 'COCO', license: null }
  })

  const benchmarks = {}
  for (const [key, byTask] of Object.entries(lin.benchmarks)) {
    for (const [task, rows] of Object.entries(byTask)) {
      if (!rows.length) continue
      const sorted = [...rows].sort((a, b) => (a.imgsz || 0) - (b.imgsz || 0) || a.size.localeCompare(b.size))
      benchmarks[task] ??= {
        metric: task === 'segment' ? 'mask mAP 50-95' : 'mAP 50-95',
        dataset: `COCO val2017, ${sorted[0].images} images`,
        source_url: 'https://www.visionanalysis.org/',
        rows: [],
      }
      for (const r of sorted) {
        benchmarks[task].rows.push({
          size: r.size, prefix_key: key, imgsz: r.imgsz, params_m: r.params_m, map: r.map,
        })
      }
    }
  }

  families[primary] = {
    key: primary,
    slug: lin.slug,
    registry_keys: lin.keys,
    display: lin.display,
    prefix: null, // filled below
    tier: lin.tier,
    tasks: lin.tasks,
    sizes_label: null,
    extra: EXTRAS[primary] ?? null,
    trainable: true,
    unsupported_train_params: lin.unsupported_train_params,
    capabilities: { train: true, val: true },
    checkpoints,
    export: merged,
    benchmarks,
    upstream: null, // merged from upstream/<slug>.json after human verification
  }
}

// Carry forward the verified rfdetr upstream block and prefix data.
const prev = current.families?.rfdetr
if (prev && families['rfdetr']) {
  families['rfdetr'].upstream = prev.upstream
  families['rfdetr'].prefix = prev.prefix
  families['rfdetr'].sizes_label = prev.sizes_label
  families['rfdetr'].added_in = prev.added_in
  families['rfdetr'].task_added_in = prev.task_added_in
  families['rfdetr'].export_reasons = prev.export_reasons
  families['rfdetr'].va_embed = prev.va_embed
  families['rfdetr'].capabilities = prev.capabilities
  // Keep the hand-verified checkpoint rows: they carry licenses and input sizes
  // the extractor cannot know yet.
  families['rfdetr'].checkpoints = prev.checkpoints
}

const out = {
  ...current,
  _comment: current._comment,
  _generated_by: 'scripts/build-registry (prototype). Mechanical fields are extracted from docs/export_support.md, each family model.py, models/registry.py, the LibreYOLO HF org listing and the vision-analysis results. Upstream metadata is merged from src/data/docs/upstream/<slug>.json after per-family human verification.',
  tasks: TASK_LABELS,
  families,
  discrepancies: ex.discrepancies,
}
fs.writeFileSync(outPath, JSON.stringify(out, null, 2))
console.log('families:', Object.keys(families).join(', '))
console.log('discrepancies:', ex.discrepancies.length)
for (const [slug, f] of Object.entries(families)) {
  console.log(`  ${slug.padEnd(13)} tier=${f.tier} tasks=${f.tasks.length} ckpts=${f.checkpoints.length} bench=${Object.keys(f.benchmarks).join('+') || 'none'} upstream=${f.upstream ? 'verified' : 'PENDING'}`)
}
