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
/*
 * Training dataset per task, ONLY where the library's own naming convention
 * makes it unambiguous. There is deliberately no default: stamping "COCO" on
 * everything put that claim on ImageNet classifiers, Cityscapes segmenters, an
 * OCR pipeline and a matting model, each of which contradicted its own page.
 * An unknown dataset renders as an empty cell, which is the house convention
 * for "not recorded".
 */
const DATASET = { detect: 'COCO', segment: 'COCO', pose: 'COCO keypoints', obb: 'DOTA' }

// "n, s, m, l at 640 px" style summary, derived from INPUT_SIZES.
function sizesLabel(lin) {
  const parts = []
  for (const key of lin.keys) {
    const table = lin.sizes[key] || {}
    const codes = Object.keys(table)
    if (!codes.length) continue
    const px = [...new Set(Object.values(table))]
    const res = px.length === 1 ? `at ${px[0]} px` : `at ${Math.min(...px)} to ${Math.max(...px)} px`
    parts.push(lin.keys.length > 1 ? `${key}: ${codes.join(', ')} ${res}` : `${codes.join(', ')} ${res}`)
  }
  return parts.join('; ') || null
}

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

  /*
   * Native input resolution for a checkpoint. TASK_INPUT_SIZES wins over
   * INPUT_SIZES where a family sets it, because several families run
   * segmentation and pose at a different resolution from detection, and
   * reading only INPUT_SIZES would report the detection number for all of them.
   * The size code is matched against the checkpoint's own suffix, longest
   * first, so "s" does not match inside "seg".
   */
  const sizeOf = (name, task) => {
    const bare = name.replace(/-(seg|pose|obb|cls|sem|point)$/, '')
    const entry = ex.lineages.find((l) => l.slug === lin.slug)
    for (const key of lin.keys) {
      const perTask = entry.task_sizes?.[key]
      // If a family declares per-task sizes at all, only its table for THIS
      // task may be used. Falling back to INPUT_SIZES would silently report
      // the detection resolution for a segmentation checkpoint.
      const table = perTask && Object.keys(perTask).length
        ? perTask[task]
        : entry.sizes[key]
      if (!table) continue
      const codes = Object.keys(table).sort((a, b) => b.length - a.length)
      for (const code of codes) {
        if (bare.toLowerCase().endsWith(code.toLowerCase())) return table[code]
      }
    }
    return null
  }

  const checkpoints = (lin.checkpoint_rows || []).map((row) => {
    const bare = row.name.replace(/\.pt$/, '')
    // A checkpoint trained on something other than the family default names
    // that dataset in its filename. When it does, the family's default input
    // size does not apply either, so leave the cell empty rather than assert
    // a resolution we have not read.
    const custom = Boolean(row.dataset)
    return {
      name: row.name,
      task: row.task,
      imgsz: custom ? null : sizeOf(bare, row.task),
      data: row.dataset || DATASET[row.task] || null,
      license: row.license,
    }
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
          id: r.id, size: r.size, prefix_key: key, imgsz: r.imgsz, params_m: r.params_m, map: r.map,
        })
      }
    }
  }

  /*
   * Vision Analysis embed. Every benchmarked model already carries its VA id,
   * so the chart is derivable: highlight this family's models against the whole
   * field. Hardware and runtime are deliberately not pinned, because pinning a
   * combination a family has no run for yields an empty chart.
   */
  const vaIds = Object.values(benchmarks).flatMap((b) => b.rows.map((r) => r.id)).filter(Boolean)
  const vaEmbed = vaIds.length
    ? {
        scatter:
          'https://www.visionanalysis.org/embed/scatter?highlight=' +
          encodeURIComponent([...new Set(vaIds)].join(',')) +
          '&title=' + encodeURIComponent(`${lin.display} on COCO`) +
          '&subtitle=' + encodeURIComponent('Accuracy against latency, every benchmarked model'),
      }
    : null

  families[primary] = {
    key: primary,
    slug: lin.slug,
    registry_keys: lin.keys,
    display: lin.display,
    prefix: lin.prefixes?.[primary] ?? null,
    // Per-key prefixes. A lineage page mixes benchmark rows from sibling keys
    // with different filename prefixes, so a row must be labelled with its own
    // key's prefix or the table credits one version's accuracy to another
    // version's filename, naming files that do not exist.
    prefixes: lin.prefixes ?? {},
    tier: lin.tier,
    tasks: lin.tasks,
    sizes_label: sizesLabel(lin),
    added_in: lin.added_in,
    task_added_in: lin.task_added_in,
    // Whether the LibreYOLO org hosts this family's weights at all. When it
    // does not, the licensing block must not claim we republish them.
    weights_hosted: (lin.checkpoint_rows || []).length > 0,
    extra: EXTRAS[primary] ?? null,
    trainable: lin.trainable,
    unsupported_train_params: lin.unsupported_train_params,
    trainable_per_key: lin.trainable_per_key,
    capabilities: { train: lin.trainable !== false, val: true },
    checkpoints,
    export: merged,
    benchmarks,
    va_embed: vaEmbed,
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

  families['rfdetr'].capabilities = prev.capabilities
  // Checkpoints are no longer carried over: the extractor now reads each
  // repository's own license tag and honors TASK_INPUT_SIZES, so it is a better
  // source than the hand-written rows it replaces.
}

const out = {
  ...current,
  _comment: current._comment,
  _generated_by: 'scripts/build-registry (prototype). Mechanical fields are extracted from docs/export_support.md, each family model.py, models/registry.py, the LibreYOLO HF org listing and the vision-analysis results. Upstream metadata is merged from src/data/docs/upstream/<slug>.json after per-family human verification.',
  tasks: TASK_LABELS,
  library: ex.library ?? null,
  families,
  discrepancies: ex.discrepancies,
}
fs.writeFileSync(outPath, JSON.stringify(out, null, 2))
console.log('families:', Object.keys(families).join(', '))
console.log('discrepancies:', ex.discrepancies.length)
for (const [slug, f] of Object.entries(families)) {
  console.log(`  ${slug.padEnd(13)} tier=${f.tier} tasks=${f.tasks.length} ckpts=${f.checkpoints.length} bench=${Object.keys(f.benchmarks).join('+') || 'none'} upstream=${f.upstream ? 'verified' : 'PENDING'}`)
}
