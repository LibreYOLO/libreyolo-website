// Rebuilds the RF100-VL result data the benchmarks page renders, straight from
// the published campaign artifacts on Hugging Face.
//
//   node scripts/build-rf100vl-results/build-rf100vl-results.mjs
//
// Every campaign run uploads a submission JSON per model under
// <model>/<run-id>/submissions/. A run directory can hold submissions for
// models other than the one it is named after, because a resumed campaign
// carries the earlier models' submissions along with it. So rather than trust
// the directory name, this walks every submission it can find, groups them by
// the model id inside the file, and keeps the newest valid sweep for each.
//
// A sweep counts only when the harness marked it valid and all 100 datasets
// scored with none skipped. Partial sweeps are debugging artifacts, and the
// article promises they are never shown.

import { writeFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

const REPO = 'LibreYOLO/rf100-vl-results'
const API = `https://huggingface.co/api/datasets/${REPO}/tree/main`
const RAW = `https://huggingface.co/datasets/${REPO}/resolve/main`

const HERE = dirname(fileURLToPath(import.meta.url))
const OUT_DIR = resolve(HERE, '../../src/components/articles/rf100vl')

// Display names and weight files are not in the submission payload in the form
// the page wants, so they are pinned here. A model absent from this map still
// renders, it just falls back to the raw campaign id.
const DISPLAY = {
  'rfdetr-s': { model: 'RF-DETR', size: 'S', weights: 'LibreRFDETRs.pt' },
  'yolov9t': { model: 'YOLOv9', size: 'T', weights: 'LibreYOLO9t.pt' },
  'yolov9s': { model: 'YOLOv9', size: 'S', weights: 'LibreYOLO9s.pt' },
  'yolov9m': { model: 'YOLOv9', size: 'M', weights: 'LibreYOLO9m.pt' },
  'yolox-nano': { model: 'YOLOX', size: 'Nano', weights: 'LibreYOLOXn.pt' },
  'yolox-tiny': { model: 'YOLOX', size: 'Tiny', weights: 'LibreYOLOXt.pt' },
  'yolox-s': { model: 'YOLOX', size: 'S', weights: 'LibreYOLOXs.pt' },
  'yolox-m': { model: 'YOLOX', size: 'M', weights: 'LibreYOLOXm.pt' },
  'yolonas-s': { model: 'YOLO-NAS', size: 'S', weights: 'LibreYOLONASs.pt' },
  'yolonas-m': { model: 'YOLO-NAS', size: 'M', weights: 'LibreYOLONASm.pt' },
  'ec-s': { model: 'EdgeCrafter', size: 'S', weights: 'LibreECs.pt' },
  'ec-m': { model: 'EdgeCrafter', size: 'M', weights: 'LibreECm.pt' },
  'ec-l': { model: 'EdgeCrafter', size: 'L', weights: 'LibreECl.pt' },
}

async function tree(path = '') {
  const res = await fetch(path ? `${API}/${path}` : API)
  if (!res.ok) throw new Error(`tree ${path || '/'} failed: ${res.status}`)
  return res.json()
}

async function json(path) {
  const res = await fetch(`${RAW}/${path}`)
  if (!res.ok) throw new Error(`fetch ${path} failed: ${res.status}`)
  return res.json()
}

async function mapLimit(items, limit, fn) {
  const results = new Array(items.length)
  let next = 0

  async function worker() {
    while (next < items.length) {
      const index = next
      next += 1
      results[index] = await fn(items[index])
    }
  }

  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker))
  return results
}

function median(values) {
  const ordered = [...values].sort((a, b) => a - b)
  const middle = Math.floor(ordered.length / 2)
  return ordered.length % 2
    ? ordered[middle]
    : (ordered[middle - 1] + ordered[middle]) / 2
}

async function trainingTimes(runPath) {
  const files = (await tree(`${runPath}/stats`)).filter(
    (entry) => entry.type === 'file' && entry.path.endsWith('.json'),
  )
  if (files.length !== 100) {
    throw new Error(`${runPath}: expected 100 training stats files, found ${files.length}`)
  }

  const stats = await mapLimit(files, 12, (file) => json(file.path))
  const seconds = stats.map((stat) => stat.wall_seconds)
  if (seconds.some((value) => !Number.isFinite(value))) {
    throw new Error(`${runPath}: a training stats file has no wall_seconds value`)
  }

  return {
    trainMin: median(seconds) / 60,
    trainHours: seconds.reduce((total, value) => total + value, 0) / 3600,
  }
}

// Every submission file in the repo, across every model and run directory.
async function findSubmissions() {
  const found = []
  const modelDirs = (await tree()).filter((e) => e.type === 'directory')

  for (const modelDir of modelDirs) {
    const runs = (await tree(modelDir.path)).filter((e) => e.type === 'directory')
    for (const run of runs) {
      // recipe/ and other bookkeeping directories have no submissions.
      let entries
      try {
        entries = await tree(`${run.path}/submissions`)
      } catch {
        continue
      }
      for (const file of entries) {
        if (file.type === 'file' && file.path.endsWith('.json')) found.push(file.path)
      }
    }
  }
  return found
}

async function main() {
  const paths = await findSubmissions()
  console.log(`found ${paths.length} submission files`)

  // Newest valid complete sweep wins for each model id.
  const best = new Map()
  const rejected = []

  for (const path of paths) {
    let sub
    try {
      sub = await json(path)
    } catch (err) {
      rejected.push(`${path}: ${err.message}`)
      continue
    }

    const id = sub?.model?.id
    const rf = sub?.rf100vl
    if (!id || !rf) {
      rejected.push(`${path}: not an RF100-VL submission`)
      continue
    }

    const datasets = rf.datasets ?? []
    const skipped = rf.skipped_datasets ?? []
    if (!rf.valid_submission || datasets.length !== 100 || skipped.length > 0) {
      rejected.push(
        `${id}: incomplete sweep (${datasets.length}/100 scored, ${skipped.length} skipped)`,
      )
      continue
    }

    const prev = best.get(id)
    const isCanonicalPath = path.startsWith(`${id}/`)
    const prevIsCanonicalPath = prev?.path.startsWith(`${id}/`)
    if (
      !prev ||
      sub.created_at > prev.sub.created_at ||
      (sub.created_at === prev.sub.created_at && isCanonicalPath && !prevIsCanonicalPath)
    ) {
      best.set(id, { sub, path })
    }
  }

  for (const line of rejected) console.log(`  skipped ${line}`)

  // Best accuracy first, which is the order both the table and the bar chart read in.
  const ordered = [...best.values()].sort(
    (a, b) => b.sub.accuracy.mAP_50_95 - a.sub.accuracy.mAP_50_95,
  )

  const byDataset = {}
  const summary = []

  for (const { sub, path } of ordered) {
    const id = sub.model.id
    const display = DISPLAY[id] ?? {
      model: sub.model.name ?? id,
      size: '',
      weights: sub.model.weights ?? '',
    }
    const paramsM = sub.model_stats?.params_millions ?? null

    const scores = {}
    for (const d of sub.rf100vl.datasets) {
      scores[d.dataset] = {
        m: Number(d.mAP_50_95.toFixed(4)),
        m50: Number(d.mAP_50.toFixed(4)),
        img: d.num_images,
        cls: d.num_classes,
      }
    }

    byDataset[id] = {
      model: id,
      label: display.size ? `${display.model}-${display.size}` : display.model,
      paramsM,
      mean: sub.accuracy.mAP_50_95,
      mean50: sub.accuracy.mAP_50,
      scores,
    }

    // Link to the real published run directory, not the submission id. The
    // two differ, and resumed campaigns can carry earlier submissions inside
    // a later model's directory.
    const runPath = path.split('/submissions/')[0]
    const times = await trainingTimes(runPath)

    summary.push({
      id,
      ...display,
      paramsM,
      map: sub.accuracy.mAP_50_95,
      map50: sub.accuracy.mAP_50,
      createdAt: sub.created_at,
      runPath,
      inputSize: sub.model.input_size,
      trainingPrecision: sub.rf100vl.training_precision,
      gpu: sub.hardware?.gpu ?? null,
      ...times,
    })
  }

  await writeFile(
    `${OUT_DIR}/results-by-dataset.json`,
    `${JSON.stringify(byDataset, null, 2)}\n`,
  )
  await writeFile(
    `${OUT_DIR}/results-summary.json`,
    `${JSON.stringify(summary, null, 2)}\n`,
  )

  console.log(`\nwrote ${ordered.length} complete sweeps:`)
  for (const s of summary) {
    console.log(
      `  ${`${s.model}-${s.size}`.padEnd(16)} mAP ${s.map.toFixed(4)}  ` +
        `mAP50 ${s.map50.toFixed(4)}  ${String(s.paramsM).padStart(6)}M  ${s.createdAt}`,
    )
  }

  // Keep a copy next to the importer as a compact audit artifact.
  await writeFile(`${HERE}/last-run-summary.json`, `${JSON.stringify(summary, null, 2)}\n`)
  console.log(`\nsummary written to scripts/build-rf100vl-results/last-run-summary.json`)
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
