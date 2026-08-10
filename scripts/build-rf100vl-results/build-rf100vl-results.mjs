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
    if (!prev || sub.created_at > prev.created_at) best.set(id, sub)
  }

  for (const line of rejected) console.log(`  skipped ${line}`)

  // Best accuracy first, which is the order both the table and the bar chart read in.
  const ordered = [...best.values()].sort(
    (a, b) => b.accuracy.mAP_50_95 - a.accuracy.mAP_50_95,
  )

  const byDataset = {}
  const summary = []

  for (const sub of ordered) {
    const id = sub.model.id
    const display = DISPLAY[id] ?? { model: id, size: '', weights: '' }
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

    // runPath is what the article links to so a reader can open the artifacts.
    const runPath = sub.rf100vl.per_dataset_results_dir
      ? `${id}/${sub.submission_id}`
      : `${id}/${sub.submission_id}`

    summary.push({
      id,
      ...display,
      paramsM,
      map: sub.accuracy.mAP_50_95,
      map50: sub.accuracy.mAP_50,
      createdAt: sub.created_at,
      runPath,
      inputSize: sub.model.input_size,
    })
  }

  await writeFile(
    `${OUT_DIR}/results-by-dataset.json`,
    `${JSON.stringify(byDataset, null, 2)}\n`,
  )

  console.log(`\nwrote ${ordered.length} complete sweeps:`)
  for (const s of summary) {
    console.log(
      `  ${`${s.model}-${s.size}`.padEnd(16)} mAP ${s.map.toFixed(4)}  ` +
        `mAP50 ${s.map50.toFixed(4)}  ${String(s.paramsM).padStart(6)}M  ${s.createdAt}`,
    )
  }

  // The article's summary table lives in RF100VLResults.jsx as hand-kept
  // source, because it carries median training times this script cannot see.
  // The summary lands next to the script as a reference for updating it by
  // hand, rather than in the component directory where it would look like
  // something the site imports.
  await writeFile(`${HERE}/last-run-summary.json`, `${JSON.stringify(summary, null, 2)}\n`)
  console.log(`\nsummary written to scripts/build-rf100vl-results/last-run-summary.json`)
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
