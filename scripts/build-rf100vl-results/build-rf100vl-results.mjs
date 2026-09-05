// Rebuilds the RF100-VL result data the benchmarks page renders, straight from
// the published campaign artifacts on Hugging Face.
//
//   node scripts/build-rf100vl-results/build-rf100vl-results.mjs
//   node scripts/build-rf100vl-results/build-rf100vl-results.mjs --artifacts-dir /path/to/download
//
// Every campaign run uploads a submission JSON per model under
// <model>/<run-id>/submissions/. A run directory can hold submissions for
// models other than the one it is named after, because a resumed campaign
// carries the earlier models' submissions along with it. This accepts only
// submissions whose model id matches their top-level model directory, then
// keeps that model's newest valid complete sweep.
//
// A sweep counts only when the harness marked it valid and all 100 datasets
// scored with none skipped. Partial sweeps are debugging artifacts, and the
// article promises they are never shown.

import { writeFile, readFile, readdir } from 'node:fs/promises'
import { readFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

const REPO = 'LibreYOLO/rf100-vl-results'
const API = `https://huggingface.co/api/datasets/${REPO}/tree/main`
const RAW = `https://huggingface.co/datasets/${REPO}/resolve/main`

const HERE = dirname(fileURLToPath(import.meta.url))
const OUT_DIR = resolve(HERE, '../../src/components/articles/rf100vl')
// An existing download can rebuild the report without fetching the same
// submissions and 1,700 stats records again.
const localArg = process.argv.indexOf('--artifacts-dir')
if (localArg !== -1 && !process.argv[localArg + 1]) {
  throw new Error('--artifacts-dir requires the root of an artifact download')
}
const ARTIFACT_DIR = localArg === -1 ? null : resolve(process.argv[localArg + 1])

// Display names and weight files are not in the submission payload in the form
// the page wants, so they are pinned here. A model absent from this map still
// renders, it just falls back to the raw campaign id.
const DISPLAY = {
  'rfdetr-n': { model: 'RF-DETR', size: 'N', weights: 'LibreRFDETRn.pt' },
  'rfdetr-s': { model: 'RF-DETR', size: 'S', weights: 'LibreRFDETRs.pt' },
  'rfdetr-s-lora': { model: 'RF-DETR', size: 'S LoRA', weights: 'LibreRFDETRs.pt' },
  'rfdetr-m': { model: 'RF-DETR', size: 'M', weights: 'LibreRFDETRm.pt' },
  'rfdetr-l': { model: 'RF-DETR', size: 'L', weights: 'LibreRFDETRl.pt' },
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

// A full rebuild walks a few thousand small files, which the Hub answers with
// 429 well before it answers with data. Anonymous requests hit that wall first,
// so an HF token is used when one is around, and every request backs off and
// retries rather than losing the whole crawl to one throttled file.
const TOKEN = process.env.HF_TOKEN || process.env.HUGGING_FACE_HUB_TOKEN || cachedToken()
const HEADERS = TOKEN ? { authorization: `Bearer ${TOKEN}` } : {}

function cachedToken() {
  for (const file of [
    resolve(homedir(), '.cache/huggingface/token'),
    resolve(homedir(), '.huggingface/token'),
  ]) {
    try {
      const value = readFileSync(file, 'utf8').trim()
      if (value) return value
    } catch {
      // no cached token here, try the next location
    }
  }
  return null
}

const sleep = (ms) => new Promise((done) => setTimeout(done, ms))

async function get(url, label, attempt = 1) {
  let res
  try {
    res = await fetch(url, { headers: HEADERS })
  } catch (err) {
    if (attempt > 5) throw new Error(`${label} failed: ${err.message}`)
    await sleep(2000 * attempt)
    return get(url, label, attempt + 1)
  }
  if (res.status === 429 || res.status >= 500) {
    if (attempt > 5) throw new Error(`${label} failed: ${res.status}`)
    const retryAfter = Number(res.headers.get('retry-after'))
    await sleep(Number.isFinite(retryAfter) && retryAfter > 0 ? retryAfter * 1000 : 2000 * attempt)
    return get(url, label, attempt + 1)
  }
  if (!res.ok) throw new Error(`${label} failed: ${res.status}`)
  return res.json()
}

async function tree(path = '') {
  if (ARTIFACT_DIR) {
    const entries = await readdir(resolve(ARTIFACT_DIR, path), { withFileTypes: true })
    return entries.map((entry) => ({
      type: entry.isDirectory() ? 'directory' : 'file',
      path: path ? `${path}/${entry.name}` : entry.name,
    }))
  }
  return get(path ? `${API}/${path}` : API, `tree ${path || '/'}`)
}

async function json(path) {
  if (ARTIFACT_DIR) return JSON.parse(await readFile(resolve(ARTIFACT_DIR, path), 'utf8'))
  return get(`${RAW}/${path}`, `fetch ${path}`)
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

  const stats = await mapLimit(files, 6, (file) => json(file.path))
  const seconds = stats.map((stat) => stat.wall_seconds)
  if (seconds.some((value) => !Number.isFinite(value))) {
    throw new Error(`${runPath}: a training stats file has no wall_seconds value`)
  }
  if (new Set(stats.map((stat) => stat.dataset)).size !== 100 ||
      stats.some((stat) => !stat.protocol_conformant || stat.epochs_requested !== 100 ||
        stat.seed !== 0 || stat.batch?.effective_batch !== 16)) {
    throw new Error(`${runPath}: training stats disagree with the campaign protocol`)
  }

  return {
    trainingPrecision: stats[0].precision,
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
    // Other models' directories contain stale copies. Only a model's own
    // run may supply the result and its corresponding training statistics.
    if (!path.startsWith(`${id}/`)) continue

    const datasets = rf.datasets ?? []
    const skipped = rf.skipped_datasets ?? []
    if (!rf.valid_submission || datasets.length !== 100 || skipped.length > 0) {
      rejected.push(
        `${id}: incomplete sweep (${datasets.length}/100 scored, ${skipped.length} skipped)`,
      )
      continue
    }
    const mean = datasets.reduce((total, d) => total + d.mAP_50_95, 0) / 100
    if (new Set(datasets.map((d) => d.dataset)).size !== 100 ||
        !Number.isFinite(mean) || Math.abs(mean - sub.accuracy.mAP_50_95) > 1e-10) {
      throw new Error(`${path}: headline AP does not match 100 distinct dataset scores`)
    }

    const prev = best.get(id)
    if (!prev || sub.created_at > prev.sub.created_at) {
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
      submissionPath: path,
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
