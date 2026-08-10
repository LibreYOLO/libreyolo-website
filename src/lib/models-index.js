// Single source of truth for /models, derived from the generated docs registry
// rather than hand-kept lists.
//
// The old showcase.js listed 38 model names across 9 tasks while the library
// shipped 82 lineages across 17 tasks, and it fell further behind every
// release because the registry regenerates and a hand-written array does not.
// Everything below reads registry.json, so a new family appears on this page
// the moment scripts/build-registry picks it up.
//
// The two URLs per model are derived here rather than emitted by
// build-registry, so there is one place to fix if either convention changes:
//
//   docs  /docs/models/<slug>            every family has one, checked in CI by
//                                        the docs tree itself
//   HF    <org>?search_models=<prefix>   the org filtered to that family's
//                                        checkpoints, because each checkpoint
//                                        is its own repo (LibreYOLO/LibreYOLO9s)
//                                        and a family has no single repo

import REGISTRY from '@/data/docs/registry.json'

const HF_ORG = REGISTRY.hf_org || 'https://huggingface.co/LibreYOLO'

// Display order: the tasks people come here for first, then the rest roughly by
// how much of the library sits behind them. Any task the registry gains that is
// missing from this list still renders, appended at the end, so a new task
// cannot silently vanish from the page.
const TASK_ORDER = [
  'detect', 'segment', 'pose', 'classify', 'obb', 'depth',
  'semantic', 'panoptic', 'point', 'ocr', 'embed', 'gaze',
  'edge', 'normal', 'matte', 'restore', 'mesh',
]

// Task presentation. `video` wins over `image` when both exist.
//
// Nine tasks have no artwork yet and fall back to a typographic card: semantic,
// panoptic, edge, normal, restore, matte, ocr, embed and mesh. Drop a file into
// public/showcase and add it here to promote one.
const TASK_META = {
  detect: {
    label: 'Detection',
    blurb: 'Boxes around objects. The task most of the library is built for.',
    video: '/showcase/parkour-detection.mp4',
    poster: '/showcase/parkour-detection-poster.jpg',
  },
  segment: {
    label: 'Segmentation',
    blurb: 'Per-instance masks rather than boxes.',
    video: '/showcase/parkour-segmentation.mp4',
    poster: '/showcase/parkour-segmentation-poster.jpg',
  },
  pose: {
    label: 'Keypoints',
    blurb: 'Skeletons and landmarks on each detected instance.',
    video: '/showcase/parkour-pose.mp4',
    poster: '/showcase/parkour-pose-poster.jpg',
  },
  classify: {
    label: 'Classification',
    blurb: 'One label for the whole image, and the backbones behind everything else.',
    image: '/showcase/task-classification.jpg',
  },
  obb: {
    label: 'Oriented boxes',
    blurb: 'Rotated boxes, for aerial imagery and anything that does not sit axis-aligned.',
    image: '/showcase/task-obb.jpg',
  },
  depth: {
    label: 'Depth',
    blurb: 'Distance per pixel from a single image.',
    image: '/showcase/depth-reveal.gif',
  },
  point: {
    label: 'Points and counting',
    blurb: 'Centroids instead of boxes, cheap enough for microcontrollers.',
    image: '/showcase/task-point.jpg',
  },
  gaze: {
    label: 'Gaze',
    blurb: 'Where a person is looking.',
    image: '/showcase/task-gaze.gif',
  },
  semantic: { label: 'Semantic segmentation', blurb: 'A class for every pixel, without separating instances.' },
  panoptic: { label: 'Panoptic segmentation', blurb: 'Semantic and instance masks in one pass.' },
  ocr: { label: 'Text recognition', blurb: 'Finding and reading text in an image.' },
  embed: { label: 'Embeddings', blurb: 'Vectors for search, clustering and retrieval.' },
  edge: { label: 'Edge detection', blurb: 'Contours and boundaries.' },
  normal: { label: 'Surface normals', blurb: 'Which way each surface faces.' },
  matte: { label: 'Matting', blurb: 'Alpha cutouts, hair and edges included.' },
  restore: { label: 'Restoration', blurb: 'Denoising, deblurring and upscaling.' },
  mesh: { label: 'Mesh', blurb: '3D geometry from an image.' },
}

// Tier copy lives in the registry, so the labels here stay in step with docs.
export function tierMeta(tier) {
  return REGISTRY.tiers?.[tier] ?? { label: tier, tone: 'libre', blurb: '' }
}

function hfUrl(family) {
  if (!family.weights_hosted) return null
  const prefix = family.prefix ?? Object.values(family.prefixes ?? {})[0]
  if (!prefix) return null
  return `${HF_ORG}?search_models=${encodeURIComponent(prefix)}`
}

function toModel(family) {
  return {
    key: family.key,
    name: family.display || family.key,
    slug: family.slug,
    tier: family.tier,
    sizesLabel: family.sizes_label ?? '',
    addedIn: family.added_in ?? null,
    trainable: family.trainable !== false,
    docsUrl: `/docs/models/${family.slug}`,
    hfUrl: hfUrl(family),
  }
}

// [{ task, label, blurb, video, poster, image, models: [...] }] in display
// order, every task the registry knows about, every family under each.
export function getModelIndex() {
  const families = Object.values(REGISTRY.families)

  const byTask = new Map()
  for (const family of families) {
    for (const task of family.tasks ?? []) {
      if (!byTask.has(task)) byTask.set(task, [])
      byTask.get(task).push(toModel(family))
    }
  }

  const known = TASK_ORDER.filter((t) => byTask.has(t))
  const extra = [...byTask.keys()].filter((t) => !TASK_ORDER.includes(t)).sort()

  return [...known, ...extra].map((task) => {
    const meta = TASK_META[task] ?? { label: task, blurb: '' }
    // Flagships first, then by name, so the tier badges read as a gradient
    // down each list instead of scattering.
    const models = byTask.get(task).sort((a, b) => {
      const tierRank = (m) => TASK_TIER_RANK[m.tier] ?? 99
      return tierRank(a) - tierRank(b) || a.name.localeCompare(b.name)
    })
    return { task, ...meta, models }
  })
}

const TASK_TIER_RANK = { g0: 0, g1: 1, g2: 2, s: 3, g3: 4, g4: 5 }

// Every family once, for the page's counts and the layout's JSON-LD.
export function getAllModels() {
  return Object.values(REGISTRY.families)
    .map(toModel)
    .sort((a, b) => a.name.localeCompare(b.name))
}

export function getModelStats() {
  const families = Object.values(REGISTRY.families)
  return {
    families: families.length,
    tasks: new Set(families.flatMap((f) => f.tasks ?? [])).size,
    hosted: families.filter((f) => f.weights_hosted).length,
    version: REGISTRY.libreyolo_version,
  }
}

// Tasks each family covers, for structured data and the model cards.
export function getTasksByModel() {
  const out = new Map()
  for (const group of getModelIndex()) {
    for (const model of group.models) {
      out.set(model.name, [...(out.get(model.name) ?? []), group.label])
    }
  }
  return out
}
