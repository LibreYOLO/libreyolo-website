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
// Every still under /showcase/task-* is a real LibreYOLO run, rendered by
// scripts/build-task-art over the library's own demo photographs. Re-run that
// script to regenerate them. All 17 tasks have artwork.
//
// Two needed weights LibreYOLO cannot host, so their recipes expect a local
// checkpoint: edge (TEED, BIPED-trained, non-commercial) and mesh (SAM 3D
// Body, which also needs Meta's own package). Running them is use, which both
// licences permit; redistributing the weights is what LibreYOLO does not do.
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
  semantic: {
    label: 'Semantic segmentation',
    blurb: 'A class for every pixel, without separating instances.',
    image: '/showcase/task-semantic.jpg',
  },
  panoptic: {
    label: 'Panoptic segmentation',
    blurb: 'Semantic and instance masks in one pass.',
    image: '/showcase/task-panoptic.jpg',
  },
  normal: {
    label: 'Surface normals',
    blurb: 'Which way each surface faces.',
    image: '/showcase/task-normal.jpg',
  },
  matte: {
    label: 'Matting',
    blurb: 'Alpha cutouts, hair and edges included.',
    image: '/showcase/task-matte.jpg',
  },
  restore: {
    label: 'Restoration',
    blurb: 'Denoising, deblurring and upscaling.',
    image: '/showcase/task-restore.jpg',
  },
  ocr: {
    label: 'Text recognition',
    blurb: 'Finding and reading text in an image.',
    image: '/showcase/task-ocr.jpg',
  },
  embed: {
    label: 'Embeddings',
    blurb: 'Vectors for search, clustering and retrieval.',
    image: '/showcase/task-embed.jpg',
  },
  edge: {
    label: 'Edge detection',
    blurb: 'Contours and boundaries.',
    image: '/showcase/task-edge.jpg',
  },
  mesh: {
    label: 'Mesh',
    blurb: '3D geometry from an image.',
    image: '/showcase/task-mesh.jpg',
  },
}

// Tier copy lives in the registry, so the labels here stay in step with docs.
export function tierMeta(tier) {
  return REGISTRY.tiers?.[tier] ?? { label: tier, tone: 'libre', blurb: '' }
}

// Where the weights for the 19 families LibreYOLO does not host actually come
// from, and why. "Not on our Hugging Face org" is never the same as "you
// cannot get them": the library downloads every one of these on first use.
//
// URLs and licence facts are taken from the verified upstream records in
// src/data/docs/upstream/<slug>.json and from each family's model.py, not
// invented here. A non-hosted family missing from this map gets no weights
// link at all rather than a guessed one.
//
// kind drives nothing but the wording of the note; the link is the point.
const WEIGHTS_SOURCE = {
  // Mirrored on the LibreYOLO org. These were upstream links until the weights
  // were checked for redistribution rights and copied across; the note says so
  // because the licence still travels with the file.
  //   sam, moge2, facerec

  // Downloaded from the upstream project on first use, permissive licence.
  dinov2: {
    url: 'https://github.com/facebookresearch/dinov2',
    note: 'Apache-2.0 weights, downloaded from the upstream project on first use.',
  },
  midas: {
    url: 'https://huggingface.co/LibreYOLO?search_models=LibreMiDaS',
    note: 'MIT. Mirrored on the LibreYOLO org. MiDaS trained on a twelve-dataset mixture whose terms are not all permissive, so check those before commercial use.',
  },
  sam: {
    url: 'https://huggingface.co/LibreYOLO?search_models=LibreSAM',
    note: 'Apache-2.0. Mirrored on the LibreYOLO org, so nothing is fetched from Meta at load time.',
  },
  moge2: {
    url: 'https://huggingface.co/LibreYOLO?search_models=LibreMoGe2',
    note: 'Mirrored on the LibreYOLO org for the s and l sizes; b still downloads from the authors on Hugging Face and is checksum-pinned.',
  },
  facerec: {
    url: 'https://huggingface.co/LibreYOLO?search_models=librefacerec',
    note: 'Apache-2.0, derived from AuraFace. Already served from the LibreYOLO org as librefacerec-l and librefacerec-det.',
  },

  // Restricted checkpoints. LibreYOLO refuses to mirror them and fetches from
  // the origin instead, printing the terms before the download starts.
  yolonas: {
    url: 'https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.YOLONAS.md',
    note: "Deci's checkpoints are non-commercial and cannot be redistributed, so LibreYOLO downloads them from Deci's CDN and prints the licence first. The architecture and training code are Apache-2.0, so weights you train yourself carry no such terms.",
  },
  l2cs: {
    url: 'https://github.com/Ahmednull/L2CS-Net',
    note: 'Trained on Gaze360, whose terms are research and non-commercial only, so the checkpoint is not mirrored. LibreYOLO downloads it from the authors.',
  },
  dexined: {
    url: 'https://github.com/xavysp/DexiNed',
    note: 'The code is MIT, but the released checkpoint is BIPED-trained and BIPED is non-commercial, so LibreYOLO does not mirror it. Convert a checkpoint you are licensed to use.',
  },
  teed: {
    url: 'https://github.com/xavysp/TEED',
    note: 'The code is MIT, but the released checkpoint is BIPED-trained and BIPED is non-commercial, so LibreYOLO does not mirror it. Convert a checkpoint you are licensed to use.',
  },
  domedetr: {
    url: 'https://github.com/RicePasteM/Dome-DETR',
    note: 'The upstream licence is unclear, so the checkpoint is not redistributed. Weights come from the authors’ repository.',
  },
  libremodus: {
    url: 'https://github.com/EPFL-VILAB/Modus',
    note: 'The code is Apache-2.0 but the upstream model card is research-only, so the weights are not mirrored.',
  },
  locateanything: {
    url: 'https://github.com/NVlabs/Eagle/tree/main/Embodied',
    note: 'Released under an NVIDIA non-commercial licence, so the weights are not mirrored.',
  },
  sam3: {
    url: 'https://github.com/facebookresearch/sam3',
    note: 'Meta gates these weights behind its own licence. Accept it upstream and the download works.',
  },

  // Wrapped rather than ported: the weights live in the upstream model repo and
  // load through that project's own loader.
  florence2: {
    url: 'https://huggingface.co/microsoft/Florence-2-large',
    note: 'LibreYOLO wraps this model rather than porting it, so the weights load from its own Hugging Face repo.',
  },
  internvl3: {
    url: 'https://github.com/OpenGVLab/InternVL',
    note: 'LibreYOLO wraps this model rather than porting it, so the weights load from its own repo. The code is MIT; the weights carry the Qwen licence.',
  },
  kosmos2: {
    url: 'https://github.com/microsoft/unilm/tree/master/kosmos-2',
    note: 'LibreYOLO wraps this model rather than porting it, so the weights load from its own repo.',
  },
  lfm2vl: {
    url: 'https://huggingface.co/LiquidAI/LFM2.5-VL-450M',
    note: 'LibreYOLO wraps this model rather than porting it, so the weights load from its own Hugging Face repo under the LFM Open License.',
  },
  qwen3vl: {
    url: 'https://github.com/QwenLM/Qwen3-VL',
    note: 'LibreYOLO wraps this model rather than porting it, so the weights load from its own repo.',
  },
  smolvlm2: {
    url: 'https://github.com/huggingface/smollm/tree/main/vision',
    note: 'LibreYOLO wraps this model rather than porting it, so the weights load from its own repo.',
  },
}

// Every family that has weights at all gets a link. Hosted families point at
// the org filtered to their prefix, because each checkpoint is its own repo and
// a family has no single one. The rest point upstream and carry a note.
function weightsLink(family) {
  if (family.weights_hosted) {
    const prefix = family.prefix ?? Object.values(family.prefixes ?? {})[0]
    if (!prefix) return { url: null, note: null }
    return { url: `${HF_ORG}?search_models=${encodeURIComponent(prefix)}`, note: null }
  }

  const keys = [family.key, ...(family.registry_keys ?? [])]
  for (const key of keys) {
    if (WEIGHTS_SOURCE[key]) return WEIGHTS_SOURCE[key]
  }
  return { url: null, note: null }
}

function toModel(family) {
  const weights = weightsLink(family)
  return {
    key: family.key,
    name: family.display || family.key,
    slug: family.slug,
    tier: family.tier,
    sizesLabel: family.sizes_label ?? '',
    addedIn: family.added_in ?? null,
    trainable: family.trainable !== false,
    docsUrl: `/docs/models/${family.slug}`,
    hfUrl: weights.url,
    weightsNote: weights.note,
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
