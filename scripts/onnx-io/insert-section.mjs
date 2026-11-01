// Insert the "ONNX inputs and outputs" block into model pages.
//
//   node scripts/onnx-io/insert-section.mjs yolov9 [rf-detr ...]
//
// For each slug it edits content/docs/models/<slug>.md and every
// <slug>.<locale>.md twin: a `### <translated heading>` followed by
// `<onnx-io />`, placed at the end of the page's Export section (just before the
// next `## ` heading). The Export section is found in English by its heading,
// and in a twin by the same position among the `## ` headings, since translated
// titles differ. Idempotent: a page that already has `<onnx-io` is left alone.
//
// Every locale keeps the anchor `onnx-inputs-and-outputs`: the site maps a
// translated heading to its English id by position (see headingIdMap in
// src/lib/docs.js), and the insert adds one `### ` to every twin.

import fs from 'node:fs'
import path from 'node:path'

const HEADINGS = {
  en: 'ONNX inputs and outputs',
  de: 'ONNX-Ein- und Ausgaben',
  es: 'Entradas y salidas ONNX',
  fr: 'Entrées et sorties ONNX',
  id: 'Input dan output ONNX',
  it: 'Input e output ONNX',
  ja: 'ONNX の入力と出力',
  ko: 'ONNX 입력과 출력',
  pl: 'Wejścia i wyjścia ONNX',
  pt: 'Entradas e saídas ONNX',
  ru: 'Входы и выходы ONNX',
  uk: 'Входи та виходи ONNX',
  vi: 'Đầu vào và đầu ra ONNX',
  zh: 'ONNX 输入与输出',
}

const dir = path.join(process.cwd(), 'content', 'docs', 'models')

function h2Lines(lines) {
  const out = []
  let fence = false
  lines.forEach((line, i) => {
    if (/^\s*```/.test(line)) fence = !fence
    else if (!fence && /^##\s+\S/.test(line)) out.push(i)
  })
  return out
}

function insert(file, locale, exportIndex) {
  const text = fs.readFileSync(file, 'utf8')
  if (text.includes('<onnx-io')) return 'already present'
  const lines = text.split('\n')
  const h2 = h2Lines(lines)
  if (exportIndex === null) {
    const i = h2.findIndex((l) => /^##\s+Export\s*$/.test(lines[l]))
    if (i < 0) throw new Error(`${file}: no "## Export" heading`)
    exportIndex = i
  }
  const start = h2[exportIndex]
  if (start === undefined) throw new Error(`${file}: no h2 at position ${exportIndex}`)
  const end = h2[exportIndex + 1] ?? lines.length
  // Back up over trailing blank lines so the block sits flush after the prose.
  let at = end
  while (at > start + 1 && lines[at - 1].trim() === '') at--
  const block = ['', `### ${HEADINGS[locale]}`, '', '<onnx-io />']
  lines.splice(at, 0, ...block, ...(end === lines.length - 0 && at === end ? [] : ['']))
  fs.writeFileSync(file, lines.join('\n').replace(/\n{3,}(?=## )/g, '\n\n'))
  return 'inserted'
}

const slugs = process.argv.slice(2)
if (!slugs.length) {
  console.error('usage: node scripts/onnx-io/insert-section.mjs <page-slug> [...]')
  process.exit(1)
}
for (const slug of slugs) {
  const english = path.join(dir, `${slug}.md`)
  if (!fs.existsSync(english)) { console.error(`missing ${english}`); process.exit(1) }
  const enLines = fs.readFileSync(english, 'utf8').split('\n')
  const exportIndex = h2Lines(enLines).findIndex((l) => /^##\s+Export\s*$/.test(enLines[l]))
  console.log(`${slug}.md: ${insert(english, 'en', null)}`)
  for (const file of fs.readdirSync(dir).sort()) {
    const m = new RegExp(`^${slug.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\.([a-z]{2})\\.md$`).exec(file)
    if (!m) continue
    if (!HEADINGS[m[1]]) { console.error(`no heading for locale ${m[1]}`); process.exit(1) }
    console.log(`${file}: ${insert(path.join(dir, file), m[1], exportIndex)}`)
  }
}
