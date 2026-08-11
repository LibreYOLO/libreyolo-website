// Structural check of a translation living at an ARBITRARY path against its
// English source. validate.mjs only handles twins sitting next to their source
// inside content/docs; this variant takes explicit paths, so two candidate
// translations of the same page (e.g. from different tools) can be compared.
//
//   node scripts/translation/compare-pair.mjs <english.md> <translation.md>
//
// Prints one PASS/FAIL line plus any structural problems, and a few quality
// signals that are cheap to measure mechanically.

import fs from 'node:fs'
import matter from 'gray-matter'

const [enPath, trPath] = process.argv.slice(2)
if (!enPath || !trPath) {
  console.error('Usage: node scripts/translation/compare-pair.mjs <english.md> <translation.md>')
  process.exit(2)
}

const read = (f) => matter(fs.readFileSync(f, 'utf8').replace(/\r\n/g, '\n'))

function headingSkeleton(md) {
  const out = []
  let inFence = false
  for (const line of md.split('\n')) {
    if (/^\s*(```|~~~)/.test(line)) inFence = !inFence
    else if (!inFence) {
      const m = line.match(/^(#{1,6})\s/)
      if (m) out.push(m[1].length)
    }
  }
  return out
}

function codeBlocks(md) {
  const blocks = []
  const re = /^```.*\n([\s\S]*?)^```/gm
  let m
  while ((m = re.exec(md))) blocks.push(m[1])
  return blocks
}

const stripComments = (code) =>
  code
    .split('\n')
    .map((l) => l.replace(/\s+(#|\/\/)\s.*$/, ''))
    .filter((l) => !/^\s*(#|\/\/)/.test(l))
    .join('\n')
    .trim()

function linkTargets(md) {
  const t = []
  const re = /\]\(([^)\s]+)/g
  let m
  while ((m = re.exec(md))) t.push(m[1])
  return t.sort()
}

const snippetCode = (d) =>
  Object.values(d.snippets ?? {}).flatMap((g) => g.map((s) => stripComments(String(s.code ?? ''))))

const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b)
const problems = []

let en, tr
try {
  en = read(enPath)
} catch (e) {
  console.log(`FAIL ${trPath}: English source unreadable (${e.message})`)
  process.exit(1)
}
try {
  tr = read(trPath)
} catch (e) {
  console.log(`FAIL ${trPath}: frontmatter does not parse (${e.message})`)
  process.exit(1)
}

const enKeys = Object.keys(en.data).sort()
const trKeys = Object.keys(tr.data).filter((k) => k !== 'source_hash').sort()
if (!eq(enKeys, trKeys)) problems.push(`frontmatter keys differ: missing=[${enKeys.filter((k) => !trKeys.includes(k))}] extra=[${trKeys.filter((k) => !enKeys.includes(k))}]`)

for (const f of ['families', 'last_verified', 'layout']) {
  if (f in en.data && !eq(en.data[f], tr.data[f])) problems.push(`frontmatter '${f}' changed`)
}

if (!eq(snippetCode(en.data), snippetCode(tr.data))) problems.push('snippet code differs beyond comments')

const enH = headingSkeleton(en.content)
const trH = headingSkeleton(tr.content)
if (!eq(enH, trH)) problems.push(`heading skeleton differs (en ${enH.length}, tr ${trH.length})`)

const enC = codeBlocks(en.content).map(stripComments)
const trC = codeBlocks(tr.content).map(stripComments)
if (enC.length !== trC.length) problems.push(`code block count differs (en ${enC.length}, tr ${trC.length})`)
else if (!eq(enC, trC)) problems.push('body code differs beyond comments')

if (!eq(linkTargets(en.content), linkTargets(tr.content))) problems.push('link targets differ')

// Quality signals (reported, never fatal).
const emDashes = (tr.content.match(/—/g) || []).length + (JSON.stringify(tr.data).match(/—/g) || []).length
const kwEn = JSON.stringify(en.data.keywords ?? [])
const kwTr = JSON.stringify(tr.data.keywords ?? [])
const signals = [
  `emdash=${emDashes}`,
  `keywords_identical_to_english=${kwEn === kwTr}`,
  `body_chars=${tr.content.length}`,
]

console.log(`${problems.length ? 'FAIL' : 'PASS'} ${trPath}  [${signals.join(' ')}]`)
for (const p of problems) console.log(`  - ${p}`)
process.exit(problems.length ? 1 : 0)
