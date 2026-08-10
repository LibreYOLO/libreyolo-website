// Structural validator for docs translations.
//
// For every content/docs/**/<slug>.<locale>.md, compares against its English
// twin and reports structural drift: frontmatter keys, heading skeleton,
// code blocks (identical modulo comment lines), and link targets.
//
// Usage:  node scripts/translation/validate.mjs es [section/slug ...]
//         (no slugs = validate every twin found for that locale)

import fs from 'node:fs'
import path from 'node:path'
import matter from 'gray-matter'

const locale = process.argv[2]
if (!locale || locale === 'en') {
  console.error('Usage: node scripts/translation/validate.mjs <locale> [section/slug ...]')
  process.exit(2)
}

const docsDir = path.join(process.cwd(), 'content', 'docs')
const only = process.argv.slice(3)

function* walk(dir) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name)
    if (entry.isDirectory()) yield* walk(full)
    else yield full
  }
}

const suffix = `.${locale}.md`
const twins = [...walk(docsDir)]
  .filter((f) => f.endsWith(suffix))
  .filter((f) => {
    if (!only.length) return true
    const rel = path.relative(docsDir, f).replaceAll(path.sep, '/')
    return only.includes(rel.slice(0, -suffix.length))
  })

if (!twins.length) {
  console.error(`No ${suffix} files found${only.length ? ' for the given slugs' : ''}.`)
  process.exit(2)
}

// Headings outside code fences, as "level:text-is-ignored" skeleton.
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

// Strip full-line comments (#, //) so translated comments don't count as drift.
function stripComments(code) {
  return code
    .split('\n')
    .map((l) => l.replace(/\s+(#|\/\/)\s.*$/, '')) // trailing comments
    .filter((l) => !/^\s*(#|\/\/)/.test(l))
    .join('\n')
    .trim()
}

function linkTargets(md) {
  const targets = []
  const re = /\]\(([^)\s]+)/g
  let m
  while ((m = re.exec(md))) targets.push(m[1])
  return targets.sort()
}

// Snippet code inside frontmatter, comments stripped, in document order.
function snippetCode(data) {
  const out = []
  for (const group of Object.values(data.snippets ?? {})) {
    for (const s of group) out.push(stripComments(String(s.code ?? '')))
  }
  return out
}

// Labels are reader-visible and may be translated; compare keys/counts/languages.
function snippetShape(data) {
  return Object.entries(data.snippets ?? {}).map(
    ([k, group]) => `${k}:${group.map((s) => s.language).join(',')}`
  )
}

let failures = 0
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b)

for (const twin of twins) {
  const enPath = twin.slice(0, -suffix.length) + '.md'
  const rel = path.relative(docsDir, twin).replaceAll(path.sep, '/')
  const problems = []

  if (!fs.existsSync(enPath)) {
    console.log(`FAIL ${rel}: English twin missing (${path.basename(enPath)})`)
    failures++
    continue
  }

  let en, tr
  try {
    // Normalize CRLF: the English sources are CRLF on this machine and JS's
    // `.` does not match `\r`, which breaks fence detection and code equality.
    en = matter(fs.readFileSync(enPath, 'utf8').replace(/\r\n/g, '\n'))
    tr = matter(fs.readFileSync(twin, 'utf8').replace(/\r\n/g, '\n'))
  } catch (e) {
    console.log(`FAIL ${rel}: frontmatter does not parse (${e.message})`)
    failures++
    continue
  }

  const enKeys = Object.keys(en.data).sort()
  const trKeys = Object.keys(tr.data).sort()
  if (!eq(enKeys, trKeys)) problems.push(`frontmatter keys differ: en=[${enKeys}] ${locale}=[${trKeys}]`)

  for (const field of ['families', 'last_verified', 'layout']) {
    if (field in en.data && !eq(en.data[field], tr.data[field]))
      problems.push(`frontmatter '${field}' must be identical to English`)
  }
  // hero.caption is reader-visible and translatable; the media paths are not.
  if (en.data.hero) {
    for (const k of ['src', 'poster']) {
      if (!eq(en.data.hero?.[k], tr.data.hero?.[k]))
        problems.push(`hero.${k} must be identical to English`)
    }
  }

  if (!eq(snippetShape(en.data), snippetShape(tr.data)))
    problems.push('snippet structure (keys/labels/languages) differs')
  else if (!eq(snippetCode(en.data), snippetCode(tr.data)))
    problems.push('snippet code differs beyond comments')

  if (!eq(headingSkeleton(en.content), headingSkeleton(tr.content)))
    problems.push(`heading skeleton differs (en: ${headingSkeleton(en.content).length}, ${locale}: ${headingSkeleton(tr.content).length})`)

  const enCode = codeBlocks(en.content).map(stripComments)
  const trCode = codeBlocks(tr.content).map(stripComments)
  if (enCode.length !== trCode.length)
    problems.push(`code block count differs (en: ${enCode.length}, ${locale}: ${trCode.length})`)
  else if (!eq(enCode, trCode)) problems.push('body code differs beyond comments')

  if (!eq(linkTargets(en.content), linkTargets(tr.content)))
    problems.push('link targets differ')

  if (problems.length) {
    failures++
    console.log(`FAIL ${rel}`)
    for (const p of problems) console.log(`  - ${p}`)
  } else {
    console.log(`ok   ${rel}`)
  }
}

console.log(`\n${twins.length - failures}/${twins.length} passed`)
process.exit(failures ? 1 : 0)
