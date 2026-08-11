// i18n readiness audit.
//
// Walks every page, layout and component and reports how much user-visible
// English is hardcoded in JSX rather than coming from messages/*.json or from
// a translated markdown twin. A page that never calls useTranslations and
// carries prose strings will render English in every locale.
//
//   node scripts/translation/i18n-audit.mjs          # summary table
//   node scripts/translation/i18n-audit.mjs --strings <file>   # list its strings

import fs from 'node:fs'
import path from 'node:path'

const roots = ['src/app', 'src/components']

function* walk(dir) {
  if (!fs.existsSync(dir)) return
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name)
    if (e.isDirectory()) {
      if (e.name === '.next' || e.name === 'node_modules') continue
      yield* walk(p)
    } else if (/\.(jsx?|tsx?)$/.test(p)) yield p
  }
}

// Strip what can never be user-visible prose so the count means something.
function stripNoise(src) {
  return src
    .replace(/\/\*[\s\S]*?\*\//g, '')            // block comments
    .replace(/^\s*\/\/.*$/gm, '')                 // line comments
    .replace(/import[^;]+;/g, '')                 // import specifiers
    .replace(/className=(["'])[\s\S]*?\1/g, '')   // tailwind classes
    .replace(/className=\{`[\s\S]*?`\}/g, '')
    .replace(/href=(["'])[^"']*\1/g, '')          // urls and paths
    .replace(/\b(d|viewBox|xmlns|fill|stroke|transform)=(["'])[^"']*\2/g, '') // svg
}

// A quoted run that reads like a sentence or label a user would see.
const PROSE = /(["'])([A-Z][A-Za-z0-9][^"'`]{11,}?)\1/g

const rows = []
for (const root of roots) {
  for (const file of walk(root)) {
    const src = fs.readFileSync(file, 'utf8')
    const translated = /useTranslations|getTranslations/.test(src)
    const body = stripNoise(src)
    const hits = [...body.matchAll(PROSE)]
      .map((m) => m[2].trim())
      .filter((s) => !/^[A-Z_]+$/.test(s))        // CONST_NAMES
      .filter((s) => !/^https?:|^\/|\.(md|json|png|jpg|svg|webp|mp4)$/i.test(s))
      .filter((s) => /\s/.test(s))                // needs at least one space
    rows.push({ file: file.replaceAll(path.sep, '/'), translated, count: hits.length, hits })
  }
}

const target = process.argv.includes('--strings') ? process.argv[process.argv.indexOf('--strings') + 1] : null
if (target) {
  const row = rows.find((r) => r.file.includes(target))
  if (!row) { console.error(`no file matching ${target}`); process.exit(2) }
  console.log(`${row.file}  (useTranslations: ${row.translated})`)
  for (const h of row.hits) console.log(`  ${h}`)
  process.exit(0)
}

rows.sort((a, b) => b.count - a.count)
const risky = rows.filter((r) => r.count > 0 && !r.translated)
const partial = rows.filter((r) => r.count > 0 && r.translated)
const clean = rows.filter((r) => r.count === 0)

console.log('UNTRANSLATABLE — hardcoded prose, no t() call at all:')
for (const r of risky) console.log(`  ${String(r.count).padStart(4)}  ${r.file}`)
console.log(`\nPARTIAL — calls t() but still holds hardcoded prose:`)
for (const r of partial) console.log(`  ${String(r.count).padStart(4)}  ${r.file}`)
console.log(`\nclean files: ${clean.length}`)
console.log(`\nTOTAL hardcoded strings: ${rows.reduce((n, r) => n + r.count, 0)} across ${risky.length + partial.length} files`)
