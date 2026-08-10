// Translation freshness check.
//
// Every `<slug>.<locale>.md` twin carries a `source_hash` in its frontmatter:
// the hash of the English source it was translated from. When the English page
// changes and a twin is not retranslated, its stored hash stops matching and
// this script fails, so a stale translation cannot reach production silently.
//
//   node scripts/translation/sync-check.mjs           # check, exit 1 if stale
//   node scripts/translation/sync-check.mjs --stamp   # record current hashes
//
// Run --stamp only after actually retranslating the affected twins. Stamping
// without retranslating just tells the check to stop reporting a real problem.

import fs from 'node:fs'
import path from 'node:path'
import crypto from 'node:crypto'
import matter from 'gray-matter'

const stamp = process.argv.includes('--stamp')
const docsDir = path.join(process.cwd(), 'content', 'docs')
const TWIN = /\.([a-z]{2}(?:-[A-Z]{2})?)\.md$/

function* walk(dir) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name)
    if (entry.isDirectory()) yield* walk(full)
    else if (full.endsWith('.md')) yield full
  }
}

// Hash the English source's meaning, not its bytes: frontmatter plus body with
// line endings normalized, so a CRLF/LF flip alone never invalidates a twin.
function sourceHash(file) {
  const raw = fs.readFileSync(file, 'utf8').replace(/\r\n/g, '\n')
  return crypto.createHash('sha256').update(raw).digest('hex').slice(0, 16)
}

const twins = [...walk(docsDir)].filter((f) => TWIN.test(f))

let stale = 0
let missing = 0
let stamped = 0

for (const twin of twins) {
  const locale = twin.match(TWIN)[1]
  const enPath = twin.replace(TWIN, '.md')
  const rel = path.relative(docsDir, twin).replaceAll(path.sep, '/')

  if (!fs.existsSync(enPath)) {
    console.log(`ORPHAN ${rel}: no English source`)
    stale++
    continue
  }

  const current = sourceHash(enPath)
  const raw = fs.readFileSync(twin, 'utf8')
  const parsed = matter(raw.replace(/\r\n/g, '\n'))
  const stored = parsed.data.source_hash

  if (stamp) {
    if (stored !== current) {
      parsed.data.source_hash = current
      fs.writeFileSync(twin, matter.stringify(parsed.content, parsed.data))
      stamped++
    }
    continue
  }

  if (!stored) {
    console.log(`UNSTAMPED ${rel}: no source_hash (run --stamp)`)
    missing++
  } else if (stored !== current) {
    console.log(`STALE ${rel}: English source changed since translation`)
    console.log(`  locale=${locale} stored=${stored} current=${current}`)
    stale++
  }
}

if (stamp) {
  console.log(`stamped ${stamped} of ${twins.length} twins`)
  process.exit(0)
}

console.log(`\n${twins.length - stale - missing}/${twins.length} twins up to date`)
if (stale || missing) {
  console.log(
    stale
      ? `\n${stale} stale. Retranslate them, then re-run with --stamp.`
      : `\n${missing} unstamped. Run --stamp to record the current sources.`
  )
}
process.exit(stale ? 1 : 0)
