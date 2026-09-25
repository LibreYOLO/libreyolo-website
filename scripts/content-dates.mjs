// Records when each content file last really changed, for sitemap <lastmod>.
//
// Production builds run on Vercel from an uploaded working tree with no git
// history, so file mtimes there are all the upload time and every page looked
// modified on every deploy. This script reads the dates from git locally and
// writes them to src/data/content-dates.json, which is committed and uploaded
// with the rest of the tree.
//
//   node scripts/content-dates.mjs
//
// Runs automatically before `next build` when a .git directory is present, and
// must run before `vercel --prod` (see skills/put-website-in-prod). Files with
// uncommitted changes get today's date.
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

const root = process.cwd()
const out = path.join(root, 'src', 'data', 'content-dates.json')

function git(args) {
  return execFileSync('git', args, { cwd: root, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 })
}

try {
  git(['rev-parse', '--git-dir'])
} catch {
  console.log('content-dates: no git repository, keeping the committed manifest')
  process.exit(0)
}

// One pass over history: the first time a path appears (newest first) is its
// last commit date.
const dates = {}
let current = null
for (const line of git(['log', '--format=@%cs', '--name-only', '--', 'content']).split('\n')) {
  if (line.startsWith('@')) current = line.slice(1)
  else if (line && !(line in dates)) dates[line] = current
}

const today = new Date().toISOString().slice(0, 10)
for (const line of git(['status', '--porcelain', '--', 'content']).split('\n')) {
  const file = line.slice(3).trim()
  if (file) dates[file] = today
}

const manifest = Object.fromEntries(
  Object.entries(dates)
    .filter(([file]) => file.endsWith('.md') && fs.existsSync(path.join(root, file)))
    .sort(([a], [b]) => a.localeCompare(b))
)
fs.writeFileSync(out, JSON.stringify(manifest, null, 2) + '\n')
console.log(`content-dates: ${Object.keys(manifest).length} files`)
