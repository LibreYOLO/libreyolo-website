import fs from 'fs'
import path from 'path'
import matter from 'gray-matter'

import registry from '@/data/docs/registry.json'
import nav from '@/data/docs/nav.json'

const docsDir = path.join(process.cwd(), 'content', 'docs')

export const DOCS_VERSION = registry.libreyolo_version
export const DOCS_NAV = nav

// Docs pages mirror the article convention: the English source is
// `content/docs/<section>/<slug>.md` and a translation sits beside it as
// `<slug>.<locale>.md`. When a locale has no twin we serve English and the
// caller consolidates the canonical, so hreflang never claims a translation
// that does not exist.
// Authors write generated blocks as self-closing tags (`<checkpoint-table />`),
// which reads naturally in markdown. HTML5 does not allow self-closing syntax on
// unknown elements: the parser ignores the slash, leaves the tag open, and the
// rest of the page ends up nested inside it. So expand them to an explicit
// open/close pair before the markdown ever reaches the parser. Only custom
// elements (a hyphen in the name) are touched, never real HTML like `<br />`.
const SELF_CLOSING_CUSTOM_TAG = /<([a-z][a-z0-9]*-[a-z0-9-]*)((?:\s[^<>]*?)?)\s*\/>/g

export function expandSelfClosingTags(markdown) {
  return String(markdown).replace(SELF_CLOSING_CUSTOM_TAG, '<$1$2></$1>')
}

function readDoc(section, slug, locale) {
  const englishPath = path.join(docsDir, section, `${slug}.md`)
  const localizedPath =
    locale && locale !== 'en' ? path.join(docsDir, section, `${slug}.${locale}.md`) : null

  let filePath = englishPath
  let translated = false
  if (localizedPath && fs.existsSync(localizedPath)) {
    filePath = localizedPath
    translated = true
  }
  if (!fs.existsSync(filePath)) return null

  const { data, content } = matter(fs.readFileSync(filePath, 'utf8'))
  return { section, slug, ...data, content: expandSelfClosingTags(content), translated }
}

export function getDoc(section, slug, locale = 'en') {
  return readDoc(section, slug, locale)
}

export function getDocSlugs(section) {
  const dir = path.join(docsDir, section)
  if (!fs.existsSync(dir)) return []
  return fs
    .readdirSync(dir)
    .filter((file) => file.endsWith('.md') && !/\.[a-z]{2}\.md$/.test(file))
    .map((file) => file.replace(/\.md$/, ''))
}

// A model page names its registry families in frontmatter. Lineage pages (the
// YOLOv9 page covering yolo9 + yolo9_e2e + yolo9_p2) list several; the first is
// the primary and supplies the page's headline facts.
export function getFamilies(keys = []) {
  return keys.map((key) => registry.families[key]).filter(Boolean)
}

export function getTierMeta(tier) {
  return registry.tiers[tier] || null
}

export function getTaskMeta(task) {
  return registry.tasks[task] || { label: task, slug: task }
}

export function getExportFormats() {
  return registry.export_formats
}

// Heading anchors have to match between the rendered markdown and the "on this
// page" rail, so both sides derive ids through this one function.
export function slugifyHeading(text) {
  return String(text)
    .toLowerCase()
    .replace(/[^\w\s-]/g, '')
    .trim()
    .replace(/\s+/g, '-')
}

// The right-hand rail is built from the markdown source rather than the DOM so
// it renders server-side with no layout shift.
export function extractHeadings(markdown, extra = []) {
  const headings = []
  let inFence = false
  for (const line of String(markdown).split('\n')) {
    if (/^\s*```/.test(line)) { inFence = !inFence; continue }
    if (inFence) continue
    const match = /^##\s+(.+?)\s*$/.exec(line)
    if (match) headings.push({ id: slugifyHeading(match[1]), title: match[1] })
  }
  return [...headings, ...extra]
}

export default registry
