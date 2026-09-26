import fs from 'fs'
import path from 'path'
import matter from 'gray-matter'

import registry from '@/data/docs/registry.json'
import nav from '@/data/docs/nav.json'
import registryV150 from '@/data/docs/archive/v1.5.0/registry.json'
import navV150 from '@/data/docs/archive/v1.5.0/nav.json'
import { routing } from '@/i18n/routing'

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

/*
 * `content/docs/start/*.md` is the prefix-free group: those serve at
 * /docs/<slug> because their URLs are meant to be short and permanent.
 */
const STANDALONE_DIR = 'start'

/* The section index routes, which have no markdown file behind them. */
export const DOCS_SECTION_INDEXES = [
  '/docs', '/docs/models', '/docs/tasks', '/docs/export',
  '/docs/train', '/docs/predict', '/docs/cli', '/docs/reference',
]

/*
 * One docs tree: its markdown, registry, sidebar manifest and upstream files.
 *
 * The current tree at /docs is one source. A frozen release is another, served
 * under /docs/vX.Y.Z by the same page components, so an archived page renders
 * exactly as it did when it was current, from the data it had then.
 *
 * Paths handed in and out of a source are logical ("/docs/models/rf-detr"),
 * the same for every version. `href()` turns one into the URL a reader should
 * follow inside that tree: identity for the current tree, and the version
 * prefix for an archive when the archive has that page. A link to something
 * the archive never had (LibreVLM, the older single-page docs) is left alone.
 */
function createDocsSource({ version, registry, nav, docsDir, upstreamDir, basePath = '/docs', archived = false }) {
  const localizedNavCache = new Map()
  let pageSet = null

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

  function getDocSlugs(section) {
    const dir = path.join(docsDir, section)
    if (!fs.existsSync(dir)) return []
    return fs
      .readdirSync(dir)
      .filter((file) => file.endsWith('.md') && !/\.[a-z]{2}\.md$/.test(file))
      .map((file) => file.replace(/\.md$/, ''))
  }

  function readUpstream(slug) {
    const file = path.join(upstreamDir, `${slug}.json`)
    if (!slug || !fs.existsSync(file)) return null
    return JSON.parse(fs.readFileSync(file, 'utf8'))
  }

  function getAllDocPages() {
    if (!fs.existsSync(docsDir)) return []
    const pages = []
    for (const section of fs.readdirSync(docsDir)) {
      const dir = path.join(docsDir, section)
      if (!fs.statSync(dir).isDirectory()) continue
      for (const slug of getDocSlugs(section)) {
        const doc = readDoc(section, slug, 'en')
        if (!doc) continue
        pages.push({
          section,
          slug,
          // The URL a reader and a crawler actually see, for the current tree.
          path: section === STANDALONE_DIR ? `/docs/${slug}` : `/docs/${section}/${slug}`,
          title: doc.title || slug,
          description: doc.description || doc.lead || '',
        })
      }
    }
    return pages.sort((a, b) => a.path.localeCompare(b.path))
  }

  /* Whether this tree serves a logical path: a section index or a page. */
  function hasPath(logicalPath) {
    if (!pageSet) pageSet = new Set([...DOCS_SECTION_INDEXES, ...getAllDocPages().map((p) => p.path)])
    return pageSet.has(logicalPath)
  }

  function href(target) {
    if (!archived || typeof target !== 'string') return target
    const match = /^(\/docs(?:\/[^?#]*)?)([?#].*)?$/.exec(target)
    if (!match) return target
    const logical = match[1].replace(/\/$/, '') || '/docs'
    if (!hasPath(logical)) return target
    return `${basePath}${logical.slice('/docs'.length)}${match[2] ?? ''}`
  }

  /*
   * Resolve translated frontmatter titles once per locale, then reuse the map for
   * every page rendered during the build. Section indexes have no markdown twin,
   * so they naturally retain the label from nav.json. An archive's slugs point
   * inside the archive.
   */
  function localizeNav(locale) {
    const key = locale || routing.defaultLocale
    if (localizedNavCache.has(key)) return localizedNavCache.get(key)

    const titles = new Map()
    if (key !== routing.defaultLocale) {
      for (const group of nav.groups) {
        for (const item of group.items) {
          const parts = item.slug.replace(/^\/docs\/?/, '').split('/').filter(Boolean)
          if (!parts.length) continue
          const section = parts.length === 1 ? STANDALONE_DIR : parts[0]
          const slug = parts.length === 1 ? parts[0] : parts.slice(1).join('/')
          const file = path.join(docsDir, section, `${slug}.${key}.md`)
          if (!fs.existsSync(file)) continue
          const { data } = matter(fs.readFileSync(file, 'utf8'))
          if (data.title) titles.set(item.slug, data.title)
        }
      }
    }

    let localized = nav
    if (titles.size || archived) {
      localized = structuredClone(nav)
      for (const group of localized.groups) {
        for (const item of group.items) {
          item.label = titles.get(item.slug) ?? item.label
          item.slug = href(item.slug)
        }
      }
    }
    localizedNavCache.set(key, localized)
    return localized
  }

  return {
    version,
    label: `v${version}`,
    basePath,
    archived,
    registry,
    nav,
    href,
    hasPath,
    localizeNav,
    getAllDocPages,
    getDocSlugs,
    getDoc(section, slug, locale = 'en') {
      return readDoc(section, slug, locale)
    },
    getDocByPath(urlPath) {
      const rest = urlPath.replace(/^\/docs\//, '')
      const parts = rest.split('/')
      if (parts.length === 1) return readDoc(STANDALONE_DIR, parts[0], 'en')
      return readDoc(parts[0], parts.slice(1).join('/'), 'en')
    },
    // A model page names its registry families in frontmatter. Lineage pages
    // (the YOLOv9 page covering yolo9 + yolo9_e2e + yolo9_p2) list only the
    // primary key; registry_keys records the siblings it covers.
    getFamilies(keys = []) {
      return keys
        .map((key) => registry.families[key])
        .filter(Boolean)
        .map((family) => ({ ...family, upstream: family.upstream ?? readUpstream(family.slug) }))
    },
    getTierMeta(tier) {
      return registry.tiers[tier] || null
    },
    getTaskMeta(task) {
      return registry.tasks[task] || { label: task, slug: task }
    },
    getExportFormats() {
      return registry.export_formats
    },
  }
}

// Upstream metadata (paper, organization, license, BibTeX) is the one part of a
// family record that cannot be extracted mechanically: a citation has to be
// copied from the authors' own block and checked against the publisher. It
// therefore lives in one small file per lineage, written after that check, and
// is merged in getFamilies rather than being regenerated by the extractor.
export const currentDocs = createDocsSource({
  version: registry.libreyolo_version,
  registry,
  nav,
  docsDir: path.join(process.cwd(), 'content', 'docs'),
  upstreamDir: path.join(process.cwd(), 'src', 'data', 'docs', 'upstream'),
})

/*
 * Frozen multi-page docs trees, one per superseded release since the v2 tree.
 * Each is a snapshot (see the README beside it) served at /docs/vX.Y.Z/..., with
 * every page canonicalised to its counterpart in the current tree and never in
 * the sitemap. Adding a release: snapshot content/docs and src/data/docs, add an
 * entry here, and copy the v1.5.0 route folder.
 */
export const DOCS_ARCHIVES = {
  'v1.5.0': createDocsSource({
    version: '1.5.0',
    registry: registryV150,
    nav: navV150,
    docsDir: path.join(process.cwd(), 'content', 'docs-archive', 'v1.5.0'),
    upstreamDir: path.join(process.cwd(), 'src', 'data', 'docs', 'archive', 'v1.5.0', 'upstream'),
    basePath: '/docs/v1.5.0',
    archived: true,
  }),
}

export const ARCHIVED_DOCS_VERSIONS = Object.keys(DOCS_ARCHIVES)

export function getDocsArchive(version) {
  return DOCS_ARCHIVES[version] ?? null
}

export const DOCS_VERSION = currentDocs.version
export const DOCS_NAV = currentDocs.nav

export const localizeNav = currentDocs.localizeNav
export const getDoc = currentDocs.getDoc
export const getDocSlugs = currentDocs.getDocSlugs
export const getFamilies = currentDocs.getFamilies
export const getTierMeta = currentDocs.getTierMeta
export const getTaskMeta = currentDocs.getTaskMeta
export const getExportFormats = currentDocs.getExportFormats
export const getAllDocPages = currentDocs.getAllDocPages
export const getDocByPath = currentDocs.getDocByPath

// Imported, used below, and re-exported so server-side callers keep importing
// it from here. The implementation moved to its own module because the search
// dialog runs in the browser and cannot import anything that touches the
// filesystem. `export ... from` alone would not bind the name in this scope,
// and extractHeadings calls it.
import { slugifyHeading } from './slugify-heading'

export { slugifyHeading }

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

/* Frozen single-page docs for releases before the v2 tree. Kept reachable and
   canonicalised to /docs, never edited again, and never in the sitemap. */
export const LEGACY_DOCS_VERSIONS = ['v1.4.0', 'v1.3.1', 'v1.3.0', 'v1.2.0', 'v1.1.0']

export default registry
