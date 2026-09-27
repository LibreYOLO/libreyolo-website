import fs from 'fs'
import path from 'path'
import matter from 'gray-matter'

import registry from '@/data/docs/registry.json'
import nav from '@/data/docs/nav.json'
import registryV150 from '@/data/docs/archive/v1.5.0/registry.json'
import navV150 from '@/data/docs/archive/v1.5.0/nav.json'
import { routing } from '@/i18n/routing'
import { getArticleBySlug } from '@/lib/articles'
import { LOCALIZED_SITE_ROUTES } from '@/lib/site-paths'

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
 *
 * `docDir`, `docFile` and `upstreamFile` build paths from a literal prefix and
 * extension at the call site, not from a directory passed in. The bundler's
 * file tracer can then narrow each tree to its own folder; a path joined from a
 * parameter makes it match every file in the project and warn. For the same
 * reason the archives live under content/archive/, not beside content/docs,
 * where the current tree's `content/docs<anything>` pattern would match them.
 */
function createDocsSource({ version, registry, nav, docsDir, docDir, docFile, upstreamFile, basePath = '/docs', archived = false }) {
  const localizedNavCache = new Map()
  let pageSet = null
  let pageIndex = null

  function readDoc(section, slug, locale) {
    const englishPath = docFile(section, slug)
    const localizedPath =
      locale && locale !== 'en' ? docFile(section, `${slug}.${locale}`) : null

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
    const dir = docDir(section)
    if (!fs.existsSync(dir)) return []
    return fs
      .readdirSync(dir)
      .filter((file) => file.endsWith('.md') && !/\.[a-z]{2}\.md$/.test(file))
      .map((file) => file.replace(/\.md$/, ''))
  }

  function readUpstream(slug) {
    const file = upstreamFile(slug)
    if (!slug || !fs.existsSync(file)) return null
    return JSON.parse(fs.readFileSync(file, 'utf8'))
  }

  function getAllDocPages() {
    if (!fs.existsSync(docsDir)) return []
    const pages = []
    for (const section of fs.readdirSync(docsDir)) {
      const dir = docDir(section)
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

  /*
   * Whether `locale` has its own version of a logical path in this tree. Section
   * indexes are message-driven, so every locale has them. A markdown page has
   * one only where its `<slug>.<locale>.md` twin exists; elsewhere the locale URL
   * serves the English page under the English canonical.
   */
  function isTranslated(logicalPath, locale) {
    if (!locale || locale === routing.defaultLocale) return true
    if (DOCS_SECTION_INDEXES.includes(logicalPath)) return true
    if (!pageIndex) pageIndex = new Map(getAllDocPages().map((p) => [p.path, p]))
    const page = pageIndex.get(logicalPath)
    return Boolean(page && fs.existsSync(docFile(page.section, `${page.slug}.${locale}`)))
  }

  /*
   * The locales that have their own version of a page: English plus every
   * locale with a twin. The sitemap and the page's hreflang alternates both use
   * this list, so the two always advertise the same set.
   */
  function docLocales(section, slug) {
    return routing.locales.filter((locale) =>
      locale === routing.defaultLocale || fs.existsSync(docFile(section, `${slug}.${locale}`)))
  }

  /*
   * The URL a reader in `locale` should follow for a link written as a logical
   * path. An archive adds its version prefix where it has the page. A locale
   * reader stays in their language when the target exists in it and lands on
   * the English page when it does not, so no link points at an English
   * fallback served under a locale prefix.
   */
  function href(target, locale = routing.defaultLocale) {
    if (typeof target !== 'string') return target
    const match = /^(\/docs(?:\/[^?#]*)?)([?#].*)?$/.exec(target)
    if (!match) return localizeHref(target, locale)
    const logical = match[1].replace(/\/$/, '') || '/docs'
    if (!hasPath(logical)) return localizeHref(target, locale)
    const suffix = match[2] ?? ''
    const path = archived ? `${basePath}${logical.slice('/docs'.length)}` : logical
    return `${prefixFor(isTranslated(logical, locale) ? locale : routing.defaultLocale)}${path}${suffix}`
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
          const file = docFile(section, `${slug}.${key}`)
          if (!fs.existsSync(file)) continue
          const { data } = matter(fs.readFileSync(file, 'utf8'))
          if (data.title) titles.set(item.slug, data.title)
        }
      }
    }

    let localized = nav
    if (titles.size || archived || key !== routing.defaultLocale) {
      localized = structuredClone(nav)
      for (const group of localized.groups) {
        for (const item of group.items) {
          item.label = titles.get(item.slug) ?? item.label
          item.slug = href(item.slug, key)
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
    isTranslated,
    docLocales,
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
  docDir: (section) => path.join(process.cwd(), 'content', 'docs', section),
  docFile: (section, name) => path.join(process.cwd(), 'content', 'docs', section, `${name}.md`),
  upstreamFile: (slug) => path.join(process.cwd(), 'src', 'data', 'docs', 'upstream', `${slug}.json`),
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
    docsDir: path.join(process.cwd(), 'content', 'archive', 'docs', 'v1.5.0'),
    docDir: (section) => path.join(process.cwd(), 'content', 'archive', 'docs', 'v1.5.0', section),
    docFile: (section, name) => path.join(process.cwd(), 'content', 'archive', 'docs', 'v1.5.0', section, `${name}.md`),
    upstreamFile: (slug) => path.join(process.cwd(), 'src', 'data', 'docs', 'archive', 'v1.5.0', 'upstream', `${slug}.json`),
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

/*
 * `docLocales` feeds both the sitemap and a page's hreflang alternates.
 * `docAlternates` is what a current-tree page route hands to buildPageMetadata:
 * the locales to list, and whether this URL is an English fallback that must
 * canonicalise to English instead. An English page is only English-only when no
 * locale has a twin; a locale URL is when its own twin is missing.
 */
export const docLocales = currentDocs.docLocales

export function docAlternates(section, slug, locale) {
  const locales = docLocales(section, slug)
  const englishOnly = locale === routing.defaultLocale
    ? locales.length === 1
    : !locales.includes(locale)
  return { locales, englishOnly }
}

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

/* ── locale-aware links ─────────────────────────────────────────── */

const LOCALIZED_PATHS = new Set(LOCALIZED_SITE_ROUTES.map((route) => route.path || '/'))
const LOCALE_PREFIX = new RegExp(`^/(${routing.locales.join('|')})(?=/|$)`)

function prefixFor(locale) {
  return locale === routing.defaultLocale ? '' : `/${locale}`
}

/*
 * Whether `locale` has its own version of a site path (no locale prefix).
 * Current docs pages and the frozen release trees answer from their twins,
 * articles from theirs, and the routes in LOCALIZED_SITE_ROUTES exist in every
 * locale. Everything else, the single-page docs before v2 included, is treated
 * as English only.
 */
function existsInLocale(pathname, locale) {
  if (LOCALIZED_PATHS.has(pathname)) return true
  const archive = /^\/docs\/(v\d+\.\d+\.\d+)(\/.*)?$/.exec(pathname)
  if (archive) {
    const source = DOCS_ARCHIVES[archive[1]]
    const logical = `/docs${archive[2] ?? ''}`
    return Boolean(source && source.hasPath(logical) && source.isTranslated(logical, locale))
  }
  if (currentDocs.hasPath(pathname)) return currentDocs.isTranslated(pathname, locale)
  const article = /^\/articles\/([^/]+)$/.exec(pathname)
  if (article) return Boolean(getArticleBySlug(article[1], locale)?.translated)
  return false
}

/*
 * Resolve an internal link for a reader in `locale`. English, external links,
 * anchors and already-prefixed paths are returned unchanged. A site path gets
 * the locale prefix when that locale has its own page there, and stays on the
 * English URL when it does not, so a link never lands on an English fallback
 * served under a locale prefix (which canonicalises to English anyway).
 */
export function localizeHref(target, locale) {
  if (!locale || locale === routing.defaultLocale) return target
  if (typeof target !== 'string' || !target.startsWith('/') || target.startsWith('//')) return target
  if (LOCALE_PREFIX.test(target)) return target
  const match = /^([^?#]*)([?#].*)?$/.exec(target)
  const pathname = match[1].replace(/\/$/, '') || '/'
  if (!existsInLocale(pathname, locale)) return target
  return `${prefixFor(locale)}${pathname === '/' ? '' : pathname}${match[2] ?? ''}`
}

export default registry
