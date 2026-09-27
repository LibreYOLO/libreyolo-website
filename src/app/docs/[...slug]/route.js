import { currentDocs, DOCS_ARCHIVES, getDocsArchive } from '@/lib/docs'
import { SITE_URL } from '@/i18n/metadata'
import { docToMarkdown } from '@/lib/docs-markdown'

/*
 * Raw markdown twin for every docs page: /docs/models/rf-detr.md returns the
 * page as markdown.
 *
 * This sits outside `[locale]` on purpose. The next-intl proxy matcher skips
 * any path containing a dot, so a `.md` URL never enters locale routing and
 * lands here instead. Same arrangement the articles pipeline already uses.
 *
 * Why it exists: a growing share of readers are agents that fetch a page, read
 * it once and act. Handing them 300 KB of HTML to recover 2 KB of facts is
 * wasteful and lossy. The twin is the same content, with the generated tables
 * expanded, in the format they actually want.
 */

export const dynamic = 'force-static'

export function generateStaticParams() {
  const toSlug = (prefix, page) =>
    // "/docs/models/rf-detr" -> ["models", "rf-detr.md"]
    [...prefix, ...`${page.path.replace(/^\/docs\//, '')}.md`.split('/')]
  return [
    ...currentDocs.getAllDocPages().map((page) => ({ slug: toSlug([], page) })),
    // Frozen trees: /docs/v1.5.0/models/rf-detr.md
    ...Object.entries(DOCS_ARCHIVES).flatMap(([version, source]) =>
      source.getAllDocPages().map((page) => ({ slug: toSlug([version], page) }))),
  ]
}

export async function GET(_request, { params }) {
  const { slug } = await params
  const parts = slug || []
  const joined = parts.join('/')
  if (!joined.endsWith('.md')) {
    return new Response('Not found', { status: 404 })
  }

  const archive = getDocsArchive(parts[0])
  const source = archive ?? currentDocs
  const logicalPath = `/docs/${(archive ? parts.slice(1) : parts).join('/').replace(/\.md$/, '')}`
  const doc = source.getDocByPath(logicalPath)
  if (!doc) return new Response('Not found', { status: 404 })

  let markdown = docToMarkdown(doc, source)
  if (archive) {
    const current = currentDocs.getDocByPath(logicalPath) ? logicalPath : '/docs'
    markdown = `> This is the documentation for LibreYOLO ${archive.version}. Current version: ${SITE_URL}${current}\n\n${markdown}`
  }

  return new Response(markdown, {
    headers: {
      'Content-Type': 'text/markdown; charset=utf-8',
      'Cache-Control': 'public, max-age=0, s-maxage=3600, stale-while-revalidate=86400',
    },
  })
}
