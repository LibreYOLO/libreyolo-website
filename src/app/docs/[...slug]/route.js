import { getAllDocPages, getDocByPath } from '@/lib/docs'
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
  return getAllDocPages().map((page) => ({
    // "/docs/models/rf-detr" -> ["models", "rf-detr.md"]
    slug: `${page.path.replace(/^\/docs\//, '')}.md`.split('/'),
  }))
}

export async function GET(_request, { params }) {
  const { slug } = await params
  const parts = slug || []
  const joined = parts.join('/')
  if (!joined.endsWith('.md')) {
    return new Response('Not found', { status: 404 })
  }

  const doc = getDocByPath(`/docs/${joined.replace(/\.md$/, '')}`)
  if (!doc) return new Response('Not found', { status: 404 })

  return new Response(docToMarkdown(doc), {
    headers: {
      'Content-Type': 'text/markdown; charset=utf-8',
      'Cache-Control': 'public, max-age=0, s-maxage=3600, stale-while-revalidate=86400',
    },
  })
}
