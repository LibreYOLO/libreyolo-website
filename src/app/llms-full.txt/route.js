import { getAllDocPages, getDocByPath } from '@/lib/docs'
import { docToMarkdown } from '@/lib/docs-markdown'
import { SITE_URL } from '@/i18n/metadata'

/*
 * /llms-full.txt: the whole documentation tree as one markdown document.
 *
 * llms.txt is an index; this is the corpus. It exists so an agent with a large
 * context can take the entire documentation in one fetch instead of crawling
 * 169 URLs, and so a reader who wants the old single-page docs back can still
 * search the lot with Ctrl-F.
 *
 * Generated from the same manifest as the sitemap and the per-page twins, so it
 * cannot drift from what is actually published.
 */

export const dynamic = 'force-static'

export function GET() {
  const pages = getAllDocPages()

  const header = `# LibreYOLO documentation

> The complete LibreYOLO documentation as a single markdown document.
> ${pages.length} pages. Source: ${SITE_URL}/docs
> Each page is also available on its own at <page-url>.md

LibreYOLO is an MIT-licensed computer vision library: one Python API across
detection, segmentation, pose, depth, OCR and more, with training, validation
and export for each. The code is MIT, so what you build with it stays yours.
Pretrained weights carry the license of whoever trained them, stated per
checkpoint.

`

  const body = pages
    .map((page) => {
      const doc = getDocByPath(page.path)
      if (!doc) return ''
      return `\n\n---\n\n<!-- ${SITE_URL}${page.path} -->\n\n${docToMarkdown(doc)}`
    })
    .filter(Boolean)
    .join('')

  return new Response(header + body, {
    headers: {
      'Content-Type': 'text/plain; charset=utf-8',
      'Cache-Control': 'public, max-age=0, s-maxage=3600, stale-while-revalidate=86400',
    },
  })
}
