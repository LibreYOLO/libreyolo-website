import { getAllDocPages, getDocByPath } from '@/lib/docs'

/*
 * The search index, built once at build time and served as one static file.
 *
 * Deliberately not a hosted search service. The whole corpus is 171 short
 * pages; the index below is a few hundred KB, which is smaller than most of
 * the images a docs site would ship without comment, and it costs nothing to
 * run and cannot go down.
 *
 * What goes in: title, description, section, every heading, and the page's
 * frontmatter keywords. Headings are the highest-signal text on a reference
 * page (they are what the writer chose to name) and they let a result
 * deep-link to the exact anchor. Keywords are already the page's declared
 * synonyms, so indexing them is free relevance: without them "torch free"
 * ranked an unrelated detector above the lightweight-install page, because
 * that page never says the two words next to each other.
 *
 * Body prose is deliberately excluded: it would multiply the payload for
 * matches that are mostly noise on pages this short.
 *
 * Dotted paths bypass the locale proxy, same trick the .md twins use, so this
 * locale-less route is reachable at /docs/search-index.json.
 */
export const dynamic = 'force-static'

// Headings only, from the raw markdown. Fenced code can contain lines that
// start with #, so track fences and skip what is inside them.
function headingsOf(body) {
  if (!body) return []
  const out = []
  let fenced = false
  for (const line of body.split('\n')) {
    if (/^\s*(```|~~~)/.test(line)) { fenced = !fenced; continue }
    if (fenced) continue
    const m = line.match(/^(#{2,3})\s+(.+?)\s*#*\s*$/)
    if (m) out.push(m[2].replace(/[*`_[\]]/g, '').trim())
  }
  return out
}

export function GET() {
  const entries = getAllDocPages().map((page) => {
    const doc = getDocByPath(page.path)
    return {
      p: page.path,
      t: page.title,
      d: page.description,
      s: page.section,
      h: headingsOf(doc?.content),
      k: Array.isArray(doc?.keywords) ? doc.keywords : [],
    }
  })

  return new Response(JSON.stringify(entries), {
    headers: {
      'Content-Type': 'application/json; charset=utf-8',
      'Cache-Control': 'public, max-age=3600',
    },
  })
}
