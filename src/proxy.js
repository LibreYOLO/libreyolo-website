import createMiddleware from 'next-intl/middleware'
import { routing } from './i18n/routing'

export default createMiddleware(routing)

export const config = {
  // Run on every path except API routes, Next internals, the dynamic OG image,
  // and any file with an extension (sitemap.xml, robots.txt, favicon.svg, images, etc.).
  // The explicit /docs/:path* entry re-includes versioned docs routes whose
  // segments contain dots (e.g. /docs/v1.1.0, /docs/v1.3.0); without it the
  // dot-in-path exclusion above would skip locale handling and 404 them.
  // The /docs re-include exists because versioned docs URLs contain dots
  // (/docs/v1.1.0) and the dot-exclusion above would otherwise 404 them. It
  // must NOT swallow the machine-readable siblings served by route handlers
  // outside [locale], which need no locale routing: the markdown twins
  // (/docs/models/rf-detr.md) and the search index
  // (/docs/search-index.json). Any future dotted sibling under /docs has to
  // be added to this exclusion or it will silently resolve to a locale page.
  // The alternation must be a NON-capturing group: Next rejects the matcher
  // outright ("Capturing groups are not allowed") and the dev server refuses
  // to start, so `(?:md|json)` rather than `(md|json)`.
  matcher: ['/', '/((?!api|_next|_vercel|opengraph-image|.*\\..*).*)', '/docs/((?!.*\\.(?:md|json)$).*)'],
}
