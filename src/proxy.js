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
  // must NOT swallow the markdown twins (/docs/models/rf-detr.md), which are
  // served by a route handler outside [locale] and need no locale routing.
  matcher: ['/', '/((?!api|_next|_vercel|opengraph-image|.*\\..*).*)', '/docs/((?!.*\\.md$).*)'],
}
