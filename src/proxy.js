import createMiddleware from 'next-intl/middleware'
import { NextResponse } from 'next/server'
import { routing, localePreferenceCookie } from './i18n/routing'

const intlMiddleware = createMiddleware(routing)

// A reader who picked a language in the menu keeps it: landing on
// /es/models with English chosen redirects to /models, and the other way
// round. Readers who never picked one are untouched, so the root still
// serves English to everyone by default and crawlers (which send no
// cookies) see every locale at its own URL.
export default function proxy(request) {
  const preferred = request.cookies.get(localePreferenceCookie)?.value
  if (routing.locales.includes(preferred)) {
    const { pathname } = request.nextUrl
    const [, first, ...rest] = pathname.split('/')
    const hasPrefix = routing.locales.includes(first)
    const current = hasPrefix ? first : routing.defaultLocale
    if (preferred !== current) {
      const bare = hasPrefix ? `/${rest.join('/')}` : pathname
      const target =
        preferred === routing.defaultLocale ? bare : `/${preferred}${bare === '/' ? '' : bare}`
      // Resolve against request.url, not nextUrl: nextUrl can report a
      // different host than the one the reader typed.
      return NextResponse.redirect(new URL(target + request.nextUrl.search, request.url))
    }
  }
  return intlMiddleware(request)
}

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
