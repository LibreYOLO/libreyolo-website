import createMiddleware from 'next-intl/middleware'
import { routing } from './i18n/routing'

export default createMiddleware(routing)

export const config = {
  // Run on every path except API routes, Next internals, the dynamic OG image,
  // and any file with an extension (sitemap.xml, robots.txt, favicon.svg, images, etc.).
  matcher: ['/', '/((?!api|_next|_vercel|opengraph-image|.*\\..*).*)'],
}
