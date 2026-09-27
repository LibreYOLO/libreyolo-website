/*
 * Non-docs routes that are translated in every locale: their messages exist for
 * all of routing.locales, so /xx/<path> is a real page in each language. The
 * sitemap lists them once per locale, and docs links to them keep the reader's
 * locale. Docs pages and articles are translated per page and are resolved from
 * their twins instead.
 */
export const LOCALIZED_SITE_ROUTES = [
  { path: '', priority: 1.0 },
  { path: '/models', priority: 0.9 },
  { path: '/commercial', priority: 0.8 },
  { path: '/sponsors', priority: 0.6 },
  { path: '/science', priority: 0.8 },
  { path: '/datasets', priority: 0.7 },
  { path: '/articles', priority: 0.9 },
  { path: '/docs/librevlm', priority: 0.8 },
  { path: '/docs/experimental', priority: 0.8 },
  { path: '/benchmarks', priority: 0.9 },
  { path: '/cursor-hackathon', priority: 0.4 },
]
