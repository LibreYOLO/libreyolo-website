import createNextIntlPlugin from 'next-intl/plugin'

const withNextIntl = createNextIntlPlugin('./src/i18n/request.js')

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Preserve the public pathname through Proxy. Otherwise the local server can
  // combine next-intl's English rewrite with its canonical redirect into a loop.
  skipProxyUrlNormalize: true,
  turbopack: {
    root: '.',
  },
  // Hand-written pages that predate the docs tree, or are placeholders and
  // past events. Each goes to the page that now covers its subject, in every
  // locale.
  async redirects() {
    const moved = {
      '/docs/librevlm': '/docs/reference/vlm-api',
      '/docs/experimental': '/docs/tasks',
      '/science': '/docs/licensing',
      '/datasets': '/docs/train/datasets',
      '/cursor-hackathon': '/docs/quickstart',
    }
    const locales = 'en|zh|es|it|pt|fr|ru|id|vi|de|pl|ja|ko|uk'
    return Object.entries(moved).flatMap(([from, to]) => [
      { source: from, destination: to, permanent: true },
      { source: `/:locale(${locales})${from}`, destination: `/:locale${to}`, permanent: true },
    ])
  },
}

export default withNextIntl(nextConfig)
