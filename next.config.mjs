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
  // The hand-written LibreVLM page predates the docs tree, which now covers
  // the same ground at /docs/reference/vlm-api plus one page per VLM family.
  async redirects() {
    const vlm = '/docs/reference/vlm-api'
    return [
      { source: '/docs/librevlm', destination: vlm, permanent: true },
      { source: '/:locale(en|zh|es|it|pt|fr|ru|id|vi|de|pl|ja|ko|uk)/docs/librevlm', destination: `/:locale${vlm}`, permanent: true },
    ]
  },
}

export default withNextIntl(nextConfig)
