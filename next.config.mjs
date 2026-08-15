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
}

export default withNextIntl(nextConfig)
