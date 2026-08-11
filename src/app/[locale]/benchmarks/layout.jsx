import { buildPageMetadata, SITE_URL } from '@/i18n/metadata'
import { getTranslations } from 'next-intl/server'

export async function generateMetadata({ params }) {
  const { locale } = await params
  const t = await getTranslations({ locale, namespace: 'Benchmarks' })
  return buildPageMetadata({
    title: t('metaTitle'),
    description: t('metaDescription'),
    path: '/benchmarks',
    locale,
    englishOnly: false,
  })
}

// Dataset + measurement provenance for the RF100-VL sweep, so the numbers on
// this page are machine-readable rather than locked inside the visualisation.
function benchmarksJsonLd(t) {
  return {
    '@context': 'https://schema.org',
    '@type': 'Dataset',
    name: t('datasetName'),
    description: t('datasetDescription'),
    url: `${SITE_URL}/benchmarks`,
    license: 'https://opensource.org/licenses/MIT',
    creator: { '@type': 'Organization', name: 'LibreYOLO', url: SITE_URL },
    isBasedOn: 'https://rf100-vl.org',
    measurementTechnique: 'pycocotools mAP@50-95, per-dataset test splits, maxDets 500',
    variableMeasured: 'mAP@50-95',
  }
}

export default async function BenchmarksLayout({ children, params }) {
  const { locale } = await params
  const t = await getTranslations({ locale, namespace: 'Benchmarks' })
  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(benchmarksJsonLd(t)) }}
      />
      {children}
    </>
  )
}
