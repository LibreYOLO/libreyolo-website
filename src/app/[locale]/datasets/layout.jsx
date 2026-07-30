import { getTranslations } from 'next-intl/server'
import { buildPageMetadata, SITE_URL } from '@/i18n/metadata'

export async function generateMetadata({ params }) {
  const { locale } = await params
  const t = await getTranslations({ locale, namespace: 'Metadata' })
  return buildPageMetadata({
    title: t('datasetsTitle'),
    description: t('datasetsDescription'),
    path: '/datasets',
    locale,
  })
}

// The page is a hub pointing at the Hugging Face collection, so DataCatalog
// (a catalog of datasets) is the honest schema; individual Dataset entries
// live on Hugging Face, not here.
const jsonLd = {
  '@context': 'https://schema.org',
  '@type': 'DataCatalog',
  name: 'LibreYOLO Dataset Zoo',
  description: 'Datasets for training and evaluating YOLO models, hosted on Hugging Face.',
  url: `${SITE_URL}/datasets`,
  sameAs: 'https://huggingface.co/LibreYOLO/datasets',
  creator: {
    '@type': 'Organization',
    name: 'LibreYOLO',
    url: SITE_URL,
  },
}

export default function DatasetsLayout({ children }) {
  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />
      {children}
    </>
  )
}
