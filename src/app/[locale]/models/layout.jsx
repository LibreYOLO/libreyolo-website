import { getTranslations } from 'next-intl/server'
import { buildPageMetadata, SITE_URL } from '@/i18n/metadata'
import { showcaseTaskGroups } from '@/lib/showcase'

export async function generateMetadata({ params }) {
  const { locale } = await params
  const t = await getTranslations({ locale, namespace: 'Metadata' })
  return buildPageMetadata({
    title: t('modelsTitle'),
    description: t('modelsDescription'),
    path: '/models',
    locale,
  })
}

// ItemList of every model family shown on the page, with the tasks it covers.
// Derived from the same showcase data the page renders, so it cannot drift.
function modelZooJsonLd() {
  const tasksByModel = new Map()
  for (const { task, models } of showcaseTaskGroups()) {
    for (const model of models) {
      tasksByModel.set(model, [...(tasksByModel.get(model) ?? []), task])
    }
  }
  return {
    '@context': 'https://schema.org',
    '@type': 'ItemList',
    name: 'LibreYOLO Model Zoo',
    description: 'Model families available in LibreYOLO, the MIT-licensed computer vision library, by task.',
    url: `${SITE_URL}/models`,
    numberOfItems: tasksByModel.size,
    itemListElement: [...tasksByModel].map(([name, tasks], i) => ({
      '@type': 'ListItem',
      position: i + 1,
      name,
      description: `Tasks: ${tasks.join(', ')}`,
    })),
  }
}

export default function ModelsLayout({ children }) {
  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(modelZooJsonLd()) }}
      />
      {children}
    </>
  )
}
