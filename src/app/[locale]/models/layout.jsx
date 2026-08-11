import { getTranslations } from 'next-intl/server'
import { buildPageMetadata, SITE_URL } from '@/i18n/metadata'
import { getTasksByModel, getAllModels } from '@/lib/models-index'

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
// Derived from the same registry-backed index the page renders, so it cannot
// drift: a family added to the library appears in both at once, and the list
// covers all 82 families rather than the 38 the old hand-kept showcase named.
function modelZooJsonLd(t) {
  const tasksByModel = getTasksByModel()
  const urlByModel = new Map(getAllModels().map((m) => [m.name, m.docsUrl]))
  return {
    '@context': 'https://schema.org',
    '@type': 'ItemList',
    name: t('zooName'),
    description: t('zooDescription'),
    url: `${SITE_URL}/models`,
    numberOfItems: tasksByModel.size,
    itemListElement: [...tasksByModel].map(([name, tasks], i) => ({
      '@type': 'ListItem',
      position: i + 1,
      name,
      url: urlByModel.has(name) ? `${SITE_URL}${urlByModel.get(name)}` : undefined,
      description: t('taskList', { tasks: tasks.join(', ') }),
    })),
  }
}

export default async function ModelsLayout({ children, params }) {
  const { locale } = await params
  const t = await getTranslations({ locale, namespace: 'Models' })
  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(modelZooJsonLd(t)) }}
      />
      {children}
    </>
  )
}
