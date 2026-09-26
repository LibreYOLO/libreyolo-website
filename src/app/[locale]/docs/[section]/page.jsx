import { getTranslations, setRequestLocale } from 'next-intl/server'

import { getDoc, getDocSlugs } from '@/lib/docs'
import { buildPageMetadata } from '@/i18n/metadata'
import { SectionIndexView, StandaloneDocView } from '@/components/docs/DocViews'

/*
 * Section index: /docs/models, /docs/export, and the rest. The page bodies
 * live in DocViews, shared with the frozen release trees.
 */

const SECTIONS = {
  models: {
    group: 'models',
  },
  tasks: {
    group: 'tasks',
  },
  export: {
    group: 'export',
  },
  train: {
    group: 'train',
  },
  predict: {
    group: 'predict',
  },
  cli: {
    group: 'cli',
  },
  reference: {
    group: 'reference',
  },
}

/*
 * The same single segment also serves the standalone pages that deliberately
 * carry no group prefix, because their URLs are permanent and short:
 * /docs/install, /docs/quickstart, /docs/licensing and the rest. Their markdown
 * lives in `content/docs/start/`, and a slug is only treated as one of these
 * when SECTIONS does not claim it first.
 */
const STANDALONE = 'start'

export function generateStaticParams() {
  return [
    ...Object.keys(SECTIONS).map((section) => ({ section })),
    ...getDocSlugs(STANDALONE).map((slug) => ({ section: slug })),
  ]
}

export async function generateMetadata({ params }) {
  const { locale, section } = await params
  const meta = SECTIONS[section]
  if (meta) {
    const t = await getTranslations({ locale, namespace: 'DocsSections' })
    return buildPageMetadata({
      title: t('metaTitle', { title: t(`${section}.title`) }),
      description: t(`${section}.description`),
      path: `/docs/${section}`,
      locale,
      englishOnly: false,
      ownImage: true,
    })
  }

  const doc = getDoc(STANDALONE, section, locale)
  if (!doc) return {}
  return {
    ...buildPageMetadata({
      title: doc.seo_title || doc.title,
      description: doc.description,
      path: `/docs/${section}`,
      locale,
      englishOnly: !doc.translated,
      ownImage: true,
    }),
    keywords: doc.keywords,
  }
}

export default async function SectionIndex({ params }) {
  const { locale, section } = await params
  setRequestLocale(locale)
  if (!SECTIONS[section]) return <StandaloneDocView locale={locale} slug={section} />
  return <SectionIndexView locale={locale} section={SECTIONS[section].group} />
}
