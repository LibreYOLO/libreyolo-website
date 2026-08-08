import { notFound } from 'next/navigation'
import { setRequestLocale } from 'next-intl/server'
import Link from 'next/link'

import { DOCS_NAV, DOCS_VERSION, getTierMeta } from '@/lib/docs'
import { buildPageMetadata, SITE_URL } from '@/i18n/metadata'
import DocsShell from '@/components/docs/DocsShell'

/*
 * Section index: /docs/models, /docs/export, and the rest.
 *
 * These exist because every page below them puts the section in its breadcrumb
 * and in its BreadcrumbList JSON-LD. A breadcrumb pointing at a 404 is a broken
 * promise to the reader and a structured-data error to a crawler.
 *
 * The listing is the nav manifest, so it cannot drift from the sidebar. Pages
 * not yet written are shown as plain text rather than hidden: the reader learns
 * what exists and what is coming, and no link leads nowhere.
 */

const SECTIONS = {
  models: {
    group: 'models',
    title: 'Models',
    lead: 'Every model family in LibreYOLO, grouped by how heavily it is supported.',
    description: 'All model families supported by LibreYOLO, from the flagship detectors to the historic ones, with their support tier.',
  },
  tasks: {
    group: 'tasks',
    title: 'Tasks',
    lead: 'What LibreYOLO can do, and which models serve each capability.',
    description: 'Every task LibreYOLO supports, from detection and segmentation to depth, OCR and gaze estimation.',
  },
  export: {
    group: 'export',
    title: 'Export and deploy',
    lead: 'Take a trained model out of PyTorch and run it somewhere else.',
    description: 'Export a LibreYOLO model to ONNX, TensorRT, OpenVINO, CoreML, TFLite and other deployment targets.',
  },
  train: {
    group: 'train',
    title: 'Train',
    lead: 'Datasets, arguments, multi-GPU and the machinery shared by every trainable family.',
    description: 'Training models in LibreYOLO: datasets, hyperparameters, augmentation, multi-GPU and experiment loggers.',
  },
  predict: {
    group: 'predict',
    title: 'Predict',
    lead: 'Running a model over images, folders, video and streams.',
    description: 'Running inference with LibreYOLO: sources, streaming, results objects and ensembling.',
  },
  cli: {
    group: 'cli',
    title: 'CLI',
    lead: 'Every command the libreyolo executable exposes.',
    description: 'The LibreYOLO command line: predict, train, val, export and the utility commands.',
  },
  reference: {
    group: 'reference',
    title: 'Reference',
    lead: 'The public Python surface, schemas and generated matrices.',
    description: 'LibreYOLO reference: Python API, results types, dataset formats and the full export matrix.',
  },
}

export function generateStaticParams() {
  return Object.keys(SECTIONS).map((section) => ({ section }))
}

export async function generateMetadata({ params }) {
  const { locale, section } = await params
  const meta = SECTIONS[section]
  if (!meta) return {}
  return buildPageMetadata({
    title: `${meta.title} | LibreYOLO docs`,
    description: meta.description,
    path: `/docs/${section}`,
    locale,
    englishOnly: true,
  })
}

export default async function SectionIndex({ params }) {
  const { locale, section } = await params
  setRequestLocale(locale)

  const meta = SECTIONS[section]
  if (!meta) notFound()

  const group = DOCS_NAV.groups.find((g) => g.id === meta.group)
  if (!group) notFound()

  // The section's own index entry would just link to this page.
  const items = group.items.filter((item) => item.slug !== `/docs/${section}`)
  const breadcrumbs = [{ label: 'Docs', href: '/docs' }, { label: meta.title }]

  const jsonLd = {
    '@context': 'https://schema.org',
    '@type': 'BreadcrumbList',
    itemListElement: breadcrumbs.map((crumb, index) => ({
      '@type': 'ListItem',
      position: index + 1,
      name: crumb.label,
      ...(crumb.href ? { item: `${SITE_URL}${crumb.href}` } : {}),
    })),
  }

  return (
    <>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }} />
      <DocsShell nav={DOCS_NAV} activePath={`/docs/${section}`} version={DOCS_VERSION} breadcrumbs={breadcrumbs}>
        <div className="max-w-3xl">
          <h1 className="text-[2.1rem] font-semibold tracking-tight text-surface-900 dark:text-white">
            {meta.title}
          </h1>
          <p className="mt-2 max-w-[62ch] text-[15px] leading-relaxed text-surface-600 dark:text-surface-400">
            {meta.lead}
          </p>

          <dl className="mt-8 text-[14px]">
            {items.map((item) => {
              const tier = item.tier ? getTierMeta(item.tier) : null
              return (
                <div
                  key={item.slug}
                  className="flex flex-col gap-x-4 border-b border-surface-200/70 py-2 last:border-0 sm:flex-row dark:border-white/[0.07]"
                >
                  <dt className="shrink-0 sm:w-56">
                    {item.built ? (
                      <Link href={item.slug} className="text-libre-700 underline-offset-2 hover:underline dark:text-libre-400">
                        {item.label}
                      </Link>
                    ) : (
                      <span className="text-surface-500 dark:text-surface-500">{item.label}</span>
                    )}
                  </dt>
                  <dd className="text-surface-500 dark:text-surface-500">
                    {tier ? tier.label : ''}
                    {tier && !item.built ? '. ' : ''}
                    {!item.built ? 'Not written yet.' : ''}
                  </dd>
                </div>
              )
            })}
          </dl>

          {group.more && (
            <p className="mt-4 text-[13px] text-surface-500 dark:text-surface-500">{group.more}.</p>
          )}
        </div>
      </DocsShell>
    </>
  )
}
