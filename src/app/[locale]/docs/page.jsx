import { getTranslations, setRequestLocale } from 'next-intl/server'
import Link from 'next/link'

import registry, { DOCS_VERSION, localizeNav } from '@/lib/docs'
import { buildPageMetadata, SITE_URL } from '@/i18n/metadata'
import DocsShell from '@/components/docs/DocsShell'
import Code from '@/components/docs/Code'

/*
 * Docs landing.
 *
 * This is a router, not an essay. Most arrivals come from a search engine with
 * one question, so the page's job is to put the install line and the right
 * onward link in front of them fast, and to establish in one pass that the
 * project is real: how many families, measured against what, under which
 * license.
 *
 * The reference-page austerity still applies. No stat tiles, no cards, no
 * pills. What sells here is the numbers being true and the routing being
 * complete, and every count below is read from the registry rather than typed.
 */

export async function generateMetadata({ params }) {
  const { locale } = await params
  const t = await getTranslations({ locale, namespace: 'DocsLanding' })
  return buildPageMetadata({
    title: t('metaTitle'),
    description: t('metaDescription'),
    path: '/docs',
    locale,
    englishOnly: false,
  })
}

function Section({ id, title, children, action }) {
  return (
    <section className="mt-14 border-t border-surface-200 pt-6 dark:border-white/[0.09]">
      <div className="mb-4 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <h2 id={id} className="scroll-mt-24 text-[1.35rem] font-semibold tracking-tight text-surface-900 dark:text-white">
          {title}
        </h2>
        {action}
      </div>
      {children}
    </section>
  )
}

/*
 * A routing list: name on the left, one line of what it is on the right.
 *
 * An entry whose page is not written yet still shows its description, because
 * the description is the useful part: it tells a reader the library does this
 * at all. Only the link is withheld. The section index pages say "not written
 * yet" because there the reader is asking about documentation progress; here
 * they are asking what the library does, and a column of that phrase would
 * answer a question nobody asked.
 */
function Routes({ items }) {
  return (
    <dl className="text-[14px]">
      {items.map((item) => (
        <div
          key={item.href || item.label}
          className="flex flex-col gap-x-4 border-b border-surface-200/70 py-2 last:border-0 sm:flex-row dark:border-white/[0.07]"
        >
          <dt className="shrink-0 sm:w-56">
            {item.built === false ? (
              <span className="text-surface-700 dark:text-surface-300">{item.label}</span>
            ) : (
              <Link href={item.href} className="font-medium text-libre-700 underline-offset-2 hover:underline dark:text-libre-400">
                {item.label}
              </Link>
            )}
          </dt>
          <dd className="text-surface-500 dark:text-surface-500">{item.note}</dd>
        </div>
      ))}
    </dl>
  )
}

export default async function DocsLanding({ params }) {
  const { locale } = await params
  setRequestLocale(locale)
  const t = await getTranslations({ locale, namespace: 'DocsLanding' })
  const docsNav = localizeNav(locale)

  const families = Object.values(registry.families)
  // Library-wide totals are counted by the registry generator, not typed here.
  const lib = registry.library ?? { families: null, tasks: null, export_formats: null }
  const checkpointCount = families.reduce((n, f) => n + f.checkpoints.length, 0)
  const benchmarked = families.filter((f) => f.va_embed)
  const groupOf = (id) => docsNav.groups.find((g) => g.id === id)?.items ?? []
  const taskBlurbs = t.raw('taskBlurbs')

  const tasks = groupOf('tasks').map((item) => ({
    href: item.slug,
    label: item.label,
    built: item.built,
    note: taskBlurbs[item.slug.split('/').at(-1)] ?? '',
  }))

  const models = groupOf('models')
    .filter((item) => item.slug !== '/docs/models' && item.built)
    .map((item) => {
      const family = families.find((f) => `/docs/models/${f.slug}` === item.slug)
      if (!family) return { href: item.slug, label: item.label, note: '' }
      const tasks = family.tasks.length === 1
        ? t('detection')
        : t('familyTaskCount', { count: family.tasks.length })
      // A family with nothing in our org is not a broken row, it is a licensing
      // fact. Saying "0 checkpoints" reads as an error and buries the reason.
      const weights = family.weights_hosted
        ? t('checkpointCount', { count: family.checkpoints.length })
        : t('weightsDistributed')
      return { href: item.slug, label: item.label, note: t('modelRouteNote', { tasks, weights }) }
    })

  const jsonLd = {
    '@context': 'https://schema.org',
    '@type': 'TechArticle',
    headline: t('metaTitle'),
    description: t('jsonLdDescription'),
    mainEntityOfPage: `${SITE_URL}/docs`,
    publisher: { '@type': 'Organization', name: 'LibreYOLO', url: SITE_URL },
  }

  return (
    <>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }} />
      <DocsShell nav={docsNav} activePath="/docs" version={DOCS_VERSION} showActions={false}>
        <div className="max-w-3xl">
          <h1 className="text-[2.4rem] font-semibold leading-tight tracking-tight text-surface-900 dark:text-white">
            {t('title')}
          </h1>
          <p className="mt-3 max-w-[60ch] text-[17px] leading-relaxed text-surface-600 dark:text-surface-400">
            {t('intro')}
          </p>

          <div className="mt-6 max-w-[420px]">
            <Code language="bash" label={t('installLabel')}>pip install libreyolo</Code>
          </div>

          <div className="mt-2 flex flex-wrap gap-x-5 gap-y-2 text-[14px]">
            <Link href="/docs/quickstart" className="font-medium text-libre-700 underline-offset-2 hover:underline dark:text-libre-400">
              {t('quickstart')}
            </Link>
            <Link href="/docs/models" className="font-medium text-libre-700 underline-offset-2 hover:underline dark:text-libre-400">
              {t('browseModels')}
            </Link>
            <Link href="/docs/tasks" className="font-medium text-libre-700 underline-offset-2 hover:underline dark:text-libre-400">
              {t('browseTasks')}
            </Link>
          </div>

          <dl className="mt-8 flex flex-col gap-y-1 border-t border-surface-200 pt-4 text-[13.5px] dark:border-white/[0.09]">
            <Fact label={t('facts.models.label')}>
              {t('facts.models.text', { count: lib.families })}
            </Fact>
            <Fact label={t('facts.tasks.label')}>
              {t('facts.tasks.text', { count: lib.tasks })}
            </Fact>
            <Fact label={t('facts.weights.label')}>
              {t.rich('facts.weights.text', {
                count: checkpointCount,
                hf: (chunks) => <a href="https://huggingface.co/LibreYOLO" target="_blank" rel="noopener noreferrer" className="text-libre-700 underline-offset-2 hover:underline dark:text-libre-400">{chunks}</a>,
              })}
            </Fact>
            <Fact label={t('facts.export.label')}>
              {t('facts.export.text', { count: lib.export_formats })}
            </Fact>
            <Fact label={t('facts.license.label')}>
              {t('facts.license.text')}
            </Fact>
          </dl>

          <Section
            id="tasks"
            title={t('whatToDo')}
            action={
              <Link href="/docs/tasks" className="text-[13px] text-libre-700 underline-offset-2 hover:underline dark:text-libre-400">
                {t('allTasks')}
              </Link>
            }
          >
            <Routes items={tasks} />
          </Section>

          <Section
            id="models"
            title={t('whichModel')}
            action={
              <Link href="/docs/models" className="text-[13px] text-libre-700 underline-offset-2 hover:underline dark:text-libre-400">
                {t('allModels')}
              </Link>
            }
          >
            <p className="mb-4 max-w-[68ch] text-[15px] leading-[1.6] text-surface-600 dark:text-surface-400">
              {t('flagship')}
            </p>
            <Routes items={models} />
          </Section>

          {benchmarked.length > 0 && (
            <Section
              id="benchmarks"
              title={t('measured')}
              action={
                <a href="https://www.visionanalysis.org/" target="_blank" rel="noopener noreferrer" className="text-[13px] text-libre-700 underline-offset-2 hover:underline dark:text-libre-400">
                  {t('visionAnalysis')}
                </a>
              }
            >
              <p className="mb-4 max-w-[68ch] text-[15px] leading-[1.6] text-surface-600 dark:text-surface-400">
                {t('measuredParagraph')}
              </p>
              <div className="relative w-full" style={{ paddingTop: '62.5%' }}>
                <iframe
                  src="https://www.visionanalysis.org/embed/scatter?title=Detection%20models%20on%20COCO&subtitle=Accuracy%20against%20latency"
                  title={t('iframeTitle')}
                  loading="lazy"
                  className="absolute inset-0 h-full w-full border border-surface-200 dark:border-white/[0.09]"
                  style={{ border: 0 }}
                />
              </div>
            </Section>
          )}

          <Section id="workflows" title={t('workflows')}>
            <Routes
              items={[
                ...groupOf('train').slice(0, 1).map((i) => ({ href: i.slug, label: t('workflowRows.train.label'), built: i.built, note: t('workflowRows.train.note') })),
                ...groupOf('predict').slice(0, 1).map((i) => ({ href: i.slug, label: t('workflowRows.predict.label'), built: i.built, note: t('workflowRows.predict.note') })),
                ...groupOf('export').slice(0, 1).map((i) => ({ href: i.slug, label: t('workflowRows.export.label'), built: i.built, note: t('workflowRows.export.note') })),
                ...groupOf('cli').slice(0, 1).map((i) => ({ href: i.slug, label: t('workflowRows.cli.label'), built: i.built, note: t('workflowRows.cli.note') })),
              ]}
            />
          </Section>

          <Section id="licensing" title={t('onTheLicense')}>
            <p className="max-w-[68ch] text-[15px] leading-[1.6] text-surface-600 dark:text-surface-400">
              {t('licensingParagraph')}
            </p>
            <p className="mt-3 text-[14px]">
              <Link href="/docs/licensing" className="font-medium text-libre-700 underline-offset-2 hover:underline dark:text-libre-400">
                {t('howLicensingWorks')}
              </Link>
            </p>
          </Section>

          <footer className="mt-14 border-t border-surface-200 pt-6 text-[13px] text-surface-500 dark:border-white/[0.09] dark:text-surface-500">
            <p>{t.rich('footer', {
              version: DOCS_VERSION,
              link: (chunks) => <Link href="/docs/versions" className="text-libre-700 underline-offset-2 hover:underline dark:text-libre-400">{chunks}</Link>,
            })}</p>
          </footer>
        </div>
      </DocsShell>
    </>
  )
}

function Fact({ label, children }) {
  return (
    <div className="flex flex-col gap-x-3 sm:flex-row">
      <dt className="shrink-0 text-surface-500 dark:text-surface-500 sm:w-24">{label}</dt>
      <dd className="min-w-0 text-surface-700 dark:text-surface-300">{children}</dd>
    </div>
  )
}
