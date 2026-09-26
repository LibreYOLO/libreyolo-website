/*
 * Page bodies for the v2 docs tree, shared by the current tree at /docs and by
 * the frozen release trees at /docs/vX.Y.Z.
 *
 * Every view takes a `source` (see createDocsSource in src/lib/docs.js): the
 * markdown, registry and sidebar of one tree. Paths inside a view are logical
 * ("/docs/models/rf-detr") and pass through `source.href()` on their way into a
 * link, which is the identity for the current tree and adds the version prefix
 * for an archive. The current routes render byte-identical output through
 * these views; the archive differs only in its data, its links, the version
 * label in the rail, and the one-line notice naming the release.
 */

import { notFound } from 'next/navigation'
import { getTranslations } from 'next-intl/server'
import Link from 'next/link'

import { currentDocs, extractHeadings } from '@/lib/docs'
import { buildPageMetadata, localeUrl, SITE_URL } from '@/i18n/metadata'
import { routing } from '@/i18n/routing'
import DocsShell from '@/components/docs/DocsShell'
import DocMarkdown from '@/components/docs/DocMarkdown'
import Code from '@/components/docs/Code'
import { getModelDiagram } from '@/lib/model-diagrams'
import ModelArchitecture from '@/components/docs/ModelArchitecture'
import { ModelHeader, HeroMedia, PageHeader } from '@/components/docs/ModelBlocks'

/* The section indexes: /docs/models, /docs/export, and the rest. */
export const DOCS_SECTIONS = ['models', 'tasks', 'export', 'train', 'predict', 'cli', 'reference']

/*
 * The same single segment also serves the standalone pages that deliberately
 * carry no group prefix, because their URLs are permanent and short:
 * /docs/install, /docs/quickstart, /docs/licensing and the rest. Their markdown
 * lives in `content/docs/start/`, and a slug is only treated as one of these
 * when DOCS_SECTIONS does not claim it first.
 */
export const STANDALONE = 'start'

/*
 * Where the current version of an archived page lives: the same logical path
 * when the current tree still has it, /docs otherwise. Locale-aware the way the
 * current pages build their own canonicals: a page without a twin in `locale`
 * canonicalises to English.
 */
export function currentCounterpart(logicalPath, locale, section, slug) {
  if (section && slug) {
    const doc = currentDocs.getDoc(section, slug, locale)
    if (doc) return { path: logicalPath, url: localeUrl(logicalPath, doc.translated ? locale : routing.defaultLocale) }
  } else if (currentDocs.hasPath(logicalPath)) {
    return { path: logicalPath, url: localeUrl(logicalPath, locale) }
  }
  return { path: '/docs', url: localeUrl('/docs', locale) }
}

/*
 * One plain line naming the release, above an archived page's content. Same
 * type as the rest of the page's quiet text: no box, no colour, no badge.
 */
async function ArchiveNotice({ locale, source, currentPath }) {
  if (!source.archived) return null
  const t = await getTranslations({ locale, namespace: 'DocsChrome' })
  return (
    <p className="mb-6 text-[13.5px] text-surface-500 dark:text-surface-500">
      {t.rich('archivedNotice', {
        version: source.version,
        link: (chunks) => (
          <Link href={currentPath} className="text-libre-700 underline-offset-2 hover:underline dark:text-libre-400">
            {chunks}
          </Link>
        ),
      })}
    </p>
  )
}

function shellProps(source) {
  return {
    version: source.version,
    archived: source.archived,
    homeHref: source.href('/docs'),
  }
}

function breadcrumbJsonLd(breadcrumbs) {
  return {
    '@context': 'https://schema.org',
    '@type': 'BreadcrumbList',
    itemListElement: breadcrumbs.map((crumb, index) => ({
      '@type': 'ListItem',
      position: index + 1,
      name: crumb.label,
      ...(crumb.href ? { item: `${SITE_URL}${crumb.href}` } : {}),
    })),
  }
}

function JsonLd({ blocks }) {
  return blocks.map((block, index) => (
    <script
      key={index}
      type="application/ld+json"
      dangerouslySetInnerHTML={{ __html: JSON.stringify(block) }}
    />
  ))
}

/* ── landing ────────────────────────────────────────────────────── */

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

function Fact({ label, children }) {
  return (
    <div className="flex flex-col gap-x-3 sm:flex-row">
      <dt className="shrink-0 text-surface-500 dark:text-surface-500 sm:w-24">{label}</dt>
      <dd className="min-w-0 text-surface-700 dark:text-surface-300">{children}</dd>
    </div>
  )
}

export async function DocsLandingView({ locale, source = currentDocs }) {
  const t = await getTranslations({ locale, namespace: 'DocsLanding' })
  const docsNav = source.localizeNav(locale)
  const href = source.href
  const registry = source.registry

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
    .filter((item) => item.slug !== href('/docs/models') && item.built)
    .map((item) => {
      const family = families.find((f) => href(`/docs/models/${f.slug}`) === item.slug)
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
      <DocsShell nav={docsNav} activePath={href('/docs')} {...shellProps(source)} showActions={false}>
        <div className="max-w-3xl">
          <ArchiveNotice locale={locale} source={source} currentPath="/docs" />
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
            <Link href={href('/docs/quickstart')} className="font-medium text-libre-700 underline-offset-2 hover:underline dark:text-libre-400">
              {t('quickstart')}
            </Link>
            <Link href={href('/docs/models')} className="font-medium text-libre-700 underline-offset-2 hover:underline dark:text-libre-400">
              {t('browseModels')}
            </Link>
            <Link href={href('/docs/tasks')} className="font-medium text-libre-700 underline-offset-2 hover:underline dark:text-libre-400">
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
              <Link href={href('/docs/tasks')} className="text-[13px] text-libre-700 underline-offset-2 hover:underline dark:text-libre-400">
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
              <Link href={href('/docs/models')} className="text-[13px] text-libre-700 underline-offset-2 hover:underline dark:text-libre-400">
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
              <Link href={href('/docs/licensing')} className="font-medium text-libre-700 underline-offset-2 hover:underline dark:text-libre-400">
                {t('howLicensingWorks')}
              </Link>
            </p>
          </Section>

          <footer className="mt-14 border-t border-surface-200 pt-6 text-[13px] text-surface-500 dark:border-white/[0.09] dark:text-surface-500">
            <p>{t.rich('footer', {
              version: source.version,
              link: (chunks) => <Link href={href('/docs/versions')} className="text-libre-700 underline-offset-2 hover:underline dark:text-libre-400">{chunks}</Link>,
            })}</p>
          </footer>
        </div>
      </DocsShell>
    </>
  )
}

/* ── section index ──────────────────────────────────────────────── */

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
export async function SectionIndexView({ locale, section, source = currentDocs }) {
  const t = await getTranslations({ locale, namespace: 'DocsSections' })
  const chrome = await getTranslations({ locale, namespace: 'DocsChrome' })
  const tiers = await getTranslations({ locale, namespace: 'Tiers' })
  const docsNav = source.localizeNav(locale)
  const href = source.href

  const group = docsNav.groups.find((g) => g.id === section)
  if (!group) notFound()

  const logicalPath = `/docs/${section}`
  // The section's own index entry would just link to this page.
  const items = group.items.filter((item) => item.slug !== href(logicalPath))
  const title = t(`${section}.title`)
  const breadcrumbs = [{ label: chrome('docsCrumb'), href: href('/docs') }, { label: title }]

  return (
    <>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(breadcrumbJsonLd(breadcrumbs)) }} />
      <DocsShell nav={docsNav} activePath={href(logicalPath)} {...shellProps(source)} breadcrumbs={breadcrumbs} showActions={false}>
        <div className="max-w-3xl">
          <ArchiveNotice locale={locale} source={source} currentPath={currentCounterpart(logicalPath, locale).path} />
          <h1 className="text-[2.1rem] font-semibold tracking-tight text-surface-900 dark:text-white">
            {title}
          </h1>
          <p className="mt-2 max-w-[62ch] text-[15px] leading-relaxed text-surface-600 dark:text-surface-400">
            {t(`${section}.lead`)}
          </p>

          <dl className="mt-8 text-[14px]">
            {items.map((item) => {
              const tier = item.tier ? source.getTierMeta(item.tier) : null
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
                    {tier ? tiers(`${item.tier}.label`) : ''}
                    {tier && !item.built ? '. ' : ''}
                    {!item.built ? chrome('notWrittenYet') : ''}
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

/* ── standalone page ────────────────────────────────────────────── */

/*
 * A prefix-free page such as /docs/install. Same shell, same header and the same
 * markdown pipeline as a sectioned page; only the breadcrumb is shorter, because
 * there is no group above it.
 */
export async function StandaloneDocView({ locale, slug, source = currentDocs }) {
  const chrome = await getTranslations({ locale, namespace: 'DocsChrome' })
  const doc = source.getDoc(STANDALONE, slug, locale)
  if (!doc) notFound()

  const href = source.href
  const path = `/docs/${slug}`
  const url = localeUrl(path, doc.translated ? locale : routing.defaultLocale)
  const headings = extractHeadings(doc.content)
  const breadcrumbs = [{ label: chrome('docsCrumb'), href: href('/docs') }, { label: doc.title }]

  const jsonLd = [
    {
      '@context': 'https://schema.org',
      '@type': 'TechArticle',
      headline: doc.seo_title || doc.title,
      description: doc.description,
      mainEntityOfPage: url,
      inLanguage: doc.translated ? locale : routing.defaultLocale,
      publisher: { '@type': 'Organization', name: 'LibreYOLO', url: SITE_URL },
      about: doc.title,
      proficiencyLevel: 'Beginner',
    },
    breadcrumbJsonLd(breadcrumbs),
  ]

  return (
    <>
      <JsonLd blocks={jsonLd} />

      <DocsShell
        nav={source.localizeNav(locale)}
        activePath={href(path)}
        {...shellProps(source)}
        headings={headings}
        breadcrumbs={breadcrumbs}
      >
        {/* An untranslated twin serves the English source under /xx/: mark it English. */}
        <article className="max-w-3xl" lang={doc.translated ? undefined : "en"}>
          <ArchiveNotice locale={locale} source={source} currentPath={currentCounterpart(path, locale, STANDALONE, slug).path} />
          <PageHeader doc={doc} />

          <DocMarkdown snippets={doc.snippets || {}} bareTables {...(source.archived ? { source } : {})}>
            {doc.content}
          </DocMarkdown>

          <footer className="mt-16 border-t border-surface-200 pt-6 text-sm text-surface-500 dark:border-white/[0.06] dark:text-surface-500">
            <p>
              {doc.verification || chrome('verified', { version: doc.last_verified })}
            </p>
          </footer>
        </article>
      </DocsShell>
    </>
  )
}

/* ── sectioned page ─────────────────────────────────────────────── */

/*
 * /docs/<section>/<slug>. Model pages carry a registry family: its header,
 * hero, generated blocks and architecture diagram. Every other section (tasks,
 * export formats, workflows, CLI, reference) has no family, so the header rows
 * come from the page's own frontmatter and the body is prose plus snippet
 * groups.
 */
export async function SectionDocView({ locale, section, slug, source = currentDocs }) {
  const t = await getTranslations({ locale, namespace: 'DocsChrome' })

  const doc = source.getDoc(section, slug, locale)
  if (!doc) notFound()

  const isModel = section === 'models'
  const diagram = isModel ? getModelDiagram(slug) : null
  const [registeredFamily] = isModel ? source.getFamilies(doc.families) : []
  if (isModel && !registeredFamily && !(doc.architecture_only && diagram)) notFound()
  const family = registeredFamily || { display: doc.title }

  const href = source.href
  const path = `/docs/${section}/${slug}`
  const url = localeUrl(path, doc.translated ? locale : routing.defaultLocale)
  // Model pages are a usage reference: install, predict, variants, train,
  // validate, export, licensing. No FAQ and no related-links section.
  const headings = extractHeadings(doc.content)
  if (diagram) {
    const labels = await getTranslations({ locale, namespace: 'ModelDiagram' })
    headings.push({ id: 'architecture', title: labels('title', { model: diagram.title }) })
  }

  const breadcrumbs = [
    { label: t('docsCrumb'), href: href('/docs') },
    { label: t(`groups.${section}`), href: href(`/docs/${section}`) },
    { label: doc.title },
  ]

  const jsonLd = [
    {
      '@context': 'https://schema.org',
      '@type': 'TechArticle',
      headline: doc.seo_title || doc.title,
      description: doc.description,
      mainEntityOfPage: url,
      inLanguage: doc.translated ? locale : routing.defaultLocale,
      publisher: { '@type': 'Organization', name: 'LibreYOLO', url: SITE_URL },
      about: isModel ? family.display : doc.title,
      proficiencyLevel: 'Beginner',
    },
    breadcrumbJsonLd(breadcrumbs),
  ]

  if (isModel) {
    jsonLd.push({
      '@context': 'https://schema.org',
      '@type': 'SoftwareSourceCode',
      name: family.display,
      description: doc.description,
      codeRepository: 'https://github.com/LibreYOLO/libreyolo',
      programmingLanguage: 'Python',
      license: 'https://opensource.org/license/mit',
      runtimePlatform: 'PyTorch',
    })
    if (doc.hero?.src) {
      jsonLd.push({
        '@context': 'https://schema.org',
        '@type': 'VideoObject',
        name: `${family.display} running in LibreYOLO`,
        description: doc.hero.caption,
        thumbnailUrl: `${SITE_URL}${doc.hero.poster}`,
        contentUrl: `${SITE_URL}${doc.hero.src}`,
        uploadDate: '2026-08-08',
      })
    }
  }

  const archivedSource = source.archived ? { source } : {}

  return (
    <>
      <JsonLd blocks={jsonLd} />

      <DocsShell
        nav={source.localizeNav(locale)}
        activePath={href(path)}
        {...shellProps(source)}
        headings={headings}
        breadcrumbs={breadcrumbs}
      >
        {/* An untranslated twin serves the English source under /xx/: mark it English. */}
        <article className="max-w-3xl" lang={doc.translated ? undefined : "en"}>
          <ArchiveNotice locale={locale} source={source} currentPath={currentCounterpart(path, locale, section, slug).path} />
          {isModel ? (
            <>
              {registeredFamily ? <ModelHeader doc={doc} family={family} {...archivedSource} /> : <PageHeader doc={doc} />}
              <HeroMedia media={doc.hero} />

              <DocMarkdown family={family} snippets={doc.snippets || {}} {...archivedSource}>
                {doc.content}
              </DocMarkdown>

              {!doc.architecture_only && <footer className="mt-16 border-t border-surface-200 pt-6 text-sm text-surface-500 dark:border-white/[0.06] dark:text-surface-500">
                <p>{t('verifiedGenerated', { version: doc.last_verified })}</p>
              </footer>}
            </>
          ) : (
            <>
              <PageHeader doc={doc} />

              <DocMarkdown snippets={doc.snippets || {}} bareTables {...archivedSource}>
                {doc.content}
              </DocMarkdown>

              <footer className="mt-16 border-t border-surface-200 pt-6 text-sm text-surface-500 dark:border-white/[0.06] dark:text-surface-500">
                <p>
                  {doc.verification || t('verified', { version: doc.last_verified })}
                </p>
              </footer>
            </>
          )}
        </article>
        {diagram && <ModelArchitecture key={slug} diagram={diagram} />}
      </DocsShell>
    </>
  )
}

/* ── frozen release trees ───────────────────────────────────────── */

/*
 * Resolve the path segments after /docs/vX.Y.Z to one of the views above.
 * Returns null for a path the archive does not serve.
 */
function resolveArchivePath(source, parts = []) {
  if (parts.length === 0) return { kind: 'landing', path: '/docs' }
  if (parts.length === 1 && DOCS_SECTIONS.includes(parts[0])) {
    return { kind: 'index', section: parts[0], path: `/docs/${parts[0]}` }
  }
  if (parts.length === 1) {
    return source.getDoc(STANDALONE, parts[0], 'en')
      ? { kind: 'standalone', section: STANDALONE, slug: parts[0], path: `/docs/${parts[0]}` }
      : null
  }
  if (parts.length === 2 && DOCS_SECTIONS.includes(parts[0]) && source.getDoc(parts[0], parts[1], 'en')) {
    return { kind: 'page', section: parts[0], slug: parts[1], path: `/docs/${parts[0]}/${parts[1]}` }
  }
  return null
}

/* Every path below /docs/vX.Y.Z except the landing, as catch-all segments. */
export function archiveStaticParams(source) {
  return [
    ...DOCS_SECTIONS.map((section) => ({ slug: [section] })),
    ...source.getAllDocPages().map((page) => ({ slug: page.path.replace(/^\/docs\//, '').split('/') })),
  ]
}

/*
 * Metadata for an archived page. The canonical points at the page's current
 * counterpart (or /docs when the current tree dropped it), so ranking stays on
 * the unversioned URLs. No hreflang: the alternates of a page that
 * canonicalises elsewhere would contradict its canonical. And no noindex:
 * pairing noindex with a canonical is a documented way to get the wrong page
 * dropped from an index.
 */
export async function archivedDocsMetadata({ locale, source, parts }) {
  const target = resolveArchivePath(source, parts)
  if (!target) return {}

  let title
  let description
  let keywords
  let englishOnly = false
  if (target.kind === 'landing') {
    const t = await getTranslations({ locale, namespace: 'DocsLanding' })
    title = t('metaTitle')
    description = t('metaDescription')
  } else if (target.kind === 'index') {
    const t = await getTranslations({ locale, namespace: 'DocsSections' })
    title = t('metaTitle', { title: t(`${target.section}.title`) })
    description = t(`${target.section}.description`)
  } else {
    const doc = source.getDoc(target.section, target.slug, locale)
    title = doc.seo_title || doc.title
    description = doc.description
    keywords = doc.keywords
    englishOnly = !doc.translated
  }

  const current = currentCounterpart(
    target.path, locale,
    target.kind === 'standalone' || target.kind === 'page' ? target.section : undefined,
    target.slug,
  )
  const meta = buildPageMetadata({
    title: `${title} (${source.label})`,
    description,
    path: source.href(target.path),
    locale,
    englishOnly,
  })
  return { ...meta, alternates: { canonical: current.url }, ...(keywords ? { keywords } : {}) }
}

export async function ArchivedDocsView({ locale, source, parts }) {
  const target = resolveArchivePath(source, parts)
  if (!target) notFound()
  if (target.kind === 'landing') return <DocsLandingView locale={locale} source={source} />
  if (target.kind === 'index') return <SectionIndexView locale={locale} section={target.section} source={source} />
  if (target.kind === 'standalone') return <StandaloneDocView locale={locale} slug={target.slug} source={source} />
  return <SectionDocView locale={locale} section={target.section} slug={target.slug} source={source} />
}
