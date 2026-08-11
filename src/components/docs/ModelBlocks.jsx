/*
 * Reference blocks for a model family page.
 *
 * House rules, taken from how mature reference sites actually present a single
 * entity with dense structured facts (rustdoc, pkg.go.dev, MDN, caniuse,
 * Wikipedia infoboxes, PyPI):
 *
 *   1. No stat tiles. A static property is never set as a figure. Counts sit at
 *      body size inline, next to their label.
 *   2. No card wrappers. Separation comes from hairline rules and whitespace.
 *      A border exists to separate rows, not to make a container.
 *   3. The label is the quieter element; emphasis goes to the value.
 *   4. Color carries state inside a matrix, plus links. Nothing else.
 *   5. Matrix cells hold one short token. State is signalled by icon SHAPE as
 *      well as color, so the table survives grayscale and colorblindness.
 *   6. The legend is a definition list of full sentences, not one-word pills.
 *   7. Anything longer than a token goes below the table, never inside a cell.
 *
 * All data comes from the generated registry; nothing here is authored prose.
 */

import { Fragment } from 'react'
import Link from 'next/link'
import { useTranslations } from 'next-intl'
import { getTaskMeta, getTierMeta, getExportFormats } from '@/lib/docs'

const HF_BASE = 'https://huggingface.co/LibreYOLO'

/* ── shared type ────────────────────────────────────────────────── */

export function SectionTitle({ id, children }) {
  return (
    <h2
      id={id}
      className="scroll-mt-24 mt-14 mb-4 border-b border-surface-200 pb-2 text-[1.35rem] font-semibold tracking-tight text-surface-900 first:mt-0 dark:border-white/[0.09] dark:text-white"
    >
      {children}
    </h2>
  )
}

function Note({ children }) {
  return <p className="mt-2 text-[13px] leading-relaxed text-surface-500 dark:text-surface-500">{children}</p>
}

function ExtLink({ href, children }) {
  return (
    <a href={href} target="_blank" rel="noopener noreferrer" className="text-libre-700 underline-offset-2 hover:underline dark:text-libre-400">
      {children}
    </a>
  )
}

/* Bare table. No wrapper border, no radius, no fill, no hover.
   Hairline row rules only, and its own horizontal scroll container. */
function Table({ children, className = '' }) {
  return (
    <div className="-mx-1 overflow-x-auto px-1">
      <table className={`w-full border-collapse text-[13.5px] ${className}`}>{children}</table>
    </div>
  )
}

function Th({ children, align = 'left', className = '' }) {
  return (
    <th
      scope="col"
      className={`border-b border-surface-300 px-3 py-1.5 text-${align} font-semibold text-surface-700 dark:border-white/20 dark:text-surface-300 ${className}`}
    >
      {children}
    </th>
  )
}

function Td({ children, className = '' }) {
  return (
    <td className={`border-b border-surface-200/70 px-3 py-1.5 align-top text-surface-700 dark:border-white/[0.07] dark:text-surface-400 ${className}`}>
      {children}
    </td>
  )
}

/* ── header ─────────────────────────────────────────────────────── */

/*
 * pkg.go.dev's convention: one wrapping run of `Label: value` in small text
 * under the title. Labels muted, values carrying the emphasis and the links.
 * Fixed field order, so the same fact is always in the same place across all
 * 80 model pages. That fixed vocabulary is what stops a dense block of facts
 * from reading as decoration.
 */
export function ModelHeader({ doc, family }) {
  const t = useTranslations('ModelBlocks')
  const tiers = useTranslations('Tiers')
  const tier = getTierMeta(family.tier)
  const u = family.upstream
  const taskNames = family.tasks.map((t) => getTaskMeta(t).label.toLowerCase()).join(', ')

  return (
    <header className="mb-8">
      <h1 className="text-[2.1rem] font-semibold tracking-tight text-surface-900 dark:text-white">
        {doc.title}
      </h1>
      <p className="mt-2 max-w-[62ch] text-[15px] leading-relaxed text-surface-600 dark:text-surface-400">
        {doc.lead}
      </p>

      <dl className="mt-5 flex flex-col gap-y-1 border-t border-surface-200 pt-4 text-[13.5px] dark:border-white/[0.09]">
        <Meta label={t('tasks')}>{taskNames}</Meta>
        <Meta label={t('sizes')}>{family.sizes_label}</Meta>
        <Meta label={t('install')}>
          {/* Most families need no extra; only quote the bracket form when
              there is actually one, or the row reads pip install "libreyolo[]". */}
          <code className="font-mono text-[12.5px]">
            {family.extra ? `pip install "libreyolo[${family.extra}]"` : 'pip install libreyolo'}
          </code>
        </Meta>
        <Meta label={t('supportTierLabel')}>
          {t('supportTier', {
            tier: tier ? tiers(`${family.tier}.label`) : '',
            version: family.added_in,
            blurb: tier ? tiers(`${family.tier}.blurb`) : '',
          })}
        </Meta>
        <Meta label={t('upstream')}>
          {t.rich('upstreamValue', {
            name: u.name,
            org: u.org,
            license: u.license,
            paper: (chunks) => <ExtLink href={u.paper_url}>{chunks}</ExtLink>,
            source: (chunks) => <ExtLink href={u.code_url}>{chunks}</ExtLink>,
          })}
        </Meta>
        <Meta label={t('licenses')}>
          {/* LibreYOLO's own code is MIT, but a vendored port keeps its
              upstream license, so the header cannot assert MIT for every
              family. `code_license` overrides where they differ. */}
          {t.rich('licensesValue', {
            codeLicense: u.code_license ?? 'MIT',
            weightsLicense: u.license,
            link: (chunks) => <Link href="#licensing" className="text-libre-700 underline-offset-2 hover:underline dark:text-libre-400">{chunks}</Link>,
          })}
        </Meta>
      </dl>
    </header>
  )
}

function Meta({ label, children }) {
  return (
    <div className="flex flex-col gap-x-3 sm:flex-row">
      <dt className="shrink-0 text-surface-500 dark:text-surface-500 sm:w-28">{label}</dt>
      <dd className="min-w-0 text-surface-700 dark:text-surface-300">{children}</dd>
    </div>
  )
}

/*
 * Header for a docs page with no registry family behind it: an export format, a
 * CLI command, a guide. Same shape as ModelHeader so the two page types read as
 * one system, but the metadata rows come from the page's own frontmatter,
 * because there is no registry slice to generate them from.
 *
 * A row is { label, value, mono?, links? }. `mono` sets the value in the
 * identifier face; `links` appends external links after it.
 */
export function PageHeader({ doc }) {
  const rows = doc.meta || []

  return (
    <header className="mb-8">
      <h1 className="text-[2.1rem] font-semibold tracking-tight text-surface-900 dark:text-white">
        {doc.title}
      </h1>
      {doc.lead && (
        <p className="mt-2 max-w-[62ch] text-[15px] leading-relaxed text-surface-600 dark:text-surface-400">
          {doc.lead}
        </p>
      )}

      {rows.length > 0 && (
        <dl className="mt-5 flex flex-col gap-y-1 border-t border-surface-200 pt-4 text-[13.5px] dark:border-white/[0.09]">
          {rows.map((row) => (
            <Meta key={row.label} label={row.label}>
              {row.mono ? <code className="font-mono text-[12.5px]">{row.value}</code> : row.value}
              {row.links?.length ? (
                <>
                  {' '}
                  {row.links.map((link, index) => (
                    <Fragment key={link.href}>
                      {index > 0 && ', '}
                      <ExtLink href={link.href}>{link.label}</ExtLink>
                    </Fragment>
                  ))}
                </>
              ) : null}
            </Meta>
          ))}
        </dl>
      )}
    </header>
  )
}

/* ── hero media ─────────────────────────────────────────────────── */

export function HeroMedia({ media }) {
  if (!media) return null
  return (
    <figure className="my-8 max-w-[560px]">
      <video
        className="block aspect-video w-full border border-surface-200 dark:border-white/[0.09]"
        poster={media.poster}
        autoPlay
        muted
        loop
        playsInline
        preload="none"
      >
        <source src={media.src} type="video/mp4" />
      </video>
      {media.caption && (
        <figcaption className="mt-2 text-[13px] text-surface-500 dark:text-surface-500">{media.caption}</figcaption>
      )}
    </figure>
  )
}

/* ── benchmarks ─────────────────────────────────────────────────── */

export function BenchmarkTable({ family, task = 'detect' }) {
  const t = useTranslations('ModelBlocks')
  const bench = family.benchmarks?.[task]
  if (!bench) return null

  return (
    <div>
      <Table className="tabular-nums">
        <thead>
          <tr>
            {/* Units live in the column head, never repeated in every cell.
                No latency columns: a millisecond figure is meaningless without
                its hardware and runtime, and Vision Analysis already compares
                those properly. The embed below carries that axis. */}
            <Th>{t('checkpoint')}</Th>
            <Th align="right">{t('inputPx')}</Th>
            <Th align="right">{bench.metric}</Th>
            <Th align="right">{t('paramsM')}</Th>
          </tr>
        </thead>
        <tbody>
          {bench.rows.map((row) => (
            <tr key={row.size}>
              {/* Label each row with its OWN key's prefix. A lineage page mixes
                  rows from sibling families, and using the primary prefix for
                  all of them credits one version's accuracy to another
                  version's filename, often one that does not exist. */}
              <Td className="font-mono text-[12.5px] text-surface-900 dark:text-surface-200">
                {(family.prefixes?.[row.prefix_key] ?? family.prefix) || ''}{row.size}
              </Td>
              <Td className="text-right">{row.imgsz}</Td>
              <Td className="text-right font-medium text-surface-900 dark:text-surface-200">{row.map.toFixed(1)}</Td>
              <Td className="text-right">{row.params_m ?? ''}</Td>
            </tr>
          ))}
        </tbody>
      </Table>
      <Note>{t.rich('benchmarkNote', {
        dataset: bench.dataset,
        link: (chunks) => <ExtLink href={bench.source_url}>{chunks}</ExtLink>,
      })}</Note>
    </div>
  )
}

export function VaEmbed({ family }) {
  const t = useTranslations('ModelBlocks')
  const src = family.va_embed?.scatter
  if (!src) return null
  return (
    <div className="my-6">
      <div className="relative w-full" style={{ paddingTop: '62.5%' }}>
        <iframe
          src={src}
          title={t('accuracyVersusLatency', { family: family.display })}
          loading="lazy"
          className="absolute inset-0 h-full w-full border border-surface-200 dark:border-white/[0.09]"
          style={{ border: 0 }}
        />
      </div>
    </div>
  )
}

/* ── checkpoints ────────────────────────────────────────────────── */

export function CheckpointTable({ family }) {
  const t = useTranslations('ModelBlocks')
  const grouped = family.tasks
    .map((task) => ({ task, rows: family.checkpoints.filter((c) => c.task === task) }))
    .filter((g) => g.rows.length)

  return (
    <div>
      <Table>
        <thead>
          <tr>
            {/* No params column: it is only recorded for a handful of rows,
                and a mostly-empty column reads as a broken table. Parameter
                counts live in the benchmark table, where they are known. */}
            {/*
              No "Trained on" column. The registry cannot source it reliably:
              filename tokens cover a handful of checkpoints, and the Hugging
              Face dataset tags are inconsistent where they exist at all. A
              task-based default put "COCO" on ImageNet classifiers and OCR
              models. The linked repository is the authority for provenance,
              which is what the licensing note already tells the reader.
            */}
            <Th>{t('file')}</Th>
            <Th align="right">{t('inputPx')}</Th>
            <Th>{t('weightsLicense')}</Th>
          </tr>
        </thead>
        <tbody>
          {grouped.map(({ task, rows }) => (
            <Fragment key={task}>
              <tr>
                <td colSpan={3} className="border-b border-surface-200/70 px-3 pb-1 pt-4 text-[11px] font-semibold uppercase tracking-wider text-surface-500 dark:border-white/[0.07] dark:text-surface-500">
                  {getTaskMeta(task).label}
                </td>
              </tr>
              {rows.map((row) => (
                <tr key={row.name}>
                  <Td>
                    <a
                      href={`${HF_BASE}/${row.name.replace(/\.pt$/, '')}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="font-mono text-[12.5px] text-libre-700 underline-offset-2 hover:underline dark:text-libre-400"
                    >
                      {row.name}
                    </a>
                  </Td>
                  <Td className="text-right tabular-nums">{row.imgsz}</Td>

                  <Td>{row.license}</Td>
                </tr>
              ))}
            </Fragment>
          ))}
        </tbody>
      </Table>
      <Note>{t.rich('hostedWeightsNote', {
        link: (chunks) => <ExtLink href={HF_BASE}>{chunks}</ExtLink>,
      })}</Note>
    </div>
  )
}

/* ── export matrix ──────────────────────────────────────────────── */

/*
 * Three states, three distinguishable icon shapes: a filled disc with a check,
 * a half-filled disc, an empty rule. Color reinforces the shape but never
 * carries the meaning alone, and the cell background stays untouched. Each mark
 * carries the full sentence in `title` plus screen-reader text; the legend
 * below repeats those sentences in a definition list.
 */
/*
 * The matrix answers one question: can I export this task to this format.
 * A tick means yes. The library's finer internal grading (parity validated
 * versus implemented but not yet parity checked) is not a distinction the
 * reader can act on in a grid, so it moves to the notes below the table,
 * where the specific measured caveat can be stated in words.
 */
const SUPPORTED_STATES = new Set(['validated', 'available'])

const TICK = (
  <svg viewBox="0 0 16 16" className="h-[13px] w-[13px]" aria-hidden="true">
    <path d="M3.2 8.6l3.1 3.1 6.5-6.8" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
)

/*
 * The tooltip and screen-reader text prefer the library's own per-cell reason
 * over the generic state sentence. Those reasons are the measured findings that
 * decided the tier (which metric drifted, and by how much), so a reader asking
 * "why is this not validated" gets the real answer rather than a category.
 */
function Mark({ state, label, reason }) {
  const t = useTranslations('ModelBlocks')
  const supported = SUPPORTED_STATES.has(state)
  if (!supported) {
    return (
      <span className="sr-only">{t('notSupported', { label })}</span>
    )
  }
  return (
    <span
      className="inline-flex text-emerald-600 dark:text-emerald-400"
      title={reason ? t('supportedWithReason', { label, reason }) : t('supported', { label })}
    >
      {TICK}
      <span className="sr-only">{reason ? t('supportedWithReason', { label, reason }) : t('supported', { label })}</span>
    </span>
  )
}

export function ExportMatrix({ family }) {
  const t = useTranslations('ModelBlocks')
  const formats = getExportFormats()

  return (
    <div>
      <div className="-mx-1 overflow-x-auto px-1">
        <table className="w-full min-w-[640px] border-collapse text-[13.5px]">
          <thead>
            <tr>
              <th scope="col" className="border-b border-surface-300 px-3 py-1.5 text-left font-semibold text-surface-700 dark:border-white/20 dark:text-surface-300">
                {t('task')}
              </th>
              {formats.map((f) => (
                <th
                  key={f.key}
                  scope="col"
                  className="border-b border-surface-300 px-1 py-1.5 text-center text-[11.5px] font-medium text-surface-600 dark:border-white/20 dark:text-surface-400"
                >
                  {f.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {family.tasks.map((task) => (
              <tr key={task}>
                <th scope="row" className="border-b border-surface-200/70 px-3 py-1.5 text-left font-normal text-surface-700 dark:border-white/[0.07] dark:text-surface-300">
                  {getTaskMeta(task).label}
                </th>
                {formats.map((f) => (
                  <td key={f.key} className="border-b border-surface-200/70 px-1 py-1.5 text-center dark:border-white/[0.07]">
                    <Mark
                      state={family.export[task]?.[f.key] || 'blocked'}
                      label={`${getTaskMeta(task).label} to ${f.label}`}
                      reason={family.export_reasons?.[task]?.[f.key]}
                    />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

    </div>
  )
}

/* ── citation ───────────────────────────────────────────────────── */

/*
 * The BibTeX is rendered from the registry, where it is stored verbatim from
 * the upstream authors' own citation block, and it always ships with a link to
 * that block. A citation is an attribution: if we retype it and drop an author
 * or change a venue, readers credit the wrong people. Never author one by hand.
 */
export function Citation({ family }) {
  const t = useTranslations('ModelBlocks')
  const u = family.upstream
  if (!u?.bibtex) return null
  return (
    <div>
      <pre className="overflow-x-auto border border-surface-200 bg-surface-50/60 px-3 py-2.5 font-mono text-[12.5px] leading-[1.7] text-surface-800 dark:border-white/[0.09] dark:bg-white/[0.02] dark:text-surface-300">
        {u.bibtex}
      </pre>
      {u.bibtex_source_url && (
        <Note>{t.rich('citationSource', {
          link: (chunks) => <ExtLink href={u.bibtex_source_url}>{chunks}</ExtLink>,
          source: u.bibtex_source_url.replace('https://', ''),
        })}</Note>
      )}
    </div>
  )
}

/* ── provenance ─────────────────────────────────────────────────── */

/*
 * PyPI's mechanic: provenance is a section boundary with labelled rows, not a
 * badge and not a tinted card. The reader can see which facts are inherited
 * from upstream and which are ours.
 */
export function Provenance({ family, children }) {
  const t = useTranslations('ModelBlocks')
  const u = family.upstream
  return (
    <div>
      {/*
        Rendered by the component, not written by the page author, so it cannot
        be forgotten on any model page. What we publish is a summary; the
        repository the reader actually downloads from is the authority, and
        licenses can differ per checkpoint inside one family.
      */}
      <div className="mb-5 border-l-2 border-surface-300 pl-4 text-[13.5px] leading-relaxed text-surface-600 dark:border-white/20 dark:text-surface-400">
        <p className="mb-2">{t.rich('licensingCaution', {
          link: (chunks) => <ExtLink href={HF_BASE}>{chunks}</ExtLink>,
        })}</p>
        <p>{t('legalCaution')}</p>
      </div>

      <dl className="flex flex-col gap-y-1 text-[13.5px]">
        <Meta label={t('originalWork')}>{u.name}, {u.org}</Meta>
        <Meta label={t('upstreamLicense')}>{u.license}</Meta>
        <Meta label={t('upstreamSource')}><ExtLink href={u.code_url}>{u.code_url.replace('https://', '')}</ExtLink></Meta>
        <Meta label={t('libreyoloCode')}>MIT</Meta>
        {/*
          Only claim we republish weights when we actually host some. Several
          families are deliberately NOT mirrored because their licenses forbid
          it (research-only, gated, or non-commercial), and asserting otherwise
          on those pages is both false and legally careless.
        */}
        <Meta label={t('weights')}>
          {family.weights_hosted === false ? (
            t('weightsNotHosted', { license: u.license })
          ) : (
            t.rich('weightsHosted', {
              license: u.license,
              link: (chunks) => <ExtLink href={HF_BASE}>{chunks}</ExtLink>,
            })
          )}
        </Meta>
        {u.license_interpretation && (
          <Meta label={t('interpretation')}>{u.license_interpretation}</Meta>
        )}
      </dl>
      {children && <div className="mt-4 text-surface-600 dark:text-surface-400">{children}</div>}
    </div>
  )
}

/* ── faq and related ────────────────────────────────────────────── */

export function Faq({ items }) {
  if (!items?.length) return null
  return (
    <dl>
      {items.map((item) => (
        <Fragment key={item.q}>
          <dt className="mt-5 font-semibold text-surface-900 first:mt-0 dark:text-white">{item.q}</dt>
          <dd className="mt-1 max-w-[68ch] leading-relaxed text-surface-600 dark:text-surface-400">{item.a}</dd>
        </Fragment>
      ))}
    </dl>
  )
}

export function RelatedGrid({ items }) {
  if (!items?.length) return null
  return (
    <dl className="text-[14px]">
      {items.map((item) => (
        <div key={item.href} className="flex flex-col gap-x-3 border-b border-surface-200/70 py-2 last:border-0 sm:flex-row dark:border-white/[0.07]">
          <dt className="shrink-0 sm:w-52">
            <Link href={item.href} className="text-libre-700 underline-offset-2 hover:underline dark:text-libre-400">
              {item.label}
            </Link>
          </dt>
          <dd className="text-surface-500 dark:text-surface-500">{item.note}</dd>
        </div>
      ))}
    </dl>
  )
}
