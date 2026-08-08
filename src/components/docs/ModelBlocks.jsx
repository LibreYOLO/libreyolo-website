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
        <Meta label="Tasks">{taskNames}</Meta>
        <Meta label="Sizes">{family.sizes_label}</Meta>
        <Meta label="Install">
          <code className="font-mono text-[12.5px]">pip install &quot;libreyolo[{family.extra}]&quot;</code>
        </Meta>
        <Meta label="Support tier">
          {tier?.label}, since v{family.added_in}. {tier?.blurb}
        </Meta>
        <Meta label="Upstream">
          {u.name} by {u.org}, {u.license}. <ExtLink href={u.paper_url}>Paper</ExtLink>
          {', '}
          <ExtLink href={u.code_url}>source</ExtLink>
        </Meta>
        <Meta label="Licenses">
          Code MIT, weights {u.license}. <Link href="#licensing" className="text-libre-700 underline-offset-2 hover:underline dark:text-libre-400">Commercial use</Link>
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

/* ── tasks ──────────────────────────────────────────────────────── */

export function TaskSupport({ family }) {
  return (
    <Table>
      <thead>
        <tr>
          <Th>Task</Th>
          <Th>Weights</Th>
          <Th>Training</Th>
          <Th>Export formats</Th>
          <Th>Since</Th>
        </tr>
      </thead>
      <tbody>
        {family.tasks.map((task) => {
          const meta = getTaskMeta(task)
          const count = family.checkpoints.filter((c) => c.task === task).length
          const validated = Object.values(family.export[task] || {}).filter((v) => v === 'validated').length
          return (
            <tr key={task}>
              <Td>
                <Link href={`/docs/tasks/${meta.slug}`} className="text-libre-700 underline-offset-2 hover:underline dark:text-libre-400">
                  {meta.label}
                </Link>
              </Td>
              <Td className="tabular-nums">{count}</Td>
              <Td>{family.capabilities.train ? 'Supported' : 'Inference only'}</Td>
              <Td className="tabular-nums">{validated} validated</Td>
              <Td className="tabular-nums">v{family.task_added_in?.[task] || family.added_in}</Td>
            </tr>
          )
        })}
      </tbody>
    </Table>
  )
}

/* ── benchmarks ─────────────────────────────────────────────────── */

export function BenchmarkTable({ family, task = 'detect' }) {
  const bench = family.benchmarks?.[task]
  if (!bench) return null

  return (
    <div>
      <Table className="tabular-nums">
        <thead>
          <tr>
            {/* Units live in the column head, never repeated in every cell. */}
            <Th>Checkpoint</Th>
            <Th align="right">Input (px)</Th>
            <Th align="right">{bench.metric}</Th>
            <Th align="right">Params (M)</Th>
            <Th align="right">PyTorch (ms)</Th>
            <Th align="right">TensorRT fp16 (ms)</Th>
          </tr>
        </thead>
        <tbody>
          {bench.rows.map((row) => (
            <tr key={row.size}>
              <Td className="font-mono text-[12.5px] text-surface-900 dark:text-surface-200">{family.prefix}{row.size}</Td>
              <Td className="text-right">{row.imgsz}</Td>
              <Td className="text-right font-medium text-surface-900 dark:text-surface-200">{row.map.toFixed(1)}</Td>
              <Td className="text-right">{row.params_m ?? ''}</Td>
              <Td className="text-right">{row.torch_ms ? row.torch_ms.toFixed(1) : ''}</Td>
              <Td className="text-right">{row.trt_ms ? row.trt_ms.toFixed(1) : ''}</Td>
            </tr>
          ))}
        </tbody>
      </Table>
      <Note>
        {bench.dataset}. {bench.hardware}, batch 1, median end-to-end latency including pre and
        postprocessing. Published on <ExtLink href={bench.source_url}>Vision Analysis</ExtLink>, where
        the full runs and hardware details are recorded.
      </Note>
    </div>
  )
}

export function VaEmbed({ family }) {
  const src = family.va_embed?.scatter
  if (!src) return null
  return (
    <div className="my-6">
      <div className="relative w-full" style={{ paddingTop: '62.5%' }}>
        <iframe
          src={src}
          title={`${family.display} accuracy versus latency`}
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
  const grouped = family.tasks
    .map((task) => ({ task, rows: family.checkpoints.filter((c) => c.task === task) }))
    .filter((g) => g.rows.length)

  return (
    <div>
      <Table>
        <thead>
          <tr>
            <Th>File</Th>
            <Th align="right">Input (px)</Th>
            <Th align="right">Params (M)</Th>
            <Th>Trained on</Th>
            <Th>Weights license</Th>
          </tr>
        </thead>
        <tbody>
          {grouped.map(({ task, rows }) => (
            <Fragment key={task}>
              <tr>
                <td colSpan={5} className="border-b border-surface-200/70 px-3 pb-1 pt-4 text-[11px] font-semibold uppercase tracking-wider text-surface-500 dark:border-white/[0.07] dark:text-surface-500">
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
                  {/* Empty means not recorded in the registry. That is the one
                      missing-data convention, used in every docs table. */}
                  <Td className="text-right tabular-nums">{row.params_m ?? ''}</Td>
                  <Td>{row.data}</Td>
                  <Td>{row.license}</Td>
                </tr>
              ))}
            </Fragment>
          ))}
        </tbody>
      </Table>
      <Note>
        Every file above exists in the <ExtLink href={HF_BASE}>LibreYOLO org</ExtLink> today and
        downloads on first use. Naming a file that is not listed here will fail rather than fall back
        to a different checkpoint.
      </Note>
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
  const supported = SUPPORTED_STATES.has(state)
  if (!supported) {
    return (
      <span className="sr-only">{`${label}: not supported`}</span>
    )
  }
  return (
    <span
      className="inline-flex text-emerald-600 dark:text-emerald-400"
      title={reason ? `${label}: supported. ${reason}` : `${label}: supported`}
    >
      {TICK}
      <span className="sr-only">{`${label}: supported. ${reason || ''}`}</span>
    </span>
  )
}

export function ExportMatrix({ family }) {
  const formats = getExportFormats()

  return (
    <div>
      <div className="-mx-1 overflow-x-auto px-1">
        <table className="w-full min-w-[640px] border-collapse text-[13.5px]">
          <thead>
            <tr>
              <th scope="col" className="border-b border-surface-300 px-3 py-1.5 text-left font-semibold text-surface-700 dark:border-white/20 dark:text-surface-300">
                Task
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

      <Note>
        A tick means the export runs and is supported. An empty cell means the
        exporter refuses that combination before it starts.
      </Note>
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
  const u = family.upstream
  if (!u?.bibtex) return null
  return (
    <div>
      <pre className="overflow-x-auto border border-surface-200 bg-surface-50/60 px-3 py-2.5 font-mono text-[12.5px] leading-[1.7] text-surface-800 dark:border-white/[0.09] dark:bg-white/[0.02] dark:text-surface-300">
        {u.bibtex}
      </pre>
      {u.bibtex_source_url && (
        <Note>
          Copied from the authors' citation block at{' '}
          <ExtLink href={u.bibtex_source_url}>{u.bibtex_source_url.replace('https://', '')}</ExtLink>.
        </Note>
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
        <p className="mb-2">
          Check the license on the Hugging Face repository of the specific weights you
          download. Every checkpoint in the{' '}
          <ExtLink href={HF_BASE}>LibreYOLO org</ExtLink> carries one, and they are not
          always the same across a family. That repository is the authoritative source;
          the summary below describes what applied when this page was last verified.
        </p>
        <p>
          This is a description of the licenses involved, not legal advice. If the
          answer matters commercially, read the licenses yourself and take your own
          counsel.
        </p>
      </div>

      <dl className="flex flex-col gap-y-1 text-[13.5px]">
        <Meta label="Original work">{u.name}, {u.org}</Meta>
        <Meta label="Upstream license">{u.license}</Meta>
        <Meta label="Upstream source"><ExtLink href={u.code_url}>{u.code_url.replace('https://', '')}</ExtLink></Meta>
        <Meta label="LibreYOLO code">MIT</Meta>
        <Meta label="Weights">{u.license}, republished at <ExtLink href={HF_BASE}>huggingface.co/LibreYOLO</ExtLink></Meta>
        {u.license_interpretation && (
          <Meta label="Interpretation">{u.license_interpretation}</Meta>
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
