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
            <Th>Checkpoint</Th>
            <Th align="right">Input</Th>
            <Th align="right">{bench.metric}</Th>
            <Th align="right">Params</Th>
            <Th align="right">PyTorch</Th>
            <Th align="right">TensorRT fp16</Th>
          </tr>
        </thead>
        <tbody>
          {bench.rows.map((row) => (
            <tr key={row.size}>
              <Td className="font-mono text-[12.5px] text-surface-900 dark:text-surface-200">{family.prefix}{row.size}</Td>
              <Td className="text-right">{row.imgsz}</Td>
              <Td className="text-right font-medium text-surface-900 dark:text-surface-200">{row.map.toFixed(1)}</Td>
              <Td className="text-right">{row.params_m ? `${row.params_m} M` : '—'}</Td>
              <Td className="text-right">{row.torch_ms ? `${row.torch_ms.toFixed(1)} ms` : '—'}</Td>
              <Td className="text-right">{row.trt_ms ? `${row.trt_ms.toFixed(1)} ms` : '—'}</Td>
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
            <Th align="right">Input</Th>
            <Th align="right">Params</Th>
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
                  <Td className="text-right tabular-nums">{row.params_m ? `${row.params_m} M` : '—'}</Td>
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
const STATES = {
  validated: {
    sentence: 'Validated. Numerically checked against the PyTorch model.',
    color: 'text-emerald-600 dark:text-emerald-400',
    icon: (
      <svg viewBox="0 0 16 16" className="h-[13px] w-[13px]" aria-hidden="true">
        <circle cx="8" cy="8" r="7" fill="currentColor" />
        <path d="M4.7 8.2l2.1 2.1 4.3-4.4" fill="none" stroke="#fff" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
  },
  experimental: {
    sentence: 'Experimental. Converts and runs, but the output is not parity checked.',
    color: 'text-amber-600 dark:text-amber-400',
    icon: (
      <svg viewBox="0 0 16 16" className="h-[13px] w-[13px]" aria-hidden="true">
        <circle cx="8" cy="8" r="6.2" fill="none" stroke="currentColor" strokeWidth="1.6" />
        <path d="M8 1.8a6.2 6.2 0 000 12.4z" fill="currentColor" />
      </svg>
    ),
  },
  blocked: {
    sentence: 'Not supported. The exporter refuses this combination before it runs.',
    color: 'text-surface-400 dark:text-surface-600',
    icon: (
      <svg viewBox="0 0 16 16" className="h-[13px] w-[13px]" aria-hidden="true">
        <path d="M3.5 8h9" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      </svg>
    ),
  },
}

function Mark({ state, label }) {
  const s = STATES[state] || STATES.blocked
  return (
    <span className={`inline-flex ${s.color}`} title={`${label}: ${s.sentence}`}>
      {s.icon}
      <span className="sr-only">{s.sentence}</span>
    </span>
  )
}

export function ExportMatrix({ family }) {
  const formats = getExportFormats()
  const used = [...new Set(family.tasks.flatMap((t) => formats.map((f) => family.export[t]?.[f.key] || 'blocked')))]

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
                    />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <dl className="mt-3 space-y-1 text-[13px]">
        {Object.entries(STATES)
          .filter(([key]) => used.includes(key))
          .map(([key, s]) => (
            <div key={key} className="flex items-baseline gap-2">
              <dt className={`shrink-0 ${s.color}`}>{s.icon}</dt>
              <dd className="text-surface-500 dark:text-surface-500">{s.sentence}</dd>
            </div>
          ))}
      </dl>

      {family.export_notes && (
        <dl className="mt-3 space-y-0.5 text-[13px]">
          {Object.entries(family.export_notes).map(([key, note]) => {
            const label = getExportFormats().find((f) => f.key === key)?.label || key
            return (
              <div key={key} className="flex flex-col gap-x-2 sm:flex-row">
                <dt className="shrink-0 text-surface-500 dark:text-surface-500 sm:w-24">{label}</dt>
                <dd className="text-surface-500 dark:text-surface-500">{note}</dd>
              </div>
            )
          })}
        </dl>
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
      <dl className="flex flex-col gap-y-1 text-[13.5px]">
        <Meta label="Original work">{u.name}, {u.org}</Meta>
        <Meta label="Upstream license">{u.license}</Meta>
        <Meta label="Upstream source"><ExtLink href={u.code_url}>{u.code_url.replace('https://', '')}</ExtLink></Meta>
        <Meta label="LibreYOLO code">MIT</Meta>
        <Meta label="Weights">{u.license}, republished at <ExtLink href={HF_BASE}>huggingface.co/LibreYOLO</ExtLink></Meta>
      </dl>
      <div className="mt-4 text-surface-600 dark:text-surface-400">{children}</div>
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
