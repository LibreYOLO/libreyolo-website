/*
 * Server-rendered blocks for a model family page.
 *
 * Everything here reads the generated registry (src/data/docs/registry.json).
 * A page author never types a support matrix, a checkpoint name or a benchmark
 * number into prose; they place the block and the pipeline fills it. That is
 * what keeps 80 model pages from drifting out of date one release later.
 */

import { Fragment } from 'react'
import Link from 'next/link'
import { ExternalLink, ShieldCheck, TriangleAlert, ArrowUpRight } from 'lucide-react'
import { getTaskMeta, getTierMeta, getExportFormats } from '@/lib/docs'

const HF_BASE = 'https://huggingface.co/LibreYOLO'

/* ── small primitives ───────────────────────────────────────────── */

const TONES = {
  emerald: 'border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300',
  libre: 'border-libre-500/30 bg-libre-500/10 text-libre-700 dark:text-libre-300',
  amber: 'border-amber-500/30 bg-amber-500/10 text-amber-700 dark:text-amber-300',
  neutral: 'border-surface-300/60 bg-surface-100 text-surface-600 dark:border-white/10 dark:bg-white/[0.04] dark:text-surface-400',
}

export function Pill({ tone = 'neutral', className = '', children }) {
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-semibold ${TONES[tone] || TONES.neutral} ${className}`}>
      {children}
    </span>
  )
}

export function SectionTitle({ id, children, kicker }) {
  return (
    <div id={id} className="scroll-mt-28 mt-16 mb-5 first:mt-0">
      {kicker && (
        <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-wider text-libre-600 dark:text-libre-400">{kicker}</p>
      )}
      <h2 className="text-2xl font-bold tracking-tight text-surface-900 dark:text-white">
        <a href={`#${id}`} className="group">
          {children}
          <span className="ml-2 text-surface-300 opacity-0 transition-opacity group-hover:opacity-100 dark:text-surface-700">#</span>
        </a>
      </h2>
    </div>
  )
}

function SourceLine({ children }) {
  return <p className="mt-2 text-xs text-surface-500 dark:text-surface-500">{children}</p>
}

/* ── header ─────────────────────────────────────────────────────── */

export function ModelHeader({ doc, family }) {
  const tier = getTierMeta(family.tier)
  const upstream = family.upstream

  return (
    <header className="mb-10">
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <Pill tone={tier?.tone}>
          <span className="h-1.5 w-1.5 rounded-full bg-current" />
          {tier?.label}
        </Pill>
        <Pill tone="neutral">MIT code</Pill>
        <Pill tone="neutral">{upstream.license} weights</Pill>
        <Pill tone="neutral">Added in v{family.added_in}</Pill>
      </div>

      <h1 className="text-4xl font-bold tracking-tight text-surface-900 dark:text-white lg:text-5xl">
        {doc.title}
      </h1>
      <p className="mt-4 max-w-2xl text-lg leading-relaxed text-surface-600 dark:text-surface-400">
        {doc.lead}
      </p>

      <div className="mt-5 flex flex-wrap items-center gap-x-5 gap-y-2 text-sm">
        <a href={upstream.paper_url} target="_blank" rel="noopener noreferrer"
          className="inline-flex items-center gap-1.5 font-medium text-libre-600 hover:underline dark:text-libre-400">
          Paper <ExternalLink className="h-3.5 w-3.5" />
        </a>
        <a href={upstream.code_url} target="_blank" rel="noopener noreferrer"
          className="inline-flex items-center gap-1.5 font-medium text-libre-600 hover:underline dark:text-libre-400">
          Original code <ExternalLink className="h-3.5 w-3.5" />
        </a>
        <a href={`${HF_BASE}?search=${family.prefix}`} target="_blank" rel="noopener noreferrer"
          className="inline-flex items-center gap-1.5 font-medium text-libre-600 hover:underline dark:text-libre-400">
          Weights on Hugging Face <ExternalLink className="h-3.5 w-3.5" />
        </a>
      </div>

      <dl className="mt-8 grid grid-cols-2 gap-px overflow-hidden rounded-xl border border-surface-200 bg-surface-200 dark:border-white/[0.08] dark:bg-white/[0.08] sm:grid-cols-4">
        <Fact label="Tasks" value={family.tasks.length} sub={family.tasks.map((t) => getTaskMeta(t).label).join(', ')} />
        <Fact label="Checkpoints" value={family.checkpoints.length} sub="published on Hugging Face" />
        <Fact label="Trainable" value={family.capabilities.train ? 'Yes' : 'No'} sub={family.capabilities.lora ? 'full or LoRA' : 'full fine-tune'} />
        <Fact label="Install" value={`libreyolo[${family.extra}]`} sub="extra required" mono />
      </dl>
    </header>
  )
}

function Fact({ label, value, sub, mono }) {
  return (
    <div className="bg-white p-4 dark:bg-surface-950">
      <dt className="text-[11px] font-semibold uppercase tracking-wider text-surface-500 dark:text-surface-500">{label}</dt>
      <dd className={`mt-1 font-bold text-surface-900 dark:text-white ${mono ? 'font-mono text-[13px] break-all' : 'text-xl'}`}>{value}</dd>
      {sub && <dd className="mt-0.5 text-xs leading-snug text-surface-500 dark:text-surface-500">{sub}</dd>}
    </div>
  )
}

/* ── hero media ─────────────────────────────────────────────────── */

export function HeroMedia({ media }) {
  if (!media) return null
  return (
    <figure className="mb-12 overflow-hidden rounded-2xl border border-surface-200 bg-surface-100 dark:border-white/[0.08] dark:bg-surface-900">
      <video
        className="block aspect-video w-full object-cover"
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
        <figcaption className="border-t border-surface-200 px-4 py-2.5 text-xs text-surface-500 dark:border-white/[0.06] dark:text-surface-500">
          {media.caption}
        </figcaption>
      )}
    </figure>
  )
}

/* ── task and mode support ──────────────────────────────────────── */

export function TaskSupport({ family }) {
  return (
    <div className="overflow-x-auto rounded-xl border border-surface-200 dark:border-white/[0.08]">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-surface-200 bg-surface-50 text-left dark:border-white/[0.08] dark:bg-white/[0.02]">
            <th className="px-4 py-3 font-semibold text-surface-700 dark:text-surface-300">Task</th>
            <th className="px-4 py-3 font-semibold text-surface-700 dark:text-surface-300">Checkpoints</th>
            <th className="px-4 py-3 font-semibold text-surface-700 dark:text-surface-300">Train</th>
            <th className="px-4 py-3 font-semibold text-surface-700 dark:text-surface-300">Export</th>
            <th className="px-4 py-3 font-semibold text-surface-700 dark:text-surface-300">Since</th>
          </tr>
        </thead>
        <tbody>
          {family.tasks.map((task) => {
            const meta = getTaskMeta(task)
            const count = family.checkpoints.filter((c) => c.task === task).length
            const validated = Object.values(family.export[task] || {}).filter((v) => v === 'validated').length
            return (
              <tr key={task} className="border-b border-surface-100 last:border-0 dark:border-white/[0.04]">
                <td className="px-4 py-3">
                  <Link href={`/docs/tasks/${meta.slug}`} className="font-medium text-libre-600 hover:underline dark:text-libre-400">
                    {meta.label}
                  </Link>
                </td>
                <td className="px-4 py-3 tabular-nums text-surface-600 dark:text-surface-400">{count}</td>
                <td className="px-4 py-3 text-surface-600 dark:text-surface-400">
                  {family.capabilities.train ? 'Yes' : 'No'}
                </td>
                <td className="px-4 py-3 tabular-nums text-surface-600 dark:text-surface-400">
                  {validated} validated {validated === 1 ? 'format' : 'formats'}
                </td>
                <td className="px-4 py-3 tabular-nums text-surface-500 dark:text-surface-500">
                  v{family.task_added_in?.[task] || family.added_in}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

/* ── benchmarks ─────────────────────────────────────────────────── */

export function BenchmarkTable({ family, task = 'detect' }) {
  const bench = family.benchmarks?.[task]
  if (!bench) return null
  const best = Math.max(...bench.rows.map((r) => r.map))

  return (
    <div>
      <div className="overflow-x-auto rounded-xl border border-surface-200 dark:border-white/[0.08]">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-surface-200 bg-surface-50 text-left dark:border-white/[0.08] dark:bg-white/[0.02]">
              <th className="px-4 py-3 font-semibold text-surface-700 dark:text-surface-300">Model</th>
              <th className="px-4 py-3 font-semibold text-surface-700 dark:text-surface-300">Input</th>
              <th className="px-4 py-3 font-semibold text-surface-700 dark:text-surface-300">{bench.metric}</th>
              <th className="px-4 py-3 font-semibold text-surface-700 dark:text-surface-300">Params</th>
              <th className="px-4 py-3 font-semibold text-surface-700 dark:text-surface-300">PyTorch</th>
              <th className="px-4 py-3 font-semibold text-surface-700 dark:text-surface-300">TensorRT fp16</th>
            </tr>
          </thead>
          <tbody className="tabular-nums">
            {bench.rows.map((row) => (
              <tr key={row.size} className="border-b border-surface-100 last:border-0 hover:bg-surface-50 dark:border-white/[0.04] dark:hover:bg-white/[0.02]">
                <td className="px-4 py-3 font-mono text-[13px] font-medium text-surface-900 dark:text-white">
                  {family.prefix}{row.size}
                </td>
                <td className="px-4 py-3 text-surface-600 dark:text-surface-400">{row.imgsz}</td>
                <td className={`px-4 py-3 font-semibold ${row.map === best ? 'text-libre-600 dark:text-libre-400' : 'text-surface-700 dark:text-surface-300'}`}>
                  {row.map.toFixed(1)}
                </td>
                <td className="px-4 py-3 text-surface-600 dark:text-surface-400">{row.params_m ? `${row.params_m}M` : '-'}</td>
                <td className="px-4 py-3 text-surface-600 dark:text-surface-400">{row.torch_ms ? `${row.torch_ms.toFixed(1)} ms` : '-'}</td>
                <td className="px-4 py-3 text-surface-600 dark:text-surface-400">
                  {row.trt_ms ? `${row.trt_ms.toFixed(1)} ms` : '-'}
                  {row.trt_fps && <span className="ml-1.5 text-xs text-surface-400 dark:text-surface-600">{row.trt_fps} FPS</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <SourceLine>
        {bench.dataset}, {bench.hardware}, batch 1, median latency end to end.
        Measured by the LibreYOLO benchmark harness and published on{' '}
        <a href={bench.source_url} target="_blank" rel="noopener noreferrer" className="text-libre-600 hover:underline dark:text-libre-400">
          Vision Analysis
        </a>.
      </SourceLine>
    </div>
  )
}

export function VaEmbed({ family }) {
  const src = family.va_embed?.scatter
  if (!src) return null
  return (
    <div className="mt-6">
      <div className="relative w-full overflow-hidden rounded-xl border border-surface-200 dark:border-white/[0.08]" style={{ paddingTop: '62.5%' }}>
        <iframe
          src={src}
          title={`${family.display} accuracy versus latency`}
          loading="lazy"
          className="absolute inset-0 h-full w-full"
          style={{ border: 0 }}
        />
      </div>
      <SourceLine>Live chart from visionanalysis.org. Drag to compare against every other benchmarked model.</SourceLine>
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
      <div className="overflow-x-auto rounded-xl border border-surface-200 dark:border-white/[0.08]">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-surface-200 bg-surface-50 text-left dark:border-white/[0.08] dark:bg-white/[0.02]">
              <th className="px-4 py-3 font-semibold text-surface-700 dark:text-surface-300">Checkpoint</th>
              <th className="px-4 py-3 font-semibold text-surface-700 dark:text-surface-300">Input</th>
              <th className="px-4 py-3 font-semibold text-surface-700 dark:text-surface-300">Params</th>
              <th className="px-4 py-3 font-semibold text-surface-700 dark:text-surface-300">Trained on</th>
              <th className="px-4 py-3 font-semibold text-surface-700 dark:text-surface-300">Weights license</th>
            </tr>
          </thead>
          <tbody>
            {grouped.map(({ task, rows }) => (
              <Fragment key={task}>
                <tr className="border-b border-surface-200 bg-surface-50/60 dark:border-white/[0.06] dark:bg-white/[0.015]">
                  <td colSpan={5} className="px-4 py-2 text-[11px] font-semibold uppercase tracking-wider text-surface-500 dark:text-surface-500">
                    {getTaskMeta(task).label}
                  </td>
                </tr>
                {rows.map((row) => (
                  <tr key={row.name} className="border-b border-surface-100 last:border-0 hover:bg-surface-50 dark:border-white/[0.04] dark:hover:bg-white/[0.02]">
                    <td className="px-4 py-3">
                      <a
                        href={`${HF_BASE}/${row.name.replace(/\.pt$/, '')}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1 font-mono text-[13px] font-medium text-libre-600 hover:underline dark:text-libre-400"
                      >
                        {row.name}
                        <ArrowUpRight className="h-3 w-3 shrink-0" />
                      </a>
                    </td>
                    <td className="px-4 py-3 tabular-nums text-surface-600 dark:text-surface-400">{row.imgsz}</td>
                    <td className="px-4 py-3 tabular-nums text-surface-600 dark:text-surface-400">{row.params_m ? `${row.params_m}M` : '-'}</td>
                    <td className="px-4 py-3 text-surface-600 dark:text-surface-400">{row.data}</td>
                    <td className="px-4 py-3 text-surface-600 dark:text-surface-400">{row.license}</td>
                  </tr>
                ))}
              </Fragment>
            ))}
          </tbody>
        </table>
      </div>
      <SourceLine>
        Every row is a repository that exists in the{' '}
        <a href={HF_BASE} target="_blank" rel="noopener noreferrer" className="text-libre-600 hover:underline dark:text-libre-400">LibreYOLO org</a>
        {' '}today. Weights download automatically the first time you name one.
      </SourceLine>
    </div>
  )
}

/* ── export matrix ──────────────────────────────────────────────── */

const CELL = {
  validated: { label: 'Validated', cls: 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-300', mark: '✓' },
  experimental: { label: 'Experimental', cls: 'bg-amber-500/15 text-amber-700 dark:text-amber-300', mark: 'exp' },
  blocked: { label: 'Not supported', cls: 'text-surface-300 dark:text-surface-700', mark: '·' },
}

export function ExportMatrix({ family }) {
  const formats = getExportFormats()
  return (
    <div>
      <div className="overflow-x-auto rounded-xl border border-surface-200 dark:border-white/[0.08]">
        <table className="w-full min-w-[720px] text-sm">
          <thead>
            <tr className="border-b border-surface-200 bg-surface-50 dark:border-white/[0.08] dark:bg-white/[0.02]">
              <th className="sticky left-0 z-10 bg-surface-50 px-4 py-3 text-left font-semibold text-surface-700 dark:bg-surface-950 dark:text-surface-300">Task</th>
              {formats.map((f) => (
                <th key={f.key} className="px-2 py-3 text-center text-[11px] font-semibold text-surface-600 dark:text-surface-400">{f.label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {family.tasks.map((task) => (
              <tr key={task} className="border-b border-surface-100 last:border-0 dark:border-white/[0.04]">
                <td className="sticky left-0 z-10 bg-white px-4 py-3 font-medium text-surface-800 dark:bg-surface-950 dark:text-surface-200">
                  {getTaskMeta(task).label}
                </td>
                {formats.map((f) => {
                  const state = family.export[task]?.[f.key] || 'blocked'
                  const cell = CELL[state]
                  return (
                    <td key={f.key} className="px-2 py-3 text-center">
                      <span
                        title={`${getTaskMeta(task).label} to ${f.label}: ${cell.label}`}
                        className={`inline-flex min-w-[2.25rem] items-center justify-center rounded-md px-1.5 py-1 text-[11px] font-semibold ${cell.cls}`}
                      >
                        {cell.mark}
                      </span>
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-4 text-xs text-surface-500 dark:text-surface-500">
        <span className="inline-flex items-center gap-1.5"><span className="rounded bg-emerald-500/15 px-1.5 py-0.5 font-semibold text-emerald-700 dark:text-emerald-300">✓</span> parity validated against PyTorch</span>
        <span className="inline-flex items-center gap-1.5"><span className="rounded bg-amber-500/15 px-1.5 py-0.5 font-semibold text-amber-700 dark:text-amber-300">exp</span> converts, no parity guarantee</span>
        <span className="inline-flex items-center gap-1.5"><span className="text-surface-300 dark:text-surface-700">·</span> blocked in preflight</span>
      </div>
      {family.export_notes && (
        <ul className="mt-3 space-y-1">
          {Object.entries(family.export_notes).map(([key, note]) => (
            <li key={key} className="text-xs text-surface-500 dark:text-surface-500">
              <span className="font-semibold text-surface-600 dark:text-surface-400">{key}:</span> {note}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

/* ── provenance ─────────────────────────────────────────────────── */

export function Provenance({ family, children }) {
  const u = family.upstream
  return (
    <div className="rounded-xl border border-surface-200 bg-surface-50/60 p-5 dark:border-white/[0.08] dark:bg-white/[0.02]">
      <div className="flex items-start gap-3">
        <ShieldCheck className="mt-0.5 h-5 w-5 shrink-0 text-emerald-600 dark:text-emerald-400" />
        <div className="min-w-0">
          <p className="font-semibold text-surface-900 dark:text-white">Where this implementation comes from</p>
          <dl className="mt-3 space-y-1.5 text-sm">
            <Row term="Original work">{u.name} by {u.org}</Row>
            <Row term="Upstream license">{u.license}</Row>
            <Row term="Upstream code">
              <a href={u.code_url} target="_blank" rel="noopener noreferrer" className="text-libre-600 hover:underline dark:text-libre-400">{u.code_url.replace('https://', '')}</a>
            </Row>
            <Row term="LibreYOLO code">MIT</Row>
          </dl>
          <div className="mt-3 text-sm text-surface-600 dark:text-surface-400">{children}</div>
        </div>
      </div>
    </div>
  )
}

function Row({ term, children }) {
  return (
    <div className="flex flex-wrap gap-x-2">
      <dt className="w-36 shrink-0 text-surface-500 dark:text-surface-500">{term}</dt>
      <dd className="min-w-0 text-surface-700 dark:text-surface-300">{children}</dd>
    </div>
  )
}

export function LicenseAnswer({ question, children, warn = false }) {
  return (
    <div className="mt-5">
      <h3 id={question.toLowerCase().replace(/[^\w\s-]/g, '').trim().replace(/\s+/g, '-')} className="scroll-mt-28 flex items-start gap-2 text-base font-semibold text-surface-900 dark:text-white">
        {warn && <TriangleAlert className="mt-0.5 h-4 w-4 shrink-0 text-amber-500" />}
        {question}
      </h3>
      <div className="mt-1.5 text-surface-600 dark:text-surface-400">{children}</div>
    </div>
  )
}

/* ── faq + related ──────────────────────────────────────────────── */

export function Faq({ items }) {
  if (!items?.length) return null
  return (
    <div className="divide-y divide-surface-200 rounded-xl border border-surface-200 dark:divide-white/[0.06] dark:border-white/[0.08]">
      {items.map((item) => (
        <details key={item.q} className="group p-5 [&_summary::-webkit-details-marker]:hidden">
          <summary className="flex cursor-pointer items-center justify-between gap-4 font-medium text-surface-900 dark:text-white">
            {item.q}
            <span className="shrink-0 text-surface-400 transition-transform group-open:rotate-45">+</span>
          </summary>
          <p className="mt-3 text-sm leading-relaxed text-surface-600 dark:text-surface-400">{item.a}</p>
        </details>
      ))}
    </div>
  )
}

export function RelatedGrid({ items }) {
  if (!items?.length) return null
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {items.map((item) => (
        <Link
          key={item.href}
          href={item.href}
          className="group rounded-xl border border-surface-200 p-4 transition-colors hover:border-libre-500/40 hover:bg-libre-500/[0.03] dark:border-white/[0.08] dark:hover:border-libre-500/40"
        >
          <p className="flex items-center justify-between font-semibold text-surface-900 dark:text-white">
            {item.label}
            <ArrowUpRight className="h-4 w-4 text-surface-300 transition-colors group-hover:text-libre-500 dark:text-surface-700" />
          </p>
          <p className="mt-1 text-sm leading-snug text-surface-500 dark:text-surface-500">{item.note}</p>
        </Link>
      ))}
    </div>
  )
}
