'use client'

import { useEffect, useRef, useState } from 'react'
import { Cpu, Layers, Repeat, Timer } from 'lucide-react'
import RESULTS from './results-summary.json'
import './rf100vl-results.css'

// Verified campaign runs only. Every row here comes from a submission with
// valid_submission: true and all 100 datasets scored, published at
// huggingface.co/datasets/LibreYOLO/rf100-vl-results.
//
// map50 / map are the campaign's reported means over the 100 test splits.
// trainMin is the median per-dataset training wall time, taken from the
// per-dataset stats files in the same run directory.
//
// Models still running are deliberately absent rather than estimated: a
// number that cannot be traced to a run is not a result.
const BEST_MAP = Math.max(...RESULTS.map((r) => r.map))
const COMPLETE_RUNS = RESULTS.length * 100
const AGGREGATE_GPU_HOURS = Math.round(
  RESULTS.reduce((total, result) => total + result.trainHours, 0) / 10,
) * 10

// The only medal colouring anywhere: a hairline of metal on each plinth's top
// edge, the engraved rank numeral, and a matching dot beside the top three
// rows of the table. Everything else stays in the site's slate and cyan.
const METALS = [
  {
    edge: 'linear-gradient(90deg, #d97706, #fef08a 50%, #d97706)',
    numeral: 'text-amber-300',
    dot: '#f59e0b',
  },
  {
    edge: 'linear-gradient(90deg, #94a3b8, #f1f5f9 50%, #94a3b8)',
    numeral: 'text-slate-300',
    dot: '#94a3b8',
  },
  {
    edge: 'linear-gradient(90deg, #7c2d12, #fb923c 50%, #7c2d12)',
    numeral: 'text-orange-300',
    dot: '#ea580c',
  },
]

function Stat({ icon: Icon, value, label }) {
  return (
    <div className="flex items-center gap-3 rounded-xl border border-surface-200 dark:border-surface-800 bg-white/60 dark:bg-surface-900/60 px-4 py-3">
      <Icon className="w-4 h-4 text-libre-500 shrink-0" />
      <div>
        <div className="text-lg font-bold leading-none text-surface-800 dark:text-white">{value}</div>
        <div className="mt-1 text-[11px] uppercase tracking-wider text-surface-500">{label}</div>
      </div>
    </div>
  )
}

// The podium is drawn to scale: each plinth's height above the third-place
// baseline is proportional to its mAP50-95 lead, so the near-tie at the top
// and the drop to third are the structure itself, not a caption. Ranks come
// straight from RESULTS, which the build script sorts by mAP50-95, so the
// podium can never disagree with the leaderboard under it.
const PLINTH_BASE_PX = 96
const PX_PER_MAP = 2200

function Podium() {
  const top = RESULTS.slice(0, 3)
  const floor = top[2].map
  // Visual order 2-1-3; the blocks rise bronze first, gold last.
  const order = ['order-2', 'order-1', 'order-3']
  const rise = [260, 130, 0]

  return (
    <div className="mb-10">
      <ol aria-label="Top three models" className="grid grid-cols-3 items-end gap-2 sm:gap-4">
        {top.map((row, idx) => {
          const metal = METALS[idx]
          const next = RESULTS[idx + 1]
          const height = Math.round(PLINTH_BASE_PX + (row.map - floor) * PX_PER_MAP)
          return (
            <li key={row.id} className={`${order[idx]} flex min-w-0 flex-col justify-end text-center`}>
              <div className="rf-plinth-top px-1" style={{ '--d': `${rise[idx] + 340}ms` }}>
                <span className="flex flex-wrap items-center justify-center gap-x-1.5 gap-y-0.5">
                  <span className="text-xs sm:text-sm font-semibold text-surface-700 dark:text-surface-200">
                    {row.model}
                  </span>
                  <span className="rounded border border-surface-300 px-1 py-px font-mono text-[9px] sm:text-[10px] text-surface-500 dark:border-surface-700">
                    {row.size}
                  </span>
                </span>
                <span
                  className={`mt-1 block font-mono font-bold tabular-nums leading-none text-surface-900 dark:text-white ${
                    idx === 0 ? 'text-2xl sm:text-4xl' : 'text-xl sm:text-3xl'
                  }`}
                >
                  {row.map.toFixed(4)}
                </span>
                <span className="mt-1 mb-2.5 block text-[9px] sm:text-[10px] uppercase tracking-wider text-surface-400 dark:text-surface-500">
                  mAP<sup>50-95</sup>
                </span>
              </div>

              <div
                className="rf-plinth-block relative overflow-hidden rounded-t-lg bg-surface-900 dark:border dark:border-b-0 dark:border-white/10 dark:bg-white/[0.06]"
                style={{ height: `${height}px`, '--d': `${rise[idx]}ms` }}
              >
                <span
                  aria-hidden="true"
                  className="absolute inset-x-0 top-0 h-[3px]"
                  style={{ background: metal.edge }}
                />
                <div
                  className="rf-plinth-face flex flex-col items-center pt-4"
                  style={{ '--d': `${rise[idx] + 300}ms` }}
                >
                  <span className={`font-mono text-4xl sm:text-5xl font-bold leading-none ${metal.numeral}`}>
                    {idx + 1}
                  </span>
                  {next && (
                    <span className="mt-2 px-1 font-mono text-[9px] sm:text-[10px] tabular-nums leading-tight text-white/50">
                      +{(row.map - next.map).toFixed(4)}
                      <span className="hidden sm:inline">
                        {' '}vs {next.model}-{next.size}
                      </span>
                    </span>
                  )}
                </div>
              </div>
            </li>
          )
        })}
      </ol>
      <div className="border-t-2 border-surface-900 dark:border-white/20" />
      <p className="mt-2 text-center text-[11px] text-surface-400 dark:text-surface-500">
        The steps are to scale: each plinth&apos;s height above third place is its mAP<sup>50-95</sup> lead.
      </p>
    </div>
  )
}

export default function RF100VLResults() {
  const rootRef = useRef(null)
  const [inView, setInView] = useState(false)

  useEffect(() => {
    const el = rootRef.current
    if (!el) return undefined
    const io = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          setInView(true)
          io.disconnect()
        }
      },
      { threshold: 0.15 }
    )
    io.observe(el)
    return () => io.disconnect()
  }, [])

  return (
    <section
      ref={rootRef}
      aria-label="RF100-VL benchmark results"
      className="not-prose my-12"
      style={{ width: 'min(96vw, 1080px)', marginLeft: 'calc(50% - min(48vw, 540px))' }}
    >
      <div className="text-center mb-8">
        <p className="text-xs font-semibold uppercase tracking-[0.25em] text-surface-500">
          Verified runs
        </p>
        <h2 className="mt-2 text-2xl md:text-3xl font-bold text-surface-800 dark:text-white tracking-tight">
          {RESULTS.length} complete campaigns
        </h2>
        <p className="mt-2 text-sm text-surface-500 dark:text-surface-400 max-w-2xl mx-auto">
          Fine-tuned for 100 epochs on each of the 100 datasets, then scored on each
          dataset&apos;s test split with pycocotools at maxDets 500. The reported figure is
          the unweighted mean across the 100 scores.
        </p>
      </div>

      {/* stat strip */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-8">
        <Stat icon={Layers} value="100" label="datasets" />
        <Stat icon={Cpu} value={RESULTS.length} label="models complete" />
        <Stat icon={Repeat} value={COMPLETE_RUNS.toLocaleString('en-US')} label="runs, none skipped" />
        <Stat
          icon={Timer}
          value={`${AGGREGATE_GPU_HOURS.toLocaleString('en-US')} h`}
          label="aggregate GPU time"
        />
      </div>

      <div className={inView ? 'rf-bars-in' : undefined}>
        <Podium />
      </div>

      {/* leaderboard */}
      <div
        className={`rounded-2xl border border-surface-200 dark:border-surface-800 bg-white dark:bg-surface-900 overflow-hidden${
          inView ? ' rf-bars-in' : ''
        }`}
      >
        {/* header row */}
        <div className="hidden lg:grid grid-cols-[1.5rem_minmax(8rem,0.8fr)_4.5rem_4rem_4.5rem_minmax(10rem,1.1fr)_5.5rem_minmax(12rem,1.2fr)] items-center gap-x-3 px-5 py-3 border-b border-surface-200 dark:border-surface-800 text-[10px] font-bold uppercase tracking-widest text-surface-500">
          <span className="text-right">#</span>
          <span>Model</span>
          <span className="text-right">params</span>
          <span className="text-right">ok/100</span>
          <span className="text-right">
            mAP<sup>50</sup>
          </span>
          <span>
            mAP<sup>50-95</sup>
          </span>
          <span className="text-right">median train</span>
          <span>run ID</span>
        </div>

        {RESULTS.map((row, idx) => {
          const pct = Math.max((row.map / BEST_MAP) * 100, 1.5)
          const runId = row.runPath.split('/').pop()
          return (
            <div
              key={row.id}
              className="rf-row grid grid-cols-[1fr_auto] lg:grid-cols-[1.5rem_minmax(8rem,0.8fr)_4.5rem_4rem_4.5rem_minmax(10rem,1.1fr)_5.5rem_minmax(12rem,1.2fr)] items-center gap-x-3 px-5 py-3 border-b last:border-b-0 border-surface-100 dark:border-surface-800/70 transition-colors hover:bg-surface-50 dark:hover:bg-white/[0.03]"
              style={{ '--d': `${idx * 45}ms` }}
            >
              <span className="hidden lg:block text-right font-mono text-xs tabular-nums text-surface-400 dark:text-surface-500">
                {idx + 1}
              </span>
              <span className="min-w-0">
                <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
                  {idx < 3 && (
                    <span
                      aria-hidden="true"
                      className="inline-block h-1.5 w-1.5 shrink-0 rounded-full"
                      style={{ background: METALS[idx].dot }}
                    />
                  )}
                  <span className="font-semibold truncate text-surface-700 dark:text-surface-200">
                    {row.model}
                  </span>
                  <span className="text-[10px] font-mono text-surface-500 border border-surface-300 dark:border-surface-700 rounded px-1 py-px">
                    {row.size}
                  </span>
                </span>
                {/* mobile-only metric line */}
                <span className="lg:hidden block mt-1 text-[11px] text-surface-500 tabular-nums">
                  mAP<sup>50-95</sup> {row.map.toFixed(4)} · ok 100/100 · ~{row.trainMin.toFixed(1)} min
                </span>
                <a
                  href={`https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/tree/main/${row.runPath}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="lg:hidden block mt-1 text-[10px] font-mono text-libre-600 dark:text-libre-400 underline underline-offset-2 break-all"
                >
                  {runId}
                </a>
              </span>

              <span className="hidden lg:block text-right text-sm text-surface-500 tabular-nums">
                {row.paramsM.toFixed(2)}M
              </span>
              <span className="hidden lg:block text-right text-sm text-surface-500 tabular-nums">
                100
              </span>
              <span className="hidden lg:block text-right text-sm text-surface-500 tabular-nums">
                {row.map50.toFixed(4)}
              </span>

              <span className="col-span-2 lg:col-span-1 order-last lg:order-none mt-2 lg:mt-0">
                <span className="flex items-center gap-2">
                  <span className="flex-1 h-2 rounded-full bg-surface-200 dark:bg-surface-800 overflow-hidden">
                    <span
                      className="rf-bar-fill block h-full rounded-full bg-libre-500"
                      style={{ '--w': `${pct}%` }}
                    />
                  </span>
                  <span className="w-12 text-right text-sm font-bold tabular-nums text-surface-600 dark:text-surface-300">
                    {row.map.toFixed(4)}
                  </span>
                </span>
              </span>

              <span className="hidden lg:block text-right text-xs text-surface-500 tabular-nums">
                ~{row.trainMin.toFixed(1)} min
              </span>

              <a
                href={`https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/tree/main/${row.runPath}`}
                target="_blank"
                rel="noopener noreferrer"
                title={runId}
                className="hidden lg:block text-[10px] font-mono leading-tight text-libre-600 dark:text-libre-400 underline underline-offset-2 break-all"
              >
                {runId}
              </a>
            </div>
          )
        })}
      </div>

      <p className="mt-4 text-xs text-surface-400 dark:text-surface-500 leading-relaxed max-w-3xl mx-auto text-center">
        All {RESULTS.length} campaigns are complete: 100 of 100 datasets trained and scored,
        none skipped, all protocol-conformant. Every campaign uses seed 0 and RTX 5060 Ti
        workers; training precision follows the published recipe for each family. Raw artifacts,
        per-epoch metrics and manifests pinning the exact commits are published at{' '}
        <a
          href="https://huggingface.co/datasets/LibreYOLO/rf100-vl-results"
          target="_blank"
          rel="noopener noreferrer"
          className="text-libre-600 dark:text-libre-400 underline underline-offset-2"
        >
          LibreYOLO/rf100-vl-results
        </a>
        . The run IDs above link to the authoritative folders. More families will be added only
        after they finish a full 100.
      </p>

      <p className="mt-3 text-[11px] text-surface-400 dark:text-surface-500 leading-relaxed max-w-3xl mx-auto text-center">
        <strong className="font-semibold text-surface-500 dark:text-surface-400">License note:</strong>{' '}
        The YOLO-NAS run starts from Deci-provided weights covered by Deci&apos;s separate{' '}
        <a
          href="https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.YOLONAS.md"
          target="_blank"
          rel="noopener noreferrer"
          className="text-libre-600 dark:text-libre-400 underline underline-offset-2"
        >
          non-commercial YOLO-NAS license
        </a>
        . RF-DETR, EdgeCrafter and YOLOX use Apache-2.0 weights; YOLOv9 uses MIT weights.
      </p>
    </section>
  )
}
