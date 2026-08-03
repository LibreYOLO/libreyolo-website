'use client'

import { useEffect, useRef, useState } from 'react'
import { Cpu, Layers, Repeat, Timer } from 'lucide-react'
import './rf100vl-results.css'

// Verified campaign runs only. Every row here comes from a submission with
// valid_submission: true and all 100 datasets scored, published at
// huggingface.co/datasets/LibreYOLO/rf100-vl-results.
//
// map50 / map are the campaign's reported means over the 100 test splits.
// trainMin is the median per-dataset training wall time, taken from the
// per-dataset stats files in the same run directory.
//
// Families still running are deliberately absent rather than estimated: a
// number that cannot be traced to a run is not a result.
const RESULTS = [
  {
    model: 'YOLOv9',
    size: 'S',
    weights: 'LibreYOLO9s.pt',
    ok: 100,
    map50: 0.814,
    map: 0.559,
    trainMin: 32.5,
    paramsM: 7.2,
  },
  {
    model: 'YOLOv9',
    size: 'T',
    weights: 'LibreYOLO9t.pt',
    ok: 100,
    map50: 0.796,
    map: 0.540,
    trainMin: 32.9,
    paramsM: 2.0,
  },
]

const BEST_MAP = Math.max(...RESULTS.map((r) => r.map))

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
          The first two complete campaigns
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
        <Stat icon={Cpu} value="2" label="models complete" />
        <Stat icon={Repeat} value="200" label="runs, none skipped" />
        <Stat icon={Timer} value="180 h" label="GPU time" />
      </div>

      {/* leaderboard */}
      <div
        className={`rounded-2xl border border-surface-200 dark:border-surface-800 bg-white dark:bg-surface-900 overflow-hidden${
          inView ? ' rf-bars-in' : ''
        }`}
      >
        {/* header row */}
        <div className="hidden md:grid grid-cols-[minmax(11rem,1.1fr)_5rem_4.5rem_4.5rem_minmax(12rem,1.4fr)_6rem] items-center gap-x-4 px-5 py-3 border-b border-surface-200 dark:border-surface-800 text-[10px] font-bold uppercase tracking-widest text-surface-500">
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
        </div>

        {RESULTS.map((row, idx) => {
          const pct = Math.max((row.map / BEST_MAP) * 100, 1.5)
          return (
            <div
              key={row.weights}
              className="rf-row grid grid-cols-[1fr_auto] md:grid-cols-[minmax(11rem,1.1fr)_5rem_4.5rem_4.5rem_minmax(12rem,1.4fr)_6rem] items-center gap-x-4 px-5 py-3 border-b last:border-b-0 border-surface-100 dark:border-surface-800/70"
              style={{ '--d': `${idx * 45}ms` }}
            >
              <span className="min-w-0">
                <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
                  <span className="font-semibold truncate text-surface-700 dark:text-surface-200">
                    {row.model}
                  </span>
                  <span className="text-[10px] font-mono text-surface-500 border border-surface-300 dark:border-surface-700 rounded px-1 py-px">
                    {row.size}
                  </span>
                </span>
                {/* mobile-only metric line */}
                <span className="md:hidden block mt-1 text-[11px] text-surface-500 tabular-nums">
                  mAP<sup>50-95</sup> {row.map.toFixed(3)} · ok {row.ok}/100 · ~{row.trainMin} min
                </span>
              </span>

              <span className="hidden md:block text-right text-sm text-surface-500 tabular-nums">
                {row.paramsM.toFixed(1)}M
              </span>
              <span className="hidden md:block text-right text-sm text-surface-500 tabular-nums">
                {row.ok}
              </span>
              <span className="hidden md:block text-right text-sm text-surface-500 tabular-nums">
                {row.map50.toFixed(3)}
              </span>

              <span className="col-span-2 md:col-span-1 order-last md:order-none mt-2 md:mt-0">
                <span className="flex items-center gap-2">
                  <span className="flex-1 h-2 rounded-full bg-surface-200 dark:bg-surface-800 overflow-hidden">
                    <span
                      className="rf-bar-fill block h-full rounded-full bg-libre-500"
                      style={{ '--w': `${pct}%` }}
                    />
                  </span>
                  <span className="w-12 text-right text-sm font-bold tabular-nums text-surface-600 dark:text-surface-300">
                    {row.map.toFixed(3)}
                  </span>
                </span>
              </span>

              <span className="hidden md:block text-right text-xs text-surface-500 tabular-nums">
                ~{row.trainMin} min
              </span>
            </div>
          )
        })}
      </div>

      <p className="mt-4 text-xs text-surface-400 dark:text-surface-500 leading-relaxed max-w-3xl mx-auto text-center">
        Both runs are complete: 100 of 100 datasets trained and scored, none skipped, all
        protocol-conformant. Single seed, 640 px, one RTX 5060 Ti. Raw artifacts, per-epoch
        metrics and the manifest pinning the exact commit are published at{' '}
        <a
          href="https://huggingface.co/datasets/LibreYOLO/rf100-vl-results"
          target="_blank"
          rel="noopener noreferrer"
          className="text-libre-600 dark:text-libre-400 underline underline-offset-2"
        >
          LibreYOLO/rf100-vl-results
        </a>
        . The remaining families are still running and will be added as each one finishes a
        full 100.
      </p>
    </section>
  )
}
