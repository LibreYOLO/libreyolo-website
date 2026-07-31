'use client'

import { useEffect, useRef, useState } from 'react'
import { Crown, Zap, AlertTriangle, Cpu, Layers, Repeat, Timer } from 'lucide-react'
import './rf100vl-results.css'

// Placeholder sweep data (issue #542 protocol): fine-tune the COCO checkpoint
// 100 epochs per dataset, evaluate on the test split, average over the
// datasets that finished. Sorted by mAP50-95.
const RESULTS = [
  { rank: 1, family: 'RF-DETR', size: 'S', ok: 94, map50: 0.784, map: 0.562, time: 198 },
  { rank: 2, family: 'EC', size: 'S', ok: 97, map50: 0.765, map: 0.555, time: 125 },
  { rank: 3, family: 'DEIMv2', size: 'S', ok: 98, map50: 0.766, map: 0.555, time: 143 },
  { rank: 4, family: 'RT-DETRv4', size: 'S', ok: 99, map50: 0.766, map: 0.554, time: 114 },
  { rank: 5, family: 'DEIM', size: 'S', ok: 99, map50: 0.760, map: 0.549, time: 126 },
  { rank: 6, family: 'YOLO-NAS', size: 'S', ok: 100, map50: 0.766, map: 0.544, time: 37 },
  { rank: 7, family: 'D-FINE', size: 'S', ok: 99, map50: 0.729, map: 0.528, time: 117 },
  { rank: 8, family: 'RT-DETR', size: 'R18', ok: 100, map50: 0.728, map: 0.517, time: 87 },
  { rank: 9, family: 'YOLOX', size: 'S', ok: 100, map50: 0.752, map: 0.510, time: 40 },
  { rank: 10, family: 'RT-DETRv2', size: 'R18', ok: 100, map50: 0.671, map: 0.479, time: 87 },
  { rank: 11, family: 'YOLOv9', size: 'S', ok: 100, map50: 0.676, map: 0.468, time: 31 },
  { rank: 12, family: 'YOLOv9 E2E', size: 'T', ok: 100, map50: 0.653, map: 0.452, time: 35 },
  { rank: 13, family: 'DAMO-YOLO', size: 'S', ok: 100, map50: 0.499, map: 0.333, time: 48 },
  { rank: 14, family: 'RTMDet', size: 'S', ok: 100, map50: 0.029, map: 0.014, time: 30 },
  { rank: 15, family: 'PicoDet', size: 'S', ok: 100, map50: 0.018, map: 0.008, time: 41 },
]

const BEST_MAP = RESULTS[0].map

function tierOf(row) {
  if (row.rank === 1) return 'champion'
  if (row.map >= 0.545) return 'sota'
  if (row.map >= 0.5) return 'strong'
  if (row.map >= 0.3) return 'lagging'
  return 'collapsed'
}

const TIER_STYLES = {
  champion: {
    bar: 'bg-gradient-to-r from-libre-500 to-emerald-400 shadow-[0_0_12px_rgba(6,182,212,0.5)]',
    badge: null,
  },
  sota: {
    bar: 'bg-libre-500',
    badge: { label: 'SOTA tier', cls: 'text-libre-600 dark:text-libre-400 border-libre-500/40 bg-libre-500/10' },
  },
  strong: {
    bar: 'bg-sky-400',
    badge: null,
  },
  lagging: {
    bar: 'bg-surface-400',
    badge: { label: 'lags behind', cls: 'text-amber-600 dark:text-amber-400 border-amber-500/40 bg-amber-500/10' },
  },
  collapsed: {
    bar: 'bg-red-500',
    badge: { label: 'collapsed', cls: 'text-red-600 dark:text-red-400 border-red-500/40 bg-red-500/10' },
  },
}

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
        <p className="text-xs font-semibold uppercase tracking-[0.25em] text-libre-600 dark:text-libre-400">
          Preliminary leaderboard
        </p>
        <h2 className="mt-2 text-2xl md:text-3xl font-bold text-surface-800 dark:text-white tracking-tight">
          Fifteen families, one clear winner
        </h2>
        <p className="mt-2 text-sm text-surface-500 dark:text-surface-400 max-w-2xl mx-auto">
          One representative model per detection family, fine-tuned for 100 epochs on each of
          the 100 datasets, evaluated on the test split. Sorted by mAP<sup>50-95</sup>.
        </p>
      </div>

      {/* stat strip */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-8">
        <Stat icon={Layers} value="100" label="datasets" />
        <Stat icon={Cpu} value="15" label="model families" />
        <Stat icon={Repeat} value="1,486" label="runs completed" />
        <Stat icon={Timer} value="100 ep" label="per dataset" />
      </div>

      {/* champion card */}
      <div className="animated-border rounded-2xl p-5 md:p-6 mb-4 bg-white dark:bg-surface-900">
        <div className="flex flex-wrap items-center gap-x-6 gap-y-4">
          <div className="flex items-center gap-4">
            <span className="flex items-center justify-center w-11 h-11 rounded-full bg-gradient-to-br from-libre-500 to-emerald-400 text-white shadow-lg shadow-libre-500/30">
              <Crown className="w-5 h-5" />
            </span>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xl font-extrabold text-surface-800 dark:text-white tracking-tight">
                  RF-DETR <span className="text-libre-500">S</span>
                </span>
                <span className="text-[10px] font-bold uppercase tracking-widest text-emerald-600 dark:text-emerald-400 border border-emerald-500/40 bg-emerald-500/10 rounded-full px-2 py-0.5">
                  #1 of 15
                </span>
              </div>
              <div className="text-xs text-surface-500 mt-0.5">Best mAP<sup>50-95</sup> on the full sweep</div>
            </div>
          </div>
          <div className="flex items-center gap-6 ml-auto">
            <div className="text-right">
              <div className="text-3xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-libre-500 to-emerald-500 tabular-nums">
                0.562
              </div>
              <div className="text-[10px] uppercase tracking-widest text-surface-500">mAP<sup>50-95</sup></div>
            </div>
            <div className="text-right">
              <div className="text-3xl font-extrabold text-surface-700 dark:text-surface-200 tabular-nums">0.784</div>
              <div className="text-[10px] uppercase tracking-widest text-surface-500">mAP<sup>50</sup></div>
            </div>
          </div>
        </div>
      </div>

      {/* insight callouts */}
      <div className="grid md:grid-cols-3 gap-3 mb-8">
        <div className="rounded-xl border border-libre-500/25 bg-libre-500/5 dark:bg-libre-500/10 px-4 py-3">
          <div className="flex items-center gap-2 text-libre-600 dark:text-libre-400 text-xs font-bold uppercase tracking-wider mb-1">
            <Crown className="w-3.5 h-3.5" /> Accuracy crown
          </div>
          <p className="text-sm text-surface-600 dark:text-surface-400 leading-relaxed">
            Transformer detectors top the chart at ~0.55 mAP, but cost 3 to 5x more training time.
          </p>
        </div>
        <div className="rounded-xl border border-emerald-500/25 bg-emerald-500/5 dark:bg-emerald-500/10 px-4 py-3">
          <div className="flex items-center gap-2 text-emerald-600 dark:text-emerald-400 text-xs font-bold uppercase tracking-wider mb-1">
            <Zap className="w-3.5 h-3.5" /> Sweet spot
          </div>
          <p className="text-sm text-surface-600 dark:text-surface-400 leading-relaxed">
            YOLO-NAS S reaches 0.544 in ~37 min per dataset, the best accuracy-per-hour of the sweep.
          </p>
        </div>
        <div className="rounded-xl border border-red-500/25 bg-red-500/5 dark:bg-red-500/10 px-4 py-3">
          <div className="flex items-center gap-2 text-red-600 dark:text-red-400 text-xs font-bold uppercase tracking-wider mb-1">
            <AlertTriangle className="w-3.5 h-3.5" /> Hard floor
          </div>
          <p className="text-sm text-surface-600 dark:text-surface-400 leading-relaxed">
            RTMDet and PicoDet collapse to near zero on most datasets: recipe choice matters more than architecture.
          </p>
        </div>
      </div>

      {/* leaderboard */}
      <div className={`rounded-2xl border border-surface-200 dark:border-surface-800 bg-white dark:bg-surface-900 overflow-hidden${inView ? ' rf-bars-in' : ''}`}>
        {/* header row */}
        <div className="hidden md:grid grid-cols-[3rem_minmax(11rem,1.1fr)_4.5rem_4.5rem_minmax(14rem,1.6fr)_5.5rem] items-center gap-x-4 px-5 py-3 border-b border-surface-200 dark:border-surface-800 text-[10px] font-bold uppercase tracking-widest text-surface-500">
          <span>#</span>
          <span>Model</span>
          <span className="text-right">ok/100</span>
          <span className="text-right">mAP<sup>50</sup></span>
          <span>mAP<sup>50-95</sup></span>
          <span className="text-right">train</span>
        </div>

        {RESULTS.map((row, idx) => {
          const tier = tierOf(row)
          const style = TIER_STYLES[tier]
          const pct = Math.max((row.map / BEST_MAP) * 100, 1.5)
          const isChampion = tier === 'champion'
          const isYoloNas = row.family === 'YOLO-NAS'
          return (
            <div
              key={row.family}
              className={`rf-row grid grid-cols-[2.5rem_1fr_auto] md:grid-cols-[3rem_minmax(11rem,1.1fr)_4.5rem_4.5rem_minmax(14rem,1.6fr)_5.5rem] items-center gap-x-4 px-5 py-3 border-b last:border-b-0 border-surface-100 dark:border-surface-800/70 ${
                isChampion ? 'bg-libre-500/5 dark:bg-libre-500/10' : ''
              }`}
              style={{ '--d': `${idx * 45}ms` }}
            >
              <span
                className={`text-sm font-bold tabular-nums ${
                  isChampion ? 'text-libre-600 dark:text-libre-400' : 'text-surface-400'
                }`}
              >
                {row.rank}
              </span>

              <span className="min-w-0">
                <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
                  <span
                    className={`font-semibold truncate ${
                      isChampion ? 'text-libre-700 dark:text-libre-300' : 'text-surface-700 dark:text-surface-200'
                    }`}
                  >
                    {row.family}
                  </span>
                  <span className="text-[10px] font-mono text-surface-500 border border-surface-300 dark:border-surface-700 rounded px-1 py-px">
                    {row.size}
                  </span>
                  {style.badge && (
                    <span className={`text-[9px] font-bold uppercase tracking-widest border rounded-full px-1.5 py-px ${style.badge.cls}`}>
                      {style.badge.label}
                    </span>
                  )}
                  {isYoloNas && (
                    <span className="inline-flex items-center gap-1 text-[9px] font-bold uppercase tracking-widest text-emerald-600 dark:text-emerald-400 border border-emerald-500/40 bg-emerald-500/10 rounded-full px-1.5 py-px">
                      <Zap className="w-2.5 h-2.5" /> best trade-off
                    </span>
                  )}
                </span>
                {/* mobile-only metric line */}
                <span className="md:hidden block mt-1 text-[11px] text-surface-500 tabular-nums">
                  mAP<sup>50-95</sup> {row.map.toFixed(3)} · ok {row.ok}/100 · ~{row.time} min
                </span>
              </span>

              <span className="hidden md:block text-right text-sm text-surface-500 tabular-nums">{row.ok}</span>
              <span className="hidden md:block text-right text-sm text-surface-500 tabular-nums">
                {row.map50.toFixed(3)}
              </span>

              <span className="col-span-3 md:col-span-1 order-last md:order-none mt-2 md:mt-0">
                <span className="flex items-center gap-2">
                  <span className="flex-1 h-2 rounded-full bg-surface-200 dark:bg-surface-800 overflow-hidden">
                    <span
                      className={`rf-bar-fill block h-full rounded-full ${style.bar}`}
                      style={{ '--w': `${pct}%` }}
                    />
                  </span>
                  <span
                    className={`w-12 text-right text-sm font-bold tabular-nums ${
                      isChampion
                        ? 'text-libre-600 dark:text-libre-400'
                        : tier === 'collapsed'
                          ? 'text-red-500'
                          : 'text-surface-600 dark:text-surface-300'
                    }`}
                  >
                    {row.map.toFixed(3)}
                  </span>
                </span>
              </span>

              <span className="hidden md:block text-right text-xs text-surface-500 tabular-nums">
                ~{row.time} min
              </span>
            </div>
          )
        })}
      </div>

      <p className="mt-4 text-xs text-surface-400 dark:text-surface-500 leading-relaxed max-w-3xl mx-auto text-center">
        Placeholder numbers from the public sweep in{' '}
        <a
          href="https://github.com/LibreYOLO/libreyolo/issues/542"
          target="_blank"
          rel="noopener noreferrer"
          className="text-libre-600 dark:text-libre-400 underline underline-offset-2"
        >
          issue #542
        </a>
        : single seed, smallest practical size per family, 8x V100. Means are over the 94 to 100
        datasets each family finished; slow transformers timed out on the six largest datasets.
        The final report will replace this table.
      </p>
    </section>
  )
}
