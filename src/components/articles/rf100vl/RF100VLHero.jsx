'use client'

import { useEffect, useRef, useState } from 'react'
import {
  FileText,
  Github,
  Globe,
  Database,
  X,
  Orbit,
  ExternalLink,
  ArrowLeft,
  // ring motifs
  Trophy,
  Target,
  Medal,
  Bike,
  Dumbbell,
  ScrollText,
  PenLine,
  Receipt,
  BookOpen,
  Plane,
  Cloud,
  Satellite,
  Wind,
  Mountain,
  HeartPulse,
  Pill,
  Brain,
  Syringe,
  Bone,
  Shapes,
  Puzzle,
  Sparkles,
  Gamepad2,
  Package,
  Cog,
  Wrench,
  Factory,
  Hammer,
  Zap,
  Leaf,
  Flower2,
  Bird,
  TreePine,
  Fish,
  Bug,
  PawPrint,
} from 'lucide-react'
import { RF100VL_DATASETS, RF100VL_DOMAINS } from './datasets'
import './rf100vl.css'

// One ring per domain, innermost first (ascending dataset count).
const RING_VARS = ['--rf-r1', '--rf-r2', '--rf-r3', '--rf-r4', '--rf-r5', '--rf-r6', '--rf-r7']
const RING_DURATIONS = [46, 56, 66, 76, 88, 100, 114]
// Staggered so the orbiting domain chips never line up on the same side.
const LABEL_PHASES = [0.5, 0.08, 0.62, 0.2, 0.76, 0.34, 0.9]
// Art mode: slow, dreamy drift.
const ART_DURATIONS = [150, 170, 190, 210, 230, 250, 270]
// Motif icons per ring, scaled to each ring's circumference.
const ICON_COUNTS = [12, 16, 21, 26, 30, 36, 42]

// Thematic motifs: [Icon, color, size]. Each ring stays mostly in its domain
// color with a few accent notes (pink flowers and golden birds in the green
// flora ring, red targets in the sports ring, and so on).
const RING_MOTIFS = {
  Sports: [
    [Trophy, '#fb923c', 16],
    [Target, '#f87171', 13],
    [Medal, '#fbbf24', 15],
    [Bike, '#fb923c', 16],
    [Dumbbell, '#fdba74', 14],
  ],
  Document: [
    [FileText, '#a78bfa', 15],
    [PenLine, '#c4b5fd', 13],
    [ScrollText, '#8b5cf6', 16],
    [Receipt, '#a78bfa', 14],
    [BookOpen, '#c4b5fd', 15],
  ],
  Aerial: [
    [Plane, '#38bdf8', 16],
    [Cloud, '#7dd3fc', 14],
    [Satellite, '#0ea5e9', 15],
    [Wind, '#7dd3fc', 13],
    [Mountain, '#38bdf8', 15],
  ],
  Medical: [
    [HeartPulse, '#f472b6', 16],
    [Pill, '#fb7185', 13],
    [Brain, '#f472b6', 15],
    [Syringe, '#f9a8d4', 14],
    [Bone, '#fda4af', 14],
  ],
  Other: [
    [Shapes, '#94a3b8', 15],
    [Puzzle, '#64748b', 14],
    [Sparkles, '#b0bcca', 13],
    [Gamepad2, '#64748b', 16],
    [Package, '#94a3b8', 14],
  ],
  Industrial: [
    [Cog, '#fbbf24', 16],
    [Wrench, '#f59e0b', 14],
    [Factory, '#fbbf24', 15],
    [Hammer, '#fcd34d', 13],
    [Zap, '#f59e0b', 14],
  ],
  'Flora and Fauna': [
    [Leaf, '#34d399', 15],
    [Flower2, '#f472b6', 15],
    [Leaf, '#10b981', 13],
    [Bird, '#fbbf24', 14],
    [TreePine, '#10b981', 16],
    [Fish, '#38bdf8', 14],
    [Bug, '#a3e635', 13],
    [PawPrint, '#34d399', 14],
  ],
}

const RINGS = RF100VL_DOMAINS.map((domain, i) => {
  const motif = RING_MOTIFS[domain.name]
  const iconCount = ICON_COUNTS[i]
  return {
    domain: domain.name,
    color: domain.color,
    count: domain.count,
    radiusVar: RING_VARS[i],
    duration: RING_DURATIONS[i],
    artDuration: ART_DURATIONS[i],
    labelPhase: LABEL_PHASES[i],
    reverse: i % 2 === 1,
    datasets: RF100VL_DATASETS.filter((d) => d.domain === domain.name),
    icons: Array.from({ length: iconCount }, (_, j) => {
      const [Icon, color, size] = motif[j % motif.length]
      return { Icon, color, size, phase: j / iconCount }
    }),
  }
})

const COLOR_BY_DOMAIN = Object.fromEntries(RF100VL_DOMAINS.map((d) => [d.name, d.color]))

const LINKS = [
  { label: 'arXiv', href: 'https://arxiv.org/abs/2505.20612', icon: FileText },
  { label: 'Code', href: 'https://github.com/LibreYOLO/libreyolo', icon: Github },
  { label: 'rf100-vl.org', href: 'https://rf100-vl.org', icon: Globe },
  { label: 'Datasets', href: 'https://universe.roboflow.com/rf100-vl/', icon: Database },
]

// Datasets with visually distinctive samples, used for the idle auto-cycle.
const FEATURED = [
  'underwater-objects',
  'nih-xray',
  'aerial-airport',
  'deeppcb',
  'jellyfish',
  'mahjong',
  'flir-camera-objects',
  'wildfire-smoke',
  'grapes-5',
  'football-player-detection',
  'aquarium-combined',
  'trail-camera',
  'taco-trash-annotations-in-context',
  'thermal-cheetah',
]
  .map((name) => RF100VL_DATASETS.find((d) => d.name === name))
  .filter(Boolean)

// Placeholder crops of the annotated sample, until the real per-dataset
// explorer ships with the final report.
const PREVIEW_CROPS = [
  { position: '0% 0%' },
  { position: '100% 0%' },
  { position: '50% 100%' },
  { position: '0% 100%' },
  { position: '100% 60%' },
  { position: '50% 30%' },
]

function ArtRings({ focusDomain, onExplore }) {
  return (
    <div className="rfa-layer" aria-label="The seven RF100-VL domains as themed rings">
      {RINGS.map((ring) => (
        <div
          key={`body-${ring.domain}`}
          className={`rfa-ring${focusDomain === ring.domain ? ' is-hot' : ''}`}
          style={{ '--r': `var(${ring.radiusVar})`, '--c': ring.color }}
        />
      ))}

      {RINGS.map((ring) => (
        <div key={`icons-${ring.domain}`} className={ring.reverse ? 'rf-rev' : ''}>
          {ring.icons.map(({ Icon, color, size, phase }, j) => (
            <div
              key={j}
              className="rf-sat rfa-icon-sat"
              style={{
                '--r': `var(${ring.radiusVar})`,
                '--dur': `${ring.artDuration}s`,
                '--phase': phase,
              }}
            >
              <div className="rf-upright">
                <span className="rfa-ic">
                  <Icon size={size} strokeWidth={1.9} style={{ color }} />
                </span>
              </div>
            </div>
          ))}

          <div
            className="rf-sat rf-label-sat"
            style={{
              '--r': `var(${ring.radiusVar})`,
              '--dur': `${ring.artDuration}s`,
              '--phase': ring.labelPhase,
            }}
          >
            <div className="rf-upright">
              <span
                className={`rfp-ring-label rfa-ring-label${
                  focusDomain === ring.domain ? ' is-hot' : ''
                }`}
                style={{ '--c': ring.color }}
              >
                {ring.domain} · {ring.count}
              </span>
            </div>
          </div>
        </div>
      ))}

      <div className="rfa-core">
        <span className="rfa-core-pill">RF100-VL</span>
        <div className="rfa-core-sub">100 real-world datasets, 7 domains</div>
        <button type="button" className="rfa-explore-btn" onClick={onExplore}>
          <Orbit className="h-4 w-4" />
          Explore the 100 datasets
        </button>
      </div>
    </div>
  )
}

function DatasetDrawer({ dataset, onClose, onSwitch }) {
  const siblings = RF100VL_DATASETS.filter(
    (d) => d.domain === dataset.domain && d !== dataset
  ).slice(0, 6)
  const color = COLOR_BY_DOMAIN[dataset.domain]

  return (
    <>
      <div className="rfd-backdrop" onClick={onClose} aria-hidden="true" />
      <aside className="rfd-drawer" role="dialog" aria-label={`${dataset.name} dataset`}>
        <div className="flex items-start justify-between gap-4 p-5 pb-4">
          <div className="min-w-0">
            <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.18em] text-surface-500 dark:text-surface-400">
              <span className="rfp-swatch" style={{ background: color }} />
              {dataset.domain}
            </div>
            <h3 className="mt-1.5 font-mono text-lg font-bold text-surface-900 dark:text-white break-words">
              {dataset.name}
            </h3>
          </div>
          <button
            type="button"
            className="rfd-close"
            aria-label="Close dataset panel"
            onClick={onClose}
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="px-5">
          <div className="rfd-main-img" style={{ '--c': color }}>
            <img src={dataset.img} alt={`Annotated sample from the ${dataset.name} dataset`} />
            <span className="rfd-img-tag">annotated sample</span>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-2 px-5 pt-4">
          <div className="rfd-stat">
            <div className="rfd-stat-value">1 / 100</div>
            <div className="rfd-stat-label">RF100-VL</div>
          </div>
          <div className="rfd-stat">
            <div className="rfd-stat-value">{RF100VL_DATASETS.filter((d) => d.domain === dataset.domain).length}</div>
            <div className="rfd-stat-label">in domain</div>
          </div>
          <div className="rfd-stat">
            <div className="rfd-stat-value">100 ep</div>
            <div className="rfd-stat-label">fine-tune</div>
          </div>
        </div>

        <div className="px-5 pt-6">
          <div className="flex items-baseline justify-between">
            <h4 className="text-xs font-bold uppercase tracking-[0.15em] text-surface-500 dark:text-surface-400">
              Sample previews
            </h4>
            <span className="text-[10px] text-surface-400 dark:text-surface-500">placeholder crops</span>
          </div>
          <div className="mt-2.5 grid grid-cols-3 gap-2">
            {PREVIEW_CROPS.map((crop, i) => (
              <div key={i} className="rfd-crop">
                <img
                  src={dataset.img}
                  alt=""
                  loading="lazy"
                  style={{ objectPosition: crop.position }}
                />
              </div>
            ))}
          </div>
          <p className="mt-2 text-[11px] leading-relaxed text-surface-400 dark:text-surface-500">
            The full per-dataset image explorer ships with the final report.
          </p>
        </div>

        {siblings.length > 0 && (
          <div className="px-5 pt-6">
            <h4 className="text-xs font-bold uppercase tracking-[0.15em] text-surface-500 dark:text-surface-400">
              More from {dataset.domain}
            </h4>
            <div className="mt-2.5 grid grid-cols-3 gap-2">
              {siblings.map((d) => (
                <button
                  key={d.name}
                  type="button"
                  className="rfd-sibling"
                  title={d.name}
                  onClick={() => onSwitch(d)}
                >
                  <img src={d.img} alt="" loading="lazy" />
                  <span>{d.name}</span>
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="p-5 pt-6">
          <a
            href={`https://universe.roboflow.com/search?q=${encodeURIComponent(dataset.name)}`}
            target="_blank"
            rel="noopener noreferrer"
            className="rfd-cta"
          >
            <ExternalLink className="h-4 w-4" />
            Open on Roboflow Universe
          </a>
        </div>
      </aside>
    </>
  )
}

export default function RF100VLHero({ title, author, dateISO, dateLabel, backLink }) {
  // Two stages: 'art' shows the seven themed domain rings around the benchmark;
  // 'explore' swaps them for the 100 dataset planets. In explore mode,
  // hovering previews a dataset and clicking opens the lateral panel.
  const [mode, setMode] = useState('art')
  const [drawerDs, setDrawerDs] = useState(null)
  const [hoverDs, setHoverDs] = useState(null)
  const [autoDs, setAutoDs] = useState(null)
  const [focusDomain, setFocusDomain] = useState(null)

  const exploring = mode === 'explore'
  const engaged = Boolean(hoverDs || drawerDs)
  const active = hoverDs || autoDs

  const stateRef = useRef({ engaged: false, exploring: false })
  useEffect(() => {
    stateRef.current = { engaged, exploring }
  }, [engaged, exploring])

  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return undefined
    let i = 0
    const id = window.setInterval(() => {
      const { engaged: busy, exploring: on } = stateRef.current
      if (!on || busy || document.hidden) return
      setAutoDs(FEATURED[i % FEATURED.length])
      i += 1
    }, 3000)
    return () => window.clearInterval(id)
  }, [])

  useEffect(() => {
    const onKey = (e) => {
      if (e.key !== 'Escape') return
      setDrawerDs((open) => {
        if (open) return null
        setMode('art')
        return open
      })
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const exitExplore = () => {
    setMode('art')
    setDrawerDs(null)
    setHoverDs(null)
    setAutoDs(null)
  }

  return (
    <section aria-label="RF100-VL benchmark report" className="rfp-hero not-prose">
      {/* backdrop: stars (dark), faint grid, and the active dataset sample */}
      <div className="rfp-stars" aria-hidden="true" />
      <div className="rfp-grid" aria-hidden="true" />
      <div className={`rfp-bg${exploring && active ? ' is-on' : ''}`} aria-hidden="true">
        {exploring && active && <img key={active.name} src={active.img} alt="" />}
      </div>

      {/* paper-style masthead */}
      <header className="relative z-10 max-w-4xl mx-auto px-6 pt-28 md:pt-36 text-center">
        {backLink && <div className="mb-8 text-left">{backLink}</div>}
        <p className="font-mono text-[10px] md:text-xs font-semibold uppercase tracking-[0.35em] text-libre-600 dark:text-libre-300/90">
          LibreYOLO · Benchmark report
        </p>
        <h1 className="mt-4 text-3xl sm:text-4xl md:text-5xl font-bold text-surface-900 dark:text-white tracking-tight leading-tight">
          {title}
        </h1>
        <div className="mt-5 flex flex-wrap items-center justify-center gap-x-2 gap-y-1 text-sm md:text-base">
          <span className="font-medium text-surface-800 dark:text-surface-100">{author}</span>
          <span className="text-surface-300 dark:text-surface-600">·</span>
          <span className="text-surface-500 dark:text-surface-400">LibreYOLO</span>
          <span className="text-surface-300 dark:text-surface-600">·</span>
          <time dateTime={dateISO} className="text-surface-500 dark:text-surface-400">
            {dateLabel}
          </time>
        </div>
        <div className="mt-3">
          <span className="inline-block rounded-full border border-emerald-500/40 bg-emerald-500/10 px-3 py-1 text-[10px] md:text-[11px] font-semibold uppercase tracking-[0.18em] text-emerald-700 dark:border-emerald-400/30 dark:bg-emerald-400/10 dark:text-emerald-300">
            Living report · updates as the sweep progresses
          </span>
        </div>
        <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
          {LINKS.map(({ label, href, icon: Icon }) => (
            <a
              key={label}
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 rounded-full border border-surface-900 bg-surface-900 px-5 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-surface-700 hover:border-surface-700 dark:border-white/15 dark:bg-white/5 dark:font-medium dark:text-surface-200 dark:hover:border-white/30 dark:hover:bg-white/10"
            >
              <Icon className="h-4 w-4" />
              {label}
            </a>
          ))}
        </div>
      </header>

      {/* the stage: themed rings by default, dataset planets when exploring */}
      <div
        className={`rfp-orbit${engaged ? ' rfp-paused' : ''}`}
        onMouseLeave={() => setHoverDs(null)}
      >
        {!exploring && (
          <ArtRings focusDomain={focusDomain} onExplore={() => setMode('explore')} />
        )}

        {exploring && (
          <div className="rfe-layer">
            {RINGS.map((ring) => (
              <div
                key={`outline-${ring.domain}`}
                className={`rfp-ring-outline${
                  (focusDomain || (engaged && active ? active.domain : null)) === ring.domain
                    ? ' is-hot'
                    : ''
                }`}
                style={{ '--r': `var(${ring.radiusVar})`, '--c': ring.color }}
              />
            ))}

            <div className="rfp-core" aria-hidden="true">
              <span className="rfp-core-pill">RF100-VL</span>
              <div className="rfp-core-sub">100 datasets · 7 domains</div>
            </div>

            {RINGS.map((ring) => (
              <div key={ring.domain} className={ring.reverse ? 'rf-rev' : ''}>
                <div
                  className="rf-sat rf-label-sat"
                  style={{
                    '--r': `var(${ring.radiusVar})`,
                    '--dur': `${ring.duration}s`,
                    '--phase': ring.labelPhase,
                  }}
                >
                  <div className="rf-upright">
                    <span
                      className={`rfp-ring-label${focusDomain === ring.domain ? ' is-hot' : ''}`}
                      style={{ '--c': ring.color }}
                    >
                      {ring.domain}
                    </span>
                  </div>
                </div>

                {ring.datasets.map((dataset, i) => {
                  const isActive = active === dataset || drawerDs === dataset
                  const isDim =
                    (engaged && !isActive) || (focusDomain && dataset.domain !== focusDomain)
                  return (
                    <div
                      key={dataset.name}
                      className="rf-sat"
                      style={{
                        '--r': `var(${ring.radiusVar})`,
                        '--dur': `${ring.duration}s`,
                        '--phase': i / ring.datasets.length,
                      }}
                    >
                      <div className="rf-upright">
                        <button
                          type="button"
                          className={`rfp-planet${isActive ? ' is-active' : ''}${isDim ? ' is-dim' : ''}`}
                          style={{ '--c': ring.color }}
                          aria-label={`Open the ${dataset.name} dataset (${ring.domain})`}
                          title={dataset.name}
                          onMouseEnter={() => setHoverDs(dataset)}
                          onMouseLeave={() => setHoverDs(null)}
                          onFocus={() => setHoverDs(dataset)}
                          onBlur={() => setHoverDs(null)}
                          onClick={() => setDrawerDs(dataset)}
                        >
                          <img src={dataset.img} alt="" loading="lazy" draggable={false} />
                        </button>
                      </div>
                    </div>
                  )
                })}
              </div>
            ))}

            {/* hover preview card */}
            <div className={`rfp-info${active && !drawerDs ? ' is-on' : ''}`} aria-live="polite">
              {active && (
                <>
                  <img src={active.img} alt={`Sample from the ${active.name} dataset`} />
                  <div className="min-w-0">
                    <div className="rfp-info-name">{active.name}</div>
                    <div className="rfp-info-domain">
                      <span
                        className="rfp-swatch"
                        style={{ background: COLOR_BY_DOMAIN[active.domain] }}
                      />
                      {active.domain}
                    </div>
                  </div>
                </>
              )}
            </div>

            <button type="button" className="rfe-exit" onClick={exitExplore}>
              <ArrowLeft className="h-3.5 w-3.5" />
              Back to overview
            </button>
          </div>
        )}
      </div>

      {exploring && (
        <p className="relative z-10 mt-2 text-center font-mono text-[10px] uppercase tracking-[0.22em] text-surface-400 dark:text-surface-600">
          hover to preview · click a planet to open the dataset
        </p>
      )}

      {/* interactive legend: hover a domain to light up its ring */}
      <div className="relative z-10 mt-4 flex flex-wrap justify-center gap-2 px-6 pb-10 md:pb-14">
        {RF100VL_DOMAINS.map((domain) => (
          <button
            key={domain.name}
            type="button"
            className={`rfp-legend-chip${focusDomain === domain.name ? ' is-hot' : ''}`}
            style={{ '--c': domain.color }}
            onMouseEnter={() => setFocusDomain(domain.name)}
            onMouseLeave={() => setFocusDomain(null)}
            onFocus={() => setFocusDomain(domain.name)}
            onBlur={() => setFocusDomain(null)}
          >
            <span className="rfp-swatch" style={{ background: domain.color }} />
            {domain.name}
            <span className="rfp-legend-count">{domain.count}</span>
          </button>
        ))}
      </div>

      {drawerDs && (
        <DatasetDrawer
          dataset={drawerDs}
          onClose={() => setDrawerDs(null)}
          onSwitch={(d) => setDrawerDs(d)}
        />
      )}
    </section>
  )
}
