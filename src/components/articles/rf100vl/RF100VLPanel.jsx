'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import {
  X, Orbit, ExternalLink, Grid3x3, Trophy, ArrowUpRight,
  // ring motifs
  Trophy as TrophyIcon, Target, Medal, Bike, Dumbbell,
  FileText, ScrollText, PenLine, Receipt, BookOpen,
  Plane, Cloud, Satellite, Wind, Mountain,
  HeartPulse, Pill, Brain, Syringe, Bone,
  Shapes, Puzzle, Sparkles, Gamepad2, Package,
  Cog, Wrench, Factory, Hammer, Zap,
  Leaf, Flower2, Bird, TreePine, Fish, Bug, PawPrint,
} from 'lucide-react'
import { RF100VL_DATASETS, RF100VL_DOMAINS } from './datasets'
import RESULTS_BY_DATASET from './results-by-dataset.json'
import './rf100vl.css'

const RING_VARS = ['--rf-r1', '--rf-r2', '--rf-r3', '--rf-r4', '--rf-r5', '--rf-r6', '--rf-r7']
const RING_DURATIONS = [46, 56, 66, 76, 88, 100, 114]
const LABEL_PHASES = [0.5, 0.08, 0.62, 0.2, 0.76, 0.34, 0.9]
const ART_DURATIONS = [150, 170, 190, 210, 230, 250, 270]
const ICON_COUNTS = [12, 16, 21, 26, 30, 36, 42]

const RING_MOTIFS = {
  Sports: [[TrophyIcon, '#fb923c', 16], [Target, '#f87171', 13], [Medal, '#fbbf24', 15], [Bike, '#fb923c', 16], [Dumbbell, '#fdba74', 14]],
  Document: [[FileText, '#a78bfa', 15], [PenLine, '#c4b5fd', 13], [ScrollText, '#8b5cf6', 16], [Receipt, '#a78bfa', 14], [BookOpen, '#c4b5fd', 15]],
  Aerial: [[Plane, '#38bdf8', 16], [Cloud, '#7dd3fc', 14], [Satellite, '#0ea5e9', 15], [Wind, '#7dd3fc', 13], [Mountain, '#38bdf8', 15]],
  Medical: [[HeartPulse, '#f472b6', 16], [Pill, '#fb7185', 13], [Brain, '#f472b6', 15], [Syringe, '#f9a8d4', 14], [Bone, '#fda4af', 14]],
  Other: [[Shapes, '#94a3b8', 15], [Puzzle, '#64748b', 14], [Sparkles, '#b0bcca', 13], [Gamepad2, '#64748b', 16], [Package, '#94a3b8', 14]],
  Industrial: [[Cog, '#fbbf24', 16], [Wrench, '#f59e0b', 14], [Factory, '#fbbf24', 15], [Hammer, '#fcd34d', 13], [Zap, '#f59e0b', 14]],
  'Flora and Fauna': [[Leaf, '#34d399', 15], [Flower2, '#f472b6', 15], [Leaf, '#10b981', 13], [Bird, '#fbbf24', 14], [TreePine, '#10b981', 16], [Fish, '#38bdf8', 14], [Bug, '#a3e635', 13], [PawPrint, '#34d399', 14]],
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

const MODELS = Object.values(RESULTS_BY_DATASET).sort((a, b) => b.mean - a.mean)

const TABS = [
  { id: 'domains', label: 'The benchmark', icon: Orbit },
  { id: 'datasets', label: 'Explore 100 datasets', icon: Grid3x3 },
  { id: 'results', label: 'Results', icon: Trophy },
]

// Score bands. RF100-VL spans a huge range, so colour carries the verdict.
// Two ramps: the bright one for fills (bars, dots, swatches) where contrast is
// against the panel, and a darker one for text, because the bright greens and
// ambers sit at ~2:1 on a white background. CSS swaps back to the bright ramp
// in dark mode, where it is the readable one.
function bandColor(m) {
  if (m >= 0.75) return '#34d399'
  if (m >= 0.6) return '#38bdf8'
  if (m >= 0.45) return '#a78bfa'
  if (m >= 0.3) return '#fbbf24'
  return '#fb7185'
}

function bandTextColor(m) {
  if (m >= 0.75) return '#047857'
  if (m >= 0.6) return '#0369a1'
  if (m >= 0.45) return '#6d28d9'
  if (m >= 0.3) return '#b45309'
  return '#be123c'
}

// Both ramps on one element; the stylesheet decides which one applies.
function bandVars(m) {
  return { '--band': bandColor(m), '--band-text': bandTextColor(m) }
}

const BANDS = [
  { label: '75+', v: 0.8 },
  { label: '60 to 75', v: 0.65 },
  { label: '45 to 60', v: 0.5 },
  { label: '30 to 45', v: 0.35 },
  { label: 'under 30', v: 0.1 },
]

function pct(v) {
  return `${(v * 100).toFixed(1)}%`
}

/* ---------------------------------------------------------------- Tab 1 */

function DomainRings({ focusDomain }) {
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
              style={{ '--r': `var(${ring.radiusVar})`, '--dur': `${ring.artDuration}s`, '--phase': phase }}
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
            style={{ '--r': `var(${ring.radiusVar})`, '--dur': `${ring.artDuration}s`, '--phase': ring.labelPhase }}
          >
            <div className="rf-upright">
              <span
                className={`rfp-ring-label rfa-ring-label${focusDomain === ring.domain ? ' is-hot' : ''}`}
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
      </div>
    </div>
  )
}

/* ---------------------------------------------------------------- Drawer */

function DatasetDrawer({ dataset, onClose, onSwitch }) {
  const siblings = RF100VL_DATASETS.filter((d) => d.domain === dataset.domain && d !== dataset).slice(0, 6)
  const color = COLOR_BY_DOMAIN[dataset.domain]
  const scores = MODELS.map((m) => ({ model: m.model, s: m.scores[dataset.name] })).filter((x) => x.s)

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
          <button type="button" className="rfd-close" aria-label="Close dataset panel" onClick={onClose}>
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="px-5">
          <div className="rfd-main-img" style={{ '--c': color }}>
            <img src={dataset.img} alt={`Annotated sample from the ${dataset.name} dataset`} />
            <span className="rfd-img-tag">annotated sample</span>
          </div>
        </div>

        {scores.length > 0 && (
          <div className="px-5 pt-4">
            <h4 className="text-xs font-bold uppercase tracking-[0.15em] text-surface-500 dark:text-surface-400">
              How LibreYOLO scores here
            </h4>
            <div className="mt-2.5 space-y-2">
              {scores.map(({ model, s }) => (
                <div key={model} className="rfd-score-row">
                  <span className="rfd-score-model">{model}</span>
                  <span className="rfd-score-bar">
                    <span style={{ width: `${Math.max(s.m * 100, 2)}%`, background: bandColor(s.m) }} />
                  </span>
                  <span className="rfd-score-value" style={bandVars(s.m)}>{pct(s.m)}</span>
                </div>
              ))}
            </div>
            <p className="mt-2 text-[11px] leading-relaxed text-surface-400 dark:text-surface-500">
              mAP@50-95 on this dataset&apos;s own test split, after fine-tuning on it.
              {scores[0]?.s && ` ${scores[0].s.img} test images, ${scores[0].s.cls} ${scores[0].s.cls === 1 ? 'class' : 'classes'}.`}
            </p>
          </div>
        )}

        {siblings.length > 0 && (
          <div className="px-5 pt-6">
            <h4 className="text-xs font-bold uppercase tracking-[0.15em] text-surface-500 dark:text-surface-400">
              More from {dataset.domain}
            </h4>
            <div className="mt-2.5 grid grid-cols-3 gap-2">
              {siblings.map((d) => (
                <button key={d.name} type="button" className="rfd-sibling" title={d.name} onClick={() => onSwitch(d)}>
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

/* ---------------------------------------------------------------- Tab 2/3 */

function PlanetRings({ colorFor, dimFor, onHover, onOpen, active, drawerDs, focusDomain }) {
  return (
    <div className="rfe-layer">
      {RINGS.map((ring) => (
        <div
          key={`outline-${ring.domain}`}
          className={`rfp-ring-outline${
            (focusDomain || (active ? active.domain : null)) === ring.domain ? ' is-hot' : ''
          }`}
          style={{ '--r': `var(${ring.radiusVar})`, '--c': ring.color }}
        />
      ))}

      {RINGS.map((ring) => (
        <div key={ring.domain} className={ring.reverse ? 'rf-rev' : ''}>
          <div
            className="rf-sat rf-label-sat"
            style={{ '--r': `var(${ring.radiusVar})`, '--dur': `${ring.duration}s`, '--phase': ring.labelPhase }}
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
            const isDim = dimFor(dataset, isActive)
            return (
              <div
                key={dataset.name}
                className="rf-sat"
                style={{ '--r': `var(${ring.radiusVar})`, '--dur': `${ring.duration}s`, '--phase': i / ring.datasets.length }}
              >
                <div className="rf-upright">
                  <button
                    type="button"
                    className={`rfp-planet${isActive ? ' is-active' : ''}${isDim ? ' is-dim' : ''}`}
                    style={{ '--c': colorFor(dataset, ring) }}
                    aria-label={`Open the ${dataset.name} dataset (${ring.domain})`}
                    title={dataset.name}
                    onMouseEnter={() => onHover(dataset)}
                    onMouseLeave={() => onHover(null)}
                    onFocus={() => onHover(dataset)}
                    onBlur={() => onHover(null)}
                    onClick={() => onOpen(dataset)}
                  >
                    <img src={dataset.img} alt="" loading="lazy" draggable={false} />
                  </button>
                </div>
              </div>
            )
          })}
        </div>
      ))}
    </div>
  )
}

/* ------------------------------------------------------- Results bar chart */

// One bar per dataset, tallest first. The shape of the curve is the story:
// a long strong shoulder, then a cliff into the datasets nothing handles well.
function ResultBars({ rows, onHover, onOpen, active }) {
  if (rows.length === 0) return null
  const mean = rows.reduce((a, r) => a + r.s.m, 0) / rows.length

  return (
    <div className="rfb-chart">
      <div className="rfb-axis">
        {[1, 0.75, 0.5, 0.25].map((t) => (
          <div key={t} className="rfb-gridline" style={{ bottom: `${t * 100}%` }}>
            <span>{Math.round(t * 100)}</span>
          </div>
        ))}
        <div className="rfb-mean" style={{ bottom: `${mean * 100}%` }}>
          <span>mean {pct(mean)}</span>
        </div>

        <div className="rfb-bars">
          {rows.map(({ ds, s }) => (
            <button
              key={ds.name}
              type="button"
              className={`rfb-bar${active === ds ? ' is-on' : ''}`}
              style={{ '--h': `${Math.max(s.m * 100, 1)}%`, '--c': bandColor(s.m) }}
              title={`${ds.name}: ${pct(s.m)}`}
              aria-label={`${ds.name}, ${pct(s.m)} mAP`}
              onMouseEnter={() => onHover(ds)}
              onMouseLeave={() => onHover(null)}
              onFocus={() => onHover(ds)}
              onBlur={() => onHover(null)}
              onClick={() => onOpen(ds)}
            />
          ))}
        </div>
      </div>

      <div className="rfb-foot">
        <span>strongest</span>
        <span className="rfb-foot-mid">{rows.length} datasets, sorted by score</span>
        <span>weakest</span>
      </div>
    </div>
  )
}

/* ---------------------------------------------------------------- Panel */

export default function RF100VLPanel() {
  const [tab, setTab] = useState('domains')
  const [drawerDs, setDrawerDs] = useState(null)
  const [hoverDs, setHoverDs] = useState(null)
  const [focusDomain, setFocusDomain] = useState(null)
  const [modelId, setModelId] = useState(MODELS[0]?.model ?? null)

  const model = useMemo(() => MODELS.find((m) => m.model === modelId) ?? null, [modelId])
  const engaged = Boolean(hoverDs || drawerDs)
  const active = hoverDs
  // Rings are for exploring what the benchmark is made of. Scores read better
  // as a sorted bar chart, so the results tab drops the orbit entirely.
  const orbiting = tab === 'datasets'

  const ranked = useMemo(() => {
    if (!model) return []
    return RF100VL_DATASETS
      .map((d) => ({ ds: d, s: model.scores[d.name] }))
      .filter((x) => x.s)
      .sort((a, b) => b.s.m - a.s.m)
  }, [model])

  const tabRef = useRef(null)
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') setDrawerDs(null)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  // Tabs are linkable: ?tab=results deep-links straight to the scores, and
  // switching tabs rewrites the URL without a navigation.
  useEffect(() => {
    const wanted = new URLSearchParams(window.location.search).get('tab')
    if (wanted && TABS.some((t) => t.id === wanted)) setTab(wanted)
  }, [])

  const selectTab = (id) => {
    setTab(id)
    setDrawerDs(null)
    setHoverDs(null)
    const url = new URL(window.location.href)
    if (id === 'domains') url.searchParams.delete('tab')
    else url.searchParams.set('tab', id)
    window.history.replaceState(null, '', url)
  }

  return (
    <section aria-label="RF100-VL benchmark" className="rfp-hero not-prose">
      <div className="rfp-stars" aria-hidden="true" />
      <div className="rfp-grid" aria-hidden="true" />
      <div className={`rfp-bg${orbiting && active ? ' is-on' : ''}`} aria-hidden="true">
        {orbiting && active && <img key={active.name} src={active.img} alt="" />}
      </div>

      {/* Tabs */}
      <div className="relative z-10 flex justify-center px-6 pt-10" ref={tabRef}>
        <div className="rfx-tabs" role="tablist" aria-label="RF100-VL views">
          {TABS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              role="tab"
              type="button"
              aria-selected={tab === id}
              className={`rfx-tab${tab === id ? ' is-on' : ''}`}
              onClick={() => selectTab(id)}
            >
              <Icon className="h-3.5 w-3.5" />
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Model picker, results tab only */}
      {tab === 'results' && MODELS.length > 0 && (
        <div className="relative z-10 mt-4 flex flex-wrap items-center justify-center gap-2 px-6">
          {MODELS.map((m) => (
            <button
              key={m.model}
              type="button"
              className={`rfx-model${modelId === m.model ? ' is-on' : ''}`}
              onClick={() => setModelId(m.model)}
            >
              <span className="rfx-model-name">{m.model}</span>
              <span className="rfx-model-score">{pct(m.mean)}</span>
            </button>
          ))}
        </div>
      )}

      {/* Stage */}
      {tab === 'results' && model && (
        <div className="relative z-10 mx-auto mt-6 max-w-5xl px-6">
          <div className="rfb-panel">
            <div className="rfb-head">
              <div>
                <div className="rfb-head-value">{pct(active ? model.scores[active.name]?.m ?? model.mean : model.mean)}</div>
                <div className="rfb-head-label">
                  {active ? active.name : `${model.model}, mean of ${ranked.length} datasets`}
                </div>
              </div>
              <div className="rfb-head-preview">
                {active && <img src={active.img} alt="" />}
              </div>
            </div>
            <ResultBars
              rows={ranked}
              active={active}
              onHover={setHoverDs}
              onOpen={setDrawerDs}
            />
          </div>
        </div>
      )}

      {tab !== 'results' && (
      <div className={`rfp-orbit${engaged ? ' rfp-paused' : ''}`} onMouseLeave={() => setHoverDs(null)}>
        {tab === 'domains' && <DomainRings focusDomain={focusDomain} />}

        {orbiting && (
          <>
            <PlanetRings
              focusDomain={focusDomain}
              active={active}
              drawerDs={drawerDs}
              onHover={setHoverDs}
              onOpen={setDrawerDs}
              colorFor={(dataset, ring) => {
                if (tab !== 'results' || !model) return ring.color
                const s = model.scores[dataset.name]
                return s ? bandColor(s.m) : ring.color
              }}
              dimFor={(dataset, isActive) =>
                (engaged && !isActive) || (focusDomain && dataset.domain !== focusDomain)
              }
            />

            <div className="rfp-core" aria-hidden="true">
              <span className="rfp-core-pill">
                {tab === 'results' && model ? pct(active ? model.scores[active.name]?.m ?? model.mean : model.mean) : 'RF100-VL'}
              </span>
              <div className="rfp-core-sub">
                {tab === 'results' && model
                  ? (active ? 'on this dataset' : `${model.model}, mean of 100`)
                  : '100 datasets · 7 domains'}
              </div>
            </div>

            <div className={`rfp-info${active && !drawerDs ? ' is-on' : ''}`} aria-live="polite">
              {active && (
                <>
                  <img src={active.img} alt={`Sample from the ${active.name} dataset`} />
                  <div className="min-w-0">
                    <div className="rfp-info-name">{active.name}</div>
                    <div className="rfp-info-domain">
                      <span className="rfp-swatch" style={{ background: COLOR_BY_DOMAIN[active.domain] }} />
                      {active.domain}
                      {tab === 'results' && model && model.scores[active.name] && (
                        <span className="ml-2 font-mono font-bold rfx-band-text" style={bandVars(model.scores[active.name].m)}>
                          {pct(model.scores[active.name].m)}
                        </span>
                      )}
                    </div>
                  </div>
                </>
              )}
            </div>
          </>
        )}
      </div>
      )}

      {orbiting && (
        <p className="relative z-10 mt-2 text-center font-mono text-[10px] uppercase tracking-[0.22em] text-surface-400 dark:text-surface-600">
          hover to preview · click a planet to open the dataset
        </p>
      )}

      {/* Legend: domains, or score bands on the results tab */}
      <div className="relative z-10 mt-4 flex flex-wrap justify-center gap-2 px-6">
        {tab === 'results'
          ? BANDS.map((b) => (
              <span key={b.label} className="rfp-legend-chip" style={{ '--c': bandColor(b.v) }}>
                <span className="rfp-swatch" style={{ background: bandColor(b.v) }} />
                {b.label}
              </span>
            ))
          : RF100VL_DOMAINS.map((domain) => (
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

      {/* Results detail */}
      {tab === 'results' && model && (
        <div className="relative z-10 mx-auto mt-8 grid max-w-4xl gap-3 px-6 pb-12 sm:grid-cols-2">
          {[
            { title: 'Strongest datasets', rows: ranked.slice(0, 5) },
            { title: 'Weakest datasets', rows: ranked.slice(-5).reverse() },
          ].map(({ title, rows }) => (
            <div key={title} className="rfx-list">
              <div className="rfx-list-head">{title}</div>
              {rows.map(({ ds, s }) => (
                <button
                  key={ds.name}
                  type="button"
                  className="rfx-list-row"
                  onClick={() => setDrawerDs(ds)}
                  onMouseEnter={() => setHoverDs(ds)}
                  onMouseLeave={() => setHoverDs(null)}
                >
                  <img src={ds.img} alt="" loading="lazy" />
                  <span className="rfx-list-name">{ds.name}</span>
                  <span className="rfx-list-value" style={bandVars(s.m)}>{pct(s.m)}</span>
                  <ArrowUpRight className="h-3 w-3 shrink-0 opacity-40" />
                </button>
              ))}
            </div>
          ))}
        </div>
      )}

      {tab !== 'results' && <div className="pb-10 md:pb-14" />}

      {drawerDs && (
        <DatasetDrawer dataset={drawerDs} onClose={() => setDrawerDs(null)} onSwitch={(d) => setDrawerDs(d)} />
      )}
    </section>
  )
}
