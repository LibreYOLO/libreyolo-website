'use client'

import { useEffect, useRef, useState } from 'react'
import { RF100VL_DATASETS, RF100VL_DOMAINS } from './datasets'
import './rf100vl.css'

// One ring per domain, innermost first (ascending dataset count).
const RING_VARS = ['--rf-r1', '--rf-r2', '--rf-r3', '--rf-r4', '--rf-r5', '--rf-r6', '--rf-r7']
const RING_DURATIONS = [42, 52, 60, 70, 82, 94, 108]

const RINGS = RF100VL_DOMAINS.map((domain, i) => ({
  domain: domain.name,
  color: domain.color,
  radiusVar: RING_VARS[i],
  duration: RING_DURATIONS[i],
  reverse: i % 2 === 1,
  datasets: RF100VL_DATASETS.filter((d) => d.domain === domain.name),
}))

const COLOR_BY_DOMAIN = Object.fromEntries(RF100VL_DOMAINS.map((d) => [d.name, d.color]))

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

function domClass(domain) {
  return domain === 'Flora and Fauna' ? 'Flora' : domain
}

export default function RF100VLHero() {
  const [active, setActive] = useState(null)
  const [hovering, setHovering] = useState(false)
  const hoverRef = useRef(false)

  // Idle auto-cycle through featured datasets; pauses while the user hovers.
  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return undefined
    let i = 0
    const id = window.setInterval(() => {
      if (hoverRef.current || document.hidden) return
      setActive(FEATURED[i % FEATURED.length])
      i += 1
    }, 3000)
    return () => window.clearInterval(id)
  }, [])

  // Preload thumbnails so hover swaps are instant.
  useEffect(() => {
    const t = window.setTimeout(() => {
      RF100VL_DATASETS.forEach((d) => {
        const img = new Image()
        img.src = d.img
      })
    }, 1500)
    return () => window.clearTimeout(t)
  }, [])

  const select = (dataset) => {
    hoverRef.current = true
    setHovering(true)
    setActive(dataset)
  }
  const release = () => {
    hoverRef.current = false
    setHovering(false)
  }

  return (
    <section
      aria-label="Interactive map of the 100 RF100-VL datasets orbiting RF-DETR"
      className="rf-hero not-prose my-10"
      style={{ width: 'min(96vw, 1280px)', marginLeft: 'calc(50% - min(48vw, 640px))' }}
    >
      {/* background: grid + the hovered dataset image */}
      <div className="rf-hero-grid" />
      <div className={`rf-bg-layer${active ? ' is-on' : ''}`} aria-hidden="true">
        {active && <img key={active.name} src={active.img} alt="" />}
      </div>

      <div className="relative z-10 px-6 pt-12 md:pt-16 text-center">
        <p className="text-[10px] md:text-xs font-semibold uppercase tracking-[0.3em] text-libre-300/80">
          The benchmark
        </p>
        <h2 className="mt-2 text-2xl md:text-4xl font-bold text-white tracking-tight">
          One model, one hundred datasets
        </h2>
        <p className="mt-2 md:mt-3 text-xs md:text-sm text-surface-400 max-w-xl mx-auto">
          RF100-VL spans seven domains, from chest x-rays to conveyor belts. Hover the
          orbiting dots to peek inside each dataset.
        </p>
      </div>

      <div
        className={`rf-orbit${hovering ? ' rf-paused' : ''}`}
        data-activedomain={hovering && active ? active.domain : undefined}
        onMouseLeave={release}
      >
        {/* ring outlines */}
        {RINGS.map((ring) => (
          <div
            key={`outline-${ring.domain}`}
            className={`rf-ring-outline rf-dom-${domClass(ring.domain)}`}
            style={{ '--r': `var(${ring.radiusVar})` }}
          />
        ))}

        {/* RF-DETR core */}
        <div className="rf-core">
          <span className="rf-core-pill">RF-DETR</span>
          <div className="rf-core-sub">100 datasets</div>
        </div>

        {/* dataset nodes */}
        {RINGS.map((ring) => (
          <div key={ring.domain} className={ring.reverse ? 'rf-rev' : ''}>
            {ring.datasets.map((dataset, i) => (
              <div
                key={dataset.name}
                className="rf-sat"
                style={{
                  '--r': `var(${ring.radiusVar})`,
                  '--dur': `${ring.duration}s`,
                  '--phase': i / ring.datasets.length,
                }}
              >
                <button
                  type="button"
                  className={`rf-node${active === dataset && hovering ? ' rf-active' : ''}`}
                  style={{ '--c': ring.color }}
                  aria-label={`${dataset.name} (${ring.domain} dataset)`}
                  title={dataset.name}
                  onMouseEnter={() => select(dataset)}
                  onFocus={() => select(dataset)}
                  onBlur={release}
                >
                  <span className="rf-dot" />
                </button>
              </div>
            ))}
          </div>
        ))}

        {/* active dataset card */}
        <div className={`rf-info${active ? ' is-on' : ''}`} aria-live="polite">
          {active && (
            <>
              <img src={active.img} alt={`Sample from the ${active.name} dataset`} />
              <div className="min-w-0">
                <div className="rf-info-name">{active.name}</div>
                <div className="rf-info-domain">
                  <span className="rf-swatch" style={{ background: COLOR_BY_DOMAIN[active.domain] }} />
                  {active.domain}
                </div>
              </div>
            </>
          )}
        </div>
      </div>

      {/* legend */}
      <div className="relative z-10 flex flex-wrap justify-center gap-x-5 gap-y-2 px-6 pb-8 md:pb-10">
        {RF100VL_DOMAINS.map((domain) => (
          <span key={domain.name} className="inline-flex items-center gap-1.5 text-[11px] text-surface-400">
            <span
              className="inline-block w-2 h-2 rounded-full"
              style={{ background: domain.color }}
            />
            {domain.name}
            <span className="text-surface-600">{domain.count}</span>
          </span>
        ))}
      </div>
    </section>
  )
}
