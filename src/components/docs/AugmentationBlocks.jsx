/*
 * Generated blocks for the augmentation pages.
 *
 * Every fact here comes from two files written by scripts/build-augmentation
 * against an installed LibreYOLO release:
 *
 *   augmentation.json          knobs, ranges, per-family support and defaults
 *   augmentation-figures.json  panels rendered by the library's own pipelines
 *
 * The same house rules as ModelBlocks apply: bare tables, state shown by icon
 * shape as well as color, a legend in full sentences, and nothing longer than
 * a token inside a matrix cell.
 */

import Link from 'next/link'
import { useTranslations } from 'next-intl'

import augmentation from '@/data/docs/augmentation.json'
import figures from '@/data/docs/augmentation-figures.json'

const KNOBS = augmentation.knobs
const FAMILIES = augmentation.families
const ARCHETYPES = ['yolox_style', 'yolonas', 'detr_style', 'classification', 'semantic', 'restore']
const DETECTION_ARCHETYPES = new Set(['yolox_style', 'yolonas', 'detr_style'])

const TH = 'border-b border-surface-300 px-3 py-1.5 text-left font-semibold text-surface-700 dark:border-white/20 dark:text-surface-300'
const TD = 'border-b border-surface-200/70 px-3 py-1.5 align-top text-surface-700 dark:border-white/[0.07] dark:text-surface-400'
const CODE = 'font-mono text-[12.5px] text-surface-800 dark:text-surface-200'

function Table({ children, minWidth = '' }) {
  return (
    <div className="relative -mx-1 my-5 overflow-x-auto px-1">
      <table className={`w-full border-collapse text-[13.5px] ${minWidth}`}>{children}</table>
    </div>
  )
}

function Note({ children }) {
  return <p className="mt-2 max-w-[68ch] text-[13px] leading-relaxed text-surface-500 dark:text-surface-500">{children}</p>
}

/* ── figures ────────────────────────────────────────────────────── */

export function AugFigure({ name }) {
  const t = useTranslations('AugmentationBlocks')
  const panels = figures.figures[name] || []
  if (!panels.length) return null
  const cols = panels.length >= 4 ? 'sm:grid-cols-4' : 'sm:grid-cols-3'
  return (
    <div className={`my-5 grid grid-cols-2 gap-x-3 gap-y-4 ${cols}`}>
      {panels.map((panel) => {
        const box = panel.largest_box
          ? t('box', { width: panel.largest_box[0], height: panel.largest_box[1] })
          : panel.boxes === 0 ? t('noBox') : null
        const alt = [panel.label, panel.note, box].filter(Boolean).join(', ')
        return (
          <figure key={panel.src} className="m-0">
            <img
              src={panel.src}
              width={panel.width}
              height={panel.height}
              alt={alt}
              loading="lazy"
              decoding="async"
              className="block h-auto w-full bg-surface-100 dark:bg-white/[0.04]"
            />
            <figcaption className="mt-1.5 text-[12.5px] leading-snug text-surface-500 dark:text-surface-400">
              <code className={`${CODE} [overflow-wrap:anywhere]`}>{panel.label}</code>
              {panel.note && <span className="tabular-nums"> {panel.note}</span>}
              {box && <span className="block tabular-nums">{box}</span>}
            </figcaption>
          </figure>
        )
      })}
    </div>
  )
}

/* ── knob reference ─────────────────────────────────────────────── */

export function AugKnobTable() {
  const t = useTranslations('AugmentationBlocks')
  return (
    <Table minWidth="min-w-[600px]">
      <thead>
        <tr>
          <th scope="col" className={TH}>{t('knob')}</th>
          <th scope="col" className={TH}>{t('meaning')}</th>
          <th scope="col" className={TH}>{t('cli')}</th>
          <th scope="col" className={TH}>{t('baseDefault')}</th>
          <th scope="col" className={TH}>{t('range')}</th>
        </tr>
      </thead>
      <tbody>
        {KNOBS.map((knob) => (
          <tr key={knob.key}>
            <th scope="row" className={`${TD} text-left font-normal`}>
              <code className={CODE}>{knob.key}</code>
            </th>
            <td className={`${TD} min-w-[16ch]`}>{t(`knobMeaning.${knob.key}`)}</td>
            <td className={TD}>
              {knob.cli.map((spelling, i) => (
                <span key={spelling}>{i > 0 && ', '}<code className={CODE}>{spelling}</code></span>
              ))}
            </td>
            <td className={`${TD} whitespace-nowrap`}><code className={CODE}>{knob.default}</code></td>
            <td className={`${TD} min-w-[14ch]`}>{knob.range}</td>
          </tr>
        ))}
      </tbody>
    </Table>
  )
}

/* ── support matrix ─────────────────────────────────────────────── */

const ICONS = {
  used: (
    <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" className="inline-block">
      <path d="M3 8.5l3 3 7-7" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  ),
  gated_by_mosaic: (
    <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" className="inline-block">
      <circle cx="8" cy="8" r="5.5" fill="none" stroke="currentColor" strokeWidth="1.6" />
      <path d="M8 2.5a5.5 5.5 0 0 1 0 11z" fill="currentColor" />
    </svg>
  ),
}

const ICON_COLOR = {
  used: 'text-emerald-600 dark:text-emerald-400',
  gated_by_mosaic: 'text-amber-600 dark:text-amber-400',
}

function StatusCell({ status, label }) {
  if (status === 'ignored') return <span className="sr-only">{label}</span>
  return (
    <span className={`inline-flex ${ICON_COLOR[status]}`} title={label}>
      {ICONS[status]}
      <span className="sr-only">{label}</span>
    </span>
  )
}

function FamilyName({ family }) {
  if (!family.page) return family.name
  return (
    <Link href={family.page} className="font-medium text-libre-700 underline-offset-2 hover:underline dark:text-libre-400">
      {family.name}
    </Link>
  )
}

export function AugMatrix() {
  const t = useTranslations('AugmentationBlocks')
  const statusLabel = (status, knob) =>
    t(status === 'used' ? 'used' : status === 'gated_by_mosaic' ? 'gated' : 'ignored', { knob })

  return (
    <div>
      <Table minWidth="min-w-[560px]">
        <thead>
          <tr>
            <th scope="col" className={`${TH} align-bottom`}>{t('family')}</th>
            {KNOBS.map((knob) => (
              <th
                key={knob.key}
                scope="col"
                className="border-b border-surface-300 px-0.5 pb-1.5 pt-1 align-bottom font-mono text-[11px] font-medium text-surface-600 dark:border-white/20 dark:text-surface-400"
              >
                <span className="mx-auto block w-4 whitespace-nowrap [writing-mode:vertical-rl] rotate-180">{knob.key}</span>
              </th>
            ))}
          </tr>
        </thead>
        {ARCHETYPES.map((archetype) => {
          const rows = FAMILIES.filter((f) => f.archetype === archetype)
          if (!rows.length) return null
          return (
            <tbody key={archetype}>
              <tr>
                <th
                  scope="colgroup"
                  colSpan={KNOBS.length + 1}
                  className="border-b border-surface-200/70 px-3 pb-1 pt-4 text-left text-[12px] font-semibold uppercase tracking-wide text-surface-500 dark:border-white/[0.07] dark:text-surface-500"
                >
                  {t(`archetype.${archetype}`)}
                </th>
              </tr>
              {rows.map((family) => (
                <tr key={family.key}>
                  <th scope="row" className={`${TD} whitespace-nowrap text-left font-normal`}>
                    <FamilyName family={family} />
                  </th>
                  {KNOBS.map((knob) => (
                    <td key={knob.key} className="border-b border-surface-200/70 px-0.5 py-1.5 text-center dark:border-white/[0.07]">
                      <StatusCell status={family.support[knob.key]} label={`${family.name}, ${statusLabel(family.support[knob.key], knob.key)}`} />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          )
        })}
      </Table>
      <dl className="mt-3 grid max-w-[68ch] grid-cols-[auto_1fr] gap-x-3 gap-y-1.5 text-[13px] leading-relaxed text-surface-600 dark:text-surface-400">
        <dt className={`pt-0.5 ${ICON_COLOR.used}`}>{ICONS.used}</dt>
        <dd>{t('legendUsed')}</dd>
        <dt className={`pt-0.5 ${ICON_COLOR.gated_by_mosaic}`}>{ICONS.gated_by_mosaic}</dt>
        <dd>{t('legendGated')}</dd>
        <dt className="pt-0.5 text-surface-400" aria-hidden="true">&nbsp;</dt>
        <dd>{t('legendIgnored')}</dd>
      </dl>
      {augmentation.corrections.map((c) => (
        <Note key={c.knob}>
          {t('correction', {
            knob: c.knob,
            families: c.families.map((key) => FAMILIES.find((f) => f.key === key)?.name ?? key).join(', '),
          })}
        </Note>
      ))}
    </div>
  )
}

/* ── defaults by family ─────────────────────────────────────────── */

/*
 * Only knobs whose starting value differs between families get a column; the
 * rest are named once below the table. A cell is empty when the family
 * ignores the knob, since its default then changes nothing.
 */
function defaultColumns() {
  const varying = []
  const uniform = []
  for (const knob of KNOBS) {
    if (knob.group === 'classification') continue
    const values = new Set(
      FAMILIES.filter((f) => DETECTION_ARCHETYPES.has(f.archetype) && f.support[knob.key] !== 'ignored')
        .map((f) => f.defaults[knob.key]),
    )
    if (values.size > 1) varying.push(knob)
    else if (values.size === 1) uniform.push({ key: knob.key, value: [...values][0] })
  }
  return { varying, uniform }
}

export function AugDefaults() {
  const t = useTranslations('AugmentationBlocks')
  const { varying, uniform } = defaultColumns()
  const rows = FAMILIES.filter((f) => DETECTION_ARCHETYPES.has(f.archetype))
  return (
    <div>
      <Table minWidth="min-w-[560px]">
        <thead>
          <tr>
            <th scope="col" className={`${TH} align-bottom`}>{t('family')}</th>
            {varying.map((knob) => (
              <th key={knob.key} scope="col" className="border-b border-surface-300 px-2 pb-1.5 pt-1 text-right align-bottom font-mono text-[11px] font-medium text-surface-600 dark:border-white/20 dark:text-surface-400">
                <span className="ml-auto block w-4 whitespace-nowrap [writing-mode:vertical-rl] rotate-180">{knob.key}</span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((family) => (
            <tr key={family.key}>
              <th scope="row" className={`${TD} whitespace-nowrap text-left font-normal`}>
                <FamilyName family={family} />
              </th>
              {varying.map((knob) => (
                <td key={knob.key} className="border-b border-surface-200/70 px-2 py-1.5 text-right font-mono text-[12.5px] tabular-nums whitespace-nowrap text-surface-700 dark:border-white/[0.07] dark:text-surface-300">
                  {family.support[knob.key] === 'ignored' ? '' : family.defaults[knob.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </Table>
      {uniform.length > 0 && (
        <Note>
          {t.rich('uniformDefaults', {
            list: () => uniform.map((u, i) => (
              <span key={u.key}>{i > 0 && ', '}<code className={CODE}>{u.key}={u.value}</code></span>
            )),
          })}
        </Note>
      )}
    </div>
  )
}

/* ── markdown, for the .md twins and llms-full.txt ──────────────── */

function mdTable(headers, rows) {
  const head = `| ${headers.join(' | ')} |`
  const rule = `| ${headers.map(() => '---').join(' | ')} |`
  return [head, rule, ...rows.map((r) => `| ${r.join(' | ')} |`)].join('\n')
}

export function augFigureMarkdown(name, siteUrl, messages) {
  const panels = figures.figures[name] || []
  return panels.map((p) => {
    const box = p.largest_box
      ? messages.box.replace('{width}', p.largest_box[0]).replace('{height}', p.largest_box[1])
      : p.boxes === 0 ? messages.noBox : ''
    const caption = [p.label, p.note, box].filter(Boolean).join(', ')
    return `- ![${caption}](${siteUrl}${p.src}) ${caption}`
  }).join('\n')
}

export function augKnobsMarkdown(messages) {
  return mdTable(
    [messages.knob, messages.meaning, messages.cli, messages.baseDefault, messages.range],
    KNOBS.map((k) => [`\`${k.key}\``, messages.knobMeaning[k.key], k.cli.map((c) => `\`${c}\``).join(', '),
      `\`${k.default}\``, k.range]),
  )
}

export function augMatrixMarkdown(messages) {
  const cell = { used: 'used', gated_by_mosaic: 'gated', ignored: '' }
  const parts = []
  for (const archetype of ARCHETYPES) {
    const rows = FAMILIES.filter((f) => f.archetype === archetype)
    if (!rows.length) continue
    parts.push(`**${messages.archetype[archetype]}**`)
    parts.push(mdTable([messages.family, ...KNOBS.map((k) => `\`${k.key}\``)],
      rows.map((f) => [f.name, ...KNOBS.map((k) => cell[f.support[k.key]])])))
  }
  parts.push(`"used": ${messages.legendUsed} "gated": ${messages.legendGated} ${messages.legendIgnored}`)
  for (const c of augmentation.corrections) {
    const names = c.families.map((key) => FAMILIES.find((f) => f.key === key)?.name ?? key).join(', ')
    parts.push(messages.correction.replace('{knob}', c.knob).replace('{families}', names))
  }
  return parts.join('\n\n')
}

export function augDefaultsMarkdown(messages) {
  const { varying, uniform } = defaultColumns()
  const rows = FAMILIES.filter((f) => DETECTION_ARCHETYPES.has(f.archetype))
  const table = mdTable([messages.family, ...varying.map((k) => `\`${k.key}\``)],
    rows.map((f) => [f.name, ...varying.map((k) => (f.support[k.key] === 'ignored' ? '' : `\`${f.defaults[k.key]}\``))]))
  const list = uniform.map((u) => `\`${u.key}=${u.value}\``).join(', ')
  return `${table}\n\n${messages.uniformDefaults.replace(/<list><\/list>/, list)}`
}
