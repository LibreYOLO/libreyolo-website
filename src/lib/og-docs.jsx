import { ImageResponse } from 'next/og'
import fs from 'fs'
import path from 'path'

/*
 * The card a docs page shows when someone pastes its link into Slack, a
 * chat, or a social post.
 *
 * Until now every page on the site shared one image, so a link to the RF-DETR
 * page and a link to the CLI reference were visually identical. The share
 * surface is often the first thing a reader sees of a page, and an identical
 * card tells them nothing about which of 171 pages they are being sent.
 *
 * House rules apply here as much as on the page: no badge rows, no stat
 * tiles, no decoration standing in for information. The card carries the
 * section it belongs to, the page title, and the one-line description. That
 * is the same hierarchy the page itself opens with.
 *
 * Satori (which renders this) supports a subset of CSS and requires every
 * element to declare display explicitly. It cannot fetch anything, so the
 * logo is inlined as a data URI.
 */

export const OG_SIZE = { width: 1200, height: 630 }
export const OG_CONTENT_TYPE = 'image/png'

const SECTION_LABEL = {
  start: 'Get started',
  tasks: 'Tasks',
  models: 'Models',
  train: 'Training',
  predict: 'Prediction',
  export: 'Export and deploy',
  cli: 'CLI',
  reference: 'Reference',
}

function logoDataUri() {
  const data = fs.readFileSync(path.join(process.cwd(), 'public', 'logo.png'))
  return `data:image/png;base64,${data.toString('base64')}`
}

// Cut on a word, never mid-word. A card reading "all MIT-licen..." looks like
// a rendering bug rather than a deliberate summary.
function clamp(text, limit) {
  const s = String(text)
  if (s.length <= limit) return s
  const cut = s.slice(0, limit)
  const lastSpace = cut.lastIndexOf(' ')
  return `${(lastSpace > limit * 0.6 ? cut.slice(0, lastSpace) : cut).replace(/[,;:.\s]+$/, '')}...`
}

// Long titles have to shrink rather than wrap into the description.
function titleSize(title) {
  const n = String(title).length
  if (n <= 22) return 82
  if (n <= 34) return 68
  if (n <= 48) return 56
  return 46
}

export function docsOgImage({ title, description, section }) {
  const eyebrow = SECTION_LABEL[section] || 'Documentation'
  const blurb = String(description || '')

  return new ImageResponse(
    (
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
          padding: '64px 72px',
          backgroundColor: '#020617',
          backgroundImage:
            'radial-gradient(circle at 12% 8%, rgba(8, 145, 178, 0.28) 0%, transparent 46%)',
          fontFamily: 'sans-serif',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <img src={logoDataUri()} width={48} height={48} style={{ borderRadius: 10 }} />
          <div style={{ display: 'flex', fontSize: 26, fontWeight: 600, color: '#ffffff' }}>
            LibreYOLO
          </div>
          <div style={{ display: 'flex', fontSize: 26, color: '#334155' }}>/</div>
          <div style={{ display: 'flex', fontSize: 24, color: '#67e8f9', letterSpacing: 0.5 }}>
            {eyebrow}
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column' }}>
          <div
            style={{
              display: 'flex',
              fontSize: titleSize(title),
              fontWeight: 700,
              color: '#ffffff',
              lineHeight: 1.12,
            }}
          >
            {title}
          </div>
          {blurb && (
            <div
              style={{
                display: 'flex',
                marginTop: 22,
                fontSize: 27,
                lineHeight: 1.4,
                color: '#94a3b8',
                maxWidth: 940,
              }}
            >
              {clamp(blurb, 150)}
            </div>
          )}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <div style={{ display: 'flex', width: 40, height: 3, backgroundColor: '#0891b2' }} />
          <div style={{ display: 'flex', fontSize: 22, color: '#64748b' }}>
            libreyolo.com/docs
          </div>
        </div>
      </div>
    ),
    { ...OG_SIZE }
  )
}
