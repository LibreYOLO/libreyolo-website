import { ImageResponse } from 'next/og'
import fs from 'fs'
import path from 'path'

export const size = { width: 1200, height: 630 }
export const contentType = 'image/png'
export const alt = 'LibreYOLO: MIT-Licensed Object Detection'

export default function OpenGraphImage() {
  const logoData = fs.readFileSync(path.join(process.cwd(), 'public', 'logo.png'))
  const logoSrc = `data:image/png;base64,${logoData.toString('base64')}`

  return new ImageResponse(
    (
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: '#020617',
          backgroundImage:
            'radial-gradient(circle at 25% 25%, rgba(8, 145, 178, 0.25) 0%, transparent 50%), radial-gradient(circle at 75% 75%, rgba(16, 185, 129, 0.15) 0%, transparent 50%)',
          fontFamily: 'sans-serif',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 36 }}>
          <img src={logoSrc} width={180} height={180} style={{ borderRadius: 24 }} />
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <div style={{ fontSize: 92, fontWeight: 700, color: '#ffffff', lineHeight: 1.1 }}>
              LibreYOLO
            </div>
            <div style={{ fontSize: 36, color: '#67e8f9', marginTop: 8 }}>
              MIT-Licensed Object Detection
            </div>
          </div>
        </div>
        <div
          style={{
            fontSize: 26,
            color: '#94a3b8',
            marginTop: 48,
            maxWidth: 900,
            textAlign: 'center',
          }}
        >
          Train and deploy state-of-the-art YOLO models in commercial applications, free from AGPL restrictions
        </div>
        <div style={{ fontSize: 24, color: '#0891b2', marginTop: 36 }}>libreyolo.com</div>
      </div>
    ),
    { ...size }
  )
}
