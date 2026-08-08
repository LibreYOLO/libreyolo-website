import { docsOgImage, OG_SIZE, OG_CONTENT_TYPE } from '@/lib/og-docs'

export const size = OG_SIZE
export const contentType = OG_CONTENT_TYPE
export const alt = 'LibreYOLO documentation'

// The export section index. A real directory shadows [section] when Next
// resolves this path, so the card has to live here rather than there.
export default function Image() {
  return docsOgImage({ title: "Export and deploy", description: '', section: 'export' })
}
