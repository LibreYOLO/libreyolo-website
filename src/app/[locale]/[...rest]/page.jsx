import { notFound } from 'next/navigation'

// Catch-all so unmatched paths under a locale (e.g. /zh/nope) render the
// localized not-found page instead of Next.js's bare global 404.
export default function CatchAllPage() {
  notFound()
}
