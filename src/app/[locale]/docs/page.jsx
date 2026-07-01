import { redirect } from '@/i18n/navigation'

// `/docs` has no content of its own: it always points at the current default
// version so the URL a reader ends up on carries the version (e.g. /docs/v1.3.0).
// Update DEFAULT_DOCS_VERSION when a newer version becomes the default.
const DEFAULT_DOCS_VERSION = 'v1.3.0'

export default async function DocsIndex({ params }) {
  const { locale } = await params
  redirect({ href: `/docs/${DEFAULT_DOCS_VERSION}`, locale })
}
