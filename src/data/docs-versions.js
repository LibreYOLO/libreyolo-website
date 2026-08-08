/*
 * The one list of documentation versions.
 *
 * This lived as five copies, one per frozen page, and drifted exactly the way
 * five copies do: every one of them still advertised v1.4.0 as "Latest" after
 * the current tree had moved on, so a reader on an old page had no way to
 * discover the new docs from the left rail. Import this instead of retyping it.
 *
 * The current version is deliberately the only entry without a version-prefixed
 * URL. The exploded docs live at /docs with no version in the path, so there is
 * one page per topic for search engines to rank rather than one per release.
 * Every other entry is frozen: still served, no longer updated, canonicalised
 * to /docs.
 */
export const CURRENT_DOCS_VERSION = 'v1.5.0'

export const docsVersions = [
  { version: 'v1.5.0', label: 'Latest', href: '/docs' },
  { version: 'v1.4.0', label: 'Frozen', href: '/docs/v1.4.0' },
  { version: 'v1.3.1', label: 'Frozen', href: '/docs/v1.3.1' },
  { version: 'v1.3.0', label: 'Frozen', href: '/docs/v1.3.0' },
  { version: 'v1.2.0', label: 'Frozen', href: '/docs/v1.2.0' },
  { version: 'v1.1.0', label: 'Frozen', href: '/docs/v1.1.0' },
]

export default docsVersions
