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
/*
 * The tree at /docs documents the development branch, which is ahead of the
 * released package: it carries model families and export formats the current
 * release does not have. Until that release is tagged, nothing on the site may
 * present these pages as documentation for a shipped version.
 *
 * On tag day: set DOCS_PRERELEASE to false, set DOCS_CURRENT_VERSION to the
 * new number, and move the previous release's entry from Latest to Frozen.
 */
export const DOCS_PRERELEASE = false

// The version a reader gets from `pip install libreyolo` today.
export const DOCS_CURRENT_VERSION = '1.5.0'

export const CURRENT_DOCS_VERSION = 'v1.5.0'

export const docsVersions = [
  { version: 'v1.5.0', label: DOCS_PRERELEASE ? 'Next release' : 'Latest', href: '/docs' },
  { version: 'v1.4.0', label: DOCS_PRERELEASE ? 'Current' : 'Frozen', href: '/docs/v1.4.0' },
  { version: 'v1.3.1', label: 'Frozen', href: '/docs/v1.3.1' },
  { version: 'v1.3.0', label: 'Frozen', href: '/docs/v1.3.0' },
  { version: 'v1.2.0', label: 'Frozen', href: '/docs/v1.2.0' },
  { version: 'v1.1.0', label: 'Frozen', href: '/docs/v1.1.0' },
]

export default docsVersions
