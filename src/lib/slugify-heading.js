/*
 * Heading anchors, derived one way for everyone.
 *
 * Three places need the same id for the same heading: the rendered markdown
 * that stamps it, the "on this page" rail that links to it, and search
 * results that deep-link to it. Two of those run in the browser, so this
 * cannot live in `lib/docs.js`, which reads the filesystem.
 *
 * Any drift between copies of this function shows up as a link that scrolls
 * nowhere, which is the kind of bug nobody files.
 */
export function slugifyHeading(text) {
  const normalized = String(text).toLowerCase()
  const existingSlug = normalized
    .replace(/[^\w\s-]/g, '')
    .trim()
    .replace(/\s+/g, '-')

  // Preserve every existing Latin/ASCII anchor. Only fall back to a
  // Unicode-aware slug when the old implementation would have returned an
  // empty string, as it did for headings written entirely in CJK and other
  // non-Latin scripts.
  if (existingSlug) return existingSlug

  return normalized
    .normalize('NFKC')
    .replace(/[^\p{Letter}\p{Number}_\s-]/gu, '')
    .trim()
    .replace(/\s+/g, '-')
}

export default slugifyHeading
