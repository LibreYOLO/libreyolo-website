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
  return String(text)
    .toLowerCase()
    .replace(/[^\w\s-]/g, '')
    .trim()
    .replace(/\s+/g, '-')
}

export default slugifyHeading
