export const REDDIT_URL = 'https://www.reddit.com/r/LibreYOLO/'
export const GITHUB_URL = 'https://github.com/LibreYOLO/libreyolo'

// Links to our own GitHub keep the referrer, so GitHub's traffic page can
// credit visits to libreyolo.com. Every other external link stays noreferrer.
export function externalRel(href) {
  return href?.startsWith('https://github.com/LibreYOLO/') ? 'noopener' : 'noopener noreferrer'
}
