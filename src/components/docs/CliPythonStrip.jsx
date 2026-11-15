/*
 * One line under the page header that links a CLI command page to the Python
 * pages covering the same ground, and each of those back to the command.
 * The pairing lives in src/data/docs/cli-python-map.json; pages are untouched,
 * so translations need nothing beyond the two labels in DocsChrome.
 *
 * Server component, no client JS.
 */

import Link from 'next/link'
import { getTranslations } from 'next-intl/server'

import map from '@/data/docs/cli-python-map.json'

function matches(pattern, path) {
  return pattern.endsWith('/*') ? path.startsWith(pattern.slice(0, -1)) : pattern === path
}

export default async function CliPythonStrip({ locale, section, slug, source }) {
  if (source.archived) return null
  const t = await getTranslations({ locale, namespace: 'DocsChrome' })
  const path = `${section}/${slug}`

  let label
  let targets = []
  if (section === 'cli') {
    label = t('cliStripPython')
    targets = (map[path]?.python || []).map((p) => {
      const [s, sl] = p.split('/')
      const doc = source.getDoc(s, sl, locale)
      return doc && { href: source.href(`/docs/${p}`), text: doc.title }
    })
  } else {
    label = t('cliStripCli')
    targets = Object.entries(map)
      .filter(([key, entry]) => key !== '_comment' && [...(entry.python || []), ...(entry.reverse || [])].some((p) => matches(p, path)))
      .map(([key]) => {
        const [s, sl] = key.split('/')
        return source.getDoc(s, sl, locale) && { href: source.href(`/docs/${key}`), text: `libreyolo ${sl}`, code: true }
      })
  }
  targets = targets.filter(Boolean)
  if (!targets.length) return null

  return (
    <nav aria-label={label} className="mb-8 flex flex-wrap items-center gap-2 text-sm">
      <span className="text-surface-500 dark:text-surface-400">{label}</span>
      {targets.map((target) => (
        <Link
          key={target.href}
          href={target.href}
          className="rounded-full border border-surface-200 px-3 py-1 text-surface-700 transition-colors hover:border-primary-500 hover:text-primary-600 dark:border-white/[0.1] dark:text-surface-300 dark:hover:text-primary-400"
        >
          {target.code ? <code className="font-mono text-[0.8125rem]">{target.text}</code> : target.text}
        </Link>
      ))}
    </nav>
  )
}
