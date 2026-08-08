import Link from 'next/link'
import { DOCS_PRERELEASE, DOCS_CURRENT_VERSION } from '@/data/docs-versions'

/*
 * Banner for the single-page docs kept at /docs/vX.Y.Z.
 *
 * These pages stay reachable because they have inbound links and readers
 * pinned to a given release. Without a banner a reader has no way to tell
 * which version they are looking at.
 *
 * There are two cases and they must not be worded the same. A superseded
 * version is frozen: no longer updated, and the reader should be sent to the
 * current tree. The CURRENT release is not frozen at all, and while the
 * exploded tree at /docs still documents an untagged version, this page is
 * the accurate one: it describes what `pip install libreyolo` actually gives
 * you. Telling that reader they are looking at outdated docs would be false
 * and would push them toward pages describing families their install lacks.
 *
 * Both variants pair with a canonical pointing at /docs. Note the deliberate
 * absence of noindex: combining noindex with a canonical is a documented way
 * to get the wrong page dropped from an index, so these pages stay indexable
 * and simply point at the current tree.
 */
export default function FrozenVersionBanner({ version }) {
  const isCurrentRelease = DOCS_PRERELEASE && version === `v${DOCS_CURRENT_VERSION}`

  if (isCurrentRelease) {
    return (
      <div className="border-b border-emerald-500/30 bg-emerald-500/[0.07] px-6 py-3">
        <p className="mx-auto max-w-4xl text-[13.5px] leading-relaxed text-surface-700 dark:text-surface-300">
          These are the docs for <strong className="font-semibold">{version}</strong>, the current
          release, and they match what <code className="font-mono text-[12.5px]">pip install libreyolo</code> installs
          today.{' '}
          <Link
            href="/docs"
            className="font-medium text-libre-700 underline underline-offset-2 dark:text-libre-400"
          >
            Preview the docs for the next release
          </Link>
          , which cover models and export formats this version does not have yet.
        </p>
      </div>
    )
  }

  return (
    <div className="border-b border-amber-500/30 bg-amber-500/[0.07] px-6 py-3">
      <p className="mx-auto max-w-4xl text-[13.5px] leading-relaxed text-surface-700 dark:text-surface-300">
        These are the frozen docs for <strong className="font-semibold">{version}</strong>. They are
        kept for anyone pinned to that release and are no longer updated.{' '}
        <Link
          href={DOCS_PRERELEASE ? `/docs/v${DOCS_CURRENT_VERSION}` : '/docs'}
          className="font-medium text-libre-700 underline underline-offset-2 dark:text-libre-400"
        >
          Read the docs for {DOCS_PRERELEASE ? `v${DOCS_CURRENT_VERSION}, the current release` : 'the current release'}
        </Link>
        .
      </p>
    </div>
  )
}
