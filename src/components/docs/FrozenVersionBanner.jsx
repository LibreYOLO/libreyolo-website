import Link from 'next/link'
import { DOCS_PRERELEASE, DOCS_CURRENT_VERSION } from '@/data/docs-versions'

/*
 * Banner for the single-page docs kept at /docs/vX.Y.Z.
 *
 * These pages stay reachable because they have inbound links and readers
 * pinned to a given release. Without a banner a reader has no way to tell
 * which version they are looking at.
 *
 * The CURRENT release gets NO banner. It is the page readers are sent to, it
 * already carries its own in-page notice naming the version, and the sidebar
 * version menu offers the next-release tree. A second bar above it repeated
 * that for no one and added a band of colour to the top of the site.
 *
 * A superseded version does get one, because nothing else on the page tells
 * the reader they are on outdated documentation.
 *
 * It pairs with a canonical pointing at /docs. Note the deliberate absence of
 * noindex: combining noindex with a canonical is a documented way to get the
 * wrong page dropped from an index, so these pages stay indexable and simply
 * point at the current tree.
 */
export default function FrozenVersionBanner({ version }) {
  if (DOCS_PRERELEASE && version === `v${DOCS_CURRENT_VERSION}`) return null

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
