import Link from 'next/link'

/*
 * Banner for the frozen single-page docs (v1.1.0 through v1.4.0).
 *
 * Those pages stay reachable because they have inbound links and readers pinned
 * to old releases, but they are no longer updated and `/docs` no longer
 * redirects to them. Without this a reader has no way to tell they are looking
 * at superseded documentation.
 *
 * It pairs with a canonical pointing at /docs. Note the deliberate absence of
 * noindex: combining noindex with a canonical is a documented way to get the
 * wrong page dropped from an index, so these pages stay indexable and simply
 * point at the current tree.
 */
export default function FrozenVersionBanner({ version }) {
  return (
    <div className="border-b border-amber-500/30 bg-amber-500/[0.07] px-6 py-3">
      <p className="mx-auto max-w-4xl text-[13.5px] leading-relaxed text-surface-700 dark:text-surface-300">
        These are the frozen docs for <strong className="font-semibold">{version}</strong>. They are
        kept for anyone pinned to that release and are no longer updated.{' '}
        <Link href="/docs" className="font-medium text-libre-700 underline underline-offset-2 dark:text-libre-400">
          Read the current documentation
        </Link>
        .
      </p>
    </div>
  )
}
