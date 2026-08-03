import { Construction } from 'lucide-react'

/**
 * Loud, unmissable banner for an article that is published but not finished.
 *
 * Deliberately not subtle: it sits above the first paragraph, uses warning
 * colors rather than the brand palette, and says both things a reader needs
 * to know before trusting anything below it, that the draft was AI-generated
 * and that it is still changing.
 */
export default function UnderConstruction() {
  return (
    <aside
      role="note"
      aria-label="This article is unfinished and AI-generated"
      className="not-prose my-8 rounded-2xl border-2 border-amber-500 bg-amber-50 dark:bg-amber-950/40 px-5 py-5 md:px-7 md:py-6"
      style={{ width: 'min(96vw, 1080px)', marginLeft: 'calc(50% - min(48vw, 540px))' }}
    >
      <div className="flex flex-col sm:flex-row sm:items-center gap-4">
        <Construction
          className="w-10 h-10 md:w-12 md:h-12 shrink-0 text-amber-600 dark:text-amber-400"
          aria-hidden="true"
        />
        <div className="min-w-0">
          <p className="text-xl md:text-3xl font-extrabold uppercase tracking-tight leading-tight text-amber-900 dark:text-amber-200">
            AI-generated article, under construction
          </p>
          <p className="mt-2 text-sm md:text-base text-amber-900/80 dark:text-amber-200/80 leading-relaxed">
            This draft was written with AI assistance and is still being edited and
            fact-checked. Wording and framing will change. The benchmark numbers shown are
            the real, published campaign results, but treat everything around them as a
            work in progress.
          </p>
        </div>
      </div>
    </aside>
  )
}
