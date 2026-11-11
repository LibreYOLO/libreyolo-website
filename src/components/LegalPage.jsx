// Plain server-rendered page for short legal texts (privacy, terms).
// English only: the sitemap lists the English URL and other locales canonicalize to it.
export default function LegalPage({ title, updated, sections }) {
  return (
    <div className="pt-24 lg:pt-32 pb-16">
      <article className="max-w-3xl mx-auto px-6 lg:px-8 text-surface-700 dark:text-surface-300">
        <h1 className="text-4xl font-bold text-surface-800 dark:text-white mb-2">{title}</h1>
        <p className="text-sm text-surface-500 mb-10">Last updated {updated}</p>
        {sections.map((s) => (
          <section key={s.heading} className="mb-8">
            <h2 className="text-xl font-semibold text-surface-800 dark:text-white mb-3">{s.heading}</h2>
            {s.body.map((p, i) => (
              <p key={i} className="leading-relaxed mb-3">{p}</p>
            ))}
          </section>
        ))}
      </article>
    </div>
  )
}
