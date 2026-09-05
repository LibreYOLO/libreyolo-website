export default function RF100VLHero({ title, author, dateISO, dateLabel, backLink }) {
  return (
    <header className="border-b border-surface-200 dark:border-surface-800 bg-surface-50 dark:bg-surface-950 pt-28 pb-10 md:pt-36 md:pb-14">
      <div className="max-w-5xl mx-auto px-6 lg:px-8">
        {backLink && <div className="mb-10">{backLink}</div>}
        <p className="font-mono text-xs uppercase tracking-[0.18em] text-libre-700 dark:text-libre-400">
          RF100-VL benchmark results
        </p>
        <h1 className="max-w-4xl mt-5 text-4xl md:text-6xl font-bold tracking-tight leading-[1.08] text-surface-900 dark:text-white">
          {title}
        </h1>
        <p className="max-w-2xl mt-6 text-lg leading-relaxed text-surface-600 dark:text-surface-300">
          Generalization beyond COCO: how well object detectors adapt to new datasets.
        </p>
        <div className="mt-6 flex flex-wrap gap-x-3 gap-y-1 text-sm text-surface-500 dark:text-surface-400">
          <span>{author} · LibreYOLO</span>
          <time dateTime={dateISO}>{dateLabel}</time>
        </div>
        <nav aria-label="Report sections" className="mt-8 flex flex-wrap gap-x-6 gap-y-3 text-sm font-semibold text-libre-700 dark:text-libre-400">
          <a className="underline underline-offset-4" href="#results">Results</a>
          <a className="underline underline-offset-4" href="#methodology">Methodology</a>
          <a className="underline underline-offset-4" href="#what-the-workload-improved">What improved</a>
          <a className="underline underline-offset-4" href="#thank-you-roboflow">Thank you, Roboflow</a>
          <a className="underline underline-offset-4" href="https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/tree/main">Public artifacts</a>
        </nav>
      </div>
    </header>
  )
}
