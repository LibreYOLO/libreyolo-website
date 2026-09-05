import { RF100VL_DATASETS } from './datasets'
import RESULTS from './results-summary.json'

const SAMPLES = ['nih-xray', 'deeppcb', 'aerial-airport', 'mahjong', 'underwater-objects', 'wildfire-smoke']
  .map((name) => RF100VL_DATASETS.find((dataset) => dataset.name === name))

export default function RF100VLHero({ title, author, dateISO, dateLabel, backLink }) {
  return (
    <header className="border-b border-surface-200 dark:border-surface-800 bg-surface-50 dark:bg-surface-950 pt-28 pb-10 md:pt-36 md:pb-14">
      <div className="max-w-5xl mx-auto px-6 lg:px-8">
        {backLink && <div className="mb-10">{backLink}</div>}
        <p className="font-mono text-xs uppercase tracking-[0.18em] text-libre-700 dark:text-libre-400">
          RF100-VL / completed campaign / August 2026
        </p>
        <h1 className="max-w-4xl mt-5 text-4xl md:text-6xl font-bold tracking-tight leading-[1.08] text-surface-900 dark:text-white">
          {title}
        </h1>
        <p className="max-w-2xl mt-6 text-lg leading-relaxed text-surface-600 dark:text-surface-300">
          How 17 detector configurations fine-tuned across 100 datasets, and what the runs taught us about LibreYOLO.
        </p>
        <div className="mt-6 flex flex-wrap gap-x-3 gap-y-1 text-sm text-surface-500 dark:text-surface-400">
          <span>{author} · LibreYOLO</span>
          <time dateTime={dateISO}>{dateLabel}</time>
        </div>
        <nav aria-label="Report sections" className="mt-8 flex flex-wrap gap-x-6 gap-y-3 text-sm font-semibold text-libre-700 dark:text-libre-400">
          <a className="underline underline-offset-4" href="#results">Results</a>
          <a className="underline underline-offset-4" href="#methodology">Methodology</a>
          <a className="underline underline-offset-4" href="#thank-you-roboflow">Thank you, Roboflow</a>
          <a className="underline underline-offset-4" href="https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/tree/main">Public artifacts</a>
        </nav>
        <div className="mt-10 grid grid-cols-3 gap-4 border-t border-surface-200 dark:border-surface-800 pt-6">
          {[[RESULTS.length, 'configurations'], [100, 'datasets each'], [RESULTS.length * 100, 'completed fine-tunes']].map(([value, label]) => (
            <div key={label}>
              <div className="text-3xl md:text-4xl font-mono tabular-nums font-semibold text-surface-900 dark:text-white">{value.toLocaleString('en-US')}</div>
              <div className="mt-2 text-xs md:text-sm text-surface-500 dark:text-surface-400">{label}</div>
            </div>
          ))}
        </div>
        <div className="mt-9 grid grid-cols-3 sm:grid-cols-6 gap-2">
          {SAMPLES.map((dataset) => (
            <figure key={dataset.name} className="min-w-0">
              <img src={dataset.img} alt={`Annotated RF100-VL sample: ${dataset.name}`} width="240" height="160" className="w-full aspect-[3/2] object-cover rounded-md bg-surface-200 dark:bg-surface-800" />
              <figcaption className="mt-2 text-[10px] sm:text-xs text-surface-500 dark:text-surface-400 break-words">{dataset.name}</figcaption>
            </figure>
          ))}
        </div>
      </div>
    </header>
  )
}
