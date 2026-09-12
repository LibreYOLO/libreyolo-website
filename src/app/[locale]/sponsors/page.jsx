import { Heart, ExternalLink } from 'lucide-react'
import { GITHUB_URL } from '@/lib/links'

const SPONSOR_URL = 'https://github.com/sponsors/EHxuban11'
const OPEN_COLLECTIVE_URL = 'https://opencollective.com/libreyolo'
const POLICY_URL = `${GITHUB_URL}/blob/HEAD/SPONSORS.md`

const ROWS = [
  ['Individuals', 'any amount', 'Name in the supporters list below.'],
  ['Companies', '$100 per month', 'Logo and link in the README and on libreyolo.com.'],
  ['Companies', '$500 per month', 'The above, plus named in every release post and in the model card of every weight released while the sponsorship is active, as "Training compute sponsored by".'],
  ['Companies', 'custom', 'Ask.'],
  ['Hardware manufacturers', 'a device', 'LibreYOLO makes the export formats that device supports work on it, or adds export support for that specific device. Logo and link in the README and on libreyolo.com. The device stays with the project and the benchmark results are published as measured.'],
]

const card =
  'rounded-xl border border-surface-200 dark:border-white/10 bg-white dark:bg-white/[0.03] p-6 sm:p-8'
const h2 = 'text-xl font-semibold text-surface-900 dark:text-white mb-3'
const p = 'text-sm leading-relaxed text-surface-600 dark:text-surface-300'
const a = 'text-libre-600 dark:text-libre-400 hover:underline'

function Ext({ href, children }) {
  return (
    <a href={href} className={a} target="_blank" rel="noopener noreferrer">
      {children}
    </a>
  )
}

export default function Sponsors() {
  return (
    <div className="pt-24 lg:pt-32 pb-16">
      <div className="max-w-3xl mx-auto px-6 lg:px-8">
        <div className="text-center mb-12">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-libre-500/10 border border-libre-500/20 text-libre-700 dark:text-libre-400 text-sm font-medium mb-6">
            <Heart className="w-4 h-4" />
            Version 1, September 2026
          </div>
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-surface-800 dark:text-white mb-6">
            The LibreYOLO <span className="text-libre-600 dark:text-libre-400">Sponsorship Program</span>
          </h1>
          <p className="text-lg text-surface-600 dark:text-surface-400 max-w-2xl mx-auto mb-8">
            LibreYOLO is an MIT-licensed computer vision library. Its goal is to
            make machine learning models more accessible, with a particular
            focus on computer vision.
          </p>
          <a
            href={SPONSOR_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="btn-primary inline-flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-libre-500 to-libre-600 rounded-xl text-white font-semibold text-lg"
          >
            Sponsor on GitHub
            <ExternalLink className="w-5 h-5" />
          </a>
          <p className="text-xs text-surface-500 dark:text-surface-400 mt-3">
            The canonical text of this program is{' '}
            <Ext href={POLICY_URL}>SPONSORS.md</Ext> in the repository.
          </p>
        </div>

        <section className={`${card} mb-6`}>
          <h2 className={h2}>Abstract</h2>
          <p className={`${p} mb-3`}>
            After approximately nine months of building in public in an
            economically lean fashion, most of the low-hanging fruit has been
            taken, while some real advancements that could revolutionize the
            computer vision field have been avoided because of their cost. Our
            ambition is to address these, thanks to the LibreYOLO Sponsorship
            Program.
          </p>
          <p className={p}>
            This document defines how the project accepts money, what it does
            with it, and what sponsors receive.
          </p>
        </section>

        <section className={`${card} mb-6`}>
          <h2 className={h2}>1. Purpose</h2>
          <p className={`${p} mb-3`}>
            The program exists to fund work that LibreYOLO cannot do without
            money. The list is detailed in section 2.
          </p>
          <p className={p}>
            The program is not a way to pay the maintainer. Funds go to compute,
            benchmarks and infrastructure, as listed in section 2.
          </p>
        </section>

        <section className={`${card} mb-6`}>
          <h2 className={h2}>2. What needs funding</h2>
          <ol className={`${p} space-y-3 list-decimal pl-5`}>
            <li>
              <strong className="text-surface-900 dark:text-white">LibreYOLO26.</strong>{' '}
              The first LibreYOLO model built to be competitive, trained from
              scratch and released under MIT in 2026. The bar is YOLOv8-class on
              two measures: accuracy, as COCO mAP at comparable size and speed,
              and generalization, as the RF100-VL score across 100 real-world
              datasets.{' '}
              <strong className="text-surface-900 dark:text-white">LibreYOLO27</strong>{' '}
              follows in 2027 with a higher bar.
            </li>
            <li>
              <strong className="text-surface-900 dark:text-white">Models retrained from scratch.</strong>{' '}
              Several families in the library have permissively licensed code
              but weights that users cannot ship freely, while the datasets they
              need are open. Retraining them on open data gives the community
              MIT weights. This work starts with YOLO-NAS.
            </li>
            <li>
              <strong className="text-surface-900 dark:text-white">Training and inference support for vision-language models.</strong>{' '}
              LibreYOLO aims to be the reference library for training VLMs, with
              the same train, val and predict API as its detectors and the same
              dataset formats.
            </li>
            <li>
              <strong className="text-surface-900 dark:text-white">Benchmarks for visionanalysis.org.</strong>{' '}
              Vision Analysis has the ambition to become the go-to reference for
              benchmarks and the model registry for computer vision. We want to
              benchmark every model on all relevant hardware and cover every
              computer vision task. Ultimately we want to become the
              &ldquo;Artificial Analysis of computer vision&rdquo;.
            </li>
          </ol>
          <p className={`${p} mt-3`}>Other needs may arise as the library advances.</p>
        </section>

        <section className={`${card} mb-6`}>
          <h2 className={h2}>3. Sponsors</h2>
          <p className={`${p} mb-4`}>
            Two words are used below. Individuals who contribute are supporters
            and are listed by name. Companies that contribute are sponsors and
            are listed by logo.
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left">
              <thead>
                <tr className="text-surface-500 dark:text-surface-400 border-b border-surface-200 dark:border-white/10">
                  <th className="py-2 pr-4 font-medium">Who</th>
                  <th className="py-2 pr-4 font-medium">Contribution</th>
                  <th className="py-2 font-medium">Recognition</th>
                </tr>
              </thead>
              <tbody className="text-surface-700 dark:text-surface-300 align-top">
                {ROWS.map(([who, amount, gets], i) => (
                  <tr key={i} className="border-b border-surface-100 dark:border-white/5">
                    <td className="py-3 pr-4 font-semibold text-surface-900 dark:text-white whitespace-nowrap">{who}</td>
                    <td className="py-3 pr-4 whitespace-nowrap">{amount}</td>
                    <td className="py-3">{gets}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className={`${p} mt-4`}>
            Contributions go through <Ext href={SPONSOR_URL}>GitHub Sponsors</Ext>,
            monthly or one-time. Companies that need an invoice or a bank
            transfer contribute through{' '}
            <Ext href={OPEN_COLLECTIVE_URL}>Open Collective</Ext>, where every
            transaction is public. GPU providers may contribute compute instead
            of money; it counts at market value.
          </p>
        </section>

        <section className={`${card} mb-6`}>
          <h2 className={h2}>4. Limits</h2>
          <ul className={`${p} space-y-3`}>
            <li>
              <span className="font-semibold text-surface-900 dark:text-white">No influence on the roadmap.</span>{' '}
              Sponsorship does not create any ability to direct what LibreYOLO
              builds, which issues get attention, or which pull requests are
              merged.
            </li>
            <li>
              <span className="font-semibold text-surface-900 dark:text-white">No endorsement.</span>{' '}
              Being listed does not mean LibreYOLO endorses a sponsor, and
              sponsoring does not mean the sponsor endorses LibreYOLO&rsquo;s
              decisions.
            </li>
          </ul>
        </section>

        <section className={`${card} mb-6`}>
          <h2 className={h2}>5. Transparency</h2>
          <p className={p}>
            All sponsorship money is loaded as credit into the project&rsquo;s
            Vast.ai account. Nothing is tracked per experiment. Once a year the
            program document gets one line: how much came in and how much was
            loaded onto Vast.ai, with the Vast.ai billing history as the record.
            The sponsor list on GitHub Sponsors is public.
          </p>
        </section>

        <section className={`${card} mb-6`}>
          <h2 className={h2}>6. Changes</h2>
          <p className={p}>
            Changes to this program are made by pull request and announced in
            the next release notes.
          </p>
        </section>

        <section className={`${card} mb-6`}>
          <h2 className={h2}>Sponsors (companies)</h2>
          <p className={`${p} mb-2`}>Companies and hardware manufacturers that contribute, listed by logo.</p>
          <p className={p}>None yet.</p>
        </section>

        <section className={card}>
          <h2 className={h2}>Supporters (individuals)</h2>
          <p className={`${p} mb-2`}>People who contribute any amount, listed by name.</p>
          <p className={p}>None yet.</p>
        </section>
      </div>
    </div>
  )
}
