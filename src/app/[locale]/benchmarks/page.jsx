'use client'

import { useTranslations } from 'next-intl'
import { ArrowRight, Gauge } from 'lucide-react'
import { Link } from '@/i18n/navigation'
import RF100VLChart from '@/components/benchmarks/RF100VLChart'
import ThemedEmbed from '@/components/ThemedEmbed'

// Defaults to the live site. Point NEXT_PUBLIC_VA_URL at a local Vision
// Analysis dev server to preview embed changes before they are deployed.
const VA = process.env.NEXT_PUBLIC_VA_URL || 'https://www.visionanalysis.org'
const UTM = 'utm_source=libreyolo&utm_medium=referral&utm_campaign=benchmarks'

// Our two flagships lit up against the grey field of every other detector.
// The embed draws a curve through each highlighted family, so the size/accuracy
// trade-off reads at a glance instead of as a cloud.
const SCATTER_HIGHLIGHT = [
  'yolov9t', 'yolov9s', 'yolov9m', 'yolov9c',
  'rfdetr-n', 'rfdetr-s', 'rfdetr-m', 'rfdetr-l',
].join(',')

// Inline link inside body copy. Vision Analysis is a separate site, so those
// open in a new tab and carry the referral campaign; the RF100-VL one is an
// anchor down this same page and stays in place.
function Ref({ href, external = true, children }) {
  const className =
    'font-medium text-libre-600 underline decoration-libre-600/30 underline-offset-2 transition-colors hover:decoration-libre-600 dark:text-libre-400 dark:decoration-libre-400/30 dark:hover:decoration-libre-400'

  if (!external) {
    return (
      <a href={href} className={className}>
        {children}
      </a>
    )
  }
  return (
    <a href={href} target="_blank" rel="noopener noreferrer" className={className}>
      {children}
    </a>
  )
}

function SectionHeading({ tag, title, children }) {
  return (
    <div className="max-w-3xl">
      <p className="font-mono text-[10px] md:text-xs font-semibold uppercase tracking-[0.3em] text-libre-600 dark:text-libre-300/90">
        {tag}
      </p>
      <h2 className="mt-3 text-2xl md:text-3xl font-bold tracking-tight text-surface-900 dark:text-white">
        {title}
      </h2>
      <p className="mt-3 text-surface-600 dark:text-surface-400 leading-relaxed">{children}</p>
    </div>
  )
}

export default function BenchmarksPage() {
  const t = useTranslations('Benchmarks')
  const rf = useTranslations('RF100VL')
  return (
    <main className="min-h-screen bg-white dark:bg-surface-950">
      {/* Hero */}
      <section className="relative overflow-hidden border-b border-surface-200 dark:border-surface-800">
        <div className="absolute inset-0 bg-gradient-to-br from-libre-500/10 via-transparent to-transparent" />
        <div className="relative mx-auto max-w-6xl px-6 pt-24 pb-8 md:pt-28 md:pb-10">
          <h1
            className="max-w-3xl text-4xl md:text-5xl font-bold tracking-tight leading-[1.05] text-surface-900 dark:text-white"
          >
            {t('title')}
          </h1>
          <p className="mt-4 max-w-3xl text-base leading-relaxed text-surface-600 dark:text-surface-400">
            {t('intro')}
          </p>
        </div>
      </section>

      {/* COCO accuracy is a separate checkpoint benchmark. */}
      <section id="coco" className="mx-auto max-w-6xl px-6 pt-12 pb-12 md:pt-16 md:pb-16">
        <SectionHeading tag="COCO" title={t('cocoTitle')}>
          {t.rich('cocoBody', {
            vision: (chunks) => <Ref href={`${VA}/?${UTM}`}>{chunks}</Ref>,
            coco: (chunks) => <Ref href={`${VA}/?${UTM}`}>{chunks}</Ref>,
            latency: (chunks) => <Ref href={`${VA}/hardware?${UTM}`}>{chunks}</Ref>,
            rf: (chunks) => <Ref href="#rf100-vl" external={false}>{chunks}</Ref>,
          })}
        </SectionHeading>

        <div className="mt-8 overflow-hidden rounded-2xl border border-surface-200 dark:border-surface-800">
          <ThemedEmbed
            src={`${VA}/embed/scatter?highlight=${SCATTER_HIGHLIGHT}&title=Accuracy%20vs%20parameters&subtitle=YOLOv9%20and%20RF-DETR%20against%20every%20other%20detector%20on%20COCO%20val2017`}
            title={t('cocoIframeTitle')}
            className="w-full block"
            style={{ border: 0, overflow: 'hidden', aspectRatio: '640 / 400' }}
          />
        </div>

        <a href={`${VA}/?${UTM}`} target="_blank" rel="noopener noreferrer" className="mt-4 inline-flex items-center gap-2 text-sm font-medium text-libre-600 hover:underline dark:text-libre-400">
          {t('openLeaderboard')}<ArrowRight className="h-4 w-4" />
        </a>
      </section>

      <section id="rf100-vl" className="mx-auto max-w-6xl scroll-mt-24 px-6 pb-12 md:pb-16">
        <SectionHeading tag="RF100-VL" title={t('rfTitle')}>
          {t('rfIntro')}
        </SectionHeading>
        <RF100VLChart />
        <div className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-3">
          <Link href="/articles/rf100vl-benchmark" className="inline-flex items-center gap-2 text-sm font-medium text-libre-600 hover:underline dark:text-libre-400">
            {rf('benchmarkReport')}<ArrowRight className="h-4 w-4" />
          </Link>
          <a href="https://github.com/LibreYOLO/vision-analysis-benchmark/tree/rf100vl-harness" target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-2 text-sm font-medium text-libre-600 hover:underline dark:text-libre-400">
            {t('rfHarness')}<ArrowRight className="h-4 w-4" />
          </a>
        </div>
        <details className="mt-5 max-w-3xl text-sm leading-relaxed text-surface-500 dark:text-surface-400">
          <summary className="w-fit cursor-pointer rounded font-medium hover:text-surface-900 focus-visible:outline focus-visible:outline-2 focus-visible:outline-libre-500 dark:hover:text-white">{t('rfMethodology')}</summary>
          <div className="mt-3 space-y-2">
            <p>{t('rfGeneralisation')}</p>
            <p>{t('rfChartNote')}</p>
            <p>{t('rfAdvice')}</p>
            <p>{t('rfCaveat')}</p>
          </div>
        </details>
      </section>

      <section className="border-t border-surface-200 dark:border-surface-800">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-6 px-6 py-10">
          <div>
            <h2 className="text-lg font-semibold text-surface-900 dark:text-white">{t('latencyTitle')}</h2>
            <p className="mt-2 text-sm text-surface-500 dark:text-surface-400">{t('latencyShort')}</p>
          </div>
          <a href={`${VA}/hardware?${UTM}`} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-2 text-sm font-medium text-libre-600 hover:underline dark:text-libre-400">
            <Gauge className="h-4 w-4" />{t('latencyCta')}<ArrowRight className="h-4 w-4" />
          </a>
        </div>
      </section>
    </main>
  )
}
