'use client'

import { motion } from 'framer-motion'
import { useTranslations } from 'next-intl'
import { ArrowRight, BarChart3, Gauge } from 'lucide-react'
import { Link } from '@/i18n/navigation'
import RF100VLPanel from '@/components/articles/rf100vl/RF100VLPanel'
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
  return (
    <main className="min-h-screen bg-white dark:bg-surface-950">
      {/* Hero */}
      <section className="relative overflow-hidden border-b border-surface-200 dark:border-surface-800">
        <div className="absolute inset-0 bg-gradient-to-br from-libre-500/10 via-transparent to-transparent" />
        <div className="relative mx-auto max-w-6xl px-6 pt-24 pb-10 md:pt-32 md:pb-14">
          <motion.h1
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            className="max-w-3xl text-4xl md:text-6xl font-bold tracking-tight leading-[1.05] text-surface-900 dark:text-white"
          >
            {t('title')}
          </motion.h1>
        </div>
      </section>

      {/* COCO first: the baseline everyone already knows, via a Vision
          Analysis embed. RF100-VL then lands as the "and off COCO?" follow-up. */}
      <section className="mx-auto max-w-6xl px-6 pt-16 pb-16 md:pt-24 md:pb-24">
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

        <div className="mt-6 flex flex-wrap gap-3">
          <a
            href={`${VA}/?${UTM}`}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 rounded-full border border-surface-900 bg-surface-900 px-5 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-surface-700 dark:border-white/15 dark:bg-white/5 dark:font-medium dark:text-surface-200 dark:hover:bg-white/10"
          >
            <BarChart3 className="h-4 w-4" />
            {t('openLeaderboard')}
          </a>
          <Link
            href="/models"
            className="inline-flex items-center gap-2 rounded-full border border-surface-200 px-5 py-2.5 text-sm font-semibold text-surface-700 transition-colors hover:border-libre-500 hover:text-libre-600 dark:border-surface-700 dark:text-surface-300 dark:hover:border-libre-400"
          >
            {t('browseModelZoo')}
            <ArrowRight className="h-4 w-4" />
          </Link>
        </div>
      </section>

      {/* RF100-VL: the follow-up to COCO, and the centrepiece of the page */}
      <section id="rf100-vl" className="mx-auto max-w-6xl scroll-mt-24 px-6">
        <SectionHeading tag="RF100-VL" title={t('rfTitle')}>
          {t('rfIntro')}
        </SectionHeading>

        <div className="mt-4 max-w-3xl">
          <p className="text-surface-600 dark:text-surface-400 leading-relaxed">
            {t('rfGeneralisation')}
          </p>
          <p className="mt-3 text-surface-600 dark:text-surface-400 leading-relaxed">
            {t('rfAdvice')}
          </p>
        </div>
      </section>

      <RF100VLPanel />

      {/* Latency lives on Vision Analysis, which measures it on real boards.
          Embedding one hardware slice here was worse than sending people to
          the page that lets them pick their own. */}
      <section className="border-t border-surface-200 dark:border-surface-800 bg-surface-50/60 dark:bg-surface-900/30">
        <div className="mx-auto max-w-6xl px-6 py-16 md:py-24">
          <SectionHeading tag={t('latencyTag')} title={t('latencyTitle')}>
            {t('latencyBody')}
          </SectionHeading>

          <a
            href={`${VA}/hardware?${UTM}`}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-8 inline-flex items-center gap-3 rounded-2xl border border-surface-900 bg-surface-900 px-7 py-4 text-base font-semibold text-white transition-colors hover:bg-surface-700 dark:border-white/15 dark:bg-white/5 dark:font-medium dark:text-surface-100 dark:hover:bg-white/10"
          >
            <Gauge className="h-5 w-5" />
            {t('latencyCta')}
            <ArrowRight className="h-5 w-5" />
          </a>
        </div>
      </section>
    </main>
  )
}
