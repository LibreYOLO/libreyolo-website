'use client'

import { motion } from 'framer-motion'
import { Activity, Cpu, Gauge, ShieldCheck, ArrowRight, BarChart3 } from 'lucide-react'
import { Link } from '@/i18n/navigation'
import RF100VLPanel from '@/components/articles/rf100vl/RF100VLPanel'
import ThemedEmbed from '@/components/ThemedEmbed'

const VA = 'https://www.visionanalysis.org'
const UTM = 'utm_source=libreyolo&utm_medium=referral&utm_campaign=benchmarks'

// Two families lit up against the grey field, rather than every model at once.
// The embed connects each highlighted family into a curve, so the size/accuracy
// trade-off reads at a glance instead of as a cloud.
const SCATTER_HIGHLIGHT = [
  'dfine-n', 'dfine-s', 'dfine-m', 'dfine-l', 'dfine-x',
  'yolov9t', 'yolov9s', 'yolov9m', 'yolov9c',
].join(',')

const HEADLINE_STATS = [
  { icon: Activity, value: '100', label: 'real-world datasets in RF100-VL' },
  { icon: Cpu, value: '6', label: 'hardware platforms measured' },
  { icon: Gauge, value: '700+', label: 'verified benchmark runs' },
  { icon: ShieldCheck, value: '0.39', label: 'worst mAP drift from the original papers' },
]

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
  return (
    <main className="min-h-screen bg-white dark:bg-surface-950">
      {/* Hero */}
      <section className="relative overflow-hidden border-b border-surface-200 dark:border-surface-800">
        <div className="absolute inset-0 bg-gradient-to-br from-libre-500/10 via-transparent to-transparent" />
        <div className="relative mx-auto max-w-6xl px-6 pt-28 pb-14 md:pt-36 md:pb-20">
          <motion.p
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            className="font-mono text-[10px] md:text-xs font-semibold uppercase tracking-[0.35em] text-libre-600 dark:text-libre-300/90"
          >
            LibreYOLO · Benchmarks
          </motion.p>

          <motion.h1
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.05 }}
            className="mt-4 max-w-3xl text-4xl md:text-6xl font-bold tracking-tight leading-[1.05] text-surface-900 dark:text-white"
          >
            Every model, measured.{' '}
            <span className="text-surface-400 dark:text-surface-500">Nothing copied from a paper.</span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="mt-6 max-w-2xl text-lg text-surface-600 dark:text-surface-400 leading-relaxed"
          >
            LibreYOLO ships dozens of detection models behind one API. These are the numbers that
            tell you which one to pick: accuracy on COCO, transfer onto 100 real-world datasets,
            and speed on hardware you can actually buy.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15 }}
            className="mt-10 grid grid-cols-2 gap-3 sm:grid-cols-4"
          >
            {HEADLINE_STATS.map(({ icon: Icon, value, label }) => (
              <div
                key={label}
                className="rounded-2xl border border-surface-200 bg-white/70 p-4 backdrop-blur-sm dark:border-surface-800 dark:bg-surface-900/60"
              >
                <Icon className="h-4 w-4 text-libre-500" />
                <div className="mt-2.5 font-mono text-2xl font-bold leading-none text-surface-900 dark:text-white">
                  {value}
                </div>
                <div className="mt-1.5 text-[11px] leading-snug text-surface-500 dark:text-surface-400">
                  {label}
                </div>
              </div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* RF100-VL: the centrepiece */}
      <section className="mx-auto max-w-6xl px-6 pt-16 md:pt-24">
        <SectionHeading tag="RF100-VL" title="What happens when you leave COCO behind?">
          COCO is 80 everyday categories: people, cars, dogs. Almost nobody ships that. The real
          question is transfer, so we fine-tuned on 100 unrelated real-world datasets and scored
          every one of them separately. Pills on a conveyor, chest X-rays, mahjong tiles, varroa
          mites, wildfire smoke.
        </SectionHeading>
      </section>

      <RF100VLPanel />

      {/* COCO, via a Vision Analysis embed */}
      <section className="mx-auto max-w-6xl px-6 pt-8 pb-16 md:pb-24">
        <SectionHeading tag="COCO" title="Accuracy against size">
          The classic view: how much accuracy each architecture buys for its parameter count.
          Live from Vision Analysis, so it stays current as new runs land.
        </SectionHeading>

        <div className="mt-8 overflow-hidden rounded-2xl border border-surface-200 dark:border-surface-800">
          <ThemedEmbed
            src={`${VA}/embed/scatter?highlight=${SCATTER_HIGHLIGHT}&title=Accuracy%20vs%20parameters&subtitle=D-FINE%20and%20YOLOv9%20against%20every%20other%20detector%20on%20COCO%20val2017`}
            title="Accuracy vs parameters, COCO val2017"
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
            Open the full leaderboard
          </a>
          <Link
            href="/models"
            className="inline-flex items-center gap-2 rounded-full border border-surface-200 px-5 py-2.5 text-sm font-semibold text-surface-700 transition-colors hover:border-libre-500 hover:text-libre-600 dark:border-surface-700 dark:text-surface-300 dark:hover:border-libre-400"
          >
            Browse the model zoo
            <ArrowRight className="h-4 w-4" />
          </Link>
        </div>
      </section>

      {/* Latency */}
      <section className="border-t border-surface-200 dark:border-surface-800 bg-surface-50/60 dark:bg-surface-900/30">
        <div className="mx-auto max-w-6xl px-6 py-16 md:py-24">
          <SectionHeading tag="Latency" title="Fast enough on what you own?">
            Accuracy does not change with hardware. Speed changes by two orders of magnitude
            between a datacentre GPU and a Raspberry Pi. Every run below is published as raw JSON.
          </SectionHeading>

          <div className="mt-8 overflow-hidden rounded-2xl border border-surface-200 dark:border-surface-800">
            <ThemedEmbed
              src={`${VA}/embed/leaderboard?hw=jetson_orin&rt=tensorrt_fp16&limit=10`}
              title="Latency leaderboard across hardware"
              className="w-full block"
              style={{ border: 0, overflow: 'hidden', aspectRatio: '640 / 460' }}
            />
          </div>
        </div>
      </section>

      {/* Trust */}
      <section className="mx-auto max-w-6xl px-6 py-16 md:py-24">
        <SectionHeading tag="Provenance" title="Why you can trust these numbers">
          Because they are ours, not the papers&apos;. Every model LibreYOLO ships was re-measured
          from scratch on the full COCO val2017 set and checked against what its original authors
          published. Across every variant, the worst disagreement is 0.39 mAP. If a port had
          quietly broken, that number would show it.
        </SectionHeading>

        <a
          href={`${VA}/parity?${UTM}`}
          target="_blank"
          rel="noopener noreferrer"
          className="mt-6 inline-flex items-center gap-2 text-sm font-semibold text-libre-600 hover:underline dark:text-libre-400"
        >
          See every variant against its original source
          <ArrowRight className="h-4 w-4" />
        </a>
      </section>
    </main>
  )
}
