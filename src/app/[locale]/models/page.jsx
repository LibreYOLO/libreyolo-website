'use client'

import { motion } from 'framer-motion'
import { useTranslations } from 'next-intl'
import { ExternalLink, ArrowRight, Boxes, Download, BarChart3 } from 'lucide-react'
import { Link } from '@/i18n/navigation'
import ThemedEmbed from '@/components/ThemedEmbed'

// Models highlighted (colored by family) on the scatter embed; every other model
// stays grey as reference context. Ids come from vision-analysis models.json.
const HIGHLIGHTED_MODEL_IDS = [
  'yolov9t', 'yolov9s', 'yolov9m', 'yolov9c',
  'rfdetr-n', 'rfdetr-s', 'rfdetr-m', 'rfdetr-l',
].join(',')

const BENCHMARK_EMBED =
  'https://visionanalysis.org/embed/scatter' +
  '?title=' + encodeURIComponent('LibreYOLO model zoo: accuracy vs. size') +
  '&highlight=' + encodeURIComponent(HIGHLIGHTED_MODEL_IDS)

const visionLink = (chunks) => (
  <a
    href="https://visionanalysis.org"
    target="_blank"
    rel="noopener noreferrer"
    className="text-libre-600 dark:text-libre-400 hover:underline"
  >
    {chunks}
  </a>
)

export default function Models() {
  const t = useTranslations('Models')
  const tc = useTranslations('Common')

  const highlights = [
    {
      icon: Boxes,
      title: t('highlight1Title'),
      description: t('highlight1Desc'),
    },
    {
      icon: Download,
      title: t('highlight2Title'),
      description: t('highlight2Desc'),
    },
    {
      icon: BarChart3,
      title: t('highlight3Title'),
      description: t.rich('highlight3Desc', { link: visionLink }),
    },
  ]

  return (
    <div className="pt-24 lg:pt-32 pb-16">
      <div className="max-w-7xl mx-auto px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-surface-800 dark:text-white mb-6">
            {t.rich('title', {
              accent: (chunks) => <span className="text-libre-500 dark:text-libre-400">{chunks}</span>,
            })}
          </h1>
          <p className="text-lg text-surface-600 dark:text-surface-400 max-w-2xl mx-auto mb-10">
            {t('subtitle')}
          </p>

          <a
            href="https://huggingface.co/LibreYOLO/models"
            target="_blank"
            rel="noopener noreferrer"
            className="btn-primary inline-flex items-center gap-3 px-10 py-5 bg-gradient-to-r from-libre-500 to-libre-600 rounded-xl text-white font-semibold text-lg"
          >
            {t('browseCta')}
            <ExternalLink className="w-5 h-5" />
          </a>
        </motion.div>

        {/* Highlights */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="grid sm:grid-cols-3 gap-6 mb-16 max-w-5xl mx-auto"
        >
          {highlights.map((item) => (
            <div
              key={item.title}
              className="bg-white dark:bg-surface-900/60 border border-surface-200 dark:border-white/10 rounded-2xl p-7 shadow-sm"
            >
              <div className="w-12 h-12 bg-libre-500/10 rounded-xl flex items-center justify-center mb-5">
                <item.icon className="w-6 h-6 text-libre-600 dark:text-libre-400" />
              </div>
              <h3 className="text-lg font-semibold text-surface-900 dark:text-white mb-2">{item.title}</h3>
              <p className="text-surface-600 dark:text-surface-400 text-sm leading-relaxed">{item.description}</p>
            </div>
          ))}
        </motion.div>

        {/* Benchmarks */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="max-w-5xl mx-auto"
        >
          <p className="text-center text-surface-600 dark:text-surface-400 max-w-2xl mx-auto mb-6">
            {t.rich('benchmarkIntro', {
              link: (chunks) => (
                <a
                  href="https://visionanalysis.org"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-libre-500 hover:text-libre-600 underline underline-offset-2"
                >
                  {chunks}
                </a>
              ),
            })}
          </p>

          {/* Aspect box matches the embed SVG (640x400) so nothing gets clipped. */}
          <div className="w-full aspect-[8/5] rounded-xl overflow-hidden">
            <ThemedEmbed
              src={BENCHMARK_EMBED}
              loading="lazy"
              title="LibreYOLO model zoo accuracy vs parameters - visionanalysis.org"
              className="block w-full h-full"
              style={{ border: 0 }}
            />
          </div>

          <div className="text-center mt-6">
            <a
              href="https://visionanalysis.org"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-6 py-3 bg-white dark:bg-white/5 hover:bg-surface-100 dark:hover:bg-white/10 border border-surface-300 dark:border-white/10 rounded-xl text-surface-800 dark:text-white font-medium transition-all"
            >
              <BarChart3 className="w-4 h-4 text-libre-500 dark:text-libre-400" />
              {t('exploreBenchmarks')}
              <ExternalLink className="w-4 h-4" />
            </a>
          </div>
        </motion.div>

        {/* Note */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center text-surface-500 text-sm max-w-xl mx-auto mt-14"
        >
          <p>
            {t('note')}
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mt-12"
        >
          <Link
            href="/docs"
            className="btn-primary inline-flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-libre-500 to-libre-600 rounded-xl text-white font-semibold text-lg"
          >
            {tc('getStarted')}
            <ArrowRight className="w-5 h-5" />
          </Link>
        </motion.div>
      </div>
    </div>
  )
}
