'use client'

import { motion } from 'framer-motion'
import { useTranslations } from 'next-intl'
import { ExternalLink, Database, ArrowRight } from 'lucide-react'
import { Link } from '@/i18n/navigation'

export default function Datasets() {
  const t = useTranslations('Datasets')
  const tc = useTranslations('Common')
  return (
    <div className="pt-24 lg:pt-32 pb-16">
      <div className="max-w-7xl mx-auto px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-700 dark:text-emerald-400 text-sm font-medium mb-6">
            <Database className="w-4 h-4" />
            {t('badge')}
          </div>
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-surface-800 dark:text-white mb-6">
            {t.rich('title', {
              accent: (chunks) => <span className="text-emerald-600 dark:text-emerald-400">{chunks}</span>,
            })}
          </h1>
          <p className="text-lg text-surface-600 dark:text-surface-400 max-w-2xl mx-auto mb-10">
            {t('subtitle')}
          </p>

          <a
            href="https://huggingface.co/LibreYOLO/datasets"
            target="_blank"
            rel="noopener noreferrer"
            className="btn-primary inline-flex items-center gap-3 px-10 py-5 bg-gradient-to-r from-emerald-500 to-emerald-600 rounded-xl text-white font-semibold text-lg"
          >
            {t('browseCta')}
            <ExternalLink className="w-5 h-5" />
          </a>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center text-surface-500 text-sm max-w-xl mx-auto mt-4"
        >
          <p>
            {t.rich('note', {
              link: (chunks) => (
                <a
                  href="https://huggingface.co/LibreYOLO/datasets"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-emerald-600 dark:text-emerald-400 hover:text-emerald-700 dark:hover:text-emerald-300 underline underline-offset-2 transition-colors"
                >
                  {chunks}
                </a>
              ),
            })}
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mt-16"
        >
          <Link
            href="/docs/v1.3.1"
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
