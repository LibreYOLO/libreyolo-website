'use client'

import { motion } from 'framer-motion'
import { useTranslations } from 'next-intl'
import { ArrowRight } from 'lucide-react'
import { Link } from '@/i18n/navigation'

export default function Commercial() {
  const t = useTranslations('Commercial')
  const tc = useTranslations('Common')
  return (
    <div className="pt-24 lg:pt-32 pb-16">
      <div className="max-w-2xl mx-auto px-6 lg:px-8 text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-surface-800 dark:text-white mb-6">
            <span className="text-emerald-600 dark:text-emerald-400">{t('titleEmphasis')}</span>{t('titleRest')}
          </h1>

          <p className="text-xl text-surface-600 dark:text-surface-400 leading-relaxed mb-12">
            {t('subtitle')}
          </p>

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
