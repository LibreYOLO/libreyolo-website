'use client'

import { motion } from 'framer-motion'
import { useTranslations } from 'next-intl'
import { Home, Search } from 'lucide-react'
import { Link } from '@/i18n/navigation'

export default function NotFound() {
  const t = useTranslations('NotFound')
  return (
    <div className="min-h-screen flex items-center justify-center px-6">
      <div className="absolute top-1/3 left-1/2 -translate-x-1/2 w-96 h-96 bg-libre-500/10 rounded-full blur-3xl" />
      
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative z-10 text-center max-w-lg"
      >
        <div className="mb-8">
          <span className="text-8xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-libre-400 to-emerald-400">
            404
          </span>
        </div>
        
        <h1 className="text-3xl font-bold text-surface-800 dark:text-white mb-4">
          {t('title')}
        </h1>

        <p className="text-surface-600 dark:text-surface-400 mb-8">
          {t('body')}
        </p>

        <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
          <Link
            href="/"
            className="btn-primary flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-libre-500 to-libre-600 rounded-xl text-white font-semibold"
          >
            <Home className="w-5 h-5" />
            {t('goHome')}
          </Link>
          <Link
            href="/docs/v1.3.1"
            className="flex items-center gap-2 px-6 py-3 bg-surface-50 dark:bg-white/5 hover:bg-surface-100 dark:hover:bg-white/10 border border-surface-200 dark:border-white/10 rounded-xl text-surface-800 dark:text-white font-medium transition-all"
          >
            <Search className="w-5 h-5" />
            {t('browseDocs')}
          </Link>
        </div>

        <div className="mt-12 code-block rounded-xl max-w-sm mx-auto">
          <pre className="p-4 text-left text-sm">
            <code className="font-mono">
              <span className="token-comment">{t('codeComment')}</span>{'\n'}
              <span className="text-surface-600 dark:text-surface-400">confidence</span><span className="text-libre-400">:</span> <span className="text-red-400">0.00</span>{'\n'}
              <span className="text-surface-600 dark:text-surface-400">class</span><span className="text-libre-400">:</span> <span className="token-string">"page_not_found"</span>{'\n'}
              <span className="text-surface-600 dark:text-surface-400">suggestion</span><span className="text-libre-400">:</span> <span className="token-string">"try_home_page"</span>
            </code>
          </pre>
        </div>
      </motion.div>
    </div>
  )
}

