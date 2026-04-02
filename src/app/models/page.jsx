'use client'

import { motion } from 'framer-motion'
import Link from 'next/link'
import { ExternalLink, Layers, ArrowRight } from 'lucide-react'

export default function Models() {
  return (
    <div className="pt-24 lg:pt-32 pb-16">
      <div className="max-w-7xl mx-auto px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-libre-500/10 border border-libre-500/20 text-libre-700 dark:text-libre-400 text-sm font-medium mb-6">
            <Layers className="w-4 h-4" />
            Pre-trained Weights
          </div>
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-surface-800 dark:text-white mb-6">
            Model <span className="text-libre-500 dark:text-libre-400">Zoo</span>
          </h1>
          <p className="text-lg text-surface-600 dark:text-surface-400 max-w-2xl mx-auto mb-10">
            Pre-trained weights for LibreYOLOX, LibreYOLO8, and LibreYOLO11 architectures.
            All models trained on COCO dataset with 80 object classes.
          </p>

          <a
            href="https://huggingface.co/LibreYOLO/models"
            target="_blank"
            rel="noopener noreferrer"
            className="btn-primary inline-flex items-center gap-3 px-10 py-5 bg-gradient-to-r from-libre-500 to-libre-600 rounded-xl text-white font-semibold text-lg"
          >
            Browse Models on HuggingFace
            <ExternalLink className="w-5 h-5" />
          </a>
        </motion.div>

        {/* Stats Bar */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="grid grid-cols-3 gap-4 mb-12 max-w-2xl mx-auto"
        >
          <div className="bg-surface-50 dark:bg-surface-900/50 border border-surface-200 dark:border-white/5 rounded-xl p-4 text-center">
            <p className="text-2xl font-bold text-surface-800 dark:text-white">15</p>
            <p className="text-surface-500 text-sm">Models</p>
          </div>
          <div className="bg-surface-50 dark:bg-surface-900/50 border border-surface-200 dark:border-white/5 rounded-xl p-4 text-center">
            <p className="text-2xl font-bold text-libre-500 dark:text-libre-400">3</p>
            <p className="text-surface-500 text-sm">Architectures</p>
          </div>
          <div className="bg-surface-50 dark:bg-surface-900/50 border border-surface-200 dark:border-white/5 rounded-xl p-4 text-center">
            <p className="text-2xl font-bold text-surface-800 dark:text-white">PyTorch</p>
            <p className="text-surface-500 text-sm">Framework</p>
          </div>
        </motion.div>

        {/* Coming soon note */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center text-surface-500 text-sm max-w-xl mx-auto mt-4"
        >
          <p>
            Individual model cards will be listed here soon. In the meanwhile, browse all available
            models on{' '}
            <a
              href="https://huggingface.co/LibreYOLO/models"
              target="_blank"
              rel="noopener noreferrer"
              className="text-libre-500 hover:text-libre-600 underline underline-offset-2 transition-colors"
            >
              LibreYOLO&apos;s HuggingFace
            </a>.
          </p>
          <p className="mt-3">
            Note: Pre-trained weights may inherit licensing terms from their original training source.
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mt-16"
        >
          <Link
            href="/docs"
            className="btn-primary inline-flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-libre-500 to-libre-600 rounded-xl text-white font-semibold text-lg"
          >
            Get Started
            <ArrowRight className="w-5 h-5" />
          </Link>
        </motion.div>
      </div>
    </div>
  )
}
