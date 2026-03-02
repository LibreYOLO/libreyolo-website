'use client'

import { motion } from 'framer-motion'
import { ExternalLink, Database } from 'lucide-react'

export default function Datasets() {
  return (
    <div className="min-h-screen pt-24 lg:pt-32 pb-16">
      <div className="max-w-7xl mx-auto px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-sm font-medium mb-6">
            <Database className="w-4 h-4" />
            Training Data
          </div>
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-white mb-6">
            Dataset <span className="text-emerald-400">Zoo</span>
          </h1>
          <p className="text-lg text-surface-400 max-w-2xl mx-auto mb-10">
            Curated datasets for training and evaluating LibreYOLO models.
            Ready to use with the LibreYOLO training pipeline.
          </p>

          <a
            href="https://huggingface.co/LibreYOLO/datasets"
            target="_blank"
            rel="noopener noreferrer"
            className="btn-primary inline-flex items-center gap-3 px-10 py-5 bg-gradient-to-r from-emerald-500 to-emerald-600 rounded-xl text-white font-semibold text-lg"
          >
            Browse Datasets on HuggingFace
            <ExternalLink className="w-5 h-5" />
          </a>
        </motion.div>
      </div>
    </div>
  )
}
