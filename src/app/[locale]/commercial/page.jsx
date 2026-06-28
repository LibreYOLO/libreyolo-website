'use client'

import { motion } from 'framer-motion'
import Link from 'next/link'
import { ArrowRight } from 'lucide-react'

export default function Commercial() {
  return (
    <div className="pt-24 lg:pt-32 pb-16">
      <div className="max-w-2xl mx-auto px-6 lg:px-8 text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-surface-800 dark:text-white mb-6">
            <span className="text-emerald-600 dark:text-emerald-400">Yes</span>, you can use this.
          </h1>

          <p className="text-xl text-surface-600 dark:text-surface-400 leading-relaxed mb-12">
            100% free, no strings attached. LibreYOLO is open source like YOLO was always supposed to be.
          </p>

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
