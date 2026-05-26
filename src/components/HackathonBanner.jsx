'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useEffect, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Sparkles, X, ArrowRight } from 'lucide-react'

export default function HackathonBanner() {
  const pathname = usePathname()
  const [isVisible, setIsVisible] = useState(true)

  // Re-open the banner on every navigation, even if the user previously
  // dismissed it. State is intentionally not persisted.
  useEffect(() => {
    setIsVisible(true)
  }, [pathname])

  const isOnHackathonRoute = pathname?.startsWith('/cursor-hackathon')
  if (isOnHackathonRoute) return null

  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 30 }}
          transition={{ duration: 0.25, ease: 'easeOut' }}
          className="fixed bottom-4 left-1/2 -translate-x-1/2 z-50 w-[calc(100%-1.5rem)] sm:w-auto sm:max-w-md px-3 sm:px-0"
          role="region"
          aria-label="Cursor Hackathon banner"
        >
          <div className="relative flex items-center gap-3 pl-4 pr-12 py-3 rounded-full bg-white dark:bg-surface-900 border border-surface-200 dark:border-white/10 shadow-lg backdrop-blur-md">
            <Link
              href="/cursor-hackathon"
              className="group flex items-center gap-3 min-w-0 flex-1"
              onClick={() => setIsVisible(false)}
            >
              <span className="flex items-center justify-center w-8 h-8 rounded-full bg-libre-500/10 border border-libre-500/20 shrink-0">
                <Sparkles className="w-4 h-4 text-libre-600 dark:text-libre-400" />
              </span>
              <span className="flex flex-col min-w-0 text-left">
                <span className="text-[11px] uppercase tracking-wider font-semibold text-libre-600 dark:text-libre-400">
                  Cursor Hackathon
                </span>
                <span className="text-sm font-medium text-surface-800 dark:text-white truncate">
                  Looking for the Hackathon page?
                </span>
              </span>
              <ArrowRight className="w-4 h-4 text-surface-400 group-hover:text-libre-500 group-hover:translate-x-0.5 transition-all shrink-0" />
            </Link>

            <button
              onClick={() => setIsVisible(false)}
              className="absolute right-2 top-1/2 -translate-y-1/2 w-8 h-8 inline-flex items-center justify-center rounded-full text-surface-400 hover:text-surface-900 dark:hover:text-white hover:bg-surface-100 dark:hover:bg-white/10 transition-colors"
              aria-label="Dismiss banner"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
