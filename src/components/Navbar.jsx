'use client'

import Image from 'next/image'
import { useState, useEffect } from 'react'
import { useTranslations } from 'next-intl'
import { motion, AnimatePresence } from 'framer-motion'
import { Menu, X, BookOpen, BarChart3, ExternalLink } from 'lucide-react'
import { Link, usePathname } from '@/i18n/navigation'
import ThemeToggle from './ThemeToggle'
import LanguageSwitcher from './LanguageSwitcher'
import { GithubIcon, RedditIcon } from './BrandIcons'
import { GITHUB_URL, REDDIT_URL } from '@/lib/links'

const repoButtonClass =
  'flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium border border-surface-200 dark:border-white/10 text-surface-600 dark:text-surface-200 hover:text-surface-900 dark:hover:text-white hover:border-surface-300 dark:hover:border-white/20 hover:bg-surface-50 dark:hover:bg-white/5 transition-all duration-200'

export default function Navbar() {
  const [isScrolled, setIsScrolled] = useState(false)
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)
  const pathname = usePathname()
  const t = useTranslations('Nav')

  const navLinks = [
    { name: t('home'), path: '/' },
    { name: t('models'), path: '/models' },
    {
      name: t('benchmarks'),
      path: 'https://www.visionanalysis.org/?utm_source=libreyolo&utm_medium=referral&utm_campaign=benchmarks',
      external: true,
      icon: BarChart3,
    },
    { name: t('commercial'), path: '/commercial' },
    { name: t('articles'), path: '/articles' },
    { name: t('docs'), path: '/docs/v1.4.0', icon: BookOpen, highlight: true },
  ]

  useEffect(() => {
    const handleScroll = () => setIsScrolled(window.scrollY > 20)
    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  useEffect(() => {
    setIsMobileMenuOpen(false)
  }, [pathname])

  return (
    <>
      <motion.nav
        initial={{ y: -100 }}
        animate={{ y: 0 }}
        transition={{ duration: 0.6, ease: 'easeOut' }}
        className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 bg-white/80 dark:bg-surface-950/80 backdrop-blur-xl border-b border-surface-200/50 dark:border-white/5 shadow-sm dark:shadow-none ${
          isScrolled
            ? ''
            : 'md:bg-transparent md:backdrop-blur-none md:border-transparent md:shadow-none'
        }`}
      >
        <div className="max-w-7xl mx-auto px-6 lg:px-8">
          <div className="flex items-center justify-between h-16 lg:h-20">
            {/* Logo */}
            <Link href="/" className="flex items-center gap-3 group">
              <div className="relative w-8 h-8">
                <Image
                  src="/logo.png"
                  alt="LibreYOLO"
                  fill
                  className="object-contain dark:invert transition-all duration-300 group-hover:opacity-80 relative z-10"
                  sizes="32px"
                />
                <div className="absolute inset-0 blur-lg bg-libre-500/20 dark:bg-libre-400/30 group-hover:bg-libre-500/30 dark:group-hover:bg-libre-300/40 transition-all duration-300" />
              </div>
              <span className="text-xl font-semibold tracking-tight">
                <span className="text-surface-900 dark:text-white">Libre</span>
                <span className="text-libre-500 dark:text-libre-400">YOLO</span>
              </span>
            </Link>

            {/* Desktop Navigation */}
            <div className="hidden md:flex items-center gap-1">
              {navLinks.map((link) => {
                const isActive = pathname === link.path
                const base = 'px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 flex items-center gap-1.5'
                const className = link.highlight
                  ? `${base} text-white bg-gradient-to-r from-libre-500 to-libre-600 hover:from-libre-400 hover:to-libre-500 shadow-sm`
                  : isActive
                    ? `${base} text-libre-600 dark:text-libre-400 bg-libre-500/10`
                    : `${base} text-surface-600 dark:text-surface-200 hover:text-surface-900 dark:hover:text-white hover:bg-surface-100 dark:hover:bg-white/5`

                if (link.external) {
                  return (
                    <a
                      key={link.path}
                      href={link.path}
                      target="_blank"
                      rel="noopener"
                      referrerPolicy="strict-origin-when-cross-origin"
                      className={className}
                    >
                      {link.icon && <link.icon className="w-4 h-4" />}
                      {link.name}
                      <ExternalLink className="w-3 h-3 opacity-60" />
                    </a>
                  )
                }

                return (
                  <Link key={link.path} href={link.path} className={className}>
                    {link.icon && <link.icon className="w-4 h-4" />}
                    {link.name}
                  </Link>
                )
              })}
            </div>

            {/* Right Side */}
            <div className="hidden md:flex items-center gap-2">
              <LanguageSwitcher />
              <ThemeToggle />
              <a
                href={REDDIT_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-semibold border border-[#FF4500]/30 bg-[#FF4500]/10 text-[#D93A00] dark:text-[#FF6A33] hover:bg-[#FF4500]/20 transition-colors"
                aria-label={t('community')}
              >
                <RedditIcon className="w-4 h-4" />
                r/LibreYOLO
              </a>
              <a
                href={GITHUB_URL}
                target="_blank"
                rel="noopener noreferrer"
                className={repoButtonClass}
              >
                <GithubIcon className="w-4 h-4" />
                {t('github')}
              </a>
            </div>

            {/* Mobile Menu Button */}
            <div className="flex items-center gap-2 md:hidden">
              <LanguageSwitcher />
              <ThemeToggle />
              <button
                onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
                className="p-2 text-surface-500 dark:text-surface-300 hover:text-surface-900 dark:hover:text-white transition-colors"
                aria-label="Toggle menu"
              >
                {isMobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
              </button>
            </div>
          </div>
        </div>
      </motion.nav>

      {/* Mobile Menu */}
      <AnimatePresence>
        {isMobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-x-0 top-16 z-40 md:hidden bg-white/95 dark:bg-surface-950/95 backdrop-blur-xl border-b border-surface-200 dark:border-white/5"
          >
            <div className="px-6 py-4 space-y-2">
              {navLinks.map((link) => {
                const isActive = pathname === link.path
                const base = 'flex items-center gap-1.5 px-4 py-3 rounded-lg text-sm font-medium transition-all'
                const className = link.highlight
                  ? `${base} text-white bg-gradient-to-r from-libre-500 to-libre-600`
                  : isActive
                    ? `${base} text-libre-600 dark:text-libre-400 bg-libre-500/10`
                    : `${base} text-surface-600 dark:text-surface-200 hover:text-surface-900 dark:hover:text-white hover:bg-surface-100 dark:hover:bg-white/5`

                if (link.external) {
                  return (
                    <a
                      key={link.path}
                      href={link.path}
                      target="_blank"
                      rel="noopener"
                      referrerPolicy="strict-origin-when-cross-origin"
                      className={className}
                    >
                      {link.icon && <link.icon className="w-4 h-4" />}
                      {link.name}
                      <ExternalLink className="w-3 h-3 opacity-60" />
                    </a>
                  )
                }

                return (
                  <Link key={link.path} href={link.path} className={className}>
                    {link.icon && <link.icon className="w-4 h-4" />}
                    {link.name}
                  </Link>
                )
              })}
              <a
                href={REDDIT_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1.5 px-4 py-3 rounded-lg text-sm font-medium text-surface-600 dark:text-surface-200 hover:text-[#FF4500] hover:bg-surface-100 dark:hover:bg-white/5 transition-all"
              >
                <RedditIcon className="w-4 h-4" />
                r/LibreYOLO
                <ExternalLink className="w-3 h-3 opacity-60" />
              </a>
              <a
                href={GITHUB_URL}
                target="_blank"
                rel="noopener noreferrer"
                className={`${repoButtonClass} mt-2 justify-center`}
              >
                <GithubIcon className="w-4 h-4" />
                {t('github')}
              </a>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}
