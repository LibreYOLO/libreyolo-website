'use client'

import Link from 'next/link'
import Image from 'next/image'
import { usePathname } from 'next/navigation'
import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Menu, X, Github, BookOpen } from 'lucide-react'
import ThemeToggle from './ThemeToggle'

const navLinks = [
  { name: 'Home', path: '/' },
  { name: 'Models', path: '/models' },
  { name: 'Datasets', path: '/datasets' },
  { name: 'Commercial', path: '/commercial' },
  { name: 'Docs', path: '/docs', icon: BookOpen },
]

export default function Navbar() {
  const [isScrolled, setIsScrolled] = useState(false)
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)
  const pathname = usePathname()

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
        className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
          isScrolled
            ? 'bg-white/80 dark:bg-surface-950/80 backdrop-blur-xl border-b border-surface-200/50 dark:border-white/5 shadow-sm dark:shadow-none'
            : 'bg-transparent'
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
              {navLinks.map((link) => (
                <Link
                  key={link.path}
                  href={link.path}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 flex items-center gap-1.5 ${
                    pathname === link.path
                      ? 'text-libre-600 dark:text-libre-400 bg-libre-500/10'
                      : 'text-surface-600 dark:text-surface-200 hover:text-surface-900 dark:hover:text-white hover:bg-surface-100 dark:hover:bg-white/5'
                  }`}
                >
                  {link.icon && <link.icon className="w-4 h-4" />}
                  {link.name}
                </Link>
              ))}
            </div>

            {/* Right Side */}
            <div className="hidden md:flex items-center gap-2">
              <ThemeToggle />
              <a
                href="https://github.com/Libre-YOLO/libreyolo"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-3 py-2 text-sm text-surface-500 dark:text-surface-300 hover:text-surface-900 dark:hover:text-white transition-colors"
              >
                <Github className="w-4 h-4" />
                GitHub
              </a>
              <div className="badge-mit px-3 py-1.5 rounded-full text-xs font-semibold flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 dark:bg-emerald-400 animate-pulse" />
                MIT Licensed
              </div>
            </div>

            {/* Mobile Menu Button */}
            <div className="flex items-center gap-2 md:hidden">
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
              {navLinks.map((link) => (
                <Link
                  key={link.path}
                  href={link.path}
                  className={`flex items-center gap-1.5 px-4 py-3 rounded-lg text-sm font-medium transition-all ${
                    pathname === link.path
                      ? 'text-libre-600 dark:text-libre-400 bg-libre-500/10'
                      : 'text-surface-600 dark:text-surface-200 hover:text-surface-900 dark:hover:text-white hover:bg-surface-100 dark:hover:bg-white/5'
                  }`}
                >
                  {link.icon && <link.icon className="w-4 h-4" />}
                  {link.name}
                </Link>
              ))}
              <a
                href="https://github.com/Libre-YOLO/libreyolo"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-4 py-3 text-sm text-surface-500 dark:text-surface-300 hover:text-surface-900 dark:hover:text-white"
              >
                <Github className="w-4 h-4" />
                GitHub
              </a>
              <div className="px-4 pt-2">
                <div className="badge-mit px-3 py-1.5 rounded-full text-xs font-semibold inline-flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 dark:bg-emerald-400 animate-pulse" />
                  MIT Licensed
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}
