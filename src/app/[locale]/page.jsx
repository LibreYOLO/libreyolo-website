'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { useTranslations } from 'next-intl'
import Image from 'next/image'
import { Link } from '@/i18n/navigation'
import SupportCallout from '@/components/SupportCallout'
import {
  Unlock, Layers, ArrowRight,
  Code2, Scale, Copy, Check, CheckCircle2,
  Upload, RefreshCw, MessageSquareQuote,
  Cpu, X as XIcon, BarChart3, ChevronLeft, ChevronRight
} from 'lucide-react'

const fadeInUp = {
  initial: { opacity: 0, y: 30 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.6, ease: 'easeOut' }
}

const stagger = {
  animate: {
    transition: {
      staggerChildren: 0.1
    }
  }
}

const codeSnippet = `from libreyolo import LibreYOLO, SAMPLE_IMAGE

model = LibreYOLO("LibreYOLOXs.pt")
results = model(SAMPLE_IMAGE, save=True)`

function HeroSection() {
  const t = useTranslations('Home')
  const tc = useTranslations('Common')
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(codeSnippet)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <section className="relative md:min-h-screen md:flex md:items-center md:justify-center overflow-hidden">
      {/* Hero Content */}
      <div className="relative z-10 max-w-7xl mx-auto px-5 md:px-6 lg:px-8 pt-24 pb-12 md:pt-32 md:pb-20">
        <motion.div
          initial="initial"
          animate="animate"
          variants={stagger}
          className="text-center"
        >
          {/* Main Headline */}
          <motion.h1
            variants={fadeInUp}
            className="text-[40px] leading-[1.05] sm:text-5xl sm:leading-tight md:text-6xl lg:text-7xl xl:text-8xl font-bold tracking-tight mb-5 md:mb-6 mt-4 md:mt-0"
          >
            <span className="text-surface-800 dark:text-white">{t('heroTitleLine1')}</span>
            <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-libre-500 via-libre-400 to-emerald-500">
              {t('heroTitleLine2')}
            </span>
          </motion.h1>

          {/* Sub-headline */}
          <motion.p
            variants={fadeInUp}
            className="text-base sm:text-xl lg:text-2xl text-surface-600 dark:text-surface-400 max-w-3xl mx-auto mb-8 md:mb-10 leading-relaxed"
          >
            {t('heroSubA')}
            <span className="hidden sm:inline text-surface-800 dark:text-white font-medium">{' '}{t('heroSubB')}</span>
          </motion.p>

          {/* CTAs */}
          <motion.div
            variants={fadeInUp}
            className="flex flex-col sm:flex-row items-stretch sm:items-center justify-center gap-3 sm:gap-4"
          >
            <Link
              href="/docs"
              className="btn-primary group flex items-center justify-center gap-2 px-6 sm:px-8 py-3.5 sm:py-4 bg-gradient-to-r from-libre-500 to-libre-600 rounded-xl text-white font-semibold text-base sm:text-lg"
            >
              {tc('getStarted')}
              <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </Link>
            <a
              href="https://github.com/LibreYOLO/libreyolo"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center justify-center gap-2 px-6 sm:px-8 py-3.5 sm:py-4 bg-white dark:bg-white/5 hover:bg-surface-100 dark:hover:bg-white/10 border border-surface-300 dark:border-white/10 rounded-xl text-surface-800 dark:text-white font-medium text-base sm:text-lg transition-all shadow-sm dark:shadow-none"
            >
              <Code2 className="w-5 h-5 text-libre-500 dark:text-libre-400" />
              {t('viewGithub')}
            </a>
          </motion.div>

          {/* Hero Visual */}
          <motion.div
            variants={fadeInUp}
            className="mt-12 md:mt-20 relative"
          >
            <div className="flex flex-col lg:flex-row items-stretch gap-0 max-w-6xl mx-auto min-w-0">
              {/* Code Preview */}
              <div className="relative flex-1 w-full min-w-0">
                <div className="relative code-block rounded-2xl lg:rounded-r-none overflow-hidden h-full">
                  <div className="flex items-center justify-between px-4 py-3 bg-surface-100 dark:bg-surface-900/50 border-b border-surface-200 dark:border-white/5">
                    <div className="flex items-center gap-2">
                      <div className="flex gap-2">
                        <span className="w-3 h-3 rounded-full bg-red-500/80" />
                        <span className="w-3 h-3 rounded-full bg-yellow-500/80" />
                        <span className="w-3 h-3 rounded-full bg-green-500/80" />
                      </div>
                      <span className="ml-4 text-surface-500 text-sm font-mono">quickstart.py</span>
                    </div>
                    <button
                      onClick={handleCopy}
                      className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium text-surface-500 dark:text-surface-400 hover:text-surface-900 dark:hover:text-white hover:bg-surface-200 dark:hover:bg-white/10 transition-all"
                      aria-label="Copy code"
                    >
                      {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                      {copied ? t('copied') : t('copy')}
                    </button>
                  </div>
                  <pre className="text-left">
                    <code className="font-mono text-sm lg:text-base">
                      <table className="border-collapse">
                        <tbody>
                          <tr>
                            <td className="pr-4 text-right select-none text-surface-600 align-top w-6">1</td>
                            <td><span className="token-keyword">from</span> <span className="text-libre-600 dark:text-libre-300">libreyolo</span> <span className="token-keyword">import</span> <span className="text-emerald-600 dark:text-emerald-400">LibreYOLO</span>, <span className="text-emerald-600 dark:text-emerald-400">SAMPLE_IMAGE</span></td>
                          </tr>
                          <tr><td className="pr-4 text-right select-none text-surface-600 w-6">2</td><td></td></tr>
                          <tr>
                            <td className="pr-4 text-right select-none text-surface-600 align-top w-6">3</td>
                            <td><span className="text-surface-700 dark:text-surface-300">model</span> <span className="text-libre-600 dark:text-libre-400">=</span> <span className="text-emerald-600 dark:text-emerald-400">LibreYOLO</span>(<span className="token-string">&quot;LibreYOLOXs.pt&quot;</span>)</td>
                          </tr>
                          <tr>
                            <td className="pr-4 text-right select-none text-surface-600 align-top w-6">4</td>
                            <td><span className="text-surface-700 dark:text-surface-300">results</span> <span className="text-libre-600 dark:text-libre-400">=</span> <span className="text-surface-700 dark:text-surface-300">model</span>(<span className="text-emerald-600 dark:text-emerald-400">SAMPLE_IMAGE</span>, <span className="text-surface-700 dark:text-surface-300">save</span><span className="text-libre-600 dark:text-libre-400">=</span><span className="text-emerald-600 dark:text-emerald-400">True</span>)</td>
                          </tr>
                        </tbody>
                      </table>
                    </code>
                  </pre>

                  {/* Mobile-only: full-width result image inside the same card, click to open full-size */}
                  <div className="lg:hidden border-t border-surface-200 dark:border-white/5">
                    <a
                      href="https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour_result.jpg"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="block relative group"
                      aria-label={t('viewFullDetection')}
                    >
                      <img
                        src="https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour_result.jpg"
                        alt={t('detectionResultAlt')}
                        className="w-full h-auto block"
                      />
                      <div className="absolute inset-0 bg-gradient-to-t from-black/40 to-transparent opacity-0 group-active:opacity-100 transition-opacity" />
                    </a>
                    <div className="flex items-center justify-between gap-3 px-4 py-3 bg-surface-100/70 dark:bg-surface-900/50">
                      <div className="flex items-center gap-2 min-w-0">
                        <div className="w-2 h-2 rounded-full bg-emerald-500 dark:bg-emerald-400 animate-pulse flex-shrink-0" />
                        <span className="text-emerald-600 dark:text-emerald-400 font-mono text-xs">{t('detected')}</span>
                      </div>
                      <span className="text-surface-500 dark:text-surface-400 font-mono text-xs flex-shrink-0">0.023s</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Arrow connector - visible on lg screens */}
              <div className="hidden lg:flex items-center justify-center relative z-10">
                <div className="w-12 h-12 rounded-full bg-white dark:bg-surface-900 border border-libre-500/30 shadow-lg dark:shadow-none flex items-center justify-center -mx-6">
                  <ArrowRight className="w-5 h-5 text-libre-500 dark:text-libre-400" />
                </div>
              </div>

              {/* Result Image as Output Panel — desktop-only */}
              <div className="hidden lg:block relative lg:max-w-sm w-full">
                <div className="relative bg-surface-50 dark:bg-surface-900/80 backdrop-blur-sm border border-surface-200 dark:border-white/10 rounded-2xl lg:rounded-l-none overflow-hidden h-full">
                  <div className="flex items-center gap-2 px-4 py-3 bg-surface-100 dark:bg-surface-900/50 border-b border-surface-200 dark:border-white/5">
                    <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                    <span className="text-surface-500 text-sm font-mono">parkour_result.jpg</span>
                  </div>
                  <div className="p-3">
                    <img
                      src="https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour_result.jpg"
                      alt={t('detectionResultAlt')}
                      className="rounded-lg w-full"
                    />
                    <div className="mt-3 flex items-center justify-between text-xs">
                      <span className="text-emerald-400 font-mono">&check; {t('detected')}</span>
                      <span className="text-surface-500 font-mono">0.023s</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </motion.div>
      </div>

    </section>
  )
}


function SocialProofSection() {
  const t = useTranslations('Home')
  const [carouselPage, setCarouselPage] = useState(0)

  const testimonials = t.raw('testimonials')

  const pageSize = 3
  const totalDesktopPages = Math.ceil(testimonials.length / pageSize)
  // carouselPage is a testimonial index (0 to testimonials.length - 1).
  // Desktop groups by pageSize; mobile shows one testimonial at a time.
  const desktopGroup = Math.floor(carouselPage / pageSize)
  const desktopVisible = testimonials.slice(desktopGroup * pageSize, desktopGroup * pageSize + pageSize)
  const mobileActive = testimonials[Math.min(carouselPage, testimonials.length - 1)]

  const goPrev = () => setCarouselPage(((desktopGroup === 0 ? totalDesktopPages - 1 : desktopGroup - 1)) * pageSize)
  const goNext = () => setCarouselPage(((desktopGroup === totalDesktopPages - 1 ? 0 : desktopGroup + 1)) * pageSize)

  return (
    <section className="relative py-16 lg:py-20 overflow-hidden">
      <div className="relative z-10 max-w-7xl mx-auto px-5 md:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-8 md:mb-10"
        >
          <span className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-amber-500/10 border border-amber-500/20 text-amber-600 dark:text-amber-400 text-xs font-medium mb-3 md:mb-4">
            <MessageSquareQuote className="w-3.5 h-3.5" />
            r/computervision
          </span>
          <h2 className="text-[26px] leading-tight sm:text-4xl font-bold text-surface-900 dark:text-white">
            {t('communityTitle')}
          </h2>
        </motion.div>

        <div className="relative max-w-5xl mx-auto">
          {/* Prev button — desktop only */}
          <button
            onClick={goPrev}
            className="hidden sm:flex absolute -left-4 lg:-left-14 top-1/2 -translate-y-1/2 z-10 w-10 h-10 rounded-full bg-white dark:bg-surface-800 border border-surface-200 dark:border-white/10 shadow-md dark:shadow-none items-center justify-center text-surface-500 dark:text-surface-300 hover:text-surface-900 dark:hover:text-white hover:border-surface-300 dark:hover:border-white/20 transition-all"
            aria-label="Previous"
          >
            <ChevronLeft className="w-5 h-5" />
          </button>

          {/* Next button — desktop only */}
          <button
            onClick={goNext}
            className="hidden sm:flex absolute -right-4 lg:-right-14 top-1/2 -translate-y-1/2 z-10 w-10 h-10 rounded-full bg-white dark:bg-surface-800 border border-surface-200 dark:border-white/10 shadow-md dark:shadow-none items-center justify-center text-surface-500 dark:text-surface-300 hover:text-surface-900 dark:hover:text-white hover:border-surface-300 dark:hover:border-white/20 transition-all"
            aria-label="Next"
          >
            <ChevronRight className="w-5 h-5" />
          </button>

          {/* Mobile: single testimonial, focal */}
          <motion.div
            key={`m-${carouselPage}`}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35 }}
            className="sm:hidden relative bg-white dark:bg-surface-900/60 border border-surface-200 dark:border-white/5 rounded-2xl p-5 shadow-sm dark:shadow-none min-h-[140px] flex flex-col justify-between"
          >
            <p className="text-surface-700 dark:text-surface-200 text-base leading-relaxed mb-4 italic">
              &ldquo;{mobileActive.quote}&rdquo;
            </p>
            <p className="text-surface-400 text-xs font-mono">{mobileActive.author}</p>
          </motion.div>

          {/* Desktop: 3-up grid */}
          <div className="hidden sm:grid sm:grid-cols-3 gap-6">
            {desktopVisible.map((t, i) => (
              <motion.div
                key={`d-${desktopGroup}-${i}`}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.08 }}
                className="relative bg-white dark:bg-surface-900/60 border border-surface-200 dark:border-white/5 rounded-2xl p-6 shadow-sm dark:shadow-none min-h-[160px] flex flex-col justify-between"
              >
                <div className="absolute top-4 left-5 text-libre-500/20 text-4xl font-serif leading-none select-none">&ldquo;</div>
                <p className="text-surface-700 dark:text-surface-200 text-sm leading-relaxed mb-4 pt-4 italic">
                  &ldquo;{t.quote}&rdquo;
                </p>
                <p className="text-surface-400 text-xs font-mono">{t.author}</p>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Mobile dots — one per testimonial */}
        <div className="sm:hidden flex items-center justify-center gap-2 mt-6 flex-wrap max-w-xs mx-auto">
          {testimonials.map((_, i) => (
            <button
              key={i}
              onClick={() => setCarouselPage(i)}
              className={`h-2.5 rounded-full transition-all duration-300 ${
                i === carouselPage
                  ? 'bg-libre-500 w-6'
                  : 'w-2.5 bg-surface-300 dark:bg-surface-600'
              }`}
              aria-label={`Testimonial ${i + 1}`}
            />
          ))}
        </div>

        {/* Desktop dots — one per page group */}
        <div className="hidden sm:flex items-center justify-center gap-3 mt-8">
          {Array.from({ length: totalDesktopPages }).map((_, i) => (
            <button
              key={i}
              onClick={() => setCarouselPage(i * pageSize)}
              className={`h-2.5 rounded-full transition-all duration-300 ${
                i === desktopGroup
                  ? 'bg-libre-500 w-8'
                  : 'w-2.5 bg-surface-300 dark:bg-surface-600 hover:bg-surface-400 dark:hover:bg-surface-500'
              }`}
              aria-label={`Page ${i + 1}`}
            />
          ))}
        </div>

        <p className="text-center text-surface-500 dark:text-surface-500 text-xs sm:text-sm mt-6 md:mt-8 max-w-lg mx-auto px-5">
          {t('communityFootnote')}
        </p>
      </div>
    </section>
  )
}


function FeaturesSection() {
  const t = useTranslations('Home')
  const features = [
    {
      icon: Unlock,
      title: t('feature1Title'),
      description: t('feature1Desc'),
      color: 'emerald'
    },
    {
      icon: Layers,
      title: t('feature2Title'),
      description: t('feature2Desc'),
      color: 'libre'
    },
    {
      icon: RefreshCw,
      title: t('feature3Title'),
      description: t('feature3Desc'),
      color: 'cyan'
    },
    {
      icon: Code2,
      title: t('feature4Title'),
      description: t('feature4Desc'),
      color: 'amber'
    },
    {
      icon: CheckCircle2,
      title: t('feature5Title'),
      description: t('feature5Desc'),
      color: 'violet'
    },
    {
      icon: Upload,
      title: t('feature6Title'),
      description: t('feature6Desc'),
      color: 'rose'
    }
  ]

  // Colors are only applied on sm+ (desktop card style). On mobile, icons all use libre for visual calm.
  const colorClasses = {
    libre: { bg: 'sm:bg-libre-500/10', text: 'sm:text-libre-700 sm:dark:text-libre-400', border: 'sm:border-libre-500/20' },
    emerald: { bg: 'sm:bg-emerald-500/10', text: 'sm:text-emerald-700 sm:dark:text-emerald-400', border: 'sm:border-emerald-500/20' },
    cyan: { bg: 'sm:bg-cyan-500/10', text: 'sm:text-cyan-700 sm:dark:text-cyan-400', border: 'sm:border-cyan-500/20' },
    amber: { bg: 'sm:bg-amber-500/10', text: 'sm:text-amber-700 sm:dark:text-amber-400', border: 'sm:border-amber-500/20' },
    violet: { bg: 'sm:bg-violet-500/10', text: 'sm:text-violet-700 sm:dark:text-violet-400', border: 'sm:border-violet-500/20' },
    rose: { bg: 'sm:bg-rose-500/10', text: 'sm:text-rose-700 sm:dark:text-rose-400', border: 'sm:border-rose-500/20' }
  }

  return (
    <section className="relative py-16 lg:py-20">
      <div className="relative z-10 max-w-7xl mx-auto px-5 md:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-8 md:mb-12"
        >
          <h2 className="text-[28px] leading-tight sm:text-4xl lg:text-5xl font-bold text-surface-900 dark:text-white mb-3 md:mb-4">
            {t.rich('featuresTitle', {
              accent: (chunks) => <span className="text-transparent bg-clip-text bg-gradient-to-r from-libre-500 to-emerald-500">{chunks}</span>,
            })}
          </h2>
          <p className="text-sm sm:text-lg text-surface-600 dark:text-surface-400 max-w-2xl mx-auto">
            {t('featuresSubtitle')}
          </p>
        </motion.div>

        {/* Mobile: dense list. sm+: card grid. */}
        <div className="sm:grid sm:grid-cols-2 lg:grid-cols-3 sm:gap-6">
          {features.map((feature, index) => {
            const colors = colorClasses[feature.color]
            return (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 12 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.06 }}
                className={`flex gap-4 items-start py-5 border-b border-surface-200 dark:border-white/5 sm:block sm:py-0 sm:p-7 sm:border sm:border-surface-200 sm:dark:border-white/10 sm:rounded-2xl sm:shadow-sm sm:bg-white sm:dark:bg-surface-900/60 sm:card-hover ${colors.border}`}
              >
                <div className={`w-10 h-10 sm:w-12 sm:h-12 bg-libre-500/10 ${colors.bg} rounded-xl flex items-center justify-center flex-shrink-0 sm:mb-5`}>
                  <feature.icon className={`w-5 h-5 sm:w-6 sm:h-6 text-libre-600 dark:text-libre-400 ${colors.text}`} />
                </div>
                <div className="flex-1 min-w-0 sm:flex-none">
                  <h3 className="text-base sm:text-lg font-semibold text-surface-900 dark:text-white mb-1.5 sm:mb-2">{feature.title}</h3>
                  <p className="text-surface-600 dark:text-surface-400 text-[13px] sm:text-sm leading-relaxed">{feature.description}</p>
                </div>
              </motion.div>
            )
          })}
        </div>
      </div>
    </section>
  )
}


function DeployAnywhereSection() {
  const t = useTranslations('Home')
  const exportFormats = [
    { name: 'ONNX', variants: ['FP32', 'FP16'] },
    { name: 'TensorRT', variants: ['FP32', 'FP16', 'INT8'] },
    { name: 'OpenVINO', variants: ['FP16', 'INT8'] },
    { name: 'ncnn', variants: ['FP16'] },
    { name: 'TorchScript', variants: ['FP32'] },
  ]

  const hardware = [
    { name: 'Jetson Nano', src: '/hardware/jetson-nano.jpg', srcLight: '/hardware/jetson-nano-light.jpg' },
    { name: 'Jetson Orin', src: '/hardware/jetson-orin.jpg', srcLight: '/hardware/jetson-orin-light.jpg' },
    { name: 'Raspberry Pi', src: '/hardware/raspberry-pi.png', srcLight: '/hardware/raspberry-pi-light.png' },
    { name: 'NVIDIA GPU', src: '/hardware/nvidia-gpu.png', srcLight: '/hardware/nvidia-gpu-light.png' },
  ]

  return (
    <section className="relative py-14 lg:py-20 overflow-hidden">
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-libre-500/5 rounded-full blur-3xl" />

      <div className="relative z-10 max-w-7xl mx-auto px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-12"
        >
          <span className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-libre-500/10 border border-libre-500/20 text-libre-600 text-xs font-medium mb-4">
            <Cpu className="w-3.5 h-3.5" />
            {t('deployBadge')}
          </span>
          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-surface-900 dark:text-white mb-4">
            {t.rich('deployTitle', {
              accent: (chunks) => <span className="text-transparent bg-clip-text bg-gradient-to-r from-libre-500 to-emerald-500">{chunks}</span>,
            })}
          </h2>
          <p className="text-lg text-surface-600 dark:text-surface-400 max-w-2xl mx-auto">
            {t('deploySubtitle')}
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-12 items-start max-w-6xl mx-auto">
          {/* Export Formats */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
          >
            <h3 className="text-sm font-semibold text-surface-500 uppercase tracking-wider mb-6">{t('exportFormats')}</h3>
            <div className="space-y-3">
              {exportFormats.map((fmt) => (
                <div
                  key={fmt.name}
                  className="flex items-center justify-between bg-surface-50 dark:bg-surface-900/60 border border-surface-200 dark:border-white/5 rounded-xl px-5 py-3.5"
                >
                  <span className="text-surface-800 dark:text-white font-medium text-sm">{fmt.name}</span>
                  <div className="flex gap-2">
                    {fmt.variants.map((v) => (
                      <span
                        key={v}
                        className="px-2.5 py-1 rounded-md bg-libre-500/10 text-libre-600 text-xs font-mono"
                      >
                        {v}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </motion.div>

          {/* Hardware Grid */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
          >
            <h3 className="text-sm font-semibold text-surface-500 uppercase tracking-wider mb-6">{t('testedHardware')}</h3>
            <div className="grid grid-cols-2 gap-4">
              {hardware.map((hw) => (
                <div
                  key={hw.name}
                  className="group relative bg-surface-50 dark:bg-surface-900/60 border border-surface-200 dark:border-white/5 rounded-xl overflow-hidden"
                >
                  <div className="aspect-[4/3] relative">
                    {/* Dark variant */}
                    <Image
                      src={hw.src}
                      alt={hw.name}
                      fill
                      className="object-cover group-hover:scale-105 transition-transform duration-500 hidden dark:block"
                    />
                    {/* Light variant */}
                    <Image
                      src={hw.srcLight}
                      alt={hw.name}
                      fill
                      className="object-cover group-hover:scale-105 transition-transform duration-500 dark:hidden"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-surface-800/60 dark:from-surface-900/80 via-transparent to-transparent" />
                  </div>
                  <div className="absolute bottom-0 left-0 right-0 px-4 py-3">
                    <span className="text-white text-sm font-medium">{hw.name}</span>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  )
}


function ComparisonSection() {
  const t = useTranslations('Home')
  const rows = [
    { feature: t('compRow1'), libre: true, ultra: false },
    { feature: t('compRow2'), libre: true, ultra: false },
    { feature: t('compRow3'), libre: true, ultra: false },
    { feature: t('compRow4'), libre: true, ultra: false },
    { feature: t('compRow5'), libre: true, ultra: false },
    { feature: t('compRow6'), libreText: t('compNone'), ultraText: t('compRequired') },
  ]

  return (
    <section className="relative py-14 lg:py-20">
      <div className="relative z-10 max-w-4xl mx-auto px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-12"
        >
          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-surface-900 dark:text-white mb-4">
            {t.rich('comparisonTitle', {
              muted: (chunks) => <span className="text-surface-400">{chunks}</span>,
            })}
          </h2>
          <p className="text-lg text-surface-600 dark:text-surface-400 max-w-2xl mx-auto">
            {t('comparisonSubtitle')}
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="bg-surface-50 dark:bg-surface-900/60 border border-surface-200 dark:border-white/5 rounded-2xl overflow-hidden"
        >
          {/* Table Header */}
          <div className="grid grid-cols-[1fr_80px_80px] sm:grid-cols-[1fr_140px_140px] items-center px-4 sm:px-6 py-4 border-b border-surface-200 dark:border-white/5 bg-surface-50 dark:bg-surface-900/80">
            <span className="text-surface-500 text-sm font-medium" />
            <span className="text-libre-500 text-xs sm:text-sm font-semibold text-center">LibreYOLO</span>
            <span className="text-surface-500 text-xs sm:text-sm font-semibold text-center">Ultralytics</span>
          </div>

          {/* Table Rows */}
          {rows.map((row, i) => (
            <div
              key={row.feature}
              className={`grid grid-cols-[1fr_80px_80px] sm:grid-cols-[1fr_140px_140px] items-center px-4 sm:px-6 py-4 ${
                i < rows.length - 1 ? 'border-b border-surface-100 dark:border-white/5' : ''
              }`}
            >
              <span className="text-surface-700 dark:text-surface-200 text-sm">{row.feature}</span>
              {row.libreText !== undefined ? (
                <>
                  <span className="text-emerald-600 dark:text-emerald-400 text-sm font-medium text-center">{row.libreText}</span>
                  <span className="text-red-500 text-sm font-medium text-center">{row.ultraText}</span>
                </>
              ) : (
                <>
                  <div className="flex justify-center">
                    <CheckCircle2 className="w-5 h-5 text-emerald-500" />
                  </div>
                  <div className="flex justify-center">
                    <XIcon className="w-5 h-5 text-red-400" />
                  </div>
                </>
              )}
            </div>
          ))}
        </motion.div>
      </div>
    </section>
  )
}


function CTASection() {
  const t = useTranslations('Home')
  return (
    <section className="relative py-16 lg:py-20 overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-b from-white dark:from-surface-950 to-surface-50 dark:to-surface-900/50" />
      <div className="absolute inset-0 gradient-mesh opacity-20 md:opacity-50" />

      <div className="relative z-10 max-w-4xl mx-auto px-5 md:px-6 lg:px-8 text-center">
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
        >
          <h2 className="text-[32px] leading-tight sm:text-4xl lg:text-5xl font-bold text-surface-900 dark:text-white mb-8 md:mb-10">
            {t.rich('ctaTitle', {
              accent: (chunks) => <span className="text-libre-500">{chunks}</span>,
            })}
          </h2>

          <div className="code-block rounded-xl max-w-md mx-auto mb-8 md:mb-10">
            <pre className="p-4 text-left">
              <code className="font-mono text-sm">
                <span className="text-surface-500">$</span> <span className="text-emerald-600 dark:text-emerald-400">pip install</span> <span className="text-libre-600 dark:text-libre-300">libreyolo</span>
              </code>
            </pre>
          </div>

          <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-center gap-3 sm:gap-4">
            <Link
              href="/docs"
              className="btn-primary flex items-center justify-center gap-2 px-6 sm:px-8 py-3.5 sm:py-4 bg-gradient-to-r from-libre-500 to-libre-600 rounded-xl text-white font-semibold"
            >
              {t('ctaReadDocs')}
              <ArrowRight className="w-5 h-5" />
            </Link>
            <Link
              href="/commercial"
              className="flex items-center justify-center gap-2 px-6 sm:px-8 py-3.5 sm:py-4 bg-white dark:bg-white/5 hover:bg-surface-100 dark:hover:bg-white/10 border border-surface-300 dark:border-white/10 rounded-xl text-surface-800 dark:text-white font-medium transition-all shadow-sm dark:shadow-none"
            >
              <Scale className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
              {t('ctaCommercialGuide')}
            </Link>
          </div>

          <div className="flex flex-col sm:flex-row sm:flex-wrap items-center justify-center gap-4 sm:gap-6 mt-8 text-sm">
            <Link href="/models" className="text-surface-500 hover:text-libre-500 transition-colors flex items-center gap-1.5">
              <Layers className="w-4 h-4" />
              {t('ctaModelZoo')}
            </Link>
            <a
              href="https://www.visionanalysis.org/?utm_source=libreyolo&utm_medium=referral&utm_campaign=benchmarks"
              target="_blank"
              rel="noopener"
              referrerPolicy="strict-origin-when-cross-origin"
              className="text-surface-500 hover:text-libre-500 transition-colors flex items-center gap-1.5"
            >
              <BarChart3 className="w-4 h-4" />
              {t('ctaBenchmarks')}
            </a>
          </div>
        </motion.div>
      </div>
    </section>
  )
}

export default function Home() {
  return (
    <>
      <HeroSection />
      <section className="px-6 pb-16">
        <div className="max-w-4xl mx-auto">
          <SupportCallout />
        </div>
      </section>
      <FeaturesSection />
      <SocialProofSection />
      {/* <DeployAnywhereSection /> */}
      {/* <ComparisonSection /> */}
      <CTASection />
    </>
  )
}
