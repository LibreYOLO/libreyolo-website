'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import Link from 'next/link'
import Image from 'next/image'
import {
  Unlock, Layers, ArrowRight,
  Code2, Scale, Copy, Check, CheckCircle2,
  Upload, RefreshCw, MessageSquareQuote,
  Cpu, X as XIcon, Map, ChevronLeft, ChevronRight
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
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(codeSnippet)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Background Effects */}
      <div className="absolute inset-0 grid-bg" />
      <div className="absolute inset-0 gradient-mesh hidden dark:block" />

      {/* Animated Orbs - hidden in light, visible in dark */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 hidden dark:block bg-libre-500/20 rounded-full blur-3xl animate-pulse-slow" />
      <div className="absolute bottom-1/4 right-1/4 w-80 h-80 hidden dark:block bg-emerald-500/15 rounded-full blur-3xl animate-pulse-slow" style={{ animationDelay: '1s' }} />

      {/* Hero Content */}
      <div className="relative z-10 max-w-7xl mx-auto px-6 lg:px-8 pt-32 pb-20">
        <motion.div
          initial="initial"
          animate="animate"
          variants={stagger}
          className="text-center"
        >
          {/* Badge */}
          <motion.div variants={fadeInUp} className="mb-8">
            <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-libre-500/10 border border-libre-500/20 text-libre-600 dark:text-libre-400 text-sm font-medium">
              <Unlock className="w-4 h-4" />
              100% MIT Licensed &bull; No AGPL Dependencies
            </span>
          </motion.div>

          {/* Main Headline */}
          <motion.h1
            variants={fadeInUp}
            className="text-5xl sm:text-6xl lg:text-7xl xl:text-8xl font-bold tracking-tight mb-6"
          >
            <span className="text-surface-800 dark:text-white">Object Detection.</span>
            <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-libre-500 via-libre-400 to-emerald-500 glow-text">
              100% MIT Licensed.
            </span>
          </motion.h1>

          {/* Sub-headline */}
          <motion.p
            variants={fadeInUp}
            className="text-lg sm:text-xl lg:text-2xl text-surface-600 dark:text-surface-400 max-w-3xl mx-auto mb-10 leading-relaxed"
          >
            Making YOLO accessible again, the way its creators always intended it to be.
            <span className="text-surface-800 dark:text-white font-medium"> A modern, MIT-licensed engine for training and deploying state-of-the-art object detection.</span>
          </motion.p>

          {/* CTAs */}
          <motion.div
            variants={fadeInUp}
            className="flex flex-col sm:flex-row items-center justify-center gap-4"
          >
            <Link
              href="/docs"
              className="btn-primary group flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-libre-500 to-libre-600 rounded-xl text-white font-semibold text-lg"
            >
              Get Started
              <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </Link>
            <a
              href="https://github.com/Libre-YOLO/libreyolo"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-8 py-4 bg-white dark:bg-white/5 hover:bg-surface-100 dark:hover:bg-white/10 border border-surface-300 dark:border-white/10 rounded-xl text-surface-800 dark:text-white font-medium text-lg transition-all shadow-sm dark:shadow-none"
            >
              <Code2 className="w-5 h-5 text-libre-500 dark:text-libre-400" />
              View on GitHub
            </a>
          </motion.div>

          {/* Roadmap Link */}
          <motion.div variants={fadeInUp} className="mt-6">
            <a
              href="https://github.com/orgs/LibreYOLO/projects/1/views/4"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 text-surface-500 hover:text-libre-500 text-sm transition-colors"
            >
              <Map className="w-4 h-4" />
              Check out our roadmap
              <ArrowRight className="w-3 h-3" />
            </a>
          </motion.div>

          {/* Hero Visual */}
          <motion.div
            variants={fadeInUp}
            className="mt-20 relative"
          >
            <div className="flex flex-col lg:flex-row items-stretch gap-0 max-w-6xl mx-auto">
              {/* Code Preview */}
              <div className="relative flex-1 w-full">
                {/* Glowing border effect */}
                <div className="absolute -inset-1 bg-gradient-to-r from-libre-500/30 via-emerald-500/30 to-libre-500/30 rounded-2xl lg:rounded-r-none blur-xl opacity-40" />

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
                      {copied ? 'Copied!' : 'Copy'}
                    </button>
                  </div>
                  <pre className="p-6 text-left">
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
                </div>
              </div>

              {/* Arrow connector - visible on lg screens */}
              <div className="hidden lg:flex items-center justify-center relative z-10">
                <div className="w-12 h-12 rounded-full bg-white dark:bg-surface-900 border border-libre-500/30 shadow-lg dark:shadow-none flex items-center justify-center -mx-6">
                  <ArrowRight className="w-5 h-5 text-libre-500 dark:text-libre-400" />
                </div>
              </div>

              {/* Result Image as Output Panel */}
              <div className="relative lg:max-w-sm w-full">
                <div className="absolute -inset-1 bg-gradient-to-r from-emerald-500/20 via-libre-500/20 to-emerald-500/20 rounded-2xl lg:rounded-l-none blur-xl opacity-40" />
                <div className="relative bg-surface-50 dark:bg-surface-900/80 backdrop-blur-sm border border-surface-200 dark:border-emerald-500/20 rounded-2xl lg:rounded-l-none overflow-hidden h-full">
                  <div className="flex items-center gap-2 px-4 py-3 bg-surface-100 dark:bg-surface-900/50 border-b border-surface-200 dark:border-white/5">
                    <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                    <span className="text-surface-500 text-sm font-mono">parkour_result.jpg</span>
                  </div>
                  <div className="p-3">
                    <img
                      src="https://raw.githubusercontent.com/LibreYOLO/libreyolo/main/libreyolo/assets/parkour_result.jpg"
                      alt="LibreYOLO Detection Result"
                      className="rounded-lg w-full"
                    />
                    <div className="mt-3 flex items-center justify-between text-xs">
                      <span className="text-emerald-400 font-mono">&check; Detected 1 object (person)</span>
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
  const [carouselPage, setCarouselPage] = useState(0)

  const testimonials = [
    {
      quote: "You have my hearty support for this. I'll be so happy when there is a good, community maintained, MIT licensed alternative to Ultralytics.",
      author: "u/Covered_in_bees_",
    },
    {
      quote: "This is so cool bro. If I see some place I can contribute, I will definitely do so!",
      author: "u/FedStan",
    },
    {
      quote: "This looks like a really good project, and I hope you are able to realize the full vision you seem to have for it. Very exciting.",
      author: "u/CalmBet",
    },
    {
      quote: "Nice work! I'll try it out this week and open issues/PRs.",
      author: "u/InternationalMany6",
    },
    {
      quote: "Looks pretty cool. It's good to see these sorts of developments.",
      author: "u/HistoricalMistake681",
    },
    {
      quote: "Thank you for sharing, this looks interesting.",
      author: "u/nemesis1836",
    },
    {
      quote: "Great initiative.",
      author: "u/Outrageous_Sort_8993",
    },
    {
      quote: "Good job! Are you hosting the models on HuggingFace?",
      author: "u/Winners-magic",
    },
    {
      quote: "Very nice, will add creating a node of it to my todo list.",
      author: "u/dr_hamilton",
    },
  ]

  const pageSize = 3
  const totalPages = Math.ceil(testimonials.length / pageSize)
  const visible = testimonials.slice(carouselPage * pageSize, carouselPage * pageSize + pageSize)

  return (
    <section className="relative py-14 lg:py-20 overflow-hidden">
      <div className="absolute inset-0 grid-bg opacity-30" />

      <div className="relative z-10 max-w-7xl mx-auto px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-10"
        >
          <span className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-amber-500/10 border border-amber-500/20 text-amber-600 dark:text-amber-400 text-xs font-medium mb-4">
            <MessageSquareQuote className="w-3.5 h-3.5" />
            r/computervision
          </span>
          <h2 className="text-3xl sm:text-4xl font-bold text-surface-900 dark:text-white">
            What the Community Says
          </h2>
        </motion.div>

        <div className="relative max-w-5xl mx-auto">
          {/* Prev button */}
          <button
            onClick={() => setCarouselPage((p) => (p === 0 ? totalPages - 1 : p - 1))}
            className="absolute -left-4 lg:-left-14 top-1/2 -translate-y-1/2 z-10 w-10 h-10 rounded-full bg-white dark:bg-surface-800 border border-surface-200 dark:border-white/10 shadow-md dark:shadow-none flex items-center justify-center text-surface-500 dark:text-surface-300 hover:text-surface-900 dark:hover:text-white hover:border-surface-300 dark:hover:border-white/20 transition-all"
            aria-label="Previous"
          >
            <ChevronLeft className="w-5 h-5" />
          </button>

          {/* Next button */}
          <button
            onClick={() => setCarouselPage((p) => (p === totalPages - 1 ? 0 : p + 1))}
            className="absolute -right-4 lg:-right-14 top-1/2 -translate-y-1/2 z-10 w-10 h-10 rounded-full bg-white dark:bg-surface-800 border border-surface-200 dark:border-white/10 shadow-md dark:shadow-none flex items-center justify-center text-surface-500 dark:text-surface-300 hover:text-surface-900 dark:hover:text-white hover:border-surface-300 dark:hover:border-white/20 transition-all"
            aria-label="Next"
          >
            <ChevronRight className="w-5 h-5" />
          </button>

          <div className="grid sm:grid-cols-3 gap-6">
            {visible.map((t, i) => (
              <motion.div
                key={`${carouselPage}-${i}`}
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

        {/* Dot indicators */}
        <div className="flex items-center justify-center gap-3 mt-8">
          {Array.from({ length: totalPages }).map((_, i) => (
            <button
              key={i}
              onClick={() => setCarouselPage(i)}
              className={`h-2.5 rounded-full transition-all duration-300 ${
                i === carouselPage
                  ? 'bg-libre-500 w-8'
                  : 'w-2.5 bg-surface-300 dark:bg-surface-600 hover:bg-surface-400 dark:hover:bg-surface-500'
              }`}
              aria-label={`Page ${i + 1}`}
            />
          ))}
        </div>
      </div>
    </section>
  )
}


function FeaturesSection() {
  const features = [
    {
      icon: Unlock,
      title: 'Truly MIT',
      description: 'No AGPL anywhere in the dependency chain. Use it in closed-source products, SaaS, embedded systems.',
      color: 'emerald'
    },
    {
      icon: Layers,
      title: 'One API, three architectures',
      description: 'YOLOX, YOLOv9, and RF-DETR behind a single LibreYOLO() call. Architecture, size, and class count auto-detected from weights.',
      color: 'libre'
    },
    {
      icon: RefreshCw,
      title: 'Batteries included',
      description: 'Any input format: paths, URLs, PIL, NumPy, OpenCV, tensors, bytes. Tiled inference for large images. Auto-download weights from HuggingFace.',
      color: 'cyan'
    },
    {
      icon: Code2,
      title: 'Train',
      description: 'Fine-tune on custom YOLO or COCO datasets with built-in augmentation, mixed precision, and early stopping. Resume from any checkpoint.',
      color: 'amber'
    },
    {
      icon: CheckCircle2,
      title: 'Validate',
      description: 'COCO-standard evaluation with mAP50, mAP50-95, precision, and recall on COCO or custom datasets. Per-class metrics and confusion matrix out of the box.',
      color: 'violet'
    },
    {
      icon: Upload,
      title: 'Export & deploy',
      description: 'One-line ONNX export with embedded metadata for easy deployment.',
      color: 'rose'
    }
  ]

  const colorClasses = {
    libre: { bg: 'bg-libre-500/10', text: 'text-libre-700 dark:text-libre-400', border: 'border-libre-500/20' },
    emerald: { bg: 'bg-emerald-500/10', text: 'text-emerald-700 dark:text-emerald-400', border: 'border-emerald-500/20' },
    cyan: { bg: 'bg-cyan-500/10', text: 'text-cyan-700 dark:text-cyan-400', border: 'border-cyan-500/20' },
    amber: { bg: 'bg-amber-500/10', text: 'text-amber-700 dark:text-amber-400', border: 'border-amber-500/20' },
    violet: { bg: 'bg-violet-500/10', text: 'text-violet-700 dark:text-violet-400', border: 'border-violet-500/20' },
    rose: { bg: 'bg-rose-500/10', text: 'text-rose-700 dark:text-rose-400', border: 'border-rose-500/20' }
  }

  return (
    <section className="relative py-14 lg:py-20">
      <div className="absolute inset-0 grid-bg opacity-50" />

      <div className="relative z-10 max-w-7xl mx-auto px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-12"
        >
          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-surface-900 dark:text-white mb-4">
            Why <span className="text-transparent bg-clip-text bg-gradient-to-r from-libre-500 to-emerald-500">LibreYOLO</span>
          </h2>
          <p className="text-lg text-surface-600 dark:text-surface-400 max-w-2xl mx-auto">
            Everything you need for object detection, nothing you don&apos;t.
          </p>
        </motion.div>

        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => {
            const colors = colorClasses[feature.color]
            return (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                className={`card-hover bg-surface-50 dark:bg-surface-900/50 border ${colors.border} rounded-2xl p-6 shadow-sm`}
              >
                <div className={`w-12 h-12 ${colors.bg} rounded-xl flex items-center justify-center mb-4`}>
                  <feature.icon className={`w-6 h-6 ${colors.text}`} />
                </div>
                <h3 className="text-lg font-semibold text-surface-900 dark:text-white mb-2">{feature.title}</h3>
                <p className="text-surface-600 dark:text-surface-400 text-sm leading-relaxed">{feature.description}</p>
              </motion.div>
            )
          })}
        </div>
      </div>
    </section>
  )
}


function DeployAnywhereSection() {
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
      <div className="absolute inset-0 grid-bg opacity-30" />
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
            Edge to Cloud
          </span>
          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-surface-900 dark:text-white mb-4">
            Deploy <span className="text-transparent bg-clip-text bg-gradient-to-r from-libre-500 to-emerald-500">Anywhere</span>
          </h2>
          <p className="text-lg text-surface-600 dark:text-surface-400 max-w-2xl mx-auto">
            Export once, run on any hardware. From $35 boards to datacenter GPUs.
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-12 items-start max-w-6xl mx-auto">
          {/* Export Formats */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
          >
            <h3 className="text-sm font-semibold text-surface-500 uppercase tracking-wider mb-6">Export Formats</h3>
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
            <h3 className="text-sm font-semibold text-surface-500 uppercase tracking-wider mb-6">Tested Hardware</h3>
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
  const rows = [
    { feature: 'Use in proprietary software', libre: true, ultra: false },
    { feature: 'Sell products containing it', libre: true, ultra: false },
    { feature: 'No source disclosure required', libre: true, ultra: false },
    { feature: 'Fine-tune & keep weights private', libre: true, ultra: false },
    { feature: 'Distill into a new model', libre: true, ultra: false },
    { feature: 'Commercial use fee', libreText: 'None', ultraText: 'Required' },
  ]

  return (
    <section className="relative py-14 lg:py-20">
      <div className="absolute inset-0 grid-bg opacity-30" />

      <div className="relative z-10 max-w-4xl mx-auto px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-12"
        >
          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-surface-900 dark:text-white mb-4">
            LibreYOLO vs <span className="text-surface-400">Ultralytics</span>
          </h2>
          <p className="text-lg text-surface-600 dark:text-surface-400 max-w-2xl mx-auto">
            MIT means you own your work. No surprises.
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
  return (
    <section className="relative py-14 lg:py-20 overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-b from-white dark:from-surface-950 to-surface-50 dark:to-surface-900/50" />
      <div className="absolute inset-0 gradient-mesh opacity-50" />

      <div className="relative z-10 max-w-4xl mx-auto px-6 lg:px-8 text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
        >
          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-surface-900 mb-10">
            Start Building <span className="text-libre-500">Today</span>
          </h2>

          <div className="code-block rounded-xl max-w-md mx-auto mb-10">
            <pre className="p-4 text-left">
              <code className="font-mono text-sm">
                <span className="text-surface-500">$</span> <span className="text-emerald-600 dark:text-emerald-400">pip install</span> <span className="text-libre-600 dark:text-libre-300">libreyolo</span>
              </code>
            </pre>
          </div>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link
              href="/docs"
              className="btn-primary flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-libre-500 to-libre-600 rounded-xl text-white font-semibold"
            >
              Read the Docs
              <ArrowRight className="w-5 h-5" />
            </Link>
            <Link
              href="/commercial"
              className="flex items-center gap-2 px-8 py-4 bg-white dark:bg-white/5 hover:bg-surface-100 dark:hover:bg-white/10 border border-surface-300 dark:border-white/10 rounded-xl text-surface-800 dark:text-white font-medium transition-all shadow-sm dark:shadow-none"
            >
              <Scale className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
              Commercial Guide
            </Link>
          </div>

          <div className="flex flex-wrap items-center justify-center gap-6 mt-8 text-sm">
            <Link href="/models" className="text-surface-500 hover:text-libre-500 transition-colors flex items-center gap-1.5">
              <Layers className="w-4 h-4" />
              Model Zoo
            </Link>
            <Link href="/datasets" className="text-surface-500 hover:text-libre-500 transition-colors flex items-center gap-1.5">
              <Layers className="w-4 h-4" />
              Dataset Zoo
            </Link>
            <a
              href="https://github.com/orgs/LibreYOLO/projects/1/views/4"
              target="_blank"
              rel="noopener noreferrer"
              className="text-surface-500 hover:text-libre-500 transition-colors flex items-center gap-1.5"
            >
              <Map className="w-4 h-4" />
              Roadmap
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
      <FeaturesSection />
      <SocialProofSection />
      <DeployAnywhereSection />
      <ComparisonSection />
      <CTASection />
    </>
  )
}
