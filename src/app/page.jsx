'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import Link from 'next/link'
import Image from 'next/image'
import {
  Unlock, Layers, ArrowRight,
  Code2, Scale, Copy, Check, CheckCircle2,
  Upload, RefreshCw, MessageSquareQuote,
  Cpu, X as XIcon
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
      <div className="absolute inset-0 gradient-mesh" />

      {/* Animated Orbs */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-libre-500/20 rounded-full blur-3xl animate-pulse-slow" />
      <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-emerald-500/15 rounded-full blur-3xl animate-pulse-slow" style={{ animationDelay: '1s' }} />

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
            <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-libre-500/10 border border-libre-500/20 text-libre-400 text-sm font-medium">
              <Unlock className="w-4 h-4" />
              100% MIT Licensed • No AGPL Dependencies
            </span>
          </motion.div>

          {/* Main Headline */}
          <motion.h1
            variants={fadeInUp}
            className="text-5xl sm:text-6xl lg:text-7xl xl:text-8xl font-bold tracking-tight mb-6"
          >
            <span className="text-white">Object Detection.</span>
            <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-libre-400 via-libre-300 to-emerald-400 glow-text">
              Unrestricted.
            </span>
          </motion.h1>

          {/* Sub-headline */}
          <motion.p
            variants={fadeInUp}
            className="text-lg sm:text-xl lg:text-2xl text-surface-300 max-w-3xl mx-auto mb-10 leading-relaxed"
          >
            A modern training and inference engine for state-of-the-art YOLO models.
            <span className="text-white"> Built for commercial applications, scientists, and the community.</span>
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
              className="flex items-center gap-2 px-8 py-4 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl text-white font-medium text-lg transition-all"
            >
              <Code2 className="w-5 h-5 text-libre-400" />
              View on GitHub
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
                <div className="absolute -inset-1 bg-gradient-to-r from-libre-500/50 via-emerald-500/50 to-libre-500/50 rounded-2xl lg:rounded-r-none blur-xl opacity-30" />

                <div className="relative code-block rounded-2xl lg:rounded-r-none overflow-hidden h-full">
                  <div className="flex items-center justify-between px-4 py-3 bg-surface-900/50 border-b border-white/5">
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
                      className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium text-surface-400 hover:text-white hover:bg-white/10 transition-all"
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
                            <td><span className="token-keyword">from</span> <span className="text-libre-300">libreyolo</span> <span className="token-keyword">import</span> <span className="text-emerald-400">LibreYOLO</span>, <span className="text-emerald-400">SAMPLE_IMAGE</span></td>
                          </tr>
                          <tr><td className="pr-4 text-right select-none text-surface-600 w-6">2</td><td></td></tr>
                          <tr>
                            <td className="pr-4 text-right select-none text-surface-600 align-top w-6">3</td>
                            <td><span className="text-surface-300">model</span> <span className="text-libre-400">=</span> <span className="text-emerald-400">LibreYOLO</span>(<span className="token-string">&quot;LibreYOLOXs.pt&quot;</span>)</td>
                          </tr>
                          <tr>
                            <td className="pr-4 text-right select-none text-surface-600 align-top w-6">4</td>
                            <td><span className="text-surface-300">results</span> <span className="text-libre-400">=</span> <span className="text-surface-300">model</span>(<span className="text-emerald-400">SAMPLE_IMAGE</span>, <span className="text-surface-300">save</span><span className="text-libre-400">=</span><span className="text-emerald-400">True</span>)</td>
                          </tr>
                        </tbody>
                      </table>
                    </code>
                  </pre>
                </div>
              </div>

              {/* Arrow connector - visible on lg screens */}
              <div className="hidden lg:flex items-center justify-center relative z-10">
                <div className="w-12 h-12 rounded-full bg-surface-900 border border-libre-500/30 flex items-center justify-center -mx-6">
                  <ArrowRight className="w-5 h-5 text-libre-400" />
                </div>
              </div>

              {/* Result Image as Output Panel */}
              <div className="relative lg:max-w-sm w-full">
                <div className="absolute -inset-1 bg-gradient-to-r from-emerald-500/30 via-libre-500/30 to-emerald-500/30 rounded-2xl lg:rounded-l-none blur-xl opacity-30" />
                <div className="relative bg-surface-900/80 backdrop-blur-sm border border-emerald-500/20 rounded-2xl lg:rounded-l-none overflow-hidden h-full">
                  <div className="flex items-center gap-2 px-4 py-3 bg-surface-900/50 border-b border-white/5">
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
                      <span className="text-emerald-400 font-mono">✓ Detected 1 object (person)</span>
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
  const testimonials = [
    {
      quote: "This is really damn good! People need to take note of this!!!",
      author: "u/InternationalMany6",
    },
    {
      quote: "I'll be so happy when there is a good, community maintained, MIT licensed alternative to Ultralytics.",
      author: "u/Covered_in_bees_",
    },
    {
      quote: "This is so cool bro. If I see some place I can contribute, I will definitely do so!",
      author: "u/FedStan",
    },
  ]

  return (
    <section className="relative py-20 lg:py-28 overflow-hidden">
      <div className="absolute inset-0 grid-bg opacity-30" />

      <div className="relative z-10 max-w-7xl mx-auto px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-12"
        >
          <span className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-amber-500/10 border border-amber-500/20 text-amber-400 text-xs font-medium mb-4">
            <MessageSquareQuote className="w-3.5 h-3.5" />
            r/computervision
          </span>
          <h2 className="text-3xl sm:text-4xl font-bold text-white">
            What the Community Says
          </h2>
        </motion.div>

        <div className="grid sm:grid-cols-3 gap-6 max-w-5xl mx-auto">
          {testimonials.map((t, i) => (
            <motion.div
              key={t.author}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.1 }}
              className="relative bg-surface-900/60 backdrop-blur-sm border border-white/5 rounded-2xl p-6"
            >
              <div className="absolute top-4 left-5 text-libre-500/20 text-4xl font-serif leading-none select-none">&ldquo;</div>
              <p className="text-surface-200 text-sm leading-relaxed mb-4 pt-4 italic">
                &ldquo;{t.quote}&rdquo;
              </p>
              <p className="text-surface-500 text-xs font-mono">{t.author}</p>
            </motion.div>
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
      description: 'No AGPL anywhere in the dependency chain. Use it in closed-source products, SaaS, embedded systems — zero licensing risk.',
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
      description: 'Any input format — paths, URLs, PIL, NumPy, OpenCV, tensors, bytes. Tiled inference for large images. Auto-download weights from HuggingFace.',
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
    libre: { bg: 'bg-libre-500/10', text: 'text-libre-400', border: 'border-libre-500/20' },
    emerald: { bg: 'bg-emerald-500/10', text: 'text-emerald-400', border: 'border-emerald-500/20' },
    cyan: { bg: 'bg-cyan-500/10', text: 'text-cyan-400', border: 'border-cyan-500/20' },
    amber: { bg: 'bg-amber-500/10', text: 'text-amber-400', border: 'border-amber-500/20' },
    violet: { bg: 'bg-violet-500/10', text: 'text-violet-400', border: 'border-violet-500/20' },
    rose: { bg: 'bg-rose-500/10', text: 'text-rose-400', border: 'border-rose-500/20' }
  }

  return (
    <section className="relative py-24 lg:py-32">
      <div className="absolute inset-0 grid-bg opacity-50" />

      <div className="relative z-10 max-w-7xl mx-auto px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-white mb-4">
            Why <span className="text-transparent bg-clip-text bg-gradient-to-r from-libre-400 to-emerald-400">LibreYOLO</span>
          </h2>
          <p className="text-lg text-surface-400 max-w-2xl mx-auto">
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
                className={`card-hover bg-surface-900/50 backdrop-blur-sm border ${colors.border} rounded-2xl p-6`}
              >
                <div className={`w-12 h-12 ${colors.bg} rounded-xl flex items-center justify-center mb-4`}>
                  <feature.icon className={`w-6 h-6 ${colors.text}`} />
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">{feature.title}</h3>
                <p className="text-surface-400 text-sm leading-relaxed">{feature.description}</p>
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
    { name: 'Jetson Nano', src: '/hardware/jetson-nano.jpg' },
    { name: 'Jetson Orin', src: '/hardware/jetson-orin.jpg' },
    { name: 'Raspberry Pi', src: '/hardware/raspberry-pi.png' },
    { name: 'NVIDIA GPU', src: '/hardware/nvidia-gpu.png' },
  ]

  return (
    <section className="relative py-24 lg:py-32 overflow-hidden">
      <div className="absolute inset-0 grid-bg opacity-30" />
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-libre-500/5 rounded-full blur-3xl" />

      <div className="relative z-10 max-w-7xl mx-auto px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <span className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-libre-500/10 border border-libre-500/20 text-libre-400 text-xs font-medium mb-4">
            <Cpu className="w-3.5 h-3.5" />
            Edge to Cloud
          </span>
          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-white mb-4">
            Deploy <span className="text-transparent bg-clip-text bg-gradient-to-r from-libre-400 to-emerald-400">Anywhere</span>
          </h2>
          <p className="text-lg text-surface-400 max-w-2xl mx-auto">
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
            <h3 className="text-sm font-semibold text-surface-400 uppercase tracking-wider mb-6">Export Formats</h3>
            <div className="space-y-3">
              {exportFormats.map((fmt) => (
                <div
                  key={fmt.name}
                  className="flex items-center justify-between bg-surface-900/60 border border-white/5 rounded-xl px-5 py-3.5"
                >
                  <span className="text-white font-medium text-sm">{fmt.name}</span>
                  <div className="flex gap-2">
                    {fmt.variants.map((v) => (
                      <span
                        key={v}
                        className="px-2.5 py-1 rounded-md bg-libre-500/10 text-libre-300 text-xs font-mono"
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
            <h3 className="text-sm font-semibold text-surface-400 uppercase tracking-wider mb-6">Tested Hardware</h3>
            <div className="grid grid-cols-2 gap-4">
              {hardware.map((hw) => (
                <div
                  key={hw.name}
                  className="group relative bg-surface-900/60 border border-white/5 rounded-xl overflow-hidden"
                >
                  <div className="aspect-[4/3] relative">
                    <Image
                      src={hw.src}
                      alt={hw.name}
                      fill
                      className="object-cover group-hover:scale-105 transition-transform duration-500"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-surface-950/90 via-surface-950/20 to-transparent" />
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
    <section className="relative py-24 lg:py-32">
      <div className="absolute inset-0 grid-bg opacity-30" />

      <div className="relative z-10 max-w-4xl mx-auto px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-white mb-4">
            LibreYOLO vs <span className="text-surface-400">Ultralytics</span>
          </h2>
          <p className="text-lg text-surface-400 max-w-2xl mx-auto">
            MIT means you own your work. No surprises.
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="bg-surface-900/60 backdrop-blur-sm border border-white/5 rounded-2xl overflow-hidden"
        >
          {/* Table Header */}
          <div className="grid grid-cols-[1fr_80px_80px] sm:grid-cols-[1fr_140px_140px] items-center px-4 sm:px-6 py-4 border-b border-white/5 bg-surface-900/80">
            <span className="text-surface-500 text-sm font-medium" />
            <span className="text-libre-400 text-xs sm:text-sm font-semibold text-center">LibreYOLO</span>
            <span className="text-surface-400 text-xs sm:text-sm font-semibold text-center">Ultralytics</span>
          </div>

          {/* Table Rows */}
          {rows.map((row, i) => (
            <div
              key={row.feature}
              className={`grid grid-cols-[1fr_80px_80px] sm:grid-cols-[1fr_140px_140px] items-center px-4 sm:px-6 py-4 ${
                i < rows.length - 1 ? 'border-b border-white/5' : ''
              }`}
            >
              <span className="text-surface-200 text-sm">{row.feature}</span>
              {row.libreText !== undefined ? (
                <>
                  <span className="text-emerald-400 text-sm font-medium text-center">{row.libreText}</span>
                  <span className="text-red-400 text-sm font-medium text-center">{row.ultraText}</span>
                </>
              ) : (
                <>
                  <div className="flex justify-center">
                    <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                  </div>
                  <div className="flex justify-center">
                    <XIcon className="w-5 h-5 text-red-400" />
                  </div>
                </>
              )}
            </div>
          ))}
        </motion.div>

        {/* Community callout */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="mt-8 text-center"
        >
          <p className="text-surface-400 text-sm">
            <span className="text-libre-400 font-medium">Community Driven</span> — built on the{' '}
            <a
              href="https://github.com/testdummyvt/yolov9mit"
              target="_blank"
              rel="noopener noreferrer"
              className="text-libre-300 hover:text-libre-200 underline underline-offset-2 transition-colors"
            >
              @testdummyvt fork
            </a>{' '}
            that added RT-DETR + NMS-free YOLOv9 under the MIT license.
          </p>
        </motion.div>
      </div>
    </section>
  )
}


function CTASection() {
  return (
    <section className="relative py-24 lg:py-32 overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-b from-surface-950 to-surface-900/50" />
      <div className="absolute inset-0 gradient-mesh opacity-50" />

      <div className="relative z-10 max-w-4xl mx-auto px-6 lg:px-8 text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
        >
          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-white mb-10">
            Start Building <span className="text-libre-400">Today</span>
          </h2>

          <div className="code-block rounded-xl max-w-md mx-auto mb-10">
            <pre className="p-4 text-left">
              <code className="font-mono text-sm">
                <span className="text-surface-500">$</span> <span className="text-emerald-400">pip install</span> <span className="text-libre-300">libreyolo</span>
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
              className="flex items-center gap-2 px-8 py-4 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl text-white font-medium transition-all"
            >
              <Scale className="w-5 h-5 text-emerald-400" />
              Commercial Guide
            </Link>
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
      <SocialProofSection />
      <FeaturesSection />
      <DeployAnywhereSection />
      <ComparisonSection />
      <CTASection />
    </>
  )
}
