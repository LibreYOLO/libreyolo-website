'use client'

import { motion } from 'framer-motion'
import {
  FlaskConical, Eye, Layers, Brain, CheckCircle2, XCircle,
  Lock, Unlock, ArrowRight, Zap, Copy, Check, ChevronLeft, ChevronRight
} from 'lucide-react'
import { useState } from 'react'

function CopyButton({ code }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <button
      onClick={handleCopy}
      className="p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
      title="Copy to clipboard"
    >
      {copied ? (
        <Check className="w-4 h-4 text-emerald-400" />
      ) : (
        <Copy className="w-4 h-4 text-surface-400" />
      )}
    </button>
  )
}

function ImageCarousel({ images, accentColor = 'cyan' }) {
  const [activeIndex, setActiveIndex] = useState(0)

  const colorClasses = {
    cyan: { border: 'border-cyan-500/20', dot: 'bg-cyan-400', activeBg: 'bg-cyan-500/20', activeText: 'text-cyan-400', hoverBg: 'hover:bg-cyan-500/20', glow: 'from-cyan-500/30 via-emerald-500/30 to-cyan-500/30' },
    fuchsia: { border: 'border-fuchsia-500/20', dot: 'bg-fuchsia-400', activeBg: 'bg-fuchsia-500/20', activeText: 'text-fuchsia-400', hoverBg: 'hover:bg-fuchsia-500/20', glow: 'from-fuchsia-500/30 via-violet-500/30 to-fuchsia-500/30' }
  }
  const colors = colorClasses[accentColor]

  const goToPrev = () => setActiveIndex((prev) => (prev === 0 ? images.length - 1 : prev - 1))
  const goToNext = () => setActiveIndex((prev) => (prev === images.length - 1 ? 0 : prev + 1))

  return (
    <div className="relative lg:max-w-sm w-full">
      <div className={`absolute -inset-1 bg-gradient-to-r ${colors.glow} rounded-2xl lg:rounded-l-none blur-xl opacity-30`} />
      <div className={`relative bg-surface-900/80 backdrop-blur-sm border ${colors.border} rounded-2xl lg:rounded-l-none overflow-hidden h-full flex flex-col`}>
        <div className="flex items-center justify-between px-4 py-3 bg-surface-900/50 border-b border-white/5">
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${colors.dot} animate-pulse`} />
            <span className="text-surface-500 text-sm font-mono">{images[activeIndex].filename}</span>
          </div>
          {/* Prev/Next buttons */}
          <div className="flex items-center gap-1">
            <button
              onClick={goToPrev}
              className={`p-1.5 rounded-lg bg-white/5 ${colors.hoverBg} transition-colors`}
              aria-label="Previous"
            >
              <ChevronLeft className={`w-4 h-4 ${colors.activeText}`} />
            </button>
            <span className="text-surface-500 text-xs font-mono px-2">{activeIndex + 1}/{images.length}</span>
            <button
              onClick={goToNext}
              className={`p-1.5 rounded-lg bg-white/5 ${colors.hoverBg} transition-colors`}
              aria-label="Next"
            >
              <ChevronRight className={`w-4 h-4 ${colors.activeText}`} />
            </button>
          </div>
        </div>

        <div className="p-3 flex-1 flex flex-col">
          <img
            src={images[activeIndex].src}
            alt={images[activeIndex].alt}
            className="rounded-lg w-full"
          />
          {/* Layer/Method selector */}
          <div className="mt-3 flex flex-wrap gap-1.5">
            {images.map((img, idx) => (
              <button
                key={idx}
                onClick={() => setActiveIndex(idx)}
                className={`px-2 py-1 rounded-md text-xs font-medium transition-all ${
                  activeIndex === idx
                    ? `${colors.activeBg} ${colors.activeText}`
                    : 'bg-white/5 text-surface-400 hover:bg-white/10'
                }`}
              >
                {img.label}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

function HeroSection() {
  return (
    <section className="relative pt-24 lg:pt-32 pb-16">
      <div className="absolute inset-0 grid-bg opacity-30" />
      <div className="absolute top-1/4 right-1/4 w-96 h-96 bg-violet-500/10 rounded-full blur-3xl" />
      
      <div className="relative z-10 max-w-7xl mx-auto px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="max-w-3xl"
        >
          <div className="flex items-center gap-2 text-violet-400 text-sm font-medium mb-4">
            <FlaskConical className="w-4 h-4" />
            Research & Science
          </div>
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-white mb-6">
            Built for <span className="text-transparent bg-clip-text bg-gradient-to-r from-violet-400 to-fuchsia-400">Discovery</span>
          </h1>
          <p className="text-lg text-surface-400 leading-relaxed">
            A codebase designed for researchers, not just users. Modify freely, inspect deeply, 
            and publish without paying for the privilege.
          </p>
        </motion.div>
      </div>
    </section>
  )
}

function NoPaywallSection() {
  return (
    <section className="py-12">
      <div className="max-w-7xl mx-auto px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="bg-gradient-to-r from-violet-500/10 to-fuchsia-500/10 border border-violet-500/30 rounded-2xl p-8"
        >
          <div className="flex items-start gap-4">
            <div className="p-3 rounded-xl bg-violet-500/20 flex-shrink-0">
              <Unlock className="w-8 h-8 text-violet-400" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-white mb-4">No "Research License" Required</h2>
              <p className="text-surface-300 mb-6">
                Some frameworks charge separate fees for "research" or "science" licenses just to 
                let you inspect the code without releasing your work. <strong className="text-white">Libre-YOLO doesn't.</strong>
              </p>
              
              <div className="grid sm:grid-cols-2 gap-6">
                <div className="bg-surface-900/50 rounded-xl p-5 border border-white/5">
                  <h3 className="text-white font-semibold mb-3 flex items-center gap-2">
                    <Lock className="w-4 h-4 text-red-400" />
                    Typical "Science Licenses"
                  </h3>
                  <ul className="space-y-2 text-sm text-surface-400">
                    <li className="flex items-start gap-2">
                      <XCircle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
                      Pay to keep research private
                    </li>
                    <li className="flex items-start gap-2">
                      <XCircle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
                      Fees for commercial R&D use
                    </li>
                    <li className="flex items-start gap-2">
                      <XCircle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
                      Complex license tiers
                    </li>
                    <li className="flex items-start gap-2">
                      <XCircle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
                      Legal review for publication
                    </li>
                  </ul>
                </div>
                
                <div className="bg-emerald-500/5 rounded-xl p-5 border border-emerald-500/20">
                  <h3 className="text-white font-semibold mb-3 flex items-center gap-2">
                    <Unlock className="w-4 h-4 text-emerald-400" />
                    Libre-YOLO (MIT)
                  </h3>
                  <ul className="space-y-2 text-sm text-surface-400">
                    <li className="flex items-start gap-2">
                      <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0 mt-0.5" />
                      <span>Investigate freely, <strong className="text-emerald-400">keep it private</strong></span>
                    </li>
                    <li className="flex items-start gap-2">
                      <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0 mt-0.5" />
                      Zero fees for any use case
                    </li>
                    <li className="flex items-start gap-2">
                      <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0 mt-0.5" />
                      One simple license: MIT
                    </li>
                    <li className="flex items-start gap-2">
                      <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0 mt-0.5" />
                      Publish without legal concerns
                    </li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  )
}

function ExplainabilitySection() {
  return (
    <section className="py-16">
      <div className="max-w-7xl mx-auto px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-12"
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-violet-500/10 border border-violet-500/20 text-violet-400 text-sm font-medium mb-4">
            <Eye className="w-4 h-4" />
            Native Explainability
          </div>
          <h2 className="text-3xl sm:text-4xl font-bold text-white mb-4">
            See Inside the Black Box
          </h2>
          <p className="text-lg text-surface-400 max-w-2xl mx-auto">
            Built-in tools for interpretability and explainability. No external dependencies,
            no complex setup - just flags and function calls.
          </p>
        </motion.div>

        <div className="space-y-12">
          {/* Feature Maps */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 rounded-lg bg-cyan-500/10">
                <Layers className="w-5 h-5 text-cyan-400" />
              </div>
              <h3 className="text-xl font-semibold text-white">Feature Map Extraction</h3>
            </div>
            <p className="text-surface-400 mb-6 max-w-2xl">
              One flag. That's all it takes to save intermediate activations from every layer.
              Perfect for understanding what your model "sees" at each stage.
            </p>

            <div className="flex flex-col lg:flex-row items-stretch gap-0 max-w-5xl mx-auto">
              {/* Code Block */}
              <div className="relative flex-1 w-full">
                <div className="absolute -inset-1 bg-gradient-to-r from-cyan-500/50 via-emerald-500/50 to-cyan-500/50 rounded-2xl lg:rounded-r-none blur-xl opacity-30" />
                <div className="relative code-block rounded-2xl lg:rounded-r-none overflow-hidden h-full">
                  <div className="flex items-center justify-between px-4 py-3 bg-surface-900/50 border-b border-white/5">
                    <div className="flex items-center gap-2">
                      <div className="flex gap-2">
                        <span className="w-3 h-3 rounded-full bg-red-500/80" />
                        <span className="w-3 h-3 rounded-full bg-yellow-500/80" />
                        <span className="w-3 h-3 rounded-full bg-green-500/80" />
                      </div>
                      <span className="ml-4 text-surface-500 text-sm font-mono">feature_maps.py</span>
                    </div>
                    <CopyButton code={`from libreyolo import LIBREYOLO

# Enable feature map saving in constructor
model = LIBREYOLO(
    model_path="libreyolo11m.pt",
    size="m",
    save_feature_maps=True
)

# Run inference - feature maps auto-saved
results = model(image="parkour.jpg")

# Saved to: runs/feature_maps/`} />
                  </div>
                  <pre className="p-6 overflow-x-auto">
                    <code className="font-mono text-sm">
                      <span className="token-keyword">from</span> <span className="text-violet-300">libreyolo</span> <span className="token-keyword">import</span> <span className="text-emerald-400">LIBREYOLO</span>{'\n\n'}
                      <span className="token-comment"># Enable feature map saving in constructor</span>{'\n'}
                      <span className="text-surface-300">model</span> <span className="text-violet-400">=</span> <span className="text-emerald-400">LIBREYOLO</span>({'\n'}
                      {'    '}<span className="text-surface-300">model_path</span><span className="text-violet-400">=</span><span className="token-string">"libreyolo11m.pt"</span>,{'\n'}
                      {'    '}<span className="text-surface-300">size</span><span className="text-violet-400">=</span><span className="token-string">"m"</span>,{'\n'}
                      {'    '}<span className="text-surface-300">save_feature_maps</span><span className="text-violet-400">=</span><span className="text-emerald-400">True</span>{'\n'}
                      ){'\n\n'}
                      <span className="token-comment"># Run inference - feature maps auto-saved</span>{'\n'}
                      <span className="text-surface-300">results</span> <span className="text-violet-400">=</span> <span className="text-surface-300">model</span>(<span className="text-surface-300">image</span><span className="text-violet-400">=</span><span className="token-string">"parkour.jpg"</span>){'\n\n'}
                      <span className="token-comment"># Saved to: runs/feature_maps/</span>
                    </code>
                  </pre>
                </div>
              </div>

              {/* Arrow connector */}
              <div className="hidden lg:flex items-center justify-center relative z-10">
                <div className="w-12 h-12 rounded-full bg-surface-900 border border-cyan-500/30 flex items-center justify-center -mx-6">
                  <ArrowRight className="w-5 h-5 text-cyan-400" />
                </div>
              </div>

              {/* Feature Map Output Carousel */}
              <ImageCarousel
                accentColor="cyan"
                images={[
                  { label: 'P1', filename: 'backbone_p1.png', src: '/feature_maps/backbone_p1.png', alt: 'Backbone P1 feature map' },
                  { label: 'P2', filename: 'backbone_p2.png', src: '/feature_maps/backbone_p2.png', alt: 'Backbone P2 feature map' },
                  { label: 'P3', filename: 'backbone_p3.png', src: '/feature_maps/backbone_p3.png', alt: 'Backbone P3 feature map' },
                  { label: 'P4', filename: 'backbone_p4.png', src: '/feature_maps/backbone_p4.png', alt: 'Backbone P4 feature map' },
                  { label: 'P5', filename: 'backbone_p5.png', src: '/feature_maps/backbone_p5.png', alt: 'Backbone P5 feature map' },
                  { label: 'SPPF', filename: 'backbone_sppf_P5.png', src: '/feature_maps/backbone_sppf_P5.png', alt: 'SPPF feature map' },
                  { label: 'Neck 1', filename: 'neck_c2f11.png', src: '/feature_maps/neck_c2f11.png', alt: 'Neck C2F11 feature map' },
                  { label: 'Neck 2', filename: 'neck_c2f22.png', src: '/feature_maps/neck_c2f22.png', alt: 'Neck C2F22 feature map' },
                  { label: 'Head 8', filename: 'head8_conv21.png', src: '/feature_maps/head8_conv21.png', alt: 'Head 8 feature map' },
                  { label: 'Head 16', filename: 'head16_conv21.png', src: '/feature_maps/head16_conv21.png', alt: 'Head 16 feature map' },
                  { label: 'Head 32', filename: 'head32_conv21.png', src: '/feature_maps/head32_conv21.png', alt: 'Head 32 feature map' },
                ]}
              />
            </div>
          </motion.div>

          {/* CAM Visualizations */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 rounded-lg bg-fuchsia-500/10">
                <Brain className="w-5 h-5 text-fuchsia-400" />
              </div>
              <h3 className="text-xl font-semibold text-white">CAM Visualizations</h3>
              <span className="px-2 py-1 rounded-md bg-amber-500/20 text-amber-400 text-xs font-medium">
                Experimental
              </span>
            </div>
            <p className="text-surface-400 mb-6 max-w-2xl">
              7 built-in Class Activation Map methods. Call <code className="text-fuchsia-400 font-mono text-sm">model.explain()</code> to
              generate heatmaps showing what your model focuses on. This feature is experimental and results may vary.
            </p>

            <div className="max-w-2xl">
              {/* Code Block */}
              <div className="relative w-full">
                <div className="absolute -inset-1 bg-gradient-to-r from-fuchsia-500/50 via-violet-500/50 to-fuchsia-500/50 rounded-2xl blur-xl opacity-30" />
                <div className="relative code-block rounded-2xl overflow-hidden">
                  <div className="flex items-center justify-between px-4 py-3 bg-surface-900/50 border-b border-white/5">
                    <div className="flex items-center gap-2">
                      <div className="flex gap-2">
                        <span className="w-3 h-3 rounded-full bg-red-500/80" />
                        <span className="w-3 h-3 rounded-full bg-yellow-500/80" />
                        <span className="w-3 h-3 rounded-full bg-green-500/80" />
                      </div>
                      <span className="ml-4 text-surface-500 text-sm font-mono">explainability.py</span>
                    </div>
                    <CopyButton code={`from libreyolo import LIBREYOLO

model = LIBREYOLO(model_path="libreyolo11m.pt", size="m")

# One-line CAM visualization
result = model.explain(
    image="parkour.jpg",
    method="gradcam",
    target_layer="neck_c2f22",
    save=True
)

# Returns heatmap and overlay
print(result["heatmap"].shape)`} />
                  </div>
                  <pre className="p-6 overflow-x-auto">
                    <code className="font-mono text-sm">
                      <span className="token-keyword">from</span> <span className="text-violet-300">libreyolo</span> <span className="token-keyword">import</span> <span className="text-emerald-400">LIBREYOLO</span>{'\n\n'}
                      <span className="text-surface-300">model</span> <span className="text-violet-400">=</span> <span className="text-emerald-400">LIBREYOLO</span>(<span className="text-surface-300">model_path</span><span className="text-violet-400">=</span><span className="token-string">"libreyolo11m.pt"</span>, <span className="text-surface-300">size</span><span className="text-violet-400">=</span><span className="token-string">"m"</span>){'\n\n'}
                      <span className="token-comment"># One-line CAM visualization</span>{'\n'}
                      <span className="text-surface-300">result</span> <span className="text-violet-400">=</span> <span className="text-surface-300">model</span>.<span className="token-function">explain</span>({'\n'}
                      {'    '}<span className="text-surface-300">image</span><span className="text-violet-400">=</span><span className="token-string">"parkour.jpg"</span>,{'\n'}
                      {'    '}<span className="text-surface-300">method</span><span className="text-violet-400">=</span><span className="token-string">"gradcam"</span>,{'\n'}
                      {'    '}<span className="text-surface-300">target_layer</span><span className="text-violet-400">=</span><span className="token-string">"neck_c2f22"</span>,{'\n'}
                      {'    '}<span className="text-surface-300">save</span><span className="text-violet-400">=</span><span className="text-emerald-400">True</span>{'\n'}
                      ){'\n\n'}
                      <span className="token-comment"># Returns heatmap and overlay</span>{'\n'}
                      <span className="token-function">print</span>(<span className="text-surface-300">result</span>[<span className="token-string">"heatmap"</span>].<span className="text-surface-300">shape</span>)
                    </code>
                  </pre>
                </div>
              </div>
            </div>
          </motion.div>

        </div>
      </div>
    </section>
  )
}


function ModificationSection() {
  return (
    <section className="py-16">
      <div className="max-w-7xl mx-auto px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
        >
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 rounded-lg bg-fuchsia-500/10">
              <Zap className="w-5 h-5 text-fuchsia-400" />
            </div>
            <h2 className="text-2xl font-semibold text-white">Designed for Modification</h2>
          </div>

          <p className="text-surface-400 mb-8 max-w-3xl">
            The codebase is written to be easy to read and modify. Want to test a new backbone?
            Experiment with a custom attention mechanism? Add a novel loss function? Go for it.
          </p>

          <div className="max-w-2xl">
            {/* Code Block */}
            <div className="code-block rounded-xl overflow-hidden">
              <div className="flex items-center justify-between px-4 py-3 bg-surface-900/50 border-b border-white/5">
                <div className="flex items-center gap-2">
                  <div className="flex gap-2">
                    <span className="w-3 h-3 rounded-full bg-red-500/80" />
                    <span className="w-3 h-3 rounded-full bg-yellow-500/80" />
                    <span className="w-3 h-3 rounded-full bg-green-500/80" />
                  </div>
                  <span className="ml-4 text-surface-500 text-sm font-mono">yolov11_custom.py</span>
                </div>
                <CopyButton code={`# Your modified architecture

from libreyolo.models.yolov11 import YOLOv11

class YOLOv11Custom(YOLOv11):
    def __init__(self):
        super().__init__()
        # Swap out the backbone
        self.backbone = MyCustomBackbone()

    def forward(self, x):
        features = self.backbone(x)
        return self.head(self.neck(features))

# Train it:
model = YOLOv11Custom()
model.train(data="my_dataset.yaml")`} />
              </div>
              <pre className="p-5 overflow-x-auto">
                <code className="font-mono text-sm">
                  <span className="token-comment"># Your modified architecture</span>{'\n\n'}
                  <span className="token-keyword">from</span> <span className="text-violet-300">libreyolo.models.yolov11</span> <span className="token-keyword">import</span> <span className="text-emerald-400">YOLOv11</span>{'\n\n'}
                  <span className="token-keyword">class</span> <span className="text-emerald-400">YOLOv11Custom</span>(<span className="text-fuchsia-400">YOLOv11</span>):{'\n'}
                  {'    '}<span className="token-keyword">def</span> <span className="token-function">__init__</span>(<span className="text-surface-300">self</span>):{'\n'}
                  {'        '}<span className="token-function">super</span>().<span className="token-function">__init__</span>(){'\n'}
                  {'        '}<span className="token-comment"># Swap out the backbone</span>{'\n'}
                  {'        '}<span className="text-surface-300">self</span>.<span className="text-surface-300">backbone</span> <span className="text-violet-400">=</span> <span className="text-emerald-400">MyCustomBackbone</span>(){'\n\n'}
                  {'    '}<span className="token-keyword">def</span> <span className="token-function">forward</span>(<span className="text-surface-300">self</span>, <span className="text-surface-300">x</span>):{'\n'}
                  {'        '}<span className="text-surface-300">features</span> <span className="text-violet-400">=</span> <span className="text-surface-300">self</span>.<span className="text-surface-300">backbone</span>(<span className="text-surface-300">x</span>){'\n'}
                  {'        '}<span className="token-keyword">return</span> <span className="text-surface-300">self</span>.<span className="token-function">head</span>(<span className="text-surface-300">self</span>.<span className="token-function">neck</span>(<span className="text-surface-300">features</span>)){'\n\n'}
                  <span className="token-comment"># Train it:</span>{'\n'}
                  <span className="text-surface-300">model</span> <span className="text-violet-400">=</span> <span className="text-emerald-400">YOLOv11Custom</span>(){'\n'}
                  <span className="text-surface-300">model</span>.<span className="token-function">train</span>(<span className="text-surface-300">data</span><span className="text-violet-400">=</span><span className="token-string">"my_dataset.yaml"</span>)
                </code>
              </pre>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  )
}

function CTASection() {
  return (
    <section className="py-16">
      <div className="max-w-7xl mx-auto px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="bg-gradient-to-r from-violet-500/10 to-fuchsia-500/10 border border-violet-500/20 rounded-2xl p-8 lg:p-12 text-center"
        >
          <FlaskConical className="w-12 h-12 text-violet-400 mx-auto mb-6" />
          <h2 className="text-2xl lg:text-3xl font-bold text-white mb-4">
            Start Your Research Today
          </h2>
          <p className="text-surface-400 mb-8 max-w-xl mx-auto">
            No license fees. No paywalls. No restrictions on keeping your work private. 
            Just install and start exploring.
          </p>
          
          <div className="code-block rounded-xl max-w-md mx-auto mb-8">
            <pre className="p-4 text-left">
              <code className="font-mono text-sm">
                <span className="text-surface-500">$</span> <span className="text-emerald-400">pip install</span> <span className="text-violet-300">libreyolo</span>
              </code>
            </pre>
          </div>

          <a
            href="https://docs.libreyolo.com"
            target="_blank"
            rel="noopener noreferrer"
            className="btn-primary inline-flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-violet-500 to-fuchsia-500 rounded-xl text-white font-semibold"
          >
            View Documentation
            <ArrowRight className="w-5 h-5" />
          </a>
        </motion.div>
      </div>
    </section>
  )
}

export default function Science() {
  return (
    <div className="min-h-screen">
      <HeroSection />
      <NoPaywallSection />
      <ExplainabilitySection />
      <ModificationSection />
      <CTASection />
    </div>
  )
}

