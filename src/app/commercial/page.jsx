'use client'

import { motion } from 'framer-motion'
import Link from 'next/link'
import {
  Scale, Shield, CheckCircle2, XCircle, AlertTriangle,
  Lock, Unlock, ArrowRight, HelpCircle
} from 'lucide-react'

function HeroSection() {
  return (
    <section className="relative pt-24 lg:pt-32 pb-16">
      <div className="max-w-7xl mx-auto px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="max-w-3xl"
        >
          <div className="flex items-center gap-2 text-emerald-700 dark:text-emerald-400 text-sm font-medium mb-4">
            <Scale className="w-4 h-4" />
            Commercial Integration Guide
          </div>
          <h1 className="text-4xl sm:text-5xl font-bold text-surface-800 dark:text-white mb-6">
            Ship Your Product <span className="text-emerald-600 dark:text-emerald-400">Without</span> Shipping Your Source
          </h1>
          <p className="text-lg text-surface-600 dark:text-surface-400 leading-relaxed">
            The definitive guide to using LibreYOLO in proprietary, closed-source applications.
            No license negotiations. No legal gray areas. Just clear, actionable guidance.
          </p>
        </motion.div>
      </div>
    </section>
  )
}

function TLDRSection() {
  return (
    <section className="py-12">
      <div className="max-w-7xl mx-auto px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="bg-gradient-to-r from-emerald-500/10 to-libre-500/10 border border-emerald-500/30 rounded-2xl p-8"
        >
          <div className="flex items-start gap-4">
            <div className="p-3 rounded-xl bg-emerald-500/20 flex-shrink-0">
              <CheckCircle2 className="w-8 h-8 text-emerald-600 dark:text-emerald-400" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-surface-800 dark:text-white mb-4">TL;DR: Yes, You Can Sell This</h2>
              <div className="space-y-3 text-surface-600 dark:text-surface-300">
                <p>
                  The LibreYOLO engine is released under the <strong className="text-surface-800 dark:text-white">MIT License</strong>.
                  This is one of the most permissive open-source licenses available.
                </p>
                <p>You are explicitly permitted to:</p>
                <ul className="grid sm:grid-cols-2 gap-2 mt-4">
                  {[
                    'Use in proprietary software',
                    'Sell products that include it',
                    'Modify the code freely',
                    'Keep your changes private',
                    'Bundle in closed-source apps',
                    'Deploy on any infrastructure',
                  ].map((item) => (
                    <li key={item} className="flex items-center gap-2">
                      <CheckCircle2 className="w-5 h-5 text-emerald-600 dark:text-emerald-400 flex-shrink-0" />
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  )
}

function WeightsSection() {
  return (
    <section className="py-12">
      <div className="max-w-7xl mx-auto px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
        >
          <h2 className="text-2xl font-bold text-surface-800 dark:text-white mb-4">
            A Note on Model Weights
          </h2>

          <div className="bg-amber-50 dark:bg-yellow-500/5 border border-amber-300 dark:border-yellow-500/20 rounded-2xl p-8 mb-8">
            <div className="flex items-start gap-4">
              <AlertTriangle className="w-6 h-6 text-amber-600 dark:text-yellow-400 flex-shrink-0 mt-1" />
              <div>
                <h3 className="text-lg font-semibold text-surface-800 dark:text-white mb-2">
                  Engine vs. Weights: Different Licenses May Apply
                </h3>
                <p className="text-surface-600 dark:text-surface-400 mb-4">
                  The LibreYOLO <strong className="text-surface-800 dark:text-white">engine</strong> (the code that runs inference and training)
                  is MIT-licensed. However, pre-trained <strong className="text-surface-800 dark:text-white">model weights</strong> may have
                  different licensing terms depending on their origin.
                </p>
                <div className="grid sm:grid-cols-2 gap-4 mt-6">
                  <div className="bg-emerald-50 dark:bg-emerald-500/5 border border-emerald-300 dark:border-emerald-500/20 rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <Shield className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
                      <span className="font-semibold text-surface-800 dark:text-white">MIT Weights</span>
                    </div>
                    <p className="text-surface-600 dark:text-surface-400 text-sm">
                      Weights trained from scratch using LibreYOLO on public datasets.
                      Fully safe for commercial use.
                    </p>
                  </div>
                  <div className="bg-amber-50 dark:bg-yellow-500/5 border border-amber-300 dark:border-yellow-500/20 rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <AlertTriangle className="w-5 h-5 text-amber-600 dark:text-yellow-400" />
                      <span className="font-semibold text-surface-800 dark:text-white">Converted Weights</span>
                    </div>
                    <p className="text-surface-600 dark:text-surface-400 text-sm">
                      Weights converted from other implementations.
                      Consult your legal team for commercial deployments.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-emerald-50 dark:bg-emerald-500/5 border border-emerald-300 dark:border-emerald-500/20 rounded-xl p-6">
            <h4 className="text-surface-800 dark:text-white font-semibold mb-3">The Safest Path for Commercial Use</h4>
            <ol className="space-y-3 text-surface-600 dark:text-surface-300">
              <li className="flex items-start gap-3">
                <span className="flex items-center justify-center w-6 h-6 rounded-full bg-emerald-500/20 text-emerald-700 dark:text-emerald-400 text-sm font-semibold flex-shrink-0">1</span>
                <span>Use MIT-licensed weights from our Model Zoo for initial development</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="flex items-center justify-center w-6 h-6 rounded-full bg-emerald-500/20 text-emerald-700 dark:text-emerald-400 text-sm font-semibold flex-shrink-0">2</span>
                <span>Fine-tune on your own proprietary dataset using LibreYOLO</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="flex items-center justify-center w-6 h-6 rounded-full bg-emerald-500/20 text-emerald-700 dark:text-emerald-400 text-sm font-semibold flex-shrink-0">3</span>
                <span>Your resulting weights are <strong className="text-surface-800 dark:text-white">entirely yours</strong> with no licensing obligations</span>
              </li>
            </ol>
          </div>
        </motion.div>
      </div>
    </section>
  )
}

function ComparisonSection() {
  const comparisons = [
    { feature: 'Use in proprietary software', mit: true, agpl: false },
    { feature: 'Sell products containing it', mit: true, agpl: false },
    { feature: 'Modify without disclosure', mit: true, agpl: false },
    { feature: 'Offer as SaaS', mit: true, agpl: false },
    { feature: 'Bundle in closed-source apps', mit: true, agpl: false },
    { feature: 'Deploy on customer infrastructure', mit: true, agpl: false },
    { feature: 'Fork without sharing changes', mit: true, agpl: false },
    { feature: 'Commercial use fee', mit: false, agpl: true },
  ]

  return (
    <section className="py-12">
      <div className="max-w-7xl mx-auto px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
        >
          <h2 className="text-2xl font-bold text-surface-800 dark:text-white mb-4">
            License Comparison: MIT vs. AGPL
          </h2>
          <p className="text-surface-600 dark:text-surface-400 mb-8">
            Understanding the practical differences for commercial applications.
          </p>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-surface-200 dark:border-white/10">
                  <th className="text-left py-4 px-4 text-surface-500 font-medium">Use Case</th>
                  <th className="py-4 px-4 text-center">
                    <div className="inline-flex items-center gap-2 badge-mit px-3 py-1 rounded-full text-xs font-semibold">
                      <Unlock className="w-3 h-3" />
                      MIT
                    </div>
                  </th>
                  <th className="py-4 px-4 text-center">
                    <div className="inline-flex items-center gap-2 badge-danger px-3 py-1 rounded-full text-xs font-semibold">
                      <Lock className="w-3 h-3" />
                      AGPL
                    </div>
                  </th>
                </tr>
              </thead>
              <tbody>
                {comparisons.map((row, index) => (
                  <tr key={index} className="border-b border-surface-100 dark:border-white/5">
                    <td className="py-4 px-4 text-surface-700 dark:text-surface-300">{row.feature}</td>
                    <td className="py-4 px-4 text-center">
                      {row.mit ? (
                        <CheckCircle2 className="w-5 h-5 text-emerald-600 dark:text-emerald-400 mx-auto" />
                      ) : (
                        <span className="text-surface-400">—</span>
                      )}
                    </td>
                    <td className="py-4 px-4 text-center">
                      {row.agpl ? (
                        <span className="text-amber-600 dark:text-yellow-400 text-xs font-medium">Required</span>
                      ) : (
                        <XCircle className="w-5 h-5 text-red-500 dark:text-red-400 mx-auto" />
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </motion.div>
      </div>
    </section>
  )
}

function FAQSection() {
  const faqs = [
    {
      question: 'Do I need to pay anything to use LibreYOLO commercially?',
      answer: 'No. The MIT license does not require any payment, royalties, or fees for commercial use. You can use, modify, and distribute LibreYOLO in commercial products at no cost.',
    },
    {
      question: 'Do I need to open-source my application if I use LibreYOLO?',
      answer: 'No. Unlike copyleft licenses (like AGPL), the MIT license has no "viral" clause. Your proprietary code remains proprietary. You only need to include the MIT license notice somewhere in your distribution.',
    },
    {
      question: 'Can I use LibreYOLO in a SaaS product?',
      answer: 'Yes. You can build and deploy SaaS products using LibreYOLO without any special licensing requirements. The MIT license explicitly permits this.',
    },
    {
      question: 'What if I modify the LibreYOLO source code?',
      answer: 'You can modify the code freely without any obligation to share your changes. Unlike AGPL, MIT does not require you to publish modifications.',
    },
    {
      question: 'Can my competitors use my improvements if I contribute back?',
      answer: 'If you choose to contribute improvements back to the LibreYOLO project (which is optional), those contributions become available under MIT license. But you\'re never required to contribute.',
    },
    {
      question: 'How do I ensure my commercial use is fully "clean"?',
      answer: 'Use MIT-licensed weights from our Model Zoo, or train your own models from scratch using LibreYOLO. Your trained weights are entirely yours with no encumbrances.',
    },
  ]

  return (
    <section className="py-12">
      <div className="max-w-7xl mx-auto px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
        >
          <div className="flex items-center gap-3 mb-8">
            <div className="p-2 rounded-lg bg-libre-500/10">
              <HelpCircle className="w-5 h-5 text-libre-600 dark:text-libre-400" />
            </div>
            <h2 className="text-2xl font-bold text-surface-800 dark:text-white">Frequently Asked Questions</h2>
          </div>

          <div className="space-y-4">
            {faqs.map((faq, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.05 }}
                className="bg-surface-50 dark:bg-surface-900/50 border border-surface-200 dark:border-white/5 rounded-xl p-6"
              >
                <h3 className="text-surface-800 dark:text-white font-semibold mb-3">{faq.question}</h3>
                <p className="text-surface-600 dark:text-surface-400 text-sm leading-relaxed">{faq.answer}</p>
              </motion.div>
            ))}
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
          className="bg-gradient-to-r from-libre-500/10 to-emerald-500/10 border border-libre-500/20 rounded-2xl p-8 lg:p-12 text-center"
        >
          <h2 className="text-2xl lg:text-3xl font-bold text-surface-800 dark:text-white mb-4">
            Ready to Build?
          </h2>
          <p className="text-surface-600 dark:text-surface-400 mb-8 max-w-xl mx-auto">
            Start integrating LibreYOLO into your commercial product today.
            No licensing calls. No legal reviews. Just code.
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link
              href="/docs"
              className="btn-primary flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-libre-500 to-libre-600 rounded-xl text-white font-semibold"
            >
              Get Started
              <ArrowRight className="w-5 h-5" />
            </Link>
          </div>
        </motion.div>
      </div>
    </section>
  )
}

export default function Commercial() {
  return (
    <div className="min-h-screen">
      <HeroSection />
      <TLDRSection />
      <WeightsSection />
      <ComparisonSection />
      <FAQSection />
      <CTASection />
    </div>
  )
}
