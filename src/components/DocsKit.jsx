'use client'

/*
 * Shared documentation primitives + layout shell.
 *
 * The original /docs page keeps its own private copies of these helpers. This
 * module exists so the satellite docs pages (LibreVLM, Experimental) can reuse
 * the exact same look and feel without bloating that 2.8k line file or coupling
 * to it. Keep the visual language identical to /docs.
 */

import Link from 'next/link'
import { useState, useEffect } from 'react'
import { useLocale, useTranslations } from 'next-intl'
import { motion, AnimatePresence } from 'framer-motion'
import {
  BookOpen, Copy, Check, Menu, X, ChevronRight, ArrowLeft, ExternalLink,
} from 'lucide-react'

/* ─── Minimal syntax highlighter (python / bash / yaml) ─── */

const PYTHON_KEYWORDS = new Set([
  'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue',
  'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from',
  'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or',
  'pass', 'raise', 'return', 'try', 'while', 'with', 'yield',
])
const PYTHON_CONSTANTS = new Set(['False', 'None', 'True'])
const PYTHON_BUILTINS = new Set([
  'all', 'any', 'bool', 'dict', 'enumerate', 'filter', 'float', 'int', 'len',
  'list', 'map', 'max', 'min', 'print', 'range', 'reversed', 'round', 'set',
  'sorted', 'str', 'sum', 'tuple', 'zip',
])
const BASH_KEYWORDS = new Set([
  'case', 'do', 'done', 'elif', 'else', 'esac', 'fi', 'for', 'function', 'if',
  'in', 'select', 'then', 'until', 'while',
])
const YAML_CONSTANTS = new Set(['false', 'no', 'null', 'off', 'on', 'true', 'yes'])

const CODE_LABELS = {
  bash: 'bash', py: 'python', python: 'python', shell: 'shell',
  sh: 'bash', text: 'text', yaml: 'yaml', yml: 'yaml',
}

function pushToken(tokens, value, className = '') {
  if (!value) return
  tokens.push({ value, className })
}

function nextNonWhitespaceChar(text, index) {
  for (let i = index; i < text.length; i += 1) {
    if (!/\s/.test(text[i])) return text[i]
  }
  return ''
}

function prevNonWhitespaceChar(text, index) {
  for (let i = index - 1; i >= 0; i -= 1) {
    if (!/\s/.test(text[i])) return text[i]
  }
  return ''
}

function isUppercaseSymbol(value) {
  return /^[A-Z][A-Za-z0-9_]*$/.test(value) || /^[A-Z0-9_]+$/.test(value)
}

function matchPythonString(segment) {
  return segment.match(/^(?:r|u|b|f|rb|br|fr|rf)?(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')/i)
}

function classifyPythonIdentifier(value, line, start, end, state) {
  if (PYTHON_KEYWORDS.has(value)) return 'token-keyword'
  if (PYTHON_CONSTANTS.has(value)) return 'token-symbol'
  if (state.expectDefinitionName === 'function') return 'token-function'
  if (state.expectDefinitionName === 'class') return 'token-symbol'
  if (state.importMode === 'module') return 'token-module'
  if (state.importMode === 'imported') {
    return isUppercaseSymbol(value) ? 'token-symbol' : 'token-module'
  }

  const nextChar = nextNonWhitespaceChar(line, end)
  const prevChar = prevNonWhitespaceChar(line, start)

  if (value === 'self' || value === 'cls') return 'token-variable'
  if (PYTHON_BUILTINS.has(value) && nextChar === '(') return 'token-function'
  if (isUppercaseSymbol(value)) return 'token-symbol'
  if (prevChar === '.') return nextChar === '(' ? 'token-function' : 'token-property'
  if (nextChar === '(') return 'token-function'
  return ''
}

function updatePythonState(state, value) {
  if (value === 'from') { state.importMode = 'module'; state.expectDefinitionName = null; return }
  if (value === 'import') { state.importMode = 'imported'; return }
  if (value === 'def') { state.expectDefinitionName = 'function'; state.importMode = null; return }
  if (value === 'class') { state.expectDefinitionName = 'class'; state.importMode = null; return }
  if (state.expectDefinitionName) { state.expectDefinitionName = null }
}

function tokenizePythonLine(line) {
  const tokens = []
  const state = { expectDefinitionName: null, importMode: null }
  let index = 0

  while (index < line.length) {
    const segment = line.slice(index)

    const whitespaceMatch = segment.match(/^\s+/)
    if (whitespaceMatch) { pushToken(tokens, whitespaceMatch[0]); index += whitespaceMatch[0].length; continue }

    if (segment.startsWith('#')) { pushToken(tokens, segment, 'token-comment'); break }

    const stringMatch = matchPythonString(segment)
    if (stringMatch) { pushToken(tokens, stringMatch[0], 'token-string'); index += stringMatch[0].length; continue }

    const decoratorMatch = segment.match(/^@[A-Za-z_]\w*/)
    if (decoratorMatch) { pushToken(tokens, decoratorMatch[0], 'token-symbol'); index += decoratorMatch[0].length; continue }

    const numberMatch = segment.match(/^(?:\d+(?:\.\d+)?|\.\d+)/)
    if (numberMatch) { pushToken(tokens, numberMatch[0], 'token-number'); index += numberMatch[0].length; continue }

    const operatorMatch = segment.match(/^(?:==|!=|<=|>=|:=|\*\*|\/\/|->|[-+*/%=&|^~<>]+)/)
    if (operatorMatch) { pushToken(tokens, operatorMatch[0], 'token-operator'); index += operatorMatch[0].length; continue }

    const identifierMatch = segment.match(/^[A-Za-z_]\w*/)
    if (identifierMatch) {
      const value = identifierMatch[0]
      pushToken(tokens, value, classifyPythonIdentifier(value, line, index, index + value.length, state))
      index += value.length
      updatePythonState(state, value)
      continue
    }

    pushToken(tokens, segment[0])
    index += 1
  }

  return tokens
}

function tokenizeBashLine(line) {
  const tokens = []
  let index = 0
  let expectCommand = true

  while (index < line.length) {
    const segment = line.slice(index)

    const whitespaceMatch = segment.match(/^\s+/)
    if (whitespaceMatch) { pushToken(tokens, whitespaceMatch[0]); index += whitespaceMatch[0].length; continue }

    if (segment.startsWith('#')) { pushToken(tokens, segment, 'token-comment'); break }

    const stringMatch = segment.match(/^(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')/)
    if (stringMatch) { pushToken(tokens, stringMatch[0], 'token-string'); index += stringMatch[0].length; expectCommand = false; continue }

    const variableMatch = segment.match(/^(?:\$\{[^}]+\}|\$[A-Za-z_]\w*)/)
    if (variableMatch) { pushToken(tokens, variableMatch[0], 'token-variable'); index += variableMatch[0].length; expectCommand = false; continue }

    const chainMatch = segment.match(/^(?:&&|\|\||[|;><=]+)/)
    if (chainMatch) { pushToken(tokens, chainMatch[0], 'token-operator'); index += chainMatch[0].length; expectCommand = true; continue }

    const flagMatch = segment.match(/^--?[A-Za-z0-9][\w-]*/)
    if (flagMatch) { pushToken(tokens, flagMatch[0], 'token-flag'); index += flagMatch[0].length; expectCommand = false; continue }

    const numberMatch = segment.match(/^(?:\d+(?:\.\d+)?|\.\d+)/)
    if (numberMatch) { pushToken(tokens, numberMatch[0], 'token-number'); index += numberMatch[0].length; expectCommand = false; continue }

    const wordMatch = segment.match(/^[A-Za-z_./:][\w./:-]*/)
    if (wordMatch) {
      const value = wordMatch[0]
      if (BASH_KEYWORDS.has(value)) pushToken(tokens, value, 'token-keyword')
      else if (expectCommand) pushToken(tokens, value, 'token-function')
      else pushToken(tokens, value)
      index += value.length
      expectCommand = false
      continue
    }

    pushToken(tokens, segment[0])
    index += 1
  }

  return tokens
}

function tokenizeYamlLine(line) {
  const tokens = []
  let index = 0
  let sawValueSeparator = false

  while (index < line.length) {
    const segment = line.slice(index)

    const whitespaceMatch = segment.match(/^\s+/)
    if (whitespaceMatch) { pushToken(tokens, whitespaceMatch[0]); index += whitespaceMatch[0].length; continue }

    if (segment.startsWith('#')) { pushToken(tokens, segment, 'token-comment'); break }

    if (segment.startsWith('- ')) { pushToken(tokens, '-', 'token-operator'); pushToken(tokens, ' '); index += 2; continue }

    const stringMatch = segment.match(/^(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')/)
    if (stringMatch) { pushToken(tokens, stringMatch[0], 'token-string'); index += stringMatch[0].length; continue }

    const numberMatch = segment.match(/^(?:\d+(?:\.\d+)?|\.\d+)/)
    if (numberMatch) { pushToken(tokens, numberMatch[0], 'token-number'); index += numberMatch[0].length; continue }

    const keywordMatch = segment.match(/^[A-Za-z0-9_.\/-]+/)
    if (keywordMatch) {
      const value = keywordMatch[0]
      const nextChar = line[index + value.length]
      if (!sawValueSeparator && nextChar === ':') pushToken(tokens, value, 'token-property')
      else if (YAML_CONSTANTS.has(value.toLowerCase())) pushToken(tokens, value, 'token-constant')
      else pushToken(tokens, value)
      index += value.length
      continue
    }

    if (segment[0] === ':') { pushToken(tokens, ':', 'token-operator'); sawValueSeparator = true; index += 1; continue }

    pushToken(tokens, segment[0])
    index += 1
  }

  return tokens
}

function highlightLine(line, language) {
  switch (language.toLowerCase()) {
    case 'py':
    case 'python':
      return tokenizePythonLine(line)
    case 'bash':
    case 'sh':
    case 'shell':
      return tokenizeBashLine(line)
    case 'yaml':
    case 'yml':
      return tokenizeYamlLine(line)
    default:
      return [{ value: line, className: '' }]
  }
}

function getCodeLabel(language, filename) {
  if (filename) return filename
  const normalizedLanguage = language.toLowerCase()
  return CODE_LABELS[normalizedLanguage] || normalizedLanguage
}

/* ─── Primitives ─── */

export function CodeBlock({ children, language = 'python', filename }) {
  const [copied, setCopied] = useState(false)
  const code = typeof children === 'string' ? children : String(children ?? '')
  const lines = code.split('\n')
  const label = getCodeLabel(language, filename)

  const handleCopy = () => {
    navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="relative my-5 code-block rounded-xl overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 bg-surface-100 dark:bg-surface-900/50 border-b border-surface-200 dark:border-white/5">
        <div className="flex items-center gap-2">
          <div className="flex gap-2">
            <span className="w-3 h-3 rounded-full bg-red-500/80" />
            <span className="w-3 h-3 rounded-full bg-yellow-500/80" />
            <span className="w-3 h-3 rounded-full bg-green-500/80" />
          </div>
          <span className="ml-4 text-surface-500 text-sm font-mono">{label}</span>
        </div>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium text-surface-500 dark:text-surface-400 hover:text-surface-900 dark:hover:text-white hover:bg-surface-200 dark:hover:bg-white/10 transition-all"
        >
          {copied ? <Check className="w-3.5 h-3.5 text-emerald-500 dark:text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
          {copied ? 'Copied!' : 'Copy'}
        </button>
      </div>
      <pre className="text-left overflow-x-auto">
        <code className="font-mono text-sm lg:text-base text-surface-700 dark:text-surface-300">
          <table className="border-collapse">
            <tbody>
              {lines.map((line, lineIndex) => (
                <tr key={`${label}-${lineIndex}`}>
                  <td className="pr-4 text-right select-none text-surface-500 dark:text-surface-600 align-top w-6">
                    {lineIndex + 1}
                  </td>
                  <td className="whitespace-pre">
                    {highlightLine(line, language).map((token, tokenIndex) => (
                      <span key={`${lineIndex}-${tokenIndex}`} className={token.className}>
                        {token.value}
                      </span>
                    ))}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </code>
      </pre>
    </div>
  )
}

export function DocTable({ headers, rows }) {
  return (
    <div className="my-5 overflow-x-auto rounded-xl border border-surface-200 dark:border-white/[0.06]">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-surface-200 dark:border-white/[0.08] bg-surface-50 dark:bg-white/[0.02]">
            {headers.map((h, i) => (
              <th key={i} className="px-4 py-3 text-left font-semibold text-surface-700 dark:text-surface-300 whitespace-nowrap">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-b border-surface-100 dark:border-white/[0.04] last:border-0 hover:bg-surface-50 dark:hover:bg-white/[0.02] transition-colors">
              {row.map((cell, j) => (
                <td key={j} className="px-4 py-3 text-surface-600 dark:text-surface-400 align-top">{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function SectionHeading({ id, icon: Icon, children }) {
  return (
    <div id={id} className="scroll-mt-28 flex items-center gap-3 mb-6 pt-2">
      <div className="w-10 h-10 rounded-xl bg-libre-500/10 border border-libre-500/20 flex items-center justify-center shrink-0">
        <Icon className="w-5 h-5 text-libre-600 dark:text-libre-400" />
      </div>
      <h2 className="text-2xl lg:text-3xl font-bold text-surface-800 dark:text-white">{children}</h2>
    </div>
  )
}

export function SubHeading({ children }) {
  return <h3 className="text-lg font-semibold text-surface-800 dark:text-white mt-10 mb-4">{children}</h3>
}

export function P({ children }) {
  return <p className="text-surface-600 dark:text-surface-400 leading-relaxed mb-4">{children}</p>
}

export function InlineCode({ children }) {
  return <code className="px-1.5 py-0.5 rounded bg-libre-500/10 dark:bg-white/[0.06] text-libre-600 dark:text-libre-300 text-sm font-mono">{children}</code>
}

export function Divider() {
  return <div className="border-t border-surface-200 dark:border-white/[0.06] my-16" />
}

export function FeatureItem({ children }) {
  return (
    <li className="flex items-start gap-3 text-surface-600 dark:text-surface-400">
      <ChevronRight className="w-4 h-4 text-libre-600 dark:text-libre-400 mt-1 shrink-0" />
      <span>{children}</span>
    </li>
  )
}

export function SupportBadge({ variant = 'experimental', children }) {
  const styles = {
    validated: 'border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300',
    experimental: 'border-amber-500/30 bg-amber-500/10 text-amber-700 dark:text-amber-300',
    wip: 'border-rose-500/30 bg-rose-500/10 text-rose-700 dark:text-rose-300',
  }
  return (
    <span className={`inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-semibold ${styles[variant] || styles.experimental}`}>
      {children}
    </span>
  )
}

const CALLOUT_STYLES = {
  libre: {
    wrap: 'border-libre-500/30 bg-libre-500/5 dark:bg-libre-500/10',
    icon: 'text-libre-600 dark:text-libre-400',
  },
  emerald: {
    wrap: 'border-emerald-500/30 bg-emerald-500/5 dark:bg-emerald-500/10',
    icon: 'text-emerald-600 dark:text-emerald-400',
  },
  amber: {
    wrap: 'border-amber-500/30 bg-amber-500/5 dark:bg-amber-500/10',
    icon: 'text-amber-600 dark:text-amber-400',
  },
  rose: {
    wrap: 'border-rose-500/30 bg-rose-500/5 dark:bg-rose-500/10',
    icon: 'text-rose-600 dark:text-rose-400',
  },
}

export function Callout({ icon: Icon, tone = 'libre', title, children, className = '' }) {
  const style = CALLOUT_STYLES[tone] || CALLOUT_STYLES.libre
  return (
    <div className={`my-6 rounded-xl border p-4 ${style.wrap} ${className}`}>
      <div className="flex items-start gap-3">
        {Icon && <Icon className={`w-5 h-5 mt-0.5 shrink-0 ${style.icon}`} />}
        <div>
          {title && <p className="font-semibold text-surface-900 dark:text-white mb-1">{title}</p>}
          <div className="text-sm text-surface-600 dark:text-surface-400 space-y-2">{children}</div>
        </div>
      </div>
    </div>
  )
}

/* ─── Layout shell (sidebar + scroll spy + mobile menu + copy docs) ─── */

function Sidebar({ sections, eyebrow, activeSection, onNavigate, relatedLinks }) {
  return (
    <nav>
      <div className="flex items-center gap-2 mb-6 px-3">
        <BookOpen className="w-5 h-5 text-libre-600 dark:text-libre-400" />
        <span className="text-sm font-semibold text-surface-800 dark:text-white tracking-wide uppercase">{eyebrow}</span>
      </div>

      <Link
        href="/docs"
        className="flex items-center gap-2 mb-6 mx-3 rounded-lg border border-surface-200 dark:border-white/[0.08] bg-surface-50 dark:bg-white/[0.03] px-3 py-2 text-sm font-medium text-surface-600 dark:text-surface-400 hover:text-surface-900 dark:hover:text-white hover:bg-white dark:hover:bg-white/[0.05] transition-colors"
      >
        <ArrowLeft className="w-4 h-4" />
        Back to core docs
      </Link>

      <ul className="space-y-0.5">
        {sections.map(({ id, title, icon: Icon }) => {
          const isActive = activeSection === id
          return (
            <li key={id}>
              <button
                onClick={() => onNavigate(id)}
                className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200 text-left ${
                  isActive
                    ? 'text-libre-600 dark:text-libre-400 bg-libre-500/10'
                    : 'text-surface-600 dark:text-surface-400 hover:text-surface-900 dark:hover:text-surface-200 hover:bg-surface-100 dark:hover:bg-white/[0.04]'
                }`}
              >
                <Icon className={`w-4 h-4 shrink-0 ${isActive ? 'text-libre-600 dark:text-libre-400' : 'text-surface-400 dark:text-surface-600'}`} />
                {title}
              </button>
            </li>
          )
        })}
      </ul>

      {relatedLinks?.length ? (
        <div className="mt-8 mx-3 pt-6 border-t border-surface-200 dark:border-white/[0.06]">
          <div className="text-[11px] font-semibold uppercase tracking-wide text-surface-500 dark:text-surface-500 mb-2">
            Related
          </div>
          <div className="space-y-1">
            {relatedLinks.map(({ href, label }) => (
              <Link
                key={href}
                href={href}
                className="block rounded-md px-2.5 py-2 text-sm font-medium text-surface-600 dark:text-surface-400 hover:bg-surface-100 dark:hover:bg-white/[0.05] hover:text-surface-900 dark:hover:text-white transition-colors"
              >
                {label}
              </Link>
            ))}
          </div>
        </div>
      ) : null}
    </nav>
  )
}

export function DocLayout({ sections, eyebrow = 'Documentation', copyTitle = 'LibreYOLO Documentation', relatedLinks, children }) {
  const locale = useLocale()
  const tNote = useTranslations('DocsNote')
  const [activeSection, setActiveSection] = useState(sections[0]?.id)
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [docsCopied, setDocsCopied] = useState(false)

  useEffect(() => {
    const handleScroll = () => {
      const threshold = window.innerHeight * 0.3
      let current = sections[0]?.id

      for (const { id } of sections) {
        const el = document.getElementById(id)
        if (el && el.getBoundingClientRect().top <= threshold) {
          current = id
        }
      }

      setActiveSection(current)
    }

    window.addEventListener('scroll', handleScroll, { passive: true })
    handleScroll()
    return () => window.removeEventListener('scroll', handleScroll)
  }, [sections])

  const navigateTo = (id) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })
    setMobileMenuOpen(false)
  }

  const copyDocs = async () => {
    const docsText = document.querySelector('[data-docs-content]')?.innerText || ''
    await navigator.clipboard.writeText(`# ${copyTitle}\n\n${docsText}`)
    setDocsCopied(true)
    setTimeout(() => setDocsCopied(false), 2000)
  }

  return (
    <div className="flex min-h-screen">
      {/* Desktop sidebar */}
      <aside className="hidden lg:block fixed left-0 top-20 bottom-0 w-64 border-r border-surface-200 dark:border-white/[0.06] bg-white/80 dark:bg-surface-950/50 backdrop-blur-sm overflow-y-auto py-8 px-4 z-30">
        <Sidebar sections={sections} eyebrow={eyebrow} activeSection={activeSection} onNavigate={navigateTo} relatedLinks={relatedLinks} />
      </aside>

      {/* Mobile sidebar toggle */}
      <button
        onClick={() => setMobileMenuOpen(true)}
        className="lg:hidden fixed bottom-6 right-6 z-40 w-12 h-12 rounded-full bg-libre-500 text-white shadow-lg shadow-libre-500/30 flex items-center justify-center hover:bg-libre-400 transition-colors"
        aria-label="Open navigation"
      >
        <Menu className="w-5 h-5" />
      </button>

      {/* Mobile sidebar overlay */}
      <AnimatePresence>
        {mobileMenuOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 lg:hidden"
              onClick={() => setMobileMenuOpen(false)}
            />
            <motion.aside
              initial={{ x: -280 }}
              animate={{ x: 0 }}
              exit={{ x: -280 }}
              transition={{ type: 'spring', damping: 25, stiffness: 250 }}
              className="fixed left-0 top-0 bottom-0 w-72 bg-white dark:bg-surface-950 border-r border-surface-200 dark:border-white/[0.06] z-50 lg:hidden overflow-y-auto py-6 px-4"
            >
              <div className="flex items-center justify-between mb-4 px-3">
                <span className="text-sm font-semibold text-surface-800 dark:text-white tracking-wide uppercase">Docs</span>
                <button
                  onClick={() => setMobileMenuOpen(false)}
                  className="p-1.5 rounded-lg text-surface-500 dark:text-surface-400 hover:text-surface-900 dark:hover:text-white hover:bg-surface-100 dark:hover:bg-white/[0.06] transition-colors"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              <Sidebar sections={sections} eyebrow={eyebrow} activeSection={activeSection} onNavigate={navigateTo} relatedLinks={relatedLinks} />
            </motion.aside>
          </>
        )}
      </AnimatePresence>

      {/* Main content */}
      <main className="flex-1 lg:ml-64 min-h-screen pt-28 lg:pt-32 pb-24 px-6 lg:px-12">
        <div className="max-w-4xl mx-auto" data-docs-content>
          {locale === 'zh' && (
            <div className="mb-6 rounded-lg border border-amber-500/30 bg-amber-500/5 px-4 py-3 text-sm text-surface-600 dark:text-surface-300">
              {tNote('text')}
            </div>
          )}
          <div className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <Link
              href="/docs"
              className="inline-flex items-center gap-1.5 text-sm font-medium text-surface-500 dark:text-surface-400 hover:text-libre-600 dark:hover:text-libre-400 transition-colors"
            >
              <ArrowLeft className="w-4 h-4" />
              Core documentation
            </Link>
            <button
              onClick={copyDocs}
              className="inline-flex items-center justify-center gap-2 rounded-lg border border-surface-200 dark:border-white/[0.08] bg-surface-950 px-3.5 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-surface-800 dark:bg-white dark:text-surface-950 dark:hover:bg-surface-200"
            >
              {docsCopied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
              {docsCopied ? 'Copied docs' : 'Copy docs'}
            </button>
          </div>

          {children}
        </div>
      </main>
    </div>
  )
}

/* Small hero used at the top of satellite pages. */
export function DocHero({ badge, badgeTone = 'experimental', title, accent, lead }) {
  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="mb-10">
      {badge && (
        <div className="mb-4">
          <SupportBadge variant={badgeTone}>{badge}</SupportBadge>
        </div>
      )}
      <h1 className="text-4xl lg:text-5xl font-bold tracking-tight text-surface-900 dark:text-white">
        {title}
        {accent && <span className="text-libre-500 dark:text-libre-400">{accent}</span>}
      </h1>
      {lead && (
        <p className="mt-4 text-lg text-surface-600 dark:text-surface-400 leading-relaxed max-w-3xl">
          {lead}
        </p>
      )}
    </motion.div>
  )
}

export function ExternalRef({ href, children }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-center gap-1 text-libre-600 dark:text-libre-400 hover:underline"
    >
      {children}
      <ExternalLink className="w-3 h-3" />
    </a>
  )
}
