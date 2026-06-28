'use client'
import { useLocale } from 'next-intl'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  BookOpen, Terminal, Rocket, Layers, Crosshair, Grid3x3,
  GraduationCap, CheckCircle2, Upload, Cpu, FileCode, Wrench,
  Database, Copy, Check, Menu, X, ChevronRight
} from 'lucide-react'

/* ─── Section metadata for sidebar ─── */
const sections = [
  { id: 'introduction', title: 'Introduction', icon: BookOpen },
  { id: 'installation', title: 'Installation', icon: Terminal },
  { id: 'quickstart', title: 'Quickstart', icon: Rocket },
  { id: 'models', title: 'Available Models', icon: Layers },
  { id: 'prediction', title: 'Prediction', icon: Crosshair },
  { id: 'tiled-inference', title: 'Tiled Inference', icon: Grid3x3 },
  { id: 'training', title: 'Training', icon: GraduationCap },
  { id: 'validation', title: 'Validation', icon: CheckCircle2 },
  { id: 'export', title: 'Export', icon: Upload },
  { id: 'onnx-inference', title: 'ONNX Inference', icon: Cpu },
  { id: 'tensorrt-inference', title: 'TensorRT Inference', icon: Cpu },
  { id: 'openvino-inference', title: 'OpenVINO Inference', icon: Cpu },
  { id: 'ncnn-inference', title: 'NCNN Inference', icon: Cpu },
  { id: 'api-reference', title: 'API Reference', icon: FileCode },
  { id: 'architecture', title: 'Architecture Guide', icon: Wrench },
  { id: 'dataset-format', title: 'Dataset Format', icon: Database },
]

const docsVersions = [
  { version: 'v1.3.0', label: 'Pre-release', href: '/docs/v1.3.0' },
  { version: 'v1.2.0', label: 'Latest', href: '/docs' },
  { version: 'v1.1.0', label: 'Archived', href: '/docs/v1.1.0' },
]

export { DocsPage }

/* ─── Reusable components ─── */

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
  bash: 'bash',
  py: 'python',
  python: 'python',
  shell: 'shell',
  sh: 'bash',
  text: 'text',
  yaml: 'yaml',
  yml: 'yaml',
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
  if (value === 'from') {
    state.importMode = 'module'
    state.expectDefinitionName = null
    return
  }

  if (value === 'import') {
    state.importMode = 'imported'
    return
  }

  if (value === 'def') {
    state.expectDefinitionName = 'function'
    state.importMode = null
    return
  }

  if (value === 'class') {
    state.expectDefinitionName = 'class'
    state.importMode = null
    return
  }

  if (state.expectDefinitionName) {
    state.expectDefinitionName = null
  }
}

function tokenizePythonLine(line) {
  const tokens = []
  const state = {
    expectDefinitionName: null,
    importMode: null,
  }

  let index = 0

  while (index < line.length) {
    const segment = line.slice(index)

    const whitespaceMatch = segment.match(/^\s+/)
    if (whitespaceMatch) {
      pushToken(tokens, whitespaceMatch[0])
      index += whitespaceMatch[0].length
      continue
    }

    if (segment.startsWith('#')) {
      pushToken(tokens, segment, 'token-comment')
      break
    }

    const stringMatch = matchPythonString(segment)
    if (stringMatch) {
      pushToken(tokens, stringMatch[0], 'token-string')
      index += stringMatch[0].length
      continue
    }

    const decoratorMatch = segment.match(/^@[A-Za-z_]\w*/)
    if (decoratorMatch) {
      pushToken(tokens, decoratorMatch[0], 'token-symbol')
      index += decoratorMatch[0].length
      continue
    }

    const numberMatch = segment.match(/^(?:\d+(?:\.\d+)?|\.\d+)/)
    if (numberMatch) {
      pushToken(tokens, numberMatch[0], 'token-number')
      index += numberMatch[0].length
      continue
    }

    const operatorMatch = segment.match(/^(?:==|!=|<=|>=|:=|\*\*|\/\/|->|[-+*/%=&|^~<>]+)/)
    if (operatorMatch) {
      pushToken(tokens, operatorMatch[0], 'token-operator')
      index += operatorMatch[0].length
      continue
    }

    const identifierMatch = segment.match(/^[A-Za-z_]\w*/)
    if (identifierMatch) {
      const value = identifierMatch[0]
      pushToken(
        tokens,
        value,
        classifyPythonIdentifier(value, line, index, index + value.length, state)
      )
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
    if (whitespaceMatch) {
      pushToken(tokens, whitespaceMatch[0])
      index += whitespaceMatch[0].length
      continue
    }

    if (segment.startsWith('#')) {
      pushToken(tokens, segment, 'token-comment')
      break
    }

    const stringMatch = segment.match(/^(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')/)
    if (stringMatch) {
      pushToken(tokens, stringMatch[0], 'token-string')
      index += stringMatch[0].length
      expectCommand = false
      continue
    }

    const variableMatch = segment.match(/^(?:\$\{[^}]+\}|\$[A-Za-z_]\w*)/)
    if (variableMatch) {
      pushToken(tokens, variableMatch[0], 'token-variable')
      index += variableMatch[0].length
      expectCommand = false
      continue
    }

    const chainMatch = segment.match(/^(?:&&|\|\||[|;><=]+)/)
    if (chainMatch) {
      pushToken(tokens, chainMatch[0], 'token-operator')
      index += chainMatch[0].length
      expectCommand = true
      continue
    }

    const flagMatch = segment.match(/^--?[A-Za-z0-9][\w-]*/)
    if (flagMatch) {
      pushToken(tokens, flagMatch[0], 'token-flag')
      index += flagMatch[0].length
      expectCommand = false
      continue
    }

    const numberMatch = segment.match(/^(?:\d+(?:\.\d+)?|\.\d+)/)
    if (numberMatch) {
      pushToken(tokens, numberMatch[0], 'token-number')
      index += numberMatch[0].length
      expectCommand = false
      continue
    }

    const wordMatch = segment.match(/^[A-Za-z_./:][\w./:-]*/)
    if (wordMatch) {
      const value = wordMatch[0]

      if (BASH_KEYWORDS.has(value)) {
        pushToken(tokens, value, 'token-keyword')
      } else if (expectCommand) {
        pushToken(tokens, value, 'token-function')
      } else {
        pushToken(tokens, value)
      }

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
    if (whitespaceMatch) {
      pushToken(tokens, whitespaceMatch[0])
      index += whitespaceMatch[0].length
      continue
    }

    if (segment.startsWith('#')) {
      pushToken(tokens, segment, 'token-comment')
      break
    }

    if (segment.startsWith('- ')) {
      pushToken(tokens, '-', 'token-operator')
      pushToken(tokens, ' ')
      index += 2
      continue
    }

    const stringMatch = segment.match(/^(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')/)
    if (stringMatch) {
      pushToken(tokens, stringMatch[0], 'token-string')
      index += stringMatch[0].length
      continue
    }

    const numberMatch = segment.match(/^(?:\d+(?:\.\d+)?|\.\d+)/)
    if (numberMatch) {
      pushToken(tokens, numberMatch[0], 'token-number')
      index += numberMatch[0].length
      continue
    }

    const keywordMatch = segment.match(/^[A-Za-z0-9_.\/-]+/)
    if (keywordMatch) {
      const value = keywordMatch[0]
      const nextChar = line[index + value.length]

      if (!sawValueSeparator && nextChar === ':') {
        pushToken(tokens, value, 'token-property')
      } else if (YAML_CONSTANTS.has(value.toLowerCase())) {
        pushToken(tokens, value, 'token-constant')
      } else {
        pushToken(tokens, value)
      }

      index += value.length
      continue
    }

    if (segment[0] === ':') {
      pushToken(tokens, ':', 'token-operator')
      sawValueSeparator = true
      index += 1
      continue
    }

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

function CodeBlock({ children, language = 'python', filename }) {
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

function DocTable({ headers, rows }) {
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
                <td key={j} className="px-4 py-3 text-surface-600 dark:text-surface-400">{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function SectionHeading({ id, icon: Icon, children }) {
  return (
    <div id={id} className="scroll-mt-28 flex items-center gap-3 mb-6 pt-2">
      <div className="w-10 h-10 rounded-xl bg-libre-500/10 border border-libre-500/20 flex items-center justify-center shrink-0">
        <Icon className="w-5 h-5 text-libre-600 dark:text-libre-400" />
      </div>
      <h2 className="text-2xl lg:text-3xl font-bold text-surface-800 dark:text-white">{children}</h2>
    </div>
  )
}

function SubHeading({ children }) {
  return <h3 className="text-lg font-semibold text-surface-800 dark:text-white mt-10 mb-4">{children}</h3>
}

function P({ children }) {
  return <p className="text-surface-600 dark:text-surface-400 leading-relaxed mb-4">{children}</p>
}

function InlineCode({ children }) {
  return <code className="px-1.5 py-0.5 rounded bg-libre-500/10 dark:bg-white/[0.06] text-libre-600 dark:text-libre-300 text-sm font-mono">{children}</code>
}

function Divider() {
  return <div className="border-t border-surface-200 dark:border-white/[0.06] my-16" />
}

function FeatureItem({ children }) {
  return (
    <li className="flex items-start gap-3 text-surface-600 dark:text-surface-400">
      <ChevronRight className="w-4 h-4 text-libre-600 dark:text-libre-400 mt-1 shrink-0" />
      <span>{children}</span>
    </li>
  )
}

/* ─── Sidebar ─── */

function Sidebar({ activeSection, onNavigate, currentVersion = 'v1.2.0', className = '' }) {
  return (
    <nav className={className}>
      <div className="flex items-center gap-2 mb-6 px-3">
        <BookOpen className="w-5 h-5 text-libre-600 dark:text-libre-400" />
        <span className="text-sm font-semibold text-surface-800 dark:text-white tracking-wide uppercase">Documentation</span>
      </div>
      <div className="mb-6 mx-3 rounded-lg border border-surface-200 dark:border-white/[0.08] bg-surface-50 dark:bg-white/[0.03] p-2">
        <div className="text-[11px] font-semibold uppercase tracking-wide text-surface-500 dark:text-surface-500 mb-2">
          Version
        </div>
        <div className="space-y-1">
          {docsVersions.map(({ version, label, href }) => {
            const isCurrent = version === currentVersion
            return (
              <a
                key={version}
                href={href}
                className={`flex items-center justify-between rounded-md px-2.5 py-2 text-sm font-medium transition-colors ${
                  isCurrent
                    ? 'bg-libre-500/10 text-libre-700 dark:text-libre-300'
                    : 'text-surface-600 dark:text-surface-400 hover:bg-white dark:hover:bg-white/[0.05]'
                }`}
              >
                <span>{version}</span>
                <span className="text-[11px] font-semibold uppercase tracking-wide">{label}</span>
              </a>
            )
          })}
        </div>
      </div>
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
    </nav>
  )
}

/* ─── Main docs page ─── */

function DocsPage({ version = 'v1.2.0', isLatest = true }) {
  const [activeSection, setActiveSection] = useState('introduction')
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [docsCopied, setDocsCopied] = useState(false)

  // Scroll spy — pick the last section whose heading has scrolled past 30% of viewport
  useEffect(() => {
    const handleScroll = () => {
      const threshold = window.innerHeight * 0.3
      let current = sections[0].id

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
  }, [])

  const navigateTo = (id) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })
    setMobileMenuOpen(false)
  }

  const copyDocs = async () => {
    const docsText = document.querySelector('[data-docs-content]')?.innerText || ''
    await navigator.clipboard.writeText(`# LibreYOLO Documentation ${version}\n\n${docsText}`)
    setDocsCopied(true)
    setTimeout(() => setDocsCopied(false), 2000)
  }

  return (
    <div className="flex min-h-screen">
      {/* Desktop sidebar */}
      <aside className="hidden lg:block fixed left-0 top-20 bottom-0 w-64 border-r border-surface-200 dark:border-white/[0.06] bg-white/80 dark:bg-surface-950/50 backdrop-blur-sm overflow-y-auto py-8 px-4 z-30">
        <Sidebar activeSection={activeSection} onNavigate={navigateTo} currentVersion={version} />
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
              <Sidebar activeSection={activeSection} onNavigate={navigateTo} currentVersion={version} />
            </motion.aside>
          </>
        )}
      </AnimatePresence>

      {/* Main content */}
      <main className="flex-1 lg:ml-64 min-h-screen pt-28 lg:pt-32 pb-24 px-6 lg:px-12">
        <div className="max-w-4xl mx-auto" data-docs-content>
          <div className="mb-8 rounded-lg border border-surface-200 dark:border-white/[0.08] bg-white/80 dark:bg-white/[0.03] p-4 shadow-sm">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <p className="text-sm text-surface-600 dark:text-surface-400">
                    {isLatest
                    ? 'These docs track the upcoming v1.2.0 dev branch. For current released docs, use v1.1.0.'
                    : 'This archived version is kept linkable so older installs, search results, and agents can target the right documentation.'}
                </p>
              </div>
              <button
                onClick={copyDocs}
                className="inline-flex items-center justify-center gap-2 rounded-lg border border-surface-200 dark:border-white/[0.08] bg-surface-950 px-3.5 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-surface-800 dark:bg-white dark:text-surface-950 dark:hover:bg-surface-200"
              >
                {docsCopied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                {docsCopied ? 'Copied docs' : 'Copy docs'}
              </button>
            </div>
          </div>

          {/* ────────────── INTRODUCTION ────────────── */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
            <SectionHeading id="introduction" icon={BookOpen}>Introduction</SectionHeading>
            <P>
              LibreYOLO is an MIT-licensed object detection library focused on two flagship model families: <strong className="text-surface-800 dark:text-white">YOLO9</strong> as the CNN flagship and <strong className="text-surface-800 dark:text-white">RF-DETR</strong> as the transformer flagship. The API stays consistent across prediction, training, validation, and export.
            </P>
            <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")
results = model("image.jpg", conf=0.25, save=True)
print(results.boxes.xyxy)`}</CodeBlock>

            <SubHeading>Key features</SubHeading>
            <ul className="space-y-2.5 mb-4">
              <FeatureItem>Flagship support for YOLO9 and RF-DETR</FeatureItem>
              <FeatureItem>Auto-detection of model architecture, size, and class count from weights</FeatureItem>
              <FeatureItem>Tiled inference for large/high-resolution images</FeatureItem>
              <FeatureItem>ONNX, TorchScript, TensorRT, OpenVINO, and NCNN export with embedded metadata</FeatureItem>
              <FeatureItem>ONNX Runtime, TensorRT, OpenVINO, and NCNN inference backends</FeatureItem>
              <FeatureItem>COCO-compatible validation with mAP metrics</FeatureItem>
              <FeatureItem>Accepts any image format: file paths, URLs, PIL, NumPy, PyTorch tensors, raw bytes</FeatureItem>
            </ul>
          </motion.div>

          <Divider />

          {/* ────────────── INSTALLATION ────────────── */}
          <SectionHeading id="installation" icon={Terminal}>Installation</SectionHeading>
          <SubHeading>Requirements</SubHeading>
          <ul className="space-y-1.5 mb-4">
            <li className="flex items-center gap-2 text-surface-600 dark:text-surface-400">
              <span className="w-1.5 h-1.5 rounded-full bg-libre-400" />Python 3.10+
            </li>
            <li className="flex items-center gap-2 text-surface-600 dark:text-surface-400">
              <span className="w-1.5 h-1.5 rounded-full bg-libre-400" />PyTorch 1.7+
            </li>
          </ul>

          <SubHeading>From PyPI</SubHeading>
          <CodeBlock language="bash">{`pip install libreyolo`}</CodeBlock>

          <SubHeading>From source</SubHeading>
          <CodeBlock language="bash">{`git clone https://github.com/Libre-YOLO/libreyolo.git
cd libreyolo
pip install -e .`}</CodeBlock>

          <SubHeading>Optional dependencies</SubHeading>
          <CodeBlock language="bash">{`# ONNX export and inference
pip install libreyolo[onnx]
# or: pip install onnx onnxsim onnxscript onnxruntime

# RT-DETR support
pip install libreyolo[rtdetr]
# or: pip install transformers timm

# RF-DETR support
pip install libreyolo[rfdetr]
# or: pip install rfdetr transformers timm supervision

# TensorRT export and inference (NVIDIA GPU)
pip install libreyolo[tensorrt]
# Note: TensorRT itself requires manual installation (depends on CUDA version)

# OpenVINO export and inference (Intel CPU/GPU/VPU)
pip install libreyolo[openvino]
# INT8 export also needs: pip install nncf

# NCNN export and inference
pip install libreyolo[ncnn]
# or: pip install pnnx ncnn`}</CodeBlock>

          <P>If using <InlineCode>uv</InlineCode>, the most reliable path is an isolated venv per extra:</P>
          <CodeBlock language="bash">{`# ONNX environment
uv venv .venv-onnx
uv pip install --python .venv-onnx/bin/python -e '.[onnx]'

# RT-DETR environment
uv venv .venv-rtdetr
uv pip install --python .venv-rtdetr/bin/python -e '.[rtdetr]'

# Repeat with .[rfdetr], .[openvino], .[ncnn], or .[tensorrt] as needed`}</CodeBlock>
          <P>
            This avoids mutating the project environment and keeps optional dependencies isolated. Vendor-specific extras such as TensorRT, OpenVINO, and NCNN may still require platform-specific native packages.
          </P>

          <Divider />

          {/* ────────────── QUICKSTART ────────────── */}
          <SectionHeading id="quickstart" icon={Rocket}>Quickstart</SectionHeading>

          <SubHeading>Load a model and run inference</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

# Auto-detects architecture and size from the weights file
model = LibreYOLO("LibreYOLO9c.pt")

# Run on a single image
result = model("photo.jpg")

print(f"Found {len(result)} objects")
print(result.boxes.xyxy)   # bounding boxes (N, 4)
print(result.boxes.conf)   # confidence scores (N,)
print(result.boxes.cls)    # class IDs (N,)`}</CodeBlock>

          <SubHeading>Save annotated output</SubHeading>
          <CodeBlock language="python">{`result = model("photo.jpg", save=True)
# Saved under runs/detect/predict*/photo.jpg by default`}</CodeBlock>

          <SubHeading>Process a directory</SubHeading>
          <CodeBlock language="python">{`results = model("images/", save=True, batch=4)
for r in results:
    print(f"{r.path}: {len(r)} detections")`}</CodeBlock>

          <Divider />

          {/* ────────────── AVAILABLE MODELS ────────────── */}
          <SectionHeading id="models" icon={Layers}>Available Models</SectionHeading>

          <P>
            v1.2.0 documentation is centered on two flagship model families. YOLO9 is the CNN flagship and RF-DETR is the transformer flagship. Other supported families are listed as compact references.
          </P>

          <SubHeading>YOLO9 flagship</SubHeading>
          <DocTable
            headers={['Size', 'Code', 'Input size', 'Use case']}
            rows={[
              ['Tiny', <InlineCode key="t">&quot;t&quot;</InlineCode>, '640', 'Fast inference'],
              ['Small', <InlineCode key="s">&quot;s&quot;</InlineCode>, '640', 'Balanced'],
              ['Medium', <InlineCode key="m">&quot;m&quot;</InlineCode>, '640', 'Higher accuracy'],
              ['Compact', <InlineCode key="c">&quot;c&quot;</InlineCode>, '640', 'Best accuracy'],
            ]}
          />
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9t.pt")
# model = LibreYOLO("LibreYOLO9s.pt")
# model = LibreYOLO("LibreYOLO9m.pt")
# model = LibreYOLO("LibreYOLO9c.pt")`}</CodeBlock>

          <SubHeading>RF-DETR flagship</SubHeading>
          <DocTable
            headers={['Size', 'Code', 'Input size', 'Use case']}
            rows={[
              ['Nano', <InlineCode key="n">&quot;n&quot;</InlineCode>, '384', 'Edge'],
              ['Small', <InlineCode key="s">&quot;s&quot;</InlineCode>, '512', 'Balanced'],
              ['Medium', <InlineCode key="m">&quot;m&quot;</InlineCode>, '576', 'Higher accuracy'],
              ['Large', <InlineCode key="l">&quot;l&quot;</InlineCode>, '704', 'Maximum accuracy'],
            ]}
          />
          <CodeBlock language="python">{`from libreyolo import LibreRFDETR

model = LibreRFDETR(size="s")`}</CodeBlock>

          <SubHeading>Additional supported models</SubHeading>
          <DocTable
            headers={['Family', 'Model class', 'Model names']}
            rows={[
              ['YOLOX', <InlineCode key="yolox">LibreYOLOX</InlineCode>, 'LibreYOLOXn.pt, LibreYOLOXt.pt, LibreYOLOXs.pt, LibreYOLOXm.pt, LibreYOLOXl.pt, LibreYOLOXx.pt'],
              ['YOLO9-E2E', <InlineCode key="yolo9e2e">LibreYOLO9E2E</InlineCode>, 'LibreYOLO9E2Et.pt, LibreYOLO9E2Es.pt, LibreYOLO9E2Em.pt, LibreYOLO9E2Ec.pt'],
              ['YOLO-NAS', <InlineCode key="yolonas">LibreYOLONAS</InlineCode>, 'LibreYOLONASs.pt, LibreYOLONASm.pt, LibreYOLONASl.pt, LibreYOLONASn-pose.pt, LibreYOLONASs-pose.pt, LibreYOLONASm-pose.pt, LibreYOLONASl-pose.pt'],
              ['D-FINE', <InlineCode key="dfine">LibreDFINE</InlineCode>, 'LibreDFINEn.pt, LibreDFINEs.pt, LibreDFINEm.pt, LibreDFINEl.pt, LibreDFINEx.pt'],
              ['DEIM', <InlineCode key="deim">LibreDEIM</InlineCode>, 'LibreDEIMn.pt, LibreDEIMs.pt, LibreDEIMm.pt, LibreDEIMl.pt, LibreDEIMx.pt'],
              ['DEIMv2', <InlineCode key="deimv2">LibreDEIMv2</InlineCode>, 'LibreDEIMv2atto.pt, LibreDEIMv2femto.pt, LibreDEIMv2pico.pt, LibreDEIMv2n.pt, LibreDEIMv2s.pt, LibreDEIMv2m.pt, LibreDEIMv2l.pt, LibreDEIMv2x.pt'],
              ['RT-DETR', <InlineCode key="rtdetr">LibreRTDETR</InlineCode>, 'LibreRTDETRr18.pt, LibreRTDETRr34.pt, LibreRTDETRr50.pt, LibreRTDETRr50m.pt, LibreRTDETRr101.pt, LibreRTDETRl.pt, LibreRTDETRx.pt'],
              ['RT-DETRv2', <InlineCode key="rtdetrv2">LibreRTDETRv2</InlineCode>, 'LibreRTDETRv2r18.pt, LibreRTDETRv2r34.pt, LibreRTDETRv2r50.pt, LibreRTDETRv2r50m.pt, LibreRTDETRv2r101.pt'],
              ['RT-DETRv4', <InlineCode key="rtdetrv4">LibreRTDETRv4</InlineCode>, 'LibreRTDETRv4s.pt, LibreRTDETRv4m.pt, LibreRTDETRv4l.pt, LibreRTDETRv4x.pt'],
              ['PicoDet', <InlineCode key="picodet">LibrePICODET</InlineCode>, 'LibrePICODETs.pt, LibrePICODETm.pt, LibrePICODETl.pt'],
              ['EdgeCrafter', <InlineCode key="ec">LibreEC</InlineCode>, 'LibreECs.pt, LibreECm.pt, LibreECl.pt, LibreECx.pt, LibreECs-pose.pt, LibreECm-pose.pt, LibreECl-pose.pt, LibreECx-pose.pt, LibreECs-seg.pt, LibreECm-seg.pt, LibreECl-seg.pt, LibreECx-seg.pt'],
              ['DAMO-YOLO', <InlineCode key="damoyolo">LibreDAMOYOLO</InlineCode>, 'LibreDAMOYOLOns.pt, LibreDAMOYOLOnm.pt, LibreDAMOYOLOnl.pt, LibreDAMOYOLOt.pt, LibreDAMOYOLOs.pt, LibreDAMOYOLOm.pt, LibreDAMOYOLOl.pt'],
              ['RTMDet', <InlineCode key="rtmdet">LibreRTMDet</InlineCode>, 'LibreRTMDett.pt, LibreRTMDets.pt, LibreRTMDetm.pt, LibreRTMDetl.pt, LibreRTMDetx.pt'],
            ]}
          />

          <SubHeading>Factory function (recommended)</SubHeading>
          <P>
            The <InlineCode>LibreYOLO()</InlineCode> factory auto-detects everything from the weights file:
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

# Auto-detects: YOLOX, size=s, 80 classes
model = LibreYOLO("LibreYOLOXs.pt")

# Auto-detects: YOLO9, size=c, 80 classes
model = LibreYOLO("LibreYOLO9c.pt")

# Auto-detects: RT-DETR
model = LibreYOLO("LibreRTDETRr50.pt")

# RF-DETR checkpoints also work when you point at an actual checkpoint file
model = LibreYOLO("/path/to/checkpoint_best_regular.pth")

# ONNX models work too
model = LibreYOLO("model.onnx")

# TensorRT engines
model = LibreYOLO("model.engine")

# OpenVINO models (directory with model.xml)
model = LibreYOLO("model_openvino/")

# NCNN models (directory with model.ncnn.param + model.ncnn.bin)
model = LibreYOLO("model_ncnn/")`}</CodeBlock>
          <P>
            For recognized official checkpoint filenames, LibreYOLO can auto-download missing weights. For custom filenames and RF-DETR checkpoints, prefer explicit local paths or the family-specific constructors.
          </P>

          <Divider />

          {/* ────────────── PREDICTION ────────────── */}
          <SectionHeading id="prediction" icon={Crosshair}>Prediction</SectionHeading>

          <SubHeading>Basic prediction</SubHeading>
          <CodeBlock language="python">{`result = model("image.jpg")`}</CodeBlock>

          <SubHeading>All prediction parameters</SubHeading>
          <CodeBlock language="python">{`result = model(
    "image.jpg",
    conf=0.25,            # confidence threshold (default: 0.25)
    iou=0.45,             # NMS IoU threshold (default: 0.45)
    imgsz=640,            # input size override (default: model's native)
    classes=[0, 2, 5],    # filter to specific class IDs (default: all)
    max_det=300,          # max detections per image (default: 300)
    save=True,            # save annotated image (default: False)
    output_path="out/",   # where to save (default: runs/detect/predict*/)
    color_format="auto",  # "auto", "rgb", or "bgr"
    output_file_format="png",  # output format: "jpg", "png", "webp"
)`}</CodeBlock>
          <P>
            <InlineCode>model.predict(...)</InlineCode> is an alias for <InlineCode>model(...)</InlineCode>.
          </P>

          <SubHeading>Supported input formats</SubHeading>
          <P>LibreYOLO accepts images in any of these formats:</P>
          <CodeBlock language="python">{`# File path (string or pathlib.Path)
result = model("photo.jpg")
result = model(Path("photo.jpg"))

# URL
result = model("https://example.com/image.jpg")

# PIL Image
from PIL import Image
img = Image.open("photo.jpg")
result = model(img)

# NumPy array (HWC or CHW, RGB or BGR, uint8 or float32)
import numpy as np
arr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
result = model(arr)

# OpenCV (BGR) — specify color_format
import cv2
frame = cv2.imread("photo.jpg")
result = model(frame, color_format="bgr")

# PyTorch tensor (CHW or NCHW)
import torch
tensor = torch.randn(3, 640, 640)
result = model(tensor)

# Raw bytes
with open("photo.jpg", "rb") as f:
    result = model(f.read())

# Directory of images
results = model("images/", batch=4)`}</CodeBlock>

          <SubHeading>Working with results</SubHeading>
          <P>
            Every prediction returns a <InlineCode>Results</InlineCode> object (or a list of them for directories):
          </P>
          <CodeBlock language="python">{`result = model("image.jpg")

# Number of detections
len(result)  # e.g., 5

# Bounding boxes in xyxy format (x1, y1, x2, y2)
result.boxes.xyxy        # tensor of shape (N, 4)

# Bounding boxes in xywh format (center_x, center_y, width, height)
result.boxes.xywh        # tensor of shape (N, 4)

# Confidence scores
result.boxes.conf        # tensor of shape (N,)

# Class IDs
result.boxes.cls         # tensor of shape (N,)

# Combined data: [x1, y1, x2, y2, conf, cls]
result.boxes.data        # tensor of shape (N, 6)

# Metadata
result.orig_shape        # (height, width) of original image
result.path              # source file path (or None)
result.names             # {0: "person", 1: "bicycle", ...}

# Move to CPU / convert to numpy
result_cpu = result.cpu()
boxes_np = result.boxes.numpy()`}</CodeBlock>

          <SubHeading>Class filtering</SubHeading>
          <P>Filter detections to specific class IDs:</P>
          <CodeBlock language="python">{`# Only detect people (class 0) and cars (class 2)
result = model("image.jpg", classes=[0, 2])`}</CodeBlock>

          <Divider />

          {/* ────────────── TILED INFERENCE ────────────── */}
          <SectionHeading id="tiled-inference" icon={Grid3x3}>Tiled Inference</SectionHeading>
          <P>
            For images much larger than the model's input size (e.g., satellite imagery, drone footage), tiled inference splits the image into overlapping tiles, runs detection on each, and merges results.
          </P>
          <CodeBlock language="python">{`result = model(
    "large_aerial_image.jpg",
    tiling=True,
    overlap_ratio=0.2,   # 20% overlap between tiles (default)
    save=True,
)

# Extra metadata on tiled results
result.tiled           # True
result.num_tiles       # number of tiles used
result.saved_path      # output directory when save=True
result.tiles_path      # directory containing per-tile crops
result.grid_path       # grid visualization image`}</CodeBlock>

          <P>
            When <InlineCode>save=True</InlineCode> with tiling, LibreYOLO saves:
          </P>
          <ul className="space-y-2 mb-4">
            <FeatureItem><InlineCode>final_image.jpg</InlineCode> — full image with all merged detections drawn</FeatureItem>
            <FeatureItem><InlineCode>grid_visualization.jpg</InlineCode> — image showing tile grid overlay</FeatureItem>
            <FeatureItem><InlineCode>tiles/</InlineCode> — individual tile crops</FeatureItem>
            <FeatureItem><InlineCode>metadata.json</InlineCode> — tiling parameters and detection counts</FeatureItem>
          </ul>
          <P>
            If the image is already smaller than the model's input size, tiling is skipped automatically.
          </P>

          <Divider />

          {/* ────────────── TRAINING ────────────── */}
          <SectionHeading id="training" icon={GraduationCap}>Training</SectionHeading>
          <P>
            The flagship training paths for v1.2.0 are YOLO9 and RF-DETR. Additional training examples remain documented for API compatibility.
          </P>

          <SubHeading>YOLOX training</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLOX

model = LibreYOLOX(size="s")

results = model.train(
    data="coco128.yaml",     # path to data.yaml (required)

    # Training parameters
    epochs=100,              # default: 100
    batch=16,
    imgsz=640,

    # Optimizer
    lr0=0.01,                # initial learning rate
    optimizer="SGD",         # "SGD", "Adam", "AdamW"

    # System
    device="0",              # GPU device ("", "cpu", "cuda", "0", "0,1")
    workers=8,
    seed=0,

    # Output
    project="runs/train",
    name="exp",
    exist_ok=False,

    # Training features
    amp=True,                # automatic mixed precision
    patience=50,             # early stopping patience
    resume=False,            # resume from loaded checkpoint
)

print(f"Best mAP50-95: {results['best_mAP50_95']:.3f}")
print(f"Best checkpoint: {results['best_checkpoint']}")`}</CodeBlock>

          <P>After training, the model instance is automatically updated with the best weights.</P>

          <SubHeading>Training results dict</SubHeading>
          <CodeBlock language="python">{`{
    "final_loss": 2.31,
    "best_mAP50": 0.682,
    "best_mAP50_95": 0.451,
    "best_epoch": 87,
    "save_dir": "runs/train/exp",
    "best_checkpoint": "runs/train/exp/weights/best.pt",
    "last_checkpoint": "runs/train/exp/weights/last.pt",
}`}</CodeBlock>

          <SubHeading>Resuming training</SubHeading>
          <CodeBlock language="python">{`model = LibreYOLOX("runs/train/exp/weights/last.pt", size="s")
results = model.train(data="coco128.yaml", resume=True)`}</CodeBlock>

          <SubHeading>Custom dataset YAML format</SubHeading>
          <CodeBlock language="yaml" filename="data.yaml">{`path: /path/to/dataset
train: images/train
val: images/val
test: images/test  # optional

nc: 3
names: ["cat", "dog", "bird"]`}</CodeBlock>

          <SubHeading>YOLO9 training</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO9

model = LibreYOLO9("LibreYOLO9c.pt", size="c")

results = model.train(
    data="coco128.yaml",
    epochs=300,              # default: 300
    batch=16,
    imgsz=640,
    lr0=0.01,
    optimizer="SGD",
    device="0",
    workers=8,
    seed=0,
    project="runs/train",
    name="yolo9_exp",        # default: "yolo9_exp"
    exist_ok=False,
    resume=False,
    amp=True,
    patience=50,
)

print(f"Best mAP50-95: {results['best_mAP50_95']:.3f}")`}</CodeBlock>
          <P>
            YOLO9 training uses the same parameter API as YOLOX but defaults to <InlineCode>epochs=300</InlineCode> and <InlineCode>name=&quot;yolo9_exp&quot;</InlineCode>. It does not have a <InlineCode>pretrained</InlineCode> parameter.
          </P>

          <SubHeading>RT-DETR training</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreRTDETR

model = LibreRTDETR(size="r50")

results = model.train(
    data="coco128.yaml",
    epochs=72,               # default: 72
    batch=4,                 # default: 4
    imgsz=640,
    lr0=1e-4,
    lr_backbone=1e-5,
    optimizer="AdamW",
    scheduler="linear",
    device="0",
    workers=4,
    seed=0,
    project="runs/train",
    name="rtdetr_exp",
    exist_ok=False,
    pretrained=True,
    resume=False,
    amp=True,
    patience=50,
)`}</CodeBlock>
          <P>
            RT-DETR training uses the YOLO-style <InlineCode>data.yaml</InlineCode> pipeline, but it has its own defaults and adds <InlineCode>lr_backbone</InlineCode> plus <InlineCode>scheduler</InlineCode>.
          </P>

          <SubHeading>RF-DETR training</SubHeading>
          <P>
            RF-DETR uses a different training API that wraps the original rfdetr implementation:
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreRFDETR

model = LibreRFDETR(size="s")

results = model.train(
    data="path/to/dataset",  # Roboflow/COCO format directory
    epochs=100,
    batch_size=4,
    lr=1e-4,
    output_dir="runs/train",
)`}</CodeBlock>

          <P>RF-DETR datasets use COCO annotation format:</P>
          <CodeBlock language="text">{`dataset/
    train/
        _annotations.coco.json
        image1.jpg
        image2.jpg
    valid/
        _annotations.coco.json
        image1.jpg`}</CodeBlock>

          <Divider />

          {/* ────────────── VALIDATION ────────────── */}
          <SectionHeading id="validation" icon={CheckCircle2}>Validation</SectionHeading>
          <P>Run COCO-standard evaluation on a validation set:</P>
          <CodeBlock language="python">{`results = model.val(
    data="coco128.yaml",   # dataset config
    batch=16,
    imgsz=640,
    conf=0.001,            # low conf for mAP calculation
    iou=0.6,               # NMS IoU threshold
    split="val",           # "val", "test", or "train"
    save_json=False,       # save predictions as COCO JSON
    verbose=True,          # print per-class metrics
)

print(f"mAP50:    {results['metrics/mAP50']:.3f}")
print(f"mAP50-95: {results['metrics/mAP50-95']:.3f}")`}</CodeBlock>

          <SubHeading>Validation results dict</SubHeading>
          <P>
            By default, LibreYOLO uses COCO evaluation and returns 12 standard metrics:
          </P>
          <CodeBlock language="python">{`{
    "metrics/mAP50-95": 0.489,   # COCO primary metric (AP@[.5:.95])
    "metrics/mAP50": 0.721,      # AP@0.5 (PASCAL VOC style)
    "metrics/mAP75": 0.534,      # AP@0.75 (strict)
    "metrics/mAP_small": 0.291,
    "metrics/mAP_medium": 0.532,
    "metrics/mAP_large": 0.648,
    "metrics/AR1": 0.362,        # Average Recall (max 1 det)
    "metrics/AR10": 0.571,
    "metrics/AR100": 0.601,
    "metrics/AR_small": 0.387,
    "metrics/AR_medium": 0.641,
    "metrics/AR_large": 0.739,
}`}</CodeBlock>
          <P>
            Set <InlineCode>use_coco_eval=False</InlineCode> in <InlineCode>ValidationConfig</InlineCode> for legacy precision/recall metrics.
          </P>

          <Divider />

          {/* ────────────── EXPORT ────────────── */}
          <SectionHeading id="export" icon={Upload}>Export</SectionHeading>
          <P>Export PyTorch models to ONNX, TorchScript, TensorRT, OpenVINO, or NCNN for deployment.</P>

          <SubHeading>Quick export</SubHeading>
          <CodeBlock language="python">{`# ONNX (default)
model.export()

# TorchScript
model.export(format="torchscript")

# TensorRT (requires NVIDIA GPU + TensorRT)
model.export(format="tensorrt")

# OpenVINO (optimized for Intel hardware)
model.export(format="openvino")

# NCNN (via PNNX)
model.export(format="ncnn")`}</CodeBlock>

          <SubHeading>All export parameters</SubHeading>
          <CodeBlock language="python">{`path = model.export(
    format="onnx",            # "onnx", "torchscript", "tensorrt", "openvino", or "ncnn"
    output_path="model.onnx", # output file (auto-generated if None)
    imgsz=640,                # input resolution (default: model's native)
    opset=13,                 # ONNX opset version (RT-DETR / RF-DETR default to 17)
    simplify=True,            # run onnxsim graph simplification
    dynamic=True,             # enable dynamic batch axis
    half=False,               # export in FP16
    batch=1,                  # batch size for static graph
    device=None,              # device to trace on (default: model's current device)
    int8=False,               # INT8 quantization (TensorRT / OpenVINO only)
    data=None,                # calibration dataset for INT8
    fraction=1.0,             # fraction of calibration data to use
    workspace=4.0,            # TensorRT workspace size (GB)
    hardware_compatibility="none", # TensorRT compatibility mode
    gpu_device=0,             # GPU device index for TensorRT
    trt_config=None,          # optional TensorRT YAML config path
    verbose=False,            # verbose logging
)`}</CodeBlock>
          <P>
            OpenVINO INT8 export additionally requires <InlineCode>nncf</InlineCode>. NCNN export writes a directory containing <InlineCode>model.ncnn.param</InlineCode>, <InlineCode>model.ncnn.bin</InlineCode>, and <InlineCode>metadata.yaml</InlineCode>.
          </P>

          <SubHeading>ONNX metadata</SubHeading>
          <P>Exported ONNX files include embedded metadata:</P>
          <DocTable
            headers={['Key', 'Example value']}
            rows={[
              [<InlineCode key="v">libreyolo_version</InlineCode>, <InlineCode key="vv">&quot;1.0.0&quot;</InlineCode>],
              [<InlineCode key="f">model_family</InlineCode>, <InlineCode key="fv">&quot;yolox&quot;</InlineCode>],
              [<InlineCode key="s">model_size</InlineCode>, <InlineCode key="sv">&quot;s&quot;</InlineCode>],
              [<InlineCode key="c">nb_classes</InlineCode>, <InlineCode key="cv">&quot;80&quot;</InlineCode>],
              [<InlineCode key="n">names</InlineCode>, <span key="nv" className="text-xs"><InlineCode>{`'{"0": "person", "1": "bicycle", ...}'`}</InlineCode></span>],
              [<InlineCode key="i">imgsz</InlineCode>, <InlineCode key="iv">&quot;640&quot;</InlineCode>],
              [<InlineCode key="d">dynamic</InlineCode>, <InlineCode key="dv">&quot;True&quot;</InlineCode>],
              [<InlineCode key="h">half</InlineCode>, <InlineCode key="hv">&quot;False&quot;</InlineCode>],
            ]}
          />
          <P>
            This metadata is automatically read back when loading the model with <InlineCode>OnnxBackend</InlineCode>.
          </P>

          <SubHeading>Using the exporter factory directly</SubHeading>
          <CodeBlock language="python">{`from libreyolo.export import BaseExporter

exporter = BaseExporter.create("onnx", model)
path = exporter(dynamic=True, simplify=True)`}</CodeBlock>

          <Divider />

          {/* ────────────── ONNX INFERENCE ────────────── */}
          <SectionHeading id="onnx-inference" icon={Cpu}>ONNX Inference</SectionHeading>
          <P>
            Run inference using ONNX Runtime instead of PyTorch. Useful for deployment environments without PyTorch.
          </P>
          <CodeBlock language="python">{`from libreyolo import OnnxBackend

model = OnnxBackend("model.onnx")

result = model("image.jpg", conf=0.25, iou=0.45, save=True)
print(result.boxes.xyxy)`}</CodeBlock>

          <SubHeading>Auto-metadata</SubHeading>
          <P>
            If the ONNX file was exported by LibreYOLO, class names and class count are read automatically from the embedded metadata:
          </P>
          <CodeBlock language="python">{`# Export with metadata
model.export(format="onnx", output_path="model.onnx")

# Load — names and nb_classes auto-populated
onnx_model = OnnxBackend("model.onnx")
print(onnx_model.names)       # {0: "person", 1: "bicycle", ...}
print(onnx_model.nb_classes)  # 80`}</CodeBlock>

          <P>
            For ONNX files without metadata (e.g., exported by other tools), specify <InlineCode>nb_classes</InlineCode> manually:
          </P>
          <CodeBlock language="python">{`model = OnnxBackend("external_model.onnx", nb_classes=20)`}</CodeBlock>

          <SubHeading>Device selection</SubHeading>
          <CodeBlock language="python">{`# Auto-detect (CUDA if available, else CPU)
model = OnnxBackend("model.onnx", device="auto")

# Force CPU
model = OnnxBackend("model.onnx", device="cpu")

# Force CUDA
model = OnnxBackend("model.onnx", device="cuda")`}</CodeBlock>

          <SubHeading>Prediction parameters</SubHeading>
          <P>
            <InlineCode>OnnxBackend</InlineCode> supports the core prediction API shared by the runtime backends:
          </P>
          <CodeBlock language="python">{`result = model(
    "image.jpg",
    conf=0.25,
    iou=0.45,
    imgsz=640,
    classes=[0, 2],
    max_det=300,
    save=True,
    output_path="output/annotated.jpg",  # final file path when save=True
    color_format="auto",
)`}</CodeBlock>
          <P>
            Runtime backends do not expose PyTorch-only options such as <InlineCode>tiling</InlineCode>, <InlineCode>overlap_ratio</InlineCode>, or <InlineCode>output_file_format</InlineCode>.
          </P>
          <P>
            Runtime backends also handle saving a little differently from the PyTorch wrappers: if you set <InlineCode>output_path</InlineCode>, pass a final file path, not a directory. If you omit it, the current backend default is under <InlineCode>runs/detections/</InlineCode>.
          </P>

          <Divider />

          {/* ────────────── TENSORRT INFERENCE ────────────── */}
          <SectionHeading id="tensorrt-inference" icon={Cpu}>TensorRT Inference</SectionHeading>
          <P>
            Run inference using TensorRT for maximum throughput on NVIDIA GPUs. Requires CUDA plus the TensorRT Python bindings.
          </P>
          <CodeBlock language="python">{`from libreyolo import TensorRTBackend

model = TensorRTBackend("model.engine")

result = model("image.jpg", conf=0.25, iou=0.45, save=True)
print(result.boxes.xyxy)`}</CodeBlock>

          <SubHeading>Auto-detection via factory</SubHeading>
          <P>
            The <InlineCode>LibreYOLO()</InlineCode> factory automatically detects <InlineCode>.engine</InlineCode> files:
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

# Auto-detects TensorRT engine
model = LibreYOLO("model.engine")`}</CodeBlock>

          <P>
            <InlineCode>TensorRTBackend</InlineCode> supports the same core runtime-backend prediction API as ONNX and OpenVINO, including the same file-path-only <InlineCode>output_path</InlineCode> behavior for <InlineCode>save=True</InlineCode>.
          </P>

          <Divider />

          {/* ────────────── OPENVINO INFERENCE ────────────── */}
          <SectionHeading id="openvino-inference" icon={Cpu}>OpenVINO Inference</SectionHeading>
          <P>
            Run inference using OpenVINO, optimized for Intel CPUs, GPUs, and VPUs.
          </P>
          <CodeBlock language="python">{`from libreyolo import OpenVINOBackend

model = OpenVINOBackend("model_openvino/")

result = model("image.jpg", conf=0.25, iou=0.45, save=True)
print(result.boxes.xyxy)`}</CodeBlock>

          <SubHeading>Auto-detection via factory</SubHeading>
          <P>
            The <InlineCode>LibreYOLO()</InlineCode> factory automatically detects OpenVINO model directories:
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

# Auto-detects OpenVINO directory
model = LibreYOLO("model_openvino/")`}</CodeBlock>

          <P>
            <InlineCode>OpenVINOBackend</InlineCode> reads <InlineCode>metadata.yaml</InlineCode> when present and supports the same core runtime-backend prediction API.
          </P>

          <Divider />

          {/* ────────────── NCNN INFERENCE ────────────── */}
          <SectionHeading id="ncnn-inference" icon={Cpu}>NCNN Inference</SectionHeading>
          <P>
            Run inference using NCNN for lightweight deployment on CPU or Vulkan-capable GPU targets.
          </P>
          <CodeBlock language="python">{`from libreyolo import NcnnBackend

model = NcnnBackend("model_ncnn/")

result = model("image.jpg", conf=0.25, iou=0.45, save=True)
print(result.boxes.xyxy)`}</CodeBlock>

          <SubHeading>Auto-detection via factory</SubHeading>
          <P>
            The <InlineCode>LibreYOLO()</InlineCode> factory automatically detects NCNN model directories:
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

# Auto-detects NCNN directory
model = LibreYOLO("model_ncnn/")`}</CodeBlock>

          <P>
            An NCNN export directory contains <InlineCode>model.ncnn.param</InlineCode>, <InlineCode>model.ncnn.bin</InlineCode>, and usually <InlineCode>metadata.yaml</InlineCode>.
          </P>

          <Divider />

          {/* ────────────── API REFERENCE ────────────── */}
          <SectionHeading id="api-reference" icon={FileCode}>API Reference</SectionHeading>

          <SubHeading>LibreYOLO (factory)</SubHeading>
          <CodeBlock language="python">{`LibreYOLO(
    model_path: str,
    size: str = None,           # auto-detected from weights
    reg_max: int = 16,          # YOLO9 only
    nb_classes: int = None,     # auto-detected from weights
    device: str = "auto",
) -> LibreYOLOX | LibreYOLO9 | LibreRTDETR | LibreRFDETR | OnnxBackend | TensorRTBackend | OpenVINOBackend | NcnnBackend`}</CodeBlock>
          <P>
            Auto-detects model architecture, size, and class count from the weights file. It also handles <InlineCode>.onnx</InlineCode>, <InlineCode>.engine</InlineCode>, OpenVINO directories containing <InlineCode>model.xml</InlineCode>, and NCNN directories containing <InlineCode>model.ncnn.param</InlineCode> plus <InlineCode>model.ncnn.bin</InlineCode>.
          </P>

          <SubHeading>Prediction (PyTorch model wrappers)</SubHeading>
          <CodeBlock language="python">{`model(
    source,                     # image input (see supported formats)
    *,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = None,
    classes: list[int] = None,
    max_det: int = 300,
    save: bool = False,
    batch: int = 1,
    output_path: str = None,
    color_format: str = "auto",
    tiling: bool = False,
    overlap_ratio: float = 0.2,
    output_file_format: str = None,
) -> Results | list[Results]`}</CodeBlock>

          <SubHeading>Prediction (runtime backends)</SubHeading>
          <CodeBlock language="python">{`backend(
    source,
    *,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = None,
    classes: list[int] = None,
    max_det: int = 300,
    save: bool = False,
    batch: int = 1,
    output_path: str = None,    # final file path when save=True
    color_format: str = "auto",
) -> Results | list[Results]`}</CodeBlock>
          <P>
            If <InlineCode>output_path</InlineCode> is omitted for a runtime backend, the current default save location is <InlineCode>runs/detections/</InlineCode>.
          </P>

          <SubHeading>Results</SubHeading>
          <CodeBlock language="python">{`result = Results(
    boxes: Boxes,
    orig_shape: tuple[int, int],  # (height, width)
    path: str | None,
    names: dict[int, str],
)

len(result)          # number of detections
result.cpu()         # copy with tensors on CPU`}</CodeBlock>

          <SubHeading>Boxes</SubHeading>
          <CodeBlock language="python">{`boxes = Boxes(boxes, conf, cls)

boxes.xyxy           # (N, 4) tensor — x1, y1, x2, y2
boxes.xywh           # (N, 4) tensor — cx, cy, w, h
boxes.conf           # (N,) tensor — confidence scores
boxes.cls            # (N,) tensor — class IDs
boxes.data           # (N, 6) tensor — [xyxy, conf, cls]

len(boxes)           # number of boxes
boxes.cpu()          # copy on CPU
boxes.numpy()        # copy as numpy arrays`}</CodeBlock>

          <SubHeading>model.export()</SubHeading>
          <CodeBlock language="python">{`model.export(
    format: str = "onnx",       # "onnx", "torchscript", "tensorrt", "openvino", or "ncnn"
    *,
    output_path: str = None,
    imgsz: int = None,
    opset: int = 13,
    simplify: bool = True,
    dynamic: bool = True,
    half: bool = False,
    batch: int = 1,
    device: str = None,
    int8: bool = False,
    data: str = None,           # calibration data for INT8
    fraction: float = 1.0,      # fraction of calibration data
    workspace: float = 4.0,     # TensorRT workspace (GB)
    hardware_compatibility: str = "none",
    gpu_device: int = 0,
    trt_config = None,          # optional TensorRT YAML config path
    verbose: bool = False,
) -> str                        # path to exported file or directory`}</CodeBlock>

          <SubHeading>BaseExporter</SubHeading>
          <CodeBlock language="python">{`from libreyolo.export import BaseExporter

exporter = BaseExporter.create("onnx", model)
path = exporter(dynamic=True, simplify=True)

BaseExporter.create("ncnn", model)(output_path="model_ncnn")`}</CodeBlock>

          <SubHeading>model.val()</SubHeading>
          <CodeBlock language="python">{`model.val(
    data: str = None,           # path to data.yaml
    batch: int = 16,
    imgsz: int = None,
    conf: float = 0.001,
    iou: float = 0.6,
    device: str = None,
    split: str = "val",         # "val", "test", or "train"
    save_json: bool = False,
    verbose: bool = True,
) -> dict`}</CodeBlock>
          <P>Returns (COCO evaluation, default):</P>
          <CodeBlock language="python">{`{
    "metrics/mAP50-95": float,   # COCO primary metric
    "metrics/mAP50": float,
    "metrics/mAP75": float,
    "metrics/mAP_small": float,
    "metrics/mAP_medium": float,
    "metrics/mAP_large": float,
    "metrics/AR1": float,
    "metrics/AR10": float,
    "metrics/AR100": float,
    "metrics/AR_small": float,
    "metrics/AR_medium": float,
    "metrics/AR_large": float,
}`}</CodeBlock>

          <SubHeading>model.train() (YOLOX)</SubHeading>
          <CodeBlock language="python">{`model.train(
    data: str,                  # path to data.yaml (required)
    *,
    epochs: int = 100,
    batch: int = 16,
    imgsz: int = 640,
    lr0: float = 0.01,
    optimizer: str = "SGD",
    device: str = "",
    workers: int = 8,
    seed: int = 0,
    project: str = "runs/train",
    name: str = "exp",
    exist_ok: bool = False,
    pretrained: bool = True,
    resume: bool = False,
    amp: bool = True,
    patience: int = 50,
) -> dict`}</CodeBlock>
          <P>Returns:</P>
          <CodeBlock language="python">{`{
    "final_loss": float,
    "best_mAP50": float,
    "best_mAP50_95": float,
    "best_epoch": int,
    "save_dir": str,
    "best_checkpoint": str,
    "last_checkpoint": str,
}`}</CodeBlock>

          <SubHeading>model.train() (YOLO9)</SubHeading>
          <CodeBlock language="python">{`model.train(
    data: str,                  # path to data.yaml (required)
    *,
    epochs: int = 300,
    batch: int = 16,
    imgsz: int = 640,
    lr0: float = 0.01,
    optimizer: str = "SGD",
    device: str = "",
    workers: int = 8,
    seed: int = 0,
    project: str = "runs/train",
    name: str = "yolo9_exp",
    exist_ok: bool = False,
    resume: bool = False,
    amp: bool = True,
    patience: int = 50,
) -> dict`}</CodeBlock>
          <P>Returns the same dict as YOLOX training.</P>

          <SubHeading>model.train() (RT-DETR)</SubHeading>
          <CodeBlock language="python">{`model.train(
    data: str,                  # path to data.yaml (required)
    *,
    epochs: int = 72,
    batch: int = 4,
    imgsz: int = 640,
    lr0: float = 1e-4,
    lr_backbone: float = 1e-5,
    optimizer: str = "AdamW",
    scheduler: str = "linear",
    device: str = "",
    workers: int = 4,
    seed: int = 0,
    project: str = "runs/train",
    name: str = "rtdetr_exp",
    exist_ok: bool = False,
    pretrained: bool = True,
    resume: bool = False,
    amp: bool = True,
    patience: int = 50,
) -> dict`}</CodeBlock>

          <SubHeading>model.train() (RF-DETR)</SubHeading>
          <CodeBlock language="python">{`model.train(
    data: str,                  # path to dataset directory
    epochs: int = 100,
    batch_size: int = 4,
    lr: float = 1e-4,
    output_dir: str = "runs/train",
    resume: str = None,
    **kwargs,                   # additional RF-DETR training args
) -> dict`}</CodeBlock>

          <SubHeading>OnnxBackend</SubHeading>
          <CodeBlock language="python">{`OnnxBackend(
    onnx_path: str,
    nb_classes: int = 80,       # auto-read from metadata if available
    device: str = "auto",
)`}</CodeBlock>
          <P>
            Runs inference on an ONNX model with ONNX Runtime. Supports the runtime-backend prediction API shown above.
          </P>

          <SubHeading>TensorRTBackend</SubHeading>
          <CodeBlock language="python">{`TensorRTBackend(
    engine_path: str,
    nb_classes: int | None = None,
    device: str = "auto",
)`}</CodeBlock>
          <P>
            Runs inference on a TensorRT <InlineCode>.engine</InlineCode> file and can read metadata from an adjacent <InlineCode>.json</InlineCode> sidecar.
          </P>

          <SubHeading>OpenVINOBackend</SubHeading>
          <CodeBlock language="python">{`OpenVINOBackend(
    model_dir: str,
    nb_classes: int | None = None,
    device: str = "auto",
)`}</CodeBlock>
          <P>
            Runs inference on an OpenVINO model directory containing <InlineCode>model.xml</InlineCode> and optionally <InlineCode>metadata.yaml</InlineCode>.
          </P>

          <SubHeading>NcnnBackend</SubHeading>
          <CodeBlock language="python">{`NcnnBackend(
    model_dir: str,
    nb_classes: int | None = None,
    device: str = "auto",
)`}</CodeBlock>
          <P>
            Runs inference on an NCNN model directory containing <InlineCode>model.ncnn.param</InlineCode>, <InlineCode>model.ncnn.bin</InlineCode>, and optionally <InlineCode>metadata.yaml</InlineCode>.
          </P>

          <SubHeading>ValidationConfig</SubHeading>
          <CodeBlock language="python">{`from libreyolo import ValidationConfig

config = ValidationConfig(
    data="coco128.yaml",
    data_dir=None,             # override dataset root directory
    batch_size=16,
    imgsz=640,
    conf_thres=0.001,
    iou_thres=0.6,
    max_det=300,
    split="val",               # "val", "test", or "train"
    device="auto",
    save_json=False,
    verbose=True,
    half=False,
    use_coco_eval=True,        # use COCO eval (12 keys); False for legacy
    num_workers=4,
)

# Load/save YAML
config = ValidationConfig.from_yaml("config.yaml")
config.to_yaml("config.yaml")`}</CodeBlock>

          <Divider />

          {/* ────────────── ARCHITECTURE GUIDE ────────────── */}
          <SectionHeading id="architecture" icon={Wrench}>Architecture Guide</SectionHeading>
          <P>
            This section is for contributors who want to understand the codebase internals.
          </P>

          <SubHeading>Base class design</SubHeading>
          <P>
            PyTorch model families inherit from <InlineCode>BaseModel</InlineCode> in <InlineCode>libreyolo/models/base/model.py</InlineCode>. Subclasses implement these abstract methods:
          </P>
          <DocTable
            headers={['Method', 'Purpose']}
            rows={[
              [<InlineCode key="init">_init_model()</InlineCode>, 'Build and return the nn.Module'],
              [<InlineCode key="layers">_get_available_layers()</InlineCode>, 'Return layer-name to module mapping'],
              [<InlineCode key="pre-np">_get_preprocess_numpy()</InlineCode>, 'Return the NumPy preprocessor used for export / calibration'],
              [<InlineCode key="pre">_preprocess()</InlineCode>, 'Image to tensor conversion'],
              [<InlineCode key="fwd">_forward()</InlineCode>, 'Model forward pass'],
              [<InlineCode key="post">_postprocess()</InlineCode>, 'Raw output to detection dicts'],
            ]}
          />
          <P>
            <InlineCode>BaseModel</InlineCode> provides the shared wrapper behavior: prediction, export, validation, size/name metadata, and training helpers. The actual single-image, batch, and tiled inference flow lives in <InlineCode>libreyolo/models/base/inference.py</InlineCode>, while deployment runtimes live under <InlineCode>libreyolo/backends/</InlineCode>.
          </P>

          <SubHeading>Package structure</SubHeading>
          <CodeBlock language="text">{`libreyolo/
    __init__.py          # Public API exports
    models/
        __init__.py      # LibreYOLO() factory + model registry bootstrap
        base/
            model.py     # BaseModel
            inference.py # Shared prediction pipeline
        yolox/
            model.py
            nn.py
            utils.py
        yolo9/
            model.py
            nn.py
            utils.py
        rtdetr/
            model.py
            nn.py
            trainer.py
            utils.py
        rfdetr/
            model.py
            utils.py
            train.py
    backends/
        base.py          # BaseBackend runtime wrapper
        onnx.py          # OnnxBackend
        tensorrt.py      # TensorRTBackend
        openvino.py      # OpenVINOBackend
        ncnn.py          # NcnnBackend
    utils/
        results.py       # Results and Boxes classes
        image_loader.py  # Unified image loading
        general.py       # Path helpers, NMS, tiling utilities
    export/
        exporter.py      # BaseExporter and format registry
        onnx.py          # ONNX export logic
        torchscript.py   # TorchScript export logic
        tensorrt.py      # TensorRT export logic
        openvino.py      # OpenVINO export logic
        ncnn.py          # NCNN export logic
    training/
        config.py        # YOLOXTrainConfig / YOLOv9TrainConfig
        trainer.py       # YOLOXTrainer
        v9_trainer.py    # YOLOv9Trainer
        dataset.py       # Training dataset
        augment.py       # Mosaic, mixup, etc.
        loss.py          # YOLOX loss functions
        scheduler.py     # LR schedulers
        ema.py           # Exponential moving average
    validation/
        config.py        # ValidationConfig
        detection_validator.py  # DetectionValidator
        metrics.py       # DetMetrics, mAP computation
        base.py          # BaseValidator
        preprocessors.py # Per-model val preprocessing
    data/
        utils.py         # Dataset loading, YAML parsing
        yolo_coco_api.py # YOLO-to-COCO annotation bridge
    config/
        datasets/        # Built-in dataset YAML configs (coco8, coco128, coco5000, coco, etc.)`}</CodeBlock>

          <SubHeading>Adding a new model family</SubHeading>
          <ol className="space-y-2.5 mb-4 list-none">
            {[
              <>Create <InlineCode>libreyolo/models/newmodel/model.py</InlineCode> with a class inheriting <InlineCode>BaseModel</InlineCode></>,
              'Implement all abstract methods',
              <>Create the supporting network and utilities under <InlineCode>libreyolo/models/newmodel/</InlineCode></>,
              <>Add the import to <InlineCode>libreyolo/models/__init__.py</InlineCode> so the registry sees it</>,
              <>Export the class from <InlineCode>libreyolo/__init__.py</InlineCode></>,
              <>(Optional) Override <InlineCode>val_preprocessor_class</InlineCode> if validation preprocessing differs from the standard path</>,
            ].map((item, i) => (
              <li key={i} className="flex items-start gap-3 text-surface-600 dark:text-surface-400">
                <span className="w-6 h-6 rounded-lg bg-libre-500/10 border border-libre-500/20 text-libre-400 text-xs font-bold flex items-center justify-center shrink-0 mt-0.5">
                  {i + 1}
                </span>
                <span>{item}</span>
              </li>
            ))}
          </ol>

          <SubHeading>Export architecture</SubHeading>
          <P>
            <InlineCode>BaseExporter</InlineCode> in <InlineCode>libreyolo/export/exporter.py</InlineCode> is the export entrypoint. Concrete exporters register themselves through subclass registration, and callers use <InlineCode>BaseExporter.create(format, model)</InlineCode> to get the right implementation:
          </P>
          <CodeBlock language="python">{`from libreyolo.export import BaseExporter

onnx_exporter = BaseExporter.create("onnx", model)
ncnn_exporter = BaseExporter.create("ncnn", model)`}</CodeBlock>
          <P>
            To add a new export format, implement a new <InlineCode>BaseExporter</InlineCode> subclass with a unique <InlineCode>format_name</InlineCode> and import it from <InlineCode>libreyolo/export/exporter.py</InlineCode> so the registry is populated.
          </P>

          <Divider />

          {/* ────────────── DATASET FORMAT ────────────── */}
          <SectionHeading id="dataset-format" icon={Database}>Dataset Format</SectionHeading>
          <P>
            YOLO-style models use datasets configured via <InlineCode>data.yaml</InlineCode>. RF-DETR uses COCO-format annotations and is documented separately below.
          </P>

          <SubHeading>data.yaml structure</SubHeading>
          <CodeBlock language="yaml" filename="data.yaml">{`path: /absolute/path/to/dataset   # dataset root
train: images/train               # directory path, relative to path
val: images/val                   # directory path, relative to path
test: images/test                 # optional

nc: 80                            # number of classes
names: [                          # class names
  "person", "bicycle", "car", "motorcycle", "airplane",
  "bus", "train", "truck", "boat", "traffic light",
  # ...
]`}</CodeBlock>

          <SubHeading>File-list variant</SubHeading>
          <P>
            The same YAML format can also point <InlineCode>train</InlineCode>, <InlineCode>val</InlineCode>, or <InlineCode>test</InlineCode> at <InlineCode>.txt</InlineCode> files containing one image path per line:
          </P>
          <CodeBlock language="yaml" filename="coco.yaml">{`path: /absolute/path/to/coco
train: train2017.txt
val: val2017.txt
test: test-dev2017.txt

nc: 80
names: ["person", "bicycle", "car", "..."]`}</CodeBlock>

          <SubHeading>Directory layout</SubHeading>
          <CodeBlock language="text">{`dataset/
    images/
        train/
            img001.jpg
            img002.jpg
        val/
            img003.jpg
    labels/
        train/
            img001.txt
            img002.txt
        val/
            img003.txt`}</CodeBlock>

          <SubHeading>Label format</SubHeading>
          <P>
            One text file per image. Each line is one object:
          </P>
          <CodeBlock language="text">{`<class_id> <center_x> <center_y> <width> <height>`}</CodeBlock>
          <P>
            All coordinates are normalized to [0, 1] relative to image dimensions.
          </P>
          <P>Example (<InlineCode>img001.txt</InlineCode>):</P>
          <CodeBlock language="text" filename="img001.txt">{`0 0.5 0.4 0.3 0.6
2 0.1 0.2 0.05 0.1`}</CodeBlock>

          <SubHeading>Built-in datasets</SubHeading>
          <P>
            LibreYOLO ships built-in dataset configs under <InlineCode>libreyolo/config/datasets/</InlineCode> and can auto-download supported datasets on first use:
          </P>
          <CodeBlock language="python">{`# These download automatically on first use
results = model.val(data="coco8.yaml")
results = model.train(data="coco128.yaml", epochs=10)`}</CodeBlock>

          <SubHeading>RF-DETR dataset format</SubHeading>
          <P>
            RF-DETR uses COCO-format annotations (JSON) instead of YOLO text labels:
          </P>
          <CodeBlock language="text">{`dataset/
    train/
        _annotations.coco.json
        image1.jpg
    valid/
        _annotations.coco.json
        image1.jpg`}</CodeBlock>

          {/* Bottom spacer */}
          <div className="h-16" />
        </div>
      </main>
    </div>
  )
}


/* ─────────────────────────────────────────────────────────────────────────
   Chinese (zh-CN) content bundle for the ARCHIVED LibreYOLO v1.1.0 docs.
   Appended to the same 'use client' file; rendered when locale === 'zh'.
   Reuses shared presentational components (P, SubHeading, SectionHeading,
   CodeBlock, DocTable, InlineCode, Divider, FeatureItem) - do NOT redefine.
   All CodeBlock bodies are byte-identical to the English source (code stays
   English). hrefs / ids / classNames / icons / class & checkpoint names kept.
   ───────────────────────────────────────────────────────────────────────── */

/* ─── Section metadata for sidebar (zh) ─── */
const sectionsZh = [
  { id: 'introduction', title: '简介', icon: BookOpen },
  { id: 'installation', title: '安装', icon: Terminal },
  { id: 'quickstart', title: '快速开始', icon: Rocket },
  { id: 'models', title: '可用模型', icon: Layers },
  { id: 'prediction', title: '预测', icon: Crosshair },
  { id: 'tiled-inference', title: '分块推理', icon: Grid3x3 },
  { id: 'training', title: '训练', icon: GraduationCap },
  { id: 'validation', title: '验证', icon: CheckCircle2 },
  { id: 'export', title: '导出', icon: Upload },
  { id: 'onnx-inference', title: 'ONNX 推理', icon: Cpu },
  { id: 'tensorrt-inference', title: 'TensorRT 推理', icon: Cpu },
  { id: 'openvino-inference', title: 'OpenVINO 推理', icon: Cpu },
  { id: 'ncnn-inference', title: 'NCNN 推理', icon: Cpu },
  { id: 'api-reference', title: 'API 参考', icon: FileCode },
  { id: 'architecture', title: '架构指南', icon: Wrench },
  { id: 'dataset-format', title: '数据集格式', icon: Database },
]

const docsVersionLabelsZh = {
  'Pre-release': '预发布',
  'Latest': '最新',
  'Archived': '已归档',
}

/* ─── Sidebar (zh) ─── */

function SidebarZh({ activeSection, onNavigate, currentVersion = 'v1.1.0', className = '' }) {
  return (
    <nav className={className}>
      <div className="flex items-center gap-2 mb-6 px-3">
        <BookOpen className="w-5 h-5 text-libre-600 dark:text-libre-400" />
        <span className="text-sm font-semibold text-surface-800 dark:text-white tracking-wide uppercase">文档</span>
      </div>
      <div className="mb-6 mx-3 rounded-lg border border-surface-200 dark:border-white/[0.08] bg-surface-50 dark:bg-white/[0.03] p-2">
        <div className="text-[11px] font-semibold uppercase tracking-wide text-surface-500 dark:text-surface-500 mb-2">
          版本
        </div>
        <div className="space-y-1">
          {docsVersions.map(({ version, label, href }) => {
            const isCurrent = version === currentVersion
            return (
              <a
                key={version}
                href={href}
                className={`flex items-center justify-between rounded-md px-2.5 py-2 text-sm font-medium transition-colors ${
                  isCurrent
                    ? 'bg-libre-500/10 text-libre-700 dark:text-libre-300'
                    : 'text-surface-600 dark:text-surface-400 hover:bg-white dark:hover:bg-white/[0.05]'
                }`}
              >
                <span>{version}</span>
                <span className="text-[11px] font-semibold uppercase tracking-wide">{docsVersionLabelsZh[label] || label}</span>
              </a>
            )
          })}
        </div>
      </div>
      <ul className="space-y-0.5">
        {sectionsZh.map(({ id, title, icon: Icon }) => {
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
    </nav>
  )
}

/* ─── Main docs page (zh) ─── */

function DocsPageZh({ version = 'v1.1.0', isLatest = false }) {
  const [activeSection, setActiveSection] = useState('introduction')
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [docsCopied, setDocsCopied] = useState(false)

  // Scroll spy — pick the last section whose heading has scrolled past 30% of viewport
  useEffect(() => {
    const handleScroll = () => {
      const threshold = window.innerHeight * 0.3
      let current = sections[0].id

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
  }, [])

  const navigateTo = (id) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })
    setMobileMenuOpen(false)
  }

  const copyDocs = async () => {
    const docsText = document.querySelector('[data-docs-content]')?.innerText || ''
    await navigator.clipboard.writeText(`# LibreYOLO 文档 ${version}\n\n${docsText}`)
    setDocsCopied(true)
    setTimeout(() => setDocsCopied(false), 2000)
  }

  return (
    <div className="flex min-h-screen">
      {/* Desktop sidebar */}
      <aside className="hidden lg:block fixed left-0 top-20 bottom-0 w-64 border-r border-surface-200 dark:border-white/[0.06] bg-white/80 dark:bg-surface-950/50 backdrop-blur-sm overflow-y-auto py-8 px-4 z-30">
        <SidebarZh activeSection={activeSection} onNavigate={navigateTo} currentVersion={version} />
      </aside>

      {/* Mobile sidebar toggle */}
      <button
        onClick={() => setMobileMenuOpen(true)}
        className="lg:hidden fixed bottom-6 right-6 z-40 w-12 h-12 rounded-full bg-libre-500 text-white shadow-lg shadow-libre-500/30 flex items-center justify-center hover:bg-libre-400 transition-colors"
        aria-label="打开导航"
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
                <span className="text-sm font-semibold text-surface-800 dark:text-white tracking-wide uppercase">文档</span>
                <button
                  onClick={() => setMobileMenuOpen(false)}
                  className="p-1.5 rounded-lg text-surface-500 dark:text-surface-400 hover:text-surface-900 dark:hover:text-white hover:bg-surface-100 dark:hover:bg-white/[0.06] transition-colors"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              <SidebarZh activeSection={activeSection} onNavigate={navigateTo} currentVersion={version} />
            </motion.aside>
          </>
        )}
      </AnimatePresence>

      {/* Main content */}
      <main className="flex-1 lg:ml-64 min-h-screen pt-28 lg:pt-32 pb-24 px-6 lg:px-12">
        <div className="max-w-4xl mx-auto" data-docs-content>
          <div className="mb-8 rounded-lg border border-surface-200 dark:border-white/[0.08] bg-white/80 dark:bg-white/[0.03] p-4 shadow-sm">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <p className="text-sm text-surface-600 dark:text-surface-400">
                    {isLatest
                    ? '本文档跟踪即将发布的 v1.2.0 开发分支。如需当前已发布的文档，请使用 v1.1.0。'
                    : '此归档版本保持可链接，以便旧版安装、搜索结果和智能体能够指向正确的文档。'}
                </p>
              </div>
              <button
                onClick={copyDocs}
                className="inline-flex items-center justify-center gap-2 rounded-lg border border-surface-200 dark:border-white/[0.08] bg-surface-950 px-3.5 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-surface-800 dark:bg-white dark:text-surface-950 dark:hover:bg-surface-200"
              >
                {docsCopied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                {docsCopied ? '已复制文档' : '复制文档'}
              </button>
            </div>
          </div>

          {/* ────────────── INTRODUCTION ────────────── */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
            <SectionHeading id="introduction" icon={BookOpen}>简介</SectionHeading>
            <P>
              LibreYOLO 是一个基于 MIT 许可的目标检测库，专注于两个旗舰模型系列：作为 CNN 旗舰的 <strong className="text-surface-800 dark:text-white">YOLO9</strong> 与作为 Transformer 旗舰的 <strong className="text-surface-800 dark:text-white">RF-DETR</strong>。其 API 在预测、训练、验证和导出之间保持一致。
            </P>
            <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")
results = model("image.jpg", conf=0.25, save=True)
print(results.boxes.xyxy)`}</CodeBlock>

            <SubHeading>核心特性</SubHeading>
            <ul className="space-y-2.5 mb-4">
              <FeatureItem>对 YOLO9 和 RF-DETR 的旗舰支持</FeatureItem>
              <FeatureItem>从权重自动检测模型架构、尺寸和类别数</FeatureItem>
              <FeatureItem>针对大尺寸/高分辨率图像的分块推理</FeatureItem>
              <FeatureItem>支持 ONNX、TorchScript、TensorRT、OpenVINO 和 NCNN 导出，并嵌入元数据</FeatureItem>
              <FeatureItem>ONNX Runtime、TensorRT、OpenVINO 和 NCNN 推理后端</FeatureItem>
              <FeatureItem>兼容 COCO 的验证，提供 mAP 指标</FeatureItem>
              <FeatureItem>接受任意图像格式：文件路径、URL、PIL、NumPy、PyTorch 张量、原始字节</FeatureItem>
            </ul>
          </motion.div>

          <Divider />

          {/* ────────────── INSTALLATION ────────────── */}
          <SectionHeading id="installation" icon={Terminal}>安装</SectionHeading>
          <SubHeading>环境要求</SubHeading>
          <ul className="space-y-1.5 mb-4">
            <li className="flex items-center gap-2 text-surface-600 dark:text-surface-400">
              <span className="w-1.5 h-1.5 rounded-full bg-libre-400" />Python 3.10+
            </li>
            <li className="flex items-center gap-2 text-surface-600 dark:text-surface-400">
              <span className="w-1.5 h-1.5 rounded-full bg-libre-400" />PyTorch 1.7+
            </li>
          </ul>

          <SubHeading>从 PyPI 安装</SubHeading>
          <CodeBlock language="bash">{`pip install libreyolo`}</CodeBlock>

          <SubHeading>从源码安装</SubHeading>
          <CodeBlock language="bash">{`git clone https://github.com/Libre-YOLO/libreyolo.git
cd libreyolo
pip install -e .`}</CodeBlock>

          <SubHeading>可选依赖</SubHeading>
          <CodeBlock language="bash">{`# ONNX export and inference
pip install libreyolo[onnx]
# or: pip install onnx onnxsim onnxscript onnxruntime

# RT-DETR support
pip install libreyolo[rtdetr]
# or: pip install transformers timm

# RF-DETR support
pip install libreyolo[rfdetr]
# or: pip install rfdetr transformers timm supervision

# TensorRT export and inference (NVIDIA GPU)
pip install libreyolo[tensorrt]
# Note: TensorRT itself requires manual installation (depends on CUDA version)

# OpenVINO export and inference (Intel CPU/GPU/VPU)
pip install libreyolo[openvino]
# INT8 export also needs: pip install nncf

# NCNN export and inference
pip install libreyolo[ncnn]
# or: pip install pnnx ncnn`}</CodeBlock>

          <P>如果使用 <InlineCode>uv</InlineCode>，最可靠的方式是为每个 extra 使用独立的虚拟环境：</P>
          <CodeBlock language="bash">{`# ONNX environment
uv venv .venv-onnx
uv pip install --python .venv-onnx/bin/python -e '.[onnx]'

# RT-DETR environment
uv venv .venv-rtdetr
uv pip install --python .venv-rtdetr/bin/python -e '.[rtdetr]'

# Repeat with .[rfdetr], .[openvino], .[ncnn], or .[tensorrt] as needed`}</CodeBlock>
          <P>
            这样可以避免改动项目环境，并使可选依赖保持隔离。TensorRT、OpenVINO 和 NCNN 等特定厂商的 extra 仍可能需要平台特定的原生包。
          </P>

          <Divider />

          {/* ────────────── QUICKSTART ────────────── */}
          <SectionHeading id="quickstart" icon={Rocket}>快速开始</SectionHeading>

          <SubHeading>加载模型并运行推理</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

# Auto-detects architecture and size from the weights file
model = LibreYOLO("LibreYOLO9c.pt")

# Run on a single image
result = model("photo.jpg")

print(f"Found {len(result)} objects")
print(result.boxes.xyxy)   # bounding boxes (N, 4)
print(result.boxes.conf)   # confidence scores (N,)
print(result.boxes.cls)    # class IDs (N,)`}</CodeBlock>

          <SubHeading>保存标注后的输出</SubHeading>
          <CodeBlock language="python">{`result = model("photo.jpg", save=True)
# Saved under runs/detect/predict*/photo.jpg by default`}</CodeBlock>

          <SubHeading>处理整个目录</SubHeading>
          <CodeBlock language="python">{`results = model("images/", save=True, batch=4)
for r in results:
    print(f"{r.path}: {len(r)} detections")`}</CodeBlock>

          <Divider />

          {/* ────────────── AVAILABLE MODELS ────────────── */}
          <SectionHeading id="models" icon={Layers}>可用模型</SectionHeading>

          <P>
            v1.2.0 文档以两个旗舰模型系列为中心。YOLO9 是 CNN 旗舰，RF-DETR 是 Transformer 旗舰。其他受支持的系列以简要参考形式列出。
          </P>

          <SubHeading>YOLO9 旗舰</SubHeading>
          <DocTable
            headers={['尺寸', '代码', '输入尺寸', '用途']}
            rows={[
              ['Tiny', <InlineCode key="t">&quot;t&quot;</InlineCode>, '640', '快速推理'],
              ['Small', <InlineCode key="s">&quot;s&quot;</InlineCode>, '640', '均衡'],
              ['Medium', <InlineCode key="m">&quot;m&quot;</InlineCode>, '640', '更高精度'],
              ['Compact', <InlineCode key="c">&quot;c&quot;</InlineCode>, '640', '最佳精度'],
            ]}
          />
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9t.pt")
# model = LibreYOLO("LibreYOLO9s.pt")
# model = LibreYOLO("LibreYOLO9m.pt")
# model = LibreYOLO("LibreYOLO9c.pt")`}</CodeBlock>

          <SubHeading>RF-DETR 旗舰</SubHeading>
          <DocTable
            headers={['尺寸', '代码', '输入尺寸', '用途']}
            rows={[
              ['Nano', <InlineCode key="n">&quot;n&quot;</InlineCode>, '384', '边缘端'],
              ['Small', <InlineCode key="s">&quot;s&quot;</InlineCode>, '512', '均衡'],
              ['Medium', <InlineCode key="m">&quot;m&quot;</InlineCode>, '576', '更高精度'],
              ['Large', <InlineCode key="l">&quot;l&quot;</InlineCode>, '704', '最高精度'],
            ]}
          />
          <CodeBlock language="python">{`from libreyolo import LibreRFDETR

model = LibreRFDETR(size="s")`}</CodeBlock>

          <SubHeading>其他受支持的模型</SubHeading>
          <DocTable
            headers={['系列', '模型类', '模型名称']}
            rows={[
              ['YOLOX', <InlineCode key="yolox">LibreYOLOX</InlineCode>, 'LibreYOLOXn.pt, LibreYOLOXt.pt, LibreYOLOXs.pt, LibreYOLOXm.pt, LibreYOLOXl.pt, LibreYOLOXx.pt'],
              ['YOLO9-E2E', <InlineCode key="yolo9e2e">LibreYOLO9E2E</InlineCode>, 'LibreYOLO9E2Et.pt, LibreYOLO9E2Es.pt, LibreYOLO9E2Em.pt, LibreYOLO9E2Ec.pt'],
              ['YOLO-NAS', <InlineCode key="yolonas">LibreYOLONAS</InlineCode>, 'LibreYOLONASs.pt, LibreYOLONASm.pt, LibreYOLONASl.pt, LibreYOLONASn-pose.pt, LibreYOLONASs-pose.pt, LibreYOLONASm-pose.pt, LibreYOLONASl-pose.pt'],
              ['D-FINE', <InlineCode key="dfine">LibreDFINE</InlineCode>, 'LibreDFINEn.pt, LibreDFINEs.pt, LibreDFINEm.pt, LibreDFINEl.pt, LibreDFINEx.pt'],
              ['DEIM', <InlineCode key="deim">LibreDEIM</InlineCode>, 'LibreDEIMn.pt, LibreDEIMs.pt, LibreDEIMm.pt, LibreDEIMl.pt, LibreDEIMx.pt'],
              ['DEIMv2', <InlineCode key="deimv2">LibreDEIMv2</InlineCode>, 'LibreDEIMv2atto.pt, LibreDEIMv2femto.pt, LibreDEIMv2pico.pt, LibreDEIMv2n.pt, LibreDEIMv2s.pt, LibreDEIMv2m.pt, LibreDEIMv2l.pt, LibreDEIMv2x.pt'],
              ['RT-DETR', <InlineCode key="rtdetr">LibreRTDETR</InlineCode>, 'LibreRTDETRr18.pt, LibreRTDETRr34.pt, LibreRTDETRr50.pt, LibreRTDETRr50m.pt, LibreRTDETRr101.pt, LibreRTDETRl.pt, LibreRTDETRx.pt'],
              ['RT-DETRv2', <InlineCode key="rtdetrv2">LibreRTDETRv2</InlineCode>, 'LibreRTDETRv2r18.pt, LibreRTDETRv2r34.pt, LibreRTDETRv2r50.pt, LibreRTDETRv2r50m.pt, LibreRTDETRv2r101.pt'],
              ['RT-DETRv4', <InlineCode key="rtdetrv4">LibreRTDETRv4</InlineCode>, 'LibreRTDETRv4s.pt, LibreRTDETRv4m.pt, LibreRTDETRv4l.pt, LibreRTDETRv4x.pt'],
              ['PicoDet', <InlineCode key="picodet">LibrePICODET</InlineCode>, 'LibrePICODETs.pt, LibrePICODETm.pt, LibrePICODETl.pt'],
              ['EdgeCrafter', <InlineCode key="ec">LibreEC</InlineCode>, 'LibreECs.pt, LibreECm.pt, LibreECl.pt, LibreECx.pt, LibreECs-pose.pt, LibreECm-pose.pt, LibreECl-pose.pt, LibreECx-pose.pt, LibreECs-seg.pt, LibreECm-seg.pt, LibreECl-seg.pt, LibreECx-seg.pt'],
              ['DAMO-YOLO', <InlineCode key="damoyolo">LibreDAMOYOLO</InlineCode>, 'LibreDAMOYOLOns.pt, LibreDAMOYOLOnm.pt, LibreDAMOYOLOnl.pt, LibreDAMOYOLOt.pt, LibreDAMOYOLOs.pt, LibreDAMOYOLOm.pt, LibreDAMOYOLOl.pt'],
              ['RTMDet', <InlineCode key="rtmdet">LibreRTMDet</InlineCode>, 'LibreRTMDett.pt, LibreRTMDets.pt, LibreRTMDetm.pt, LibreRTMDetl.pt, LibreRTMDetx.pt'],
            ]}
          />

          <SubHeading>工厂函数（推荐）</SubHeading>
          <P>
            <InlineCode>LibreYOLO()</InlineCode> 工厂函数会从权重文件自动检测一切：
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

# Auto-detects: YOLOX, size=s, 80 classes
model = LibreYOLO("LibreYOLOXs.pt")

# Auto-detects: YOLO9, size=c, 80 classes
model = LibreYOLO("LibreYOLO9c.pt")

# Auto-detects: RT-DETR
model = LibreYOLO("LibreRTDETRr50.pt")

# RF-DETR checkpoints also work when you point at an actual checkpoint file
model = LibreYOLO("/path/to/checkpoint_best_regular.pth")

# ONNX models work too
model = LibreYOLO("model.onnx")

# TensorRT engines
model = LibreYOLO("model.engine")

# OpenVINO models (directory with model.xml)
model = LibreYOLO("model_openvino/")

# NCNN models (directory with model.ncnn.param + model.ncnn.bin)
model = LibreYOLO("model_ncnn/")`}</CodeBlock>
          <P>
            对于可识别的官方检查点文件名，LibreYOLO 可以自动下载缺失的权重。对于自定义文件名和 RF-DETR 检查点，建议使用明确的本地路径或特定系列的构造函数。
          </P>

          <Divider />

          {/* ────────────── PREDICTION ────────────── */}
          <SectionHeading id="prediction" icon={Crosshair}>预测</SectionHeading>

          <SubHeading>基本预测</SubHeading>
          <CodeBlock language="python">{`result = model("image.jpg")`}</CodeBlock>

          <SubHeading>全部预测参数</SubHeading>
          <CodeBlock language="python">{`result = model(
    "image.jpg",
    conf=0.25,            # confidence threshold (default: 0.25)
    iou=0.45,             # NMS IoU threshold (default: 0.45)
    imgsz=640,            # input size override (default: model's native)
    classes=[0, 2, 5],    # filter to specific class IDs (default: all)
    max_det=300,          # max detections per image (default: 300)
    save=True,            # save annotated image (default: False)
    output_path="out/",   # where to save (default: runs/detect/predict*/)
    color_format="auto",  # "auto", "rgb", or "bgr"
    output_file_format="png",  # output format: "jpg", "png", "webp"
)`}</CodeBlock>
          <P>
            <InlineCode>model.predict(...)</InlineCode> 是 <InlineCode>model(...)</InlineCode> 的别名。
          </P>

          <SubHeading>支持的输入格式</SubHeading>
          <P>LibreYOLO 接受以下任意格式的图像：</P>
          <CodeBlock language="python">{`# File path (string or pathlib.Path)
result = model("photo.jpg")
result = model(Path("photo.jpg"))

# URL
result = model("https://example.com/image.jpg")

# PIL Image
from PIL import Image
img = Image.open("photo.jpg")
result = model(img)

# NumPy array (HWC or CHW, RGB or BGR, uint8 or float32)
import numpy as np
arr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
result = model(arr)

# OpenCV (BGR) — specify color_format
import cv2
frame = cv2.imread("photo.jpg")
result = model(frame, color_format="bgr")

# PyTorch tensor (CHW or NCHW)
import torch
tensor = torch.randn(3, 640, 640)
result = model(tensor)

# Raw bytes
with open("photo.jpg", "rb") as f:
    result = model(f.read())

# Directory of images
results = model("images/", batch=4)`}</CodeBlock>

          <SubHeading>处理结果</SubHeading>
          <P>
            每次预测都会返回一个 <InlineCode>Results</InlineCode> 对象（处理目录时则返回它们的列表）：
          </P>
          <CodeBlock language="python">{`result = model("image.jpg")

# Number of detections
len(result)  # e.g., 5

# Bounding boxes in xyxy format (x1, y1, x2, y2)
result.boxes.xyxy        # tensor of shape (N, 4)

# Bounding boxes in xywh format (center_x, center_y, width, height)
result.boxes.xywh        # tensor of shape (N, 4)

# Confidence scores
result.boxes.conf        # tensor of shape (N,)

# Class IDs
result.boxes.cls         # tensor of shape (N,)

# Combined data: [x1, y1, x2, y2, conf, cls]
result.boxes.data        # tensor of shape (N, 6)

# Metadata
result.orig_shape        # (height, width) of original image
result.path              # source file path (or None)
result.names             # {0: "person", 1: "bicycle", ...}

# Move to CPU / convert to numpy
result_cpu = result.cpu()
boxes_np = result.boxes.numpy()`}</CodeBlock>

          <SubHeading>类别过滤</SubHeading>
          <P>将检测结果过滤到特定的类别 ID：</P>
          <CodeBlock language="python">{`# Only detect people (class 0) and cars (class 2)
result = model("image.jpg", classes=[0, 2])`}</CodeBlock>

          <Divider />

          {/* ────────────── TILED INFERENCE ────────────── */}
          <SectionHeading id="tiled-inference" icon={Grid3x3}>分块推理</SectionHeading>
          <P>
            对于远大于模型输入尺寸的图像（例如卫星影像、无人机航拍），分块推理会将图像切分为相互重叠的图块，对每个图块运行检测，然后合并结果。
          </P>
          <CodeBlock language="python">{`result = model(
    "large_aerial_image.jpg",
    tiling=True,
    overlap_ratio=0.2,   # 20% overlap between tiles (default)
    save=True,
)

# Extra metadata on tiled results
result.tiled           # True
result.num_tiles       # number of tiles used
result.saved_path      # output directory when save=True
result.tiles_path      # directory containing per-tile crops
result.grid_path       # grid visualization image`}</CodeBlock>

          <P>
            在分块时设置 <InlineCode>save=True</InlineCode>，LibreYOLO 会保存：
          </P>
          <ul className="space-y-2 mb-4">
            <FeatureItem><InlineCode>final_image.jpg</InlineCode> - 绘制了所有合并检测结果的完整图像</FeatureItem>
            <FeatureItem><InlineCode>grid_visualization.jpg</InlineCode> - 显示图块网格叠加的图像</FeatureItem>
            <FeatureItem><InlineCode>tiles/</InlineCode> - 单个图块裁剪</FeatureItem>
            <FeatureItem><InlineCode>metadata.json</InlineCode> - 分块参数和检测计数</FeatureItem>
          </ul>
          <P>
            如果图像已经小于模型的输入尺寸，则会自动跳过分块。
          </P>

          <Divider />

          {/* ────────────── TRAINING ────────────── */}
          <SectionHeading id="training" icon={GraduationCap}>训练</SectionHeading>
          <P>
            v1.2.0 的旗舰训练路径是 YOLO9 和 RF-DETR。为保持 API 兼容性，文档仍保留了其他训练示例。
          </P>

          <SubHeading>YOLOX 训练</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLOX

model = LibreYOLOX(size="s")

results = model.train(
    data="coco128.yaml",     # path to data.yaml (required)

    # Training parameters
    epochs=100,              # default: 100
    batch=16,
    imgsz=640,

    # Optimizer
    lr0=0.01,                # initial learning rate
    optimizer="SGD",         # "SGD", "Adam", "AdamW"

    # System
    device="0",              # GPU device ("", "cpu", "cuda", "0", "0,1")
    workers=8,
    seed=0,

    # Output
    project="runs/train",
    name="exp",
    exist_ok=False,

    # Training features
    amp=True,                # automatic mixed precision
    patience=50,             # early stopping patience
    resume=False,            # resume from loaded checkpoint
)

print(f"Best mAP50-95: {results['best_mAP50_95']:.3f}")
print(f"Best checkpoint: {results['best_checkpoint']}")`}</CodeBlock>

          <P>训练完成后，模型实例会自动更新为最佳权重。</P>

          <SubHeading>训练结果字典</SubHeading>
          <CodeBlock language="python">{`{
    "final_loss": 2.31,
    "best_mAP50": 0.682,
    "best_mAP50_95": 0.451,
    "best_epoch": 87,
    "save_dir": "runs/train/exp",
    "best_checkpoint": "runs/train/exp/weights/best.pt",
    "last_checkpoint": "runs/train/exp/weights/last.pt",
}`}</CodeBlock>

          <SubHeading>恢复训练</SubHeading>
          <CodeBlock language="python">{`model = LibreYOLOX("runs/train/exp/weights/last.pt", size="s")
results = model.train(data="coco128.yaml", resume=True)`}</CodeBlock>

          <SubHeading>自定义数据集 YAML 格式</SubHeading>
          <CodeBlock language="yaml" filename="data.yaml">{`path: /path/to/dataset
train: images/train
val: images/val
test: images/test  # optional

nc: 3
names: ["cat", "dog", "bird"]`}</CodeBlock>

          <SubHeading>YOLO9 训练</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO9

model = LibreYOLO9("LibreYOLO9c.pt", size="c")

results = model.train(
    data="coco128.yaml",
    epochs=300,              # default: 300
    batch=16,
    imgsz=640,
    lr0=0.01,
    optimizer="SGD",
    device="0",
    workers=8,
    seed=0,
    project="runs/train",
    name="yolo9_exp",        # default: "yolo9_exp"
    exist_ok=False,
    resume=False,
    amp=True,
    patience=50,
)

print(f"Best mAP50-95: {results['best_mAP50_95']:.3f}")`}</CodeBlock>
          <P>
            YOLO9 训练使用与 YOLOX 相同的参数 API，但默认 <InlineCode>epochs=300</InlineCode> 和 <InlineCode>name=&quot;yolo9_exp&quot;</InlineCode>。它没有 <InlineCode>pretrained</InlineCode> 参数。
          </P>

          <SubHeading>RT-DETR 训练</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreRTDETR

model = LibreRTDETR(size="r50")

results = model.train(
    data="coco128.yaml",
    epochs=72,               # default: 72
    batch=4,                 # default: 4
    imgsz=640,
    lr0=1e-4,
    lr_backbone=1e-5,
    optimizer="AdamW",
    scheduler="linear",
    device="0",
    workers=4,
    seed=0,
    project="runs/train",
    name="rtdetr_exp",
    exist_ok=False,
    pretrained=True,
    resume=False,
    amp=True,
    patience=50,
)`}</CodeBlock>
          <P>
            RT-DETR 训练使用 YOLO 风格的 <InlineCode>data.yaml</InlineCode> 流程，但有自己的默认值，并额外增加了 <InlineCode>lr_backbone</InlineCode> 和 <InlineCode>scheduler</InlineCode>。
          </P>

          <SubHeading>RF-DETR 训练</SubHeading>
          <P>
            RF-DETR 使用不同的训练 API，它封装了原始的 rfdetr 实现：
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreRFDETR

model = LibreRFDETR(size="s")

results = model.train(
    data="path/to/dataset",  # Roboflow/COCO format directory
    epochs=100,
    batch_size=4,
    lr=1e-4,
    output_dir="runs/train",
)`}</CodeBlock>

          <P>RF-DETR 数据集使用 COCO 标注格式：</P>
          <CodeBlock language="text">{`dataset/
    train/
        _annotations.coco.json
        image1.jpg
        image2.jpg
    valid/
        _annotations.coco.json
        image1.jpg`}</CodeBlock>

          <Divider />

          {/* ────────────── VALIDATION ────────────── */}
          <SectionHeading id="validation" icon={CheckCircle2}>验证</SectionHeading>
          <P>在验证集上运行 COCO 标准评估：</P>
          <CodeBlock language="python">{`results = model.val(
    data="coco128.yaml",   # dataset config
    batch=16,
    imgsz=640,
    conf=0.001,            # low conf for mAP calculation
    iou=0.6,               # NMS IoU threshold
    split="val",           # "val", "test", or "train"
    save_json=False,       # save predictions as COCO JSON
    verbose=True,          # print per-class metrics
)

print(f"mAP50:    {results['metrics/mAP50']:.3f}")
print(f"mAP50-95: {results['metrics/mAP50-95']:.3f}")`}</CodeBlock>

          <SubHeading>验证结果字典</SubHeading>
          <P>
            默认情况下，LibreYOLO 使用 COCO 评估并返回 12 个标准指标：
          </P>
          <CodeBlock language="python">{`{
    "metrics/mAP50-95": 0.489,   # COCO primary metric (AP@[.5:.95])
    "metrics/mAP50": 0.721,      # AP@0.5 (PASCAL VOC style)
    "metrics/mAP75": 0.534,      # AP@0.75 (strict)
    "metrics/mAP_small": 0.291,
    "metrics/mAP_medium": 0.532,
    "metrics/mAP_large": 0.648,
    "metrics/AR1": 0.362,        # Average Recall (max 1 det)
    "metrics/AR10": 0.571,
    "metrics/AR100": 0.601,
    "metrics/AR_small": 0.387,
    "metrics/AR_medium": 0.641,
    "metrics/AR_large": 0.739,
}`}</CodeBlock>
          <P>
            在 <InlineCode>ValidationConfig</InlineCode> 中设置 <InlineCode>use_coco_eval=False</InlineCode> 可获得旧版的 precision/recall 指标。
          </P>

          <Divider />

          {/* ────────────── EXPORT ────────────── */}
          <SectionHeading id="export" icon={Upload}>导出</SectionHeading>
          <P>将 PyTorch 模型导出为 ONNX、TorchScript、TensorRT、OpenVINO 或 NCNN 以便部署。</P>

          <SubHeading>快速导出</SubHeading>
          <CodeBlock language="python">{`# ONNX (default)
model.export()

# TorchScript
model.export(format="torchscript")

# TensorRT (requires NVIDIA GPU + TensorRT)
model.export(format="tensorrt")

# OpenVINO (optimized for Intel hardware)
model.export(format="openvino")

# NCNN (via PNNX)
model.export(format="ncnn")`}</CodeBlock>

          <SubHeading>全部导出参数</SubHeading>
          <CodeBlock language="python">{`path = model.export(
    format="onnx",            # "onnx", "torchscript", "tensorrt", "openvino", or "ncnn"
    output_path="model.onnx", # output file (auto-generated if None)
    imgsz=640,                # input resolution (default: model's native)
    opset=13,                 # ONNX opset version (RT-DETR / RF-DETR default to 17)
    simplify=True,            # run onnxsim graph simplification
    dynamic=True,             # enable dynamic batch axis
    half=False,               # export in FP16
    batch=1,                  # batch size for static graph
    device=None,              # device to trace on (default: model's current device)
    int8=False,               # INT8 quantization (TensorRT / OpenVINO only)
    data=None,                # calibration dataset for INT8
    fraction=1.0,             # fraction of calibration data to use
    workspace=4.0,            # TensorRT workspace size (GB)
    hardware_compatibility="none", # TensorRT compatibility mode
    gpu_device=0,             # GPU device index for TensorRT
    trt_config=None,          # optional TensorRT YAML config path
    verbose=False,            # verbose logging
)`}</CodeBlock>
          <P>
            OpenVINO INT8 导出还需要 <InlineCode>nncf</InlineCode>。NCNN 导出会写入一个目录，其中包含 <InlineCode>model.ncnn.param</InlineCode>、<InlineCode>model.ncnn.bin</InlineCode> 和 <InlineCode>metadata.yaml</InlineCode>。
          </P>

          <SubHeading>ONNX 元数据</SubHeading>
          <P>导出的 ONNX 文件包含嵌入的元数据：</P>
          <DocTable
            headers={['键', '示例值']}
            rows={[
              [<InlineCode key="v">libreyolo_version</InlineCode>, <InlineCode key="vv">&quot;1.0.0&quot;</InlineCode>],
              [<InlineCode key="f">model_family</InlineCode>, <InlineCode key="fv">&quot;yolox&quot;</InlineCode>],
              [<InlineCode key="s">model_size</InlineCode>, <InlineCode key="sv">&quot;s&quot;</InlineCode>],
              [<InlineCode key="c">nb_classes</InlineCode>, <InlineCode key="cv">&quot;80&quot;</InlineCode>],
              [<InlineCode key="n">names</InlineCode>, <span key="nv" className="text-xs"><InlineCode>{`'{"0": "person", "1": "bicycle", ...}'`}</InlineCode></span>],
              [<InlineCode key="i">imgsz</InlineCode>, <InlineCode key="iv">&quot;640&quot;</InlineCode>],
              [<InlineCode key="d">dynamic</InlineCode>, <InlineCode key="dv">&quot;True&quot;</InlineCode>],
              [<InlineCode key="h">half</InlineCode>, <InlineCode key="hv">&quot;False&quot;</InlineCode>],
            ]}
          />
          <P>
            使用 <InlineCode>OnnxBackend</InlineCode> 加载模型时，会自动读回这些元数据。
          </P>

          <SubHeading>直接使用导出器工厂</SubHeading>
          <CodeBlock language="python">{`from libreyolo.export import BaseExporter

exporter = BaseExporter.create("onnx", model)
path = exporter(dynamic=True, simplify=True)`}</CodeBlock>

          <Divider />

          {/* ────────────── ONNX INFERENCE ────────────── */}
          <SectionHeading id="onnx-inference" icon={Cpu}>ONNX 推理</SectionHeading>
          <P>
            使用 ONNX Runtime 而非 PyTorch 运行推理。适用于没有 PyTorch 的部署环境。
          </P>
          <CodeBlock language="python">{`from libreyolo import OnnxBackend

model = OnnxBackend("model.onnx")

result = model("image.jpg", conf=0.25, iou=0.45, save=True)
print(result.boxes.xyxy)`}</CodeBlock>

          <SubHeading>自动元数据</SubHeading>
          <P>
            如果 ONNX 文件由 LibreYOLO 导出，则会自动从嵌入的元数据中读取类别名称和类别数：
          </P>
          <CodeBlock language="python">{`# Export with metadata
model.export(format="onnx", output_path="model.onnx")

# Load — names and nb_classes auto-populated
onnx_model = OnnxBackend("model.onnx")
print(onnx_model.names)       # {0: "person", 1: "bicycle", ...}
print(onnx_model.nb_classes)  # 80`}</CodeBlock>

          <P>
            对于没有元数据的 ONNX 文件（例如由其他工具导出），请手动指定 <InlineCode>nb_classes</InlineCode>：
          </P>
          <CodeBlock language="python">{`model = OnnxBackend("external_model.onnx", nb_classes=20)`}</CodeBlock>

          <SubHeading>设备选择</SubHeading>
          <CodeBlock language="python">{`# Auto-detect (CUDA if available, else CPU)
model = OnnxBackend("model.onnx", device="auto")

# Force CPU
model = OnnxBackend("model.onnx", device="cpu")

# Force CUDA
model = OnnxBackend("model.onnx", device="cuda")`}</CodeBlock>

          <SubHeading>预测参数</SubHeading>
          <P>
            <InlineCode>OnnxBackend</InlineCode> 支持各运行时后端共享的核心预测 API：
          </P>
          <CodeBlock language="python">{`result = model(
    "image.jpg",
    conf=0.25,
    iou=0.45,
    imgsz=640,
    classes=[0, 2],
    max_det=300,
    save=True,
    output_path="output/annotated.jpg",  # final file path when save=True
    color_format="auto",
)`}</CodeBlock>
          <P>
            运行时后端不暴露仅限 PyTorch 的选项，例如 <InlineCode>tiling</InlineCode>、<InlineCode>overlap_ratio</InlineCode> 或 <InlineCode>output_file_format</InlineCode>。
          </P>
          <P>
            运行时后端处理保存的方式也与 PyTorch 包装器略有不同：如果设置 <InlineCode>output_path</InlineCode>，请传入最终文件路径，而非目录。如果省略它，当前后端默认保存位置在 <InlineCode>runs/detections/</InlineCode> 下。
          </P>

          <Divider />

          {/* ────────────── TENSORRT INFERENCE ────────────── */}
          <SectionHeading id="tensorrt-inference" icon={Cpu}>TensorRT 推理</SectionHeading>
          <P>
            使用 TensorRT 在 NVIDIA GPU 上以最大吞吐量运行推理。需要 CUDA 以及 TensorRT 的 Python 绑定。
          </P>
          <CodeBlock language="python">{`from libreyolo import TensorRTBackend

model = TensorRTBackend("model.engine")

result = model("image.jpg", conf=0.25, iou=0.45, save=True)
print(result.boxes.xyxy)`}</CodeBlock>

          <SubHeading>通过工厂自动检测</SubHeading>
          <P>
            <InlineCode>LibreYOLO()</InlineCode> 工厂函数会自动检测 <InlineCode>.engine</InlineCode> 文件：
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

# Auto-detects TensorRT engine
model = LibreYOLO("model.engine")`}</CodeBlock>

          <P>
            <InlineCode>TensorRTBackend</InlineCode> 支持与 ONNX 和 OpenVINO 相同的核心运行时后端预测 API，包括 <InlineCode>save=True</InlineCode> 时同样的仅文件路径 <InlineCode>output_path</InlineCode> 行为。
          </P>

          <Divider />

          {/* ────────────── OPENVINO INFERENCE ────────────── */}
          <SectionHeading id="openvino-inference" icon={Cpu}>OpenVINO 推理</SectionHeading>
          <P>
            使用 OpenVINO 运行推理，针对 Intel CPU、GPU 和 VPU 进行了优化。
          </P>
          <CodeBlock language="python">{`from libreyolo import OpenVINOBackend

model = OpenVINOBackend("model_openvino/")

result = model("image.jpg", conf=0.25, iou=0.45, save=True)
print(result.boxes.xyxy)`}</CodeBlock>

          <SubHeading>通过工厂自动检测</SubHeading>
          <P>
            <InlineCode>LibreYOLO()</InlineCode> 工厂函数会自动检测 OpenVINO 模型目录：
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

# Auto-detects OpenVINO directory
model = LibreYOLO("model_openvino/")`}</CodeBlock>

          <P>
            <InlineCode>OpenVINOBackend</InlineCode> 在存在 <InlineCode>metadata.yaml</InlineCode> 时会读取它，并支持相同的核心运行时后端预测 API。
          </P>

          <Divider />

          {/* ────────────── NCNN INFERENCE ────────────── */}
          <SectionHeading id="ncnn-inference" icon={Cpu}>NCNN 推理</SectionHeading>
          <P>
            使用 NCNN 在 CPU 或支持 Vulkan 的 GPU 目标上进行轻量级部署推理。
          </P>
          <CodeBlock language="python">{`from libreyolo import NcnnBackend

model = NcnnBackend("model_ncnn/")

result = model("image.jpg", conf=0.25, iou=0.45, save=True)
print(result.boxes.xyxy)`}</CodeBlock>

          <SubHeading>通过工厂自动检测</SubHeading>
          <P>
            <InlineCode>LibreYOLO()</InlineCode> 工厂函数会自动检测 NCNN 模型目录：
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

# Auto-detects NCNN directory
model = LibreYOLO("model_ncnn/")`}</CodeBlock>

          <P>
            NCNN 导出目录包含 <InlineCode>model.ncnn.param</InlineCode>、<InlineCode>model.ncnn.bin</InlineCode>，通常还有 <InlineCode>metadata.yaml</InlineCode>。
          </P>

          <Divider />

          {/* ────────────── API REFERENCE ────────────── */}
          <SectionHeading id="api-reference" icon={FileCode}>API 参考</SectionHeading>

          <SubHeading>LibreYOLO（工厂）</SubHeading>
          <CodeBlock language="python">{`LibreYOLO(
    model_path: str,
    size: str = None,           # auto-detected from weights
    reg_max: int = 16,          # YOLO9 only
    nb_classes: int = None,     # auto-detected from weights
    device: str = "auto",
) -> LibreYOLOX | LibreYOLO9 | LibreRTDETR | LibreRFDETR | OnnxBackend | TensorRTBackend | OpenVINOBackend | NcnnBackend`}</CodeBlock>
          <P>
            从权重文件自动检测模型架构、尺寸和类别数。它还可处理 <InlineCode>.onnx</InlineCode>、<InlineCode>.engine</InlineCode>、包含 <InlineCode>model.xml</InlineCode> 的 OpenVINO 目录，以及包含 <InlineCode>model.ncnn.param</InlineCode> 和 <InlineCode>model.ncnn.bin</InlineCode> 的 NCNN 目录。
          </P>

          <SubHeading>预测（PyTorch 模型包装器）</SubHeading>
          <CodeBlock language="python">{`model(
    source,                     # image input (see supported formats)
    *,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = None,
    classes: list[int] = None,
    max_det: int = 300,
    save: bool = False,
    batch: int = 1,
    output_path: str = None,
    color_format: str = "auto",
    tiling: bool = False,
    overlap_ratio: float = 0.2,
    output_file_format: str = None,
) -> Results | list[Results]`}</CodeBlock>

          <SubHeading>预测（运行时后端）</SubHeading>
          <CodeBlock language="python">{`backend(
    source,
    *,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = None,
    classes: list[int] = None,
    max_det: int = 300,
    save: bool = False,
    batch: int = 1,
    output_path: str = None,    # final file path when save=True
    color_format: str = "auto",
) -> Results | list[Results]`}</CodeBlock>
          <P>
            如果运行时后端省略 <InlineCode>output_path</InlineCode>，当前默认保存位置为 <InlineCode>runs/detections/</InlineCode>。
          </P>

          <SubHeading>Results</SubHeading>
          <CodeBlock language="python">{`result = Results(
    boxes: Boxes,
    orig_shape: tuple[int, int],  # (height, width)
    path: str | None,
    names: dict[int, str],
)

len(result)          # number of detections
result.cpu()         # copy with tensors on CPU`}</CodeBlock>

          <SubHeading>Boxes</SubHeading>
          <CodeBlock language="python">{`boxes = Boxes(boxes, conf, cls)

boxes.xyxy           # (N, 4) tensor — x1, y1, x2, y2
boxes.xywh           # (N, 4) tensor — cx, cy, w, h
boxes.conf           # (N,) tensor — confidence scores
boxes.cls            # (N,) tensor — class IDs
boxes.data           # (N, 6) tensor — [xyxy, conf, cls]

len(boxes)           # number of boxes
boxes.cpu()          # copy on CPU
boxes.numpy()        # copy as numpy arrays`}</CodeBlock>

          <SubHeading>model.export()</SubHeading>
          <CodeBlock language="python">{`model.export(
    format: str = "onnx",       # "onnx", "torchscript", "tensorrt", "openvino", or "ncnn"
    *,
    output_path: str = None,
    imgsz: int = None,
    opset: int = 13,
    simplify: bool = True,
    dynamic: bool = True,
    half: bool = False,
    batch: int = 1,
    device: str = None,
    int8: bool = False,
    data: str = None,           # calibration data for INT8
    fraction: float = 1.0,      # fraction of calibration data
    workspace: float = 4.0,     # TensorRT workspace (GB)
    hardware_compatibility: str = "none",
    gpu_device: int = 0,
    trt_config = None,          # optional TensorRT YAML config path
    verbose: bool = False,
) -> str                        # path to exported file or directory`}</CodeBlock>

          <SubHeading>BaseExporter</SubHeading>
          <CodeBlock language="python">{`from libreyolo.export import BaseExporter

exporter = BaseExporter.create("onnx", model)
path = exporter(dynamic=True, simplify=True)

BaseExporter.create("ncnn", model)(output_path="model_ncnn")`}</CodeBlock>

          <SubHeading>model.val()</SubHeading>
          <CodeBlock language="python">{`model.val(
    data: str = None,           # path to data.yaml
    batch: int = 16,
    imgsz: int = None,
    conf: float = 0.001,
    iou: float = 0.6,
    device: str = None,
    split: str = "val",         # "val", "test", or "train"
    save_json: bool = False,
    verbose: bool = True,
) -> dict`}</CodeBlock>
          <P>返回值（COCO 评估，默认）：</P>
          <CodeBlock language="python">{`{
    "metrics/mAP50-95": float,   # COCO primary metric
    "metrics/mAP50": float,
    "metrics/mAP75": float,
    "metrics/mAP_small": float,
    "metrics/mAP_medium": float,
    "metrics/mAP_large": float,
    "metrics/AR1": float,
    "metrics/AR10": float,
    "metrics/AR100": float,
    "metrics/AR_small": float,
    "metrics/AR_medium": float,
    "metrics/AR_large": float,
}`}</CodeBlock>

          <SubHeading>model.train()（YOLOX）</SubHeading>
          <CodeBlock language="python">{`model.train(
    data: str,                  # path to data.yaml (required)
    *,
    epochs: int = 100,
    batch: int = 16,
    imgsz: int = 640,
    lr0: float = 0.01,
    optimizer: str = "SGD",
    device: str = "",
    workers: int = 8,
    seed: int = 0,
    project: str = "runs/train",
    name: str = "exp",
    exist_ok: bool = False,
    pretrained: bool = True,
    resume: bool = False,
    amp: bool = True,
    patience: int = 50,
) -> dict`}</CodeBlock>
          <P>返回值：</P>
          <CodeBlock language="python">{`{
    "final_loss": float,
    "best_mAP50": float,
    "best_mAP50_95": float,
    "best_epoch": int,
    "save_dir": str,
    "best_checkpoint": str,
    "last_checkpoint": str,
}`}</CodeBlock>

          <SubHeading>model.train()（YOLO9）</SubHeading>
          <CodeBlock language="python">{`model.train(
    data: str,                  # path to data.yaml (required)
    *,
    epochs: int = 300,
    batch: int = 16,
    imgsz: int = 640,
    lr0: float = 0.01,
    optimizer: str = "SGD",
    device: str = "",
    workers: int = 8,
    seed: int = 0,
    project: str = "runs/train",
    name: str = "yolo9_exp",
    exist_ok: bool = False,
    resume: bool = False,
    amp: bool = True,
    patience: int = 50,
) -> dict`}</CodeBlock>
          <P>返回与 YOLOX 训练相同的字典。</P>

          <SubHeading>model.train()（RT-DETR）</SubHeading>
          <CodeBlock language="python">{`model.train(
    data: str,                  # path to data.yaml (required)
    *,
    epochs: int = 72,
    batch: int = 4,
    imgsz: int = 640,
    lr0: float = 1e-4,
    lr_backbone: float = 1e-5,
    optimizer: str = "AdamW",
    scheduler: str = "linear",
    device: str = "",
    workers: int = 4,
    seed: int = 0,
    project: str = "runs/train",
    name: str = "rtdetr_exp",
    exist_ok: bool = False,
    pretrained: bool = True,
    resume: bool = False,
    amp: bool = True,
    patience: int = 50,
) -> dict`}</CodeBlock>

          <SubHeading>model.train()（RF-DETR）</SubHeading>
          <CodeBlock language="python">{`model.train(
    data: str,                  # path to dataset directory
    epochs: int = 100,
    batch_size: int = 4,
    lr: float = 1e-4,
    output_dir: str = "runs/train",
    resume: str = None,
    **kwargs,                   # additional RF-DETR training args
) -> dict`}</CodeBlock>

          <SubHeading>OnnxBackend</SubHeading>
          <CodeBlock language="python">{`OnnxBackend(
    onnx_path: str,
    nb_classes: int = 80,       # auto-read from metadata if available
    device: str = "auto",
)`}</CodeBlock>
          <P>
            使用 ONNX Runtime 对 ONNX 模型运行推理。支持上面所示的运行时后端预测 API。
          </P>

          <SubHeading>TensorRTBackend</SubHeading>
          <CodeBlock language="python">{`TensorRTBackend(
    engine_path: str,
    nb_classes: int | None = None,
    device: str = "auto",
)`}</CodeBlock>
          <P>
            对 TensorRT <InlineCode>.engine</InlineCode> 文件运行推理，并可从相邻的 <InlineCode>.json</InlineCode> 附属文件读取元数据。
          </P>

          <SubHeading>OpenVINOBackend</SubHeading>
          <CodeBlock language="python">{`OpenVINOBackend(
    model_dir: str,
    nb_classes: int | None = None,
    device: str = "auto",
)`}</CodeBlock>
          <P>
            对包含 <InlineCode>model.xml</InlineCode> 以及可选 <InlineCode>metadata.yaml</InlineCode> 的 OpenVINO 模型目录运行推理。
          </P>

          <SubHeading>NcnnBackend</SubHeading>
          <CodeBlock language="python">{`NcnnBackend(
    model_dir: str,
    nb_classes: int | None = None,
    device: str = "auto",
)`}</CodeBlock>
          <P>
            对包含 <InlineCode>model.ncnn.param</InlineCode>、<InlineCode>model.ncnn.bin</InlineCode> 以及可选 <InlineCode>metadata.yaml</InlineCode> 的 NCNN 模型目录运行推理。
          </P>

          <SubHeading>ValidationConfig</SubHeading>
          <CodeBlock language="python">{`from libreyolo import ValidationConfig

config = ValidationConfig(
    data="coco128.yaml",
    data_dir=None,             # override dataset root directory
    batch_size=16,
    imgsz=640,
    conf_thres=0.001,
    iou_thres=0.6,
    max_det=300,
    split="val",               # "val", "test", or "train"
    device="auto",
    save_json=False,
    verbose=True,
    half=False,
    use_coco_eval=True,        # use COCO eval (12 keys); False for legacy
    num_workers=4,
)

# Load/save YAML
config = ValidationConfig.from_yaml("config.yaml")
config.to_yaml("config.yaml")`}</CodeBlock>

          <Divider />

          {/* ────────────── ARCHITECTURE GUIDE ────────────── */}
          <SectionHeading id="architecture" icon={Wrench}>架构指南</SectionHeading>
          <P>
            本节面向希望了解代码库内部实现的贡献者。
          </P>

          <SubHeading>基类设计</SubHeading>
          <P>
            PyTorch 模型系列继承自 <InlineCode>libreyolo/models/base/model.py</InlineCode> 中的 <InlineCode>BaseModel</InlineCode>。子类需实现以下抽象方法：
          </P>
          <DocTable
            headers={['方法', '作用']}
            rows={[
              [<InlineCode key="init">_init_model()</InlineCode>, '构建并返回 nn.Module'],
              [<InlineCode key="layers">_get_available_layers()</InlineCode>, '返回层名到模块的映射'],
              [<InlineCode key="pre-np">_get_preprocess_numpy()</InlineCode>, '返回用于导出/校准的 NumPy 预处理器'],
              [<InlineCode key="pre">_preprocess()</InlineCode>, '图像到张量的转换'],
              [<InlineCode key="fwd">_forward()</InlineCode>, '模型前向传播'],
              [<InlineCode key="post">_postprocess()</InlineCode>, '原始输出到检测字典'],
            ]}
          />
          <P>
            <InlineCode>BaseModel</InlineCode> 提供共享的包装行为：预测、导出、验证、尺寸/名称元数据以及训练辅助函数。实际的单图、批量和分块推理流程位于 <InlineCode>libreyolo/models/base/inference.py</InlineCode>，而部署运行时位于 <InlineCode>libreyolo/backends/</InlineCode> 下。
          </P>

          <SubHeading>包结构</SubHeading>
          <CodeBlock language="text">{`libreyolo/
    __init__.py          # Public API exports
    models/
        __init__.py      # LibreYOLO() factory + model registry bootstrap
        base/
            model.py     # BaseModel
            inference.py # Shared prediction pipeline
        yolox/
            model.py
            nn.py
            utils.py
        yolo9/
            model.py
            nn.py
            utils.py
        rtdetr/
            model.py
            nn.py
            trainer.py
            utils.py
        rfdetr/
            model.py
            utils.py
            train.py
    backends/
        base.py          # BaseBackend runtime wrapper
        onnx.py          # OnnxBackend
        tensorrt.py      # TensorRTBackend
        openvino.py      # OpenVINOBackend
        ncnn.py          # NcnnBackend
    utils/
        results.py       # Results and Boxes classes
        image_loader.py  # Unified image loading
        general.py       # Path helpers, NMS, tiling utilities
    export/
        exporter.py      # BaseExporter and format registry
        onnx.py          # ONNX export logic
        torchscript.py   # TorchScript export logic
        tensorrt.py      # TensorRT export logic
        openvino.py      # OpenVINO export logic
        ncnn.py          # NCNN export logic
    training/
        config.py        # YOLOXTrainConfig / YOLOv9TrainConfig
        trainer.py       # YOLOXTrainer
        v9_trainer.py    # YOLOv9Trainer
        dataset.py       # Training dataset
        augment.py       # Mosaic, mixup, etc.
        loss.py          # YOLOX loss functions
        scheduler.py     # LR schedulers
        ema.py           # Exponential moving average
    validation/
        config.py        # ValidationConfig
        detection_validator.py  # DetectionValidator
        metrics.py       # DetMetrics, mAP computation
        base.py          # BaseValidator
        preprocessors.py # Per-model val preprocessing
    data/
        utils.py         # Dataset loading, YAML parsing
        yolo_coco_api.py # YOLO-to-COCO annotation bridge
    config/
        datasets/        # Built-in dataset YAML configs (coco8, coco128, coco5000, coco, etc.)`}</CodeBlock>

          <SubHeading>添加新的模型系列</SubHeading>
          <ol className="space-y-2.5 mb-4 list-none">
            {[
              <>在 <InlineCode>libreyolo/models/newmodel/model.py</InlineCode> 中创建一个继承 <InlineCode>BaseModel</InlineCode> 的类</>,
              '实现所有抽象方法',
              <>在 <InlineCode>libreyolo/models/newmodel/</InlineCode> 下创建配套的网络和工具</>,
              <>将导入添加到 <InlineCode>libreyolo/models/__init__.py</InlineCode>，以便注册表能识别它</>,
              <>从 <InlineCode>libreyolo/__init__.py</InlineCode> 导出该类</>,
              <>（可选）如果验证预处理与标准流程不同，可重写 <InlineCode>val_preprocessor_class</InlineCode></>,
            ].map((item, i) => (
              <li key={i} className="flex items-start gap-3 text-surface-600 dark:text-surface-400">
                <span className="w-6 h-6 rounded-lg bg-libre-500/10 border border-libre-500/20 text-libre-400 text-xs font-bold flex items-center justify-center shrink-0 mt-0.5">
                  {i + 1}
                </span>
                <span>{item}</span>
              </li>
            ))}
          </ol>

          <SubHeading>导出架构</SubHeading>
          <P>
            <InlineCode>libreyolo/export/exporter.py</InlineCode> 中的 <InlineCode>BaseExporter</InlineCode> 是导出入口。具体的导出器通过子类注册机制自行注册，调用方使用 <InlineCode>BaseExporter.create(format, model)</InlineCode> 获取正确的实现：
          </P>
          <CodeBlock language="python">{`from libreyolo.export import BaseExporter

onnx_exporter = BaseExporter.create("onnx", model)
ncnn_exporter = BaseExporter.create("ncnn", model)`}</CodeBlock>
          <P>
            要添加新的导出格式，请实现一个具有唯一 <InlineCode>format_name</InlineCode> 的新 <InlineCode>BaseExporter</InlineCode> 子类，并从 <InlineCode>libreyolo/export/exporter.py</InlineCode> 导入它，以填充注册表。
          </P>

          <Divider />

          {/* ────────────── DATASET FORMAT ────────────── */}
          <SectionHeading id="dataset-format" icon={Database}>数据集格式</SectionHeading>
          <P>
            YOLO 风格的模型使用通过 <InlineCode>data.yaml</InlineCode> 配置的数据集。RF-DETR 使用 COCO 格式的标注，将在下文单独说明。
          </P>

          <SubHeading>data.yaml 结构</SubHeading>
          <CodeBlock language="yaml" filename="data.yaml">{`path: /absolute/path/to/dataset   # dataset root
train: images/train               # directory path, relative to path
val: images/val                   # directory path, relative to path
test: images/test                 # optional

nc: 80                            # number of classes
names: [                          # class names
  "person", "bicycle", "car", "motorcycle", "airplane",
  "bus", "train", "truck", "boat", "traffic light",
  # ...
]`}</CodeBlock>

          <SubHeading>文件列表变体</SubHeading>
          <P>
            相同的 YAML 格式也可以让 <InlineCode>train</InlineCode>、<InlineCode>val</InlineCode> 或 <InlineCode>test</InlineCode> 指向每行包含一个图像路径的 <InlineCode>.txt</InlineCode> 文件：
          </P>
          <CodeBlock language="yaml" filename="coco.yaml">{`path: /absolute/path/to/coco
train: train2017.txt
val: val2017.txt
test: test-dev2017.txt

nc: 80
names: ["person", "bicycle", "car", "..."]`}</CodeBlock>

          <SubHeading>目录布局</SubHeading>
          <CodeBlock language="text">{`dataset/
    images/
        train/
            img001.jpg
            img002.jpg
        val/
            img003.jpg
    labels/
        train/
            img001.txt
            img002.txt
        val/
            img003.txt`}</CodeBlock>

          <SubHeading>标签格式</SubHeading>
          <P>
            每张图像对应一个文本文件。每一行表示一个目标：
          </P>
          <CodeBlock language="text">{`<class_id> <center_x> <center_y> <width> <height>`}</CodeBlock>
          <P>
            所有坐标都相对于图像尺寸归一化到 [0, 1]。
          </P>
          <P>示例（<InlineCode>img001.txt</InlineCode>）：</P>
          <CodeBlock language="text" filename="img001.txt">{`0 0.5 0.4 0.3 0.6
2 0.1 0.2 0.05 0.1`}</CodeBlock>

          <SubHeading>内置数据集</SubHeading>
          <P>
            LibreYOLO 在 <InlineCode>libreyolo/config/datasets/</InlineCode> 下附带内置数据集配置，并可在首次使用时自动下载受支持的数据集：
          </P>
          <CodeBlock language="python">{`# These download automatically on first use
results = model.val(data="coco8.yaml")
results = model.train(data="coco128.yaml", epochs=10)`}</CodeBlock>

          <SubHeading>RF-DETR 数据集格式</SubHeading>
          <P>
            RF-DETR 使用 COCO 格式的标注（JSON），而非 YOLO 文本标签：
          </P>
          <CodeBlock language="text">{`dataset/
    train/
        _annotations.coco.json
        image1.jpg
    valid/
        _annotations.coco.json
        image1.jpg`}</CodeBlock>

          {/* Bottom spacer */}
          <div className="h-16" />
        </div>
      </main>
    </div>
  )
}


export default function Docs() {
  const locale = useLocale()
  if (locale === 'zh') return <DocsPageZh version="v1.1.0" isLatest={false} />
  return <DocsPage version="v1.1.0" isLatest={false} />
}
