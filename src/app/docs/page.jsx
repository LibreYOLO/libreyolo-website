'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  BookOpen, Terminal, Rocket, Layers, Crosshair, Grid3x3,
  GraduationCap, CheckCircle2, Upload, Cpu, FileCode, Wrench,
  Database, Copy, Check, Menu, X, ChevronRight,
  Sparkles, Tags, Video, Activity, Scissors, PersonStanding, Eye, SquareTerminal,
} from 'lucide-react'

/* ─── Section metadata for sidebar ─── */
const sections = [
  { id: 'introduction', title: 'Introduction', icon: BookOpen },
  { id: 'installation', title: 'Installation', icon: Terminal },
  { id: 'quickstart', title: 'Quickstart', icon: Rocket },
  { id: 'models', title: 'Available Models', icon: Layers },
  { id: 'tasks', title: 'Tasks & Filenames', icon: Tags },
  { id: 'prediction', title: 'Prediction', icon: Crosshair },
  { id: 'tiled-inference', title: 'Tiled Inference', icon: Grid3x3 },
  { id: 'video-inference', title: 'Video Inference', icon: Video },
  { id: 'tracking', title: 'Tracking', icon: Activity },
  { id: 'segmentation', title: 'Segmentation', icon: Scissors },
  { id: 'pose', title: 'Pose Estimation', icon: PersonStanding },
  { id: 'gaze', title: 'Gaze Estimation', icon: Eye },
  { id: 'training', title: 'Training', icon: GraduationCap },
  { id: 'validation', title: 'Validation', icon: CheckCircle2 },
  { id: 'export', title: 'Export', icon: Upload },
  { id: 'onnx-inference', title: 'ONNX Inference', icon: Cpu },
  { id: 'tensorrt-inference', title: 'TensorRT Inference', icon: Cpu },
  { id: 'openvino-inference', title: 'OpenVINO Inference', icon: Cpu },
  { id: 'ncnn-inference', title: 'NCNN Inference', icon: Cpu },
  { id: 'cli', title: 'CLI', icon: SquareTerminal },
  { id: 'api-reference', title: 'API Reference', icon: FileCode },
  { id: 'architecture', title: 'Architecture Guide', icon: Wrench },
  { id: 'dataset-format', title: 'Dataset Format', icon: Database },
]

const docsVersions = [
  { version: 'v1.2.0', label: 'Dev branch', href: '/docs' },
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

function HFLink({ name }) {
  const stem = name.replace(/\.(pt|pth|onnx|engine|torchscript)$/, '')
  return (
    <a
      href={`https://huggingface.co/LibreYOLO/${stem}`}
      target="_blank"
      rel="noopener noreferrer"
      className="font-mono text-xs text-libre-600 dark:text-libre-400 hover:underline whitespace-nowrap"
    >
      {name}
    </a>
  )
}

function Checkpoints({ names, link = true }) {
  return (
    <span className="leading-relaxed">
      {names.map((n, i) => (
        <span key={n}>
          {i > 0 && ', '}
          {link
            ? <HFLink name={n} />
            : <span className="font-mono text-xs text-surface-700 dark:text-surface-300 whitespace-nowrap">{n}</span>}
        </span>
      ))}
    </span>
  )
}

function FlagshipCallout({ className = '' }) {
  return (
    <div className={`my-6 rounded-xl border border-libre-500/30 bg-libre-500/5 dark:bg-libre-500/10 p-4 ${className}`}>
      <div className="flex items-start gap-3">
        <Sparkles className="w-5 h-5 text-libre-600 dark:text-libre-400 mt-0.5 shrink-0" />
        <div>
          <p className="font-semibold text-surface-900 dark:text-white mb-1">
            Recommended for new projects: YOLO9 (CNN) or RF-DETR (transformer)
          </p>
          <p className="text-sm text-surface-600 dark:text-surface-400">
            Heavily tested across every supported export format and runtime backend. Broadest deployment coverage. Best accuracy / speed balance.
          </p>
        </div>
      </div>
    </div>
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

  // Scroll spy - pick the last section whose heading has scrolled past 30% of viewport
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
                <div className="flex flex-wrap items-center gap-2">
                  {docsVersions.map(({ version: itemVersion, label, href }) => {
                    const isCurrent = itemVersion === version
                    return (
                      <a
                        key={itemVersion}
                        href={href}
                        className={`rounded-md px-2.5 py-1 text-sm font-semibold transition-colors ${
                          isCurrent
                            ? 'bg-libre-500/10 text-libre-700 dark:text-libre-300'
                            : 'text-surface-500 dark:text-surface-400 hover:bg-surface-100 dark:hover:bg-white/[0.06] hover:text-surface-900 dark:hover:text-white'
                        }`}
                      >
                        {itemVersion} {label.toLowerCase()}
                      </a>
                    )
                  })}
                </div>
                <p className="mt-2 text-sm text-surface-600 dark:text-surface-400">
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
            <FlagshipCallout />
            <P>
              LibreYOLO is an MIT-licensed object detection toolkit. We focus development, testing, and benchmarking on two flagship model families:
            </P>
            <ul className="space-y-2 mb-4">
              <FeatureItem><strong className="text-surface-800 dark:text-white">YOLO9</strong> - the CNN flagship.</FeatureItem>
              <FeatureItem><strong className="text-surface-800 dark:text-white">RF-DETR</strong> - the transformer flagship.</FeatureItem>
            </ul>
            <P>
              Both are heavily tested across every supported export format (ONNX, TorchScript, TensorRT, OpenVINO, NCNN) and runtime backend we ship. <strong className="text-surface-800 dark:text-white">We recommend them as the default choice for new projects</strong> - they offer the best balance of accuracy, inference speed, and deployment maturity. Other supported families work the same way through the unified <InlineCode>LibreYOLO()</InlineCode> factory; reach for them when you have a specific reason (existing checkpoint, hardware constraint, paper reproduction).
            </P>
            <CodeBlock language="python">{`from libreyolo import LibreYOLO, SAMPLE_IMAGE

# YOLO9 - the CNN flagship
model = LibreYOLO("LibreYOLO9c.pt")
result = model(SAMPLE_IMAGE, conf=0.25, save=True)

print(f"Detected {len(result)} objects")
print(result.boxes.xyxy)
print(result.saved_path)`}</CodeBlock>

            <SubHeading>Key features</SubHeading>
            <ul className="space-y-2.5 mb-4">
              <FeatureItem>Heavy testing and recommended defaults around the YOLO9 and RF-DETR flagships</FeatureItem>
              <FeatureItem>Unified <InlineCode>LibreYOLO()</InlineCode> factory auto-detects family, size, classes, and task from the weights file</FeatureItem>
              <FeatureItem>Detection, segmentation, pose, and gaze tasks through one consistent API</FeatureItem>
              <FeatureItem>Image, directory, and video inference (with optional tiled inference for large frames)</FeatureItem>
              <FeatureItem>Built-in multi-object tracking via ByteTrack</FeatureItem>
              <FeatureItem>ONNX, TorchScript, TensorRT, OpenVINO, and NCNN export with embedded metadata, plus matching runtime backends</FeatureItem>
              <FeatureItem>COCO-compatible validation with mAP metrics, plus segmentation and pose validators</FeatureItem>
              <FeatureItem>Ultralytics-style <InlineCode>libreyolo</InlineCode> command-line tool for predict / train / val / export</FeatureItem>
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
          <P>
            Pick a flagship. Both load through the same factory, accept the same inputs, and return the same <InlineCode>Results</InlineCode> object - so you can swap between them without changing surrounding code.
          </P>

          <SubHeading>YOLO9 - CNN flagship</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO, SAMPLE_IMAGE

# Auto-detects family, size, classes, and task from the weights file
model = LibreYOLO("LibreYOLO9c.pt")

# Run on a single image (SAMPLE_IMAGE ships with the package)
result = model(SAMPLE_IMAGE)

print(f"Found {len(result)} objects")
print(result.boxes.xyxy)   # bounding boxes (N, 4)
print(result.boxes.conf)   # confidence scores (N,)
print(result.boxes.cls)    # class IDs (N,)`}</CodeBlock>

          <SubHeading>RF-DETR - transformer flagship</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO, SAMPLE_IMAGE

# Same factory, same call shape - just point at an RF-DETR checkpoint
model = LibreYOLO("LibreRFDETRs.pt")
result = model(SAMPLE_IMAGE)

print(f"Found {len(result)} objects")
print(result.boxes.xyxy)`}</CodeBlock>

          <SubHeading>Save annotated output</SubHeading>
          <CodeBlock language="python">{`result = model(SAMPLE_IMAGE, save=True)
print(result.saved_path)   # e.g. runs/detect/predict/parkour.jpg`}</CodeBlock>

          <SubHeading>Process a directory</SubHeading>
          <CodeBlock language="python">{`results = model("images/", save=True, batch=4)
for r in results:
    print(f"{r.path}: {len(r)} detections")`}</CodeBlock>

          <Divider />

          {/* ────────────── AVAILABLE MODELS ────────────── */}
          <SectionHeading id="models" icon={Layers}>Available Models</SectionHeading>
          <FlagshipCallout />
          <P>
            LibreYOLO ships two flagship families plus a broader catalogue of supported detection models. Every model loads through the same <InlineCode>LibreYOLO()</InlineCode> factory - pick by checkpoint name, the rest is auto-detected.
          </P>

          <SubHeading>YOLO9 - CNN flagship</SubHeading>
          <DocTable
            headers={['Size', 'Code', 'Input size', 'Use case', 'Detection checkpoint']}
            rows={[
              ['Tiny', <InlineCode key="t">&quot;t&quot;</InlineCode>, '640', 'Fast inference', <HFLink key="cp-t" name="LibreYOLO9t.pt" />],
              ['Small', <InlineCode key="s">&quot;s&quot;</InlineCode>, '640', 'Balanced', <HFLink key="cp-s" name="LibreYOLO9s.pt" />],
              ['Medium', <InlineCode key="m">&quot;m&quot;</InlineCode>, '640', 'Higher accuracy', <HFLink key="cp-m" name="LibreYOLO9m.pt" />],
              ['Compact', <InlineCode key="c">&quot;c&quot;</InlineCode>, '640', 'Best accuracy', <HFLink key="cp-c" name="LibreYOLO9c.pt" />],
            ]}
          />
          <P>
            <strong className="text-surface-800 dark:text-white">Segmentation checkpoints:</strong>{' '}
            <Checkpoints names={['LibreYOLO9t-seg.pt', 'LibreYOLO9s-seg.pt', 'LibreYOLO9m-seg.pt', 'LibreYOLO9c-seg.pt']} />
            . See the <a href="#segmentation" className="text-libre-600 dark:text-libre-400 hover:underline">Segmentation</a> section.
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")
# Segmentation variants share the same factory call
# model = LibreYOLO("LibreYOLO9c-seg.pt")`}</CodeBlock>

          <SubHeading>RF-DETR - transformer flagship</SubHeading>
          <DocTable
            headers={['Size', 'Code', 'Input size', 'Use case', 'Detection checkpoint']}
            rows={[
              ['Nano', <InlineCode key="n">&quot;n&quot;</InlineCode>, '384', 'Edge', <HFLink key="cp-n" name="LibreRFDETRn.pt" />],
              ['Small', <InlineCode key="s">&quot;s&quot;</InlineCode>, '512', 'Balanced', <HFLink key="cp-s" name="LibreRFDETRs.pt" />],
              ['Medium', <InlineCode key="m">&quot;m&quot;</InlineCode>, '576', 'Higher accuracy', <HFLink key="cp-m" name="LibreRFDETRm.pt" />],
              ['Large', <InlineCode key="l">&quot;l&quot;</InlineCode>, '704', 'Maximum accuracy', <HFLink key="cp-l" name="LibreRFDETRl.pt" />],
            ]}
          />
          <P>
            <strong className="text-surface-800 dark:text-white">Segmentation checkpoints:</strong>{' '}
            <Checkpoints names={['LibreRFDETRn-seg.pt', 'LibreRFDETRs-seg.pt', 'LibreRFDETRm-seg.pt', 'LibreRFDETRl-seg.pt', 'LibreRFDETRx-seg.pt', 'LibreRFDETRxx-seg.pt']} />
            . See the <a href="#segmentation" className="text-libre-600 dark:text-libre-400 hover:underline">Segmentation</a> section.
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreRFDETRs.pt")
# Segmentation variants exist for every RF-DETR size
# model = LibreYOLO("LibreRFDETRs-seg.pt")`}</CodeBlock>

          <SubHeading>Additional supported families</SubHeading>
          <P>
            Detection-capable families that share the same factory and API surface as the flagships. Each checkpoint name links to its Hugging Face model card on the <a href="https://huggingface.co/LibreYOLO" target="_blank" rel="noopener noreferrer" className="text-libre-600 dark:text-libre-400 hover:underline">LibreYOLO org</a>; pass any name to <InlineCode>LibreYOLO()</InlineCode> and the factory will fetch it on first use.
          </P>
          <DocTable
            headers={['Family', 'Tasks', 'Checkpoints']}
            rows={[
              ['YOLOX', 'detect', <Checkpoints key="yolox" names={['LibreYOLOXn.pt', 'LibreYOLOXt.pt', 'LibreYOLOXs.pt', 'LibreYOLOXm.pt', 'LibreYOLOXl.pt', 'LibreYOLOXx.pt']} />],
              ['YOLO9-E2E', 'detect', <Checkpoints key="y9e2e" names={['LibreYOLO9E2Et.pt', 'LibreYOLO9E2Es.pt', 'LibreYOLO9E2Em.pt', 'LibreYOLO9E2Ec.pt']} />],
              ['YOLO-NAS', 'detect, pose', <Checkpoints key="ynas" link={false} names={['LibreYOLONASs.pt', 'LibreYOLONASm.pt', 'LibreYOLONASl.pt', 'LibreYOLONASn-pose.pt', 'LibreYOLONASs-pose.pt', 'LibreYOLONASm-pose.pt', 'LibreYOLONASl-pose.pt']} />],
              ['D-FINE', 'detect', <Checkpoints key="dfine" names={['LibreDFINEn.pt', 'LibreDFINEs.pt', 'LibreDFINEm.pt', 'LibreDFINEl.pt', 'LibreDFINEx.pt']} />],
              ['DEIM', 'detect', <Checkpoints key="deim" names={['LibreDEIMn.pt', 'LibreDEIMs.pt', 'LibreDEIMm.pt', 'LibreDEIMl.pt', 'LibreDEIMx.pt']} />],
              ['DEIMv2', 'detect', <Checkpoints key="deimv2" names={['LibreDEIMv2atto.pt', 'LibreDEIMv2femto.pt', 'LibreDEIMv2pico.pt', 'LibreDEIMv2n.pt', 'LibreDEIMv2s.pt', 'LibreDEIMv2m.pt', 'LibreDEIMv2l.pt', 'LibreDEIMv2x.pt']} />],
              ['RT-DETR', 'detect', <Checkpoints key="rtdetr" names={['LibreRTDETRr18.pt', 'LibreRTDETRr34.pt', 'LibreRTDETRr50.pt', 'LibreRTDETRr50m.pt', 'LibreRTDETRr101.pt', 'LibreRTDETRl.pt', 'LibreRTDETRx.pt']} />],
              ['RT-DETRv2', 'detect', <Checkpoints key="rtdetrv2" names={['LibreRTDETRv2r18.pt', 'LibreRTDETRv2r34.pt', 'LibreRTDETRv2r50.pt', 'LibreRTDETRv2r50m.pt', 'LibreRTDETRv2r101.pt']} />],
              ['RT-DETRv4', 'detect', <Checkpoints key="rtdetrv4" names={['LibreRTDETRv4s.pt', 'LibreRTDETRv4m.pt', 'LibreRTDETRv4l.pt', 'LibreRTDETRv4x.pt']} />],
              ['PicoDet', 'detect', <Checkpoints key="picodet" names={['LibrePICODETs.pt', 'LibrePICODETm.pt', 'LibrePICODETl.pt']} />],
              ['EdgeCrafter', 'detect, pose, segment', <Checkpoints key="ec" names={['LibreECs.pt', 'LibreECm.pt', 'LibreECl.pt', 'LibreECx.pt', 'LibreECs-pose.pt', 'LibreECm-pose.pt', 'LibreECl-pose.pt', 'LibreECx-pose.pt', 'LibreECs-seg.pt', 'LibreECm-seg.pt', 'LibreECl-seg.pt', 'LibreECx-seg.pt']} />],
              ['DAMO-YOLO', 'detect', <Checkpoints key="damo" names={['LibreDAMOYOLOns.pt', 'LibreDAMOYOLOnm.pt', 'LibreDAMOYOLOnl.pt', 'LibreDAMOYOLOt.pt', 'LibreDAMOYOLOs.pt', 'LibreDAMOYOLOm.pt', 'LibreDAMOYOLOl.pt']} />],
              ['RTMDet', 'detect', <Checkpoints key="rtmdet" names={['LibreRTMDett.pt', 'LibreRTMDets.pt', 'LibreRTMDetm.pt', 'LibreRTMDetl.pt', 'LibreRTMDetx.pt']} />],
            ]}
          />
          <P className="text-sm">
            <strong className="text-surface-800 dark:text-white">Hosting note:</strong> YOLO-NAS checkpoints (plain text above) are hosted on Deci&apos;s CDN under their proprietary weights license, not on the LibreYOLO Hugging Face org. The factory still downloads them automatically on first use.
          </P>

          <SubHeading>Specialized models</SubHeading>
          <DocTable
            headers={['Family', 'Tasks', 'Checkpoints']}
            rows={[
              ['L2CS', <span key="t">gaze (inference-only) - see <a href="#gaze" className="text-libre-600 dark:text-libre-400 hover:underline">Gaze Estimation</a></span>, <Checkpoints key="l2cs" link={false} names={['LibreL2CSr18.pt', 'LibreL2CSr34.pt', 'LibreL2CSr50.pt', 'LibreL2CSr101.pt', 'LibreL2CSr152.pt']} />],
            ]}
          />
          <P className="text-sm">
            L2CS weights are hosted on Google Drive (Gaze360 dataset license forbids redistribution); LibreYOLO does not mirror them on Hugging Face. With <InlineCode>pip install libreyolo[gaze]</InlineCode>, the factory fetches them automatically on first use.
          </P>

          <SubHeading>Factory function (recommended)</SubHeading>
          <P>
            The <InlineCode>LibreYOLO()</InlineCode> factory auto-detects family, size, classes, and task from the weights file. It also handles every runtime backend format - point it at a <InlineCode>.pt</InlineCode>, <InlineCode>.onnx</InlineCode>, <InlineCode>.engine</InlineCode>, or a directory of OpenVINO/NCNN artifacts:
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

# Flagship: YOLO9 (auto-detected: yolo9, size=c, 80 classes)
model = LibreYOLO("LibreYOLO9c.pt")

# Flagship: RF-DETR (auto-detected: rfdetr, size=s)
model = LibreYOLO("LibreRFDETRs.pt")

# Any other supported family - same call
model = LibreYOLO("LibreYOLOXs.pt")
model = LibreYOLO("LibreRTDETRr50.pt")

# Multi-task checkpoints: task is inferred from the -seg / -pose suffix
model = LibreYOLO("LibreYOLO9c-seg.pt")        # segmentation
model = LibreYOLO("LibreYOLONASm-pose.pt")     # pose

# Exported deployment formats
model = LibreYOLO("model.onnx")                # ONNX Runtime
model = LibreYOLO("model.engine")              # TensorRT
model = LibreYOLO("model_openvino/")           # OpenVINO (directory)
model = LibreYOLO("model_ncnn/")               # NCNN (directory)`}</CodeBlock>
          <P>
            For recognized official checkpoint filenames, LibreYOLO can auto-download missing weights. For custom filenames, point at an explicit local path. To override task inference, pass <InlineCode>task=&quot;segment&quot;</InlineCode> / <InlineCode>&quot;pose&quot;</InlineCode> / <InlineCode>&quot;detect&quot;</InlineCode> - see <a href="#tasks" className="text-libre-600 dark:text-libre-400 hover:underline">Tasks &amp; Filenames</a>.
          </P>

          <Divider />

          {/* ────────────── TASKS & FILENAMES ────────────── */}
          <SectionHeading id="tasks" icon={Tags}>Tasks &amp; Filenames</SectionHeading>
          <P>
            LibreYOLO uses a uniform filename convention so the factory can detect family, size, and task from the checkpoint name alone:
          </P>
          <CodeBlock language="text">{`Libre<FAMILY><size>[-<task>].pt`}</CodeBlock>

          <SubHeading>Task suffixes</SubHeading>
          <DocTable
            headers={['Task', 'Canonical name', 'Filename suffix']}
            rows={[
              ['Detection', <InlineCode key="d">&quot;detect&quot;</InlineCode>, '(none - implicit)'],
              ['Instance segmentation', <InlineCode key="s">&quot;segment&quot;</InlineCode>, <InlineCode key="ss">-seg</InlineCode>],
              ['Pose estimation', <InlineCode key="p">&quot;pose&quot;</InlineCode>, <InlineCode key="ps">-pose</InlineCode>],
              ['Classification', <InlineCode key="c">&quot;classify&quot;</InlineCode>, <InlineCode key="cs">-cls</InlineCode>],
              ['Gaze estimation', <InlineCode key="g">&quot;gaze&quot;</InlineCode>, <InlineCode key="gs">-gaze</InlineCode>],
            ]}
          />
          <P>
            The factory accepts aliases at the API boundary (<InlineCode>&quot;detection&quot;</InlineCode>, <InlineCode>&quot;seg&quot;</InlineCode>, <InlineCode>&quot;keypoints&quot;</InlineCode>, etc.) - only the canonical names appear in filenames.
          </P>

          <SubHeading>Resolution precedence</SubHeading>
          <P>
            When you load a model, the task is resolved in this order:
          </P>
          <CodeBlock language="text">{`explicit task=  →  checkpoint["task"]  →  filename suffix  →  family default`}</CodeBlock>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

# 1. Filename suffix decides → segment
model = LibreYOLO("LibreYOLO9c-seg.pt")

# 2. Override regardless of filename
model = LibreYOLO("custom_weights.pt", task="segment")

# 3. Detection is implicit
model = LibreYOLO("LibreYOLO9c.pt")  # task="detect"`}</CodeBlock>

          <SubHeading>Per-family task support</SubHeading>
          <DocTable
            headers={['Family', 'Default', 'Supported tasks']}
            rows={[
              [<strong key="y9">YOLO9 (flagship)</strong>, 'detect', 'detect, segment'],
              [<strong key="rfd">RF-DETR (flagship)</strong>, 'detect', 'detect, segment'],
              ['YOLOX', 'detect', 'detect'],
              ['YOLO9-E2E', 'detect', 'detect'],
              ['YOLO-NAS', 'detect', 'detect, pose'],
              ['D-FINE / DEIM / DEIMv2', 'detect', 'detect'],
              ['RT-DETR / RT-DETRv2 / RT-DETRv4', 'detect', 'detect'],
              ['PicoDet', 'detect', 'detect'],
              ['EdgeCrafter (EC)', 'detect', 'detect, pose, segment'],
              ['DAMO-YOLO / RTMDet', 'detect', 'detect'],
              ['L2CS', 'gaze', 'gaze (inference-only)'],
            ]}
          />

          <SubHeading>Examples</SubHeading>
          <CodeBlock language="text">{`# Detection (implicit)
LibreYOLO9c.pt
LibreRFDETRs.pt
LibreRTDETRr50.pt

# Segmentation
LibreYOLO9c-seg.pt
LibreRFDETRs-seg.pt
LibreECm-seg.pt

# Pose
LibreYOLONASn-pose.pt
LibreECs-pose.pt

# Gaze
LibreL2CSr50.pt   # gaze is L2CS's only task - suffix optional`}</CodeBlock>

          <SubHeading>Deprecated aliases</SubHeading>
          <P>
            <InlineCode>LibreYOLORTDETR</InlineCode> and <InlineCode>LibreYOLORFDETR</InlineCode> are old names for <InlineCode>LibreRTDETR</InlineCode> and <InlineCode>LibreRFDETR</InlineCode> respectively. They still resolve but emit a <InlineCode>DeprecationWarning</InlineCode> - update imports when convenient.
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

# OpenCV (BGR) - specify color_format
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
            <FeatureItem><InlineCode>final_image.jpg</InlineCode> - full image with all merged detections drawn</FeatureItem>
            <FeatureItem><InlineCode>grid_visualization.jpg</InlineCode> - image showing tile grid overlay</FeatureItem>
            <FeatureItem><InlineCode>tiles/</InlineCode> - individual tile crops</FeatureItem>
            <FeatureItem><InlineCode>metadata.json</InlineCode> - tiling parameters and detection counts</FeatureItem>
          </ul>
          <P>
            If the image is already smaller than the model's input size, tiling is skipped automatically.
          </P>

          <Divider />

          {/* ────────────── VIDEO INFERENCE ────────────── */}
          <SectionHeading id="video-inference" icon={Video}>Video Inference</SectionHeading>
          <P>
            Pass any video file to a flagship and LibreYOLO auto-detects the format from the extension. Supported: <InlineCode>.mp4</InlineCode>, <InlineCode>.avi</InlineCode>, <InlineCode>.mov</InlineCode>, <InlineCode>.mkv</InlineCode>, <InlineCode>.webm</InlineCode>, <InlineCode>.gif</InlineCode>, and other common containers.
          </P>

          <SubHeading>Save annotated video</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")
results = model("clip.mp4", save=True)
# Saved under runs/detect/predict*/clip.mp4`}</CodeBlock>

          <SubHeading>Stream results (memory-flat)</SubHeading>
          <P>
            For long videos, pass <InlineCode>stream=True</InlineCode> to get a generator. Each iteration yields the <InlineCode>Results</InlineCode> for one frame - no full list buffered in RAM.
          </P>
          <CodeBlock language="python">{`for result in model("long_clip.mp4", stream=True):
    print(f"frame {result.frame_idx}: {len(result)} detections")`}</CodeBlock>

          <SubHeading>Frame subsampling</SubHeading>
          <CodeBlock language="python">{`# Process every 2nd frame (halves compute and saved fps)
results = model("clip.mp4", vid_stride=2, save=True)`}</CodeBlock>

          <SubHeading>Live preview</SubHeading>
          <CodeBlock language="python">{`# Display annotated frames in an OpenCV window while processing
results = model("clip.mp4", show=True)`}</CodeBlock>

          <SubHeading>VideoSource / VideoWriter for custom pipelines</SubHeading>
          <P>
            When you need full control of decoding and encoding - custom frame transforms, mixing tracker output, writing to a non-default codec - use the building blocks directly:
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO
from libreyolo.utils.video import VideoSource, VideoWriter

model = LibreYOLO("LibreYOLO9c.pt")

with VideoSource("clip.mp4", vid_stride=1) as src, \\
     VideoWriter("out.mp4", fps=src.fps, width=src.width, height=src.height) as out:
    for frame_bgr, frame_idx in src:
        result = model(frame_bgr, color_format="bgr")
        # ... draw, transform, etc.
        out.write_frame(frame_bgr)`}</CodeBlock>

          <Divider />

          {/* ────────────── TRACKING ────────────── */}
          <SectionHeading id="tracking" icon={Activity}>Tracking</SectionHeading>
          <P>
            LibreYOLO ships a ByteTrack multi-object tracker that consumes <InlineCode>Results</InlineCode> from any detector and adds persistent track IDs. Works out of the box with the flagships - and any other detection model.
          </P>

          <SubHeading>Install</SubHeading>
          <CodeBlock language="bash">{`pip install libreyolo[tracking]   # pulls scipy`}</CodeBlock>

          <SubHeading>Basic loop</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO, ByteTracker
from libreyolo.utils.video import VideoSource

model = LibreYOLO("LibreYOLO9c.pt")
tracker = ByteTracker()

with VideoSource("clip.mp4") as src:
    for frame_bgr, frame_idx in src:
        result = model(frame_bgr, color_format="bgr", conf=0.1)
        tracked = tracker.update(result)

        for i in range(len(tracked.boxes)):
            track_id = int(tracked.boxes.id[i])
            xyxy = tracked.boxes.xyxy[i].tolist()
            cls = int(tracked.boxes.cls[i])
            print(f"frame {frame_idx} - id {track_id} cls {cls} {xyxy}")`}</CodeBlock>
          <P>
            After <InlineCode>tracker.update()</InlineCode>, <InlineCode>result.boxes.id</InlineCode> holds the track IDs and <InlineCode>result.boxes.is_track</InlineCode> is <InlineCode>True</InlineCode>.
          </P>

          <SubHeading>TrackConfig knobs</SubHeading>
          <CodeBlock language="python">{`from libreyolo import ByteTracker, TrackConfig

cfg = TrackConfig(
    track_high_thresh=0.25,           # first-stage match threshold
    track_low_thresh=0.1,             # second-stage (low-conf recovery)
    new_track_thresh=0.25,            # minimum conf to start a new track
    match_thresh=0.8,                 # IoU cost cutoff (stage 1)
    match_thresh_low=0.5,             # IoU cost cutoff (stage 2)
    match_thresh_unconfirmed=0.7,     # IoU cost cutoff for unconfirmed tracks
    track_buffer=30,                  # frames to keep lost tracks before removal
    frame_rate=30,                    # scales track_buffer
    fuse_score=True,                  # multiply IoU by detection score
    minimum_consecutive_frames=1,     # frames to confirm a new track
)
tracker = ByteTracker(config=cfg)`}</CodeBlock>

          <SubHeading>Reset between clips</SubHeading>
          <CodeBlock language="python">{`tracker.reset()   # clears tracked / lost / removed lists and the ID counter`}</CodeBlock>

          <Divider />

          {/* ────────────── SEGMENTATION ────────────── */}
          <SectionHeading id="segmentation" icon={Scissors}>Segmentation</SectionHeading>
          <FlagshipCallout />
          <P>
            Instance segmentation is supported on both flagships - <strong className="text-surface-800 dark:text-white">YOLO9 (<InlineCode>-seg</InlineCode>)</strong> and <strong className="text-surface-800 dark:text-white">RF-DETR (<InlineCode>-seg</InlineCode>)</strong> - plus EdgeCrafter (<InlineCode>-seg</InlineCode>). The factory routes to the segmentation head automatically from the filename suffix.
          </P>

          <SubHeading>Run segmentation</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

# YOLO9 segmentation flagship
model = LibreYOLO("LibreYOLO9c-seg.pt")
result = model("photo.jpg")

# RF-DETR segmentation flagship
# model = LibreYOLO("LibreRFDETRs-seg.pt")

# Both return boxes + masks
print(result.boxes.xyxy)        # bounding boxes (N, 4)
print(result.boxes.cls)         # class IDs (N,)
print(result.masks.data.shape)  # (N, H, W) tensor of binary masks`}</CodeBlock>

          <SubHeading>Mask representations</SubHeading>
          <CodeBlock language="python">{`# Raw bitmasks
result.masks.data        # tensor (N, H, W) - original image resolution

# Polygon contours (one ndarray of (M, 2) per instance)
result.masks.xy          # absolute pixel coords
result.masks.xyn         # normalized to [0, 1]

# Move / convert like Boxes
result.masks.cpu()
result.masks.numpy()`}</CodeBlock>

          <SubHeading>Save annotated output</SubHeading>
          <P>
            <InlineCode>save=True</InlineCode> draws boxes and translucent mask overlays automatically.
          </P>
          <CodeBlock language="python">{`model("photo.jpg", save=True)`}</CodeBlock>

          <SubHeading>Training segmentation</SubHeading>
          <P>
            YOLO9 segmentation trains through the same <InlineCode>model.train(data=&quot;...yaml&quot;)</InlineCode> path as detection - load the <InlineCode>-seg</InlineCode> checkpoint and call train. RF-DETR segmentation uses the RF-DETR COCO-format training pipeline.
          </P>

          <Divider />

          {/* ────────────── POSE ESTIMATION ────────────── */}
          <SectionHeading id="pose" icon={PersonStanding}>Pose Estimation</SectionHeading>
          <P>
            Pose (human keypoint) estimation is supported on <InlineCode>YOLO-NAS (-pose)</InlineCode> and <InlineCode>EdgeCrafter (-pose)</InlineCode>. Each pose model is single-class (&quot;person&quot;) with 17 COCO keypoints.
          </P>

          <SubHeading>Run pose</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

# YOLO-NAS pose
model = LibreYOLO("LibreYOLONASs-pose.pt")
result = model("people.jpg")

# EdgeCrafter pose
# model = LibreYOLO("LibreECs-pose.pt")

# Per-person bbox + 17 keypoints
print(result.boxes.xyxy)          # person boxes (N, 4)
print(result.keypoints.xy.shape)  # (N, 17, 2) pixel coordinates`}</CodeBlock>

          <SubHeading>Keypoint API</SubHeading>
          <CodeBlock language="python">{`result.keypoints.xy        # (N, K, 2) absolute pixel coords
result.keypoints.xyn       # (N, K, 2) normalized to [0, 1]
result.keypoints.conf      # (N, K) per-keypoint confidence (None if model doesn't emit it)
result.keypoints.has_visible  # (N, K) bool - conf > 0

result.keypoints.cpu()
result.keypoints.numpy()`}</CodeBlock>

          <SubHeading>Save annotated output</SubHeading>
          <CodeBlock language="python">{`model("people.jpg", save=True)  # draws boxes + skeleton`}</CodeBlock>

          <P>
            Pose training is supported for YOLO-NAS; EdgeCrafter pose is currently inference-only. YOLO9 and RF-DETR don&apos;t ship pose checkpoints yet.
          </P>

          <Divider />

          {/* ────────────── GAZE ESTIMATION ────────────── */}
          <SectionHeading id="gaze" icon={Eye}>Gaze Estimation</SectionHeading>
          <P>
            Gaze direction estimation is provided by the <InlineCode>LibreL2CS</InlineCode> family, an L2CS-Net port (ResNet trunk with two angle-bin classification heads). It&apos;s a two-stage model: an upstream face detector locates faces, then the gaze head predicts per-face pitch and yaw in radians. Inference-only.
          </P>

          <SubHeading>Install</SubHeading>
          <CodeBlock language="bash">{`pip install libreyolo[gaze]   # enables best-effort auto-download of the Gaze360 weights`}</CodeBlock>
          <P>
            Weights are licensed under the Gaze360 dataset license (research / non-commercial only); LibreYOLO doesn&apos;t mirror them on its HF org. With <InlineCode>libreyolo[gaze]</InlineCode>, LibreYOLO fetches them from the original authors&apos; Google Drive on first use. Otherwise, see the printed manual-download instructions.
          </P>

          <SubHeading>Two-stage inference</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO
from libreyolo.models.l2cs.face import resolve_face_detector

# Gaze head
gaze = LibreYOLO("LibreL2CSr50.pt")

# Wire any LibreYOLO detector trained on faces
face = LibreYOLO("path/to/face-detector.pt")
gaze.face_detector = resolve_face_detector(face)

result = gaze("portrait.jpg")
print(result.boxes.xyxy)    # face boxes
print(result.gaze.data)     # (N, 2) tensor - pitch, yaw in radians`}</CodeBlock>

          <SubHeading>Decode angles</SubHeading>
          <CodeBlock language="python">{`import math

for i in range(len(result.gaze)):
    pitch_rad, yaw_rad = result.gaze.data[i].tolist()
    pitch_deg = pitch_rad * 180.0 / math.pi
    yaw_deg = yaw_rad * 180.0 / math.pi
    print(f"face {i}: pitch={pitch_deg:.1f}°, yaw={yaw_deg:.1f}°")`}</CodeBlock>

          <P>
            From the CLI: <InlineCode>libreyolo predict model=LibreL2CSr50.pt source=portrait.jpg --face-detector path/to/face.pt</InlineCode>.
          </P>

          <Divider />

          {/* ────────────── TRAINING ────────────── */}
          <SectionHeading id="training" icon={GraduationCap}>Training</SectionHeading>
          <FlagshipCallout />
          <P>
            Every model trains through the same pattern: load with the <InlineCode>LibreYOLO()</InlineCode> factory (or the family constructor for from-scratch runs), then call <InlineCode>model.train(data=&quot;...yaml&quot;)</InlineCode>. We document the two flagships in full and collapse the rest into short snippets - they share the same parameter shape.
          </P>

          <SubHeading>YOLO9 - CNN flagship training</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

# Fine-tune from a pretrained checkpoint (recommended)
model = LibreYOLO("LibreYOLO9c.pt")

# Advanced: start from a fresh model
# from libreyolo import LibreYOLO9
# model = LibreYOLO9(model_path=None, size="c")

results = model.train(
    data="coco128.yaml",     # path to data.yaml (required)

    # Schedule
    epochs=300,              # default: 300
    batch=16,
    imgsz=640,

    # Optimizer
    lr0=0.01,                # initial learning rate
    optimizer="SGD",         # "SGD", "Adam", "AdamW"

    # System
    device="0",              # "0" | "0,1" | "0,1,2,3" — comma-separated for multi-GPU
    workers=8,
    seed=0,

    # Output
    project="runs/train",
    name="yolo9_exp",
    exist_ok=False,

    # Training features
    amp=True,                # automatic mixed precision
    patience=50,             # early stopping patience
    resume=False,            # True = resume from the loaded checkpoint path
)

print(f"Best mAP50-95: {results['best_mAP50_95']:.3f}")
print(f"Best checkpoint: {results['best_checkpoint']}")`}</CodeBlock>
          <P>
            After training completes, the model instance is automatically reloaded with the best weights so you can call <InlineCode>model(...)</InlineCode> immediately. YOLO9 segmentation training is supported via <InlineCode>LibreYOLO(&quot;LibreYOLO9c-seg.pt&quot;)</InlineCode>.
          </P>

          <SubHeading>RF-DETR - transformer flagship training</SubHeading>
          <CodeBlock language="python">{`from libreyolo.models.rfdetr.model import LibreRFDETR

# Fine-tune from pretrained weights (None = use built-in pretrained)
model = LibreRFDETR(None, size="n")  # or "s", "m", "l"

results = model.train(
    data="path/to/data.yaml",
    epochs=100,
    batch=-1,          # -1 = AutoBatch (targets 60% VRAM)
    nbs=16,            # nominal batch for grad accumulation (RF-DETR default: 16)
    device="0",        # "0" | "0,1" | "0,1,2,3" — comma-separated for multi-GPU
    workers=4,
    output_dir="runs/train/rfdetr_exp",
    amp=True,
    seed=42,
    exist_ok=True,
)

print(f"Best mAP50-95: {results['best_mAP50_95']:.3f}")
print(f"Weights saved: {results['output_dir']}")`}</CodeBlock>
          <CodeBlock language="bash">{`# CLI
libreyolo train --model rfdetr-n --data path/to/data.yaml --device 0 --batch -1`}</CodeBlock>
          <SubHeading>Resuming RF-DETR training</SubHeading>
          <CodeBlock language="python">{`# Pass the checkpoint path as the first argument to resume
model = LibreRFDETR("runs/train/rfdetr_exp/best.pt", size="n")
results = model.train(data="path/to/data.yaml", resume="runs/train/rfdetr_exp/best.pt")`}</CodeBlock>

          <SubHeading>Training results dict</SubHeading>
          <CodeBlock language="python">{`{
    "final_loss": 2.31,
    "best_mAP50": 0.682,
    "best_mAP50_95": 0.451,
    "best_epoch": 87,
    "save_dir": "runs/train/yolo9_exp",
    "best_checkpoint": "runs/train/yolo9_exp/weights/best.pt",
    "last_checkpoint": "runs/train/yolo9_exp/weights/last.pt",
}`}</CodeBlock>

          <SubHeading>Resuming training</SubHeading>
          <CodeBlock language="python">{`# Load the checkpoint with the factory, then resume
model = LibreYOLO("runs/train/yolo9_exp/weights/last.pt")
results = model.train(data="coco128.yaml", resume=True)`}</CodeBlock>

          <SubHeading>Custom dataset YAML format</SubHeading>
          <CodeBlock language="yaml" filename="data.yaml">{`path: /path/to/dataset
train: images/train
val: images/val
test: images/test  # optional

nc: 3
names: ["cat", "dog", "bird"]`}</CodeBlock>

          <SubHeading>Additional training paths</SubHeading>
          <P>
            Same factory pattern, same call shape. The defaults differ - see the API Reference for each family&apos;s full signature.
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

# YOLOX
model = LibreYOLO("LibreYOLOXs.pt")
model.train(data="coco128.yaml", epochs=100, batch=16, lr0=0.01, optimizer="SGD")

# RT-DETR (note: adds lr_backbone and scheduler)
model = LibreYOLO("LibreRTDETRr50.pt")
model.train(
    data="coco128.yaml",
    epochs=72, batch=4, lr0=1e-4, lr_backbone=1e-5,
    optimizer="AdamW", scheduler="linear", pretrained=True,
)

# EdgeCrafter (experimental fine-tune path)
model = LibreYOLO("LibreECs.pt")
model.train(data="coco128.yaml", allow_experimental=True)`}</CodeBlock>

          <SubHeading>Training from a YAML config</SubHeading>
          <P>
            Every <InlineCode>model.train(...)</InlineCode> accepts <InlineCode>cfg=&quot;train.yaml&quot;</InlineCode> to load all parameters from a file. Explicit kwargs still win over yaml values, so you can use a yaml for the baseline and override individual fields per run.
          </P>
          <CodeBlock language="python">{`model = LibreYOLO("LibreYOLO9c.pt")
results = model.train(cfg="configs/yolo9_finetune.yaml")
# Override individual fields:
# results = model.train(cfg="configs/yolo9_finetune.yaml", epochs=50)`}</CodeBlock>

          <SubHeading>Gradient accumulation</SubHeading>
          <P>
            Pass <InlineCode>nbs</InlineCode> (nominal batch size) to opt into gradient accumulation. The trainer steps the optimizer every <InlineCode>nbs / batch</InlineCode> forward passes, which lets you train at the recipe&apos;s reference batch size on smaller hardware.
          </P>
          <CodeBlock language="python">{`# Effective batch 64 on a single GPU that only fits batch=8
model.train(data="coco128.yaml", batch=8, nbs=64)`}</CodeBlock>

          <SubHeading>Multi-GPU training</SubHeading>
          <P>
            Both YOLO9 and RF-DETR support multi-GPU training by passing comma-separated GPU indices to <InlineCode>device</InlineCode>. No <InlineCode>torchrun</InlineCode> wrapper needed.
          </P>
          <CodeBlock language="python">{`# YOLO9 — 2 GPUs
from libreyolo import LibreYOLO9

model = LibreYOLO9(None, size="c")
model.train(data="coco128.yaml", epochs=300, batch=-1, nbs=64, device="0,1", amp=True)

# RF-DETR — 2 GPUs
from libreyolo.models.rfdetr.model import LibreRFDETR

model = LibreRFDETR(None, size="n")
model.train(data="path/to/data.yaml", epochs=100, batch=-1, nbs=16, device="0,1", amp=True)`}</CodeBlock>
          <P>
            Pass as many indices as you have available GPUs: <InlineCode>&quot;0,1,2,3&quot;</InlineCode> for a 4-GPU node. The trainer distributes batches across ranks automatically.
          </P>

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
            Segmentation and pose validation return additional <InlineCode>(M)</InlineCode> (mask) and pose-keypoint metrics - see <InlineCode>SegmentationValidator</InlineCode> and <InlineCode>PoseValidator</InlineCode> in the API reference.
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
    opset=None,               # ONNX opset (auto: 13 CNN, 17 DETR families)
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
              [<InlineCode key="v">libreyolo_version</InlineCode>, <InlineCode key="vv">&quot;1.2.0&quot;</InlineCode>],
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

# Load - names and nb_classes auto-populated
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

          {/* ────────────── CLI ────────────── */}
          <SectionHeading id="cli" icon={SquareTerminal}>CLI</SectionHeading>
          <P>
            Installing LibreYOLO registers a <InlineCode>libreyolo</InlineCode> command on your PATH (entry point in <InlineCode>pyproject.toml</InlineCode>). The CLI mirrors the Python API and follows Ultralytics-style <InlineCode>key=value</InlineCode> syntax.
          </P>

          <SubHeading>Subcommands</SubHeading>
          <DocTable
            headers={['Command', 'Purpose']}
            rows={[
              [<InlineCode key="p">predict</InlineCode>, 'Run inference on images, directories, or videos'],
              [<InlineCode key="t">train</InlineCode>, 'Train a model on a dataset'],
              [<InlineCode key="v">val</InlineCode>, 'Evaluate a model on a dataset'],
              [<InlineCode key="e">export</InlineCode>, 'Export to ONNX / TorchScript / TensorRT / OpenVINO / NCNN'],
              [<InlineCode key="c">checks</InlineCode>, 'Print Python, torch, CUDA, GPU, and optional-package info'],
              [<InlineCode key="m">models</InlineCode>, 'List registered model families and CLI shortcut names'],
              [<InlineCode key="f">formats</InlineCode>, 'List supported export formats'],
              [<InlineCode key="i">info</InlineCode>, 'Inspect a checkpoint file'],
              [<InlineCode key="ver">version</InlineCode>, 'Print LibreYOLO + Python + torch versions'],
            ]}
          />

          <SubHeading>Model name shortcuts</SubHeading>
          <P>
            The CLI accepts short names (<InlineCode>yolo9-c</InlineCode>) that resolve to weight filenames (<InlineCode>LibreYOLO9c.pt</InlineCode>) - discoverable via <InlineCode>libreyolo models</InlineCode>. You can also pass any explicit checkpoint path.
          </P>

          <SubHeading>Predict</SubHeading>
          <CodeBlock language="bash">{`# Flagship: YOLO9
libreyolo predict model=yolo9-c source=image.jpg conf=0.25 save=true

# Flagship: RF-DETR
libreyolo predict model=rfdetr-s source=image.jpg save=true

# Video - saved under runs/detect/predict*/
libreyolo predict model=yolo9-c source=clip.mp4 save=true

# Tiled inference for very large images
libreyolo predict model=yolo9-c source=aerial.jpg tiling=true save=true

# Gaze (requires a face detector)
libreyolo predict model=LibreL2CSr50.pt source=portrait.jpg \\
    --face-detector path/to/face.pt save=true`}</CodeBlock>

          <SubHeading>Train</SubHeading>
          <CodeBlock language="bash">{`libreyolo train model=yolo9-c data=coco128.yaml epochs=300 batch=16 device=0

# Dry-run prints the resolved config without launching training
libreyolo train model=yolo9-c data=coco128.yaml --dry-run`}</CodeBlock>

          <SubHeading>Validate</SubHeading>
          <CodeBlock language="bash">{`libreyolo val model=runs/train/exp/weights/best.pt data=coco128.yaml split=val`}</CodeBlock>

          <SubHeading>Export</SubHeading>
          <CodeBlock language="bash">{`libreyolo export model=runs/train/exp/weights/best.pt format=onnx dynamic=true
libreyolo export model=best.pt format=tensorrt half=true
libreyolo export model=best.pt format=openvino int8=true data=coco128.yaml`}</CodeBlock>

          <SubHeading>Machine-readable output</SubHeading>
          <P>
            Every command accepts <InlineCode>--json</InlineCode> (structured stdout for piping into scripts or agents) and <InlineCode>--quiet</InlineCode> (suppress stderr progress lines). Use <InlineCode>--help-json</InlineCode> on any subcommand to dump its parameter schema as JSON - handy for tooling.
          </P>
          <CodeBlock language="bash">{`libreyolo predict model=yolo9-c source=img.jpg --json | jq .

libreyolo train --help-json > train_schema.json`}</CodeBlock>

          <Divider />

          {/* ────────────── API REFERENCE ────────────── */}
          <SectionHeading id="api-reference" icon={FileCode}>API Reference</SectionHeading>

          <SubHeading>LibreYOLO (factory)</SubHeading>
          <CodeBlock language="python">{`LibreYOLO(
    model_path: str,
    size: str | None = None,    # auto-detected from weights
    reg_max: int = 16,          # YOLO9 only
    nb_classes: int | None = None,  # auto-detected from weights
    device: str = "auto",
    task: str | None = None,    # explicit task override: "detect" | "segment" | "pose" | "gaze"
) -> (
    LibreYOLOX | LibreYOLO9 | LibreYOLO9E2E | LibreYOLONAS
    | LibreDFINE | LibreDEIM | LibreDEIMv2 | LibreEC | LibrePICODET
    | LibreDAMOYOLO | LibreRTDETR | LibreRTDETRv2 | LibreRTDETRv4
    | LibreRTMDet | LibreRFDETR | LibreL2CS
    | OnnxBackend | TorchScriptBackend | TensorRTBackend
    | OpenVINOBackend | NcnnBackend
)`}</CodeBlock>
          <P>
            Auto-detects family, size, class count, and task from the weights file. It also handles <InlineCode>.onnx</InlineCode>, <InlineCode>.torchscript</InlineCode>, <InlineCode>.engine</InlineCode>, OpenVINO directories containing <InlineCode>model.xml</InlineCode>, and NCNN directories containing <InlineCode>model.ncnn.param</InlineCode> plus <InlineCode>model.ncnn.bin</InlineCode>. The <InlineCode>task</InlineCode> argument overrides everything else; otherwise resolution is checkpoint metadata → filename suffix → family default.
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

boxes.xyxy           # (N, 4) tensor - x1, y1, x2, y2
boxes.xywh           # (N, 4) tensor - cx, cy, w, h
boxes.conf           # (N,) tensor - confidence scores
boxes.cls            # (N,) tensor - class IDs
boxes.data           # (N, 6) tensor -[xyxy, conf, cls]

len(boxes)           # number of boxes
boxes.cpu()          # copy on CPU
boxes.numpy()        # copy as numpy arrays`}</CodeBlock>

          <SubHeading>model.export()</SubHeading>
          <CodeBlock language="python">{`model.export(
    format: str = "onnx",       # "onnx", "torchscript", "tensorrt", "openvino", or "ncnn"
    *,
    output_path: str | None = None,
    imgsz: int | None = None,
    opset: int | None = None,   # auto: 13 for CNN families, 17 for DETR families
    simplify: bool = True,
    dynamic: bool = True,
    half: bool = False,
    batch: int = 1,
    device: str | None = None,
    int8: bool = False,
    data: str | None = None,    # calibration data for INT8
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
    data: str,                            # path to YOLO-format data.yaml
    epochs: int = 100,
    batch: int = -1,                      # -1 = AutoBatch (targets 60% VRAM)
    nbs: int = 16,                        # nominal batch for grad accumulation (RFDETRConfig default)
    device: str = "0",                    # "0" | "0,1" | "0,1,2,3"
    workers: int = 4,
    output_dir: str = "runs/train",
    amp: bool = True,
    seed: int = 42,
    exist_ok: bool = True,
    resume: str | None = None,            # path to checkpoint to resume from
    allow_download_scripts: bool = False, # allow embedded download blocks in data YAML
    # imgsz: derived from size variant (n→384 s→512 m→576 l→704), not overridable
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
    split="val",               # "val", "test", or "train"
    batch_size=16,
    imgsz=640,
    conf_thres=0.001,
    iou_thres=0.6,
    max_det=300,
    iou_thresholds=(           # mAP IoU sweep
        0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
    ),
    device="auto",
    save_dir=None,
    save_json=False,
    verbose=True,
    num_workers=4,
    half=False,
    augment=False,             # test-time augmentation (TTA)
    allow_download_scripts=False,
    # Pose-only fields (PoseValidator)
    keypoints_json=None,
    images_dir=None,
    oks_sigmas=None,
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
    __init__.py          # Public API exports + deprecated-alias resolver
    tasks.py             # Task types, suffix conventions, resolution rules
    assets/parkour.jpg   # SAMPLE_IMAGE
    models/
        __init__.py      # LibreYOLO() factory + model registry bootstrap
        base/
            model.py     # BaseModel - shared wrapper behaviour
            inference.py # Shared prediction pipeline (image/dir/video/tiled)
        yolox/           # LibreYOLOX (detect)
        yolo9/           # LibreYOLO9 (detect, segment)
        yolo9_e2e/       # LibreYOLO9E2E (detect)
        yolonas/         # LibreYOLONAS (detect, pose)
        dfine/           # LibreDFINE (detect)
        deim/            # LibreDEIM (detect)
        deimv2/          # LibreDEIMv2 (detect)
        rtdetr/          # LibreRTDETR (detect)
        rtdetrv2/        # LibreRTDETRv2 (detect)
        rtdetrv4/        # LibreRTDETRv4 (detect)
        rfdetr/          # LibreRFDETR (detect, segment) - lazy-loaded
        ec/              # LibreEC / EdgeCrafter (detect, pose, segment)
        picodet/         # LibrePICODET (detect)
        damoyolo/        # LibreDAMOYOLO (detect)
        rtmdet/          # LibreRTMDet (detect)
        l2cs/            # LibreL2CS (gaze, inference-only)
    backends/
        base.py
        onnx.py          # OnnxBackend
        torchscript.py   # TorchScriptBackend
        tensorrt.py      # TensorRTBackend
        openvino.py      # OpenVINOBackend
        ncnn.py          # NcnnBackend
    export/
        exporter.py      # BaseExporter and format registry
        onnx.py / torchscript.py / tensorrt.py / openvino.py / ncnn.py
        config.py / calibration.py
    training/
        trainer.py       # Shared trainer scaffolding
        config.py        # TrainConfig dataclass (single source of truth)
        augment.py / callbacks.py / distributed.py / ema.py / scheduler.py
        artifacts.py / train_config.yaml
        # Per-family trainers live in models/<family>/trainer.py
    validation/
        config.py                # ValidationConfig
        base.py / preprocessors.py
        detection_validator.py   # DetectionValidator, SegmentationValidator
        pose_validator.py        # PoseValidator
        coco_evaluator.py        # COCOEvaluator
    tracking/
        tracker.py       # ByteTracker
        config.py        # TrackConfig
        kalman_filter.py / matching.py / strack.py
    cli/
        __init__.py      # libreyolo entrypoint (Typer app)
        commands/        # predict / train / val / export / special
        aliases.py / config.py / parsing.py / output.py / errors.py
    utils/
        results.py       # Results, Boxes, Masks, Keypoints, Probs, OBB, Gaze
        image_loader.py  # Unified image loading
        video.py         # VideoSource, VideoWriter, video inference loop
        general.py       # Path helpers, NMS, tiling utilities
        download.py / drawing.py / logging.py / predict_args.py
        serialization.py / box_ops.py
    data/
        dataset.py / pose_dataset.py / utils.py / yolo_coco_api.py
    config/
        datasets/        # Built-in dataset YAML configs (coco8, coco128, coco5000, coco, etc.)
        export/          # TensorRT default YAML`}</CodeBlock>

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
            All models — including RF-DETR — use datasets configured via <InlineCode>data.yaml</InlineCode>.
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

          {/* Bottom spacer */}
          <div className="h-16" />
        </div>
      </main>
    </div>
  )
}

export default function Docs() {
  return <DocsPage />
}
