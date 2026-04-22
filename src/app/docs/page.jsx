'use client'

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

function Sidebar({ activeSection, onNavigate, className = '' }) {
  return (
    <nav className={className}>
      <div className="flex items-center gap-2 mb-6 px-3">
        <BookOpen className="w-5 h-5 text-libre-600 dark:text-libre-400" />
        <span className="text-sm font-semibold text-surface-800 dark:text-white tracking-wide uppercase">Documentation</span>
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

export default function Docs() {
  const [activeSection, setActiveSection] = useState('introduction')
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

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

  return (
    <div className="flex min-h-screen">
      {/* Desktop sidebar */}
      <aside className="hidden lg:block fixed left-0 top-20 bottom-0 w-64 border-r border-surface-200 dark:border-white/[0.06] bg-white/80 dark:bg-surface-950/50 backdrop-blur-sm overflow-y-auto py-8 px-4 z-30">
        <Sidebar activeSection={activeSection} onNavigate={navigateTo} />
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
              <Sidebar activeSection={activeSection} onNavigate={navigateTo} />
            </motion.aside>
          </>
        )}
      </AnimatePresence>

      {/* Main content */}
      <main className="flex-1 lg:ml-64 min-h-screen pt-28 lg:pt-32 pb-24 px-6 lg:px-12">
        <div className="max-w-4xl mx-auto">

          {/* ────────────── INTRODUCTION ────────────── */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
            <SectionHeading id="introduction" icon={BookOpen}>Introduction</SectionHeading>
            <P>
              LibreYOLO is an MIT-licensed object detection library that provides a unified Python API across four architectures: <strong className="text-surface-800 dark:text-white">YOLOX</strong>, <strong className="text-surface-800 dark:text-white">YOLOv9</strong>, <strong className="text-surface-800 dark:text-white">RT-DETR</strong>, and <strong className="text-surface-800 dark:text-white">RF-DETR</strong>. One interface for prediction, training, validation, and export, regardless of which model family you use.
            </P>
            <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLOXs.pt")
results = model("image.jpg", conf=0.25, save=True)
print(results.boxes.xyxy)`}</CodeBlock>

            <SubHeading>Key features</SubHeading>
            <ul className="space-y-2.5 mb-4">
              <FeatureItem>Unified API across YOLOX, YOLOv9, RT-DETR, and RF-DETR</FeatureItem>
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
model = LibreYOLO("LibreYOLOXs.pt")

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

          <SubHeading>YOLOX</SubHeading>
          <DocTable
            headers={['Size', 'Code', 'Input size', 'Use case']}
            rows={[
              ['Nano', <InlineCode key="n">&quot;n&quot;</InlineCode>, '416', 'Edge devices, mobile'],
              ['Tiny', <InlineCode key="t">&quot;t&quot;</InlineCode>, '416', 'Edge devices, faster'],
              ['Small', <InlineCode key="s">&quot;s&quot;</InlineCode>, '640', 'Balanced speed/accuracy'],
              ['Medium', <InlineCode key="m">&quot;m&quot;</InlineCode>, '640', 'Higher accuracy'],
              ['Large', <InlineCode key="l">&quot;l&quot;</InlineCode>, '640', 'High accuracy'],
              ['XLarge', <InlineCode key="x">&quot;x&quot;</InlineCode>, '640', 'Maximum accuracy'],
            ]}
          />
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLOXn.pt")
# model = LibreYOLO("LibreYOLOXt.pt")
# model = LibreYOLO("LibreYOLOXs.pt")
# model = LibreYOLO("LibreYOLOXm.pt")
# model = LibreYOLO("LibreYOLOXl.pt")
# model = LibreYOLO("LibreYOLOXx.pt")`}</CodeBlock>

          <SubHeading>YOLOv9</SubHeading>
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

          <SubHeading>RT-DETR</SubHeading>
          <DocTable
            headers={['Size', 'Code', 'Input size', 'Use case']}
            rows={[
              ['ResNet-18', <InlineCode key="r18">&quot;r18&quot;</InlineCode>, '640', 'Fastest RT-DETR variant'],
              ['ResNet-34', <InlineCode key="r34">&quot;r34&quot;</InlineCode>, '640', 'Balanced speed/accuracy'],
              ['ResNet-50', <InlineCode key="r50">&quot;r50&quot;</InlineCode>, '640', 'General-purpose default'],
              ['ResNet-50-m', <InlineCode key="r50m">&quot;r50m&quot;</InlineCode>, '640', 'Higher accuracy'],
              ['ResNet-101', <InlineCode key="r101">&quot;r101&quot;</InlineCode>, '640', 'Largest RT-DETR variant'],
            ]}
          />
          <CodeBlock language="python">{`from libreyolo import LibreYOLORTDETR

model = LibreYOLORTDETR(size="r50")`}</CodeBlock>

          <SubHeading>RF-DETR</SubHeading>
          <DocTable
            headers={['Size', 'Code', 'Input size', 'Use case']}
            rows={[
              ['Nano', <InlineCode key="n">&quot;n&quot;</InlineCode>, '384', 'Edge'],
              ['Small', <InlineCode key="s">&quot;s&quot;</InlineCode>, '512', 'Balanced'],
              ['Medium', <InlineCode key="m">&quot;m&quot;</InlineCode>, '576', 'Higher accuracy'],
              ['Large', <InlineCode key="l">&quot;l&quot;</InlineCode>, '704', 'Maximum accuracy'],
            ]}
          />
          <CodeBlock language="python">{`from libreyolo import LibreYOLORFDETR

model = LibreYOLORFDETR(size="s")`}</CodeBlock>

          <SubHeading>Factory function (recommended)</SubHeading>
          <P>
            The <InlineCode>LibreYOLO()</InlineCode> factory auto-detects everything from the weights file:
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

# Auto-detects: YOLOX, size=s, 80 classes
model = LibreYOLO("LibreYOLOXs.pt")

# Auto-detects: YOLOv9, size=c, 80 classes
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

          <SubHeading>YOLOv9 training</SubHeading>
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
            YOLOv9 training uses the same parameter API as YOLOX but defaults to <InlineCode>epochs=300</InlineCode> and <InlineCode>name=&quot;yolo9_exp&quot;</InlineCode>. It does not have a <InlineCode>pretrained</InlineCode> parameter.
          </P>

          <SubHeading>RT-DETR training</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLORTDETR

model = LibreYOLORTDETR(size="r50")

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
          <CodeBlock language="python">{`from libreyolo import LibreYOLORFDETR

model = LibreYOLORFDETR(size="s")

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
    reg_max: int = 16,          # YOLOv9 only
    nb_classes: int = None,     # auto-detected from weights
    device: str = "auto",
) -> LibreYOLOX | LibreYOLO9 | LibreYOLORTDETR | LibreYOLORFDETR | OnnxBackend | TensorRTBackend | OpenVINOBackend | NcnnBackend`}</CodeBlock>
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

          <SubHeading>model.train() (YOLOv9)</SubHeading>
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
            YOLOX, YOLOv9, and RT-DETR use YOLO-style datasets configured via <InlineCode>data.yaml</InlineCode>. RF-DETR uses COCO-format annotations and is documented separately below.
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
