'use client'

import { useState, useEffect } from 'react'
import { useLocale, useTranslations } from 'next-intl'
import { motion, AnimatePresence } from 'framer-motion'
import {
  BookOpen, Terminal, Rocket, Layers, Crosshair, Grid3x3,
  GraduationCap, CheckCircle2, Upload, Cpu, FileCode, Wrench,
  Database, Copy, Check, Menu, X, ChevronRight,
  Sparkles, Tags, Video, Activity, Scissors, PersonStanding, Eye, SquareTerminal,
  ShieldCheck, Mountain, MapPin, Rotate3d, ScanSearch, Layers2, AlertTriangle,
} from 'lucide-react'

/* ─── Section metadata for sidebar ─── */
const sections = [
  { id: 'introduction', title: 'Introduction', icon: BookOpen },
  { id: 'compatibility', title: 'Compatibility', icon: CheckCircle2 },
  { id: 'installation', title: 'Installation', icon: Terminal },
  { id: 'quickstart', title: 'Quickstart', icon: Rocket },
  { id: 'models', title: 'Available Models', icon: Layers },
  { id: 'tasks', title: 'Tasks & Filenames', icon: Tags },
  { id: 'prediction', title: 'Prediction', icon: Crosshair },
  { id: 'tiled-inference', title: 'Tiled Inference', icon: Grid3x3 },
  { id: 'video-inference', title: 'Video Inference', icon: Video },
  { id: 'tracking', title: 'Tracking', icon: Activity },
  { id: 'segmentation', title: 'Segmentation', icon: Scissors },
  { id: 'obb', title: 'Oriented Boxes (OBB)', icon: Rotate3d },
  { id: 'pose', title: 'Pose Estimation', icon: PersonStanding },
  { id: 'gaze', title: 'Gaze Estimation', icon: Eye },
  { id: 'open-vocabulary', title: 'Open-Vocabulary Detection', icon: ScanSearch },
  { id: 'classification', title: 'Classification', icon: Tags },
  { id: 'depth', title: 'Depth Estimation', icon: Mountain },
  { id: 'point-localization', title: 'Point Localization', icon: MapPin },
  { id: 'training', title: 'Training', icon: GraduationCap },
  { id: 'lora', title: 'LoRA / DoRA', icon: Layers2 },
  { id: 'validation', title: 'Validation', icon: CheckCircle2 },
  { id: 'export', title: 'Export', icon: Upload },
  { id: 'torchscript-inference', title: 'TorchScript Inference', icon: Cpu },
  { id: 'onnx-inference', title: 'ONNX Inference', icon: Cpu },
  { id: 'tensorrt-inference', title: 'TensorRT Inference', icon: Cpu },
  { id: 'openvino-inference', title: 'OpenVINO Inference', icon: Cpu },
  { id: 'ncnn-inference', title: 'NCNN Inference', icon: Cpu },
  { id: 'coreml-inference', title: 'CoreML Inference', icon: Cpu },
  { id: 'cli', title: 'CLI', icon: SquareTerminal },
  { id: 'api-reference', title: 'API Reference', icon: FileCode },
  { id: 'architecture', title: 'Architecture Guide', icon: Wrench },
  { id: 'dataset-format', title: 'Dataset Format', icon: Database },
]

const docsVersions = [
  { version: 'v1.3.0', label: 'Latest', href: '/docs/v1.3.0' },
  { version: 'v1.2.0', label: 'Previous', href: '/docs/v1.2.0' },
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

function SupportBadge({ variant = 'experimental', children }) {
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

function Callout({ icon: Icon, tone = 'libre', title, children, className = '' }) {
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

function ValidatedModelHeader({ title, children }) {
  return (
    <div className="mt-10 mb-5 rounded-lg border border-emerald-500/30 bg-emerald-500/[0.08] dark:bg-emerald-500/[0.12] px-4 py-3">
      <div className="flex flex-wrap items-center gap-3">
        <h3 className="text-lg font-semibold text-surface-900 dark:text-white underline decoration-emerald-500 decoration-2 underline-offset-4">
          {title}
        </h3>
        <SupportBadge variant="validated">Recommended</SupportBadge>
      </div>
      <div className="mt-3 flex flex-wrap gap-2">
        {children}
      </div>
    </div>
  )
}

function MatrixMark({ value }) {
  if (value === 'yes') {
    return (
      <span className="font-semibold text-emerald-600 dark:text-emerald-400" aria-label="supported">
        ✓
      </span>
    )
  }

  if (value === 'exp') {
    return (
      <span className="font-semibold text-amber-600 dark:text-amber-400" aria-label="experimental">
        exp
      </span>
    )
  }

  if (value === 'preview') {
    return (
      <span className="font-semibold text-sky-600 dark:text-sky-400" aria-label="research preview">
        prev
      </span>
    )
  }

  return <span className="sr-only">Not currently supported</span>
}

function CompatibilityMatrix() {
  const rows = [
    {
      family: 'YOLO9', status: 'Validated detect, single GPU',
      inference: 'yes', training: 'yes',
      detect: 'yes', segment: '', semantic: '', classify: '', pose: '', obb: '', depth: '', point: '', gaze: '',
      onnx: 'yes', torchscript: 'yes', tensorrt: 'yes', openvino: 'yes', ncnn: 'yes', coreml: 'yes', tflite: 'exp',
    },
    {
      family: 'RF-DETR', status: 'Validated detect, segment, and pose; OBB experimental',
      inference: 'yes', training: 'yes',
      detect: 'yes', segment: 'yes', semantic: '', classify: '', pose: 'yes', obb: 'exp', depth: '', point: '', gaze: '',
      onnx: 'yes', torchscript: 'yes', tensorrt: 'yes', openvino: 'yes', ncnn: '', coreml: 'exp', tflite: 'exp',
    },
    {
      family: 'YOLOX', status: 'Experimental',
      inference: 'exp', training: 'exp',
      detect: 'exp', segment: '', semantic: '', classify: '', pose: '', obb: '', depth: '', point: '', gaze: '',
      onnx: 'exp', torchscript: 'exp', tensorrt: 'exp', openvino: 'exp', ncnn: 'exp', coreml: 'exp', tflite: '',
    },
    {
      family: 'YOLO9-E2E', status: 'Experimental',
      inference: 'exp', training: 'exp',
      detect: 'exp', segment: '', semantic: '', classify: '', pose: '', obb: '', depth: '', point: '', gaze: '',
      onnx: 'exp', torchscript: 'exp', tensorrt: 'exp', openvino: 'exp', ncnn: '', coreml: '', tflite: '',
    },
    {
      family: 'YOLO-NAS', status: 'Experimental',
      inference: 'exp', training: 'exp',
      detect: 'exp', segment: '', semantic: '', classify: '', pose: 'exp', obb: '', depth: '', point: '', gaze: '',
      onnx: 'exp', torchscript: 'exp', tensorrt: 'exp', openvino: 'exp', ncnn: 'exp', coreml: '', tflite: '',
    },
    {
      family: 'D-FINE', status: 'Experimental',
      inference: 'exp', training: 'exp',
      detect: 'exp', segment: '', semantic: '', classify: '', pose: '', obb: '', depth: '', point: '', gaze: '',
      onnx: 'exp', torchscript: 'exp', tensorrt: 'exp', openvino: 'exp', ncnn: 'exp', coreml: '', tflite: '',
    },
    {
      family: 'DEIM', status: 'Experimental',
      inference: 'exp', training: 'exp',
      detect: 'exp', segment: '', semantic: '', classify: '', pose: '', obb: '', depth: '', point: '', gaze: '',
      onnx: 'exp', torchscript: 'exp', tensorrt: 'exp', openvino: 'exp', ncnn: 'exp', coreml: '', tflite: '',
    },
    {
      family: 'DEIMv2', status: 'Experimental',
      inference: 'exp', training: 'exp',
      detect: 'exp', segment: '', semantic: '', classify: '', pose: '', obb: '', depth: '', point: '', gaze: '',
      onnx: 'exp', torchscript: 'exp', tensorrt: 'exp', openvino: 'exp', ncnn: 'exp', coreml: '', tflite: '',
    },
    {
      family: 'RT-DETR', status: 'Experimental',
      inference: 'exp', training: 'exp',
      detect: 'exp', segment: '', semantic: '', classify: '', pose: '', obb: '', depth: '', point: '', gaze: '',
      onnx: 'exp', torchscript: 'exp', tensorrt: 'exp', openvino: 'exp', ncnn: 'exp', coreml: 'exp', tflite: '',
    },
    {
      family: 'RT-DETRv2', status: 'Experimental',
      inference: 'exp', training: 'exp',
      detect: 'exp', segment: '', semantic: '', classify: '', pose: '', obb: '', depth: '', point: '', gaze: '',
      onnx: 'exp', torchscript: '', tensorrt: '', openvino: '', ncnn: '', coreml: '', tflite: '',
    },
    {
      family: 'RT-DETRv4', status: 'Experimental',
      inference: 'exp', training: 'exp',
      detect: 'exp', segment: '', semantic: '', classify: '', pose: '', obb: '', depth: '', point: '', gaze: '',
      onnx: 'exp', torchscript: '', tensorrt: '', openvino: '', ncnn: '', coreml: '', tflite: '',
    },
    {
      family: 'PicoDet', status: 'Experimental',
      inference: 'exp', training: 'exp',
      detect: 'exp', segment: '', semantic: '', classify: '', pose: '', obb: '', depth: '', point: '', gaze: '',
      onnx: 'exp', torchscript: 'exp', tensorrt: 'exp', openvino: '', ncnn: '', coreml: '', tflite: '',
    },
    {
      family: 'RTMDet', status: 'Experimental',
      inference: 'exp', training: 'exp',
      detect: 'exp', segment: '', semantic: '', classify: '', pose: '', obb: '', depth: '', point: '', gaze: '',
      onnx: 'exp', torchscript: '', tensorrt: '', openvino: '', ncnn: '', coreml: '', tflite: '',
    },
    {
      family: 'EC', status: 'Experimental',
      inference: 'exp', training: 'exp',
      detect: 'exp', segment: 'exp', semantic: '', classify: '', pose: 'exp', obb: '', depth: '', point: '', gaze: '',
      onnx: 'exp', torchscript: '', tensorrt: '', openvino: '', ncnn: '', coreml: '', tflite: '',
    },
    {
      family: 'DINOv2', status: 'New, experimental (needs transformers)',
      inference: 'exp', training: 'exp',
      detect: '', segment: '', semantic: 'exp', classify: 'exp', pose: '', obb: '', depth: '', point: '', gaze: '',
      onnx: '', torchscript: '', tensorrt: '', openvino: '', ncnn: '', coreml: '', tflite: '',
    },
    {
      family: 'MobileNetV4', status: 'New, experimental classifier (Apache)',
      inference: 'exp', training: 'exp',
      detect: '', segment: '', semantic: '', classify: 'exp', pose: '', obb: '', depth: '', point: '', gaze: '',
      onnx: 'exp', torchscript: '', tensorrt: '', openvino: '', ncnn: '', coreml: '', tflite: '',
    },
    {
      family: 'ConvNeXt', status: 'New, experimental classifier (Apache)',
      inference: 'exp', training: 'exp',
      detect: '', segment: '', semantic: '', classify: 'exp', pose: '', obb: '', depth: '', point: '', gaze: '',
      onnx: 'exp', torchscript: '', tensorrt: '', openvino: '', ncnn: '', coreml: '', tflite: '',
    },
    {
      family: 'EfficientNetV2', status: 'New, experimental classifier (Apache)',
      inference: 'exp', training: 'exp',
      detect: '', segment: '', semantic: '', classify: 'exp', pose: '', obb: '', depth: '', point: '', gaze: '',
      onnx: 'exp', torchscript: '', tensorrt: '', openvino: '', ncnn: '', coreml: '', tflite: '',
    },
    {
      family: 'Depth Anything V2', status: 'New, experimental; no export',
      inference: 'exp', training: '',
      detect: '', segment: '', semantic: '', classify: '', pose: '', obb: '', depth: 'exp', point: '', gaze: '',
      onnx: '', torchscript: '', tensorrt: '', openvino: '', ncnn: '', coreml: '', tflite: '',
    },
    {
      family: 'FOMO', status: 'New, experimental; no auto-download',
      inference: 'exp', training: 'exp',
      detect: '', segment: '', semantic: '', classify: '', pose: '', obb: '', depth: '', point: 'exp', gaze: '',
      onnx: '', torchscript: '', tensorrt: '', openvino: '', ncnn: '', coreml: '', tflite: '',
    },
    {
      family: 'L2CS', status: 'Experimental, inference-only',
      inference: 'exp', training: '',
      detect: '', segment: '', semantic: '', classify: '', pose: '', obb: '', depth: '', point: '', gaze: 'exp',
      onnx: '', torchscript: '', tensorrt: '', openvino: '', ncnn: '', coreml: '', tflite: '',
    },
  ]

  const headers = ['Model family', 'v1.3.0 status', 'Inference', 'Training', 'Detect', 'Segment', 'Semantic', 'Classify', 'Pose', 'OBB', 'Depth', 'Point', 'Gaze', 'ONNX', 'TorchScript', 'TensorRT', 'OpenVINO', 'NCNN', 'CoreML', 'TFLite']
  const columns = ['inference', 'training', 'detect', 'segment', 'semantic', 'classify', 'pose', 'obb', 'depth', 'point', 'gaze', 'onnx', 'torchscript', 'tensorrt', 'openvino', 'ncnn', 'coreml', 'tflite']

  return (
    <DocTable
      headers={headers}
      rows={rows.map((row) => [
        <strong key={`${row.family}-family`} className="text-surface-800 dark:text-white whitespace-nowrap">{row.family}</strong>,
        <span key={`${row.family}-status`} className="text-xs leading-relaxed">{row.status}</span>,
        ...columns.map((column) => <MatrixMark key={`${row.family}-${column}`} value={row[column]} />),
      ])}
    />
  )
}

function ValidationScopeCallout({ className = '' }) {
  return (
    <div className={`my-6 rounded-xl border border-emerald-500/30 bg-emerald-500/5 dark:bg-emerald-500/10 p-4 ${className}`}>
      <div className="flex items-start gap-3">
        <ShieldCheck className="w-5 h-5 text-emerald-600 dark:text-emerald-400 mt-0.5 shrink-0" />
        <div>
          <p className="font-semibold text-surface-900 dark:text-white mb-2">
            v1.3.0 validation scope
          </p>
          <p className="text-sm text-surface-600 dark:text-surface-400 mb-2">
            The heavily tested path is detection, training and inference for YOLO9 and RF-DETR, including RF-DETR segmentation.
          </p>
          <p className="text-sm text-surface-600 dark:text-surface-400">
            For production we recommend starting with YOLO9 or RF-DETR.
          </p>
        </div>
      </div>
    </div>
  )
}

function FlagshipCallout({ className = '' }) {
  return (
    <div className={`my-6 rounded-xl border border-libre-500/30 bg-libre-500/5 dark:bg-libre-500/10 p-4 ${className}`}>
      <div className="flex items-start gap-3">
        <Sparkles className="w-5 h-5 text-libre-600 dark:text-libre-400 mt-0.5 shrink-0" />
        <div>
          <p className="font-semibold text-surface-900 dark:text-white mb-1">
            Recommended validated path: YOLO9 detection or RF-DETR detection / segmentation
          </p>
          <p className="text-sm text-surface-600 dark:text-surface-400">
            Detection, training and inference for these models receive the heaviest testing. Treat other families, tasks, and multi-GPU workflows as experimental in v1.3.0.
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
  const locale = useLocale()
  const tNote = useTranslations('DocsNote')
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

  // On load, honor a #section deep link so a shared URL lands on that section.
  useEffect(() => {
    const id = decodeURIComponent((window.location.hash || '').replace(/^#/, ''))
    if (id && sections.some((s) => s.id === id)) {
      requestAnimationFrame(() => {
        document.getElementById(id)?.scrollIntoView()
        setActiveSection(id)
      })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const navigateTo = (id) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })
    setActiveSection(id)
    window.history.replaceState(null, '', `#${id}`)
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
        className="lg:hidden fixed top-16 left-4 z-30 inline-flex items-center gap-2 rounded-lg border border-surface-200 dark:border-white/[0.1] bg-white/90 dark:bg-surface-900/90 backdrop-blur px-3 py-2 text-sm font-semibold text-surface-700 dark:text-surface-200 shadow-sm hover:bg-white dark:hover:bg-surface-800 transition-colors"
        aria-label="Open documentation navigation"
      >
        <Menu className="w-4 h-4" />
        Menu
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
          {locale === 'zh' && (
            <div className="mb-6 rounded-lg border border-amber-500/30 bg-amber-500/5 px-4 py-3 text-sm text-surface-600 dark:text-surface-300">
              {tNote('text')}
            </div>
          )}
          <div className="mb-8 rounded-lg border border-surface-200 dark:border-white/[0.08] bg-white/80 dark:bg-white/[0.03] p-4 shadow-sm">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <p className="text-sm text-surface-600 dark:text-surface-400">
                    {isLatest
                    ? 'These are the latest LibreYOLO docs (v1.3.0). For the previous release, see v1.2.0.'
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
            <ValidationScopeCallout />
            <CodeBlock language="python">{`from libreyolo import LibreYOLO, SAMPLE_IMAGE

# Default: YOLO9 detection
model = LibreYOLO("LibreYOLO9c.pt")
result = model(SAMPLE_IMAGE, conf=0.25, save=True)

print(f"Detected {len(result)} objects")
print(result.boxes.xyxy)
print(result.saved_path)`}</CodeBlock>

          </motion.div>


          <Divider />

          {/* ────────────── COMPATIBILITY ────────────── */}
          <SectionHeading id="compatibility" icon={CheckCircle2}>Compatibility</SectionHeading>
          <P>
            Use this matrix as the quick v1.3.0 support map. <InlineCode>&#10003;</InlineCode>{' '}
            marks a validated path, <InlineCode>exp</InlineCode> is experimental,{' '}
            and empty cells are not currently supported. YOLO9 detection and
            RF-DETR detection, segmentation, and pose are the heavily tested paths;
            everything else, including the new classification, semantic, depth and
            point families, is experimental.
          </P>
          <CompatibilityMatrix />
          <p className="text-sm text-surface-600 dark:text-surface-400 leading-relaxed mb-4">
            Depth Anything V2 has no export path. TFLite export is experimental and
            limited to YOLO9 detection and RF-DETR detect / segment / pose. The
            classification families (MobileNetV4, ConvNeXt, EfficientNetV2) support
            ONNX export. CoreML exports produce <InlineCode>.mlpackage</InlineCode>{' '}
            bundles and require <InlineCode>libreyolo[coreml]</InlineCode>: macOS only,
            no INT8, and no embedded NMS for RF-DETR, D-FINE, DEIM, DEIMv2, or EC.
          </p>

          <Divider />

          <SectionHeading id="installation" icon={Terminal}>Installation</SectionHeading>
          <SubHeading>Requirements</SubHeading>
          <ul className="space-y-1.5 mb-4">
            <li className="flex items-center gap-2 text-surface-600 dark:text-surface-400">
              <span className="w-1.5 h-1.5 rounded-full bg-libre-400" />Python 3.10+
            </li>
            <li className="flex items-center gap-2 text-surface-600 dark:text-surface-400">
              <span className="w-1.5 h-1.5 rounded-full bg-libre-400" />PyTorch 2.4+ and torchvision 0.19+
            </li>
          </ul>

          <SubHeading>From PyPI</SubHeading>
          <CodeBlock language="bash">{`pip install libreyolo`}</CodeBlock>
          <P>
            v1.3.0 is on PyPI, so a plain pip install libreyolo installs the features documented on this page.
          </P>

          <SubHeading>From source</SubHeading>
          <CodeBlock language="bash">{`git clone https://github.com/LibreYOLO/libreyolo.git
cd libreyolo
git checkout dev
pip install -e .`}</CodeBlock>

          <SubHeading>Optional dependencies</SubHeading>
          <CodeBlock language="bash">{`# ONNX export and inference
pip install libreyolo[onnx]
# or: pip install onnx onnxsim onnxruntime

# RT-DETR compatibility extra (currently no extra packages)
pip install libreyolo[rtdetr]

# RF-DETR support
pip install libreyolo[rfdetr]
# or: pip install transformers

# TensorRT export and inference (NVIDIA GPU)
pip install libreyolo[tensorrt]
# Installs TensorRT CUDA 12 Python packages on Linux/Windows.
# Host driver/CUDA compatibility still matters.

# OpenVINO export and inference (Intel CPU/GPU/VPU)
pip install libreyolo[openvino]
# INT8 export also needs: pip install nncf

# NCNN export and inference
pip install libreyolo[ncnn]
# or: pip install pnnx ncnn

# ByteTrack API compatibility extra
pip install libreyolo[tracking]
# Tracking dependencies are part of the base install in v1.3.0.

# CoreML export and inference (macOS only for runtime)
pip install libreyolo[coreml]
# or: pip install coremltools

# L2CS gaze optional auto-download helper
pip install libreyolo[gaze]

# Install every optional LibreYOLO extra
pip install libreyolo[all]`}</CodeBlock>

          <P>If using <InlineCode>uv</InlineCode>, the most reliable path is an isolated venv per extra:</P>
          <CodeBlock language="bash">{`# ONNX environment
uv venv .venv-onnx
uv pip install --python .venv-onnx/bin/python -e '.[onnx]'

# RT-DETR environment
uv venv .venv-rtdetr
uv pip install --python .venv-rtdetr/bin/python -e '.[rtdetr]'

# Repeat with .[rfdetr], .[openvino], .[ncnn], .[coreml], .[gaze], .[tracking], or .[tensorrt] as needed`}</CodeBlock>
          <P>
            This avoids mutating the project environment and keeps optional dependencies isolated. Vendor-specific extras such as TensorRT, OpenVINO, NCNN, and CoreML may still require platform-specific native packages.
          </P>

          <Divider />

          {/* ────────────── QUICKSTART ────────────── */}
          <SectionHeading id="quickstart" icon={Rocket}>Quickstart</SectionHeading>
          <P>
            For the most tested path, pick single-GPU YOLO9 detection, RF-DETR detection, or RF-DETR segmentation. They load through the same factory, accept the same inputs, and return the same <InlineCode>Results</InlineCode> object, so you can swap between them without changing surrounding code.
          </P>

          <SubHeading>YOLO9 - CNN flagship</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO, SAMPLE_IMAGE

# Use the official checkpoint name and let the factory resolve the details
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
            LibreYOLO v1.3.0 ships two validated flagship families plus a broader
            catalogue of supported and freshly added models. Every model loads
            through the same <InlineCode>LibreYOLO()</InlineCode> factory, but only
            the validated paths below should be treated as heavily tested.
          </P>

          <ValidatedModelHeader title="YOLO9 - CNN flagship">
            <SupportBadge variant="validated">Default: LibreYOLO9c.pt</SupportBadge>
            <SupportBadge variant="validated">Heavily tested: detection, training and inference</SupportBadge>
            <SupportBadge>Detect-only in v1.3.0</SupportBadge>
            <SupportBadge>Experimental: multi-GPU</SupportBadge>
          </ValidatedModelHeader>
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
            YOLO9 is detection-only in v1.3.0. The non-detect flagship variants
            (including the old <InlineCode>-seg</InlineCode> checkpoints) were
            removed; for segmentation use RF-DETR or EdgeCrafter below.
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")   # detection`}</CodeBlock>

          <ValidatedModelHeader title="RF-DETR - transformer flagship">
            <SupportBadge variant="validated">Recommended transformer path</SupportBadge>
            <SupportBadge variant="validated">Heavily tested: detection, segmentation, pose, training and inference</SupportBadge>
            <SupportBadge>Experimental: OBB</SupportBadge>
            <SupportBadge>Experimental: multi-GPU</SupportBadge>
          </ValidatedModelHeader>
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
            LibreYOLO ships the Apache-clean RF-DETR detect sizes N/S/M/L on the
            Hugging Face org. The XL/2XL tiers are intentionally not shipped.
          </P>
          <P>
            <SupportBadge variant="validated">Heavily tested</SupportBadge>{' '}
            <strong className="text-surface-800 dark:text-white">Segmentation:</strong>{' '}
            <Checkpoints names={['LibreRFDETRn-seg.pt', 'LibreRFDETRs-seg.pt', 'LibreRFDETRm-seg.pt', 'LibreRFDETRl-seg.pt']} />.
            {' '}Larger <InlineCode>-seg</InlineCode> sizes (<InlineCode>x</InlineCode>, <InlineCode>xx</InlineCode>)
            pull upstream RF-DETR seg-XL / seg-2XL weights under a non-commercial
            license and are not hosted on the LibreYOLO org. See the{' '}
            <a href="#segmentation" className="text-libre-600 dark:text-libre-400 hover:underline">Segmentation</a> section.
          </P>
          <P>
            <SupportBadge variant="validated">Supported</SupportBadge>{' '}
            <strong className="text-surface-800 dark:text-white">Pose:</strong>{' '}
            <Checkpoints names={['LibreRFDETRx-pose.pt']} link={false} /> (ported
            from RF-DETR v1.8.0 GroupPose; only size <InlineCode>x</InlineCode> at
            576 ships).
          </P>
          <P>
            <SupportBadge>Experimental</SupportBadge>{' '}
            <strong className="text-surface-800 dark:text-white">OBB:</strong>{' '}
            <Checkpoints names={['LibreRFDETRn-obb.pt', 'LibreRFDETRs-obb.pt', 'LibreRFDETRm-obb.pt', 'LibreRFDETRl-obb.pt']} link={false} />{' '}
            (oriented boxes, uses detection input sizes).
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreRFDETRs.pt")           # detect (validated)
# model = LibreYOLO("LibreRFDETRs-seg.pt")     # segment (validated)
# model = LibreYOLO("LibreRFDETRx-pose.pt")    # pose  (size x)
# model = LibreYOLO("LibreRFDETRn-obb.pt")     # obb   (experimental)`}</CodeBlock>

          <SubHeading>Additional detection families</SubHeading>
          <P>
            Detection-capable families that share the same factory and API surface
            as the validated paths. These are experimental in v1.3.0. Each
            checkpoint name links to its model card on the{' '}
            <a href="https://huggingface.co/LibreYOLO" target="_blank" rel="noopener noreferrer" className="text-libre-600 dark:text-libre-400 hover:underline">LibreYOLO org</a>;
            pass any name to <InlineCode>LibreYOLO()</InlineCode> and the factory
            fetches it on first use.
          </P>
          <DocTable
            headers={['Family', 'Status', 'Tasks', 'Checkpoints']}
            rows={[
              ['YOLOX', <SupportBadge key="b">Experimental</SupportBadge>, 'detect', <Checkpoints key="yolox" names={['LibreYOLOXn.pt', 'LibreYOLOXt.pt', 'LibreYOLOXs.pt', 'LibreYOLOXm.pt', 'LibreYOLOXl.pt', 'LibreYOLOXx.pt']} />],
              ['YOLO9-E2E', <SupportBadge key="b">Experimental</SupportBadge>, 'detect', <Checkpoints key="y9e2e" names={['LibreYOLO9E2Et.pt', 'LibreYOLO9E2Es.pt', 'LibreYOLO9E2Em.pt', 'LibreYOLO9E2Ec.pt']} />],
              ['YOLO-NAS', <SupportBadge key="b">Experimental</SupportBadge>, 'detect, pose', <Checkpoints key="ynas" link={false} names={['LibreYOLONASs.pt', 'LibreYOLONASm.pt', 'LibreYOLONASl.pt', 'LibreYOLONASn-pose.pt', 'LibreYOLONASs-pose.pt', 'LibreYOLONASm-pose.pt', 'LibreYOLONASl-pose.pt']} />],
              ['D-FINE', <SupportBadge key="b">Experimental</SupportBadge>, 'detect', <Checkpoints key="dfine" names={['LibreDFINEn.pt', 'LibreDFINEs.pt', 'LibreDFINEm.pt', 'LibreDFINEl.pt', 'LibreDFINEx.pt']} />],
              ['DEIM', <SupportBadge key="b">Experimental</SupportBadge>, 'detect', <Checkpoints key="deim" names={['LibreDEIMn.pt', 'LibreDEIMs.pt', 'LibreDEIMm.pt', 'LibreDEIMl.pt', 'LibreDEIMx.pt']} />],
              ['DEIMv2', <SupportBadge key="b">Experimental</SupportBadge>, 'detect', <Checkpoints key="deimv2" names={['LibreDEIMv2atto.pt', 'LibreDEIMv2femto.pt', 'LibreDEIMv2pico.pt', 'LibreDEIMv2n.pt', 'LibreDEIMv2s.pt', 'LibreDEIMv2m.pt', 'LibreDEIMv2l.pt', 'LibreDEIMv2x.pt']} />],
              ['RT-DETR', <SupportBadge key="b">Experimental</SupportBadge>, 'detect', <Checkpoints key="rtdetr" names={['LibreRTDETRr18.pt', 'LibreRTDETRr34.pt', 'LibreRTDETRr50.pt', 'LibreRTDETRr50m.pt', 'LibreRTDETRr101.pt', 'LibreRTDETRl.pt', 'LibreRTDETRx.pt']} />],
              ['RT-DETRv2', <SupportBadge key="b">Experimental</SupportBadge>, 'detect', <Checkpoints key="rtdetrv2" names={['LibreRTDETRv2r18.pt', 'LibreRTDETRv2r34.pt', 'LibreRTDETRv2r50.pt', 'LibreRTDETRv2r50m.pt', 'LibreRTDETRv2r101.pt']} />],
              ['RT-DETRv4', <SupportBadge key="b">Experimental</SupportBadge>, 'detect', <Checkpoints key="rtdetrv4" names={['LibreRTDETRv4s.pt', 'LibreRTDETRv4m.pt', 'LibreRTDETRv4l.pt', 'LibreRTDETRv4x.pt']} />],
              ['PicoDet', <SupportBadge key="b">Experimental</SupportBadge>, 'detect', <Checkpoints key="picodet" names={['LibrePICODETs.pt', 'LibrePICODETm.pt', 'LibrePICODETl.pt']} />],
              ['RTMDet', <SupportBadge key="b">Experimental</SupportBadge>, 'detect', <Checkpoints key="rtmdet" names={['LibreRTMDett.pt', 'LibreRTMDets.pt', 'LibreRTMDetm.pt', 'LibreRTMDetl.pt', 'LibreRTMDetx.pt']} />],
              ['EdgeCrafter', <SupportBadge key="b">Experimental</SupportBadge>, 'detect, pose, segment', <Checkpoints key="ec" names={['LibreECs.pt', 'LibreECm.pt', 'LibreECl.pt', 'LibreECx.pt', 'LibreECs-pose.pt', 'LibreECm-pose.pt', 'LibreECl-pose.pt', 'LibreECx-pose.pt', 'LibreECs-seg.pt', 'LibreECm-seg.pt', 'LibreECl-seg.pt', 'LibreECx-seg.pt']} />],
            ]}
          />
          <P className="text-sm">
            <strong className="text-surface-800 dark:text-white">Hosting note:</strong>{' '}
            YOLO-NAS checkpoints (plain text above) are hosted on Deci&apos;s CDN
            under their proprietary weights license, not on the LibreYOLO Hugging
            Face org. The factory still downloads them automatically on first use.
            DAMO-YOLO was removed in v1.3.0 and is no longer loadable.
          </P>

          <SubHeading>New model families in v1.3.0</SubHeading>
          <P>
            v1.3.0 adds classification, dense semantic segmentation, monocular depth
            and point-localization families. They load through the same factory but
            are newly added and experimental. DINOv2 needs{' '}
            <InlineCode>pip install libreyolo[rfdetr]</InlineCode> (transformers).
          </P>
          <DocTable
            headers={['Family', 'Status', 'Task', 'Checkpoints']}
            rows={[
              ['MobileNetV4', <SupportBadge key="b">Experimental</SupportBadge>, 'classify', <Checkpoints key="mn4" names={['LibreMobileNetV4s-cls.pt', 'LibreMobileNetV4m-cls.pt', 'LibreMobileNetV4l-cls.pt']} />],
              ['ConvNeXt', <SupportBadge key="b">Experimental</SupportBadge>, 'classify', <Checkpoints key="cnx" link={false} names={['LibreConvNeXtt-cls.pt', 'LibreConvNeXts-cls.pt', 'LibreConvNeXtb-cls.pt']} />],
              ['EfficientNetV2', <SupportBadge key="b">Experimental</SupportBadge>, 'classify', <Checkpoints key="env2" link={false} names={['LibreEfficientNetV2b0-cls.pt', 'LibreEfficientNetV2b1-cls.pt', 'LibreEfficientNetV2b2-cls.pt', 'LibreEfficientNetV2b3-cls.pt']} />],
              ['DINOv2', <SupportBadge key="b">Experimental</SupportBadge>, 'semantic, classify', <Checkpoints key="dino" link={false} names={['LibreDINOv2n.pt', 'LibreDINOv2s.pt', 'LibreDINOv2m.pt', 'LibreDINOv2l.pt', 'LibreDINOv2n-cls.pt', 'LibreDINOv2s-cls.pt', 'LibreDINOv2m-cls.pt', 'LibreDINOv2l-cls.pt']} />],
              ['Depth Anything V2', <SupportBadge key="b">Experimental</SupportBadge>, 'depth', <Checkpoints key="depth" link={false} names={['LibreDepthAnythingV2s-depth.pt', 'LibreDepthAnythingV2b-depth.pt', 'LibreDepthAnythingV2l-depth.pt', 'LibreDepthAnythingV2g-depth.pt']} />],
              ['FOMO', <SupportBadge key="b">Experimental</SupportBadge>, 'point', <Checkpoints key="fomo" link={false} names={['LibreFOMOs-point.pt', 'LibreFOMOm-point.pt', 'LibreFOMOl-point.pt']} />],
            ]}
          />
          <ul className="space-y-2 my-4">
            <FeatureItem><strong className="text-surface-800 dark:text-white">MobileNetV4</strong> is the commercially clean classification path: Apache-2.0 ImageNet-1k weights (s/m/l at 224/224/256), with predict, top-1/top-5 validation, fine-tune training and ONNX export.</FeatureItem>
            <FeatureItem><strong className="text-surface-800 dark:text-white">ConvNeXt</strong> (V1 Tiny/Small/Base, 224) and <strong className="text-surface-800 dark:text-white">EfficientNetV2</strong> (b0-b3, 224-300) are additional Apache-2.0 ImageNet-1k classifiers (the accuracy tier).</FeatureItem>
            <FeatureItem><strong className="text-surface-800 dark:text-white">DINOv2</strong> is a DINOv2 backbone with a task head: dense semantic segmentation at 518 (default) and a classification linear probe at 224. It is not the RF-DETR detector. Classification was moved here from RF-DETR in v1.3.0.</FeatureItem>
            <FeatureItem><strong className="text-surface-800 dark:text-white">Depth Anything V2</strong> does monocular depth (sizes s/b/l/g, all at 518). ViT-S weights are Apache-2.0; ViT-B/L/G are CC-BY-NC-4.0 (non-commercial). Inference and zero-shot validation only: not trainable and with no export.</FeatureItem>
            <FeatureItem><strong className="text-surface-800 dark:text-white">FOMO</strong> is a point-localizer emitting <InlineCode>(x, y, class, confidence)</InlineCode> per object. Pretrained weights are not redistributed: pass a local checkpoint or train from scratch.</FeatureItem>
          </ul>
          <P className="text-sm">
            <strong className="text-surface-800 dark:text-white">Promptable and VLM tiers:</strong>{' '}
            LibreSAM (promptable segmentation, <InlineCode>libreyolo[sam]</InlineCode>)
            and the LibreVLM tier of vision-language detectors
            (<InlineCode>libreyolo[vlm]</InlineCode>) are separate categories that load
            upstream Hugging Face snapshots and are not routed through the{' '}
            <InlineCode>LibreYOLO()</InlineCode> detector factory. Their weights inherit
            each upstream model&apos;s license.
          </P>

          <SubHeading>Specialized models</SubHeading>
          <DocTable
            headers={['Family', 'Status', 'Tasks', 'Checkpoints']}
            rows={[
              ['L2CS', <SupportBadge key="b">Experimental</SupportBadge>, <span key="t">gaze (inference-only) - see <a href="#gaze" className="text-libre-600 dark:text-libre-400 hover:underline">Gaze Estimation</a></span>, <Checkpoints key="l2cs" link={false} names={['LibreL2CSr50.pt']} />],
            ]}
          />
          <P className="text-sm">
            L2CS architecture sizes include r18, r34, r50, r101, and r152, but the
            upstream-published Gaze360 checkpoint is ResNet-50. Install{' '}
            <InlineCode>libreyolo[gaze]</InlineCode> for the optional download helper,
            or pass a local checkpoint path for other sizes. L2CS weights are not
            hosted by LibreYOLO (the Gaze360 dataset license forbids redistribution).
          </P>

          <SubHeading>Factory function</SubHeading>
          <P>
            Use the <InlineCode>LibreYOLO()</InlineCode> factory for every model and
            runtime. Give it an official checkpoint name or exported artifact path,
            then let it choose the right model family, task, class count, and runtime:
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

# Default: YOLO9 detection
model = LibreYOLO("LibreYOLO9c.pt")

# Flagship transformer: RF-DETR
model = LibreYOLO("LibreRFDETRs.pt")
model = LibreYOLO("LibreRFDETRs-seg.pt")        # validated segmentation

# New in v1.3.0
model = LibreYOLO("LibreMobileNetV4s-cls.pt")   # classification (Apache, ImageNet-1k)
model = LibreYOLO("LibreDINOv2n.pt")            # semantic segmentation
model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")  # monocular depth
model = LibreYOLO("LibreFOMOs-point.pt")        # point localization (local weights)

# Exported deployment formats
model = LibreYOLO("model.onnx")                 # ONNX Runtime
model = LibreYOLO("model.engine")               # TensorRT
model = LibreYOLO("model.mlpackage")            # CoreML (macOS)
model = LibreYOLO("model_openvino/")            # OpenVINO (directory)
model = LibreYOLO("model_ncnn/")                # NCNN (directory)`}</CodeBlock>
          <P>
            For recognized official checkpoint filenames, LibreYOLO can auto-download
            missing weights. For custom filenames, point at an explicit local path.
            Keep new projects on YOLO9 detection or RF-DETR detection / segmentation;
            other families, tasks, and the new families are experimental in v1.3.0.
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
            headers={['Task', 'Canonical name', 'Filename suffix', 'Owned by']}
            rows={[
              ['Detection', <InlineCode key="d">&quot;detect&quot;</InlineCode>, '(none - implicit)', 'most families (default)'],
              ['Instance segmentation', <InlineCode key="s">&quot;segment&quot;</InlineCode>, <InlineCode key="ss">-seg</InlineCode>, 'RF-DETR, EdgeCrafter'],
              ['Semantic segmentation', <InlineCode key="se">&quot;semantic&quot;</InlineCode>, <InlineCode key="ses">-sem</InlineCode>, 'DINOv2'],
              ['Pose estimation', <InlineCode key="p">&quot;pose&quot;</InlineCode>, <InlineCode key="ps">-pose</InlineCode>, 'YOLO-NAS, EdgeCrafter, RF-DETR'],
              ['Oriented boxes', <InlineCode key="o">&quot;obb&quot;</InlineCode>, <InlineCode key="os">-obb</InlineCode>, 'RF-DETR (experimental)'],
              ['Classification', <InlineCode key="c">&quot;classify&quot;</InlineCode>, <InlineCode key="cs">-cls</InlineCode>, 'MobileNetV4, ConvNeXt, EfficientNetV2, DINOv2'],
              ['Monocular depth', <InlineCode key="de">&quot;depth&quot;</InlineCode>, <InlineCode key="des">-depth</InlineCode>, 'Depth Anything V2'],
              ['Point localization', <InlineCode key="pt">&quot;point&quot;</InlineCode>, <InlineCode key="pts">-point</InlineCode>, 'FOMO'],
              ['Gaze estimation', <InlineCode key="g">&quot;gaze&quot;</InlineCode>, <InlineCode key="gs">-gaze</InlineCode>, 'L2CS'],
            ]}
          />
          <P>
            Detection is implicit (no suffix), following the common YOLO convention.
            The factory accepts aliases at the API boundary
            (<InlineCode>&quot;detection&quot;</InlineCode>, <InlineCode>&quot;seg&quot;</InlineCode>,
            <InlineCode>&quot;keypoints&quot;</InlineCode>, <InlineCode>&quot;cls&quot;</InlineCode>, etc.);
            only the canonical names above appear in filenames. A task is available
            only when it is in that family&apos;s supported-task set.
          </P>

          <SubHeading>Resolution precedence</SubHeading>
          <P>
            When you load a model, the task is resolved in this order:
          </P>
          <CodeBlock language="text">{`explicit task=  →  checkpoint["task"]  →  filename suffix  →  family default`}</CodeBlock>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

# 1. Filename suffix decides → segment
model = LibreYOLO("LibreRFDETRs-seg.pt")

# 2. Override regardless of filename
model = LibreYOLO("custom_weights.pt", task="segment")

# 3. Detection is implicit
model = LibreYOLO("LibreYOLO9c.pt")  # task="detect"`}</CodeBlock>

          <SubHeading>Per-family task support</SubHeading>
          <DocTable
            headers={['Family', 'v1.3.0 status', 'Default', 'Supported tasks']}
            rows={[
              [<strong key="y9">YOLO9</strong>, 'detect single-GPU heavily tested; multi-GPU experimental', 'detect', 'detect'],
              [<strong key="rfd">RF-DETR</strong>, 'detect, segment, and pose single-GPU heavily tested; OBB experimental', 'detect', 'detect, segment, pose, obb'],
              ['YOLOX', 'experimental', 'detect', 'detect'],
              ['YOLO9-E2E', 'experimental', 'detect', 'detect'],
              ['YOLO-NAS', 'experimental', 'detect', 'detect, pose'],
              ['D-FINE / DEIM / DEIMv2', 'experimental', 'detect', 'detect'],
              ['RT-DETR / RT-DETRv2 / RT-DETRv4', 'experimental', 'detect', 'detect'],
              ['PicoDet / RTMDet', 'experimental', 'detect', 'detect'],
              ['EdgeCrafter (EC)', 'experimental', 'detect', 'detect, pose, segment'],
              ['DINOv2', 'new, experimental', 'semantic', 'semantic, classify'],
              ['MobileNetV4 / ConvNeXt / EfficientNetV2', 'new, experimental', 'classify', 'classify'],
              ['Depth Anything V2', 'new, experimental', 'depth', 'depth'],
              ['FOMO', 'new, experimental', 'point', 'point'],
              ['L2CS', 'experimental', 'gaze', 'gaze (inference-only)'],
            ]}
          />

          <SubHeading>Examples</SubHeading>
          <CodeBlock language="text">{`# Detection (implicit)
LibreYOLO9c.pt
LibreRFDETRs.pt
LibreRTDETRr50.pt

# Instance segmentation (-seg)
LibreRFDETRs-seg.pt
LibreECm-seg.pt

# Semantic segmentation (-sem)
LibreDINOv2n.pt          # semantic is DINOv2's default; -sem optional

# Pose (-pose)
LibreYOLONASn-pose.pt
LibreECs-pose.pt
LibreRFDETRx-pose.pt     # pose; size x

# Oriented boxes (-obb)
LibreRFDETRn-obb.pt      # obb; experimental

# Classification (-cls)
LibreMobileNetV4s-cls.pt
LibreConvNeXtt-cls.pt
LibreEfficientNetV2b0-cls.pt
LibreDINOv2n-cls.pt      # DINOv2 linear probe

# Depth (-depth)
LibreDepthAnythingV2s-depth.pt

# Point (-point)
LibreFOMOs-point.pt

# Gaze (-gaze optional; only task for L2CS)
LibreL2CSr50.pt`}</CodeBlock>

          <SubHeading>Deprecated aliases</SubHeading>
          <P>
            <InlineCode>LibreYOLORTDETR</InlineCode> and <InlineCode>LibreYOLORFDETR</InlineCode> are old names for <InlineCode>LibreRTDETR</InlineCode> and <InlineCode>LibreRFDETR</InlineCode> respectively. They still resolve with a <InlineCode>DeprecationWarning</InlineCode> - update imports when convenient.
          </P>

          <Divider />

          {/* ────────────── PREDICTION ────────────── */}
          <SectionHeading id="prediction" icon={Crosshair}>Prediction</SectionHeading>
          <P>
            The single-GPU prediction path is heavily tested for YOLO9 detection, RF-DETR detection, and RF-DETR segmentation. Other families and tasks use the same API but are experimental in v1.3.0.
          </P>

          <SubHeading>Basic prediction</SubHeading>
          <CodeBlock language="python">{`result = model("image.jpg")`}</CodeBlock>

          <SubHeading>All prediction parameters</SubHeading>
          <CodeBlock language="python">{`result = model(
    "image.jpg",
    conf=0.25,            # confidence threshold (default: 0.25)
    iou=0.45,             # NMS IoU threshold (default: 0.45)
    imgsz=640,            # input size override (default: model's native)
    device="auto",        # "auto", "cpu", "mps", "0", "cuda:0", ...
    classes=[0, 2, 5],    # filter to specific class IDs (default: all)
    max_det=300,          # max detections per image (default: 300)
    augment=False,        # test-time augmentation where implemented
    save=True,            # save annotated image (default: False)
    batch=4,              # directory batch size
    stream=False,         # video only: yield frame results instead of a list
    vid_stride=1,         # video only: process every N-th frame
    show=False,           # video only: display annotated frames
    tiling=False,         # large-image tiled detection
    overlap_ratio=0.2,    # tile overlap ratio
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
result = model("s3://bucket/image.jpg")
result = model("gs://bucket/image.jpg")

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

# BytesIO
from io import BytesIO
result = model(BytesIO(open("photo.jpg", "rb").read()))

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
# Tracking adds a track_id column before conf/cls.
result.boxes.data        # shape (N, 6), or (N, 7) when tracked

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

          <SubHeading>Batched in-memory inference</SubHeading>
          <P>
            New in v1.3.0: <InlineCode>model.predict()</InlineCode> accepts a list or tuple of
            in-memory images (NumPy arrays, PIL images, or tensors) and runs them as a true
            stacked-forward batch. Set <InlineCode>batch &gt; 1</InlineCode> to actually batch the
            forward pass on families that support it; a list of results is returned, one per input.
          </P>
          <CodeBlock language="python">{`import numpy as np
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")

frames = [
    np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
    np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
    np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
]

results = model(frames, batch=4)   # list/tuple -> true batched inference
for r in results:
    print(len(r), r.boxes.xyxy.shape)`}</CodeBlock>

          <SubHeading>Model info</SubHeading>
          <P>
            <InlineCode>model.info()</InlineCode> (new in v1.3.0) returns a JSON-friendly dict of
            family, size, task, parameter counts, input size, and class names, and logs a
            human-readable summary when <InlineCode>verbose=True</InlineCode>.
          </P>
          <CodeBlock language="python">{`meta = model.info(detailed=False, verbose=True)
# meta -> {"family": ..., "size": ..., "task": ..., "params": ..., "imgsz": ..., "names": {...}, ...}`}</CodeBlock>

          <Divider />

          {/* ────────────── TILED INFERENCE ────────────── */}
          <SectionHeading id="tiled-inference" icon={Grid3x3}>Tiled Inference</SectionHeading>
          <P>
            For images much larger than the model's input size (e.g., satellite imagery, drone footage), tiled inference splits the image into overlapping tiles, runs detection on each, and merges results.
          </P>
          <P>
            Tiling is detection-only in v1.3.0. It rejects segmentation masks, and it cannot be combined with <InlineCode>augment=True</InlineCode>.
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
            LibreYOLO ships two motion trackers that consume <InlineCode>Results</InlineCode> from any
            detector and add persistent track IDs: <strong className="text-surface-800 dark:text-white">ByteTrack</strong> (default)
            and <strong className="text-surface-800 dark:text-white">OC-SORT</strong> (new in v1.3.0), which is more robust to
            occlusion and non-linear motion. Tracking is most tested with single-GPU YOLO9 detection and
            RF-DETR detection; other detection families are experimental in v1.3.0.
          </P>

          <SubHeading>Install</SubHeading>
          <CodeBlock language="bash">{`pip install libreyolo[tracking]   # compatibility extra; tracking deps ship in base dev install`}</CodeBlock>

          <SubHeading>Video tracking helper</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")

for result in model.track(
    "clip.mp4",
    track_conf=0.25,
    iou=0.45,
    save=True,             # writes runs/track/<video_stem>.mp4 by default
    vid_stride=1,
):
    print(result.frame_idx, result.track_id)`}</CodeBlock>
          <P>
            <InlineCode>model.track()</InlineCode> is a generator for video files. It runs detection frame by frame, uses the lower ByteTrack confidence internally for recovery, and yields <InlineCode>Results</InlineCode> with <InlineCode>result.track_id</InlineCode> and <InlineCode>result.boxes.id</InlineCode> populated.
          </P>

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

          <SubHeading>OC-SORT (occlusion-robust)</SubHeading>
          <P>
            Select OC-SORT with <InlineCode>tracker=&quot;ocsort&quot;</InlineCode> on{' '}
            <InlineCode>model.track()</InlineCode>. ByteTrack stays the default. With OC-SORT,{' '}
            <InlineCode>track_conf</InlineCode> maps to the tracker&apos;s{' '}
            <InlineCode>det_thresh</InlineCode> (for ByteTrack it maps to{' '}
            <InlineCode>track_high_thresh</InlineCode>).
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")

for result in model.track(
    "clip.mp4",
    tracker="ocsort",      # "bytetrack" (default) or "ocsort"
    track_conf=0.25,       # maps to OC-SORT det_thresh
    iou=0.45,
    save=True,
):
    print(result.frame_idx, result.track_id)`}</CodeBlock>

          <P>
            Pass an <InlineCode>OCSortConfig</InlineCode> for full control. Supplying a config
            instance selects the tracker by type, so the <InlineCode>tracker=</InlineCode> string
            is then ignored.
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO, OCSortConfig

cfg = OCSortConfig(
    det_thresh=0.25,     # boxes above this drive association and spawn new tracks
    max_age=30,          # frames a track survives without an observation
    min_hits=3,          # consecutive hits before a track is reported
    iou_threshold=0.3,   # minimum IoU for a valid association
    delta_t=3,           # frame span used to estimate velocity direction
    inertia=0.2,         # weight of the velocity-direction (momentum) term
    use_byte=False,      # enable the BYTE low-score recovery pass
)

model = LibreYOLO("LibreYOLO9c.pt")
for result in model.track("clip.mp4", tracker_config=cfg, save=True):
    print(result.frame_idx, result.track_id)`}</CodeBlock>

          <Divider />

          {/* ────────────── SEGMENTATION ────────────── */}
          <SectionHeading id="segmentation" icon={Scissors}>Segmentation</SectionHeading>
          <ValidationScopeCallout />
          <P>
            RF-DETR segmentation is the segmentation path in v1.3.0 and is the heavily tested option. EdgeCrafter (<InlineCode>-seg</InlineCode>) also exposes a segmentation head but is experimental. YOLO9 no longer ships a segmentation head: it is detect-only as of v1.3.0.
          </P>

          <SubHeading>Run segmentation</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

# RF-DETR segmentation, the heavily tested segmentation path
model = LibreYOLO("LibreRFDETRs-seg.pt")
result = model("photo.jpg")

# EdgeCrafter segmentation is also available but experimental
# model = LibreYOLO("LibreECs-seg.pt")

# Segmentation returns boxes + masks
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
            RF-DETR segmentation uses the RF-DETR COCO-format training pipeline and is part of the heavily tested single-GPU scope. EdgeCrafter segmentation training is available but experimental. YOLO9 segmentation training was removed in v1.3.0.
          </P>

          <Divider />

          {/* ────────────── ORIENTED BOXES (OBB) ────────────── */}
          <SectionHeading id="obb" icon={Rotate3d}>Oriented Boxes (OBB)</SectionHeading>
          <div className="flex flex-wrap gap-2 mb-5">
            <SupportBadge variant="experimental">YOLO9: t, s, m, c</SupportBadge>
            <SupportBadge variant="experimental">RF-DETR: n, s, m, l</SupportBadge>
          </div>
          <P>
            Oriented boxes carry a rotation angle, which is what aerial imagery, documents, and densely packed scenes need. YOLO9 adds an angle branch to its detect head; RF-DETR adds a learnable angle embedding to its decoder.
          </P>

          <SubHeading>Inference and the OBB result</SubHeading>
          <P>
            Results expose an <InlineCode>obb</InlineCode> field. Angles are in <strong className="text-surface-800 dark:text-white">radians</strong>.
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9t-obb.pt")
r = model.predict("aerial.jpg")

for i in range(len(r.obb.cls)):
    cx, cy, w, h, angle = r.obb.xywhr[i]  # angle in radians
    corners = r.obb.xyxyxyxy[i]           # 4 (x, y) corner points
    conf, cls = r.obb.conf[i], r.obb.cls[i]`}</CodeBlock>
          <DocTable
            headers={['Field', 'Shape', 'Meaning']}
            rows={[
              [<InlineCode key="a">obb.xywhr</InlineCode>, 'N x 5', '[cx, cy, w, h, angle], angle in radians.'],
              [<InlineCode key="b">obb.xyxyxyxy</InlineCode>, 'N x 4 x 2', 'Four corner points per box.'],
              [<InlineCode key="c">obb.conf</InlineCode>, 'N', 'Confidence per box.'],
              [<InlineCode key="d">obb.cls</InlineCode>, 'N', 'Class id per box.'],
            ]}
          />

          <SubHeading>Dataset format and training</SubHeading>
          <P>
            OBB uses a standard detect-style data YAML, but labels are YOLO-OBB text files with <strong className="text-surface-800 dark:text-white">exactly nine fields</strong> per row: a class id followed by four normalized corner points. The angle is derived from the corners, not stored.
          </P>
          <CodeBlock language="text" filename="labels/aerial_001.txt">{`# class_id  x1 y1  x2 y2  x3 y3  x4 y4   (all normalized to [0, 1])
0  0.51 0.32  0.66 0.38  0.62 0.55  0.47 0.49
2  0.10 0.71  0.18 0.69  0.20 0.80  0.12 0.82`}</CodeBlock>
          <P>
            A plain detection checkpoint cannot be loaded directly into an OBB model. Going from detect to OBB is only allowed as a training warm-start: pass <InlineCode>pretrained=True</InlineCode> (YOLO9) or the explicit transfer flag on RF-DETR. Mosaic and mixup are disabled for OBB until corner-aware augmentation lands, and tiled inference is not supported.
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO9

model = LibreYOLO9(None, size="t", task="obb")
# Warm-start the backbone from a same-family detect checkpoint
result = model.train(data="dota8.yaml", pretrained=True, epochs=100, imgsz=640)

# CLI equivalent
# libreyolo train model=LibreYOLO9t.pt data=dota8.yaml --task obb`}</CodeBlock>
          <P>
            Validation uses rotated-IoU AP, reported as mAP50 and mAP50-95 under the OBB metric group.
          </P>

          <Divider />

          {/* ────────────── POSE ESTIMATION ────────────── */}
          <SectionHeading id="pose" icon={PersonStanding}>Pose Estimation</SectionHeading>
          <P>
            Pose (human keypoint) estimation runs on <InlineCode>YOLO-NAS (-pose)</InlineCode>,{' '}
            <InlineCode>EdgeCrafter (-pose)</InlineCode>, and, new in v1.3.0,{' '}
            <InlineCode>RF-DETR (-pose)</InlineCode>. Each pose model is single-class
            (&quot;person&quot;) with 17 COCO keypoints.
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

          <P>
            <SupportBadge variant="validated">Supported</SupportBadge>{' '}
            RF-DETR pose ships a single checkpoint at size <InlineCode>x</InlineCode> only:{' '}
            <InlineCode>LibreRFDETRx-pose.pt</InlineCode>.
          </P>
          <CodeBlock language="python">{`# RF-DETR pose (size x only)
model = LibreYOLO("LibreRFDETRx-pose.pt")
result = model("people.jpg")
print(result.keypoints.xy.shape)  # (N, 17, 2)`}</CodeBlock>

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
            Pose training is supported for YOLO-NAS; EdgeCrafter pose is currently inference-only. RF-DETR pose ships the size <InlineCode>x</InlineCode> checkpoint only. YOLO9 is detect-only and ships no pose checkpoints.
          </P>

          <Divider />

          {/* ────────────── GAZE ESTIMATION ────────────── */}
          <SectionHeading id="gaze" icon={Eye}>Gaze Estimation</SectionHeading>
          <P>
            Gaze direction estimation is provided by the <InlineCode>LibreL2CS</InlineCode> family, an L2CS-Net port with a ResNet trunk and two angle-bin classification heads. It is a two-stage model: an upstream face detector locates faces, then the gaze head predicts per-face pitch and yaw in radians. It is inference-only and experimental in v1.3.0.
          </P>

          <SubHeading>Install</SubHeading>
          <CodeBlock language="bash">{`pip install libreyolo[gaze]   # optional Google Drive helper for Gaze360 weights`}</CodeBlock>
          <P>
            The published L2CS ResNet-50 weights are trained on Gaze360 and are not mirrored by LibreYOLO. Without the optional helper, pass a local checkpoint path or follow the manual download instructions printed by <InlineCode>LibreL2CS</InlineCode>.
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
    print(f"face {i}: pitch={pitch_deg:.1f} deg, yaw={yaw_deg:.1f} deg")`}</CodeBlock>

          <P>
            From the CLI: <InlineCode>libreyolo predict model=LibreL2CSr50.pt source=portrait.jpg --face-detector path/to/face.pt</InlineCode>.
          </P>

          <Divider />

          {/* ────────────── OPEN-VOCABULARY DETECTION (LibreVLM) ────────────── */}
          <SectionHeading id="open-vocabulary" icon={ScanSearch}>Open-Vocabulary Detection</SectionHeading>
          <div className="flex flex-wrap gap-2 mb-5">
            <SupportBadge variant="experimental">Inference only</SupportBadge>
            <SupportBadge variant="experimental">Python API</SupportBadge>
          </div>
          <P>
            A classic detector ships with a fixed list of classes baked into its head. <InlineCode>LibreVLM</InlineCode> throws that constraint away: it wraps modern instruction-tuned vision-language models, prompts them to emit bounding boxes, parses the generated text, and returns the same <InlineCode>Results</InlineCode> object you already use for YOLO9 and RF-DETR. The class list is just a list of words you supply at runtime, so adding a new category costs nothing and works zero-shot.
          </P>
          <ul className="space-y-2 mb-4">
            <FeatureItem><strong className="text-surface-800 dark:text-white">Open vocabulary.</strong> Detect <InlineCode>&quot;pink car&quot;</InlineCode>, <InlineCode>&quot;license plate&quot;</InlineCode>, or <InlineCode>&quot;the small island&quot;</InlineCode> without ever training a head for them.</FeatureItem>
            <FeatureItem><strong className="text-surface-800 dark:text-white">One factory, one contract.</strong> <InlineCode>LibreVLM(...)</InlineCode> returns the standard <InlineCode>Results</InlineCode> with <InlineCode>boxes.xyxy</InlineCode>, <InlineCode>boxes.cls</InlineCode>, <InlineCode>boxes.conf</InlineCode>, plus <InlineCode>.plot()</InlineCode> and <InlineCode>.save()</InlineCode>.</FeatureItem>
            <FeatureItem><strong className="text-surface-800 dark:text-white">Swappable backends.</strong> Six model families behind one alias string, from a 230M Florence-2 to an 8B Qwen3-VL.</FeatureItem>
            <FeatureItem><strong className="text-surface-800 dark:text-white">A raw escape hatch.</strong> <InlineCode>chat()</InlineCode> gives you free-form image question answering when you need more than boxes.</FeatureItem>
          </ul>
          <Callout icon={AlertTriangle} tone="amber" title="Inference-only tier">
            <p>
              LibreVLM is a Python-only inference tier: there is no training, validation, export, or CLI path yet, and confidence scores are placeholders. Read the limitations at the end of this section before you build on top of it.
            </p>
          </Callout>

          <SubHeading>Installation</SubHeading>
          <P>
            LibreVLM lives behind the optional <InlineCode>vlm</InlineCode> extra. It pulls in a recent <InlineCode>transformers</InlineCode> and the helpers a couple of processors need.
          </P>
          <CodeBlock language="bash">{`pip install 'libreyolo[vlm]'`}</CodeBlock>
          <P>
            Weights download from the Hugging Face Hub on first use into a local <InlineCode>weights/</InlineCode> folder. A GPU is recommended for the larger backends, but every model also runs on CPU with <InlineCode>device=&quot;cpu&quot;</InlineCode>.
          </P>

          <SubHeading>Quickstart</SubHeading>
          <P>
            Construct a model, declare the words you care about, and predict. The default backend is Qwen3-VL-4B, the strongest detector in the tier and Apache-2.0 licensed.
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreVLM

# Qwen3-VL-4B by default; weights autodownload on first use
model = LibreVLM()

# The vocabulary is just words. Any words.
model.set_classes(["pink car", "wheel"])

result = model.predict("street.jpg")

print(result.boxes.xyxy)   # pixel [x1, y1, x2, y2]
print(result.boxes.cls)    # ids into ["pink car", "wheel"]
result.plot()              # same drawing helpers as any LibreYOLO model
result.save("out.jpg")`}</CodeBlock>

          <SubHeading>Supported backends</SubHeading>
          <P>
            Pick a backend with the alias you pass to <InlineCode>LibreVLM(...)</InlineCode>. A bare family name resolves to its default size. The strongest detectors are <strong className="text-surface-800 dark:text-white">Qwen3-VL</strong>, <strong className="text-surface-800 dark:text-white">LFM2-VL</strong>, and <strong className="text-surface-800 dark:text-white">Florence-2</strong>.
          </P>
          <DocTable
            headers={['Family', 'Alias', 'Sizes (params)', 'License', 'Notes']}
            rows={[
              [
                <strong key="q" className="text-surface-800 dark:text-white whitespace-nowrap">Qwen3-VL</strong>,
                <code key="qa" className="font-mono text-xs">qwen3-vl-2b / -4b / -8b</code>,
                '2B / 4B / 8B',
                'Apache-2.0',
                <span key="qn">Default and strongest. Recommended starting point.</span>,
              ],
              [
                <strong key="l" className="text-surface-800 dark:text-white whitespace-nowrap">LFM2-VL</strong>,
                <code key="la" className="font-mono text-xs">lfm2-vl-450m / -1.6b</code>,
                '450M / 1.6B',
                'LFM Open License',
                <span key="ln">Edge sized, surprisingly strong small detector. Notice gated.</span>,
              ],
              [
                <strong key="i" className="text-surface-800 dark:text-white whitespace-nowrap">InternVL3</strong>,
                <code key="ia" className="font-mono text-xs">internvl3-1b / -2b / -8b</code>,
                '1B / 2B / 8B',
                'Qwen License',
                <span key="in">Good grounding at 8B; small sizes are weak. Notice gated.</span>,
              ],
              [
                <strong key="f" className="text-surface-800 dark:text-white whitespace-nowrap">Florence-2</strong>,
                <code key="fa" className="font-mono text-xs">florence-2-base / -large</code>,
                '0.23B / 0.77B',
                'MIT',
                <span key="fn">Purpose-built grounding model. Tight boxes, no <InlineCode>chat()</InlineCode>.</span>,
              ],
              [
                <strong key="s" className="text-surface-800 dark:text-white whitespace-nowrap">SmolVLM2</strong>,
                <code key="sa" className="font-mono text-xs">smolvlm2-500m / -2.2b</code>,
                '500M / 2.2B',
                'Apache-2.0',
                <span key="sn">Tiny and fast; weaker detector. Good for quick trials.</span>,
              ],
              [
                <strong key="k" className="text-surface-800 dark:text-white whitespace-nowrap">Kosmos-2</strong>,
                <code key="ka" className="font-mono text-xs">kosmos-2</code>,
                '~1.6B',
                'MIT',
                <span key="kn">2023 grounder. Coarser boxes, no <InlineCode>chat()</InlineCode>.</span>,
              ],
            ]}
          />
          <ul className="space-y-2 mb-4">
            <FeatureItem><strong className="text-surface-800 dark:text-white">Best quality:</strong> <InlineCode>qwen3-vl-8b</InlineCode> or <InlineCode>qwen3-vl-4b</InlineCode> (the default).</FeatureItem>
            <FeatureItem><strong className="text-surface-800 dark:text-white">Tight boxes, small footprint:</strong> <InlineCode>florence-2-large</InlineCode>.</FeatureItem>
            <FeatureItem><strong className="text-surface-800 dark:text-white">Edge / CPU:</strong> <InlineCode>lfm2-vl-450m</InlineCode> or <InlineCode>smolvlm2-500m</InlineCode>.</FeatureItem>
            <FeatureItem><strong className="text-surface-800 dark:text-white">Fully permissive license:</strong> any Qwen3-VL, SmolVLM2, Florence-2 or Kosmos-2 size.</FeatureItem>
          </ul>
          <Callout icon={ShieldCheck} tone="emerald" title="Licensing">
            <p>
              Qwen3-VL and SmolVLM2 are Apache-2.0; Florence-2 and Kosmos-2 are MIT. LFM2-VL and InternVL3 carry non-OSI licenses and emit a one-time notice before their first download, so you can make an informed choice for commercial use.
            </p>
          </Callout>

          <SubHeading>Setting the vocabulary</SubHeading>
          <P>
            The vocabulary is the heart of open-vocabulary detection. Call <InlineCode>set_classes()</InlineCode> with a list of label strings. It is sticky: it persists across every later <InlineCode>predict()</InlineCode> and <InlineCode>track()</InlineCode> call until you set it again. It returns <InlineCode>self</InlineCode>, so it chains.
          </P>
          <CodeBlock language="python">{`# Sticky and chainable
model = LibreVLM("qwen3-vl-2b").set_classes(["person", "dog", "cat"])

# Set it once at construction instead
model = LibreVLM("lfm2-vl-450m", names=["boat"], device="cpu")

# Re-set any time to change what you are looking for
model.set_classes(["a red car", "a blue truck"])`}</CodeBlock>
          <P>
            Labels can be any phrase. They must be unique case-insensitively, and you must pass a list, not a bare string. If you never call <InlineCode>set_classes()</InlineCode>, the model falls back to the COCO-80 vocabulary so a bare <InlineCode>predict()</InlineCode> still does something sensible.
          </P>

          <SubHeading>Prediction and results</SubHeading>
          <P>
            <InlineCode>predict()</InlineCode> (and the equivalent <InlineCode>model(...)</InlineCode> call) accepts the same source types as any LibreYOLO detector: a path, a PIL image, a numpy array, a URL, a folder, or a video. <InlineCode>stream=True</InlineCode> and <InlineCode>track()</InlineCode> work too.
          </P>
          <CodeBlock language="python">{`result = model.predict(
    source="image.jpg",  # path | PIL | ndarray | URL | folder | video
    conf=0.25,           # see note below: scoring is synthetic
    classes=[0],         # optional: keep only these vocabulary ids
    max_det=300,
)`}</CodeBlock>
          <DocTable
            headers={['Field', 'Shape / type', 'Meaning']}
            rows={[
              [<InlineCode key="a">result.boxes.xyxy</InlineCode>, 'N x 4', 'Pixel boxes [x1, y1, x2, y2], scaled to the original image.'],
              [<InlineCode key="b">result.boxes.cls</InlineCode>, 'N', 'Class ids indexing into your set_classes() vocabulary.'],
              [<InlineCode key="c">result.boxes.conf</InlineCode>, 'N', 'Synthetic confidence: 1.0 for every box (see Limitations).'],
              [<InlineCode key="d">result.plot() / .save()</InlineCode>, '-', 'The usual drawing and saving helpers.'],
            ]}
          />
          <P>
            Under the hood, LibreVLM tolerantly parses the model output (handling markdown fences, stray prose, duplicated boxes, and truncated arrays), maps free-text labels back to your class ids, and drops any label that is not in your vocabulary. That last step is what makes a free-form generator behave like a closed-set detector.
          </P>

          <SubHeading>Examples</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreVLM

# Detect a specific colored object
model = LibreVLM("qwen3-vl-4b")
model.set_classes(["red car"])
result = model.predict("parking_lot.jpg")
print(f"Found {len(result.boxes.cls)} red car(s)")
result.save("red_cars.jpg")`}</CodeBlock>
          <CodeBlock language="python">{`# Tight boxes with Florence-2 (a purpose-built grounder)
model = LibreVLM("florence-2-large")
model.set_classes(["a red car", "license plate"])
result = model.predict("car.jpg")
result.plot()`}</CodeBlock>
          <CodeBlock language="python">{`# Run on CPU with a built-in sample image
from libreyolo import LibreVLM, SAMPLE_IMAGE

model = LibreVLM("lfm2-vl-450m", device="cpu")
# No set_classes() -> falls back to the COCO-80 vocabulary
result = model.predict(SAMPLE_IMAGE)
print(model.names[result.boxes.cls[0]])  # e.g. "person"`}</CodeBlock>
          <CodeBlock language="python">{`# Batches, folders, and video
model = LibreVLM().set_classes(["forklift", "pallet"])

# A whole folder
for result in model.predict("warehouse_frames/", stream=True):
    result.save()

# A video file (frames are processed one at a time)
model.predict("warehouse.mp4", save=True)`}</CodeBlock>

          <SubHeading>Raw chat</SubHeading>
          <P>
            Sometimes you want the model, not the detector. The chat-template families expose <InlineCode>chat()</InlineCode>, which takes an image and a free-form prompt and returns the decoded text verbatim. Use it for counting, captioning, or quick visual questions.
          </P>
          <CodeBlock language="python">{`model = LibreVLM("qwen3-vl-4b")

answer = model.chat("harbor.jpg", "How many boats are docked? Answer with a number.")
print(answer)`}</CodeBlock>
          <Callout icon={AlertTriangle} tone="amber">
            <p>
              <InlineCode>chat()</InlineCode> is available on the chat-template families (Qwen3-VL, LFM2-VL, SmolVLM2, InternVL3). Florence-2 and Kosmos-2 are task-token grounders and raise <InlineCode>NotImplementedError</InlineCode>; use <InlineCode>predict()</InlineCode> with them.
            </p>
          </Callout>

          <SubHeading>How backends differ</SubHeading>
          <P>
            Every family returns the same <InlineCode>Results</InlineCode>, but they reach it differently. The chat families are prompted for a JSON array of boxes; the grounders use dedicated task tokens.
          </P>
          <DocTable
            headers={['Family', 'Prompting', 'Coordinate space', 'chat()']}
            rows={[
              ['Qwen3-VL', 'JSON box prompt', '0 to 1000, rescaled', 'Yes'],
              ['LFM2-VL', 'JSON box prompt', 'Normalized 0 to 1', 'Yes'],
              ['SmolVLM2', 'JSON box prompt', 'Normalized 0 to 1', 'Yes'],
              ['InternVL3', 'JSON box prompt', '0 to 1000, rescaled', 'Yes'],
              ['Florence-2', 'Task token', 'Native pixels', 'No'],
              ['Kosmos-2', 'Grounding prompt', 'Normalized, rescaled', 'No'],
            ]}
          />
          <P>
            For the chat families you can override the detection prompt with the <InlineCode>prompt=</InlineCode> constructor argument, and cap generation length with <InlineCode>max_new_tokens=</InlineCode>. Device and dtype are resolved automatically: bf16 or fp16 on CUDA, fp32 on CPU.
          </P>

          <SubHeading>Limitations</SubHeading>
          <ul className="space-y-2 mb-4">
            <FeatureItem><strong className="text-surface-800 dark:text-white">Synthetic confidence.</strong> Every box is scored 1.0. The <InlineCode>conf=</InlineCode> filter therefore behaves as all-or-nothing rather than a real threshold.</FeatureItem>
            <FeatureItem><strong className="text-surface-800 dark:text-white">No mAP / validation.</strong> <InlineCode>val()</InlineCode> raises, because synthetic scores would make COCO mAP misleading.</FeatureItem>
            <FeatureItem><strong className="text-surface-800 dark:text-white">No training or export.</strong> <InlineCode>train()</InlineCode> and <InlineCode>export()</InlineCode> raise. Fine-tune the VLM upstream and load the resulting weights instead.</FeatureItem>
            <FeatureItem><strong className="text-surface-800 dark:text-white">Tracking is degraded.</strong> <InlineCode>track()</InlineCode> runs, but uniform scores make the tracker&apos;s low-confidence recovery stage inert.</FeatureItem>
            <FeatureItem><strong className="text-surface-800 dark:text-white">One image at a time.</strong> Generation is sequential in v1, so larger <InlineCode>batch=</InlineCode> values give no speedup.</FeatureItem>
            <FeatureItem><strong className="text-surface-800 dark:text-white">Python API only.</strong> The <InlineCode>libreyolo</InlineCode> CLI does not resolve VLM aliases yet.</FeatureItem>
          </ul>
          <Callout icon={Eye} tone="libre" title="Where it shines">
            <p>
              Use LibreVLM when the class set is open ended, changes often, or is hard to label up front: rapid prototyping, long-tail or rare categories, and &quot;find the thing I describe in words&quot; workflows. When you need calibrated confidence, throughput, or a deployable artifact, train a closed-vocabulary YOLO9 or RF-DETR with the <a href="#training" className="text-libre-600 dark:text-libre-400 hover:underline">Training</a> section above.
            </p>
          </Callout>

          <Divider />

          {/* ────────────── CLASSIFICATION ────────────── */}
          <SectionHeading id="classification" icon={Tags}>Classification</SectionHeading>
          <P>
            New in v1.3.0: whole-image classification. Two families ship, and they target different needs. <InlineCode>LibreMobileNetV4</InlineCode> is the production classifier (Apache-2.0 ImageNet-1k weights, exportable to ONNX). <InlineCode>LibreDINOv2</InlineCode> with <InlineCode>task=classify</InlineCode> is a DINOv2 backbone plus linear probe, ideal for transfer learning, but its published weights are demo-grade and it cannot export yet. This is a pre-release task, so details may change before launch.
          </P>

          <DocTable
            headers={['Family', 'Checkpoints', 'Input', 'Weights', 'Fine-tune', 'ONNX export']}
            rows={[
              ['LibreMobileNetV4', 'LibreMobileNetV4{s,m,l}-cls.pt', '224 / 224 / 256', 'Apache-2.0 ImageNet-1k (production)', 'Cross-entropy', 'Yes'],
              ['LibreDINOv2 (classify)', 'LibreDINOv2{n,s,m,l}-cls.pt', '224', 'Imagenette demo-grade (10 classes)', 'Linear probe', 'Not supported'],
            ]}
          />

          <SubHeading>LibreMobileNetV4 (production classifier)</SubHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="validated">Apache-2.0 ImageNet-1k weights</SupportBadge>
            <SupportBadge variant="experimental">New in v1.3.0</SupportBadge>
          </div>
          <P>
            A native MobileNetV4-conv port (derived from timm) whose 1000-class ImageNet-1k weights load bit-identically. Sizes <InlineCode>s</InlineCode> / <InlineCode>m</InlineCode> run at 224, <InlineCode>l</InlineCode> at 256. Checkpoints:
          </P>
          <Checkpoints names={['LibreMobileNetV4s-cls.pt', 'LibreMobileNetV4m-cls.pt', 'LibreMobileNetV4l-cls.pt']} />

          <P>Load and predict. A single image returns one <InlineCode>Results</InlineCode>; read <InlineCode>.probs</InlineCode> directly off it (pass a list to get a list back).</P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

# MobileNetV4-conv-Small, Apache-2.0 ImageNet-1k weights (auto-downloaded if missing)
model = LibreYOLO("LibreMobileNetV4s-cls.pt")
result = model("cat.jpg")            # single image -> one Results

probs = result.probs                 # whole-image class vector, length = num classes
print(probs.top1, probs.top1conf)    # top-1 class id (int) and its confidence
print(probs.top5, probs.top5conf)    # 5 class ids and 5 confidences
print(result.names[probs.top1])      # human-readable class name`}</CodeBlock>

          <P>Fine-tune to a custom class set (ImageFolder layout). The head is rebuilt to the dataset class count automatically; the ImageNet-pretrained backbone transfers cleanly.</P>
          <CodeBlock language="python">{`from libreyolo import LibreMobileNetV4

model = LibreMobileNetV4(size="s")   # ImageNet-pretrained backbone
model.train(
    data="imagenette160",            # known name, dataset root, or .zip URL
    epochs=5,
    batch=64,
    lr0=1e-3,                        # AdamW + cosine, 1-epoch warmup
    imgsz=224,
)`}</CodeBlock>

          <P>Validate (top-1 / top-5 accuracy):</P>
          <CodeBlock language="python">{`model = LibreYOLO("LibreMobileNetV4s-cls.pt")
metrics = model.val(data="imagenette160")
print(metrics["metrics/accuracy_top1"])
print(metrics["metrics/accuracy_top5"])`}</CodeBlock>

          <P>Export to ONNX (verified bit-exact against eager). The ONNX graph emits a single logits tensor.</P>
          <CodeBlock language="python">{`model = LibreYOLO("LibreMobileNetV4s-cls.pt")
path = model.export(format="onnx", imgsz=224)   # single output: logits [batch, num_classes]

# Interop note: the ONNX output is RAW LOGITS, not softmaxed. The PyTorch
# predict path applies softmax for you; non-Python consumers must apply it
# themselves before reading probabilities.`}</CodeBlock>

          <SubHeading>LibreDINOv2 classify (linear probe / transfer)</SubHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="experimental">Demo-grade weights (Imagenette)</SupportBadge>
            <SupportBadge variant="experimental">No export</SupportBadge>
          </div>
          <P>
            A frozen-style DINOv2-S encoder with a trainable linear head, run at 224. The <InlineCode>n</InlineCode> / <InlineCode>s</InlineCode> / <InlineCode>m</InlineCode> / <InlineCode>l</InlineCode> sizes control only the projector width: all four share the same DINOv2-S encoder, so the published checkpoints land at near-identical accuracy. The shipped <InlineCode>-cls</InlineCode> weights are demo-grade (trained on Imagenette, 10 classes), so treat this family as the transfer-learning option, not a drop-in 1000-class classifier. Checkpoints:
          </P>
          <Checkpoints names={['LibreDINOv2n-cls.pt', 'LibreDINOv2s-cls.pt', 'LibreDINOv2m-cls.pt', 'LibreDINOv2l-cls.pt']} link={false} />

          <P>Load and predict (same <InlineCode>Probs</InlineCode> surface as MobileNetV4):</P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreDINOv2s-cls.pt")   # DINOv2-S backbone + linear probe (224)
result = model("springer.jpg")
print(result.probs.top1, result.probs.top1conf)`}</CodeBlock>

          <P>
            Fine-tune for transfer. Build a fresh model with <InlineCode>task=&quot;classify&quot;</InlineCode> for a brand-new head, or load a shipped <InlineCode>-cls</InlineCode> checkpoint and continue training. For the best accuracy, fine-tune from a shipped checkpoint rather than a fresh head, and keep the default <InlineCode>lr=1e-4</InlineCode> (higher learning rates converge worse).
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreDINOv2

# Fresh DINOv2 backbone + random linear head, sized to the dataset
model = LibreDINOv2(size="s", task="classify", nb_classes=3)
model.train(data="path/to/imagefolder", epochs=5, lr=1e-4, batch=4)

# Validate the same way (top-1 / top-5)
metrics = model.val(data="path/to/imagefolder")
print(metrics["metrics/accuracy_top1"])`}</CodeBlock>

          <P>Export is not implemented for LibreDINOv2. If you need an exportable classifier, use LibreMobileNetV4.</P>
          <CodeBlock language="python">{`model = LibreYOLO("LibreDINOv2s-cls.pt")
model.export(format="onnx")
# raises NotImplementedError: Export is not yet implemented for LibreDINOv2.`}</CodeBlock>

          <SubHeading>Dataset layout (both families)</SubHeading>
          <P>
            Classification uses an ImageNet-style ImageFolder tree (folders, not label files). Class index is assigned by sorted folder name. <InlineCode>data=</InlineCode> accepts a dataset root, a known name (e.g. <InlineCode>imagenette160</InlineCode>), or a <InlineCode>.zip</InlineCode> URL.
          </P>
          <CodeBlock language="text">{`dataset_root/
  train/                # required; one subfolder per class
    class_a/img001.jpg
    class_a/img002.jpg
    class_b/img003.jpg
  val/                  # required for validation; same class folders as train
    class_a/img010.jpg
    class_b/img011.jpg`}</CodeBlock>

          <SubHeading>Results.probs reference</SubHeading>
          <CodeBlock language="python">{`probs = result.probs        # Probs payload, 1-D vector of length = num classes
probs.data                  # raw tensor / ndarray of class probabilities
probs.top1                  # int   - argmax class id
probs.top5                  # list  - 5 class ids, highest first
probs.top1conf              # float - confidence of the top-1 class
probs.top5conf              # 5 confidences, aligned with probs.top5`}</CodeBlock>

          <ul className="space-y-2 my-4">
            <FeatureItem>MobileNetV4 weights are production grade (Apache-2.0 ImageNet-1k, bit-identical load). DINOv2 classify weights are demo-grade (Imagenette, 10 classes).</FeatureItem>
            <FeatureItem>There is no LibreRFDETR classifier in v1.3.0. Classification moved into the LibreMobileNetV4 and LibreDINOv2 families; legacy LibreRFDETR*-cls checkpoints are rejected on load.</FeatureItem>
            <FeatureItem>A fresh DINOv2 fine-tune with the default recipe tops out around 0.93 top-1 on Imagenette, below the shipped 0.976. Fine-tune from a shipped -cls checkpoint to recover accuracy.</FeatureItem>
            <FeatureItem>ONNX classify output is raw logits. Apply softmax in non-Python consumers.</FeatureItem>
            <FeatureItem>Predicting a single image returns one Results. Read result.probs directly, or pass a list and index the list: model([&quot;a.jpg&quot;])[0].probs.</FeatureItem>
          </ul>

          <Divider />

          {/* ────────────── DEPTH ESTIMATION ────────────── */}
          <SectionHeading id="depth" icon={Mountain}>Depth Estimation</SectionHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="experimental">New in v1.3.0</SupportBadge>
            <SupportBadge variant="experimental">Inference and val only</SupportBadge>
          </div>
          <P>
            New in v1.3.0: monocular depth via <InlineCode>LibreDepthAnythingV2</InlineCode>, a Depth Anything V2 port (DINOv2 encoder plus DPT head, NeurIPS 2024). It predicts a dense relative inverse-depth map: higher values are closer to the camera, with no metric unit implied. Sizes <InlineCode>s</InlineCode> / <InlineCode>b</InlineCode> / <InlineCode>l</InlineCode> / <InlineCode>g</InlineCode> map to ViT-S / B / L / G and all run at 518. This is a pre-release task and supports inference and zero-shot validation only: no training and no export.
          </P>
          <P>
            Checkpoints. Only the ViT-S checkpoint is Apache-2.0 and auto-hosted: <Checkpoints names={['LibreDepthAnythingV2s-depth.pt']} link={false} />. The larger encoders <Checkpoints names={['LibreDepthAnythingV2b-depth.pt', 'LibreDepthAnythingV2l-depth.pt', 'LibreDepthAnythingV2g-depth.pt']} link={false} /> are CC-BY-NC-4.0 and are not redistributed by LibreYOLO; convert the official upstream checkpoints with <InlineCode>weights/convert_depth_anything_v2_weights.py</InlineCode>.
          </P>

          <SubHeading>Run depth estimation</SubHeading>
          <P>Input <InlineCode>imgsz</InlineCode> must be divisible by 14 (the DINOv2 patch grid). The depth map is returned on the original image canvas.</P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

# ViT-S encoder, Apache-2.0 weights (commercial use OK)
model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
result = model("street.jpg")

depth = result.depth_map          # DepthMap payload, (H, W) float on the original canvas
print(depth.data.shape)           # (H, W)
print(depth.min, depth.max, depth.mean)   # relative inverse depth: higher = closer
norm = depth.normalized()         # rescaled to [0, 1] over finite values`}</CodeBlock>

          <SubHeading>DepthMap API</SubHeading>
          <CodeBlock language="python">{`depth = result.depth_map
depth.data          # (H, W) float tensor / ndarray, relative inverse depth
depth.min           # min over finite values
depth.max           # max over finite values
depth.mean          # mean over finite values
depth.normalized()  # (H, W) rescaled to [0, 1]; non-finite pixels become 0

depth.cpu()
depth.numpy()`}</CodeBlock>

          <SubHeading>Zero-shot validation</SubHeading>
          <P>
            Validation runs zero-shot through the shared depth validator and reports standard depth metrics (AbsRel, RMSE, and delta thresholds). The validator letterboxes to a fixed square and excludes padded pixels; because predict uses Depth Anything&apos;s native keep-aspect resize, non-square val metrics are a documented approximation of predict.
          </P>
          <CodeBlock language="python">{`metrics = model.val(data="depth_dataset.yaml")
print(metrics["metrics/abs_rel"])   # absolute relative error (lower is better)
print(metrics["metrics/rmse"])      # root mean squared error
print(metrics["metrics/delta1"])    # fraction within a 1.25x ratio (higher is better)`}</CodeBlock>

          <SubHeading>Not supported</SubHeading>
          <CodeBlock language="python">{`model.train(data="...")          # raises NotImplementedError - DA V2 is inference + val only
model.export(format="onnx")      # raises NotImplementedError - depth export is out of scope`}</CodeBlock>

          <ul className="space-y-2 my-4">
            <FeatureItem>Licensing is split: ViT-S (size s) weights are Apache-2.0 and fine for commercial use. ViT-B / ViT-L / ViT-G (sizes b / l / g) are CC-BY-NC-4.0 (non-commercial) and are not redistributed by LibreYOLO.</FeatureItem>
            <FeatureItem>For commercial use, stick to size s.</FeatureItem>
            <FeatureItem>Depth is relative inverse depth with no metric unit. Calibrate on your side if you need meters.</FeatureItem>
            <FeatureItem>imgsz must be divisible by 14. Batched predict is disabled because keep-aspect resize yields variable per-image sizes.</FeatureItem>
          </ul>

          <Divider />

          {/* ────────────── POINT LOCALIZATION ────────────── */}
          <SectionHeading id="point-localization" icon={MapPin}>Point Localization</SectionHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="experimental">Experimental</SupportBadge>
          </div>
          <P>
            <InlineCode>LibreFOMO</InlineCode> is a FOMO-style point localizer (sizes <InlineCode>s</InlineCode> / <InlineCode>m</InlineCode> / <InlineCode>l</InlineCode>) for centroid-style detection: instead of boxes, each detection is a single image coordinate. Predictions arrive as <InlineCode>result.points</InlineCode>. Pretrained LibreFOMO weights are not auto-downloaded, so pass a local checkpoint path (or train from scratch, which is experimental and requires <InlineCode>allow_experimental=True</InlineCode>).
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

# LibreFOMO weights are not hosted by LibreYOLO - pass a local checkpoint
model = LibreYOLO("path/to/LibreFOMOm-point.pt")
result = model("scene.jpg")

points = result.points       # Points payload, (N, 4) rows: x, y, class, confidence
print(points.xy)             # (N, 2) absolute pixel coords
print(points.xyn)            # (N, 2) normalized to [0, 1]
print(points.cls, points.conf)`}</CodeBlock>

          <Divider />

          <SectionHeading id="training" icon={GraduationCap}>Training</SectionHeading>
          <ValidationScopeCallout />
          <P>
            The heavily tested training paths are single-GPU YOLO9 detection, RF-DETR detection, and RF-DETR segmentation. Other model-family trainers and multi-GPU workflows are also available, though less extensively tested. YOLO9 is detect-only in v1.3.0, so there is no YOLO9 segmentation or pose training.
          </P>

          <SubHeading>YOLO9 - CNN flagship training</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

# Fine-tune from a pretrained checkpoint (recommended)
model = LibreYOLO("LibreYOLO9c.pt")

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
    device="0",              # "" | "cpu" | "cuda" | "0" | "0,1"
    workers=8,
    seed=0,

    # Output
    project="runs/train",
    name="yolo9_exp",
    exist_ok=False,

    # Training features
    amp=True,                # automatic mixed precision
    patience=50,             # early stopping patience
    resume=False,            # resume from loaded checkpoint
    pretrained=True,         # transfer-learning init (True, a path, or None)
    cache="disk",            # cache decoded images: False | True/"ram" | "disk"
    freeze=10,               # freeze first N groups, or a list of indices / module names
    save_plots=True,         # write final validation plots to the run dir
)

print(f"Best mAP50-95: {results['best_mAP50_95']:.3f}")
print(f"Best checkpoint: {results['best_checkpoint']}")`}</CodeBlock>
          <P>
            After training completes, the model instance is automatically reloaded with the best weights so you can call <InlineCode>model(...)</InlineCode> immediately. <InlineCode>freeze</InlineCode>, <InlineCode>cache</InlineCode>, <InlineCode>pretrained</InlineCode>, and <InlineCode>save_plots</InlineCode> are new in v1.3.0 and accepted across the trainer-backed families.
          </P>

          <SubHeading>RF-DETR - transformer flagship training</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreRFDETRs.pt")

results = model.train(
    data="path/to/data.yaml",
    epochs=100,
    batch_size=4,            # NOTE: RF-DETR uses batch_size, not batch
    lr=1e-4,
    output_dir="runs/train/rfdetr_exp",
)`}</CodeBlock>
          <P>
            RF-DETR has its own training signature (<InlineCode>batch_size</InlineCode>, <InlineCode>lr</InlineCode>, <InlineCode>output_dir</InlineCode>) but it uses LibreYOLO&apos;s dataset config loader. Pass a <InlineCode>data.yaml</InlineCode> for detection or segmentation; COCO/Roboflow-style annotation layouts can be referenced from that config.
          </P>

          <SubHeading>LoRA fine-tuning (RF-DETR)</SubHeading>
          <P>
            <SupportBadge variant="experimental">Experimental</SupportBadge>{' '}
            <InlineCode>lora=True</InlineCode> injects LoRA adapters into the RF-DETR backbone for
            low-VRAM fine-tuning. It requires the optional <InlineCode>peft</InlineCode> dependency
            (<InlineCode>pip install &quot;libreyolo[lora]&quot;</InlineCode>) and is currently limited to
            RF-DETR; other families raise a clear error rather than ignoring the flag.
          </P>
          <CodeBlock language="python">{`model = LibreYOLO("LibreRFDETRs.pt")
results = model.train(data="data.yaml", epochs=50, lora=True)`}</CodeBlock>

          <SubHeading>Experiment loggers</SubHeading>
          <P>
            New in v1.3.0: pass <InlineCode>loggers=</InlineCode> to stream metrics to TensorBoard,
            MLflow, or Weights &amp; Biases. Accepts a name (<InlineCode>&quot;tensorboard&quot;</InlineCode>,{' '}
            <InlineCode>&quot;mlflow&quot;</InlineCode>, <InlineCode>&quot;wandb&quot;</InlineCode>), a configured logger
            instance, or an iterable mixing both. Each backend is an optional extra
            (<InlineCode>libreyolo[tensorboard]</InlineCode>, <InlineCode>[mlflow]</InlineCode>,{' '}
            <InlineCode>[wandb]</InlineCode>).
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO
from libreyolo.training.loggers import MLflowLogger

model = LibreYOLO("LibreYOLO9c.pt")

# By name
model.train(data="coco128.yaml", loggers="tensorboard")

# Mix configured instances and names
model.train(
    data="coco128.yaml",
    loggers=[MLflowLogger(experiment_name="my-exp"), "tensorboard"],
)`}</CodeBlock>
          <P>
            Loggers are a Python-API feature only. There is no CLI flag for them; the rest of the
            new training knobs (<InlineCode>--task</InlineCode>, <InlineCode>--cache</InlineCode>,{' '}
            <InlineCode>--lora</InlineCode>, <InlineCode>--freeze</InlineCode>,{' '}
            <InlineCode>--save-plots</InlineCode>) are exposed on the CLI.
          </P>

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
            Other families have trainer hooks, but they are not the recommended path in v1.3.0. Keep new work on YOLO9 detection or RF-DETR detection/segmentation; use experimental trainers only for compatibility, benchmark reproduction, or targeted research. PicoDet, RTMDet, and EC training require an explicit <InlineCode>allow_experimental=True</InlineCode> acknowledgement.
          </P>

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

          <SubHeading>Distributed training (DDP, experimental)</SubHeading>
          <P>
            YOLO9 and RF-DETR support multi-GPU training through PyTorch DistributedDataParallel, but multi-GPU is outside the heavily tested v1.3.0 scope. Launch the training script with <InlineCode>torchrun</InlineCode>:
          </P>
          <CodeBlock language="bash">{`# 4-GPU node
torchrun --nproc_per_node=4 train_yolo9.py

# Multi-node - see PyTorch's torchrun docs for --nnodes / --rdzv_endpoint`}</CodeBlock>
          <CodeBlock language="python" filename="train_yolo9.py">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")
# Pass device="" (auto-detect) and let torchrun set the rank
model.train(data="coco128.yaml", epochs=300, batch=16)`}</CodeBlock>

          <Divider />

          {/* ────────────── LoRA / DoRA ────────────── */}
          <SectionHeading id="lora" icon={Layers2}>LoRA / DoRA Fine-Tuning</SectionHeading>
          <div className="flex flex-wrap gap-2 mb-5">
            <SupportBadge variant="experimental">RF-DETR: n, s, m, l</SupportBadge>
          </div>
          <P>
            LoRA-style adapters let you fine-tune RF-DETR&apos;s transformer backbone by training a small set of low-rank matrices while the base weights stay frozen. That cuts optimizer and gradient memory, which is ideal for adapting a strong checkpoint to a new domain on modest hardware.
          </P>

          <SubHeading>Enabling it</SubHeading>
          <P>
            The whole public API is a single flag on <InlineCode>train()</InlineCode>. There are no rank, alpha, or target-module knobs to tune; the recipe is fixed to a well-tested configuration. Under the hood the implementation uses <strong className="text-surface-800 dark:text-white">DoRA</strong> (weight-decomposed LoRA, rank 16) applied to the DINOv2 attention query, key, and value projections.
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("rf-detr-nano.pth")   # sizes n, s, m, l
result = model.train(
    data="data.yaml",
    lora=True,        # DoRA on the frozen DINOv2 backbone
    epochs=100, batch_size=4, lr=1e-4,
)

# Resume: LoRA is auto-detected from the checkpoint, no need to repeat the flag
model.train(data="data.yaml", resume=True)`}</CodeBlock>
          <CodeBlock language="bash">{`# CLI equivalent
libreyolo train --model rf-detr-nano.pth --data data.yaml --lora`}</CodeBlock>

          <SubHeading>Checkpoints and export</SubHeading>
          <ul className="space-y-2 mb-4">
            <FeatureItem>Training checkpoints keep the adapter tensors, and the config records that LoRA was used, so loading and resuming rebuild the adapter graph automatically.</FeatureItem>
            <FeatureItem>The detection head always stays trainable, so you can still adapt to a new class count.</FeatureItem>
            <FeatureItem><InlineCode>export()</InlineCode> merges the adapters back into dense weights. Exported models are plain and carry no <InlineCode>peft</InlineCode> dependency.</FeatureItem>
            <FeatureItem>LoRA is RF-DETR only; passing <InlineCode>lora=True</InlineCode> to other families raises a clear error.</FeatureItem>
          </ul>
          <Callout icon={ShieldCheck} tone="emerald" title="Install extra">
            <p>
              LoRA training needs the adapter dependency: <InlineCode>pip install &quot;libreyolo[lora]&quot;</InlineCode>, which pulls in the RF-DETR stack and <InlineCode>peft</InlineCode>. Exported (merged) models do not need it at inference time.
            </p>
          </Callout>

          <Divider />

          {/* ────────────── VALIDATION ────────────── */}
          <SectionHeading id="validation" icon={CheckCircle2}>Validation</SectionHeading>
          <P>
            Run COCO-standard evaluation on a validation set. The heavily tested validation paths are single-GPU YOLO9 detection, RF-DETR detection, and RF-DETR segmentation.
          </P>
          <CodeBlock language="python">{`results = model.val(
    data="coco128.yaml",   # dataset config
    batch=16,
    imgsz=640,
    conf=0.001,            # low conf for mAP calculation
    iou=0.6,               # NMS IoU threshold
    split="val",           # "val", "test", or "train"
    save_json=False,       # save predictions as COCO JSON
    verbose=True,          # print per-class metrics
    plots=True,            # save validation plots (metrics, per-class AP, confusion matrix); alias for save_plots
)

print(f"mAP50:    {results['metrics/mAP50']:.3f}")
print(f"mAP50-95: {results['metrics/mAP50-95']:.3f}")`}</CodeBlock>

          <SubHeading>Validation results dict</SubHeading>
          <P>
            By default, LibreYOLO uses COCO evaluation and returns precision, recall, AP/AR metrics, and per-image timing:
          </P>
          <CodeBlock language="python">{`{
    "metrics/mAP50-95": 0.489,   # COCO primary metric (AP@[.5:.95])
    "metrics/mAP50": 0.721,      # AP@0.5 (PASCAL VOC style)
    "metrics/mAP75": 0.534,      # AP@0.75 (strict)
    "metrics/precision": 0.68,
    "metrics/recall": 0.61,
    "metrics/precision(B)": 0.68, # bbox aliases
    "metrics/recall(B)": 0.61,
    "metrics/mAP50(B)": 0.721,
    "metrics/mAP50-95(B)": 0.489,
    "metrics/mAP_small": 0.291,
    "metrics/mAP_medium": 0.532,
    "metrics/mAP_large": 0.648,
    "metrics/AR1": 0.362,        # Average Recall (max 1 det)
    "metrics/AR10": 0.571,
    "metrics/AR100": 0.601,
    "metrics/AR_small": 0.387,
    "metrics/AR_medium": 0.641,
    "metrics/AR_large": 0.739,
    "speed/preprocess_ms": 1.2,
    "speed/inference_ms": 6.8,
    "speed/postprocess_ms": 0.9,
    "speed/total_ms": 8.9,
    "speed/total_s": 12.3,
    "speed/images_seen": 1382,
}`}</CodeBlock>
          <P>
            Segmentation validation returns mask metrics with <InlineCode>(M)</InlineCode> suffixes alongside
            bbox metrics with <InlineCode>(B)</InlineCode> suffixes; OBB validation adds{' '}
            <InlineCode>(OBB)</InlineCode> metrics. Pose validation returns COCO keypoint metrics through{' '}
            <InlineCode>PoseValidator</InlineCode>. v1.3.0 adds validators for classify (top-1 / top-5),
            semantic (mIoU / pixel accuracy), point, and depth (zero-shot). Pass{' '}
            <InlineCode>plots=True</InlineCode> (or <InlineCode>--save-plots</InlineCode> on the CLI) to write
            metric, per-class AP, confusion-matrix, and sample plots to the run directory.
          </P>

          <Divider />

          {/* ────────────── EXPORT ────────────── */}
          <SectionHeading id="export" icon={Upload}>Export</SectionHeading>
          <P>
            Export PyTorch models to ONNX, TorchScript, TensorRT, OpenVINO, NCNN, CoreML, or (new in v1.3.0) TFLite for deployment. The heavily tested export paths are single-GPU YOLO9 detection, RF-DETR detection, and RF-DETR segmentation. Some newer families, including DINOv2, Depth Anything V2, SAM, VLM, and L2CS, do not support export yet.
          </P>

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
model.export(format="ncnn")

# CoreML (.mlpackage, macOS runtime)
model.export(format="coreml")

# TFLite (RF-DETR detect/seg/pose + YOLO9 detect; experimental, needs Python 3.12+)
model.export(format="tflite")`}</CodeBlock>

          <SubHeading>All export parameters</SubHeading>
          <CodeBlock language="python">{`path = model.export(
    format="onnx",            # "onnx", "torchscript", "tensorrt", "openvino", "ncnn", "coreml", or "tflite"
    output_path="model.onnx", # output file (auto-generated if None)
    imgsz=640,                # input resolution (default: model's native); also accepts (h, w) for rectangular
    opset=None,               # ONNX opset (auto: 13, or 17 for wrappers that need it)
    simplify=True,            # run onnxsim graph simplification
    dynamic=True,             # enable dynamic batch axis (ONNX); TFLite requires static shapes
    half=False,               # export in FP16
    batch=1,                  # batch size for static graph
    device=None,              # device to trace on (default: model's current device)
    int8=False,               # INT8 quantization: TensorRT, OpenVINO, or ONNX (YOLO9 detection only)
    data=None,                # calibration dataset for INT8
    fraction=1.0,             # fraction of calibration data to use
    allow_download_scripts=False, # allow data.yaml download hooks during calibration
    workspace=4.0,            # TensorRT workspace size (GB)
    min_batch=1,              # TensorRT dynamic profile minimum batch
    opt_batch=1,              # TensorRT dynamic profile optimal batch
    max_batch=8,              # TensorRT dynamic profile maximum batch
    hardware_compatibility="none", # TensorRT compatibility mode
    gpu_device=0,             # GPU device index for TensorRT
    trt_config=None,          # optional TensorRT YAML config path
    compute_units="all",      # CoreML routing: all, cpu_only, cpu_and_gpu, cpu_and_ne
    nms=False,                # embed NMS in the graph (ONNX YOLO9 detection, or CoreML)
    iou=0.45,                 # embedded-NMS IoU threshold
    conf=0.25,                # embedded-NMS confidence threshold
    max_det=300,              # embedded-NMS max detections (ONNX only)
    verbose=False,            # verbose logging
)`}</CodeBlock>
          <P>
            OpenVINO INT8 export additionally requires <InlineCode>nncf</InlineCode>. NCNN export writes a directory containing <InlineCode>model.ncnn.param</InlineCode>, <InlineCode>model.ncnn.bin</InlineCode>, and <InlineCode>metadata.yaml</InlineCode>. CoreML export writes a <InlineCode>.mlpackage</InlineCode> bundle, requires <InlineCode>coremltools</InlineCode>, and does not support INT8.
          </P>

          <SubHeading>ONNX embedded NMS (YOLO9 detection)</SubHeading>
          <P>
            New in v1.3.0: pass <InlineCode>nms=True</InlineCode> to bake NMS into an exported ONNX
            graph so the model emits final boxes directly. This is currently limited to the{' '}
            <InlineCode>yolo9</InlineCode> family on the <InlineCode>detect</InlineCode> task (other
            families/tasks raise). It forces a fixed batch-1 graph
            (<InlineCode>dynamic=False</InlineCode>) and records{' '}
            <InlineCode>nms</InlineCode> / <InlineCode>nms_conf</InlineCode> /{' '}
            <InlineCode>nms_iou</InlineCode> / <InlineCode>max_det</InlineCode> in the ONNX metadata.
          </P>
          <CodeBlock language="python">{`model = LibreYOLO("LibreYOLO9c.pt")
model.export(format="onnx", nms=True, conf=0.25, iou=0.45, max_det=300)`}</CodeBlock>
          <P>
            <InlineCode>int8=True</InlineCode> is now also supported for ONNX (in addition to
            TensorRT and OpenVINO), again limited to YOLO9 detection; it needs a calibration{' '}
            <InlineCode>data=</InlineCode> dataset.
          </P>

          <SubHeading>TFLite export</SubHeading>
          <P>
            <SupportBadge variant="experimental">Experimental</SupportBadge>{' '}
            v1.3.0 adds a TFLite export path built on <InlineCode>onnx2tf</InlineCode>. It is
            validated for RF-DETR detect / segment / pose and YOLO9 detect. It requires{' '}
            <strong className="text-surface-800 dark:text-white">Python 3.12+</strong> (the{' '}
            <InlineCode>onnx2tf 2.4.x</InlineCode> wheels do not target older Python) plus the
            optional extra <InlineCode>libreyolo[tflite]</InlineCode>
            (<InlineCode>onnx2tf&gt;=2.4.3</InlineCode>, onnx-graphsurgeon, onnx-simplifier). Export
            is FP32 and static-shape only (no <InlineCode>half</InlineCode>,{' '}
            <InlineCode>int8</InlineCode>, or <InlineCode>dynamic</InlineCode> yet).
          </P>
          <CodeBlock language="bash">{`pip install "libreyolo[tflite]"   # Python 3.12+`}</CodeBlock>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreRFDETRs-seg.pt")
model.export(format="tflite")   # writes a .tflite file`}</CodeBlock>
          <P>
            For RF-DETR, the exporter rewrites each GridSample node into a TFLite-safe bilinear
            subgraph because onnx2tf&apos;s default lowering is numerically broken. In v1.3.0 the old
            runtime monkeypatches against onnx2tf were removed now that{' '}
            <InlineCode>onnx2tf&gt;=2.4.3</InlineCode> ships the RF-DETR fixes upstream; only the
            static ONNX-graph rewrite remains.
          </P>
          <P>
            <strong className="text-surface-800 dark:text-white">No TFLite runtime backend.</strong>{' '}
            LibreYOLO cannot load or run a <InlineCode>.tflite</InlineCode> file; this format is
            export-only. Run the exported model with a TF Lite runtime
            (<InlineCode>ai-edge-litert</InlineCode> / <InlineCode>tflite-runtime</InlineCode>) on
            your target device.
          </P>

          <SubHeading>ONNX metadata</SubHeading>
          <P>Exported ONNX files include embedded metadata:</P>
          <DocTable
            headers={['Key', 'Example value']}
            rows={[
              [<InlineCode key="v">libreyolo_version</InlineCode>, <InlineCode key="vv">&quot;1.3.0&quot;</InlineCode>],
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
            This metadata is automatically read back when loading the exported file with <InlineCode>LibreYOLO(&quot;model.onnx&quot;)</InlineCode>.
          </P>

          <Divider />

          {/* ────────────── TORCHSCRIPT INFERENCE ────────────── */}
          <SectionHeading id="torchscript-inference" icon={Cpu}>TorchScript Inference</SectionHeading>
          <P>
            Run an exported <InlineCode>.torchscript</InlineCode> model through the same runtime-backend prediction API.
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("model.torchscript")

result = model("image.jpg", conf=0.25, iou=0.45, save=True)
print(result.boxes.xyxy)`}</CodeBlock>

          <Divider />

          {/* ────────────── ONNX INFERENCE ────────────── */}
          <SectionHeading id="onnx-inference" icon={Cpu}>ONNX Inference</SectionHeading>
          <P>
            Run inference using ONNX Runtime instead of PyTorch. Useful for deployment environments without PyTorch.
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("model.onnx")

result = model("image.jpg", conf=0.25, iou=0.45, save=True)
print(result.boxes.xyxy)`}</CodeBlock>

          <SubHeading>Auto-metadata</SubHeading>
          <P>
            If the ONNX file was exported by LibreYOLO, class names and class count are read automatically from the embedded metadata:
          </P>
          <CodeBlock language="python">{`# Export with metadata
model.export(format="onnx", output_path="model.onnx")

# Load - names and nb_classes auto-populated
onnx_model = LibreYOLO("model.onnx")
print(onnx_model.names)       # {0: "person", 1: "bicycle", ...}
print(onnx_model.nb_classes)  # 80`}</CodeBlock>

          <P>
            For ONNX files without metadata (e.g., exported by other tools), specify <InlineCode>nb_classes</InlineCode> manually:
          </P>
          <CodeBlock language="python">{`model = LibreYOLO("external_model.onnx", nb_classes=20)`}</CodeBlock>

          <SubHeading>Device selection</SubHeading>
          <CodeBlock language="python">{`# Auto-detect (CUDA if available, else CPU)
model = LibreYOLO("model.onnx", device="auto")

# Force CPU
model = LibreYOLO("model.onnx", device="cpu")

# Force CUDA
model = LibreYOLO("model.onnx", device="cuda")`}</CodeBlock>

          <SubHeading>Prediction parameters</SubHeading>
          <P>
            Runtime artifacts loaded through <InlineCode>LibreYOLO()</InlineCode> support the shared runtime prediction API:
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
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("model.engine")

result = model("image.jpg", conf=0.25, iou=0.45, save=True)
print(result.boxes.xyxy)`}</CodeBlock>

          <P>
            TensorRT artifacts loaded through <InlineCode>LibreYOLO()</InlineCode> support the same core runtime prediction API as ONNX and OpenVINO, including the same file-path-only <InlineCode>output_path</InlineCode> behavior for <InlineCode>save=True</InlineCode>.
          </P>

          <Divider />

          {/* ────────────── OPENVINO INFERENCE ────────────── */}
          <SectionHeading id="openvino-inference" icon={Cpu}>OpenVINO Inference</SectionHeading>
          <P>
            Run inference using OpenVINO, optimized for Intel CPUs, GPUs, and VPUs.
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("model_openvino/")

result = model("image.jpg", conf=0.25, iou=0.45, save=True)
print(result.boxes.xyxy)`}</CodeBlock>

          <P>
            OpenVINO directories loaded through <InlineCode>LibreYOLO()</InlineCode> read <InlineCode>metadata.yaml</InlineCode> when present and support the same core runtime prediction API.
          </P>

          <Divider />

          {/* ────────────── NCNN INFERENCE ────────────── */}
          <SectionHeading id="ncnn-inference" icon={Cpu}>NCNN Inference</SectionHeading>
          <P>
            Run inference using NCNN for lightweight deployment on CPU or Vulkan-capable GPU targets.
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("model_ncnn/")

result = model("image.jpg", conf=0.25, iou=0.45, save=True)
print(result.boxes.xyxy)`}</CodeBlock>

          <P>
            An NCNN export directory contains <InlineCode>model.ncnn.param</InlineCode>, <InlineCode>model.ncnn.bin</InlineCode>, and usually <InlineCode>metadata.yaml</InlineCode>.
          </P>

          <Divider />

          {/* ────────────── COREML INFERENCE ────────────── */}
          <SectionHeading id="coreml-inference" icon={Cpu}>CoreML Inference</SectionHeading>
          <P>
            Run an exported <InlineCode>.mlpackage</InlineCode> through CoreML on macOS. CoreML routes execution with <InlineCode>compute_units</InlineCode> instead of PyTorch device strings.
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("model.mlpackage", compute_units="all")

result = model("image.jpg", conf=0.25, iou=0.45, save=True)
print(result.boxes.xyxy)`}</CodeBlock>

          <P>
            Supported <InlineCode>compute_units</InlineCode> values are <InlineCode>all</InlineCode>, <InlineCode>cpu_only</InlineCode>, <InlineCode>cpu_and_gpu</InlineCode>, and <InlineCode>cpu_and_ne</InlineCode>.
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
              [<InlineCode key="e">export</InlineCode>, 'Export to ONNX / TorchScript / TensorRT / OpenVINO / NCNN / CoreML / TFLite'],
              [<InlineCode key="ui">ui</InlineCode>, 'Launch a local drag-and-drop / paste browser inference UI'],
              [<InlineCode key="dr">doctor</InlineCode>, 'Run pre-training dataset health checks (YOLO detection format)'],
              [<InlineCode key="c">checks</InlineCode>, 'Print Python, torch, CUDA, GPU, and optional-package info'],
              [<InlineCode key="m">models</InlineCode>, 'List registered model families and CLI shortcut names'],
              [<InlineCode key="f">formats</InlineCode>, 'List supported export formats'],
              [<InlineCode key="cfg">cfg</InlineCode>, 'Print the default training configuration YAML'],
              [<InlineCode key="i">info</InlineCode>, 'Load a model and print resolved family, size, task, device, and classes'],
              [<InlineCode key="md">metadata</InlineCode>, 'Inspect raw checkpoint metadata from a .pt file'],
              [<InlineCode key="ver">version</InlineCode>, 'Print LibreYOLO + Python + torch versions'],
            ]}
          />

          <SubHeading>Model name shortcuts</SubHeading>
          <P>
            The CLI accepts short names (<InlineCode>yolo9-c</InlineCode>) that resolve to weight filenames (<InlineCode>LibreYOLO9c.pt</InlineCode>) - discoverable via <InlineCode>libreyolo models</InlineCode>. You can also pass any explicit checkpoint path.
          </P>

          <SubHeading>Common options</SubHeading>
          <DocTable
            headers={['Command', 'Important options']}
            rows={[
              [<InlineCode key="p">predict</InlineCode>, <span key="pv"><InlineCode>conf</InlineCode>, <InlineCode>iou</InlineCode>, <InlineCode>imgsz</InlineCode>, <InlineCode>classes</InlineCode>, <InlineCode>max_det</InlineCode>, <InlineCode>half</InlineCode>, <InlineCode>batch</InlineCode>, <InlineCode>tiling</InlineCode>, <InlineCode>overlap_ratio</InlineCode>, <InlineCode>output_file_format</InlineCode>, <InlineCode>project</InlineCode>, <InlineCode>name</InlineCode>, <InlineCode>exist_ok</InlineCode>, <InlineCode>face_detector</InlineCode></span>],
              [<InlineCode key="t">train</InlineCode>, <span key="tv"><InlineCode>epochs</InlineCode>, <InlineCode>batch</InlineCode>, <InlineCode>imgsz</InlineCode>, <InlineCode>lr0</InlineCode>, <InlineCode>optimizer</InlineCode>, <InlineCode>scheduler</InlineCode>, <InlineCode>workers</InlineCode>, <InlineCode>seed</InlineCode>, <InlineCode>resume</InlineCode>, <InlineCode>amp</InlineCode>, <InlineCode>task</InlineCode>, <InlineCode>cache</InlineCode>, <InlineCode>lora</InlineCode>, <InlineCode>freeze</InlineCode>, <InlineCode>save_plots</InlineCode>, <InlineCode>allow_download_scripts</InlineCode>, <InlineCode>dry_run</InlineCode></span>],
              [<InlineCode key="v">val</InlineCode>, <span key="vv"><InlineCode>split</InlineCode>, <InlineCode>batch</InlineCode>, <InlineCode>imgsz</InlineCode>, <InlineCode>conf</InlineCode>, <InlineCode>iou</InlineCode>, <InlineCode>max_det</InlineCode>, <InlineCode>half</InlineCode>, <InlineCode>save_plots</InlineCode>, <InlineCode>data_dir</InlineCode>, <InlineCode>use_coco_eval</InlineCode>, <InlineCode>project</InlineCode>, <InlineCode>name</InlineCode>, <InlineCode>exist_ok</InlineCode>, <InlineCode>save_json</InlineCode>, <InlineCode>allow_download_scripts</InlineCode></span>],
              [<InlineCode key="e">export</InlineCode>, <span key="ev"><InlineCode>format</InlineCode>, <InlineCode>imgsz</InlineCode>, <InlineCode>batch</InlineCode>, <InlineCode>half</InlineCode>, <InlineCode>int8</InlineCode>, <InlineCode>dynamic</InlineCode>, <InlineCode>simplify</InlineCode>, <InlineCode>nms</InlineCode>, <InlineCode>conf</InlineCode>, <InlineCode>iou</InlineCode>, <InlineCode>max_det</InlineCode>, <InlineCode>opset</InlineCode>, <InlineCode>data</InlineCode>, <InlineCode>fraction</InlineCode>, <InlineCode>device</InlineCode>, <InlineCode>allow_download_scripts</InlineCode>, <InlineCode>verbose</InlineCode></span>],
            ]}
          />

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
libreyolo export model=best.pt format=openvino int8=true data=coco128.yaml
libreyolo export model=best.pt format=coreml`}</CodeBlock>

          <SubHeading>Export with embedded NMS and rectangular size</SubHeading>
          <CodeBlock language="bash">{`# Embed NMS into an ONNX YOLO9 detection graph
libreyolo export model=yolo9-c format=onnx nms=true conf=0.25 iou=0.45 max_det=300

# Rectangular export size (imgsz accepts a single value or two comma-separated dims)
libreyolo export model=yolo9-c format=onnx imgsz=640,480

# TFLite (Python 3.12+, libreyolo[tflite])
libreyolo export model=rfdetr-s format=tflite`}</CodeBlock>

          <SubHeading>Local inference UI</SubHeading>
          <P>
            <InlineCode>libreyolo ui</InlineCode> serves a local browser page where you drop, paste,
            or pick images, choose a model, and view results. It binds{' '}
            <InlineCode>127.0.0.1:8000</InlineCode> by default and auto-bumps the port if taken.
          </P>
          <CodeBlock language="bash">{`libreyolo ui                       # opens http://127.0.0.1:8000
libreyolo ui --port 9000 --no-browser --device 0`}</CodeBlock>

          <SubHeading>Dataset health check</SubHeading>
          <P>
            <InlineCode>libreyolo doctor</InlineCode> runs pre-training checks on a YOLO
            detection-format dataset and exits non-zero when errors are found
            (<InlineCode>--strict</InlineCode> also fails on warnings), so it can gate CI.
          </P>
          <CodeBlock language="bash">{`libreyolo doctor coco8.yaml
libreyolo doctor --data coco8.yaml --strict --json
libreyolo doctor coco8.yaml --fast --only labels   # skip image decoding, run one check family`}</CodeBlock>

          <SubHeading>Machine-readable output</SubHeading>
          <P>
            Every command accepts <InlineCode>--json</InlineCode> (structured stdout for piping into scripts or agents) and <InlineCode>--quiet</InlineCode> (suppress stderr progress lines). The core <InlineCode>predict</InlineCode>, <InlineCode>train</InlineCode>, <InlineCode>val</InlineCode>, and <InlineCode>export</InlineCode> commands also accept <InlineCode>--help-json</InlineCode> to dump their parameter schema as JSON.
          </P>
          <CodeBlock language="bash">{`libreyolo predict model=yolo9-c source=img.jpg --json | jq .

libreyolo train --help-json > train_schema.json`}</CodeBlock>

          <Divider />

          {/* ────────────── API REFERENCE ────────────── */}
          <SectionHeading id="api-reference" icon={FileCode}>API Reference</SectionHeading>

          <SubHeading>LibreYOLO (factory)</SubHeading>
          <CodeBlock language="python">{`LibreYOLO(
    model_path: str,
    *,
    device: str = "auto",
    task: str | None = None,    # override only when a custom artifact is ambiguous
    nb_classes: int | None = None,  # mainly for external exported artifacts
    compute_units: str = "all", # CoreML only: all, cpu_only, cpu_and_gpu, cpu_and_ne
) -> model wrapper or runtime backend`}</CodeBlock>
          <P>
            Prefer official checkpoint filenames and exported artifact paths, then let the factory resolve the details. It handles PyTorch checkpoints, <InlineCode>.onnx</InlineCode>, <InlineCode>.torchscript</InlineCode>, <InlineCode>.engine</InlineCode>, <InlineCode>.tensorrt</InlineCode>, <InlineCode>.mlpackage</InlineCode>, OpenVINO directories containing <InlineCode>model.xml</InlineCode>, and NCNN directories containing <InlineCode>model.ncnn.param</InlineCode> plus <InlineCode>model.ncnn.bin</InlineCode>. The <InlineCode>task</InlineCode> argument is for ambiguous custom artifacts; otherwise resolution comes from checkpoint metadata, filename suffix, and family default.
          </P>

          <SubHeading>Prediction (PyTorch model wrappers)</SubHeading>
          <CodeBlock language="python">{`model(
    source,                     # image input (see supported formats)
    *,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = None,
    device: str = "auto",
    classes: list[int] = None,
    max_det: int = 300,
    augment: bool = False,
    save: bool = False,
    batch: int = 1,
    stream: bool = False,
    vid_stride: int = 1,
    show: bool = False,
    output_path: str = None,
    color_format: str = "auto",
    tiling: bool = False,
    overlap_ratio: float = 0.2,
    output_file_format: str = None,
) -> Results | list[Results] | Generator[Results, None, None]`}</CodeBlock>

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
    boxes: Boxes | None,
    orig_shape: tuple[int, int],  # (height, width)
    path: str | None,
    names: dict[int, str],
    masks: Masks | None = None,
    keypoints: Keypoints | None = None,
    probs: Probs | None = None,
    obb: OBB | None = None,
    gaze: Gaze | None = None,
    speed: dict[str, float] | None = None,
    track_id = None,
    frame_idx: int | None = None,
)

len(result)          # number of detections
result.cpu()         # copy with tensors on CPU
result.cuda()        # copy with tensors on CUDA
result.numpy()       # copy with numpy arrays
result.summary()     # list[dict] with boxes, masks, gaze, and track_id when present
result.to_json()     # JSON string from summary()`}</CodeBlock>

          <SubHeading>Boxes</SubHeading>
          <CodeBlock language="python">{`boxes = Boxes(boxes, conf, cls)

boxes.xyxy           # (N, 4) tensor - x1, y1, x2, y2
boxes.xywh           # (N, 4) tensor - cx, cy, w, h
boxes.conf           # (N,) tensor - confidence scores
boxes.cls            # (N,) tensor - class IDs
boxes.id             # (N,) track IDs when tracking, else None
boxes.is_track       # True when track IDs are attached
boxes.data           # (N, 6) [xyxy, conf, cls], or (N, 7) with track IDs

len(boxes)           # number of boxes
boxes.cpu()          # copy on CPU
boxes.numpy()        # copy as numpy arrays`}</CodeBlock>

          <SubHeading>Task payloads</SubHeading>
          <CodeBlock language="python">{`result.masks.data        # segmentation masks, (N, H, W)
result.masks.xy          # list of mask contours in pixel coordinates
result.masks.xyn         # normalized mask contours

result.keypoints.xy      # pose keypoint coordinates
result.keypoints.xyn     # normalized keypoint coordinates
result.keypoints.conf    # keypoint confidence when present

result.gaze.data         # (N, 2): pitch, yaw in radians
result.gaze.pitch_deg    # pitch in degrees
result.gaze.yaw_deg      # yaw in degrees
result.gaze.direction_3d # approximate 3D direction vectors`}</CodeBlock>

          <SubHeading>model.export()</SubHeading>
          <CodeBlock language="python">{`model.export(
    format: str = "onnx",       # "onnx", "torchscript", "tensorrt", "openvino", "ncnn", or "coreml"
    *,
    output_path: str | None = None,
    imgsz: int | None = None,
    opset: int | None = None,   # auto: 13, or 17 for wrappers that need it
    simplify: bool = True,
    dynamic: bool = True,
    half: bool = False,
    batch: int = 1,
    device: str | None = None,
    int8: bool = False,
    data: str | None = None,    # calibration data for INT8
    fraction: float = 1.0,      # fraction of calibration data
    allow_download_scripts: bool = False,
    workspace: float = 4.0,     # TensorRT workspace (GB)
    min_batch: int = 1,         # TensorRT dynamic profile minimum batch
    opt_batch: int = 1,         # TensorRT dynamic profile optimal batch
    max_batch: int = 8,         # TensorRT dynamic profile maximum batch
    hardware_compatibility: str = "none",
    gpu_device: int = 0,
    trt_config = None,          # optional TensorRT YAML config path
    compute_units: str = "all", # CoreML only
    nms: bool = False,          # CoreML embedded NMS where supported
    iou: float = 0.45,          # CoreML embedded NMS IoU threshold
    conf: float = 0.25,         # CoreML embedded NMS confidence threshold
    verbose: bool = False,
) -> str                        # path to exported file or directory`}</CodeBlock>

          <SubHeading>model.val()</SubHeading>
          <CodeBlock language="python">{`model.val(
    data: str = None,           # path to data.yaml
    batch: int = 16,
    imgsz: int = None,
    conf: float = 0.001,
    iou: float = 0.6,
    workers: int = 4,
    allow_download_scripts: bool = False,
    device: str = None,
    split: str = "val",         # "val", "test", or "train"
    augment: bool = False,
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
    allow_download_scripts: bool = False,
    callbacks = None,
) -> dict`}</CodeBlock>
          <P>Returns the standard LibreYOLO training dict with <InlineCode>final_loss</InlineCode>, <InlineCode>best_mAP50</InlineCode>, <InlineCode>best_mAP50_95</InlineCode>, <InlineCode>best_epoch</InlineCode>, <InlineCode>save_dir</InlineCode>, <InlineCode>best_checkpoint</InlineCode>, and <InlineCode>last_checkpoint</InlineCode>.</P>

          <SubHeading>model.train() (RF-DETR)</SubHeading>
          <CodeBlock language="python">{`model.train(
    data: str,                  # path to data.yaml
    epochs: int = 100,
    batch_size: int = 4,
    lr: float = 1e-4,
    output_dir: str = "runs/train",
    resume: str = None,
    **kwargs,                   # additional RF-DETR training args
) -> dict`}</CodeBlock>
          <P>
            Additional experimental trainers exist for YOLO-NAS, D-FINE, DEIM, DEIMv2, EC, PicoDet, RT-DETRv2/v4, and RTMDet, plus the new classification (MobileNetV4, ConvNeXt, EfficientNetV2, DINOv2), semantic-segmentation (DINOv2), and point (FOMO) families. They follow the same <InlineCode>model.train(data=&quot;...yaml&quot;, ...)</InlineCode> shape but their defaults and experimental gates are family-specific.
          </P>

          <SubHeading>Runtime artifact loading</SubHeading>
          <P>
            Load exported artifacts through <InlineCode>LibreYOLO()</InlineCode>, the same way you load PyTorch checkpoints. The factory chooses ONNX Runtime, TorchScript, TensorRT, OpenVINO, NCNN, or CoreML from the path:
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("model.onnx")
model = LibreYOLO("model.torchscript")
model = LibreYOLO("model.engine")
model = LibreYOLO("model_openvino/")
model = LibreYOLO("model_ncnn/")
model = LibreYOLO("model.mlpackage", compute_units="all")`}</CodeBlock>
          <P>
            Advanced integrations can reach lower-level runtime modules, but normal application code should stay on the factory path.
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
        yolo9/           # LibreYOLO9 (detect)
        yolo9_e2e/       # LibreYOLO9E2E (detect)
        yolonas/         # LibreYOLONAS (detect, pose)
        dfine/           # LibreDFINE (detect)
        deim/            # LibreDEIM (detect)
        deimv2/          # LibreDEIMv2 (detect)
        rtdetr/          # LibreRTDETR (detect)
        rtdetrv2/        # LibreRTDETRv2 (detect)
        rtdetrv4/        # LibreRTDETRv4 (detect)
        rfdetr/          # LibreRFDETR (detect, segment, pose, obb) - lazy-loaded
        ec/              # LibreEC / EdgeCrafter (detect, pose, segment)
        picodet/         # LibrePICODET (detect)
        rtmdet/          # LibreRTMDet (detect)
        dinov2/          # LibreDINOv2 (semantic, classify) - lazy-loaded
        mobilenetv4/     # LibreMobileNetV4 (classify)
        convnext/        # LibreConvNeXt (classify)
        efficientnetv2/  # LibreEfficientNetV2 (classify)
        depth_anything/  # LibreDepthAnythingV2 (depth)
        fomo/            # LibreFOMO (point)
        l2cs/            # LibreL2CS (gaze, inference-only)
    backends/
        base.py
        onnx.py          # ONNX Runtime loader
        torchscript.py   # TorchScript loader
        tensorrt.py      # TensorRT loader
        openvino.py      # OpenVINO loader
        ncnn.py          # NCNN loader
        coreml.py        # CoreML loader
    export/
        exporter.py      # BaseExporter and format registry
        onnx.py / torchscript.py / tensorrt.py / openvino.py / ncnn.py / coreml.py
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
              <>Set <InlineCode>FAMILY</InlineCode>, <InlineCode>FILENAME_PREFIX</InlineCode>, <InlineCode>INPUT_SIZES</InlineCode>, <InlineCode>SUPPORTED_TASKS</InlineCode>, and <InlineCode>DEFAULT_TASK</InlineCode> as needed</>,
              <>Implement registry hooks such as <InlineCode>can_load()</InlineCode>, <InlineCode>detect_size()</InlineCode>, <InlineCode>detect_nb_classes()</InlineCode>, and <InlineCode>detect_size_from_filename()</InlineCode></>,
              'Implement the model init, preprocess, forward, postprocess, train, and validation hooks that the family needs',
              <>Create the supporting network and utilities under <InlineCode>libreyolo/models/newmodel/</InlineCode></>,
              <>Add the import to <InlineCode>libreyolo/models/__init__.py</InlineCode>; subclass registration happens when the import runs</>,
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
            User code should export through <InlineCode>model.export(...)</InlineCode>. Internally, <InlineCode>BaseExporter</InlineCode> in <InlineCode>libreyolo/export/exporter.py</InlineCode> owns the format registry, and concrete exporters register themselves through subclass registration.
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")
model.export(format="onnx")`}</CodeBlock>
          <P>
            To add a new export format, implement a new <InlineCode>BaseExporter</InlineCode> subclass with a unique <InlineCode>format_name</InlineCode> and import it from <InlineCode>libreyolo/export/exporter.py</InlineCode> so the registry is populated.
          </P>

          <Divider />

          {/* ────────────── DATASET FORMAT ────────────── */}
          <SectionHeading id="dataset-format" icon={Database}>Dataset Format</SectionHeading>
          <P>
            Training and validation use dataset configs loaded through <InlineCode>data.yaml</InlineCode>. Detection, segmentation, pose, and RF-DETR training all enter through this loader; the label file contents differ by task.
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

          <SubHeading>Config resolution and downloads</SubHeading>
          <P>
            Dataset configs resolve from an explicit path, the current working directory, then built-ins under <InlineCode>libreyolo/config/datasets/</InlineCode>. Dataset roots default under <InlineCode>~/datasets</InlineCode> and can be overridden with <InlineCode>LIBREYOLO_DATASETS_DIR</InlineCode>.
          </P>
          <P>
            <InlineCode>train</InlineCode>, <InlineCode>val</InlineCode>, and <InlineCode>test</InlineCode> may be directories, <InlineCode>.txt</InlineCode> files, or lists of paths. YAML download hooks are guarded; pass <InlineCode>allow_download_scripts=True</InlineCode> only for trusted configs.
          </P>

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

          <SubHeading>Detection label format</SubHeading>
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

          <SubHeading>Segmentation label format</SubHeading>
          <P>
            Segmentation uses YOLO polygon rows. The dataset loader derives the bounding box from the polygon vertices and keeps the polygon rings when segment loading is enabled:
          </P>
          <CodeBlock language="text">{`<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>`}</CodeBlock>

          <SubHeading>Pose label format</SubHeading>
          <P>
            Pose labels append keypoints after the box. Add <InlineCode>kpt_shape</InlineCode> and <InlineCode>flip_idx</InlineCode> to <InlineCode>data.yaml</InlineCode> so the loader knows the keypoint count and horizontal flip permutation.
          </P>
          <CodeBlock language="yaml">{`kpt_shape: [17, 3]
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]`}</CodeBlock>
          <CodeBlock language="text">{`<class_id> <cx> <cy> <w> <h> <kx1> <ky1> <v1> ... <kxK> <kyK> <vK>`}</CodeBlock>

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


/* ============================================================================
 * Chinese (zh-CN) content bundle for the LibreYOLO v1.3.0 docs page.
 * Append to src/app/[locale]/docs/v1.3.0/page.jsx and render <DocsPageZh /> when
 * locale === 'zh'. Reuses all shared presentational components from that file.
 * ========================================================================== */

/* ─── 1. Section metadata (Chinese titles, same id + icon) ─── */
const sectionsZh = [
  { id: 'introduction', title: '简介', icon: BookOpen },
  { id: 'compatibility', title: '兼容性', icon: CheckCircle2 },
  { id: 'installation', title: '安装', icon: Terminal },
  { id: 'quickstart', title: '快速开始', icon: Rocket },
  { id: 'models', title: '可用模型', icon: Layers },
  { id: 'tasks', title: '任务与文件名', icon: Tags },
  { id: 'prediction', title: '预测', icon: Crosshair },
  { id: 'tiled-inference', title: '分块推理', icon: Grid3x3 },
  { id: 'video-inference', title: '视频推理', icon: Video },
  { id: 'tracking', title: '跟踪', icon: Activity },
  { id: 'segmentation', title: '分割', icon: Scissors },
  { id: 'obb', title: '旋转框 (OBB)', icon: Rotate3d },
  { id: 'pose', title: '姿态估计', icon: PersonStanding },
  { id: 'gaze', title: '视线估计', icon: Eye },
  { id: 'open-vocabulary', title: '开放词表检测', icon: ScanSearch },
  { id: 'classification', title: '分类', icon: Tags },
  { id: 'depth', title: '深度估计', icon: Mountain },
  { id: 'point-localization', title: '点定位', icon: MapPin },
  { id: 'training', title: '训练', icon: GraduationCap },
  { id: 'lora', title: 'LoRA / DoRA', icon: Layers2 },
  { id: 'validation', title: '验证', icon: CheckCircle2 },
  { id: 'export', title: '导出', icon: Upload },
  { id: 'torchscript-inference', title: 'TorchScript 推理', icon: Cpu },
  { id: 'onnx-inference', title: 'ONNX 推理', icon: Cpu },
  { id: 'tensorrt-inference', title: 'TensorRT 推理', icon: Cpu },
  { id: 'openvino-inference', title: 'OpenVINO 推理', icon: Cpu },
  { id: 'ncnn-inference', title: 'NCNN 推理', icon: Cpu },
  { id: 'coreml-inference', title: 'CoreML 推理', icon: Cpu },
  { id: 'cli', title: '命令行（CLI）', icon: SquareTerminal },
  { id: 'api-reference', title: 'API 参考', icon: FileCode },
  { id: 'architecture', title: '架构指南', icon: Wrench },
  { id: 'dataset-format', title: '数据集格式', icon: Database },
]

/* ─── 2. Sidebar (Chinese) ─── */
function SidebarZh({ activeSection, onNavigate, currentVersion = 'v1.3.0', className = '' }) {
  const versionLabelZh = {
    'Pre-release': '预发布',
    'Latest': '最新',
    'Archived': '已归档',
  }

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
                <span className="text-[11px] font-semibold uppercase tracking-wide">{versionLabelZh[label] || label}</span>
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

/* ─── 3. Content-bearing callouts + compatibility matrix (Chinese) ─── */
function ValidationScopeCalloutZh({ className = '' }) {
  return (
    <div className={`my-6 rounded-xl border border-emerald-500/30 bg-emerald-500/5 dark:bg-emerald-500/10 p-4 ${className}`}>
      <div className="flex items-start gap-3">
        <ShieldCheck className="w-5 h-5 text-emerald-600 dark:text-emerald-400 mt-0.5 shrink-0" />
        <div>
          <p className="font-semibold text-surface-900 dark:text-white mb-2">
            v1.3.0 验证范围
          </p>
          <p className="text-sm text-surface-600 dark:text-surface-400 mb-2">
            经过充分测试的路径是 YOLO9 和 RF-DETR 的检测、训练与推理，包括 RF-DETR 分割。
          </p>
          <p className="text-sm text-surface-600 dark:text-surface-400">
            在生产环境中，我们建议从 YOLO9 或 RF-DETR 开始。
          </p>
        </div>
      </div>
    </div>
  )
}

function FlagshipCalloutZh({ className = '' }) {
  return (
    <div className={`my-6 rounded-xl border border-libre-500/30 bg-libre-500/5 dark:bg-libre-500/10 p-4 ${className}`}>
      <div className="flex items-start gap-3">
        <Sparkles className="w-5 h-5 text-libre-600 dark:text-libre-400 mt-0.5 shrink-0" />
        <div>
          <p className="font-semibold text-surface-900 dark:text-white mb-1">
            推荐的已验证路径：YOLO9 检测或 RF-DETR 检测 / 分割
          </p>
          <p className="text-sm text-surface-600 dark:text-surface-400">
            这些模型的检测、训练与推理经过最充分的测试。在 v1.3.0 中，请将其他系列、任务以及多 GPU 工作流视为实验性。
          </p>
        </div>
      </div>
    </div>
  )
}

function CompatibilityMatrixZh() {
  const rows = [
    {
      family: 'YOLO9', status: '已验证 detect，单 GPU',
      inference: 'yes', training: 'yes',
      detect: 'yes', segment: '', semantic: '', classify: '', pose: '', obb: '', depth: '', point: '', gaze: '',
      onnx: 'yes', torchscript: 'yes', tensorrt: 'yes', openvino: 'yes', ncnn: 'yes', coreml: 'yes', tflite: 'exp',
    },
    {
      family: 'RF-DETR', status: '已验证 detect、segment 和 pose；OBB 实验性',
      inference: 'yes', training: 'yes',
      detect: 'yes', segment: 'yes', semantic: '', classify: '', pose: 'yes', obb: 'exp', depth: '', point: '', gaze: '',
      onnx: 'yes', torchscript: 'yes', tensorrt: 'yes', openvino: 'yes', ncnn: '', coreml: 'exp', tflite: 'exp',
    },
    {
      family: 'YOLOX', status: '实验性',
      inference: 'exp', training: 'exp',
      detect: 'exp', segment: '', semantic: '', classify: '', pose: '', obb: '', depth: '', point: '', gaze: '',
      onnx: 'exp', torchscript: 'exp', tensorrt: 'exp', openvino: 'exp', ncnn: 'exp', coreml: 'exp', tflite: '',
    },
    {
      family: 'YOLO9-E2E', status: '实验性',
      inference: 'exp', training: 'exp',
      detect: 'exp', segment: '', semantic: '', classify: '', pose: '', obb: '', depth: '', point: '', gaze: '',
      onnx: 'exp', torchscript: 'exp', tensorrt: 'exp', openvino: 'exp', ncnn: '', coreml: '', tflite: '',
    },
    {
      family: 'YOLO-NAS', status: '实验性',
      inference: 'exp', training: 'exp',
      detect: 'exp', segment: '', semantic: '', classify: '', pose: 'exp', obb: '', depth: '', point: '', gaze: '',
      onnx: 'exp', torchscript: 'exp', tensorrt: 'exp', openvino: 'exp', ncnn: 'exp', coreml: '', tflite: '',
    },
    {
      family: 'D-FINE', status: '实验性',
      inference: 'exp', training: 'exp',
      detect: 'exp', segment: '', semantic: '', classify: '', pose: '', obb: '', depth: '', point: '', gaze: '',
      onnx: 'exp', torchscript: 'exp', tensorrt: 'exp', openvino: 'exp', ncnn: 'exp', coreml: '', tflite: '',
    },
    {
      family: 'DEIM', status: '实验性',
      inference: 'exp', training: 'exp',
      detect: 'exp', segment: '', semantic: '', classify: '', pose: '', obb: '', depth: '', point: '', gaze: '',
      onnx: 'exp', torchscript: 'exp', tensorrt: 'exp', openvino: 'exp', ncnn: 'exp', coreml: '', tflite: '',
    },
    {
      family: 'DEIMv2', status: '实验性',
      inference: 'exp', training: 'exp',
      detect: 'exp', segment: '', semantic: '', classify: '', pose: '', obb: '', depth: '', point: '', gaze: '',
      onnx: 'exp', torchscript: 'exp', tensorrt: 'exp', openvino: 'exp', ncnn: 'exp', coreml: '', tflite: '',
    },
    {
      family: 'RT-DETR', status: '实验性',
      inference: 'exp', training: 'exp',
      detect: 'exp', segment: '', semantic: '', classify: '', pose: '', obb: '', depth: '', point: '', gaze: '',
      onnx: 'exp', torchscript: 'exp', tensorrt: 'exp', openvino: 'exp', ncnn: 'exp', coreml: 'exp', tflite: '',
    },
    {
      family: 'RT-DETRv2', status: '实验性',
      inference: 'exp', training: 'exp',
      detect: 'exp', segment: '', semantic: '', classify: '', pose: '', obb: '', depth: '', point: '', gaze: '',
      onnx: 'exp', torchscript: '', tensorrt: '', openvino: '', ncnn: '', coreml: '', tflite: '',
    },
    {
      family: 'RT-DETRv4', status: '实验性',
      inference: 'exp', training: 'exp',
      detect: 'exp', segment: '', semantic: '', classify: '', pose: '', obb: '', depth: '', point: '', gaze: '',
      onnx: 'exp', torchscript: '', tensorrt: '', openvino: '', ncnn: '', coreml: '', tflite: '',
    },
    {
      family: 'PicoDet', status: '实验性',
      inference: 'exp', training: 'exp',
      detect: 'exp', segment: '', semantic: '', classify: '', pose: '', obb: '', depth: '', point: '', gaze: '',
      onnx: 'exp', torchscript: 'exp', tensorrt: 'exp', openvino: '', ncnn: '', coreml: '', tflite: '',
    },
    {
      family: 'RTMDet', status: '实验性',
      inference: 'exp', training: 'exp',
      detect: 'exp', segment: '', semantic: '', classify: '', pose: '', obb: '', depth: '', point: '', gaze: '',
      onnx: 'exp', torchscript: '', tensorrt: '', openvino: '', ncnn: '', coreml: '', tflite: '',
    },
    {
      family: 'EC', status: '实验性',
      inference: 'exp', training: 'exp',
      detect: 'exp', segment: 'exp', semantic: '', classify: '', pose: 'exp', obb: '', depth: '', point: '', gaze: '',
      onnx: 'exp', torchscript: '', tensorrt: '', openvino: '', ncnn: '', coreml: '', tflite: '',
    },
    {
      family: 'DINOv2', status: '新增，实验性（需要 transformers）',
      inference: 'exp', training: 'exp',
      detect: '', segment: '', semantic: 'exp', classify: 'exp', pose: '', obb: '', depth: '', point: '', gaze: '',
      onnx: '', torchscript: '', tensorrt: '', openvino: '', ncnn: '', coreml: '', tflite: '',
    },
    {
      family: 'MobileNetV4', status: '新增，实验性分类器（Apache）',
      inference: 'exp', training: 'exp',
      detect: '', segment: '', semantic: '', classify: 'exp', pose: '', obb: '', depth: '', point: '', gaze: '',
      onnx: 'exp', torchscript: '', tensorrt: '', openvino: '', ncnn: '', coreml: '', tflite: '',
    },
    {
      family: 'ConvNeXt', status: '新增，实验性分类器（Apache）',
      inference: 'exp', training: 'exp',
      detect: '', segment: '', semantic: '', classify: 'exp', pose: '', obb: '', depth: '', point: '', gaze: '',
      onnx: 'exp', torchscript: '', tensorrt: '', openvino: '', ncnn: '', coreml: '', tflite: '',
    },
    {
      family: 'EfficientNetV2', status: '新增，实验性分类器（Apache）',
      inference: 'exp', training: 'exp',
      detect: '', segment: '', semantic: '', classify: 'exp', pose: '', obb: '', depth: '', point: '', gaze: '',
      onnx: 'exp', torchscript: '', tensorrt: '', openvino: '', ncnn: '', coreml: '', tflite: '',
    },
    {
      family: 'Depth Anything V2', status: '新增，实验性；不支持导出',
      inference: 'exp', training: '',
      detect: '', segment: '', semantic: '', classify: '', pose: '', obb: '', depth: 'exp', point: '', gaze: '',
      onnx: '', torchscript: '', tensorrt: '', openvino: '', ncnn: '', coreml: '', tflite: '',
    },
    {
      family: 'FOMO', status: '新增，实验性；不自动下载',
      inference: 'exp', training: 'exp',
      detect: '', segment: '', semantic: '', classify: '', pose: '', obb: '', depth: '', point: 'exp', gaze: '',
      onnx: '', torchscript: '', tensorrt: '', openvino: '', ncnn: '', coreml: '', tflite: '',
    },
    {
      family: 'L2CS', status: '实验性，仅推理',
      inference: 'exp', training: '',
      detect: '', segment: '', semantic: '', classify: '', pose: '', obb: '', depth: '', point: '', gaze: 'exp',
      onnx: '', torchscript: '', tensorrt: '', openvino: '', ncnn: '', coreml: '', tflite: '',
    },
  ]

  const headers = ['模型系列', 'v1.3.0 状态', '推理', '训练', '检测', '分割', '语义', '分类', '姿态', 'OBB', '深度', '点', '视线', 'ONNX', 'TorchScript', 'TensorRT', 'OpenVINO', 'NCNN', 'CoreML', 'TFLite']
  const columns = ['inference', 'training', 'detect', 'segment', 'semantic', 'classify', 'pose', 'obb', 'depth', 'point', 'gaze', 'onnx', 'torchscript', 'tensorrt', 'openvino', 'ncnn', 'coreml', 'tflite']

  return (
    <DocTable
      headers={headers}
      rows={rows.map((row) => [
        <strong key={`${row.family}-family`} className="text-surface-800 dark:text-white whitespace-nowrap">{row.family}</strong>,
        <span key={`${row.family}-status`} className="text-xs leading-relaxed">{row.status}</span>,
        ...columns.map((column) => <MatrixMark key={`${row.family}-${column}`} value={row[column]} />),
      ])}
    />
  )
}

/* ─── 4. Main docs page (Chinese) ─── */
function DocsPageZh({ version = 'v1.3.0', isLatest = true }) {
  const [activeSection, setActiveSection] = useState('introduction')
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [docsCopied, setDocsCopied] = useState(false)

  // Scroll spy - pick the last section whose heading has scrolled past 30% of viewport
  // NOTE: keep the English `sections` ids here so scroll-spy matches the rendered ids.
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

  // On load, honor a #section deep link so a shared URL lands on that section.
  useEffect(() => {
    const id = decodeURIComponent((window.location.hash || '').replace(/^#/, ''))
    if (id && sections.some((s) => s.id === id)) {
      requestAnimationFrame(() => {
        document.getElementById(id)?.scrollIntoView()
        setActiveSection(id)
      })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const navigateTo = (id) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })
    setActiveSection(id)
    window.history.replaceState(null, '', `#${id}`)
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
        <SidebarZh activeSection={activeSection} onNavigate={navigateTo} currentVersion={version} />
      </aside>

      {/* Mobile sidebar toggle */}
      <button
        onClick={() => setMobileMenuOpen(true)}
        className="lg:hidden fixed top-16 left-4 z-30 inline-flex items-center gap-2 rounded-lg border border-surface-200 dark:border-white/[0.1] bg-white/90 dark:bg-surface-900/90 backdrop-blur px-3 py-2 text-sm font-semibold text-surface-700 dark:text-surface-200 shadow-sm hover:bg-white dark:hover:bg-surface-800 transition-colors"
        aria-label="打开文档导航"
      >
        <Menu className="w-4 h-4" />
        菜单
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
                    ? '这是最新的 LibreYOLO 文档（v1.3.0）。如需上一版本，请查看 v1.2.0。'
                    : '保留此归档版本的可链接性，以便较旧的安装、搜索结果和智能体能够定位到正确的文档。'}
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
            <ValidationScopeCalloutZh />
            <CodeBlock language="python">{`from libreyolo import LibreYOLO, SAMPLE_IMAGE

# Default: YOLO9 detection
model = LibreYOLO("LibreYOLO9c.pt")
result = model(SAMPLE_IMAGE, conf=0.25, save=True)

print(f"Detected {len(result)} objects")
print(result.boxes.xyxy)
print(result.saved_path)`}</CodeBlock>

          </motion.div>


          <Divider />

          {/* ────────────── COMPATIBILITY ────────────── */}
          <SectionHeading id="compatibility" icon={CheckCircle2}>兼容性</SectionHeading>
          <P>
            可将此矩阵作为 v1.3.0 的快速支持速查表。<InlineCode>&#10003;</InlineCode>{' '}
            表示已验证路径，<InlineCode>exp</InlineCode> 表示实验性，{' '}
            空白单元格表示当前不支持。经过充分测试的路径是 YOLO9 检测，以及 RF-DETR
            的检测、分割和姿态；其余全部，包括新增的分类、语义、深度和点定位系列，均为实验性。
          </P>
          <CompatibilityMatrixZh />
          <p className="text-sm text-surface-600 dark:text-surface-400 leading-relaxed mb-4">
            Depth Anything V2 没有导出路径。TFLite 导出为实验性，仅限 YOLO9 检测和 RF-DETR
            detect / segment / pose。分类系列（MobileNetV4、ConvNeXt、EfficientNetV2）支持
            ONNX 导出。CoreML 导出会生成 <InlineCode>.mlpackage</InlineCode>{' '}
            包并需要 <InlineCode>libreyolo[coreml]</InlineCode>：仅限 macOS、不支持 INT8，
            且对 RF-DETR、D-FINE、DEIM、DEIMv2 或 EC 不内嵌 NMS。
          </p>

          <Divider />

          <SectionHeading id="installation" icon={Terminal}>安装</SectionHeading>
          <SubHeading>环境要求</SubHeading>
          <ul className="space-y-1.5 mb-4">
            <li className="flex items-center gap-2 text-surface-600 dark:text-surface-400">
              <span className="w-1.5 h-1.5 rounded-full bg-libre-400" />Python 3.10+
            </li>
            <li className="flex items-center gap-2 text-surface-600 dark:text-surface-400">
              <span className="w-1.5 h-1.5 rounded-full bg-libre-400" />PyTorch 2.4+ 和 torchvision 0.19+
            </li>
          </ul>

          <SubHeading>从 PyPI 安装</SubHeading>
          <CodeBlock language="bash">{`pip install libreyolo`}</CodeBlock>
          <P>
            v1.3.0 已发布到 PyPI，因此直接 pip install libreyolo 即可获得本页所述的功能。
          </P>

          <SubHeading>从源码安装</SubHeading>
          <CodeBlock language="bash">{`git clone https://github.com/LibreYOLO/libreyolo.git
cd libreyolo
git checkout dev
pip install -e .`}</CodeBlock>

          <SubHeading>可选依赖</SubHeading>
          <CodeBlock language="bash">{`# ONNX export and inference
pip install libreyolo[onnx]
# or: pip install onnx onnxsim onnxruntime

# RT-DETR compatibility extra (currently no extra packages)
pip install libreyolo[rtdetr]

# RF-DETR support
pip install libreyolo[rfdetr]
# or: pip install transformers

# TensorRT export and inference (NVIDIA GPU)
pip install libreyolo[tensorrt]
# Installs TensorRT CUDA 12 Python packages on Linux/Windows.
# Host driver/CUDA compatibility still matters.

# OpenVINO export and inference (Intel CPU/GPU/VPU)
pip install libreyolo[openvino]
# INT8 export also needs: pip install nncf

# NCNN export and inference
pip install libreyolo[ncnn]
# or: pip install pnnx ncnn

# ByteTrack API compatibility extra
pip install libreyolo[tracking]
# Tracking dependencies are part of the base install in v1.3.0.

# CoreML export and inference (macOS only for runtime)
pip install libreyolo[coreml]
# or: pip install coremltools

# L2CS gaze optional auto-download helper
pip install libreyolo[gaze]

# Install every optional LibreYOLO extra
pip install libreyolo[all]`}</CodeBlock>

          <P>如果使用 <InlineCode>uv</InlineCode>，最可靠的做法是为每个 extra 建立独立的 venv：</P>
          <CodeBlock language="bash">{`# ONNX environment
uv venv .venv-onnx
uv pip install --python .venv-onnx/bin/python -e '.[onnx]'

# RT-DETR environment
uv venv .venv-rtdetr
uv pip install --python .venv-rtdetr/bin/python -e '.[rtdetr]'

# Repeat with .[rfdetr], .[openvino], .[ncnn], .[coreml], .[gaze], .[tracking], or .[tensorrt] as needed`}</CodeBlock>
          <P>
            这样可以避免改动项目环境，并保持可选依赖相互隔离。TensorRT、OpenVINO、NCNN 和 CoreML 等特定厂商 extra 可能仍需平台相关的原生包。
          </P>

          <Divider />

          {/* ────────────── QUICKSTART ────────────── */}
          <SectionHeading id="quickstart" icon={Rocket}>快速开始</SectionHeading>
          <P>
            若要使用测试最充分的路径，请选择单 GPU 的 YOLO9 检测、RF-DETR 检测或 RF-DETR 分割。它们通过同一个工厂加载、接受相同的输入并返回相同的 <InlineCode>Results</InlineCode> 对象，因此你可以在它们之间切换而无需改动周边代码。
          </P>

          <SubHeading>YOLO9 - CNN 旗舰</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO, SAMPLE_IMAGE

# Use the official checkpoint name and let the factory resolve the details
model = LibreYOLO("LibreYOLO9c.pt")

# Run on a single image (SAMPLE_IMAGE ships with the package)
result = model(SAMPLE_IMAGE)

print(f"Found {len(result)} objects")
print(result.boxes.xyxy)   # bounding boxes (N, 4)
print(result.boxes.conf)   # confidence scores (N,)
print(result.boxes.cls)    # class IDs (N,)`}</CodeBlock>

          <SubHeading>RF-DETR - transformer 旗舰</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO, SAMPLE_IMAGE

# Same factory, same call shape - just point at an RF-DETR checkpoint
model = LibreYOLO("LibreRFDETRs.pt")
result = model(SAMPLE_IMAGE)

print(f"Found {len(result)} objects")
print(result.boxes.xyxy)`}</CodeBlock>

          <SubHeading>保存标注后的输出</SubHeading>
          <CodeBlock language="python">{`result = model(SAMPLE_IMAGE, save=True)
print(result.saved_path)   # e.g. runs/detect/predict/parkour.jpg`}</CodeBlock>

          <SubHeading>处理目录</SubHeading>
          <CodeBlock language="python">{`results = model("images/", save=True, batch=4)
for r in results:
    print(f"{r.path}: {len(r)} detections")`}</CodeBlock>

          <Divider />

          {/* ────────────── AVAILABLE MODELS ────────────── */}
          <SectionHeading id="models" icon={Layers}>可用模型</SectionHeading>
          <FlagshipCalloutZh />
          <P>
            LibreYOLO v1.3.0 提供两个已验证的旗舰系列，以及更广泛的受支持和新增模型目录。每个模型都通过同一个 <InlineCode>LibreYOLO()</InlineCode> 工厂加载，但只有下面的已验证路径才应被视为经过充分测试。
          </P>

          <ValidatedModelHeader title="YOLO9 - CNN 旗舰">
            <SupportBadge variant="validated">默认：LibreYOLO9c.pt</SupportBadge>
            <SupportBadge variant="validated">充分测试：检测、训练与推理</SupportBadge>
            <SupportBadge>v1.3.0 中仅检测</SupportBadge>
            <SupportBadge>实验性：多 GPU</SupportBadge>
          </ValidatedModelHeader>
          <DocTable
            headers={['尺寸', '代号', '输入尺寸', '适用场景', '检测检查点']}
            rows={[
              ['Tiny', <InlineCode key="t">&quot;t&quot;</InlineCode>, '640', '快速推理', <HFLink key="cp-t" name="LibreYOLO9t.pt" />],
              ['Small', <InlineCode key="s">&quot;s&quot;</InlineCode>, '640', '均衡', <HFLink key="cp-s" name="LibreYOLO9s.pt" />],
              ['Medium', <InlineCode key="m">&quot;m&quot;</InlineCode>, '640', '更高精度', <HFLink key="cp-m" name="LibreYOLO9m.pt" />],
              ['Compact', <InlineCode key="c">&quot;c&quot;</InlineCode>, '640', '最佳精度', <HFLink key="cp-c" name="LibreYOLO9c.pt" />],
            ]}
          />
          <P>
            在 v1.3.0 中 YOLO9 仅支持检测。非检测的旗舰变体（包括旧的 <InlineCode>-seg</InlineCode> 检查点）已移除；如需分割，请使用下文的 RF-DETR 或 EdgeCrafter。
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")   # detection`}</CodeBlock>

          <ValidatedModelHeader title="RF-DETR - transformer 旗舰">
            <SupportBadge variant="validated">推荐的 transformer 路径</SupportBadge>
            <SupportBadge variant="validated">充分测试：检测、分割、姿态、训练与推理</SupportBadge>
            <SupportBadge>实验性：OBB</SupportBadge>
            <SupportBadge>实验性：多 GPU</SupportBadge>
          </ValidatedModelHeader>
          <DocTable
            headers={['尺寸', '代号', '输入尺寸', '适用场景', '检测检查点']}
            rows={[
              ['Nano', <InlineCode key="n">&quot;n&quot;</InlineCode>, '384', '边缘端', <HFLink key="cp-n" name="LibreRFDETRn.pt" />],
              ['Small', <InlineCode key="s">&quot;s&quot;</InlineCode>, '512', '均衡', <HFLink key="cp-s" name="LibreRFDETRs.pt" />],
              ['Medium', <InlineCode key="m">&quot;m&quot;</InlineCode>, '576', '更高精度', <HFLink key="cp-m" name="LibreRFDETRm.pt" />],
              ['Large', <InlineCode key="l">&quot;l&quot;</InlineCode>, '704', '最高精度', <HFLink key="cp-l" name="LibreRFDETRl.pt" />],
            ]}
          />
          <P>
            LibreYOLO 在 Hugging Face 组织上提供 Apache 干净授权的 RF-DETR 检测尺寸 N/S/M/L。XL/2XL 档位有意不提供。
          </P>
          <P>
            <SupportBadge variant="validated">充分测试</SupportBadge>{' '}
            <strong className="text-surface-800 dark:text-white">分割：</strong>{' '}
            <Checkpoints names={['LibreRFDETRn-seg.pt', 'LibreRFDETRs-seg.pt', 'LibreRFDETRm-seg.pt', 'LibreRFDETRl-seg.pt']} />。
            {' '}更大的 <InlineCode>-seg</InlineCode> 尺寸（<InlineCode>x</InlineCode>、<InlineCode>xx</InlineCode>）
            会拉取上游 RF-DETR seg-XL / seg-2XL 权重，采用非商业许可，
            且不托管在 LibreYOLO 组织上。请参见{' '}
            <a href="#segmentation" className="text-libre-600 dark:text-libre-400 hover:underline">分割</a> 章节。
          </P>
          <P>
            <SupportBadge variant="validated">已支持</SupportBadge>{' '}
            <strong className="text-surface-800 dark:text-white">姿态：</strong>{' '}
            <Checkpoints names={['LibreRFDETRx-pose.pt']} link={false} />（移植自
            RF-DETR v1.8.0 的 GroupPose；仅提供 576 下的 <InlineCode>x</InlineCode> 尺寸）。
          </P>
          <P>
            <SupportBadge>实验性</SupportBadge>{' '}
            <strong className="text-surface-800 dark:text-white">OBB：</strong>{' '}
            <Checkpoints names={['LibreRFDETRn-obb.pt', 'LibreRFDETRs-obb.pt', 'LibreRFDETRm-obb.pt', 'LibreRFDETRl-obb.pt']} link={false} />{' '}
            （旋转框，使用检测的输入尺寸）。
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreRFDETRs.pt")           # detect (validated)
# model = LibreYOLO("LibreRFDETRs-seg.pt")     # segment (validated)
# model = LibreYOLO("LibreRFDETRx-pose.pt")    # pose  (size x)
# model = LibreYOLO("LibreRFDETRn-obb.pt")     # obb   (experimental)`}</CodeBlock>

          <SubHeading>其他检测系列</SubHeading>
          <P>
            与已验证路径共享同一工厂和 API 接口的可检测系列。它们在 v1.3.0 中为实验性。每个检查点名称都链接到{' '}
            <a href="https://huggingface.co/LibreYOLO" target="_blank" rel="noopener noreferrer" className="text-libre-600 dark:text-libre-400 hover:underline">LibreYOLO 组织</a>上的模型卡；
            将任意名称传给 <InlineCode>LibreYOLO()</InlineCode>，工厂会在首次使用时自动获取。
          </P>
          <DocTable
            headers={['系列', '状态', '任务', '检查点']}
            rows={[
              ['YOLOX', <SupportBadge key="b">实验性</SupportBadge>, 'detect', <Checkpoints key="yolox" names={['LibreYOLOXn.pt', 'LibreYOLOXt.pt', 'LibreYOLOXs.pt', 'LibreYOLOXm.pt', 'LibreYOLOXl.pt', 'LibreYOLOXx.pt']} />],
              ['YOLO9-E2E', <SupportBadge key="b">实验性</SupportBadge>, 'detect', <Checkpoints key="y9e2e" names={['LibreYOLO9E2Et.pt', 'LibreYOLO9E2Es.pt', 'LibreYOLO9E2Em.pt', 'LibreYOLO9E2Ec.pt']} />],
              ['YOLO-NAS', <SupportBadge key="b">实验性</SupportBadge>, 'detect, pose', <Checkpoints key="ynas" link={false} names={['LibreYOLONASs.pt', 'LibreYOLONASm.pt', 'LibreYOLONASl.pt', 'LibreYOLONASn-pose.pt', 'LibreYOLONASs-pose.pt', 'LibreYOLONASm-pose.pt', 'LibreYOLONASl-pose.pt']} />],
              ['D-FINE', <SupportBadge key="b">实验性</SupportBadge>, 'detect', <Checkpoints key="dfine" names={['LibreDFINEn.pt', 'LibreDFINEs.pt', 'LibreDFINEm.pt', 'LibreDFINEl.pt', 'LibreDFINEx.pt']} />],
              ['DEIM', <SupportBadge key="b">实验性</SupportBadge>, 'detect', <Checkpoints key="deim" names={['LibreDEIMn.pt', 'LibreDEIMs.pt', 'LibreDEIMm.pt', 'LibreDEIMl.pt', 'LibreDEIMx.pt']} />],
              ['DEIMv2', <SupportBadge key="b">实验性</SupportBadge>, 'detect', <Checkpoints key="deimv2" names={['LibreDEIMv2atto.pt', 'LibreDEIMv2femto.pt', 'LibreDEIMv2pico.pt', 'LibreDEIMv2n.pt', 'LibreDEIMv2s.pt', 'LibreDEIMv2m.pt', 'LibreDEIMv2l.pt', 'LibreDEIMv2x.pt']} />],
              ['RT-DETR', <SupportBadge key="b">实验性</SupportBadge>, 'detect', <Checkpoints key="rtdetr" names={['LibreRTDETRr18.pt', 'LibreRTDETRr34.pt', 'LibreRTDETRr50.pt', 'LibreRTDETRr50m.pt', 'LibreRTDETRr101.pt', 'LibreRTDETRl.pt', 'LibreRTDETRx.pt']} />],
              ['RT-DETRv2', <SupportBadge key="b">实验性</SupportBadge>, 'detect', <Checkpoints key="rtdetrv2" names={['LibreRTDETRv2r18.pt', 'LibreRTDETRv2r34.pt', 'LibreRTDETRv2r50.pt', 'LibreRTDETRv2r50m.pt', 'LibreRTDETRv2r101.pt']} />],
              ['RT-DETRv4', <SupportBadge key="b">实验性</SupportBadge>, 'detect', <Checkpoints key="rtdetrv4" names={['LibreRTDETRv4s.pt', 'LibreRTDETRv4m.pt', 'LibreRTDETRv4l.pt', 'LibreRTDETRv4x.pt']} />],
              ['PicoDet', <SupportBadge key="b">实验性</SupportBadge>, 'detect', <Checkpoints key="picodet" names={['LibrePICODETs.pt', 'LibrePICODETm.pt', 'LibrePICODETl.pt']} />],
              ['RTMDet', <SupportBadge key="b">实验性</SupportBadge>, 'detect', <Checkpoints key="rtmdet" names={['LibreRTMDett.pt', 'LibreRTMDets.pt', 'LibreRTMDetm.pt', 'LibreRTMDetl.pt', 'LibreRTMDetx.pt']} />],
              ['EdgeCrafter', <SupportBadge key="b">实验性</SupportBadge>, 'detect, pose, segment', <Checkpoints key="ec" names={['LibreECs.pt', 'LibreECm.pt', 'LibreECl.pt', 'LibreECx.pt', 'LibreECs-pose.pt', 'LibreECm-pose.pt', 'LibreECl-pose.pt', 'LibreECx-pose.pt', 'LibreECs-seg.pt', 'LibreECm-seg.pt', 'LibreECl-seg.pt', 'LibreECx-seg.pt']} />],
            ]}
          />
          <P className="text-sm">
            <strong className="text-surface-800 dark:text-white">托管说明：</strong>{' '}
            YOLO-NAS 检查点（上方纯文本）托管在 Deci 的 CDN 上，采用其专有权重许可，
            而非 LibreYOLO 的 Hugging Face 组织。工厂仍会在首次使用时自动下载它们。
            DAMO-YOLO 已在 v1.3.0 中移除，不再可加载。
          </P>

          <SubHeading>v1.3.0 中的新模型系列</SubHeading>
          <P>
            v1.3.0 新增了分类、密集语义分割、单目深度和点定位系列。它们通过同一个工厂加载，但属于新增且实验性。DINOv2 需要{' '}
            <InlineCode>pip install libreyolo[rfdetr]</InlineCode>（transformers）。
          </P>
          <DocTable
            headers={['系列', '状态', '任务', '检查点']}
            rows={[
              ['MobileNetV4', <SupportBadge key="b">实验性</SupportBadge>, 'classify', <Checkpoints key="mn4" names={['LibreMobileNetV4s-cls.pt', 'LibreMobileNetV4m-cls.pt', 'LibreMobileNetV4l-cls.pt']} />],
              ['ConvNeXt', <SupportBadge key="b">实验性</SupportBadge>, 'classify', <Checkpoints key="cnx" link={false} names={['LibreConvNeXtt-cls.pt', 'LibreConvNeXts-cls.pt', 'LibreConvNeXtb-cls.pt']} />],
              ['EfficientNetV2', <SupportBadge key="b">实验性</SupportBadge>, 'classify', <Checkpoints key="env2" link={false} names={['LibreEfficientNetV2b0-cls.pt', 'LibreEfficientNetV2b1-cls.pt', 'LibreEfficientNetV2b2-cls.pt', 'LibreEfficientNetV2b3-cls.pt']} />],
              ['DINOv2', <SupportBadge key="b">实验性</SupportBadge>, 'semantic, classify', <Checkpoints key="dino" link={false} names={['LibreDINOv2n.pt', 'LibreDINOv2s.pt', 'LibreDINOv2m.pt', 'LibreDINOv2l.pt', 'LibreDINOv2n-cls.pt', 'LibreDINOv2s-cls.pt', 'LibreDINOv2m-cls.pt', 'LibreDINOv2l-cls.pt']} />],
              ['Depth Anything V2', <SupportBadge key="b">实验性</SupportBadge>, 'depth', <Checkpoints key="depth" link={false} names={['LibreDepthAnythingV2s-depth.pt', 'LibreDepthAnythingV2b-depth.pt', 'LibreDepthAnythingV2l-depth.pt', 'LibreDepthAnythingV2g-depth.pt']} />],
              ['FOMO', <SupportBadge key="b">实验性</SupportBadge>, 'point', <Checkpoints key="fomo" link={false} names={['LibreFOMOs-point.pt', 'LibreFOMOm-point.pt', 'LibreFOMOl-point.pt']} />],
            ]}
          />
          <ul className="space-y-2 my-4">
            <FeatureItem><strong className="text-surface-800 dark:text-white">MobileNetV4</strong> 是商用干净的分类路径：Apache-2.0 ImageNet-1k 权重（s/m/l 分别为 224/224/256），支持预测、top-1/top-5 验证、微调训练和 ONNX 导出。</FeatureItem>
            <FeatureItem><strong className="text-surface-800 dark:text-white">ConvNeXt</strong>（V1 Tiny/Small/Base，224）和 <strong className="text-surface-800 dark:text-white">EfficientNetV2</strong>（b0-b3，224-300）是另外的 Apache-2.0 ImageNet-1k 分类器（精度档）。</FeatureItem>
            <FeatureItem><strong className="text-surface-800 dark:text-white">DINOv2</strong> 是带任务头的 DINOv2 主干：默认在 518 下进行密集语义分割，并在 224 下提供分类线性探针。它不是 RF-DETR 检测器。分类在 v1.3.0 中从 RF-DETR 迁移到这里。</FeatureItem>
            <FeatureItem><strong className="text-surface-800 dark:text-white">Depth Anything V2</strong> 进行单目深度估计（尺寸 s/b/l/g，均在 518）。ViT-S 权重为 Apache-2.0；ViT-B/L/G 为 CC-BY-NC-4.0（非商业）。仅支持推理和零样本验证：不可训练，且不支持导出。</FeatureItem>
            <FeatureItem><strong className="text-surface-800 dark:text-white">FOMO</strong> 是点定位器，为每个目标输出 <InlineCode>(x, y, class, confidence)</InlineCode>。预训练权重不再分发：请传入本地检查点或从头训练。</FeatureItem>
          </ul>
          <P className="text-sm">
            <strong className="text-surface-800 dark:text-white">可提示与 VLM 档位：</strong>{' '}
            LibreSAM（可提示分割，<InlineCode>libreyolo[sam]</InlineCode>）
            和 LibreVLM 视觉语言检测器档位
            （<InlineCode>libreyolo[vlm]</InlineCode>）是单独的类别，它们加载
            上游的 Hugging Face 快照，且不通过{' '}
            <InlineCode>LibreYOLO()</InlineCode> 检测器工厂路由。它们的权重继承
            各上游模型的许可。
          </P>

          <SubHeading>专用模型</SubHeading>
          <DocTable
            headers={['系列', '状态', '任务', '检查点']}
            rows={[
              ['L2CS', <SupportBadge key="b">实验性</SupportBadge>, <span key="t">gaze（仅推理）- 参见 <a href="#gaze" className="text-libre-600 dark:text-libre-400 hover:underline">视线估计</a></span>, <Checkpoints key="l2cs" link={false} names={['LibreL2CSr50.pt']} />],
            ]}
          />
          <P className="text-sm">
            L2CS 架构尺寸包括 r18、r34、r50、r101 和 r152，但上游发布的 Gaze360 检查点为
            ResNet-50。安装 <InlineCode>libreyolo[gaze]</InlineCode> 以获取可选的下载助手，
            或为其他尺寸传入本地检查点路径。L2CS 权重不由 LibreYOLO 托管（Gaze360 数据集
            许可禁止再分发）。
          </P>

          <SubHeading>工厂函数</SubHeading>
          <P>
            对每个模型和运行时都使用 <InlineCode>LibreYOLO()</InlineCode> 工厂。给它一个官方检查点名称或导出产物路径，
            然后让它选择正确的模型系列、任务、类别数和运行时：
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

# Default: YOLO9 detection
model = LibreYOLO("LibreYOLO9c.pt")

# Flagship transformer: RF-DETR
model = LibreYOLO("LibreRFDETRs.pt")
model = LibreYOLO("LibreRFDETRs-seg.pt")        # validated segmentation

# New in v1.3.0
model = LibreYOLO("LibreMobileNetV4s-cls.pt")   # classification (Apache, ImageNet-1k)
model = LibreYOLO("LibreDINOv2n.pt")            # semantic segmentation
model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")  # monocular depth
model = LibreYOLO("LibreFOMOs-point.pt")        # point localization (local weights)

# Exported deployment formats
model = LibreYOLO("model.onnx")                 # ONNX Runtime
model = LibreYOLO("model.engine")               # TensorRT
model = LibreYOLO("model.mlpackage")            # CoreML (macOS)
model = LibreYOLO("model_openvino/")            # OpenVINO (directory)
model = LibreYOLO("model_ncnn/")                # NCNN (directory)`}</CodeBlock>
          <P>
            对于可识别的官方检查点文件名，LibreYOLO 可以自动下载缺失的权重。对于自定义文件名，请指向明确的本地路径。
            新项目请保持使用 YOLO9 检测或 RF-DETR 检测 / 分割；
            其他系列、任务以及新增系列在 v1.3.0 中为实验性。
          </P>

          <Divider />

          {/* ────────────── TASKS & FILENAMES ────────────── */}
          <SectionHeading id="tasks" icon={Tags}>任务与文件名</SectionHeading>
          <P>
            LibreYOLO 使用统一的文件名约定，因此工厂仅凭检查点名称即可识别系列、尺寸和任务：
          </P>
          <CodeBlock language="text">{`Libre<FAMILY><size>[-<task>].pt`}</CodeBlock>

          <SubHeading>任务后缀</SubHeading>
          <DocTable
            headers={['任务', '规范名称', '文件名后缀', '所属系列']}
            rows={[
              ['检测', <InlineCode key="d">&quot;detect&quot;</InlineCode>, '（无 - 隐式）', '大多数系列（默认）'],
              ['实例分割', <InlineCode key="s">&quot;segment&quot;</InlineCode>, <InlineCode key="ss">-seg</InlineCode>, 'RF-DETR、EdgeCrafter'],
              ['语义分割', <InlineCode key="se">&quot;semantic&quot;</InlineCode>, <InlineCode key="ses">-sem</InlineCode>, 'DINOv2'],
              ['姿态估计', <InlineCode key="p">&quot;pose&quot;</InlineCode>, <InlineCode key="ps">-pose</InlineCode>, 'YOLO-NAS、EdgeCrafter、RF-DETR'],
              ['旋转框', <InlineCode key="o">&quot;obb&quot;</InlineCode>, <InlineCode key="os">-obb</InlineCode>, 'RF-DETR（实验性）'],
              ['分类', <InlineCode key="c">&quot;classify&quot;</InlineCode>, <InlineCode key="cs">-cls</InlineCode>, 'MobileNetV4、ConvNeXt、EfficientNetV2、DINOv2'],
              ['单目深度', <InlineCode key="de">&quot;depth&quot;</InlineCode>, <InlineCode key="des">-depth</InlineCode>, 'Depth Anything V2'],
              ['点定位', <InlineCode key="pt">&quot;point&quot;</InlineCode>, <InlineCode key="pts">-point</InlineCode>, 'FOMO'],
              ['视线估计', <InlineCode key="g">&quot;gaze&quot;</InlineCode>, <InlineCode key="gs">-gaze</InlineCode>, 'L2CS'],
            ]}
          />
          <P>
            检测是隐式的（无后缀），遵循常见的 YOLO 约定。
            工厂在 API 边界接受别名
            （<InlineCode>&quot;detection&quot;</InlineCode>、<InlineCode>&quot;seg&quot;</InlineCode>、
            <InlineCode>&quot;keypoints&quot;</InlineCode>、<InlineCode>&quot;cls&quot;</InlineCode> 等）；
            只有上面的规范名称才会出现在文件名中。只有当某个任务在该系列的
            受支持任务集中时，它才可用。
          </P>

          <SubHeading>解析优先级</SubHeading>
          <P>
            加载模型时，任务按以下顺序解析：
          </P>
          <CodeBlock language="text">{`explicit task=  →  checkpoint["task"]  →  filename suffix  →  family default`}</CodeBlock>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

# 1. Filename suffix decides → segment
model = LibreYOLO("LibreRFDETRs-seg.pt")

# 2. Override regardless of filename
model = LibreYOLO("custom_weights.pt", task="segment")

# 3. Detection is implicit
model = LibreYOLO("LibreYOLO9c.pt")  # task="detect"`}</CodeBlock>

          <SubHeading>各系列任务支持</SubHeading>
          <DocTable
            headers={['系列', 'v1.3.0 状态', '默认', '受支持任务']}
            rows={[
              [<strong key="y9">YOLO9</strong>, '单 GPU detect 充分测试；多 GPU 实验性', 'detect', 'detect'],
              [<strong key="rfd">RF-DETR</strong>, '单 GPU detect、segment 和 pose 充分测试；OBB 实验性', 'detect', 'detect, segment, pose, obb'],
              ['YOLOX', '实验性', 'detect', 'detect'],
              ['YOLO9-E2E', '实验性', 'detect', 'detect'],
              ['YOLO-NAS', '实验性', 'detect', 'detect, pose'],
              ['D-FINE / DEIM / DEIMv2', '实验性', 'detect', 'detect'],
              ['RT-DETR / RT-DETRv2 / RT-DETRv4', '实验性', 'detect', 'detect'],
              ['PicoDet / RTMDet', '实验性', 'detect', 'detect'],
              ['EdgeCrafter (EC)', '实验性', 'detect', 'detect, pose, segment'],
              ['DINOv2', '新增，实验性', 'semantic', 'semantic, classify'],
              ['MobileNetV4 / ConvNeXt / EfficientNetV2', '新增，实验性', 'classify', 'classify'],
              ['Depth Anything V2', '新增，实验性', 'depth', 'depth'],
              ['FOMO', '新增，实验性', 'point', 'point'],
              ['L2CS', '实验性', 'gaze', 'gaze（仅推理）'],
            ]}
          />

          <SubHeading>示例</SubHeading>
          <CodeBlock language="text">{`# Detection (implicit)
LibreYOLO9c.pt
LibreRFDETRs.pt
LibreRTDETRr50.pt

# Instance segmentation (-seg)
LibreRFDETRs-seg.pt
LibreECm-seg.pt

# Semantic segmentation (-sem)
LibreDINOv2n.pt          # semantic is DINOv2's default; -sem optional

# Pose (-pose)
LibreYOLONASn-pose.pt
LibreECs-pose.pt
LibreRFDETRx-pose.pt     # pose; size x

# Oriented boxes (-obb)
LibreRFDETRn-obb.pt      # obb; experimental

# Classification (-cls)
LibreMobileNetV4s-cls.pt
LibreConvNeXtt-cls.pt
LibreEfficientNetV2b0-cls.pt
LibreDINOv2n-cls.pt      # DINOv2 linear probe

# Depth (-depth)
LibreDepthAnythingV2s-depth.pt

# Point (-point)
LibreFOMOs-point.pt

# Gaze (-gaze optional; only task for L2CS)
LibreL2CSr50.pt`}</CodeBlock>

          <SubHeading>已弃用别名</SubHeading>
          <P>
            <InlineCode>LibreYOLORTDETR</InlineCode> 和 <InlineCode>LibreYOLORFDETR</InlineCode> 分别是 <InlineCode>LibreRTDETR</InlineCode> 和 <InlineCode>LibreRFDETR</InlineCode> 的旧名称。它们仍可解析，但会发出 <InlineCode>DeprecationWarning</InlineCode> - 方便时请更新导入。
          </P>

          <Divider />

          {/* ────────────── PREDICTION ────────────── */}
          <SectionHeading id="prediction" icon={Crosshair}>预测</SectionHeading>
          <P>
            单 GPU 预测路径在 YOLO9 检测、RF-DETR 检测和 RF-DETR 分割上经过充分测试。其他系列和任务使用相同的 API，但在 v1.3.0 中为实验性。
          </P>

          <SubHeading>基础预测</SubHeading>
          <CodeBlock language="python">{`result = model("image.jpg")`}</CodeBlock>

          <SubHeading>全部预测参数</SubHeading>
          <CodeBlock language="python">{`result = model(
    "image.jpg",
    conf=0.25,            # confidence threshold (default: 0.25)
    iou=0.45,             # NMS IoU threshold (default: 0.45)
    imgsz=640,            # input size override (default: model's native)
    device="auto",        # "auto", "cpu", "mps", "0", "cuda:0", ...
    classes=[0, 2, 5],    # filter to specific class IDs (default: all)
    max_det=300,          # max detections per image (default: 300)
    augment=False,        # test-time augmentation where implemented
    save=True,            # save annotated image (default: False)
    batch=4,              # directory batch size
    stream=False,         # video only: yield frame results instead of a list
    vid_stride=1,         # video only: process every N-th frame
    show=False,           # video only: display annotated frames
    tiling=False,         # large-image tiled detection
    overlap_ratio=0.2,    # tile overlap ratio
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
result = model("s3://bucket/image.jpg")
result = model("gs://bucket/image.jpg")

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

# BytesIO
from io import BytesIO
result = model(BytesIO(open("photo.jpg", "rb").read()))

# Directory of images
results = model("images/", batch=4)`}</CodeBlock>

          <SubHeading>处理结果</SubHeading>
          <P>
            每次预测都返回一个 <InlineCode>Results</InlineCode> 对象（对目录则返回它们的列表）：
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
# Tracking adds a track_id column before conf/cls.
result.boxes.data        # shape (N, 6), or (N, 7) when tracked

# Metadata
result.orig_shape        # (height, width) of original image
result.path              # source file path (or None)
result.names             # {0: "person", 1: "bicycle", ...}

# Move to CPU / convert to numpy
result_cpu = result.cpu()
boxes_np = result.boxes.numpy()`}</CodeBlock>

          <SubHeading>类别过滤</SubHeading>
          <P>将检测结果过滤到特定类别 ID：</P>
          <CodeBlock language="python">{`# Only detect people (class 0) and cars (class 2)
result = model("image.jpg", classes=[0, 2])`}</CodeBlock>

          <SubHeading>内存中的批量推理</SubHeading>
          <P>
            v1.3.0 新增：<InlineCode>model.predict()</InlineCode> 接受由内存中图像组成的列表或元组
            （NumPy 数组、PIL 图像或张量），并将它们作为真正的堆叠前向批次运行。设置 <InlineCode>batch &gt; 1</InlineCode> 可在支持的系列上
            真正批处理前向计算；返回结果列表，每个输入对应一个。
          </P>
          <CodeBlock language="python">{`import numpy as np
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")

frames = [
    np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
    np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
    np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
]

results = model(frames, batch=4)   # list/tuple -> true batched inference
for r in results:
    print(len(r), r.boxes.xyxy.shape)`}</CodeBlock>

          <SubHeading>模型信息</SubHeading>
          <P>
            <InlineCode>model.info()</InlineCode>（v1.3.0 新增）返回一个 JSON 友好的字典，包含
            系列、尺寸、任务、参数量、输入尺寸和类别名称，并在 <InlineCode>verbose=True</InlineCode> 时
            打印可读的摘要。
          </P>
          <CodeBlock language="python">{`meta = model.info(detailed=False, verbose=True)
# meta -> {"family": ..., "size": ..., "task": ..., "params": ..., "imgsz": ..., "names": {...}, ...}`}</CodeBlock>

          <Divider />

          {/* ────────────── TILED INFERENCE ────────────── */}
          <SectionHeading id="tiled-inference" icon={Grid3x3}>分块推理</SectionHeading>
          <P>
            对于远大于模型输入尺寸的图像（如卫星影像、无人机画面），分块推理会将图像切分为重叠的分块，对每个分块运行检测，再合并结果。
          </P>
          <P>
            在 v1.3.0 中分块仅支持检测。它会拒绝分割掩码，且不能与 <InlineCode>augment=True</InlineCode> 组合使用。
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
            <FeatureItem><InlineCode>final_image.jpg</InlineCode> - 绘制了所有合并检测的完整图像</FeatureItem>
            <FeatureItem><InlineCode>grid_visualization.jpg</InlineCode> - 显示分块网格叠加的图像</FeatureItem>
            <FeatureItem><InlineCode>tiles/</InlineCode> - 各个分块裁剪图</FeatureItem>
            <FeatureItem><InlineCode>metadata.json</InlineCode> - 分块参数和检测数量</FeatureItem>
          </ul>
          <P>
            如果图像已经小于模型的输入尺寸，则会自动跳过分块。
          </P>

          <Divider />

          {/* ────────────── VIDEO INFERENCE ────────────── */}
          <SectionHeading id="video-inference" icon={Video}>视频推理</SectionHeading>
          <P>
            将任意视频文件传给旗舰模型，LibreYOLO 会根据扩展名自动检测格式。支持：<InlineCode>.mp4</InlineCode>、<InlineCode>.avi</InlineCode>、<InlineCode>.mov</InlineCode>、<InlineCode>.mkv</InlineCode>、<InlineCode>.webm</InlineCode>、<InlineCode>.gif</InlineCode> 以及其他常见容器。
          </P>

          <SubHeading>保存标注后的视频</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")
results = model("clip.mp4", save=True)
# Saved under runs/detect/predict*/clip.mp4`}</CodeBlock>

          <SubHeading>流式结果（内存平稳）</SubHeading>
          <P>
            对于长视频，传入 <InlineCode>stream=True</InlineCode> 可获得一个生成器。每次迭代产出一帧的 <InlineCode>Results</InlineCode> - 不会在内存中缓存完整列表。
          </P>
          <CodeBlock language="python">{`for result in model("long_clip.mp4", stream=True):
    print(f"frame {result.frame_idx}: {len(result)} detections")`}</CodeBlock>

          <SubHeading>帧抽样</SubHeading>
          <CodeBlock language="python">{`# Process every 2nd frame (halves compute and saved fps)
results = model("clip.mp4", vid_stride=2, save=True)`}</CodeBlock>

          <SubHeading>实时预览</SubHeading>
          <CodeBlock language="python">{`# Display annotated frames in an OpenCV window while processing
results = model("clip.mp4", show=True)`}</CodeBlock>

          <SubHeading>用于自定义流水线的 VideoSource / VideoWriter</SubHeading>
          <P>
            当你需要完全控制解码和编码时 - 自定义帧变换、混入跟踪输出、写入非默认编解码器 - 可直接使用这些构建块：
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
          <SectionHeading id="tracking" icon={Activity}>跟踪</SectionHeading>
          <P>
            LibreYOLO 提供两种运动跟踪器，它们消费任意检测器的 <InlineCode>Results</InlineCode> 并添加持久的轨迹 ID：<strong className="text-surface-800 dark:text-white">ByteTrack</strong>（默认）
            和 <strong className="text-surface-800 dark:text-white">OC-SORT</strong>（v1.3.0 新增），后者对遮挡和
            非线性运动更鲁棒。跟踪在单 GPU 的 YOLO9 检测和
            RF-DETR 检测上测试最充分；其他检测系列在 v1.3.0 中为实验性。
          </P>

          <SubHeading>安装</SubHeading>
          <CodeBlock language="bash">{`pip install libreyolo[tracking]   # compatibility extra; tracking deps ship in base dev install`}</CodeBlock>

          <SubHeading>视频跟踪助手</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")

for result in model.track(
    "clip.mp4",
    track_conf=0.25,
    iou=0.45,
    save=True,             # writes runs/track/<video_stem>.mp4 by default
    vid_stride=1,
):
    print(result.frame_idx, result.track_id)`}</CodeBlock>
          <P>
            <InlineCode>model.track()</InlineCode> 是用于视频文件的生成器。它逐帧运行检测，内部使用较低的 ByteTrack 置信度进行恢复，并产出已填充 <InlineCode>result.track_id</InlineCode> 和 <InlineCode>result.boxes.id</InlineCode> 的 <InlineCode>Results</InlineCode>。
          </P>

          <SubHeading>基础循环</SubHeading>
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
            调用 <InlineCode>tracker.update()</InlineCode> 之后，<InlineCode>result.boxes.id</InlineCode> 保存轨迹 ID，且 <InlineCode>result.boxes.is_track</InlineCode> 为 <InlineCode>True</InlineCode>。
          </P>

          <SubHeading>TrackConfig 参数</SubHeading>
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

          <SubHeading>片段之间重置</SubHeading>
          <CodeBlock language="python">{`tracker.reset()   # clears tracked / lost / removed lists and the ID counter`}</CodeBlock>

          <SubHeading>OC-SORT（抗遮挡）</SubHeading>
          <P>
            在 <InlineCode>model.track()</InlineCode> 上用 <InlineCode>tracker=&quot;ocsort&quot;</InlineCode> 选择 OC-SORT。ByteTrack 仍为默认。使用 OC-SORT 时，{' '}
            <InlineCode>track_conf</InlineCode> 映射到跟踪器的{' '}
            <InlineCode>det_thresh</InlineCode>（在 ByteTrack 中它映射到{' '}
            <InlineCode>track_high_thresh</InlineCode>）。
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")

for result in model.track(
    "clip.mp4",
    tracker="ocsort",      # "bytetrack" (default) or "ocsort"
    track_conf=0.25,       # maps to OC-SORT det_thresh
    iou=0.45,
    save=True,
):
    print(result.frame_idx, result.track_id)`}</CodeBlock>

          <P>
            传入 <InlineCode>OCSortConfig</InlineCode> 以完全控制。提供配置实例会按类型选择跟踪器，
            因此此时会忽略 <InlineCode>tracker=</InlineCode> 字符串。
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO, OCSortConfig

cfg = OCSortConfig(
    det_thresh=0.25,     # boxes above this drive association and spawn new tracks
    max_age=30,          # frames a track survives without an observation
    min_hits=3,          # consecutive hits before a track is reported
    iou_threshold=0.3,   # minimum IoU for a valid association
    delta_t=3,           # frame span used to estimate velocity direction
    inertia=0.2,         # weight of the velocity-direction (momentum) term
    use_byte=False,      # enable the BYTE low-score recovery pass
)

model = LibreYOLO("LibreYOLO9c.pt")
for result in model.track("clip.mp4", tracker_config=cfg, save=True):
    print(result.frame_idx, result.track_id)`}</CodeBlock>

          <Divider />

          {/* ────────────── SEGMENTATION ────────────── */}
          <SectionHeading id="segmentation" icon={Scissors}>分割</SectionHeading>
          <ValidationScopeCalloutZh />
          <P>
            RF-DETR 分割是 v1.3.0 中的分割路径，也是经过充分测试的选项。EdgeCrafter（<InlineCode>-seg</InlineCode>）也提供分割头，但为实验性。YOLO9 不再提供分割头：自 v1.3.0 起它仅支持检测。
          </P>

          <SubHeading>运行分割</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

# RF-DETR segmentation, the heavily tested segmentation path
model = LibreYOLO("LibreRFDETRs-seg.pt")
result = model("photo.jpg")

# EdgeCrafter segmentation is also available but experimental
# model = LibreYOLO("LibreECs-seg.pt")

# Segmentation returns boxes + masks
print(result.boxes.xyxy)        # bounding boxes (N, 4)
print(result.boxes.cls)         # class IDs (N,)
print(result.masks.data.shape)  # (N, H, W) tensor of binary masks`}</CodeBlock>

          <SubHeading>掩码表示</SubHeading>
          <CodeBlock language="python">{`# Raw bitmasks
result.masks.data        # tensor (N, H, W) - original image resolution

# Polygon contours (one ndarray of (M, 2) per instance)
result.masks.xy          # absolute pixel coords
result.masks.xyn         # normalized to [0, 1]

# Move / convert like Boxes
result.masks.cpu()
result.masks.numpy()`}</CodeBlock>

          <SubHeading>保存标注后的输出</SubHeading>
          <P>
            <InlineCode>save=True</InlineCode> 会自动绘制框和半透明掩码叠加。
          </P>
          <CodeBlock language="python">{`model("photo.jpg", save=True)`}</CodeBlock>

          <SubHeading>训练分割</SubHeading>
          <P>
            RF-DETR 分割使用 RF-DETR 的 COCO 格式训练流水线，属于经过充分测试的单 GPU 范围。EdgeCrafter 分割训练可用，但为实验性。YOLO9 分割训练已在 v1.3.0 中移除。
          </P>

          <Divider />

          {/* ────────────── ORIENTED BOXES (OBB) ────────────── */}
          <SectionHeading id="obb" icon={Rotate3d}>旋转边界框 (OBB)</SectionHeading>
          <div className="flex flex-wrap gap-2 mb-5">
            <SupportBadge variant="experimental">YOLO9: t, s, m, c</SupportBadge>
            <SupportBadge variant="experimental">RF-DETR: n, s, m, l</SupportBadge>
          </div>
          <P>
            旋转框带有一个旋转角，这正是航拍图像、文档以及密集排布场景所需要的。YOLO9 在其检测头上增加了一个角度分支；RF-DETR 则在其解码器中加入了一个可学习的角度嵌入。
          </P>

          <SubHeading>推理与 OBB 结果</SubHeading>
          <P>
            Results 暴露一个 <InlineCode>obb</InlineCode> 字段。角度以 <strong className="text-surface-800 dark:text-white">弧度</strong>为单位。
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9t-obb.pt")
r = model.predict("aerial.jpg")

for i in range(len(r.obb.cls)):
    cx, cy, w, h, angle = r.obb.xywhr[i]  # angle in radians
    corners = r.obb.xyxyxyxy[i]           # 4 (x, y) corner points
    conf, cls = r.obb.conf[i], r.obb.cls[i]`}</CodeBlock>
          <DocTable
            headers={['字段', '形状', '含义']}
            rows={[
              [<InlineCode key="a">obb.xywhr</InlineCode>, 'N x 5', '[cx, cy, w, h, angle]，angle 以弧度为单位。'],
              [<InlineCode key="b">obb.xyxyxyxy</InlineCode>, 'N x 4 x 2', '每个框的四个角点。'],
              [<InlineCode key="c">obb.conf</InlineCode>, 'N', '每个框的置信度。'],
              [<InlineCode key="d">obb.cls</InlineCode>, 'N', '每个框的类别 id。'],
            ]}
          />

          <SubHeading>数据集格式与训练</SubHeading>
          <P>
            OBB 使用标准的检测式数据 YAML，但标签是 YOLO-OBB 文本文件，每行 <strong className="text-surface-800 dark:text-white">恰好九个字段</strong>：一个类别 id，后跟四个归一化角点。角度由角点推导得出，并不存储。
          </P>
          <CodeBlock language="text" filename="labels/aerial_001.txt">{`# class_id  x1 y1  x2 y2  x3 y3  x4 y4   (all normalized to [0, 1])
0  0.51 0.32  0.66 0.38  0.62 0.55  0.47 0.49
2  0.10 0.71  0.18 0.69  0.20 0.80  0.12 0.82`}</CodeBlock>
          <P>
            普通的检测检查点无法直接加载到 OBB 模型中。从检测转到 OBB 仅允许作为训练时的热启动：传入 <InlineCode>pretrained=True</InlineCode>（YOLO9）或 RF-DETR 上的显式迁移标志。在角点感知增强落地之前，OBB 会禁用 Mosaic 与 mixup，且不支持分块推理。
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO9

model = LibreYOLO9(None, size="t", task="obb")
# Warm-start the backbone from a same-family detect checkpoint
result = model.train(data="dota8.yaml", pretrained=True, epochs=100, imgsz=640)

# CLI equivalent
# libreyolo train model=LibreYOLO9t.pt data=dota8.yaml --task obb`}</CodeBlock>
          <P>
            验证使用旋转 IoU 的 AP，在 OBB 指标组下以 mAP50 与 mAP50-95 报告。
          </P>

          <Divider />

          {/* ────────────── POSE ESTIMATION ────────────── */}
          <SectionHeading id="pose" icon={PersonStanding}>姿态估计</SectionHeading>
          <P>
            姿态（人体关键点）估计可在 <InlineCode>YOLO-NAS (-pose)</InlineCode>、{' '}
            <InlineCode>EdgeCrafter (-pose)</InlineCode> 以及 v1.3.0 新增的{' '}
            <InlineCode>RF-DETR (-pose)</InlineCode> 上运行。每个姿态模型都是单类别
            （&quot;person&quot;），具有 17 个 COCO 关键点。
          </P>

          <SubHeading>运行姿态</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

# YOLO-NAS pose
model = LibreYOLO("LibreYOLONASs-pose.pt")
result = model("people.jpg")

# EdgeCrafter pose
# model = LibreYOLO("LibreECs-pose.pt")

# Per-person bbox + 17 keypoints
print(result.boxes.xyxy)          # person boxes (N, 4)
print(result.keypoints.xy.shape)  # (N, 17, 2) pixel coordinates`}</CodeBlock>

          <P>
            <SupportBadge variant="validated">已支持</SupportBadge>{' '}
            RF-DETR 姿态仅提供一个 <InlineCode>x</InlineCode> 尺寸的检查点：{' '}
            <InlineCode>LibreRFDETRx-pose.pt</InlineCode>。
          </P>
          <CodeBlock language="python">{`# RF-DETR pose (size x only)
model = LibreYOLO("LibreRFDETRx-pose.pt")
result = model("people.jpg")
print(result.keypoints.xy.shape)  # (N, 17, 2)`}</CodeBlock>

          <SubHeading>关键点 API</SubHeading>
          <CodeBlock language="python">{`result.keypoints.xy        # (N, K, 2) absolute pixel coords
result.keypoints.xyn       # (N, K, 2) normalized to [0, 1]
result.keypoints.conf      # (N, K) per-keypoint confidence (None if model doesn't emit it)
result.keypoints.has_visible  # (N, K) bool - conf > 0

result.keypoints.cpu()
result.keypoints.numpy()`}</CodeBlock>

          <SubHeading>保存标注后的输出</SubHeading>
          <CodeBlock language="python">{`model("people.jpg", save=True)  # draws boxes + skeleton`}</CodeBlock>

          <P>
            YOLO-NAS 支持姿态训练；EdgeCrafter 姿态目前仅推理。RF-DETR 姿态仅提供 <InlineCode>x</InlineCode> 尺寸的检查点。YOLO9 仅检测，不提供姿态检查点。
          </P>

          <Divider />

          {/* ────────────── GAZE ESTIMATION ────────────── */}
          <SectionHeading id="gaze" icon={Eye}>视线估计</SectionHeading>
          <P>
            视线方向估计由 <InlineCode>LibreL2CS</InlineCode> 系列提供，它是 L2CS-Net 的移植，具有 ResNet 主干和两个角度分箱分类头。这是一个两阶段模型：上游的人脸检测器定位人脸，然后视线头以弧度预测每张人脸的 pitch 和 yaw。它在 v1.3.0 中仅推理且为实验性。
          </P>

          <SubHeading>安装</SubHeading>
          <CodeBlock language="bash">{`pip install libreyolo[gaze]   # optional Google Drive helper for Gaze360 weights`}</CodeBlock>
          <P>
            已发布的 L2CS ResNet-50 权重在 Gaze360 上训练，LibreYOLO 不做镜像。在没有可选助手的情况下，请传入本地检查点路径，或按照 <InlineCode>LibreL2CS</InlineCode> 打印的手动下载说明操作。
          </P>

          <SubHeading>两阶段推理</SubHeading>
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

          <SubHeading>解码角度</SubHeading>
          <CodeBlock language="python">{`import math

for i in range(len(result.gaze)):
    pitch_rad, yaw_rad = result.gaze.data[i].tolist()
    pitch_deg = pitch_rad * 180.0 / math.pi
    yaw_deg = yaw_rad * 180.0 / math.pi
    print(f"face {i}: pitch={pitch_deg:.1f} deg, yaw={yaw_deg:.1f} deg")`}</CodeBlock>

          <P>
            在命令行中：<InlineCode>libreyolo predict model=LibreL2CSr50.pt source=portrait.jpg --face-detector path/to/face.pt</InlineCode>。
          </P>

          <Divider />

          {/* ────────────── OPEN-VOCABULARY DETECTION (LibreVLM) ────────────── */}
          <SectionHeading id="open-vocabulary" icon={ScanSearch}>开放词表检测</SectionHeading>
          <div className="flex flex-wrap gap-2 mb-5">
            <SupportBadge variant="experimental">仅推理</SupportBadge>
            <SupportBadge variant="experimental">Python API</SupportBadge>
          </div>
          <P>
            传统检测器在检测头中固化了一份固定的类别列表。<InlineCode>LibreVLM</InlineCode> 抛弃了这个限制：它封装现代指令微调的视觉语言模型，提示它们输出边界框，解析生成的文本，并返回你在 YOLO9 和 RF-DETR 中已经用过的同一个 <InlineCode>Results</InlineCode> 对象。类别列表只是你在运行时提供的一组词，因此新增一个类别毫无成本，而且是零样本生效。
          </P>
          <ul className="space-y-2 mb-4">
            <FeatureItem><strong className="text-surface-800 dark:text-white">开放词表。</strong> 检测 <InlineCode>&quot;pink car&quot;</InlineCode>、<InlineCode>&quot;license plate&quot;</InlineCode> 或 <InlineCode>&quot;the small island&quot;</InlineCode>，无需为它们训练任何检测头。</FeatureItem>
            <FeatureItem><strong className="text-surface-800 dark:text-white">一个工厂，一份契约。</strong> <InlineCode>LibreVLM(...)</InlineCode> 返回标准的 <InlineCode>Results</InlineCode>，含 <InlineCode>boxes.xyxy</InlineCode>、<InlineCode>boxes.cls</InlineCode>、<InlineCode>boxes.conf</InlineCode>，以及 <InlineCode>.plot()</InlineCode> 和 <InlineCode>.save()</InlineCode>。</FeatureItem>
            <FeatureItem><strong className="text-surface-800 dark:text-white">可替换的后端。</strong> 一个别名字符串背后是六个模型系列，从 230M 的 Florence-2 到 8B 的 Qwen3-VL。</FeatureItem>
            <FeatureItem><strong className="text-surface-800 dark:text-white">一个原始的逃生通道。</strong> 当你需要的不止是框时，<InlineCode>chat()</InlineCode> 提供自由形式的图像问答。</FeatureItem>
          </ul>
          <Callout icon={AlertTriangle} tone="amber" title="仅推理层级">
            <p>
              LibreVLM 是一个纯 Python 的推理层级：尚无训练、验证、导出或 CLI 路径，且置信度分数是占位符。在此之上构建之前，请先阅读本节末尾的局限性。
            </p>
          </Callout>

          <SubHeading>安装</SubHeading>
          <P>
            LibreVLM 位于可选的 <InlineCode>vlm</InlineCode> extra 之后。它会引入较新的 <InlineCode>transformers</InlineCode> 以及一些处理器所需的辅助库。
          </P>
          <CodeBlock language="bash">{`pip install 'libreyolo[vlm]'`}</CodeBlock>
          <P>
            权重在首次使用时从 Hugging Face Hub 下载到本地的 <InlineCode>weights/</InlineCode> 文件夹。较大的后端推荐使用 GPU，但每个模型也都能通过 <InlineCode>device=&quot;cpu&quot;</InlineCode> 在 CPU 上运行。
          </P>

          <SubHeading>快速开始</SubHeading>
          <P>
            构造一个模型，声明你关心的词，然后预测。默认后端是 Qwen3-VL-4B，它是该层级中最强的检测器，并采用 Apache-2.0 许可证。
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreVLM

# Qwen3-VL-4B by default; weights autodownload on first use
model = LibreVLM()

# The vocabulary is just words. Any words.
model.set_classes(["pink car", "wheel"])

result = model.predict("street.jpg")

print(result.boxes.xyxy)   # pixel [x1, y1, x2, y2]
print(result.boxes.cls)    # ids into ["pink car", "wheel"]
result.plot()              # same drawing helpers as any LibreYOLO model
result.save("out.jpg")`}</CodeBlock>

          <SubHeading>支持的后端</SubHeading>
          <P>
            通过传给 <InlineCode>LibreVLM(...)</InlineCode> 的别名来选择后端。仅给出系列名会解析为其默认尺寸。最强的检测器是 <strong className="text-surface-800 dark:text-white">Qwen3-VL</strong>、<strong className="text-surface-800 dark:text-white">LFM2-VL</strong> 和 <strong className="text-surface-800 dark:text-white">Florence-2</strong>。
          </P>
          <DocTable
            headers={['系列', '别名', '尺寸（参数量）', '许可证', '说明']}
            rows={[
              [
                <strong key="q" className="text-surface-800 dark:text-white whitespace-nowrap">Qwen3-VL</strong>,
                <code key="qa" className="font-mono text-xs">qwen3-vl-2b / -4b / -8b</code>,
                '2B / 4B / 8B',
                'Apache-2.0',
                <span key="qn">默认且最强。推荐作为起点。</span>,
              ],
              [
                <strong key="l" className="text-surface-800 dark:text-white whitespace-nowrap">LFM2-VL</strong>,
                <code key="la" className="font-mono text-xs">lfm2-vl-450m / -1.6b</code>,
                '450M / 1.6B',
                'LFM Open License',
                <span key="ln">边缘尺寸，小型检测器表现意外出色。下载前有提示。</span>,
              ],
              [
                <strong key="i" className="text-surface-800 dark:text-white whitespace-nowrap">InternVL3</strong>,
                <code key="ia" className="font-mono text-xs">internvl3-1b / -2b / -8b</code>,
                '1B / 2B / 8B',
                'Qwen License',
                <span key="in">8B 定位效果好；小尺寸较弱。下载前有提示。</span>,
              ],
              [
                <strong key="f" className="text-surface-800 dark:text-white whitespace-nowrap">Florence-2</strong>,
                <code key="fa" className="font-mono text-xs">florence-2-base / -large</code>,
                '0.23B / 0.77B',
                'MIT',
                <span key="fn">专为定位打造的模型。框很紧致，无 <InlineCode>chat()</InlineCode>。</span>,
              ],
              [
                <strong key="s" className="text-surface-800 dark:text-white whitespace-nowrap">SmolVLM2</strong>,
                <code key="sa" className="font-mono text-xs">smolvlm2-500m / -2.2b</code>,
                '500M / 2.2B',
                'Apache-2.0',
                <span key="sn">小巧快速；检测能力较弱。适合快速试用。</span>,
              ],
              [
                <strong key="k" className="text-surface-800 dark:text-white whitespace-nowrap">Kosmos-2</strong>,
                <code key="ka" className="font-mono text-xs">kosmos-2</code>,
                '~1.6B',
                'MIT',
                <span key="kn">2023 年的定位模型。框较粗糙，无 <InlineCode>chat()</InlineCode>。</span>,
              ],
            ]}
          />
          <ul className="space-y-2 mb-4">
            <FeatureItem><strong className="text-surface-800 dark:text-white">最佳质量：</strong> <InlineCode>qwen3-vl-8b</InlineCode> 或 <InlineCode>qwen3-vl-4b</InlineCode>（默认）。</FeatureItem>
            <FeatureItem><strong className="text-surface-800 dark:text-white">紧致的框、占用小：</strong> <InlineCode>florence-2-large</InlineCode>。</FeatureItem>
            <FeatureItem><strong className="text-surface-800 dark:text-white">边缘 / CPU：</strong> <InlineCode>lfm2-vl-450m</InlineCode> 或 <InlineCode>smolvlm2-500m</InlineCode>。</FeatureItem>
            <FeatureItem><strong className="text-surface-800 dark:text-white">完全宽松的许可证：</strong> 任意尺寸的 Qwen3-VL、SmolVLM2、Florence-2 或 Kosmos-2。</FeatureItem>
          </ul>
          <Callout icon={ShieldCheck} tone="emerald" title="许可证">
            <p>
              Qwen3-VL 和 SmolVLM2 采用 Apache-2.0；Florence-2 和 Kosmos-2 采用 MIT。LFM2-VL 和 InternVL3 采用非 OSI 许可证，会在首次下载前发出一次性提示，以便你为商业用途做出知情选择。
            </p>
          </Callout>

          <SubHeading>设置词表</SubHeading>
          <P>
            词表是开放词表检测的核心。用一组标签字符串调用 <InlineCode>set_classes()</InlineCode>。它是持久的：会在之后每一次 <InlineCode>predict()</InlineCode> 和 <InlineCode>track()</InlineCode> 调用中保留，直到你再次设置。它返回 <InlineCode>self</InlineCode>，因此可以链式调用。
          </P>
          <CodeBlock language="python">{`# Sticky and chainable
model = LibreVLM("qwen3-vl-2b").set_classes(["person", "dog", "cat"])

# Set it once at construction instead
model = LibreVLM("lfm2-vl-450m", names=["boat"], device="cpu")

# Re-set any time to change what you are looking for
model.set_classes(["a red car", "a blue truck"])`}</CodeBlock>
          <P>
            标签可以是任意短语。它们在不区分大小写时必须唯一，并且你必须传入一个列表，而不是单个字符串。如果你从不调用 <InlineCode>set_classes()</InlineCode>，模型会回退到 COCO-80 词表，这样即使是裸 <InlineCode>predict()</InlineCode> 也能给出合理的结果。
          </P>

          <SubHeading>预测与结果</SubHeading>
          <P>
            <InlineCode>predict()</InlineCode>（以及等价的 <InlineCode>model(...)</InlineCode> 调用）接受与任何 LibreYOLO 检测器相同的输入类型：路径、PIL 图像、numpy 数组、URL、文件夹或视频。<InlineCode>stream=True</InlineCode> 和 <InlineCode>track()</InlineCode> 也都能用。
          </P>
          <CodeBlock language="python">{`result = model.predict(
    source="image.jpg",  # path | PIL | ndarray | URL | folder | video
    conf=0.25,           # see note below: scoring is synthetic
    classes=[0],         # optional: keep only these vocabulary ids
    max_det=300,
)`}</CodeBlock>
          <DocTable
            headers={['字段', '形状 / 类型', '含义']}
            rows={[
              [<InlineCode key="a">result.boxes.xyxy</InlineCode>, 'N x 4', '像素框 [x1, y1, x2, y2]，缩放到原始图像尺寸。'],
              [<InlineCode key="b">result.boxes.cls</InlineCode>, 'N', '类别 id，索引到你的 set_classes() 词表。'],
              [<InlineCode key="c">result.boxes.conf</InlineCode>, 'N', '合成置信度：每个框都是 1.0（见“局限性”）。'],
              [<InlineCode key="d">result.plot() / .save()</InlineCode>, '-', '常用的绘制与保存辅助方法。'],
            ]}
          />
          <P>
            在底层，LibreVLM 会宽容地解析模型输出（处理 markdown 代码围栏、多余的散文、重复的框以及被截断的数组），把自由文本标签映射回你的类别 id，并丢弃任何不在你词表中的标签。正是最后这一步，让一个自由生成的模型表现得像一个闭集检测器。
          </P>

          <SubHeading>示例</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreVLM

# Detect a specific colored object
model = LibreVLM("qwen3-vl-4b")
model.set_classes(["red car"])
result = model.predict("parking_lot.jpg")
print(f"Found {len(result.boxes.cls)} red car(s)")
result.save("red_cars.jpg")`}</CodeBlock>
          <CodeBlock language="python">{`# Tight boxes with Florence-2 (a purpose-built grounder)
model = LibreVLM("florence-2-large")
model.set_classes(["a red car", "license plate"])
result = model.predict("car.jpg")
result.plot()`}</CodeBlock>
          <CodeBlock language="python">{`# Run on CPU with a built-in sample image
from libreyolo import LibreVLM, SAMPLE_IMAGE

model = LibreVLM("lfm2-vl-450m", device="cpu")
# No set_classes() -> falls back to the COCO-80 vocabulary
result = model.predict(SAMPLE_IMAGE)
print(model.names[result.boxes.cls[0]])  # e.g. "person"`}</CodeBlock>
          <CodeBlock language="python">{`# Batches, folders, and video
model = LibreVLM().set_classes(["forklift", "pallet"])

# A whole folder
for result in model.predict("warehouse_frames/", stream=True):
    result.save()

# A video file (frames are processed one at a time)
model.predict("warehouse.mp4", save=True)`}</CodeBlock>

          <SubHeading>原始对话</SubHeading>
          <P>
            有时你想要的是模型本身，而不是检测器。采用对话模板的系列暴露了 <InlineCode>chat()</InlineCode>，它接受一张图像和一个自由形式的提示，并原样返回解码后的文本。可用于计数、生成描述或快速的视觉问答。
          </P>
          <CodeBlock language="python">{`model = LibreVLM("qwen3-vl-4b")

answer = model.chat("harbor.jpg", "How many boats are docked? Answer with a number.")
print(answer)`}</CodeBlock>
          <Callout icon={AlertTriangle} tone="amber">
            <p>
              <InlineCode>chat()</InlineCode> 在采用对话模板的系列上可用（Qwen3-VL、LFM2-VL、SmolVLM2、InternVL3）。Florence-2 和 Kosmos-2 是基于任务 token 的定位模型，会抛出 <InlineCode>NotImplementedError</InlineCode>；请对它们使用 <InlineCode>predict()</InlineCode>。
            </p>
          </Callout>

          <SubHeading>后端差异</SubHeading>
          <P>
            每个系列都返回同样的 <InlineCode>Results</InlineCode>，但抵达方式各不相同。对话系列被提示输出一个 JSON 框数组；定位模型则使用专门的任务 token。
          </P>
          <DocTable
            headers={['系列', '提示方式', '坐标空间', 'chat()']}
            rows={[
              ['Qwen3-VL', 'JSON 框提示', '0 到 1000，重新缩放', '是'],
              ['LFM2-VL', 'JSON 框提示', '归一化 0 到 1', '是'],
              ['SmolVLM2', 'JSON 框提示', '归一化 0 到 1', '是'],
              ['InternVL3', 'JSON 框提示', '0 到 1000，重新缩放', '是'],
              ['Florence-2', '任务 token', '原生像素', '否'],
              ['Kosmos-2', 'Grounding 提示', '归一化，重新缩放', '否'],
            ]}
          />
          <P>
            对于对话系列，你可以用构造函数参数 <InlineCode>prompt=</InlineCode> 覆盖检测提示，并用 <InlineCode>max_new_tokens=</InlineCode> 限制生成长度。设备和 dtype 会自动解析：CUDA 上为 bf16 或 fp16，CPU 上为 fp32。
          </P>

          <SubHeading>局限性</SubHeading>
          <ul className="space-y-2 mb-4">
            <FeatureItem><strong className="text-surface-800 dark:text-white">合成置信度。</strong> 每个框的得分都是 1.0。因此 <InlineCode>conf=</InlineCode> 过滤表现为全有或全无，而非真正的阈值。</FeatureItem>
            <FeatureItem><strong className="text-surface-800 dark:text-white">无 mAP / 验证。</strong> <InlineCode>val()</InlineCode> 会抛出异常，因为合成分数会让 COCO mAP 产生误导。</FeatureItem>
            <FeatureItem><strong className="text-surface-800 dark:text-white">无训练或导出。</strong> <InlineCode>train()</InlineCode> 和 <InlineCode>export()</InlineCode> 会抛出异常。请在上游微调 VLM，然后加载得到的权重。</FeatureItem>
            <FeatureItem><strong className="text-surface-800 dark:text-white">跟踪能力受限。</strong> <InlineCode>track()</InlineCode> 可以运行，但统一的分数会让跟踪器的低置信度恢复阶段失效。</FeatureItem>
            <FeatureItem><strong className="text-surface-800 dark:text-white">一次一张图像。</strong> 在 v1 中生成是串行的，因此更大的 <InlineCode>batch=</InlineCode> 值不会带来加速。</FeatureItem>
            <FeatureItem><strong className="text-surface-800 dark:text-white">仅 Python API。</strong> <InlineCode>libreyolo</InlineCode> CLI 尚不能解析 VLM 别名。</FeatureItem>
          </ul>
          <Callout icon={Eye} tone="libre" title="它的优势所在">
            <p>
              当类别集合是开放式的、经常变化，或难以提前标注时，就使用 LibreVLM：快速原型、长尾或稀有类别，以及“用文字描述要找的东西”这类工作流。当你需要校准过的置信度、吞吐量或可部署的产物时，请用上文的 <a href="#training" className="text-libre-600 dark:text-libre-400 hover:underline">训练</a> 一节训练闭合词表的 YOLO9 或 RF-DETR。
            </p>
          </Callout>

          <Divider />

          {/* ────────────── CLASSIFICATION ────────────── */}
          <SectionHeading id="classification" icon={Tags}>分类</SectionHeading>
          <P>
            v1.3.0 新增：整图分类。提供两个系列，面向不同需求。<InlineCode>LibreMobileNetV4</InlineCode> 是生产级分类器（Apache-2.0 ImageNet-1k 权重，可导出为 ONNX）。<InlineCode>LibreDINOv2</InlineCode> 配合 <InlineCode>task=classify</InlineCode> 是 DINOv2 主干加线性探针，非常适合迁移学习，但其发布的权重为演示级，且暂不支持导出。这是一个预发布任务，因此细节可能在正式发布前变化。
          </P>

          <DocTable
            headers={['系列', '检查点', '输入', '权重', '微调', 'ONNX 导出']}
            rows={[
              ['LibreMobileNetV4', 'LibreMobileNetV4{s,m,l}-cls.pt', '224 / 224 / 256', 'Apache-2.0 ImageNet-1k（生产级）', '交叉熵', '支持'],
              ['LibreDINOv2（classify）', 'LibreDINOv2{n,s,m,l}-cls.pt', '224', 'Imagenette 演示级（10 类）', '线性探针', '不支持'],
            ]}
          />

          <SubHeading>LibreMobileNetV4（生产级分类器）</SubHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="validated">Apache-2.0 ImageNet-1k 权重</SupportBadge>
            <SupportBadge variant="experimental">v1.3.0 新增</SupportBadge>
          </div>
          <P>
            原生 MobileNetV4-conv 移植（源自 timm），其 1000 类 ImageNet-1k 权重可逐比特一致地加载。尺寸 <InlineCode>s</InlineCode> / <InlineCode>m</InlineCode> 在 224 运行，<InlineCode>l</InlineCode> 在 256 运行。检查点：
          </P>
          <Checkpoints names={['LibreMobileNetV4s-cls.pt', 'LibreMobileNetV4m-cls.pt', 'LibreMobileNetV4l-cls.pt']} />

          <P>加载并预测。单张图像返回一个 <InlineCode>Results</InlineCode>；直接从中读取 <InlineCode>.probs</InlineCode>（传入列表则返回列表）。</P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

# MobileNetV4-conv-Small, Apache-2.0 ImageNet-1k weights (auto-downloaded if missing)
model = LibreYOLO("LibreMobileNetV4s-cls.pt")
result = model("cat.jpg")            # single image -> one Results

probs = result.probs                 # whole-image class vector, length = num classes
print(probs.top1, probs.top1conf)    # top-1 class id (int) and its confidence
print(probs.top5, probs.top5conf)    # 5 class ids and 5 confidences
print(result.names[probs.top1])      # human-readable class name`}</CodeBlock>

          <P>微调到自定义类别集（ImageFolder 布局）。分类头会自动按数据集类别数重建；ImageNet 预训练主干可干净地迁移。</P>
          <CodeBlock language="python">{`from libreyolo import LibreMobileNetV4

model = LibreMobileNetV4(size="s")   # ImageNet-pretrained backbone
model.train(
    data="imagenette160",            # known name, dataset root, or .zip URL
    epochs=5,
    batch=64,
    lr0=1e-3,                        # AdamW + cosine, 1-epoch warmup
    imgsz=224,
)`}</CodeBlock>

          <P>验证（top-1 / top-5 精度）：</P>
          <CodeBlock language="python">{`model = LibreYOLO("LibreMobileNetV4s-cls.pt")
metrics = model.val(data="imagenette160")
print(metrics["metrics/accuracy_top1"])
print(metrics["metrics/accuracy_top5"])`}</CodeBlock>

          <P>导出为 ONNX（已验证与 eager 模式逐比特一致）。ONNX 图输出单个 logits 张量。</P>
          <CodeBlock language="python">{`model = LibreYOLO("LibreMobileNetV4s-cls.pt")
path = model.export(format="onnx", imgsz=224)   # single output: logits [batch, num_classes]

# Interop note: the ONNX output is RAW LOGITS, not softmaxed. The PyTorch
# predict path applies softmax for you; non-Python consumers must apply it
# themselves before reading probabilities.`}</CodeBlock>

          <SubHeading>LibreDINOv2 分类（线性探针 / 迁移）</SubHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="experimental">演示级权重（Imagenette）</SupportBadge>
            <SupportBadge variant="experimental">不支持导出</SupportBadge>
          </div>
          <P>
            冻结风格的 DINOv2-S 编码器加可训练线性头，在 224 运行。<InlineCode>n</InlineCode> / <InlineCode>s</InlineCode> / <InlineCode>m</InlineCode> / <InlineCode>l</InlineCode> 尺寸只控制投影头宽度：四者共享同一个 DINOv2-S 编码器，因此发布的检查点精度几乎相同。随附的 <InlineCode>-cls</InlineCode> 权重为演示级（在 Imagenette 上训练，10 类），因此请将该系列视为迁移学习选项，而非可直接替换的 1000 类分类器。检查点：
          </P>
          <Checkpoints names={['LibreDINOv2n-cls.pt', 'LibreDINOv2s-cls.pt', 'LibreDINOv2m-cls.pt', 'LibreDINOv2l-cls.pt']} link={false} />

          <P>加载并预测（与 MobileNetV4 相同的 <InlineCode>Probs</InlineCode> 接口）：</P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreDINOv2s-cls.pt")   # DINOv2-S backbone + linear probe (224)
result = model("springer.jpg")
print(result.probs.top1, result.probs.top1conf)`}</CodeBlock>

          <P>
            微调用于迁移。用 <InlineCode>task=&quot;classify&quot;</InlineCode> 构建全新模型以获得全新分类头，或加载随附的 <InlineCode>-cls</InlineCode> 检查点继续训练。为获得最佳精度，请从随附检查点而非全新分类头开始微调，并保持默认的 <InlineCode>lr=1e-4</InlineCode>（更高的学习率收敛更差）。
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreDINOv2

# Fresh DINOv2 backbone + random linear head, sized to the dataset
model = LibreDINOv2(size="s", task="classify", nb_classes=3)
model.train(data="path/to/imagefolder", epochs=5, lr=1e-4, batch=4)

# Validate the same way (top-1 / top-5)
metrics = model.val(data="path/to/imagefolder")
print(metrics["metrics/accuracy_top1"])`}</CodeBlock>

          <P>LibreDINOv2 未实现导出。如果你需要可导出的分类器，请使用 LibreMobileNetV4。</P>
          <CodeBlock language="python">{`model = LibreYOLO("LibreDINOv2s-cls.pt")
model.export(format="onnx")
# raises NotImplementedError: Export is not yet implemented for LibreDINOv2.`}</CodeBlock>

          <SubHeading>数据集布局（两个系列）</SubHeading>
          <P>
            分类使用 ImageNet 风格的 ImageFolder 目录树（按文件夹组织，而非标签文件）。类别索引按文件夹名称排序分配。<InlineCode>data=</InlineCode> 接受数据集根目录、已知名称（如 <InlineCode>imagenette160</InlineCode>）或 <InlineCode>.zip</InlineCode> URL。
          </P>
          <CodeBlock language="text">{`dataset_root/
  train/                # required; one subfolder per class
    class_a/img001.jpg
    class_a/img002.jpg
    class_b/img003.jpg
  val/                  # required for validation; same class folders as train
    class_a/img010.jpg
    class_b/img011.jpg`}</CodeBlock>

          <SubHeading>Results.probs 参考</SubHeading>
          <CodeBlock language="python">{`probs = result.probs        # Probs payload, 1-D vector of length = num classes
probs.data                  # raw tensor / ndarray of class probabilities
probs.top1                  # int   - argmax class id
probs.top5                  # list  - 5 class ids, highest first
probs.top1conf              # float - confidence of the top-1 class
probs.top5conf              # 5 confidences, aligned with probs.top5`}</CodeBlock>

          <ul className="space-y-2 my-4">
            <FeatureItem>MobileNetV4 权重为生产级（Apache-2.0 ImageNet-1k，逐比特一致加载）。DINOv2 分类权重为演示级（Imagenette，10 类）。</FeatureItem>
            <FeatureItem>v1.3.0 中没有 LibreRFDETR 分类器。分类已迁移到 LibreMobileNetV4 和 LibreDINOv2 系列；旧的 LibreRFDETR*-cls 检查点在加载时会被拒绝。</FeatureItem>
            <FeatureItem>使用默认配方从头微调 DINOv2 在 Imagenette 上的 top-1 上限约为 0.93，低于随附的 0.976。从随附的 -cls 检查点开始微调以恢复精度。</FeatureItem>
            <FeatureItem>ONNX 分类输出为原始 logits。请在非 Python 消费方中自行应用 softmax。</FeatureItem>
            <FeatureItem>预测单张图像返回一个 Results。直接读取 result.probs，或传入列表并对列表取索引：model([&quot;a.jpg&quot;])[0].probs。</FeatureItem>
          </ul>

          <Divider />

          {/* ────────────── DEPTH ESTIMATION ────────────── */}
          <SectionHeading id="depth" icon={Mountain}>深度估计</SectionHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="experimental">v1.3.0 新增</SupportBadge>
            <SupportBadge variant="experimental">仅推理与验证</SupportBadge>
          </div>
          <P>
            v1.3.0 新增：通过 <InlineCode>LibreDepthAnythingV2</InlineCode> 进行单目深度估计，它是 Depth Anything V2 的移植（DINOv2 编码器加 DPT 头，NeurIPS 2024）。它预测一张密集的相对逆深度图：值越大越靠近相机，不隐含任何度量单位。尺寸 <InlineCode>s</InlineCode> / <InlineCode>b</InlineCode> / <InlineCode>l</InlineCode> / <InlineCode>g</InlineCode> 对应 ViT-S / B / L / G，均在 518 运行。这是一个预发布任务，仅支持推理和零样本验证：不支持训练，也不支持导出。
          </P>
          <P>
            检查点。只有 ViT-S 检查点为 Apache-2.0 并自动托管：<Checkpoints names={['LibreDepthAnythingV2s-depth.pt']} link={false} />。更大的编码器 <Checkpoints names={['LibreDepthAnythingV2b-depth.pt', 'LibreDepthAnythingV2l-depth.pt', 'LibreDepthAnythingV2g-depth.pt']} link={false} /> 为 CC-BY-NC-4.0，LibreYOLO 不做再分发；请使用 <InlineCode>weights/convert_depth_anything_v2_weights.py</InlineCode> 转换官方上游检查点。
          </P>

          <SubHeading>运行深度估计</SubHeading>
          <P>输入 <InlineCode>imgsz</InlineCode> 必须能被 14 整除（DINOv2 的 patch 网格）。深度图在原始图像画布上返回。</P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

# ViT-S encoder, Apache-2.0 weights (commercial use OK)
model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
result = model("street.jpg")

depth = result.depth_map          # DepthMap payload, (H, W) float on the original canvas
print(depth.data.shape)           # (H, W)
print(depth.min, depth.max, depth.mean)   # relative inverse depth: higher = closer
norm = depth.normalized()         # rescaled to [0, 1] over finite values`}</CodeBlock>

          <SubHeading>DepthMap API</SubHeading>
          <CodeBlock language="python">{`depth = result.depth_map
depth.data          # (H, W) float tensor / ndarray, relative inverse depth
depth.min           # min over finite values
depth.max           # max over finite values
depth.mean          # mean over finite values
depth.normalized()  # (H, W) rescaled to [0, 1]; non-finite pixels become 0

depth.cpu()
depth.numpy()`}</CodeBlock>

          <SubHeading>零样本验证</SubHeading>
          <P>
            验证通过共享的深度验证器以零样本方式运行，并报告标准深度指标（AbsRel、RMSE 和 delta 阈值）。验证器会信箱填充到固定正方形并排除填充像素；由于预测使用 Depth Anything 原生的保持纵横比缩放，非正方形的验证指标是预测的有据近似。
          </P>
          <CodeBlock language="python">{`metrics = model.val(data="depth_dataset.yaml")
print(metrics["metrics/abs_rel"])   # absolute relative error (lower is better)
print(metrics["metrics/rmse"])      # root mean squared error
print(metrics["metrics/delta1"])    # fraction within a 1.25x ratio (higher is better)`}</CodeBlock>

          <SubHeading>不支持</SubHeading>
          <CodeBlock language="python">{`model.train(data="...")          # raises NotImplementedError - DA V2 is inference + val only
model.export(format="onnx")      # raises NotImplementedError - depth export is out of scope`}</CodeBlock>

          <ul className="space-y-2 my-4">
            <FeatureItem>许可是分开的：ViT-S（尺寸 s）权重为 Apache-2.0，可用于商业用途。ViT-B / ViT-L / ViT-G（尺寸 b / l / g）为 CC-BY-NC-4.0（非商业），LibreYOLO 不做再分发。</FeatureItem>
            <FeatureItem>若用于商业用途，请坚持使用尺寸 s。</FeatureItem>
            <FeatureItem>深度为相对逆深度，没有度量单位。如果你需要以米为单位，请自行标定。</FeatureItem>
            <FeatureItem>imgsz 必须能被 14 整除。批量预测被禁用，因为保持纵横比缩放会导致每张图像尺寸不一。</FeatureItem>
          </ul>

          <Divider />

          {/* ────────────── POINT LOCALIZATION ────────────── */}
          <SectionHeading id="point-localization" icon={MapPin}>点定位</SectionHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="experimental">实验性</SupportBadge>
          </div>
          <P>
            <InlineCode>LibreFOMO</InlineCode> 是 FOMO 风格的点定位器（尺寸 <InlineCode>s</InlineCode> / <InlineCode>m</InlineCode> / <InlineCode>l</InlineCode>），用于质心式检测：每个检测不是框，而是一个图像坐标。预测以 <InlineCode>result.points</InlineCode> 形式返回。预训练的 LibreFOMO 权重不会自动下载，因此请传入本地检查点路径（或从头训练，这属于实验性，需要 <InlineCode>allow_experimental=True</InlineCode>）。
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

# LibreFOMO weights are not hosted by LibreYOLO - pass a local checkpoint
model = LibreYOLO("path/to/LibreFOMOm-point.pt")
result = model("scene.jpg")

points = result.points       # Points payload, (N, 4) rows: x, y, class, confidence
print(points.xy)             # (N, 2) absolute pixel coords
print(points.xyn)            # (N, 2) normalized to [0, 1]
print(points.cls, points.conf)`}</CodeBlock>

          <Divider />

          <SectionHeading id="training" icon={GraduationCap}>训练</SectionHeading>
          <ValidationScopeCalloutZh />
          <P>
            经过充分测试的训练路径是单 GPU 的 YOLO9 检测、RF-DETR 检测和 RF-DETR 分割。其他模型系列的训练器和多 GPU 工作流也可使用，但测试较少。在 v1.3.0 中 YOLO9 仅检测，因此没有 YOLO9 分割或姿态训练。
          </P>

          <SubHeading>YOLO9 - CNN 旗舰训练</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

# Fine-tune from a pretrained checkpoint (recommended)
model = LibreYOLO("LibreYOLO9c.pt")

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
    device="0",              # "" | "cpu" | "cuda" | "0" | "0,1"
    workers=8,
    seed=0,

    # Output
    project="runs/train",
    name="yolo9_exp",
    exist_ok=False,

    # Training features
    amp=True,                # automatic mixed precision
    patience=50,             # early stopping patience
    resume=False,            # resume from loaded checkpoint
    pretrained=True,         # transfer-learning init (True, a path, or None)
    cache="disk",            # cache decoded images: False | True/"ram" | "disk"
    freeze=10,               # freeze first N groups, or a list of indices / module names
    save_plots=True,         # write final validation plots to the run dir
)

print(f"Best mAP50-95: {results['best_mAP50_95']:.3f}")
print(f"Best checkpoint: {results['best_checkpoint']}")`}</CodeBlock>
          <P>
            训练完成后，模型实例会自动以最优权重重新加载，因此你可以立即调用 <InlineCode>model(...)</InlineCode>。<InlineCode>freeze</InlineCode>、<InlineCode>cache</InlineCode>、<InlineCode>pretrained</InlineCode> 和 <InlineCode>save_plots</InlineCode> 为 v1.3.0 新增，并在所有基于训练器的系列中受支持。
          </P>

          <SubHeading>RF-DETR - transformer 旗舰训练</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreRFDETRs.pt")

results = model.train(
    data="path/to/data.yaml",
    epochs=100,
    batch_size=4,            # NOTE: RF-DETR uses batch_size, not batch
    lr=1e-4,
    output_dir="runs/train/rfdetr_exp",
)`}</CodeBlock>
          <P>
            RF-DETR 有自己的训练签名（<InlineCode>batch_size</InlineCode>、<InlineCode>lr</InlineCode>、<InlineCode>output_dir</InlineCode>），但它使用 LibreYOLO 的数据集配置加载器。为检测或分割传入 <InlineCode>data.yaml</InlineCode>；COCO/Roboflow 风格的标注布局可从该配置中引用。
          </P>

          <SubHeading>LoRA 微调（RF-DETR）</SubHeading>
          <P>
            <SupportBadge variant="experimental">实验性</SupportBadge>{' '}
            <InlineCode>lora=True</InlineCode> 会向 RF-DETR 主干注入 LoRA 适配器，用于
            低显存微调。它需要可选的 <InlineCode>peft</InlineCode> 依赖
            （<InlineCode>pip install &quot;libreyolo[lora]&quot;</InlineCode>），目前仅限
            RF-DETR；其他系列会抛出明确错误，而不是忽略该标志。
          </P>
          <CodeBlock language="python">{`model = LibreYOLO("LibreRFDETRs.pt")
results = model.train(data="data.yaml", epochs=50, lora=True)`}</CodeBlock>

          <SubHeading>实验日志记录器</SubHeading>
          <P>
            v1.3.0 新增：传入 <InlineCode>loggers=</InlineCode> 可将指标流式发送到 TensorBoard、
            MLflow 或 Weights &amp; Biases。接受名称（<InlineCode>&quot;tensorboard&quot;</InlineCode>、{' '}
            <InlineCode>&quot;mlflow&quot;</InlineCode>、<InlineCode>&quot;wandb&quot;</InlineCode>）、已配置的日志记录器
            实例，或两者混合的可迭代对象。每个后端都是可选 extra
            （<InlineCode>libreyolo[tensorboard]</InlineCode>、<InlineCode>[mlflow]</InlineCode>、{' '}
            <InlineCode>[wandb]</InlineCode>）。
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO
from libreyolo.training.loggers import MLflowLogger

model = LibreYOLO("LibreYOLO9c.pt")

# By name
model.train(data="coco128.yaml", loggers="tensorboard")

# Mix configured instances and names
model.train(
    data="coco128.yaml",
    loggers=[MLflowLogger(experiment_name="my-exp"), "tensorboard"],
)`}</CodeBlock>
          <P>
            日志记录器仅为 Python API 功能。命令行没有相应标志；其余新增的训练参数
            （<InlineCode>--task</InlineCode>、<InlineCode>--cache</InlineCode>、{' '}
            <InlineCode>--lora</InlineCode>、<InlineCode>--freeze</InlineCode>、{' '}
            <InlineCode>--save-plots</InlineCode>）在命令行中可用。
          </P>

          <SubHeading>训练结果字典</SubHeading>
          <CodeBlock language="python">{`{
    "final_loss": 2.31,
    "best_mAP50": 0.682,
    "best_mAP50_95": 0.451,
    "best_epoch": 87,
    "save_dir": "runs/train/yolo9_exp",
    "best_checkpoint": "runs/train/yolo9_exp/weights/best.pt",
    "last_checkpoint": "runs/train/yolo9_exp/weights/last.pt",
}`}</CodeBlock>

          <SubHeading>恢复训练</SubHeading>
          <CodeBlock language="python">{`# Load the checkpoint with the factory, then resume
model = LibreYOLO("runs/train/yolo9_exp/weights/last.pt")
results = model.train(data="coco128.yaml", resume=True)`}</CodeBlock>

          <SubHeading>自定义数据集 YAML 格式</SubHeading>
          <CodeBlock language="yaml" filename="data.yaml">{`path: /path/to/dataset
train: images/train
val: images/val
test: images/test  # optional

nc: 3
names: ["cat", "dog", "bird"]`}</CodeBlock>

          <SubHeading>其他训练路径</SubHeading>
          <P>
            其他系列也有训练器钩子，但在 v1.3.0 中它们不是推荐路径。新工作请保持使用 YOLO9 检测或 RF-DETR 检测/分割；仅出于兼容性、基准复现或针对性研究使用实验性训练器。PicoDet、RTMDet 和 EC 训练需要显式的 <InlineCode>allow_experimental=True</InlineCode> 确认。
          </P>

          <SubHeading>从 YAML 配置训练</SubHeading>
          <P>
            每个 <InlineCode>model.train(...)</InlineCode> 都接受 <InlineCode>cfg=&quot;train.yaml&quot;</InlineCode>，以从文件加载全部参数。显式 kwargs 仍优先于 yaml 中的值，因此你可以用 yaml 作为基线，并按运行覆盖单个字段。
          </P>
          <CodeBlock language="python">{`model = LibreYOLO("LibreYOLO9c.pt")
results = model.train(cfg="configs/yolo9_finetune.yaml")
# Override individual fields:
# results = model.train(cfg="configs/yolo9_finetune.yaml", epochs=50)`}</CodeBlock>

          <SubHeading>梯度累积</SubHeading>
          <P>
            传入 <InlineCode>nbs</InlineCode>（名义批大小）以启用梯度累积。训练器每 <InlineCode>nbs / batch</InlineCode> 次前向后步进一次优化器，从而让你在较小硬件上以配方的参考批大小进行训练。
          </P>
          <CodeBlock language="python">{`# Effective batch 64 on a single GPU that only fits batch=8
model.train(data="coco128.yaml", batch=8, nbs=64)`}</CodeBlock>

          <SubHeading>分布式训练（DDP，实验性）</SubHeading>
          <P>
            YOLO9 和 RF-DETR 通过 PyTorch DistributedDataParallel 支持多 GPU 训练，但多 GPU 不在 v1.3.0 经过充分测试的范围内。使用 <InlineCode>torchrun</InlineCode> 启动训练脚本：
          </P>
          <CodeBlock language="bash">{`# 4-GPU node
torchrun --nproc_per_node=4 train_yolo9.py

# Multi-node - see PyTorch's torchrun docs for --nnodes / --rdzv_endpoint`}</CodeBlock>
          <CodeBlock language="python" filename="train_yolo9.py">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")
# Pass device="" (auto-detect) and let torchrun set the rank
model.train(data="coco128.yaml", epochs=300, batch=16)`}</CodeBlock>

          <Divider />

          {/* ────────────── LoRA / DoRA ────────────── */}
          <SectionHeading id="lora" icon={Layers2}>LoRA / DoRA 微调</SectionHeading>
          <div className="flex flex-wrap gap-2 mb-5">
            <SupportBadge variant="experimental">RF-DETR: n, s, m, l</SupportBadge>
          </div>
          <P>
            LoRA 式适配器让你通过训练一小组低秩矩阵来微调 RF-DETR 的 Transformer 主干，同时保持基础权重冻结。这能削减优化器与梯度的显存占用，非常适合在普通硬件上将一个强力检查点适配到新领域。
          </P>

          <SubHeading>启用方式</SubHeading>
          <P>
            整个公开 API 就是 <InlineCode>train()</InlineCode> 上的一个标志。没有 rank、alpha 或目标模块等参数可调；配方固定为一套经过充分测试的配置。底层实现使用 <strong className="text-surface-800 dark:text-white">DoRA</strong>（权重分解的 LoRA，秩 16），应用于 DINOv2 注意力的 query、key 与 value 投影。
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("rf-detr-nano.pth")   # sizes n, s, m, l
result = model.train(
    data="data.yaml",
    lora=True,        # DoRA on the frozen DINOv2 backbone
    epochs=100, batch_size=4, lr=1e-4,
)

# Resume: LoRA is auto-detected from the checkpoint, no need to repeat the flag
model.train(data="data.yaml", resume=True)`}</CodeBlock>
          <CodeBlock language="bash">{`# CLI equivalent
libreyolo train --model rf-detr-nano.pth --data data.yaml --lora`}</CodeBlock>

          <SubHeading>检查点与导出</SubHeading>
          <ul className="space-y-2 mb-4">
            <FeatureItem>训练检查点会保留适配器张量，配置中也会记录已使用 LoRA，因此加载与续训会自动重建适配器图。</FeatureItem>
            <FeatureItem>检测头始终保持可训练，因此你仍可适配到新的类别数。</FeatureItem>
            <FeatureItem><InlineCode>export()</InlineCode> 会将适配器合并回稠密权重。导出的模型是普通模型，不带 <InlineCode>peft</InlineCode> 依赖。</FeatureItem>
            <FeatureItem>LoRA 仅限 RF-DETR；向其他家族传入 <InlineCode>lora=True</InlineCode> 会抛出明确的错误。</FeatureItem>
          </ul>
          <Callout icon={ShieldCheck} tone="emerald" title="安装额外依赖">
            <p>
              LoRA 训练需要适配器依赖：<InlineCode>pip install &quot;libreyolo[lora]&quot;</InlineCode>，它会引入 RF-DETR 相关组件与 <InlineCode>peft</InlineCode>。导出（已合并）的模型在推理时无需该依赖。
            </p>
          </Callout>

          <Divider />

          {/* ────────────── VALIDATION ────────────── */}
          <SectionHeading id="validation" icon={CheckCircle2}>验证</SectionHeading>
          <P>
            在验证集上运行 COCO 标准评估。经过充分测试的验证路径是单 GPU 的 YOLO9 检测、RF-DETR 检测和 RF-DETR 分割。
          </P>
          <CodeBlock language="python">{`results = model.val(
    data="coco128.yaml",   # dataset config
    batch=16,
    imgsz=640,
    conf=0.001,            # low conf for mAP calculation
    iou=0.6,               # NMS IoU threshold
    split="val",           # "val", "test", or "train"
    save_json=False,       # save predictions as COCO JSON
    verbose=True,          # print per-class metrics
    plots=True,            # save validation plots (metrics, per-class AP, confusion matrix); alias for save_plots
)

print(f"mAP50:    {results['metrics/mAP50']:.3f}")
print(f"mAP50-95: {results['metrics/mAP50-95']:.3f}")`}</CodeBlock>

          <SubHeading>验证结果字典</SubHeading>
          <P>
            默认情况下，LibreYOLO 使用 COCO 评估，返回精确率、召回率、AP/AR 指标以及每张图像的耗时：
          </P>
          <CodeBlock language="python">{`{
    "metrics/mAP50-95": 0.489,   # COCO primary metric (AP@[.5:.95])
    "metrics/mAP50": 0.721,      # AP@0.5 (PASCAL VOC style)
    "metrics/mAP75": 0.534,      # AP@0.75 (strict)
    "metrics/precision": 0.68,
    "metrics/recall": 0.61,
    "metrics/precision(B)": 0.68, # bbox aliases
    "metrics/recall(B)": 0.61,
    "metrics/mAP50(B)": 0.721,
    "metrics/mAP50-95(B)": 0.489,
    "metrics/mAP_small": 0.291,
    "metrics/mAP_medium": 0.532,
    "metrics/mAP_large": 0.648,
    "metrics/AR1": 0.362,        # Average Recall (max 1 det)
    "metrics/AR10": 0.571,
    "metrics/AR100": 0.601,
    "metrics/AR_small": 0.387,
    "metrics/AR_medium": 0.641,
    "metrics/AR_large": 0.739,
    "speed/preprocess_ms": 1.2,
    "speed/inference_ms": 6.8,
    "speed/postprocess_ms": 0.9,
    "speed/total_ms": 8.9,
    "speed/total_s": 12.3,
    "speed/images_seen": 1382,
}`}</CodeBlock>
          <P>
            分割验证返回带 <InlineCode>(M)</InlineCode> 后缀的掩码指标，以及带 <InlineCode>(B)</InlineCode> 后缀的
            框指标；OBB 验证会添加{' '}
            <InlineCode>(OBB)</InlineCode> 指标。姿态验证通过{' '}
            <InlineCode>PoseValidator</InlineCode> 返回 COCO 关键点指标。v1.3.0 新增了用于分类（top-1 / top-5）、
            语义（mIoU / 像素精度）、点定位和深度（零样本）的验证器。传入{' '}
            <InlineCode>plots=True</InlineCode>（或命令行的 <InlineCode>--save-plots</InlineCode>）可将
            指标图、各类 AP、混淆矩阵和样本图写入运行目录。
          </P>

          <Divider />

          {/* ────────────── EXPORT ────────────── */}
          <SectionHeading id="export" icon={Upload}>导出</SectionHeading>
          <P>
            将 PyTorch 模型导出为 ONNX、TorchScript、TensorRT、OpenVINO、NCNN、CoreML 或（v1.3.0 新增）TFLite 以进行部署。经过充分测试的导出路径是单 GPU 的 YOLO9 检测、RF-DETR 检测和 RF-DETR 分割。部分较新的系列（包括 DINOv2、Depth Anything V2、SAM、VLM 和 L2CS）尚不支持导出。
          </P>

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
model.export(format="ncnn")

# CoreML (.mlpackage, macOS runtime)
model.export(format="coreml")

# TFLite (RF-DETR detect/seg/pose + YOLO9 detect; experimental, needs Python 3.12+)
model.export(format="tflite")`}</CodeBlock>

          <SubHeading>全部导出参数</SubHeading>
          <CodeBlock language="python">{`path = model.export(
    format="onnx",            # "onnx", "torchscript", "tensorrt", "openvino", "ncnn", "coreml", or "tflite"
    output_path="model.onnx", # output file (auto-generated if None)
    imgsz=640,                # input resolution (default: model's native); also accepts (h, w) for rectangular
    opset=None,               # ONNX opset (auto: 13, or 17 for wrappers that need it)
    simplify=True,            # run onnxsim graph simplification
    dynamic=True,             # enable dynamic batch axis (ONNX); TFLite requires static shapes
    half=False,               # export in FP16
    batch=1,                  # batch size for static graph
    device=None,              # device to trace on (default: model's current device)
    int8=False,               # INT8 quantization: TensorRT, OpenVINO, or ONNX (YOLO9 detection only)
    data=None,                # calibration dataset for INT8
    fraction=1.0,             # fraction of calibration data to use
    allow_download_scripts=False, # allow data.yaml download hooks during calibration
    workspace=4.0,            # TensorRT workspace size (GB)
    min_batch=1,              # TensorRT dynamic profile minimum batch
    opt_batch=1,              # TensorRT dynamic profile optimal batch
    max_batch=8,              # TensorRT dynamic profile maximum batch
    hardware_compatibility="none", # TensorRT compatibility mode
    gpu_device=0,             # GPU device index for TensorRT
    trt_config=None,          # optional TensorRT YAML config path
    compute_units="all",      # CoreML routing: all, cpu_only, cpu_and_gpu, cpu_and_ne
    nms=False,                # embed NMS in the graph (ONNX YOLO9 detection, or CoreML)
    iou=0.45,                 # embedded-NMS IoU threshold
    conf=0.25,                # embedded-NMS confidence threshold
    max_det=300,              # embedded-NMS max detections (ONNX only)
    verbose=False,            # verbose logging
)`}</CodeBlock>
          <P>
            OpenVINO INT8 导出额外需要 <InlineCode>nncf</InlineCode>。NCNN 导出会写出一个目录，包含 <InlineCode>model.ncnn.param</InlineCode>、<InlineCode>model.ncnn.bin</InlineCode> 和 <InlineCode>metadata.yaml</InlineCode>。CoreML 导出会写出 <InlineCode>.mlpackage</InlineCode> 包，需要 <InlineCode>coremltools</InlineCode>，且不支持 INT8。
          </P>

          <SubHeading>ONNX 内嵌 NMS（YOLO9 检测）</SubHeading>
          <P>
            v1.3.0 新增：传入 <InlineCode>nms=True</InlineCode> 可将 NMS 烘焙进导出的 ONNX
            图，使模型直接输出最终框。目前仅限 <InlineCode>detect</InlineCode> 任务上的{' '}
            <InlineCode>yolo9</InlineCode> 系列（其他系列/任务会报错）。它会强制固定的 batch-1 图
            （<InlineCode>dynamic=False</InlineCode>），并在 ONNX 元数据中记录{' '}
            <InlineCode>nms</InlineCode> / <InlineCode>nms_conf</InlineCode> /{' '}
            <InlineCode>nms_iou</InlineCode> / <InlineCode>max_det</InlineCode>。
          </P>
          <CodeBlock language="python">{`model = LibreYOLO("LibreYOLO9c.pt")
model.export(format="onnx", nms=True, conf=0.25, iou=0.45, max_det=300)`}</CodeBlock>
          <P>
            <InlineCode>int8=True</InlineCode> 现在也支持 ONNX（除 TensorRT 和 OpenVINO 之外），
            同样仅限 YOLO9 检测；它需要一个用于标定的{' '}
            <InlineCode>data=</InlineCode> 数据集。
          </P>

          <SubHeading>TFLite 导出</SubHeading>
          <P>
            <SupportBadge variant="experimental">实验性</SupportBadge>{' '}
            v1.3.0 新增了基于 <InlineCode>onnx2tf</InlineCode> 的 TFLite 导出路径。它已在
            RF-DETR detect / segment / pose 和 YOLO9 detect 上验证。它需要{' '}
            <strong className="text-surface-800 dark:text-white">Python 3.12+</strong>（{' '}
            <InlineCode>onnx2tf 2.4.x</InlineCode> wheels 不面向更老的 Python），以及
            可选 extra <InlineCode>libreyolo[tflite]</InlineCode>
            （<InlineCode>onnx2tf&gt;=2.4.3</InlineCode>、onnx-graphsurgeon、onnx-simplifier）。导出
            为 FP32 且仅静态形状（暂不支持 <InlineCode>half</InlineCode>、{' '}
            <InlineCode>int8</InlineCode> 或 <InlineCode>dynamic</InlineCode>）。
          </P>
          <CodeBlock language="bash">{`pip install "libreyolo[tflite]"   # Python 3.12+`}</CodeBlock>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreRFDETRs-seg.pt")
model.export(format="tflite")   # writes a .tflite file`}</CodeBlock>
          <P>
            对于 RF-DETR，导出器会将每个 GridSample 节点改写为 TFLite 安全的双线性
            子图，因为 onnx2tf 的默认 lowering 在数值上有问题。在 v1.3.0 中，由于{' '}
            <InlineCode>onnx2tf&gt;=2.4.3</InlineCode> 已在上游提供 RF-DETR 修复，旧的针对 onnx2tf
            的运行时 monkeypatch 已被移除；只保留静态 ONNX 图改写。
          </P>
          <P>
            <strong className="text-surface-800 dark:text-white">没有 TFLite 运行时后端。</strong>{' '}
            LibreYOLO 无法加载或运行 <InlineCode>.tflite</InlineCode> 文件；该格式仅用于
            导出。请在目标设备上使用 TF Lite 运行时
            （<InlineCode>ai-edge-litert</InlineCode> / <InlineCode>tflite-runtime</InlineCode>）
            运行导出的模型。
          </P>

          <SubHeading>ONNX 元数据</SubHeading>
          <P>导出的 ONNX 文件包含内嵌元数据：</P>
          <DocTable
            headers={['键', '示例值']}
            rows={[
              [<InlineCode key="v">libreyolo_version</InlineCode>, <InlineCode key="vv">&quot;1.3.0&quot;</InlineCode>],
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
            使用 <InlineCode>LibreYOLO(&quot;model.onnx&quot;)</InlineCode> 加载导出文件时，会自动读回该元数据。
          </P>

          <Divider />

          {/* ────────────── TORCHSCRIPT INFERENCE ────────────── */}
          <SectionHeading id="torchscript-inference" icon={Cpu}>TorchScript 推理</SectionHeading>
          <P>
            通过相同的运行时后端预测 API 运行导出的 <InlineCode>.torchscript</InlineCode> 模型。
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("model.torchscript")

result = model("image.jpg", conf=0.25, iou=0.45, save=True)
print(result.boxes.xyxy)`}</CodeBlock>

          <Divider />

          {/* ────────────── ONNX INFERENCE ────────────── */}
          <SectionHeading id="onnx-inference" icon={Cpu}>ONNX 推理</SectionHeading>
          <P>
            使用 ONNX Runtime 而非 PyTorch 运行推理。适用于没有 PyTorch 的部署环境。
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("model.onnx")

result = model("image.jpg", conf=0.25, iou=0.45, save=True)
print(result.boxes.xyxy)`}</CodeBlock>

          <SubHeading>自动元数据</SubHeading>
          <P>
            如果 ONNX 文件由 LibreYOLO 导出，类别名称和类别数会自动从内嵌元数据中读取：
          </P>
          <CodeBlock language="python">{`# Export with metadata
model.export(format="onnx", output_path="model.onnx")

# Load - names and nb_classes auto-populated
onnx_model = LibreYOLO("model.onnx")
print(onnx_model.names)       # {0: "person", 1: "bicycle", ...}
print(onnx_model.nb_classes)  # 80`}</CodeBlock>

          <P>
            对于没有元数据的 ONNX 文件（例如由其他工具导出），请手动指定 <InlineCode>nb_classes</InlineCode>：
          </P>
          <CodeBlock language="python">{`model = LibreYOLO("external_model.onnx", nb_classes=20)`}</CodeBlock>

          <SubHeading>设备选择</SubHeading>
          <CodeBlock language="python">{`# Auto-detect (CUDA if available, else CPU)
model = LibreYOLO("model.onnx", device="auto")

# Force CPU
model = LibreYOLO("model.onnx", device="cpu")

# Force CUDA
model = LibreYOLO("model.onnx", device="cuda")`}</CodeBlock>

          <SubHeading>预测参数</SubHeading>
          <P>
            通过 <InlineCode>LibreYOLO()</InlineCode> 加载的运行时产物支持共享的运行时预测 API：
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
            运行时后端不暴露仅 PyTorch 的选项，如 <InlineCode>tiling</InlineCode>、<InlineCode>overlap_ratio</InlineCode> 或 <InlineCode>output_file_format</InlineCode>。
          </P>
          <P>
            运行时后端在保存方面与 PyTorch 包装器略有不同：如果你设置 <InlineCode>output_path</InlineCode>，请传入最终文件路径，而非目录。如果省略它，当前后端默认保存在 <InlineCode>runs/detections/</InlineCode> 下。
          </P>

          <Divider />

          {/* ────────────── TENSORRT INFERENCE ────────────── */}
          <SectionHeading id="tensorrt-inference" icon={Cpu}>TensorRT 推理</SectionHeading>
          <P>
            使用 TensorRT 在 NVIDIA GPU 上获得最大吞吐。需要 CUDA 以及 TensorRT 的 Python 绑定。
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("model.engine")

result = model("image.jpg", conf=0.25, iou=0.45, save=True)
print(result.boxes.xyxy)`}</CodeBlock>

          <P>
            通过 <InlineCode>LibreYOLO()</InlineCode> 加载的 TensorRT 产物支持与 ONNX 和 OpenVINO 相同的核心运行时预测 API，包括 <InlineCode>save=True</InlineCode> 时同样仅接受文件路径的 <InlineCode>output_path</InlineCode> 行为。
          </P>

          <Divider />

          {/* ────────────── OPENVINO INFERENCE ────────────── */}
          <SectionHeading id="openvino-inference" icon={Cpu}>OpenVINO 推理</SectionHeading>
          <P>
            使用 OpenVINO 运行推理，针对 Intel CPU、GPU 和 VPU 优化。
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("model_openvino/")

result = model("image.jpg", conf=0.25, iou=0.45, save=True)
print(result.boxes.xyxy)`}</CodeBlock>

          <P>
            通过 <InlineCode>LibreYOLO()</InlineCode> 加载的 OpenVINO 目录会在存在时读取 <InlineCode>metadata.yaml</InlineCode>，并支持相同的核心运行时预测 API。
          </P>

          <Divider />

          {/* ────────────── NCNN INFERENCE ────────────── */}
          <SectionHeading id="ncnn-inference" icon={Cpu}>NCNN 推理</SectionHeading>
          <P>
            使用 NCNN 运行推理，用于在 CPU 或支持 Vulkan 的 GPU 目标上进行轻量部署。
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("model_ncnn/")

result = model("image.jpg", conf=0.25, iou=0.45, save=True)
print(result.boxes.xyxy)`}</CodeBlock>

          <P>
            NCNN 导出目录包含 <InlineCode>model.ncnn.param</InlineCode>、<InlineCode>model.ncnn.bin</InlineCode>，通常还有 <InlineCode>metadata.yaml</InlineCode>。
          </P>

          <Divider />

          {/* ────────────── COREML INFERENCE ────────────── */}
          <SectionHeading id="coreml-inference" icon={Cpu}>CoreML 推理</SectionHeading>
          <P>
            在 macOS 上通过 CoreML 运行导出的 <InlineCode>.mlpackage</InlineCode>。CoreML 使用 <InlineCode>compute_units</InlineCode> 而非 PyTorch 设备字符串来路由执行。
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("model.mlpackage", compute_units="all")

result = model("image.jpg", conf=0.25, iou=0.45, save=True)
print(result.boxes.xyxy)`}</CodeBlock>

          <P>
            支持的 <InlineCode>compute_units</InlineCode> 取值为 <InlineCode>all</InlineCode>、<InlineCode>cpu_only</InlineCode>、<InlineCode>cpu_and_gpu</InlineCode> 和 <InlineCode>cpu_and_ne</InlineCode>。
          </P>

          <Divider />

          {/* ────────────── CLI ────────────── */}
          <SectionHeading id="cli" icon={SquareTerminal}>命令行（CLI）</SectionHeading>
          <P>
            安装 LibreYOLO 会在你的 PATH 中注册一个 <InlineCode>libreyolo</InlineCode> 命令（入口点在 <InlineCode>pyproject.toml</InlineCode> 中）。命令行镜像 Python API，并遵循 Ultralytics 风格的 <InlineCode>key=value</InlineCode> 语法。
          </P>

          <SubHeading>子命令</SubHeading>
          <DocTable
            headers={['命令', '用途']}
            rows={[
              [<InlineCode key="p">predict</InlineCode>, '对图像、目录或视频运行推理'],
              [<InlineCode key="t">train</InlineCode>, '在数据集上训练模型'],
              [<InlineCode key="v">val</InlineCode>, '在数据集上评估模型'],
              [<InlineCode key="e">export</InlineCode>, '导出为 ONNX / TorchScript / TensorRT / OpenVINO / NCNN / CoreML / TFLite'],
              [<InlineCode key="ui">ui</InlineCode>, '启动本地拖放 / 粘贴的浏览器推理界面'],
              [<InlineCode key="dr">doctor</InlineCode>, '运行训练前的数据集健康检查（YOLO 检测格式）'],
              [<InlineCode key="c">checks</InlineCode>, '打印 Python、torch、CUDA、GPU 和可选包信息'],
              [<InlineCode key="m">models</InlineCode>, '列出已注册的模型系列和命令行简称'],
              [<InlineCode key="f">formats</InlineCode>, '列出支持的导出格式'],
              [<InlineCode key="cfg">cfg</InlineCode>, '打印默认的训练配置 YAML'],
              [<InlineCode key="i">info</InlineCode>, '加载模型并打印解析出的系列、尺寸、任务、设备和类别'],
              [<InlineCode key="md">metadata</InlineCode>, '检查 .pt 文件中的原始检查点元数据'],
              [<InlineCode key="ver">version</InlineCode>, '打印 LibreYOLO + Python + torch 版本'],
            ]}
          />

          <SubHeading>模型名称简称</SubHeading>
          <P>
            命令行接受简称（<InlineCode>yolo9-c</InlineCode>），它们会解析为权重文件名（<InlineCode>LibreYOLO9c.pt</InlineCode>）- 可通过 <InlineCode>libreyolo models</InlineCode> 查看。你也可以传入任意明确的检查点路径。
          </P>

          <SubHeading>常用选项</SubHeading>
          <DocTable
            headers={['命令', '重要选项']}
            rows={[
              [<InlineCode key="p">predict</InlineCode>, <span key="pv"><InlineCode>conf</InlineCode>, <InlineCode>iou</InlineCode>, <InlineCode>imgsz</InlineCode>, <InlineCode>classes</InlineCode>, <InlineCode>max_det</InlineCode>, <InlineCode>half</InlineCode>, <InlineCode>batch</InlineCode>, <InlineCode>tiling</InlineCode>, <InlineCode>overlap_ratio</InlineCode>, <InlineCode>output_file_format</InlineCode>, <InlineCode>project</InlineCode>, <InlineCode>name</InlineCode>, <InlineCode>exist_ok</InlineCode>, <InlineCode>face_detector</InlineCode></span>],
              [<InlineCode key="t">train</InlineCode>, <span key="tv"><InlineCode>epochs</InlineCode>, <InlineCode>batch</InlineCode>, <InlineCode>imgsz</InlineCode>, <InlineCode>lr0</InlineCode>, <InlineCode>optimizer</InlineCode>, <InlineCode>scheduler</InlineCode>, <InlineCode>workers</InlineCode>, <InlineCode>seed</InlineCode>, <InlineCode>resume</InlineCode>, <InlineCode>amp</InlineCode>, <InlineCode>task</InlineCode>, <InlineCode>cache</InlineCode>, <InlineCode>lora</InlineCode>, <InlineCode>freeze</InlineCode>, <InlineCode>save_plots</InlineCode>, <InlineCode>allow_download_scripts</InlineCode>, <InlineCode>dry_run</InlineCode></span>],
              [<InlineCode key="v">val</InlineCode>, <span key="vv"><InlineCode>split</InlineCode>, <InlineCode>batch</InlineCode>, <InlineCode>imgsz</InlineCode>, <InlineCode>conf</InlineCode>, <InlineCode>iou</InlineCode>, <InlineCode>max_det</InlineCode>, <InlineCode>half</InlineCode>, <InlineCode>save_plots</InlineCode>, <InlineCode>data_dir</InlineCode>, <InlineCode>use_coco_eval</InlineCode>, <InlineCode>project</InlineCode>, <InlineCode>name</InlineCode>, <InlineCode>exist_ok</InlineCode>, <InlineCode>save_json</InlineCode>, <InlineCode>allow_download_scripts</InlineCode></span>],
              [<InlineCode key="e">export</InlineCode>, <span key="ev"><InlineCode>format</InlineCode>, <InlineCode>imgsz</InlineCode>, <InlineCode>batch</InlineCode>, <InlineCode>half</InlineCode>, <InlineCode>int8</InlineCode>, <InlineCode>dynamic</InlineCode>, <InlineCode>simplify</InlineCode>, <InlineCode>nms</InlineCode>, <InlineCode>conf</InlineCode>, <InlineCode>iou</InlineCode>, <InlineCode>max_det</InlineCode>, <InlineCode>opset</InlineCode>, <InlineCode>data</InlineCode>, <InlineCode>fraction</InlineCode>, <InlineCode>device</InlineCode>, <InlineCode>allow_download_scripts</InlineCode>, <InlineCode>verbose</InlineCode></span>],
            ]}
          />

          <SubHeading>预测</SubHeading>
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

          <SubHeading>训练</SubHeading>
          <CodeBlock language="bash">{`libreyolo train model=yolo9-c data=coco128.yaml epochs=300 batch=16 device=0

# Dry-run prints the resolved config without launching training
libreyolo train model=yolo9-c data=coco128.yaml --dry-run`}</CodeBlock>

          <SubHeading>验证</SubHeading>
          <CodeBlock language="bash">{`libreyolo val model=runs/train/exp/weights/best.pt data=coco128.yaml split=val`}</CodeBlock>

          <SubHeading>导出</SubHeading>
          <CodeBlock language="bash">{`libreyolo export model=runs/train/exp/weights/best.pt format=onnx dynamic=true
libreyolo export model=best.pt format=tensorrt half=true
libreyolo export model=best.pt format=openvino int8=true data=coco128.yaml
libreyolo export model=best.pt format=coreml`}</CodeBlock>

          <SubHeading>导出内嵌 NMS 与矩形尺寸</SubHeading>
          <CodeBlock language="bash">{`# Embed NMS into an ONNX YOLO9 detection graph
libreyolo export model=yolo9-c format=onnx nms=true conf=0.25 iou=0.45 max_det=300

# Rectangular export size (imgsz accepts a single value or two comma-separated dims)
libreyolo export model=yolo9-c format=onnx imgsz=640,480

# TFLite (Python 3.12+, libreyolo[tflite])
libreyolo export model=rfdetr-s format=tflite`}</CodeBlock>

          <SubHeading>本地推理界面</SubHeading>
          <P>
            <InlineCode>libreyolo ui</InlineCode> 提供一个本地浏览器页面，你可以在其中拖放、粘贴
            或选择图像、选择模型并查看结果。它默认绑定{' '}
            <InlineCode>127.0.0.1:8000</InlineCode>，若端口被占用会自动递增。
          </P>
          <CodeBlock language="bash">{`libreyolo ui                       # opens http://127.0.0.1:8000
libreyolo ui --port 9000 --no-browser --device 0`}</CodeBlock>

          <SubHeading>数据集健康检查</SubHeading>
          <P>
            <InlineCode>libreyolo doctor</InlineCode> 对 YOLO 检测格式的数据集运行训练前检查，
            发现错误时以非零退出
            （<InlineCode>--strict</InlineCode> 还会在出现警告时失败），因此可用于门控 CI。
          </P>
          <CodeBlock language="bash">{`libreyolo doctor coco8.yaml
libreyolo doctor --data coco8.yaml --strict --json
libreyolo doctor coco8.yaml --fast --only labels   # skip image decoding, run one check family`}</CodeBlock>

          <SubHeading>机器可读输出</SubHeading>
          <P>
            每个命令都接受 <InlineCode>--json</InlineCode>（结构化的标准输出，便于管道传入脚本或智能体）和 <InlineCode>--quiet</InlineCode>（抑制 stderr 进度行）。核心的 <InlineCode>predict</InlineCode>、<InlineCode>train</InlineCode>、<InlineCode>val</InlineCode> 和 <InlineCode>export</InlineCode> 命令还接受 <InlineCode>--help-json</InlineCode>，可将其参数 schema 以 JSON 形式输出。
          </P>
          <CodeBlock language="bash">{`libreyolo predict model=yolo9-c source=img.jpg --json | jq .

libreyolo train --help-json > train_schema.json`}</CodeBlock>

          <Divider />

          {/* ────────────── API REFERENCE ────────────── */}
          <SectionHeading id="api-reference" icon={FileCode}>API 参考</SectionHeading>

          <SubHeading>LibreYOLO（工厂）</SubHeading>
          <CodeBlock language="python">{`LibreYOLO(
    model_path: str,
    *,
    device: str = "auto",
    task: str | None = None,    # override only when a custom artifact is ambiguous
    nb_classes: int | None = None,  # mainly for external exported artifacts
    compute_units: str = "all", # CoreML only: all, cpu_only, cpu_and_gpu, cpu_and_ne
) -> model wrapper or runtime backend`}</CodeBlock>
          <P>
            优先使用官方检查点文件名和导出产物路径，然后让工厂解析细节。它可处理 PyTorch 检查点、<InlineCode>.onnx</InlineCode>、<InlineCode>.torchscript</InlineCode>、<InlineCode>.engine</InlineCode>、<InlineCode>.tensorrt</InlineCode>、<InlineCode>.mlpackage</InlineCode>、包含 <InlineCode>model.xml</InlineCode> 的 OpenVINO 目录，以及包含 <InlineCode>model.ncnn.param</InlineCode> 和 <InlineCode>model.ncnn.bin</InlineCode> 的 NCNN 目录。<InlineCode>task</InlineCode> 参数用于有歧义的自定义产物；否则解析来自检查点元数据、文件名后缀和系列默认值。
          </P>

          <SubHeading>预测（PyTorch 模型包装器）</SubHeading>
          <CodeBlock language="python">{`model(
    source,                     # image input (see supported formats)
    *,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = None,
    device: str = "auto",
    classes: list[int] = None,
    max_det: int = 300,
    augment: bool = False,
    save: bool = False,
    batch: int = 1,
    stream: bool = False,
    vid_stride: int = 1,
    show: bool = False,
    output_path: str = None,
    color_format: str = "auto",
    tiling: bool = False,
    overlap_ratio: float = 0.2,
    output_file_format: str = None,
) -> Results | list[Results] | Generator[Results, None, None]`}</CodeBlock>

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
            如果运行时后端省略了 <InlineCode>output_path</InlineCode>，当前默认保存位置是 <InlineCode>runs/detections/</InlineCode>。
          </P>

          <SubHeading>Results</SubHeading>
          <CodeBlock language="python">{`result = Results(
    boxes: Boxes | None,
    orig_shape: tuple[int, int],  # (height, width)
    path: str | None,
    names: dict[int, str],
    masks: Masks | None = None,
    keypoints: Keypoints | None = None,
    probs: Probs | None = None,
    obb: OBB | None = None,
    gaze: Gaze | None = None,
    speed: dict[str, float] | None = None,
    track_id = None,
    frame_idx: int | None = None,
)

len(result)          # number of detections
result.cpu()         # copy with tensors on CPU
result.cuda()        # copy with tensors on CUDA
result.numpy()       # copy with numpy arrays
result.summary()     # list[dict] with boxes, masks, gaze, and track_id when present
result.to_json()     # JSON string from summary()`}</CodeBlock>

          <SubHeading>Boxes</SubHeading>
          <CodeBlock language="python">{`boxes = Boxes(boxes, conf, cls)

boxes.xyxy           # (N, 4) tensor - x1, y1, x2, y2
boxes.xywh           # (N, 4) tensor - cx, cy, w, h
boxes.conf           # (N,) tensor - confidence scores
boxes.cls            # (N,) tensor - class IDs
boxes.id             # (N,) track IDs when tracking, else None
boxes.is_track       # True when track IDs are attached
boxes.data           # (N, 6) [xyxy, conf, cls], or (N, 7) with track IDs

len(boxes)           # number of boxes
boxes.cpu()          # copy on CPU
boxes.numpy()        # copy as numpy arrays`}</CodeBlock>

          <SubHeading>任务载荷</SubHeading>
          <CodeBlock language="python">{`result.masks.data        # segmentation masks, (N, H, W)
result.masks.xy          # list of mask contours in pixel coordinates
result.masks.xyn         # normalized mask contours

result.keypoints.xy      # pose keypoint coordinates
result.keypoints.xyn     # normalized keypoint coordinates
result.keypoints.conf    # keypoint confidence when present

result.gaze.data         # (N, 2): pitch, yaw in radians
result.gaze.pitch_deg    # pitch in degrees
result.gaze.yaw_deg      # yaw in degrees
result.gaze.direction_3d # approximate 3D direction vectors`}</CodeBlock>

          <SubHeading>model.export()</SubHeading>
          <CodeBlock language="python">{`model.export(
    format: str = "onnx",       # "onnx", "torchscript", "tensorrt", "openvino", "ncnn", or "coreml"
    *,
    output_path: str | None = None,
    imgsz: int | None = None,
    opset: int | None = None,   # auto: 13, or 17 for wrappers that need it
    simplify: bool = True,
    dynamic: bool = True,
    half: bool = False,
    batch: int = 1,
    device: str | None = None,
    int8: bool = False,
    data: str | None = None,    # calibration data for INT8
    fraction: float = 1.0,      # fraction of calibration data
    allow_download_scripts: bool = False,
    workspace: float = 4.0,     # TensorRT workspace (GB)
    min_batch: int = 1,         # TensorRT dynamic profile minimum batch
    opt_batch: int = 1,         # TensorRT dynamic profile optimal batch
    max_batch: int = 8,         # TensorRT dynamic profile maximum batch
    hardware_compatibility: str = "none",
    gpu_device: int = 0,
    trt_config = None,          # optional TensorRT YAML config path
    compute_units: str = "all", # CoreML only
    nms: bool = False,          # CoreML embedded NMS where supported
    iou: float = 0.45,          # CoreML embedded NMS IoU threshold
    conf: float = 0.25,         # CoreML embedded NMS confidence threshold
    verbose: bool = False,
) -> str                        # path to exported file or directory`}</CodeBlock>

          <SubHeading>model.val()</SubHeading>
          <CodeBlock language="python">{`model.val(
    data: str = None,           # path to data.yaml
    batch: int = 16,
    imgsz: int = None,
    conf: float = 0.001,
    iou: float = 0.6,
    workers: int = 4,
    allow_download_scripts: bool = False,
    device: str = None,
    split: str = "val",         # "val", "test", or "train"
    augment: bool = False,
    save_json: bool = False,
    verbose: bool = True,
) -> dict`}</CodeBlock>
          <P>返回（COCO 评估，默认）：</P>
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
    allow_download_scripts: bool = False,
    callbacks = None,
) -> dict`}</CodeBlock>
          <P>返回标准的 LibreYOLO 训练字典，包含 <InlineCode>final_loss</InlineCode>、<InlineCode>best_mAP50</InlineCode>、<InlineCode>best_mAP50_95</InlineCode>、<InlineCode>best_epoch</InlineCode>、<InlineCode>save_dir</InlineCode>、<InlineCode>best_checkpoint</InlineCode> 和 <InlineCode>last_checkpoint</InlineCode>。</P>

          <SubHeading>model.train()（RF-DETR）</SubHeading>
          <CodeBlock language="python">{`model.train(
    data: str,                  # path to data.yaml
    epochs: int = 100,
    batch_size: int = 4,
    lr: float = 1e-4,
    output_dir: str = "runs/train",
    resume: str = None,
    **kwargs,                   # additional RF-DETR training args
) -> dict`}</CodeBlock>
          <P>
            YOLO-NAS、D-FINE、DEIM、DEIMv2、EC、PicoDet、RT-DETRv2/v4 和 RTMDet 还有额外的实验性训练器，外加新增的分类（MobileNetV4、ConvNeXt、EfficientNetV2、DINOv2）、语义分割（DINOv2）和点定位（FOMO）系列。它们遵循相同的 <InlineCode>model.train(data=&quot;...yaml&quot;, ...)</InlineCode> 形式，但默认值和实验性门控因系列而异。
          </P>

          <SubHeading>运行时产物加载</SubHeading>
          <P>
            通过 <InlineCode>LibreYOLO()</InlineCode> 加载导出产物，方式与加载 PyTorch 检查点相同。工厂会根据路径选择 ONNX Runtime、TorchScript、TensorRT、OpenVINO、NCNN 或 CoreML：
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("model.onnx")
model = LibreYOLO("model.torchscript")
model = LibreYOLO("model.engine")
model = LibreYOLO("model_openvino/")
model = LibreYOLO("model_ncnn/")
model = LibreYOLO("model.mlpackage", compute_units="all")`}</CodeBlock>
          <P>
            高级集成可以触及更底层的运行时模块，但普通应用代码应保持使用工厂路径。
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
          <SectionHeading id="architecture" icon={Wrench}>架构指南</SectionHeading>
          <P>
            本节面向希望了解代码库内部机制的贡献者。
          </P>

          <SubHeading>基类设计</SubHeading>
          <P>
            PyTorch 模型系列继承自 <InlineCode>libreyolo/models/base/model.py</InlineCode> 中的 <InlineCode>BaseModel</InlineCode>。子类实现以下抽象方法：
          </P>
          <DocTable
            headers={['方法', '用途']}
            rows={[
              [<InlineCode key="init">_init_model()</InlineCode>, '构建并返回 nn.Module'],
              [<InlineCode key="layers">_get_available_layers()</InlineCode>, '返回层名到模块的映射'],
              [<InlineCode key="pre-np">_get_preprocess_numpy()</InlineCode>, '返回用于导出 / 标定的 NumPy 预处理器'],
              [<InlineCode key="pre">_preprocess()</InlineCode>, '图像到张量的转换'],
              [<InlineCode key="fwd">_forward()</InlineCode>, '模型前向传播'],
              [<InlineCode key="post">_postprocess()</InlineCode>, '原始输出到检测字典'],
            ]}
          />
          <P>
            <InlineCode>BaseModel</InlineCode> 提供共享的包装行为：预测、导出、验证、尺寸/名称元数据和训练辅助。实际的单图、批量和分块推理流程位于 <InlineCode>libreyolo/models/base/inference.py</InlineCode>，而部署运行时位于 <InlineCode>libreyolo/backends/</InlineCode> 下。
          </P>

          <SubHeading>包结构</SubHeading>
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
        yolo9/           # LibreYOLO9 (detect)
        yolo9_e2e/       # LibreYOLO9E2E (detect)
        yolonas/         # LibreYOLONAS (detect, pose)
        dfine/           # LibreDFINE (detect)
        deim/            # LibreDEIM (detect)
        deimv2/          # LibreDEIMv2 (detect)
        rtdetr/          # LibreRTDETR (detect)
        rtdetrv2/        # LibreRTDETRv2 (detect)
        rtdetrv4/        # LibreRTDETRv4 (detect)
        rfdetr/          # LibreRFDETR (detect, segment, pose, obb) - lazy-loaded
        ec/              # LibreEC / EdgeCrafter (detect, pose, segment)
        picodet/         # LibrePICODET (detect)
        rtmdet/          # LibreRTMDet (detect)
        dinov2/          # LibreDINOv2 (semantic, classify) - lazy-loaded
        mobilenetv4/     # LibreMobileNetV4 (classify)
        convnext/        # LibreConvNeXt (classify)
        efficientnetv2/  # LibreEfficientNetV2 (classify)
        depth_anything/  # LibreDepthAnythingV2 (depth)
        fomo/            # LibreFOMO (point)
        l2cs/            # LibreL2CS (gaze, inference-only)
    backends/
        base.py
        onnx.py          # ONNX Runtime loader
        torchscript.py   # TorchScript loader
        tensorrt.py      # TensorRT loader
        openvino.py      # OpenVINO loader
        ncnn.py          # NCNN loader
        coreml.py        # CoreML loader
    export/
        exporter.py      # BaseExporter and format registry
        onnx.py / torchscript.py / tensorrt.py / openvino.py / ncnn.py / coreml.py
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

          <SubHeading>添加新的模型系列</SubHeading>
          <ol className="space-y-2.5 mb-4 list-none">
            {[
              <>创建 <InlineCode>libreyolo/models/newmodel/model.py</InlineCode>，其中包含一个继承 <InlineCode>BaseModel</InlineCode> 的类</>,
              <>按需设置 <InlineCode>FAMILY</InlineCode>、<InlineCode>FILENAME_PREFIX</InlineCode>、<InlineCode>INPUT_SIZES</InlineCode>、<InlineCode>SUPPORTED_TASKS</InlineCode> 和 <InlineCode>DEFAULT_TASK</InlineCode></>,
              <>实现注册表钩子，如 <InlineCode>can_load()</InlineCode>、<InlineCode>detect_size()</InlineCode>、<InlineCode>detect_nb_classes()</InlineCode> 和 <InlineCode>detect_size_from_filename()</InlineCode></>,
              '实现该系列所需的模型初始化、预处理、前向、后处理、训练和验证钩子',
              <>在 <InlineCode>libreyolo/models/newmodel/</InlineCode> 下创建配套的网络和工具</>,
              <>将导入添加到 <InlineCode>libreyolo/models/__init__.py</InlineCode>；子类注册会在导入运行时发生</>,
              <>从 <InlineCode>libreyolo/__init__.py</InlineCode> 导出该类</>,
              <>（可选）如果验证预处理与标准路径不同，覆盖 <InlineCode>val_preprocessor_class</InlineCode></>,
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
            用户代码应通过 <InlineCode>model.export(...)</InlineCode> 导出。在内部，<InlineCode>libreyolo/export/exporter.py</InlineCode> 中的 <InlineCode>BaseExporter</InlineCode> 拥有格式注册表，具体的导出器通过子类注册来注册自身。
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")
model.export(format="onnx")`}</CodeBlock>
          <P>
            要添加新的导出格式，请实现一个具有唯一 <InlineCode>format_name</InlineCode> 的新 <InlineCode>BaseExporter</InlineCode> 子类，并从 <InlineCode>libreyolo/export/exporter.py</InlineCode> 导入它，以填充注册表。
          </P>

          <Divider />

          {/* ────────────── DATASET FORMAT ────────────── */}
          <SectionHeading id="dataset-format" icon={Database}>数据集格式</SectionHeading>
          <P>
            训练和验证使用通过 <InlineCode>data.yaml</InlineCode> 加载的数据集配置。检测、分割、姿态和 RF-DETR 训练都通过该加载器进入；标签文件内容因任务而异。
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

          <SubHeading>配置解析与下载</SubHeading>
          <P>
            数据集配置按以下顺序解析：明确路径、当前工作目录，然后是 <InlineCode>libreyolo/config/datasets/</InlineCode> 下的内置配置。数据集根目录默认在 <InlineCode>~/datasets</InlineCode> 下，可用 <InlineCode>LIBREYOLO_DATASETS_DIR</InlineCode> 覆盖。
          </P>
          <P>
            <InlineCode>train</InlineCode>、<InlineCode>val</InlineCode> 和 <InlineCode>test</InlineCode> 可以是目录、<InlineCode>.txt</InlineCode> 文件或路径列表。YAML 下载钩子受保护；仅对可信配置传入 <InlineCode>allow_download_scripts=True</InlineCode>。
          </P>

          <SubHeading>文件列表变体</SubHeading>
          <P>
            同样的 YAML 格式也可以让 <InlineCode>train</InlineCode>、<InlineCode>val</InlineCode> 或 <InlineCode>test</InlineCode> 指向每行一个图像路径的 <InlineCode>.txt</InlineCode> 文件：
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

          <SubHeading>检测标签格式</SubHeading>
          <P>
            每张图像一个文本文件。每行是一个目标：
          </P>
          <CodeBlock language="text">{`<class_id> <center_x> <center_y> <width> <height>`}</CodeBlock>
          <P>
            所有坐标都相对于图像尺寸归一化到 [0, 1]。
          </P>
          <P>示例（<InlineCode>img001.txt</InlineCode>）：</P>
          <CodeBlock language="text" filename="img001.txt">{`0 0.5 0.4 0.3 0.6
2 0.1 0.2 0.05 0.1`}</CodeBlock>

          <SubHeading>分割标签格式</SubHeading>
          <P>
            分割使用 YOLO 多边形行。数据集加载器从多边形顶点导出边界框，并在启用分割加载时保留多边形环：
          </P>
          <CodeBlock language="text">{`<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>`}</CodeBlock>

          <SubHeading>姿态标签格式</SubHeading>
          <P>
            姿态标签在框之后追加关键点。在 <InlineCode>data.yaml</InlineCode> 中添加 <InlineCode>kpt_shape</InlineCode> 和 <InlineCode>flip_idx</InlineCode>，以便加载器知道关键点数量和水平翻转排列。
          </P>
          <CodeBlock language="yaml">{`kpt_shape: [17, 3]
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]`}</CodeBlock>
          <CodeBlock language="text">{`<class_id> <cx> <cy> <w> <h> <kx1> <ky1> <v1> ... <kxK> <kyK> <vK>`}</CodeBlock>

          <SubHeading>内置数据集</SubHeading>
          <P>
            LibreYOLO 在 <InlineCode>libreyolo/config/datasets/</InlineCode> 下提供内置数据集配置，并可在首次使用时自动下载受支持的数据集：
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
  const locale = useLocale()
  if (locale === 'zh') return <DocsPageZh version="v1.3.0" />
  return <DocsPage version="v1.3.0" />
}
