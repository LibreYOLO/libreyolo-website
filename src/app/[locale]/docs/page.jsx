'use client'

import { useState, useEffect } from 'react'
import { useLocale, useTranslations } from 'next-intl'
import { motion, AnimatePresence } from 'framer-motion'
import {
  BookOpen, Terminal, Rocket, Layers, Crosshair, Grid3x3,
  GraduationCap, CheckCircle2, Upload, Cpu, FileCode, Wrench,
  Database, Copy, Check, Menu, X, ChevronRight,
  Sparkles, Tags, Video, Activity, Scissors, PersonStanding, Eye, SquareTerminal,
  ShieldCheck,
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
  { id: 'pose', title: 'Pose Estimation', icon: PersonStanding },
  { id: 'gaze', title: 'Gaze Estimation', icon: Eye },
  { id: 'training', title: 'Training', icon: GraduationCap },
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
  }

  return (
    <span className={`inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-semibold ${styles[variant]}`}>
      {children}
    </span>
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

  return <span className="sr-only">Not currently supported</span>
}

function CompatibilityMatrix() {
  const rows = [
    {
      family: 'YOLO9',
      status: 'Validated detect, single GPU',
      inference: 'yes',
      training: 'yes',
      detect: 'yes',
      segment: 'exp',
      pose: '',
      gaze: '',
      onnx: 'yes',
      torchscript: 'yes',
      tensorrt: 'yes',
      openvino: 'yes',
      ncnn: 'yes',
      coreml: 'yes',
    },
    {
      family: 'RF-DETR',
      status: 'Validated detect + segment, single GPU',
      inference: 'yes',
      training: 'yes',
      detect: 'yes',
      segment: 'yes',
      pose: '',
      gaze: '',
      onnx: 'yes',
      torchscript: 'exp',
      tensorrt: 'yes',
      openvino: 'yes',
      ncnn: '',
      coreml: 'exp',
    },
    {
      family: 'YOLOX',
      status: 'Experimental',
      inference: 'exp',
      training: 'exp',
      detect: 'exp',
      segment: '',
      pose: '',
      gaze: '',
      onnx: 'exp',
      torchscript: 'exp',
      tensorrt: 'exp',
      openvino: 'exp',
      ncnn: 'exp',
      coreml: 'exp',
    },
    {
      family: 'YOLO9-E2E',
      status: 'Experimental',
      inference: 'exp',
      training: 'exp',
      detect: 'exp',
      segment: '',
      pose: '',
      gaze: '',
      onnx: 'exp',
      torchscript: 'exp',
      tensorrt: 'exp',
      openvino: 'exp',
      ncnn: 'exp',
      coreml: '',
    },
    {
      family: 'YOLO-NAS',
      status: 'Experimental',
      inference: 'exp',
      training: 'exp',
      detect: 'exp',
      segment: '',
      pose: 'exp',
      gaze: '',
      onnx: 'exp',
      torchscript: 'exp',
      tensorrt: 'exp',
      openvino: 'exp',
      ncnn: 'exp',
      coreml: '',
    },
    {
      family: 'D-FINE',
      status: 'Experimental',
      inference: 'exp',
      training: 'exp',
      detect: 'exp',
      segment: '',
      pose: '',
      gaze: '',
      onnx: 'exp',
      torchscript: 'exp',
      tensorrt: 'exp',
      openvino: 'exp',
      ncnn: '',
      coreml: '',
    },
    {
      family: 'DEIM',
      status: 'Experimental',
      inference: 'exp',
      training: 'exp',
      detect: 'exp',
      segment: '',
      pose: '',
      gaze: '',
      onnx: 'exp',
      torchscript: 'exp',
      tensorrt: 'exp',
      openvino: 'exp',
      ncnn: '',
      coreml: '',
    },
    {
      family: 'DEIMv2',
      status: 'Experimental',
      inference: 'exp',
      training: 'exp',
      detect: 'exp',
      segment: '',
      pose: '',
      gaze: '',
      onnx: 'exp',
      torchscript: 'exp',
      tensorrt: 'exp',
      openvino: 'exp',
      ncnn: '',
      coreml: '',
    },
    {
      family: 'RT-DETR',
      status: 'Experimental',
      inference: 'exp',
      training: 'exp',
      detect: 'exp',
      segment: '',
      pose: '',
      gaze: '',
      onnx: 'exp',
      torchscript: 'exp',
      tensorrt: 'exp',
      openvino: 'exp',
      ncnn: '',
      coreml: 'exp',
    },
    {
      family: 'PicoDet',
      status: 'Experimental',
      inference: 'exp',
      training: 'exp',
      detect: 'exp',
      segment: '',
      pose: '',
      gaze: '',
      onnx: 'exp',
      torchscript: 'exp',
      tensorrt: 'exp',
      openvino: 'exp',
      ncnn: 'exp',
      coreml: '',
    },
    {
      family: 'EC',
      status: 'Experimental',
      inference: 'exp',
      training: 'exp',
      detect: 'exp',
      segment: 'exp',
      pose: 'exp',
      gaze: '',
      onnx: 'exp',
      torchscript: 'exp',
      tensorrt: 'exp',
      openvino: 'exp',
      ncnn: '',
      coreml: '',
    },
    {
      family: 'RT-DETRv2',
      status: 'Experimental',
      inference: 'exp',
      training: 'exp',
      detect: 'exp',
      segment: '',
      pose: '',
      gaze: '',
      onnx: 'exp',
      torchscript: 'exp',
      tensorrt: 'exp',
      openvino: 'exp',
      ncnn: 'exp',
      coreml: '',
    },
    {
      family: 'RT-DETRv4',
      status: 'Experimental',
      inference: 'exp',
      training: 'exp',
      detect: 'exp',
      segment: '',
      pose: '',
      gaze: '',
      onnx: 'exp',
      torchscript: 'exp',
      tensorrt: 'exp',
      openvino: 'exp',
      ncnn: 'exp',
      coreml: '',
    },
    {
      family: 'DAMO-YOLO',
      status: 'Experimental',
      inference: 'exp',
      training: 'exp',
      detect: 'exp',
      segment: '',
      pose: '',
      gaze: '',
      onnx: 'exp',
      torchscript: 'exp',
      tensorrt: 'exp',
      openvino: 'exp',
      ncnn: 'exp',
      coreml: '',
    },
    {
      family: 'RTMDet',
      status: 'Experimental',
      inference: 'exp',
      training: 'exp',
      detect: 'exp',
      segment: '',
      pose: '',
      gaze: '',
      onnx: 'exp',
      torchscript: 'exp',
      tensorrt: 'exp',
      openvino: 'exp',
      ncnn: 'exp',
      coreml: '',
    },
    {
      family: 'L2CS',
      status: 'Experimental, inference-only',
      inference: 'exp',
      training: '',
      detect: '',
      segment: '',
      pose: '',
      gaze: 'exp',
      onnx: '',
      torchscript: '',
      tensorrt: '',
      openvino: '',
      ncnn: '',
      coreml: '',
    },
  ]

  const headers = ['Model family', 'v1.2.0 status', 'Inference', 'Training', 'Detection', 'Segmentation', 'Pose', 'Gaze', 'ONNX', 'TorchScript', 'TensorRT', 'OpenVINO', 'NCNN', 'CoreML']
  const columns = ['inference', 'training', 'detect', 'segment', 'pose', 'gaze', 'onnx', 'torchscript', 'tensorrt', 'openvino', 'ncnn', 'coreml']

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
            v1.2.0 validation scope
          </p>
          <p className="text-sm text-surface-600 dark:text-surface-400 mb-2">
            The heavily tested path is detection, training and inference for YOLO9 and RF-DETR, including RF-DETR segmentation.
          </p>
          <p className="text-sm text-surface-600 dark:text-surface-400">
            Other model families, tasks, and multi-GPU workflows are available but experimental.
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
            Detection, training and inference for these models receive the heaviest testing. Treat other families, tasks, and multi-GPU workflows as experimental in v1.2.0.
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
                    ? 'These are the docs for the current stable release, v1.2.0. A pre-release of the upcoming v1.3.0 is also available from the version menu.'
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
            <P>
              LibreYOLO is an MIT-licensed object detection toolkit. v1.2.0 ships a broad catalogue, but the validated support surface is intentionally narrow:
            </P>
            <ul className="space-y-2 mb-4">
              <FeatureItem><strong className="text-surface-800 dark:text-white">YOLO9 detection</strong> - the CNN path.</FeatureItem>
              <FeatureItem><strong className="text-surface-800 dark:text-white">RF-DETR detection</strong> - the transformer path.</FeatureItem>
              <FeatureItem><strong className="text-surface-800 dark:text-white">RF-DETR segmentation</strong> - the heavily tested segmentation path.</FeatureItem>
            </ul>
            <P>
              We recommend those paths as the default choice for new projects because they receive the heaviest testing around detection, training and inference. Other supported families and tasks work through the same unified <InlineCode>LibreYOLO()</InlineCode> factory, but they are experimental in v1.2.0. Use them if you have a specific reason.
            </P>
            <CodeBlock language="python">{`from libreyolo import LibreYOLO, SAMPLE_IMAGE

# Default: YOLO9 detection
model = LibreYOLO("LibreYOLO9c.pt")
result = model(SAMPLE_IMAGE, conf=0.25, save=True)

print(f"Detected {len(result)} objects")
print(result.boxes.xyxy)
print(result.saved_path)`}</CodeBlock>

            <SubHeading>Key features</SubHeading>
            <ul className="space-y-2.5 mb-4">
              <FeatureItem>Heavy testing and recommended defaults for YOLO9 detection, RF-DETR detection, and RF-DETR segmentation</FeatureItem>
              <FeatureItem>Unified <InlineCode>LibreYOLO()</InlineCode> factory for checkpoints, exported artifacts, and runtime loading</FeatureItem>
              <FeatureItem>Detection, segmentation, pose, and gaze tasks through one consistent API</FeatureItem>
              <FeatureItem>Image, directory, and video inference (with optional tiled inference for large frames)</FeatureItem>
              <FeatureItem>Built-in multi-object tracking via ByteTrack</FeatureItem>
              <FeatureItem>ONNX, TorchScript, TensorRT, OpenVINO, NCNN, and CoreML export with embedded metadata, plus matching runtime backends</FeatureItem>
              <FeatureItem>COCO-compatible validation with mAP metrics, plus segmentation and pose validators</FeatureItem>
              <FeatureItem>Ultralytics-style <InlineCode>libreyolo</InlineCode> command-line tool for predict / train / val / export</FeatureItem>
              <FeatureItem>Accepts any image format: file paths, URLs, PIL, NumPy, PyTorch tensors, raw bytes</FeatureItem>
            </ul>
          </motion.div>

          <Divider />

          {/* ────────────── INSTALLATION ────────────── */}
          <SectionHeading id="compatibility" icon={CheckCircle2}>Compatibility</SectionHeading>
          <P>
            Use this matrix as the quick v1.2.0 support map. A checkmark means the path is supported in the validated documentation surface, <InlineCode>exp</InlineCode> means the path exists but is experimental, and empty cells are not currently supported or should not be relied on.
          </P>
          <CompatibilityMatrix />
          <p className="text-sm text-surface-600 dark:text-surface-400 leading-relaxed mb-4">
            CoreML exports produce <InlineCode>.mlpackage</InlineCode> bundles and require <InlineCode>libreyolo[coreml]</InlineCode>. CoreML inference is macOS only, INT8 is not supported, and embedded CoreML NMS is not available for RF-DETR, D-FINE, DEIM, DEIMv2, or EC.
          </p>

          <Divider />

          <SectionHeading id="installation" icon={Terminal}>Installation</SectionHeading>
          <SubHeading>Requirements</SubHeading>
          <ul className="space-y-1.5 mb-4">
            <li className="flex items-center gap-2 text-surface-600 dark:text-surface-400">
              <span className="w-1.5 h-1.5 rounded-full bg-libre-400" />Python 3.10+
            </li>
            <li className="flex items-center gap-2 text-surface-600 dark:text-surface-400">
              <span className="w-1.5 h-1.5 rounded-full bg-libre-400" />PyTorch 1.13+ and torchvision 0.11+
            </li>
          </ul>

          <SubHeading>From PyPI</SubHeading>
          <CodeBlock language="bash">{`pip install libreyolo`}</CodeBlock>
          <P>
            These docs track the upcoming v1.2.0 dev branch. Until v1.2.0 is published to PyPI, use a source install for the features documented on this page.
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
# Tracking dependencies are part of the base install in v1.2.0 dev.

# CoreML export and inference (macOS only for runtime)
pip install libreyolo[coreml]
# or: pip install coremltools

# L2CS gaze optional auto-download helper
pip install libreyolo[gaze]
# Optional parity with the upstream RetinaFace-based L2CS pipeline
pip install libreyolo[gaze-retinaface]

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
            LibreYOLO ships a small validated v1.2.0 surface plus a broader catalogue of supported models. Every model loads through the same <InlineCode>LibreYOLO()</InlineCode> factory, but only the validated paths below should be treated as heavily tested.
          </P>

          <ValidatedModelHeader title="YOLO9 - CNN flagship">
            <SupportBadge variant="validated">Default: LibreYOLO9c.pt</SupportBadge>
            <SupportBadge variant="validated">Heavily tested: detection, training and inference</SupportBadge>
            <SupportBadge>Experimental: segment, multi-GPU</SupportBadge>
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
            <SupportBadge>Experimental</SupportBadge>{' '}
            <strong className="text-surface-800 dark:text-white">Segmentation checkpoints:</strong>{' '}
            <Checkpoints names={['LibreYOLO9t-seg.pt', 'LibreYOLO9s-seg.pt', 'LibreYOLO9m-seg.pt', 'LibreYOLO9c-seg.pt']} />
            . See the <a href="#segmentation" className="text-libre-600 dark:text-libre-400 hover:underline">Segmentation</a> section.
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")
# Experimental segmentation variant
# model = LibreYOLO("LibreYOLO9c-seg.pt")`}</CodeBlock>

          <ValidatedModelHeader title="RF-DETR - transformer flagship">
            <SupportBadge variant="validated">Recommended transformer path</SupportBadge>
            <SupportBadge variant="validated">Heavily tested: detection, segmentation, training and inference</SupportBadge>
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
            <SupportBadge variant="validated">Heavily tested</SupportBadge>{' '}
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
            Detection-capable families that share the same factory and API surface as the validated paths. These are experimental in v1.2.0. Each checkpoint name links to its Hugging Face model card on the <a href="https://huggingface.co/LibreYOLO" target="_blank" rel="noopener noreferrer" className="text-libre-600 dark:text-libre-400 hover:underline">LibreYOLO org</a>; pass any name to <InlineCode>LibreYOLO()</InlineCode> and the factory will fetch it on first use.
          </P>
          <DocTable
            headers={['Family', 'Status', 'Tasks', 'Checkpoints']}
            rows={[
              ['YOLOX', <SupportBadge>Experimental</SupportBadge>, 'detect', <Checkpoints key="yolox" names={['LibreYOLOXn.pt', 'LibreYOLOXt.pt', 'LibreYOLOXs.pt', 'LibreYOLOXm.pt', 'LibreYOLOXl.pt', 'LibreYOLOXx.pt']} />],
              ['YOLO9-E2E', <SupportBadge>Experimental</SupportBadge>, 'detect', <Checkpoints key="y9e2e" names={['LibreYOLO9E2Et.pt', 'LibreYOLO9E2Es.pt', 'LibreYOLO9E2Em.pt', 'LibreYOLO9E2Ec.pt']} />],
              ['YOLO-NAS', <SupportBadge>Experimental</SupportBadge>, 'detect, pose', <Checkpoints key="ynas" link={false} names={['LibreYOLONASs.pt', 'LibreYOLONASm.pt', 'LibreYOLONASl.pt', 'LibreYOLONASn-pose.pt', 'LibreYOLONASs-pose.pt', 'LibreYOLONASm-pose.pt', 'LibreYOLONASl-pose.pt']} />],
              ['D-FINE', <SupportBadge>Experimental</SupportBadge>, 'detect', <Checkpoints key="dfine" names={['LibreDFINEn.pt', 'LibreDFINEs.pt', 'LibreDFINEm.pt', 'LibreDFINEl.pt', 'LibreDFINEx.pt']} />],
              ['DEIM', <SupportBadge>Experimental</SupportBadge>, 'detect', <Checkpoints key="deim" names={['LibreDEIMn.pt', 'LibreDEIMs.pt', 'LibreDEIMm.pt', 'LibreDEIMl.pt', 'LibreDEIMx.pt']} />],
              ['DEIMv2', <SupportBadge>Experimental</SupportBadge>, 'detect', <Checkpoints key="deimv2" names={['LibreDEIMv2atto.pt', 'LibreDEIMv2femto.pt', 'LibreDEIMv2pico.pt', 'LibreDEIMv2n.pt', 'LibreDEIMv2s.pt', 'LibreDEIMv2m.pt', 'LibreDEIMv2l.pt', 'LibreDEIMv2x.pt']} />],
              ['RT-DETR', <SupportBadge>Experimental</SupportBadge>, 'detect', <Checkpoints key="rtdetr" names={['LibreRTDETRr18.pt', 'LibreRTDETRr34.pt', 'LibreRTDETRr50.pt', 'LibreRTDETRr50m.pt', 'LibreRTDETRr101.pt', 'LibreRTDETRl.pt', 'LibreRTDETRx.pt']} />],
              ['RT-DETRv2', <SupportBadge>Experimental</SupportBadge>, 'detect', <Checkpoints key="rtdetrv2" names={['LibreRTDETRv2r18.pt', 'LibreRTDETRv2r34.pt', 'LibreRTDETRv2r50.pt', 'LibreRTDETRv2r50m.pt', 'LibreRTDETRv2r101.pt']} />],
              ['RT-DETRv4', <SupportBadge>Experimental</SupportBadge>, 'detect', <Checkpoints key="rtdetrv4" names={['LibreRTDETRv4s.pt', 'LibreRTDETRv4m.pt', 'LibreRTDETRv4l.pt', 'LibreRTDETRv4x.pt']} />],
              ['PicoDet', <SupportBadge>Experimental</SupportBadge>, 'detect', <Checkpoints key="picodet" names={['LibrePICODETs.pt', 'LibrePICODETm.pt', 'LibrePICODETl.pt']} />],
              ['EdgeCrafter', <SupportBadge>Experimental</SupportBadge>, 'detect, pose, segment', <Checkpoints key="ec" names={['LibreECs.pt', 'LibreECm.pt', 'LibreECl.pt', 'LibreECx.pt', 'LibreECs-pose.pt', 'LibreECm-pose.pt', 'LibreECl-pose.pt', 'LibreECx-pose.pt', 'LibreECs-seg.pt', 'LibreECm-seg.pt', 'LibreECl-seg.pt', 'LibreECx-seg.pt']} />],
              ['DAMO-YOLO', <SupportBadge>Experimental</SupportBadge>, 'detect', <Checkpoints key="damo" names={['LibreDAMOYOLOns.pt', 'LibreDAMOYOLOnm.pt', 'LibreDAMOYOLOnl.pt', 'LibreDAMOYOLOt.pt', 'LibreDAMOYOLOs.pt', 'LibreDAMOYOLOm.pt', 'LibreDAMOYOLOl.pt']} />],
              ['RTMDet', <SupportBadge>Experimental</SupportBadge>, 'detect', <Checkpoints key="rtmdet" names={['LibreRTMDett.pt', 'LibreRTMDets.pt', 'LibreRTMDetm.pt', 'LibreRTMDetl.pt', 'LibreRTMDetx.pt']} />],
            ]}
          />
          <P className="text-sm">
            <strong className="text-surface-800 dark:text-white">Hosting note:</strong> YOLO-NAS checkpoints (plain text above) are hosted on Deci&apos;s CDN under their proprietary weights license, not on the LibreYOLO Hugging Face org. The factory still downloads them automatically on first use.
          </P>

          <SubHeading>Specialized models</SubHeading>
          <DocTable
            headers={['Family', 'Status', 'Tasks', 'Checkpoints']}
            rows={[
              ['L2CS', <SupportBadge>Experimental</SupportBadge>, <span key="t">gaze (inference-only) - see <a href="#gaze" className="text-libre-600 dark:text-libre-400 hover:underline">Gaze Estimation</a></span>, <Checkpoints key="l2cs" link={false} names={['LibreL2CSr50.pt']} />],
            ]}
          />
          <P className="text-sm">
            L2CS architecture sizes include r18, r34, r50, r101, and r152, but the upstream-published Gaze360 checkpoint is ResNet-50. Install <InlineCode>libreyolo[gaze]</InlineCode> for the optional Google Drive helper, or pass a local checkpoint path for other sizes.
          </P>

          <SubHeading>Factory function</SubHeading>
          <P>
            Use the <InlineCode>LibreYOLO()</InlineCode> factory for every model and runtime. Give it an official checkpoint name or exported artifact path, then let it choose the right model family, task, class count, and runtime:
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

# Default: YOLO9 detection
model = LibreYOLO("LibreYOLO9c.pt")

# Flagship: RF-DETR
model = LibreYOLO("LibreRFDETRs.pt")

# Segmentation checkpoints use the same factory path
model = LibreYOLO("LibreRFDETRs-seg.pt")       # validated segmentation
model = LibreYOLO("LibreYOLO9c-seg.pt")        # experimental segmentation

# Exported deployment formats
model = LibreYOLO("model.onnx")                # ONNX Runtime
model = LibreYOLO("model.engine")              # TensorRT
model = LibreYOLO("model.mlpackage")           # CoreML (macOS)
model = LibreYOLO("model_openvino/")           # OpenVINO (directory)
model = LibreYOLO("model_ncnn/")               # NCNN (directory)`}</CodeBlock>
          <P>
            For recognized official checkpoint filenames, LibreYOLO can auto-download missing weights. For custom filenames, point at an explicit local path. Experimental families still load through the same factory, but keep new projects on YOLO9 detection or RF-DETR detection/segmentation. Use them if you have a specific reason.
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
model = LibreYOLO("LibreRFDETRs-seg.pt")

# 2. Override regardless of filename
model = LibreYOLO("custom_weights.pt", task="segment")

# 3. Detection is implicit
model = LibreYOLO("LibreYOLO9c.pt")  # task="detect"`}</CodeBlock>

          <SubHeading>Per-family task support</SubHeading>
          <DocTable
            headers={['Family', 'v1.2.0 status', 'Default', 'Supported tasks']}
            rows={[
              [<strong key="y9">YOLO9</strong>, 'detect single-GPU heavily tested; segment and multi-GPU experimental', 'detect', 'detect, segment'],
              [<strong key="rfd">RF-DETR</strong>, 'detect and segment single-GPU heavily tested; multi-GPU experimental', 'detect', 'detect, segment'],
              ['YOLOX', 'experimental', 'detect', 'detect'],
              ['YOLO9-E2E', 'experimental', 'detect', 'detect'],
              ['YOLO-NAS', 'experimental', 'detect', 'detect, pose'],
              ['D-FINE / DEIM / DEIMv2', 'experimental', 'detect', 'detect'],
              ['RT-DETR / RT-DETRv2 / RT-DETRv4', 'experimental', 'detect', 'detect'],
              ['PicoDet', 'experimental', 'detect', 'detect'],
              ['EdgeCrafter (EC)', 'experimental', 'detect', 'detect, pose, segment'],
              ['DAMO-YOLO / RTMDet', 'experimental', 'detect', 'detect'],
              ['L2CS', 'experimental', 'gaze', 'gaze (inference-only)'],
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
            <InlineCode>LibreYOLORTDETR</InlineCode> and <InlineCode>LibreYOLORFDETR</InlineCode> are old names for <InlineCode>LibreRTDETR</InlineCode> and <InlineCode>LibreRFDETR</InlineCode> respectively. They still resolve with a <InlineCode>DeprecationWarning</InlineCode> - update imports when convenient.
          </P>

          <Divider />

          {/* ────────────── PREDICTION ────────────── */}
          <SectionHeading id="prediction" icon={Crosshair}>Prediction</SectionHeading>
          <P>
            The single-GPU prediction path is heavily tested for YOLO9 detection, RF-DETR detection, and RF-DETR segmentation. Other families and tasks use the same API but are experimental in v1.2.0.
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

          <Divider />

          {/* ────────────── TILED INFERENCE ────────────── */}
          <SectionHeading id="tiled-inference" icon={Grid3x3}>Tiled Inference</SectionHeading>
          <P>
            For images much larger than the model's input size (e.g., satellite imagery, drone footage), tiled inference splits the image into overlapping tiles, runs detection on each, and merges results.
          </P>
          <P>
            Tiling is detection-only in v1.2.0 dev. It rejects segmentation masks, and it cannot be combined with <InlineCode>augment=True</InlineCode>.
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
            LibreYOLO ships a ByteTrack multi-object tracker that consumes <InlineCode>Results</InlineCode> from any detector and adds persistent track IDs. It is most tested with single-GPU YOLO9 detection and RF-DETR detection; other detection families are experimental in v1.2.0.
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

          <Divider />

          {/* ────────────── SEGMENTATION ────────────── */}
          <SectionHeading id="segmentation" icon={Scissors}>Segmentation</SectionHeading>
          <ValidationScopeCallout />
          <P>
            RF-DETR segmentation is the heavily tested segmentation path in v1.2.0. YOLO9 segmentation and EdgeCrafter segmentation are available through the same <InlineCode>-seg</InlineCode> suffix, but they are experimental.
          </P>

          <SubHeading>Run segmentation</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

# RF-DETR segmentation, heavily tested on single GPU
model = LibreYOLO("LibreRFDETRs-seg.pt")
result = model("photo.jpg")

# YOLO9 segmentation is available but experimental in v1.2.0
# model = LibreYOLO("LibreYOLO9c-seg.pt")

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
            RF-DETR segmentation uses the RF-DETR COCO-format training pipeline and is part of the heavily tested single-GPU scope. YOLO9 segmentation and EdgeCrafter segmentation training are available but experimental in v1.2.0.
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
            Gaze direction estimation is provided by the <InlineCode>LibreL2CS</InlineCode> family, an L2CS-Net port with a ResNet trunk and two angle-bin classification heads. It is a two-stage model: an upstream face detector locates faces, then the gaze head predicts per-face pitch and yaw in radians. It is inference-only and experimental in v1.2.0.
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

          <SectionHeading id="training" icon={GraduationCap}>Training</SectionHeading>
          <ValidationScopeCallout />
          <P>
            The heavily tested training paths are single-GPU YOLO9 detection, RF-DETR detection, and RF-DETR segmentation. Other model-family trainers, YOLO9 segmentation training, and multi-GPU workflows are available but experimental in v1.2.0.
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
)

print(f"Best mAP50-95: {results['best_mAP50_95']:.3f}")
print(f"Best checkpoint: {results['best_checkpoint']}")`}</CodeBlock>
          <P>
            After training completes, the model instance is automatically reloaded with the best weights so you can call <InlineCode>model(...)</InlineCode> immediately. YOLO9 segmentation training is supported via <InlineCode>LibreYOLO(&quot;LibreYOLO9c-seg.pt&quot;)</InlineCode>, but it is experimental in v1.2.0.
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
            Other families have trainer hooks, but they are not the recommended path in v1.2.0. Keep new work on YOLO9 detection or RF-DETR detection/segmentation; use experimental trainers only for compatibility, benchmark reproduction, or targeted research. DAMO-YOLO, PicoDet, RTMDet, and EC training require an explicit <InlineCode>allow_experimental=True</InlineCode> acknowledgement.
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
            YOLO9 and RF-DETR support multi-GPU training through PyTorch DistributedDataParallel, but multi-GPU is outside the heavily tested v1.2.0 scope. Launch the training script with <InlineCode>torchrun</InlineCode>:
          </P>
          <CodeBlock language="bash">{`# 4-GPU node
torchrun --nproc_per_node=4 train_yolo9.py

# Multi-node - see PyTorch's torchrun docs for --nnodes / --rdzv_endpoint`}</CodeBlock>
          <CodeBlock language="python" filename="train_yolo9.py">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")
# Pass device="" (auto-detect) and let torchrun set the rank
model.train(data="coco128.yaml", epochs=300, batch=16)`}</CodeBlock>

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
            Segmentation validation returns mask metrics with <InlineCode>(M)</InlineCode> suffixes alongside bbox metrics with <InlineCode>(B)</InlineCode> suffixes. Pose validation returns COCO keypoint metrics through <InlineCode>PoseValidator</InlineCode>.
          </P>

          <Divider />

          {/* ────────────── EXPORT ────────────── */}
          <SectionHeading id="export" icon={Upload}>Export</SectionHeading>
          <P>
            Export PyTorch models to ONNX, TorchScript, TensorRT, OpenVINO, NCNN, or CoreML for deployment. The heavily tested export and runtime-backend paths are single-GPU YOLO9 detection, RF-DETR detection, and RF-DETR segmentation. Other families and tasks are experimental.
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
model.export(format="coreml")`}</CodeBlock>

          <SubHeading>All export parameters</SubHeading>
          <CodeBlock language="python">{`path = model.export(
    format="onnx",            # "onnx", "torchscript", "tensorrt", "openvino", "ncnn", or "coreml"
    output_path="model.onnx", # output file (auto-generated if None)
    imgsz=640,                # input resolution (default: model's native)
    opset=None,               # ONNX opset (auto: 13, or 17 for wrappers that need it)
    simplify=True,            # run onnxsim graph simplification
    dynamic=True,             # enable dynamic batch axis
    half=False,               # export in FP16
    batch=1,                  # batch size for static graph
    device=None,              # device to trace on (default: model's current device)
    int8=False,               # INT8 quantization (TensorRT / OpenVINO only)
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
    nms=False,                # CoreML embedded NMS where supported
    iou=0.45,                 # CoreML embedded NMS IoU threshold
    conf=0.25,                # CoreML embedded NMS confidence threshold
    verbose=False,            # verbose logging
)`}</CodeBlock>
          <P>
            OpenVINO INT8 export additionally requires <InlineCode>nncf</InlineCode>. NCNN export writes a directory containing <InlineCode>model.ncnn.param</InlineCode>, <InlineCode>model.ncnn.bin</InlineCode>, and <InlineCode>metadata.yaml</InlineCode>. CoreML export writes a <InlineCode>.mlpackage</InlineCode> bundle, requires <InlineCode>coremltools</InlineCode>, and does not support INT8.
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
              [<InlineCode key="e">export</InlineCode>, 'Export to ONNX / TorchScript / TensorRT / OpenVINO / NCNN / CoreML'],
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
              [<InlineCode key="t">train</InlineCode>, <span key="tv"><InlineCode>epochs</InlineCode>, <InlineCode>batch</InlineCode>, <InlineCode>imgsz</InlineCode>, <InlineCode>lr0</InlineCode>, <InlineCode>optimizer</InlineCode>, <InlineCode>scheduler</InlineCode>, <InlineCode>workers</InlineCode>, <InlineCode>seed</InlineCode>, <InlineCode>resume</InlineCode>, <InlineCode>amp</InlineCode>, <InlineCode>allow_download_scripts</InlineCode>, <InlineCode>dry_run</InlineCode></span>],
              [<InlineCode key="v">val</InlineCode>, <span key="vv"><InlineCode>split</InlineCode>, <InlineCode>batch</InlineCode>, <InlineCode>imgsz</InlineCode>, <InlineCode>conf</InlineCode>, <InlineCode>iou</InlineCode>, <InlineCode>max_det</InlineCode>, <InlineCode>half</InlineCode>, <InlineCode>data_dir</InlineCode>, <InlineCode>use_coco_eval</InlineCode>, <InlineCode>project</InlineCode>, <InlineCode>name</InlineCode>, <InlineCode>exist_ok</InlineCode>, <InlineCode>save_json</InlineCode>, <InlineCode>allow_download_scripts</InlineCode></span>],
              [<InlineCode key="e">export</InlineCode>, <span key="ev"><InlineCode>format</InlineCode>, <InlineCode>imgsz</InlineCode>, <InlineCode>batch</InlineCode>, <InlineCode>half</InlineCode>, <InlineCode>int8</InlineCode>, <InlineCode>dynamic</InlineCode>, <InlineCode>simplify</InlineCode>, <InlineCode>opset</InlineCode>, <InlineCode>data</InlineCode>, <InlineCode>fraction</InlineCode>, <InlineCode>device</InlineCode>, <InlineCode>allow_download_scripts</InlineCode>, <InlineCode>verbose</InlineCode></span>],
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
            Additional experimental trainers exist for YOLO-NAS, D-FINE, DEIM, DEIMv2, EC, PicoDet, DAMO-YOLO, RT-DETRv2/v4, and RTMDet. They follow the same <InlineCode>model.train(data=&quot;...yaml&quot;, ...)</InlineCode> shape but their defaults and experimental gates are family-specific.
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


/* ============================================================
   Chinese (zh-CN) content bundle for LibreYOLO docs v1.2.0
   Append to src/app/[locale]/docs/page.jsx.
   Reuses shared presentational components already defined in that file.
   ============================================================ */

/* ─── 1. Section metadata (Chinese titles) ─── */
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
  { id: 'pose', title: '姿态估计', icon: PersonStanding },
  { id: 'gaze', title: '视线估计', icon: Eye },
  { id: 'training', title: '训练', icon: GraduationCap },
  { id: 'validation', title: '验证', icon: CheckCircle2 },
  { id: 'export', title: '导出', icon: Upload },
  { id: 'torchscript-inference', title: 'TorchScript 推理', icon: Cpu },
  { id: 'onnx-inference', title: 'ONNX 推理', icon: Cpu },
  { id: 'tensorrt-inference', title: 'TensorRT 推理', icon: Cpu },
  { id: 'openvino-inference', title: 'OpenVINO 推理', icon: Cpu },
  { id: 'ncnn-inference', title: 'NCNN 推理', icon: Cpu },
  { id: 'coreml-inference', title: 'CoreML 推理', icon: Cpu },
  { id: 'cli', title: '命令行工具', icon: SquareTerminal },
  { id: 'api-reference', title: 'API 参考', icon: FileCode },
  { id: 'architecture', title: '架构指南', icon: Wrench },
  { id: 'dataset-format', title: '数据集格式', icon: Database },
]

/* ─── 2. Sidebar (Chinese) ─── */
function SidebarZh({ activeSection, onNavigate, currentVersion = 'v1.2.0', className = '' }) {
  const versionLabelsZh = {
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
                <span className="text-[11px] font-semibold uppercase tracking-wide">{versionLabelsZh[label] || label}</span>
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

/* ─── 3. Content-bearing twins ─── */

function ValidatedModelHeaderZh({ title, children }) {
  return (
    <div className="mt-10 mb-5 rounded-lg border border-emerald-500/30 bg-emerald-500/[0.08] dark:bg-emerald-500/[0.12] px-4 py-3">
      <div className="flex flex-wrap items-center gap-3">
        <h3 className="text-lg font-semibold text-surface-900 dark:text-white underline decoration-emerald-500 decoration-2 underline-offset-4">
          {title}
        </h3>
        <SupportBadge variant="validated">推荐</SupportBadge>
      </div>
      <div className="mt-3 flex flex-wrap gap-2">
        {children}
      </div>
    </div>
  )
}

function CompatibilityMatrixZh() {
  const rows = [
    {
      family: 'YOLO9',
      status: '已验证 detect，单 GPU',
      inference: 'yes',
      training: 'yes',
      detect: 'yes',
      segment: 'exp',
      pose: '',
      gaze: '',
      onnx: 'yes',
      torchscript: 'yes',
      tensorrt: 'yes',
      openvino: 'yes',
      ncnn: 'yes',
      coreml: 'yes',
    },
    {
      family: 'RF-DETR',
      status: '已验证 detect + segment，单 GPU',
      inference: 'yes',
      training: 'yes',
      detect: 'yes',
      segment: 'yes',
      pose: '',
      gaze: '',
      onnx: 'yes',
      torchscript: 'exp',
      tensorrt: 'yes',
      openvino: 'yes',
      ncnn: '',
      coreml: 'exp',
    },
    {
      family: 'YOLOX',
      status: '实验性',
      inference: 'exp',
      training: 'exp',
      detect: 'exp',
      segment: '',
      pose: '',
      gaze: '',
      onnx: 'exp',
      torchscript: 'exp',
      tensorrt: 'exp',
      openvino: 'exp',
      ncnn: 'exp',
      coreml: 'exp',
    },
    {
      family: 'YOLO9-E2E',
      status: '实验性',
      inference: 'exp',
      training: 'exp',
      detect: 'exp',
      segment: '',
      pose: '',
      gaze: '',
      onnx: 'exp',
      torchscript: 'exp',
      tensorrt: 'exp',
      openvino: 'exp',
      ncnn: 'exp',
      coreml: '',
    },
    {
      family: 'YOLO-NAS',
      status: '实验性',
      inference: 'exp',
      training: 'exp',
      detect: 'exp',
      segment: '',
      pose: 'exp',
      gaze: '',
      onnx: 'exp',
      torchscript: 'exp',
      tensorrt: 'exp',
      openvino: 'exp',
      ncnn: 'exp',
      coreml: '',
    },
    {
      family: 'D-FINE',
      status: '实验性',
      inference: 'exp',
      training: 'exp',
      detect: 'exp',
      segment: '',
      pose: '',
      gaze: '',
      onnx: 'exp',
      torchscript: 'exp',
      tensorrt: 'exp',
      openvino: 'exp',
      ncnn: '',
      coreml: '',
    },
    {
      family: 'DEIM',
      status: '实验性',
      inference: 'exp',
      training: 'exp',
      detect: 'exp',
      segment: '',
      pose: '',
      gaze: '',
      onnx: 'exp',
      torchscript: 'exp',
      tensorrt: 'exp',
      openvino: 'exp',
      ncnn: '',
      coreml: '',
    },
    {
      family: 'DEIMv2',
      status: '实验性',
      inference: 'exp',
      training: 'exp',
      detect: 'exp',
      segment: '',
      pose: '',
      gaze: '',
      onnx: 'exp',
      torchscript: 'exp',
      tensorrt: 'exp',
      openvino: 'exp',
      ncnn: '',
      coreml: '',
    },
    {
      family: 'RT-DETR',
      status: '实验性',
      inference: 'exp',
      training: 'exp',
      detect: 'exp',
      segment: '',
      pose: '',
      gaze: '',
      onnx: 'exp',
      torchscript: 'exp',
      tensorrt: 'exp',
      openvino: 'exp',
      ncnn: '',
      coreml: 'exp',
    },
    {
      family: 'PicoDet',
      status: '实验性',
      inference: 'exp',
      training: 'exp',
      detect: 'exp',
      segment: '',
      pose: '',
      gaze: '',
      onnx: 'exp',
      torchscript: 'exp',
      tensorrt: 'exp',
      openvino: 'exp',
      ncnn: 'exp',
      coreml: '',
    },
    {
      family: 'EC',
      status: '实验性',
      inference: 'exp',
      training: 'exp',
      detect: 'exp',
      segment: 'exp',
      pose: 'exp',
      gaze: '',
      onnx: 'exp',
      torchscript: 'exp',
      tensorrt: 'exp',
      openvino: 'exp',
      ncnn: '',
      coreml: '',
    },
    {
      family: 'RT-DETRv2',
      status: '实验性',
      inference: 'exp',
      training: 'exp',
      detect: 'exp',
      segment: '',
      pose: '',
      gaze: '',
      onnx: 'exp',
      torchscript: 'exp',
      tensorrt: 'exp',
      openvino: 'exp',
      ncnn: 'exp',
      coreml: '',
    },
    {
      family: 'RT-DETRv4',
      status: '实验性',
      inference: 'exp',
      training: 'exp',
      detect: 'exp',
      segment: '',
      pose: '',
      gaze: '',
      onnx: 'exp',
      torchscript: 'exp',
      tensorrt: 'exp',
      openvino: 'exp',
      ncnn: 'exp',
      coreml: '',
    },
    {
      family: 'DAMO-YOLO',
      status: '实验性',
      inference: 'exp',
      training: 'exp',
      detect: 'exp',
      segment: '',
      pose: '',
      gaze: '',
      onnx: 'exp',
      torchscript: 'exp',
      tensorrt: 'exp',
      openvino: 'exp',
      ncnn: 'exp',
      coreml: '',
    },
    {
      family: 'RTMDet',
      status: '实验性',
      inference: 'exp',
      training: 'exp',
      detect: 'exp',
      segment: '',
      pose: '',
      gaze: '',
      onnx: 'exp',
      torchscript: 'exp',
      tensorrt: 'exp',
      openvino: 'exp',
      ncnn: 'exp',
      coreml: '',
    },
    {
      family: 'L2CS',
      status: '实验性，仅推理',
      inference: 'exp',
      training: '',
      detect: '',
      segment: '',
      pose: '',
      gaze: 'exp',
      onnx: '',
      torchscript: '',
      tensorrt: '',
      openvino: '',
      ncnn: '',
      coreml: '',
    },
  ]

  const headers = ['模型系列', 'v1.2.0 状态', '推理', '训练', '检测', '分割', '姿态', '视线', 'ONNX', 'TorchScript', 'TensorRT', 'OpenVINO', 'NCNN', 'CoreML']
  const columns = ['inference', 'training', 'detect', 'segment', 'pose', 'gaze', 'onnx', 'torchscript', 'tensorrt', 'openvino', 'ncnn', 'coreml']

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

function ValidationScopeCalloutZh({ className = '' }) {
  return (
    <div className={`my-6 rounded-xl border border-emerald-500/30 bg-emerald-500/5 dark:bg-emerald-500/10 p-4 ${className}`}>
      <div className="flex items-start gap-3">
        <ShieldCheck className="w-5 h-5 text-emerald-600 dark:text-emerald-400 mt-0.5 shrink-0" />
        <div>
          <p className="font-semibold text-surface-900 dark:text-white mb-2">
            v1.2.0 验证范围
          </p>
          <p className="text-sm text-surface-600 dark:text-surface-400 mb-2">
            经过深度测试的路径是 YOLO9 和 RF-DETR 的检测、训练与推理，包括 RF-DETR 分割。
          </p>
          <p className="text-sm text-surface-600 dark:text-surface-400">
            其他模型系列、任务和多 GPU 工作流均可用，但属实验性。
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
            这些模型的检测、训练与推理获得了最充分的测试。在 v1.2.0 中，请将其他系列、任务和多 GPU 工作流视为实验性。
          </p>
        </div>
      </div>
    </div>
  )
}

/* ─── 4. Main docs page (Chinese) ─── */
function DocsPageZh({ version = 'v1.2.0', isLatest = true }) {
  const [activeSection, setActiveSection] = useState('introduction')
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [docsCopied, setDocsCopied] = useState(false)

  // Scroll spy - keeps looping over the English `sections` so ids match the DOM
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
                    ? '这是当前稳定版本 v1.2.0 的文档。即将发布的 v1.3.0 预发布版也可在版本菜单中找到。'
                    : '此归档版本保留可链接状态，以便旧版安装、搜索结果和智能体能够定位到正确的文档。'}
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
            <P>
              LibreYOLO 是一个采用 MIT 许可证的目标检测工具包。v1.2.0 提供了广泛的模型目录，但经过验证的支持范围有意保持精简：
            </P>
            <ul className="space-y-2 mb-4">
              <FeatureItem><strong className="text-surface-800 dark:text-white">YOLO9 检测</strong> - CNN 路径。</FeatureItem>
              <FeatureItem><strong className="text-surface-800 dark:text-white">RF-DETR 检测</strong> - Transformer 路径。</FeatureItem>
              <FeatureItem><strong className="text-surface-800 dark:text-white">RF-DETR 分割</strong> - 经过深度测试的分割路径。</FeatureItem>
            </ul>
            <P>
              我们推荐将这些路径作为新项目的默认选择，因为它们在检测、训练和推理方面经过了最充分的测试。其他受支持的系列和任务通过同一个统一的 <InlineCode>LibreYOLO()</InlineCode> 工厂函数工作，但在 v1.2.0 中仍属实验性。如有特定理由再使用它们。
            </P>
            <CodeBlock language="python">{`from libreyolo import LibreYOLO, SAMPLE_IMAGE

# Default: YOLO9 detection
model = LibreYOLO("LibreYOLO9c.pt")
result = model(SAMPLE_IMAGE, conf=0.25, save=True)

print(f"Detected {len(result)} objects")
print(result.boxes.xyxy)
print(result.saved_path)`}</CodeBlock>

            <SubHeading>核心特性</SubHeading>
            <ul className="space-y-2.5 mb-4">
              <FeatureItem>针对 YOLO9 检测、RF-DETR 检测和 RF-DETR 分割的深度测试与推荐默认配置</FeatureItem>
              <FeatureItem>统一的 <InlineCode>LibreYOLO()</InlineCode> 工厂函数，用于加载检查点、导出产物和运行时</FeatureItem>
              <FeatureItem>通过一致的 API 支持检测、分割、姿态和视线任务</FeatureItem>
              <FeatureItem>支持图像、目录和视频推理（大尺寸帧可选用分块推理）</FeatureItem>
              <FeatureItem>通过 ByteTrack 内置多目标跟踪</FeatureItem>
              <FeatureItem>支持 ONNX、TorchScript、TensorRT、OpenVINO、NCNN 和 CoreML 导出（含内嵌元数据），并提供匹配的运行时后端</FeatureItem>
              <FeatureItem>兼容 COCO 的验证与 mAP 指标，并提供分割和姿态验证器</FeatureItem>
              <FeatureItem>Ultralytics 风格的 <InlineCode>libreyolo</InlineCode> 命令行工具，用于 predict / train / val / export</FeatureItem>
              <FeatureItem>接受任意图像格式：文件路径、URL、PIL、NumPy、PyTorch 张量、原始字节</FeatureItem>
            </ul>
          </motion.div>

          <Divider />

          {/* ────────────── COMPATIBILITY ────────────── */}
          <SectionHeading id="compatibility" icon={CheckCircle2}>兼容性</SectionHeading>
          <P>
            可将此矩阵用作 v1.2.0 的快速支持速查表。对勾表示该路径在经过验证的文档范围内受支持，<InlineCode>exp</InlineCode> 表示该路径存在但属实验性，空白单元格表示当前不受支持或不应依赖。
          </P>
          <CompatibilityMatrixZh />
          <p className="text-sm text-surface-600 dark:text-surface-400 leading-relaxed mb-4">
            CoreML 导出会生成 <InlineCode>.mlpackage</InlineCode> 包，并需要 <InlineCode>libreyolo[coreml]</InlineCode>。CoreML 推理仅限 macOS，不支持 INT8，且 RF-DETR、D-FINE、DEIM、DEIMv2 或 EC 不提供内嵌的 CoreML NMS。
          </p>

          <Divider />

          {/* ────────────── INSTALLATION ────────────── */}
          <SectionHeading id="installation" icon={Terminal}>安装</SectionHeading>
          <SubHeading>环境要求</SubHeading>
          <ul className="space-y-1.5 mb-4">
            <li className="flex items-center gap-2 text-surface-600 dark:text-surface-400">
              <span className="w-1.5 h-1.5 rounded-full bg-libre-400" />Python 3.10+
            </li>
            <li className="flex items-center gap-2 text-surface-600 dark:text-surface-400">
              <span className="w-1.5 h-1.5 rounded-full bg-libre-400" />PyTorch 1.13+ 与 torchvision 0.11+
            </li>
          </ul>

          <SubHeading>从 PyPI 安装</SubHeading>
          <CodeBlock language="bash">{`pip install libreyolo`}</CodeBlock>
          <P>
            本文档对应即将发布的 v1.2.0 开发分支。在 v1.2.0 发布到 PyPI 之前，请使用源码安装以获得本页所述功能。
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
# Tracking dependencies are part of the base install in v1.2.0 dev.

# CoreML export and inference (macOS only for runtime)
pip install libreyolo[coreml]
# or: pip install coremltools

# L2CS gaze optional auto-download helper
pip install libreyolo[gaze]
# Optional parity with the upstream RetinaFace-based L2CS pipeline
pip install libreyolo[gaze-retinaface]

# Install every optional LibreYOLO extra
pip install libreyolo[all]`}</CodeBlock>

          <P>如果使用 <InlineCode>uv</InlineCode>，最可靠的方式是为每个 extra 创建独立的虚拟环境：</P>
          <CodeBlock language="bash">{`# ONNX environment
uv venv .venv-onnx
uv pip install --python .venv-onnx/bin/python -e '.[onnx]'

# RT-DETR environment
uv venv .venv-rtdetr
uv pip install --python .venv-rtdetr/bin/python -e '.[rtdetr]'

# Repeat with .[rfdetr], .[openvino], .[ncnn], .[coreml], .[gaze], .[tracking], or .[tensorrt] as needed`}</CodeBlock>
          <P>
            这可避免改动项目环境，并使可选依赖保持隔离。TensorRT、OpenVINO、NCNN 和 CoreML 等特定厂商的 extra 仍可能需要平台专属的原生包。
          </P>

          <Divider />

          {/* ────────────── QUICKSTART ────────────── */}
          <SectionHeading id="quickstart" icon={Rocket}>快速开始</SectionHeading>
          <P>
            若要使用经过最充分测试的路径，请选择单 GPU 的 YOLO9 检测、RF-DETR 检测或 RF-DETR 分割。它们通过同一个工厂函数加载，接受相同的输入，并返回相同的 <InlineCode>Results</InlineCode> 对象，因此你可以在它们之间切换而无需改动周边代码。
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

          <SubHeading>RF-DETR - Transformer 旗舰</SubHeading>
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
            LibreYOLO 在 v1.2.0 中提供一小部分经过验证的范围，外加更广泛的受支持模型目录。每个模型都通过同一个 <InlineCode>LibreYOLO()</InlineCode> 工厂函数加载，但只有下方经过验证的路径才应视为经过深度测试。
          </P>

          <ValidatedModelHeaderZh title="YOLO9 - CNN 旗舰">
            <SupportBadge variant="validated">默认：LibreYOLO9c.pt</SupportBadge>
            <SupportBadge variant="validated">深度测试：检测、训练与推理</SupportBadge>
            <SupportBadge>实验性：分割、多 GPU</SupportBadge>
          </ValidatedModelHeaderZh>
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
            <SupportBadge>实验性</SupportBadge>{' '}
            <strong className="text-surface-800 dark:text-white">分割检查点：</strong>{' '}
            <Checkpoints names={['LibreYOLO9t-seg.pt', 'LibreYOLO9s-seg.pt', 'LibreYOLO9m-seg.pt', 'LibreYOLO9c-seg.pt']} />
            。参见 <a href="#segmentation" className="text-libre-600 dark:text-libre-400 hover:underline">分割</a> 章节。
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")
# Experimental segmentation variant
# model = LibreYOLO("LibreYOLO9c-seg.pt")`}</CodeBlock>

          <ValidatedModelHeaderZh title="RF-DETR - Transformer 旗舰">
            <SupportBadge variant="validated">推荐的 Transformer 路径</SupportBadge>
            <SupportBadge variant="validated">深度测试：检测、分割、训练与推理</SupportBadge>
            <SupportBadge>实验性：多 GPU</SupportBadge>
          </ValidatedModelHeaderZh>
          <DocTable
            headers={['尺寸', '代号', '输入尺寸', '适用场景', '检测检查点']}
            rows={[
              ['Nano', <InlineCode key="n">&quot;n&quot;</InlineCode>, '384', '边缘设备', <HFLink key="cp-n" name="LibreRFDETRn.pt" />],
              ['Small', <InlineCode key="s">&quot;s&quot;</InlineCode>, '512', '均衡', <HFLink key="cp-s" name="LibreRFDETRs.pt" />],
              ['Medium', <InlineCode key="m">&quot;m&quot;</InlineCode>, '576', '更高精度', <HFLink key="cp-m" name="LibreRFDETRm.pt" />],
              ['Large', <InlineCode key="l">&quot;l&quot;</InlineCode>, '704', '最高精度', <HFLink key="cp-l" name="LibreRFDETRl.pt" />],
            ]}
          />
          <P>
            <SupportBadge variant="validated">深度测试</SupportBadge>{' '}
            <strong className="text-surface-800 dark:text-white">分割检查点：</strong>{' '}
            <Checkpoints names={['LibreRFDETRn-seg.pt', 'LibreRFDETRs-seg.pt', 'LibreRFDETRm-seg.pt', 'LibreRFDETRl-seg.pt', 'LibreRFDETRx-seg.pt', 'LibreRFDETRxx-seg.pt']} />
            。参见 <a href="#segmentation" className="text-libre-600 dark:text-libre-400 hover:underline">分割</a> 章节。
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreRFDETRs.pt")
# Segmentation variants exist for every RF-DETR size
# model = LibreYOLO("LibreRFDETRs-seg.pt")`}</CodeBlock>

          <SubHeading>其他受支持的系列</SubHeading>
          <P>
            具备检测能力的系列，与经过验证的路径共享同一个工厂函数和 API。它们在 v1.2.0 中属实验性。每个检查点名称都链接到 <a href="https://huggingface.co/LibreYOLO" target="_blank" rel="noopener noreferrer" className="text-libre-600 dark:text-libre-400 hover:underline">LibreYOLO 组织</a> 在 Hugging Face 上的模型卡；将任意名称传给 <InlineCode>LibreYOLO()</InlineCode>，工厂函数会在首次使用时自动获取。
          </P>
          <DocTable
            headers={['系列', '状态', '任务', '检查点']}
            rows={[
              ['YOLOX', <SupportBadge>实验性</SupportBadge>, 'detect', <Checkpoints key="yolox" names={['LibreYOLOXn.pt', 'LibreYOLOXt.pt', 'LibreYOLOXs.pt', 'LibreYOLOXm.pt', 'LibreYOLOXl.pt', 'LibreYOLOXx.pt']} />],
              ['YOLO9-E2E', <SupportBadge>实验性</SupportBadge>, 'detect', <Checkpoints key="y9e2e" names={['LibreYOLO9E2Et.pt', 'LibreYOLO9E2Es.pt', 'LibreYOLO9E2Em.pt', 'LibreYOLO9E2Ec.pt']} />],
              ['YOLO-NAS', <SupportBadge>实验性</SupportBadge>, 'detect, pose', <Checkpoints key="ynas" link={false} names={['LibreYOLONASs.pt', 'LibreYOLONASm.pt', 'LibreYOLONASl.pt', 'LibreYOLONASn-pose.pt', 'LibreYOLONASs-pose.pt', 'LibreYOLONASm-pose.pt', 'LibreYOLONASl-pose.pt']} />],
              ['D-FINE', <SupportBadge>实验性</SupportBadge>, 'detect', <Checkpoints key="dfine" names={['LibreDFINEn.pt', 'LibreDFINEs.pt', 'LibreDFINEm.pt', 'LibreDFINEl.pt', 'LibreDFINEx.pt']} />],
              ['DEIM', <SupportBadge>实验性</SupportBadge>, 'detect', <Checkpoints key="deim" names={['LibreDEIMn.pt', 'LibreDEIMs.pt', 'LibreDEIMm.pt', 'LibreDEIMl.pt', 'LibreDEIMx.pt']} />],
              ['DEIMv2', <SupportBadge>实验性</SupportBadge>, 'detect', <Checkpoints key="deimv2" names={['LibreDEIMv2atto.pt', 'LibreDEIMv2femto.pt', 'LibreDEIMv2pico.pt', 'LibreDEIMv2n.pt', 'LibreDEIMv2s.pt', 'LibreDEIMv2m.pt', 'LibreDEIMv2l.pt', 'LibreDEIMv2x.pt']} />],
              ['RT-DETR', <SupportBadge>实验性</SupportBadge>, 'detect', <Checkpoints key="rtdetr" names={['LibreRTDETRr18.pt', 'LibreRTDETRr34.pt', 'LibreRTDETRr50.pt', 'LibreRTDETRr50m.pt', 'LibreRTDETRr101.pt', 'LibreRTDETRl.pt', 'LibreRTDETRx.pt']} />],
              ['RT-DETRv2', <SupportBadge>实验性</SupportBadge>, 'detect', <Checkpoints key="rtdetrv2" names={['LibreRTDETRv2r18.pt', 'LibreRTDETRv2r34.pt', 'LibreRTDETRv2r50.pt', 'LibreRTDETRv2r50m.pt', 'LibreRTDETRv2r101.pt']} />],
              ['RT-DETRv4', <SupportBadge>实验性</SupportBadge>, 'detect', <Checkpoints key="rtdetrv4" names={['LibreRTDETRv4s.pt', 'LibreRTDETRv4m.pt', 'LibreRTDETRv4l.pt', 'LibreRTDETRv4x.pt']} />],
              ['PicoDet', <SupportBadge>实验性</SupportBadge>, 'detect', <Checkpoints key="picodet" names={['LibrePICODETs.pt', 'LibrePICODETm.pt', 'LibrePICODETl.pt']} />],
              ['EdgeCrafter', <SupportBadge>实验性</SupportBadge>, 'detect, pose, segment', <Checkpoints key="ec" names={['LibreECs.pt', 'LibreECm.pt', 'LibreECl.pt', 'LibreECx.pt', 'LibreECs-pose.pt', 'LibreECm-pose.pt', 'LibreECl-pose.pt', 'LibreECx-pose.pt', 'LibreECs-seg.pt', 'LibreECm-seg.pt', 'LibreECl-seg.pt', 'LibreECx-seg.pt']} />],
              ['DAMO-YOLO', <SupportBadge>实验性</SupportBadge>, 'detect', <Checkpoints key="damo" names={['LibreDAMOYOLOns.pt', 'LibreDAMOYOLOnm.pt', 'LibreDAMOYOLOnl.pt', 'LibreDAMOYOLOt.pt', 'LibreDAMOYOLOs.pt', 'LibreDAMOYOLOm.pt', 'LibreDAMOYOLOl.pt']} />],
              ['RTMDet', <SupportBadge>实验性</SupportBadge>, 'detect', <Checkpoints key="rtmdet" names={['LibreRTMDett.pt', 'LibreRTMDets.pt', 'LibreRTMDetm.pt', 'LibreRTMDetl.pt', 'LibreRTMDetx.pt']} />],
            ]}
          />
          <P className="text-sm">
            <strong className="text-surface-800 dark:text-white">托管说明：</strong> YOLO-NAS 检查点（上方以纯文本显示）托管在 Deci 的 CDN 上，遵循其专有权重许可证，而非 LibreYOLO 的 Hugging Face 组织。工厂函数仍会在首次使用时自动下载它们。
          </P>

          <SubHeading>专用模型</SubHeading>
          <DocTable
            headers={['系列', '状态', '任务', '检查点']}
            rows={[
              ['L2CS', <SupportBadge>实验性</SupportBadge>, <span key="t">gaze（仅推理） - 参见 <a href="#gaze" className="text-libre-600 dark:text-libre-400 hover:underline">视线估计</a></span>, <Checkpoints key="l2cs" link={false} names={['LibreL2CSr50.pt']} />],
            ]}
          />
          <P className="text-sm">
            L2CS 架构尺寸包括 r18、r34、r50、r101 和 r152，但上游发布的 Gaze360 检查点是 ResNet-50。安装 <InlineCode>libreyolo[gaze]</InlineCode> 以使用可选的 Google Drive 辅助下载，或为其他尺寸传入本地检查点路径。
          </P>

          <SubHeading>工厂函数</SubHeading>
          <P>
            对所有模型和运行时都使用 <InlineCode>LibreYOLO()</InlineCode> 工厂函数。传入官方检查点名称或导出产物路径，让它自动选择正确的模型系列、任务、类别数和运行时：
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

# Default: YOLO9 detection
model = LibreYOLO("LibreYOLO9c.pt")

# Flagship: RF-DETR
model = LibreYOLO("LibreRFDETRs.pt")

# Segmentation checkpoints use the same factory path
model = LibreYOLO("LibreRFDETRs-seg.pt")       # validated segmentation
model = LibreYOLO("LibreYOLO9c-seg.pt")        # experimental segmentation

# Exported deployment formats
model = LibreYOLO("model.onnx")                # ONNX Runtime
model = LibreYOLO("model.engine")              # TensorRT
model = LibreYOLO("model.mlpackage")           # CoreML (macOS)
model = LibreYOLO("model_openvino/")           # OpenVINO (directory)
model = LibreYOLO("model_ncnn/")               # NCNN (directory)`}</CodeBlock>
          <P>
            对于可识别的官方检查点文件名，LibreYOLO 可自动下载缺失的权重。对于自定义文件名，请指向明确的本地路径。实验性系列仍通过同一个工厂函数加载，但新项目请坚持使用 YOLO9 检测或 RF-DETR 检测/分割。如有特定理由再使用它们。
          </P>

          <Divider />

          {/* ────────────── TASKS & FILENAMES ────────────── */}
          <SectionHeading id="tasks" icon={Tags}>任务与文件名</SectionHeading>
          <P>
            LibreYOLO 使用统一的文件名约定，使工厂函数仅凭检查点名称即可识别系列、尺寸和任务：
          </P>
          <CodeBlock language="text">{`Libre<FAMILY><size>[-<task>].pt`}</CodeBlock>

          <SubHeading>任务后缀</SubHeading>
          <DocTable
            headers={['任务', '规范名称', '文件名后缀']}
            rows={[
              ['检测', <InlineCode key="d">&quot;detect&quot;</InlineCode>, '（无 - 隐式）'],
              ['实例分割', <InlineCode key="s">&quot;segment&quot;</InlineCode>, <InlineCode key="ss">-seg</InlineCode>],
              ['姿态估计', <InlineCode key="p">&quot;pose&quot;</InlineCode>, <InlineCode key="ps">-pose</InlineCode>],
              ['分类', <InlineCode key="c">&quot;classify&quot;</InlineCode>, <InlineCode key="cs">-cls</InlineCode>],
              ['视线估计', <InlineCode key="g">&quot;gaze&quot;</InlineCode>, <InlineCode key="gs">-gaze</InlineCode>],
            ]}
          />
          <P>
            工厂函数在 API 边界接受别名（<InlineCode>&quot;detection&quot;</InlineCode>、<InlineCode>&quot;seg&quot;</InlineCode>、<InlineCode>&quot;keypoints&quot;</InlineCode> 等） - 但文件名中只出现规范名称。
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
            headers={['系列', 'v1.2.0 状态', '默认', '支持的任务']}
            rows={[
              [<strong key="y9">YOLO9</strong>, 'detect 单 GPU 深度测试；segment 与多 GPU 实验性', 'detect', 'detect, segment'],
              [<strong key="rfd">RF-DETR</strong>, 'detect 与 segment 单 GPU 深度测试；多 GPU 实验性', 'detect', 'detect, segment'],
              ['YOLOX', '实验性', 'detect', 'detect'],
              ['YOLO9-E2E', '实验性', 'detect', 'detect'],
              ['YOLO-NAS', '实验性', 'detect', 'detect, pose'],
              ['D-FINE / DEIM / DEIMv2', '实验性', 'detect', 'detect'],
              ['RT-DETR / RT-DETRv2 / RT-DETRv4', '实验性', 'detect', 'detect'],
              ['PicoDet', '实验性', 'detect', 'detect'],
              ['EdgeCrafter (EC)', '实验性', 'detect', 'detect, pose, segment'],
              ['DAMO-YOLO / RTMDet', '实验性', 'detect', 'detect'],
              ['L2CS', '实验性', 'gaze', 'gaze（仅推理）'],
            ]}
          />

          <SubHeading>示例</SubHeading>
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

          <SubHeading>已弃用的别名</SubHeading>
          <P>
            <InlineCode>LibreYOLORTDETR</InlineCode> 和 <InlineCode>LibreYOLORFDETR</InlineCode> 分别是 <InlineCode>LibreRTDETR</InlineCode> 和 <InlineCode>LibreRFDETR</InlineCode> 的旧名称。它们仍可解析，但会触发 <InlineCode>DeprecationWarning</InlineCode> - 请在方便时更新导入。
          </P>

          <Divider />

          {/* ────────────── PREDICTION ────────────── */}
          <SectionHeading id="prediction" icon={Crosshair}>预测</SectionHeading>
          <P>
            单 GPU 预测路径针对 YOLO9 检测、RF-DETR 检测和 RF-DETR 分割经过深度测试。其他系列和任务使用相同的 API，但在 v1.2.0 中属实验性。
          </P>

          <SubHeading>基本预测</SubHeading>
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
            每次预测都会返回一个 <InlineCode>Results</InlineCode> 对象（处理目录时返回其列表）：
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
          <P>将检测结果过滤为特定类别 ID：</P>
          <CodeBlock language="python">{`# Only detect people (class 0) and cars (class 2)
result = model("image.jpg", classes=[0, 2])`}</CodeBlock>

          <Divider />

          {/* ────────────── TILED INFERENCE ────────────── */}
          <SectionHeading id="tiled-inference" icon={Grid3x3}>分块推理</SectionHeading>
          <P>
            对于远大于模型输入尺寸的图像（如卫星影像、无人机航拍），分块推理会将图像切分为相互重叠的小块，分别在每块上运行检测，再合并结果。
          </P>
          <P>
            在 v1.2.0 dev 中分块仅支持检测。它不接受分割掩码，且无法与 <InlineCode>augment=True</InlineCode> 组合使用。
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
            <FeatureItem><InlineCode>final_image.jpg</InlineCode> - 绘制了全部合并检测结果的完整图像</FeatureItem>
            <FeatureItem><InlineCode>grid_visualization.jpg</InlineCode> - 显示分块网格叠加的图像</FeatureItem>
            <FeatureItem><InlineCode>tiles/</InlineCode> - 各个分块裁剪图</FeatureItem>
            <FeatureItem><InlineCode>metadata.json</InlineCode> - 分块参数和检测数量</FeatureItem>
          </ul>
          <P>
            如果图像本身已小于模型的输入尺寸，则会自动跳过分块。
          </P>

          <Divider />

          {/* ────────────── VIDEO INFERENCE ────────────── */}
          <SectionHeading id="video-inference" icon={Video}>视频推理</SectionHeading>
          <P>
            将任意视频文件传给旗舰模型，LibreYOLO 会根据扩展名自动识别格式。支持：<InlineCode>.mp4</InlineCode>、<InlineCode>.avi</InlineCode>、<InlineCode>.mov</InlineCode>、<InlineCode>.mkv</InlineCode>、<InlineCode>.webm</InlineCode>、<InlineCode>.gif</InlineCode> 以及其他常见容器格式。
          </P>

          <SubHeading>保存标注后的视频</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")
results = model("clip.mp4", save=True)
# Saved under runs/detect/predict*/clip.mp4`}</CodeBlock>

          <SubHeading>流式结果（内存恒定）</SubHeading>
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
            当你需要完全掌控解码和编码时 - 自定义帧变换、混入跟踪器输出、写入非默认编解码器 - 可直接使用这些构建模块：
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
            LibreYOLO 内置 ByteTrack 多目标跟踪器，可消费任意检测器输出的 <InlineCode>Results</InlineCode> 并添加持久的跟踪 ID。它在单 GPU 的 YOLO9 检测和 RF-DETR 检测上测试最充分；其他检测系列在 v1.2.0 中属实验性。
          </P>

          <SubHeading>安装</SubHeading>
          <CodeBlock language="bash">{`pip install libreyolo[tracking]   # compatibility extra; tracking deps ship in base dev install`}</CodeBlock>

          <SubHeading>视频跟踪辅助函数</SubHeading>
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

          <SubHeading>基本循环</SubHeading>
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
            调用 <InlineCode>tracker.update()</InlineCode> 后，<InlineCode>result.boxes.id</InlineCode> 保存跟踪 ID，且 <InlineCode>result.boxes.is_track</InlineCode> 为 <InlineCode>True</InlineCode>。
          </P>

          <SubHeading>TrackConfig 可调参数</SubHeading>
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

          <Divider />

          {/* ────────────── SEGMENTATION ────────────── */}
          <SectionHeading id="segmentation" icon={Scissors}>分割</SectionHeading>
          <ValidationScopeCalloutZh />
          <P>
            RF-DETR 分割是 v1.2.0 中经过深度测试的分割路径。YOLO9 分割和 EdgeCrafter 分割可通过相同的 <InlineCode>-seg</InlineCode> 后缀使用，但属实验性。
          </P>

          <SubHeading>运行分割</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

# RF-DETR segmentation, heavily tested on single GPU
model = LibreYOLO("LibreRFDETRs-seg.pt")
result = model("photo.jpg")

# YOLO9 segmentation is available but experimental in v1.2.0
# model = LibreYOLO("LibreYOLO9c-seg.pt")

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
            <InlineCode>save=True</InlineCode> 会自动绘制边界框和半透明掩码叠加层。
          </P>
          <CodeBlock language="python">{`model("photo.jpg", save=True)`}</CodeBlock>

          <SubHeading>训练分割模型</SubHeading>
          <P>
            RF-DETR 分割使用 RF-DETR 的 COCO 格式训练流水线，属于经过深度测试的单 GPU 范围。YOLO9 分割和 EdgeCrafter 分割训练可用，但在 v1.2.0 中属实验性。
          </P>

          <Divider />

          {/* ────────────── POSE ESTIMATION ────────────── */}
          <SectionHeading id="pose" icon={PersonStanding}>姿态估计</SectionHeading>
          <P>
            姿态（人体关键点）估计在 <InlineCode>YOLO-NAS (-pose)</InlineCode> 和 <InlineCode>EdgeCrafter (-pose)</InlineCode> 上受支持。每个姿态模型均为单类别（&quot;person&quot;），包含 17 个 COCO 关键点。
          </P>

          <SubHeading>运行姿态估计</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

# YOLO-NAS pose
model = LibreYOLO("LibreYOLONASs-pose.pt")
result = model("people.jpg")

# EdgeCrafter pose
# model = LibreYOLO("LibreECs-pose.pt")

# Per-person bbox + 17 keypoints
print(result.boxes.xyxy)          # person boxes (N, 4)
print(result.keypoints.xy.shape)  # (N, 17, 2) pixel coordinates`}</CodeBlock>

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
            YOLO-NAS 支持姿态训练；EdgeCrafter 姿态目前仅支持推理。YOLO9 和 RF-DETR 尚未提供姿态检查点。
          </P>

          <Divider />

          {/* ────────────── GAZE ESTIMATION ────────────── */}
          <SectionHeading id="gaze" icon={Eye}>视线估计</SectionHeading>
          <P>
            视线方向估计由 <InlineCode>LibreL2CS</InlineCode> 系列提供，它是 L2CS-Net 的移植版，采用 ResNet 主干和两个角度分箱分类头。这是一个两阶段模型：上游的人脸检测器先定位人脸，然后视线头以弧度预测每张人脸的俯仰角（pitch）和偏航角（yaw）。它仅支持推理，且在 v1.2.0 中属实验性。
          </P>

          <SubHeading>安装</SubHeading>
          <CodeBlock language="bash">{`pip install libreyolo[gaze]   # optional Google Drive helper for Gaze360 weights`}</CodeBlock>
          <P>
            已发布的 L2CS ResNet-50 权重在 Gaze360 上训练，LibreYOLO 不进行镜像。若没有可选辅助工具，请传入本地检查点路径，或按照 <InlineCode>LibreL2CS</InlineCode> 打印的手动下载说明操作。
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
            通过命令行：<InlineCode>libreyolo predict model=LibreL2CSr50.pt source=portrait.jpg --face-detector path/to/face.pt</InlineCode>。
          </P>

          <Divider />

          <SectionHeading id="training" icon={GraduationCap}>训练</SectionHeading>
          <ValidationScopeCalloutZh />
          <P>
            经过深度测试的训练路径是单 GPU 的 YOLO9 检测、RF-DETR 检测和 RF-DETR 分割。其他模型系列的训练器、YOLO9 分割训练和多 GPU 工作流可用，但在 v1.2.0 中属实验性。
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
)

print(f"Best mAP50-95: {results['best_mAP50_95']:.3f}")
print(f"Best checkpoint: {results['best_checkpoint']}")`}</CodeBlock>
          <P>
            训练完成后，模型实例会自动以最佳权重重新加载，因此你可以立即调用 <InlineCode>model(...)</InlineCode>。YOLO9 分割训练可通过 <InlineCode>LibreYOLO(&quot;LibreYOLO9c-seg.pt&quot;)</InlineCode> 使用，但在 v1.2.0 中属实验性。
          </P>

          <SubHeading>RF-DETR - Transformer 旗舰训练</SubHeading>
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
            RF-DETR 有自己的训练签名（<InlineCode>batch_size</InlineCode>、<InlineCode>lr</InlineCode>、<InlineCode>output_dir</InlineCode>），但它使用 LibreYOLO 的数据集配置加载器。为检测或分割传入 <InlineCode>data.yaml</InlineCode>；该配置可引用 COCO/Roboflow 风格的标注布局。
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
            其他系列也有训练器钩子，但它们不是 v1.2.0 中的推荐路径。新工作请坚持使用 YOLO9 检测或 RF-DETR 检测/分割；仅在兼容、复现基准或定向研究时使用实验性训练器。DAMO-YOLO、PicoDet、RTMDet 和 EC 的训练需要显式的 <InlineCode>allow_experimental=True</InlineCode> 确认。
          </P>

          <SubHeading>通过 YAML 配置训练</SubHeading>
          <P>
            每个 <InlineCode>model.train(...)</InlineCode> 都接受 <InlineCode>cfg=&quot;train.yaml&quot;</InlineCode>，以从文件加载全部参数。显式关键字参数仍优先于 yaml 中的值，因此你可以用 yaml 作为基线，并在每次运行时覆盖单个字段。
          </P>
          <CodeBlock language="python">{`model = LibreYOLO("LibreYOLO9c.pt")
results = model.train(cfg="configs/yolo9_finetune.yaml")
# Override individual fields:
# results = model.train(cfg="configs/yolo9_finetune.yaml", epochs=50)`}</CodeBlock>

          <SubHeading>梯度累积</SubHeading>
          <P>
            传入 <InlineCode>nbs</InlineCode>（名义批量大小）以启用梯度累积。训练器每 <InlineCode>nbs / batch</InlineCode> 次前向传播执行一次优化器步进，使你能在较小的硬件上以配方的参考批量大小进行训练。
          </P>
          <CodeBlock language="python">{`# Effective batch 64 on a single GPU that only fits batch=8
model.train(data="coco128.yaml", batch=8, nbs=64)`}</CodeBlock>

          <SubHeading>分布式训练（DDP，实验性）</SubHeading>
          <P>
            YOLO9 和 RF-DETR 通过 PyTorch DistributedDataParallel 支持多 GPU 训练，但多 GPU 不在 v1.2.0 的深度测试范围内。使用 <InlineCode>torchrun</InlineCode> 启动训练脚本：
          </P>
          <CodeBlock language="bash">{`# 4-GPU node
torchrun --nproc_per_node=4 train_yolo9.py

# Multi-node - see PyTorch's torchrun docs for --nnodes / --rdzv_endpoint`}</CodeBlock>
          <CodeBlock language="python" filename="train_yolo9.py">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")
# Pass device="" (auto-detect) and let torchrun set the rank
model.train(data="coco128.yaml", epochs=300, batch=16)`}</CodeBlock>

          <Divider />

          {/* ────────────── VALIDATION ────────────── */}
          <SectionHeading id="validation" icon={CheckCircle2}>验证</SectionHeading>
          <P>
            在验证集上运行 COCO 标准评估。经过深度测试的验证路径是单 GPU 的 YOLO9 检测、RF-DETR 检测和 RF-DETR 分割。
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
            分割验证返回带 <InlineCode>(M)</InlineCode> 后缀的掩码指标，以及带 <InlineCode>(B)</InlineCode> 后缀的边界框指标。姿态验证通过 <InlineCode>PoseValidator</InlineCode> 返回 COCO 关键点指标。
          </P>

          <Divider />

          {/* ────────────── EXPORT ────────────── */}
          <SectionHeading id="export" icon={Upload}>导出</SectionHeading>
          <P>
            将 PyTorch 模型导出为 ONNX、TorchScript、TensorRT、OpenVINO、NCNN 或 CoreML 以便部署。经过深度测试的导出与运行时后端路径是单 GPU 的 YOLO9 检测、RF-DETR 检测和 RF-DETR 分割。其他系列和任务属实验性。
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
model.export(format="coreml")`}</CodeBlock>

          <SubHeading>全部导出参数</SubHeading>
          <CodeBlock language="python">{`path = model.export(
    format="onnx",            # "onnx", "torchscript", "tensorrt", "openvino", "ncnn", or "coreml"
    output_path="model.onnx", # output file (auto-generated if None)
    imgsz=640,                # input resolution (default: model's native)
    opset=None,               # ONNX opset (auto: 13, or 17 for wrappers that need it)
    simplify=True,            # run onnxsim graph simplification
    dynamic=True,             # enable dynamic batch axis
    half=False,               # export in FP16
    batch=1,                  # batch size for static graph
    device=None,              # device to trace on (default: model's current device)
    int8=False,               # INT8 quantization (TensorRT / OpenVINO only)
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
    nms=False,                # CoreML embedded NMS where supported
    iou=0.45,                 # CoreML embedded NMS IoU threshold
    conf=0.25,                # CoreML embedded NMS confidence threshold
    verbose=False,            # verbose logging
)`}</CodeBlock>
          <P>
            OpenVINO INT8 导出还需要 <InlineCode>nncf</InlineCode>。NCNN 导出会写出一个目录，包含 <InlineCode>model.ncnn.param</InlineCode>、<InlineCode>model.ncnn.bin</InlineCode> 和 <InlineCode>metadata.yaml</InlineCode>。CoreML 导出会写出 <InlineCode>.mlpackage</InlineCode> 包，需要 <InlineCode>coremltools</InlineCode>，且不支持 INT8。
          </P>

          <SubHeading>ONNX 元数据</SubHeading>
          <P>导出的 ONNX 文件包含内嵌元数据：</P>
          <DocTable
            headers={['键', '示例值']}
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
            使用 <InlineCode>LibreYOLO(&quot;model.onnx&quot;)</InlineCode> 加载导出文件时，会自动读回此元数据。
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
            对于没有元数据的 ONNX 文件（如由其他工具导出），请手动指定 <InlineCode>nb_classes</InlineCode>：
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
            运行时后端不暴露仅限 PyTorch 的选项，如 <InlineCode>tiling</InlineCode>、<InlineCode>overlap_ratio</InlineCode> 或 <InlineCode>output_file_format</InlineCode>。
          </P>
          <P>
            运行时后端的保存方式与 PyTorch 封装略有不同：如果设置 <InlineCode>output_path</InlineCode>，请传入最终文件路径，而非目录。如果省略它，当前后端默认保存在 <InlineCode>runs/detections/</InlineCode> 下。
          </P>

          <Divider />

          {/* ────────────── TENSORRT INFERENCE ────────────── */}
          <SectionHeading id="tensorrt-inference" icon={Cpu}>TensorRT 推理</SectionHeading>
          <P>
            使用 TensorRT 在 NVIDIA GPU 上以最大吞吐量运行推理。需要 CUDA 以及 TensorRT 的 Python 绑定。
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
            使用 OpenVINO 运行推理，针对 Intel CPU、GPU 和 VPU 进行了优化。
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
            使用 NCNN 运行推理，以在 CPU 或支持 Vulkan 的 GPU 目标上实现轻量化部署。
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
          <SectionHeading id="cli" icon={SquareTerminal}>命令行工具</SectionHeading>
          <P>
            安装 LibreYOLO 会在你的 PATH 中注册一个 <InlineCode>libreyolo</InlineCode> 命令（入口点在 <InlineCode>pyproject.toml</InlineCode> 中）。该命令行镜像 Python API，并遵循 Ultralytics 风格的 <InlineCode>key=value</InlineCode> 语法。
          </P>

          <SubHeading>子命令</SubHeading>
          <DocTable
            headers={['命令', '用途']}
            rows={[
              [<InlineCode key="p">predict</InlineCode>, '对图像、目录或视频运行推理'],
              [<InlineCode key="t">train</InlineCode>, '在数据集上训练模型'],
              [<InlineCode key="v">val</InlineCode>, '在数据集上评估模型'],
              [<InlineCode key="e">export</InlineCode>, '导出为 ONNX / TorchScript / TensorRT / OpenVINO / NCNN / CoreML'],
              [<InlineCode key="c">checks</InlineCode>, '打印 Python、torch、CUDA、GPU 及可选包信息'],
              [<InlineCode key="m">models</InlineCode>, '列出已注册的模型系列和 CLI 快捷名称'],
              [<InlineCode key="f">formats</InlineCode>, '列出支持的导出格式'],
              [<InlineCode key="cfg">cfg</InlineCode>, '打印默认的训练配置 YAML'],
              [<InlineCode key="i">info</InlineCode>, '加载模型并打印解析出的系列、尺寸、任务、设备和类别'],
              [<InlineCode key="md">metadata</InlineCode>, '检视 .pt 文件中的原始检查点元数据'],
              [<InlineCode key="ver">version</InlineCode>, '打印 LibreYOLO + Python + torch 版本'],
            ]}
          />

          <SubHeading>模型名称快捷方式</SubHeading>
          <P>
            命令行接受短名称（<InlineCode>yolo9-c</InlineCode>），它们会解析为权重文件名（<InlineCode>LibreYOLO9c.pt</InlineCode>） - 可通过 <InlineCode>libreyolo models</InlineCode> 查看。你也可以传入任意明确的检查点路径。
          </P>

          <SubHeading>常用选项</SubHeading>
          <DocTable
            headers={['命令', '重要选项']}
            rows={[
              [<InlineCode key="p">predict</InlineCode>, <span key="pv"><InlineCode>conf</InlineCode>, <InlineCode>iou</InlineCode>, <InlineCode>imgsz</InlineCode>, <InlineCode>classes</InlineCode>, <InlineCode>max_det</InlineCode>, <InlineCode>half</InlineCode>, <InlineCode>batch</InlineCode>, <InlineCode>tiling</InlineCode>, <InlineCode>overlap_ratio</InlineCode>, <InlineCode>output_file_format</InlineCode>, <InlineCode>project</InlineCode>, <InlineCode>name</InlineCode>, <InlineCode>exist_ok</InlineCode>, <InlineCode>face_detector</InlineCode></span>],
              [<InlineCode key="t">train</InlineCode>, <span key="tv"><InlineCode>epochs</InlineCode>, <InlineCode>batch</InlineCode>, <InlineCode>imgsz</InlineCode>, <InlineCode>lr0</InlineCode>, <InlineCode>optimizer</InlineCode>, <InlineCode>scheduler</InlineCode>, <InlineCode>workers</InlineCode>, <InlineCode>seed</InlineCode>, <InlineCode>resume</InlineCode>, <InlineCode>amp</InlineCode>, <InlineCode>allow_download_scripts</InlineCode>, <InlineCode>dry_run</InlineCode></span>],
              [<InlineCode key="v">val</InlineCode>, <span key="vv"><InlineCode>split</InlineCode>, <InlineCode>batch</InlineCode>, <InlineCode>imgsz</InlineCode>, <InlineCode>conf</InlineCode>, <InlineCode>iou</InlineCode>, <InlineCode>max_det</InlineCode>, <InlineCode>half</InlineCode>, <InlineCode>data_dir</InlineCode>, <InlineCode>use_coco_eval</InlineCode>, <InlineCode>project</InlineCode>, <InlineCode>name</InlineCode>, <InlineCode>exist_ok</InlineCode>, <InlineCode>save_json</InlineCode>, <InlineCode>allow_download_scripts</InlineCode></span>],
              [<InlineCode key="e">export</InlineCode>, <span key="ev"><InlineCode>format</InlineCode>, <InlineCode>imgsz</InlineCode>, <InlineCode>batch</InlineCode>, <InlineCode>half</InlineCode>, <InlineCode>int8</InlineCode>, <InlineCode>dynamic</InlineCode>, <InlineCode>simplify</InlineCode>, <InlineCode>opset</InlineCode>, <InlineCode>data</InlineCode>, <InlineCode>fraction</InlineCode>, <InlineCode>device</InlineCode>, <InlineCode>allow_download_scripts</InlineCode>, <InlineCode>verbose</InlineCode></span>],
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

          <SubHeading>机器可读输出</SubHeading>
          <P>
            每个命令都接受 <InlineCode>--json</InlineCode>（结构化的标准输出，便于管道传入脚本或智能体）和 <InlineCode>--quiet</InlineCode>（抑制 stderr 进度行）。核心的 <InlineCode>predict</InlineCode>、<InlineCode>train</InlineCode>、<InlineCode>val</InlineCode> 和 <InlineCode>export</InlineCode> 命令还接受 <InlineCode>--help-json</InlineCode>，可将其参数 schema 以 JSON 形式导出。
          </P>
          <CodeBlock language="bash">{`libreyolo predict model=yolo9-c source=img.jpg --json | jq .

libreyolo train --help-json > train_schema.json`}</CodeBlock>

          <Divider />

          {/* ────────────── API REFERENCE ────────────── */}
          <SectionHeading id="api-reference" icon={FileCode}>API 参考</SectionHeading>

          <SubHeading>LibreYOLO（工厂函数）</SubHeading>
          <CodeBlock language="python">{`LibreYOLO(
    model_path: str,
    *,
    device: str = "auto",
    task: str | None = None,    # override only when a custom artifact is ambiguous
    nb_classes: int | None = None,  # mainly for external exported artifacts
    compute_units: str = "all", # CoreML only: all, cpu_only, cpu_and_gpu, cpu_and_ne
) -> model wrapper or runtime backend`}</CodeBlock>
          <P>
            优先使用官方检查点文件名和导出产物路径，再让工厂函数解析细节。它可处理 PyTorch 检查点、<InlineCode>.onnx</InlineCode>、<InlineCode>.torchscript</InlineCode>、<InlineCode>.engine</InlineCode>、<InlineCode>.tensorrt</InlineCode>、<InlineCode>.mlpackage</InlineCode>、包含 <InlineCode>model.xml</InlineCode> 的 OpenVINO 目录，以及包含 <InlineCode>model.ncnn.param</InlineCode> 和 <InlineCode>model.ncnn.bin</InlineCode> 的 NCNN 目录。<InlineCode>task</InlineCode> 参数用于含糊的自定义产物；否则解析来自检查点元数据、文件名后缀和系列默认值。
          </P>

          <SubHeading>预测（PyTorch 模型封装）</SubHeading>
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
            如果运行时后端省略 <InlineCode>output_path</InlineCode>，当前默认保存位置为 <InlineCode>runs/detections/</InlineCode>。
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

          <SubHeading>任务负载</SubHeading>
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
          <P>返回标准的 LibreYOLO 训练字典，包含 <InlineCode>final_loss</InlineCode>、<InlineCode>best_mAP50</InlineCode>、<InlineCode>best_mAP50_95</InlineCode>、<InlineCode>best_epoch</InlineCode>、<InlineCode>save_dir</InlineCode>、<InlineCode>best_checkpoint</InlineCode> 和 <InlineCode>last_checkpoint</InlineCode>。</P>

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
            YOLO-NAS、D-FINE、DEIM、DEIMv2、EC、PicoDet、DAMO-YOLO、RT-DETRv2/v4 和 RTMDet 还提供额外的实验性训练器。它们遵循相同的 <InlineCode>model.train(data=&quot;...yaml&quot;, ...)</InlineCode> 形式，但其默认值和实验性门控因系列而异。
          </P>

          <SubHeading>运行时产物加载</SubHeading>
          <P>
            通过 <InlineCode>LibreYOLO()</InlineCode> 加载导出产物，方式与加载 PyTorch 检查点相同。工厂函数会根据路径选择 ONNX Runtime、TorchScript、TensorRT、OpenVINO、NCNN 或 CoreML：
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("model.onnx")
model = LibreYOLO("model.torchscript")
model = LibreYOLO("model.engine")
model = LibreYOLO("model_openvino/")
model = LibreYOLO("model_ncnn/")
model = LibreYOLO("model.mlpackage", compute_units="all")`}</CodeBlock>
          <P>
            高级集成可以访问更底层的运行时模块，但普通应用代码应坚持使用工厂函数路径。
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
            本节面向希望了解代码库内部实现的贡献者。
          </P>

          <SubHeading>基类设计</SubHeading>
          <P>
            PyTorch 模型系列继承自 <InlineCode>libreyolo/models/base/model.py</InlineCode> 中的 <InlineCode>BaseModel</InlineCode>。子类需实现以下抽象方法：
          </P>
          <DocTable
            headers={['方法', '用途']}
            rows={[
              [<InlineCode key="init">_init_model()</InlineCode>, '构建并返回 nn.Module'],
              [<InlineCode key="layers">_get_available_layers()</InlineCode>, '返回层名称到模块的映射'],
              [<InlineCode key="pre-np">_get_preprocess_numpy()</InlineCode>, '返回用于导出/校准的 NumPy 预处理器'],
              [<InlineCode key="pre">_preprocess()</InlineCode>, '图像到张量的转换'],
              [<InlineCode key="fwd">_forward()</InlineCode>, '模型前向传播'],
              [<InlineCode key="post">_postprocess()</InlineCode>, '原始输出到检测字典'],
            ]}
          />
          <P>
            <InlineCode>BaseModel</InlineCode> 提供共享的封装行为：预测、导出、验证、尺寸/名称元数据和训练辅助函数。实际的单图、批量和分块推理流程位于 <InlineCode>libreyolo/models/base/inference.py</InlineCode>，而部署运行时位于 <InlineCode>libreyolo/backends/</InlineCode> 下。
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
              <>将该导入添加到 <InlineCode>libreyolo/models/__init__.py</InlineCode>；导入运行时会进行子类注册</>,
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
            用户代码应通过 <InlineCode>model.export(...)</InlineCode> 导出。在内部，<InlineCode>libreyolo/export/exporter.py</InlineCode> 中的 <InlineCode>BaseExporter</InlineCode> 持有格式注册表，具体的导出器通过子类注册来注册自身。
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
            训练和验证使用通过 <InlineCode>data.yaml</InlineCode> 加载的数据集配置。检测、分割、姿态和 RF-DETR 训练都经由该加载器；标签文件内容因任务而异。
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
            数据集配置按以下顺序解析：明确路径、当前工作目录，然后是 <InlineCode>libreyolo/config/datasets/</InlineCode> 下的内置配置。数据集根目录默认在 <InlineCode>~/datasets</InlineCode> 下，可通过 <InlineCode>LIBREYOLO_DATASETS_DIR</InlineCode> 覆盖。
          </P>
          <P>
            <InlineCode>train</InlineCode>、<InlineCode>val</InlineCode> 和 <InlineCode>test</InlineCode> 可以是目录、<InlineCode>.txt</InlineCode> 文件或路径列表。YAML 下载钩子受保护；仅对可信配置传入 <InlineCode>allow_download_scripts=True</InlineCode>。
          </P>

          <SubHeading>文件列表变体</SubHeading>
          <P>
            同样的 YAML 格式也可将 <InlineCode>train</InlineCode>、<InlineCode>val</InlineCode> 或 <InlineCode>test</InlineCode> 指向每行一个图像路径的 <InlineCode>.txt</InlineCode> 文件：
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
            每张图像对应一个文本文件。每行表示一个目标：
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
            分割使用 YOLO 多边形行。数据集加载器从多边形顶点推导边界框，并在启用分割加载时保留多边形环：
          </P>
          <CodeBlock language="text">{`<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>`}</CodeBlock>

          <SubHeading>姿态标签格式</SubHeading>
          <P>
            姿态标签在边界框之后追加关键点。在 <InlineCode>data.yaml</InlineCode> 中添加 <InlineCode>kpt_shape</InlineCode> 和 <InlineCode>flip_idx</InlineCode>，使加载器知道关键点数量和水平翻转排列。
          </P>
          <CodeBlock language="yaml">{`kpt_shape: [17, 3]
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]`}</CodeBlock>
          <CodeBlock language="text">{`<class_id> <cx> <cy> <w> <h> <kx1> <ky1> <v1> ... <kxK> <kyK> <vK>`}</CodeBlock>

          <SubHeading>内置数据集</SubHeading>
          <P>
            LibreYOLO 在 <InlineCode>libreyolo/config/datasets/</InlineCode> 下提供内置数据集配置，并可在首次使用时自动下载支持的数据集：
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
  if (locale === 'zh') return <DocsPageZh />
  return <DocsPage />
}
