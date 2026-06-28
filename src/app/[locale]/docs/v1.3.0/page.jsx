'use client'

import { useState, useEffect } from 'react'
import { useLocale, useTranslations } from 'next-intl'
import { motion, AnimatePresence } from 'framer-motion'
import {
  BookOpen, Terminal, Rocket, Layers, Crosshair, Grid3x3,
  GraduationCap, CheckCircle2, Upload, Cpu, FileCode, Wrench,
  Database, Copy, Check, Menu, X, ChevronRight,
  Sparkles, Tags, Video, Activity, Scissors, PersonStanding, Eye, SquareTerminal,
  ShieldCheck, Mountain, MapPin,
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
  { id: 'classification', title: 'Classification', icon: Tags },
  { id: 'depth', title: 'Depth Estimation', icon: Mountain },
  { id: 'point-localization', title: 'Point Localization', icon: MapPin },
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
      family: 'RF-DETR', status: 'Validated detect + segment; pose / OBB preview',
      inference: 'yes', training: 'yes',
      detect: 'yes', segment: 'yes', semantic: '', classify: '', pose: 'preview', obb: 'preview', depth: '', point: '', gaze: '',
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
                    ? 'These docs cover v1.3.0, an upcoming pre-release that is still being finalized, so some details may change before launch. For the current stable docs, use v1.2.0.'
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

          {/* ────────────── SPECIALIZED GUIDES ────────────── */}
          <div className="mb-10 grid gap-4 sm:grid-cols-2">
            <a
              href="/docs/librevlm"
              className="group rounded-xl border border-surface-200 dark:border-white/[0.08] bg-white/80 dark:bg-white/[0.03] p-5 transition-colors hover:border-libre-500/40 hover:bg-libre-500/[0.03]"
            >
              <div className="flex items-center gap-2.5 mb-2">
                <span className="flex items-center justify-center w-9 h-9 rounded-lg bg-libre-500/10 border border-libre-500/20">
                  <Eye className="w-5 h-5 text-libre-600 dark:text-libre-400" />
                </span>
                <span className="font-semibold text-surface-900 dark:text-white">LibreVLM</span>
                <span className="ml-auto inline-flex items-center rounded-full border border-amber-500/30 bg-amber-500/10 px-2 py-0.5 text-[11px] font-semibold text-amber-700 dark:text-amber-300">Experimental</span>
              </div>
              <p className="text-sm text-surface-600 dark:text-surface-400">
                Open-vocabulary detection that wraps vision language models like Qwen3-VL and Florence-2. Detect anything you can name.
              </p>
              <span className="mt-3 inline-flex items-center gap-1 text-sm font-medium text-libre-600 dark:text-libre-400">
                Read the guide <ChevronRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
              </span>
            </a>
            <a
              href="/docs/experimental"
              className="group rounded-xl border border-surface-200 dark:border-white/[0.08] bg-white/80 dark:bg-white/[0.03] p-5 transition-colors hover:border-libre-500/40 hover:bg-libre-500/[0.03]"
            >
              <div className="flex items-center gap-2.5 mb-2">
                <span className="flex items-center justify-center w-9 h-9 rounded-lg bg-libre-500/10 border border-libre-500/20">
                  <Sparkles className="w-5 h-5 text-libre-600 dark:text-libre-400" />
                </span>
                <span className="font-semibold text-surface-900 dark:text-white">Experimental tasks</span>
                <span className="ml-auto inline-flex items-center rounded-full border border-amber-500/30 bg-amber-500/10 px-2 py-0.5 text-[11px] font-semibold text-amber-700 dark:text-amber-300">New</span>
              </div>
              <p className="text-sm text-surface-600 dark:text-surface-400">
                Classification, oriented boxes, pose, and LoRA / DoRA fine-tuning for YOLO9 and RF-DETR.
              </p>
              <span className="mt-3 inline-flex items-center gap-1 text-sm font-medium text-libre-600 dark:text-libre-400">
                Read the guide <ChevronRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
              </span>
            </a>
          </div>

          {/* ────────────── INTRODUCTION ────────────── */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
            <SectionHeading id="introduction" icon={BookOpen}>Introduction</SectionHeading>
            <ValidationScopeCallout />
            <P>
              LibreYOLO is an MIT-licensed computer-vision toolkit. v1.3.0 ships a broad catalogue across detection, segmentation, classification, depth and more, but the validated support surface is intentionally narrow:
            </P>
            <ul className="space-y-2 mb-4">
              <FeatureItem><strong className="text-surface-800 dark:text-white">YOLO9 detection</strong> - the CNN path.</FeatureItem>
              <FeatureItem><strong className="text-surface-800 dark:text-white">RF-DETR detection</strong> - the transformer path.</FeatureItem>
              <FeatureItem><strong className="text-surface-800 dark:text-white">RF-DETR segmentation</strong> - the heavily tested segmentation path.</FeatureItem>
            </ul>
            <P>
              We recommend those paths as the default choice for new projects because they receive the heaviest testing around detection, training and inference. Other supported families and tasks work through the same unified <InlineCode>LibreYOLO()</InlineCode> factory, but they are experimental in v1.3.0. Use them if you have a specific reason.
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
            Use this matrix as the quick v1.3.0 support map. <InlineCode>&#10003;</InlineCode>{' '}
            marks a validated path, <InlineCode>exp</InlineCode> is experimental,{' '}
            <InlineCode>prev</InlineCode> is a research preview, and empty cells are
            not currently supported. Only YOLO9 and RF-DETR detection (plus RF-DETR
            segmentation) are heavily tested; everything else, including the new
            classification, semantic, depth and point families, is experimental.
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
              <span className="w-1.5 h-1.5 rounded-full bg-libre-400" />PyTorch 1.13+ and torchvision 0.11+
            </li>
          </ul>

          <SubHeading>From PyPI</SubHeading>
          <CodeBlock language="bash">{`pip install libreyolo`}</CodeBlock>
          <P>
            These docs track the upcoming v1.3.0 release. Until v1.3.0 is published to PyPI, use a source install for the features documented on this page.
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
            <SupportBadge variant="validated">Heavily tested: detection, segmentation, training and inference</SupportBadge>
            <SupportBadge>Research preview: pose, OBB</SupportBadge>
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
            <SupportBadge>Research preview</SupportBadge>{' '}
            <strong className="text-surface-800 dark:text-white">Pose:</strong>{' '}
            <Checkpoints names={['LibreRFDETRx-pose.pt']} link={false} /> (ported
            from RF-DETR v1.8.0 GroupPose; only size <InlineCode>x</InlineCode> at
            576 ships).{' '}
            <strong className="text-surface-800 dark:text-white">OBB:</strong>{' '}
            <Checkpoints names={['LibreRFDETRn-obb.pt', 'LibreRFDETRs-obb.pt', 'LibreRFDETRm-obb.pt', 'LibreRFDETRl-obb.pt']} link={false} />{' '}
            (oriented boxes, uses detection input sizes). Treat both as research
            previews, not validated paths.
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreRFDETRs.pt")           # detect (validated)
# model = LibreYOLO("LibreRFDETRs-seg.pt")     # segment (validated)
# model = LibreYOLO("LibreRFDETRx-pose.pt")    # pose  (research preview)
# model = LibreYOLO("LibreRFDETRn-obb.pt")     # obb   (research preview)`}</CodeBlock>

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
              ['Pose estimation', <InlineCode key="p">&quot;pose&quot;</InlineCode>, <InlineCode key="ps">-pose</InlineCode>, 'YOLO-NAS, EdgeCrafter, RF-DETR (preview)'],
              ['Oriented boxes', <InlineCode key="o">&quot;obb&quot;</InlineCode>, <InlineCode key="os">-obb</InlineCode>, 'RF-DETR (preview)'],
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
              [<strong key="rfd">RF-DETR</strong>, 'detect and segment single-GPU heavily tested; pose and OBB research preview', 'detect', 'detect, segment, pose, obb'],
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
LibreRFDETRx-pose.pt     # preview; size x only

# Oriented boxes (-obb)
LibreRFDETRn-obb.pt      # preview

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
            LibreYOLO ships a ByteTrack multi-object tracker that consumes <InlineCode>Results</InlineCode> from any detector and adds persistent track IDs. It is most tested with single-GPU YOLO9 detection and RF-DETR detection; other detection families are experimental in v1.3.0.
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

          {/* ────────────── POSE ESTIMATION ────────────── */}
          <SectionHeading id="pose" icon={PersonStanding}>Pose Estimation</SectionHeading>
          <P>
            Pose (human keypoint) estimation runs on <InlineCode>YOLO-NAS (-pose)</InlineCode>,{' '}
            <InlineCode>EdgeCrafter (-pose)</InlineCode>, and, new in v1.3.0, an{' '}
            <InlineCode>RF-DETR (-pose)</InlineCode> preview. Each pose model is single-class
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
            <SupportBadge variant="experimental">Preview</SupportBadge>{' '}
            RF-DETR pose ships a single checkpoint at size <InlineCode>x</InlineCode> only:{' '}
            <InlineCode>LibreRFDETRx-pose.pt</InlineCode>. It is a research preview in v1.3.0.
          </P>
          <CodeBlock language="python">{`# RF-DETR pose preview (size x only)
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
            Pose training is supported for YOLO-NAS; EdgeCrafter pose is currently inference-only. RF-DETR pose is a preview (size <InlineCode>x</InlineCode> only). YOLO9 is detect-only and ships no pose checkpoints.
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
            The heavily tested training paths are single-GPU YOLO9 detection, RF-DETR detection, and RF-DETR segmentation. Other model-family trainers, YOLO9 segmentation training, and multi-GPU workflows are available but experimental in v1.3.0.
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
            After training completes, the model instance is automatically reloaded with the best weights so you can call <InlineCode>model(...)</InlineCode> immediately. YOLO9 segmentation training is supported via <InlineCode>LibreYOLO(&quot;LibreYOLO9c-seg.pt&quot;)</InlineCode>, but it is experimental in v1.3.0.
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

export default function Docs() {
  return <DocsPage version="v1.3.0" />
}
