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
  Palette, WandSparkles, MousePointerClick, Search, PenTool, FlaskConical,
  Boxes, Gauge, Timer, Combine, Eraser, ScanText, Dices, Binary,
} from 'lucide-react'
import SupportCallout from '@/components/SupportCallout'

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
  { id: 'ensembling', title: 'Ensembling', icon: Boxes },
  { id: 'segmentation', title: 'Instance Segmentation', icon: Scissors },
  { id: 'semantic-segmentation', title: 'Semantic Segmentation', icon: Palette },
  { id: 'panoptic-segmentation', title: 'Panoptic Segmentation', icon: Combine },
  { id: 'promptable-segmentation', title: 'Promptable Segmentation', icon: MousePointerClick },
  { id: 'open-vocabulary', title: 'Open-Vocabulary Detection', icon: Search },
  { id: 'pose', title: 'Pose Estimation', icon: PersonStanding },
  { id: 'gaze', title: 'Gaze Estimation', icon: Eye },
  { id: 'classification', title: 'Classification', icon: Tags },
  { id: 'depth', title: 'Depth Estimation', icon: Mountain },
  { id: 'restoration', title: 'Restoration & Upscaling', icon: WandSparkles },
  { id: 'background-removal', title: 'Background Removal', icon: Eraser },
  { id: 'ocr', title: 'OCR', icon: ScanText },
  { id: 'point-localization', title: 'Point Localization', icon: MapPin },
  { id: 'annotation', title: 'Annotation (LibreLabel)', icon: PenTool },
  { id: 'training', title: 'Training', icon: GraduationCap },
  { id: 'augmentation', title: 'Data Augmentation', icon: Dices },
  { id: 'distillation', title: 'Distillation', icon: FlaskConical },
  { id: 'monitoring', title: 'Training Monitor', icon: Gauge },
  { id: 'profiling', title: 'Profiling', icon: Timer },
  { id: 'validation', title: 'Validation', icon: CheckCircle2 },
  { id: 'quantization', title: 'Quantization', icon: Binary },
  { id: 'export', title: 'Export', icon: Upload },
  { id: 'torchscript-inference', title: 'TorchScript Inference', icon: Cpu },
  { id: 'onnx-inference', title: 'ONNX Inference', icon: Cpu },
  { id: 'tensorrt-inference', title: 'TensorRT Inference', icon: Cpu },
  { id: 'openvino-inference', title: 'OpenVINO Inference', icon: Cpu },
  { id: 'ncnn-inference', title: 'NCNN Inference', icon: Cpu },
  { id: 'coreml-inference', title: 'CoreML Inference', icon: Cpu },
  { id: 'tflite-inference', title: 'TFLite Inference', icon: Cpu },
  { id: 'cli', title: 'CLI', icon: SquareTerminal },
  { id: 'api-reference', title: 'API Reference', icon: FileCode },
  { id: 'architecture', title: 'Architecture Guide', icon: Wrench },
  { id: 'dataset-format', title: 'Dataset Format', icon: Database },
]

const docsVersions = [
  { version: 'v1.4.0', label: 'Latest', href: '/docs/v1.4.0' },
  { version: 'v1.3.1', label: 'Previous', href: '/docs/v1.3.1' },
  { version: 'v1.3.0', label: 'Archived', href: '/docs/v1.3.0' },
  { version: 'v1.2.0', label: 'Archived', href: '/docs/v1.2.0' },
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

  if (value === 'preview') {
    return (
      <span className="font-semibold text-sky-600 dark:text-sky-400" aria-label="research preview">
        prev
      </span>
    )
  }

  return <span className="sr-only">Not currently supported</span>
}

const compatibilityColumns = [
  'inference', 'training', 'detect', 'segment', 'semantic', 'panoptic', 'classify', 'pose', 'obb',
  'depth', 'point', 'restore', 'matte', 'ocr', 'gaze',
  'onnx', 'torchscript', 'tensorrt', 'openvino', 'ncnn', 'coreml', 'tflite',
]

// One row per family, shared by the English and Chinese matrices. Only set
// the columns a family supports; missing keys render as "not supported".
const compatibilityRows = [
  {
    family: 'YOLO9', status: 'Recommended detect path; int8 / fp8 quantization', statusZh: '推荐的 detect 路径；支持 int8 / fp8 量化',
    inference: 'yes', training: 'yes', detect: 'yes',
    onnx: 'yes', torchscript: 'yes', tensorrt: 'yes', openvino: 'yes', ncnn: 'yes', coreml: 'yes', tflite: 'yes',
  },
  {
    family: 'RF-DETR', status: 'Recommended detect + segment; pose / OBB preview; TFLite detect experimental, segment / pose blocked', statusZh: '推荐的 detect + segment；pose / OBB 预览；TFLite detect 为实验性，segment / pose 被拦截',
    inference: 'yes', training: 'yes', detect: 'yes', segment: 'yes', pose: 'preview', obb: 'preview',
    onnx: 'yes', torchscript: 'yes', tensorrt: 'yes', openvino: 'yes', coreml: 'yes', tflite: 'yes',
  },
  {
    family: 'YOLOX', status: '', statusZh: '',
    inference: 'yes', training: 'yes', detect: 'yes',
    onnx: 'yes', torchscript: 'yes', tensorrt: 'yes', openvino: 'yes', ncnn: 'yes', coreml: 'yes', tflite: 'yes',
  },
  {
    family: 'YOLOv7', status: 'Trainable as of v1.4.0 (SimOTA); was inference-only', statusZh: 'v1.4.0 起可训练（SimOTA）；此前仅推理',
    inference: 'yes', training: 'yes', detect: 'yes',
    onnx: 'yes', torchscript: 'yes', tensorrt: 'yes', openvino: 'yes', ncnn: 'yes',
  },
  {
    family: 'YOLO9-E2E', status: '', statusZh: '',
    inference: 'yes', training: 'yes', detect: 'yes',
    onnx: 'yes', torchscript: 'yes', tensorrt: 'yes', openvino: 'yes', ncnn: 'yes',
  },
  {
    family: 'YOLO9-P2', status: 'Small objects; VisDrone weights only (non-commercial)', statusZh: '小目标；仅 VisDrone 权重（非商用）',
    inference: 'yes', training: 'yes', detect: 'yes',
    onnx: 'yes', torchscript: 'yes', tensorrt: 'yes', openvino: 'yes', ncnn: 'yes',
  },
  {
    family: 'YOLO-NAS', status: 'multi-class pose training new in v1.4.0', statusZh: 'v1.4.0 新增多类别姿态训练',
    inference: 'yes', training: 'yes', detect: 'yes', pose: 'yes',
    onnx: 'yes', torchscript: 'yes', tensorrt: 'yes', openvino: 'yes', ncnn: 'yes',
  },
  {
    family: 'D-FINE', status: 'segmentation + dynamic eval sizes new in v1.4.0; LoRA', statusZh: 'v1.4.0 新增分割与动态验证尺寸；支持 LoRA',
    inference: 'yes', training: 'yes', detect: 'yes', segment: 'yes',
    onnx: 'yes', torchscript: 'yes', tensorrt: 'yes', openvino: 'yes',
  },
  {
    family: 'DEIM', status: 'fine-tune defaults fixed in v1.4.0; LoRA', statusZh: 'v1.4.0 修正微调默认值；支持 LoRA',
    inference: 'yes', training: 'yes', detect: 'yes',
    onnx: 'yes', torchscript: 'yes', tensorrt: 'yes', openvino: 'yes',
  },
  {
    family: 'DEIMv2', status: 'LoRA', statusZh: '支持 LoRA',
    inference: 'yes', training: 'yes', detect: 'yes',
    onnx: 'yes', torchscript: 'yes', tensorrt: 'yes', openvino: 'yes',
  },
  {
    family: 'RT-DETR', status: 'LoRA', statusZh: '支持 LoRA',
    inference: 'yes', training: 'yes', detect: 'yes',
    onnx: 'yes', torchscript: 'yes', tensorrt: 'yes', openvino: 'yes', coreml: 'yes',
  },
  {
    family: 'RT-DETRv2', status: 'LoRA', statusZh: '支持 LoRA',
    inference: 'yes', training: 'yes', detect: 'yes',
    onnx: 'yes', torchscript: 'yes', tensorrt: 'yes', openvino: 'yes',
  },
  {
    family: 'RT-DETRv4', status: 'dynamic eval sizes new in v1.4.0; LoRA', statusZh: 'v1.4.0 新增动态验证尺寸；支持 LoRA',
    inference: 'yes', training: 'yes', detect: 'yes',
    onnx: 'yes', torchscript: 'yes', tensorrt: 'yes', openvino: 'yes',
  },
  {
    family: 'PicoDet', status: 'fine-tune defaults fixed in v1.4.0', statusZh: 'v1.4.0 修正微调默认值',
    inference: 'yes', training: 'yes', detect: 'yes',
    onnx: 'yes', torchscript: 'yes', tensorrt: 'yes', openvino: 'yes', ncnn: 'yes',
  },
  {
    family: 'RTMDet', status: 'RTMDet-Ins segmentation (inference + val) new in v1.4.0', statusZh: 'v1.4.0 新增 RTMDet-Ins 实例分割（仅推理与验证）',
    inference: 'yes', training: 'yes', detect: 'yes', segment: 'yes',
    onnx: 'yes', torchscript: 'yes', tensorrt: 'yes', openvino: 'yes',
  },
  {
    family: 'EC', status: 'LoRA (detect)', statusZh: '支持 LoRA（仅 detect）',
    inference: 'yes', training: 'yes', detect: 'yes', segment: 'yes', pose: 'yes',
    onnx: 'yes', torchscript: 'yes', tensorrt: 'yes', openvino: 'yes',
  },
  {
    family: 'EoMT', status: 'Semantic + instance + panoptic; instance and panoptic new in v1.4.0; inference and val only', statusZh: '语义 + 实例 + 全景；实例与全景为 v1.4.0 新增；仅推理与验证',
    inference: 'yes', segment: 'yes', semantic: 'yes', panoptic: 'yes',
    onnx: 'yes', torchscript: 'yes',
  },
  {
    family: 'SegFormer', status: 'New in v1.4.0; semantic b0-b5; ADE20K weights non-commercial', statusZh: 'v1.4.0 新增；语义分割 b0-b5；ADE20K 权重非商用',
    inference: 'yes', training: 'yes', semantic: 'yes',
  },
  {
    family: 'PIDNet', status: 'Semantic; inference and val only; ONNX / TorchScript / NCNN / TFLite', statusZh: '语义分割；仅推理与验证；支持 ONNX / TorchScript / NCNN / TFLite',
    inference: 'yes', semantic: 'yes',
    onnx: 'yes', torchscript: 'yes', ncnn: 'yes', tflite: 'yes',
  },
  {
    family: 'DINOv2', status: 'semantic / classify / detect (needs transformers)', statusZh: '语义 / 分类 / 检测（需要 transformers）',
    inference: 'yes', training: 'yes', detect: 'yes', semantic: 'yes', classify: 'yes',
    onnx: 'yes', torchscript: 'yes',
  },
  {
    family: 'MobileNetV4', status: 'Classifier (Apache-2.0)', statusZh: '分类器（Apache-2.0）',
    inference: 'yes', training: 'yes', classify: 'yes',
    onnx: 'yes', torchscript: 'yes', ncnn: 'yes', tflite: 'yes',
  },
  {
    family: 'ConvNeXt', status: 'Classifier (Apache-2.0); LoRA', statusZh: '分类器（Apache-2.0）；支持 LoRA',
    inference: 'yes', training: 'yes', classify: 'yes',
    onnx: 'yes', torchscript: 'yes', ncnn: 'yes', tflite: 'yes',
  },
  {
    family: 'EfficientNetV2', status: 'Classifier (Apache-2.0)', statusZh: '分类器（Apache-2.0）',
    inference: 'yes', training: 'yes', classify: 'yes',
    onnx: 'yes', torchscript: 'yes', ncnn: 'yes', tflite: 'yes',
  },
  {
    family: 'ResNet', status: 'Classifier', statusZh: '分类器',
    inference: 'yes', training: 'yes', classify: 'yes',
    onnx: 'yes', torchscript: 'yes', ncnn: 'yes', tflite: 'yes',
  },
  {
    family: 'CLIP', status: 'Zero-shot classification', statusZh: '零样本分类',
    inference: 'yes', classify: 'yes',
    onnx: 'yes',
  },
  {
    family: 'SigLIP2', status: 'New in v1.4.0; zero-shot classification, inference-only', statusZh: 'v1.4.0 新增；零样本分类，仅推理',
    inference: 'yes', classify: 'yes',
    onnx: 'yes',
  },
  {
    family: 'Depth Anything V2', status: 'ONNX export new in v1.4.0 (fixed resolution, batch 1)', statusZh: 'v1.4.0 新增 ONNX 导出（固定分辨率、batch 1）',
    inference: 'yes', depth: 'yes',
    onnx: 'yes', torchscript: 'yes',
  },
  {
    family: 'Depth Anything 3', status: 'New in v1.4.0; size l at 504, Apache-2.0', statusZh: 'v1.4.0 新增；l 尺寸、504 输入，Apache-2.0',
    inference: 'yes', depth: 'yes',
  },
  {
    family: 'ZipDepth', status: 'New in v1.4.0; b / bnpu at 384, MIT', statusZh: 'v1.4.0 新增；b / bnpu、384 输入，MIT',
    inference: 'yes', depth: 'yes',
    onnx: 'yes',
  },
  {
    family: 'FOMO', status: 'no auto-download; ONNX export new in v1.4.0', statusZh: '不自动下载；v1.4.0 新增 ONNX 导出',
    inference: 'yes', training: 'yes', point: 'yes',
    onnx: 'yes', torchscript: 'yes',
  },
  {
    family: 'NAFNet', status: 'Restoration (denoise / deblur); SIDD denoise weights', statusZh: '图像修复（去噪 / 去模糊）；提供 SIDD 去噪权重',
    inference: 'yes', training: 'yes', restore: 'yes',
    onnx: 'yes', torchscript: 'yes', ncnn: 'yes',
  },
  {
    family: 'SwinIR', status: 'New in v1.4.0; 4x super-resolution s / m / l, Apache-2.0; inference and val', statusZh: 'v1.4.0 新增；4 倍超分 s / m / l，Apache-2.0；仅推理与验证',
    inference: 'yes', restore: 'yes',
    onnx: 'yes', torchscript: 'yes',
  },
  {
    family: 'Real-ESRGAN', status: 'New in v1.4.0; x4 / x2 / x4t super-resolution; inference and val', statusZh: 'v1.4.0 新增；x4 / x2 / x4t 超分；仅推理与验证',
    inference: 'yes', restore: 'yes',
    onnx: 'yes', torchscript: 'yes', ncnn: 'yes', tflite: 'yes',
  },
  {
    family: 'BiRefNet', status: 'New in v1.4.0; background removal (matte) t / l at 1024', statusZh: 'v1.4.0 新增；背景移除（matte）t / l，1024 输入',
    inference: 'yes', matte: 'yes',
    onnx: 'yes', torchscript: 'yes',
  },
  {
    family: 'PP-OCR', status: 'New in v1.4.0; text detection + recognition; inference and val', statusZh: 'v1.4.0 新增；文本检测 + 识别；仅推理与验证',
    inference: 'yes', ocr: 'yes',
  },
  {
    family: 'YOLO1 / YOLO2 / YOLO3 / YOLO4', status: 'Museum tier, inference-only (YOLO1 new in v1.4.0)', statusZh: '博物馆级历史基线，仅推理（YOLO1 为 v1.4.0 新增）',
    inference: 'yes', detect: 'yes',
    onnx: 'yes', torchscript: 'yes',
  },
  {
    family: 'L2CS', status: 'Inference-only', statusZh: '仅推理',
    inference: 'yes', gaze: 'yes',
    onnx: 'yes',
  },
]

function CompatibilityMatrix({ zh = false }) {
  const headers = zh
    ? ['模型系列', '备注', '推理', '训练', '检测', '分割', '语义', '全景', '分类', '姿态', 'OBB', '深度', '点', '修复', '抠图', 'OCR', '视线', 'ONNX', 'TorchScript', 'TensorRT', 'OpenVINO', 'NCNN', 'CoreML', 'TFLite']
    : ['Model family', 'Notes', 'Inference', 'Training', 'Detect', 'Segment', 'Semantic', 'Panoptic', 'Classify', 'Pose', 'OBB', 'Depth', 'Point', 'Restore', 'Matte', 'OCR', 'Gaze', 'ONNX', 'TorchScript', 'TensorRT', 'OpenVINO', 'NCNN', 'CoreML', 'TFLite']

  return (
    <DocTable
      headers={headers}
      rows={compatibilityRows.map((row) => [
        <strong key={`${row.family}-family`} className="text-surface-800 dark:text-white whitespace-nowrap">{row.family}</strong>,
        <span key={`${row.family}-status`} className="text-xs leading-relaxed">{zh ? row.statusZh : row.status}</span>,
        ...compatibilityColumns.map((column) => <MatrixMark key={`${row.family}-${column}`} value={row[column]} />),
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
            v1.4.0 validation scope
          </p>
          <p className="text-sm text-surface-600 dark:text-surface-400 mb-2">
            The heavily tested path is detection, training and inference for YOLO9 and RF-DETR, including RF-DETR segmentation.
          </p>
          <p className="text-sm text-surface-600 dark:text-surface-400">
            Other model families and tasks are available but experimental. Multi-GPU training got a correctness overhaul in v1.4.0 and is much stronger than in v1.3.x, but it is still outside the validated scope.
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
            Detection, training and inference for these models receive the heaviest testing. Treat other families, tasks, and multi-GPU workflows as experimental in v1.4.0.
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
                    ? 'These are the docs for v1.4.0, the current stable release. Earlier versions stay available from the version menu.'
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

          <SupportCallout className="mb-8" community={false} />

          {/* ────────────── SPECIALIZED GUIDES ────────────── */}
          <P>
            Two companion guides go deeper on specialized topics. The{' '}
            <a href="/docs/librevlm" className="text-libre-600 dark:text-libre-400 hover:underline">LibreVLM guide</a>{' '}
            covers the vision-language tier (Qwen3-VL, Florence-2), which generates text that LibreYOLO parses into boxes. That is a different thing from{' '}
            <a href="#open-vocabulary" className="text-libre-600 dark:text-libre-400 hover:underline">open-vocabulary detection</a>, new in v1.3.1, which uses purpose-built detectors conditioned on text and is documented on this page. The{' '}
            <a href="/docs/experimental" className="text-libre-600 dark:text-libre-400 hover:underline">experimental tasks guide</a>{' '}
            covers additional experimental workflows, including LoRA / DoRA fine-tuning.
          </P>

          {/* ────────────── INTRODUCTION ────────────── */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
            <SectionHeading id="introduction" icon={BookOpen}>Introduction</SectionHeading>
            <ValidationScopeCallout />
            <P>
              LibreYOLO is an MIT-licensed computer-vision toolkit. v1.4.0 ships a broad catalogue across detection, segmentation, classification, depth, restoration, OCR and more, but the validated support surface is intentionally narrow:
            </P>
            <ul className="space-y-2 mb-4">
              <FeatureItem><strong className="text-surface-800 dark:text-white">YOLO9 detection</strong> - the CNN path.</FeatureItem>
              <FeatureItem><strong className="text-surface-800 dark:text-white">RF-DETR detection</strong> - the transformer path.</FeatureItem>
              <FeatureItem><strong className="text-surface-800 dark:text-white">RF-DETR segmentation</strong> - the heavily tested segmentation path.</FeatureItem>
            </ul>
            <P>
              We recommend those paths as the default choice for new projects because they receive the heaviest testing around detection, training and inference. Other supported families and tasks work through the same unified <InlineCode>LibreYOLO()</InlineCode> factory, but they are experimental in v1.4.0. Use them if you have a specific reason.
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
              <FeatureItem>Detection, instance / semantic / panoptic segmentation, pose, classification, depth, restoration, background removal, OCR, point localization, and gaze through one consistent API</FeatureItem>
              <FeatureItem>Image, directory, and video inference (with optional tiled inference for large frames)</FeatureItem>
              <FeatureItem>Built-in multi-object tracking: ByteTrack, OC-SORT, BoT-SORT, and Deep OC-SORT with ReID</FeatureItem>
              <FeatureItem>Per-family <a href="#augmentation" className="text-libre-600 dark:text-libre-400 hover:underline">training augmentation control</a> with a declarative support spec that warns when a knob is ignored</FeatureItem>
              <FeatureItem>PyTorch-native <a href="#quantization" className="text-libre-600 dark:text-libre-400 hover:underline">quantization</a>: fp16 / bf16 / fp8 / int8 / int4 recipes with QAT and QAD recovery</FeatureItem>
              <FeatureItem>ONNX, TorchScript, TensorRT, OpenVINO, NCNN, CoreML, and TFLite export with embedded metadata, plus matching runtime backends</FeatureItem>
              <FeatureItem>COCO-compatible validation with mAP metrics, plus segmentation, pose, panoptic, matte, and OCR validators</FeatureItem>
              <FeatureItem>A <InlineCode>libreyolo</InlineCode> command-line tool for predict / train / val / export / quantize</FeatureItem>
              <FeatureItem>Accepts any image format: file paths, URLs, PIL, NumPy, PyTorch tensors, raw bytes</FeatureItem>
            </ul>
          </motion.div>

          <SubHeading>What&apos;s new in v1.4.0</SubHeading>
          <ul className="space-y-2 my-4">
            <FeatureItem>
              <strong className="text-surface-800 dark:text-white">15 new model families</strong>, including SegFormer (semantic), SwinIR and Real-ESRGAN (super-resolution), BiRefNet (background removal), ZipDepth and Depth Anything 3 (depth), PP-OCR (text), SigLIP2 (zero-shot classification), SAM 3, EdgeTAM and PicoSAM3 (promptable segmentation), and OmDet-Turbo and OV-DEIM (open-vocabulary detection).
            </FeatureItem>
            <FeatureItem>
              <strong className="text-surface-800 dark:text-white">Three new tasks</strong>: <InlineCode>panoptic</InlineCode>, <InlineCode>matte</InlineCode>, and <InlineCode>ocr</InlineCode>, each with its own result payload and validator.
            </FeatureItem>
            <FeatureItem>
              <strong className="text-surface-800 dark:text-white">A documented augmentation system.</strong> Every training augmentation knob now has a per-family support spec, the CLI warns when a family ignores a parameter you set, and this page finally has a full <a href="#augmentation" className="text-libre-600 dark:text-libre-400 hover:underline">Data Augmentation</a> section.
            </FeatureItem>
            <FeatureItem>
              <strong className="text-surface-800 dark:text-white">A quantization stack</strong>: <InlineCode>model.quantize()</InlineCode> and <InlineCode>libreyolo quantize</InlineCode> with nine recipes, honest simulation-based accuracy, QAT / QAD recovery, and packed low-bit checkpoints.
            </FeatureItem>
            <FeatureItem>
              <strong className="text-surface-800 dark:text-white">Two new trackers</strong>: BoT-SORT and Deep OC-SORT (appearance ReID), alongside ByteTrack and OC-SORT.
            </FeatureItem>
            <FeatureItem>
              <strong className="text-surface-800 dark:text-white">A multi-GPU correctness overhaul</strong>: correct DDP sharding everywhere, globally reduced loss normalizers, SyncBatchNorm defaults for BatchNorm-heavy families, and loud setup errors instead of silently wrong runs.
            </FeatureItem>
            <FeatureItem>
              <strong className="text-surface-800 dark:text-white">YOLOv7 training</strong> (the family was inference-only in v1.3.1), LoRA fine-tuning across seven more families, DINOv2 foundation-teacher distillation, and a TFLite runtime backend.
            </FeatureItem>
          </ul>

          <SubHeading>Compatibility notes in v1.4.0</SubHeading>
          <ul className="space-y-2 my-4">
            <FeatureItem>
              <strong className="text-surface-800 dark:text-white">Checkpoints only move forward.</strong> Checkpoints that use the new task strings (<InlineCode>panoptic</InlineCode>, <InlineCode>matte</InlineCode>, <InlineCode>ocr</InlineCode>) or finalized quantization state are not loadable by v1.3.1. Everything v1.3.x wrote still loads in v1.4.0, and <InlineCode>Results</InlineCode> and <InlineCode>LibreEoMT</InlineCode> keep full v1.3 positional-argument compatibility.
            </FeatureItem>
            <FeatureItem>
              <strong className="text-surface-800 dark:text-white">Some fine-tune defaults changed</strong> because the old ones were harmful: PicoDet (<InlineCode>lr0</InlineCode> 0.1 to 0.01) and DEIM (<InlineCode>lr0</InlineCode> 4e-4 to 1e-4, <InlineCode>min_lr_ratio</InlineCode> 0.5 to 0.05). Pass the old values explicitly to reproduce upstream COCO recipes.
            </FeatureItem>
            <FeatureItem>
              <strong className="text-surface-800 dark:text-white">Training results can shift</strong> where augmentation defaults were fixed: semantic segmentation now applies HSV jitter by default, restoration training adds coupled vertical flip and rot90, and AdamW no longer decays BatchNorm / bias parameters.
            </FeatureItem>
            <FeatureItem>
              <strong className="text-surface-800 dark:text-white"><InlineCode>model.train(profile=True)</InlineCode> keeps training</strong> after the profiled window instead of stopping. Pass <InlineCode>profile_then_stop=True</InlineCode> for the old behavior.
            </FeatureItem>
            <FeatureItem>
              <strong className="text-surface-800 dark:text-white"><InlineCode>libreyolo models --json</InlineCode> schema changed</strong> (task-suffixed CLI names, new keys); <InlineCode>formats</InlineCode> and <InlineCode>info</InlineCode> JSON gained keys. Update scripts that parse them.
            </FeatureItem>
          </ul>

          <Divider />

          {/* ────────────── COMPATIBILITY ────────────── */}
          <SectionHeading id="compatibility" icon={CheckCircle2}>Compatibility</SectionHeading>
          <P>
            Use this matrix as the quick v1.4.0 support map. <InlineCode>&#10003;</InlineCode>{' '}
            marks a supported path,{' '}
            <InlineCode>prev</InlineCode> is a research preview, and empty cells are
            not currently supported. YOLO9 and RF-DETR detection (plus RF-DETR
            segmentation) get the heaviest testing and are the recommended starting
            point; the other families are supported too, so please report an issue if something misbehaves.
          </P>
          <CompatibilityMatrix />
          <p className="text-sm text-surface-600 dark:text-surface-400 leading-relaxed mb-4">
            The export columns summarize the canonical support matrix that ships in
            v1.4.0; query it exactly with <InlineCode>libreyolo formats --family ...</InlineCode>{' '}
            or <InlineCode>libreyolo info --model ... --json</InlineCode> (see{' '}
            <a href="#export" className="text-libre-600 dark:text-libre-400 hover:underline">Export</a>).
            The promptable-segmentation tier (SAM, SAM 2, SAM 3, MobileSAM, EdgeTAM, PicoSAM3),
            the open-vocabulary tier (Grounding DINO, OWLv2, OmDet-Turbo, OV-DEIM), and the VLM
            tier live outside the <InlineCode>LibreYOLO()</InlineCode> factory and are not rows
            here; see their sections. CoreML exports produce <InlineCode>.mlpackage</InlineCode>{' '}
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
            v1.4.0 is the current release on PyPI, and it is what these docs describe. Everything on this page works from the published package: you do not need a source install.
          </P>

          <SubHeading>From source</SubHeading>
          <CodeBlock language="bash">{`git clone https://github.com/LibreYOLO/libreyolo.git
cd libreyolo
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

# TFLite export + LiteRT runtime backend (Python 3.12+)
pip install libreyolo[tflite]
# "litert" is an alias extra: pip install libreyolo[litert]

# Tracking API compatibility extra
pip install libreyolo[tracking]
# Tracking dependencies are part of the base install; Deep OC-SORT's ReID
# embedder weights auto-download on first use.

# CoreML export and inference (macOS only for runtime)
pip install libreyolo[coreml]
# or: pip install coremltools

# L2CS gaze optional auto-download helper
pip install libreyolo[gaze]

# Promptable segmentation (LibreSAM: SAM-1, SAM-2, SAM 3, MobileSAM,
# EdgeTAM, PicoSAM3)
pip install libreyolo[sam]

# Open-vocabulary detection (Grounding DINO, OWLv2, OmDet-Turbo, OV-DEIM)
pip install libreyolo[openvocab]

# LibreLabel AI assist (SAM click-to-mask)
pip install libreyolo[label]

# Zero-shot classification
pip install libreyolo[clip]       # CLIP
pip install libreyolo[siglip2]    # SigLIP2 tokenizer (SentencePiece)

# Validation and training plots
pip install libreyolo[plots]

# SenseNova Vision preview
pip install libreyolo[sensenova]

# Converter-only dependencies for CLIP and SigLIP2 checkpoints
pip install libreyolo[clip-convert]
pip install libreyolo[siglip2-convert]

# LoRA fine-tuning (peft)
pip install libreyolo[lora]

# Experiment loggers
pip install libreyolo[tensorboard]   # or [mlflow], [wandb]

# EoMT instance / panoptic segmentation
pip install libreyolo[eomt]

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
            LibreYOLO v1.4.0 ships two validated flagship families plus a broad
            catalogue of supported models: fifteen families are new in this
            release alone. Every checkpoint-based model loads through the same{' '}
            <InlineCode>LibreYOLO()</InlineCode> factory, but only the validated
            paths below should be treated as heavily tested.
          </P>

          <ValidatedModelHeader title="YOLO9 - CNN flagship">
            <SupportBadge variant="validated">Default: LibreYOLO9c.pt</SupportBadge>
            <SupportBadge variant="validated">Heavily tested: detection, training and inference</SupportBadge>
            <SupportBadge>Detect-only in v1.4.0</SupportBadge>
            <SupportBadge>Quantizable: int8 / fp8</SupportBadge>
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
            YOLO9 is detection-only in v1.4.0. The non-detect flagship variants
            (including the old <InlineCode>-seg</InlineCode> checkpoints) were
            removed in v1.3.0; for segmentation use RF-DETR or the experimental
            segmentation families below.
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")   # detection`}</CodeBlock>

          <ValidatedModelHeader title="RF-DETR - transformer flagship">
            <SupportBadge variant="validated">Recommended transformer path</SupportBadge>
            <SupportBadge variant="validated">Heavily tested: detection, segmentation, training and inference</SupportBadge>
            <SupportBadge>Research preview: pose, OBB</SupportBadge>
            <SupportBadge>LoRA + all quantization recipes</SupportBadge>
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
            {' '}The larger <InlineCode>-seg</InlineCode> sizes (<InlineCode>x</InlineCode>, <InlineCode>xx</InlineCode>)
            carry the upstream RF-DETR seg-XL / seg-2XL weights under a
            non-commercial license: check the model card before commercial use. See the{' '}
            <a href="#segmentation" className="text-libre-600 dark:text-libre-400 hover:underline">Segmentation</a> section.
          </P>
          <P>
            <SupportBadge>Research preview</SupportBadge>{' '}
            <strong className="text-surface-800 dark:text-white">Pose:</strong>{' '}
            <Checkpoints names={['LibreRFDETRx-pose.pt']} /> (ported from RF-DETR
            GroupPose).{' '}
            <strong className="text-surface-800 dark:text-white">OBB:</strong>{' '}
            <Checkpoints names={['LibreRFDETRn-obb.pt', 'LibreRFDETRs-obb.pt', 'LibreRFDETRm-obb.pt', 'LibreRFDETRl-obb.pt']} />{' '}
            (oriented boxes, uses detection input sizes). These checkpoints are
            trained for six vehicle classes: <InlineCode>bike</InlineCode>,{' '}
            <InlineCode>bus</InlineCode>, <InlineCode>car</InlineCode>,{' '}
            <InlineCode>other_vehicle</InlineCode>, <InlineCode>taxi</InlineCode>,
            and <InlineCode>truck</InlineCode>. They are not COCO-80 models.
            Treat both pose and OBB as research previews, not validated paths.
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreRFDETRs.pt")           # detect (validated)
# model = LibreYOLO("LibreRFDETRs-seg.pt")     # segment (validated)
# model = LibreYOLO("LibreRFDETRx-pose.pt")    # pose  (research preview)
# model = LibreYOLO("LibreRFDETRn-obb.pt")     # obb   (research preview)`}</CodeBlock>

          <SubHeading>Additional detection families</SubHeading>
          <P>
            Detection-capable families that share the same factory and API surface
            as the validated paths. These are experimental in v1.4.0. Each
            checkpoint name links to its model card on the{' '}
            <a href="https://huggingface.co/LibreYOLO" target="_blank" rel="noopener noreferrer" className="text-libre-600 dark:text-libre-400 hover:underline">LibreYOLO org</a>;
            pass any name to <InlineCode>LibreYOLO()</InlineCode> and the factory
            fetches it on first use.
          </P>
          <DocTable
            headers={['Family', 'Status', 'Tasks', 'Checkpoints']}
            rows={[
              ['YOLOX', <SupportBadge key="b">Experimental</SupportBadge>, 'detect', <Checkpoints key="yolox" names={['LibreYOLOXn.pt', 'LibreYOLOXt.pt', 'LibreYOLOXs.pt', 'LibreYOLOXm.pt', 'LibreYOLOXl.pt', 'LibreYOLOXx.pt']} />],
              ['YOLOv7', <SupportBadge key="b">Experimental</SupportBadge>, 'detect (trainable as of v1.4.0)', <Checkpoints key="y7" names={['LibreYOLO7b.pt']} />],
              ['YOLO9-E2E', <SupportBadge key="b">Experimental</SupportBadge>, 'detect', <Checkpoints key="y9e2e" names={['LibreYOLO9E2Et.pt', 'LibreYOLO9E2Es.pt', 'LibreYOLO9E2Em.pt', 'LibreYOLO9E2Ec.pt']} />],
              ['YOLO-NAS', <SupportBadge key="b">Experimental</SupportBadge>, 'detect, pose', <Checkpoints key="ynas" link={false} names={['LibreYOLONASs.pt', 'LibreYOLONASm.pt', 'LibreYOLONASl.pt', 'LibreYOLONASn-pose.pt', 'LibreYOLONASs-pose.pt', 'LibreYOLONASm-pose.pt', 'LibreYOLONASl-pose.pt']} />],
              ['D-FINE', <SupportBadge key="b">Experimental</SupportBadge>, 'detect, segment (new in v1.4.0)', <Checkpoints key="dfine" names={['LibreDFINEn.pt', 'LibreDFINEs.pt', 'LibreDFINEm.pt', 'LibreDFINEl.pt', 'LibreDFINEx.pt', 'LibreDFINEn-seg.pt', 'LibreDFINEs-seg.pt', 'LibreDFINEm-seg.pt', 'LibreDFINEl-seg.pt', 'LibreDFINEx-seg.pt']} />],
              ['DEIM', <SupportBadge key="b">Experimental</SupportBadge>, 'detect', <Checkpoints key="deim" names={['LibreDEIMn.pt', 'LibreDEIMs.pt', 'LibreDEIMm.pt', 'LibreDEIMl.pt', 'LibreDEIMx.pt']} />],
              ['DEIMv2', <SupportBadge key="b">Experimental</SupportBadge>, 'detect', <Checkpoints key="deimv2" names={['LibreDEIMv2atto.pt', 'LibreDEIMv2femto.pt', 'LibreDEIMv2pico.pt', 'LibreDEIMv2n.pt', 'LibreDEIMv2s.pt', 'LibreDEIMv2m.pt', 'LibreDEIMv2l.pt', 'LibreDEIMv2x.pt']} />],
              ['RT-DETR', <SupportBadge key="b">Experimental</SupportBadge>, 'detect', <Checkpoints key="rtdetr" names={['LibreRTDETRr18.pt', 'LibreRTDETRr34.pt', 'LibreRTDETRr50.pt', 'LibreRTDETRr50m.pt', 'LibreRTDETRr101.pt', 'LibreRTDETRl.pt', 'LibreRTDETRx.pt']} />],
              ['RT-DETRv2', <SupportBadge key="b">Experimental</SupportBadge>, 'detect', <Checkpoints key="rtdetrv2" names={['LibreRTDETRv2r18.pt', 'LibreRTDETRv2r34.pt', 'LibreRTDETRv2r50.pt', 'LibreRTDETRv2r50m.pt', 'LibreRTDETRv2r101.pt']} />],
              ['RT-DETRv4', <SupportBadge key="b">Experimental</SupportBadge>, 'detect', <Checkpoints key="rtdetrv4" names={['LibreRTDETRv4s.pt', 'LibreRTDETRv4m.pt', 'LibreRTDETRv4l.pt', 'LibreRTDETRv4x.pt']} />],
              ['PicoDet', <SupportBadge key="b">Experimental</SupportBadge>, 'detect', <Checkpoints key="picodet" names={['LibrePICODETs.pt', 'LibrePICODETm.pt', 'LibrePICODETl.pt']} />],
              ['RTMDet', <SupportBadge key="b">Experimental</SupportBadge>, 'detect, segment (RTMDet-Ins, inference + val)', <Checkpoints key="rtmdet" names={['LibreRTMDett.pt', 'LibreRTMDets.pt', 'LibreRTMDetm.pt', 'LibreRTMDetl.pt', 'LibreRTMDetx.pt', 'LibreRTMDett-seg.pt', 'LibreRTMDets-seg.pt', 'LibreRTMDetm-seg.pt', 'LibreRTMDetl-seg.pt', 'LibreRTMDetx-seg.pt']} />],
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

          <SubHeading>New model families in v1.4.0</SubHeading>
          <P>
            v1.4.0 adds fifteen model families. The checkpoint-based ones load
            through the same <InlineCode>LibreYOLO()</InlineCode> factory; the
            promptable-segmentation and open-vocabulary entries are constructed
            directly (see their sections). All of them are experimental.
          </P>
          <DocTable
            headers={['Family', 'Task', 'Sizes', 'Checkpoints / how to load']}
            rows={[
              ['SegFormer', 'semantic', 'b0-b5 (512; b5 at 640)', <Checkpoints key="segf" names={['LibreSegformerb0-sem.pt', 'LibreSegformerb1-sem.pt', 'LibreSegformerb2-sem.pt', 'LibreSegformerb3-sem.pt', 'LibreSegformerb4-sem.pt', 'LibreSegformerb5-sem.pt']} />],
              ['SwinIR', 'restore (4x super-resolution)', 's / m / l', <Checkpoints key="swinir" names={['LibreSwinIRs-restore.pt', 'LibreSwinIRm-restore.pt', 'LibreSwinIRl-restore.pt']} />],
              ['Real-ESRGAN', 'restore (super-resolution)', 'x4 / x2 / x4t', <Checkpoints key="resr" names={['LibreRealESRGANx4-restore.pt', 'LibreRealESRGANx2-restore.pt', 'LibreRealESRGANx4t-restore.pt']} />],
              ['BiRefNet', 'matte (background removal)', 't / l (1024)', <span key="birefnet"><Checkpoints names={['LibreBiRefNetl-matte.pt']} />; the t weights are not rehosted yet</span>],
              ['ZipDepth', 'depth', 'b / bnpu (384)', <Checkpoints key="zip" names={['LibreZipDepthb-depth.pt', 'LibreZipDepthbnpu-depth.pt']} />],
              ['Depth Anything 3', 'depth', 'l (504)', <Checkpoints key="da3" names={['LibreDepthAnything3l-depth.pt']} />],
              ['PP-OCR (v5)', 'ocr', 't / l (960)', <Checkpoints key="ppocr" names={['LibrePPOCRt-ocr.pt', 'LibrePPOCRl-ocr.pt']} />],
              ['SigLIP2', 'zero-shot classify', 'b16 / so400m', <Checkpoints key="siglip" names={['LibreSigLIP2b16-cls.pt', 'LibreSigLIP2so400m-cls.pt']} />],
              ['YOLOv1', 'detect (museum tier)', 't / b (448, VOC-20)', <span key="y1"><Checkpoints names={['LibreYOLO1b.pt']} />; the tiny weights are lost upstream</span>],
              ['SAM 3', 'promptable segmentation', 'large (1008)', <span key="sam3"><InlineCode>LibreSAM3()</InlineCode>: gated Hugging Face weights, Meta SAM license</span>],
              ['EdgeTAM', 'promptable segmentation', 'edge (1024)', <span key="edgetam"><InlineCode>LibreEdgeTAM()</InlineCode>: image inference only, Apache-2.0</span>],
              ['PicoSAM3', 'promptable ROI segmentation', '96 px', <span key="picosam"><InlineCode>LibrePicoSAM3()</InlineCode>: native port, ONNX-only export</span>],
              ['OmDet-Turbo', 'open-vocabulary detect', 't', <span key="omdet"><InlineCode>LibreOpenVocab(&quot;omdet-turbo&quot;)</InlineCode></span>],
              ['OV-DEIM', 'open-vocabulary detect (NMS-free)', 's / m / l', <span key="ovdeim"><InlineCode>LibreOpenVocab(&quot;ov-deim&quot;)</InlineCode>; weights CC BY-NC 4.0</span>],
              ['SenseNova Vision', '7-task multimodal preview', '7B', <span key="sense">experimental: not yet in the CLI, UI, or model inventory; weights CC BY-NC 4.0</span>],
            ]}
          />
          <ul className="space-y-2 my-4">
            <FeatureItem><strong className="text-surface-800 dark:text-white">Existing families also grew tasks:</strong> EoMT adds instance segmentation and panoptic checkpoints, RTMDet adds RTMDet-Ins instance segmentation (inference and validation), and D-FINE adds experimental segmentation with published weights.</FeatureItem>
            <FeatureItem><strong className="text-surface-800 dark:text-white">Licensing varies per family.</strong> SwinIR, EdgeTAM and Depth Anything 3 are Apache-2.0 end to end. SegFormer code is Apache-2.0 but the converted NVIDIA ADE20K weights are non-commercial (the download shows a license notice first). OV-DEIM and SenseNova weights are CC BY-NC 4.0. SAM 3 weights are gated on Hugging Face under the Meta SAM license. Check the model card before commercial use.</FeatureItem>
          </ul>

          <SubHeading>Task families beyond detection</SubHeading>
          <P>
            Families carried over from earlier releases, each documented in its
            task section. DINOv2 needs{' '}
            <InlineCode>pip install libreyolo[rfdetr]</InlineCode> (transformers).
          </P>
          <DocTable
            headers={['Family', 'Task', 'Documented in']}
            rows={[
              ['MobileNetV4 / ConvNeXt / EfficientNetV2 / ResNet', 'classify', <a key="l" href="#classification" className="text-libre-600 dark:text-libre-400 hover:underline">Classification</a>],
              ['CLIP / SigLIP2', 'zero-shot classify', <a key="l" href="#classification" className="text-libre-600 dark:text-libre-400 hover:underline">Classification</a>],
              ['DINOv2', 'semantic, classify, detect', <a key="l" href="#semantic-segmentation" className="text-libre-600 dark:text-libre-400 hover:underline">Semantic Segmentation</a>],
              ['PIDNet / EoMT / SegFormer', 'semantic', <a key="l" href="#semantic-segmentation" className="text-libre-600 dark:text-libre-400 hover:underline">Semantic Segmentation</a>],
              ['EoMT', 'panoptic', <a key="l" href="#panoptic-segmentation" className="text-libre-600 dark:text-libre-400 hover:underline">Panoptic Segmentation</a>],
              ['Depth Anything V2 / Depth Anything 3 / ZipDepth', 'depth', <a key="l" href="#depth" className="text-libre-600 dark:text-libre-400 hover:underline">Depth Estimation</a>],
              ['NAFNet / SwinIR / Real-ESRGAN', 'restore', <a key="l" href="#restoration" className="text-libre-600 dark:text-libre-400 hover:underline">Restoration &amp; Upscaling</a>],
              ['BiRefNet', 'matte', <a key="l" href="#background-removal" className="text-libre-600 dark:text-libre-400 hover:underline">Background Removal</a>],
              ['PP-OCR', 'ocr', <a key="l" href="#ocr" className="text-libre-600 dark:text-libre-400 hover:underline">OCR</a>],
              ['FOMO', 'point', <a key="l" href="#point-localization" className="text-libre-600 dark:text-libre-400 hover:underline">Point Localization</a>],
            ]}
          />
          <P className="text-sm">
            <strong className="text-surface-800 dark:text-white">Promptable, open-vocabulary and VLM tiers:</strong>{' '}
            LibreSAM (promptable segmentation, <InlineCode>libreyolo[sam]</InlineCode>),
            LibreOpenVocab (open-vocabulary detection, <InlineCode>libreyolo[openvocab]</InlineCode>)
            and the LibreVLM tier of vision-language detectors
            (<InlineCode>libreyolo[vlm]</InlineCode>) are separate categories that load
            Hugging Face snapshots and are not routed through the{' '}
            <InlineCode>LibreYOLO()</InlineCode> checkpoint factory. Their weights inherit
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

# The task suffix selects the task
model = LibreYOLO("LibreMobileNetV4s-cls.pt")   # classification (Apache, ImageNet-1k)
model = LibreYOLO("LibreDINOv2n.pt")            # semantic segmentation
model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")  # monocular depth
model = LibreYOLO("LibreFOMOs-point.pt")        # point localization (local weights)

# New in v1.4.0
model = LibreYOLO("LibreSegformerb2-sem.pt")    # semantic segmentation
model = LibreYOLO("LibreEoMTb-panoptic.pt")     # panoptic segmentation
model = LibreYOLO("LibreSwinIRm-restore.pt")    # 4x super-resolution
model = LibreYOLO("LibreBiRefNetl-matte.pt")    # background removal
model = LibreYOLO("LibrePPOCRt-ocr.pt")         # OCR (text detection + recognition)
model = LibreYOLO("LibreZipDepthb-depth.pt")    # depth

# Exported deployment formats
model = LibreYOLO("model.onnx")                 # ONNX Runtime
model = LibreYOLO("model.engine")               # TensorRT
model = LibreYOLO("model.mlpackage")            # CoreML (macOS)
model = LibreYOLO("model_openvino/")            # OpenVINO (directory)
model = LibreYOLO("model_ncnn/")                # NCNN (directory)
model = LibreYOLO("model.tflite")               # TFLite / LiteRT (new in v1.4.0)`}</CodeBlock>
          <P>
            For recognized official checkpoint filenames, LibreYOLO can auto-download
            missing weights. For custom filenames, point at an explicit local path.
            Keep new projects on YOLO9 detection or RF-DETR detection / segmentation;
            other families, tasks, and the new families are experimental in v1.4.0.
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
              ['Instance segmentation', <InlineCode key="s">&quot;segment&quot;</InlineCode>, <InlineCode key="ss">-seg</InlineCode>, 'RF-DETR, EdgeCrafter, RTMDet-Ins, D-FINE, EoMT'],
              ['Semantic segmentation', <InlineCode key="se">&quot;semantic&quot;</InlineCode>, <InlineCode key="ses">-sem</InlineCode>, 'DINOv2, PIDNet, EoMT, SegFormer'],
              ['Panoptic segmentation', <InlineCode key="pa">&quot;panoptic&quot;</InlineCode>, <InlineCode key="pas">-panoptic</InlineCode>, 'EoMT (new in v1.4.0)'],
              ['Pose estimation', <InlineCode key="p">&quot;pose&quot;</InlineCode>, <InlineCode key="ps">-pose</InlineCode>, 'YOLO-NAS, EdgeCrafter, RF-DETR (preview)'],
              ['Oriented boxes', <InlineCode key="o">&quot;obb&quot;</InlineCode>, <InlineCode key="os">-obb</InlineCode>, 'RF-DETR (preview)'],
              ['Classification', <InlineCode key="c">&quot;classify&quot;</InlineCode>, <InlineCode key="cs">-cls</InlineCode>, 'MobileNetV4, ConvNeXt, EfficientNetV2, ResNet, DINOv2; CLIP and SigLIP2 zero-shot'],
              ['Monocular depth', <InlineCode key="de">&quot;depth&quot;</InlineCode>, <InlineCode key="des">-depth</InlineCode>, 'Depth Anything V2 / 3, ZipDepth'],
              ['Image restoration', <InlineCode key="r">&quot;restore&quot;</InlineCode>, <InlineCode key="rs">-restore</InlineCode>, 'NAFNet, SwinIR, Real-ESRGAN'],
              ['Background removal', <InlineCode key="ma">&quot;matte&quot;</InlineCode>, <InlineCode key="mas">-matte</InlineCode>, 'BiRefNet (new in v1.4.0)'],
              ['OCR', <InlineCode key="oc">&quot;ocr&quot;</InlineCode>, <InlineCode key="ocs">-ocr</InlineCode>, 'PP-OCR (new in v1.4.0)'],
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
            headers={['Family', 'v1.3.1 status', 'Default', 'Supported tasks']}
            rows={[
              [<strong key="y9">YOLO9</strong>, 'detect single-GPU heavily tested; multi-GPU experimental', 'detect', 'detect'],
              [<strong key="rfd">RF-DETR</strong>, 'detect and segment single-GPU heavily tested; pose and OBB research preview', 'detect', 'detect, segment, pose, obb'],
              ['YOLOX', 'experimental', 'detect', 'detect'],
              ['YOLOv7', 'experimental; trainable as of v1.4.0', 'detect', 'detect'],
              ['YOLO9-E2E', 'experimental', 'detect', 'detect'],
              ['YOLO9-P2', 'experimental (small objects)', 'detect', 'detect'],
              ['YOLO-NAS', 'experimental; multi-class pose training new in v1.4.0', 'detect', 'detect, pose'],
              ['D-FINE', 'experimental; segment new in v1.4.0', 'detect', 'detect, segment'],
              ['DEIM / DEIMv2', 'experimental', 'detect', 'detect'],
              ['RT-DETR / RT-DETRv2 / RT-DETRv4', 'experimental', 'detect', 'detect'],
              ['PicoDet', 'experimental', 'detect', 'detect'],
              ['RTMDet', 'experimental; RTMDet-Ins segment (inference and val) new in v1.4.0', 'detect', 'detect, segment'],
              ['EdgeCrafter (EC)', 'experimental', 'detect', 'detect, pose, segment'],
              ['YOLO1 / YOLO2 / YOLO3 / YOLO4', 'museum tier (inference-only); YOLO1 new in v1.4.0', 'detect', 'detect'],
              ['PIDNet', 'experimental', 'semantic', 'semantic (inference and val only)'],
              ['EoMT', 'experimental; instance and panoptic new in v1.4.0', 'semantic', 'semantic, segment, panoptic (inference and val only)'],
              ['SegFormer', 'new in v1.4.0, experimental', 'semantic', 'semantic'],
              ['DINOv2', 'experimental', 'semantic', 'semantic, classify, detect'],
              ['MobileNetV4 / ConvNeXt / EfficientNetV2 / ResNet', 'experimental', 'classify', 'classify'],
              ['CLIP / SigLIP2', 'experimental; SigLIP2 new in v1.4.0', 'classify', 'zero-shot classify (inference-only)'],
              ['Depth Anything V2 / Depth Anything 3 / ZipDepth', 'experimental; DA3 and ZipDepth new in v1.4.0', 'depth', 'depth (inference and val only)'],
              ['NAFNet', 'experimental', 'restore', 'restore'],
              ['SwinIR / Real-ESRGAN', 'new in v1.4.0, experimental', 'restore', 'restore (inference and val only)'],
              ['BiRefNet', 'new in v1.4.0, experimental', 'matte', 'matte (inference and val only)'],
              ['PP-OCR', 'new in v1.4.0, experimental', 'ocr', 'ocr (inference and val only)'],
              ['FOMO', 'experimental', 'point', 'point'],
              ['L2CS', 'experimental', 'gaze', 'gaze (inference-only)'],
            ]}
          />
          <P>
            Three tiers sit <em>outside</em> the <InlineCode>LibreYOLO()</InlineCode> factory and are imported directly instead: <a href="#promptable-segmentation" className="text-libre-600 dark:text-libre-400 hover:underline">LibreSAM</a> (promptable segmentation, now including SAM 3, EdgeTAM and PicoSAM3), <a href="#open-vocabulary" className="text-libre-600 dark:text-libre-400 hover:underline">LibreOpenVocab</a> (open-vocabulary detection, now including OmDet-Turbo and OV-DEIM), and <a href="/docs/librevlm" className="text-libre-600 dark:text-libre-400 hover:underline">LibreVLM</a>. They are not checkpoint families, so <InlineCode>LibreYOLO(&quot;sam_b&quot;)</InlineCode> and friends will not resolve.
          </P>

          <SubHeading>Legacy YOLO baselines</SubHeading>
          <P>
            The museum tier holds the historical lineage so you can reproduce old baselines against modern ones with one API. These are <strong>inference-only</strong>: none of them can be trained in LibreYOLO, and they are not the path to pick for new work. Reach for YOLO9 or RF-DETR instead.
          </P>
          <DocTable
            headers={['Family', 'Checkpoints', 'Input size', 'Weights license']}
            rows={[
              ['LibreYOLO1 (new in v1.4.0)', 'LibreYOLO1b.pt', '448 (fixed)', 'Public domain'],
              ['LibreYOLO2', 'LibreYOLO2{t,b}.pt', '416 / 608', 'Public domain'],
              ['LibreYOLO3', 'LibreYOLO3{t,b,spp}.pt', '416 / 416 / 608', 'Public domain'],
              ['LibreYOLO4', 'LibreYOLO4{t,b}.pt', '416 / 608', 'Public domain'],
            ]}
          />
          <P>
            YOLO1 predicts the 20 VOC classes; YOLO2 / 3 / 4 are COCO-80. YOLO1 ships size <InlineCode>b</InlineCode> only: the original tiny-yolov1 weights are lost upstream. YOLOv7 left this list in v1.4.0: it is now trainable (SimOTA loss) and lives with the detection families above. It is ported from the MIT-licensed <InlineCode>MultimediaTechLab/YOLO</InlineCode>, deliberately <strong>not</strong> from the GPL-3.0 reference implementation, so it is safe to use commercially.
          </P>

          <SubHeading>YOLO9-P2, for small objects</SubHeading>
          <P>
            <InlineCode>LibreYOLO9P2</InlineCode> adds a stride-4 detection scale to YOLO9. That extra high-resolution head is what makes it worth the cost when your objects are tiny in frame, which is the classic aerial and drone-footage problem. It trains and exports like YOLO9.
          </P>
          <P>
            One published checkpoint ships: <Checkpoints names={['LibreYOLO9P2s-visdrone.pt']} link={false} />, trained on VisDrone. There is <strong>no COCO-pretrained P2 checkpoint</strong>. Note the licence carefully: the VisDrone weights are <strong>CC BY-NC-SA 3.0, so they are non-commercial</strong>. Train your own P2 weights on a permissive dataset if you need commercial use.
          </P>
          <P>
            Two rough edges remain in v1.4.0. TFLite export is not available for P2 or the museum families, and the CLI cannot resolve the variant filename, so load the VisDrone checkpoint from Python.
          </P>

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
LibreSegformerb0-sem.pt  # SegFormer requires the -sem suffix

# Panoptic segmentation (-panoptic)
LibreEoMTs-panoptic.pt

# Pose (-pose)
LibreYOLONASn-pose.pt
LibreECs-pose.pt
LibreRFDETRx-pose.pt     # preview

# Oriented boxes (-obb)
LibreRFDETRn-obb.pt      # preview

# Classification (-cls)
LibreMobileNetV4s-cls.pt
LibreConvNeXtt-cls.pt
LibreEfficientNetV2b0-cls.pt
# LibreDINOv2 classify checkpoints are not publicly shipped in v1.4.0
LibreSigLIP2b16-cls.pt   # zero-shot

# Depth (-depth)
LibreDepthAnythingV2s-depth.pt
LibreZipDepthb-depth.pt

# Restoration / super-resolution (-restore)
LibreNAFNetl-restore-sidd.pt
LibreSwinIRm-restore.pt
LibreRealESRGANx4-restore.pt

# Background removal (-matte)
LibreBiRefNetl-matte.pt

# OCR (-ocr)
LibrePPOCRt-ocr.pt

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
            The single-GPU prediction path is heavily tested for YOLO9 detection, RF-DETR detection, and RF-DETR segmentation. Other families and tasks use the same API but are experimental in v1.4.0.
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
    output_path="out/",   # images: directory; video: final file path
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
            <InlineCode>model.predict()</InlineCode> accepts a list or tuple of
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
            <InlineCode>model.info()</InlineCode> returns a JSON-friendly dict of
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
            Tiling is detection-only in v1.4.0. It rejects segmentation masks, and it cannot be combined with <InlineCode>augment=True</InlineCode>.
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
          <P>
            For video, <InlineCode>output_path</InlineCode> must be a complete
            filename such as <InlineCode>out/clip.mp4</InlineCode>, not a directory.
            In v1.4.0 the per-frame <InlineCode>Results</InlineCode> objects do not
            populate <InlineCode>saved_path</InlineCode>; use the requested path or
            the default shown above.
          </P>

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
            LibreYOLO ships four motion trackers that consume <InlineCode>Results</InlineCode> from any
            detector and add persistent track IDs: <strong className="text-surface-800 dark:text-white">ByteTrack</strong> (default),{' '}
            <strong className="text-surface-800 dark:text-white">OC-SORT</strong> (more robust to occlusion and
            non-linear motion), and, new in v1.4.0, <strong className="text-surface-800 dark:text-white">BoT-SORT</strong>{' '}
            (camera-motion compensation) and <strong className="text-surface-800 dark:text-white">Deep OC-SORT</strong>{' '}
            (appearance ReID on top of OC-SORT). Select one with{' '}
            <InlineCode>tracker=&quot;bytetrack&quot;</InlineCode> / <InlineCode>&quot;ocsort&quot;</InlineCode> /{' '}
            <InlineCode>&quot;botsort&quot;</InlineCode> / <InlineCode>&quot;deepocsort&quot;</InlineCode> on{' '}
            <InlineCode>model.track()</InlineCode>. Tracking is most tested with single-GPU YOLO9 detection and
            RF-DETR detection; other detection families are experimental in v1.4.0.
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

          <SubHeading>BoT-SORT (camera-motion compensation)</SubHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="experimental">New in v1.4.0</SupportBadge>
          </div>
          <P>
            BoT-SORT extends the ByteTrack association scheme with camera-motion
            compensation (CMC): it estimates global frame motion with sparse optical
            flow and warps predicted track positions before matching. That makes it
            the tracker to try when the <em>camera</em> moves: handheld footage,
            drones, vehicle-mounted cameras. The LibreYOLO port is motion-only
            (no ReID branch).
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO, BoTSortConfig

model = LibreYOLO("LibreYOLO9c.pt")

# Select by name with defaults
for result in model.track("drone.mp4", tracker="botsort", save=True):
    print(result.frame_idx, result.track_id)

# Or configure it: a config instance selects the tracker by type
cfg = BoTSortConfig(
    track_high_thresh=0.25,
    track_buffer=30,
    enable_cmc=True,              # camera-motion compensation on (default)
    cmc_method="sparseOptFlow",   # the shipped CMC estimator
    cmc_downscale=2,              # estimate flow at half resolution
)
for result in model.track("drone.mp4", tracker_config=cfg, save=True):
    print(result.frame_idx, result.track_id)`}</CodeBlock>
          <P>
            <InlineCode>BoTSortTracker</InlineCode> and <InlineCode>BoTSortConfig</InlineCode> are
            exported at the top level for the manual <InlineCode>tracker.update(result)</InlineCode> loop,
            same as <InlineCode>ByteTracker</InlineCode>.
          </P>

          <SubHeading>Deep OC-SORT (appearance ReID)</SubHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="experimental">New in v1.4.0</SupportBadge>
          </div>
          <P>
            Deep OC-SORT adds an appearance-embedding branch to OC-SORT, so tracks
            that disappear behind an occluder can be re-identified by how they{' '}
            <em>look</em>, not just where they were heading. The default embedder is
            an OSNet-AIN model auto-downloaded from{' '}
            <a href="https://huggingface.co/LibreYOLO/LibreReID-osnet" target="_blank" rel="noopener noreferrer" className="text-libre-600 dark:text-libre-400 hover:underline">LibreYOLO/LibreReID-osnet</a>{' '}
            on first use; it runs on the same device as the detector. That extra
            forward pass makes Deep OC-SORT the slowest of the four trackers: reach
            for it when identity switches, not speed, are your problem.
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO
from libreyolo.tracking import DeepOCSortConfig

model = LibreYOLO("LibreYOLO9c.pt")

# Defaults: OSNet-AIN embedder, auto-downloaded
for result in model.track("mall.mp4", tracker="deepocsort", save=True):
    print(result.frame_idx, result.track_id)

# Tune the appearance term, or plug in your own embedder
cfg = DeepOCSortConfig(
    det_thresh=0.25,
    embedder="osnet_ain_x0_25",   # or a callable: (frame, boxes_xyxy) -> (N, D) features
    w_association_emb=0.75,       # weight of appearance vs motion in matching
    alpha_fixed_emb=0.95,         # EMA smoothing of per-track embeddings
)
for result in model.track("mall.mp4", tracker_config=cfg, save=True):
    print(result.frame_idx, result.track_id)`}</CodeBlock>
          <P>
            <InlineCode>DeepOCSortTracker</InlineCode> and <InlineCode>DeepOCSortConfig</InlineCode>{' '}
            live in <InlineCode>libreyolo.tracking</InlineCode> (they are not top-level exports).
            Custom embedder callables are supported: pass any function that maps a frame and{' '}
            <InlineCode>(N, 4)</InlineCode> boxes to <InlineCode>(N, D)</InlineCode> features.
          </P>

          <SubHeading>Choosing a tracker</SubHeading>
          <DocTable
            headers={['Tracker', 'Select with', 'Strength', 'Cost']}
            rows={[
              ['ByteTrack', <InlineCode key="s">tracker=&quot;bytetrack&quot;</InlineCode>, 'Fast, simple, the default', 'Lowest'],
              ['OC-SORT', <InlineCode key="s">tracker=&quot;ocsort&quot;</InlineCode>, 'Occlusion and non-linear motion', 'Low'],
              ['BoT-SORT', <InlineCode key="s">tracker=&quot;botsort&quot;</InlineCode>, 'Moving cameras (CMC)', 'Medium (optical flow per frame)'],
              ['Deep OC-SORT', <InlineCode key="s">tracker=&quot;deepocsort&quot;</InlineCode>, 'Identity switches, re-ID after long occlusion', 'Highest (embedder forward pass)'],
            ]}
          />

          <Divider />

          {/* ────────────── ENSEMBLING ────────────── */}
          <SectionHeading id="ensembling" icon={Boxes}>Ensembling</SectionHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="experimental">Detection only</SupportBadge>
            <SupportBadge variant="experimental">Python API only</SupportBadge>
          </div>
          <P>
            <InlineCode>LibreEnsemble</InlineCode> runs two or more detection models and fuses their detections into one ordinary <InlineCode>Results</InlineCode>. Fusion happens at the detection level, never at the tensor level, so every member keeps its own input size, normalization and NMS. That is what lets you mix a grid detector with a DETR, or a <InlineCode>.pt</InlineCode> checkpoint with an exported backend, in the same ensemble.
          </P>
          <P>
            Class spaces do not have to match. Members are unified by class <em>name</em>: identical name maps pass straight through, otherwise LibreYOLO builds the union and remaps each member into it. Boxes are only fused with boxes of the same unified class, and a class that only one member knows passes through unfused.
          </P>

          <SubHeading>Fuse two detectors</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreEnsemble

# Weighted Boxes Fusion (the default), keep only boxes BOTH models found
ens = LibreEnsemble(["LibreYOLO9s.pt", "LibreRFDETRs.pt"], min_votes=2)

result = ens("image.jpg", conf=0.25)
print(result.boxes.xyxy)
print(result.names)     # the unified (union) class map
print(result.speed)     # per-member timings plus fusion`}</CodeBlock>

          <SubHeading>Trust weights and per-member settings</SubHeading>
          <P>
            <InlineCode>weights</InlineCode> expresses how much you trust each member (set it proportional to each model&apos;s validation mAP). <InlineCode>conf</InlineCode>, <InlineCode>iou</InlineCode> and <InlineCode>device</InlineCode> accept either one value for everyone or one value per member.
          </P>
          <CodeBlock language="python">{`ens = LibreEnsemble(
    ["LibreYOLO9s.pt", "LibreRFDETRs.pt"],
    weights=[1.0, 1.4],     # pull fused coordinates and scores toward member 2
    fusion="wbf",           # "wbf" | "wbf_seeded" | "nms" | your own callable
    fusion_iou=0.55,        # IoU used to CLUSTER boxes for fusion, not member NMS
    min_votes=1,            # keep boxes confirmed by at least N members
)

result = ens("image.jpg", conf=[0.25, 0.4])   # per-member confidence`}</CodeBlock>

          <SubHeading>Bring an outside detector</SubHeading>
          <P>
            <InlineCode>ExternalDetector</InlineCode> wraps any callable that returns boxes, so a model that is not a LibreYOLO model can still join the ensemble. The function receives a PIL image and must return boxes in original-image pixels.
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreEnsemble, ExternalDetector

def my_detector(image):
    # -> (boxes_xyxy, scores, labels) in ORIGINAL-image pixels
    return boxes, scores, labels

member = ExternalDetector(my_detector, names={0: "person"})
ens = LibreEnsemble(["LibreYOLO9s.pt", member])`}</CodeBlock>

          <SubHeading>Limits</SubHeading>
          <ul className="space-y-2 my-4">
            <FeatureItem>Detection members only. Any member whose task is not <InlineCode>detect</InlineCode> raises. Segmentation and pose models cannot be ensembled.</FeatureItem>
            <FeatureItem>At least two members are required.</FeatureItem>
            <FeatureItem><InlineCode>min_votes</InlineCode> above 1 requires a voting fusion. It raises with <InlineCode>fusion=&quot;nms&quot;</InlineCode>; use <InlineCode>wbf</InlineCode> or <InlineCode>wbf_seeded</InlineCode>.</FeatureItem>
            <FeatureItem>Images and image directories only. Video sources and <InlineCode>stream=True</InlineCode> raise: run the members individually for video.</FeatureItem>
            <FeatureItem><InlineCode>ens.val()</InlineCode> and <InlineCode>ens.export()</InlineCode> both raise. Validate and export the members individually.</FeatureItem>
            <FeatureItem><InlineCode>batch</InlineCode> is accepted for API parity but images are still processed one at a time.</FeatureItem>
          </ul>

          <Divider />

          {/* ────────────── INSTANCE SEGMENTATION ────────────── */}
          <SectionHeading id="segmentation" icon={Scissors}>Instance Segmentation</SectionHeading>
          <ValidationScopeCallout />
          <P>
            RF-DETR segmentation is the heavily tested segmentation path in v1.4.0. Around it sit four experimental options: EdgeCrafter (<InlineCode>-seg</InlineCode>), and, new in v1.4.0, RTMDet-Ins, EoMT instance segmentation, and D-FINE segmentation. YOLO9 does not ship a segmentation head: it is detect-only.
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

          <SubHeading>New segmentation families in v1.4.0</SubHeading>
          <P>
            Three families gained instance segmentation in v1.4.0, all experimental
            and all returning the same <InlineCode>boxes + masks</InlineCode> results:
          </P>
          <DocTable
            headers={['Family', 'Checkpoints', 'Scope']}
            rows={[
              ['RTMDet-Ins', <Checkpoints key="c" names={['LibreRTMDett-seg.pt', 'LibreRTMDets-seg.pt', 'LibreRTMDetm-seg.pt', 'LibreRTMDetl-seg.pt', 'LibreRTMDetx-seg.pt']} />, 'Inference and validation; training not implemented'],
              ['EoMT (instance)', <Checkpoints key="c" names={['LibreEoMTl-seg.pt', 'LibreEoMTl-seg-1280.pt']} />, 'Inference and validation; the -1280 variant trades speed for high-resolution masks'],
              ['D-FINE', <Checkpoints key="c" names={['LibreDFINEn-seg.pt', 'LibreDFINEs-seg.pt', 'LibreDFINEm-seg.pt', 'LibreDFINEl-seg.pt', 'LibreDFINEx-seg.pt']} />, 'Inference, validation and experimental training; CLI train auto-transfers detect weights to segment'],
            ]}
          />
          <P>
            D-FINE segmentation has verified parity with its ONNX and TensorRT
            exports. Note that tiled inference (<InlineCode>tiling=True</InlineCode>)
            rejects segmentation models loudly rather than silently dropping masks.
          </P>

          <SubHeading>Training segmentation</SubHeading>
          <P>
            RF-DETR segmentation uses the RF-DETR COCO-format training pipeline and is part of the heavily tested single-GPU scope. EdgeCrafter and D-FINE segmentation training are available but experimental. For segmentation-specific augmentation (copy-paste), see{' '}
            <a href="#augmentation" className="text-libre-600 dark:text-libre-400 hover:underline">Data Augmentation</a>.
          </P>

          <Divider />

          {/* ────────────── SEMANTIC SEGMENTATION ────────────── */}
          <SectionHeading id="semantic-segmentation" icon={Palette}>Semantic Segmentation</SectionHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="experimental">Experimental</SupportBadge>
            <SupportBadge variant="experimental">SegFormer + TTA new in v1.4.0</SupportBadge>
          </div>
          <P>
            Semantic segmentation labels <em>every pixel</em> with a class. It is a different task from instance segmentation: there are no object instances and no boxes, just one dense class map. Pass <InlineCode>task=&quot;semantic&quot;</InlineCode> (aliases: <InlineCode>semseg</InlineCode>, <InlineCode>sem</InlineCode>), and read the result from <InlineCode>result.semantic_mask</InlineCode>. On a semantic model <InlineCode>result.boxes</InlineCode> and <InlineCode>result.masks</InlineCode> are both <InlineCode>None</InlineCode>.
          </P>

          <SubHeading>Models</SubHeading>
          <DocTable
            headers={['Family', 'Checkpoints', 'Backbone', 'Trained on', 'Classes', 'Train?']}
            rows={[
              ['LibrePIDNet', 'LibrePIDNet{s,m,l}-sem.pt', 'PIDNet 3-branch CNN', 'Cityscapes', '19', 'No'],
              ['LibreEoMT', 'LibreEoMTl-sem.pt', 'DINOv2 ViT-L', 'ADE20K', '150', 'No'],
              ['LibreSegformer (new in v1.4.0)', 'LibreSegformer{b0..b5}-sem.pt', 'MiT hierarchical transformer', 'ADE20K', '150', 'Yes (fine-tune)'],
              ['LibreDINOv2', 'none published: you train it', 'DINOv2 + dense head', 'your data', 'you choose', 'Yes'],
            ]}
          />
          <P>
            The families behave quite differently, so pick deliberately. <InlineCode>LibrePIDNet</InlineCode> is a fast real-time CNN carrying Cityscapes road-scene classes. <InlineCode>LibreEoMT</InlineCode> carries ADE20K&apos;s 150 general scene classes. Both ship pretrained weights and <strong>cannot be trained</strong> inside LibreYOLO: fine-tune them upstream and convert the result.
          </P>
          <P>
            <InlineCode>LibreSegformer</InlineCode>, new in v1.4.0, is the middle path: six sizes (b0 to b5, 512 px; b5 at 640) of the SegFormer architecture with a bit-exact reference port, ADE20K pretrained weights, <strong>and</strong> a fine-tune trainer, so you can start from 150 general classes and adapt to your own. One licensing caveat: the code is Apache-2.0 but the converted NVIDIA ADE20K weights are <strong>non-commercial</strong>, and the download shows a license notice before fetching them.
          </P>
          <P>
            <InlineCode>LibreDINOv2</InlineCode> is the fine-tuning family without pretrained-head baggage: there is <strong>no published LibreDINOv2 semantic checkpoint</strong>. You construct it from the pretrained DINOv2 backbone with a fresh dense head and train it on your own masks. Reach for it when your classes are not Cityscapes or ADE20K and you want the strongest features under a fresh head.
          </P>

          <SubHeading>Run semantic segmentation</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibrePIDNets-sem.pt")   # Cityscapes, 19 classes
result = model.predict("street.jpg", save=True)

sm = result.semantic_mask     # SemanticMask
print(sm.data.shape)          # (H, W) int class ids, on the ORIGINAL image canvas
print(sm.classes)             # sorted class ids present, 255 (ignore) excluded
print(model.names[13])        # 'car'

car = sm.class_mask(13)       # (H, W) bool mask for one class

print(result.saved_path)      # saved semantic overlay

print(result.boxes, result.masks)   # None None: semantic has no instances`}</CodeBlock>

          <SubHeading>SemanticMask API</SubHeading>
          <CodeBlock language="python">{`sm = result.semantic_mask

sm.data               # (H, W) integer class ids at original resolution
sm.orig_shape         # (H, W)
sm.classes            # list[int] of ids present, excluding the ignore index
sm.class_mask(cid)    # (H, W) bool
SemanticMask.IGNORE_INDEX   # 255: the void label, never counted as a class

sm.cpu(); sm.numpy()`}</CodeBlock>

          <SubHeading>Validate</SubHeading>
          <P>Validation reports mean IoU and pixel accuracy. Classes never seen in either the prediction or the ground truth are excluded from the mean rather than scored as zero. <InlineCode>fitness</InlineCode> is an alias of mIoU, so it is what drives best-checkpoint selection during training.</P>
          <CodeBlock language="python">{`metrics = model.val(data="cityscapes.yaml")
print(metrics["metrics/mIoU"])
print(metrics["metrics/pixel_accuracy"])`}</CodeBlock>
          <CodeBlock language="bash">{`libreyolo val model=LibrePIDNets-sem.pt data=cityscapes.yaml split=val`}</CodeBlock>

          <SubHeading>Train (LibreDINOv2)</SubHeading>
          <P>
            Masks are single-channel lossless images whose pixel value is the class id, paired to each image by filename stem. <InlineCode>255</InlineCode> means ignore and is excluded from both loss and metrics.
          </P>
          <CodeBlock language="bash">{`dataset/
    images/train/*.jpg
    images/val/*.jpg
    masks/train/*.png      # same stem as the image; pixel value = class id
    masks/val/*.png`}</CodeBlock>
          <CodeBlock language="python">{`from libreyolo import LibreDINOv2

# model_path=None -> pretrained DINOv2 backbone + a fresh dense head
model = LibreDINOv2(model_path=None, size="s", task="semantic", nb_classes=19)
model.train(data="cityscapes.yaml", epochs=100, batch_size=4, lr=1e-4)`}</CodeBlock>
          <P>
            In the dataset YAML, <InlineCode>masks_dir</InlineCode> names the mask directory (default <InlineCode>masks</InlineCode>). If you omit it, LibreYOLO rasterizes masks from YOLO polygon labels at load time and appends a <InlineCode>background</InlineCode> class. <InlineCode>label_mapping</InlineCode> remaps source pixel values to training ids, and anything unmapped becomes ignore.
          </P>

          <SubHeading>Test-time augmentation (new in v1.4.0)</SubHeading>
          <P>
            Semantic models now accept <InlineCode>augment=True</InlineCode> on both{' '}
            <InlineCode>predict()</InlineCode> and <InlineCode>val()</InlineCode> (this raised in
            v1.3.1). TTA runs a horizontal-flip pass and averages logits, trading
            roughly 2x inference cost for a small, reliable mIoU gain. It is
            implemented for PIDNet, SegFormer, EoMT and DINOv2 semantic.
          </P>
          <CodeBlock language="python">{`model = LibreYOLO("LibreSegformerb2-sem.pt")
result = model.predict("street.jpg", augment=True)   # flip-TTA
metrics = model.val(data="ade20k.yaml", augment=True)`}</CodeBlock>

          <SubHeading>Limits</SubHeading>
          <ul className="space-y-2 my-4">
            <FeatureItem><strong>Export support is family-specific.</strong> PIDNet supports ONNX, TorchScript, NCNN, and TFLite. DINOv2 and EoMT semantic support ONNX and TorchScript. SegFormer export remains blocked. Check <InlineCode>libreyolo formats --family ...</InlineCode> for the exact tier.</FeatureItem>
            <FeatureItem><strong>Only LibreDINOv2 and LibreSegformer train.</strong> <InlineCode>LibrePIDNet.train()</InlineCode> and <InlineCode>LibreEoMT.train()</InlineCode> raise.</FeatureItem>
            <FeatureItem>Semantic training now applies <strong>HSV color jitter by default</strong> (new in v1.4.0), so retrained mIoU can shift slightly versus v1.3.1 runs. The knob comes from the family, not <InlineCode>hsv_prob</InlineCode>; see <a href="#augmentation" className="text-libre-600 dark:text-libre-400 hover:underline">Data Augmentation</a>.</FeatureItem>
            <FeatureItem><strong>EoMT semantic is size <InlineCode>l</InlineCode> only and locked to <InlineCode>imgsz=512</InlineCode></strong> (its checkpoint uses fixed position embeddings), and it cannot batch: <InlineCode>val(batch=N)</InlineCode> warns and still runs one image at a time.</FeatureItem>
            <FeatureItem><InlineCode>imgsz</InlineCode> divisibility differs per family: PIDNet needs a multiple of 8, EoMT of 16, DINOv2 of 14, and SegFormer of 32. Violations raise.</FeatureItem>
            <FeatureItem>No tracking for semantic models.</FeatureItem>
            <FeatureItem>Cityscapes, ADE20K and COCO-Stuff all require a manual download. LibreYOLO ships the dataset YAMLs, not the data.</FeatureItem>
            <FeatureItem>Raw upstream checkpoints are rejected. Convert with the <InlineCode>weights/convert_*_weights.py</InlineCode> scripts.</FeatureItem>
          </ul>

          <Divider />

          {/* ────────────── PANOPTIC SEGMENTATION ────────────── */}
          <SectionHeading id="panoptic-segmentation" icon={Combine}>Panoptic Segmentation</SectionHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="experimental">New in v1.4.0</SupportBadge>
            <SupportBadge variant="experimental">Inference and val only</SupportBadge>
          </div>
          <P>
            Panoptic segmentation answers both questions at once: every pixel gets a
            class (like semantic segmentation), <em>and</em> countable objects are
            separated into instances (like instance segmentation). Roads and sky come
            back as single &quot;stuff&quot; segments; each car and person comes back as its
            own &quot;thing&quot; segment. The result is one segment-id map plus a per-segment
            info list, read from <InlineCode>result.panoptic</InlineCode>.
          </P>
          <P>
            One family ships the task in v1.4.0: <InlineCode>LibreEoMT</InlineCode> with
            COCO-panoptic checkpoints (133 classes, 640 px) in three sizes:{' '}
            <Checkpoints names={['LibreEoMTs-panoptic.pt', 'LibreEoMTb-panoptic.pt', 'LibreEoMTl-panoptic.pt']} />.
          </P>

          <SubHeading>Run panoptic segmentation</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreEoMTb-panoptic.pt")   # task resolved from the -panoptic suffix
result = model.predict("street.jpg", save=True)

pan = result.panoptic             # PanopticSegmentation
print(pan.data.shape)             # (H, W) integer segment ids, original canvas
for seg in pan.segments_info:     # one dict per segment
    print(seg["id"], seg["category_id"], model.names[seg["category_id"]])

car_mask = pan.segment_mask(3)    # (H, W) bool mask for one segment id
print(result.saved_path)          # saved panoptic overlay

# Flip-TTA works here too (new in v1.4.0)
result = model.predict("street.jpg", augment=True)`}</CodeBlock>

          <SubHeading>PanopticSegmentation API</SubHeading>
          <CodeBlock language="python">{`pan = result.panoptic
pan.data                 # (H, W) int segment-id map at original resolution
pan.segments_info        # list of {"id", "category_id", ...} dicts
pan.segment_ids          # ids present in the map
pan.segment_mask(sid)    # (H, W) bool mask for one segment

pan.cpu(); pan.numpy()`}</CodeBlock>

          <SubHeading>Validate with Panoptic Quality</SubHeading>
          <P>
            Validation runs through <InlineCode>PanopticValidator</InlineCode> and reports
            Panoptic Quality (PQ), the standard metric that multiplies segmentation
            quality (average IoU of matched segments) by recognition quality
            (F1 over segments). <InlineCode>augment=True</InlineCode> is accepted for
            flip-TTA.
          </P>
          <CodeBlock language="python">{`metrics = model.val(data="coco_panoptic.yaml")
print(metrics["metrics/PQ"])`}</CodeBlock>

          <SubHeading>Limits</SubHeading>
          <ul className="space-y-2 my-4">
            <FeatureItem><strong>Inference and validation only.</strong> Panoptic training and export both raise in v1.4.0.</FeatureItem>
            <FeatureItem>Checkpoints written with the <InlineCode>panoptic</InlineCode> task string are not loadable by v1.3.1.</FeatureItem>
            <FeatureItem>On a panoptic model, <InlineCode>result.boxes</InlineCode> and <InlineCode>result.masks</InlineCode> are <InlineCode>None</InlineCode>: everything lives in <InlineCode>result.panoptic</InlineCode>.</FeatureItem>
          </ul>

          <Divider />

          {/* ────────────── PROMPTABLE SEGMENTATION (SAM) ────────────── */}
          <SectionHeading id="promptable-segmentation" icon={MousePointerClick}>Promptable Segmentation</SectionHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="experimental">Python API only</SupportBadge>
            <SupportBadge variant="experimental">Inference only</SupportBadge>
            <SupportBadge variant="experimental">SAM 3, EdgeTAM, PicoSAM3 new in v1.4.0</SupportBadge>
          </div>
          <P>
            LibreSAM is a separate tier from the detector factory, because a promptable segmenter has a different contract: it runs a heavy image encoder once, then answers cheap spatial prompts (a click, a box) with a mask. There is no fixed class list. Install it with <InlineCode>pip install &quot;libreyolo[sam]&quot;</InlineCode>.
          </P>
          <P>
            Two things surprise people. First, <InlineCode>LibreSAM</InlineCode> is a <strong>factory function, not a class</strong>, and it is deliberately kept <strong>outside</strong> the <InlineCode>LibreYOLO()</InlineCode> loader, so <InlineCode>LibreYOLO(&quot;sam_b&quot;)</InlineCode> does not work. Import it directly. Second, the whole tier is <strong>Python-only</strong>: there is no CLI path to it.
          </P>

          <SubHeading>Models</SubHeading>
          <DocTable
            headers={['Family', 'Pass to LibreSAM()', 'Encoder', 'Notes']}
            rows={[
              ['SAM-1', '"base" (default), "large", "huge"', 'ViT-B / L / H', 'Apache-2.0'],
              ['SAM-2.1', '"sam2-tiny", "sam2-small", "sam2-base-plus", "sam2-large"', 'Hiera', 'Images only, no video'],
              ['MobileSAM', '"mobilesam"', 'TinyViT', 'Fastest; native LibreYOLO port'],
            ]}
          />
          <P>
            Those short aliases only work through the <InlineCode>LibreSAM()</InlineCode> factory. The concrete classes take canonical sizes, so <InlineCode>LibreSAM1(&quot;base&quot;)</InlineCode> is right and <InlineCode>LibreSAM1(&quot;sam_b&quot;)</InlineCode> raises.
          </P>

          <SubHeading>New in v1.4.0: SAM 3, EdgeTAM, PicoSAM3</SubHeading>
          <P>
            Three additions span the quality / speed spectrum. Construct them
            directly; snapshots download on first use.
          </P>
          <DocTable
            headers={['Model', 'Construct', 'Working size', 'Notes']}
            rows={[
              ['SAM 3', <InlineCode key="c">LibreSAM3()</InlineCode>, '1008', 'Highest quality; transformers-backed. Weights are gated on Hugging Face under the Meta SAM license: accept the terms and log in before first use.'],
              ['EdgeTAM', <InlineCode key="c">LibreEdgeTAM()</InlineCode>, '1024', 'Edge-oriented; image inference only; Apache-2.0 end to end.'],
              ['PicoSAM3', <InlineCode key="c">LibrePicoSAM3()</InlineCode>, '96', 'Native tiny port for ROI segmentation on very small crops; the only SAM-tier model with export (ONNX only).'],
            ]}
          />
          <CodeBlock language="python">{`from libreyolo import LibreSAM3, LibreEdgeTAM, LibrePicoSAM3

model = LibreSAM3()               # gated HF weights (Meta SAM license)
r = model.predict("img.jpg", points=[640, 360], labels=[1])

model = LibreEdgeTAM()            # Apache-2.0, edge-friendly
r = model.predict("img.jpg", bboxes=[100, 100, 500, 500])

model = LibrePicoSAM3()           # 96 px ROI segmenter, ONNX-exportable
r = model.predict("crop.jpg", bboxes=[8, 8, 88, 88])`}</CodeBlock>

          <SubHeading>Prompt with a click or a box</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreSAM

model = LibreSAM("base")          # SAM-1 ViT-B

# a single click
r = model.predict("img.jpg", points=[640, 360], labels=[1])
print(r.masks.data.shape)   # (1, H, W) bool, at the original resolution
print(r.boxes.xyxy)         # a tight box derived from the mask
print(r.boxes.conf)         # SAM's predicted mask quality, NOT a detection score

# a box prompt
r = model.predict("img.jpg", bboxes=[100, 100, 500, 500])

# segment everything (a coarse grid); lower the grid on CPU, it is slow
r = model.predict("img.jpg", points_per_side=16)`}</CodeBlock>

          <SubHeading>Encode once, prompt many times</SubHeading>
          <P>This is the pattern that makes interactive use fast. The expensive encoder runs once per image and every later prompt reuses the embedding.</P>
          <CodeBlock language="python">{`model.set_image("img.jpg")                        # heavy encoder runs ONCE
a = model.predict(points=[500, 375], labels=[1])  # cheap: decoder only
b = model.predict(bboxes=[100, 100, 200, 200])    # cheap: reuses the embedding
model.reset_image()`}</CodeBlock>

          <SubHeading>How prompts are shaped</SubHeading>
          <P>
            Nesting depth carries meaning, and this is the single easiest thing to get wrong. Points are plain <InlineCode>[x, y]</InlineCode> pixels. A label of <InlineCode>1</InlineCode> means include, <InlineCode>0</InlineCode> means exclude.
          </P>
          <DocTable
            headers={['You pass', 'It means']}
            rows={[
              ['points=[x, y]', 'one object, one point'],
              ['points=[[x, y], [x, y]]', 'TWO objects, one point each'],
              ['points=[[[x, y], [x, y]]]', 'ONE object, two points'],
            ]}
          />
          <CodeBlock language="python">{`# refine ONE object with a positive and a negative click
r = model.predict(
    "img.jpg",
    points=[[[500, 375], [620, 400]]],   # one object, two points
    labels=[1, 0],                        # include, then exclude
)

# all three whole-vs-part candidate masks for an ambiguous click
r = model.predict("img.jpg", points=[640, 360], labels=[1], multimask=True)`}</CodeBlock>

          <SubHeading>Limits</SubHeading>
          <ul className="space-y-2 my-4">
            <FeatureItem><strong>Images only, across the tier.</strong> There is no video segmentation and no memory propagation across frames in v1.4.0 (this includes SAM 2, SAM 3 and EdgeTAM): <InlineCode>track()</InlineCode> raises. Call <InlineCode>predict()</InlineCode> per frame.</FeatureItem>
            <FeatureItem><strong>No training and no validation</strong> for any SAM family. Export raises everywhere except PicoSAM3, which exports to ONNX only.</FeatureItem>
            <FeatureItem><strong>PicoSAM3 accepts only <InlineCode>bboxes=</InlineCode> ROI prompts.</strong> Use LibreSAM2 or LibreSAM3 for points, text, masks, or segment-everything.</FeatureItem>
            <FeatureItem>Mask prompts (<InlineCode>masks=</InlineCode>) are not supported and raise. Use points or boxes.</FeatureItem>
            <FeatureItem><InlineCode>conf</InlineCode> here filters on SAM&apos;s predicted <em>mask quality</em>, not on detection confidence. Detector intuition does not transfer.</FeatureItem>
            <FeatureItem>Everything runs in fp32, even on CUDA. This is deliberate: half precision rounds prompt coordinates by several pixels at SAM&apos;s 1024px working size, which silently moves where you clicked.</FeatureItem>
            <FeatureItem>Segment-everything is a simplified grid, not the reference automatic mask generator. It under-segments crowded scenes.</FeatureItem>
            <FeatureItem>Weights download into <InlineCode>./weights/</InlineCode> relative to your working directory, so running from elsewhere re-downloads.</FeatureItem>
          </ul>

          <Divider />

          {/* ────────────── OPEN-VOCABULARY DETECTION ────────────── */}
          <SectionHeading id="open-vocabulary" icon={Search}>Open-Vocabulary Detection</SectionHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="experimental">Python API only</SupportBadge>
            <SupportBadge variant="experimental">OmDet-Turbo + OV-DEIM new in v1.4.0</SupportBadge>
          </div>
          <P>
            Give the model a list of class names as text and get real detection boxes back. No training, no labelled data. Change the list and you change what it detects. Install with <InlineCode>pip install &quot;libreyolo[openvocab]&quot;</InlineCode>.
          </P>
          <P>
            This is not the same as the <a href="/docs/librevlm" className="text-libre-600 dark:text-libre-400 hover:underline">LibreVLM</a> tier, and the difference matters. These are purpose-built <em>detectors</em> conditioned on text: the detector head returns boxes with real model scores. A VLM instead generates text that LibreYOLO parses into boxes. The rule of thumb: <strong>boxes for named classes, use open-vocab; describe or instruct, use a VLM</strong>. On licensing, check per family: Grounding DINO, OWLv2 and OmDet-Turbo weights are Apache-2.0, but the OV-DEIM weights are CC BY-NC 4.0 (non-commercial), confirmed with the upstream author.
          </P>

          <SubHeading>Models</SubHeading>
          <DocTable
            headers={['Pass to LibreOpenVocab()', 'Class', 'Backbone', 'Default conf']}
            rows={[
              ['"grounding-dino" (default, tiny)', 'LibreGroundingDINO', 'Swin-T + BERT', '0.25'],
              ['"grounding-dino-base"', 'LibreGroundingDINO', 'Swin-B + BERT', '0.25'],
              ['"owlv2"', 'LibreOWLv2', 'ViT-B/16', '0.1'],
              ['"owlv2-large"', 'LibreOWLv2', 'ViT-L/14', '0.1'],
              ['"omdet-turbo" (new in v1.4.0)', 'LibreOMDetTurbo', 'Swin-T, transformers-backed', '0.25'],
              ['"ov-deim" / "-m" / "-l" (new in v1.4.0)', 'LibreOVDEIM', 'DEIM, native NMS-free port', '0.25'],
            ]}
          />

          <SubHeading>Detect anything you can name</SubHeading>
          <P>
            The vocabulary is set on the <em>model</em>, with <InlineCode>set_classes()</InlineCode>, and it is sticky across later calls. There is no <InlineCode>prompts=</InlineCode> or <InlineCode>text=</InlineCode> argument on <InlineCode>predict()</InlineCode>.
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreOpenVocab

model = LibreOpenVocab("grounding-dino")
model.set_classes(["person", "dog", "skateboard"])   # sticky vocabulary

result = model.predict("street.jpg", conf=0.25, text_threshold=0.25)
print(result.boxes.xyxy, result.boxes.conf)
print(result.names)     # {0: 'person', 1: 'dog', 2: 'skateboard'}

result = model.predict("another.jpg")   # same vocabulary, still set

# or set it at construction
model = LibreOpenVocab("owlv2", names=["forklift", "pallet"])`}</CodeBlock>
          <P>
            <strong>Watch out for the lookalike.</strong> <InlineCode>predict(classes=...)</InlineCode> is <em>not</em> the text API. It is the standard integer class-id filter and takes a list of ints. The text vocabulary goes through <InlineCode>set_classes()</InlineCode>.
          </P>

          <SubHeading>Practical notes</SubHeading>
          <ul className="space-y-2 my-4">
            <FeatureItem>Short noun phrases work best. &quot;remote control&quot; beats &quot;remote&quot;. Phrases that cannot be mapped back to one of your class names unambiguously are dropped, so a missing detection is sometimes a mapping drop rather than a detector miss.</FeatureItem>
            <FeatureItem>There is no cap on how many classes you may pass. Grounding DINO automatically splits a long vocabulary into chunks that fit its text encoder and runs one forward pass per chunk, so <strong>cost grows with vocabulary size</strong>. That is the main latency knob you control.</FeatureItem>
            <FeatureItem><InlineCode>text_threshold</InlineCode> is Grounding DINO only. Passing it to the other families raises.</FeatureItem>
            <FeatureItem>The families score differently, so tune <InlineCode>conf</InlineCode> per family rather than reusing a number.</FeatureItem>
            <FeatureItem><strong>OV-DEIM</strong> is the interesting speed option: a native, NMS-free port (not a transformers pipeline) in three sizes. Its text features are cached per vocabulary, and v1.4.0 fixed the device-switch crash on that cache. Remember the weights are non-commercial.</FeatureItem>
            <FeatureItem><strong>OmDet-Turbo</strong> is transformers-backed and respects <InlineCode>iou=</InlineCode> (it did not before v1.4.0).</FeatureItem>
            <FeatureItem>Expect this to be far slower than a LibreYOLO detector. The honest workflow: use open-vocab to explore or auto-label an open vocabulary, then train a fast detector on the result.</FeatureItem>
          </ul>

          <SubHeading>Limits</SubHeading>
          <ul className="space-y-2 my-4">
            <FeatureItem><strong>No CLI.</strong> <InlineCode>libreyolo predict model=grounding-dino</InlineCode> does not work. This tier is reachable only from Python.</FeatureItem>
            <FeatureItem><strong>No training, no validation, no export, no tracking.</strong> All four raise.</FeatureItem>
            <FeatureItem><InlineCode>imgsz</InlineCode> and <InlineCode>augment=True</InlineCode> are rejected: the processor owns resizing. <InlineCode>iou</InlineCode> is accepted but ignored, since no LibreYOLO NMS runs here.</FeatureItem>
            <FeatureItem>Batching gives no speedup: images run one at a time. Everything is fp32.</FeatureItem>
          </ul>

          <Divider />

          {/* ────────────── POSE ESTIMATION ────────────── */}
          <SectionHeading id="pose" icon={PersonStanding}>Pose Estimation</SectionHeading>
          <P>
            Pose (keypoint) estimation runs on <InlineCode>YOLO-NAS (-pose)</InlineCode>,{' '}
            <InlineCode>EdgeCrafter (-pose)</InlineCode>, and an{' '}
            <InlineCode>RF-DETR (-pose)</InlineCode> preview. The published checkpoints are
            single-class (&quot;person&quot;) with 17 COCO keypoints. New in v1.4.0, YOLO-NAS
            pose supports <strong>multi-class keypoint training</strong>: train on a dataset
            with several object classes, and multi-class checkpoints now load with their
            real class count and return real class ids (they were previously forced to
            single-class person).
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
            RF-DETR pose (ported from GroupPose) remains a research preview in v1.4.0.
          </P>
          <CodeBlock language="python">{`# RF-DETR pose preview
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
            Pose training is supported for YOLO-NAS (including multi-class datasets as of v1.4.0); EdgeCrafter pose is currently inference-only. RF-DETR pose is a preview. YOLO9 is detect-only and ships no pose checkpoints. Pose validation under multi-GPU DDP was fixed in v1.4.0 (per-rank file clobbering and a collective deadlock).
          </P>

          <Divider />

          {/* ────────────── GAZE ESTIMATION ────────────── */}
          <SectionHeading id="gaze" icon={Eye}>Gaze Estimation</SectionHeading>
          <P>
            Gaze direction estimation is provided by the <InlineCode>LibreL2CS</InlineCode> family, an L2CS-Net port with a ResNet trunk and two angle-bin classification heads. It is a two-stage model: an upstream face detector locates faces, then the gaze head predicts per-face pitch and yaw in radians. It is inference-only and experimental in v1.4.0. (v1.4.0 also fixed face detection on OpenCV 5 by adding a YuNet detector.)
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
            Whole-image classification spans two supervised paths and a zero-shot path. <InlineCode>LibreMobileNetV4</InlineCode> is the production classifier (Apache-2.0 ImageNet-1k weights, exportable to ONNX), with <InlineCode>LibreConvNeXt</InlineCode>, <InlineCode>LibreEfficientNetV2</InlineCode> and <InlineCode>LibreResNet</InlineCode> as alternatives on the same API. <InlineCode>LibreDINOv2</InlineCode> with <InlineCode>task=classify</InlineCode> is a DINOv2 backbone plus linear probe for transfer learning. v1.4.0 does not publish public <InlineCode>-cls</InlineCode> checkpoints for it, so construct and train a fresh head; a trained classifier can export to ONNX. For zero-shot (no training, labels as text), use CLIP or, new in v1.4.0, SigLIP2. Classification training gained its own <a href="#augmentation" className="text-libre-600 dark:text-libre-400 hover:underline">augmentation pack</a> in v1.4.0: <InlineCode>auto_augment</InlineCode>, <InlineCode>erasing</InlineCode>, <InlineCode>mixup</InlineCode> and <InlineCode>cutmix</InlineCode>.
          </P>

          <DocTable
            headers={['Family', 'Checkpoints', 'Input', 'Weights', 'Fine-tune', 'ONNX export']}
            rows={[
              ['LibreMobileNetV4', 'LibreMobileNetV4{s,m,l}-cls.pt', '224 / 224 / 256', 'Apache-2.0 ImageNet-1k (production)', 'Cross-entropy', 'Yes'],
              ['LibreDINOv2 (classify)', 'Not publicly shipped in v1.4.0', '224', 'Fresh linear head', 'Linear probe', 'Yes'],
            ]}
          />

          <SubHeading>LibreMobileNetV4 (production classifier)</SubHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="validated">Apache-2.0 ImageNet-1k weights</SupportBadge>
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
            <SupportBadge variant="experimental">No public v1.4.0 checkpoints</SupportBadge>
            <SupportBadge variant="validated">ONNX export</SupportBadge>
          </div>
          <P>
            A DINOv2-S encoder with a trainable linear head, run at 224. The <InlineCode>n</InlineCode> / <InlineCode>s</InlineCode> / <InlineCode>m</InlineCode> / <InlineCode>l</InlineCode> sizes control the projector width; all four share the same DINOv2-S encoder. LibreYOLO v1.4.0 does not publicly host <InlineCode>LibreDINOv2*-cls.pt</InlineCode> checkpoints. Build a fresh classifier and train it on your own ImageFolder dataset.
          </P>

          <P>
            Build a fresh model with <InlineCode>task=&quot;classify&quot;</InlineCode>, train the new head, then use the same <InlineCode>Probs</InlineCode> prediction surface as MobileNetV4.
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreDINOv2

# Fresh DINOv2 backbone + random linear head, sized to the dataset
model = LibreDINOv2(size="s", task="classify", nb_classes=3)
model.train(data="path/to/imagefolder", epochs=5, lr=1e-4, batch=4)

# Validate the same way (top-1 / top-5)
metrics = model.val(data="path/to/imagefolder")
print(metrics["metrics/accuracy_top1"])

result = model("springer.jpg")
print(result.probs.top1, result.probs.top1conf)

model.export(format="onnx", imgsz=224)`}</CodeBlock>

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

          <SubHeading>Zero-shot classification: SigLIP2 and CLIP</SubHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="experimental">SigLIP2 new in v1.4.0</SupportBadge>
            <SupportBadge variant="experimental">Inference-only</SupportBadge>
          </div>
          <P>
            Zero-shot classifiers score an image against <em>text labels you choose at
            runtime</em>: no training, no fixed class list. v1.4.0 adds{' '}
            <InlineCode>LibreSigLIP2</InlineCode> (sizes <InlineCode>b16</InlineCode> and{' '}
            <InlineCode>so400m</InlineCode>, native torch port) alongside the existing CLIP
            family. Both load through the factory and set their vocabulary with{' '}
            <InlineCode>set_classes()</InlineCode>; SigLIP2 needs{' '}
            <InlineCode>pip install &quot;libreyolo[siglip2]&quot;</InlineCode> for its
            SentencePiece tokenizer.
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreSigLIP2b16-cls.pt")
model.set_classes(["a forklift", "an empty aisle", "a spill"])

result = model.predict("warehouse.jpg")
print(model.names[result.probs.top1], float(result.probs.top1conf))

# Independent per-label probabilities (sigmoid) instead of softmax
result = model.predict("warehouse.jpg", multi_label=True)`}</CodeBlock>
          <P>
            SigLIP2&apos;s sigmoid training objective makes its{' '}
            <InlineCode>multi_label=True</InlineCode> scores meaningful on their own, which
            CLIP-style softmax scores are not: use it when several labels can be true at
            once. <InlineCode>train()</InlineCode> raises for both zero-shot families.
          </P>

          <ul className="space-y-2 my-4">
            <FeatureItem>MobileNetV4 weights are production grade (Apache-2.0 ImageNet-1k, bit-identical load). LibreDINOv2 classify has no publicly hosted v1.4.0 checkpoint; train a fresh head.</FeatureItem>
            <FeatureItem>There is no LibreRFDETR classifier since v1.3.0. Classification moved into the dedicated classifier families; legacy LibreRFDETR*-cls checkpoints are rejected on load.</FeatureItem>
            <FeatureItem>ONNX classify output is raw logits. Apply softmax in non-Python consumers.</FeatureItem>
            <FeatureItem>Predicting a single image returns one Results. Read result.probs directly, or pass a list and index the list: model([&quot;a.jpg&quot;])[0].probs.</FeatureItem>
            <FeatureItem>New in v1.4.0: <InlineCode>square_resize</InlineCode> together with <InlineCode>augment</InlineCode> now raises instead of silently misbehaving, and the classifier families train multi-GPU via the spawn path.</FeatureItem>
          </ul>

          <Divider />

          {/* ────────────── DEPTH ESTIMATION ────────────── */}
          <SectionHeading id="depth" icon={Mountain}>Depth Estimation</SectionHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="experimental">ZipDepth + Depth Anything 3 new in v1.4.0</SupportBadge>
            <SupportBadge variant="experimental">Inference and val only</SupportBadge>
          </div>
          <P>
            Monocular depth predicts a dense relative inverse-depth map: higher values
            are closer to the camera, with no metric unit implied. v1.4.0 has three
            depth families behind one API, differing mainly in size, license and
            target hardware:
          </P>
          <DocTable
            headers={['Family', 'Sizes (input)', 'License', 'Checkpoints']}
            rows={[
              ['LibreDepthAnythingV2', 's / b / l / g (518)', 's Apache-2.0; b / l / g CC-BY-NC-4.0', <span key="c"><Checkpoints names={['LibreDepthAnythingV2s-depth.pt', 'LibreDepthAnythingV2b-depth.pt', 'LibreDepthAnythingV2l-depth.pt']} />; g converts from upstream</span>],
              ['LibreDepthAnything3 (new)', 'l (504)', 'Apache-2.0', <Checkpoints key="c" names={['LibreDepthAnything3l-depth.pt']} />],
              ['LibreZipDepth (new)', 'b / bnpu (384)', 'MIT', <Checkpoints key="c" names={['LibreZipDepthb-depth.pt', 'LibreZipDepthbnpu-depth.pt']} />],
            ]}
          />
          <P>
            <InlineCode>LibreDepthAnything3</InlineCode> is a separate family from V2 (not an
            upgrade in place), with a single Apache-2.0 large checkpoint: the quality
            pick when the license matters. <InlineCode>LibreZipDepth</InlineCode> is the
            efficiency pick: MIT-licensed, 384 px, with a <InlineCode>bnpu</InlineCode>{' '}
            variant whose decoder avoids NPU-hostile ops for edge accelerators.
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

          <SubHeading>Export (new in v1.4.0)</SubHeading>
          <P>
            Depth export was unblocked in v1.4.0 for Depth Anything V2 and ZipDepth
            under a <strong>fixed-resolution, batch-1 contract</strong>: the exported graph
            bakes in one input size and no dynamic axes. Depth Anything 3 does not
            export yet.
          </P>
          <CodeBlock language="python">{`model = LibreYOLO("LibreZipDepthb-depth.pt")
model.export(format="onnx")   # fixed resolution, batch 1`}</CodeBlock>

          <SubHeading>Not supported</SubHeading>
          <CodeBlock language="python">{`model.train(data="...")   # raises NotImplementedError - all depth families are inference + val only`}</CodeBlock>

          <ul className="space-y-2 my-4">
            <FeatureItem>Depth Anything V2 licensing is split: size s is Apache-2.0 and fine for commercial use; b / l / g are CC-BY-NC-4.0 (non-commercial). For commercial use pick V2 size s, Depth Anything 3, or ZipDepth.</FeatureItem>
            <FeatureItem>Depth is relative inverse depth with no metric unit. Calibrate on your side if you need meters.</FeatureItem>
            <FeatureItem>For Depth Anything V2, imgsz must be divisible by 14 (the DINOv2 patch grid). Batched predict is disabled because keep-aspect resize yields variable per-image sizes.</FeatureItem>
            <FeatureItem>Video input works for depth models as of v1.4.0 (it crashed in v1.3.1).</FeatureItem>
          </ul>

          <Divider />

          {/* ────────────── IMAGE RESTORATION ────────────── */}
          <SectionHeading id="restoration" icon={WandSparkles}>Restoration &amp; Upscaling</SectionHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="experimental">SwinIR + Real-ESRGAN new in v1.4.0</SupportBadge>
            <SupportBadge variant="experimental">NAFNet trainable</SupportBadge>
          </div>
          <P>
            The <InlineCode>restore</InlineCode> task takes a degraded image and returns a
            better one. Unlike most tasks here there is nothing to detect: the output is
            an image, returned as <InlineCode>result.restored</InlineCode>. In v1.4.0 the
            task covers two jobs: <strong>cleaning</strong> (denoise / deblur, output at input
            resolution) and, new, <strong>super-resolution</strong> (output 2x or 4x larger per
            axis; <InlineCode>result.restore_scale</InlineCode> tells you the factor).
          </P>
          <DocTable
            headers={['Family', 'Job', 'Sizes', 'Output scale', 'Train?']}
            rows={[
              ['LibreNAFNet', 'denoise / deblur', 's / l', '1x', 'Yes'],
              ['LibreSwinIR (new)', 'super-resolution', 's / m / l', '4x', 'No'],
              ['LibreRealESRGAN (new)', 'super-resolution', 'x4 / x2 / x4t', '4x / 2x / 4x (fast)', 'No'],
            ]}
          />
          <P>
            What a cleaning model actually fixes, whether it denoises or deblurs, is a
            property of <em>the weights it was trained on</em>, not of the model size.
            For super-resolution the scale is baked into the checkpoint: there is no
            scale argument at predict time.
          </P>

          <SubHeading>Checkpoints</SubHeading>
          <P>
            NAFNet publishes one checkpoint: <Checkpoints names={['LibreNAFNetl-restore-sidd.pt']} />, a
            real-image <strong>denoiser</strong> trained on SIDD, converted bit-exactly from
            upstream NAFNet, MIT licensed. For <strong>deblurring</strong> there is no published
            checkpoint: convert the upstream GoPro weights with{' '}
            <InlineCode>weights/convert_nafnet_weights.py</InlineCode>. The plain names{' '}
            <InlineCode>LibreNAFNets-restore.pt</InlineCode> and <InlineCode>LibreNAFNetl-restore.pt</InlineCode>{' '}
            are <strong>not</strong> hosted, so asking for them will fail to download.
          </P>
          <P>
            Super-resolution ships fully hosted: SwinIR{' '}
            <Checkpoints names={['LibreSwinIRs-restore.pt', 'LibreSwinIRm-restore.pt', 'LibreSwinIRl-restore.pt']} />{' '}
            (Apache-2.0 code and weights) and Real-ESRGAN{' '}
            <Checkpoints names={['LibreRealESRGANx4-restore.pt', 'LibreRealESRGANx2-restore.pt', 'LibreRealESRGANx4t-restore.pt']} />.
            The <InlineCode>x4t</InlineCode> size is the compact SRVGG variant: much faster,
            visibly softer.
          </P>

          <SubHeading>Clean up an image</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")   # SIDD denoiser
result = model("noisy.jpg")

img = result.restored           # RestoredImage
print(img.array.shape)          # (H, W, 3) uint8 RGB, at the original resolution
print(result.restore_scale)     # 1 for cleaning models
img.save("clean.png")           # save lossless`}</CodeBlock>
          <P>
            Cleaning runs at the image&apos;s native resolution: the input is padded to a multiple of 16 and cropped back afterwards, so you get the same size out that you put in.
          </P>

          <SubHeading>Upscale an image (new in v1.4.0)</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreSwinIRm-restore.pt")        # 4x super-resolution
result = model("small.jpg")
print(result.restore_scale)         # 4
result.restored.save("big.png")     # 4x height, 4x width

# Real-ESRGAN: seam-free tiled upscaling for large inputs
model = LibreYOLO("LibreRealESRGANx4-restore.pt")
result = model("photo.jpg", tile=512)   # process in 512px tiles, bounded VRAM`}</CodeBlock>

          <SubHeading>Save losslessly, or you undo the work</SubHeading>
          <P>
            This is the one thing to get right. <InlineCode>libreyolo predict --save</InlineCode> writes <strong>JPEG</strong> by default, which re-introduces compression artefacts into an image you just spent a model cleaning up. Ask for PNG.
          </P>
          <CodeBlock language="bash">{`libreyolo predict model=LibreNAFNetl-restore-sidd.pt source=noisy.jpg \\
  save=true output-file-format=png`}</CodeBlock>

          <SubHeading>Train and validate (NAFNet)</SubHeading>
          <P>
            NAFNet training takes paired degraded and clean images; SwinIR and
            Real-ESRGAN are inference and validation only. Validation reports PSNR and
            SSIM. New in v1.4.0, restoration training applies coupled vertical flip and
            90-degree rotation by default (input and target transformed together), so
            retrained results move slightly versus v1.3.1; see{' '}
            <a href="#augmentation" className="text-libre-600 dark:text-libre-400 hover:underline">Data Augmentation</a>.
          </P>
          <CodeBlock language="python">{`model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
model.train(data="gopro.yaml", epochs=100)

metrics = model.val(data="gopro.yaml")
print(metrics["metrics/psnr"], metrics["metrics/ssim"])`}</CodeBlock>
          <ul className="space-y-2 my-4">
            <FeatureItem>Two reporting quirks to expect while training: the console prints PSNR under the <InlineCode>mAP50</InlineCode> column heading (a labelling bug, the number is PSNR), and PSNR/SSIM are computed with no border crop, so they are not directly comparable to published NAFNet benchmark figures.</FeatureItem>
            <FeatureItem>Export: NAFNet supports ONNX (static shapes, <InlineCode>imgsz</InlineCode> multiple of 16) and TorchScript. SwinIR supports ONNX experimentally and TorchScript. Real-ESRGAN supports ONNX, TorchScript, NCNN, and TFLite. NAFNet TFLite and CoreML remain blocked.</FeatureItem>
          </ul>

          <Divider />

          {/* ────────────── BACKGROUND REMOVAL (MATTE) ────────────── */}
          <SectionHeading id="background-removal" icon={Eraser}>Background Removal</SectionHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="experimental">New in v1.4.0</SupportBadge>
            <SupportBadge variant="experimental">Inference and val only</SupportBadge>
          </div>
          <P>
            The <InlineCode>matte</InlineCode> task predicts a per-pixel alpha value in{' '}
            <InlineCode>[0, 1]</InlineCode>: how much each pixel belongs to the foreground
            subject. Unlike a binary segmentation mask, a matte captures soft edges
            (hair, fur, motion blur), which is what makes cutouts look right. v1.4.0
            ships <InlineCode>LibreBiRefNet</InlineCode>, a BiRefNet port at 1024 px in sizes{' '}
            <InlineCode>t</InlineCode> and <InlineCode>l</InlineCode>. The{' '}
            <Checkpoints names={['LibreBiRefNetl-matte.pt']} /> weights are hosted; the{' '}
            <InlineCode>t</InlineCode> (lite) weights are not rehosted yet pending license
            confirmation, so convert them locally if you need the small one.
          </P>

          <SubHeading>Cut out a subject</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO
from PIL import Image

model = LibreYOLO("LibreBiRefNetl-matte.pt")
result = model("portrait.jpg")

matte = result.matte            # Matte payload
print(matte.data.shape)         # (H, W) float32 alpha in [0, 1], original canvas

# RGBA cutout: original pixels with the matte as the alpha channel
rgba = result.cutout()          # (H, W, 4) uint8
Image.fromarray(rgba).save("subject.png")   # transparent background

# Or composite yourself
alpha = matte.array[..., None]  # (H, W, 1)`}</CodeBlock>
          <P>
            <InlineCode>save=True</InlineCode> writes the matte overlay; on video sources the
            overlay renders per frame (matte video overlays work as of v1.4.0). Validation runs
            through <InlineCode>MatteValidator</InlineCode> against ground-truth alpha maps.
          </P>

          <SubHeading>Limits</SubHeading>
          <ul className="space-y-2 my-4">
            <FeatureItem><strong>Inference and validation only.</strong> Matte training raises in v1.4.0. Export supports experimental ONNX and fixed-1024 TorchScript; NCNN remains blocked.</FeatureItem>
            <FeatureItem>Checkpoints written with the <InlineCode>matte</InlineCode> task string are not loadable by v1.3.1.</FeatureItem>
            <FeatureItem>Save the cutout as PNG or WebP. JPEG has no alpha channel, so saving a cutout as JPEG silently flattens it.</FeatureItem>
          </ul>

          <Divider />

          {/* ────────────── OCR ────────────── */}
          <SectionHeading id="ocr" icon={ScanText}>OCR</SectionHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="experimental">New in v1.4.0</SupportBadge>
            <SupportBadge variant="experimental">Inference and val only</SupportBadge>
          </div>
          <P>
            The <InlineCode>ocr</InlineCode> task reads text: a detection stage finds text
            regions as four-point polygons, then a recognition stage transcribes each
            one. v1.4.0 ships <InlineCode>LibrePPOCR</InlineCode>, a PP-OCRv5 port at 960 px
            in sizes <InlineCode>t</InlineCode> and <InlineCode>l</InlineCode>:{' '}
            <Checkpoints names={['LibrePPOCRt-ocr.pt', 'LibrePPOCRl-ocr.pt']} />. Results
            arrive as <InlineCode>result.ocr</InlineCode>, an <InlineCode>OCRRegions</InlineCode>{' '}
            payload pairing every polygon with its text and two confidences (one from
            the detector, one from the recognizer).
          </P>

          <SubHeading>Read text from an image</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibrePPOCRt-ocr.pt")
result = model("receipt.jpg", save=True)

ocr = result.ocr                 # OCRRegions
print(ocr.polygons.shape)        # (N, 4, 2) quad corners in pixels
for text, conf in zip(ocr.texts, ocr.conf):
    print(f"{conf:.2f}  {text}")

ocr.det_conf                      # (N,) detector scores, separate from recognition
print(result.saved_path)          # saved OCR overlay`}</CodeBlock>

          <SubHeading>CLI and validation</SubHeading>
          <P>
            <InlineCode>libreyolo predict --json</InlineCode> emits an <InlineCode>ocr</InlineCode>{' '}
            array (polygon, text, confidences per region), which makes the CLI directly
            scriptable for document pipelines. Validation runs through{' '}
            <InlineCode>OCRValidator</InlineCode>, which matches predictions to ground truth
            with an optimal one-to-one assignment before scoring.
          </P>
          <CodeBlock language="bash">{`libreyolo predict model=LibrePPOCRt-ocr.pt source=receipt.jpg --json | jq .ocr`}</CodeBlock>

          <SubHeading>Limits</SubHeading>
          <ul className="space-y-2 my-4">
            <FeatureItem><strong>Inference and validation only.</strong> OCR training and export raise in v1.4.0.</FeatureItem>
            <FeatureItem>Checkpoints written with the <InlineCode>ocr</InlineCode> task string are not loadable by v1.3.1.</FeatureItem>
            <FeatureItem>On an OCR model, <InlineCode>result.boxes</InlineCode> is <InlineCode>None</InlineCode>: regions are polygons in <InlineCode>result.ocr</InlineCode>, not axis-aligned boxes.</FeatureItem>
          </ul>

          <Divider />

          {/* ────────────── POINT LOCALIZATION ────────────── */}
          <SectionHeading id="point-localization" icon={MapPin}>Point Localization</SectionHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="experimental">Experimental</SupportBadge>
          </div>
          <P>
            <InlineCode>LibreFOMO</InlineCode> is a FOMO-style point localizer (sizes <InlineCode>s</InlineCode> / <InlineCode>m</InlineCode> / <InlineCode>l</InlineCode>) for centroid-style detection: instead of boxes, each detection is a single image coordinate. Predictions arrive as <InlineCode>result.points</InlineCode>. Pretrained LibreFOMO weights are not auto-downloaded, so pass a local checkpoint path (or train from scratch, which is experimental and requires <InlineCode>allow_experimental=True</InlineCode>). New in v1.4.0: FOMO exports to ONNX under the fixed-resolution contract, so a trained point model can leave Python for edge deployment.
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

          {/* ────────────── ANNOTATION (LIBRELABEL) ────────────── */}
          <SectionHeading id="annotation" icon={PenTool}>Annotation (LibreLabel)</SectionHeading>
          <P>
            <InlineCode>libreyolo label</InlineCode> starts a local, browser-based annotation tool. It writes LibreYOLO-native label files exactly where the trainer already reads them, so a folder of images becomes a trainable dataset with no conversion step, no cloud account and no database. The server is Python standard library only, and it runs entirely on your machine.
          </P>

          <SubHeading>Label a folder of images</SubHeading>
          <CodeBlock language="bash">{`# open an existing dataset
libreyolo label data=path/to/data.yaml

# a bare folder works too: LibreYOLO scaffolds the dataset around it
libreyolo label data=path/to/images

# start on the project home screen and create a project in the browser
libreyolo label`}</CodeBlock>

          <SubHeading>Options</SubHeading>
          <DocTable
            headers={['Option', 'Default', 'What it does']}
            rows={[
              ['data', '(none)', 'Dataset YAML or a folder. Omit to open the project home screen.'],
              ['host', '127.0.0.1', 'Interface to bind. See the sharing note below before changing this.'],
              ['port', '8000', 'Port to bind. Auto-bumps up to port+19 if taken.'],
              ['device', 'auto', 'Device used by the AI assist features.'],
              ['no_assist', 'false', 'Hard-disable every AI assist feature.'],
              ['no_browser', 'false', 'Do not auto-open a browser.'],
              ['share', 'false', 'Bind 0.0.0.0 so teammates on your LAN can label with you.'],
            ]}
          />

          <SubHeading>What you can label</SubHeading>
          <P>
            Bounding boxes (<InlineCode>detect</InlineCode>), polygons (<InlineCode>segment</InlineCode>) and oriented boxes (<InlineCode>obb</InlineCode>, with a rotate handle). Keypoints, masks and depth files open <strong>read-only</strong>, so a save can never silently drop fields it does not understand. Classification labelling is not available yet.
          </P>

          <SubHeading>AI assist, and the one rule it never breaks</SubHeading>
          <P>
            LibreLabel can pre-label with one of your own detectors, turn a click into a mask with SAM, audit your existing labels for likely mistakes, find near-duplicate images, and detect train/val leakage. <strong>No AI path ever writes a label file.</strong> Every suggestion is held in memory until a human accepts it. AI assist also never downloads weights: if a checkpoint is not already on disk it refuses and tells you, rather than pulling hundreds of megabytes behind your back.
          </P>
          <ul className="space-y-2 my-4">
            <FeatureItem>Box pre-labelling with any in-package detector works on the base install, no extra needed.</FeatureItem>
            <FeatureItem>SAM click-to-mask needs <InlineCode>pip install &quot;libreyolo[label]&quot;</InlineCode> and the LibreSAM weights already downloaded.</FeatureItem>
            <FeatureItem>Assist is task-aware: on an OBB project it is refused entirely, and on a segmentation project only the mask tools stay available.</FeatureItem>
          </ul>

          <SubHeading>Export</SubHeading>
          <P>
            Export to YOLO, COCO or VOC (or several at once) from the Export dialog in the browser, with reproducible train/val/test splits. Note it is a browser action: there is no CLI export flag. Import is YOLO only, so COCO and VOC are export formats, not entry points.
          </P>

          <SubHeading>Sharing, and a trap worth knowing</SubHeading>
          <P>
            There is <strong>no authentication of any kind</strong>. Access is controlled purely by network position, so only share on a network you trust.
          </P>
          <P>
            The counter-intuitive part: <InlineCode>share=true</InlineCode> is the <em>safe</em> way to let teammates in. It binds a wildcard address, and because admin rights require a loopback connection, you keep admin on your machine while teammates get a labelling-only view. Binding a specific address instead (<InlineCode>host=192.168.1.50</InlineCode>) makes your machine indistinguishable from a teammate, which hands <strong>full admin to every client on the LAN</strong>. Prefer <InlineCode>share=true</InlineCode>.
          </P>

          <Divider />

          <SectionHeading id="training" icon={GraduationCap}>Training</SectionHeading>
          <ValidationScopeCallout />
          <P>
            The heavily tested training paths are single-GPU YOLO9 detection, RF-DETR detection, and RF-DETR segmentation. Other model-family trainers and multi-GPU workflows are available but experimental. YOLO9 is detect-only, so there is no YOLO9 segmentation or pose training. New in v1.4.0: YOLOv7 trains (SimOTA loss), SegFormer fine-tunes, and every augmentation knob is documented per family in the <a href="#augmentation" className="text-libre-600 dark:text-libre-400 hover:underline">Data Augmentation</a> section.
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
            After training completes, the model instance is automatically reloaded with the best weights so you can call <InlineCode>model(...)</InlineCode> immediately. <InlineCode>freeze</InlineCode>, <InlineCode>cache</InlineCode>, <InlineCode>pretrained</InlineCode>, and <InlineCode>save_plots</InlineCode> are accepted across the trainer-backed families.
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
            RF-DETR has its own training signature (<InlineCode>batch_size</InlineCode>, <InlineCode>lr</InlineCode>, <InlineCode>output_dir</InlineCode>) but shares LibreYOLO&apos;s dataset loader. Pass a <InlineCode>data.yaml</InlineCode> for detection or segmentation in either YOLO TXT or native COCO JSON layout - see <a href="#dataset-format" className="text-libre-600 dark:text-libre-400 hover:underline">Dataset Format</a>.
          </P>

          <SubHeading>LoRA fine-tuning</SubHeading>
          <P>
            <SupportBadge variant="experimental">Experimental</SupportBadge>{' '}
            <InlineCode>lora=True</InlineCode> injects LoRA adapters for low-VRAM fine-tuning:
            only the adapters (plus the parts that must stay trainable, like detection
            heads) receive gradients. It requires the optional <InlineCode>peft</InlineCode>{' '}
            dependency (<InlineCode>pip install &quot;libreyolo[lora]&quot;</InlineCode>). v1.4.0
            extends LoRA well beyond RF-DETR: the supported families are{' '}
            <strong className="text-surface-800 dark:text-white">RF-DETR, D-FINE, DEIM, DEIMv2, RT-DETR v1 / v2 / v4, EC, and ConvNeXt</strong>{' '}
            (D-FINE and EC detect-only). Unsupported families still raise a clear error
            rather than ignoring the flag. On <InlineCode>export()</InlineCode>, adapters are
            merged into the dense weights, so deployed artifacts need no peft at runtime.
          </P>
          <CodeBlock language="python">{`model = LibreYOLO("LibreRFDETRs.pt")
results = model.train(data="data.yaml", epochs=50, lora=True)

# Works the same on the newly supported families
model = LibreYOLO("LibreDEIMs.pt")
results = model.train(data="data.yaml", epochs=50, lora=True)`}</CodeBlock>

          <SubHeading>Experiment loggers</SubHeading>
          <P>
            Pass <InlineCode>loggers=</InlineCode> to stream metrics to TensorBoard,
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
            Other families have trainer hooks, but they are not the recommended path in v1.4.0. Keep new work on YOLO9 detection or RF-DETR detection/segmentation; use experimental trainers only for compatibility, benchmark reproduction, or targeted research. PicoDet, RTMDet, and EC training require an explicit <InlineCode>allow_experimental=True</InlineCode> acknowledgement. Note that v1.4.0 fixed harmful fine-tune defaults for PicoDet (<InlineCode>lr0</InlineCode> 0.1 to 0.01) and DEIM (<InlineCode>lr0</InlineCode> 4e-4 to 1e-4): pass the old values explicitly if you need to reproduce upstream COCO recipes.
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

          <SubHeading>Distributed training (DDP, overhauled in v1.4.0)</SubHeading>
          <P>
            Multi-GPU training got a correctness overhaul in v1.4.0. It is still outside
            the heavily tested scope, but the failure modes changed from silent to loud:
          </P>
          <ul className="space-y-2 my-4">
            <FeatureItem><strong>Correct sharding everywhere.</strong> DEIM, D-FINE and YOLO-NAS pose previously trained the <em>full</em> dataset at the <em>full</em> batch on every rank (so extra GPUs bought nothing); they now shard correctly, and loss normalizers are all-reduced globally so gradients match single-GPU training.</FeatureItem>
            <FeatureItem><strong>SyncBatchNorm defaults on</strong> under DDP for the BatchNorm-heavy families (YOLO9, YOLOX, YOLOv7, YOLO-NAS, PicoDet, RTMDet, FOMO), fixing a real multi-GPU convergence degradation from per-rank BN statistics.</FeatureItem>
            <FeatureItem><strong>Hard setup errors instead of silently wrong runs</strong>: a global batch that does not divide by the world size, a per-rank batch below 1 after AutoBatch, or a custom loader that does not shard now raise at setup.</FeatureItem>
            <FeatureItem><strong>Spawn-path multi-GPU</strong> reaches the classifier families (ResNet, ConvNeXt, EfficientNetV2, MobileNetV4) and NAFNet: pass <InlineCode>device=&quot;0,1&quot;</InlineCode> and workers are spawned for you, no torchrun needed.</FeatureItem>
          </ul>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")
# Two GPUs: the global batch is split per rank (16 -> 8 + 8)
model.train(data="coco128.yaml", epochs=300, batch=16, device="0,1")`}</CodeBlock>
          <P>
            <InlineCode>device=&quot;0,1&quot;</InlineCode> (or a list <InlineCode>[0, 1]</InlineCode>)
            selects multi-GPU. Under <InlineCode>torchrun</InlineCode> the launcher owns the
            process group; outside it, LibreYOLO spawns DDP workers itself. Both paths
            run the same trainer.
          </P>
          <CodeBlock language="bash">{`# Explicit torchrun launch also works
torchrun --nproc_per_node=2 train_yolo9.py`}</CodeBlock>

          <Divider />

          {/* ────────────── DATA AUGMENTATION ────────────── */}
          <SectionHeading id="augmentation" icon={Dices}>Data Augmentation</SectionHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="experimental">Documented + spec-checked in v1.4.0</SupportBadge>
          </div>
          <P>
            Training-time augmentation is configured directly on{' '}
            <InlineCode>model.train()</InlineCode>: mosaic, MixUp, HSV jitter, flips, and
            the affine warp (rotation, translation, scale, shear, perspective) are all
            plain keyword arguments. The same knobs work as{' '}
            <InlineCode>key=value</InlineCode> pairs on <InlineCode>libreyolo train</InlineCode>,
            where <InlineCode>mosaic=</InlineCode> and <InlineCode>mixup=</InlineCode> are
            CLI shorthands for <InlineCode>mosaic_prob</InlineCode> and{' '}
            <InlineCode>mixup_prob</InlineCode>.
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")
model.train(
    data="coco128.yaml",
    epochs=100,

    # Augmentation knobs (defaults shown in the table below)
    mosaic_prob=1.0,       # 4-image mosaic
    mixup_prob=0.5,        # blend in a second sample
    hsv_prob=1.0,          # HSV color jitter
    flip_prob=0.5,         # horizontal flip
    flipud=0.1,            # vertical flip (good for aerial imagery)
    degrees=10.0,          # random rotation range for the affine warp
    translate=0.1,         # random translation fraction
    shear=2.0,             # random shear, degrees
    perspective=0.0005,    # projective warp magnitude (0 = pure affine)
    no_aug_epochs=15,      # final epochs with strong augmentation off
)`}</CodeBlock>
          <CodeBlock language="bash">{`libreyolo train model=yolo9-c data=coco128.yaml epochs=100 \\
  mosaic=1.0 mixup=0.5 hsv_prob=1.0 flip_prob=0.5 degrees=10 translate=0.1`}</CodeBlock>

          <SubHeading>The augmentation knobs</SubHeading>
          <P>
            These are the base <InlineCode>TrainConfig</InlineCode> fields. The defaults
            below are the base values: <strong>families override them with tuned
            recipes</strong> (YOLO9 defaults to <InlineCode>degrees=0</InlineCode>,{' '}
            <InlineCode>shear=0</InlineCode> and mixup off, while YOLOX keeps all three
            on, for example). Print the exact resolved defaults for your model with{' '}
            <InlineCode>libreyolo cfg</InlineCode>.
          </P>
          <DocTable
            headers={['Knob', 'Base default', 'What it does']}
            rows={[
              [<InlineCode key="k">mosaic_prob</InlineCode>, '1.0', 'Probability of building a 4-image mosaic sample.'],
              [<InlineCode key="k">mixup_prob</InlineCode>, '1.0', 'Probability of blending in a second sample (MixUp).'],
              [<InlineCode key="k">hsv_prob</InlineCode>, '1.0', 'Probability of HSV color jitter.'],
              [<InlineCode key="k">flip_prob</InlineCode>, '0.5', 'Horizontal-flip probability.'],
              [<InlineCode key="k">flipud</InlineCode>, '0.0', 'Vertical-flip probability. Off by default; useful when the scene has no fixed up (aerial, microscopy).'],
              [<InlineCode key="k">degrees</InlineCode>, '10.0', 'Random-rotation range for the affine warp, in degrees.'],
              [<InlineCode key="k">translate</InlineCode>, '0.1', 'Random-translation fraction for the affine warp.'],
              [<InlineCode key="k">mosaic_scale</InlineCode>, '(0.1, 2.0)', 'Random-scale range for the affine warp.'],
              [<InlineCode key="k">shear</InlineCode>, '2.0', 'Random-shear range for the affine warp, in degrees.'],
              [<InlineCode key="k">perspective</InlineCode>, '0.0', 'Projective warp magnitude, sampled in [-p, +p]; around 0.0005 is typical. 0 keeps the warp purely affine.'],
              [<InlineCode key="k">mixup_scale</InlineCode>, '(0.5, 1.5)', 'Jitter-scale range applied to the MixUp partner image.'],
              [<InlineCode key="k">no_aug_epochs</InlineCode>, '15', 'Final epochs trained with strong augmentation disabled, so the model converges on clean images.'],
            ]}
          />

          <SubHeading>Which families honor which knobs</SubHeading>
          <P>
            Not every family runs every augmentation: each one trains through the
            pipeline its recipe came with. v1.4.0 makes this explicit with a
            declarative spec (<InlineCode>libreyolo/data/augment/spec.py</InlineCode>) that
            is pinned to the real pipelines by tests. Every knob has one of three
            statuses per family, and <strong>the CLI now warns whenever you explicitly set
            a parameter the selected family ignores</strong>, so a typo or a wrong
            assumption no longer fails silently.
          </P>
          <DocTable
            headers={['Status', 'Meaning']}
            rows={[
              [<strong key="s" className="text-emerald-600 dark:text-emerald-400">used</strong>, 'The knob reaches the training pipeline and changes samples.'],
              [<strong key="s" className="text-amber-600 dark:text-amber-400">gated by mosaic</strong>, 'The knob only applies to samples that took the mosaic branch; with mosaic_prob=0 it never fires.'],
              [<strong key="s" className="text-surface-500">ignored</strong>, 'The knob never reaches this family’s pipeline; setting it does nothing (and the CLI warns).'],
            ]}
          />
          <DocTable
            headers={['Pipeline', 'Families', 'What actually runs']}
            rows={[
              ['YOLOX-style mosaic', 'YOLO9, YOLO9-E2E, YOLO9-P2, YOLOX, YOLOv7, RTMDet, PicoDet, RT-DETR, RT-DETRv2, FOMO', 'HSV jitter and flips run per sample. The affine warp (degrees / translate / mosaic_scale / shear / perspective) and MixUp run on the mosaic canvas only, so they are gated by mosaic_prob. RTMDet, PicoDet, RT-DETR, RT-DETRv2 and FOMO have no vertical flip; FOMO also drops perspective.'],
              ['YOLO-NAS', 'YOLO-NAS', 'No mosaic (mosaic_prob is ignored). Instead a per-sample affine is always on, so degrees / translate / shear / perspective apply directly, and MixUp is independent of mosaic.'],
              ['DETR-style pass-through', 'D-FINE, DEIM, DEIMv2, RT-DETRv4, EC', 'Only flip_prob and no_aug_epochs are yours to tune. Color jitter, zoom-out and IoU-crop are fixed recipe constants, and there is no mosaic, MixUp or affine warp. Exception: EC pose honors hsv_prob, degrees and translate through its keypoint-aware affine.'],
              ['RF-DETR native', 'RF-DETR', 'Flip, scale jitter and random crop from the native recipe; flip_prob and no_aug_epochs are configurable, HSV is not.'],
              ['Classification', 'ResNet, ConvNeXt, MobileNetV4, EfficientNetV2, DINOv2 (classify)', 'The detection knobs never apply (horizontal flip is a fixed 0.5). Use the classification pack below.'],
              ['Semantic', 'SegFormer (and the shared semantic pipeline)', 'Scale jitter and HSV come from family attributes rather than TrainConfig knobs; flip is a fixed 0.5. HSV jitter defaults on as of v1.4.0.'],
              ['Restoration', 'NAFNet', 'Coupled input / target crop, flips and rot90 at fixed probabilities (vertical flip + rot90 new in v1.4.0). TrainConfig knobs are ignored.'],
            ]}
          />

          <SubHeading>Mosaic gating, explained</SubHeading>
          <P>
            In the YOLOX-style pipelines, MixUp and the affine warp ride <em>inside</em>{' '}
            the mosaic branch: a sample first becomes a 4-image mosaic (with probability{' '}
            <InlineCode>mosaic_prob</InlineCode>), and only then is the mosaic canvas
            warped and optionally blended with another sample. Two practical
            consequences:
          </P>
          <ul className="space-y-2 my-4">
            <FeatureItem>Setting <InlineCode>mosaic_prob=0</InlineCode> also turns off MixUp and the affine warp for these families, whatever their knobs say. v1.4.0 warns at training start when <InlineCode>mixup_prob &gt; 0</InlineCode> can never fire because <InlineCode>mosaic_prob=0</InlineCode>.</FeatureItem>
            <FeatureItem>To train with light augmentation but keep some geometry, lower <InlineCode>mosaic_prob</InlineCode> rather than zeroing it, or switch the geometry off explicitly with <InlineCode>degrees=0 translate=0 shear=0</InlineCode>.</FeatureItem>
          </ul>
          <CodeBlock language="python">{`# Minimal augmentation: flips only
model.train(
    data="data.yaml",
    mosaic_prob=0.0,   # also disables mixup + affine in mosaic-gated families
    mixup_prob=0.0,
    hsv_prob=0.0,
    flip_prob=0.5,
    no_aug_epochs=0,
)`}</CodeBlock>

          <SubHeading>Classification augmentation pack (new in v1.4.0)</SubHeading>
          <P>
            The classification ImageFolder pipeline has its own four knobs, all off by
            default. At most one of MixUp / CutMix runs per batch: MixUp fires with
            probability <InlineCode>mixup</InlineCode>, otherwise CutMix with probability{' '}
            <InlineCode>cutmix</InlineCode>, so the two should sum to at most 1.
          </P>
          <DocTable
            headers={['Knob', 'Default', 'What it does']}
            rows={[
              [<InlineCode key="k">auto_augment</InlineCode>, 'None', 'Policy name: "randaugment", "autoaugment" or "augmix".'],
              [<InlineCode key="k">erasing</InlineCode>, '0.0', 'RandomErasing probability.'],
              [<InlineCode key="k">mixup</InlineCode>, '0.0', 'Batch-MixUp probability with soft labels. Python API only: on the CLI, --mixup is the detection mixup_prob alias.'],
              [<InlineCode key="k">cutmix</InlineCode>, '0.0', 'Batch-CutMix probability with soft labels.'],
            ]}
          />
          <CodeBlock language="python">{`from libreyolo import LibreMobileNetV4

model = LibreMobileNetV4(size="s")
model.train(
    data="imagenette160",
    epochs=20,
    auto_augment="randaugment",
    erasing=0.25,
    mixup=0.2,
    cutmix=0.2,
)`}</CodeBlock>

          <SubHeading>Task-specific extras</SubHeading>
          <P>
            A few knobs live on family <InlineCode>TrainConfig</InlineCode> subclasses
            rather than the base config, and are reachable from Python or a training
            YAML (the CLI does not expose them):
          </P>
          <DocTable
            headers={['Knob', 'Families', 'What it does']}
            rows={[
              [<InlineCode key="k">copy_paste</InlineCode>, 'RF-DETR (segment), YOLO9 lineage', 'Copy-paste instance augmentation probability for segmentation training: instances are cut out and pasted into the sample.'],
              [<InlineCode key="k">copy_paste_mode</InlineCode>, 'same', 'Source of pasted instances: "flip" mirrors the same sample; "mixup" pulls a second sample (RF-DETR supports "flip" only).'],
              [<InlineCode key="k">rot90</InlineCode>, 'YOLO9 lineage (OBB path)', 'Random 90-degree rotation probability for oriented-box training; ignored for axis-aligned detection.'],
              [<InlineCode key="k">crop_resize_prob</InlineCode>, 'RF-DETR, D-FINE (segment), EC (segment)', 'Random crop-resize probability in the native pipelines.'],
              [<InlineCode key="k">brightness_contrast_prob</InlineCode>, 'YOLO-NAS (pose), EC (pose)', 'Brightness / contrast jitter probability for keypoint training.'],
              [<InlineCode key="k">affine_prob</InlineCode>, 'YOLO-NAS (pose), EC (pose)', 'Keypoint-aware affine probability.'],
            ]}
          />

          <SubHeading>Training augmentation vs test-time augmentation</SubHeading>
          <P>
            Everything above happens during <InlineCode>train()</InlineCode>. Test-time
            augmentation is separate: <InlineCode>predict(augment=True)</InlineCode> /{' '}
            <InlineCode>val(augment=True)</InlineCode> run extra augmented forward passes
            and merge the outputs at inference time. In v1.4.0 TTA covers detection
            families where implemented, the four semantic families (PIDNet, SegFormer,
            EoMT, DINOv2), and EoMT panoptic.
          </P>

          <Divider />

          {/* ────────────── DISTILLATION ────────────── */}
          <SectionHeading id="distillation" icon={FlaskConical}>Distillation</SectionHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="experimental">YOLO9 and YOLOX students</SupportBadge>
            <SupportBadge variant="experimental">DINOv2 teacher new in v1.4.0</SupportBadge>
          </div>
          <P>
            Knowledge distillation trains a small student model against a larger frozen teacher, so the student learns from the teacher&apos;s intermediate features on top of its own labels. You get a model that runs at the student&apos;s speed but recovers some of the teacher&apos;s accuracy. Point <InlineCode>distill_model</InlineCode> at a teacher checkpoint and distillation turns on.
          </P>

          <SubHeading>Distill a big model into a small one</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

student = LibreYOLO("LibreYOLO9t.pt")     # small student

student.train(
    data="coco.yaml",
    epochs=100,
    distill_model="LibreYOLO9c.pt",   # the frozen teacher: this turns distillation ON
    distill_loss_type="mgd",          # "mgd" (default) or "cwd"
    dis=2e-5,                         # global weight; omit to take the per-loss default
)`}</CodeBlock>
          <CodeBlock language="bash">{`libreyolo train model=LibreYOLO9t.pt data=coco.yaml epochs=100 \\
  distill-model=LibreYOLO9c.pt distill-loss-type=mgd dis=2e-5`}</CodeBlock>
          <P>
            During training the distillation term shows up as a <InlineCode>distill</InlineCode> loss component alongside the usual ones.
          </P>

          <SubHeading>Arguments</SubHeading>
          <DocTable
            headers={['Argument', 'Default', 'Meaning']}
            rows={[
              ['distill_model', 'None', 'Teacher checkpoint path. Setting it enables distillation.'],
              ['dis', 'None', 'Global distillation loss weight. Falls back to 2e-5 for MGD, 1.0 for CWD.'],
              ['distill_loss_type', '"mgd"', 'Feature loss: "mgd" or "cwd".'],
              ['distill_mask_ratio', '0.65', 'MGD only: fraction of spatial positions masked. Python API only.'],
              ['distill_tau', '1.0', 'CWD only: softmax temperature. Python API only.'],
            ]}
          />
          <P>
            Note the short name: the weight argument is <InlineCode>dis</InlineCode>, not <InlineCode>distill_loss_weight</InlineCode>. Three of these have CLI flags (<InlineCode>distill-model</InlineCode>, <InlineCode>dis</InlineCode>, <InlineCode>distill-loss-type</InlineCode>); <InlineCode>distill_mask_ratio</InlineCode> and <InlineCode>distill_tau</InlineCode> are reachable from Python or a training YAML only.
          </P>

          <SubHeading>Foundation-teacher distillation: DINOv2 (new in v1.4.0)</SubHeading>
          <P>
            Beyond checkpoint teachers, v1.4.0 adds <InlineCode>distill_model=&quot;dinov2&quot;</InlineCode>:
            the student&apos;s backbone features are regressed against a frozen DINOv2
            foundation encoder with a <InlineCode>feat_mse</InlineCode> loss. No teacher
            checkpoint of your own is needed, which makes this the cheapest way to add a
            distillation signal to a YOLO9-backbone training run.
          </P>
          <CodeBlock language="python">{`student = LibreYOLO("LibreYOLO9s.pt")
student.train(
    data="coco128.yaml",
    epochs=100,
    distill_model="dinov2",       # frozen foundation teacher
    distill_loss_type="feat_mse", # feature regression against DINOv2
    distill_normalize=True,       # normalize features before the loss
)`}</CodeBlock>
          <P>
            The teacher runs at a DINOv2-compatible resolution internally; v1.4.0 fixed
            a border-cropping bug at sizes that are not multiples of 14, so odd input
            sizes distill correctly.
          </P>

          <SubHeading>MGD or CWD</SubHeading>
          <P>
            <strong>MGD</strong> (Masked Generative Distillation, the default) masks random spatial positions in the student features and asks it to regenerate the teacher&apos;s. Because it regresses raw feature magnitudes, its default weight is small: <InlineCode>2e-5</InlineCode>.
          </P>
          <P>
            <strong>CWD</strong> (Channel-Wise Distillation) turns each channel into a spatial distribution and matches them with a KL divergence. Normalizing per channel makes it scale invariant, so it copes better when teacher and student feature magnitudes are far apart. Its default weight is <InlineCode>1.0</InlineCode>.
          </P>
          <P>
            We do not publish a head-to-head accuracy comparison of the two, so treat MGD as the default and try CWD if the loss scale looks unhealthy.
          </P>

          <SubHeading>Limits</SubHeading>
          <ul className="space-y-2 my-4">
            <FeatureItem><strong>Only the YOLO9 and YOLOX families can be students.</strong> Every other family raises at setup, because distillation needs feature tap points that only these two declare. The DINOv2 foundation teacher targets YOLO9 backbones.</FeatureItem>
            <FeatureItem><strong>Teacher and student strides must match exactly.</strong> Both supported families use strides 8/16/32, so in practice you distill within a family, across sizes. Channel widths may differ freely: a 1x1 adapter bridges them.</FeatureItem>
            <FeatureItem>Multi-GPU, mixed precision and gradient accumulation all work with distillation on.</FeatureItem>
            <FeatureItem>Resuming works, and the adapter state is restored, but the teacher is not stored in the checkpoint: pass <InlineCode>distill_model</InlineCode> again when you resume.</FeatureItem>
          </ul>

          <Divider />

          {/* ────────────── TRAINING MONITOR ────────────── */}
          <SectionHeading id="monitoring" icon={Gauge}>Training Monitor</SectionHeading>
          <P>
            Every training run, for every model family, writes machine-readable progress files into its run directory. You do not have to enable anything. <InlineCode>libreyolo monitor</InlineCode> serves them as a live dashboard, and because it only reads files it works equally well on a running job, a finished one, or one that crashed.
          </P>
          <CodeBlock language="bash">{`libreyolo monitor                     # watch runs/ on http://127.0.0.1:8420
libreyolo monitor runs/train/exp      # open one run directly
libreyolo monitor --port 9000 --no-browser`}</CodeBlock>

          <SubHeading>Run artifacts</SubHeading>
          <P>
            The two files below are the contract, and they are the reason this is useful to scripts and agents as well as to humans: you can poll a run&apos;s state without parsing logs.
          </P>
          <DocTable
            headers={['File', 'What it is']}
            rows={[
              ['status.json', 'Current state of the run, rewritten atomically every epoch.'],
              ['metrics.jsonl', 'Append-only, one JSON object per epoch. The full metric history.'],
              ['train.log', 'The run log.'],
            ]}
          />
          <P>
            <InlineCode>status.json</InlineCode> always carries <InlineCode>state</InlineCode> (<InlineCode>running</InlineCode>, <InlineCode>completed</InlineCode> or <InlineCode>failed</InlineCode>), <InlineCode>pid</InlineCode>, <InlineCode>progress</InlineCode>, <InlineCode>eta_seconds</InlineCode>, and the current and best metric. If the run dies it records <InlineCode>state: &quot;failed&quot;</InlineCode> plus an <InlineCode>error</InlineCode> object with the exception type and message, so a crash is visible in the file rather than only in a terminal you have closed.
          </P>
          <CodeBlock language="python">{`import json, time

def wait_for_run(run_dir):
    while True:
        status = json.load(open(f"{run_dir}/status.json"))
        if status["state"] != "running":
            return status
        print(f'{status["progress"]:.0%}  eta {status["eta_seconds"]:.0f}s  '
              f'best {status["best_metric"]}')
        time.sleep(30)

final = wait_for_run("runs/train/exp")
if final["state"] == "failed":
    print(final["error"]["type"], final["error"]["message"])
else:
    print(final["checkpoints"]["best"])`}</CodeBlock>
          <P>
            The monitor also exposes the same data over HTTP (<InlineCode>/api/status</InlineCode>, <InlineCode>/api/metrics</InlineCode>, <InlineCode>/api/log</InlineCode>, <InlineCode>/api/images</InlineCode>), so you can drive a dashboard of your own from it.
          </P>

          <Divider />

          {/* ────────────── PROFILING ────────────── */}
          <SectionHeading id="profiling" icon={Timer}>Profiling</SectionHeading>
          <P>
            <InlineCode>libreyolo profile</InlineCode> measures where the time actually goes, in training and in inference. It is deliberately a measuring tool and nothing else: it never edits your config or tunes anything for you. It tells you what is slow and leaves the decision to you.
          </P>
          <P>
            Behavior change in v1.4.0: <InlineCode>model.train(profile=True)</InlineCode> now{' '}
            <strong>keeps training</strong> after the profiled window instead of stopping (and it
            no longer corrupts resume state). Pass <InlineCode>profile_then_stop=True</InlineCode>{' '}
            to restore the old capture-and-exit behavior.
          </P>

          <SubHeading>Profile training or inference</SubHeading>
          <CodeBlock language="bash">{`# training: is the GPU actually busy, or am I dataloader-bound?
libreyolo profile run coco128 --weights LibreYOLO9t.pt --batch 16 --repeat 3

# inference: latency percentiles and where they are spent
libreyolo profile infer bus.jpg --weights LibreYOLO9t.pt --runs 200`}</CodeBlock>
          <P>
            <InlineCode>profile infer</InlineCode> reports p50, p90 and p99 latency, throughput, and a split across preprocess, forward and postprocess (NMS), plus a verdict on what is bounding you. That split is usually the punchline: a model that looks slow is often spending its time in NMS or in preprocessing rather than in the network.
          </P>

          <SubHeading>Then look closer</SubHeading>
          <P>
            Both commands write the same <InlineCode>profile.json</InlineCode>, and the analysis subcommands all read it, so you profile once and then interrogate the result from several angles.
          </P>
          <DocTable
            headers={['Subcommand', 'What it answers']}
            rows={[
              ['summary', 'The high-level diagnosis: utilisation, what is bounding you, the kernel mix.'],
              ['phases', 'Where the time went: forward, backward, dataload, optimizer.'],
              ['kernels', 'Which individual GPU kernels dominate.'],
              ['ops', 'The framework view: which operations cost the most CPU time.'],
              ['get', 'Print one metric, for use inside a script.'],
              ['compare', 'Diff two profiles, before and after a change.'],
              ['what-if', 'Estimate the payoff of a change before you write it.'],
            ]}
          />
          <CodeBlock language="bash">{`libreyolo profile summary runs/profile/prof/profile.json
libreyolo profile kernels runs/profile/prof/profile.json --top 20
libreyolo profile compare before.json after.json`}</CodeBlock>
          <P>
            Two practical notes. Every subcommand takes <InlineCode>--json</InlineCode>, which makes the profiler usable inside an automated optimize loop. And <InlineCode>compare</InlineCode> will only report statistical significance if both profiles were captured with <InlineCode>--repeat 2</InlineCode> or higher: a single run is noisy enough to mislead you, especially when the job is launch-bound.
          </P>

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
            <InlineCode>PoseValidator</InlineCode>. Beyond those, validators cover classify (top-1 / top-5),
            semantic (mIoU / pixel accuracy), point, depth (zero-shot), and, new in v1.4.0,
            panoptic (<InlineCode>PanopticValidator</InlineCode> with the Panoptic Quality metric),
            matte (<InlineCode>MatteValidator</InlineCode>), and OCR (<InlineCode>OCRValidator</InlineCode>{' '}
            with optimal one-to-one assignment). Semantic and panoptic validation accept{' '}
            <InlineCode>augment=True</InlineCode> for flip-TTA as of v1.4.0 (this raised before). Pass{' '}
            <InlineCode>plots=True</InlineCode> (or <InlineCode>--save-plots</InlineCode> on the CLI) to write
            metric, per-class AP, confusion-matrix, and sample plots to the run directory.
          </P>

          <Divider />

          {/* ────────────── QUANTIZATION ────────────── */}
          <SectionHeading id="quantization" icon={Binary}>Quantization</SectionHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="experimental">New in v1.4.0</SupportBadge>
            <SupportBadge variant="experimental">YOLO9 and RF-DETR</SupportBadge>
          </div>
          <P>
            LibreYOLO quantizes models directly in PyTorch. A quantized model keeps the
            normal <InlineCode>predict</InlineCode> / <InlineCode>val</InlineCode> /{' '}
            <InlineCode>train</InlineCode> / <InlineCode>save</InlineCode> contract, so accuracy
            is measured with the same validators as float models, and accuracy recovery is
            just <InlineCode>train()</InlineCode> on the quantized model (QAT), optionally with
            the existing distillation kwargs (QAD).
          </P>

          <SubHeading>The grammar: quantize, then optionally recover</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9s.pt")

# Step 1: quantize. calib is a small UNLABELED image set, used forward-only
# to derive activation ranges and scales.
qmodel = model.quantize(recipe="int8", calib="coco128.yaml", samples=128)

qmodel.val(data="coco8.yaml")            # honest accuracy, same validators
qmodel.predict("bus.jpg")
qmodel.save("LibreYOLO9s-int8.pt")       # manifest-carrying checkpoint

# Step 2 (optional): QAT is plain train() on the quantized model
qmodel.train(data="coco.yaml", epochs=5)

# QAD: same, plus the existing distillation kwargs
qmodel.train(data="coco.yaml", epochs=5, distill_model="LibreYOLO9m.pt")`}</CodeBlock>
          <CodeBlock language="bash">{`libreyolo quantize --model LibreYOLO9s.pt --recipe int8 --calib coco8.yaml
libreyolo train model=LibreYOLO9s-int8.pt data=coco.yaml epochs=5`}</CodeBlock>
          <P>
            <InlineCode>LibreYOLO(&quot;LibreYOLO9s-int8.pt&quot;)</InlineCode> restores the quantized
            structure and scales automatically: checkpoints carry a <InlineCode>quant</InlineCode>{' '}
            manifest, and trainer checkpoints written during QAT / QAD carry it too, so{' '}
            <InlineCode>best.pt</InlineCode> from a QAT run is itself a quantized checkpoint.{' '}
            <InlineCode>model.quant_info()</InlineCode> reports the recipe, module counts,
            calibration state and execution tier; <InlineCode>model.dequantize()</InlineCode>{' '}
            restores float modules in place.
          </P>

          <SubHeading>Recipes</SubHeading>
          <DocTable
            headers={['Recipe', 'What it does', 'Families', 'Calibration']}
            rows={[
              [<InlineCode key="r">fp16</InlineCode>, 'Half-precision cast with a float32 I/O contract. Inference-only.', 'yolo9, rfdetr', 'none'],
              [<InlineCode key="r">bf16</InlineCode>, 'bfloat16 cast: fp32 exponent range at half storage; the fix when fp16 overflows on DETR-style models. Inference-only.', 'yolo9, rfdetr', 'none'],
              [<InlineCode key="r">fp8</InlineCode>, 'E4M3 weight + activation simulation on Conv2d and Linear.', 'yolo9, rfdetr', 'required'],
              [<InlineCode key="r">int8</InlineCode>, 'W8A8: per-channel INT8 weights, per-tensor affine INT8 activations.', 'yolo9, rfdetr', 'required (calib=None gives weights-only)'],
              [<InlineCode key="r">w4a16</InlineCode>, 'Grouped INT4 weights, float activations, Linear only.', 'rfdetr', 'not needed'],
              [<InlineCode key="r">w4a8</InlineCode>, 'Grouped INT4 weights + INT8 activations; maps to NPU W4A8 deployments.', 'rfdetr', 'required'],
              [<InlineCode key="r">nvfp4</InlineCode>, 'NVFP4 W4A4: E2M1 elements, 16-element blocks, FP8 block scales.', 'rfdetr', 'not needed (dynamic)'],
              [<InlineCode key="r">mxfp4</InlineCode>, 'OCP MXFP4: E2M1 elements, 32-element blocks, power-of-two scales.', 'rfdetr', 'not needed (dynamic)'],
              [<InlineCode key="r">int2</InlineCode>, 'Research preview: grouped 2-bit weights + INT8 activations. PTQ alone is unusable; QAT / QAD required.', 'rfdetr', 'required'],
            ]}
          />
          <P>
            The split is deliberate: sub-8-bit acceleration is GEMM-only on current
            hardware, so the Linear-only recipes are rejected for conv-heavy families
            like YOLO9 (use <InlineCode>int8</InlineCode> or <InlineCode>fp8</InlineCode> there);
            transformer families (RF-DETR) are the target for the 4-bit recipes.
            Per-family <InlineCode>keep_high_precision</InlineCode> defaults protect the first
            layer and the heads; override with{' '}
            <InlineCode>quantize(..., keep_high_precision=(&quot;head.&quot;,))</InlineCode> if you
            know what you are doing.
          </P>

          <SubHeading>Calibration data is not training data</SubHeading>
          <ul className="space-y-2 my-4">
            <FeatureItem><InlineCode>calib=</InlineCode> is a few hundred images, no labels read, forward-only. Its job is activation ranges and scales. Default <InlineCode>coco128.yaml</InlineCode> (auto-downloaded); multiple batches matter because ranges are estimated across them.</FeatureItem>
            <FeatureItem><InlineCode>data=</InlineCode> on train / val is the labeled dataset, for gradients and metrics. Different argument, different job.</FeatureItem>
            <FeatureItem>The default range estimator is <InlineCode>minmax</InlineCode>; <InlineCode>algorithm=&quot;percentile&quot;</InlineCode> exists but measured worse everywhere, and it collapses DETR-family accuracy because transformer activation outliers are load-bearing. What actually fixes small-model int8 sensitivity is calibrating on enough batches: with the coco128 default, YOLO9-t lands within about one mAP point of fp32.</FeatureItem>
          </ul>

          <SubHeading>Honest numbers: simulation first</SubHeading>
          <P>
            v1.4.0 executes quantized arithmetic in <strong>simulation</strong> (fake-quantize
            with straight-through gradients, computed in fp32 islands). Simulation is
            numerics-true: a <InlineCode>val()</InlineCode> score on any device is a real claim
            about the quantized arithmetic. It is <em>not</em> a speed claim; packed low-bit
            kernels are a deployment concern. The <InlineCode>fp16</InlineCode> and{' '}
            <InlineCode>bf16</InlineCode> casts are the exception: they execute natively.
          </P>

          <SubHeading>Deploying quantized models</SubHeading>
          <CodeBlock language="python">{`# Finalize: pack real low-bit weights, strip fp32 masters
qmodel.export(format="pt")    # -> <name>-final.pt

# int8 exports straight to QDQ ONNX with the model's own calibrated scales
qmodel = LibreYOLO("LibreYOLO9s-int8.pt")
qmodel.export(format="onnx")  # ONNX Runtime / TensorRT consume real INT8 kernels`}</CodeBlock>
          <ul className="space-y-2 my-4">
            <FeatureItem>Finalized checkpoints store packed weights and shrink accordingly (measured: YOLO9-s int8 29.5 to 9.6 MB; RF-DETR-n nvfp4 122 to 26 MB), and unpacking reproduces the simulation bit for bit on the device you finalized on. Loading one gives an inference-ready model, and <InlineCode>train()</InlineCode> on it re-prepares masters automatically.</FeatureItem>
            <FeatureItem>For <InlineCode>fp16</InlineCode> / <InlineCode>bf16</InlineCode>, call <InlineCode>dequantize()</InlineCode> and use the float exporters (<InlineCode>half=True</InlineCode> gives fp16 ONNX).</FeatureItem>
            <FeatureItem>The sub-8-bit Linear recipes and <InlineCode>fp8</InlineCode> have no deployable ONNX form yet: they execute in PyTorch and crystallize via <InlineCode>format=&quot;pt&quot;</InlineCode>.</FeatureItem>
            <FeatureItem>In-tree Triton kernels back the simulation, with a pluggable registry and a <InlineCode>LIBREYOLO_QUANT_KERNELS</InlineCode> override.</FeatureItem>
            <FeatureItem>Checkpoints with finalized quant state are not loadable by v1.3.1.</FeatureItem>
          </ul>

          <Divider />

          {/* ────────────── EXPORT ────────────── */}
          <SectionHeading id="export" icon={Upload}>Export</SectionHeading>
          <P>
            Export PyTorch models to ONNX, TorchScript, TensorRT, OpenVINO, NCNN, CoreML, or TFLite for deployment. The heavily tested export paths remain single-GPU YOLO9 detection, RF-DETR detection, and RF-DETR segmentation.
          </P>

          <SubHeading>The support matrix is canonical (new in v1.4.0)</SubHeading>
          <P>
            v1.4.0 replaces guesswork with a canonical export-support matrix. Every
            family / task / format combination has a tier:{' '}
            <strong className="text-emerald-600 dark:text-emerald-400">validated</strong>{' '}
            (tested, safe), <strong className="text-amber-600 dark:text-amber-400">experimental</strong>{' '}
            (exports with a warning), or <strong>blocked</strong> (raises up front instead of
            producing a broken artifact). Query it before you build a pipeline around a
            format:
          </P>
          <CodeBlock language="bash">{`# Formats and tiers for one family / task
libreyolo formats --family yolo9
libreyolo formats --family rfdetr --task segment

# Everything about one model, including its export_support map
libreyolo info --model LibreYOLO9s.pt --json | jq .export_support`}</CodeBlock>
          <P>
            v1.4.0 also unblocked whole task groups under a fixed-resolution, batch-1
            contract: PIDNet (semantic), FOMO (point), and ZipDepth / Depth Anything V2
            (depth) now export to ONNX. Two behavior guarantees landed with the matrix:
            export never mutates the live model before the request is accepted (LoRA
            adapters fold and quantized models re-prepare only after format lookup and
            option preflight succeed), and LoRA adapters are merged into dense weights
            on export, so deployed artifacts need no peft at runtime.
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

# TFLite (needs Python 3.12+); "litert" is an accepted alias
model.export(format="tflite")

# Quantized checkpoints: pack low-bit weights, or emit QDQ INT8 ONNX
qmodel.export(format="pt")      # finalized packed checkpoint
qmodel.export(format="onnx")    # int8 -> QDQ ONNX (see Quantization)`}</CodeBlock>

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
            Pass <InlineCode>nms=True</InlineCode> to bake NMS into an exported ONNX
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

          <SubHeading>TFLite (LiteRT) export</SubHeading>
          <P>
            <SupportBadge variant="experimental">Runtime backend new in v1.4.0</SupportBadge>{' '}
            LibreYOLO has a TFLite export path built on <InlineCode>onnx2tf</InlineCode>. TFLite is
            the format of Google&apos;s LiteRT runtime (TensorFlow Lite was renamed LiteRT in 2024;
            the <InlineCode>.tflite</InlineCode> file format is unchanged). It supports
            YOLO9 and YOLOX detection, the MobileNetV4 / ConvNeXt / EfficientNetV2 /
            ResNet classifiers, PIDNet semantic segmentation, and Real-ESRGAN restoration.
            RF-DETR detection is experimental; RF-DETR segmentation and pose are blocked.
            It requires{' '}
            <strong className="text-surface-800 dark:text-white">Python 3.12+</strong> (the{' '}
            <InlineCode>onnx2tf 2.4.x</InlineCode> wheels do not target older Python) plus the
            optional extra <InlineCode>libreyolo[tflite]</InlineCode> (alias:{' '}
            <InlineCode>libreyolo[litert]</InlineCode>). Export is FP32 and static-shape only
            (no <InlineCode>half</InlineCode>, <InlineCode>int8</InlineCode>, or{' '}
            <InlineCode>dynamic</InlineCode> yet).
          </P>
          <CodeBlock language="bash">{`pip install "libreyolo[tflite]"   # Python 3.12+; [litert] is the same extra`}</CodeBlock>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")
model.export(format="tflite")   # writes a .tflite file; format="litert" also works`}</CodeBlock>
          <P>
            For RF-DETR, the exporter rewrites each GridSample node into a TFLite-safe bilinear
            subgraph because onnx2tf&apos;s default lowering is numerically broken.
          </P>
          <P>
            <strong className="text-surface-800 dark:text-white">TFLite is no longer export-only.</strong>{' '}
            New in v1.4.0, <InlineCode>LibreYOLO(&quot;model.tflite&quot;)</InlineCode> loads the file
            through a LiteRT runtime backend (<InlineCode>ai-edge-litert</InlineCode>), so the same
            factory that runs your ONNX and TensorRT artifacts now runs TFLite too; see{' '}
            <a href="#tflite-inference" className="text-libre-600 dark:text-libre-400 hover:underline">TFLite Inference</a>.
          </P>

          <SubHeading>ONNX metadata</SubHeading>
          <P>Exported ONNX files include embedded metadata:</P>
          <DocTable
            headers={['Key', 'Example value']}
            rows={[
              [<InlineCode key="v">libreyolo_version</InlineCode>, <InlineCode key="vv">&quot;1.4.0&quot;</InlineCode>],
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

          {/* ────────────── TFLITE INFERENCE ────────────── */}
          <SectionHeading id="tflite-inference" icon={Cpu}>TFLite Inference</SectionHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="experimental">New in v1.4.0</SupportBadge>
          </div>
          <P>
            Run an exported <InlineCode>.tflite</InlineCode> file through Google&apos;s LiteRT
            interpreter, the runtime formerly named TensorFlow Lite. Requires Python 3.12+
            and <InlineCode>pip install &quot;libreyolo[tflite]&quot;</InlineCode> (or the{' '}
            <InlineCode>[litert]</InlineCode> alias), which brings in{' '}
            <InlineCode>ai-edge-litert</InlineCode>.
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("model.tflite")

result = model("image.jpg", conf=0.25, iou=0.45, save=True)
print(result.boxes.xyxy)`}</CodeBlock>
          <P>
            TFLite artifacts support the same core runtime prediction API as the other
            backends, including the file-path-only <InlineCode>output_path</InlineCode> behavior
            for <InlineCode>save=True</InlineCode>. Exported graphs are static-shape, so run at
            the exported <InlineCode>imgsz</InlineCode>.
          </P>

          <Divider />

          {/* ────────────── CLI ────────────── */}
          <SectionHeading id="cli" icon={SquareTerminal}>CLI</SectionHeading>
          <P>
            Installing LibreYOLO registers a <InlineCode>libreyolo</InlineCode> command on your PATH (entry point in <InlineCode>pyproject.toml</InlineCode>). The CLI mirrors the Python API and accepts <InlineCode>key=value</InlineCode> syntax.
          </P>

          <SubHeading>Subcommands</SubHeading>
          <DocTable
            headers={['Command', 'Purpose']}
            rows={[
              [<InlineCode key="p">predict</InlineCode>, 'Run inference on images, directories, or videos'],
              [<InlineCode key="t">train</InlineCode>, 'Train a model on a dataset'],
              [<InlineCode key="v">val</InlineCode>, 'Evaluate a model on a dataset'],
              [<InlineCode key="e">export</InlineCode>, 'Export to ONNX / TorchScript / TensorRT / OpenVINO / NCNN / CoreML / TFLite'],
              [<InlineCode key="q">quantize</InlineCode>, 'Quantize a model with a recipe + calibration set (new in v1.4.0)'],
              [<InlineCode key="lb">label</InlineCode>, 'Launch LibreLabel, the local browser annotation tool'],
              [<InlineCode key="mo">monitor</InlineCode>, 'Serve a live dashboard over training runs'],
              [<InlineCode key="pr">profile</InlineCode>, 'Profile training or inference, then analyse the result'],
              [<InlineCode key="ui">ui</InlineCode>, 'Launch a local drag-and-drop / paste browser inference UI'],
              [<InlineCode key="dr">doctor</InlineCode>, 'Run pre-training dataset health checks (YOLO detection format)'],
              [<InlineCode key="c">checks</InlineCode>, 'Print Python, torch, CUDA, GPU, and optional-package info'],
              [<InlineCode key="m">models</InlineCode>, 'List registered model families and CLI shortcut names (enriched in v1.4.0; --json schema changed)'],
              [<InlineCode key="f">formats</InlineCode>, 'List export formats; --family / --task filter by support tier (new in v1.4.0)'],
              [<InlineCode key="cfg">cfg</InlineCode>, 'Print the default training configuration YAML'],
              [<InlineCode key="i">info</InlineCode>, 'Load a model and print family, size, task, device, classes, and its export_support map'],
              [<InlineCode key="md">metadata</InlineCode>, 'Inspect raw checkpoint metadata from a .pt file'],
              [<InlineCode key="ver">version</InlineCode>, 'Print LibreYOLO + Python + torch versions'],
            ]}
          />

          <SubHeading>Model name shortcuts</SubHeading>
          <P>
            The CLI accepts short names (<InlineCode>yolo9-c</InlineCode>) that resolve to weight filenames (<InlineCode>LibreYOLO9c.pt</InlineCode>) - discoverable via <InlineCode>libreyolo models</InlineCode>. You can also pass any explicit checkpoint path.
          </P>
          <P>
            Released-v1.4.0 caveats: without <InlineCode>transformers</InlineCode>,
            <InlineCode>libreyolo models</InlineCode> omits RF-DETR and DINOv2 instead
            of listing them as unavailable. Some task-suffixed shortcuts for
            non-detection-default families also fail to resolve. Install{' '}
            <InlineCode>libreyolo[rfdetr]</InlineCode> for the transformer families,
            and pass the full official checkpoint filename when a shortcut fails.
          </P>

          <SubHeading>Common options</SubHeading>
          <DocTable
            headers={['Command', 'Important options']}
            rows={[
              [<InlineCode key="p">predict</InlineCode>, <span key="pv"><InlineCode>conf</InlineCode>, <InlineCode>iou</InlineCode>, <InlineCode>imgsz</InlineCode>, <InlineCode>classes</InlineCode>, <InlineCode>max_det</InlineCode>, <InlineCode>half</InlineCode>, <InlineCode>batch</InlineCode>, <InlineCode>tiling</InlineCode>, <InlineCode>overlap_ratio</InlineCode>, <InlineCode>output_file_format</InlineCode>, <InlineCode>project</InlineCode>, <InlineCode>name</InlineCode>, <InlineCode>exist_ok</InlineCode>, <InlineCode>face_detector</InlineCode></span>],
              [<InlineCode key="t">train</InlineCode>, <span key="tv"><InlineCode>epochs</InlineCode>, <InlineCode>batch</InlineCode>, <InlineCode>imgsz</InlineCode>, <InlineCode>lr0</InlineCode>, <InlineCode>optimizer</InlineCode>, <InlineCode>scheduler</InlineCode>, <InlineCode>workers</InlineCode>, <InlineCode>seed</InlineCode>, <InlineCode>resume</InlineCode>, <InlineCode>amp</InlineCode>, <InlineCode>task</InlineCode>, <InlineCode>cache</InlineCode>, <InlineCode>lora</InlineCode>, <InlineCode>freeze</InlineCode>, <InlineCode>save_plots</InlineCode>, <InlineCode>allow_download_scripts</InlineCode>, <InlineCode>dry_run</InlineCode>, plus the <a href="#augmentation" className="text-libre-600 dark:text-libre-400 hover:underline">augmentation knobs</a> (<InlineCode>mosaic</InlineCode>, <InlineCode>mixup</InlineCode>, <InlineCode>hsv_prob</InlineCode>, <InlineCode>flip_prob</InlineCode>, <InlineCode>degrees</InlineCode>, ...)</span>],
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

          <SubHeading>Quantize (new in v1.4.0)</SubHeading>
          <CodeBlock language="bash">{`# PTQ: int8 with a small calibration set
libreyolo quantize --model LibreYOLO9s.pt --recipe int8 --calib coco8.yaml --samples 128

# Write to an explicit path, machine-readable output
libreyolo quantize --model LibreRFDETRn.pt --recipe nvfp4 --out rfdetr-nvfp4.pt --json`}</CodeBlock>
          <P>
            The quantized checkpoint then flows through the normal commands:{' '}
            <InlineCode>libreyolo val</InlineCode> for honest accuracy,{' '}
            <InlineCode>libreyolo train</InlineCode> for QAT, and{' '}
            <InlineCode>libreyolo export</InlineCode> for deployment. See{' '}
            <a href="#quantization" className="text-libre-600 dark:text-libre-400 hover:underline">Quantization</a>.
          </P>

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
    points: Points | None = None,
    semantic_mask: SemanticMask | None = None,
    depth_map: DepthMap | None = None,
    restored: RestoredImage | None = None,
    speed: dict[str, float] | None = None,
    track_id = None,
    frame_idx: int | None = None,
    # New in v1.4.0 (placed after the complete v1.3 signature, so
    # positional v1.3 call sites keep working):
    panoptic: PanopticSegmentation | None = None,
    matte: Matte | None = None,
    ocr: OCRRegions | None = None,
    restore_scale: int = 1,
)

len(result)          # number of detections
result.cpu()         # copy with tensors on CPU
result.cuda()        # copy with tensors on CUDA
result.numpy()       # copy with numpy arrays
result.summary()     # list[dict] with the payloads present
result.to_json()     # JSON string from summary()
result.cutout()      # (H, W, 4) RGBA ndarray; matte results only`}</CodeBlock>

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

result.obb.xywhr         # (N, 5): center x/y, width, height, rotation
result.obb.xyxyxyxy      # (N, 4, 2): four oriented box corners
result.obb.conf          # (N,) confidence scores
result.obb.cls           # (N,) class IDs

result.gaze.data         # (N, 2): pitch, yaw in radians
result.gaze.pitch_deg    # pitch in degrees
result.gaze.yaw_deg      # yaw in degrees
result.gaze.direction_3d # approximate 3D direction vectors

result.semantic_mask.data      # (H, W) class-id map (semantic)
result.depth_map.data          # (H, W) relative inverse depth
result.points.xy               # (N, 2) point detections (FOMO)
result.restored.array          # (H, W, 3) uint8 restored image
result.restore_scale           # int upscale factor; 1 unless super-resolution

# New in v1.4.0
result.panoptic.data           # (H, W) segment-id map
result.panoptic.segments_info  # per-segment {"id", "category_id", ...}
result.matte.data              # (H, W) float32 alpha in [0, 1]
result.ocr.polygons            # (N, 4, 2) text-region quads
result.ocr.texts               # list[str] transcriptions
result.ocr.conf                # (N,) recognition scores
result.ocr.det_conf            # (N,) detection scores`}</CodeBlock>

          <SubHeading>model.track()</SubHeading>
          <CodeBlock language="python">{`model.track(
    source,                       # video file path
    *,
    track_conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = None,
    classes: list[int] = None,
    max_det: int = 300,
    save: bool = False,
    show: bool = False,
    vid_stride: int = 1,
    output_path: str = None,
    tracker: str = "bytetrack",   # "bytetrack" | "ocsort" | "botsort" | "deepocsort"
    tracker_config = None,        # a config instance selects the tracker by type
    augment: bool = False,
    **tracker_kwargs,
) -> Generator[Results, None, None]`}</CodeBlock>

          <SubHeading>model.quantize() (new in v1.4.0)</SubHeading>
          <CodeBlock language="python">{`model.quantize(
    recipe: str,                  # "fp16" | "bf16" | "fp8" | "int8" | "w4a16"
                                  # | "w4a8" | "nvfp4" | "mxfp4" | "int2"
    calib: str = "coco128.yaml",  # unlabeled calibration images (forward-only)
    samples: int = 128,
    batch: int = 8,
    algorithm: str = "auto",      # "auto" (minmax) | "minmax" | "percentile"
    keep_high_precision = None,   # module-name substrings to keep in float
    verbose: bool = True,
) -> model                        # quantized in place

model.quant_info()                # dict describing the quant state, or None
model.dequantize()                # restore float modules in place`}</CodeBlock>

          <SubHeading>model.export()</SubHeading>
          <CodeBlock language="python">{`model.export(
    format: str = "onnx",       # "onnx", "torchscript", "tensorrt", "openvino",
                                # "ncnn", "coreml", "tflite" (alias "litert"), or "pt"
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
            Every task loads through one <InlineCode>data.yaml</InlineCode>. Detection, instance segmentation, and OBB accept <strong className="text-surface-800 dark:text-white">two interchangeable label formats</strong> (YOLO TXT or native COCO JSON), and the loader picks the right one from the config. Pose, semantic segmentation, depth, and classification each add a small format of their own. The table maps every task to its layout.
          </P>

          <SubHeading>Formats by task</SubHeading>
          <DocTable
            headers={['Task', 'Data layout', 'Labels']}
            rows={[
              ['Detection', <span key="l"><InlineCode>data.yaml</InlineCode> + <InlineCode>labels/*.txt</InlineCode>, or COCO JSON</span>, 'One box per line'],
              ['Instance segmentation', <span key="l"><InlineCode>data.yaml</InlineCode> + polygon <InlineCode>.txt</InlineCode>, or COCO JSON</span>, 'Polygon per line (TXT) / polygons + RLE (COCO)'],
              ['OBB', <span key="l"><InlineCode>data.yaml</InlineCode> + rotated-box <InlineCode>.txt</InlineCode>, or COCO JSON</span>, 'One rotated box per line'],
              ['Pose', <span key="l"><InlineCode>data.yaml</InlineCode> + <InlineCode>.txt</InlineCode> + <InlineCode>kpt_shape</InlineCode>/<InlineCode>flip_idx</InlineCode></span>, 'Box + keypoints per line'],
              ['Semantic segmentation', <span key="l"><InlineCode>data.yaml</InlineCode> + <InlineCode>masks_dir/</InlineCode> PNGs</span>, 'Per-pixel class ID (255 = ignore)'],
              ['Depth', <span key="l"><InlineCode>data.yaml</InlineCode> + <InlineCode>depths_dir/</InlineCode> maps</span>, 'Per-pixel depth (0 = invalid)'],
              ['Classification', <span key="l">ImageFolder (<InlineCode>train/&lt;class&gt;/</InlineCode>)</span>, 'Folder name = class'],
            ]}
          />

          <SubHeading>data.yaml structure</SubHeading>
          <P>
            The shared contract for detection, segmentation, OBB, and pose. <InlineCode>train</InlineCode>/<InlineCode>val</InlineCode>/<InlineCode>test</InlineCode> may be a directory, a <InlineCode>.txt</InlineCode> file list (one image path per line), or a list of paths. <InlineCode>nc</InlineCode> is optional: when omitted it is inferred from <InlineCode>names</InlineCode>.
          </P>
          <CodeBlock language="yaml" filename="data.yaml">{`path: /absolute/path/to/dataset   # dataset root
train: images/train               # dir, .txt file list, or list of paths
val: images/val
test: images/test                 # optional

nc: 80                            # optional; inferred from names if absent
names: ["person", "bicycle", "car", "..."]`}</CodeBlock>
          <P>
            Configs resolve from an explicit path, the working directory, then the built-ins under <InlineCode>libreyolo/config/datasets/</InlineCode>. Roots default under <InlineCode>~/datasets</InlineCode>; override with <InlineCode>LIBREYOLO_DATASETS_DIR</InlineCode>.
          </P>

          <SubHeading>YOLO TXT labels</SubHeading>
          <P>
            The default layout: one <InlineCode>.txt</InlineCode> per image under <InlineCode>labels/</InlineCode>, mirroring the <InlineCode>images/</InlineCode> tree with the same file stem. All coordinates are normalized to [0, 1].
          </P>
          <CodeBlock language="text">{`dataset/
    images/train/img001.jpg
    labels/train/img001.txt        # same stem as the image`}</CodeBlock>
          <CodeBlock language="text">{`# Detection      one box per line
<class_id> <cx> <cy> <w> <h>

# Segmentation   one polygon per line (box derived from the vertices)
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>

# Pose           box, then K keypoints (needs kpt_shape / flip_idx below)
<class_id> <cx> <cy> <w> <h> <kx1> <ky1> <v1> ... <kxK> <kyK> <vK>

# OBB            four rotated-box corners
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>`}</CodeBlock>
          <CodeBlock language="yaml" filename="data.yaml (pose)">{`kpt_shape: [17, 3]   # K keypoints, 3 values each: x, y, visibility
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]`}</CodeBlock>

          <SubHeading>Native COCO JSON</SubHeading>
          <P>
            Detection, segmentation, and OBB also load COCO JSON directly: add an <InlineCode>annotations:</InlineCode> block mapping each split to its JSON file. <InlineCode>train</InlineCode>/<InlineCode>val</InlineCode> then point at image <em>directories</em> (not <InlineCode>.txt</InlineCode> lists). Requires <InlineCode>pycocotools</InlineCode>; class names come from the JSON categories, so <InlineCode>nc</InlineCode>/<InlineCode>names</InlineCode> are optional.
          </P>
          <CodeBlock language="yaml" filename="data.yaml (COCO)">{`path: dataset
train: images/train               # image directory
val: images/val
annotations:
  train: annotations/train.json   # COCO instances JSON
  val: annotations/val.json`}</CodeBlock>
          <P>
            The same switch feeds YOLO9, RF-DETR, DEIM, and D-FINE training and the detection, OBB, and pose validators. A COCO layout with <InlineCode>annotations/instances_train2017.json</InlineCode> on disk is also detected automatically, without the <InlineCode>annotations:</InlineCode> key.
          </P>
          <P>
            <strong className="text-surface-800 dark:text-white">Which segmentation format?</strong> A YOLO polygon row is a single ring per instance: it cannot express a hole or a split (multi-part) mask. COCO JSON keeps every polygon of an instance and decodes RLE masks, holes included. Use COCO JSON when instances have holes or disconnected parts; either format is fine for simple blobs. Crowd annotations (<InlineCode>iscrowd: 1</InlineCode>) are skipped.
          </P>

          <SubHeading>Semantic segmentation masks</SubHeading>
          <P>
            Pair each image with a single-channel mask whose pixel values are class IDs; <InlineCode>255</InlineCode> marks ignored pixels. <InlineCode>masks_dir</InlineCode> is substituted for <InlineCode>images</InlineCode> in each path (default <InlineCode>masks</InlineCode>), and masks must be lossless (PNG) with the same stem as their image. Optional <InlineCode>label_mapping</InlineCode> remaps source IDs to train IDs (unmapped values become ignore). Omit <InlineCode>masks_dir</InlineCode> to rasterize masks from YOLO polygon labels at load time, with a <InlineCode>background</InlineCode> class appended.
          </P>
          <CodeBlock language="text">{`dataset/
    images/train/scene001.jpg
    masks/train/scene001.png       # single-channel class IDs, 255 = ignore`}</CodeBlock>
          <CodeBlock language="yaml" filename="data.yaml (semantic)">{`path: /path/to/dataset
train: images/train
val: images/val
masks_dir: masks
nc: 3
names: ["road", "building", "vegetation"]`}</CodeBlock>

          <SubHeading>Depth maps</SubHeading>
          <P>
            Pair each image with a single-channel depth map under <InlineCode>depths_dir</InlineCode> (default <InlineCode>depths</InlineCode>). 16-bit PNG/TIF is divided by <InlineCode>depth_scale</InlineCode> (default <InlineCode>256.0</InlineCode>); <InlineCode>.npy</InlineCode> float files are used as-is. Zero, negative, and non-finite pixels are invalid. An optional <InlineCode>depth_stem_suffix</InlineCode> and a <InlineCode>*_mask</InlineCode> validity map are honored automatically. Depth is validation-only.
          </P>
          <CodeBlock language="yaml" filename="data.yaml (depth)">{`path: /path/to/dataset
val: images/val
depths_dir: depths
depth_scale: 256.0                # 16-bit PNG encoding: value / 256 = depth`}</CodeBlock>

          <SubHeading>Classification</SubHeading>
          <P>
            Classification uses an ImageNet-style ImageFolder tree instead of a <InlineCode>data.yaml</InlineCode> - see <a href="#classification" className="text-libre-600 dark:text-libre-400 hover:underline">Classification</a> for the layout. <InlineCode>data=</InlineCode> takes a dataset root, a <InlineCode>.zip</InlineCode> URL, or a known name.
          </P>

          <SubHeading>Built-in datasets</SubHeading>
          <P>
            Configs ship under <InlineCode>libreyolo/config/datasets/</InlineCode>. Download behavior differs per config: URL-backed sets fetch on first use, script-backed sets need <InlineCode>allow_download_scripts=True</InlineCode>, and a few must be placed locally.
          </P>
          <DocTable
            headers={['Config', 'Task', 'Download']}
            rows={[
              [<InlineCode key="c">coco8</InlineCode>, 'Detection (8 images)', 'Automatic'],
              [<InlineCode key="c">coco128</InlineCode>, 'Detection (128 images)', 'Automatic'],
              [<InlineCode key="c">coco5000</InlineCode>, 'Detection', <span key="d">Script: <InlineCode>allow_download_scripts=True</InlineCode></span>],
              [<span key="c"><InlineCode>coco</InlineCode> / <InlineCode>coco-val-only</InlineCode></span>, 'Detection (full)', <span key="d">Script: <InlineCode>allow_download_scripts=True</InlineCode></span>],
              [<span key="c"><InlineCode>coco8-pose</InlineCode> / <InlineCode>coco-pose</InlineCode></span>, 'Pose', <span key="d">Script: <InlineCode>allow_download_scripts=True</InlineCode></span>],
              [<InlineCode key="c">cocostuff</InlineCode>, 'Semantic (182 classes)', 'Manual: place locally'],
            ]}
          />
          <CodeBlock language="python">{`results = model.val(data="coco8.yaml")                          # auto-downloads
results = model.train(data="coco128.yaml", epochs=10)           # auto-downloads
model.train(data="coco8-pose.yaml", allow_download_scripts=True)  # script config`}</CodeBlock>

          {/* Bottom spacer */}
          <div className="h-16" />
        </div>
      </main>
    </div>
  )
}


/* ============================================================================
 * Chinese (zh-CN) content bundle for the LibreYOLO v1.4.0 docs page.
 * Rendered as <DocsPageZh /> when locale === 'zh'. Reuses all shared
 * presentational components from this file.
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
  { id: 'ensembling', title: '模型集成', icon: Boxes },
  { id: 'segmentation', title: '实例分割', icon: Scissors },
  { id: 'semantic-segmentation', title: '语义分割', icon: Palette },
  { id: 'panoptic-segmentation', title: '全景分割', icon: Combine },
  { id: 'promptable-segmentation', title: '可提示分割', icon: MousePointerClick },
  { id: 'open-vocabulary', title: '开放词表检测', icon: Search },
  { id: 'pose', title: '姿态估计', icon: PersonStanding },
  { id: 'gaze', title: '视线估计', icon: Eye },
  { id: 'classification', title: '分类', icon: Tags },
  { id: 'depth', title: '深度估计', icon: Mountain },
  { id: 'restoration', title: '图像修复与超分', icon: WandSparkles },
  { id: 'background-removal', title: '背景移除', icon: Eraser },
  { id: 'ocr', title: '文字识别（OCR）', icon: ScanText },
  { id: 'point-localization', title: '点定位', icon: MapPin },
  { id: 'annotation', title: '标注（LibreLabel）', icon: PenTool },
  { id: 'training', title: '训练', icon: GraduationCap },
  { id: 'augmentation', title: '数据增强', icon: Dices },
  { id: 'distillation', title: '知识蒸馏', icon: FlaskConical },
  { id: 'monitoring', title: '训练监控', icon: Gauge },
  { id: 'profiling', title: '性能分析', icon: Timer },
  { id: 'validation', title: '验证', icon: CheckCircle2 },
  { id: 'quantization', title: '模型量化', icon: Binary },
  { id: 'export', title: '导出', icon: Upload },
  { id: 'torchscript-inference', title: 'TorchScript 推理', icon: Cpu },
  { id: 'onnx-inference', title: 'ONNX 推理', icon: Cpu },
  { id: 'tensorrt-inference', title: 'TensorRT 推理', icon: Cpu },
  { id: 'openvino-inference', title: 'OpenVINO 推理', icon: Cpu },
  { id: 'ncnn-inference', title: 'NCNN 推理', icon: Cpu },
  { id: 'coreml-inference', title: 'CoreML 推理', icon: Cpu },
  { id: 'tflite-inference', title: 'TFLite 推理', icon: Cpu },
  { id: 'cli', title: '命令行（CLI）', icon: SquareTerminal },
  { id: 'api-reference', title: 'API 参考', icon: FileCode },
  { id: 'architecture', title: '架构指南', icon: Wrench },
  { id: 'dataset-format', title: '数据集格式', icon: Database },
]

/* ─── 2. Sidebar (Chinese) ─── */
function SidebarZh({ activeSection, onNavigate, currentVersion = 'v1.3.1', className = '' }) {
  const versionLabelZh = {
    'Pre-release': '预发布',
    'Latest': '最新',
    'Previous': '上一版本',
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
            v1.4.0 验证范围
          </p>
          <p className="text-sm text-surface-600 dark:text-surface-400 mb-2">
            经过充分测试的路径是 YOLO9 和 RF-DETR 的检测、训练与推理，包括 RF-DETR 分割。
          </p>
          <p className="text-sm text-surface-600 dark:text-surface-400">
            其他模型系列和任务均可使用，但仍属实验性。多 GPU 训练在 v1.4.0 中经历了正确性大修，比 v1.3.x 可靠得多，但仍在已验证范围之外。
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
            这些模型的检测、训练与推理经过最充分的测试。在 v1.4.0 中，请将其他系列、任务以及多 GPU 工作流视为实验性。
          </p>
        </div>
      </div>
    </div>
  )
}

function CompatibilityMatrixZh() {
  return <CompatibilityMatrix zh />
}

/* ─── 4. Main docs page (Chinese) ─── */
function DocsPageZh({ version = 'v1.3.1', isLatest = true }) {
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
                    ? '这是当前稳定版本 v1.4.0 的文档。早期版本仍可在版本菜单中找到。'
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

          <SupportCallout className="mb-8" community={false} />

          {/* ────────────── SPECIALIZED GUIDES ────────────── */}
          <P>
            两份配套指南更深入地介绍专门主题。{' '}
            <a href="/docs/librevlm" className="text-libre-600 dark:text-libre-400 hover:underline">LibreVLM 指南</a>{' '}
            介绍视觉语言模型层（Qwen3-VL、Florence-2），它生成文本，再由 LibreYOLO 解析成框。这与{' '}
            <a href="#open-vocabulary" className="text-libre-600 dark:text-libre-400 hover:underline">开放词表检测</a>{' '}
            是两回事：后者使用以文本为条件的专用检测器，并在本页中说明。{' '}
            <a href="/docs/experimental" className="text-libre-600 dark:text-libre-400 hover:underline">实验性任务指南</a>{' '}
            介绍更多实验性工作流，包括 LoRA / DoRA 微调。
          </P>

          {/* ────────────── INTRODUCTION ────────────── */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
            <SectionHeading id="introduction" icon={BookOpen}>简介</SectionHeading>
            <ValidationScopeCalloutZh />
            <P>
              LibreYOLO 是一个采用 MIT 许可的计算机视觉工具包。v1.4.0 提供了涵盖检测、分割、分类、深度、图像修复、OCR 等的广泛模型目录，但其经过验证的支持范围是有意收窄的：
            </P>
            <ul className="space-y-2 mb-4">
              <FeatureItem><strong className="text-surface-800 dark:text-white">YOLO9 检测</strong> - CNN 路径。</FeatureItem>
              <FeatureItem><strong className="text-surface-800 dark:text-white">RF-DETR 检测</strong> - transformer 路径。</FeatureItem>
              <FeatureItem><strong className="text-surface-800 dark:text-white">RF-DETR 分割</strong> - 经过充分测试的分割路径。</FeatureItem>
            </ul>
            <P>
              我们建议将这些路径作为新项目的默认选择，因为它们在检测、训练与推理方面经过了最充分的测试。其他受支持的系列和任务通过同一个统一的 <InlineCode>LibreYOLO()</InlineCode> 工厂工作，但在 v1.4.0 中属于实验性。如果你有特定理由，可以使用它们。
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
              <FeatureItem>对 YOLO9 检测、RF-DETR 检测和 RF-DETR 分割进行了充分测试并作为推荐默认项</FeatureItem>
              <FeatureItem>统一的 <InlineCode>LibreYOLO()</InlineCode> 工厂，用于加载检查点、导出产物和运行时</FeatureItem>
              <FeatureItem>通过一致的 API 完成检测、实例 / 语义 / 全景分割、姿态、分类、深度、图像修复、背景移除、OCR、点定位和视线任务</FeatureItem>
              <FeatureItem>支持图像、目录和视频推理（大尺寸帧可选分块推理）</FeatureItem>
              <FeatureItem>内置多目标跟踪：ByteTrack、OC-SORT、BoT-SORT，以及带 ReID 的 Deep OC-SORT</FeatureItem>
              <FeatureItem>按系列细化的<a href="#augmentation" className="text-libre-600 dark:text-libre-400 hover:underline">训练数据增强控制</a>，配有声明式支持规范，参数被忽略时会发出警告</FeatureItem>
              <FeatureItem>PyTorch 原生<a href="#quantization" className="text-libre-600 dark:text-libre-400 hover:underline">量化</a>：fp16 / bf16 / fp8 / int8 / int4 配方，支持 QAT 与 QAD 精度恢复</FeatureItem>
              <FeatureItem>ONNX、TorchScript、TensorRT、OpenVINO、NCNN、CoreML 和 TFLite 导出，内嵌元数据，并配有相应的运行时后端</FeatureItem>
              <FeatureItem>兼容 COCO 的验证，提供 mAP 指标，并包含分割、姿态、全景、抠图和 OCR 验证器</FeatureItem>
              <FeatureItem><InlineCode>libreyolo</InlineCode> 命令行工具，用于 predict / train / val / export / quantize</FeatureItem>
              <FeatureItem>接受任意图像格式：文件路径、URL、PIL、NumPy、PyTorch 张量、原始字节</FeatureItem>
            </ul>
          </motion.div>

          <SubHeading>v1.4.0 新特性</SubHeading>
          <ul className="space-y-2 my-4">
            <FeatureItem>
              <strong className="text-surface-800 dark:text-white">15 个新模型系列</strong>，包括 SegFormer（语义分割）、SwinIR 和 Real-ESRGAN（超分辨率）、BiRefNet（背景移除）、ZipDepth 和 Depth Anything 3（深度）、PP-OCR（文字）、SigLIP2（零样本分类）、SAM 3、EdgeTAM 和 PicoSAM3（提示式分割），以及 OmDet-Turbo 和 OV-DEIM（开放词表检测）。
            </FeatureItem>
            <FeatureItem>
              <strong className="text-surface-800 dark:text-white">三个新任务</strong>：<InlineCode>panoptic</InlineCode>、<InlineCode>matte</InlineCode> 和 <InlineCode>ocr</InlineCode>，各自拥有独立的结果类型和验证器。
            </FeatureItem>
            <FeatureItem>
              <strong className="text-surface-800 dark:text-white">成体系的数据增强文档。</strong>每个训练增强参数现在都有按系列的支持规范，CLI 会在所选系列忽略你显式设置的参数时发出警告，并且本页终于有了完整的<a href="#augmentation" className="text-libre-600 dark:text-libre-400 hover:underline">数据增强</a>章节。
            </FeatureItem>
            <FeatureItem>
              <strong className="text-surface-800 dark:text-white">量化栈</strong>：<InlineCode>model.quantize()</InlineCode> 与 <InlineCode>libreyolo quantize</InlineCode>，提供九种配方、基于仿真的可信精度、QAT / QAD 恢复，以及打包低比特权重的检查点。
            </FeatureItem>
            <FeatureItem>
              <strong className="text-surface-800 dark:text-white">两个新跟踪器</strong>：BoT-SORT 和 Deep OC-SORT（外观 ReID），与 ByteTrack、OC-SORT 并列可选。
            </FeatureItem>
            <FeatureItem>
              <strong className="text-surface-800 dark:text-white">多 GPU 正确性大修</strong>：所有系列的 DDP 分片都已修正，损失归一化因子做全局规约，BatchNorm 密集的系列默认开启 SyncBatchNorm，配置错误会在启动时大声报错而不是悄悄跑错。
            </FeatureItem>
            <FeatureItem>
              <strong className="text-surface-800 dark:text-white">YOLOv7 可训练</strong>（该系列在 v1.3.1 中仅支持推理），LoRA 微调扩展到七个新系列，新增 DINOv2 基础模型蒸馏教师和 TFLite 运行时后端。
            </FeatureItem>
          </ul>

          <SubHeading>v1.4.0 兼容性说明</SubHeading>
          <ul className="space-y-2 my-4">
            <FeatureItem>
              <strong className="text-surface-800 dark:text-white">检查点只向前兼容。</strong>使用新任务字符串（<InlineCode>panoptic</InlineCode>、<InlineCode>matte</InlineCode>、<InlineCode>ocr</InlineCode>）或已定型量化状态的检查点无法被 v1.3.1 加载。v1.3.x 写出的所有内容都能在 v1.4.0 中加载，且 <InlineCode>Results</InlineCode> 和 <InlineCode>LibreEoMT</InlineCode> 完整保留 v1.3 的位置参数兼容性。
            </FeatureItem>
            <FeatureItem>
              <strong className="text-surface-800 dark:text-white">部分微调默认值已修改</strong>，因为旧值有害：PicoDet（<InlineCode>lr0</InlineCode> 从 0.1 改为 0.01）和 DEIM（<InlineCode>lr0</InlineCode> 从 4e-4 改为 1e-4，<InlineCode>min_lr_ratio</InlineCode> 从 0.5 改为 0.05）。如需复现上游 COCO 配方，请显式传入旧值。
            </FeatureItem>
            <FeatureItem>
              <strong className="text-surface-800 dark:text-white">训练结果可能有所偏移</strong>：语义分割训练现在默认应用 HSV 抖动，图像修复训练新增成对的垂直翻转和 rot90，AdamW 不再对 BatchNorm / bias 参数施加权重衰减。
            </FeatureItem>
            <FeatureItem>
              <strong className="text-surface-800 dark:text-white"><InlineCode>model.train(profile=True)</InlineCode> 在分析窗口结束后继续训练</strong>，而不再停止。传入 <InlineCode>profile_then_stop=True</InlineCode> 可恢复旧行为。
            </FeatureItem>
            <FeatureItem>
              <strong className="text-surface-800 dark:text-white"><InlineCode>libreyolo models --json</InlineCode> 的模式已变更</strong>（带任务后缀的 CLI 名称、新增键）；<InlineCode>formats</InlineCode> 和 <InlineCode>info</InlineCode> 的 JSON 也增加了键。请更新解析它们的脚本。
            </FeatureItem>
          </ul>

          <Divider />

          {/* ────────────── COMPATIBILITY ────────────── */}
          <SectionHeading id="compatibility" icon={CheckCircle2}>兼容性</SectionHeading>
          <P>
            可将此矩阵作为 v1.4.0 的快速支持速查表。<InlineCode>&#10003;</InlineCode>{' '}
            表示受支持的路径，{' '}
            <InlineCode>prev</InlineCode> 表示研究预览，空白单元格表示当前不支持。YOLO9 和 RF-DETR 检测（外加 RF-DETR
            分割）测试最充分，也是推荐的起点；其他系列同样受支持，如遇问题请提交 issue。
          </P>
          <CompatibilityMatrixZh />
          <p className="text-sm text-surface-600 dark:text-surface-400 leading-relaxed mb-4">
            导出各列汇总了 v1.4.0 内置的规范化导出支持矩阵；可用{' '}
            <InlineCode>libreyolo formats --family ...</InlineCode> 或{' '}
            <InlineCode>libreyolo info --model ... --json</InlineCode> 精确查询（见
            <a href="#export" className="text-libre-600 dark:text-libre-400 hover:underline">导出</a>）。
            提示式分割层（SAM、SAM 2、SAM 3、MobileSAM、EdgeTAM、PicoSAM3）、开放词表层（Grounding
            DINO、OWLv2、OmDet-Turbo、OV-DEIM）和 VLM 层在 <InlineCode>LibreYOLO()</InlineCode>{' '}
            工厂之外，不列入此表；请参见各自章节。CoreML 导出会生成 <InlineCode>.mlpackage</InlineCode>{' '}
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
            v1.4.0 是 PyPI 上的当前版本，也是本文档所描述的版本。本页所有内容都可直接使用已发布的软件包，无需源码安装。
          </P>

          <SubHeading>从源码安装</SubHeading>
          <CodeBlock language="bash">{`git clone https://github.com/LibreYOLO/libreyolo.git
cd libreyolo
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

# TFLite export + LiteRT runtime backend (Python 3.12+)
pip install libreyolo[tflite]
# "litert" is an alias extra: pip install libreyolo[litert]

# Tracking API compatibility extra
pip install libreyolo[tracking]
# Tracking dependencies are part of the base install; Deep OC-SORT's ReID
# embedder weights auto-download on first use.

# CoreML export and inference (macOS only for runtime)
pip install libreyolo[coreml]
# or: pip install coremltools

# L2CS gaze optional auto-download helper
pip install libreyolo[gaze]

# Promptable segmentation (LibreSAM: SAM-1, SAM-2, SAM 3, MobileSAM,
# EdgeTAM, PicoSAM3)
pip install libreyolo[sam]

# Open-vocabulary detection (Grounding DINO, OWLv2, OmDet-Turbo, OV-DEIM)
pip install libreyolo[openvocab]

# LibreLabel AI assist (SAM click-to-mask)
pip install libreyolo[label]

# Zero-shot classification
pip install libreyolo[clip]       # CLIP
pip install libreyolo[siglip2]    # SigLIP2 tokenizer (SentencePiece)

# Validation and training plots
pip install libreyolo[plots]

# SenseNova Vision preview
pip install libreyolo[sensenova]

# Converter-only dependencies for CLIP and SigLIP2 checkpoints
pip install libreyolo[clip-convert]
pip install libreyolo[siglip2-convert]

# LoRA fine-tuning (peft)
pip install libreyolo[lora]

# Experiment loggers
pip install libreyolo[tensorboard]   # or [mlflow], [wandb]

# EoMT instance / panoptic segmentation
pip install libreyolo[eomt]

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
            LibreYOLO v1.4.0 提供两个已验证的旗舰系列，以及广泛的受支持模型目录：仅本次发布就新增了十五个系列。每个基于检查点的模型都通过同一个 <InlineCode>LibreYOLO()</InlineCode> 工厂加载，但只有下面的已验证路径才应被视为经过充分测试。
          </P>

          <ValidatedModelHeader title="YOLO9 - CNN 旗舰">
            <SupportBadge variant="validated">默认：LibreYOLO9c.pt</SupportBadge>
            <SupportBadge variant="validated">充分测试：检测、训练与推理</SupportBadge>
            <SupportBadge>v1.4.0 中仅检测</SupportBadge>
            <SupportBadge>可量化：int8 / fp8</SupportBadge>
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
            在 v1.4.0 中 YOLO9 仅支持检测。非检测的旗舰变体（包括旧的 <InlineCode>-seg</InlineCode> 检查点）已在 v1.3.0 中移除；如需分割，请使用 RF-DETR 或下文的实验性分割系列。
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")   # detection`}</CodeBlock>

          <ValidatedModelHeader title="RF-DETR - transformer 旗舰">
            <SupportBadge variant="validated">推荐的 transformer 路径</SupportBadge>
            <SupportBadge variant="validated">充分测试：检测、分割、训练与推理</SupportBadge>
            <SupportBadge>研究预览：pose、OBB</SupportBadge>
            <SupportBadge>支持 LoRA 与全部量化配方</SupportBadge>
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
            使用上游 RF-DETR seg-XL / seg-2XL 权重，采用非商业许可：商用前请查看模型卡。请参见{' '}
            <a href="#segmentation" className="text-libre-600 dark:text-libre-400 hover:underline">分割</a> 章节。
          </P>
          <P>
            <SupportBadge>研究预览</SupportBadge>{' '}
            <strong className="text-surface-800 dark:text-white">姿态：</strong>{' '}
            <Checkpoints names={['LibreRFDETRx-pose.pt']} />（移植自 RF-DETR 的 GroupPose）。{' '}
            <strong className="text-surface-800 dark:text-white">OBB：</strong>{' '}
            <Checkpoints names={['LibreRFDETRn-obb.pt', 'LibreRFDETRs-obb.pt', 'LibreRFDETRm-obb.pt', 'LibreRFDETRl-obb.pt']} />{' '}
            （旋转框，使用检测的输入尺寸）。这些检查点针对六个车辆类别训练：
            <InlineCode>bike</InlineCode>、<InlineCode>bus</InlineCode>、
            <InlineCode>car</InlineCode>、<InlineCode>other_vehicle</InlineCode>、
            <InlineCode>taxi</InlineCode> 和 <InlineCode>truck</InlineCode>，并非 COCO-80 模型。
            请将姿态与 OBB 都视为研究预览，而非已验证路径。
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreRFDETRs.pt")           # detect (validated)
# model = LibreYOLO("LibreRFDETRs-seg.pt")     # segment (validated)
# model = LibreYOLO("LibreRFDETRx-pose.pt")    # pose  (research preview)
# model = LibreYOLO("LibreRFDETRn-obb.pt")     # obb   (research preview)`}</CodeBlock>

          <SubHeading>其他检测系列</SubHeading>
          <P>
            与已验证路径共享同一工厂和 API 接口的可检测系列。它们在 v1.4.0 中为实验性。每个检查点名称都链接到{' '}
            <a href="https://huggingface.co/LibreYOLO" target="_blank" rel="noopener noreferrer" className="text-libre-600 dark:text-libre-400 hover:underline">LibreYOLO 组织</a>上的模型卡；
            将任意名称传给 <InlineCode>LibreYOLO()</InlineCode>，工厂会在首次使用时自动获取。
          </P>
          <DocTable
            headers={['系列', '状态', '任务', '检查点']}
            rows={[
              ['YOLOX', <SupportBadge key="b">实验性</SupportBadge>, 'detect', <Checkpoints key="yolox" names={['LibreYOLOXn.pt', 'LibreYOLOXt.pt', 'LibreYOLOXs.pt', 'LibreYOLOXm.pt', 'LibreYOLOXl.pt', 'LibreYOLOXx.pt']} />],
              ['YOLOv7', <SupportBadge key="b">实验性</SupportBadge>, 'detect（v1.4.0 起可训练）', <Checkpoints key="y7" names={['LibreYOLO7b.pt']} />],
              ['YOLO9-E2E', <SupportBadge key="b">实验性</SupportBadge>, 'detect', <Checkpoints key="y9e2e" names={['LibreYOLO9E2Et.pt', 'LibreYOLO9E2Es.pt', 'LibreYOLO9E2Em.pt', 'LibreYOLO9E2Ec.pt']} />],
              ['YOLO-NAS', <SupportBadge key="b">实验性</SupportBadge>, 'detect, pose', <Checkpoints key="ynas" link={false} names={['LibreYOLONASs.pt', 'LibreYOLONASm.pt', 'LibreYOLONASl.pt', 'LibreYOLONASn-pose.pt', 'LibreYOLONASs-pose.pt', 'LibreYOLONASm-pose.pt', 'LibreYOLONASl-pose.pt']} />],
              ['D-FINE', <SupportBadge key="b">实验性</SupportBadge>, 'detect, segment（v1.4.0 新增）', <Checkpoints key="dfine" names={['LibreDFINEn.pt', 'LibreDFINEs.pt', 'LibreDFINEm.pt', 'LibreDFINEl.pt', 'LibreDFINEx.pt', 'LibreDFINEn-seg.pt', 'LibreDFINEs-seg.pt', 'LibreDFINEm-seg.pt', 'LibreDFINEl-seg.pt', 'LibreDFINEx-seg.pt']} />],
              ['DEIM', <SupportBadge key="b">实验性</SupportBadge>, 'detect', <Checkpoints key="deim" names={['LibreDEIMn.pt', 'LibreDEIMs.pt', 'LibreDEIMm.pt', 'LibreDEIMl.pt', 'LibreDEIMx.pt']} />],
              ['DEIMv2', <SupportBadge key="b">实验性</SupportBadge>, 'detect', <Checkpoints key="deimv2" names={['LibreDEIMv2atto.pt', 'LibreDEIMv2femto.pt', 'LibreDEIMv2pico.pt', 'LibreDEIMv2n.pt', 'LibreDEIMv2s.pt', 'LibreDEIMv2m.pt', 'LibreDEIMv2l.pt', 'LibreDEIMv2x.pt']} />],
              ['RT-DETR', <SupportBadge key="b">实验性</SupportBadge>, 'detect', <Checkpoints key="rtdetr" names={['LibreRTDETRr18.pt', 'LibreRTDETRr34.pt', 'LibreRTDETRr50.pt', 'LibreRTDETRr50m.pt', 'LibreRTDETRr101.pt', 'LibreRTDETRl.pt', 'LibreRTDETRx.pt']} />],
              ['RT-DETRv2', <SupportBadge key="b">实验性</SupportBadge>, 'detect', <Checkpoints key="rtdetrv2" names={['LibreRTDETRv2r18.pt', 'LibreRTDETRv2r34.pt', 'LibreRTDETRv2r50.pt', 'LibreRTDETRv2r50m.pt', 'LibreRTDETRv2r101.pt']} />],
              ['RT-DETRv4', <SupportBadge key="b">实验性</SupportBadge>, 'detect', <Checkpoints key="rtdetrv4" names={['LibreRTDETRv4s.pt', 'LibreRTDETRv4m.pt', 'LibreRTDETRv4l.pt', 'LibreRTDETRv4x.pt']} />],
              ['PicoDet', <SupportBadge key="b">实验性</SupportBadge>, 'detect', <Checkpoints key="picodet" names={['LibrePICODETs.pt', 'LibrePICODETm.pt', 'LibrePICODETl.pt']} />],
              ['RTMDet', <SupportBadge key="b">实验性</SupportBadge>, 'detect, segment（RTMDet-Ins，仅推理与验证）', <Checkpoints key="rtmdet" names={['LibreRTMDett.pt', 'LibreRTMDets.pt', 'LibreRTMDetm.pt', 'LibreRTMDetl.pt', 'LibreRTMDetx.pt', 'LibreRTMDett-seg.pt', 'LibreRTMDets-seg.pt', 'LibreRTMDetm-seg.pt', 'LibreRTMDetl-seg.pt', 'LibreRTMDetx-seg.pt']} />],
              ['EdgeCrafter', <SupportBadge key="b">实验性</SupportBadge>, 'detect, pose, segment', <Checkpoints key="ec" names={['LibreECs.pt', 'LibreECm.pt', 'LibreECl.pt', 'LibreECx.pt', 'LibreECs-pose.pt', 'LibreECm-pose.pt', 'LibreECl-pose.pt', 'LibreECx-pose.pt', 'LibreECs-seg.pt', 'LibreECm-seg.pt', 'LibreECl-seg.pt', 'LibreECx-seg.pt']} />],
            ]}
          />
          <P className="text-sm">
            <strong className="text-surface-800 dark:text-white">托管说明：</strong>{' '}
            YOLO-NAS 检查点（上方纯文本）托管在 Deci 的 CDN 上，采用其专有权重许可，
            而非 LibreYOLO 的 Hugging Face 组织。工厂仍会在首次使用时自动下载它们。
            DAMO-YOLO 已在 v1.3.0 中移除，不再可加载。
          </P>

          <SubHeading>v1.4.0 中的新模型系列</SubHeading>
          <P>
            v1.4.0 新增了十五个模型系列。基于检查点的系列通过同一个{' '}
            <InlineCode>LibreYOLO()</InlineCode> 工厂加载；提示式分割和开放词表条目则直接构造
            （见各自章节）。它们全部为实验性。
          </P>
          <DocTable
            headers={['系列', '任务', '尺寸', '检查点 / 加载方式']}
            rows={[
              ['SegFormer', 'semantic', 'b0-b5（512；b5 为 640）', <Checkpoints key="segf" names={['LibreSegformerb0-sem.pt', 'LibreSegformerb1-sem.pt', 'LibreSegformerb2-sem.pt', 'LibreSegformerb3-sem.pt', 'LibreSegformerb4-sem.pt', 'LibreSegformerb5-sem.pt']} />],
              ['SwinIR', 'restore（4 倍超分）', 's / m / l', <Checkpoints key="swinir" names={['LibreSwinIRs-restore.pt', 'LibreSwinIRm-restore.pt', 'LibreSwinIRl-restore.pt']} />],
              ['Real-ESRGAN', 'restore（超分辨率）', 'x4 / x2 / x4t', <Checkpoints key="resr" names={['LibreRealESRGANx4-restore.pt', 'LibreRealESRGANx2-restore.pt', 'LibreRealESRGANx4t-restore.pt']} />],
              ['BiRefNet', 'matte（背景移除）', 't / l（1024）', <span key="birefnet"><Checkpoints names={['LibreBiRefNetl-matte.pt']} />；t 权重暂未重新托管</span>],
              ['ZipDepth', 'depth', 'b / bnpu（384）', <Checkpoints key="zip" names={['LibreZipDepthb-depth.pt', 'LibreZipDepthbnpu-depth.pt']} />],
              ['Depth Anything 3', 'depth', 'l（504）', <Checkpoints key="da3" names={['LibreDepthAnything3l-depth.pt']} />],
              ['PP-OCR (v5)', 'ocr', 't / l（960）', <Checkpoints key="ppocr" names={['LibrePPOCRt-ocr.pt', 'LibrePPOCRl-ocr.pt']} />],
              ['SigLIP2', '零样本 classify', 'b16 / so400m', <Checkpoints key="siglip" names={['LibreSigLIP2b16-cls.pt', 'LibreSigLIP2so400m-cls.pt']} />],
              ['YOLOv1', 'detect（博物馆级）', 't / b（448，VOC-20）', <span key="y1"><Checkpoints names={['LibreYOLO1b.pt']} />；tiny 权重已在上游遗失</span>],
              ['SAM 3', '提示式分割', 'large（1008）', <span key="sam3"><InlineCode>LibreSAM3()</InlineCode>：Hugging Face 门控权重，Meta SAM 许可</span>],
              ['EdgeTAM', '提示式分割', 'edge（1024）', <span key="edgetam"><InlineCode>LibreEdgeTAM()</InlineCode>：仅图像推理，Apache-2.0</span>],
              ['PicoSAM3', '提示式 ROI 分割', '96 px', <span key="picosam"><InlineCode>LibrePicoSAM3()</InlineCode>：原生移植，仅支持 ONNX 导出</span>],
              ['OmDet-Turbo', '开放词表检测', 't', <span key="omdet"><InlineCode>LibreOpenVocab(&quot;omdet-turbo&quot;)</InlineCode></span>],
              ['OV-DEIM', '开放词表检测（无 NMS）', 's / m / l', <span key="ovdeim"><InlineCode>LibreOpenVocab(&quot;ov-deim&quot;)</InlineCode>；权重为 CC BY-NC 4.0</span>],
              ['SenseNova Vision', '七任务多模态预览', '7B', <span key="sense">实验性：尚未进入 CLI、UI 或模型清单；权重为 CC BY-NC 4.0</span>],
            ]}
          />
          <ul className="space-y-2 my-4">
            <FeatureItem><strong className="text-surface-800 dark:text-white">既有系列也扩展了任务：</strong>EoMT 新增实例分割和全景分割检查点，RTMDet 新增 RTMDet-Ins 实例分割（推理与验证），D-FINE 新增实验性分割并发布了权重。</FeatureItem>
            <FeatureItem><strong className="text-surface-800 dark:text-white">许可因系列而异。</strong>SwinIR、EdgeTAM 和 Depth Anything 3 从代码到权重都是 Apache-2.0。SegFormer 代码为 Apache-2.0，但转换自 NVIDIA 的 ADE20K 权重为非商业许可（下载前会显示许可提示）。OV-DEIM 和 SenseNova 权重为 CC BY-NC 4.0。SAM 3 权重在 Hugging Face 上门控，采用 Meta SAM 许可。商用前请查看模型卡。</FeatureItem>
          </ul>

          <SubHeading>检测之外的任务系列</SubHeading>
          <P>
            这些系列沿袭自早期版本，各自在对应任务章节中有完整说明。DINOv2 需要{' '}
            <InlineCode>pip install libreyolo[rfdetr]</InlineCode>（transformers）。
          </P>
          <DocTable
            headers={['系列', '任务', '文档位置']}
            rows={[
              ['MobileNetV4 / ConvNeXt / EfficientNetV2 / ResNet', 'classify', <a key="l" href="#classification" className="text-libre-600 dark:text-libre-400 hover:underline">分类</a>],
              ['CLIP / SigLIP2', '零样本 classify', <a key="l" href="#classification" className="text-libre-600 dark:text-libre-400 hover:underline">分类</a>],
              ['DINOv2', 'semantic, classify, detect', <a key="l" href="#semantic-segmentation" className="text-libre-600 dark:text-libre-400 hover:underline">语义分割</a>],
              ['PIDNet / EoMT / SegFormer', 'semantic', <a key="l" href="#semantic-segmentation" className="text-libre-600 dark:text-libre-400 hover:underline">语义分割</a>],
              ['EoMT', 'panoptic', <a key="l" href="#panoptic-segmentation" className="text-libre-600 dark:text-libre-400 hover:underline">全景分割</a>],
              ['Depth Anything V2 / Depth Anything 3 / ZipDepth', 'depth', <a key="l" href="#depth" className="text-libre-600 dark:text-libre-400 hover:underline">深度估计</a>],
              ['NAFNet / SwinIR / Real-ESRGAN', 'restore', <a key="l" href="#restoration" className="text-libre-600 dark:text-libre-400 hover:underline">图像修复与超分</a>],
              ['BiRefNet', 'matte', <a key="l" href="#background-removal" className="text-libre-600 dark:text-libre-400 hover:underline">背景移除</a>],
              ['PP-OCR', 'ocr', <a key="l" href="#ocr" className="text-libre-600 dark:text-libre-400 hover:underline">文字识别（OCR）</a>],
              ['FOMO', 'point', <a key="l" href="#point-localization" className="text-libre-600 dark:text-libre-400 hover:underline">点定位</a>],
            ]}
          />
          <P className="text-sm">
            <strong className="text-surface-800 dark:text-white">提示式、开放词表与 VLM 档位：</strong>{' '}
            LibreSAM（提示式分割，<InlineCode>libreyolo[sam]</InlineCode>）、
            LibreOpenVocab（开放词表检测，<InlineCode>libreyolo[openvocab]</InlineCode>）
            和 LibreVLM 视觉语言检测器档位
            （<InlineCode>libreyolo[vlm]</InlineCode>）是单独的类别，它们加载
            Hugging Face 快照，且不通过{' '}
            <InlineCode>LibreYOLO()</InlineCode> 检查点工厂路由。它们的权重继承
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

# The task suffix selects the task
model = LibreYOLO("LibreMobileNetV4s-cls.pt")   # classification (Apache, ImageNet-1k)
model = LibreYOLO("LibreDINOv2n.pt")            # semantic segmentation
model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")  # monocular depth
model = LibreYOLO("LibreFOMOs-point.pt")        # point localization (local weights)

# New in v1.4.0
model = LibreYOLO("LibreSegformerb2-sem.pt")    # semantic segmentation
model = LibreYOLO("LibreEoMTb-panoptic.pt")     # panoptic segmentation
model = LibreYOLO("LibreSwinIRm-restore.pt")    # 4x super-resolution
model = LibreYOLO("LibreBiRefNetl-matte.pt")    # background removal
model = LibreYOLO("LibrePPOCRt-ocr.pt")         # OCR (text detection + recognition)
model = LibreYOLO("LibreZipDepthb-depth.pt")    # depth

# Exported deployment formats
model = LibreYOLO("model.onnx")                 # ONNX Runtime
model = LibreYOLO("model.engine")               # TensorRT
model = LibreYOLO("model.mlpackage")            # CoreML (macOS)
model = LibreYOLO("model_openvino/")            # OpenVINO (directory)
model = LibreYOLO("model_ncnn/")                # NCNN (directory)
model = LibreYOLO("model.tflite")               # TFLite / LiteRT (new in v1.4.0)`}</CodeBlock>
          <P>
            对于可识别的官方检查点文件名，LibreYOLO 可以自动下载缺失的权重。对于自定义文件名，请指向明确的本地路径。
            新项目请保持使用 YOLO9 检测或 RF-DETR 检测 / 分割；
            其他系列、任务以及新增系列在 v1.4.0 中为实验性。
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
              ['实例分割', <InlineCode key="s">&quot;segment&quot;</InlineCode>, <InlineCode key="ss">-seg</InlineCode>, 'RF-DETR、EdgeCrafter、RTMDet-Ins、D-FINE、EoMT'],
              ['语义分割', <InlineCode key="se">&quot;semantic&quot;</InlineCode>, <InlineCode key="ses">-sem</InlineCode>, 'DINOv2、PIDNet、EoMT、SegFormer'],
              ['全景分割', <InlineCode key="pa">&quot;panoptic&quot;</InlineCode>, <InlineCode key="pas">-panoptic</InlineCode>, 'EoMT（v1.4.0 新增）'],
              ['姿态估计', <InlineCode key="p">&quot;pose&quot;</InlineCode>, <InlineCode key="ps">-pose</InlineCode>, 'YOLO-NAS、EdgeCrafter、RF-DETR（预览）'],
              ['旋转框', <InlineCode key="o">&quot;obb&quot;</InlineCode>, <InlineCode key="os">-obb</InlineCode>, 'RF-DETR（预览）'],
              ['分类', <InlineCode key="c">&quot;classify&quot;</InlineCode>, <InlineCode key="cs">-cls</InlineCode>, 'MobileNetV4、ConvNeXt、EfficientNetV2、ResNet、DINOv2；CLIP 与 SigLIP2 零样本'],
              ['单目深度', <InlineCode key="de">&quot;depth&quot;</InlineCode>, <InlineCode key="des">-depth</InlineCode>, 'Depth Anything V2 / 3、ZipDepth'],
              ['图像修复', <InlineCode key="r">&quot;restore&quot;</InlineCode>, <InlineCode key="rs">-restore</InlineCode>, 'NAFNet、SwinIR、Real-ESRGAN'],
              ['背景移除', <InlineCode key="ma">&quot;matte&quot;</InlineCode>, <InlineCode key="mas">-matte</InlineCode>, 'BiRefNet（v1.4.0 新增）'],
              ['文字识别', <InlineCode key="oc">&quot;ocr&quot;</InlineCode>, <InlineCode key="ocs">-ocr</InlineCode>, 'PP-OCR（v1.4.0 新增）'],
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
            headers={['系列', 'v1.4.0 状态', '默认', '受支持任务']}
            rows={[
              [<strong key="y9">YOLO9</strong>, '单 GPU detect 充分测试；多 GPU 实验性', 'detect', 'detect'],
              [<strong key="rfd">RF-DETR</strong>, '单 GPU detect 和 segment 充分测试；pose 和 OBB 研究预览', 'detect', 'detect, segment, pose, obb'],
              ['YOLOX', '实验性', 'detect', 'detect'],
              ['YOLOv7', '实验性；v1.4.0 起可训练', 'detect', 'detect'],
              ['YOLO9-E2E', '实验性', 'detect', 'detect'],
              ['YOLO-NAS', '实验性；v1.4.0 新增多类别姿态训练', 'detect', 'detect, pose'],
              ['D-FINE', '实验性；segment 为 v1.4.0 新增', 'detect', 'detect, segment'],
              ['DEIM / DEIMv2', '实验性', 'detect', 'detect'],
              ['RT-DETR / RT-DETRv2 / RT-DETRv4', '实验性', 'detect', 'detect'],
              ['PicoDet', '实验性', 'detect', 'detect'],
              ['RTMDet', '实验性；RTMDet-Ins segment（仅推理与验证）为 v1.4.0 新增', 'detect', 'detect, segment'],
              ['EdgeCrafter (EC)', '实验性', 'detect', 'detect, pose, segment'],
              ['YOLO1 / YOLO2 / YOLO3 / YOLO4', '博物馆级（仅推理）；YOLO1 为 v1.4.0 新增', 'detect', 'detect'],
              ['PIDNet', '实验性', 'semantic', 'semantic（仅推理与验证）'],
              ['EoMT', '实验性；实例与全景为 v1.4.0 新增', 'semantic', 'semantic, segment, panoptic（仅推理与验证）'],
              ['SegFormer', 'v1.4.0 新增，实验性', 'semantic', 'semantic'],
              ['DINOv2', '实验性', 'semantic', 'semantic, classify, detect'],
              ['MobileNetV4 / ConvNeXt / EfficientNetV2 / ResNet', '实验性', 'classify', 'classify'],
              ['CLIP / SigLIP2', '实验性；SigLIP2 为 v1.4.0 新增', 'classify', '零样本 classify（仅推理）'],
              ['Depth Anything V2 / Depth Anything 3 / ZipDepth', '实验性；DA3 与 ZipDepth 为 v1.4.0 新增', 'depth', 'depth（仅推理与验证）'],
              ['NAFNet', '实验性', 'restore', 'restore'],
              ['SwinIR / Real-ESRGAN', 'v1.4.0 新增，实验性', 'restore', 'restore（仅推理与验证）'],
              ['BiRefNet', 'v1.4.0 新增，实验性', 'matte', 'matte（仅推理与验证）'],
              ['PP-OCR', 'v1.4.0 新增，实验性', 'ocr', 'ocr（仅推理与验证）'],
              ['FOMO', '实验性', 'point', 'point'],
              ['L2CS', '实验性', 'gaze', 'gaze（仅推理）'],
            ]}
          />
          <P>
            有三个档位位于 <InlineCode>LibreYOLO()</InlineCode> 工厂<em>之外</em>，需要直接导入：<a href="#promptable-segmentation" className="text-libre-600 dark:text-libre-400 hover:underline">LibreSAM</a>（提示式分割，现已包含 SAM 3、EdgeTAM 和 PicoSAM3）、<a href="#open-vocabulary" className="text-libre-600 dark:text-libre-400 hover:underline">LibreOpenVocab</a>（开放词表检测，现已包含 OmDet-Turbo 和 OV-DEIM）和 <a href="/docs/librevlm" className="text-libre-600 dark:text-libre-400 hover:underline">LibreVLM</a>。它们不是检查点系列，因此 <InlineCode>LibreYOLO(&quot;sam_b&quot;)</InlineCode> 之类的写法不会解析。
          </P>

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
LibreSegformerb0-sem.pt  # SegFormer requires the -sem suffix

# Panoptic segmentation (-panoptic)
LibreEoMTs-panoptic.pt

# Pose (-pose)
LibreYOLONASn-pose.pt
LibreECs-pose.pt
LibreRFDETRx-pose.pt     # preview

# Oriented boxes (-obb)
LibreRFDETRn-obb.pt      # preview

# Classification (-cls)
LibreMobileNetV4s-cls.pt
LibreConvNeXtt-cls.pt
LibreEfficientNetV2b0-cls.pt
# LibreDINOv2 classify checkpoints are not publicly shipped in v1.4.0
LibreSigLIP2b16-cls.pt   # zero-shot

# Depth (-depth)
LibreDepthAnythingV2s-depth.pt
LibreZipDepthb-depth.pt

# Restoration / super-resolution (-restore)
LibreNAFNetl-restore-sidd.pt
LibreSwinIRm-restore.pt
LibreRealESRGANx4-restore.pt

# Background removal (-matte)
LibreBiRefNetl-matte.pt

# OCR (-ocr)
LibrePPOCRt-ocr.pt

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
            单 GPU 预测路径在 YOLO9 检测、RF-DETR 检测和 RF-DETR 分割上经过充分测试。其他系列和任务使用相同的 API，但在 v1.4.0 中为实验性。
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
    output_path="out/",   # images: directory; video: final file path
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
            <InlineCode>model.predict()</InlineCode> 接受由内存中图像组成的列表或元组
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
            <InlineCode>model.info()</InlineCode> 返回一个 JSON 友好的字典，包含
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
            在 v1.4.0 中分块仅支持检测。它会拒绝分割掩码，且不能与 <InlineCode>augment=True</InlineCode> 组合使用。
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
          <P>
            对于视频，<InlineCode>output_path</InlineCode> 必须是完整文件名，例如
            <InlineCode>out/clip.mp4</InlineCode>，不能只传目录。v1.4.0 的逐帧
            <InlineCode>Results</InlineCode> 不会填充 <InlineCode>saved_path</InlineCode>；
            请使用传入的路径或上方所示的默认位置。
          </P>

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
            LibreYOLO 提供四种运动跟踪器，它们消费任意检测器的 <InlineCode>Results</InlineCode> 并添加持久的轨迹 ID：<strong className="text-surface-800 dark:text-white">ByteTrack</strong>（默认）、
            <strong className="text-surface-800 dark:text-white">OC-SORT</strong>（对遮挡和非线性运动更鲁棒），以及 v1.4.0 新增的{' '}
            <strong className="text-surface-800 dark:text-white">BoT-SORT</strong>（相机运动补偿）和{' '}
            <strong className="text-surface-800 dark:text-white">Deep OC-SORT</strong>（在 OC-SORT 之上加入外观 ReID）。在{' '}
            <InlineCode>model.track()</InlineCode> 上用{' '}
            <InlineCode>tracker=&quot;bytetrack&quot;</InlineCode> / <InlineCode>&quot;ocsort&quot;</InlineCode> /{' '}
            <InlineCode>&quot;botsort&quot;</InlineCode> / <InlineCode>&quot;deepocsort&quot;</InlineCode> 选择。
            跟踪在单 GPU 的 YOLO9 检测和 RF-DETR 检测上测试最充分；其他检测系列在 v1.4.0 中为实验性。
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

          <SubHeading>BoT-SORT（相机运动补偿）</SubHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="experimental">v1.4.0 新增</SupportBadge>
          </div>
          <P>
            BoT-SORT 在 ByteTrack 的关联框架上加入相机运动补偿（CMC）：用稀疏光流估计整帧的全局运动，并在匹配前对轨迹预测位置做相应的变换。当移动的是<em>相机</em>本身时（手持拍摄、无人机、车载相机），优先尝试它。LibreYOLO 的移植是纯运动版（不含 ReID 分支）。
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO, BoTSortConfig

model = LibreYOLO("LibreYOLO9c.pt")

# Select by name with defaults
for result in model.track("drone.mp4", tracker="botsort", save=True):
    print(result.frame_idx, result.track_id)

# Or configure it: a config instance selects the tracker by type
cfg = BoTSortConfig(
    track_high_thresh=0.25,
    track_buffer=30,
    enable_cmc=True,              # camera-motion compensation on (default)
    cmc_method="sparseOptFlow",   # the shipped CMC estimator
    cmc_downscale=2,              # estimate flow at half resolution
)
for result in model.track("drone.mp4", tracker_config=cfg, save=True):
    print(result.frame_idx, result.track_id)`}</CodeBlock>
          <P>
            <InlineCode>BoTSortTracker</InlineCode> 和 <InlineCode>BoTSortConfig</InlineCode> 与{' '}
            <InlineCode>ByteTracker</InlineCode> 一样从顶层导出，可用于手动的{' '}
            <InlineCode>tracker.update(result)</InlineCode> 循环。
          </P>

          <SubHeading>Deep OC-SORT（外观 ReID）</SubHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="experimental">v1.4.0 新增</SupportBadge>
          </div>
          <P>
            Deep OC-SORT 在 OC-SORT 之上加入外观嵌入分支：被遮挡后消失的目标可以按<em>长相</em>重新识别，而不只是按运动轨迹外推。默认嵌入器是 OSNet-AIN，首次使用时从{' '}
            <a href="https://huggingface.co/LibreYOLO/LibreReID-osnet" target="_blank" rel="noopener noreferrer" className="text-libre-600 dark:text-libre-400 hover:underline">LibreYOLO/LibreReID-osnet</a>{' '}
            自动下载，并运行在与检测器相同的设备上。额外的前向计算使 Deep OC-SORT 成为四者中最慢的：当你的问题是 ID 切换而非速度时再选它。
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO
from libreyolo.tracking import DeepOCSortConfig

model = LibreYOLO("LibreYOLO9c.pt")

# Defaults: OSNet-AIN embedder, auto-downloaded
for result in model.track("mall.mp4", tracker="deepocsort", save=True):
    print(result.frame_idx, result.track_id)

# Tune the appearance term, or plug in your own embedder
cfg = DeepOCSortConfig(
    det_thresh=0.25,
    embedder="osnet_ain_x0_25",   # or a callable: (frame, boxes_xyxy) -> (N, D) features
    w_association_emb=0.75,       # weight of appearance vs motion in matching
    alpha_fixed_emb=0.95,         # EMA smoothing of per-track embeddings
)
for result in model.track("mall.mp4", tracker_config=cfg, save=True):
    print(result.frame_idx, result.track_id)`}</CodeBlock>
          <P>
            <InlineCode>DeepOCSortTracker</InlineCode> 和 <InlineCode>DeepOCSortConfig</InlineCode> 位于{' '}
            <InlineCode>libreyolo.tracking</InlineCode>（不是顶层导出）。支持自定义嵌入器可调用对象：任何把一帧图像和{' '}
            <InlineCode>(N, 4)</InlineCode> 框映射为 <InlineCode>(N, D)</InlineCode> 特征的函数都可以。
          </P>

          <SubHeading>如何选择跟踪器</SubHeading>
          <DocTable
            headers={['跟踪器', '选择方式', '优势', '开销']}
            rows={[
              ['ByteTrack', <InlineCode key="s">tracker=&quot;bytetrack&quot;</InlineCode>, '快速、简单、默认', '最低'],
              ['OC-SORT', <InlineCode key="s">tracker=&quot;ocsort&quot;</InlineCode>, '遮挡与非线性运动', '低'],
              ['BoT-SORT', <InlineCode key="s">tracker=&quot;botsort&quot;</InlineCode>, '移动相机（CMC）', '中（每帧光流）'],
              ['Deep OC-SORT', <InlineCode key="s">tracker=&quot;deepocsort&quot;</InlineCode>, 'ID 切换、长时间遮挡后重识别', '最高（嵌入器前向）'],
            ]}
          />

          <Divider />

          {/* ────────────── SEGMENTATION ────────────── */}
          <SectionHeading id="ensembling" icon={Boxes}>模型集成</SectionHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="experimental">仅检测</SupportBadge>
            <SupportBadge variant="experimental">仅 Python API</SupportBadge>
          </div>
          <P>
            <InlineCode>LibreEnsemble</InlineCode> 同时运行两个或更多检测模型，并把它们的检测结果融合成一个普通的 <InlineCode>Results</InlineCode>。融合发生在检测层面，而不是张量层面，因此每个成员都保留自己的输入尺寸、归一化和 NMS。这正是你可以把网格检测器与 DETR、或把 <InlineCode>.pt</InlineCode> 检查点与已导出后端混在同一个集成里的原因。
          </P>
          <P>
            类别空间不必一致。成员按类别<em>名称</em>统一：名称映射相同则直接通过，否则 LibreYOLO 会构建并集并把每个成员重映射进去。只有属于同一统一类别的框才会相互融合。
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreEnsemble

# 加权框融合（WBF，默认），只保留两个模型都找到的框
ens = LibreEnsemble(["LibreYOLO9s.pt", "LibreRFDETRs.pt"], min_votes=2)

result = ens("image.jpg", conf=0.25)
print(result.boxes.xyxy)
print(result.names)     # 统一后的（并集）类别表`}</CodeBlock>
          <CodeBlock language="python">{`ens = LibreEnsemble(
    ["LibreYOLO9s.pt", "LibreRFDETRs.pt"],
    weights=[1.0, 1.4],     # 让融合结果更偏向第二个成员
    fusion="wbf",           # "wbf" | "wbf_seeded" | "nms" | 自定义可调用对象
    fusion_iou=0.55,        # 用于聚类的 IoU，不是成员的 NMS
    min_votes=1,
)

result = ens("image.jpg", conf=[0.25, 0.4])   # 每个成员单独的置信度阈值`}</CodeBlock>
          <SubHeading>限制</SubHeading>
          <ul className="space-y-2 my-4">
            <FeatureItem>只支持检测成员。任何非 <InlineCode>detect</InlineCode> 任务的成员都会报错，分割与姿态模型无法集成。</FeatureItem>
            <FeatureItem>至少需要两个成员。</FeatureItem>
            <FeatureItem><InlineCode>min_votes</InlineCode> 大于 1 时必须使用投票式融合，配合 <InlineCode>fusion=&quot;nms&quot;</InlineCode> 会报错，请改用 <InlineCode>wbf</InlineCode>。</FeatureItem>
            <FeatureItem>仅支持图像与图像目录。视频源与 <InlineCode>stream=True</InlineCode> 会报错，请分别对每个成员单独处理视频。</FeatureItem>
            <FeatureItem><InlineCode>ens.val()</InlineCode> 与 <InlineCode>ens.export()</InlineCode> 都会报错，请对成员单独验证与导出。</FeatureItem>
          </ul>

          <Divider />

          <SectionHeading id="segmentation" icon={Scissors}>实例分割</SectionHeading>
          <ValidationScopeCalloutZh />
          <P>
            RF-DETR 分割是 v1.4.0 中经过充分测试的分割路径。它周围还有四个实验性选项：EdgeCrafter（<InlineCode>-seg</InlineCode>），以及 v1.4.0 新增的 RTMDet-Ins、EoMT 实例分割和 D-FINE 分割。YOLO9 不提供分割头：它仅支持检测。
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

          <SubHeading>v1.4.0 中的新分割系列</SubHeading>
          <P>
            v1.4.0 有三个系列获得了实例分割能力，全部为实验性，且都返回同样的{' '}
            <InlineCode>boxes + masks</InlineCode> 结果：
          </P>
          <DocTable
            headers={['系列', '检查点', '支持范围']}
            rows={[
              ['RTMDet-Ins', <Checkpoints key="c" names={['LibreRTMDett-seg.pt', 'LibreRTMDets-seg.pt', 'LibreRTMDetm-seg.pt', 'LibreRTMDetl-seg.pt', 'LibreRTMDetx-seg.pt']} />, '推理与验证；训练未实现'],
              ['EoMT（实例）', <Checkpoints key="c" names={['LibreEoMTl-seg.pt', 'LibreEoMTl-seg-1280.pt']} />, '推理与验证；-1280 变体以速度换取高分辨率掩码'],
              ['D-FINE', <Checkpoints key="c" names={['LibreDFINEn-seg.pt', 'LibreDFINEs-seg.pt', 'LibreDFINEm-seg.pt', 'LibreDFINEl-seg.pt', 'LibreDFINEx-seg.pt']} />, '推理、验证与实验性训练；CLI 训练会自动把 detect 权重迁移到 segment'],
            ]}
          />
          <P>
            D-FINE 分割与其 ONNX 和 TensorRT 导出已验证一致。注意分块推理（<InlineCode>tiling=True</InlineCode>）会对分割模型直接报错，而不是悄悄丢掉掩码。
          </P>

          <SubHeading>训练分割</SubHeading>
          <P>
            RF-DETR 分割使用 RF-DETR 的 COCO 格式训练流水线，属于经过充分测试的单 GPU 范围。EdgeCrafter 与 D-FINE 分割训练可用，但为实验性。分割专用的增强（copy-paste）见<a href="#augmentation" className="text-libre-600 dark:text-libre-400 hover:underline">数据增强</a>。
          </P>

          <Divider />

          {/* ────────────── POSE ESTIMATION ────────────── */}
          <SectionHeading id="semantic-segmentation" icon={Palette}>语义分割</SectionHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="experimental">实验性</SupportBadge>
            <SupportBadge variant="experimental">SegFormer 与 TTA 为 v1.4.0 新增</SupportBadge>
          </div>
          <P>
            语义分割为<em>每个像素</em>赋予一个类别。它与实例分割是不同的任务：没有目标实例，也没有检测框，只有一张稠密的类别图。使用 <InlineCode>task=&quot;semantic&quot;</InlineCode>，并从 <InlineCode>result.semantic_mask</InlineCode> 读取结果。对语义模型而言，<InlineCode>result.boxes</InlineCode> 与 <InlineCode>result.masks</InlineCode> 均为 <InlineCode>None</InlineCode>。
          </P>
          <DocTable
            headers={['系列', '检查点', '主干', '训练数据', '类别数', '可训练？']}
            rows={[
              ['LibrePIDNet', 'LibrePIDNet{s,m,l}-sem.pt', 'PIDNet 三分支 CNN', 'Cityscapes', '19', '否'],
              ['LibreEoMT', 'LibreEoMTl-sem.pt', 'DINOv2 ViT-L', 'ADE20K', '150', '否'],
              ['LibreSegformer（v1.4.0 新增）', 'LibreSegformer{b0..b5}-sem.pt', 'MiT 分层 transformer', 'ADE20K', '150', '是（可微调）'],
              ['LibreDINOv2', '未发布权重：需自行训练', 'DINOv2 + 稠密头', '你的数据', '自定义', '是'],
            ]}
          />
          <P>
            各系列差别很大，请有意识地选择。<InlineCode>LibrePIDNet</InlineCode> 是快速的实时 CNN，携带 Cityscapes 街景类别；<InlineCode>LibreEoMT</InlineCode> 携带 ADE20K 的 150 个通用场景类别。两者都提供预训练权重，但<strong>无法在 LibreYOLO 内训练</strong>：请在上游微调后再转换权重。
          </P>
          <P>
            v1.4.0 新增的 <InlineCode>LibreSegformer</InlineCode> 走中间路线：SegFormer 架构的逐位一致移植，六个尺寸（b0 到 b5，512 px；b5 为 640），带 ADE20K 预训练权重，<strong>并且</strong>提供微调训练器，可以从 150 个通用类别出发适配你自己的数据。许可上有一点要注意：代码为 Apache-2.0，但转换自 NVIDIA 的 ADE20K 权重为<strong>非商业许可</strong>，下载前会先显示许可提示。
          </P>
          <P>
            <InlineCode>LibreDINOv2</InlineCode> 是不带预训练头的微调系列：<strong>没有发布任何 LibreDINOv2 语义检查点</strong>。你需要基于预训练的 DINOv2 主干加一个全新的稠密头，用自己的掩码训练它。当你的类别既不是 Cityscapes 也不是 ADE20K，又想在全新头下用最强的特征时，就选它。
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibrePIDNets-sem.pt")   # Cityscapes，19 类
result = model.predict("street.jpg", save=True)

sm = result.semantic_mask     # SemanticMask
print(sm.data.shape)          # (H, W) 整数类别 id，位于原图尺寸上
print(sm.classes)             # 出现过的类别 id，已排除 255（忽略）
car = sm.class_mask(13)       # (H, W) 布尔掩码

print(result.saved_path)      # 已保存的语义分割叠加图
print(result.boxes, result.masks)   # None None：语义分割没有实例`}</CodeBlock>
          <SubHeading>验证</SubHeading>
          <CodeBlock language="python">{`metrics = model.val(data="cityscapes.yaml")
print(metrics["metrics/mIoU"])
print(metrics["metrics/pixel_accuracy"])`}</CodeBlock>
          <SubHeading>训练（仅 LibreDINOv2）</SubHeading>
          <P>掩码是单通道无损图像，像素值即类别 id，按文件名与图像配对。<InlineCode>255</InlineCode> 表示忽略，不参与损失与指标。</P>
          <CodeBlock language="bash">{`dataset/
    images/train/*.jpg
    masks/train/*.png      # 与图像同名；像素值 = 类别 id`}</CodeBlock>
          <CodeBlock language="python">{`from libreyolo import LibreDINOv2

model = LibreDINOv2(model_path=None, size="s", task="semantic", nb_classes=19)
model.train(data="cityscapes.yaml", epochs=100, batch_size=4, lr=1e-4)`}</CodeBlock>
          <SubHeading>测试时增强（v1.4.0 新增）</SubHeading>
          <P>
            语义模型现在在 <InlineCode>predict()</InlineCode> 和 <InlineCode>val()</InlineCode> 上都接受{' '}
            <InlineCode>augment=True</InlineCode>（v1.3.1 中会报错）。TTA 额外运行一次水平翻转前向并对 logits
            取平均，用约 2 倍的推理开销换取小而稳定的 mIoU 提升。PIDNet、SegFormer、EoMT 和 DINOv2 语义分割均已实现。
          </P>
          <CodeBlock language="python">{`model = LibreYOLO("LibreSegformerb2-sem.pt")
result = model.predict("street.jpg", augment=True)   # flip-TTA
metrics = model.val(data="ade20k.yaml", augment=True)`}</CodeBlock>

          <SubHeading>限制</SubHeading>
          <ul className="space-y-2 my-4">
            <FeatureItem><strong>导出支持因系列而异。</strong>PIDNet 支持 ONNX、TorchScript、NCNN 和 TFLite；DINOv2 与 EoMT 语义分割支持 ONNX 和 TorchScript；SegFormer 导出仍被拦截。用 <InlineCode>libreyolo formats --family ...</InlineCode> 查询准确档位。</FeatureItem>
            <FeatureItem><strong>只有 LibreDINOv2 和 LibreSegformer 可训练。</strong><InlineCode>LibrePIDNet.train()</InlineCode> 与 <InlineCode>LibreEoMT.train()</InlineCode> 都会报错。</FeatureItem>
            <FeatureItem>语义分割训练现在<strong>默认应用 HSV 颜色抖动</strong>（v1.4.0 新增），重新训练的 mIoU 可能与 v1.3.1 略有偏移。该开关来自系列本身而非 <InlineCode>hsv_prob</InlineCode>；见<a href="#augmentation" className="text-libre-600 dark:text-libre-400 hover:underline">数据增强</a>。</FeatureItem>
            <FeatureItem><strong>EoMT 语义分割仅有 <InlineCode>l</InlineCode> 尺寸，且 <InlineCode>imgsz</InlineCode> 固定为 512</strong>（其检查点使用固定位置编码），并且不支持批处理。</FeatureItem>
            <FeatureItem><InlineCode>imgsz</InlineCode> 的整除要求因系列而异：PIDNet 需被 8 整除，EoMT 为 16，DINOv2 为 14，SegFormer 为 32。</FeatureItem>
            <FeatureItem>语义模型不支持跟踪。</FeatureItem>
            <FeatureItem>Cityscapes、ADE20K 与 COCO-Stuff 均需手动下载，LibreYOLO 只提供数据集 YAML。</FeatureItem>
          </ul>

          <Divider />

          {/* ────────────── PANOPTIC SEGMENTATION ────────────── */}
          <SectionHeading id="panoptic-segmentation" icon={Combine}>全景分割</SectionHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="experimental">v1.4.0 新增</SupportBadge>
            <SupportBadge variant="experimental">仅推理与验证</SupportBadge>
          </div>
          <P>
            全景分割同时回答两个问题：每个像素都有类别（像语义分割），可数目标又被拆成独立实例（像实例分割）。道路和天空作为单个 &quot;stuff&quot; 段返回；每辆车、每个人作为独立的 &quot;thing&quot; 段返回。结果是一张段 id 图加一份逐段信息列表，从 <InlineCode>result.panoptic</InlineCode> 读取。
          </P>
          <P>
            v1.4.0 中由一个系列提供该任务：<InlineCode>LibreEoMT</InlineCode> 的 COCO-panoptic 检查点（133 类，640 px），共三个尺寸：{' '}
            <Checkpoints names={['LibreEoMTs-panoptic.pt', 'LibreEoMTb-panoptic.pt', 'LibreEoMTl-panoptic.pt']} />。
          </P>

          <SubHeading>运行全景分割</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreEoMTb-panoptic.pt")   # task resolved from the -panoptic suffix
result = model.predict("street.jpg", save=True)

pan = result.panoptic             # PanopticSegmentation
print(pan.data.shape)             # (H, W) integer segment ids, original canvas
for seg in pan.segments_info:     # one dict per segment
    print(seg["id"], seg["category_id"], model.names[seg["category_id"]])

car_mask = pan.segment_mask(3)    # (H, W) bool mask for one segment id
print(result.saved_path)          # saved panoptic overlay

# Flip-TTA works here too (new in v1.4.0)
result = model.predict("street.jpg", augment=True)`}</CodeBlock>

          <SubHeading>PanopticSegmentation API</SubHeading>
          <CodeBlock language="python">{`pan = result.panoptic
pan.data                 # (H, W) int segment-id map at original resolution
pan.segments_info        # list of {"id", "category_id", ...} dicts
pan.segment_ids          # ids present in the map
pan.segment_mask(sid)    # (H, W) bool mask for one segment

pan.cpu(); pan.numpy()`}</CodeBlock>

          <SubHeading>用 Panoptic Quality 验证</SubHeading>
          <P>
            验证通过 <InlineCode>PanopticValidator</InlineCode> 运行，报告 Panoptic Quality（PQ）：这一标准指标把分割质量（匹配段的平均 IoU）与识别质量（段级 F1）相乘。<InlineCode>augment=True</InlineCode> 可启用翻转 TTA。
          </P>
          <CodeBlock language="python">{`metrics = model.val(data="coco_panoptic.yaml")
print(metrics["metrics/PQ"])`}</CodeBlock>

          <SubHeading>限制</SubHeading>
          <ul className="space-y-2 my-4">
            <FeatureItem><strong>仅推理与验证。</strong>v1.4.0 中全景分割的训练与导出都会报错。</FeatureItem>
            <FeatureItem>使用 <InlineCode>panoptic</InlineCode> 任务字符串写出的检查点无法被 v1.3.1 加载。</FeatureItem>
            <FeatureItem>对全景模型而言，<InlineCode>result.boxes</InlineCode> 与 <InlineCode>result.masks</InlineCode> 均为 <InlineCode>None</InlineCode>：一切都在 <InlineCode>result.panoptic</InlineCode> 里。</FeatureItem>
          </ul>

          <Divider />

          <SectionHeading id="promptable-segmentation" icon={MousePointerClick}>可提示分割</SectionHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="experimental">仅 Python API</SupportBadge>
            <SupportBadge variant="experimental">仅推理</SupportBadge>
            <SupportBadge variant="experimental">SAM 3、EdgeTAM、PicoSAM3 为 v1.4.0 新增</SupportBadge>
          </div>
          <P>
            LibreSAM 是独立于检测器工厂的一层，因为可提示分割器的调用契约不同：它先运行一次重量级图像编码器，随后用廉价的空间提示（点击、框）换取掩码。没有固定的类别表。安装：<InlineCode>pip install &quot;libreyolo[sam]&quot;</InlineCode>。
          </P>
          <P>
            有两点常令人意外。其一，<InlineCode>LibreSAM</InlineCode> 是<strong>工厂函数而非类</strong>，并且刻意<strong>不在</strong> <InlineCode>LibreYOLO()</InlineCode> 加载器之内，因此 <InlineCode>LibreYOLO(&quot;sam_b&quot;)</InlineCode> 无效，请直接导入。其二，整个层<strong>仅支持 Python</strong>，没有任何 CLI 入口。
          </P>
          <DocTable
            headers={['系列', '传给 LibreSAM()', '编码器', '备注']}
            rows={[
              ['SAM-1', '"base"（默认）, "large", "huge"', 'ViT-B / L / H', 'Apache-2.0'],
              ['SAM-2.1', '"sam2-tiny", "sam2-small", "sam2-base-plus", "sam2-large"', 'Hiera', '仅图像，不支持视频'],
              ['MobileSAM', '"mobilesam"', 'TinyViT', '最快'],
            ]}
          />

          <SubHeading>v1.4.0 新增：SAM 3、EdgeTAM、PicoSAM3</SubHeading>
          <P>
            三个新成员覆盖了从最高质量到最小体积的光谱。它们直接构造，快照在首次使用时自动下载。
          </P>
          <DocTable
            headers={['模型', '构造方式', '工作尺寸', '备注']}
            rows={[
              ['SAM 3', <InlineCode key="c">LibreSAM3()</InlineCode>, '1008', '质量最高；基于 transformers。权重在 Hugging Face 上门控，采用 Meta SAM 许可：首次使用前需接受条款并登录。'],
              ['EdgeTAM', <InlineCode key="c">LibreEdgeTAM()</InlineCode>, '1024', '面向边缘端；仅图像推理；从代码到权重均为 Apache-2.0。'],
              ['PicoSAM3', <InlineCode key="c">LibrePicoSAM3()</InlineCode>, '96', '面向极小裁剪图的原生 ROI 分割移植；SAM 层中唯一支持导出的模型（仅 ONNX）。'],
            ]}
          />
          <CodeBlock language="python">{`from libreyolo import LibreSAM3, LibreEdgeTAM, LibrePicoSAM3

model = LibreSAM3()               # gated HF weights (Meta SAM license)
r = model.predict("img.jpg", points=[640, 360], labels=[1])

model = LibreEdgeTAM()            # Apache-2.0, edge-friendly
r = model.predict("img.jpg", bboxes=[100, 100, 500, 500])

model = LibrePicoSAM3()           # 96 px ROI segmenter, ONNX-exportable
r = model.predict("crop.jpg", bboxes=[8, 8, 88, 88])`}</CodeBlock>
          <CodeBlock language="python">{`from libreyolo import LibreSAM

model = LibreSAM("base")

# 一次点击
r = model.predict("img.jpg", points=[640, 360], labels=[1])
print(r.masks.data.shape)   # (1, H, W) 布尔掩码，原图分辨率
print(r.boxes.conf)         # SAM 预测的掩码质量，不是检测置信度

# 框提示
r = model.predict("img.jpg", bboxes=[100, 100, 500, 500])`}</CodeBlock>
          <SubHeading>编码一次，多次提示</SubHeading>
          <CodeBlock language="python">{`model.set_image("img.jpg")                        # 重量级编码器只运行一次
a = model.predict(points=[500, 375], labels=[1])  # 廉价：只跑解码器
b = model.predict(bboxes=[100, 100, 200, 200])    # 廉价：复用图像嵌入
model.reset_image()`}</CodeBlock>
          <SubHeading>提示的形状</SubHeading>
          <P>嵌套层级是有含义的，这是最容易出错的地方。标签 <InlineCode>1</InlineCode> 表示包含，<InlineCode>0</InlineCode> 表示排除。</P>
          <DocTable
            headers={['你传入', '含义']}
            rows={[
              ['points=[x, y]', '一个目标，一个点'],
              ['points=[[x, y], [x, y]]', '两个目标，各一个点'],
              ['points=[[[x, y], [x, y]]]', '一个目标，两个点'],
            ]}
          />
          <SubHeading>限制</SubHeading>
          <ul className="space-y-2 my-4">
            <FeatureItem><strong>整层仅支持图像。</strong>v1.4.0 没有视频分割，也没有跨帧记忆传播（SAM 2、SAM 3 和 EdgeTAM 亦然），<InlineCode>track()</InlineCode> 会报错，请逐帧调用 <InlineCode>predict()</InlineCode>。</FeatureItem>
            <FeatureItem><strong>不支持训练与验证</strong>，所有 SAM 系列均如此。导出除 PicoSAM3（仅 ONNX）外全部报错。</FeatureItem>
            <FeatureItem><strong>PicoSAM3 仅接受 <InlineCode>bboxes=</InlineCode> ROI 提示。</strong>点、文本、掩码或全图分割请使用 LibreSAM2 或 LibreSAM3。</FeatureItem>
            <FeatureItem>不支持掩码提示（<InlineCode>masks=</InlineCode>），请使用点或框。</FeatureItem>
            <FeatureItem>全程使用 fp32，即便在 CUDA 上也是如此。这是有意为之：半精度会在 SAM 的 1024px 工作尺度上把提示坐标舍入若干像素，悄悄挪动你点击的位置。</FeatureItem>
          </ul>

          <Divider />

          <SectionHeading id="open-vocabulary" icon={Search}>开放词表检测</SectionHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="experimental">仅 Python API</SupportBadge>
            <SupportBadge variant="experimental">OmDet-Turbo 与 OV-DEIM 为 v1.4.0 新增</SupportBadge>
          </div>
          <P>
            用文本给出类别名列表，就能得到真实的检测框。无需训练，也无需标注数据。改变列表，就改变它检测的东西。安装：<InlineCode>pip install &quot;libreyolo[openvocab]&quot;</InlineCode>。
          </P>
          <P>
            它与 <a href="/docs/librevlm" className="text-libre-600 dark:text-libre-400 hover:underline">LibreVLM</a> 不同，这一点很关键。这些是以文本为条件的<em>专用检测器</em>：检测头直接输出带真实模型分数的框；而 VLM 是生成文本，再由 LibreYOLO 解析成框。经验法则：<strong>要“给已命名类别画框”就用开放词表；要“描述或指令”就用 VLM</strong>。许可请按系列核对：Grounding DINO、OWLv2 和 OmDet-Turbo 的权重为 Apache-2.0，但 OV-DEIM 的权重为 CC BY-NC 4.0（非商用，已与上游作者确认）。
          </P>
          <DocTable
            headers={['传给 LibreOpenVocab()', '类', '主干', '默认 conf']}
            rows={[
              ['"grounding-dino"（默认，tiny）', 'LibreGroundingDINO', 'Swin-T + BERT', '0.25'],
              ['"grounding-dino-base"', 'LibreGroundingDINO', 'Swin-B + BERT', '0.25'],
              ['"owlv2"', 'LibreOWLv2', 'ViT-B/16', '0.1'],
              ['"owlv2-large"', 'LibreOWLv2', 'ViT-L/14', '0.1'],
              ['"omdet-turbo"（v1.4.0 新增）', 'LibreOMDetTurbo', 'Swin-T，基于 transformers', '0.25'],
              ['"ov-deim" / "-m" / "-l"（v1.4.0 新增）', 'LibreOVDEIM', 'DEIM，原生无 NMS 移植', '0.25'],
            ]}
          />
          <P>
            词表设置在<em>模型</em>上，通过 <InlineCode>set_classes()</InlineCode>，并且会在后续调用中保持。<InlineCode>predict()</InlineCode> 上<strong>没有</strong> <InlineCode>prompts=</InlineCode> 或 <InlineCode>text=</InlineCode> 参数。
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreOpenVocab

model = LibreOpenVocab("grounding-dino")
model.set_classes(["person", "dog", "skateboard"])   # 持久词表

result = model.predict("street.jpg", conf=0.25, text_threshold=0.25)
print(result.names)     # {0: 'person', 1: 'dog', 2: 'skateboard'}

result = model.predict("another.jpg")   # 词表仍然有效`}</CodeBlock>
          <P>
            <strong>小心同名陷阱。</strong><InlineCode>predict(classes=...)</InlineCode> <em>不是</em>文本接口，它是标准的整数类别 id 过滤器。文本词表只能通过 <InlineCode>set_classes()</InlineCode> 设置。
          </P>
          <SubHeading>实用提示</SubHeading>
          <ul className="space-y-2 my-4">
            <FeatureItem>简短的名词短语效果最好，&quot;remote control&quot; 优于 &quot;remote&quot;。无法明确映射回你类别名的短语会被丢弃，因此漏检有时是映射被丢弃，而不是检测器没看到。</FeatureItem>
            <FeatureItem>类别数量没有上限。Grounding DINO 会自动把长词表切分成多个块，每块跑一次前向，因此<strong>开销随词表增大而增长</strong>，这是你能控制的主要延迟因素。</FeatureItem>
            <FeatureItem><InlineCode>text_threshold</InlineCode> 仅适用于 Grounding DINO，传给其他系列会报错。</FeatureItem>
            <FeatureItem>各系列的打分尺度不同，请分别调 <InlineCode>conf</InlineCode>，不要复用同一个数值。</FeatureItem>
            <FeatureItem><strong>OV-DEIM</strong> 是值得关注的速度选项：原生的无 NMS 移植（不是 transformers 流水线），提供三个尺寸。它的文本特征按词表缓存，v1.4.0 修复了该缓存在设备切换时的崩溃。注意权重为非商用许可。</FeatureItem>
            <FeatureItem><strong>OmDet-Turbo</strong> 基于 transformers，并会遵循 <InlineCode>iou=</InlineCode>（v1.4.0 之前会被忽略）。</FeatureItem>
            <FeatureItem>它比 LibreYOLO 检测器慢得多。务实的用法：先用开放词表探索或自动标注，再训练一个快速检测器。</FeatureItem>
          </ul>
          <SubHeading>限制</SubHeading>
          <ul className="space-y-2 my-4">
            <FeatureItem><strong>没有 CLI。</strong><InlineCode>libreyolo predict model=grounding-dino</InlineCode> 无效，本层只能从 Python 使用。</FeatureItem>
            <FeatureItem><strong>不支持训练、验证、导出与跟踪</strong>，四者都会报错。</FeatureItem>
            <FeatureItem><InlineCode>imgsz</InlineCode> 与 <InlineCode>augment=True</InlineCode> 会被拒绝；<InlineCode>iou</InlineCode> 会被接受但忽略。</FeatureItem>
            <FeatureItem>批处理不会带来加速，且全程 fp32。</FeatureItem>
          </ul>

          <Divider />

          <SectionHeading id="pose" icon={PersonStanding}>姿态估计</SectionHeading>
          <P>
            姿态（关键点）估计可在 <InlineCode>YOLO-NAS (-pose)</InlineCode>、{' '}
            <InlineCode>EdgeCrafter (-pose)</InlineCode> 以及{' '}
            <InlineCode>RF-DETR (-pose)</InlineCode> 预览上运行。已发布的检查点都是单类别
            （&quot;person&quot;），具有 17 个 COCO 关键点。v1.4.0 新增：YOLO-NAS 姿态支持<strong>多类别关键点训练</strong>，
            并且多类别检查点会以真实的类别数加载、返回真实的类别 id（此前会被强制视为单类别 person）。
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
            <SupportBadge variant="experimental">预览</SupportBadge>{' '}
            RF-DETR 姿态（移植自 GroupPose）在 v1.4.0 中仍是研究预览。
          </P>
          <CodeBlock language="python">{`# RF-DETR pose preview
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
            YOLO-NAS 支持姿态训练（v1.4.0 起包括多类别数据集）；EdgeCrafter 姿态目前仅推理。RF-DETR 姿态为预览。YOLO9 仅检测，不提供姿态检查点。多 GPU DDP 下的姿态验证已在 v1.4.0 修复（各 rank 互相覆盖文件与集合通信死锁）。
          </P>

          <Divider />

          {/* ────────────── GAZE ESTIMATION ────────────── */}
          <SectionHeading id="gaze" icon={Eye}>视线估计</SectionHeading>
          <P>
            视线方向估计由 <InlineCode>LibreL2CS</InlineCode> 系列提供，它是 L2CS-Net 的移植，具有 ResNet 主干和两个角度分箱分类头。这是一个两阶段模型：上游的人脸检测器定位人脸，然后视线头以弧度预测每张人脸的 pitch 和 yaw。它在 v1.4.0 中仅推理且为实验性。（v1.4.0 还通过加入 YuNet 检测器修复了 OpenCV 5 上的人脸检测。）
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

          {/* ────────────── CLASSIFICATION ────────────── */}
          <SectionHeading id="classification" icon={Tags}>分类</SectionHeading>
          <P>
            整图分类覆盖两条监督路径和一条零样本路径。<InlineCode>LibreMobileNetV4</InlineCode> 是生产级分类器（Apache-2.0 ImageNet-1k 权重，可导出为 ONNX)，<InlineCode>LibreConvNeXt</InlineCode>、<InlineCode>LibreEfficientNetV2</InlineCode> 和 <InlineCode>LibreResNet</InlineCode> 提供同一 API 下的替代选择。<InlineCode>LibreDINOv2</InlineCode> 配合 <InlineCode>task=classify</InlineCode> 是用于迁移学习的 DINOv2 主干加线性探针。v1.4.0 没有为它公开发布 <InlineCode>-cls</InlineCode> 检查点，因此需要构建并训练全新分类头；训练后的分类器可导出为 ONNX。零样本分类（无需训练，标签即文本）请用 CLIP 或 v1.4.0 新增的 SigLIP2。分类训练在 v1.4.0 中获得了自己的<a href="#augmentation" className="text-libre-600 dark:text-libre-400 hover:underline">增强参数包</a>：<InlineCode>auto_augment</InlineCode>、<InlineCode>erasing</InlineCode>、<InlineCode>mixup</InlineCode> 和 <InlineCode>cutmix</InlineCode>。
          </P>

          <DocTable
            headers={['系列', '检查点', '输入', '权重', '微调', 'ONNX 导出']}
            rows={[
              ['LibreMobileNetV4', 'LibreMobileNetV4{s,m,l}-cls.pt', '224 / 224 / 256', 'Apache-2.0 ImageNet-1k（生产级）', '交叉熵', '支持'],
              ['LibreDINOv2（classify）', 'v1.4.0 未公开发布', '224', '全新线性头', '线性探针', '支持'],
            ]}
          />

          <SubHeading>LibreMobileNetV4（生产级分类器）</SubHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="validated">Apache-2.0 ImageNet-1k 权重</SupportBadge>
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
            <SupportBadge variant="experimental">无公开 v1.4.0 检查点</SupportBadge>
            <SupportBadge variant="validated">支持 ONNX 导出</SupportBadge>
          </div>
          <P>
            DINOv2-S 编码器加可训练线性头，在 224 运行。<InlineCode>n</InlineCode> / <InlineCode>s</InlineCode> / <InlineCode>m</InlineCode> / <InlineCode>l</InlineCode> 尺寸控制投影头宽度，四者共享同一个 DINOv2-S 编码器。LibreYOLO v1.4.0 没有公开托管 <InlineCode>LibreDINOv2*-cls.pt</InlineCode> 检查点。请构建全新分类器，并在自己的 ImageFolder 数据集上训练。
          </P>

          <P>
            用 <InlineCode>task=&quot;classify&quot;</InlineCode> 构建全新模型，训练新分类头，然后使用与 MobileNetV4 相同的 <InlineCode>Probs</InlineCode> 预测接口。
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreDINOv2

# Fresh DINOv2 backbone + random linear head, sized to the dataset
model = LibreDINOv2(size="s", task="classify", nb_classes=3)
model.train(data="path/to/imagefolder", epochs=5, lr=1e-4, batch=4)

# Validate the same way (top-1 / top-5)
metrics = model.val(data="path/to/imagefolder")
print(metrics["metrics/accuracy_top1"])

result = model("springer.jpg")
print(result.probs.top1, result.probs.top1conf)

model.export(format="onnx", imgsz=224)`}</CodeBlock>

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

          <SubHeading>零样本分类：SigLIP2 与 CLIP</SubHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="experimental">SigLIP2 为 v1.4.0 新增</SupportBadge>
            <SupportBadge variant="experimental">仅推理</SupportBadge>
          </div>
          <P>
            零样本分类器把图像与<em>运行时给定的文本标签</em>逐一打分：无需训练，也没有固定类别表。v1.4.0 在原有 CLIP 系列之外新增了{' '}
            <InlineCode>LibreSigLIP2</InlineCode>（尺寸 <InlineCode>b16</InlineCode> 与{' '}
            <InlineCode>so400m</InlineCode>，原生 torch 移植）。两者都通过工厂加载，并用{' '}
            <InlineCode>set_classes()</InlineCode> 设置词表；SigLIP2 的 SentencePiece 分词器需要{' '}
            <InlineCode>pip install &quot;libreyolo[siglip2]&quot;</InlineCode>。
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreSigLIP2b16-cls.pt")
model.set_classes(["a forklift", "an empty aisle", "a spill"])

result = model.predict("warehouse.jpg")
print(model.names[result.probs.top1], float(result.probs.top1conf))

# Independent per-label probabilities (sigmoid) instead of softmax
result = model.predict("warehouse.jpg", multi_label=True)`}</CodeBlock>
          <P>
            SigLIP2 的 sigmoid 训练目标让 <InlineCode>multi_label=True</InlineCode> 的分数本身就有意义（CLIP 式 softmax 分数则不然）：当多个标签可能同时为真时用它。两种零样本系列的 <InlineCode>train()</InlineCode> 都会报错。
          </P>

          <ul className="space-y-2 my-4">
            <FeatureItem>MobileNetV4 权重为生产级（Apache-2.0 ImageNet-1k，逐比特一致加载）。LibreDINOv2 分类没有公开托管的 v1.4.0 检查点，请训练全新分类头。</FeatureItem>
            <FeatureItem>自 v1.3.0 起不再有 LibreRFDETR 分类器。分类已迁移到专门的分类器系列；旧的 LibreRFDETR*-cls 检查点在加载时会被拒绝。</FeatureItem>
            <FeatureItem>ONNX 分类输出为原始 logits。请在非 Python 消费方中自行应用 softmax。</FeatureItem>
            <FeatureItem>预测单张图像返回一个 Results。直接读取 result.probs，或传入列表并对列表取索引：model([&quot;a.jpg&quot;])[0].probs。</FeatureItem>
            <FeatureItem>v1.4.0 新增：<InlineCode>square_resize</InlineCode> 与 <InlineCode>augment</InlineCode> 同时使用现在会直接报错而不是悄悄出错，并且分类器系列支持 spawn 路径的多 GPU 训练。</FeatureItem>
          </ul>

          <Divider />

          {/* ────────────── DEPTH ESTIMATION ────────────── */}
          <SectionHeading id="depth" icon={Mountain}>深度估计</SectionHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="experimental">ZipDepth 与 Depth Anything 3 为 v1.4.0 新增</SupportBadge>
            <SupportBadge variant="experimental">仅推理与验证</SupportBadge>
          </div>
          <P>
            单目深度估计预测一张密集的相对逆深度图：值越大越靠近相机，不隐含任何度量单位。v1.4.0 在同一 API 下提供三个深度系列，主要差别在体量、许可和目标硬件：
          </P>
          <DocTable
            headers={['系列', '尺寸（输入）', '许可', '检查点']}
            rows={[
              ['LibreDepthAnythingV2', 's / b / l / g（518）', 's 为 Apache-2.0；b / l / g 为 CC-BY-NC-4.0', <span key="c"><Checkpoints names={['LibreDepthAnythingV2s-depth.pt', 'LibreDepthAnythingV2b-depth.pt', 'LibreDepthAnythingV2l-depth.pt']} />；g 需从上游转换</span>],
              ['LibreDepthAnything3（新增）', 'l（504）', 'Apache-2.0', <Checkpoints key="c" names={['LibreDepthAnything3l-depth.pt']} />],
              ['LibreZipDepth（新增）', 'b / bnpu（384）', 'MIT', <Checkpoints key="c" names={['LibreZipDepthb-depth.pt', 'LibreZipDepthbnpu-depth.pt']} />],
            ]}
          />
          <P>
            <InlineCode>LibreDepthAnything3</InlineCode> 是与 V2 并列的独立系列（不是原地升级），只有一个 Apache-2.0 的 large 检查点：在意许可时的质量之选。<InlineCode>LibreZipDepth</InlineCode> 是效率之选：MIT 许可、384 px，其 <InlineCode>bnpu</InlineCode> 变体的解码器避开了对 NPU 不友好的算子，适合边缘加速器。
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

          <SubHeading>导出（v1.4.0 新增）</SubHeading>
          <P>
            v1.4.0 为 Depth Anything V2 和 ZipDepth 解锁了深度导出，采用<strong>固定分辨率、batch 1 的契约</strong>：导出的图固定一个输入尺寸，没有动态轴。Depth Anything 3 暂不支持导出。
          </P>
          <CodeBlock language="python">{`model = LibreYOLO("LibreZipDepthb-depth.pt")
model.export(format="onnx")   # fixed resolution, batch 1`}</CodeBlock>

          <SubHeading>不支持</SubHeading>
          <CodeBlock language="python">{`model.train(data="...")   # raises NotImplementedError - all depth families are inference + val only`}</CodeBlock>

          <ul className="space-y-2 my-4">
            <FeatureItem>Depth Anything V2 的许可是分开的：尺寸 s 为 Apache-2.0，可商用；b / l / g 为 CC-BY-NC-4.0（非商业）。商用请选择 V2 的尺寸 s、Depth Anything 3 或 ZipDepth。</FeatureItem>
            <FeatureItem>深度为相对逆深度，没有度量单位。如果你需要以米为单位，请自行标定。</FeatureItem>
            <FeatureItem>Depth Anything V2 的 imgsz 必须能被 14 整除（DINOv2 patch 网格）。批量预测被禁用，因为保持纵横比缩放会导致每张图像尺寸不一。</FeatureItem>
            <FeatureItem>v1.4.0 起深度模型支持视频输入（v1.3.1 中会崩溃）。</FeatureItem>
          </ul>

          <Divider />

          {/* ────────────── POINT LOCALIZATION ────────────── */}
          <SectionHeading id="restoration" icon={WandSparkles}>图像修复与超分</SectionHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="experimental">SwinIR 与 Real-ESRGAN 为 v1.4.0 新增</SupportBadge>
            <SupportBadge variant="experimental">NAFNet 可训练</SupportBadge>
          </div>
          <P>
            <InlineCode>restore</InlineCode> 任务接收退化的图像，返回更好的图像。与这里的多数任务不同，它没有要检测的东西：输出就是一张图像，通过 <InlineCode>result.restored</InlineCode> 获取。在 v1.4.0 中该任务覆盖两类工作：<strong>清理</strong>（去噪 / 去模糊，输出与输入同分辨率）和新增的<strong>超分辨率</strong>（每个方向放大 2 倍或 4 倍；<InlineCode>result.restore_scale</InlineCode> 会告诉你倍数）。
          </P>
          <DocTable
            headers={['系列', '工作', '尺寸', '输出倍数', '可训练？']}
            rows={[
              ['LibreNAFNet', '去噪 / 去模糊', 's / l', '1x', '是'],
              ['LibreSwinIR（新增）', '超分辨率', 's / m / l', '4x', '否'],
              ['LibreRealESRGAN（新增）', '超分辨率', 'x4 / x2 / x4t', '4x / 2x / 4x（快速）', '否'],
            ]}
          />
          <P>
            清理模型究竟修复什么（去噪还是去模糊）取决于<em>它训练所用的权重</em>，而不是模型尺寸。超分模型的倍数则固化在检查点里：预测时没有倍数参数。
          </P>

          <SubHeading>检查点</SubHeading>
          <P>
            NAFNet 发布了一个检查点：<Checkpoints names={['LibreNAFNetl-restore-sidd.pt']} />，一个基于 SIDD 训练的真实图像<strong>去噪</strong>模型，从上游 NAFNet 逐位精确转换而来，MIT 许可。<strong>去模糊</strong>没有已发布的检查点：请用 <InlineCode>weights/convert_nafnet_weights.py</InlineCode> 自行转换上游 GoPro 权重。注意 <InlineCode>LibreNAFNets-restore.pt</InlineCode> 与 <InlineCode>LibreNAFNetl-restore.pt</InlineCode> 这两个名字<strong>并未托管</strong>，请求它们会下载失败。
          </P>
          <P>
            超分辨率则全部托管：SwinIR{' '}
            <Checkpoints names={['LibreSwinIRs-restore.pt', 'LibreSwinIRm-restore.pt', 'LibreSwinIRl-restore.pt']} />
            （代码与权重均为 Apache-2.0）以及 Real-ESRGAN{' '}
            <Checkpoints names={['LibreRealESRGANx4-restore.pt', 'LibreRealESRGANx2-restore.pt', 'LibreRealESRGANx4t-restore.pt']} />。
            <InlineCode>x4t</InlineCode> 是紧凑的 SRVGG 变体：快得多，也明显更软。
          </P>

          <SubHeading>清理一张图像</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")   # SIDD 去噪
result = model("noisy.jpg")

img = result.restored           # RestoredImage
print(img.array.shape)          # (H, W, 3) uint8 RGB，原始分辨率
print(result.restore_scale)     # 清理模型为 1
img.save("clean.png")           # 无损保存`}</CodeBlock>
          <P>
            清理在图像的原生分辨率上进行：输入会先填充到 16 的倍数，之后再裁回原尺寸，因此输出与输入尺寸一致。
          </P>

          <SubHeading>放大一张图像（v1.4.0 新增）</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreSwinIRm-restore.pt")        # 4x super-resolution
result = model("small.jpg")
print(result.restore_scale)         # 4
result.restored.save("big.png")     # 4x height, 4x width

# Real-ESRGAN: seam-free tiled upscaling for large inputs
model = LibreYOLO("LibreRealESRGANx4-restore.pt")
result = model("photo.jpg", tile=512)   # process in 512px tiles, bounded VRAM`}</CodeBlock>

          <SubHeading>务必无损保存</SubHeading>
          <P>
            这一点最容易出错。<InlineCode>libreyolo predict --save</InlineCode> 默认写出 <strong>JPEG</strong>，会把压缩伪影重新引入你刚刚花模型清理干净的图像。请显式要求 PNG。
          </P>
          <CodeBlock language="bash">{`libreyolo predict model=LibreNAFNetl-restore-sidd.pt source=noisy.jpg \\
  save=true output-file-format=png`}</CodeBlock>

          <SubHeading>训练与验证（NAFNet）</SubHeading>
          <P>
            NAFNet 训练使用成对的退化 / 干净图像；SwinIR 与 Real-ESRGAN 仅支持推理与验证。验证报告 PSNR 与 SSIM。v1.4.0 新增：修复训练默认应用成对的垂直翻转和 90 度旋转（输入与目标一起变换），因此重新训练的结果与 v1.3.1 会略有差异；见<a href="#augmentation" className="text-libre-600 dark:text-libre-400 hover:underline">数据增强</a>。
          </P>
          <CodeBlock language="python">{`model.train(data="gopro.yaml", epochs=100)

metrics = model.val(data="gopro.yaml")
print(metrics["metrics/psnr"], metrics["metrics/ssim"])`}</CodeBlock>
          <ul className="space-y-2 my-4">
            <FeatureItem>训练时有两处显示上的怪癖：控制台会把 PSNR 打印在 <InlineCode>mAP50</InlineCode> 这一列标题下（标签错误，数值确实是 PSNR）；并且 PSNR/SSIM 计算时不做边界裁剪，因此不能直接与公开的 NAFNet 基准数字对比。</FeatureItem>
            <FeatureItem>导出：NAFNet 支持 ONNX（静态尺寸，<InlineCode>imgsz</InlineCode> 为 16 的倍数）与 TorchScript；SwinIR 实验性支持 ONNX，并支持 TorchScript；Real-ESRGAN 支持 ONNX、TorchScript、NCNN 和 TFLite。NAFNet 的 TFLite 与 CoreML 仍被拦截。</FeatureItem>
          </ul>

          <Divider />

          {/* ────────────── BACKGROUND REMOVAL (MATTE) ────────────── */}
          <SectionHeading id="background-removal" icon={Eraser}>背景移除</SectionHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="experimental">v1.4.0 新增</SupportBadge>
            <SupportBadge variant="experimental">仅推理与验证</SupportBadge>
          </div>
          <P>
            <InlineCode>matte</InlineCode> 任务为每个像素预测一个 <InlineCode>[0, 1]</InlineCode> 区间的 alpha 值：该像素属于前景主体的程度。与二值分割掩码不同，matte 能保留柔和的边缘（发丝、毛发、运动模糊），抠图效果的好坏正取决于此。v1.4.0 提供 <InlineCode>LibreBiRefNet</InlineCode>，BiRefNet 的 1024 px 移植，尺寸为 <InlineCode>t</InlineCode> 与 <InlineCode>l</InlineCode>。<Checkpoints names={['LibreBiRefNetl-matte.pt']} /> 权重已托管；<InlineCode>t</InlineCode>（lite）权重在许可确认前暂未重新托管，如需小模型请自行转换。
          </P>

          <SubHeading>抠出主体</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO
from PIL import Image

model = LibreYOLO("LibreBiRefNetl-matte.pt")
result = model("portrait.jpg")

matte = result.matte            # Matte payload
print(matte.data.shape)         # (H, W) float32 alpha in [0, 1], original canvas

# RGBA cutout: original pixels with the matte as the alpha channel
rgba = result.cutout()          # (H, W, 4) uint8
Image.fromarray(rgba).save("subject.png")   # transparent background

# Or composite yourself
alpha = matte.array[..., None]  # (H, W, 1)`}</CodeBlock>
          <P>
            <InlineCode>save=True</InlineCode> 会写出 matte 叠加图；对视频源则逐帧渲染（matte 视频叠加自 v1.4.0 起可用）。验证通过 <InlineCode>MatteValidator</InlineCode> 与真值 alpha 图对比。
          </P>

          <SubHeading>限制</SubHeading>
          <ul className="space-y-2 my-4">
            <FeatureItem><strong>仅推理与验证。</strong>v1.4.0 中 matte 训练会报错。导出支持实验性 ONNX 和固定 1024 输入的 TorchScript；NCNN 仍被拦截。</FeatureItem>
            <FeatureItem>使用 <InlineCode>matte</InlineCode> 任务字符串写出的检查点无法被 v1.3.1 加载。</FeatureItem>
            <FeatureItem>抠图请保存为 PNG 或 WebP。JPEG 没有 alpha 通道，存成 JPEG 会悄悄拍平透明度。</FeatureItem>
          </ul>

          <Divider />

          {/* ────────────── OCR ────────────── */}
          <SectionHeading id="ocr" icon={ScanText}>文字识别（OCR）</SectionHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="experimental">v1.4.0 新增</SupportBadge>
            <SupportBadge variant="experimental">仅推理与验证</SupportBadge>
          </div>
          <P>
            <InlineCode>ocr</InlineCode> 任务读取文字：检测阶段以四点多边形找出文本区域，识别阶段逐一转写。v1.4.0 提供 <InlineCode>LibrePPOCR</InlineCode>，PP-OCRv5 的 960 px 移植，尺寸为 <InlineCode>t</InlineCode> 与 <InlineCode>l</InlineCode>：<Checkpoints names={['LibrePPOCRt-ocr.pt', 'LibrePPOCRl-ocr.pt']} />。结果通过 <InlineCode>result.ocr</InlineCode> 获取，这是一个 <InlineCode>OCRRegions</InlineCode> 负载，把每个多边形与其文本和两个置信度（检测一个、识别一个）配对。
          </P>

          <SubHeading>从图像读取文字</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibrePPOCRt-ocr.pt")
result = model("receipt.jpg", save=True)

ocr = result.ocr                 # OCRRegions
print(ocr.polygons.shape)        # (N, 4, 2) quad corners in pixels
for text, conf in zip(ocr.texts, ocr.conf):
    print(f"{conf:.2f}  {text}")

ocr.det_conf                      # (N,) detector scores, separate from recognition
print(result.saved_path)          # saved OCR overlay`}</CodeBlock>

          <SubHeading>CLI 与验证</SubHeading>
          <P>
            <InlineCode>libreyolo predict --json</InlineCode> 会输出一个 <InlineCode>ocr</InlineCode> 数组（每个区域的多边形、文本与置信度），让 CLI 可以直接用于文档处理流水线。验证通过 <InlineCode>OCRValidator</InlineCode> 运行，它先用最优一对一匹配把预测与真值配对，再计分。
          </P>
          <CodeBlock language="bash">{`libreyolo predict model=LibrePPOCRt-ocr.pt source=receipt.jpg --json | jq .ocr`}</CodeBlock>

          <SubHeading>限制</SubHeading>
          <ul className="space-y-2 my-4">
            <FeatureItem><strong>仅推理与验证。</strong>v1.4.0 中 OCR 的训练与导出都会报错。</FeatureItem>
            <FeatureItem>使用 <InlineCode>ocr</InlineCode> 任务字符串写出的检查点无法被 v1.3.1 加载。</FeatureItem>
            <FeatureItem>对 OCR 模型而言 <InlineCode>result.boxes</InlineCode> 为 <InlineCode>None</InlineCode>：区域是 <InlineCode>result.ocr</InlineCode> 中的多边形，不是轴对齐的框。</FeatureItem>
          </ul>

          <Divider />

          <SectionHeading id="point-localization" icon={MapPin}>点定位</SectionHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="experimental">实验性</SupportBadge>
          </div>
          <P>
            <InlineCode>LibreFOMO</InlineCode> 是 FOMO 风格的点定位器（尺寸 <InlineCode>s</InlineCode> / <InlineCode>m</InlineCode> / <InlineCode>l</InlineCode>），用于质心式检测：每个检测不是框，而是一个图像坐标。预测以 <InlineCode>result.points</InlineCode> 形式返回。预训练的 LibreFOMO 权重不会自动下载，因此请传入本地检查点路径（或从头训练，这属于实验性，需要 <InlineCode>allow_experimental=True</InlineCode>）。v1.4.0 新增：FOMO 支持固定分辨率契约下的 ONNX 导出，训练好的点模型可以离开 Python 部署到边缘端。
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

          <SectionHeading id="annotation" icon={PenTool}>标注（LibreLabel）</SectionHeading>
          <P>
            <InlineCode>libreyolo label</InlineCode> 启动一个本地的、基于浏览器的标注工具。它把 LibreYOLO 原生格式的标签文件直接写到训练器已经会读取的位置，因此一个图像文件夹无需任何转换就能变成可训练的数据集，不需要云账号，也不需要数据库。服务端只用 Python 标准库，全程在你自己的机器上运行。
          </P>
          <CodeBlock language="bash">{`# 打开已有数据集
libreyolo label data=path/to/data.yaml

# 也可以直接给一个图像文件夹
libreyolo label data=path/to/images

# 从项目主页开始，在浏览器里创建项目
libreyolo label`}</CodeBlock>
          <DocTable
            headers={['选项', '默认值', '作用']}
            rows={[
              ['data', '（无）', '数据集 YAML 或文件夹。省略则打开项目主页。'],
              ['host', '127.0.0.1', '绑定的网卡地址。修改前请先看下面的共享说明。'],
              ['port', '8000', '绑定端口，被占用时自动向后顺延最多 19 个端口。'],
              ['device', 'auto', 'AI 辅助功能使用的设备。'],
              ['no_assist', 'false', '彻底关闭所有 AI 辅助。'],
              ['no_browser', 'false', '不自动打开浏览器。'],
              ['share', 'false', '绑定 0.0.0.0，让局域网内的同事一起标注。'],
            ]}
          />
          <SubHeading>可以标注什么</SubHeading>
          <P>
            检测框（<InlineCode>detect</InlineCode>）、多边形（<InlineCode>segment</InlineCode>）与旋转框（<InlineCode>obb</InlineCode>，带旋转手柄）。关键点、掩码与深度文件以<strong>只读</strong>方式打开，这样保存时绝不会悄悄丢掉它不理解的字段。分类标注尚不可用。
          </P>
          <SubHeading>AI 辅助，以及它绝不打破的一条规则</SubHeading>
          <P>
            LibreLabel 可以用你自己的检测器预标注、用 SAM 把一次点击变成掩码、审计已有标签中可能的错误、找出近似重复的图像，并检测训练/验证集泄漏。<strong>任何 AI 路径都不会写入标签文件。</strong>所有建议都只存在内存中，直到人工确认。AI 辅助也绝不会下载权重：如果检查点不在本地，它会拒绝并提示你，而不是在背后拉取数百 MB。
          </P>
          <SubHeading>导出</SubHeading>
          <P>
            在浏览器的导出对话框中可导出为 YOLO、COCO 或 VOC（也可一次导出多种），并提供可复现的训练/验证/测试划分。注意这是浏览器里的操作，<strong>没有对应的 CLI 参数</strong>。导入仅支持 YOLO，因此 COCO 与 VOC 只是导出格式，不是入口。
          </P>
          <SubHeading>共享，以及一个值得注意的陷阱</SubHeading>
          <P>
            这里<strong>完全没有任何身份认证</strong>，访问控制纯粹依赖网络位置，因此只在你信任的网络上共享。
          </P>
          <P>
            反直觉的地方在于：<InlineCode>share=true</InlineCode> 才是让同事加入的<em>安全</em>方式。它绑定通配地址，而管理员权限要求来自回环连接，因此你在本机保留管理员权限，同事只得到一个仅供标注的视图。反过来，绑定具体地址（<InlineCode>host=192.168.1.50</InlineCode>）会让你的机器与同事无法区分，从而把<strong>完整管理员权限交给局域网内的每一个客户端</strong>。请优先使用 <InlineCode>share=true</InlineCode>。
          </P>

          <Divider />

          <SectionHeading id="training" icon={GraduationCap}>训练</SectionHeading>
          <ValidationScopeCalloutZh />
          <P>
            经过充分测试的训练路径是单 GPU 的 YOLO9 检测、RF-DETR 检测和 RF-DETR 分割。其他模型系列的训练器和多 GPU 工作流可用，但为实验性。YOLO9 仅检测，因此没有 YOLO9 分割或姿态训练。v1.4.0 新增：YOLOv7 可训练（SimOTA 损失）、SegFormer 可微调，并且每个增强参数都在<a href="#augmentation" className="text-libre-600 dark:text-libre-400 hover:underline">数据增强</a>章节中按系列写明。
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
            训练完成后，模型实例会自动以最优权重重新加载，因此你可以立即调用 <InlineCode>model(...)</InlineCode>。<InlineCode>freeze</InlineCode>、<InlineCode>cache</InlineCode>、<InlineCode>pretrained</InlineCode> 和 <InlineCode>save_plots</InlineCode> 在所有基于训练器的系列中受支持。
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
            RF-DETR 有自己的训练签名（<InlineCode>batch_size</InlineCode>、<InlineCode>lr</InlineCode>、<InlineCode>output_dir</InlineCode>），但共用 LibreYOLO 的数据集加载器。为检测或分割传入 <InlineCode>data.yaml</InlineCode>，可采用 YOLO TXT 或原生 COCO JSON 布局，参见 <a href="#dataset-format" className="text-libre-600 dark:text-libre-400 hover:underline">数据集格式</a>。
          </P>

          <SubHeading>LoRA 微调</SubHeading>
          <P>
            <SupportBadge variant="experimental">实验性</SupportBadge>{' '}
            <InlineCode>lora=True</InlineCode> 注入 LoRA 适配器进行低显存微调：只有适配器
            （加上必须保持可训练的部分，如检测头）接收梯度。它需要可选的{' '}
            <InlineCode>peft</InlineCode> 依赖（<InlineCode>pip install &quot;libreyolo[lora]&quot;</InlineCode>）。
            v1.4.0 把 LoRA 扩展到 RF-DETR 之外：受支持的系列为{' '}
            <strong className="text-surface-800 dark:text-white">RF-DETR、D-FINE、DEIM、DEIMv2、RT-DETR v1 / v2 / v4、EC 和 ConvNeXt</strong>
            （D-FINE 与 EC 仅限 detect）。不支持的系列仍会抛出明确错误，而不是忽略该标志。
            <InlineCode>export()</InlineCode> 时适配器会合并进稠密权重，部署产物在运行时不需要 peft。
          </P>
          <CodeBlock language="python">{`model = LibreYOLO("LibreRFDETRs.pt")
results = model.train(data="data.yaml", epochs=50, lora=True)

# Works the same on the newly supported families
model = LibreYOLO("LibreDEIMs.pt")
results = model.train(data="data.yaml", epochs=50, lora=True)`}</CodeBlock>

          <SubHeading>实验日志记录器</SubHeading>
          <P>
            传入 <InlineCode>loggers=</InlineCode> 可将指标流式发送到 TensorBoard、
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
            其他系列也有训练器钩子，但在 v1.4.0 中它们不是推荐路径。新工作请保持使用 YOLO9 检测或 RF-DETR 检测/分割；仅出于兼容性、基准复现或针对性研究使用实验性训练器。PicoDet、RTMDet 和 EC 训练需要显式的 <InlineCode>allow_experimental=True</InlineCode> 确认。注意 v1.4.0 修正了有害的微调默认值：PicoDet（<InlineCode>lr0</InlineCode> 从 0.1 改为 0.01）和 DEIM（<InlineCode>lr0</InlineCode> 从 4e-4 改为 1e-4）；如需复现上游 COCO 配方，请显式传入旧值。
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

          <SubHeading>分布式训练（DDP，v1.4.0 大修）</SubHeading>
          <P>
            多 GPU 训练在 v1.4.0 经历了正确性大修。它仍在充分测试范围之外，但失败模式从“悄悄跑错”变成了“大声报错”：
          </P>
          <ul className="space-y-2 my-4">
            <FeatureItem><strong>所有系列的分片都已修正。</strong>DEIM、D-FINE 和 YOLO-NAS 姿态此前每个 rank 都在用<em>完整</em>批大小训练<em>完整</em>数据集（多买 GPU 毫无收益）；现在它们正确分片，损失归一化因子做全局 all-reduce，梯度与单 GPU 训练一致。</FeatureItem>
            <FeatureItem><strong>SyncBatchNorm 默认开启</strong>：BatchNorm 密集的系列（YOLO9、YOLOX、YOLOv7、YOLO-NAS、PicoDet、RTMDet、FOMO）在 DDP 下默认同步 BN 统计，修复了各 rank 独立统计导致的真实收敛退化。</FeatureItem>
            <FeatureItem><strong>启动时硬性报错</strong>：全局批大小不能被卡数整除、AutoBatch 之后每 rank 批大小小于 1、自定义加载器不分片，这些情况现在都会在启动时报错。</FeatureItem>
            <FeatureItem><strong>spawn 路径的多 GPU</strong> 覆盖到分类器系列（ResNet、ConvNeXt、EfficientNetV2、MobileNetV4）和 NAFNet：传入 <InlineCode>device=&quot;0,1&quot;</InlineCode> 即自动派生工作进程，无需 torchrun。</FeatureItem>
          </ul>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")
# Two GPUs: the global batch is split per rank (16 -> 8 + 8)
model.train(data="coco128.yaml", epochs=300, batch=16, device="0,1")`}</CodeBlock>
          <P>
            <InlineCode>device=&quot;0,1&quot;</InlineCode>（或列表 <InlineCode>[0, 1]</InlineCode>）即选择多 GPU。在{' '}
            <InlineCode>torchrun</InlineCode> 之下由启动器管理进程组；在它之外，LibreYOLO 会自行派生 DDP 工作进程。两条路径运行同一个训练器。
          </P>
          <CodeBlock language="bash">{`# Explicit torchrun launch also works
torchrun --nproc_per_node=2 train_yolo9.py`}</CodeBlock>

          <Divider />

          {/* ────────────── DATA AUGMENTATION ────────────── */}
          <SectionHeading id="augmentation" icon={Dices}>数据增强</SectionHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="experimental">v1.4.0 完成文档化并由规范测试保障</SupportBadge>
          </div>
          <P>
            训练时的数据增强直接在 <InlineCode>model.train()</InlineCode> 上配置：mosaic、MixUp、HSV
            抖动、翻转，以及仿射变换（旋转、平移、缩放、错切、透视）都是普通的关键字参数。同样的参数在{' '}
            <InlineCode>libreyolo train</InlineCode> 上以 <InlineCode>key=value</InlineCode> 形式可用，其中{' '}
            <InlineCode>mosaic=</InlineCode> 和 <InlineCode>mixup=</InlineCode> 是{' '}
            <InlineCode>mosaic_prob</InlineCode> 与 <InlineCode>mixup_prob</InlineCode> 的 CLI 简写。
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")
model.train(
    data="coco128.yaml",
    epochs=100,

    # Augmentation knobs (defaults shown in the table below)
    mosaic_prob=1.0,       # 4-image mosaic
    mixup_prob=0.5,        # blend in a second sample
    hsv_prob=1.0,          # HSV color jitter
    flip_prob=0.5,         # horizontal flip
    flipud=0.1,            # vertical flip (good for aerial imagery)
    degrees=10.0,          # random rotation range for the affine warp
    translate=0.1,         # random translation fraction
    shear=2.0,             # random shear, degrees
    perspective=0.0005,    # projective warp magnitude (0 = pure affine)
    no_aug_epochs=15,      # final epochs with strong augmentation off
)`}</CodeBlock>
          <CodeBlock language="bash">{`libreyolo train model=yolo9-c data=coco128.yaml epochs=100 \\
  mosaic=1.0 mixup=0.5 hsv_prob=1.0 flip_prob=0.5 degrees=10 translate=0.1`}</CodeBlock>

          <SubHeading>增强参数一览</SubHeading>
          <P>
            以下是基础 <InlineCode>TrainConfig</InlineCode> 字段。表中是基类默认值：<strong>各系列会用调优过的配方覆盖它们</strong>（例如
            YOLO9 默认 <InlineCode>degrees=0</InlineCode>、<InlineCode>shear=0</InlineCode> 且关闭
            mixup，而 YOLOX 三者全开）。用 <InlineCode>libreyolo cfg</InlineCode> 可打印你的模型实际解析出的默认值。
          </P>
          <DocTable
            headers={['参数', '基类默认值', '作用']}
            rows={[
              [<InlineCode key="k">mosaic_prob</InlineCode>, '1.0', '把 4 张图拼成一张 mosaic 样本的概率。'],
              [<InlineCode key="k">mixup_prob</InlineCode>, '1.0', '混入第二个样本（MixUp）的概率。'],
              [<InlineCode key="k">hsv_prob</InlineCode>, '1.0', 'HSV 颜色抖动的概率。'],
              [<InlineCode key="k">flip_prob</InlineCode>, '0.5', '水平翻转概率。'],
              [<InlineCode key="k">flipud</InlineCode>, '0.0', '垂直翻转概率。默认关闭；适合没有固定上下方向的场景（航拍、显微镜）。'],
              [<InlineCode key="k">degrees</InlineCode>, '10.0', '仿射变换的随机旋转范围（度）。'],
              [<InlineCode key="k">translate</InlineCode>, '0.1', '仿射变换的随机平移比例。'],
              [<InlineCode key="k">mosaic_scale</InlineCode>, '(0.1, 2.0)', '仿射变换的随机缩放范围。'],
              [<InlineCode key="k">shear</InlineCode>, '2.0', '仿射变换的随机错切范围（度）。'],
              [<InlineCode key="k">perspective</InlineCode>, '0.0', '透视变换幅度，在 [-p, +p] 内采样；0.0005 左右是典型值。0 表示纯仿射。'],
              [<InlineCode key="k">mixup_scale</InlineCode>, '(0.5, 1.5)', '施加在 MixUp 伙伴图像上的缩放抖动范围。'],
              [<InlineCode key="k">no_aug_epochs</InlineCode>, '15', '最后若干个 epoch 关闭强增强，让模型在干净图像上收敛。'],
            ]}
          />

          <SubHeading>哪些系列使用哪些参数</SubHeading>
          <P>
            并非每个系列都运行所有增强：每个系列都沿用其配方自带的流水线。v1.4.0 用一份声明式规范
            （<InlineCode>libreyolo/data/augment/spec.py</InlineCode>）把这一点显式化，并用测试钉在真实流水线上。每个参数在每个系列下有三种状态之一，并且 <strong>CLI 现在会在你显式设置了所选系列忽略的参数时发出警告</strong>：手误或错误假设不会再无声失败。
          </P>
          <DocTable
            headers={['状态', '含义']}
            rows={[
              [<strong key="s" className="text-emerald-600 dark:text-emerald-400">used</strong>, '参数进入训练流水线并改变样本。'],
              [<strong key="s" className="text-amber-600 dark:text-amber-400">gated by mosaic</strong>, '参数只作用于走了 mosaic 分支的样本；mosaic_prob=0 时它永远不会生效。'],
              [<strong key="s" className="text-surface-500">ignored</strong>, '参数不会进入该系列的流水线；设置它没有任何效果（CLI 会警告）。'],
            ]}
          />
          <DocTable
            headers={['流水线', '系列', '实际发生的事']}
            rows={[
              ['YOLOX 式 mosaic', 'YOLO9、YOLO9-E2E、YOLO9-P2、YOLOX、YOLOv7、RTMDet、PicoDet、RT-DETR、RT-DETRv2、FOMO', 'HSV 抖动与翻转逐样本运行。仿射变换（degrees / translate / mosaic_scale / shear / perspective）与 MixUp 只在 mosaic 画布上运行，因此受 mosaic_prob 门控。RTMDet、PicoDet、RT-DETR、RT-DETRv2 和 FOMO 没有垂直翻转；FOMO 还去掉了 perspective。'],
              ['YOLO-NAS', 'YOLO-NAS', '没有 mosaic（mosaic_prob 被忽略），取而代之的是常开的逐样本仿射，因此 degrees / translate / shear / perspective 直接生效，且 MixUp 独立于 mosaic。'],
              ['DETR 式直通', 'D-FINE、DEIM、DEIMv2、RT-DETRv4、EC', '只有 flip_prob 和 no_aug_epochs 可调。颜色抖动、zoom-out 和 IoU 裁剪是固定的配方常量，没有 mosaic、MixUp 或仿射。例外：EC 姿态通过关键点感知仿射使用 hsv_prob、degrees 和 translate。'],
              ['RF-DETR 原生', 'RF-DETR', '原生配方中的翻转、尺度抖动与随机裁剪；flip_prob 与 no_aug_epochs 可配置，HSV 不可。'],
              ['分类', 'ResNet、ConvNeXt、MobileNetV4、EfficientNetV2、DINOv2（classify）', '检测参数一律不生效（水平翻转固定为 0.5）。请使用下面的分类增强包。'],
              ['语义', 'SegFormer（及共享的语义流水线）', '尺度抖动与 HSV 来自系列属性而非 TrainConfig 参数；翻转固定为 0.5。v1.4.0 起 HSV 抖动默认开启。'],
              ['修复', 'NAFNet', '输入 / 目标成对的裁剪、翻转与 rot90，概率固定（垂直翻转 + rot90 为 v1.4.0 新增）。TrainConfig 参数被忽略。'],
            ]}
          />

          <SubHeading>mosaic 门控的含义</SubHeading>
          <P>
            在 YOLOX 式流水线中，MixUp 与仿射变换搭乘在 mosaic 分支<em>内部</em>：样本先（以{' '}
            <InlineCode>mosaic_prob</InlineCode> 的概率）拼成 4 图 mosaic，然后才对 mosaic
            画布做仿射并可选地与另一个样本混合。两个实际后果：
          </P>
          <ul className="space-y-2 my-4">
            <FeatureItem>设 <InlineCode>mosaic_prob=0</InlineCode> 会连带关闭这些系列的 MixUp 与仿射，无论其他参数写了什么。v1.4.0 会在 <InlineCode>mixup_prob &gt; 0</InlineCode> 但 <InlineCode>mosaic_prob=0</InlineCode>（mixup 永远无法生效）时于训练开始时发出警告。</FeatureItem>
            <FeatureItem>想减弱增强又保留一些几何变换时，调低 <InlineCode>mosaic_prob</InlineCode> 而不是归零，或者用 <InlineCode>degrees=0 translate=0 shear=0</InlineCode> 显式关掉几何部分。</FeatureItem>
          </ul>
          <CodeBlock language="python">{`# Minimal augmentation: flips only
model.train(
    data="data.yaml",
    mosaic_prob=0.0,   # also disables mixup + affine in mosaic-gated families
    mixup_prob=0.0,
    hsv_prob=0.0,
    flip_prob=0.5,
    no_aug_epochs=0,
)`}</CodeBlock>

          <SubHeading>分类增强包（v1.4.0 新增）</SubHeading>
          <P>
            分类的 ImageFolder 流水线有自己的四个参数，默认全部关闭。每个 batch 最多运行 MixUp / CutMix
            之一：先以 <InlineCode>mixup</InlineCode> 的概率尝试 MixUp，否则以{' '}
            <InlineCode>cutmix</InlineCode> 的概率尝试 CutMix，因此两者之和不应超过 1。
          </P>
          <DocTable
            headers={['参数', '默认值', '作用']}
            rows={[
              [<InlineCode key="k">auto_augment</InlineCode>, 'None', '策略名："randaugment"、"autoaugment" 或 "augmix"。'],
              [<InlineCode key="k">erasing</InlineCode>, '0.0', 'RandomErasing 概率。'],
              [<InlineCode key="k">mixup</InlineCode>, '0.0', 'batch 级 MixUp 概率（软标签）。仅限 Python API：CLI 上的 --mixup 是检测的 mixup_prob 别名。'],
              [<InlineCode key="k">cutmix</InlineCode>, '0.0', 'batch 级 CutMix 概率（软标签）。'],
            ]}
          />
          <CodeBlock language="python">{`from libreyolo import LibreMobileNetV4

model = LibreMobileNetV4(size="s")
model.train(
    data="imagenette160",
    epochs=20,
    auto_augment="randaugment",
    erasing=0.25,
    mixup=0.2,
    cutmix=0.2,
)`}</CodeBlock>

          <SubHeading>任务专属参数</SubHeading>
          <P>
            少数参数位于各系列的 <InlineCode>TrainConfig</InlineCode> 子类而非基础配置上，可从 Python
            或训练 YAML 使用（CLI 不暴露它们）：
          </P>
          <DocTable
            headers={['参数', '系列', '作用']}
            rows={[
              [<InlineCode key="k">copy_paste</InlineCode>, 'RF-DETR（segment）、YOLO9 系', '分割训练的 copy-paste 实例增强概率：把实例剪出并粘贴到样本中。'],
              [<InlineCode key="k">copy_paste_mode</InlineCode>, '同上', '粘贴实例的来源："flip" 镜像同一样本；"mixup" 抽取第二个样本（RF-DETR 仅支持 "flip"）。'],
              [<InlineCode key="k">rot90</InlineCode>, 'YOLO9 系（OBB 路径）', '旋转框训练的随机 90 度旋转概率；对轴对齐检测忽略。'],
              [<InlineCode key="k">crop_resize_prob</InlineCode>, 'RF-DETR、D-FINE（segment）、EC（segment）', '原生流水线中的随机裁剪缩放概率。'],
              [<InlineCode key="k">brightness_contrast_prob</InlineCode>, 'YOLO-NAS（pose）、EC（pose）', '关键点训练的亮度 / 对比度抖动概率。'],
              [<InlineCode key="k">affine_prob</InlineCode>, 'YOLO-NAS（pose）、EC（pose）', '关键点感知仿射的概率。'],
            ]}
          />

          <SubHeading>训练增强与测试时增强</SubHeading>
          <P>
            以上一切都发生在 <InlineCode>train()</InlineCode> 期间。测试时增强（TTA）是另一回事：{' '}
            <InlineCode>predict(augment=True)</InlineCode> / <InlineCode>val(augment=True)</InlineCode>{' '}
            在推理时额外运行增强过的前向并合并输出。v1.4.0 中 TTA 覆盖已实现的检测系列、四个语义分割系列
            （PIDNet、SegFormer、EoMT、DINOv2）以及 EoMT 全景分割。
          </P>

          <Divider />

          {/* ────────────── DISTILLATION ────────────── */}
          <SectionHeading id="distillation" icon={FlaskConical}>知识蒸馏</SectionHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="experimental">学生仅限 YOLO9 与 YOLOX</SupportBadge>
            <SupportBadge variant="experimental">DINOv2 教师为 v1.4.0 新增</SupportBadge>
          </div>
          <P>
            知识蒸馏让小模型（学生）在自身标签之外，额外向一个更大的冻结教师模型的中间特征学习。你会得到一个以学生速度运行、却找回部分教师精度的模型。把 <InlineCode>distill_model</InlineCode> 指向一个教师检查点即可开启蒸馏。
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

student = LibreYOLO("LibreYOLO9t.pt")     # 小学生模型

student.train(
    data="coco.yaml",
    epochs=100,
    distill_model="LibreYOLO9c.pt",   # 冻结的教师：这一项开启蒸馏
    distill_loss_type="mgd",          # "mgd"（默认）或 "cwd"
    dis=2e-5,                         # 全局权重；省略则使用各损失的默认值
)`}</CodeBlock>
          <CodeBlock language="bash">{`libreyolo train model=LibreYOLO9t.pt data=coco.yaml epochs=100 \\
  distill-model=LibreYOLO9c.pt distill-loss-type=mgd dis=2e-5`}</CodeBlock>
          <DocTable
            headers={['参数', '默认值', '含义']}
            rows={[
              ['distill_model', 'None', '教师检查点路径。设置即开启蒸馏。'],
              ['dis', 'None', '蒸馏损失的全局权重。默认回退为 MGD 的 2e-5、CWD 的 1.0。'],
              ['distill_loss_type', '"mgd"', '特征损失："mgd" 或 "cwd"。'],
              ['distill_mask_ratio', '0.65', '仅 MGD：被遮蔽的空间位置比例。仅 Python API。'],
              ['distill_tau', '1.0', '仅 CWD：softmax 温度。仅 Python API。'],
            ]}
          />
          <P>
            注意参数名很短：权重参数叫 <InlineCode>dis</InlineCode>，而不是 <InlineCode>distill_loss_weight</InlineCode>。
          </P>

          <SubHeading>基础模型蒸馏教师：DINOv2（v1.4.0 新增）</SubHeading>
          <P>
            除了检查点教师，v1.4.0 新增 <InlineCode>distill_model=&quot;dinov2&quot;</InlineCode>：学生主干的特征以{' '}
            <InlineCode>feat_mse</InlineCode> 损失向一个冻结的 DINOv2 基础编码器回归。不需要你自己的教师检查点，这是给
            YOLO9 主干的训练加上蒸馏信号最省事的方式。
          </P>
          <CodeBlock language="python">{`student = LibreYOLO("LibreYOLO9s.pt")
student.train(
    data="coco128.yaml",
    epochs=100,
    distill_model="dinov2",       # frozen foundation teacher
    distill_loss_type="feat_mse", # feature regression against DINOv2
    distill_normalize=True,       # normalize features before the loss
)`}</CodeBlock>
          <P>
            教师内部以 DINOv2 兼容的分辨率运行；v1.4.0 修复了非 14 倍数尺寸下的边缘裁剪 bug，奇数输入尺寸也能正确蒸馏。
          </P>

          <SubHeading>MGD 还是 CWD</SubHeading>
          <P>
            <strong>MGD</strong>（掩码生成式蒸馏，默认）随机遮蔽学生特征的空间位置，并要求它重建教师的特征。由于它回归的是原始特征幅值，默认权重很小：<InlineCode>2e-5</InlineCode>。
          </P>
          <P>
            <strong>CWD</strong>（逐通道蒸馏）把每个通道变成一个空间分布，再用 KL 散度对齐。逐通道归一化使它对尺度不敏感，因此在师生特征幅值差距很大时表现更稳。默认权重为 <InlineCode>1.0</InlineCode>。
          </P>
          <SubHeading>限制</SubHeading>
          <ul className="space-y-2 my-4">
            <FeatureItem><strong>学生只能是 YOLO9 与 YOLOX 系列。</strong>其他系列会在 setup 阶段报错，因为蒸馏需要只有这两个系列声明的特征抽取点。DINOv2 基础教师面向 YOLO9 主干。</FeatureItem>
            <FeatureItem><strong>师生的 stride 必须完全一致。</strong>两个支持的系列都使用 8/16/32，因此实际做法是在同一系列内跨尺寸蒸馏。通道宽度可以自由不同，会由 1x1 适配层桥接。</FeatureItem>
            <FeatureItem>多卡、混合精度与梯度累积都可与蒸馏同时使用。</FeatureItem>
            <FeatureItem>支持断点续训并恢复适配层状态，但教师不会存入检查点：续训时请重新传入 <InlineCode>distill_model</InlineCode>。</FeatureItem>
          </ul>

          <Divider />

          <SectionHeading id="monitoring" icon={Gauge}>训练监控</SectionHeading>
          <P>
            每一次训练、每一个模型系列，都会把机器可读的进度文件写入运行目录，无需任何开关。<InlineCode>libreyolo monitor</InlineCode> 把它们展示为实时看板；由于它只读取文件，对正在运行、已完成、甚至已崩溃的任务同样有效。
          </P>
          <CodeBlock language="bash">{`libreyolo monitor                     # 在 http://127.0.0.1:8420 监视 runs/
libreyolo monitor runs/train/exp      # 直接打开某一次运行
libreyolo monitor --port 9000 --no-browser`}</CodeBlock>
          <DocTable
            headers={['文件', '内容']}
            rows={[
              ['status.json', '本次运行的当前状态，每个 epoch 原子性重写。'],
              ['metrics.jsonl', '仅追加，每个 epoch 一个 JSON 对象，完整指标历史。'],
              ['train.log', '运行日志。'],
            ]}
          />
          <P>
            <InlineCode>status.json</InlineCode> 始终包含 <InlineCode>state</InlineCode>（<InlineCode>running</InlineCode>、<InlineCode>completed</InlineCode> 或 <InlineCode>failed</InlineCode>）、<InlineCode>pid</InlineCode>、<InlineCode>progress</InlineCode>、<InlineCode>eta_seconds</InlineCode>，以及当前与最佳指标。如果任务崩溃，它会记录 <InlineCode>state: &quot;failed&quot;</InlineCode> 以及包含异常类型与信息的 <InlineCode>error</InlineCode> 对象，因此崩溃会留在文件里，而不是只出现在你已经关掉的终端里。
          </P>
          <CodeBlock language="python">{`import json, time

def wait_for_run(run_dir):
    while True:
        status = json.load(open(f"{run_dir}/status.json"))
        if status["state"] != "running":
            return status
        print(f'{status["progress"]:.0%}  eta {status["eta_seconds"]:.0f}s')
        time.sleep(30)

final = wait_for_run("runs/train/exp")
if final["state"] == "failed":
    print(final["error"]["type"], final["error"]["message"])`}</CodeBlock>

          <Divider />

          <SectionHeading id="profiling" icon={Timer}>性能分析</SectionHeading>
          <P>
            <InlineCode>libreyolo profile</InlineCode> 测量时间究竟花在哪里，训练与推理都支持。它刻意只做测量：绝不修改你的配置，也不替你调参。它只告诉你什么慢，决定权留给你。
          </P>
          <P>
            v1.4.0 行为变化：<InlineCode>model.train(profile=True)</InlineCode> 现在会在分析窗口结束后<strong>继续训练</strong>而不是停止（也不再破坏断点续训状态）。传入 <InlineCode>profile_then_stop=True</InlineCode> 可恢复旧的“采集完即退出”行为。
          </P>
          <CodeBlock language="bash">{`# 训练：GPU 真的在忙，还是被数据加载卡住了？
libreyolo profile run coco128 --weights LibreYOLO9t.pt --batch 16 --repeat 3

# 推理：延迟分位数，以及时间花在哪一段
libreyolo profile infer bus.jpg --weights LibreYOLO9t.pt --runs 200`}</CodeBlock>
          <P>
            <InlineCode>profile infer</InlineCode> 报告 p50、p90、p99 延迟、吞吐，以及在预处理 / 前向 / 后处理（NMS）之间的耗时拆分。这个拆分往往才是重点：一个看起来很慢的模型，时间常常花在 NMS 或预处理上，而不是网络本身。
          </P>
          <DocTable
            headers={['子命令', '回答什么问题']}
            rows={[
              ['summary', '总体诊断：利用率、瓶颈在哪、kernel 构成。'],
              ['phases', '时间去向：前向、反向、数据加载、优化器。'],
              ['kernels', '哪些 GPU kernel 占主导。'],
              ['ops', '框架视角：哪些算子最耗 CPU 时间。'],
              ['compare', '对比两份 profile，看改动前后。'],
              ['what-if', '在动手改代码之前，先估算收益。'],
            ]}
          />
          <P>
            两点提示：每个子命令都支持 <InlineCode>--json</InlineCode>，便于放进自动化的优化循环；而 <InlineCode>compare</InlineCode> 只有在两份 profile 都用 <InlineCode>--repeat 2</InlineCode> 或更高采集时才会报告统计显著性，单次运行的噪声足以误导你。
          </P>

          <Divider />

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
            <InlineCode>PoseValidator</InlineCode> 返回 COCO 关键点指标。除此之外还有分类（top-1 / top-5）、
            语义（mIoU / 像素精度）、点定位和深度（零样本）验证器，以及 v1.4.0 新增的全景
            （<InlineCode>PanopticValidator</InlineCode>，Panoptic Quality 指标）、抠图
            （<InlineCode>MatteValidator</InlineCode>）和 OCR（<InlineCode>OCRValidator</InlineCode>，
            最优一对一匹配）验证器。语义与全景验证自 v1.4.0 起接受 <InlineCode>augment=True</InlineCode>{' '}
            启用翻转 TTA（此前会报错）。传入{' '}
            <InlineCode>plots=True</InlineCode>（或命令行的 <InlineCode>--save-plots</InlineCode>）可将
            指标图、各类 AP、混淆矩阵和样本图写入运行目录。
          </P>

          <Divider />

          {/* ────────────── QUANTIZATION ────────────── */}
          <SectionHeading id="quantization" icon={Binary}>模型量化</SectionHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="experimental">v1.4.0 新增</SupportBadge>
            <SupportBadge variant="experimental">YOLO9 与 RF-DETR</SupportBadge>
          </div>
          <P>
            LibreYOLO 直接在 PyTorch 中量化模型。量化后的模型保持正常的{' '}
            <InlineCode>predict</InlineCode> / <InlineCode>val</InlineCode> /{' '}
            <InlineCode>train</InlineCode> / <InlineCode>save</InlineCode> 契约，因此精度用与浮点模型相同的验证器测量；精度恢复就是在量化模型上直接{' '}
            <InlineCode>train()</InlineCode>（QAT），或再加上现有的蒸馏参数（QAD）。
          </P>

          <SubHeading>语法：先量化，可选恢复</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9s.pt")

# Step 1: quantize. calib is a small UNLABELED image set, used forward-only
# to derive activation ranges and scales.
qmodel = model.quantize(recipe="int8", calib="coco128.yaml", samples=128)

qmodel.val(data="coco8.yaml")            # honest accuracy, same validators
qmodel.predict("bus.jpg")
qmodel.save("LibreYOLO9s-int8.pt")       # manifest-carrying checkpoint

# Step 2 (optional): QAT is plain train() on the quantized model
qmodel.train(data="coco.yaml", epochs=5)

# QAD: same, plus the existing distillation kwargs
qmodel.train(data="coco.yaml", epochs=5, distill_model="LibreYOLO9m.pt")`}</CodeBlock>
          <CodeBlock language="bash">{`libreyolo quantize --model LibreYOLO9s.pt --recipe int8 --calib coco8.yaml
libreyolo train model=LibreYOLO9s-int8.pt data=coco.yaml epochs=5`}</CodeBlock>
          <P>
            <InlineCode>LibreYOLO(&quot;LibreYOLO9s-int8.pt&quot;)</InlineCode> 会自动恢复量化结构和缩放系数：检查点携带{' '}
            <InlineCode>quant</InlineCode> 清单，QAT / QAD 期间训练器写出的检查点同样携带，因此 QAT 跑出的{' '}
            <InlineCode>best.pt</InlineCode> 本身就是量化检查点。<InlineCode>model.quant_info()</InlineCode>{' '}
            报告配方、模块数量、校准状态和执行层级；<InlineCode>model.dequantize()</InlineCode> 原地恢复浮点模块。
          </P>

          <SubHeading>配方</SubHeading>
          <DocTable
            headers={['配方', '做什么', '系列', '校准']}
            rows={[
              [<InlineCode key="r">fp16</InlineCode>, '半精度转换，保持 float32 输入输出契约。仅推理。', 'yolo9, rfdetr', '不需要'],
              [<InlineCode key="r">bf16</InlineCode>, 'bfloat16 转换：以一半存储保留 fp32 的指数范围；DETR 式模型 fp16 溢出时的解法。仅推理。', 'yolo9, rfdetr', '不需要'],
              [<InlineCode key="r">fp8</InlineCode>, 'E4M3 权重 + 激活仿真，作用于 Conv2d 与 Linear。', 'yolo9, rfdetr', '需要'],
              [<InlineCode key="r">int8</InlineCode>, 'W8A8：逐通道 INT8 权重、逐张量仿射 INT8 激活。', 'yolo9, rfdetr', '需要（calib=None 时仅量化权重）'],
              [<InlineCode key="r">w4a16</InlineCode>, '分组 INT4 权重、浮点激活，仅 Linear。', 'rfdetr', '不需要'],
              [<InlineCode key="r">w4a8</InlineCode>, '分组 INT4 权重 + INT8 激活；对应 NPU 的 W4A8 部署。', 'rfdetr', '需要'],
              [<InlineCode key="r">nvfp4</InlineCode>, 'NVFP4 W4A4：E2M1 元素、16 元素块、FP8 块缩放。', 'rfdetr', '不需要（动态）'],
              [<InlineCode key="r">mxfp4</InlineCode>, 'OCP MXFP4：E2M1 元素、32 元素块、二的幂次缩放。', 'rfdetr', '不需要（动态）'],
              [<InlineCode key="r">int2</InlineCode>, '研究预览：分组 2 比特权重 + INT8 激活。仅 PTQ 不可用；必须 QAT / QAD。', 'rfdetr', '需要'],
            ]}
          />
          <P>
            这样划分是有意的：当前硬件上低于 8 比特的加速只在 GEMM 上成立，因此仅作用于 Linear
            的配方会对 YOLO9 这类以卷积为主的系列直接拒绝（它们用{' '}
            <InlineCode>int8</InlineCode> 或 <InlineCode>fp8</InlineCode>）；4 比特配方的目标是
            transformer 系列（RF-DETR）。各系列的 <InlineCode>keep_high_precision</InlineCode>{' '}
            默认值保护第一层和各个头；确有把握时可用{' '}
            <InlineCode>quantize(..., keep_high_precision=(&quot;head.&quot;,))</InlineCode> 覆盖。
          </P>

          <SubHeading>校准数据不是训练数据</SubHeading>
          <ul className="space-y-2 my-4">
            <FeatureItem><InlineCode>calib=</InlineCode> 是几百张图像，不读标签、只做前向。它的工作是激活范围与缩放系数。默认 <InlineCode>coco128.yaml</InlineCode>（自动下载）；多个 batch 很重要，因为范围是跨 batch 估计的。</FeatureItem>
            <FeatureItem>train / val 上的 <InlineCode>data=</InlineCode> 才是带标签的数据集，负责梯度和指标。参数不同，职责不同。</FeatureItem>
            <FeatureItem>默认的范围估计是 <InlineCode>minmax</InlineCode>；<InlineCode>algorithm=&quot;percentile&quot;</InlineCode> 存在但实测处处更差，还会让 DETR 系精度崩塌，因为 transformer 的激活离群值是承重的。真正解决小模型 int8 敏感性的是用足够多的 batch 校准：用 coco128 默认值时，YOLO9-t 与 fp32 的差距约在 1 个 mAP 点以内。</FeatureItem>
          </ul>

          <SubHeading>诚实的数字：先仿真</SubHeading>
          <P>
            v1.4.0 以<strong>仿真</strong>执行量化算术（fake-quantize 加直通梯度，在 fp32 岛内计算）。仿真在数值上是真实的：任何设备上的{' '}
            <InlineCode>val()</InlineCode> 分数都是对量化算术的真实断言。但它<em>不是</em>速度断言；打包的低比特 kernel
            属于部署层面。<InlineCode>fp16</InlineCode> 与 <InlineCode>bf16</InlineCode> 转换是例外：它们原生执行。
          </P>

          <SubHeading>部署量化模型</SubHeading>
          <CodeBlock language="python">{`# Finalize: pack real low-bit weights, strip fp32 masters
qmodel.export(format="pt")    # -> <name>-final.pt

# int8 exports straight to QDQ ONNX with the model's own calibrated scales
qmodel = LibreYOLO("LibreYOLO9s-int8.pt")
qmodel.export(format="onnx")  # ONNX Runtime / TensorRT consume real INT8 kernels`}</CodeBlock>
          <ul className="space-y-2 my-4">
            <FeatureItem>定型后的检查点存储真实打包权重并相应变小（实测：YOLO9-s int8 从 29.5 降到 9.6 MB；RF-DETR-n nvfp4 从 122 降到 26 MB），在定型设备上解包与仿真逐比特一致。加载即得到可推理的模型，对它 <InlineCode>train()</InlineCode> 会自动重新准备主权重。</FeatureItem>
            <FeatureItem><InlineCode>fp16</InlineCode> / <InlineCode>bf16</InlineCode> 请先 <InlineCode>dequantize()</InlineCode> 再走浮点导出器（<InlineCode>half=True</InlineCode> 得到 fp16 ONNX）。</FeatureItem>
            <FeatureItem>低于 8 比特的 Linear 配方与 <InlineCode>fp8</InlineCode> 目前没有可部署的 ONNX 形式：它们在 PyTorch 中执行，通过 <InlineCode>format=&quot;pt&quot;</InlineCode> 定型。</FeatureItem>
            <FeatureItem>仿真由内置 Triton kernel 支撑，带可插拔注册表和 <InlineCode>LIBREYOLO_QUANT_KERNELS</InlineCode> 覆盖开关。</FeatureItem>
            <FeatureItem>携带定型量化状态的检查点无法被 v1.3.1 加载。</FeatureItem>
          </ul>

          <Divider />

          {/* ────────────── EXPORT ────────────── */}
          <SectionHeading id="export" icon={Upload}>导出</SectionHeading>
          <P>
            将 PyTorch 模型导出为 ONNX、TorchScript、TensorRT、OpenVINO、NCNN、CoreML 或 TFLite 以进行部署。经过充分测试的导出路径仍是单 GPU 的 YOLO9 检测、RF-DETR 检测和 RF-DETR 分割。
          </P>

          <SubHeading>支持矩阵是权威依据（v1.4.0 新增）</SubHeading>
          <P>
            v1.4.0 用一份规范化的导出支持矩阵取代了猜测。每个系列 / 任务 / 格式组合都有一个档位：{' '}
            <strong className="text-emerald-600 dark:text-emerald-400">validated</strong>（已测试，放心用）、{' '}
            <strong className="text-amber-600 dark:text-amber-400">experimental</strong>（可导出，带警告）或{' '}
            <strong>blocked</strong>（预先报错，而不是产出坏产物）。在围绕某个格式搭建流水线之前先查询它：
          </P>
          <CodeBlock language="bash">{`# Formats and tiers for one family / task
libreyolo formats --family yolo9
libreyolo formats --family rfdetr --task segment

# Everything about one model, including its export_support map
libreyolo info --model LibreYOLO9s.pt --json | jq .export_support`}</CodeBlock>
          <P>
            v1.4.0 还在固定分辨率、batch 1 的契约下解锁了整组任务的导出：PIDNet（语义）、FOMO（点定位）、ZipDepth / Depth Anything V2（深度）现在都能导出 ONNX。伴随矩阵还有两条行为保证：在请求被接受之前，导出绝不修改活动模型（LoRA 适配器折叠与量化模型重准备只在格式查询和选项预检通过之后发生）；LoRA 适配器在导出时合并进稠密权重，部署产物运行时不需要 peft。
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

# TFLite (needs Python 3.12+); "litert" is an accepted alias
model.export(format="tflite")

# Quantized checkpoints: pack low-bit weights, or emit QDQ INT8 ONNX
qmodel.export(format="pt")      # finalized packed checkpoint
qmodel.export(format="onnx")    # int8 -> QDQ ONNX (see Quantization)`}</CodeBlock>

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
            传入 <InlineCode>nms=True</InlineCode> 可将 NMS 烘焙进导出的 ONNX
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

          <SubHeading>TFLite (LiteRT) 导出</SubHeading>
          <P>
            <SupportBadge variant="experimental">运行时后端为 v1.4.0 新增</SupportBadge>{' '}
            LibreYOLO 提供基于 <InlineCode>onnx2tf</InlineCode> 的 TFLite 导出路径。TFLite 是
            Google LiteRT 运行时的格式（TensorFlow Lite 于 2024 年更名为 LiteRT，
            <InlineCode>.tflite</InlineCode> 文件格式不变）。它支持 YOLO9 与 YOLOX 检测、
            MobileNetV4 / ConvNeXt / EfficientNetV2 / ResNet 分类器、PIDNet 语义分割，
            以及 Real-ESRGAN 修复。RF-DETR 检测为实验性支持；RF-DETR 分割与姿态被拦截。
            它需要{' '}
            <strong className="text-surface-800 dark:text-white">Python 3.12+</strong>（{' '}
            <InlineCode>onnx2tf 2.4.x</InlineCode> wheels 不面向更老的 Python），以及
            可选 extra <InlineCode>libreyolo[tflite]</InlineCode>（别名{' '}
            <InlineCode>libreyolo[litert]</InlineCode>）。导出
            为 FP32 且仅静态形状（暂不支持 <InlineCode>half</InlineCode>、{' '}
            <InlineCode>int8</InlineCode> 或 <InlineCode>dynamic</InlineCode>）。
          </P>
          <CodeBlock language="bash">{`pip install "libreyolo[tflite]"   # Python 3.12+; [litert] is the same extra`}</CodeBlock>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")
model.export(format="tflite")   # writes a .tflite file; format="litert" also works`}</CodeBlock>
          <P>
            对于 RF-DETR，导出器会将每个 GridSample 节点改写为 TFLite 安全的双线性
            子图，因为 onnx2tf 的默认 lowering 在数值上有问题。
          </P>
          <P>
            <strong className="text-surface-800 dark:text-white">TFLite 不再只是导出格式。</strong>{' '}
            v1.4.0 新增：<InlineCode>LibreYOLO(&quot;model.tflite&quot;)</InlineCode> 会通过 LiteRT
            运行时后端（<InlineCode>ai-edge-litert</InlineCode>）加载该文件，运行 ONNX 与 TensorRT
            产物的同一个工厂现在也能运行 TFLite；见{' '}
            <a href="#tflite-inference" className="text-libre-600 dark:text-libre-400 hover:underline">TFLite 推理</a>。
          </P>

          <SubHeading>ONNX 元数据</SubHeading>
          <P>导出的 ONNX 文件包含内嵌元数据：</P>
          <DocTable
            headers={['键', '示例值']}
            rows={[
              [<InlineCode key="v">libreyolo_version</InlineCode>, <InlineCode key="vv">&quot;1.4.0&quot;</InlineCode>],
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

          {/* ────────────── TFLITE INFERENCE ────────────── */}
          <SectionHeading id="tflite-inference" icon={Cpu}>TFLite 推理</SectionHeading>
          <div className="flex flex-wrap gap-2 mb-4">
            <SupportBadge variant="experimental">v1.4.0 新增</SupportBadge>
          </div>
          <P>
            通过 Google 的 LiteRT 解释器（即原 TensorFlow Lite 运行时）运行导出的{' '}
            <InlineCode>.tflite</InlineCode> 文件。需要 Python 3.12+ 与{' '}
            <InlineCode>pip install &quot;libreyolo[tflite]&quot;</InlineCode>（或别名{' '}
            <InlineCode>[litert]</InlineCode>），它会安装 <InlineCode>ai-edge-litert</InlineCode>。
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("model.tflite")

result = model("image.jpg", conf=0.25, iou=0.45, save=True)
print(result.boxes.xyxy)`}</CodeBlock>
          <P>
            TFLite 产物支持与其他后端相同的核心运行时预测 API，包括 <InlineCode>save=True</InlineCode>{' '}
            时 <InlineCode>output_path</InlineCode> 只接受文件路径的行为。导出的计算图为静态形状，请以导出时的{' '}
            <InlineCode>imgsz</InlineCode> 运行。
          </P>

          <Divider />

          {/* ────────────── CLI ────────────── */}
          <SectionHeading id="cli" icon={SquareTerminal}>命令行（CLI）</SectionHeading>
          <P>
            安装 LibreYOLO 会在你的 PATH 中注册一个 <InlineCode>libreyolo</InlineCode> 命令（入口点在 <InlineCode>pyproject.toml</InlineCode> 中）。命令行镜像 Python API，并接受 <InlineCode>key=value</InlineCode> 语法。
          </P>

          <SubHeading>子命令</SubHeading>
          <DocTable
            headers={['命令', '用途']}
            rows={[
              [<InlineCode key="p">predict</InlineCode>, '对图像、目录或视频运行推理'],
              [<InlineCode key="t">train</InlineCode>, '在数据集上训练模型'],
              [<InlineCode key="v">val</InlineCode>, '在数据集上评估模型'],
              [<InlineCode key="e">export</InlineCode>, '导出为 ONNX / TorchScript / TensorRT / OpenVINO / NCNN / CoreML / TFLite'],
              [<InlineCode key="q">quantize</InlineCode>, '用配方 + 校准集量化模型（v1.4.0 新增）'],
              [<InlineCode key="lb">label</InlineCode>, '启动 LibreLabel 本地浏览器标注工具'],
              [<InlineCode key="mo">monitor</InlineCode>, '为训练任务提供实时看板'],
              [<InlineCode key="pr">profile</InlineCode>, '分析训练或推理性能，并解读结果'],
              [<InlineCode key="ui">ui</InlineCode>, '启动本地拖放 / 粘贴的浏览器推理界面'],
              [<InlineCode key="dr">doctor</InlineCode>, '运行训练前的数据集健康检查（YOLO 检测格式）'],
              [<InlineCode key="c">checks</InlineCode>, '打印 Python、torch、CUDA、GPU 和可选包信息'],
              [<InlineCode key="m">models</InlineCode>, '列出已注册的模型系列和命令行简称（v1.4.0 内容更丰富；--json 模式已变更）'],
              [<InlineCode key="f">formats</InlineCode>, '列出导出格式；--family / --task 按支持档位过滤（v1.4.0 新增）'],
              [<InlineCode key="cfg">cfg</InlineCode>, '打印默认的训练配置 YAML'],
              [<InlineCode key="i">info</InlineCode>, '加载模型并打印系列、尺寸、任务、设备、类别及其 export_support 映射'],
              [<InlineCode key="md">metadata</InlineCode>, '检查 .pt 文件中的原始检查点元数据'],
              [<InlineCode key="ver">version</InlineCode>, '打印 LibreYOLO + Python + torch 版本'],
            ]}
          />

          <SubHeading>模型名称简称</SubHeading>
          <P>
            命令行接受简称（<InlineCode>yolo9-c</InlineCode>），它们会解析为权重文件名（<InlineCode>LibreYOLO9c.pt</InlineCode>）- 可通过 <InlineCode>libreyolo models</InlineCode> 查看。你也可以传入任意明确的检查点路径。
          </P>
          <P>
            v1.4.0 已发布版本有两个限制：未安装 <InlineCode>transformers</InlineCode> 时，
            <InlineCode>libreyolo models</InlineCode> 会省略 RF-DETR 与 DINOv2，而不是将其标为不可用；
            某些默认任务并非检测的系列，其带任务后缀的简称也可能无法解析。需要 transformer 系列时请安装
            <InlineCode>libreyolo[rfdetr]</InlineCode>，简称失败时请传入完整的官方检查点文件名。
          </P>

          <SubHeading>常用选项</SubHeading>
          <DocTable
            headers={['命令', '重要选项']}
            rows={[
              [<InlineCode key="p">predict</InlineCode>, <span key="pv"><InlineCode>conf</InlineCode>, <InlineCode>iou</InlineCode>, <InlineCode>imgsz</InlineCode>, <InlineCode>classes</InlineCode>, <InlineCode>max_det</InlineCode>, <InlineCode>half</InlineCode>, <InlineCode>batch</InlineCode>, <InlineCode>tiling</InlineCode>, <InlineCode>overlap_ratio</InlineCode>, <InlineCode>output_file_format</InlineCode>, <InlineCode>project</InlineCode>, <InlineCode>name</InlineCode>, <InlineCode>exist_ok</InlineCode>, <InlineCode>face_detector</InlineCode></span>],
              [<InlineCode key="t">train</InlineCode>, <span key="tv"><InlineCode>epochs</InlineCode>, <InlineCode>batch</InlineCode>, <InlineCode>imgsz</InlineCode>, <InlineCode>lr0</InlineCode>, <InlineCode>optimizer</InlineCode>, <InlineCode>scheduler</InlineCode>, <InlineCode>workers</InlineCode>, <InlineCode>seed</InlineCode>, <InlineCode>resume</InlineCode>, <InlineCode>amp</InlineCode>, <InlineCode>task</InlineCode>, <InlineCode>cache</InlineCode>, <InlineCode>lora</InlineCode>, <InlineCode>freeze</InlineCode>, <InlineCode>save_plots</InlineCode>, <InlineCode>allow_download_scripts</InlineCode>, <InlineCode>dry_run</InlineCode>，以及各<a href="#augmentation" className="text-libre-600 dark:text-libre-400 hover:underline">增强参数</a>（<InlineCode>mosaic</InlineCode>, <InlineCode>mixup</InlineCode>, <InlineCode>hsv_prob</InlineCode>, <InlineCode>flip_prob</InlineCode>, <InlineCode>degrees</InlineCode>, ...）</span>],
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

          <SubHeading>量化（v1.4.0 新增）</SubHeading>
          <CodeBlock language="bash">{`# PTQ: int8 with a small calibration set
libreyolo quantize --model LibreYOLO9s.pt --recipe int8 --calib coco8.yaml --samples 128

# Write to an explicit path, machine-readable output
libreyolo quantize --model LibreRFDETRn.pt --recipe nvfp4 --out rfdetr-nvfp4.pt --json`}</CodeBlock>
          <P>
            量化后的检查点接入常规命令：<InlineCode>libreyolo val</InlineCode> 得到可信精度、{' '}
            <InlineCode>libreyolo train</InlineCode> 做 QAT、<InlineCode>libreyolo export</InlineCode>{' '}
            做部署。见<a href="#quantization" className="text-libre-600 dark:text-libre-400 hover:underline">模型量化</a>。
          </P>

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
    points: Points | None = None,
    semantic_mask: SemanticMask | None = None,
    depth_map: DepthMap | None = None,
    restored: RestoredImage | None = None,
    speed: dict[str, float] | None = None,
    track_id = None,
    frame_idx: int | None = None,
    # New in v1.4.0 (placed after the complete v1.3 signature, so
    # positional v1.3 call sites keep working):
    panoptic: PanopticSegmentation | None = None,
    matte: Matte | None = None,
    ocr: OCRRegions | None = None,
    restore_scale: int = 1,
)

len(result)          # number of detections
result.cpu()         # copy with tensors on CPU
result.cuda()        # copy with tensors on CUDA
result.numpy()       # copy with numpy arrays
result.summary()     # list[dict] with the payloads present
result.to_json()     # JSON string from summary()
result.cutout()      # (H, W, 4) RGBA ndarray; matte results only`}</CodeBlock>

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

result.obb.xywhr         # (N, 5): center x/y, width, height, rotation
result.obb.xyxyxyxy      # (N, 4, 2): four oriented box corners
result.obb.conf          # (N,) confidence scores
result.obb.cls           # (N,) class IDs

result.gaze.data         # (N, 2): pitch, yaw in radians
result.gaze.pitch_deg    # pitch in degrees
result.gaze.yaw_deg      # yaw in degrees
result.gaze.direction_3d # approximate 3D direction vectors

result.semantic_mask.data      # (H, W) class-id map (semantic)
result.depth_map.data          # (H, W) relative inverse depth
result.points.xy               # (N, 2) point detections (FOMO)
result.restored.array          # (H, W, 3) uint8 restored image
result.restore_scale           # int upscale factor; 1 unless super-resolution

# New in v1.4.0
result.panoptic.data           # (H, W) segment-id map
result.panoptic.segments_info  # per-segment {"id", "category_id", ...}
result.matte.data              # (H, W) float32 alpha in [0, 1]
result.ocr.polygons            # (N, 4, 2) text-region quads
result.ocr.texts               # list[str] transcriptions
result.ocr.conf                # (N,) recognition scores
result.ocr.det_conf            # (N,) detection scores`}</CodeBlock>

          <SubHeading>model.track()</SubHeading>
          <CodeBlock language="python">{`model.track(
    source,                       # video file path
    *,
    track_conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = None,
    classes: list[int] = None,
    max_det: int = 300,
    save: bool = False,
    show: bool = False,
    vid_stride: int = 1,
    output_path: str = None,
    tracker: str = "bytetrack",   # "bytetrack" | "ocsort" | "botsort" | "deepocsort"
    tracker_config = None,        # a config instance selects the tracker by type
    augment: bool = False,
    **tracker_kwargs,
) -> Generator[Results, None, None]`}</CodeBlock>

          <SubHeading>model.quantize()（v1.4.0 新增）</SubHeading>
          <CodeBlock language="python">{`model.quantize(
    recipe: str,                  # "fp16" | "bf16" | "fp8" | "int8" | "w4a16"
                                  # | "w4a8" | "nvfp4" | "mxfp4" | "int2"
    calib: str = "coco128.yaml",  # unlabeled calibration images (forward-only)
    samples: int = 128,
    batch: int = 8,
    algorithm: str = "auto",      # "auto" (minmax) | "minmax" | "percentile"
    keep_high_precision = None,   # module-name substrings to keep in float
    verbose: bool = True,
) -> model                        # quantized in place

model.quant_info()                # dict describing the quant state, or None
model.dequantize()                # restore float modules in place`}</CodeBlock>

          <SubHeading>model.export()</SubHeading>
          <CodeBlock language="python">{`model.export(
    format: str = "onnx",       # "onnx", "torchscript", "tensorrt", "openvino",
                                # "ncnn", "coreml", "tflite" (alias "litert"), or "pt"
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
            每个任务都通过一个 <InlineCode>data.yaml</InlineCode> 加载。检测、实例分割和 OBB 接受<strong className="text-surface-800 dark:text-white">两种可互换的标签格式</strong>（YOLO TXT 或原生 COCO JSON），加载器会根据配置自动选择。姿态、语义分割、深度和分类各自增加一个小型格式。下表将每个任务映射到其布局。
          </P>

          <SubHeading>按任务划分的格式</SubHeading>
          <DocTable
            headers={['任务', '数据布局', '标签']}
            rows={[
              ['检测', <span key="l"><InlineCode>data.yaml</InlineCode> + <InlineCode>labels/*.txt</InlineCode>，或 COCO JSON</span>, '每行一个框'],
              ['实例分割', <span key="l"><InlineCode>data.yaml</InlineCode> + 多边形 <InlineCode>.txt</InlineCode>，或 COCO JSON</span>, '每行一个多边形（TXT）/ 多边形 + RLE（COCO）'],
              ['OBB', <span key="l"><InlineCode>data.yaml</InlineCode> + 旋转框 <InlineCode>.txt</InlineCode>，或 COCO JSON</span>, '每行一个旋转框'],
              ['姿态', <span key="l"><InlineCode>data.yaml</InlineCode> + <InlineCode>.txt</InlineCode> + <InlineCode>kpt_shape</InlineCode>/<InlineCode>flip_idx</InlineCode></span>, '每行一个框 + 关键点'],
              ['语义分割', <span key="l"><InlineCode>data.yaml</InlineCode> + <InlineCode>masks_dir/</InlineCode> PNG</span>, '逐像素类别 ID（255 = 忽略）'],
              ['深度', <span key="l"><InlineCode>data.yaml</InlineCode> + <InlineCode>depths_dir/</InlineCode> 深度图</span>, '逐像素深度（0 = 无效）'],
              ['分类', <span key="l">ImageFolder（<InlineCode>train/&lt;class&gt;/</InlineCode>）</span>, '文件夹名 = 类别'],
            ]}
          />

          <SubHeading>data.yaml 结构</SubHeading>
          <P>
            检测、分割、OBB 和姿态的通用契约。<InlineCode>train</InlineCode>/<InlineCode>val</InlineCode>/<InlineCode>test</InlineCode> 可以是目录、<InlineCode>.txt</InlineCode> 文件列表（每行一个图像路径）或路径列表。<InlineCode>nc</InlineCode> 可选：省略时从 <InlineCode>names</InlineCode> 推断。
          </P>
          <CodeBlock language="yaml" filename="data.yaml">{`path: /absolute/path/to/dataset   # dataset root
train: images/train               # dir, .txt file list, or list of paths
val: images/val
test: images/test                 # optional

nc: 80                            # optional; inferred from names if absent
names: ["person", "bicycle", "car", "..."]`}</CodeBlock>
          <P>
            配置按以下顺序解析：明确路径、当前工作目录，然后是 <InlineCode>libreyolo/config/datasets/</InlineCode> 下的内置配置。根目录默认在 <InlineCode>~/datasets</InlineCode> 下；用 <InlineCode>LIBREYOLO_DATASETS_DIR</InlineCode> 覆盖。
          </P>

          <SubHeading>YOLO TXT 标签</SubHeading>
          <P>
            默认布局：每张图像在 <InlineCode>labels/</InlineCode> 下有一个 <InlineCode>.txt</InlineCode>，镜像 <InlineCode>images/</InlineCode> 树并使用相同的文件名主干。所有坐标都归一化到 [0, 1]。
          </P>
          <CodeBlock language="text">{`dataset/
    images/train/img001.jpg
    labels/train/img001.txt        # same stem as the image`}</CodeBlock>
          <CodeBlock language="text">{`# Detection      one box per line
<class_id> <cx> <cy> <w> <h>

# Segmentation   one polygon per line (box derived from the vertices)
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>

# Pose           box, then K keypoints (needs kpt_shape / flip_idx below)
<class_id> <cx> <cy> <w> <h> <kx1> <ky1> <v1> ... <kxK> <kyK> <vK>

# OBB            four rotated-box corners
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>`}</CodeBlock>
          <CodeBlock language="yaml" filename="data.yaml (pose)">{`kpt_shape: [17, 3]   # K keypoints, 3 values each: x, y, visibility
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]`}</CodeBlock>

          <SubHeading>原生 COCO JSON</SubHeading>
          <P>
            检测、分割和 OBB 也可直接加载 COCO JSON：添加一个 <InlineCode>annotations:</InlineCode> 块，将每个划分映射到其 JSON 文件。此时 <InlineCode>train</InlineCode>/<InlineCode>val</InlineCode> 指向图像<em>目录</em>（而非 <InlineCode>.txt</InlineCode> 列表）。需要 <InlineCode>pycocotools</InlineCode>；类别名来自 JSON 的 categories，因此 <InlineCode>nc</InlineCode>/<InlineCode>names</InlineCode> 可选。
          </P>
          <CodeBlock language="yaml" filename="data.yaml (COCO)">{`path: dataset
train: images/train               # image directory
val: images/val
annotations:
  train: annotations/train.json   # COCO instances JSON
  val: annotations/val.json`}</CodeBlock>
          <P>
            同一套切换逻辑服务于 YOLO9、RF-DETR、DEIM 和 D-FINE 训练，以及检测、OBB 和姿态验证器。磁盘上带有 <InlineCode>annotations/instances_train2017.json</InlineCode> 的 COCO 布局也会被自动识别，无需 <InlineCode>annotations:</InlineCode> 键。
          </P>
          <P>
            <strong className="text-surface-800 dark:text-white">选择哪种分割格式？</strong> 一个 YOLO 多边形行是每个实例一个环，无法表示带孔或分离（多部分）的掩码。COCO JSON 会保留实例的每个多边形并解码 RLE 掩码（含孔）。当实例带孔或存在不相连的部分时使用 COCO JSON；简单的单块掩码用任一格式均可。人群标注（<InlineCode>iscrowd: 1</InlineCode>）会被跳过。
          </P>

          <SubHeading>语义分割掩码</SubHeading>
          <P>
            为每张图像配一张单通道掩码，其像素值为类别 ID；<InlineCode>255</InlineCode> 标记被忽略的像素。<InlineCode>masks_dir</InlineCode> 会替换每个路径中的 <InlineCode>images</InlineCode>（默认 <InlineCode>masks</InlineCode>），掩码必须是无损（PNG）且与其图像同名主干。可选的 <InlineCode>label_mapping</InlineCode> 将源 ID 重映射为训练 ID（未映射的值变为忽略）。省略 <InlineCode>masks_dir</InlineCode> 则在加载时从 YOLO 多边形标签栅格化掩码，并追加一个 <InlineCode>background</InlineCode> 类。
          </P>
          <CodeBlock language="text">{`dataset/
    images/train/scene001.jpg
    masks/train/scene001.png       # single-channel class IDs, 255 = ignore`}</CodeBlock>
          <CodeBlock language="yaml" filename="data.yaml (semantic)">{`path: /path/to/dataset
train: images/train
val: images/val
masks_dir: masks
nc: 3
names: ["road", "building", "vegetation"]`}</CodeBlock>

          <SubHeading>深度图</SubHeading>
          <P>
            为每张图像配一张单通道深度图，放在 <InlineCode>depths_dir</InlineCode> 下（默认 <InlineCode>depths</InlineCode>）。16 位 PNG/TIF 会除以 <InlineCode>depth_scale</InlineCode>（默认 <InlineCode>256.0</InlineCode>）；<InlineCode>.npy</InlineCode> 浮点文件按原样使用。零、负值和非有限像素为无效。可选的 <InlineCode>depth_stem_suffix</InlineCode> 和 <InlineCode>*_mask</InlineCode> 有效性掩码会被自动识别。深度仅支持验证。
          </P>
          <CodeBlock language="yaml" filename="data.yaml (depth)">{`path: /path/to/dataset
val: images/val
depths_dir: depths
depth_scale: 256.0                # 16-bit PNG encoding: value / 256 = depth`}</CodeBlock>

          <SubHeading>分类</SubHeading>
          <P>
            分类使用 ImageNet 风格的 ImageFolder 树，而非 <InlineCode>data.yaml</InlineCode>，布局参见 <a href="#classification" className="text-libre-600 dark:text-libre-400 hover:underline">分类</a>。<InlineCode>data=</InlineCode> 接受数据集根目录、<InlineCode>.zip</InlineCode> URL 或已知名称。
          </P>

          <SubHeading>内置数据集</SubHeading>
          <P>
            配置位于 <InlineCode>libreyolo/config/datasets/</InlineCode> 下。下载行为因配置而异：基于 URL 的数据集在首次使用时获取，基于脚本的数据集需要 <InlineCode>allow_download_scripts=True</InlineCode>，少数需要在本地放置。
          </P>
          <DocTable
            headers={['配置', '任务', '下载']}
            rows={[
              [<InlineCode key="c">coco8</InlineCode>, '检测（8 张）', '自动'],
              [<InlineCode key="c">coco128</InlineCode>, '检测（128 张）', '自动'],
              [<InlineCode key="c">coco5000</InlineCode>, '检测', <span key="d">脚本：<InlineCode>allow_download_scripts=True</InlineCode></span>],
              [<span key="c"><InlineCode>coco</InlineCode> / <InlineCode>coco-val-only</InlineCode></span>, '检测（完整）', <span key="d">脚本：<InlineCode>allow_download_scripts=True</InlineCode></span>],
              [<span key="c"><InlineCode>coco8-pose</InlineCode> / <InlineCode>coco-pose</InlineCode></span>, '姿态', <span key="d">脚本：<InlineCode>allow_download_scripts=True</InlineCode></span>],
              [<InlineCode key="c">cocostuff</InlineCode>, '语义（182 类）', '手动：本地放置'],
            ]}
          />
          <CodeBlock language="python">{`results = model.val(data="coco8.yaml")                          # auto-downloads
results = model.train(data="coco128.yaml", epochs=10)           # auto-downloads
model.train(data="coco8-pose.yaml", allow_download_scripts=True)  # script config`}</CodeBlock>

          {/* Bottom spacer */}
          <div className="h-16" />
        </div>
      </main>
    </div>
  )
}


export default function Docs() {
  const locale = useLocale()
  if (locale === 'zh') return <DocsPageZh version="v1.4.0" />
  return <DocsPage version="v1.4.0" />
}
