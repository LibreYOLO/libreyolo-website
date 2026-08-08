'use client'

/*
 * Code block for the v2 docs.
 *
 * Deliberately plainer than the one on the versioned docs pages: no fake window
 * chrome, no traffic-light dots, no gradient fill, no line numbers. A hairline
 * border, a muted language label, and a text copy control that only gains
 * contrast on hover. Reference sites present code as text, not as a screenshot
 * of an editor.
 */

import { useState } from 'react'
import { highlightLine } from '@/components/DocsKit'

const LABELS = {
  bash: 'bash', sh: 'bash', shell: 'shell',
  py: 'python', python: 'python',
  yaml: 'yaml', yml: 'yaml', text: '', json: 'json',
}

export default function Code({ children, language = 'text', label }) {
  const [copied, setCopied] = useState(false)
  const code = String(children ?? '').replace(/\n$/, '')
  // When a snippet carries its own name ("CLI", "Run the ONNX"), that name is
  // more useful than the language, so it wins the header slot.
  const caption = label ?? LABELS[language.toLowerCase()] ?? language.toLowerCase()

  const copy = () => {
    navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 1600)
  }

  return (
    <div className="group my-4 border border-surface-200 bg-surface-50/60 dark:border-white/[0.09] dark:bg-white/[0.02]">
      <div className="flex items-center justify-between border-b border-surface-200/70 px-3 py-1 dark:border-white/[0.07]">
        <span className="font-mono text-[11px] text-surface-500 dark:text-surface-500">{caption}</span>
        <button
          onClick={copy}
          className="font-mono text-[11px] text-surface-400 transition-colors hover:text-surface-900 dark:text-surface-600 dark:hover:text-surface-200"
        >
          {copied ? 'copied' : 'copy'}
        </button>
      </div>
      <pre className="overflow-x-auto px-3 py-2.5">
        <code className="font-mono text-[12.5px] leading-[1.7] text-surface-800 dark:text-surface-300">
          {code.split('\n').map((line, i) => (
            <span key={i} className="block">
              {highlightLine(line, language).map((token, j) => (
                <span key={j} className={token.className}>{token.value}</span>
              ))}
              {line === '' ? ' ' : ''}
            </span>
          ))}
        </code>
      </pre>
    </div>
  )
}
