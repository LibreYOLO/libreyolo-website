/*
 * Markdown renderer for docs pages.
 *
 * Prose is markdown. Facts are components: the author drops a tag such as
 * <checkpoint-table /> where the generated block belongs, rehype-raw keeps the
 * unknown tag, and the map below swaps in the real component. The same trick
 * the articles pipeline already uses for its inline widgets.
 */

import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeRaw from 'rehype-raw'

import Code from '@/components/docs/Code'
import { slugifyHeading } from '@/lib/docs'
import CodeTabs from '@/components/docs/CodeTabs'
import {
  SectionTitle, BenchmarkTable, VaEmbed, CheckpointTable, ExportMatrix,
  TaskSupport, Provenance, Citation,
} from '@/components/docs/ModelBlocks'

function textOf(children) {
  if (typeof children === 'string') return children
  if (Array.isArray(children)) return children.map(textOf).join('')
  if (children?.props?.children) return textOf(children.props.children)
  return ''
}

export default function DocMarkdown({ children, family, snippets = {} }) {
  const components = {
    h2: ({ children: kids }) => <SectionTitle id={slugifyHeading(textOf(kids))}>{kids}</SectionTitle>,
    h3: ({ children: kids }) => (
      <h3 id={slugifyHeading(textOf(kids))} className="scroll-mt-24 mt-7 mb-2 text-[15px] font-semibold text-surface-900 dark:text-white">
        {kids}
      </h3>
    ),
    // Paragraph spacing must stay clearly larger than intra-paragraph leading,
    // or paragraph boundaries dissolve and the page reads as one grey field.
    // This is the most-cited complaint about the 2022 MDN redesign (1.75 leading).
    p: (props) => <p className="mb-5 max-w-[68ch] text-[15px] leading-[1.6] text-surface-600 dark:text-surface-400" {...props} />,
    a: (props) => (
      <a
        className="font-medium text-libre-600 underline-offset-2 hover:underline dark:text-libre-400"
        target={props.href?.startsWith('http') ? '_blank' : undefined}
        rel={props.href?.startsWith('http') ? 'noopener noreferrer' : undefined}
        {...props}
      />
    ),
    ul: (props) => <ul className="mb-4 max-w-[68ch] list-disc space-y-1 pl-5 text-[15px] text-surface-600 dark:text-surface-400" {...props} />,
    ol: (props) => <ol className="mb-4 max-w-[68ch] list-decimal space-y-1 pl-5 text-[15px] text-surface-600 dark:text-surface-400" {...props} />,
    li: (props) => <li className="leading-[1.6]" {...props} />,
    strong: (props) => <strong className="font-semibold text-surface-800 dark:text-surface-200" {...props} />,
    hr: () => <hr className="my-10 border-surface-200 dark:border-white/[0.06]" />,
    blockquote: (props) => (
      <blockquote className="my-5 border-l-2 border-libre-500 pl-4 text-surface-500 dark:text-surface-400" {...props} />
    ),
    table: (props) => (
      <div className="my-5 overflow-x-auto rounded-xl border border-surface-200 dark:border-white/[0.08]">
        <table className="w-full text-sm" {...props} />
      </div>
    ),
    th: (props) => <th className="border-b border-surface-200 bg-surface-50 px-4 py-3 text-left font-semibold text-surface-700 dark:border-white/[0.08] dark:bg-white/[0.02] dark:text-surface-300" {...props} />,
    td: (props) => <td className="border-b border-surface-100 px-4 py-3 text-surface-600 last:border-0 dark:border-white/[0.04] dark:text-surface-400" {...props} />,
    code: ({ className, children: kids, ...props }) => {
      if (/language-/.test(className || '')) {
        return <code className={className} {...props}>{kids}</code>
      }
      return (
        <code className="rounded bg-libre-500/10 px-1.5 py-0.5 font-mono text-[0.85em] text-libre-700 dark:bg-white/[0.06] dark:text-libre-300" {...props}>
          {kids}
        </code>
      )
    },
    pre: ({ children: kids }) => {
      const node = Array.isArray(kids) ? kids[0] : kids
      const className = node?.props?.className || ''
      const language = (/language-(\w+)/.exec(className) || [, 'text'])[1]
      return <Code language={language}>{textOf(node?.props?.children).replace(/\n$/, '')}</Code>
    },

    /* Generated blocks. The author writes the tag; the pipeline supplies data. */
    'task-support': () => <TaskSupport family={family} />,
    'benchmark-table': ({ task }) => <BenchmarkTable family={family} task={task || 'detect'} />,
    'va-embed': () => <VaEmbed family={family} />,
    'checkpoint-table': () => <CheckpointTable family={family} />,
    'export-matrix': () => <ExportMatrix family={family} />,
    'code-tabs': ({ name }) => <CodeTabs tabs={snippets[name] || []} />,
    'provenance-box': ({ children: kids }) => <Provenance family={family}>{kids}</Provenance>,
    'citation-block': () => <Citation family={family} />,
  }

  return (
    <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw]} components={components}>
      {children}
    </ReactMarkdown>
  )
}
