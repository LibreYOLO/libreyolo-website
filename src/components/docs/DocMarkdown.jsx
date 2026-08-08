/*
 * Markdown renderer for docs pages.
 *
 * Prose is markdown. Facts are components: the author drops a tag such as
 * <checkpoint-table /> where the generated block belongs, rehype-raw keeps the
 * unknown tag, and the map below swaps in the real component.
 */

import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeRaw from 'rehype-raw'

import Code from '@/components/docs/Code'
import { slugifyHeading } from '@/lib/docs'
import CodeTabs from '@/components/docs/CodeTabs'
import {
  SectionTitle, BenchmarkTable, VaEmbed, CheckpointTable, ExportMatrix,
  Provenance, Citation,
} from '@/components/docs/ModelBlocks'

function textOf(children) {
  if (typeof children === 'string') return children
  if (Array.isArray(children)) return children.map(textOf).join('')
  if (children?.props?.children) return textOf(children.props.children)
  return ''
}

/*
 * react-markdown hands each renderer the hast node in `node`. Spreading props
 * straight onto a DOM element therefore emits node="[object Object]" on every
 * tag, which is invalid HTML and noise in the .md twin and for crawlers. Strip
 * it once here rather than destructuring it in a dozen renderers.
 */
function dom({ node, ...rest }) {
  return rest
}

export default function DocMarkdown({ children, family, snippets = {} }) {
  const components = {
    h2: ({ children: kids }) => <SectionTitle id={slugifyHeading(textOf(kids))}>{kids}</SectionTitle>,
    h3: ({ children: kids }) => (
      <h3 id={slugifyHeading(textOf(kids))} className="scroll-mt-24 mt-7 mb-2 text-[15px] font-semibold text-surface-900 dark:text-white">
        {kids}
      </h3>
    ),
    // Paragraph spacing stays clearly larger than intra-paragraph leading, or
    // paragraph boundaries dissolve and the page reads as one grey field.
    p: (props) => <p className="mb-5 max-w-[68ch] text-[15px] leading-[1.6] text-surface-600 dark:text-surface-400" {...dom(props)} />,
    a: (props) => (
      <a
        className="font-medium text-libre-600 underline-offset-2 hover:underline dark:text-libre-400"
        target={props.href?.startsWith('http') ? '_blank' : undefined}
        rel={props.href?.startsWith('http') ? 'noopener noreferrer' : undefined}
        {...dom(props)}
      />
    ),
    ul: (props) => <ul className="mb-4 max-w-[68ch] list-disc space-y-1 pl-5 text-[15px] text-surface-600 dark:text-surface-400" {...dom(props)} />,
    ol: (props) => <ol className="mb-4 max-w-[68ch] list-decimal space-y-1 pl-5 text-[15px] text-surface-600 dark:text-surface-400" {...dom(props)} />,
    li: (props) => <li className="leading-[1.6]" {...dom(props)} />,
    strong: (props) => <strong className="font-semibold text-surface-800 dark:text-surface-200" {...dom(props)} />,
    hr: () => <hr className="my-10 border-surface-200 dark:border-white/[0.06]" />,
    blockquote: (props) => (
      <blockquote className="my-5 border-l-2 border-libre-500 pl-4 text-surface-500 dark:text-surface-400" {...dom(props)} />
    ),

    /*
     * One table skin, matching the generated blocks: hairline row rules, a
     * stronger rule under the header, no container border, no fill, no radius,
     * no hover, no zebra. An authored table and a generated one have to look
     * like one system, and the house rules forbid the boxed alternative.
     */
    table: (props) => (
      <div className="-mx-1 my-5 overflow-x-auto px-1">
        <table className="w-full border-collapse text-[13.5px]" {...dom(props)} />
      </div>
    ),
    th: (props) => (
      <th className="border-b border-surface-300 px-3 py-1.5 text-left font-semibold text-surface-700 dark:border-white/20 dark:text-surface-300" {...dom(props)} />
    ),
    td: (props) => (
      <td className="border-b border-surface-200/70 px-3 py-1.5 align-top text-surface-700 dark:border-white/[0.07] dark:text-surface-400" {...dom(props)} />
    ),

    code: ({ className, children: kids, ...props }) => {
      if (/language-/.test(className || '')) {
        return <code className={className} {...dom(props)}>{kids}</code>
      }
      return (
        <code className="rounded bg-libre-500/10 px-1.5 py-0.5 font-mono text-[0.85em] text-libre-700 dark:bg-white/[0.06] dark:text-libre-300" {...dom(props)}>
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

    /* Generated blocks. The author writes the tag; the registry supplies data. */
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
