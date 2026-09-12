import fs from 'node:fs'
import path from 'node:path'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const components = {
  h1: ({ node, ...props }) => (
    <h1 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-surface-800 dark:text-white mb-4 leading-tight" {...props} />
  ),
  h2: ({ node, ...props }) => (
    <h2 className="text-2xl font-bold text-surface-800 dark:text-white mt-12 mb-4" {...props} />
  ),
  p: ({ node, ...props }) => <p className="text-surface-600 dark:text-surface-400 leading-relaxed mb-5" {...props} />,
  a: ({ node, ...props }) => (
    <a
      className="text-libre-600 dark:text-libre-400 hover:text-libre-700 dark:hover:text-libre-300 underline underline-offset-2 transition-colors"
      target={props.href?.startsWith('http') ? '_blank' : undefined}
      rel={props.href?.startsWith('http') ? 'noopener noreferrer' : undefined}
      {...props}
    />
  ),
  ul: ({ node, ...props }) => <ul className="list-disc pl-6 mb-5 space-y-2 text-surface-600 dark:text-surface-400" {...props} />,
  ol: ({ node, ...props }) => <ol className="list-decimal pl-6 mb-5 space-y-2 text-surface-600 dark:text-surface-400" {...props} />,
  li: ({ node, ...props }) => <li className="leading-relaxed" {...props} />,
  strong: ({ node, ...props }) => <strong className="font-semibold text-surface-800 dark:text-white" {...props} />,
  table: ({ node, ...props }) => (
    <div className="overflow-x-auto mb-6">
      <table className="w-full text-sm border-collapse" {...props} />
    </div>
  ),
  th: ({ node, ...props }) => <th className="text-left font-semibold text-surface-800 dark:text-white border-b border-surface-300 dark:border-surface-700 px-3 py-2" {...props} />,
  td: ({ node, ...props }) => <td className="text-surface-600 dark:text-surface-400 border-b border-surface-200 dark:border-surface-800 px-3 py-2 align-top" {...props} />,
  code: ({ node, ...props }) => (
    <code className="font-mono text-sm px-1.5 py-0.5 rounded bg-surface-100 dark:bg-surface-800 text-libre-700 dark:text-libre-300" {...props} />
  ),
}

export default function Sponsors() {
  const content = fs.readFileSync(path.join(process.cwd(), 'content', 'sponsors.md'), 'utf8')
  return (
    <div className="pt-24 lg:pt-32 pb-20">
      <article className="max-w-3xl mx-auto px-6 lg:px-8">
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
          {content}
        </ReactMarkdown>
      </article>
    </div>
  )
}
