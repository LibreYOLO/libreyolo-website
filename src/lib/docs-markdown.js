import registry, { getFamilies, getTaskMeta, getExportFormats, expandSelfClosingTags } from '@/lib/docs'

/*
 * Render a docs page as plain markdown, for the `.md` twin and for
 * llms-full.txt.
 *
 * The generated blocks are expanded into real markdown tables here. A twin that
 * still contained `<checkpoint-table />` would hand an agent a page with the
 * facts missing, which is worse than no twin at all: the prose says "the table
 * below" and there is no table. Everything a reader of the HTML page can see,
 * a reader of the twin can see.
 */

function table(headers, rows) {
  if (!rows.length) return ''
  const head = `| ${headers.join(' | ')} |`
  const rule = `| ${headers.map(() => '---').join(' | ')} |`
  const body = rows.map((r) => `| ${r.map((c) => (c === null || c === undefined ? '' : String(c))).join(' | ')} |`)
  return [head, rule, ...body].join('\n')
}

function benchmarkTable(family, task = 'detect') {
  const bench = family?.benchmarks?.[task]
  if (!bench) return ''
  const rows = bench.rows.map((r) => [
    `\`${(family.prefixes?.[r.prefix_key] ?? family.prefix ?? '')}${r.size}\``,
    r.imgsz, r.map?.toFixed?.(1) ?? r.map, r.params_m ?? '',
  ])
  return `${table(['Checkpoint', 'Input (px)', bench.metric, 'Params (M)'], rows)}\n\n${bench.dataset}. Published on ${bench.source_url}`
}

function checkpointTable(family) {
  if (!family?.checkpoints?.length) return ''
  const rows = family.checkpoints.map((c) => [
    `\`${c.name}\``, c.imgsz ?? '', getTaskMeta(c.task).label, c.license ?? '',
  ])
  return table(['File', 'Input (px)', 'Task', 'Weights license'], rows)
}

function exportMatrix(family) {
  if (!family?.export) return ''
  const formats = getExportFormats()
  const rows = family.tasks.map((t) => [
    getTaskMeta(t).label,
    ...formats.map((f) => {
      const s = family.export[t]?.[f.key]
      return s === 'validated' || s === 'available' ? 'yes' : ''
    }),
  ])
  return `${table(['Task', ...formats.map((f) => f.label)], rows)}\n\nA "yes" means the export is supported. An empty cell means the exporter refuses that combination.`
}

function provenance(family) {
  const u = family?.upstream
  if (!u) return ''
  const lines = [
    'Check the license on the Hugging Face repository of the specific weights you download. That repository is authoritative and licenses are not always uniform across a family. This is a description of the licenses involved, not legal advice.',
    '',
    `- Original work: ${u.name}, ${u.org}`,
    `- Upstream license: ${u.license}`,
    `- Upstream source: ${u.code_url}`,
    `- LibreYOLO code: ${u.code_license ?? 'MIT'}`,
    family.weights_hosted === false
      ? `- Weights: ${u.license}, distributed by their authors. LibreYOLO does not host or mirror them.`
      : `- Weights: ${u.license}, republished at https://huggingface.co/LibreYOLO`,
  ]
  if (u.license_interpretation) lines.push(`- Interpretation: ${u.license_interpretation}`)
  return lines.join('\n')
}

function citation(family) {
  const u = family?.upstream
  if (!u?.bibtex) return ''
  const src = u.bibtex_source_url ? `\n\nCopied from ${u.bibtex_source_url}` : ''
  return '```bibtex\n' + u.bibtex + '\n```' + src
}

function codeTabs(snippets, name) {
  const group = snippets?.[name]
  if (!group?.length) return ''
  return group
    .map((t) => {
      const out = t.expect ? `\n\nOutput:\n\n\`\`\`\n${t.expect.replace(/\n$/, '')}\n\`\`\`` : ''
      return `**${t.label}**\n\n\`\`\`${t.language}\n${t.code.replace(/\n$/, '')}\n\`\`\`${out}`
    })
    .join('\n\n')
}

export function docToMarkdown(doc) {
  const [family] = getFamilies(doc.families || [])
  const snippets = doc.snippets || {}

  let body = expandSelfClosingTags(doc.content || '')

  const replace = (tag, fn) => {
    body = body.replace(new RegExp(`<${tag}([^>]*)>[\\s\\S]*?</${tag}>`, 'g'), (_m, attrs) => fn(attrs || ''))
  }

  replace('benchmark-table', (a) => benchmarkTable(family, /task="(\w+)"/.exec(a)?.[1] || 'detect'))
  replace('checkpoint-table', () => checkpointTable(family))
  replace('export-matrix', () => exportMatrix(family))
  replace('citation-block', () => citation(family))
  replace('code-tabs', (a) => codeTabs(snippets, /name="([\w-]+)"/.exec(a)?.[1]))
  replace('task-support', () => '')
  // The chart is an iframe; a twin gets the link, since the numbers it plots
  // are already in the benchmark table above it.
  replace('va-embed', () => (family?.va_embed?.scatter ? `Interactive chart: ${family.va_embed.scatter}` : ''))
  // provenance-box wraps author prose, so keep the prose and prepend the rows.
  body = body.replace(/<provenance-box>([\s\S]*?)<\/provenance-box>/g, (_m, inner) =>
    [provenance(family), inner.trim()].filter(Boolean).join('\n\n'))

  const header = [
    `# ${doc.title}`,
    '',
    doc.lead || doc.description || '',
    '',
    family
      ? `Tasks: ${family.tasks.map((t) => getTaskMeta(t).label).join(', ')}. Install: ${family.extra ? `pip install "libreyolo[${family.extra}]"` : 'pip install libreyolo'}.`
      : '',
    doc.last_verified ? `Verified against LibreYOLO v${doc.last_verified}.` : '',
  ].filter(Boolean).join('\n')

  return `${header}\n\n${body.trim()}\n`
}

export { registry }
