import { getAllArticles } from '@/lib/articles'
import { getAllDocPages } from '@/lib/docs'
import { SITE_URL } from '@/i18n/metadata'
import { routing, localeLabels } from '@/i18n/routing'
import { GITHUB_URL, REDDIT_URL } from '@/lib/links'

// Derived from routing.locales so adding a locale never leaves this stale.
const LOCALE_SUMMARY = routing.locales
  .filter((locale) => locale !== routing.defaultLocale)
  .map((locale) => `/${locale} (${localeLabels[locale]})`)
  .join(', ')

// Serves /llms.txt, the emerging "sitemap for AI" convention: a markdown index
// that states what the site is and points LLMs at the high-signal pages. It
// mirrors the human-visible claims (no cloaking). Articles are read from disk
// so that list never drifts; the static page list below must be kept in sync
// with src/app/sitemap.js (see AGENTS.md).
export const dynamic = 'force-static'

export function GET() {
  /*
   * The docs index, grouped by section and read from disk. A hand-kept list at
   * this size goes stale on the first page added, and a stale llms.txt is worse
   * than none: it tells an agent a page exists when it does not.
   */
  const SECTION_TITLES = {
    start: 'Getting started', tasks: 'Tasks', models: 'Models',
    train: 'Training', predict: 'Prediction', export: 'Export and deploy',
    cli: 'Command line', reference: 'Reference',
  }
  const bySection = {}
  for (const page of getAllDocPages()) (bySection[page.section] ??= []).push(page)
  const docsLines = Object.keys(SECTION_TITLES)
    .filter((s) => bySection[s]?.length)
    .map((s) => {
      const lines = bySection[s]
        .map((p) => `- [${p.title}](${SITE_URL}${p.path})${p.description ? `: ${p.description}` : ''}`)
        .join('\n')
      return `### ${SECTION_TITLES[s]}\n${lines}`
    })
    .join('\n\n')

  const articleLines = getAllArticles()
    .map((a) => `- [${a.title}](${SITE_URL}/articles/${a.slug})${a.description ? `: ${a.description}` : ''}`)
    .join('\n')

  const body = `# LibreYOLO: MIT-Licensed Object Detection

> LibreYOLO is the MIT-licensed training and inference engine for state-of-the-art YOLO models. Built for commercial applications, free from AGPL restrictions.

libreyolo.com is the official site of the LibreYOLO open-source library: documentation, model zoo, dataset zoo, commercial licensing guide, and articles. Pages are also served under a locale path prefix (e.g. ${SITE_URL}/zh/models) for: ${LOCALE_SUMMARY}. Documentation is fully translated into Spanish, Chinese, Portuguese and Italian; the other locales are still being translated and serve English on the pages they have not reached yet.

## Get started
- Install: \`pip install libreyolo\`
- [Documentation](${SITE_URL}/docs): install, quickstart, every model family, every task, training, validation and export
- Every documentation page is also served as raw markdown by appending \`.md\` to its URL, for example ${SITE_URL}/docs/models/rf-detr.md
- [The whole documentation as one file](${SITE_URL}/llms-full.txt)

## Key pages
- [Home](${SITE_URL}/): project overview and quickstart
- [Model Zoo](${SITE_URL}/models): pre-trained weights with MIT licensing, browsable per family
- [Benchmarks](${SITE_URL}/benchmarks): measured COCO accuracy, RF100-VL transfer across 100 real-world datasets, and latency on real hardware
- [Dataset Zoo](${SITE_URL}/datasets): datasets for training and evaluating models, hosted on Hugging Face
- [Commercial Guide](${SITE_URL}/commercial): how to use LibreYOLO in proprietary, closed-source commercial applications under the MIT license
- [Research & Science](${SITE_URL}/science): native explainability, feature maps, and a codebase designed for discovery
- [Articles](${SITE_URL}/articles): tutorials, comparisons, and news about MIT-licensed object detection
- [LibreVLM](${SITE_URL}/docs/librevlm): vision-language model documentation
- [Experimental tasks](${SITE_URL}/docs/experimental): research previews and experimental model documentation
- [Cursor Hackathon track](${SITE_URL}/cursor-hackathon): setup tutorial and working examples for the LibreYOLO track

## Documentation versions
${SITE_URL}/docs always describes the current release. These single-page docs for
earlier releases stay reachable for anyone pinned to them, but they are frozen,
no longer updated, and canonicalised to ${SITE_URL}/docs. Prefer /docs.
- [v1.4.0 (frozen)](${SITE_URL}/docs/v1.4.0)
- [v1.3.1 (frozen)](${SITE_URL}/docs/v1.3.1)
- [v1.3.0 (frozen)](${SITE_URL}/docs/v1.3.0)
- [v1.2.0 (frozen)](${SITE_URL}/docs/v1.2.0)
- [v1.1.0 (frozen)](${SITE_URL}/docs/v1.1.0)

## Documentation
Generated from the docs tree, so this list cannot drift from what is published.
${docsLines}

## Articles
Every article is also available as raw markdown: append .md to its URL (or .zh.md for the Chinese translation).
${articleLines}

## Project links
- [LibreYOLO on GitHub](${GITHUB_URL}): source code, issues, releases
- [libreyolo on PyPI](https://pypi.org/project/libreyolo/): official package
- [LibreYOLO on Hugging Face](https://huggingface.co/LibreYOLO): model weights and datasets
- [Live demo](https://huggingface.co/spaces/LibreYOLO/libreyolo-demo): try the models in the browser
- [Vision Analysis benchmarks](https://www.visionanalysis.org/): accuracy and speed leaderboard for every LibreYOLO model (its own index: https://www.visionanalysis.org/llms.txt)
- [r/LibreYOLO](${REDDIT_URL}): community subreddit
`

  return new Response(body, {
    headers: {
      'Content-Type': 'text/plain; charset=utf-8',
      'Cache-Control': 'public, max-age=3600',
    },
  })
}
