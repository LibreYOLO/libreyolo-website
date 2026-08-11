import fs from 'fs'
import path from 'path'
import { getAllArticles } from '@/lib/articles'
import { routing } from '@/i18n/routing'

// Raw-markdown endpoints for articles: /articles/<slug>.md (English source)
// and /articles/<slug>.<locale>.md for each translation that exists. Gives
// LLMs and agents the article without the app shell; llms.txt points here.
// Dotted paths bypass the locale proxy (see src/proxy.js), so this
// locale-less route only ever receives *.md requests.
export const dynamic = 'force-static'
export const dynamicParams = false

const articlesDir = path.join(process.cwd(), 'content', 'articles')

export function generateStaticParams() {
  const translations = routing.locales.filter((l) => l !== routing.defaultLocale)
  return getAllArticles().flatMap(({ slug }) => {
    const params = [{ file: `${slug}.md` }]
    for (const locale of translations) {
      if (fs.existsSync(path.join(articlesDir, `${slug}.${locale}.md`))) {
        params.push({ file: `${slug}.${locale}.md` })
      }
    }
    return params
  })
}

export async function GET(request, { params }) {
  const { file } = await params
  const raw = fs.readFileSync(path.join(articlesDir, file), 'utf8')
  return new Response(raw, {
    headers: {
      'Content-Type': 'text/markdown; charset=utf-8',
      'Cache-Control': 'public, max-age=3600',
    },
  })
}
