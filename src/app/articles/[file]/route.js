import fs from 'fs'
import path from 'path'
import { getAllArticles } from '@/lib/articles'

// Raw-markdown endpoints for articles: /articles/<slug>.md (English source)
// and /articles/<slug>.zh.md (Chinese translation, where one exists). Gives
// LLMs and agents the article without the app shell; llms.txt points here.
// Dotted paths bypass the locale proxy (see src/proxy.js), so this
// locale-less route only ever receives *.md requests.
export const dynamic = 'force-static'
export const dynamicParams = false

const articlesDir = path.join(process.cwd(), 'content', 'articles')

export function generateStaticParams() {
  return getAllArticles().flatMap(({ slug }) => {
    const params = [{ file: `${slug}.md` }]
    if (fs.existsSync(path.join(articlesDir, `${slug}.zh.md`))) {
      params.push({ file: `${slug}.zh.md` })
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
