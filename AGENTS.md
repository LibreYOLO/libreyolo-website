# Agent Instructions

- Keep `src/app/sitemap.js` synchronized with the public site. Every canonical, indexable public page must appear in the sitemap, including each localized URL whose main content is translated and independently indexable.
- Do not add redirects, 404 pages, `noindex` pages, duplicate URLs, or locale fallbacks that canonicalize to another URL to the sitemap.
- Whenever a route or article is added, removed, renamed, translated, or changes canonical behavior, update the sitemap in the same change and verify the generated production sitemap.
- `src/app/llms.txt/route.js` serves /llms.txt, an LLM-facing index of the site. Its article list is generated, but its page and docs-version lists are static: update them whenever routes or docs versions change (same trigger as the sitemap).

- Don't use em dashes "—" in this website.
