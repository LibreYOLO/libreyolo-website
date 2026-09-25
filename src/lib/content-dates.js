import dates from '@/data/content-dates.json'

// Last real change of a content file, from git (see scripts/content-dates.mjs).
// `file` is repo-relative, e.g. `content/docs/models/yolox.ja.md`. Returns
// undefined for files the manifest does not know, so the sitemap omits
// <lastmod> rather than inventing one.
export function contentDate(file) {
  const date = dates[file]
  return date ? new Date(date) : undefined
}

// The file behind a page in a given locale: the `<name>.<locale>.md` twin for
// translations, `<name>.md` for English.
export function localeFile(base, locale) {
  return locale === 'en' ? `${base}.md` : `${base}.${locale}.md`
}
