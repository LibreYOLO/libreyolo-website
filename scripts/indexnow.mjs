// Tells IndexNow engines (Bing, Yandex, Seznam, Naver, ...) which URLs changed,
// so they recrawl them without waiting for their own schedule. Google does not
// use IndexNow; it reads the sitemap.
//
//   node scripts/indexnow.mjs                   # URLs whose sitemap lastmod is within 2 days
//   node scripts/indexnow.mjs --since 2026-09-20
//   node scripts/indexnow.mjs --all             # every URL in the sitemap
//   node scripts/indexnow.mjs --dry-run         # print what would be sent
//
// Run it after `vercel --prod` has finished, against the live sitemap, so the
// engines fetch pages that already exist. The key file is public by design
// (public/<key>.txt); IndexNow checks it to confirm we own the host.
const HOST = 'www.libreyolo.com'
const KEY = '28805d7bc50f0909dbb5cbbe4eecd5d9'
const ENDPOINT = 'https://api.indexnow.org/indexnow'
const BATCH = 10000

const args = process.argv.slice(2)
const dryRun = args.includes('--dry-run')
const all = args.includes('--all')
const sinceArg = args[args.indexOf('--since') + 1]
const since = args.includes('--since')
  ? new Date(sinceArg)
  : new Date(Date.now() - 2 * 24 * 60 * 60 * 1000)
if (Number.isNaN(since.getTime())) {
  console.error(`indexnow: bad --since date: ${sinceArg}`)
  process.exit(1)
}

const xml = await (await fetch(`https://${HOST}/sitemap.xml`)).text()
const entries = [...xml.matchAll(/<url>([\s\S]*?)<\/url>/g)].map(([, block]) => ({
  loc: block.match(/<loc>(.*?)<\/loc>/)?.[1],
  lastmod: block.match(/<lastmod>(.*?)<\/lastmod>/)?.[1],
}))
const urls = entries
  .filter(({ loc, lastmod }) => loc && (all || (lastmod && new Date(lastmod) >= since)))
  .map(({ loc }) => loc)

console.log(`indexnow: ${urls.length} of ${entries.length} sitemap URLs selected`)
if (dryRun || !urls.length) {
  if (dryRun) urls.forEach((url) => console.log(url))
  process.exit(0)
}

for (let i = 0; i < urls.length; i += BATCH) {
  const res = await fetch(ENDPOINT, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json; charset=utf-8' },
    body: JSON.stringify({
      host: HOST,
      key: KEY,
      keyLocation: `https://${HOST}/${KEY}.txt`,
      urlList: urls.slice(i, i + BATCH),
    }),
  })
  // 200 and 202 both mean accepted; 202 while the key is still being verified.
  console.log(`indexnow: batch ${i / BATCH + 1} -> HTTP ${res.status}`)
  if (!res.ok) {
    console.error(await res.text())
    process.exit(1)
  }
}
