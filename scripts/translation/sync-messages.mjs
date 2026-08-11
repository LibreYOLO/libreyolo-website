import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..')
const messagesDir = path.join(root, 'messages')
const englishPath = path.join(messagesDir, 'en.json')
const english = JSON.parse(fs.readFileSync(englishPath, 'utf8'))

function isObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function countLeaves(value) {
  if (isObject(value)) return Object.values(value).reduce((total, child) => total + countLeaves(child), 0)
  return 1
}

function mergeInEnglishOrder(template, current, stats) {
  if (!isObject(template)) return current === undefined ? template : current

  const source = isObject(current) ? current : {}
  const merged = {}
  for (const [key, englishValue] of Object.entries(template)) {
    if (!Object.hasOwn(source, key)) {
      stats.added += countLeaves(englishValue)
      merged[key] = englishValue
      continue
    }
    merged[key] = mergeInEnglishOrder(englishValue, source[key], stats)
  }
  return merged
}

const files = fs.readdirSync(messagesDir)
  .filter((file) => file.endsWith('.json') && file !== 'en.json')
  .sort()

for (const file of files) {
  const filePath = path.join(messagesDir, file)
  const current = JSON.parse(fs.readFileSync(filePath, 'utf8'))
  const stats = { added: 0 }
  const merged = mergeInEnglishOrder(english, current, stats)
  fs.writeFileSync(filePath, `${JSON.stringify(merged, null, 2)}\n`)
  console.log(`${file}: ${stats.added} additions`)
}
