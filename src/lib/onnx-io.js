import fs from 'fs'
import path from 'path'

/*
 * ONNX inputs and outputs, one JSON file per family under
 * src/data/docs/onnx-io/<family.key>.json.
 *
 * Read from disk on demand, the same way upstream metadata is: a new file is
 * picked up with no code change and no generated index. Pages are static, so the
 * read happens once per page at build time. The path keeps a literal prefix so
 * the bundler's file tracer narrows to this folder.
 *
 * A family with no file renders no block. A malformed file throws, which fails
 * the build rather than shipping a half-documented contract.
 */
export function getOnnxIo(familyKey) {
  if (!familyKey || !/^[\w-]+$/.test(familyKey)) return null
  const file = path.join(process.cwd(), 'src', 'data', 'docs', 'onnx-io', `${familyKey}.json`)
  if (!fs.existsSync(file)) return null
  return JSON.parse(fs.readFileSync(file, 'utf8'))
}

/* Values both the React block and the markdown twin derive the same way. */
export function scalingOf(input) {
  return input.scale && input.scale !== 1 ? input.scale : null
}

export function normalizationOf(input) {
  const hasMean = input.mean != null
  const hasStd = input.std != null
  if (!hasMean && !hasStd) return null
  return { mean: hasMean ? JSON.stringify(input.mean) : null, std: hasStd ? JSON.stringify(input.std) : null }
}
