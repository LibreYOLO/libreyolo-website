import fs from 'fs'
import path from 'path'

// The manifest is written after the family's SVG/HTML files have been built.
export function getModelDiagram(slug) {
  if (!/^[a-z0-9][a-z0-9-]*$/.test(slug)) return null
  const file = path.join(process.cwd(), 'public', 'diagrams', 'models', slug, 'manifest.json')
  if (fs.existsSync(file)) {
    const diagram = JSON.parse(fs.readFileSync(file, 'utf8'))
    if (!Array.isArray(diagram.views) || !diagram.views.length) return null
    return diagram
  }
  // Keep the approved first diagram available while its family expands.
  if (slug === 'yolov9') return {
    family: 'yolo9', slug, title: 'YOLOv9', default_view: 't-detect',
    views: [{ id: 't-detect', label: 'YOLO9-T', kind: 'concrete', task: 'detect', size: 't',
      html: '/diagrams/yolo9-t.html', svg: '/diagrams/yolo9-t.svg', input: '3 × 640 × 640' }],
  }
  return null
}
