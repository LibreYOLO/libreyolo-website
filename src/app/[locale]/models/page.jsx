'use client'

import { useRef, useEffect, useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useTranslations } from 'next-intl'
import { X, Maximize2 } from 'lucide-react'
import { Link } from '@/i18n/navigation'

// Same parkour clip, three tasks, in motion. Each carries its model list, shown
// under the video (so these tasks don't need their own section below).
const VIDEOS = [
  { src: '/showcase/parkour-detection.mp4', poster: '/showcase/parkour-detection-poster.jpg', label: 'Detection', models: ['YOLO9', 'RF-DETR', 'YOLOX', 'YOLO9-E2E', 'YOLO-NAS', 'D-FINE', 'DEIM', 'DEIMv2', 'RT-DETR', 'RT-DETRv2', 'RT-DETRv4', 'PicoDet', 'RTMDet', 'EdgeCrafter'] },
  { src: '/showcase/parkour-segmentation.mp4', poster: '/showcase/parkour-segmentation-poster.jpg', label: 'Segmentation', models: ['RF-DETR', 'EdgeCrafter', 'SAM', 'MobileSAM', 'SAM2'] },
  { src: '/showcase/parkour-pose.mp4', poster: '/showcase/parkour-pose-poster.jpg', label: 'Keypoints', models: ['RF-DETR', 'EdgeCrafter', 'YOLO-NAS'] },
]

// The remaining tasks, each with an example image + model list.
const TASKS = [
  { title: 'Classification', image: '/showcase/task-classification.jpg', models: ['ConvNeXt', 'EfficientNetV2', 'MobileNetV4', 'DINOv2', 'ResNet', 'CLIP'] },
  { title: 'Oriented boxes', image: '/showcase/task-obb.jpg', models: ['RF-DETR'] },
  { title: 'Point & counting', image: '/showcase/task-point.jpg', models: ['FOMO'] },
  { title: 'Gaze', image: '/showcase/task-gaze.gif', models: ['L2CS'] },
  { title: 'Vision-language', image: '/showcase/task-vlm.jpg', models: ['Florence-2', 'InternVL3', 'Kosmos-2', 'LFM2-VL', 'Qwen3-VL', 'SmolVLM2'] },
  { title: 'Depth', image: '/showcase/depth-reveal.gif', models: ['Depth Anything V2'] },
]

// Small corner hint so people (especially on touch) know a tile opens larger.
function ExpandBadge() {
  return (
    <span className="pointer-events-none absolute top-2 right-2 w-8 h-8 rounded-lg bg-black/40 text-white flex items-center justify-center opacity-70 group-hover:opacity-100 transition-opacity backdrop-blur-sm">
      <Maximize2 className="w-4 h-4" />
    </span>
  )
}

// Connection-friendly video: poster shows instantly, the clip is only fetched
// when scrolled into view (skipped entirely on Data Saver / 2G), pauses off-screen.
// Clicking opens the lightbox rather than navigating away.
function VideoTile({ src, poster, label, models, onOpen }) {
  const ref = useRef(null)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    el.muted = true

    const c = typeof navigator !== 'undefined' ? navigator.connection : null
    const saveData = !!(c && (c.saveData || /(^|\b)(slow-2g|2g)$/.test(c.effectiveType || '')))
    if (saveData) return // stay on the poster; the lightbox loads it on demand

    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          if (!el.getAttribute('src')) el.setAttribute('src', src)
          el.play().catch(() => {})
        } else {
          el.pause()
        }
      },
      { rootMargin: '250px' },
    )
    io.observe(el)
    return () => io.disconnect()
  }, [src])

  return (
    <div>
      <p className="mb-2 text-center text-sm font-medium text-surface-700 dark:text-surface-200">{label}</p>
      <button
        type="button"
        onClick={() => onOpen({ kind: 'video', src, poster, alt: label })}
        aria-label={`Enlarge ${label} demo`}
        className="group relative block w-full overflow-hidden rounded-xl border border-surface-200 dark:border-white/10 bg-surface-100 dark:bg-surface-900/60 cursor-zoom-in"
      >
        <video ref={ref} className="w-full aspect-video object-cover block" poster={poster} loop muted playsInline preload="none" />
        <ExpandBadge />
      </button>
      <p className="mt-2 text-center text-xs text-surface-500 dark:text-surface-400 leading-relaxed">
        {models.join(', ')}
      </p>
    </div>
  )
}

function TaskSection({ title, models, image, alt, onOpen }) {
  return (
    <div className="py-6 border-t border-surface-200 dark:border-white/10">
      <div className={`grid gap-5 items-center ${image ? 'md:grid-cols-[minmax(0,300px)_1fr]' : ''}`}>
        {image && (
          <button
            type="button"
            onClick={() => onOpen({ kind: 'image', src: image, alt: alt || title })}
            aria-label={`Enlarge ${title} example`}
            className="group relative block overflow-hidden rounded-xl border border-surface-200 dark:border-white/10 bg-surface-100 dark:bg-surface-900/60 cursor-zoom-in"
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={image}
              alt={alt || title}
              loading="lazy"
              className="w-full aspect-video object-cover block transition-transform duration-500 group-hover:scale-[1.02]"
            />
            <ExpandBadge />
          </button>
        )}
        <div>
          <h3 className="text-lg font-semibold text-surface-800 dark:text-white">{title}</h3>
          <p className="mt-1.5 text-sm text-surface-500 dark:text-surface-400 leading-relaxed">
            {models.join(', ')}
          </p>
        </div>
      </div>
    </div>
  )
}

// In-page zoom overlay. Closes on backdrop click, the X button, or Escape.
function Lightbox({ item, onClose }) {
  useEffect(() => {
    if (!item) return
    const onKey = (e) => {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', onKey)
    const prevOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', onKey)
      document.body.style.overflow = prevOverflow
    }
  }, [item, onClose])

  return (
    <AnimatePresence>
      {item && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          className="fixed inset-0 z-[100] flex items-center justify-center bg-black/85 backdrop-blur-sm p-4 sm:p-8"
          onClick={onClose}
          role="dialog"
          aria-modal="true"
        >
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="absolute top-4 right-4 w-11 h-11 rounded-full bg-white/10 hover:bg-white/25 text-white flex items-center justify-center backdrop-blur-sm transition-colors"
          >
            <X className="w-6 h-6" />
          </button>
          <motion.div
            initial={{ scale: 0.94, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.96, opacity: 0 }}
            transition={{ type: 'spring', damping: 26, stiffness: 300 }}
            className="relative"
            onClick={(e) => e.stopPropagation()}
          >
            {item.kind === 'video' ? (
              <video
                src={item.src}
                poster={item.poster}
                autoPlay
                loop
                muted
                playsInline
                controls
                className="max-w-[92vw] max-h-[86vh] rounded-xl block shadow-2xl"
              />
            ) : (
              /* eslint-disable-next-line @next/next/no-img-element */
              <img
                src={item.src}
                alt={item.alt}
                className="max-w-[92vw] max-h-[86vh] rounded-xl block object-contain shadow-2xl"
              />
            )}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

export default function Models() {
  const t = useTranslations('Models')
  const [lightbox, setLightbox] = useState(null)
  const closeLightbox = useCallback(() => setLightbox(null), [])

  return (
    <div className="pt-28 lg:pt-36 pb-20">
      <div className="max-w-6xl mx-auto px-5 md:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-10 md:mb-14"
        >
          <h1 className="text-2xl sm:text-3xl font-semibold text-surface-800 dark:text-white mb-2">
            {t('title')}
          </h1>
          <p className="text-sm text-surface-500 dark:text-surface-400">{t('subtitle')}</p>
        </motion.div>

        {/* Three tasks in motion, each with its model list underneath */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 md:gap-6">
          {VIDEOS.map((v) => (
            <VideoTile key={v.src} src={v.src} poster={v.poster} label={v.label} models={v.models} onOpen={setLightbox} />
          ))}
        </div>

        {/* Remaining tasks */}
        <div className="mt-16">
          <h2 className="text-center text-xs font-semibold uppercase tracking-widest text-surface-400 dark:text-surface-500 mb-2">
            {t('byTask')}
          </h2>
          <div>
            {TASKS.map((task) => (
              <TaskSection key={task.title} title={task.title} models={task.models} image={task.image} alt={task.title} onOpen={setLightbox} />
            ))}
          </div>
        </div>

        <div className="mt-16 flex items-center justify-center gap-8 text-sm">
          <a
            href="https://huggingface.co/LibreYOLO/models"
            target="_blank"
            rel="noopener noreferrer"
            className="text-surface-500 hover:text-libre-600 dark:hover:text-libre-400 transition-colors"
          >
            {t('weights')} &#8599;
          </a>
          <Link
            href="/docs"
            className="text-surface-500 hover:text-libre-600 dark:hover:text-libre-400 transition-colors"
          >
            {t('docs')} &#8594;
          </Link>
        </div>
      </div>

      <Lightbox item={lightbox} onClose={closeLightbox} />
    </div>
  )
}
