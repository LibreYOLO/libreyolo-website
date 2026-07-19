import { buildPageMetadata } from '@/i18n/metadata'

const metadataByLocale = {
  en: {
    title: 'LibreVLM | Open-Vocabulary Detection',
    description: 'LibreVLM wraps modern vision language models (Qwen3-VL, Florence-2, LFM2-VL, InternVL3, SmolVLM2, Kosmos-2) and exposes them as open-vocabulary object detectors behind the familiar LibreYOLO Results API.',
    keywords: ['LibreVLM', 'open vocabulary detection', 'vision language model', 'Qwen3-VL', 'Florence-2', 'LFM2-VL', 'InternVL3', 'SmolVLM2', 'Kosmos-2', 'zero-shot detection'],
  },
  zh: {
    title: 'LibreVLM | 开放词表检测',
    description: 'LibreVLM 封装 Qwen3-VL、Florence-2、LFM2-VL、InternVL3、SmolVLM2 和 Kosmos-2 等现代视觉语言模型，并通过熟悉的 LibreYOLO Results API 将其用于开放词表目标检测。',
    keywords: ['LibreVLM', '开放词表检测', '视觉语言模型', 'Qwen3-VL', 'Florence-2', 'LFM2-VL', 'InternVL3', 'SmolVLM2', 'Kosmos-2', '零样本检测'],
  },
}

export async function generateMetadata({ params }) {
  const { locale } = await params
  const copy = metadataByLocale[locale] ?? metadataByLocale.en
  return {
    ...buildPageMetadata({
      title: copy.title,
      description: copy.description,
      path: '/docs/librevlm',
      locale,
    }),
    keywords: copy.keywords,
  }
}

export default function LibreVLMDocsLayout({ children }) {
  return children
}
