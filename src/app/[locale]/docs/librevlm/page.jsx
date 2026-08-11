'use client'
import { useLocale, useTranslations } from 'next-intl'

import { motion } from 'framer-motion'
import {
  ScanSearch, Sparkles, Terminal, Rocket, Layers, Tags, Crosshair,
  MessageSquare, Settings2, AlertTriangle, Eye, ShieldCheck,
} from 'lucide-react'
import {
  DocLayout, DocHero, SectionHeading, SubHeading, P, InlineCode, Divider,
  CodeBlock, DocTable, FeatureItem, Callout, SupportBadge, ExternalRef,
} from '@/components/DocsKit'

const sectionDefs = [
  { id: 'introduction', titleKey: 'introduction', icon: ScanSearch },
  { id: 'installation', titleKey: 'installation', icon: Terminal },
  { id: 'quickstart', titleKey: 'quickstart', icon: Rocket },
  { id: 'models', titleKey: 'models', icon: Layers },
  { id: 'vocabulary', titleKey: 'vocabulary', icon: Tags },
  { id: 'prediction', titleKey: 'prediction', icon: Crosshair },
  { id: 'examples', titleKey: 'examples', icon: Sparkles },
  { id: 'chat', titleKey: 'chat', icon: MessageSquare },
  { id: 'backends', titleKey: 'backends', icon: Settings2 },
  { id: 'limitations', titleKey: 'limitations', icon: AlertTriangle },
]

const relatedLinkDefs = [
  { href: '/docs', labelKey: 'coreDocs' },
  { href: '/docs/experimental', labelKey: 'experimentalTasks' },
  { href: '/models', labelKey: 'modelZoo' },
]

function LibreVLMPage() {
  const t = useTranslations('LibreVLM')
  const sections = sectionDefs.map(({ titleKey, ...section }) => ({ ...section, title: t(`sections.${titleKey}`) }))
  const relatedLinks = relatedLinkDefs.map(({ labelKey, ...link }) => ({ ...link, label: t(`related.${labelKey}`) }))
  const rich = {
    code: (chunks) => <InlineCode>{chunks}</InlineCode>,
    strong: (chunks) => <strong className="text-surface-800 dark:text-white">{chunks}</strong>,
  }
  return (
    <DocLayout
      sections={sections}
      eyebrow={t('eyebrow')}
      copyTitle={t('copyTitle')}
      relatedLinks={relatedLinks}
    >
      <DocHero
        badge={t('badge')}
        badgeTone="experimental"
        title={t('heroTitle')}
        accent={t('heroAccent')}
        lead={t('heroLead')}
      />

      {/* ───────── INTRODUCTION ───────── */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <SectionHeading id="introduction" icon={ScanSearch}>{t('sections.introduction')}</SectionHeading>
        <P>{t.rich('introduction.body', rich)}</P>
        <ul className="space-y-2 mb-4">
          <FeatureItem>{t.rich('introduction.features.openVocabulary', rich)}</FeatureItem>
          <FeatureItem>{t.rich('introduction.features.contract', rich)}</FeatureItem>
          <FeatureItem>{t.rich('introduction.features.backends', rich)}</FeatureItem>
          <FeatureItem>{t.rich('introduction.features.chat', rich)}</FeatureItem>
        </ul>

        <Callout icon={Sparkles} tone="libre" title={t('introduction.tierTitle')}>
          <p>{t.rich('introduction.tierBody', rich)}</p>
        </Callout>

        <Callout icon={AlertTriangle} tone="amber" title={t('introduction.devTitle')}>
          <p>{t.rich('introduction.devBody', {
            ...rich,
            link: (chunks) => <a href="#limitations" className="text-libre-600 dark:text-libre-400 hover:underline">{chunks}</a>,
          })}</p>
        </Callout>
      </motion.div>

      <Divider />

      {/* ───────── INSTALLATION ───────── */}
      <SectionHeading id="installation" icon={Terminal}>{t('sections.installation')}</SectionHeading>
      <P>{t.rich('installation.body', rich)}</P>
      <CodeBlock language="bash">{`pip install 'libreyolo[vlm]'`}</CodeBlock>
      <P>{t.rich('installation.weights', rich)}</P>

      <Divider />

      {/* ───────── QUICKSTART ───────── */}
      <SectionHeading id="quickstart" icon={Rocket}>{t('sections.quickstart')}</SectionHeading>
      <P>{t('quickstart.body')}</P>
      <CodeBlock language="python">{`from libreyolo import LibreVLM

# Qwen3-VL-4B by default; weights autodownload on first use
model = LibreVLM()

# The vocabulary is just words. Any words.
model.set_classes(["pink car", "wheel"])

result = model.predict("street.jpg")

print(result.boxes.xyxy)   # pixel [x1, y1, x2, y2]
print(result.boxes.cls)    # ids into ["pink car", "wheel"]
result.plot()              # same drawing helpers as any LibreYOLO model
result.save("out.jpg")`}</CodeBlock>
      <P>{t.rich('quickstart.after', rich)}</P>

      <Divider />

      {/* ───────── MODELS ───────── */}
      <SectionHeading id="models" icon={Layers}>{t('sections.models')}</SectionHeading>
      <P>{t.rich('models.body', rich)}</P>

      <DocTable
        headers={[t('models.table.family'), t('models.table.alias'), t('models.table.sizes'), t('models.table.license'), t('models.table.notes')]}
        rows={[
          [
            <strong key="q" className="text-surface-800 dark:text-white whitespace-nowrap">Qwen3-VL</strong>,
            <code key="qa" className="font-mono text-xs">qwen3-vl-2b / -4b / -8b</code>,
            '2B / 4B / 8B',
            'Apache-2.0',
            <span key="qn">{t('models.table.qwenNote')}</span>,
          ],
          [
            <strong key="l" className="text-surface-800 dark:text-white whitespace-nowrap">LFM2-VL</strong>,
            <code key="la" className="font-mono text-xs">lfm2-vl-450m / -1.6b</code>,
            '450M / 1.6B',
            'LFM Open License',
            <span key="ln">{t('models.table.lfmNote')}</span>,
          ],
          [
            <strong key="i" className="text-surface-800 dark:text-white whitespace-nowrap">InternVL3</strong>,
            <code key="ia" className="font-mono text-xs">internvl3-1b / -2b / -8b</code>,
            '1B / 2B / 8B',
            'Qwen License',
            <span key="in">{t('models.table.internNote')}</span>,
          ],
          [
            <strong key="f" className="text-surface-800 dark:text-white whitespace-nowrap">Florence-2</strong>,
            <code key="fa" className="font-mono text-xs">florence-2-base / -large</code>,
            '0.23B / 0.77B',
            'MIT',
            <span key="fn">{t.rich('models.table.florenceNote', rich)}</span>,
          ],
          [
            <strong key="s" className="text-surface-800 dark:text-white whitespace-nowrap">SmolVLM2</strong>,
            <code key="sa" className="font-mono text-xs">smolvlm2-500m / -2.2b</code>,
            '500M / 2.2B',
            'Apache-2.0',
            <span key="sn">{t('models.table.smolNote')}</span>,
          ],
          [
            <strong key="k" className="text-surface-800 dark:text-white whitespace-nowrap">Kosmos-2</strong>,
            <code key="ka" className="font-mono text-xs">kosmos-2</code>,
            '~1.6B',
            'MIT',
            <span key="kn">{t.rich('models.table.kosmosNote', rich)}</span>,
          ],
        ]}
      />

      <SubHeading>{t('models.choosing')}</SubHeading>
      <ul className="space-y-2 mb-4">
        <FeatureItem>{t.rich('models.choices.quality', rich)}</FeatureItem>
        <FeatureItem>{t.rich('models.choices.tight', rich)}</FeatureItem>
        <FeatureItem>{t.rich('models.choices.edge', rich)}</FeatureItem>
        <FeatureItem>{t.rich('models.choices.license', rich)}</FeatureItem>
      </ul>

      <Callout icon={ShieldCheck} tone="emerald" title={t('models.licensingTitle')}>
        <p>{t('models.licensingBody')}</p>
      </Callout>

      <Divider />

      {/* ───────── VOCABULARY ───────── */}
      <SectionHeading id="vocabulary" icon={Tags}>{t('sections.vocabulary')}</SectionHeading>
      <P>{t.rich('vocabulary.body', rich)}</P>
      <CodeBlock language="python">{`# Sticky and chainable
model = LibreVLM("qwen3-vl-2b").set_classes(["person", "dog", "cat"])

# Set it once at construction instead
model = LibreVLM("lfm2-vl-450m", names=["boat"], device="cpu")

# Re-set any time to change what you are looking for
model.set_classes(["a red car", "a blue truck"])`}</CodeBlock>
      <P>{t.rich('vocabulary.rules', rich)}</P>

      <Divider />

      {/* ───────── PREDICTION ───────── */}
      <SectionHeading id="prediction" icon={Crosshair}>{t('sections.prediction')}</SectionHeading>
      <P>{t.rich('prediction.body', rich)}</P>
      <CodeBlock language="python">{`result = model.predict(
    source="image.jpg",  # path | PIL | ndarray | URL | folder | video
    conf=0.25,           # see note below: scoring is synthetic
    classes=[0],         # optional: keep only these vocabulary ids
    max_det=300,
)`}</CodeBlock>

      <SubHeading>{t('prediction.returnShape')}</SubHeading>
      <P>{t.rich('prediction.returnBody', rich)}</P>
      <DocTable
        headers={[t('prediction.table.field'), t('prediction.table.shapeType'), t('prediction.table.meaning')]}
        rows={[
          [<InlineCode key="a">result.boxes.xyxy</InlineCode>, 'N x 4', t('prediction.table.xyxy')],
          [<InlineCode key="b">result.boxes.cls</InlineCode>, 'N', t('prediction.table.cls')],
          [<InlineCode key="c">result.boxes.conf</InlineCode>, 'N', t('prediction.table.conf')],
          [<InlineCode key="d">result.plot() / .save()</InlineCode>, '-', t('prediction.table.helpers')],
        ]}
      />
      <P>{t('prediction.parsing')}</P>

      <Divider />

      {/* ───────── EXAMPLES ───────── */}
      <SectionHeading id="examples" icon={Sparkles}>{t('sections.examples')}</SectionHeading>

      <SubHeading>{t('examples.coloredObject')}</SubHeading>
      <CodeBlock language="python">{`from libreyolo import LibreVLM

model = LibreVLM("qwen3-vl-4b")
model.set_classes(["red car"])

result = model.predict("parking_lot.jpg")
print(f"Found {len(result.boxes.cls)} red car(s)")
result.save("red_cars.jpg")`}</CodeBlock>

      <SubHeading>{t('examples.florence')}</SubHeading>
      <CodeBlock language="python">{`# Florence-2 is a purpose-built grounder: very tight pixel boxes.
model = LibreVLM("florence-2-large")
model.set_classes(["a red car", "license plate"])

result = model.predict("car.jpg")
result.plot()`}</CodeBlock>

      <SubHeading>{t('examples.filter')}</SubHeading>
      <CodeBlock language="python">{`model = LibreVLM("qwen3-vl-2b").set_classes(["person", "dog", "cat"])

# classes= filters the configured vocabulary by id
people_only = model.predict("street.jpg", classes=[0])`}</CodeBlock>

      <SubHeading>{t('examples.cpu')}</SubHeading>
      <CodeBlock language="python">{`from libreyolo import LibreVLM, SAMPLE_IMAGE

model = LibreVLM("lfm2-vl-450m", device="cpu")
# No set_classes() -> falls back to the COCO-80 vocabulary
result = model.predict(SAMPLE_IMAGE)
print(model.names[result.boxes.cls[0]])  # e.g. "person"`}</CodeBlock>

      <SubHeading>{t('examples.batches')}</SubHeading>
      <CodeBlock language="python">{`model = LibreVLM().set_classes(["forklift", "pallet"])

# A whole folder
for result in model.predict("warehouse_frames/", stream=True):
    result.save()

# A video file (frames are processed one at a time)
model.predict("warehouse.mp4", save=True)`}</CodeBlock>

      <Divider />

      {/* ───────── CHAT ───────── */}
      <SectionHeading id="chat" icon={MessageSquare}>{t('sections.chat')}</SectionHeading>
      <P>{t.rich('chat.body', rich)}</P>
      <CodeBlock language="python">{`model = LibreVLM("qwen3-vl-4b")

answer = model.chat("harbor.jpg", "How many boats are docked? Answer with a number.")
print(answer)`}</CodeBlock>
      <Callout icon={AlertTriangle} tone="amber">
        <p>{t.rich('chat.callout', rich)}</p>
      </Callout>

      <Divider />

      {/* ───────── BACKENDS ───────── */}
      <SectionHeading id="backends" icon={Settings2}>{t('sections.backends')}</SectionHeading>
      <P>{t.rich('backends.body', rich)}</P>
      <DocTable
        headers={[t('backends.table.family'), t('backends.table.prompting'), t('backends.table.coordinates'), 'chat()']}
        rows={[
          ['Qwen3-VL', t('backends.table.jsonPrompt'), t('backends.table.rescaled1000'), t('backends.table.yes')],
          ['LFM2-VL', t('backends.table.jsonPrompt'), t('backends.table.normalized01'), t('backends.table.yes')],
          ['SmolVLM2', t('backends.table.jsonPrompt'), t('backends.table.normalized01'), t('backends.table.yes')],
          ['InternVL3', t('backends.table.jsonPrompt'), t('backends.table.rescaled1000'), t('backends.table.yes')],
          ['Florence-2', t('backends.table.taskToken'), t('backends.table.nativePixels'), t('backends.table.no')],
          ['Kosmos-2', t('backends.table.groundingPrompt'), t('backends.table.normalizedRescaled'), t('backends.table.no')],
        ]}
      />
      <P>{t.rich('backends.options', rich)}</P>

      <Divider />

      {/* ───────── LIMITATIONS ───────── */}
      <SectionHeading id="limitations" icon={AlertTriangle}>{t('sections.limitations')}</SectionHeading>
      <P>{t('limitations.body')}</P>
      <ul className="space-y-2 mb-4">
        <FeatureItem>{t.rich('limitations.items.confidence', rich)}</FeatureItem>
        <FeatureItem>{t.rich('limitations.items.validation', rich)}</FeatureItem>
        <FeatureItem>{t.rich('limitations.items.training', rich)}</FeatureItem>
        <FeatureItem>{t.rich('limitations.items.tracking', rich)}</FeatureItem>
        <FeatureItem>{t.rich('limitations.items.sequential', rich)}</FeatureItem>
        <FeatureItem>{t.rich('limitations.items.python', rich)}</FeatureItem>
      </ul>
      <Callout icon={Eye} tone="libre" title={t('limitations.shinesTitle')}>
        <p>{t.rich('limitations.shinesBody', {
          link: (chunks) => <a href="/docs" className="text-libre-600 dark:text-libre-400 hover:underline">{chunks}</a>,
        })}</p>
      </Callout>

      <div className="mt-12 flex flex-wrap gap-3">
        <SupportBadge variant="experimental">{t('limitations.inferenceOnly')}</SupportBadge>
        <SupportBadge variant="experimental">{t('limitations.devTarget')}</SupportBadge>
        <span className="text-sm text-surface-500 dark:text-surface-400 self-center">
          {t.rich('limitations.source', {
            link: (chunks) => <ExternalRef href="https://github.com/LibreYOLO/libreyolo">{chunks}</ExternalRef>,
          })}
        </span>
      </div>
    </DocLayout>
  )
}

/* ─────────────────────────────────────────────────────────────────────────
   Chinese (zh-CN) content bundle for the LibreVLM docs page.
   APPEND this to src/app/[locale]/docs/librevlm/page.jsx and render
   <LibreVLMPageZh /> when locale === 'zh'.

   Reuses the existing imports already at the top of the file:
   - motion (framer-motion)
   - icons: ScanSearch, Sparkles, Terminal, Rocket, Layers, Tags, Crosshair,
     MessageSquare, Settings2, AlertTriangle, Eye, ShieldCheck
   - DocsKit: DocLayout, DocHero, SectionHeading, SubHeading, P, InlineCode,
     Divider, CodeBlock, DocTable, FeatureItem, Callout, SupportBadge, ExternalRef
   Do NOT redefine any of those here.
   ───────────────────────────────────────────────────────────────────────── */

const sectionsZh = [
  { id: 'introduction', title: '简介', icon: ScanSearch },
  { id: 'installation', title: '安装', icon: Terminal },
  { id: 'quickstart', title: '快速开始', icon: Rocket },
  { id: 'models', title: '支持的模型', icon: Layers },
  { id: 'vocabulary', title: '设置词表', icon: Tags },
  { id: 'prediction', title: '预测', icon: Crosshair },
  { id: 'examples', title: '示例', icon: Sparkles },
  { id: 'chat', title: '原始对话', icon: MessageSquare },
  { id: 'backends', title: '后端差异', icon: Settings2 },
  { id: 'limitations', title: '局限性', icon: AlertTriangle },
]

const relatedLinksZh = [
  { href: '/zh/docs', label: '核心文档' },
  { href: '/docs/experimental', label: '实验性任务' },
  { href: '/models', label: '模型库' },
]

function LibreVLMPageZh() {
  return (
    <DocLayout
      sections={sectionsZh}
      eyebrow="LibreVLM"
      copyTitle="LibreVLM 文档"
      relatedLinks={relatedLinksZh}
    >
      <DocHero
        badge="实验性层级"
        badgeTone="experimental"
        title="Libre"
        accent="VLM"
        lead="把视觉语言模型对准一张图像，递给它一组词，就能得到检测框。LibreVLM 让 Qwen3-VL、Florence-2 等模型变成开放词表的目标检测器，并使用与其他所有 LibreYOLO 模型完全相同的 Results API。"
      />

      {/* ───────── INTRODUCTION ───────── */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <SectionHeading id="introduction" icon={ScanSearch}>简介</SectionHeading>
        <P>
          传统检测器在检测头中固化了一份固定的类别列表。LibreVLM 抛弃了这个限制。它封装现代指令微调的视觉语言模型，
          提示它们输出边界框，解析生成的文本，并返回你在 YOLO9 和 RF-DETR 中已经用过的同一个{' '}
          <InlineCode>Results</InlineCode> 对象。类别列表只是你在运行时提供的一组词，因此新增一个类别毫无成本，
          而且是零样本生效。
        </P>
        <ul className="space-y-2 mb-4">
          <FeatureItem><strong className="text-surface-800 dark:text-white">开放词表。</strong> 检测 <InlineCode>"pink car"</InlineCode>、<InlineCode>"license plate"</InlineCode> 或 <InlineCode>"the small island"</InlineCode>，无需为它们训练任何检测头。</FeatureItem>
          <FeatureItem><strong className="text-surface-800 dark:text-white">一个工厂，一份契约。</strong> <InlineCode>LibreVLM(...)</InlineCode> 返回标准的 <InlineCode>Results</InlineCode>，含 <InlineCode>boxes.xyxy</InlineCode>、<InlineCode>boxes.cls</InlineCode>、<InlineCode>boxes.conf</InlineCode>，以及 <InlineCode>.plot()</InlineCode> 和 <InlineCode>.save()</InlineCode>。</FeatureItem>
          <FeatureItem><strong className="text-surface-800 dark:text-white">可替换的后端。</strong> 一个别名字符串背后是六个模型系列，从 230M 的 Florence-2 到 8B 的 Qwen3-VL。</FeatureItem>
          <FeatureItem><strong className="text-surface-800 dark:text-white">一个原始的逃生通道。</strong> 当你需要的不止是框时，<InlineCode>chat()</InlineCode> 提供自由形式的图像问答。</FeatureItem>
        </ul>

        <Callout icon={Sparkles} tone="libre" title="为什么单独分出一个层级（和一个页面）">
          <p>
            LibreVLM 被刻意排除在闭合词表的 <InlineCode>LibreYOLO(...)</InlineCode>{' '}
            工厂及其 <InlineCode>.pt</InlineCode> 注册表之外。这些模型由提示驱动、开放词表，
            并报告合成的置信度，因此遵循的是另一份契约。把它们作为独立的层级，
            可以让核心检测文档保持整洁，并对所测量的内容保持诚实。
          </p>
        </Callout>

        <Callout icon={AlertTriangle} tone="amber" title="处于 dev 分支">
          <p>
            LibreVLM 目前位于 <InlineCode>dev</InlineCode> 分支，计划在 v1.3 版本发布；它不属于 v1.2.0。
            它是一个纯 Python 的推理层级：尚无训练、验证、导出或 CLI 路径，且置信度分数是占位符。在此之上构建之前，请先阅读{' '}
            <a href="#limitations" className="text-libre-600 dark:text-libre-400 hover:underline">局限性</a>{' '}
            一节。
          </p>
        </Callout>
      </motion.div>

      <Divider />

      {/* ───────── INSTALLATION ───────── */}
      <SectionHeading id="installation" icon={Terminal}>安装</SectionHeading>
      <P>
        LibreVLM 位于可选的 <InlineCode>vlm</InlineCode> extra 之后。它会引入较新的{' '}
        <InlineCode>transformers</InlineCode> 以及一些处理器所需的辅助库。没有该 extra 时，
        导入某个 VLM 系列会抛出 <InlineCode>ImportError</InlineCode> 并将你指引到这里。
      </P>
      <CodeBlock language="bash">{`pip install 'libreyolo[vlm]'`}</CodeBlock>
      <P>
        权重在首次使用时从 Hugging Face Hub 下载到本地的{' '}
        <InlineCode>weights/</InlineCode> 文件夹。少数系列采用非 OSI 许可证，会在下载前记录一次性提示。
        较大的后端推荐使用 GPU，但每个模型也都能通过 <InlineCode>device="cpu"</InlineCode> 在 CPU 上运行。
      </P>

      <Divider />

      {/* ───────── QUICKSTART ───────── */}
      <SectionHeading id="quickstart" icon={Rocket}>快速开始</SectionHeading>
      <P>
        构造一个模型，声明你关心的词，然后预测。默认后端是 Qwen3-VL-4B，
        它是该层级中最强的检测器，并采用 Apache-2.0 许可证。
      </P>
      <CodeBlock language="python">{`from libreyolo import LibreVLM

# Qwen3-VL-4B by default; weights autodownload on first use
model = LibreVLM()

# The vocabulary is just words. Any words.
model.set_classes(["pink car", "wheel"])

result = model.predict("street.jpg")

print(result.boxes.xyxy)   # pixel [x1, y1, x2, y2]
print(result.boxes.cls)    # ids into ["pink car", "wheel"]
result.plot()              # same drawing helpers as any LibreYOLO model
result.save("out.jpg")`}</CodeBlock>
      <P>
        这就是整个流程。<InlineCode>predict()</InlineCode> 之后的一切都表现得像一个普通检测器，
        因此现有的可视化、裁剪和跟踪代码都能继续工作。
      </P>

      <Divider />

      {/* ───────── MODELS ───────── */}
      <SectionHeading id="models" icon={Layers}>支持的模型</SectionHeading>
      <P>
        通过传给 <InlineCode>LibreVLM(...)</InlineCode> 的别名来选择后端。仅给出系列名会解析为其默认尺寸。
        整体默认后端是{' '}
        <InlineCode>qwen3-vl-4b</InlineCode>。实践中最强的检测器是{' '}
        <strong className="text-surface-800 dark:text-white">Qwen3-VL</strong>、{' '}
        <strong className="text-surface-800 dark:text-white">LFM2-VL</strong> 和{' '}
        <strong className="text-surface-800 dark:text-white">Florence-2</strong>。
      </P>

      <DocTable
        headers={['系列', '别名', '尺寸（参数量）', '许可证', '说明']}
        rows={[
          [
            <strong key="q" className="text-surface-800 dark:text-white whitespace-nowrap">Qwen3-VL</strong>,
            <code key="qa" className="font-mono text-xs">qwen3-vl-2b / -4b / -8b</code>,
            '2B / 4B / 8B',
            'Apache-2.0',
            <span key="qn">默认且最强。推荐作为起点。</span>,
          ],
          [
            <strong key="l" className="text-surface-800 dark:text-white whitespace-nowrap">LFM2-VL</strong>,
            <code key="la" className="font-mono text-xs">lfm2-vl-450m / -1.6b</code>,
            '450M / 1.6B',
            'LFM Open License',
            <span key="ln">边缘尺寸，小型检测器表现意外出色。下载前有提示。</span>,
          ],
          [
            <strong key="i" className="text-surface-800 dark:text-white whitespace-nowrap">InternVL3</strong>,
            <code key="ia" className="font-mono text-xs">internvl3-1b / -2b / -8b</code>,
            '1B / 2B / 8B',
            'Qwen License',
            <span key="in">8B 定位效果好；小尺寸较弱。下载前有提示。</span>,
          ],
          [
            <strong key="f" className="text-surface-800 dark:text-white whitespace-nowrap">Florence-2</strong>,
            <code key="fa" className="font-mono text-xs">florence-2-base / -large</code>,
            '0.23B / 0.77B',
            'MIT',
            <span key="fn">专为定位打造的模型。框很紧致，无 <InlineCode>chat()</InlineCode>。</span>,
          ],
          [
            <strong key="s" className="text-surface-800 dark:text-white whitespace-nowrap">SmolVLM2</strong>,
            <code key="sa" className="font-mono text-xs">smolvlm2-500m / -2.2b</code>,
            '500M / 2.2B',
            'Apache-2.0',
            <span key="sn">小巧快速；检测能力较弱。适合快速试用。</span>,
          ],
          [
            <strong key="k" className="text-surface-800 dark:text-white whitespace-nowrap">Kosmos-2</strong>,
            <code key="ka" className="font-mono text-xs">kosmos-2</code>,
            '~1.6B',
            'MIT',
            <span key="kn">2023 年的定位模型。框较粗糙，无 <InlineCode>chat()</InlineCode>。</span>,
          ],
        ]}
      />

      <SubHeading>选择后端</SubHeading>
      <ul className="space-y-2 mb-4">
        <FeatureItem><strong className="text-surface-800 dark:text-white">最佳质量：</strong> <InlineCode>qwen3-vl-8b</InlineCode> 或 <InlineCode>qwen3-vl-4b</InlineCode>（默认）。</FeatureItem>
        <FeatureItem><strong className="text-surface-800 dark:text-white">紧致的框、占用小：</strong> <InlineCode>florence-2-large</InlineCode>。</FeatureItem>
        <FeatureItem><strong className="text-surface-800 dark:text-white">边缘 / CPU：</strong> <InlineCode>lfm2-vl-450m</InlineCode> 或 <InlineCode>smolvlm2-500m</InlineCode>。</FeatureItem>
        <FeatureItem><strong className="text-surface-800 dark:text-white">完全宽松的许可证：</strong> 任意尺寸的 Qwen3-VL、SmolVLM2、Florence-2 或 Kosmos-2。</FeatureItem>
      </ul>

      <Callout icon={ShieldCheck} tone="emerald" title="许可证">
        <p>
          Qwen3-VL 和 SmolVLM2 采用 Apache-2.0；Florence-2 和 Kosmos-2 采用 MIT。LFM2-VL 和 InternVL3
          采用非 OSI 许可证，会在首次下载前发出一次性提示，以便你为商业用途做出知情选择。
        </p>
      </Callout>

      <Divider />

      {/* ───────── VOCABULARY ───────── */}
      <SectionHeading id="vocabulary" icon={Tags}>设置词表</SectionHeading>
      <P>
        词表是开放词表检测的核心。用一组标签字符串调用{' '}
        <InlineCode>set_classes()</InlineCode>。它是持久的：会在之后每一次 <InlineCode>predict()</InlineCode> 和 <InlineCode>track()</InlineCode>{' '}
        调用中保留，直到你再次设置。它返回 <InlineCode>self</InlineCode>，因此可以链式调用。
      </P>
      <CodeBlock language="python">{`# Sticky and chainable
model = LibreVLM("qwen3-vl-2b").set_classes(["person", "dog", "cat"])

# Set it once at construction instead
model = LibreVLM("lfm2-vl-450m", names=["boat"], device="cpu")

# Re-set any time to change what you are looking for
model.set_classes(["a red car", "a blue truck"])`}</CodeBlock>
      <P>
        标签可以是任意短语。它们在不区分大小写时必须唯一，并且你必须传入一个列表，而不是单个字符串。
        如果你从不调用 <InlineCode>set_classes()</InlineCode>，模型会回退到 COCO-80 词表，这样即使是裸{' '}
        <InlineCode>predict()</InlineCode> 也能给出合理的结果。
      </P>

      <Divider />

      {/* ───────── PREDICTION ───────── */}
      <SectionHeading id="prediction" icon={Crosshair}>预测</SectionHeading>
      <P>
        <InlineCode>predict()</InlineCode>（以及等价的 <InlineCode>model(...)</InlineCode> 调用）
        接受与任何 LibreYOLO 检测器相同的输入类型：路径、PIL 图像、numpy 数组、URL、文件夹或视频。{' '}
        <InlineCode>stream=True</InlineCode> 和{' '}
        <InlineCode>track()</InlineCode> 也都能用。
      </P>
      <CodeBlock language="python">{`result = model.predict(
    source="image.jpg",  # path | PIL | ndarray | URL | folder | video
    conf=0.25,           # see note below: scoring is synthetic
    classes=[0],         # optional: keep only these vocabulary ids
    max_det=300,
)`}</CodeBlock>

      <SubHeading>返回结构</SubHeading>
      <P>
        你会得到标准的 <InlineCode>Results</InlineCode> 对象，与闭合词表检测器完全一致：
      </P>
      <DocTable
        headers={['字段', '形状 / 类型', '含义']}
        rows={[
          [<InlineCode key="a">result.boxes.xyxy</InlineCode>, 'N x 4', '像素框 [x1, y1, x2, y2]，缩放到原始图像尺寸。'],
          [<InlineCode key="b">result.boxes.cls</InlineCode>, 'N', '类别 id，索引到你的 set_classes() 词表。'],
          [<InlineCode key="c">result.boxes.conf</InlineCode>, 'N', '合成置信度：每个框都是 1.0（见“局限性”）。'],
          [<InlineCode key="d">result.plot() / .save()</InlineCode>, '-', '常用的绘制与保存辅助方法。'],
        ]}
      />
      <P>
        在底层，LibreVLM 会宽容地解析模型输出（处理 markdown 代码围栏、多余的散文、重复的框以及被截断的数组），
        把自由文本标签映射回你的类别 id，并丢弃任何不在你词表中的标签。正是最后这一步，
        让一个自由生成的模型表现得像一个闭集检测器。
      </P>

      <Divider />

      {/* ───────── EXAMPLES ───────── */}
      <SectionHeading id="examples" icon={Sparkles}>示例</SectionHeading>

      <SubHeading>检测特定颜色的物体</SubHeading>
      <CodeBlock language="python">{`from libreyolo import LibreVLM

model = LibreVLM("qwen3-vl-4b")
model.set_classes(["red car"])

result = model.predict("parking_lot.jpg")
print(f"Found {len(result.boxes.cls)} red car(s)")
result.save("red_cars.jpg")`}</CodeBlock>

      <SubHeading>用 Florence-2 获得紧致的框</SubHeading>
      <CodeBlock language="python">{`# Florence-2 is a purpose-built grounder: very tight pixel boxes.
model = LibreVLM("florence-2-large")
model.set_classes(["a red car", "license plate"])

result = model.predict("car.jpg")
result.plot()`}</CodeBlock>

      <SubHeading>动态过滤到单个类别</SubHeading>
      <CodeBlock language="python">{`model = LibreVLM("qwen3-vl-2b").set_classes(["person", "dog", "cat"])

# classes= filters the configured vocabulary by id
people_only = model.predict("street.jpg", classes=[0])`}</CodeBlock>

      <SubHeading>用内置示例图像在 CPU 上运行</SubHeading>
      <CodeBlock language="python">{`from libreyolo import LibreVLM, SAMPLE_IMAGE

model = LibreVLM("lfm2-vl-450m", device="cpu")
# No set_classes() -> falls back to the COCO-80 vocabulary
result = model.predict(SAMPLE_IMAGE)
print(model.names[result.boxes.cls[0]])  # e.g. "person"`}</CodeBlock>

      <SubHeading>批量、文件夹与视频</SubHeading>
      <CodeBlock language="python">{`model = LibreVLM().set_classes(["forklift", "pallet"])

# A whole folder
for result in model.predict("warehouse_frames/", stream=True):
    result.save()

# A video file (frames are processed one at a time)
model.predict("warehouse.mp4", save=True)`}</CodeBlock>

      <Divider />

      {/* ───────── CHAT ───────── */}
      <SectionHeading id="chat" icon={MessageSquare}>原始对话</SectionHeading>
      <P>
        有时你想要的是模型本身，而不是检测器。采用对话模板的系列暴露了{' '}
        <InlineCode>chat()</InlineCode>，它接受一张图像和一个自由形式的提示，并原样返回解码后的文本。
        可用于计数、生成描述或快速的视觉问答。
      </P>
      <CodeBlock language="python">{`model = LibreVLM("qwen3-vl-4b")

answer = model.chat("harbor.jpg", "How many boats are docked? Answer with a number.")
print(answer)`}</CodeBlock>
      <Callout icon={AlertTriangle} tone="amber">
        <p>
          <InlineCode>chat()</InlineCode> 在采用对话模板的系列上可用（Qwen3-VL、LFM2-VL、
          SmolVLM2、InternVL3）。Florence-2 和 Kosmos-2 是基于任务 token 的定位模型，会抛出{' '}
          <InlineCode>NotImplementedError</InlineCode>；请对它们使用 <InlineCode>predict()</InlineCode>。
        </p>
      </Callout>

      <Divider />

      {/* ───────── BACKENDS ───────── */}
      <SectionHeading id="backends" icon={Settings2}>后端差异</SectionHeading>
      <P>
        每个系列都返回同样的 <InlineCode>Results</InlineCode>，但抵达方式各不相同。你很少需要关心这一点，
        不过了解某些后端为何如此表现会有帮助。对话系列被提示输出一个 JSON 框数组；定位模型则使用专门的任务 token。
      </P>
      <DocTable
        headers={['系列', '提示方式', '坐标空间', 'chat()']}
        rows={[
          ['Qwen3-VL', 'JSON 框提示', '0 到 1000，重新缩放', '是'],
          ['LFM2-VL', 'JSON 框提示', '归一化 0 到 1', '是'],
          ['SmolVLM2', 'JSON 框提示', '归一化 0 到 1', '是'],
          ['InternVL3', 'JSON 框提示', '0 到 1000，重新缩放', '是'],
          ['Florence-2', '任务 token', '原生像素', '否'],
          ['Kosmos-2', 'Grounding 提示', '归一化，重新缩放', '否'],
        ]}
      />
      <P>
        对于对话系列，你可以用构造函数参数 <InlineCode>prompt=</InlineCode> 覆盖检测提示，并用{' '}
        <InlineCode>max_new_tokens=</InlineCode> 限制生成长度。设备和 dtype 会自动解析：CUDA 上为 bf16 或
        fp16，CPU 上为 fp32。
      </P>

      <Divider />

      {/* ───────── LIMITATIONS ───────── */}
      <SectionHeading id="limitations" icon={AlertTriangle}>局限性</SectionHeading>
      <P>
        LibreVLM 强大但尚不成熟。提前了解它的边界能避免日后意外。
      </P>
      <ul className="space-y-2 mb-4">
        <FeatureItem><strong className="text-surface-800 dark:text-white">合成置信度。</strong> 每个框的得分都是 1.0。因此 <InlineCode>conf=</InlineCode> 过滤表现为全有或全无，而非真正的阈值。</FeatureItem>
        <FeatureItem><strong className="text-surface-800 dark:text-white">无 mAP / 验证。</strong> <InlineCode>val()</InlineCode> 会抛出异常，因为合成分数会让 COCO mAP 产生误导。</FeatureItem>
        <FeatureItem><strong className="text-surface-800 dark:text-white">无训练或导出。</strong> <InlineCode>train()</InlineCode> 和 <InlineCode>export()</InlineCode> 会抛出异常。请在上游微调 VLM，然后加载得到的权重。</FeatureItem>
        <FeatureItem><strong className="text-surface-800 dark:text-white">跟踪能力受限。</strong> <InlineCode>track()</InlineCode> 可以运行，但统一的分数会让跟踪器的低置信度恢复阶段失效。</FeatureItem>
        <FeatureItem><strong className="text-surface-800 dark:text-white">一次一张图像。</strong> 在 v1 中生成是串行的，因此更大的 <InlineCode>batch=</InlineCode> 值不会带来加速。</FeatureItem>
        <FeatureItem><strong className="text-surface-800 dark:text-white">仅 Python API。</strong> <InlineCode>libreyolo</InlineCode> CLI 尚不能解析 VLM 别名。</FeatureItem>
      </ul>
      <Callout icon={Eye} tone="libre" title="它的优势所在">
        <p>
          当类别集合是开放式的、经常变化，或难以提前标注时，就使用 LibreVLM：快速原型、长尾或稀有类别，
          以及“用文字描述要找的东西”这类工作流。当你需要校准过的置信度、吞吐量或可部署的产物时，请按{' '}
          <a href="/zh/docs" className="text-libre-600 dark:text-libre-400 hover:underline">核心文档</a>{' '}
          训练闭合词表的 YOLO9 或 RF-DETR。
        </p>
      </Callout>

      <div className="mt-12 flex flex-wrap gap-3">
        <SupportBadge variant="experimental">仅推理</SupportBadge>
        <SupportBadge variant="experimental">dev 分支 / 面向 v1.3</SupportBadge>
        <span className="text-sm text-surface-500 dark:text-surface-400 self-center">
          源码见 <ExternalRef href="https://github.com/LibreYOLO/libreyolo">GitHub</ExternalRef>
        </span>
      </div>
    </DocLayout>
  )
}


export default function Page() {
  const locale = useLocale()
  if (locale === 'zh') return <LibreVLMPageZh />
  return <LibreVLMPage />
}
