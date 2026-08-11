'use client'
import { useLocale, useTranslations } from 'next-intl'

import { motion } from 'framer-motion'
import {
  FlaskConical, Tags, Rotate3d, PersonStanding, Layers2, Sparkles,
  AlertTriangle, GitBranch, Crosshair, GraduationCap, ShieldCheck,
} from 'lucide-react'
import {
  DocLayout, DocHero, SectionHeading, SubHeading, P, InlineCode, Divider,
  CodeBlock, DocTable, FeatureItem, Callout, SupportBadge, ExternalRef,
} from '@/components/DocsKit'

const sectionDefs = [
  { id: 'overview', titleKey: 'overview', icon: FlaskConical },
  { id: 'task-selection', titleKey: 'taskSelection', icon: GitBranch },
  { id: 'classification', titleKey: 'classification', icon: Tags },
  { id: 'obb', titleKey: 'obb', icon: Rotate3d },
  { id: 'pose', titleKey: 'pose', icon: PersonStanding },
  { id: 'small-object', titleKey: 'smallObject', icon: Crosshair },
  { id: 'lora', titleKey: 'lora', icon: Layers2 },
  { id: 'status', titleKey: 'status', icon: AlertTriangle },
]

const relatedLinkDefs = [
  { href: '/docs', labelKey: 'coreDocs' },
  { href: '/docs/librevlm', labelKey: 'librevlm' },
  { href: '/models', labelKey: 'modelZoo' },
]

function ExperimentalPage() {
  const t = useTranslations('Experimental')
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

      {/* ───────── OVERVIEW ───────── */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <SectionHeading id="overview" icon={FlaskConical}>{t('sections.overview')}</SectionHeading>
        <P>{t.rich('overview.body', rich)}</P>
        <ul className="space-y-2 mb-4">
          <FeatureItem>{t.rich('overview.features.classification', rich)}</FeatureItem>
          <FeatureItem>{t.rich('overview.features.obb', rich)}</FeatureItem>
          <FeatureItem>{t.rich('overview.features.pose', rich)}</FeatureItem>
          <FeatureItem>{t.rich('overview.features.smallObject', rich)}</FeatureItem>
          <FeatureItem>{t.rich('overview.features.lora', rich)}</FeatureItem>
        </ul>
        <Callout icon={AlertTriangle} tone="amber" title={t('overview.calloutTitle')}>
          <p>{t.rich('overview.calloutBody', {
            link: (chunks) => <a href="#status" className="text-libre-600 dark:text-libre-400 hover:underline">{chunks}</a>,
          })}</p>
        </Callout>
      </motion.div>

      <Divider />

      {/* ───────── TASK SELECTION ───────── */}
      <SectionHeading id="task-selection" icon={GitBranch}>{t('sections.taskSelection')}</SectionHeading>
      <P>{t('taskSelection.body')}</P>
      <DocTable
        headers={[t('taskSelection.table.priority'), t('taskSelection.table.mechanism'), t('taskSelection.table.example')]}
        rows={[
          ['1', t('taskSelection.table.explicit'), <InlineCode key="a">task="obb"</InlineCode>],
          ['2', t('taskSelection.table.metadata'), t('taskSelection.table.recorded')],
          ['3', t('taskSelection.table.suffix'), <span key="c"><InlineCode>-cls</InlineCode>, <InlineCode>-obb</InlineCode>, <InlineCode>-pose</InlineCode></span>],
          ['4', t('taskSelection.table.default'), 'detect'],
        ]}
      />
      <P>{t.rich('taskSelection.after', rich)}</P>
      <CodeBlock language="python">{`from libreyolo import LibreYOLO, LibreYOLO9, LibreRFDETR

# Start a task from scratch via the family class
m = LibreYOLO9(None, size="t", task="classify", nb_classes=10)

# Load a trained checkpoint via the unified factory (task auto-detected)
m = LibreYOLO("LibreYOLO9t-obb.pt")`}</CodeBlock>

      <Divider />

      {/* ───────── CLASSIFICATION ───────── */}
      <SectionHeading id="classification" icon={Tags}>{t('classification.title')}</SectionHeading>
      <div className="flex flex-wrap gap-2 mb-5">
        <SupportBadge variant="experimental">YOLO9: t, s, m, c</SupportBadge>
        <SupportBadge variant="experimental">RF-DETR: n, s, m, l</SupportBadge>
      </div>
      <P>{t('classification.body')}</P>

      <SubHeading>{t('classification.inferenceTitle')}</SubHeading>
      <P>{t.rich('classification.inferenceBody', rich)}</P>
      <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9t-cls.pt")
r = model.predict("cat.jpg")

print(r.probs.top1)        # class id of the argmax
print(r.probs.top1conf)    # its probability
print(r.probs.top5)        # [id, id, id, id, id]
print(model.names[r.probs.top1])  # human-readable label`}</CodeBlock>
      <DocTable
        headers={[t('commonTable.field'), t('commonTable.type'), t('commonTable.meaning')]}
        rows={[
          [<InlineCode key="a">probs.top1</InlineCode>, 'int', t('classification.table.top1')],
          [<InlineCode key="b">probs.top5</InlineCode>, 'list[int]', t('classification.table.top5')],
          [<InlineCode key="c">probs.top1conf</InlineCode>, 'float', t('classification.table.top1conf')],
          [<InlineCode key="d">probs.top5conf</InlineCode>, 'tensor', t('classification.table.top5conf')],
          [<InlineCode key="e">probs.data</InlineCode>, 'tensor', t('classification.table.data')],
        ]}
      />

      <SubHeading>{t('classification.trainingTitle')}</SubHeading>
      <P>{t.rich('classification.trainingBody', rich)}</P>
      <CodeBlock language="text" filename="dataset/">{`dataset/
  train/
    cat/   img001.jpg ...
    dog/   img104.jpg ...
  val/
    cat/   ...
    dog/   ...`}</CodeBlock>
      <P>{t.rich('classification.dataBody', rich)}</P>
      <CodeBlock language="python">{`from libreyolo import LibreYOLO9

model = LibreYOLO9(None, size="t", task="classify", nb_classes=10)
result = model.train(
    data="imagenette160",   # folder, .zip URL, or known name
    epochs=10, batch=64, imgsz=224,
    optimizer="adamw", lr0=1e-3,
)
# Validation reports metrics/accuracy_top1 and metrics/accuracy_top5`}</CodeBlock>
      <Callout icon={Sparkles} tone="emerald" title={t('classification.referenceTitle')}>
        <p>{t('classification.referenceBody')}</p>
      </Callout>

      <Divider />

      {/* ───────── OBB ───────── */}
      <SectionHeading id="obb" icon={Rotate3d}>{t('obb.title')}</SectionHeading>
      <div className="flex flex-wrap gap-2 mb-5">
        <SupportBadge variant="experimental">YOLO9: t, s, m, c</SupportBadge>
        <SupportBadge variant="experimental">RF-DETR: n, s, m, l</SupportBadge>
      </div>
      <P>{t('obb.body')}</P>

      <SubHeading>{t('obb.inferenceTitle')}</SubHeading>
      <P>{t.rich('obb.inferenceBody', rich)}</P>
      <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9t-obb.pt")
r = model.predict("aerial.jpg")

for i in range(len(r.obb.cls)):
    cx, cy, w, h, angle = r.obb.xywhr[i]  # angle in radians
    corners = r.obb.xyxyxyxy[i]           # 4 (x, y) corner points
    conf, cls = r.obb.conf[i], r.obb.cls[i]`}</CodeBlock>
      <DocTable
        headers={[t('commonTable.field'), t('commonTable.shape'), t('commonTable.meaning')]}
        rows={[
          [<InlineCode key="a">obb.xywhr</InlineCode>, 'N x 5', t('obb.table.xywhr')],
          [<InlineCode key="b">obb.xyxyxyxy</InlineCode>, 'N x 4 x 2', t('obb.table.corners')],
          [<InlineCode key="c">obb.conf</InlineCode>, 'N', t('obb.table.conf')],
          [<InlineCode key="d">obb.cls</InlineCode>, 'N', t('obb.table.cls')],
        ]}
      />

      <SubHeading>{t('obb.trainingTitle')}</SubHeading>
      <P>{t.rich('obb.trainingBody', rich)}</P>
      <CodeBlock language="text" filename="labels/aerial_001.txt">{`# class_id  x1 y1  x2 y2  x3 y3  x4 y4   (all normalized to [0, 1])
0  0.51 0.32  0.66 0.38  0.62 0.55  0.47 0.49
2  0.10 0.71  0.18 0.69  0.20 0.80  0.12 0.82`}</CodeBlock>
      <P>{t.rich('obb.warmStart', rich)}</P>
      <CodeBlock language="python">{`from libreyolo import LibreYOLO9

model = LibreYOLO9(None, size="t", task="obb")
# Warm-start the backbone from a same-family detect checkpoint
result = model.train(data="dota8.yaml", pretrained=True, epochs=100, imgsz=640)

# CLI equivalent
# libreyolo train model=LibreYOLO9t.pt data=dota8.yaml --task obb`}</CodeBlock>
      <P>{t('obb.validation')}</P>

      <Divider />

      {/* ───────── POSE ───────── */}
      <SectionHeading id="pose" icon={PersonStanding}>{t('pose.title')}</SectionHeading>
      <div className="flex flex-wrap gap-2 mb-5">
        <SupportBadge variant="wip">YOLO9 + RF-DETR: landing soon</SupportBadge>
        <SupportBadge variant="experimental">YOLO-NAS, EdgeCrafter: available</SupportBadge>
      </div>
      <P>{t('pose.body')}</P>

      <SubHeading>{t('pose.inferenceTitle')}</SubHeading>
      <P>{t.rich('pose.inferenceBody', rich)}</P>
      <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9t-pose.pt")
r = model.predict("athletes.jpg")

kp = r.keypoints
print(kp.xy.shape)   # (N, 17, 2) pixel coordinates
print(kp.conf)       # (N, 17)    per-keypoint visibility / confidence
print(kp.xyn)        # normalized coordinates
print(r.boxes.xyxy)  # person boxes still come along`}</CodeBlock>
      <DocTable
        headers={[t('commonTable.field'), t('commonTable.shape'), t('commonTable.meaning')]}
        rows={[
          [<InlineCode key="a">keypoints.xy</InlineCode>, 'N x K x 2', t('pose.table.xy')],
          [<InlineCode key="b">keypoints.xyn</InlineCode>, 'N x K x 2', t('pose.table.xyn')],
          [<InlineCode key="c">keypoints.conf</InlineCode>, 'N x K', t('pose.table.conf')],
          [<InlineCode key="d">keypoints.has_visible</InlineCode>, 'N x K', t('pose.table.visible')],
        ]}
      />

      <SubHeading>{t('pose.trainingTitle')}</SubHeading>
      <P>{t.rich('pose.trainingBody', rich)}</P>
      <CodeBlock language="yaml" filename="coco8-pose.yaml">{`path: coco8-pose
train: images/train
val: images/val
nc: 1
names:
  0: person
kpt_shape: [17, 3]
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]`}</CodeBlock>
      <CodeBlock language="python">{`from libreyolo import LibreYOLO9

# Warm-start from a detection checkpoint; the keypoint head is reinitialized
model = LibreYOLO9("LibreYOLO9t.pt", size="t", task="pose")
model.train(data="coco8-pose.yaml", epochs=100, imgsz=640)

# Validation reports OKS-based AP via the pose validator`}</CodeBlock>
      <Callout icon={AlertTriangle} tone="rose" title={t('pose.devTitle')}>
        <p>{t('pose.devBody')}</p>
      </Callout>

      <Divider />

      {/* ───────── SMALL-OBJECT ───────── */}
      <SectionHeading id="small-object" icon={Crosshair}>{t('smallObject.title')}</SectionHeading>
      <div className="flex flex-wrap gap-2 mb-5">
        <SupportBadge variant="experimental">YOLO9-P2: t, s</SupportBadge>
        <SupportBadge variant="experimental">VisDrone research preview</SupportBadge>
      </div>
      <P>{t.rich('smallObject.body', rich)}</P>
      <P>{t.rich('smallObject.resultsIntro', rich)}</P>
      <DocTable
        headers={[t('smallObject.table.model'), 'AP', 'AP50', 'AP_small']}
        rows={[
          [t('smallObject.table.control'), '0.123', '0.220', '0.047'],
          [t('smallObject.table.ab'), '0.138', '0.254', '0.070'],
          [<strong key="s">{t('smallObject.table.preview')}</strong>, '0.226', '0.385', '0.141'],
        ]}
      />
      <P className="text-sm">
        {t('smallObject.tableNote')}
      </P>

      <SubHeading>{t('smallObject.previewTitle')}</SubHeading>
      <P>{t.rich('smallObject.previewBody', {
        ...rich,
        link: (chunks) => <a href="https://huggingface.co/LibreYOLO/LibreYOLO9P2s-visdrone" target="_blank" rel="noopener noreferrer" className="text-libre-600 dark:text-libre-400 hover:underline">{chunks}</a>,
      })}</P>
      <CodeBlock language="python">{`from libreyolo import LibreYOLO

# Auto-downloads from the LibreYOLO Hugging Face org
model = LibreYOLO("LibreYOLO9P2s-visdrone.pt")

# Evaluate/predict at 768 - the resolution it was trained at
results = model.predict("aerial.jpg", imgsz=768, conf=0.25)`}</CodeBlock>
      <Callout icon={AlertTriangle} tone="amber" title={t('smallObject.licenseTitle')}>
        <p>{t.rich('smallObject.licenseBody', {
          strong: (chunks) => <strong>{chunks}</strong>,
        })}</p>
      </Callout>

      <SubHeading>{t('smallObject.whenTitle')}</SubHeading>
      <P>{t.rich('smallObject.whenBody', rich)}</P>

      <SubHeading>{t('smallObject.trainingTitle')}</SubHeading>
      <P>{t('smallObject.trainingBody')}</P>
      <CodeBlock language="python">{`from libreyolo import LibreYOLO9P2

model = LibreYOLO9P2(None, size="s")
model.train(
    data="/abs/path/tiny_objects.yaml",
    imgsz=768,                # resolution is the biggest lever for tiny objects
    lr0=0.005,                # the family default 0.01 diverges on transfer init
    mosaic_prob=0.0,          # mosaic tiling shrinks tiny objects below detectability
    mixup_prob=0.0,
    hsv_prob=1.0, flip_prob=0.5,
    max_labels=600,           # dense aerial frames exceed the default 100-box cap
    pretrained="LibreYOLO9s.pt",  # transfer init from stock YOLOv9
    epochs=60,
)`}</CodeBlock>

      <Divider />

      {/* ───────── LoRA ───────── */}
      <SectionHeading id="lora" icon={Layers2}>{t('lora.title')}</SectionHeading>
      <div className="flex flex-wrap gap-2 mb-5">
        <SupportBadge variant="experimental">RF-DETR: n, s, m, l</SupportBadge>
      </div>
      <P>{t('lora.body')}</P>

      <SubHeading>{t('lora.enablingTitle')}</SubHeading>
      <P>{t.rich('lora.enablingBody', rich)}</P>
      <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("rf-detr-nano.pth")   # sizes n, s, m, l
result = model.train(
    data="data.yaml",
    lora=True,        # DoRA on the frozen DINOv2 backbone
    epochs=100, batch_size=4, lr=1e-4,
)

# Resume: LoRA is auto-detected from the checkpoint, no need to repeat the flag
model.train(data="data.yaml", resume=True)`}</CodeBlock>
      <CodeBlock language="bash">{`# CLI equivalent
libreyolo train --model rf-detr-nano.pth --data data.yaml --lora`}</CodeBlock>

      <SubHeading>{t('lora.checkpointsTitle')}</SubHeading>
      <ul className="space-y-2 mb-4">
        <FeatureItem>{t('lora.items.checkpoints')}</FeatureItem>
        <FeatureItem>{t('lora.items.head')}</FeatureItem>
        <FeatureItem>{t.rich('lora.items.export', rich)}</FeatureItem>
        <FeatureItem>{t.rich('lora.items.only', rich)}</FeatureItem>
      </ul>
      <Callout icon={ShieldCheck} tone="emerald" title={t('lora.installTitle')}>
        <p>{t.rich('lora.installBody', rich)}</p>
      </Callout>

      <Divider />

      {/* ───────── STATUS ───────── */}
      <SectionHeading id="status" icon={AlertTriangle}>{t('sections.status')}</SectionHeading>
      <P>{t('status.body')}</P>
      <DocTable
        headers={[t('status.table.feature'), t('status.table.families'), t('status.table.state')]}
        rows={[
          [t('status.table.classification'), 'YOLO9, RF-DETR', <SupportBadge key="a" variant="experimental">{t('status.table.prOpen')}</SupportBadge>],
          [t('status.table.obb'), 'YOLO9, RF-DETR', <SupportBadge key="b" variant="experimental">{t('status.table.experimental')}</SupportBadge>],
          [t('status.table.pose'), 'YOLO9, RF-DETR', <SupportBadge key="c" variant="wip">{t('status.table.landingSoon')}</SupportBadge>],
          [t('status.table.pose'), 'YOLO-NAS, EdgeCrafter', <SupportBadge key="d" variant="experimental">{t('status.table.available')}</SupportBadge>],
          [t('status.table.smallObject'), 'YOLO9-P2', <SupportBadge key="f" variant="experimental">{t('status.table.researchPreview')}</SupportBadge>],
          ['LoRA / DoRA', 'RF-DETR', <SupportBadge key="e" variant="experimental">{t('status.table.reviewed')}</SupportBadge>],
        ]}
      />
      <Callout icon={Crosshair} tone="libre" title={t('status.stableTitle')}>
        <p>{t.rich('status.stableBody', {
          docs: (chunks) => <a href="/docs" className="text-libre-600 dark:text-libre-400 hover:underline">{chunks}</a>,
          vlm: (chunks) => <a href="/docs/librevlm" className="text-libre-600 dark:text-libre-400 hover:underline">{chunks}</a>,
        })}</p>
      </Callout>

      <div className="mt-12 flex flex-wrap gap-3 items-center">
        <GraduationCap className="w-5 h-5 text-surface-400" />
        <span className="text-sm text-surface-500 dark:text-surface-400">
          {t.rich('status.source', {
            link: (chunks) => <ExternalRef href="https://github.com/LibreYOLO/libreyolo">{chunks}</ExternalRef>,
          })}
        </span>
      </div>
    </DocLayout>
  )
}

/* ============================================================================
   zh-CN content bundle for src/app/[locale]/docs/experimental/page.jsx
   APPEND this to the END of that file. All imports it relies on
   (motion, lucide icons, DocsKit components) are already imported there.
   Render ExperimentalPageZh (no props) when locale === 'zh'.
   ============================================================================ */

const sectionsZh = [
  { id: 'overview', title: '概述', icon: FlaskConical },
  { id: 'task-selection', title: '选择任务', icon: GitBranch },
  { id: 'classification', title: '分类', icon: Tags },
  { id: 'obb', title: '旋转框 (OBB)', icon: Rotate3d },
  { id: 'pose', title: '关键点 / 姿态', icon: PersonStanding },
  { id: 'small-object', title: '小目标检测', icon: Crosshair },
  { id: 'lora', title: 'LoRA / DoRA', icon: Layers2 },
  { id: 'status', title: '稳定性', icon: AlertTriangle },
]

const relatedLinksZh = [
  { href: '/zh/docs', label: '核心文档' },
  { href: '/docs/librevlm', label: 'LibreVLM' },
  { href: '/models', label: '模型库' },
]

function ExperimentalPageZh() {
  return (
    <DocLayout
      sections={sectionsZh}
      eyebrow="实验性"
      copyTitle="LibreYOLO 实验性任务"
      relatedLinks={relatedLinksZh}
    >
      <DocHero
        badge="实验性任务"
        badgeTone="experimental"
        title="下一步是"
        accent="什么"
        lead="检测与分割路径是经过验证的核心。本页介绍我们正在其之上积极构建的新任务头与训练技巧：分类、旋转框、姿态，以及参数高效微调。"
      />

      {/* ───────── OVERVIEW ───────── */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <SectionHeading id="overview" icon={FlaskConical}>概述</SectionHeading>
        <P>
          LibreYOLO 是一个多任务框架：同一个模型家族可以搭配不同的任务头。除了经过验证的检测与分割路径外，
          两大旗舰家族 YOLO9 和 RF-DETR 还在陆续加入若干新任务。它们都接入相同的{' '}
          <InlineCode>LibreYOLO(...)</InlineCode> 工厂函数与相同的 <InlineCode>Results</InlineCode>{' '}
          容器，因此只要你熟悉核心 API，这些都只是小幅扩展。
        </P>
        <ul className="space-y-2 mb-4">
          <FeatureItem><strong className="text-surface-800 dark:text-white">分类</strong>，支持 YOLO9 和 RF-DETR。输出整图标签及 top-1 / top-5 概率。</FeatureItem>
          <FeatureItem><strong className="text-surface-800 dark:text-white">旋转边界框 (OBB)</strong>，支持 YOLO9 和 RF-DETR。为航拍与文档图像提供带旋转角的检测框。</FeatureItem>
          <FeatureItem><strong className="text-surface-800 dark:text-white">关键点 / 姿态</strong>，支持 YOLO9 和 RF-DETR。COCO-17 人体关键点。</FeatureItem>
          <FeatureItem><strong className="text-surface-800 dark:text-white">小目标检测</strong>：YOLO9-P2，为 YOLOv9 增加 stride-4 检测尺度，面向航拍/无人机图像中 4-16 像素的目标，并提供 VisDrone 研究预览权重。</FeatureItem>
          <FeatureItem><strong className="text-surface-800 dark:text-white">LoRA / DoRA</strong> 微调，支持 RF-DETR。以极少的显存适配 Transformer 主干。</FeatureItem>
        </ul>
        <Callout icon={AlertTriangle} tone="amber" title="请先阅读">
          <p>
            本页所有内容均为实验性，其中部分仍在功能分支上开发中。在被提升进经过验证的核心之前，API、默认值与标签格式都可能发生变化。<a href="#status" className="text-libre-600 dark:text-libre-400 hover:underline">稳定性</a>{' '}
            一节准确记录了各项功能当前所处的阶段。
          </p>
        </Callout>
      </motion.div>

      <Divider />

      {/* ───────── TASK SELECTION ───────── */}
      <SectionHeading id="task-selection" icon={GitBranch}>选择任务</SectionHeading>
      <P>
        每个家族默认执行检测。你可以通过以下三种方式之一选择其他任务，并按以下优先级顺序解析：
      </P>
      <DocTable
        headers={['优先级', '机制', '示例']}
        rows={[
          ['1', '显式参数', <InlineCode key="a">task="obb"</InlineCode>],
          ['2', '检查点元数据', '记录在已训练 .pt 文件中的 task'],
          ['3', '文件名后缀', <span key="c"><InlineCode>-cls</InlineCode>, <InlineCode>-obb</InlineCode>, <InlineCode>-pose</InlineCode></span>],
          ['4', '家族默认值', 'detect'],
        ]}
      />
      <P>
        由于公开的 <InlineCode>LibreYOLO(...)</InlineCode> 工厂函数需要一个真实的权重文件，
        从零开始启动这些任务最简洁的方式是直接构造家族类并传入 <InlineCode>task=</InlineCode>。
        已训练的检查点可通过统一工厂函数加载，并自动检测其任务。
      </P>
      <CodeBlock language="python">{`from libreyolo import LibreYOLO, LibreYOLO9, LibreRFDETR

# Start a task from scratch via the family class
m = LibreYOLO9(None, size="t", task="classify", nb_classes=10)

# Load a trained checkpoint via the unified factory (task auto-detected)
m = LibreYOLO("LibreYOLO9t-obb.pt")`}</CodeBlock>

      <Divider />

      {/* ───────── CLASSIFICATION ───────── */}
      <SectionHeading id="classification" icon={Tags}>图像分类</SectionHeading>
      <div className="flex flex-wrap gap-2 mb-5">
        <SupportBadge variant="experimental">YOLO9: t, s, m, c</SupportBadge>
        <SupportBadge variant="experimental">RF-DETR: n, s, m, l</SupportBadge>
      </div>
      <P>
        分类为整张图像给出单一标签。YOLO9 保留其主干并加装一个轻量分类头；RF-DETR 复用其 DINOv2 编码器
        并添加一个池化线性头。两者均在 224×224 分辨率下运行。
      </P>

      <SubHeading>推理与 Probs 结果</SubHeading>
      <P>
        预测返回一个 <InlineCode>Results</InlineCode> 对象，其 <InlineCode>probs</InlineCode>{' '}
        字段携带各类别的 softmax 概率。
      </P>
      <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9t-cls.pt")
r = model.predict("cat.jpg")

print(r.probs.top1)        # class id of the argmax
print(r.probs.top1conf)    # its probability
print(r.probs.top5)        # [id, id, id, id, id]
print(model.names[r.probs.top1])  # human-readable label`}</CodeBlock>
      <DocTable
        headers={['字段', '类型', '含义']}
        rows={[
          [<InlineCode key="a">probs.top1</InlineCode>, 'int', 'argmax 的类别 id。'],
          [<InlineCode key="b">probs.top5</InlineCode>, 'list[int]', '按降序排列的 top-5 类别 id。'],
          [<InlineCode key="c">probs.top1conf</InlineCode>, 'float', 'top-1 类别的概率。'],
          [<InlineCode key="d">probs.top5conf</InlineCode>, 'tensor', 'top-5 类别的概率。'],
          [<InlineCode key="e">probs.data</InlineCode>, 'tensor', '完整的 softmax 向量。'],
        ]}
      />

      <SubHeading>数据集格式与训练</SubHeading>
      <P>
        分类使用 <strong className="text-surface-800 dark:text-white">ImageFolder</strong>{' '}
        布局，而非 YAML。类别名称取自排序后的子文件夹名，以 train 划分为准。
      </P>
      <CodeBlock language="text" filename="dataset/">{`dataset/
  train/
    cat/   img001.jpg ...
    dog/   img104.jpg ...
  val/
    cat/   ...
    dog/   ...`}</CodeBlock>
      <P>
        <InlineCode>data=</InlineCode> 参数接受一个文件夹、一个 <InlineCode>.zip</InlineCode> URL，
        或一个已知的自动下载名称（<InlineCode>imagenette160</InlineCode> 与{' '}
        <InlineCode>imagenet10</InlineCode>）。分类头会自动重建以匹配数据集的类别数。
      </P>
      <CodeBlock language="python">{`from libreyolo import LibreYOLO9

model = LibreYOLO9(None, size="t", task="classify", nb_classes=10)
result = model.train(
    data="imagenette160",   # folder, .zip URL, or known name
    epochs=10, batch=64, imgsz=224,
    optimizer="adamw", lr0=1e-3,
)
# Validation reports metrics/accuracy_top1 and metrics/accuracy_top5`}</CodeBlock>
      <Callout icon={Sparkles} tone="emerald" title="参考运行结果">
        <p>
          开发期间的快速验证：YOLO9-t 在 imagenette160 上达到 top-1 0.79 / top-5 0.975（10 个 epoch），
          RF-DETR-n 达到 top-1 0.69 / top-5 0.96（6 个 epoch）。RF-DETR 在首次运行时如能联网获取其 DINOv2
          主干会更好；离线时则回退到随机初始化。
        </p>
      </Callout>

      <Divider />

      {/* ───────── OBB ───────── */}
      <SectionHeading id="obb" icon={Rotate3d}>旋转边界框 (OBB)</SectionHeading>
      <div className="flex flex-wrap gap-2 mb-5">
        <SupportBadge variant="experimental">YOLO9: t, s, m, c</SupportBadge>
        <SupportBadge variant="experimental">RF-DETR: n, s, m, l</SupportBadge>
      </div>
      <P>
        旋转框带有一个旋转角，这正是航拍图像、文档以及密集排布场景所需要的。YOLO9 在其检测头上增加了一个
        角度分支；RF-DETR 则在其解码器中加入了一个可学习的角度嵌入。
      </P>

      <SubHeading>推理与 OBB 结果</SubHeading>
      <P>
        Results 暴露一个 <InlineCode>obb</InlineCode> 字段。角度以{' '}
        <strong className="text-surface-800 dark:text-white">弧度</strong>为单位。
      </P>
      <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9t-obb.pt")
r = model.predict("aerial.jpg")

for i in range(len(r.obb.cls)):
    cx, cy, w, h, angle = r.obb.xywhr[i]  # angle in radians
    corners = r.obb.xyxyxyxy[i]           # 4 (x, y) corner points
    conf, cls = r.obb.conf[i], r.obb.cls[i]`}</CodeBlock>
      <DocTable
        headers={['字段', '形状', '含义']}
        rows={[
          [<InlineCode key="a">obb.xywhr</InlineCode>, 'N x 5', '[cx, cy, w, h, angle]，angle 以弧度为单位。'],
          [<InlineCode key="b">obb.xyxyxyxy</InlineCode>, 'N x 4 x 2', '每个框的四个角点。'],
          [<InlineCode key="c">obb.conf</InlineCode>, 'N', '每个框的置信度。'],
          [<InlineCode key="d">obb.cls</InlineCode>, 'N', '每个框的类别 id。'],
        ]}
      />

      <SubHeading>数据集格式与训练</SubHeading>
      <P>
        OBB 使用标准的检测式数据 YAML，但标签是 YOLO-OBB 文本文件，每行{' '}
        <strong className="text-surface-800 dark:text-white">恰好九个字段</strong>：一个类别 id，
        后跟四个归一化角点。角度由角点推导得出，并不存储。
      </P>
      <CodeBlock language="text" filename="labels/aerial_001.txt">{`# class_id  x1 y1  x2 y2  x3 y3  x4 y4   (all normalized to [0, 1])
0  0.51 0.32  0.66 0.38  0.62 0.55  0.47 0.49
2  0.10 0.71  0.18 0.69  0.20 0.80  0.12 0.82`}</CodeBlock>
      <P>
        普通的检测检查点无法直接加载到 OBB 模型中。从检测转到 OBB 仅允许作为训练时的热启动：传入{' '}
        <InlineCode>pretrained=True</InlineCode>（YOLO9）或 RF-DETR 上的显式迁移标志。在角点感知增强落地之前，
        OBB 会禁用 Mosaic 与 mixup，且不支持分块推理。
      </P>
      <CodeBlock language="python">{`from libreyolo import LibreYOLO9

model = LibreYOLO9(None, size="t", task="obb")
# Warm-start the backbone from a same-family detect checkpoint
result = model.train(data="dota8.yaml", pretrained=True, epochs=100, imgsz=640)

# CLI equivalent
# libreyolo train model=LibreYOLO9t.pt data=dota8.yaml --task obb`}</CodeBlock>
      <P>
        验证使用旋转 IoU 的 AP，在 OBB 指标组下以 mAP50 与 mAP50-95 报告。
      </P>

      <Divider />

      {/* ───────── POSE ───────── */}
      <SectionHeading id="pose" icon={PersonStanding}>关键点 / 姿态</SectionHeading>
      <div className="flex flex-wrap gap-2 mb-5">
        <SupportBadge variant="wip">YOLO9 + RF-DETR：即将上线</SupportBadge>
        <SupportBadge variant="experimental">YOLO-NAS、EdgeCrafter：已可用</SupportBadge>
      </div>
      <P>
        姿态估计为每个检测到的实例预测关键点。默认布局为 COCO-17 人体关键点。YOLO9 和 RF-DETR 姿态在首个版本中
        仅支持人体单类别；YOLO-NAS 与 EdgeCrafter 姿态在代码库中已经可用。
      </P>

      <SubHeading>推理与 Keypoints 结果</SubHeading>
      <P>
        Results 暴露一个形状为 <InlineCode>(N, K, 3)</InlineCode> 的 <InlineCode>keypoints</InlineCode>{' '}
        字段，其中最后一个通道为可见性或置信度，坐标为原始图像的像素坐标。
      </P>
      <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9t-pose.pt")
r = model.predict("athletes.jpg")

kp = r.keypoints
print(kp.xy.shape)   # (N, 17, 2) pixel coordinates
print(kp.conf)       # (N, 17)    per-keypoint visibility / confidence
print(kp.xyn)        # normalized coordinates
print(r.boxes.xyxy)  # person boxes still come along`}</CodeBlock>
      <DocTable
        headers={['字段', '形状', '含义']}
        rows={[
          [<InlineCode key="a">keypoints.xy</InlineCode>, 'N x K x 2', '关键点像素坐标。'],
          [<InlineCode key="b">keypoints.xyn</InlineCode>, 'N x K x 2', '归一化关键点坐标。'],
          [<InlineCode key="c">keypoints.conf</InlineCode>, 'N x K', '每个关键点的可见性 / 置信度。'],
          [<InlineCode key="d">keypoints.has_visible</InlineCode>, 'N x K', '布尔型可见性掩码。'],
        ]}
      />

      <SubHeading>数据集格式与训练</SubHeading>
      <P>
        姿态使用的数据 YAML 必须声明 <InlineCode>kpt_shape: [K, 2|3]</InlineCode>，若需水平翻转增强还需声明{' '}
        <InlineCode>flip_idx</InlineCode>。标签为 YOLO-pose 文本行：一个类别 id、一个归一化框，然后是{' '}
        <InlineCode>K</InlineCode> 个关键点三元组 <InlineCode>(x, y, v)</InlineCode>，其中可见性{' '}
        <InlineCode>v</InlineCode> 取值于 <InlineCode>{'{0, 1, 2}'}</InlineCode>。
      </P>
      <CodeBlock language="yaml" filename="coco8-pose.yaml">{`path: coco8-pose
train: images/train
val: images/val
nc: 1
names:
  0: person
kpt_shape: [17, 3]
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]`}</CodeBlock>
      <CodeBlock language="python">{`from libreyolo import LibreYOLO9

# Warm-start from a detection checkpoint; the keypoint head is reinitialized
model = LibreYOLO9("LibreYOLO9t.pt", size="t", task="pose")
model.train(data="coco8-pose.yaml", epochs=100, imgsz=640)

# Validation reports OKS-based AP via the pose validator`}</CodeBlock>
      <Callout icon={AlertTriangle} tone="rose" title="正在积极开发中">
        <p>
          YOLO9 和 RF-DETR 姿态尚在功能分支上，尚未合并；请将上述 API 视为预期约定而非冻结契约。
          YOLO-NAS 姿态权重是从上游链接而非镜像，需要手动准备。
        </p>
      </Callout>

      <Divider />

      {/* ───────── SMALL-OBJECT ───────── */}
      <SectionHeading id="small-object" icon={Crosshair}>小目标检测（YOLO9-P2）</SectionHeading>
      <div className="flex flex-wrap gap-2 mb-5">
        <SupportBadge variant="experimental">YOLO9-P2: t, s</SupportBadge>
        <SupportBadge variant="experimental">VisDrone 研究预览</SupportBadge>
      </div>
      <P>
        YOLO9-P2 是在 YOLOv9 上增加了第四个检测尺度（<strong className="text-surface-800 dark:text-white">stride 4</strong>）的变体。
        标准 YOLOv9 在 stride 8/16/32 上检测，约 16 像素以下的目标会落在其最细网格之下；P2
        检测头正好覆盖航拍和无人机画面中占主导的 4-16 像素范围。
      </P>
      <P>
        在 VisDrone 上的受控 A/B 实验中（相同配方、相同分辨率、相同初始化，唯一差别是 P2 头），
        小目标 AP 比同级标准 YOLOv9 提升了{' '}
        <strong className="text-surface-800 dark:text-white">+49%</strong>。再叠加更高的训练分辨率与更大的
        s 尺寸后，小目标 AP 在整个项目中大约翻倍：
      </P>
      <DocTable
        headers={['模型', 'AP', 'AP50', 'AP_small']}
        rows={[
          ['标准 YOLO9-t @640（对照）', '0.123', '0.220', '0.047'],
          ['YOLO9-P2-t @640（同配方 A/B）', '0.138', '0.254', '0.070'],
          [<strong key="s">YOLO9-P2-s @768（已发布预览）</strong>, '0.226', '0.385', '0.141'],
        ]}
      />
      <P className="text-sm">
        VisDrone2019-DET 验证集（548 张），pycocotools，单一随机种子；请将 &plusmn;1 个点视为噪声。
      </P>

      <SubHeading>VisDrone 研究预览权重</SubHeading>
      <P>
        已发布的训练权重：{' '}
        <a href="https://huggingface.co/LibreYOLO/LibreYOLO9P2s-visdrone" target="_blank" rel="noopener noreferrer" className="text-libre-600 dark:text-libre-400 hover:underline">LibreYOLO9P2s-visdrone</a>。
        该模型家族已合并到 <InlineCode>dev</InlineCode>，但尚未进入 PyPI 发行版；在下个版本发布前请从源码安装。
      </P>
      <CodeBlock language="python">{`from libreyolo import LibreYOLO

# Auto-downloads from the LibreYOLO Hugging Face org
model = LibreYOLO("LibreYOLO9P2s-visdrone.pt")

# Evaluate/predict at 768 - the resolution it was trained at
results = model.predict("aerial.jpg", imgsz=768, conf=0.25)`}</CodeBlock>
      <Callout icon={AlertTriangle} tone="amber" title="非商业许可证">
        <p>
          该预览权重在 VisDrone2019-DET（天津大学 AISKYEYE 团队）上训练，数据集许可证为
          CC BY-NC-SA 3.0：<strong>仅限非商业用途</strong>，与 LibreYOLO 的 MIT 代码和 COCO
          默认权重不同。它检测的是 VisDrone 的 10 个航拍类别，而非 COCO。Hugging Face
          模型卡附带完整训练配方、逐 epoch 指标以及净室实现的数据集转换脚本，方便复现或在自己的数据上重训。
        </p>
      </Callout>

      <SubHeading>何时（不）使用它</SubHeading>
      <P>
        让架构匹配场景。在类 COCO 数据上（&quot;小目标&quot;指 16-32 像素），P2 头并
        <strong className="text-surface-800 dark:text-white">不会</strong>带来提升，
        那里标准 YOLOv9 是更好的选择。当目标小于约 16 像素时才选 YOLO9-P2：无人机与航拍画面、远距离监控、卫星切片。
        额外的尺度会使计算量与 anchor 数量大约翻倍，这是 stride-4 网格的代价。
      </P>

      <SubHeading>训练自己的模型</SubHeading>
      <P>
        YOLO9-P2 从标准 YOLOv9 检测权重迁移初始化：主干、共享的 neck 和已有检测塔直接加载，新增的 P2
        模块从头初始化。下面的配方浓缩了我们在微小目标数据上踩过的坑：
      </P>
      <CodeBlock language="python">{`from libreyolo import LibreYOLO9P2

model = LibreYOLO9P2(None, size="s")
model.train(
    data="/abs/path/tiny_objects.yaml",
    imgsz=768,                # resolution is the biggest lever for tiny objects
    lr0=0.005,                # the family default 0.01 diverges on transfer init
    mosaic_prob=0.0,          # mosaic tiling shrinks tiny objects below detectability
    mixup_prob=0.0,
    hsv_prob=1.0, flip_prob=0.5,
    max_labels=600,           # dense aerial frames exceed the default 100-box cap
    pretrained="LibreYOLO9s.pt",  # transfer init from stock YOLOv9
    epochs=60,
)`}</CodeBlock>

      <Divider />

      {/* ───────── LoRA ───────── */}
      <SectionHeading id="lora" icon={Layers2}>LoRA / DoRA 微调</SectionHeading>
      <div className="flex flex-wrap gap-2 mb-5">
        <SupportBadge variant="experimental">RF-DETR: n, s, m, l</SupportBadge>
      </div>
      <P>
        LoRA 式适配器让你通过训练一小组低秩矩阵来微调 RF-DETR 的 Transformer 主干，同时保持基础权重冻结。
        这能削减优化器与梯度的显存占用，非常适合在普通硬件上将一个强力检查点适配到新领域。
      </P>

      <SubHeading>启用方式</SubHeading>
      <P>
        整个公开 API 就是 <InlineCode>train()</InlineCode> 上的一个标志。没有 rank、alpha 或目标模块等参数可调；
        配方固定为一套经过充分测试的配置。底层实现使用 <strong className="text-surface-800 dark:text-white">DoRA</strong>{' '}
        （权重分解的 LoRA，秩 16），应用于 DINOv2 注意力的 query、key 与 value 投影。
      </P>
      <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("rf-detr-nano.pth")   # sizes n, s, m, l
result = model.train(
    data="data.yaml",
    lora=True,        # DoRA on the frozen DINOv2 backbone
    epochs=100, batch_size=4, lr=1e-4,
)

# Resume: LoRA is auto-detected from the checkpoint, no need to repeat the flag
model.train(data="data.yaml", resume=True)`}</CodeBlock>
      <CodeBlock language="bash">{`# CLI equivalent
libreyolo train --model rf-detr-nano.pth --data data.yaml --lora`}</CodeBlock>

      <SubHeading>检查点与导出</SubHeading>
      <ul className="space-y-2 mb-4">
        <FeatureItem>训练检查点会保留适配器张量，配置中也会记录已使用 LoRA，因此加载与续训会自动重建适配器图。</FeatureItem>
        <FeatureItem>检测头始终保持可训练，因此你仍可适配到新的类别数。</FeatureItem>
        <FeatureItem><InlineCode>export()</InlineCode> 会将适配器合并回稠密权重。导出的模型是普通模型，不带 <InlineCode>peft</InlineCode> 依赖。</FeatureItem>
        <FeatureItem>LoRA 仅限 RF-DETR；向其他家族传入 <InlineCode>lora=True</InlineCode> 会抛出明确的错误。</FeatureItem>
      </ul>
      <Callout icon={ShieldCheck} tone="emerald" title="安装额外依赖">
        <p>
          LoRA 训练需要适配器依赖：<InlineCode>pip install "libreyolo[lora]"</InlineCode>，
          它会引入 RF-DETR 相关组件与 <InlineCode>peft</InlineCode>。导出（已合并）的模型在推理时无需该依赖。
        </p>
      </Callout>

      <Divider />

      {/* ───────── STATUS ───────── */}
      <SectionHeading id="status" icon={AlertTriangle}>稳定性</SectionHeading>
      <P>
        各项功能当前所处的阶段。这里的一切都是实验性的；此表是真实情况的映射。
      </P>
      <DocTable
        headers={['功能', '家族', '状态']}
        rows={[
          ['分类', 'YOLO9, RF-DETR', <SupportBadge key="a" variant="experimental">PR 已开启</SupportBadge>],
          ['旋转框 (OBB)', 'YOLO9, RF-DETR', <SupportBadge key="b" variant="experimental">实验性</SupportBadge>],
          ['关键点 / 姿态', 'YOLO9, RF-DETR', <SupportBadge key="c" variant="wip">即将上线</SupportBadge>],
          ['关键点 / 姿态', 'YOLO-NAS, EdgeCrafter', <SupportBadge key="d" variant="experimental">已可用</SupportBadge>],
          ['小目标检测', 'YOLO9-P2', <SupportBadge key="f" variant="experimental">研究预览</SupportBadge>],
          ['LoRA / DoRA', 'RF-DETR', <SupportBadge key="e" variant="experimental">已评审</SupportBadge>],
        ]}
      />
      <Callout icon={Crosshair} tone="libre" title="在寻找稳定的方案？">
        <p>
          对于生产工作，经过验证的核心是 YOLO9 检测以及 RF-DETR 检测与分割。相关内容请参阅{' '}
          <a href="/zh/docs" className="text-libre-600 dark:text-libre-400 hover:underline">核心文档</a>，
          开放词表检测请参阅 <a href="/docs/librevlm" className="text-libre-600 dark:text-libre-400 hover:underline">LibreVLM</a>。
        </p>
      </Callout>

      <div className="mt-12 flex flex-wrap gap-3 items-center">
        <GraduationCap className="w-5 h-5 text-surface-400" />
        <span className="text-sm text-surface-500 dark:text-surface-400">
          在 <ExternalRef href="https://github.com/LibreYOLO/libreyolo">GitHub</ExternalRef> 上追踪进展与源码
        </span>
      </div>
    </DocLayout>
  )
}


export default function Page() {
  const locale = useLocale()
  if (locale === 'zh') return <ExperimentalPageZh />
  return <ExperimentalPage />
}
