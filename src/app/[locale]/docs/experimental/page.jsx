'use client'

import { motion } from 'framer-motion'
import {
  FlaskConical, Tags, Rotate3d, PersonStanding, Layers2, Sparkles,
  AlertTriangle, GitBranch, Crosshair, GraduationCap, ShieldCheck,
} from 'lucide-react'
import {
  DocLayout, DocHero, SectionHeading, SubHeading, P, InlineCode, Divider,
  CodeBlock, DocTable, FeatureItem, Callout, SupportBadge, ExternalRef,
} from '@/components/DocsKit'

const sections = [
  { id: 'overview', title: 'Overview', icon: FlaskConical },
  { id: 'task-selection', title: 'Selecting a Task', icon: GitBranch },
  { id: 'classification', title: 'Classification', icon: Tags },
  { id: 'obb', title: 'Oriented Boxes (OBB)', icon: Rotate3d },
  { id: 'pose', title: 'Keypoints / Pose', icon: PersonStanding },
  { id: 'lora', title: 'LoRA / DoRA', icon: Layers2 },
  { id: 'status', title: 'Stability', icon: AlertTriangle },
]

const relatedLinks = [
  { href: '/docs', label: 'Core documentation' },
  { href: '/docs/librevlm', label: 'LibreVLM' },
  { href: '/models', label: 'Model Zoo' },
]

export default function ExperimentalPage() {
  return (
    <DocLayout
      sections={sections}
      eyebrow="Experimental"
      copyTitle="LibreYOLO Experimental Tasks"
      relatedLinks={relatedLinks}
    >
      <DocHero
        badge="Experimental tasks"
        badgeTone="experimental"
        title="What's "
        accent="next"
        lead="The detection and segmentation paths are the validated core. This page documents the new task heads and training tricks we are actively building on top of them: classification, oriented boxes, pose, and parameter-efficient fine-tuning."
      />

      {/* ───────── OVERVIEW ───────── */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <SectionHeading id="overview" icon={FlaskConical}>Overview</SectionHeading>
        <P>
          LibreYOLO is a multi-task framework: the same model family can wear different heads. Alongside
          the validated detect and segment paths, several new tasks are landing for the two flagship
          families, YOLO9 and RF-DETR. They all plug into the same{' '}
          <InlineCode>LibreYOLO(...)</InlineCode> factory and the same <InlineCode>Results</InlineCode>{' '}
          container, so once you know the core API these are small additions.
        </P>
        <ul className="space-y-2 mb-4">
          <FeatureItem><strong className="text-surface-800 dark:text-white">Classification</strong> for YOLO9 and RF-DETR. Whole-image labels with top-1 / top-5 probabilities.</FeatureItem>
          <FeatureItem><strong className="text-surface-800 dark:text-white">Oriented bounding boxes (OBB)</strong> for YOLO9 and RF-DETR. Rotated boxes for aerial and document imagery.</FeatureItem>
          <FeatureItem><strong className="text-surface-800 dark:text-white">Keypoints / pose</strong> for YOLO9 and RF-DETR. COCO-17 person keypoints.</FeatureItem>
          <FeatureItem><strong className="text-surface-800 dark:text-white">LoRA / DoRA</strong> fine-tuning for RF-DETR. Adapt the transformer backbone with a fraction of the memory.</FeatureItem>
        </ul>
        <Callout icon={AlertTriangle} tone="amber" title="Read this first">
          <p>
            Everything on this page is experimental and some of it is still in flight on feature
            branches. APIs, defaults, and label formats can change before they are promoted into the
            validated core. The <a href="#status" className="text-libre-600 dark:text-libre-400 hover:underline">Stability</a>{' '}
            section tracks exactly where each feature stands.
          </p>
        </Callout>
      </motion.div>

      <Divider />

      {/* ───────── TASK SELECTION ───────── */}
      <SectionHeading id="task-selection" icon={GitBranch}>Selecting a Task</SectionHeading>
      <P>
        Every family defaults to detection. You opt into another task in one of three ways, resolved in
        this order of precedence:
      </P>
      <DocTable
        headers={['Priority', 'Mechanism', 'Example']}
        rows={[
          ['1', 'Explicit argument', <InlineCode key="a">task="obb"</InlineCode>],
          ['2', 'Checkpoint metadata', 'task recorded inside a trained .pt'],
          ['3', 'Filename suffix', <span key="c"><InlineCode>-cls</InlineCode>, <InlineCode>-obb</InlineCode>, <InlineCode>-pose</InlineCode></span>],
          ['4', 'Family default', 'detect'],
        ]}
      />
      <P>
        Because the public <InlineCode>LibreYOLO(...)</InlineCode> factory expects a real weights file,
        the cleanest way to start one of these tasks from scratch is to construct the family class
        directly and pass <InlineCode>task=</InlineCode>. Trained checkpoints load back through the unified
        factory and auto-detect their task.
      </P>
      <CodeBlock language="python">{`from libreyolo import LibreYOLO, LibreYOLO9, LibreRFDETR

# Start a task from scratch via the family class
m = LibreYOLO9(None, size="t", task="classify", nb_classes=10)

# Load a trained checkpoint via the unified factory (task auto-detected)
m = LibreYOLO("LibreYOLO9t-obb.pt")`}</CodeBlock>

      <Divider />

      {/* ───────── CLASSIFICATION ───────── */}
      <SectionHeading id="classification" icon={Tags}>Image Classification</SectionHeading>
      <div className="flex flex-wrap gap-2 mb-5">
        <SupportBadge variant="experimental">YOLO9: t, s, m, c</SupportBadge>
        <SupportBadge variant="experimental">RF-DETR: n, s, m, l</SupportBadge>
      </div>
      <P>
        Classification gives a single label for a whole image. YOLO9 keeps its backbone and bolts on a
        lightweight classification head; RF-DETR reuses its DINOv2 encoder and adds a pooled linear head.
        Both run at 224 by 224.
      </P>

      <SubHeading>Inference and the Probs result</SubHeading>
      <P>
        Prediction returns a <InlineCode>Results</InlineCode> object whose <InlineCode>probs</InlineCode>{' '}
        field carries a softmax over the classes.
      </P>
      <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9t-cls.pt")
r = model.predict("cat.jpg")

print(r.probs.top1)        # class id of the argmax
print(r.probs.top1conf)    # its probability
print(r.probs.top5)        # [id, id, id, id, id]
print(model.names[r.probs.top1])  # human-readable label`}</CodeBlock>
      <DocTable
        headers={['Field', 'Type', 'Meaning']}
        rows={[
          [<InlineCode key="a">probs.top1</InlineCode>, 'int', 'Argmax class id.'],
          [<InlineCode key="b">probs.top5</InlineCode>, 'list[int]', 'Top-5 class ids, descending.'],
          [<InlineCode key="c">probs.top1conf</InlineCode>, 'float', 'Probability of the top-1 class.'],
          [<InlineCode key="d">probs.top5conf</InlineCode>, 'tensor', 'Probabilities of the top-5 classes.'],
          [<InlineCode key="e">probs.data</InlineCode>, 'tensor', 'Full softmax vector.'],
        ]}
      />

      <SubHeading>Dataset format and training</SubHeading>
      <P>
        Classification uses an <strong className="text-surface-800 dark:text-white">ImageFolder</strong>{' '}
        layout, not a YAML. Class names are the sorted subfolder names, pinned to the train split.
      </P>
      <CodeBlock language="text" filename="dataset/">{`dataset/
  train/
    cat/   img001.jpg ...
    dog/   img104.jpg ...
  val/
    cat/   ...
    dog/   ...`}</CodeBlock>
      <P>
        The <InlineCode>data=</InlineCode> argument accepts a folder, a <InlineCode>.zip</InlineCode> URL,
        or a known auto-download name (<InlineCode>imagenette160</InlineCode> and{' '}
        <InlineCode>imagenet10</InlineCode>). The head is rebuilt to match the dataset's class count
        automatically.
      </P>
      <CodeBlock language="python">{`from libreyolo import LibreYOLO9

model = LibreYOLO9(None, size="t", task="classify", nb_classes=10)
result = model.train(
    data="imagenette160",   # folder, .zip URL, or known name
    epochs=10, batch=64, imgsz=224,
    optimizer="adamw", lr0=1e-3,
)
# Validation reports metrics/accuracy_top1 and metrics/accuracy_top5`}</CodeBlock>
      <Callout icon={Sparkles} tone="emerald" title="Reference runs">
        <p>
          Quick sanity checks from development: YOLO9-t reached top-1 0.79 / top-5 0.975 on imagenette160
          (10 epochs), and RF-DETR-n reached top-1 0.69 / top-5 0.96 (6 epochs). RF-DETR benefits from
          internet access on first run to fetch its DINOv2 backbone; offline it falls back to random init.
        </p>
      </Callout>

      <Divider />

      {/* ───────── OBB ───────── */}
      <SectionHeading id="obb" icon={Rotate3d}>Oriented Bounding Boxes (OBB)</SectionHeading>
      <div className="flex flex-wrap gap-2 mb-5">
        <SupportBadge variant="experimental">YOLO9: t, s, m, c</SupportBadge>
        <SupportBadge variant="experimental">RF-DETR: n, s, m, l</SupportBadge>
      </div>
      <P>
        Oriented boxes carry a rotation angle, which is what aerial imagery, documents, and densely packed
        scenes need. YOLO9 adds an angle branch to its detect head; RF-DETR adds a learnable angle
        embedding to its decoder.
      </P>

      <SubHeading>Inference and the OBB result</SubHeading>
      <P>
        Results expose an <InlineCode>obb</InlineCode> field. Angles are in{' '}
        <strong className="text-surface-800 dark:text-white">radians</strong>.
      </P>
      <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9t-obb.pt")
r = model.predict("aerial.jpg")

for i in range(len(r.obb.cls)):
    cx, cy, w, h, angle = r.obb.xywhr[i]  # angle in radians
    corners = r.obb.xyxyxyxy[i]           # 4 (x, y) corner points
    conf, cls = r.obb.conf[i], r.obb.cls[i]`}</CodeBlock>
      <DocTable
        headers={['Field', 'Shape', 'Meaning']}
        rows={[
          [<InlineCode key="a">obb.xywhr</InlineCode>, 'N x 5', '[cx, cy, w, h, angle], angle in radians.'],
          [<InlineCode key="b">obb.xyxyxyxy</InlineCode>, 'N x 4 x 2', 'Four corner points per box.'],
          [<InlineCode key="c">obb.conf</InlineCode>, 'N', 'Confidence per box.'],
          [<InlineCode key="d">obb.cls</InlineCode>, 'N', 'Class id per box.'],
        ]}
      />

      <SubHeading>Dataset format and training</SubHeading>
      <P>
        OBB uses a standard detect-style data YAML, but labels are YOLO-OBB text files with{' '}
        <strong className="text-surface-800 dark:text-white">exactly nine fields</strong> per row: a class
        id followed by four normalized corner points. The angle is derived from the corners, not stored.
      </P>
      <CodeBlock language="text" filename="labels/aerial_001.txt">{`# class_id  x1 y1  x2 y2  x3 y3  x4 y4   (all normalized to [0, 1])
0  0.51 0.32  0.66 0.38  0.62 0.55  0.47 0.49
2  0.10 0.71  0.18 0.69  0.20 0.80  0.12 0.82`}</CodeBlock>
      <P>
        A plain detection checkpoint cannot be loaded directly into an OBB model. Going from detect to OBB
        is only allowed as a training warm-start: pass <InlineCode>pretrained=True</InlineCode> (YOLO9) or
        the explicit transfer flag on RF-DETR. Mosaic and mixup are disabled for OBB until corner-aware
        augmentation lands, and tiled inference is not supported.
      </P>
      <CodeBlock language="python">{`from libreyolo import LibreYOLO9

model = LibreYOLO9(None, size="t", task="obb")
# Warm-start the backbone from a same-family detect checkpoint
result = model.train(data="dota8.yaml", pretrained=True, epochs=100, imgsz=640)

# CLI equivalent
# libreyolo train model=LibreYOLO9t.pt data=dota8.yaml --task obb`}</CodeBlock>
      <P>
        Validation uses rotated-IoU AP, reported as mAP50 and mAP50-95 under the OBB metric group.
      </P>

      <Divider />

      {/* ───────── POSE ───────── */}
      <SectionHeading id="pose" icon={PersonStanding}>Keypoints / Pose</SectionHeading>
      <div className="flex flex-wrap gap-2 mb-5">
        <SupportBadge variant="wip">YOLO9 + RF-DETR: landing soon</SupportBadge>
        <SupportBadge variant="experimental">YOLO-NAS, EdgeCrafter: available</SupportBadge>
      </div>
      <P>
        Pose estimation predicts keypoints per detected instance. The default layout is COCO-17 person
        keypoints. YOLO9 and RF-DETR pose are person-only single-class in their first version; YOLO-NAS
        and EdgeCrafter pose are already available in the tree.
      </P>

      <SubHeading>Inference and the Keypoints result</SubHeading>
      <P>
        Results expose a <InlineCode>keypoints</InlineCode> field of shape{' '}
        <InlineCode>(N, K, 3)</InlineCode>, where the last channel is visibility or confidence, in
        original-image pixel coordinates.
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
        headers={['Field', 'Shape', 'Meaning']}
        rows={[
          [<InlineCode key="a">keypoints.xy</InlineCode>, 'N x K x 2', 'Pixel keypoint coordinates.'],
          [<InlineCode key="b">keypoints.xyn</InlineCode>, 'N x K x 2', 'Normalized keypoint coordinates.'],
          [<InlineCode key="c">keypoints.conf</InlineCode>, 'N x K', 'Per-keypoint visibility / confidence.'],
          [<InlineCode key="d">keypoints.has_visible</InlineCode>, 'N x K', 'Boolean visible mask.'],
        ]}
      />

      <SubHeading>Dataset format and training</SubHeading>
      <P>
        Pose uses a data YAML that must declare <InlineCode>kpt_shape: [K, 2|3]</InlineCode> and, for
        horizontal-flip augmentation, a <InlineCode>flip_idx</InlineCode>. Labels are YOLO-pose text rows:
        a class id, a normalized box, then <InlineCode>K</InlineCode> keypoint triplets{' '}
        <InlineCode>(x, y, v)</InlineCode> with visibility <InlineCode>v</InlineCode> in{' '}
        <InlineCode>{'{0, 1, 2}'}</InlineCode>.
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
      <Callout icon={AlertTriangle} tone="rose" title="In active development">
        <p>
          YOLO9 and RF-DETR pose are on a feature branch and have not been merged yet; treat the API
          above as the intended contract rather than a frozen one. YOLO-NAS pose weights are linked from
          upstream rather than mirrored and must be staged manually.
        </p>
      </Callout>

      <Divider />

      {/* ───────── LoRA ───────── */}
      <SectionHeading id="lora" icon={Layers2}>LoRA / DoRA Fine-Tuning</SectionHeading>
      <div className="flex flex-wrap gap-2 mb-5">
        <SupportBadge variant="experimental">RF-DETR: n, s, m, l</SupportBadge>
      </div>
      <P>
        LoRA-style adapters let you fine-tune RF-DETR's transformer backbone by training a small set of
        low-rank matrices while the base weights stay frozen. That cuts optimizer and gradient memory,
        which is ideal for adapting a strong checkpoint to a new domain on modest hardware.
      </P>

      <SubHeading>Enabling it</SubHeading>
      <P>
        The whole public API is a single flag on <InlineCode>train()</InlineCode>. There are no rank,
        alpha, or target-module knobs to tune; the recipe is fixed to a well-tested configuration. Under
        the hood the implementation uses <strong className="text-surface-800 dark:text-white">DoRA</strong>{' '}
        (weight-decomposed LoRA, rank 16) applied to the DINOv2 attention query, key, and value
        projections.
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

      <SubHeading>Checkpoints and export</SubHeading>
      <ul className="space-y-2 mb-4">
        <FeatureItem>Training checkpoints keep the adapter tensors, and the config records that LoRA was used, so loading and resuming rebuild the adapter graph automatically.</FeatureItem>
        <FeatureItem>The detection head always stays trainable, so you can still adapt to a new class count.</FeatureItem>
        <FeatureItem><InlineCode>export()</InlineCode> merges the adapters back into dense weights. Exported models are plain and carry no <InlineCode>peft</InlineCode> dependency.</FeatureItem>
        <FeatureItem>LoRA is RF-DETR only; passing <InlineCode>lora=True</InlineCode> to other families raises a clear error.</FeatureItem>
      </ul>
      <Callout icon={ShieldCheck} tone="emerald" title="Install extra">
        <p>
          LoRA training needs the adapter dependency: <InlineCode>pip install "libreyolo[lora]"</InlineCode>,
          which pulls in the RF-DETR stack and <InlineCode>peft</InlineCode>. Exported (merged) models do
          not need it at inference time.
        </p>
      </Callout>

      <Divider />

      {/* ───────── STATUS ───────── */}
      <SectionHeading id="status" icon={AlertTriangle}>Stability</SectionHeading>
      <P>
        Where each feature stands today. Everything here is experimental; this table is the honest map.
      </P>
      <DocTable
        headers={['Feature', 'Families', 'State']}
        rows={[
          ['Classification', 'YOLO9, RF-DETR', <SupportBadge key="a" variant="experimental">PR open</SupportBadge>],
          ['Oriented boxes (OBB)', 'YOLO9, RF-DETR', <SupportBadge key="b" variant="experimental">Experimental</SupportBadge>],
          ['Keypoints / pose', 'YOLO9, RF-DETR', <SupportBadge key="c" variant="wip">Landing soon</SupportBadge>],
          ['Keypoints / pose', 'YOLO-NAS, EdgeCrafter', <SupportBadge key="d" variant="experimental">Available</SupportBadge>],
          ['LoRA / DoRA', 'RF-DETR', <SupportBadge key="e" variant="experimental">Reviewed</SupportBadge>],
        ]}
      />
      <Callout icon={Crosshair} tone="libre" title="Looking for the stable path?">
        <p>
          For production work, the validated core is YOLO9 detection and RF-DETR detection and
          segmentation. See the{' '}
          <a href="/docs" className="text-libre-600 dark:text-libre-400 hover:underline">core documentation</a>{' '}
          for those, and <a href="/docs/librevlm" className="text-libre-600 dark:text-libre-400 hover:underline">LibreVLM</a>{' '}
          for open-vocabulary detection.
        </p>
      </Callout>

      <div className="mt-12 flex flex-wrap gap-3 items-center">
        <GraduationCap className="w-5 h-5 text-surface-400" />
        <span className="text-sm text-surface-500 dark:text-surface-400">
          Track progress and source on <ExternalRef href="https://github.com/Libre-YOLO/libreyolo">GitHub</ExternalRef>
        </span>
      </div>
    </DocLayout>
  )
}
