'use client'
import { useLocale } from 'next-intl'

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
  { id: 'small-object', title: 'Small-Object Detection', icon: Crosshair },
  { id: 'lora', title: 'LoRA / DoRA', icon: Layers2 },
  { id: 'status', title: 'Stability', icon: AlertTriangle },
]

const relatedLinks = [
  { href: '/docs', label: 'Core documentation' },
  { href: '/docs/librevlm', label: 'LibreVLM' },
  { href: '/models', label: 'Model Zoo' },
]

function ExperimentalPage() {
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
          <FeatureItem><strong className="text-surface-800 dark:text-white">Small-object detection</strong> with YOLO9-P2, a YOLOv9 variant with a stride-4 scale for the 4-16 px objects of aerial and drone imagery, including a VisDrone research-preview checkpoint.</FeatureItem>
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

      {/* ───────── SMALL-OBJECT ───────── */}
      <SectionHeading id="small-object" icon={Crosshair}>Small-Object Detection (YOLO9-P2)</SectionHeading>
      <div className="flex flex-wrap gap-2 mb-5">
        <SupportBadge variant="experimental">YOLO9-P2: t, s</SupportBadge>
        <SupportBadge variant="experimental">VisDrone research preview</SupportBadge>
      </div>
      <P>
        YOLO9-P2 is YOLOv9 with a fourth detection scale at{' '}
        <strong className="text-surface-800 dark:text-white">stride 4</strong>. Stock YOLOv9 detects at
        strides 8/16/32, so objects below ~16 px fall under its finest grid; the P2 head catches the
        4-16 px range that dominates aerial and drone footage.
      </P>
      <P>
        In a controlled A/B on VisDrone (same recipe, same resolution, same init; the only change was
        the P2 head), small-object AP improved by{' '}
        <strong className="text-surface-800 dark:text-white">+49%</strong> over stock YOLOv9 of the same
        size. Adding higher training resolution and the bigger s size roughly doubled small-object AP
        across the project:
      </P>
      <DocTable
        headers={['Model', 'AP', 'AP50', 'AP_small']}
        rows={[
          ['Stock YOLO9-t @640 (control)', '0.123', '0.220', '0.047'],
          ['YOLO9-P2-t @640 (same-recipe A/B)', '0.138', '0.254', '0.070'],
          [<strong key="s">YOLO9-P2-s @768 (released preview)</strong>, '0.226', '0.385', '0.141'],
        ]}
      />
      <P className="text-sm">
        VisDrone2019-DET val (548 images), pycocotools, single seed; treat &plusmn;1 point as noise.
      </P>

      <SubHeading>The VisDrone research preview</SubHeading>
      <P>
        A trained checkpoint is published as{' '}
        <a href="https://huggingface.co/LibreYOLO/LibreYOLO9P2s-visdrone" target="_blank" rel="noopener noreferrer" className="text-libre-600 dark:text-libre-400 hover:underline">LibreYOLO9P2s-visdrone</a>.
        The family is merged on <InlineCode>dev</InlineCode> but not yet in a PyPI release, so install
        from source until the next release.
      </P>
      <CodeBlock language="python">{`from libreyolo import LibreYOLO

# Auto-downloads from the LibreYOLO Hugging Face org
model = LibreYOLO("LibreYOLO9P2s-visdrone.pt")

# Evaluate/predict at 768 - the resolution it was trained at
results = model.predict("aerial.jpg", imgsz=768, conf=0.25)`}</CodeBlock>
      <Callout icon={AlertTriangle} tone="amber" title="Non-commercial license">
        <p>
          The preview checkpoint is trained on VisDrone2019-DET (AISKYEYE, Tianjin University), licensed
          CC BY-NC-SA 3.0: <strong>non-commercial use only</strong>, unlike LibreYOLO&apos;s MIT
          code and COCO-default weights. It detects the 10 VisDrone aerial classes, not COCO. The model
          card ships the exact training recipe, the per-epoch metrics, and a clean-room dataset converter
          so you can reproduce it or retrain on your own data.
        </p>
      </Callout>

      <SubHeading>When (not) to use it</SubHeading>
      <P>
        Match the architecture to the arena. On COCO-like data (&quot;small&quot; means 16-32 px)
        the P2 head does <strong className="text-surface-800 dark:text-white">not</strong> help;
        stock YOLOv9 is the better pick there. Reach for YOLO9-P2 when your objects live under ~16 px:
        drone and aerial footage, distant CCTV, satellite tiles. The extra scale roughly doubles compute
        and anchor count. That is the price of the stride-4 grid.
      </P>

      <SubHeading>Training your own</SubHeading>
      <P>
        YOLO9-P2 transfer-initializes from stock YOLOv9 detect checkpoints: the backbone, shared neck,
        and existing head towers load; the new P2 modules start fresh. The recipe below encodes what we
        learned the hard way on tiny-object data:
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
          ['Small-object detection', 'YOLO9-P2', <SupportBadge key="f" variant="experimental">Research preview</SupportBadge>],
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
  { href: '/docs', label: '核心文档' },
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
          <a href="/docs" className="text-libre-600 dark:text-libre-400 hover:underline">核心文档</a>，
          开放词表检测请参阅 <a href="/docs/librevlm" className="text-libre-600 dark:text-libre-400 hover:underline">LibreVLM</a>。
        </p>
      </Callout>

      <div className="mt-12 flex flex-wrap gap-3 items-center">
        <GraduationCap className="w-5 h-5 text-surface-400" />
        <span className="text-sm text-surface-500 dark:text-surface-400">
          在 <ExternalRef href="https://github.com/Libre-YOLO/libreyolo">GitHub</ExternalRef> 上追踪进展与源码
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
