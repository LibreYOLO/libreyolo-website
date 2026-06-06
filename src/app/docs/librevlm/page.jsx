'use client'

import { motion } from 'framer-motion'
import {
  ScanSearch, Sparkles, Terminal, Rocket, Layers, Tags, Crosshair,
  MessageSquare, Settings2, AlertTriangle, Eye, ShieldCheck,
} from 'lucide-react'
import {
  DocLayout, DocHero, SectionHeading, SubHeading, P, InlineCode, Divider,
  CodeBlock, DocTable, FeatureItem, Callout, SupportBadge, ExternalRef,
} from '@/components/DocsKit'

const sections = [
  { id: 'introduction', title: 'Introduction', icon: ScanSearch },
  { id: 'installation', title: 'Installation', icon: Terminal },
  { id: 'quickstart', title: 'Quickstart', icon: Rocket },
  { id: 'models', title: 'Supported Models', icon: Layers },
  { id: 'vocabulary', title: 'Setting the Vocabulary', icon: Tags },
  { id: 'prediction', title: 'Prediction', icon: Crosshair },
  { id: 'examples', title: 'Examples', icon: Sparkles },
  { id: 'chat', title: 'Raw Chat', icon: MessageSquare },
  { id: 'backends', title: 'How Backends Differ', icon: Settings2 },
  { id: 'limitations', title: 'Limitations', icon: AlertTriangle },
]

const relatedLinks = [
  { href: '/docs', label: 'Core documentation' },
  { href: '/docs/experimental', label: 'Experimental tasks' },
  { href: '/models', label: 'Model Zoo' },
]

export default function LibreVLMPage() {
  return (
    <DocLayout
      sections={sections}
      eyebrow="LibreVLM"
      copyTitle="LibreVLM Documentation"
      relatedLinks={relatedLinks}
    >
      <DocHero
        badge="Experimental tier"
        badgeTone="experimental"
        title="Libre"
        accent="VLM"
        lead="Point a vision language model at an image, hand it a list of words, and get back boxes. LibreVLM turns Qwen3-VL, Florence-2 and friends into open-vocabulary object detectors that speak the exact same Results API as every other LibreYOLO model."
      />

      {/* ───────── INTRODUCTION ───────── */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <SectionHeading id="introduction" icon={ScanSearch}>Introduction</SectionHeading>
        <P>
          A classic detector ships with a fixed list of classes baked into its head. LibreVLM throws
          that constraint away. It wraps modern instruction-tuned vision language models, prompts them
          to emit bounding boxes, parses the generated text, and returns the same{' '}
          <InlineCode>Results</InlineCode> object you already use for YOLO9 and RF-DETR. The class list
          is just a list of words you supply at runtime, so adding a new category costs nothing and
          works zero-shot.
        </P>
        <ul className="space-y-2 mb-4">
          <FeatureItem><strong className="text-surface-800 dark:text-white">Open vocabulary.</strong> Detect <InlineCode>"pink car"</InlineCode>, <InlineCode>"license plate"</InlineCode>, or <InlineCode>"the small island"</InlineCode> without ever training a head for them.</FeatureItem>
          <FeatureItem><strong className="text-surface-800 dark:text-white">One factory, one contract.</strong> <InlineCode>LibreVLM(...)</InlineCode> returns the standard <InlineCode>Results</InlineCode> with <InlineCode>boxes.xyxy</InlineCode>, <InlineCode>boxes.cls</InlineCode>, <InlineCode>boxes.conf</InlineCode>, plus <InlineCode>.plot()</InlineCode> and <InlineCode>.save()</InlineCode>.</FeatureItem>
          <FeatureItem><strong className="text-surface-800 dark:text-white">Swappable backends.</strong> Six model families behind one alias string, from a 230M Florence-2 to an 8B Qwen3-VL.</FeatureItem>
          <FeatureItem><strong className="text-surface-800 dark:text-white">A raw escape hatch.</strong> <InlineCode>chat()</InlineCode> gives you free-form image question answering when you need more than boxes.</FeatureItem>
        </ul>

        <Callout icon={Sparkles} tone="libre" title="Why a separate tier (and a separate page)">
          <p>
            LibreVLM is deliberately kept out of the closed-vocabulary <InlineCode>LibreYOLO(...)</InlineCode>{' '}
            factory and its <InlineCode>.pt</InlineCode> registry. These models are prompt-driven, open
            vocabulary, and report synthetic confidence, so they honor a different contract. Treating
            them as their own tier keeps the core detection docs clean and honest about what is being
            measured.
          </p>
        </Callout>

        <Callout icon={AlertTriangle} tone="amber" title="On the dev branch">
          <p>
            LibreVLM currently lives on the <InlineCode>dev</InlineCode> branch and is targeted for the
            v1.3 release; it is not part of v1.2.0. It is a Python-only inference tier: there is no
            training, validation, export, or CLI path yet, and confidence scores are placeholders. Read the{' '}
            <a href="#limitations" className="text-libre-600 dark:text-libre-400 hover:underline">Limitations</a>{' '}
            section before you build on top of it.
          </p>
        </Callout>
      </motion.div>

      <Divider />

      {/* ───────── INSTALLATION ───────── */}
      <SectionHeading id="installation" icon={Terminal}>Installation</SectionHeading>
      <P>
        LibreVLM lives behind the optional <InlineCode>vlm</InlineCode> extra. It pulls in a recent{' '}
        <InlineCode>transformers</InlineCode> and the helpers a couple of processors need. Without the
        extra, importing a VLM family raises an <InlineCode>ImportError</InlineCode> that points you here.
      </P>
      <CodeBlock language="bash">{`pip install 'libreyolo[vlm]'`}</CodeBlock>
      <P>
        Weights are downloaded from the Hugging Face Hub on first use into a local{' '}
        <InlineCode>weights/</InlineCode> folder. A few families ship under non-OSI licenses and log a
        one-time notice before downloading. A GPU is recommended for the larger backends, but every
        model also runs on CPU with <InlineCode>device="cpu"</InlineCode>.
      </P>

      <Divider />

      {/* ───────── QUICKSTART ───────── */}
      <SectionHeading id="quickstart" icon={Rocket}>Quickstart</SectionHeading>
      <P>
        Construct a model, declare the words you care about, and predict. The default backend is
        Qwen3-VL-4B, the strongest detector in the tier and Apache-2.0 licensed.
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
        That is the whole loop. Everything downstream of <InlineCode>predict()</InlineCode> behaves like
        a normal detector, so existing visualization, cropping, and tracking code keeps working.
      </P>

      <Divider />

      {/* ───────── MODELS ───────── */}
      <SectionHeading id="models" icon={Layers}>Supported Models</SectionHeading>
      <P>
        Pick a backend with the alias you pass to <InlineCode>LibreVLM(...)</InlineCode>. A bare family
        name resolves to its default size. The default backend overall is{' '}
        <InlineCode>qwen3-vl-4b</InlineCode>. In practice the strongest detectors are{' '}
        <strong className="text-surface-800 dark:text-white">Qwen3-VL</strong>,{' '}
        <strong className="text-surface-800 dark:text-white">LFM2-VL</strong>, and{' '}
        <strong className="text-surface-800 dark:text-white">Florence-2</strong>.
      </P>

      <DocTable
        headers={['Family', 'Alias', 'Sizes (params)', 'License', 'Notes']}
        rows={[
          [
            <strong key="q" className="text-surface-800 dark:text-white whitespace-nowrap">Qwen3-VL</strong>,
            <code key="qa" className="font-mono text-xs">qwen3-vl-2b / -4b / -8b</code>,
            '2B / 4B / 8B',
            'Apache-2.0',
            <span key="qn">Default and strongest. Recommended starting point.</span>,
          ],
          [
            <strong key="l" className="text-surface-800 dark:text-white whitespace-nowrap">LFM2-VL</strong>,
            <code key="la" className="font-mono text-xs">lfm2-vl-450m / -1.6b</code>,
            '450M / 1.6B',
            'LFM Open License',
            <span key="ln">Edge sized, surprisingly strong small detector. Notice gated.</span>,
          ],
          [
            <strong key="i" className="text-surface-800 dark:text-white whitespace-nowrap">InternVL3</strong>,
            <code key="ia" className="font-mono text-xs">internvl3-1b / -2b / -8b</code>,
            '1B / 2B / 8B',
            'Qwen License',
            <span key="in">Good grounding at 8B; small sizes are weak. Notice gated.</span>,
          ],
          [
            <strong key="f" className="text-surface-800 dark:text-white whitespace-nowrap">Florence-2</strong>,
            <code key="fa" className="font-mono text-xs">florence-2-base / -large</code>,
            '0.23B / 0.77B',
            'MIT',
            <span key="fn">Purpose-built grounding model. Tight boxes, no <InlineCode>chat()</InlineCode>.</span>,
          ],
          [
            <strong key="s" className="text-surface-800 dark:text-white whitespace-nowrap">SmolVLM2</strong>,
            <code key="sa" className="font-mono text-xs">smolvlm2-500m / -2.2b</code>,
            '500M / 2.2B',
            'Apache-2.0',
            <span key="sn">Tiny and fast; weaker detector. Good for quick trials.</span>,
          ],
          [
            <strong key="k" className="text-surface-800 dark:text-white whitespace-nowrap">Kosmos-2</strong>,
            <code key="ka" className="font-mono text-xs">kosmos-2</code>,
            '~1.6B',
            'MIT',
            <span key="kn">2023 grounder. Coarser boxes, no <InlineCode>chat()</InlineCode>.</span>,
          ],
        ]}
      />

      <SubHeading>Choosing a backend</SubHeading>
      <ul className="space-y-2 mb-4">
        <FeatureItem><strong className="text-surface-800 dark:text-white">Best quality:</strong> <InlineCode>qwen3-vl-8b</InlineCode> or <InlineCode>qwen3-vl-4b</InlineCode> (the default).</FeatureItem>
        <FeatureItem><strong className="text-surface-800 dark:text-white">Tight boxes, small footprint:</strong> <InlineCode>florence-2-large</InlineCode>.</FeatureItem>
        <FeatureItem><strong className="text-surface-800 dark:text-white">Edge / CPU:</strong> <InlineCode>lfm2-vl-450m</InlineCode> or <InlineCode>smolvlm2-500m</InlineCode>.</FeatureItem>
        <FeatureItem><strong className="text-surface-800 dark:text-white">Fully permissive license:</strong> any Qwen3-VL, SmolVLM2, Florence-2 or Kosmos-2 size.</FeatureItem>
      </ul>

      <Callout icon={ShieldCheck} tone="emerald" title="Licensing">
        <p>
          Qwen3-VL and SmolVLM2 are Apache-2.0; Florence-2 and Kosmos-2 are MIT. LFM2-VL and InternVL3
          carry non-OSI licenses and emit a one-time notice before their first download, so you can make
          an informed choice for commercial use.
        </p>
      </Callout>

      <Divider />

      {/* ───────── VOCABULARY ───────── */}
      <SectionHeading id="vocabulary" icon={Tags}>Setting the Vocabulary</SectionHeading>
      <P>
        The vocabulary is the heart of open-vocabulary detection. Call{' '}
        <InlineCode>set_classes()</InlineCode> with a list of label strings. It is sticky: it persists
        across every later <InlineCode>predict()</InlineCode> and <InlineCode>track()</InlineCode> call
        until you set it again. It returns <InlineCode>self</InlineCode>, so it chains.
      </P>
      <CodeBlock language="python">{`# Sticky and chainable
model = LibreVLM("qwen3-vl-2b").set_classes(["person", "dog", "cat"])

# Set it once at construction instead
model = LibreVLM("lfm2-vl-450m", names=["boat"], device="cpu")

# Re-set any time to change what you are looking for
model.set_classes(["a red car", "a blue truck"])`}</CodeBlock>
      <P>
        Labels can be any phrase. They must be unique case-insensitively, and you must pass a list, not a
        bare string. If you never call <InlineCode>set_classes()</InlineCode>, the model falls back to the
        COCO-80 vocabulary so a bare <InlineCode>predict()</InlineCode> still does something sensible.
      </P>

      <Divider />

      {/* ───────── PREDICTION ───────── */}
      <SectionHeading id="prediction" icon={Crosshair}>Prediction</SectionHeading>
      <P>
        <InlineCode>predict()</InlineCode> (and the equivalent <InlineCode>model(...)</InlineCode> call)
        accepts the same source types as any LibreYOLO detector: a path, a PIL image, a numpy array, a
        URL, a folder, or a video. <InlineCode>stream=True</InlineCode> and{' '}
        <InlineCode>track()</InlineCode> work too.
      </P>
      <CodeBlock language="python">{`result = model.predict(
    source="image.jpg",  # path | PIL | ndarray | URL | folder | video
    conf=0.25,           # see note below: scoring is synthetic
    classes=[0],         # optional: keep only these vocabulary ids
    max_det=300,
)`}</CodeBlock>

      <SubHeading>Return shape</SubHeading>
      <P>
        You get back the standard <InlineCode>Results</InlineCode> object, identical to a closed-vocabulary
        detector:
      </P>
      <DocTable
        headers={['Field', 'Shape / type', 'Meaning']}
        rows={[
          [<InlineCode key="a">result.boxes.xyxy</InlineCode>, 'N x 4', 'Pixel boxes [x1, y1, x2, y2], scaled to the original image.'],
          [<InlineCode key="b">result.boxes.cls</InlineCode>, 'N', 'Class ids indexing into your set_classes() vocabulary.'],
          [<InlineCode key="c">result.boxes.conf</InlineCode>, 'N', 'Synthetic confidence: 1.0 for every box (see Limitations).'],
          [<InlineCode key="d">result.plot() / .save()</InlineCode>, '-', 'The usual drawing and saving helpers.'],
        ]}
      />
      <P>
        Under the hood, LibreVLM tolerantly parses the model output (handling markdown fences, stray
        prose, duplicated boxes, and truncated arrays), maps free-text labels back to your class ids, and
        drops any label that is not in your vocabulary. That last step is what makes a free-form generator
        behave like a closed-set detector.
      </P>

      <Divider />

      {/* ───────── EXAMPLES ───────── */}
      <SectionHeading id="examples" icon={Sparkles}>Examples</SectionHeading>

      <SubHeading>Detect a specific colored object</SubHeading>
      <CodeBlock language="python">{`from libreyolo import LibreVLM

model = LibreVLM("qwen3-vl-4b")
model.set_classes(["red car"])

result = model.predict("parking_lot.jpg")
print(f"Found {len(result.boxes.cls)} red car(s)")
result.save("red_cars.jpg")`}</CodeBlock>

      <SubHeading>Tight boxes with Florence-2</SubHeading>
      <CodeBlock language="python">{`# Florence-2 is a purpose-built grounder: very tight pixel boxes.
model = LibreVLM("florence-2-large")
model.set_classes(["a red car", "license plate"])

result = model.predict("car.jpg")
result.plot()`}</CodeBlock>

      <SubHeading>Filter to a single class on the fly</SubHeading>
      <CodeBlock language="python">{`model = LibreVLM("qwen3-vl-2b").set_classes(["person", "dog", "cat"])

# classes= filters the configured vocabulary by id
people_only = model.predict("street.jpg", classes=[0])`}</CodeBlock>

      <SubHeading>Run on CPU with a built-in sample image</SubHeading>
      <CodeBlock language="python">{`from libreyolo import LibreVLM, SAMPLE_IMAGE

model = LibreVLM("lfm2-vl-450m", device="cpu")
# No set_classes() -> falls back to the COCO-80 vocabulary
result = model.predict(SAMPLE_IMAGE)
print(model.names[result.boxes.cls[0]])  # e.g. "person"`}</CodeBlock>

      <SubHeading>Batches, folders, and video</SubHeading>
      <CodeBlock language="python">{`model = LibreVLM().set_classes(["forklift", "pallet"])

# A whole folder
for result in model.predict("warehouse_frames/", stream=True):
    result.save()

# A video file (frames are processed one at a time)
model.predict("warehouse.mp4", save=True)`}</CodeBlock>

      <Divider />

      {/* ───────── CHAT ───────── */}
      <SectionHeading id="chat" icon={MessageSquare}>Raw Chat</SectionHeading>
      <P>
        Sometimes you want the model, not the detector. The chat-template families expose{' '}
        <InlineCode>chat()</InlineCode>, which takes an image and a free-form prompt and returns the
        decoded text verbatim. Use it for counting, captioning, or quick visual questions.
      </P>
      <CodeBlock language="python">{`model = LibreVLM("qwen3-vl-4b")

answer = model.chat("harbor.jpg", "How many boats are docked? Answer with a number.")
print(answer)`}</CodeBlock>
      <Callout icon={AlertTriangle} tone="amber">
        <p>
          <InlineCode>chat()</InlineCode> is available on the chat-template families (Qwen3-VL, LFM2-VL,
          SmolVLM2, InternVL3). Florence-2 and Kosmos-2 are task-token grounders and raise{' '}
          <InlineCode>NotImplementedError</InlineCode>; use <InlineCode>predict()</InlineCode> with them.
        </p>
      </Callout>

      <Divider />

      {/* ───────── BACKENDS ───────── */}
      <SectionHeading id="backends" icon={Settings2}>How Backends Differ</SectionHeading>
      <P>
        Every family returns the same <InlineCode>Results</InlineCode>, but they reach it differently. You
        rarely need to care, yet it helps to know why some backends behave the way they do. The chat
        families are prompted for a JSON array of boxes; the grounders use dedicated task tokens.
      </P>
      <DocTable
        headers={['Family', 'Prompting', 'Coordinate space', 'chat()']}
        rows={[
          ['Qwen3-VL', 'JSON box prompt', '0 to 1000, rescaled', 'Yes'],
          ['LFM2-VL', 'JSON box prompt', 'Normalized 0 to 1', 'Yes'],
          ['SmolVLM2', 'JSON box prompt', 'Normalized 0 to 1', 'Yes'],
          ['InternVL3', 'JSON box prompt', '0 to 1000, rescaled', 'Yes'],
          ['Florence-2', 'Task token', 'Native pixels', 'No'],
          ['Kosmos-2', 'Grounding prompt', 'Normalized, rescaled', 'No'],
        ]}
      />
      <P>
        For the chat families you can override the detection prompt with the{' '}
        <InlineCode>prompt=</InlineCode> constructor argument, and cap generation length with{' '}
        <InlineCode>max_new_tokens=</InlineCode>. Device and dtype are resolved automatically: bf16 or
        fp16 on CUDA, fp32 on CPU.
      </P>

      <Divider />

      {/* ───────── LIMITATIONS ───────── */}
      <SectionHeading id="limitations" icon={AlertTriangle}>Limitations</SectionHeading>
      <P>
        LibreVLM is powerful but young. Knowing the boundaries up front saves surprises later.
      </P>
      <ul className="space-y-2 mb-4">
        <FeatureItem><strong className="text-surface-800 dark:text-white">Synthetic confidence.</strong> Every box is scored 1.0. The <InlineCode>conf=</InlineCode> filter therefore behaves as all-or-nothing rather than a real threshold.</FeatureItem>
        <FeatureItem><strong className="text-surface-800 dark:text-white">No mAP / validation.</strong> <InlineCode>val()</InlineCode> raises, because synthetic scores would make COCO mAP misleading.</FeatureItem>
        <FeatureItem><strong className="text-surface-800 dark:text-white">No training or export.</strong> <InlineCode>train()</InlineCode> and <InlineCode>export()</InlineCode> raise. Fine-tune the VLM upstream and load the resulting weights instead.</FeatureItem>
        <FeatureItem><strong className="text-surface-800 dark:text-white">Tracking is degraded.</strong> <InlineCode>track()</InlineCode> runs, but uniform scores make the tracker's low-confidence recovery stage inert.</FeatureItem>
        <FeatureItem><strong className="text-surface-800 dark:text-white">One image at a time.</strong> Generation is sequential in v1, so larger <InlineCode>batch=</InlineCode> values give no speedup.</FeatureItem>
        <FeatureItem><strong className="text-surface-800 dark:text-white">Python API only.</strong> The <InlineCode>libreyolo</InlineCode> CLI does not resolve VLM aliases yet.</FeatureItem>
      </ul>
      <Callout icon={Eye} tone="libre" title="Where it shines">
        <p>
          Use LibreVLM when the class set is open ended, changes often, or is hard to label up front:
          rapid prototyping, long-tail or rare categories, and "find the thing I describe in words"
          workflows. When you need calibrated confidence, throughput, or a deployable artifact, train a
          closed-vocabulary YOLO9 or RF-DETR from the{' '}
          <a href="/docs" className="text-libre-600 dark:text-libre-400 hover:underline">core docs</a>.
        </p>
      </Callout>

      <div className="mt-12 flex flex-wrap gap-3">
        <SupportBadge variant="experimental">Inference only</SupportBadge>
        <SupportBadge variant="experimental">dev branch / targeting v1.3</SupportBadge>
        <span className="text-sm text-surface-500 dark:text-surface-400 self-center">
          Source on <ExternalRef href="https://github.com/Libre-YOLO/libreyolo">GitHub</ExternalRef>
        </span>
      </div>
    </DocLayout>
  )
}
