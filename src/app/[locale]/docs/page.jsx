import { setRequestLocale } from 'next-intl/server'
import Link from 'next/link'

import registry, { DOCS_NAV, DOCS_VERSION } from '@/lib/docs'
import { buildPageMetadata, SITE_URL } from '@/i18n/metadata'
import DocsShell from '@/components/docs/DocsShell'
import Code from '@/components/docs/Code'

/*
 * Docs landing.
 *
 * This is a router, not an essay. Most arrivals come from a search engine with
 * one question, so the page's job is to put the install line and the right
 * onward link in front of them fast, and to establish in one pass that the
 * project is real: how many families, measured against what, under which
 * license.
 *
 * The reference-page austerity still applies. No stat tiles, no cards, no
 * pills. What sells here is the numbers being true and the routing being
 * complete, and every count below is read from the registry rather than typed.
 */

const TASK_BLURBS = {
  '/docs/tasks/object-detection': 'Boxes and classes. The default task, and the one most families serve.',
  '/docs/tasks/instance-segmentation': 'Per-object masks, one mask per detection.',
  '/docs/tasks/semantic-segmentation': 'A class for every pixel, without separating objects.',
  '/docs/tasks/panoptic-segmentation': 'Semantic and instance segmentation in one output.',
  '/docs/tasks/pose-estimation': 'Keypoints per object, top-down or bottom-up.',
  '/docs/tasks/image-classification': 'One label per image, including zero-shot with CLIP and SigLIP2.',
  '/docs/tasks/oriented-detection': 'Rotated boxes, for aerial and document imagery.',
  '/docs/tasks/point-detection': 'Centroids instead of boxes. Counting, and very small models.',
  '/docs/tasks/depth-estimation': 'Per-pixel depth from a single image.',
  '/docs/tasks/image-restoration': 'Denoising, deblurring and upscaling.',
  '/docs/tasks/background-removal': 'Alpha mattes for cutouts.',
  '/docs/tasks/ocr': 'Text detection and recognition in one pass.',
  '/docs/tasks/face-recognition': 'Face embeddings and identity galleries.',
  '/docs/tasks/gaze-estimation': 'Where a person is looking.',
  '/docs/tasks/object-tracking': 'Identities across video frames, over any detector.',
  '/docs/tasks/open-vocabulary-detection': 'Detect classes you name at runtime, with no training.',
  '/docs/tasks/promptable-segmentation': 'Segment anything you point at, with SAM and friends.',
}

export async function generateMetadata({ params }) {
  const { locale } = await params
  return buildPageMetadata({
    title: 'LibreYOLO documentation',
    description:
      'Install, train, evaluate and export every model in LibreYOLO, the MIT-licensed computer vision library. One API across detection, segmentation, pose, depth, OCR and more.',
    path: '/docs',
    locale,
    englishOnly: true,
  })
}

function Section({ id, title, children, action }) {
  return (
    <section className="mt-14 border-t border-surface-200 pt-6 dark:border-white/[0.09]">
      <div className="mb-4 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <h2 id={id} className="scroll-mt-24 text-[1.35rem] font-semibold tracking-tight text-surface-900 dark:text-white">
          {title}
        </h2>
        {action}
      </div>
      {children}
    </section>
  )
}

/*
 * A routing list: name on the left, one line of what it is on the right.
 *
 * An entry whose page is not written yet still shows its description, because
 * the description is the useful part: it tells a reader the library does this
 * at all. Only the link is withheld. The section index pages say "not written
 * yet" because there the reader is asking about documentation progress; here
 * they are asking what the library does, and a column of that phrase would
 * answer a question nobody asked.
 */
function Routes({ items }) {
  return (
    <dl className="text-[14px]">
      {items.map((item) => (
        <div
          key={item.href || item.label}
          className="flex flex-col gap-x-4 border-b border-surface-200/70 py-2 last:border-0 sm:flex-row dark:border-white/[0.07]"
        >
          <dt className="shrink-0 sm:w-56">
            {item.built === false ? (
              <span className="text-surface-700 dark:text-surface-300">{item.label}</span>
            ) : (
              <Link href={item.href} className="font-medium text-libre-700 underline-offset-2 hover:underline dark:text-libre-400">
                {item.label}
              </Link>
            )}
          </dt>
          <dd className="text-surface-500 dark:text-surface-500">{item.note}</dd>
        </div>
      ))}
    </dl>
  )
}

export default async function DocsLanding({ params }) {
  const { locale } = await params
  setRequestLocale(locale)

  const families = Object.values(registry.families)
  // Library-wide totals are counted by the registry generator, not typed here.
  const lib = registry.library ?? { families: null, tasks: null, export_formats: null }
  const checkpointCount = families.reduce((n, f) => n + f.checkpoints.length, 0)
  const benchmarked = families.filter((f) => f.va_embed)
  const groupOf = (id) => DOCS_NAV.groups.find((g) => g.id === id)?.items ?? []

  const tasks = groupOf('tasks').map((item) => ({
    href: item.slug,
    label: item.label,
    built: item.built,
    note: TASK_BLURBS[item.slug] ?? '',
  }))

  const models = groupOf('models')
    .filter((item) => item.slug !== '/docs/models' && item.built)
    .map((item) => {
      const family = families.find((f) => `/docs/models/${f.slug}` === item.slug)
      if (!family) return { href: item.slug, label: item.label, note: '' }
      const tasks = family.tasks.length === 1 ? 'Detection' : `${family.tasks.length} tasks`
      // A family with nothing in our org is not a broken row, it is a licensing
      // fact. Saying "0 checkpoints" reads as an error and buries the reason.
      const weights = family.weights_hosted
        ? `${family.checkpoints.length} checkpoint${family.checkpoints.length === 1 ? '' : 's'}.`
        : 'weights distributed by their authors, not by us.'
      return { href: item.slug, label: item.label, note: `${tasks}, ${weights}` }
    })

  const jsonLd = {
    '@context': 'https://schema.org',
    '@type': 'TechArticle',
    headline: 'LibreYOLO documentation',
    description: 'Documentation for LibreYOLO, the MIT-licensed computer vision library.',
    mainEntityOfPage: `${SITE_URL}/docs`,
    publisher: { '@type': 'Organization', name: 'LibreYOLO', url: SITE_URL },
  }

  return (
    <>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }} />
      <DocsShell nav={DOCS_NAV} activePath="/docs" version={DOCS_VERSION} showActions={false}>
        <div className="max-w-3xl">
          <h1 className="text-[2.4rem] font-semibold leading-tight tracking-tight text-surface-900 dark:text-white">
            LibreYOLO documentation
          </h1>
          <p className="mt-3 max-w-[60ch] text-[17px] leading-relaxed text-surface-600 dark:text-surface-400">
            One Python API for computer vision: detection, segmentation, pose, depth, OCR and more,
            with training, validation and export for each. The code is MIT licensed, so what you
            build with it stays yours.
          </p>

          <div className="mt-6 max-w-[420px]">
            <Code language="bash" label="install">pip install libreyolo</Code>
          </div>

          <div className="mt-2 flex flex-wrap gap-x-5 gap-y-2 text-[14px]">
            <Link href="/docs/quickstart" className="font-medium text-libre-700 underline-offset-2 hover:underline dark:text-libre-400">
              Quickstart
            </Link>
            <Link href="/docs/models" className="font-medium text-libre-700 underline-offset-2 hover:underline dark:text-libre-400">
              Browse models
            </Link>
            <Link href="/docs/tasks" className="font-medium text-libre-700 underline-offset-2 hover:underline dark:text-libre-400">
              Browse tasks
            </Link>
          </div>

          <dl className="mt-8 flex flex-col gap-y-1 border-t border-surface-200 pt-4 text-[13.5px] dark:border-white/[0.09]">
            <Fact label="Models">
              {lib.families} families, from the current flagships back to the historic detectors,
              behind one factory
            </Fact>
            <Fact label="Tasks">
              {lib.tasks}, from detection to gaze estimation and human mesh recovery
            </Fact>
            <Fact label="Weights">
              {checkpointCount ? `${checkpointCount} published checkpoints for the families documented here, ` : ''}
              downloaded on first use from{' '}
              <a href="https://huggingface.co/LibreYOLO" target="_blank" rel="noopener noreferrer" className="text-libre-700 underline-offset-2 hover:underline dark:text-libre-400">
                huggingface.co/LibreYOLO
              </a>
            </Fact>
            <Fact label="Export">
              {lib.export_formats} formats, ONNX through TensorRT, CoreML, TFLite and Core AI
            </Fact>
            <Fact label="License">
              MIT for the code. Weights carry their upstream license, stated per checkpoint
            </Fact>
          </dl>

          <Section
            id="tasks"
            title="What do you want to do?"
            action={
              <Link href="/docs/tasks" className="text-[13px] text-libre-700 underline-offset-2 hover:underline dark:text-libre-400">
                All tasks
              </Link>
            }
          >
            <Routes items={tasks} />
          </Section>

          <Section
            id="models"
            title="Which model?"
            action={
              <Link href="/docs/models" className="text-[13px] text-libre-700 underline-offset-2 hover:underline dark:text-libre-400">
                All models
              </Link>
            }
          >
            <p className="mb-4 max-w-[68ch] text-[15px] leading-[1.6] text-surface-600 dark:text-surface-400">
              Start with a flagship unless you have a reason not to. Every feature is designed and
              GPU-validated against those first, and each page carries the checkpoints, the export
              matrix and the licensing for that family.
            </p>
            <Routes items={models} />
          </Section>

          {benchmarked.length > 0 && (
            <Section
              id="benchmarks"
              title="Measured, not claimed"
              action={
                <a href="https://www.visionanalysis.org/" target="_blank" rel="noopener noreferrer" className="text-[13px] text-libre-700 underline-offset-2 hover:underline dark:text-libre-400">
                  Vision Analysis
                </a>
              }
            >
              <p className="mb-4 max-w-[68ch] text-[15px] leading-[1.6] text-surface-600 dark:text-surface-400">
                Every accuracy and latency number in these docs comes from a run we publish, with the
                hardware, runtime and precision recorded alongside it. The chart below is live.
              </p>
              <div className="relative w-full" style={{ paddingTop: '62.5%' }}>
                <iframe
                  src="https://www.visionanalysis.org/embed/scatter?title=Detection%20models%20on%20COCO&subtitle=Accuracy%20against%20latency"
                  title="Detection models on COCO, accuracy against latency"
                  loading="lazy"
                  className="absolute inset-0 h-full w-full border border-surface-200 dark:border-white/[0.09]"
                  style={{ border: 0 }}
                />
              </div>
            </Section>
          )}

          <Section id="workflows" title="Working with a model">
            <Routes
              items={[
                ...groupOf('train').slice(0, 1).map((i) => ({ href: i.slug, label: 'Train', built: i.built, note: 'Datasets, arguments, augmentation, multi-GPU and experiment loggers.' })),
                ...groupOf('predict').slice(0, 1).map((i) => ({ href: i.slug, label: 'Predict', built: i.built, note: 'Images, folders, video, webcams and RTSP streams.' })),
                ...groupOf('export').slice(0, 1).map((i) => ({ href: i.slug, label: 'Export and deploy', built: i.built, note: 'Twelve targets, with a support matrix per family.' })),
                ...groupOf('cli').slice(0, 1).map((i) => ({ href: i.slug, label: 'Command line', built: i.built, note: 'Everything the Python API does, without writing Python.' })),
              ]}
            />
          </Section>

          <Section id="licensing" title="On the license">
            <p className="max-w-[68ch] text-[15px] leading-[1.6] text-surface-600 dark:text-surface-400">
              LibreYOLO's code is MIT. It does not require you to open source your application, it
              does not reach your model weights, and it does not change if you sell what you build.
              Pretrained weights are separate: each one carries the license of whoever trained it,
              and the docs state that per checkpoint rather than averaging it into a claim. A few
              are non-commercial, and those say so.
            </p>
            <p className="mt-3 text-[14px]">
              <Link href="/docs/licensing" className="font-medium text-libre-700 underline-offset-2 hover:underline dark:text-libre-400">
                How licensing works here
              </Link>
            </p>
          </Section>

          <footer className="mt-14 border-t border-surface-200 pt-6 text-[13px] text-surface-500 dark:border-white/[0.09] dark:text-surface-500">
            <p>
              These docs describe LibreYOLO v{DOCS_VERSION}. Documentation for earlier releases stays
              available at{' '}
              <Link href="/docs/versions" className="text-libre-700 underline-offset-2 hover:underline dark:text-libre-400">
                /docs/versions
              </Link>
              .
            </p>
          </footer>
        </div>
      </DocsShell>
    </>
  )
}

function Fact({ label, children }) {
  return (
    <div className="flex flex-col gap-x-3 sm:flex-row">
      <dt className="shrink-0 text-surface-500 dark:text-surface-500 sm:w-24">{label}</dt>
      <dd className="min-w-0 text-surface-700 dark:text-surface-300">{children}</dd>
    </div>
  )
}
