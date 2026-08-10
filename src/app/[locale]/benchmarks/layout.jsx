import { buildPageMetadata, SITE_URL } from '@/i18n/metadata'

const TITLE = 'Benchmarks'
const DESCRIPTION =
  'Measured benchmarks for every model LibreYOLO ships: COCO accuracy, RF100-VL transfer across 100 real-world datasets, and latency on real hardware. Nothing copied from a paper.'

export async function generateMetadata({ params }) {
  const { locale } = await params
  // The page copy is not translated yet, so /zh/benchmarks canonicalises to the
  // English URL rather than advertising a translation that does not exist.
  return buildPageMetadata({
    title: TITLE,
    description: DESCRIPTION,
    path: '/benchmarks',
    locale,
    englishOnly: true,
  })
}

// Dataset + measurement provenance for the RF100-VL sweep, so the numbers on
// this page are machine-readable rather than locked inside the visualisation.
function benchmarksJsonLd() {
  return {
    '@context': 'https://schema.org',
    '@type': 'Dataset',
    name: 'LibreYOLO RF100-VL benchmark results',
    description:
      'Per-dataset mAP@50-95 for LibreYOLO models fine-tuned and evaluated on each of the 100 RF100-VL datasets, plus COCO val2017 accuracy and measured latency across hardware.',
    url: `${SITE_URL}/benchmarks`,
    license: 'https://opensource.org/licenses/MIT',
    creator: { '@type': 'Organization', name: 'LibreYOLO', url: SITE_URL },
    isBasedOn: 'https://rf100-vl.org',
    measurementTechnique: 'pycocotools mAP@50-95, per-dataset test splits, maxDets 500',
    variableMeasured: 'mAP@50-95',
  }
}

export default function BenchmarksLayout({ children }) {
  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(benchmarksJsonLd()) }}
      />
      {children}
    </>
  )
}
