import LegalPage from '@/components/LegalPage'
import { buildPageMetadata } from '@/i18n/metadata'

export function generateMetadata() {
  return buildPageMetadata({
    title: 'Privacy Policy',
    description: 'What data the LibreYOLO library, its agent plugins and libreyolo.com collect: the library and plugins collect none.',
    path: '/privacy',
    locale: 'en',
    englishOnly: true,
  })
}

const sections = [
  {
    heading: 'Summary',
    body: [
      'LibreYOLO is an open-source Python library. The library and the LibreYOLO plugins for coding agents (for example the LibreYOLO plugin for ChatGPT and Codex) do not collect, store or send any personal data or usage data to the LibreYOLO project.',
    ],
  },
  {
    heading: 'The libreyolo Python package',
    body: [
      'The package runs entirely on your own machine. It has no telemetry and no accounts. Your images, videos, datasets and trained models never leave your machine unless you send them somewhere yourself.',
      'When you load a model for the first time, the package downloads its weights from Hugging Face (huggingface.co, or the mirror set in the HF_ENDPOINT environment variable). That download is a request to Hugging Face and is covered by the Hugging Face privacy policy. Installing the package from PyPI is covered by the PyPI privacy policy.',
    ],
  },
  {
    heading: 'Agent plugins and skills',
    body: [
      'The LibreYOLO plugins contain instructions (skills) that tell an AI agent how to use the libreyolo package. They have no server of their own. The agent runs libreyolo commands on your machine or in your sandbox.',
      'Your conversations with the agent are handled by the agent provider (for example OpenAI for ChatGPT and Codex) under that provider\'s privacy policy. The LibreYOLO project receives none of it.',
    ],
  },
  {
    heading: 'This website',
    body: [
      'libreyolo.com is hosted on Vercel and uses Vercel Web Analytics to count page views. It does not use cookies for analytics and does not identify individual visitors; we see aggregated numbers such as page views, referring sites and countries. We also use Google Search Console and Bing Webmaster Tools, which report aggregated search statistics.',
      'The website has no accounts, forms or newsletter, and we do not sell or share data.',
    ],
  },
  {
    heading: 'GitHub',
    body: [
      'Issues, discussions and pull requests on github.com/LibreYOLO are public and are governed by GitHub\'s privacy statement. Do not post private data, images or credentials there.',
    ],
  },
  {
    heading: 'Changes and contact',
    body: [
      'If this policy changes, the new version is published on this page with a new date. Questions: open an issue at github.com/LibreYOLO/libreyolo/issues.',
    ],
  },
]

export default function PrivacyPage() {
  return <LegalPage title="Privacy Policy" updated="November 11, 2026" sections={sections} />
}
