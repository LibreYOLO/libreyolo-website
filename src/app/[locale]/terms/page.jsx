import LegalPage from '@/components/LegalPage'
import { buildPageMetadata } from '@/i18n/metadata'

export function generateMetadata() {
  return buildPageMetadata({
    title: 'Terms of Use',
    description: 'Terms for using the LibreYOLO library, its agent plugins, model weights and libreyolo.com.',
    path: '/terms',
    locale: 'en',
    englishOnly: true,
  })
}

const sections = [
  {
    heading: 'The software',
    body: [
      'The LibreYOLO library and the LibreYOLO agent plugins and skills are free, open-source software released under the MIT License. Your use of them is governed by that license, which you can read at github.com/LibreYOLO/libreyolo/blob/release/LICENSE. These terms do not restrict any right the MIT License gives you.',
    ],
  },
  {
    heading: 'Model weights',
    body: [
      'Pretrained weights are published on huggingface.co/LibreYOLO. Each set of weights keeps the license of its upstream model, which is stated on its model card. Check that license before you redistribute weights or use them commercially.',
    ],
  },
  {
    heading: 'No warranty',
    body: [
      'The software, the weights and this website are provided "as is", without warranty of any kind, as stated in the MIT License. Model predictions can be wrong. You are responsible for checking that results are fit for your purpose, especially in safety-critical, medical, legal or security uses.',
    ],
  },
  {
    heading: 'Acceptable use',
    body: [
      'Use LibreYOLO in line with the law and with the terms of the services you run it on, including the policies of the AI agent or platform that runs the plugin. You are responsible for having the rights to the images, videos and data you process.',
    ],
  },
  {
    heading: 'Third-party services',
    body: [
      'LibreYOLO downloads weights from Hugging Face and is installed from PyPI. The plugins run inside third-party agents such as ChatGPT and Codex. Those services have their own terms, which apply to your use of them.',
    ],
  },
  {
    heading: 'Changes and contact',
    body: [
      'If these terms change, the new version is published on this page with a new date. Questions: open an issue at github.com/LibreYOLO/libreyolo/issues.',
    ],
  },
]

export default function TermsPage() {
  return <LegalPage title="Terms of Use" updated="November 11, 2026" sections={sections} />
}
