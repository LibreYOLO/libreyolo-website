import './globals.css'
import Navbar from '@/components/Navbar'
import Footer from '@/components/Footer'
import { Outfit, JetBrains_Mono } from 'next/font/google'
import { Analytics } from '@vercel/analytics/next'

const outfit = Outfit({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-outfit',
  weight: ['400', '500', '600', '700'],
})

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-jetbrains',
  weight: ['400', '500'],
})

export const metadata = {
  metadataBase: new URL('https://libreyolo.com'),
  title: {
    default: 'LibreYOLO | MIT-Licensed Object Detection',
    template: '%s | LibreYOLO',
  },
  description: 'The MIT-licensed training and inference engine for state-of-the-art YOLO models. Built for commercial applications, free from AGPL restrictions.',
  keywords: ['YOLO', 'object detection', 'MIT license', 'machine learning', 'computer vision', 'open source', 'AI'],
  authors: [{ name: 'LibreYOLO Team' }],
  alternates: {
    canonical: './',
  },
  openGraph: {
    title: 'LibreYOLO | MIT-Licensed Object Detection',
    description: 'The MIT-licensed training and inference engine for state-of-the-art YOLO models. Built for commercial applications, free from AGPL restrictions.',
    url: 'https://libreyolo.com',
    siteName: 'LibreYOLO',
    locale: 'en_US',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'LibreYOLO | MIT-Licensed Object Detection',
    description: 'The MIT-licensed training and inference engine for state-of-the-art YOLO models.',
  },
  icons: {
    icon: '/favicon.svg',
  },
}

const jsonLd = {
  '@context': 'https://schema.org',
  '@type': 'SoftwareApplication',
  name: 'LibreYOLO',
  description: 'MIT-licensed training and inference engine for state-of-the-art YOLO object detection models.',
  url: 'https://libreyolo.com',
  applicationCategory: 'DeveloperApplication',
  operatingSystem: 'Windows, macOS, Linux',
  license: 'https://opensource.org/license/mit',
  offers: {
    '@type': 'Offer',
    price: '0',
    priceCurrency: 'USD',
  },
  sameAs: [
    'https://github.com/Libre-YOLO/libreyolo',
    'https://pypi.org/project/libreyolo/',
    'https://huggingface.co/LibreYOLO',
  ],
}

export default function RootLayout({ children }) {
  return (
    <html lang="en" className={`${outfit.variable} ${jetbrainsMono.variable}`} suppressHydrationWarning>
      <head>
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
        />
        <script dangerouslySetInnerHTML={{ __html: `
          (function() {
            var t = localStorage.getItem('theme');
            if (t === 'dark') {
              document.documentElement.classList.add('dark');
            }
          })();
        `}} />
      </head>
      <body>
        <div className="min-h-screen flex flex-col font-sans">
          <Navbar />
          <main className="flex-1">
            {children}
          </main>
          <Footer />
        </div>
        <Analytics />
      </body>
    </html>
  )
}

