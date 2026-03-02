import './globals.css'
import Navbar from '@/components/Navbar'
import Footer from '@/components/Footer'
import { Outfit, JetBrains_Mono } from 'next/font/google'

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
  title: 'LibreYOLO | MIT-Licensed Object Detection',
  description: 'The MIT-licensed training and inference engine for state-of-the-art YOLO models. Built for commercial applications, free from AGPL restrictions.',
  keywords: ['YOLO', 'object detection', 'MIT license', 'machine learning', 'computer vision', 'open source', 'AI'],
  authors: [{ name: 'LibreYOLO Team' }],
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

export default function RootLayout({ children }) {
  return (
    <html lang="en" className={`${outfit.variable} ${jetbrainsMono.variable}`}>
      <body>
        <div className="min-h-screen flex flex-col font-sans">
          <Navbar />
          <main className="flex-1">
            {children}
          </main>
          <Footer />
        </div>
      </body>
    </html>
  )
}

