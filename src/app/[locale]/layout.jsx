import '../globals.css'
import Navbar from '@/components/Navbar'
import Footer from '@/components/Footer'
import { Outfit, JetBrains_Mono } from 'next/font/google'
import { Analytics } from '@vercel/analytics/next'
import { NextIntlClientProvider, hasLocale } from 'next-intl'
import { getTranslations, setRequestLocale } from 'next-intl/server'
import { notFound } from 'next/navigation'
import { routing, localeHtmlLang } from '@/i18n/routing'
import { buildAlternates, ogLocale, SITE_URL, OG_IMAGE } from '@/i18n/metadata'

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

export function generateStaticParams() {
  return routing.locales.map((locale) => ({ locale }))
}

export async function generateMetadata({ params }) {
  const { locale } = await params
  const t = await getTranslations({ locale, namespace: 'Metadata' })

  return {
    metadataBase: new URL(SITE_URL),
    title: {
      default: t('defaultTitle'),
      template: `%s | LibreYOLO`,
    },
    description: t('description'),
    keywords: t('keywords').split('|'),
    authors: [{ name: 'LibreYOLO Team' }],
    alternates: buildAlternates('/', locale),
    openGraph: {
      title: t('defaultTitle'),
      description: t('description'),
      url: locale === 'zh' ? `${SITE_URL}/zh` : SITE_URL,
      siteName: 'LibreYOLO',
      locale: ogLocale(locale),
      type: 'website',
      images: [OG_IMAGE],
    },
    twitter: {
      card: 'summary_large_image',
      title: t('defaultTitle'),
      description: t('twitterDescription'),
      images: [OG_IMAGE.url],
    },
    icons: {
      icon: '/favicon.svg',
    },
  }
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

export default async function LocaleLayout({ children, params }) {
  const { locale } = await params
  if (!hasLocale(routing.locales, locale)) {
    notFound()
  }
  setRequestLocale(locale)

  return (
    <html
      lang={localeHtmlLang[locale]}
      className={`${outfit.variable} ${jetbrainsMono.variable}`}
      suppressHydrationWarning
    >
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
        <NextIntlClientProvider>
          <div className="min-h-screen flex flex-col font-sans">
            <Navbar />
            <main className="flex-1">
              {children}
            </main>
            <Footer />
          </div>
          <Analytics />
        </NextIntlClientProvider>
      </body>
    </html>
  )
}
