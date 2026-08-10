import Image from 'next/image'
import { useTranslations } from 'next-intl'
import { ExternalLink, Linkedin } from 'lucide-react'
import { Link } from '@/i18n/navigation'
import { GithubIcon, RedditIcon } from './BrandIcons'
import { GITHUB_URL, REDDIT_URL } from '@/lib/links'

export default function Footer() {
  const t = useTranslations('Footer')
  return (
    <footer className="relative border-t border-surface-200 dark:border-white/5 bg-surface-50 dark:bg-surface-950">
      <div className="max-w-7xl mx-auto px-6 lg:px-8 py-12 lg:py-16">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-10">
          {/* Brand */}
          <div className="md:col-span-2">
            <Link href="/" className="flex items-center gap-3 mb-4">
              <div className="relative w-7 h-7">
                <Image
                  src="/logo.png"
                  alt="LibreYOLO"
                  fill
                  className="object-contain dark:invert"
                  sizes="28px"
                />
              </div>
              <span className="text-lg font-semibold">
                <span className="text-surface-900 dark:text-white">Libre</span>
                <span className="text-libre-500 dark:text-libre-400">YOLO</span>
              </span>
            </Link>
            <p className="text-surface-500 dark:text-surface-400 text-sm leading-relaxed max-w-md">
              {t('tagline')}
            </p>
            <div className="flex items-center gap-4 mt-6">
              <a
                href={GITHUB_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="text-surface-500 dark:text-surface-400 hover:text-libre-500 transition-colors"
                aria-label="GitHub"
              >
                <GithubIcon className="w-5 h-5" />
              </a>
              <a
                href={REDDIT_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="text-surface-500 dark:text-surface-400 hover:text-[#FF4500] transition-colors"
                aria-label={t('community')}
              >
                <RedditIcon className="w-5 h-5" />
              </a>
            </div>
          </div>

          {/* Links */}
          <div>
            <h4 className="text-surface-900 dark:text-white font-semibold mb-4">{t('resources')}</h4>
            <ul className="space-y-3">
              <li>
                <Link href="/docs" className="text-surface-500 dark:text-surface-400 hover:text-libre-500 dark:hover:text-libre-400 text-sm transition-colors">

                  {t('documentation')}
                </Link>
              </li>
              <li>
                <Link href="/docs/librevlm" className="text-surface-500 dark:text-surface-400 hover:text-libre-500 dark:hover:text-libre-400 text-sm transition-colors">
                  LibreVLM
                </Link>
              </li>
              <li>
                <Link href="/docs/experimental" className="text-surface-500 dark:text-surface-400 hover:text-libre-500 dark:hover:text-libre-400 text-sm transition-colors">
                  {t('experimental')}
                </Link>
              </li>
              <li>
                <Link href="/models" className="text-surface-500 dark:text-surface-400 hover:text-libre-500 dark:hover:text-libre-400 text-sm transition-colors">
                  {t('modelZoo')}
                </Link>
              </li>
              <li>
                <a
                  href="https://huggingface.co/spaces/LibreYOLO/libreyolo-demo"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-surface-500 dark:text-surface-400 hover:text-libre-500 dark:hover:text-libre-400 text-sm transition-colors inline-flex items-center gap-1"
                >
                  {t('liveDemo')} <ExternalLink className="w-3 h-3" />
                </a>
              </li>
              <li>
                <a
                  href="https://www.visionanalysis.org/?utm_source=libreyolo&utm_medium=referral&utm_campaign=benchmarks"
                  target="_blank"
                  rel="noopener"
                  referrerPolicy="strict-origin-when-cross-origin"
                  className="text-surface-500 dark:text-surface-400 hover:text-libre-500 dark:hover:text-libre-400 text-sm transition-colors inline-flex items-center gap-1"
                >
                  {t('benchmarks')} <ExternalLink className="w-3 h-3" />
                </a>
              </li>
              <li>
                <Link href="/commercial" className="text-surface-500 dark:text-surface-400 hover:text-libre-500 dark:hover:text-libre-400 text-sm transition-colors">
                  {t('commercialGuide')}
                </Link>
              </li>
              <li>
                <Link href="/articles" className="text-surface-500 dark:text-surface-400 hover:text-libre-500 dark:hover:text-libre-400 text-sm transition-colors">
                  {t('articles')}
                </Link>
              </li>
              <li>
                <a
                  href={REDDIT_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-surface-500 dark:text-surface-400 hover:text-libre-500 dark:hover:text-libre-400 text-sm transition-colors inline-flex items-center gap-1"
                >
                  {t('community')} <ExternalLink className="w-3 h-3" />
                </a>
              </li>
            </ul>
          </div>

          <div>
            <h4 className="text-surface-900 dark:text-white font-semibold mb-4">{t('contact')}</h4>
            <div className="flex items-center gap-3 mb-4">
              <Image
                src="/photo.jpg"
                alt="Xuban Ceccon"
                width={44}
                height={44}
                className="rounded-full object-cover w-11 h-11"
              />
              <div>
                <p className="text-surface-800 dark:text-surface-200 text-sm font-medium">Xuban Ceccon</p>
                <p className="text-surface-500 dark:text-surface-400 text-xs">{t('role')}</p>
              </div>
            </div>
            <a
              href="https://www.linkedin.com/in/xuban-ceccon"
              target="_blank"
              rel="noopener noreferrer"
              className="text-libre-600 dark:text-libre-400 hover:text-libre-700 dark:hover:text-libre-300 text-sm font-medium transition-colors inline-flex items-center gap-1.5"
            >
              <Linkedin className="w-4 h-4" />
              {t('chat')}
            </a>
            <div className="mt-4">
              <a
                href="https://opensource.org/licenses/MIT"
                target="_blank"
                rel="noopener noreferrer"
                className="text-surface-500 dark:text-surface-400 hover:text-libre-500 dark:hover:text-libre-400 text-sm transition-colors inline-flex items-center gap-1"
              >
                {t('mitLicense')} <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          </div>
        </div>

        <div className="mt-12 pt-8 border-t border-surface-200 dark:border-white/5">
          <p className="text-surface-500 dark:text-surface-400 text-sm text-center md:text-left">
            {t('copyright', { year: new Date().getFullYear() })}
          </p>
        </div>
      </div>
    </footer>
  )
}
