import Link from 'next/link'
import Image from 'next/image'
import { Github, ExternalLink } from 'lucide-react'

export default function Footer() {
  return (
    <footer className="relative border-t border-white/5 bg-surface-950">
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
                  className="object-contain invert"
                  sizes="28px"
                />
              </div>
              <span className="text-lg font-semibold">
                <span className="text-white">Libre</span>
                <span className="text-libre-400">YOLO</span>
              </span>
            </Link>
            <p className="text-surface-400 text-sm leading-relaxed max-w-md">
              An independent open-source project providing state-of-the-art object detection.
              Built for developers, researchers, and commercial applications.
            </p>
            <div className="flex items-center gap-4 mt-6">
              <a
                href="https://github.com/Libre-YOLO/libreyolo"
                target="_blank"
                rel="noopener noreferrer"
                className="text-surface-400 hover:text-libre-400 transition-colors"
                aria-label="GitHub"
              >
                <Github className="w-5 h-5" />
              </a>
            </div>
          </div>

          {/* Links */}
          <div>
            <h4 className="text-white font-semibold mb-4">Resources</h4>
            <ul className="space-y-3">
              <li>
                <Link href="/docs" className="text-surface-400 hover:text-libre-400 text-sm transition-colors">
                  Documentation
                </Link>
              </li>
              <li>
                <Link href="/models" className="text-surface-400 hover:text-libre-400 text-sm transition-colors">
                  Model Zoo
                </Link>
              </li>
              <li>
                <Link href="/datasets" className="text-surface-400 hover:text-libre-400 text-sm transition-colors">
                  Dataset Zoo
                </Link>
              </li>
              <li>
                <Link href="/commercial" className="text-surface-400 hover:text-libre-400 text-sm transition-colors">
                  Commercial Guide
                </Link>
              </li>
            </ul>
          </div>

          <div>
            <h4 className="text-white font-semibold mb-4">Legal</h4>
            <ul className="space-y-3">
              <li>
                <a
                  href="https://opensource.org/licenses/MIT"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-surface-400 hover:text-libre-400 text-sm transition-colors inline-flex items-center gap-1"
                >
                  MIT License <ExternalLink className="w-3 h-3" />
                </a>
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-12 pt-8 border-t border-white/5">
          <p className="text-surface-500 text-sm text-center md:text-left">
            © {new Date().getFullYear()} LibreYOLO. An independent open-source project.
          </p>
        </div>
      </div>
    </footer>
  )
}

