import { createNavigation } from 'next-intl/navigation'
import { routing } from './routing'

// Locale-aware navigation helpers. <Link href="/models"> automatically resolves
// to `/models` for English and `/zh/models` for Chinese.
export const { Link, redirect, usePathname, useRouter, getPathname } =
  createNavigation(routing)
