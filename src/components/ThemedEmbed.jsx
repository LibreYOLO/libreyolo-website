'use client'

import { useEffect, useState } from 'react'

// Renders iframes embedded in articles. For Vision Analysis widgets (any
// `/embed/` URL) it mirrors the site's current light/dark theme into the widget
// through a `theme` query param, and keeps it in sync if the site theme toggles.
// Any other iframe is passed straight through, unchanged.
export default function ThemedEmbed({ node, src = '', className, style, ...props }) {
  const [theme, setTheme] = useState('light')

  useEffect(() => {
    const read = () =>
      setTheme(document.documentElement.classList.contains('dark') ? 'dark' : 'light')
    read()
    const observer = new MutationObserver(read)
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class'],
    })
    return () => observer.disconnect()
  }, [])

  let finalSrc = src
  if (src && src.includes('/embed/')) {
    try {
      const url = new URL(src, 'https://visionanalysis.org')
      url.searchParams.set('theme', theme)
      finalSrc = url.toString()
    } catch {
      finalSrc = src
    }
  }

  return (
    <iframe
      src={finalSrc}
      className={className ?? 'rounded-xl my-6 w-full'}
      style={style ?? { border: 0, overflow: 'hidden' }}
      {...props}
    />
  )
}
