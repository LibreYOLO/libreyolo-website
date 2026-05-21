'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import Image from 'next/image'
import { Copy, Check, Terminal, BookOpen, Github } from 'lucide-react'

const SETUP_PROMPT = `# Install LibreYOLO (Cursor Hackathon, LibreYOLO Track)
#
# Your job: get LibreYOLO installed from source on the user's machine and
# verify it imports. The user is on Windows, macOS, or Linux. Adapt the
# venv activation line to match. Everything else is identical across OSes.
#
# Prereqs the user must already have:
#   - Python 3.10 or newer  (check: python --version  or  python3 --version)
#   - git
# If either is missing, stop and tell the user. Do not silently install.
#
# If LibreYOLO is already installed and importable in the current environment,
# skip the install steps and jump straight to VERIFY.

# 1. Clone the dev branch (we will move to main later)
git clone -b dev https://github.com/Libre-YOLO/libreyolo.git
cd libreyolo

# 2. Create a virtual environment (stdlib only, no extra tools required)
python -m venv .venv

# 3. Activate it
# macOS / Linux:
source .venv/bin/activate
# Windows PowerShell (run instead of the line above):
# .venv\\Scripts\\Activate.ps1
# Windows cmd.exe:
# .venv\\Scripts\\activate.bat

# 4. Upgrade pip (avoids resolver bugs on older Python installs)
python -m pip install --upgrade pip

# 5. Install LibreYOLO in editable mode
pip install -e .

# 6. VERIFY (should print: LibreYOLO ready)
python -c "from libreyolo import LibreYOLO; print('LibreYOLO ready')"

# OPTIONAL: smoke test on a real image (downloads small weights on first run)
python -c "from libreyolo import LibreYOLO; m = LibreYOLO('LibreYOLO9t.pt'); r = m('https://raw.githubusercontent.com/LibreYOLO/libreyolo/main/libreyolo/assets/parkour.jpg'); print(len(r), 'detections')"

# === FAST PATH (only if the user already has \`uv\` installed) ===============
# Replaces steps 2-5 with a much faster resolver. Skip if uv is not present.
# Do NOT install uv just for this; pip is already fine.
#   uv venv
#   source .venv/bin/activate           # or .venv\\Scripts\\Activate.ps1 on Windows
#   uv pip install -e .

# === GPU NOTES =============================================================
# Step 5 installs the default PyTorch wheel from PyPI:
#   - Linux:   CUDA 12.x build (works with NVIDIA GPU out of the box)
#   - macOS:   CPU + MPS (Apple Silicon GPU works automatically)
#   - Windows: CPU only by default
# Windows + NVIDIA GPU: install a CUDA torch wheel BEFORE step 5:
#   pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
# Hackathon-safe default: the CPU build is fine for small images and YOLO9t.

# === OPTIONAL EXTRAS (install only if your project needs them) =============
# pip install -e ".[onnx]"        # ONNX export + ONNX Runtime inference
# pip install -e ".[rfdetr]"      # RF-DETR transformer flagship
# pip install -e ".[tensorrt]"    # NVIDIA TensorRT (Linux/Windows + CUDA)
# pip install -e ".[openvino]"    # Intel CPU/GPU/VPU acceleration
# pip install -e ".[ncnn]"        # Lightweight CPU/Vulkan deployment

# === IF THINGS BREAK =======================================================
# Windows: "running scripts is disabled on this system" when activating venv:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   (run once, then retry)
# macOS / Linux: "python: command not found":
#   use  python3 -m venv .venv  instead, and  python3 -m pip ... for step 4
# torch wheel mismatch / ImportError:
#   find the right wheel at https://pytorch.org/get-started/locally/
# Anything else: ask the user to paste the full error, then debug.`

function CopyableBlock({ content }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    await navigator.clipboard.writeText(content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="mb-12">
      <div className="flex items-center justify-between mb-3 gap-3">
        <div className="flex items-center gap-2 min-w-0">
          <Terminal className="w-4 h-4 text-surface-500 dark:text-surface-400 shrink-0" />
          <span className="text-xs sm:text-sm font-semibold uppercase tracking-wider text-surface-500 dark:text-surface-400 truncate">
            Setup prompt
          </span>
        </div>
        <button
          onClick={handleCopy}
          className="inline-flex items-center gap-2 px-4 sm:px-5 py-2.5 rounded-lg bg-gradient-to-r from-libre-500 to-libre-600 text-white text-sm font-semibold shadow-sm hover:shadow-md transition-all shrink-0"
          aria-label="Copy setup prompt"
        >
          {copied ? (
            <>
              <Check className="w-4 h-4" />
              Copied!
            </>
          ) : (
            <>
              <Copy className="w-4 h-4" />
              Copy
            </>
          )}
        </button>
      </div>

      <div className="relative code-block rounded-2xl overflow-hidden">
        <pre className="select-all p-4 sm:p-5 text-left text-[13px] sm:text-sm leading-relaxed text-surface-800 dark:text-surface-200 whitespace-pre-wrap font-mono cursor-text overflow-x-auto">
          {content}
        </pre>
      </div>

      <p className="mt-2 text-xs text-surface-500 dark:text-surface-500">
        Click anywhere inside the box to select all, or use the Copy button.
      </p>
    </div>
  )
}

function CoBrandedLockup() {
  return (
    <div className="inline-flex items-center gap-5 sm:gap-7 px-6 sm:px-8 py-4 sm:py-5 rounded-2xl bg-white dark:bg-surface-900 border border-surface-200 dark:border-white/10 shadow-sm">
      {/* Cursor official logo (native ratio ~4.1:1) */}
      <div className="relative h-7 sm:h-8 w-[115px] sm:w-[131px]">
        <Image
          src="/cursor-logo.png"
          alt="Cursor"
          fill
          className="object-contain dark:invert"
          sizes="(min-width: 640px) 131px, 115px"
          priority
        />
      </div>

      <span className="text-2xl sm:text-3xl text-surface-300 dark:text-surface-700 font-light leading-none">×</span>

      {/* LibreYOLO mark */}
      <div className="flex items-center gap-2.5">
        <div className="relative w-8 h-8 sm:w-9 sm:h-9">
          <Image
            src="/logo.png"
            alt="LibreYOLO"
            fill
            className="object-contain dark:invert"
            sizes="36px"
          />
        </div>
        <span className="text-xl sm:text-2xl font-semibold tracking-tight">
          <span className="text-surface-900 dark:text-white">Libre</span>
          <span className="text-libre-500 dark:text-libre-400">YOLO</span>
        </span>
      </div>
    </div>
  )
}

export default function CursorHackathon() {
  return (
    <div className="pt-24 lg:pt-32 pb-20">
      <div className="max-w-3xl mx-auto px-5 sm:px-6 lg:px-8">

        {/* Big-ass title (Cursor theme: black, geometric, uppercase) */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-10"
        >
          <h1 className="text-4xl sm:text-6xl lg:text-7xl font-black uppercase tracking-tight text-surface-900 dark:text-white leading-[0.95] mb-4">
            Cursor Madrid<br />Hackathon 3
          </h1>
          <p className="text-xl sm:text-2xl font-semibold text-surface-600 dark:text-surface-300 tracking-tight">
            LibreYOLO Track
          </p>
        </motion.div>

        {/* Co-branded lockup */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.05 }}
          className="text-center mb-12"
        >
          <div className="flex justify-center">
            <CoBrandedLockup />
          </div>
        </motion.div>

        {/* Single copy block */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <CopyableBlock content={SETUP_PROMPT} />
        </motion.div>

        {/* Resources */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="grid sm:grid-cols-2 gap-3 mb-12"
        >
          <a
            href="https://github.com/Libre-YOLO/libreyolo/tree/dev"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-3 px-4 py-3 rounded-xl bg-surface-50 dark:bg-surface-900/60 border border-surface-200 dark:border-white/10 hover:border-libre-500/40 transition-colors"
          >
            <Github className="w-5 h-5 text-surface-600 dark:text-surface-400 shrink-0" />
            <div className="min-w-0">
              <div className="text-sm font-semibold text-surface-800 dark:text-white">Dev branch</div>
              <div className="text-xs text-surface-500 truncate">github.com/Libre-YOLO/libreyolo</div>
            </div>
          </a>
          <a
            href="/docs"
            className="flex items-center gap-3 px-4 py-3 rounded-xl bg-surface-50 dark:bg-surface-900/60 border border-surface-200 dark:border-white/10 hover:border-libre-500/40 transition-colors"
          >
            <BookOpen className="w-5 h-5 text-surface-600 dark:text-surface-400 shrink-0" />
            <div className="min-w-0">
              <div className="text-sm font-semibold text-surface-800 dark:text-white">Documentation</div>
              <div className="text-xs text-surface-500 truncate">API reference, training, export</div>
            </div>
          </a>
        </motion.div>

      </div>
    </div>
  )
}
