'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import Image from 'next/image'
import {
  Sparkles,
  Terminal,
  Code2,
  Github,
  Menu,
  X,
  Copy,
  Check,
  ExternalLink,
  Rocket,
  Scissors,
  PersonStanding,
  Video,
  Activity,
} from 'lucide-react'

const SECTIONS = [
  { id: 'overview', title: 'Overview', icon: Rocket },
  { id: 'what-is-libreyolo', title: 'What is LibreYOLO', icon: Sparkles },
  { id: 'setup-tutorial', title: 'Setup tutorial', icon: Terminal },
  { id: 'example-detection', title: 'Example 1', icon: Code2 },
  { id: 'example-segmentation', title: 'Example 2', icon: Scissors },
  { id: 'example-keypoints', title: 'Example 3', icon: PersonStanding },
  { id: 'example-video', title: 'Example 4', icon: Video },
  { id: 'example-tracking', title: 'Example 5', icon: Activity },
]

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
python -c "from libreyolo import LibreYOLO; m = LibreYOLO('LibreYOLO9t.pt'); r = m('https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg'); print(len(r), 'detections')"

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

const EXAMPLE_DETECTION_CODE = `from libreyolo import LibreYOLO, SAMPLE_IMAGE

# One factory, any architecture. Auto-detects family, size, and classes.
model = LibreYOLO("LibreYOLO9t.pt")

# Accepts file paths, URLs, PIL, NumPy, tensors, or raw bytes.
result = model(SAMPLE_IMAGE, save=True)

print(result.boxes.xyxy)        # (N, 4) tensor of bounding boxes
print(result.boxes.conf)        # (N,) confidence scores
print(result.names[int(result.boxes.cls[0].item())])  # first class name
print(result.saved_path)        # where the annotated image was saved
`

const RFDETR_EXTRA_CODE = `# Required once for RF-DETR examples
pip install -e ".[rfdetr]"
`

const EXAMPLE_SEGMENTATION_CODE = `from libreyolo import LibreYOLO, SAMPLE_IMAGE

# RF-DETR is the transformer flagship. The "-seg" suffix tells the factory
# to load the segmentation head. Same call shape as detection.
model = LibreYOLO("LibreRFDETRs-seg.pt")

# save=True draws boxes plus translucent mask overlays on top of the image.
result = model(SAMPLE_IMAGE, save=True)

# Boxes still work the same as in detection
print(result.boxes.xyxy)            # (N, 4) bounding boxes
print(result.boxes.cls)             # (N,) class IDs

# Masks are the new bit
print(result.masks.data.shape)      # (N, H, W) binary masks at image resolution
print(result.masks.xy[0].shape)     # polygon contour for the first instance
print(result.saved_path)            # annotated output path
`

const EXAMPLE_KEYPOINTS_CODE = `from libreyolo import LibreYOLO, SAMPLE_IMAGE

# YOLO-NAS pose predicts one person box plus 17 COCO keypoints per person.
model = LibreYOLO("LibreYOLONASs-pose.pt")
result = model(SAMPLE_IMAGE, save=True)

print(result.boxes.xyxy)             # (N, 4) person boxes
print(result.keypoints.xy.shape)     # (N, 17, 2) pixel coordinates
print(result.keypoints.conf.shape)   # (N, 17) keypoint confidence
if len(result):
    print(result.keypoints.xy[0, 0])  # first person's nose keypoint
print(result.saved_path)             # annotated output path
`

const EXAMPLE_VIDEO_CODE = `from libreyolo import LibreYOLO

# Pick one model:
model = LibreYOLO("LibreYOLO9t.pt")          # detection
# model = LibreYOLO("LibreRFDETRs-seg.pt")   # segmentation
# model = LibreYOLO("LibreYOLONASs-pose.pt") # keypoints

for frame in model("clip.mp4", stream=True, save=True):
    print(frame.frame_idx, len(frame))
`

const TRACKING_EXTRA_CODE = `pip install libreyolo[tracking]
`

const EXAMPLE_TRACKING_CODE = `from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9t.pt")

for result in model.track(
    "clip.mp4",
    track_conf=0.25,
    iou=0.45,
    save=True,      # writes runs/track/<video_stem>.mp4 by default
    vid_stride=1,
):
    print(result.frame_idx, result.track_id)
`

function CopyButton({ value, label = 'Copy' }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    await navigator.clipboard.writeText(value)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <button
      onClick={handleCopy}
      className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-libre-500 to-libre-600 text-white text-sm font-semibold shadow-sm hover:shadow-md transition-all shrink-0"
      aria-label={label}
    >
      {copied ? (
        <>
          <Check className="w-4 h-4" />
          Copied!
        </>
      ) : (
        <>
          <Copy className="w-4 h-4" />
          {label}
        </>
      )}
    </button>
  )
}

function CodeBlockCard({ icon: Icon, label, content, copyLabel = 'Copy' }) {
  return (
    <div>
      <div className="flex items-center justify-between mb-3 gap-3">
        <div className="flex items-center gap-2 min-w-0">
          <Icon className="w-4 h-4 text-surface-500 dark:text-surface-400 shrink-0" />
          <span className="text-xs sm:text-sm font-semibold uppercase tracking-wider text-surface-500 dark:text-surface-400 truncate">
            {label}
          </span>
        </div>
        <CopyButton value={content} label={copyLabel} />
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

function SectionHeading({ id, icon: Icon, children }) {
  return (
    <h2
      id={id}
      className="scroll-mt-28 lg:scroll-mt-32 flex items-center gap-3 text-2xl sm:text-3xl font-black uppercase tracking-tight text-surface-900 dark:text-white mb-5"
    >
      <span className="inline-flex items-center justify-center w-10 h-10 rounded-xl bg-libre-500/10 border border-libre-500/20 shrink-0">
        <Icon className="w-5 h-5 text-libre-600 dark:text-libre-400" />
      </span>
      {children}
    </h2>
  )
}

function Sidebar({ activeSection, onNavigate }) {
  return (
    <nav>
      <div className="flex items-center gap-2 mb-6 px-3">
        <Sparkles className="w-5 h-5 text-libre-600 dark:text-libre-400" />
        <span className="text-sm font-semibold text-surface-800 dark:text-white tracking-wide uppercase">
          Cursor Hackathon
        </span>
      </div>

      <ul className="space-y-0.5 mb-4">
        {SECTIONS.map(({ id, title, icon: Icon }) => {
          const isActive = activeSection === id
          return (
            <li key={id}>
              <button
                onClick={() => onNavigate(id)}
                className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200 text-left ${
                  isActive
                    ? 'text-libre-600 dark:text-libre-400 bg-libre-500/10'
                    : 'text-surface-600 dark:text-surface-400 hover:text-surface-900 dark:hover:text-surface-200 hover:bg-surface-100 dark:hover:bg-white/[0.04]'
                }`}
              >
                <Icon
                  className={`w-4 h-4 shrink-0 ${
                    isActive ? 'text-libre-600 dark:text-libre-400' : 'text-surface-400 dark:text-surface-600'
                  }`}
                />
                {title}
              </button>
            </li>
          )
        })}
      </ul>

      <div className="border-t border-surface-200 dark:border-white/[0.06] pt-4 px-3">
        <div className="text-[11px] font-semibold uppercase tracking-wide text-surface-500 dark:text-surface-500 mb-2">
          External
        </div>
        <a
          href="https://github.com/Libre-YOLO/libreyolo"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center justify-between gap-2 px-2 py-2 rounded-lg text-sm font-medium text-surface-600 dark:text-surface-400 hover:text-surface-900 dark:hover:text-white hover:bg-surface-100 dark:hover:bg-white/[0.04] transition-colors"
        >
          <span className="flex items-center gap-2.5">
            <Github className="w-4 h-4 text-surface-400 dark:text-surface-600" />
            Repo
          </span>
          <ExternalLink className="w-3.5 h-3.5 text-surface-400" />
        </a>
      </div>
    </nav>
  )
}

export default function CursorHackathon() {
  const [activeSection, setActiveSection] = useState(SECTIONS[0].id)
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  useEffect(() => {
    const handleScroll = () => {
      const threshold = window.innerHeight * 0.3
      let current = SECTIONS[0].id

      for (const { id } of SECTIONS) {
        const el = document.getElementById(id)
        if (el && el.getBoundingClientRect().top <= threshold) {
          current = id
        }
      }

      setActiveSection(current)
    }

    window.addEventListener('scroll', handleScroll, { passive: true })
    handleScroll()

    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  const navigateTo = (id) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })
    setMobileMenuOpen(false)
  }

  return (
    <div className="flex min-h-screen">
      {/* Desktop sidebar */}
      <aside className="hidden lg:block fixed left-0 top-20 bottom-0 w-64 border-r border-surface-200 dark:border-white/[0.06] bg-white/80 dark:bg-surface-950/50 backdrop-blur-sm overflow-y-auto py-8 px-4 z-30">
        <Sidebar activeSection={activeSection} onNavigate={navigateTo} />
      </aside>

      {/* Mobile sidebar toggle */}
      <button
        onClick={() => setMobileMenuOpen(true)}
        className="lg:hidden fixed bottom-6 right-6 z-40 w-12 h-12 rounded-full bg-libre-500 text-white shadow-lg shadow-libre-500/30 flex items-center justify-center hover:bg-libre-400 transition-colors"
        aria-label="Open navigation"
      >
        <Menu className="w-5 h-5" />
      </button>

      {/* Mobile sidebar overlay */}
      <AnimatePresence>
        {mobileMenuOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 lg:hidden"
              onClick={() => setMobileMenuOpen(false)}
            />
            <motion.aside
              initial={{ x: -280 }}
              animate={{ x: 0 }}
              exit={{ x: -280 }}
              transition={{ type: 'spring', damping: 25, stiffness: 250 }}
              className="fixed left-0 top-0 bottom-0 w-72 bg-white dark:bg-surface-950 border-r border-surface-200 dark:border-white/[0.06] z-50 lg:hidden overflow-y-auto py-6 px-4"
            >
              <div className="flex items-center justify-between mb-4 px-3">
                <span className="text-sm font-semibold text-surface-800 dark:text-white tracking-wide uppercase">
                  Hackathon
                </span>
                <button
                  onClick={() => setMobileMenuOpen(false)}
                  className="p-1.5 rounded-lg text-surface-500 dark:text-surface-400 hover:text-surface-900 dark:hover:text-white hover:bg-surface-100 dark:hover:bg-white/[0.06] transition-colors"
                  aria-label="Close navigation"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              <Sidebar activeSection={activeSection} onNavigate={navigateTo} />
            </motion.aside>
          </>
        )}
      </AnimatePresence>

      {/* Main content */}
      <main className="flex-1 lg:ml-64 min-h-screen pt-28 lg:pt-32 pb-24 px-5 sm:px-6 lg:px-12">
        <div className="max-w-3xl mx-auto">

          {/* ────────── OVERVIEW (hero) ────────── */}
          <section id="overview" className="scroll-mt-28 lg:scroll-mt-32 mb-16">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-center"
            >
              <h1 className="text-4xl sm:text-6xl lg:text-7xl font-black uppercase tracking-tight text-surface-900 dark:text-white leading-[0.95] mb-4">
                Cursor Madrid<br />Hackathon 3
              </h1>
              <p className="text-xl sm:text-2xl font-semibold text-surface-600 dark:text-surface-300 tracking-tight mb-8">
                LibreYOLO Track
              </p>

              <div className="flex justify-center mb-8">
                <CoBrandedLockup />
              </div>

              <p className="text-base sm:text-lg text-surface-600 dark:text-surface-400 max-w-xl mx-auto">
                Welcome, hackers. Read what LibreYOLO is, follow the setup
                tutorial, copy the example, and start building.
              </p>
            </motion.div>
          </section>

          {/* ────────── WHAT IS LIBREYOLO ────────── */}
          <section className="mb-16">
            <SectionHeading id="what-is-libreyolo" icon={Sparkles}>
              What is LibreYOLO
            </SectionHeading>

            <p className="text-base sm:text-lg text-surface-700 dark:text-surface-300 leading-relaxed mb-8">
              LibreYOLO is a modern, 100% MIT-licensed engine for training and
              deploying state-of-the-art object detection. It exists to make
              YOLO accessible again, the way its creators always intended.
            </p>

            <div className="relative max-w-lg mx-auto">
              <div className="absolute -inset-1 bg-gradient-to-r from-emerald-500/20 via-libre-500/20 to-emerald-500/20 rounded-2xl blur-xl opacity-40" />
              <div className="relative bg-surface-50 dark:bg-surface-900/80 backdrop-blur-sm border border-surface-200 dark:border-emerald-500/20 rounded-2xl overflow-hidden">
                <div className="flex items-center gap-2 px-4 py-3 bg-surface-100 dark:bg-surface-900/50 border-b border-surface-200 dark:border-white/5">
                  <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                  <span className="text-surface-500 text-sm font-mono">parkour_result.jpg</span>
                </div>
                <div className="p-3">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src="https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour_result.jpg"
                    alt="LibreYOLO detection result"
                    className="rounded-lg w-full"
                  />
                  <div className="mt-3 flex items-center justify-between text-xs">
                    <span className="text-emerald-600 dark:text-emerald-400 font-mono">&check; Detected 1 object (person)</span>
                    <span className="text-surface-500 font-mono">0.023s</span>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* ────────── SETUP TUTORIAL ────────── */}
          <section className="mb-16">
            <SectionHeading id="setup-tutorial" icon={Terminal}>
              Setup tutorial
            </SectionHeading>

            <p className="text-base text-surface-600 dark:text-surface-400 leading-relaxed mb-6">
              Paste the prompt below into Cursor. It walks the agent through
              installing LibreYOLO from source and verifying the install. Works
              on Windows, macOS, and Linux.
            </p>

            <CodeBlockCard
              icon={Terminal}
              label="Setup prompt"
              content={SETUP_PROMPT}
            />
          </section>

          {/* ────────── EXAMPLE 1: DETECTION ────────── */}
          <section className="mb-16">
            <SectionHeading id="example-detection" icon={Code2}>
              Example 1: Object detection
            </SectionHeading>

            <p className="text-base text-surface-600 dark:text-surface-400 leading-relaxed mb-6">
              A minimal end-to-end detection example with the lightweight
              YOLO9t flagship. Once LibreYOLO is installed, paste this into a
              Python file or a notebook to confirm everything works and to see
              the shape of the results object.
            </p>

            <CodeBlockCard
              icon={Code2}
              label="detect.py"
              content={EXAMPLE_DETECTION_CODE}
              copyLabel="Copy example"
            />
          </section>

          {/* ────────── EXAMPLE 2: SEGMENTATION ────────── */}
          <section className="mb-16">
            <SectionHeading id="example-segmentation" icon={Scissors}>
              Example 2: Segmentation with RF-DETR
            </SectionHeading>

            <p className="text-base text-surface-600 dark:text-surface-400 leading-relaxed mb-6">
              Instance segmentation with the RF-DETR transformer flagship. The
              <code className="mx-1 px-1.5 py-0.5 rounded bg-surface-100 dark:bg-white/[0.06] text-libre-700 dark:text-libre-300 font-mono text-[0.9em]">-seg</code>
              suffix tells the factory to load the segmentation head, so you
              get bounding boxes plus per-instance binary masks from the same
              call.
            </p>

            <div className="mb-6">
              <CodeBlockCard
                icon={Terminal}
                label="Install RF-DETR extra"
                content={RFDETR_EXTRA_CODE}
                copyLabel="Copy install"
              />
            </div>

            <CodeBlockCard
              icon={Scissors}
              label="segment.py"
              content={EXAMPLE_SEGMENTATION_CODE}
              copyLabel="Copy example"
            />
          </section>

          {/* â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ EXAMPLE 3: KEYPOINTS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
          <section className="mb-16">
            <SectionHeading id="example-keypoints" icon={PersonStanding}>
              Example 3: Human keypoints
            </SectionHeading>

            <p className="text-base text-surface-600 dark:text-surface-400 leading-relaxed mb-6">
              Human pose estimation with YOLO-NAS pose. The
              <code className="mx-1 px-1.5 py-0.5 rounded bg-surface-100 dark:bg-white/[0.06] text-libre-700 dark:text-libre-300 font-mono text-[0.9em]">-pose</code>
              suffix loads the keypoint head, returning person boxes plus 17
              COCO keypoints for each detected person.
            </p>

            <CodeBlockCard
              icon={PersonStanding}
              label="keypoints.py"
              content={EXAMPLE_KEYPOINTS_CODE}
              copyLabel="Copy example"
            />
          </section>

          {/* Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬ EXAMPLE 4: VIDEO Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬ */}
          <section className="mb-16">
            <SectionHeading id="example-video" icon={Video}>
              Example 4: Video inference
            </SectionHeading>

            <p className="text-base text-surface-600 dark:text-surface-400 leading-relaxed mb-6">
              Run the same LibreYOLO call on a video. Swap one model line to do
              detection, segmentation, or keypoints; RF-DETR segmentation still
              needs the <code className="mx-1 px-1.5 py-0.5 rounded bg-surface-100 dark:bg-white/[0.06] text-libre-700 dark:text-libre-300 font-mono text-[0.9em]">rfdetr</code>
              extra from Example 2.
            </p>

            <CodeBlockCard
              icon={Video}
              label="video.py"
              content={EXAMPLE_VIDEO_CODE}
              copyLabel="Copy example"
            />
          </section>

          {/* EXAMPLE 5: TRACKING */}
          <section className="mb-16">
            <SectionHeading id="example-tracking" icon={Activity}>
              Example 5: Object tracking
            </SectionHeading>

            <p className="text-base text-surface-600 dark:text-surface-400 leading-relaxed mb-6">
              ByteTrack adds stable IDs across video frames. This is the fastest
              path from detection to people counting, sports clips, traffic
              analysis, or anything that needs to know whether the same object
              is still on screen.
            </p>

            <div className="mb-6">
              <CodeBlockCard
                icon={Terminal}
                label="Install tracking extra"
                content={TRACKING_EXTRA_CODE}
                copyLabel="Copy install"
              />
            </div>

            <CodeBlockCard
              icon={Activity}
              label="track.py"
              content={EXAMPLE_TRACKING_CODE}
              copyLabel="Copy example"
            />

            <div className="mt-8 flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
              <a
                href="https://github.com/Libre-YOLO/libreyolo"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center justify-center gap-2 px-5 py-3 rounded-xl bg-white dark:bg-white/5 hover:bg-surface-100 dark:hover:bg-white/10 border border-surface-300 dark:border-white/10 text-surface-800 dark:text-white font-medium transition-all shadow-sm dark:shadow-none"
              >
                <Github className="w-5 h-5 text-libre-500 dark:text-libre-400" />
                Open the repo
              </a>
              <a
                href="/docs"
                className="inline-flex items-center justify-center gap-2 px-5 py-3 rounded-xl bg-white dark:bg-white/5 hover:bg-surface-100 dark:hover:bg-white/10 border border-surface-300 dark:border-white/10 text-surface-800 dark:text-white font-medium transition-all shadow-sm dark:shadow-none"
              >
                <Code2 className="w-5 h-5 text-libre-500 dark:text-libre-400" />
                Read the full docs
              </a>
            </div>
          </section>

        </div>
      </main>
    </div>
  )
}
