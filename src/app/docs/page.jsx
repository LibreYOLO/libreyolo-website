'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  BookOpen, Terminal, Rocket, Layers, Crosshair, Grid3x3,
  GraduationCap, CheckCircle2, Upload, Cpu, FileCode, Wrench,
  Database, Copy, Check, Menu, X, ChevronRight
} from 'lucide-react'

/* ─── Section metadata for sidebar ─── */
const sections = [
  { id: 'introduction', title: 'Introduction', icon: BookOpen },
  { id: 'installation', title: 'Installation', icon: Terminal },
  { id: 'quickstart', title: 'Quickstart', icon: Rocket },
  { id: 'models', title: 'Available Models', icon: Layers },
  { id: 'prediction', title: 'Prediction', icon: Crosshair },
  { id: 'tiled-inference', title: 'Tiled Inference', icon: Grid3x3 },
  { id: 'training', title: 'Training', icon: GraduationCap },
  { id: 'validation', title: 'Validation', icon: CheckCircle2 },
  { id: 'export', title: 'Export', icon: Upload },
  { id: 'onnx-inference', title: 'ONNX Inference', icon: Cpu },
  { id: 'tensorrt-inference', title: 'TensorRT Inference', icon: Cpu },
  { id: 'openvino-inference', title: 'OpenVINO Inference', icon: Cpu },
  { id: 'api-reference', title: 'API Reference', icon: FileCode },
  { id: 'architecture', title: 'Architecture Guide', icon: Wrench },
  { id: 'dataset-format', title: 'Dataset Format', icon: Database },
]

/* ─── Reusable components ─── */

function CodeBlock({ children, language = 'python', filename }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(children)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="relative my-5 rounded-xl overflow-hidden border border-white/[0.06]">
      <div className="flex items-center justify-between px-4 py-2.5 bg-white/[0.03] border-b border-white/[0.06]">
        <div className="flex items-center gap-2">
          <span className="text-[11px] font-mono text-surface-500 uppercase tracking-wider">{language}</span>
          {filename && (
            <>
              <span className="text-surface-700">·</span>
              <span className="text-xs font-mono text-surface-500">{filename}</span>
            </>
          )}
        </div>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1.5 px-2 py-1 rounded-md text-[11px] font-medium text-surface-500 hover:text-surface-300 hover:bg-white/[0.06] transition-all"
        >
          {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <div className="overflow-x-auto bg-[#0a0f1a]">
        <pre className="p-5 text-[13px] leading-relaxed">
          <code className="font-mono text-surface-300">{children}</code>
        </pre>
      </div>
    </div>
  )
}

function DocTable({ headers, rows }) {
  return (
    <div className="my-5 overflow-x-auto rounded-xl border border-white/[0.06]">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-white/[0.08] bg-white/[0.02]">
            {headers.map((h, i) => (
              <th key={i} className="px-4 py-3 text-left font-semibold text-surface-300 whitespace-nowrap">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-b border-white/[0.04] last:border-0 hover:bg-white/[0.02] transition-colors">
              {row.map((cell, j) => (
                <td key={j} className="px-4 py-3 text-surface-400">{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function SectionHeading({ id, icon: Icon, children }) {
  return (
    <div id={id} className="scroll-mt-28 flex items-center gap-3 mb-6 pt-2">
      <div className="w-10 h-10 rounded-xl bg-libre-500/10 border border-libre-500/20 flex items-center justify-center shrink-0">
        <Icon className="w-5 h-5 text-libre-400" />
      </div>
      <h2 className="text-2xl lg:text-3xl font-bold text-white">{children}</h2>
    </div>
  )
}

function SubHeading({ children }) {
  return <h3 className="text-lg font-semibold text-white mt-10 mb-4">{children}</h3>
}

function P({ children }) {
  return <p className="text-surface-400 leading-relaxed mb-4">{children}</p>
}

function InlineCode({ children }) {
  return <code className="px-1.5 py-0.5 rounded bg-white/[0.06] text-libre-300 text-sm font-mono">{children}</code>
}

function Divider() {
  return <div className="border-t border-white/[0.06] my-16" />
}

function FeatureItem({ children }) {
  return (
    <li className="flex items-start gap-3 text-surface-400">
      <ChevronRight className="w-4 h-4 text-libre-400 mt-1 shrink-0" />
      <span>{children}</span>
    </li>
  )
}

/* ─── Sidebar ─── */

function Sidebar({ activeSection, onNavigate, className = '' }) {
  return (
    <nav className={className}>
      <div className="flex items-center gap-2 mb-6 px-3">
        <BookOpen className="w-5 h-5 text-libre-400" />
        <span className="text-sm font-semibold text-white tracking-wide uppercase">Documentation</span>
      </div>
      <ul className="space-y-0.5">
        {sections.map(({ id, title, icon: Icon }) => {
          const isActive = activeSection === id
          return (
            <li key={id}>
              <button
                onClick={() => onNavigate(id)}
                className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200 text-left ${
                  isActive
                    ? 'text-libre-400 bg-libre-500/10'
                    : 'text-surface-400 hover:text-surface-200 hover:bg-white/[0.04]'
                }`}
              >
                <Icon className={`w-4 h-4 shrink-0 ${isActive ? 'text-libre-400' : 'text-surface-600'}`} />
                {title}
              </button>
            </li>
          )
        })}
      </ul>
    </nav>
  )
}

/* ─── Main docs page ─── */

export default function Docs() {
  const [activeSection, setActiveSection] = useState('introduction')
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  // Scroll spy — pick the last section whose heading has scrolled past 30% of viewport
  useEffect(() => {
    const handleScroll = () => {
      const threshold = window.innerHeight * 0.3
      let current = sections[0].id

      for (const { id } of sections) {
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
      <aside className="hidden lg:block fixed left-0 top-20 bottom-0 w-64 border-r border-white/[0.06] bg-surface-950/50 backdrop-blur-sm overflow-y-auto py-8 px-4 z-30">
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
              className="fixed left-0 top-0 bottom-0 w-72 bg-surface-950 border-r border-white/[0.06] z-50 lg:hidden overflow-y-auto py-6 px-4"
            >
              <div className="flex items-center justify-between mb-4 px-3">
                <span className="text-sm font-semibold text-white tracking-wide uppercase">Docs</span>
                <button
                  onClick={() => setMobileMenuOpen(false)}
                  className="p-1.5 rounded-lg text-surface-400 hover:text-white hover:bg-white/[0.06] transition-colors"
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
      <main className="flex-1 lg:ml-64 min-h-screen pt-28 lg:pt-32 pb-24 px-6 lg:px-12">
        <div className="max-w-4xl mx-auto">

          {/* ────────────── INTRODUCTION ────────────── */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
            <SectionHeading id="introduction" icon={BookOpen}>Introduction</SectionHeading>
            <P>
              LibreYOLO is an MIT-licensed object detection library that provides a unified Python API across three architectures: <strong className="text-white">YOLOX</strong>, <strong className="text-white">YOLOv9</strong>, and <strong className="text-white">RF-DETR</strong>. One interface for prediction, training, validation, and export — regardless of which model family you use.
            </P>
            <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLOXs.pt")
results = model("image.jpg", conf=0.25, save=True)
print(results.boxes.xyxy)`}</CodeBlock>

            <SubHeading>Key features</SubHeading>
            <ul className="space-y-2.5 mb-4">
              <FeatureItem>Unified API across YOLOX, YOLOv9, and RF-DETR</FeatureItem>
              <FeatureItem>Auto-detection of model architecture, size, and class count from weights</FeatureItem>
              <FeatureItem>Tiled inference for large/high-resolution images</FeatureItem>
              <FeatureItem>ONNX, TorchScript, TensorRT, and OpenVINO export with embedded metadata</FeatureItem>
              <FeatureItem>ONNX Runtime, TensorRT, and OpenVINO inference backends</FeatureItem>
              <FeatureItem>COCO-compatible validation with mAP metrics</FeatureItem>
              <FeatureItem>Accepts any image format: file paths, URLs, PIL, NumPy, PyTorch tensors, raw bytes</FeatureItem>
            </ul>
          </motion.div>

          <Divider />

          {/* ────────────── INSTALLATION ────────────── */}
          <SectionHeading id="installation" icon={Terminal}>Installation</SectionHeading>
          <SubHeading>Requirements</SubHeading>
          <ul className="space-y-1.5 mb-4">
            <li className="flex items-center gap-2 text-surface-400">
              <span className="w-1.5 h-1.5 rounded-full bg-libre-400" />Python 3.10+
            </li>
            <li className="flex items-center gap-2 text-surface-400">
              <span className="w-1.5 h-1.5 rounded-full bg-libre-400" />PyTorch 1.7+
            </li>
          </ul>

          <SubHeading>From PyPI</SubHeading>
          <CodeBlock language="bash">{`pip install libreyolo`}</CodeBlock>

          <SubHeading>From source</SubHeading>
          <CodeBlock language="bash">{`git clone https://github.com/Libre-YOLO/libreyolo.git
cd libreyolo
pip install -e .`}</CodeBlock>

          <SubHeading>Optional dependencies</SubHeading>
          <CodeBlock language="bash">{`# ONNX export and inference
pip install libreyolo[onnx]
# or: pip install onnx onnxsim onnxscript onnxruntime

# RF-DETR support
pip install libreyolo[rfdetr]
# or: pip install rfdetr transformers timm supervision

# TensorRT export and inference (NVIDIA GPU)
pip install libreyolo[tensorrt]
# Note: TensorRT itself requires manual installation (depends on CUDA version)

# OpenVINO export and inference (Intel CPU/GPU/VPU)
pip install libreyolo[openvino]`}</CodeBlock>

          <P>If using <InlineCode>uv</InlineCode>:</P>
          <CodeBlock language="bash">{`uv sync --extra onnx
uv sync --extra rfdetr
uv sync --extra tensorrt
uv sync --extra openvino`}</CodeBlock>

          <Divider />

          {/* ────────────── QUICKSTART ────────────── */}
          <SectionHeading id="quickstart" icon={Rocket}>Quickstart</SectionHeading>

          <SubHeading>Load a model and run inference</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

# Auto-detects architecture and size from the weights file
model = LibreYOLO("LibreYOLOXs.pt")

# Run on a single image
result = model("photo.jpg")

print(f"Found {len(result)} objects")
print(result.boxes.xyxy)   # bounding boxes (N, 4)
print(result.boxes.conf)   # confidence scores (N,)
print(result.boxes.cls)    # class IDs (N,)`}</CodeBlock>

          <SubHeading>Save annotated output</SubHeading>
          <CodeBlock language="python">{`result = model("photo.jpg", save=True)
# Saved to runs/detections/photo_LibreYOLOX_s_<timestamp>.jpg`}</CodeBlock>

          <SubHeading>Process a directory</SubHeading>
          <CodeBlock language="python">{`results = model("images/", save=True, batch=4)
for r in results:
    print(f"{r.path}: {len(r)} detections")`}</CodeBlock>

          <Divider />

          {/* ────────────── AVAILABLE MODELS ────────────── */}
          <SectionHeading id="models" icon={Layers}>Available Models</SectionHeading>

          <SubHeading>YOLOX</SubHeading>
          <DocTable
            headers={['Size', 'Code', 'Input size', 'Use case']}
            rows={[
              ['Nano', <InlineCode key="n">&quot;n&quot;</InlineCode>, '416', 'Edge devices, mobile'],
              ['Tiny', <InlineCode key="t">&quot;t&quot;</InlineCode>, '416', 'Edge devices, faster'],
              ['Small', <InlineCode key="s">&quot;s&quot;</InlineCode>, '640', 'Balanced speed/accuracy'],
              ['Medium', <InlineCode key="m">&quot;m&quot;</InlineCode>, '640', 'Higher accuracy'],
              ['Large', <InlineCode key="l">&quot;l&quot;</InlineCode>, '640', 'High accuracy'],
              ['XLarge', <InlineCode key="x">&quot;x&quot;</InlineCode>, '640', 'Maximum accuracy'],
            ]}
          />
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLOXn.pt")
# model = LibreYOLO("LibreYOLOXt.pt")
# model = LibreYOLO("LibreYOLOXs.pt")
# model = LibreYOLO("LibreYOLOXm.pt")
# model = LibreYOLO("LibreYOLOXl.pt")
# model = LibreYOLO("LibreYOLOXx.pt")`}</CodeBlock>

          <SubHeading>YOLOv9</SubHeading>
          <DocTable
            headers={['Size', 'Code', 'Input size', 'Use case']}
            rows={[
              ['Tiny', <InlineCode key="t">&quot;t&quot;</InlineCode>, '640', 'Fast inference'],
              ['Small', <InlineCode key="s">&quot;s&quot;</InlineCode>, '640', 'Balanced'],
              ['Medium', <InlineCode key="m">&quot;m&quot;</InlineCode>, '640', 'Higher accuracy'],
              ['Compact', <InlineCode key="c">&quot;c&quot;</InlineCode>, '640', 'Best accuracy'],
            ]}
          />
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9t.pt")
# model = LibreYOLO("LibreYOLO9s.pt")
# model = LibreYOLO("LibreYOLO9m.pt")
# model = LibreYOLO("LibreYOLO9c.pt")`}</CodeBlock>

          <SubHeading>RF-DETR</SubHeading>
          <DocTable
            headers={['Size', 'Code', 'Input size', 'Use case']}
            rows={[
              ['Nano', <InlineCode key="n">&quot;n&quot;</InlineCode>, '384', 'Edge'],
              ['Small', <InlineCode key="s">&quot;s&quot;</InlineCode>, '512', 'Balanced'],
              ['Medium', <InlineCode key="m">&quot;m&quot;</InlineCode>, '576', 'Higher accuracy'],
              ['Large', <InlineCode key="l">&quot;l&quot;</InlineCode>, '704', 'Maximum accuracy'],
            ]}
          />
          <CodeBlock language="python">{`from libreyolo import LibreYOLORFDETR

model = LibreYOLORFDETR(size="s")`}</CodeBlock>

          <SubHeading>Factory function (recommended)</SubHeading>
          <P>
            The <InlineCode>LibreYOLO()</InlineCode> factory auto-detects everything from the weights file:
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

# Auto-detects: YOLOX, size=s, 80 classes
model = LibreYOLO("LibreYOLOXs.pt")

# Auto-detects: YOLOv9, size=c, 80 classes
model = LibreYOLO("LibreYOLO9c.pt")

# Auto-detects: RF-DETR
model = LibreYOLO("LibreRFDETRs.pth")

# ONNX models work too
model = LibreYOLO("model.onnx")

# TensorRT engines
model = LibreYOLO("model.engine")

# OpenVINO models (directory with model.xml)
model = LibreYOLO("model_openvino/")`}</CodeBlock>
          <P>
            If weights are not found locally, LibreYOLO attempts to download them from Hugging Face automatically.
          </P>

          <Divider />

          {/* ────────────── PREDICTION ────────────── */}
          <SectionHeading id="prediction" icon={Crosshair}>Prediction</SectionHeading>

          <SubHeading>Basic prediction</SubHeading>
          <CodeBlock language="python">{`result = model("image.jpg")`}</CodeBlock>

          <SubHeading>All prediction parameters</SubHeading>
          <CodeBlock language="python">{`result = model(
    "image.jpg",
    conf=0.25,            # confidence threshold (default: 0.25)
    iou=0.45,             # NMS IoU threshold (default: 0.45)
    imgsz=640,            # input size override (default: model's native)
    classes=[0, 2, 5],    # filter to specific class IDs (default: all)
    max_det=300,          # max detections per image (default: 300)
    save=True,            # save annotated image (default: False)
    output_path="out/",   # where to save (default: runs/detections/)
    color_format="auto",  # "auto", "rgb", or "bgr"
    output_file_format="png",  # output format: "jpg", "png", "webp"
)`}</CodeBlock>
          <P>
            <InlineCode>model.predict(...)</InlineCode> is an alias for <InlineCode>model(...)</InlineCode>.
          </P>

          <SubHeading>Supported input formats</SubHeading>
          <P>LibreYOLO accepts images in any of these formats:</P>
          <CodeBlock language="python">{`# File path (string or pathlib.Path)
result = model("photo.jpg")
result = model(Path("photo.jpg"))

# URL
result = model("https://example.com/image.jpg")

# PIL Image
from PIL import Image
img = Image.open("photo.jpg")
result = model(img)

# NumPy array (HWC or CHW, RGB or BGR, uint8 or float32)
import numpy as np
arr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
result = model(arr)

# OpenCV (BGR) — specify color_format
import cv2
frame = cv2.imread("photo.jpg")
result = model(frame, color_format="bgr")

# PyTorch tensor (CHW or NCHW)
import torch
tensor = torch.randn(3, 640, 640)
result = model(tensor)

# Raw bytes
with open("photo.jpg", "rb") as f:
    result = model(f.read())

# Directory of images
results = model("images/", batch=4)`}</CodeBlock>

          <SubHeading>Working with results</SubHeading>
          <P>
            Every prediction returns a <InlineCode>Results</InlineCode> object (or a list of them for directories):
          </P>
          <CodeBlock language="python">{`result = model("image.jpg")

# Number of detections
len(result)  # e.g., 5

# Bounding boxes in xyxy format (x1, y1, x2, y2)
result.boxes.xyxy        # tensor of shape (N, 4)

# Bounding boxes in xywh format (center_x, center_y, width, height)
result.boxes.xywh        # tensor of shape (N, 4)

# Confidence scores
result.boxes.conf        # tensor of shape (N,)

# Class IDs
result.boxes.cls         # tensor of shape (N,)

# Combined data: [x1, y1, x2, y2, conf, cls]
result.boxes.data        # tensor of shape (N, 6)

# Metadata
result.orig_shape        # (height, width) of original image
result.path              # source file path (or None)
result.names             # {0: "person", 1: "bicycle", ...}

# Move to CPU / convert to numpy
result_cpu = result.cpu()
boxes_np = result.boxes.numpy()`}</CodeBlock>

          <SubHeading>Class filtering</SubHeading>
          <P>Filter detections to specific class IDs:</P>
          <CodeBlock language="python">{`# Only detect people (class 0) and cars (class 2)
result = model("image.jpg", classes=[0, 2])`}</CodeBlock>

          <Divider />

          {/* ────────────── TILED INFERENCE ────────────── */}
          <SectionHeading id="tiled-inference" icon={Grid3x3}>Tiled Inference</SectionHeading>
          <P>
            For images much larger than the model's input size (e.g., satellite imagery, drone footage), tiled inference splits the image into overlapping tiles, runs detection on each, and merges results.
          </P>
          <CodeBlock language="python">{`result = model(
    "large_aerial_image.jpg",
    tiling=True,
    overlap_ratio=0.2,   # 20% overlap between tiles (default)
    save=True,
)

# Extra metadata on tiled results
result.tiled           # True
result.num_tiles       # number of tiles used`}</CodeBlock>

          <P>
            When <InlineCode>save=True</InlineCode> with tiling, LibreYOLO saves:
          </P>
          <ul className="space-y-2 mb-4">
            <FeatureItem><InlineCode>final_image.jpg</InlineCode> — full image with all merged detections drawn</FeatureItem>
            <FeatureItem><InlineCode>grid_visualization.jpg</InlineCode> — image showing tile grid overlay</FeatureItem>
            <FeatureItem><InlineCode>tiles/</InlineCode> — individual tile crops</FeatureItem>
            <FeatureItem><InlineCode>metadata.json</InlineCode> — tiling parameters and detection counts</FeatureItem>
          </ul>
          <P>
            If the image is already smaller than the model's input size, tiling is skipped automatically.
          </P>

          <Divider />

          {/* ────────────── TRAINING ────────────── */}
          <SectionHeading id="training" icon={GraduationCap}>Training</SectionHeading>

          <SubHeading>YOLOX training</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLOX

model = LibreYOLOX(size="s")

results = model.train(
    data="coco128.yaml",     # path to data.yaml (required)

    # Training parameters
    epochs=100,              # default: 100
    batch=16,
    imgsz=640,

    # Optimizer
    lr0=0.01,                # initial learning rate
    optimizer="SGD",         # "SGD", "Adam", "AdamW"

    # System
    device="0",              # GPU device ("", "cpu", "cuda", "0", "0,1")
    workers=8,
    seed=0,

    # Output
    project="runs/train",
    name="exp",
    exist_ok=False,

    # Training features
    amp=True,                # automatic mixed precision
    patience=50,             # early stopping patience
    resume=False,            # resume from loaded checkpoint
)

print(f"Best mAP50-95: {results['best_mAP50_95']:.3f}")
print(f"Best checkpoint: {results['best_checkpoint']}")`}</CodeBlock>

          <P>After training, the model instance is automatically updated with the best weights.</P>

          <SubHeading>Training results dict</SubHeading>
          <CodeBlock language="python">{`{
    "final_loss": 2.31,
    "best_mAP50": 0.682,
    "best_mAP50_95": 0.451,
    "best_epoch": 87,
    "save_dir": "runs/train/exp",
    "best_checkpoint": "runs/train/exp/weights/best.pt",
    "last_checkpoint": "runs/train/exp/weights/last.pt",
}`}</CodeBlock>

          <SubHeading>Resuming training</SubHeading>
          <CodeBlock language="python">{`model = LibreYOLOX("runs/train/exp/weights/last.pt", size="s")
results = model.train(data="coco128.yaml", resume=True)`}</CodeBlock>

          <SubHeading>Custom dataset YAML format</SubHeading>
          <CodeBlock language="yaml" filename="data.yaml">{`path: /path/to/dataset
train: images/train
val: images/val
test: images/test  # optional

nc: 3
names: ["cat", "dog", "bird"]`}</CodeBlock>

          <SubHeading>YOLOv9 training</SubHeading>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO9

model = LibreYOLO9("LibreYOLO9c.pt", size="c")

results = model.train(
    data="coco128.yaml",
    epochs=300,              # default: 300
    batch=16,
    imgsz=640,
    lr0=0.01,
    optimizer="SGD",
    device="0",
    workers=8,
    seed=0,
    project="runs/train",
    name="v9_exp",           # default: "v9_exp"
    exist_ok=False,
    resume=False,
    amp=True,
    patience=50,
)

print(f"Best mAP50-95: {results['best_mAP50_95']:.3f}")`}</CodeBlock>
          <P>
            YOLOv9 training uses the same parameter API as YOLOX but defaults to <InlineCode>epochs=300</InlineCode> and <InlineCode>name=&quot;v9_exp&quot;</InlineCode>. It does not have a <InlineCode>pretrained</InlineCode> parameter.
          </P>

          <SubHeading>RF-DETR training</SubHeading>
          <P>
            RF-DETR uses a different training API that wraps the original rfdetr implementation:
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLORFDETR

model = LibreYOLORFDETR(size="s")

results = model.train(
    data="path/to/dataset",  # Roboflow/COCO format directory
    epochs=100,
    batch_size=4,
    lr=1e-4,
    output_dir="runs/train",
)`}</CodeBlock>

          <P>RF-DETR datasets use COCO annotation format:</P>
          <CodeBlock language="text">{`dataset/
    train/
        _annotations.coco.json
        image1.jpg
        image2.jpg
    valid/
        _annotations.coco.json
        image1.jpg`}</CodeBlock>

          <Divider />

          {/* ────────────── VALIDATION ────────────── */}
          <SectionHeading id="validation" icon={CheckCircle2}>Validation</SectionHeading>
          <P>Run COCO-standard evaluation on a validation set:</P>
          <CodeBlock language="python">{`results = model.val(
    data="coco128.yaml",   # dataset config
    batch=16,
    imgsz=640,
    conf=0.001,            # low conf for mAP calculation
    iou=0.6,               # NMS IoU threshold
    split="val",           # "val", "test", or "train"
    save_json=False,       # save predictions as COCO JSON
    plots=True,            # reserved for future visualization
    verbose=True,          # print per-class metrics
)

print(f"mAP50:    {results['metrics/mAP50']:.3f}")
print(f"mAP50-95: {results['metrics/mAP50-95']:.3f}")`}</CodeBlock>

          <SubHeading>Validation results dict</SubHeading>
          <P>
            By default, LibreYOLO uses COCO evaluation and returns 12 standard metrics:
          </P>
          <CodeBlock language="python">{`{
    "metrics/mAP50-95": 0.489,   # COCO primary metric (AP@[.5:.95])
    "metrics/mAP50": 0.721,      # AP@0.5 (PASCAL VOC style)
    "metrics/mAP75": 0.534,      # AP@0.75 (strict)
    "metrics/mAP_small": 0.291,
    "metrics/mAP_medium": 0.532,
    "metrics/mAP_large": 0.648,
    "metrics/AR1": 0.362,        # Average Recall (max 1 det)
    "metrics/AR10": 0.571,
    "metrics/AR100": 0.601,
    "metrics/AR_small": 0.387,
    "metrics/AR_medium": 0.641,
    "metrics/AR_large": 0.739,
}`}</CodeBlock>
          <P>
            Set <InlineCode>use_coco_eval=False</InlineCode> in <InlineCode>ValidationConfig</InlineCode> for legacy precision/recall metrics.
          </P>

          <Divider />

          {/* ────────────── EXPORT ────────────── */}
          <SectionHeading id="export" icon={Upload}>Export</SectionHeading>
          <P>Export models to ONNX, TorchScript, TensorRT, or OpenVINO for deployment.</P>

          <SubHeading>Quick export</SubHeading>
          <CodeBlock language="python">{`# ONNX (default)
model.export()

# TorchScript
model.export(format="torchscript")

# TensorRT (requires NVIDIA GPU + TensorRT)
model.export(format="tensorrt")

# OpenVINO (optimized for Intel hardware)
model.export(format="openvino")`}</CodeBlock>

          <SubHeading>All export parameters</SubHeading>
          <CodeBlock language="python">{`path = model.export(
    format="onnx",            # "onnx", "torchscript", "tensorrt", or "openvino"
    output_path="model.onnx", # output file (auto-generated if None)
    imgsz=640,                # input resolution (default: model's native)
    opset=13,                 # ONNX opset version (default: 13)
    simplify=True,            # run onnxsim graph simplification
    dynamic=True,             # enable dynamic batch axis
    half=False,               # export in FP16
    batch=1,                  # batch size for static graph
    device=None,              # device to trace on (default: model's current device)
    int8=False,               # INT8 quantization (TensorRT)
    data=None,                # calibration dataset for INT8
    fraction=1.0,             # fraction of calibration data to use
    workspace=4,              # TensorRT workspace size (GB)
    hardware_compatibility=False,  # TensorRT hardware compatibility mode
    gpu_device=0,             # GPU device index for TensorRT
    trt_config=None,          # custom TensorRT BuilderConfig
    verbose=False,            # verbose logging
)`}</CodeBlock>

          <SubHeading>ONNX metadata</SubHeading>
          <P>Exported ONNX files include embedded metadata:</P>
          <DocTable
            headers={['Key', 'Example value']}
            rows={[
              [<InlineCode key="v">libreyolo_version</InlineCode>, <InlineCode key="vv">&quot;0.1.4&quot;</InlineCode>],
              [<InlineCode key="f">model_family</InlineCode>, <InlineCode key="fv">&quot;LibreYOLOX&quot;</InlineCode>],
              [<InlineCode key="s">model_size</InlineCode>, <InlineCode key="sv">&quot;s&quot;</InlineCode>],
              [<InlineCode key="c">nb_classes</InlineCode>, <InlineCode key="cv">&quot;80&quot;</InlineCode>],
              [<InlineCode key="n">names</InlineCode>, <span key="nv" className="text-xs"><InlineCode>{`'{"0": "person", "1": "bicycle", ...}'`}</InlineCode></span>],
              [<InlineCode key="i">imgsz</InlineCode>, <InlineCode key="iv">&quot;640&quot;</InlineCode>],
              [<InlineCode key="d">dynamic</InlineCode>, <InlineCode key="dv">&quot;True&quot;</InlineCode>],
              [<InlineCode key="h">half</InlineCode>, <InlineCode key="hv">&quot;False&quot;</InlineCode>],
            ]}
          />
          <P>
            This metadata is automatically read back when loading the model with <InlineCode>LibreYOLOOnnx</InlineCode>.
          </P>

          <SubHeading>Using the Exporter class directly</SubHeading>
          <CodeBlock language="python">{`from libreyolo.export import Exporter

exporter = Exporter(model)
path = exporter("onnx", dynamic=True, simplify=True)`}</CodeBlock>

          <Divider />

          {/* ────────────── ONNX INFERENCE ────────────── */}
          <SectionHeading id="onnx-inference" icon={Cpu}>ONNX Inference</SectionHeading>
          <P>
            Run inference using ONNX Runtime instead of PyTorch. Useful for deployment environments without PyTorch.
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLOOnnx

model = LibreYOLOOnnx("model.onnx")

result = model("image.jpg", conf=0.25, iou=0.45, save=True)
print(result.boxes.xyxy)`}</CodeBlock>

          <SubHeading>Auto-metadata</SubHeading>
          <P>
            If the ONNX file was exported by LibreYOLO, class names and class count are read automatically from the embedded metadata:
          </P>
          <CodeBlock language="python">{`# Export with metadata
model.export(format="onnx", output_path="model.onnx")

# Load — names and nb_classes auto-populated
onnx_model = LibreYOLOOnnx("model.onnx")
print(onnx_model.names)       # {0: "person", 1: "bicycle", ...}
print(onnx_model.nb_classes)  # 80`}</CodeBlock>

          <P>
            For ONNX files without metadata (e.g., exported by other tools), specify <InlineCode>nb_classes</InlineCode> manually:
          </P>
          <CodeBlock language="python">{`model = LibreYOLOOnnx("external_model.onnx", nb_classes=20)`}</CodeBlock>

          <SubHeading>Device selection</SubHeading>
          <CodeBlock language="python">{`# Auto-detect (CUDA if available, else CPU)
model = LibreYOLOOnnx("model.onnx", device="auto")

# Force CPU
model = LibreYOLOOnnx("model.onnx", device="cpu")

# Force CUDA
model = LibreYOLOOnnx("model.onnx", device="cuda")`}</CodeBlock>

          <SubHeading>Prediction parameters</SubHeading>
          <P>
            <InlineCode>LibreYOLOOnnx</InlineCode> supports the same prediction API:
          </P>
          <CodeBlock language="python">{`result = model(
    "image.jpg",
    conf=0.25,
    iou=0.45,
    imgsz=640,
    classes=[0, 2],
    max_det=300,
    save=True,
    output_path="output/",
    color_format="auto",
)`}</CodeBlock>

          <Divider />

          {/* ────────────── TENSORRT INFERENCE ────────────── */}
          <SectionHeading id="tensorrt-inference" icon={Cpu}>TensorRT Inference</SectionHeading>
          <P>
            Run inference using TensorRT for maximum throughput on NVIDIA GPUs. Requires TensorRT and pycuda.
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLOTensorRT

model = LibreYOLOTensorRT("model.engine")

result = model("image.jpg", conf=0.25, iou=0.45, save=True)
print(result.boxes.xyxy)`}</CodeBlock>

          <SubHeading>Auto-detection via factory</SubHeading>
          <P>
            The <InlineCode>LibreYOLO()</InlineCode> factory automatically detects <InlineCode>.engine</InlineCode> files:
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

# Auto-detects TensorRT engine
model = LibreYOLO("model.engine")`}</CodeBlock>

          <P>
            <InlineCode>LibreYOLOTensorRT</InlineCode> supports the same prediction API as PyTorch models (except tiling).
          </P>

          <Divider />

          {/* ────────────── OPENVINO INFERENCE ────────────── */}
          <SectionHeading id="openvino-inference" icon={Cpu}>OpenVINO Inference</SectionHeading>
          <P>
            Run inference using OpenVINO, optimized for Intel CPUs, GPUs, and VPUs.
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLOOpenVINO

model = LibreYOLOOpenVINO("model_openvino/")

result = model("image.jpg", conf=0.25, iou=0.45, save=True)
print(result.boxes.xyxy)`}</CodeBlock>

          <SubHeading>Auto-detection via factory</SubHeading>
          <P>
            The <InlineCode>LibreYOLO()</InlineCode> factory automatically detects OpenVINO model directories:
          </P>
          <CodeBlock language="python">{`from libreyolo import LibreYOLO

# Auto-detects OpenVINO directory
model = LibreYOLO("model_openvino/")`}</CodeBlock>

          <P>
            <InlineCode>LibreYOLOOpenVINO</InlineCode> supports the same prediction API as PyTorch models (except tiling).
          </P>

          <Divider />

          {/* ────────────── API REFERENCE ────────────── */}
          <SectionHeading id="api-reference" icon={FileCode}>API Reference</SectionHeading>

          <SubHeading>LibreYOLO (factory)</SubHeading>
          <CodeBlock language="python">{`LibreYOLO(
    model_path: str,
    size: str = None,           # auto-detected from weights
    reg_max: int = 16,          # YOLOv9 only
    nb_classes: int = None,     # auto-detected from weights
    device: str = "auto",
) -> LibreYOLOX | LibreYOLO9 | LibreYOLORFDETR | LibreYOLOOnnx | LibreYOLOTensorRT | LibreYOLOOpenVINO`}</CodeBlock>
          <P>
            Auto-detects model architecture, size, and class count from the weights file. Returns the appropriate model class. Also handles <InlineCode>.onnx</InlineCode>, <InlineCode>.engine</InlineCode> (TensorRT), and OpenVINO model directories. Downloads weights from Hugging Face if not found locally.
          </P>

          <SubHeading>Prediction (all models)</SubHeading>
          <CodeBlock language="python">{`model(
    source,                     # image input (see supported formats)
    *,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = None,
    classes: list[int] = None,
    max_det: int = 300,
    save: bool = False,
    batch: int = 1,
    output_path: str = None,
    color_format: str = "auto",
    tiling: bool = False,
    overlap_ratio: float = 0.2,
    output_file_format: str = None,
) -> Results | list[Results]`}</CodeBlock>

          <SubHeading>Results</SubHeading>
          <CodeBlock language="python">{`result = Results(
    boxes: Boxes,
    orig_shape: tuple[int, int],  # (height, width)
    path: str | None,
    names: dict[int, str],
)

len(result)          # number of detections
result.cpu()         # copy with tensors on CPU`}</CodeBlock>

          <SubHeading>Boxes</SubHeading>
          <CodeBlock language="python">{`boxes = Boxes(boxes, conf, cls)

boxes.xyxy           # (N, 4) tensor — x1, y1, x2, y2
boxes.xywh           # (N, 4) tensor — cx, cy, w, h
boxes.conf           # (N,) tensor — confidence scores
boxes.cls            # (N,) tensor — class IDs
boxes.data           # (N, 6) tensor — [xyxy, conf, cls]

len(boxes)           # number of boxes
boxes.cpu()          # copy on CPU
boxes.numpy()        # copy as numpy arrays`}</CodeBlock>

          <SubHeading>model.export()</SubHeading>
          <CodeBlock language="python">{`model.export(
    format: str = "onnx",       # "onnx", "torchscript", "tensorrt", or "openvino"
    *,
    output_path: str = None,
    imgsz: int = None,
    opset: int = 13,
    simplify: bool = True,
    dynamic: bool = True,
    half: bool = False,
    batch: int = 1,
    device: str = None,
    int8: bool = False,
    data: str = None,           # calibration data for INT8
    fraction: float = 1.0,      # fraction of calibration data
    workspace: int = 4,         # TensorRT workspace (GB)
    hardware_compatibility: bool = False,
    gpu_device: int = 0,
    trt_config = None,          # custom TensorRT BuilderConfig
    verbose: bool = False,
) -> str                        # path to exported file`}</CodeBlock>

          <SubHeading>Exporter</SubHeading>
          <CodeBlock language="python">{`from libreyolo.export import Exporter

exporter = Exporter(model)
path = exporter(
    format,                     # same parameters as model.export()
    **kwargs,
)

Exporter.FORMATS               # dict of supported formats
# {
#   "onnx":        {"suffix": ".onnx",        "requires": "onnx"},
#   "torchscript": {"suffix": ".torchscript", "requires": None},
#   "tensorrt":    {"suffix": ".engine",      "requires": "tensorrt"},
#   "openvino":    {"suffix": "_openvino",    "requires": "openvino-dev"},
# }`}</CodeBlock>

          <SubHeading>model.val()</SubHeading>
          <CodeBlock language="python">{`model.val(
    data: str = None,           # path to data.yaml
    batch: int = 16,
    imgsz: int = None,
    conf: float = 0.001,
    iou: float = 0.6,
    device: str = None,
    split: str = "val",         # "val", "test", or "train"
    save_json: bool = False,
    plots: bool = True,
    verbose: bool = True,
) -> dict`}</CodeBlock>
          <P>Returns (COCO evaluation, default):</P>
          <CodeBlock language="python">{`{
    "metrics/mAP50-95": float,   # COCO primary metric
    "metrics/mAP50": float,
    "metrics/mAP75": float,
    "metrics/mAP_small": float,
    "metrics/mAP_medium": float,
    "metrics/mAP_large": float,
    "metrics/AR1": float,
    "metrics/AR10": float,
    "metrics/AR100": float,
    "metrics/AR_small": float,
    "metrics/AR_medium": float,
    "metrics/AR_large": float,
}`}</CodeBlock>

          <SubHeading>model.train() (YOLOX)</SubHeading>
          <CodeBlock language="python">{`model.train(
    data: str,                  # path to data.yaml (required)
    *,
    epochs: int = 100,
    batch: int = 16,
    imgsz: int = 640,
    lr0: float = 0.01,
    optimizer: str = "SGD",
    device: str = "",
    workers: int = 8,
    seed: int = 0,
    project: str = "runs/train",
    name: str = "exp",
    exist_ok: bool = False,
    pretrained: bool = True,
    resume: bool = False,
    amp: bool = True,
    patience: int = 50,
) -> dict`}</CodeBlock>
          <P>Returns:</P>
          <CodeBlock language="python">{`{
    "final_loss": float,
    "best_mAP50": float,
    "best_mAP50_95": float,
    "best_epoch": int,
    "save_dir": str,
    "best_checkpoint": str,
    "last_checkpoint": str,
}`}</CodeBlock>

          <SubHeading>model.train() (YOLOv9)</SubHeading>
          <CodeBlock language="python">{`model.train(
    data: str,                  # path to data.yaml (required)
    *,
    epochs: int = 300,
    batch: int = 16,
    imgsz: int = 640,
    lr0: float = 0.01,
    optimizer: str = "SGD",
    device: str = "",
    workers: int = 8,
    seed: int = 0,
    project: str = "runs/train",
    name: str = "v9_exp",
    exist_ok: bool = False,
    resume: bool = False,
    amp: bool = True,
    patience: int = 50,
) -> dict`}</CodeBlock>
          <P>Returns the same dict as YOLOX training.</P>

          <SubHeading>model.train() (RF-DETR)</SubHeading>
          <CodeBlock language="python">{`model.train(
    data: str,                  # path to dataset directory
    epochs: int = 100,
    batch_size: int = 4,
    lr: float = 1e-4,
    output_dir: str = "runs/train",
    resume: str = None,
    **kwargs,                   # additional RF-DETR training args
) -> dict`}</CodeBlock>

          <SubHeading>LibreYOLOOnnx</SubHeading>
          <CodeBlock language="python">{`LibreYOLOOnnx(
    onnx_path: str,
    nb_classes: int = 80,       # auto-read from metadata if available
    device: str = "auto",
)`}</CodeBlock>
          <P>
            Supports the same prediction API as PyTorch models (except tiling).
          </P>

          <SubHeading>LibreYOLOTensorRT</SubHeading>
          <CodeBlock language="python">{`LibreYOLOTensorRT(
    engine_path: str,
    nb_classes: int = 80,
    device: str = "auto",
)`}</CodeBlock>
          <P>
            Runs inference on a TensorRT <InlineCode>.engine</InlineCode> file. Same prediction API as PyTorch models (except tiling).
          </P>

          <SubHeading>LibreYOLOOpenVINO</SubHeading>
          <CodeBlock language="python">{`LibreYOLOOpenVINO(
    model_dir: str,
    nb_classes: int = 80,
    device: str = "auto",
)`}</CodeBlock>
          <P>
            Runs inference on an OpenVINO model directory. Same prediction API as PyTorch models (except tiling).
          </P>

          <SubHeading>ValidationConfig</SubHeading>
          <CodeBlock language="python">{`from libreyolo import ValidationConfig

config = ValidationConfig(
    data="coco128.yaml",
    data_dir=None,             # override dataset root directory
    batch_size=16,
    imgsz=640,
    conf_thres=0.001,
    iou_thres=0.6,
    max_det=300,
    split="val",               # "val", "test", or "train"
    device="auto",
    save_json=False,
    plots=True,
    verbose=True,
    half=False,
    use_coco_eval=True,        # use COCO eval (12 keys); False for legacy
    num_workers=8,
)

# Load/save YAML
config = ValidationConfig.from_yaml("config.yaml")
config.to_yaml("config.yaml")`}</CodeBlock>

          <Divider />

          {/* ────────────── ARCHITECTURE GUIDE ────────────── */}
          <SectionHeading id="architecture" icon={Wrench}>Architecture Guide</SectionHeading>
          <P>
            This section is for contributors who want to understand the codebase internals.
          </P>

          <SubHeading>Base class design</SubHeading>
          <P>
            All model families inherit from <InlineCode>LibreYOLOBase</InlineCode> (in <InlineCode>libreyolo/common/base_model.py</InlineCode>). Subclasses implement these abstract methods:
          </P>
          <DocTable
            headers={['Method', 'Purpose']}
            rows={[
              [<InlineCode key="init">_init_model()</InlineCode>, 'Build and return the nn.Module'],
              [<InlineCode key="sizes">_get_valid_sizes()</InlineCode>, 'Return list of valid size codes'],
              [<InlineCode key="name">_get_model_name()</InlineCode>, 'Return model name string'],
              [<InlineCode key="input">_get_input_size()</InlineCode>, 'Return default input resolution'],
              [<InlineCode key="pre">_preprocess()</InlineCode>, 'Image to tensor conversion'],
              [<InlineCode key="fwd">_forward()</InlineCode>, 'Model forward pass'],
              [<InlineCode key="post">_postprocess()</InlineCode>, 'Raw output to detection dicts'],
              [<InlineCode key="layers">_get_available_layers()</InlineCode>, 'Map of named layers'],
            ]}
          />
          <P>
            The base class provides the shared pipeline: <InlineCode>__call__</InlineCode> / <InlineCode>predict</InlineCode>, <InlineCode>export</InlineCode>, <InlineCode>val</InlineCode>, tiled inference, results wrapping, and saving.
          </P>

          <SubHeading>Package structure</SubHeading>
          <CodeBlock language="text">{`libreyolo/
    __init__.py          # Public API exports
    factory.py           # LibreYOLO() auto-detection factory
    common/
        base_model.py    # LibreYOLOBase abstract class
        onnx.py          # LibreYOLOOnnx runtime backend
        tensorrt.py      # LibreYOLOTensorRT runtime backend
        openvino.py      # LibreYOLOOpenVINO runtime backend
        results.py       # Results and Boxes classes
        image_loader.py  # Unified image loading
        utils.py         # NMS, drawing, preprocessing
    export/
        exporter.py      # Unified Exporter class
        onnx.py          # ONNX export logic
        torchscript.py   # TorchScript export logic
        tensorrt.py      # TensorRT export logic
        openvino.py      # OpenVINO export logic
    yolox/
        model.py         # LibreYOLOX
        nn.py            # YOLOX network architecture
        utils.py         # YOLOX-specific pre/postprocessing
    v9/
        model.py         # LibreYOLO9
        nn.py            # YOLOv9 network architecture
        utils.py         # v9-specific pre/postprocessing
    rfdetr/
        model.py         # LibreYOLORFDETR
        nn.py            # RF-DETR network + configs
        utils.py         # RF-DETR postprocessing
        train.py         # RF-DETR training wrapper
    training/
        config.py        # YOLOXTrainConfig / YOLOv9TrainConfig
        trainer.py       # YOLOXTrainer
        v9_trainer.py    # YOLOv9Trainer
        dataset.py       # Training dataset
        augment.py       # Mosaic, mixup, etc.
        loss.py          # YOLOX loss functions
        scheduler.py     # LR schedulers
        ema.py           # Exponential moving average
    validation/
        config.py        # ValidationConfig
        detection_validator.py  # DetectionValidator
        metrics.py       # DetMetrics, mAP computation
        base.py          # BaseValidator
        preprocessors.py # Per-model val preprocessing
    data/
        utils.py         # Dataset loading, YAML parsing
        yolo_coco_api.py # YOLO-to-COCO annotation bridge
    cfg/
        datasets/        # Built-in dataset YAML configs (coco8, coco128, coco5000, coco, etc.)`}</CodeBlock>

          <SubHeading>Adding a new model family</SubHeading>
          <ol className="space-y-2.5 mb-4 list-none">
            {[
              <>Create <InlineCode>libreyolo/newmodel/model.py</InlineCode> with a class inheriting <InlineCode>LibreYOLOBase</InlineCode></>,
              'Implement all abstract methods',
              <>Create <InlineCode>libreyolo/newmodel/nn.py</InlineCode> with the actual network architecture</>,
              <>Add detection logic to <InlineCode>factory.py</InlineCode> for auto-detection from weights</>,
              <>Export the class from <InlineCode>libreyolo/__init__.py</InlineCode></>,
              <>(Optional) Override <InlineCode>_get_val_preprocessor()</InlineCode> if the model needs non-standard validation preprocessing</>,
            ].map((item, i) => (
              <li key={i} className="flex items-start gap-3 text-surface-400">
                <span className="w-6 h-6 rounded-lg bg-libre-500/10 border border-libre-500/20 text-libre-400 text-xs font-bold flex items-center justify-center shrink-0 mt-0.5">
                  {i + 1}
                </span>
                <span>{item}</span>
              </li>
            ))}
          </ol>

          <SubHeading>Export architecture</SubHeading>
          <P>
            The <InlineCode>Exporter</InlineCode> class (in <InlineCode>libreyolo/export/exporter.py</InlineCode>) is format-agnostic. It uses a <InlineCode>FORMATS</InlineCode> dict for dispatch:
          </P>
          <CodeBlock language="python">{`FORMATS = {
    "onnx":        {"suffix": ".onnx",        "requires": "onnx"},
    "torchscript": {"suffix": ".torchscript", "requires": None},
    "tensorrt":    {"suffix": ".engine",      "requires": "tensorrt"},
    "openvino":    {"suffix": "_openvino",    "requires": "openvino-dev"},
}`}</CodeBlock>
          <P>
            To add a new export format, add an entry to <InlineCode>FORMATS</InlineCode> with the file suffix and required package, then implement the corresponding export module in <InlineCode>libreyolo/export/</InlineCode>.
          </P>

          <Divider />

          {/* ────────────── DATASET FORMAT ────────────── */}
          <SectionHeading id="dataset-format" icon={Database}>Dataset Format</SectionHeading>
          <P>
            LibreYOLO uses YOLO-format datasets configured via YAML files.
          </P>

          <SubHeading>data.yaml structure</SubHeading>
          <CodeBlock language="yaml" filename="data.yaml">{`path: /absolute/path/to/dataset   # dataset root
train: images/train               # relative to path
val: images/val                   # relative to path
test: images/test                 # optional

nc: 80                            # number of classes
names: [                          # class names
  "person", "bicycle", "car", "motorcycle", "airplane",
  "bus", "train", "truck", "boat", "traffic light",
  # ...
]`}</CodeBlock>

          <SubHeading>Directory layout</SubHeading>
          <CodeBlock language="text">{`dataset/
    images/
        train/
            img001.jpg
            img002.jpg
        val/
            img003.jpg
    labels/
        train/
            img001.txt
            img002.txt
        val/
            img003.txt`}</CodeBlock>

          <SubHeading>Label format</SubHeading>
          <P>
            One text file per image. Each line is one object:
          </P>
          <CodeBlock language="text">{`<class_id> <center_x> <center_y> <width> <height>`}</CodeBlock>
          <P>
            All coordinates are normalized to [0, 1] relative to image dimensions.
          </P>
          <P>Example (<InlineCode>img001.txt</InlineCode>):</P>
          <CodeBlock language="text" filename="img001.txt">{`0 0.5 0.4 0.3 0.6
2 0.1 0.2 0.05 0.1`}</CodeBlock>

          <SubHeading>Built-in datasets</SubHeading>
          <P>LibreYOLO includes configs for common datasets that auto-download:</P>
          <CodeBlock language="python">{`# These download automatically on first use
results = model.val(data="coco8.yaml")
results = model.train(data="coco128.yaml", epochs=10)`}</CodeBlock>

          <SubHeading>RF-DETR dataset format</SubHeading>
          <P>
            RF-DETR uses COCO-format annotations (JSON) instead of YOLO text labels:
          </P>
          <CodeBlock language="text">{`dataset/
    train/
        _annotations.coco.json
        image1.jpg
    valid/
        _annotations.coco.json
        image1.jpg`}</CodeBlock>

          {/* Bottom spacer */}
          <div className="h-16" />
        </div>
      </main>
    </div>
  )
}
