---
title: 安装
seo_title: 安装 LibreYOLO
description: 从 PyPI 安装 LibreYOLO，挑出某个模型家族或某个导出目标需要的可选 extra，并确认 PyTorch 能看到你的 GPU。
lead: >-
  LibreYOLO 发布在 PyPI 上，包名是 libreyolo。基础包覆盖预测、训练、验证，以及那些除 PyTorch
  之外不需要任何东西的模型家族；可选 extra 补上其余部分。
keywords:
  - libreyolo 安装
  - pip install libreyolo
  - libreyolo extra
  - libreyolo cuda
  - libreyolo gpu
  - libreyolo 环境要求
last_verified: 1.5.0
meta:
  - label: 包名
    value: libreyolo
    mono: true
  - label: Python
    value: 3.10 或更高版本
  - label: 代码许可
    value: MIT
  - label: 核心依赖
    value: PyTorch 2.4 或更高版本
snippets:
  install:
    - label: pip
      language: bash
      code: |
        pip install libreyolo
    - label: 带 extra
      language: bash
      code: |
        # 用逗号分隔，可以在一次安装里组合多个
        pip install "libreyolo[rfdetr,onnx]"
    - label: 全部装上
      language: bash
      code: |
        pip install "libreyolo[all]"
    - label: 从源码安装
      language: bash
      code: |
        git clone https://github.com/LibreYOLO/libreyolo.git
        cd libreyolo
        pip install -e .
  verify:
    - label: CLI
      language: bash
      code: |
        # Python、Torch、CUDA、cuDNN、每一块可见的 GPU，以及哪些
        # 可选包已经装上了
        libreyolo checks
    - label: Python
      language: python
      code: |
        import libreyolo

        print(libreyolo.__version__)
    - label: 模型清单
      language: bash
      code: |
        # 每个已注册的家族，连同它的任务、尺寸和输入分辨率。
        # 缺少对应 extra 的家族会连同启用它们的那条 pip
        # 命令一起列出
        libreyolo models
source_hash: 34fc6d3e24d03fb4
---

## 安装

<code-tabs name="install" />

需要 Python 3.10 或更高版本。基础安装会拉取 PyTorch、torchvision、NumPy、Pillow、
OpenCV、PyYAML、requests、mss、tqdm、pycocotools、typer、click、safetensors 和
SciPy，所以 YOLOv9 以及其他不需要更多依赖的家族，`pip install libreyolo` 之后就能
直接用。

克隆下来默认签出 `release`，那是代码与这份文档对应的稳定分支。承载未发布工作的集成
分支是 `dev`。

## 可选 extra

extra 是写在方括号里的一个名字，用来补上某个模型家族或某个导出目标需要的依赖。别的
什么都不变：装不装 extra，API 都一样。

### 模型家族

| Extra | 新增 |
|---|---|
| `rfdetr` | `transformers`，它提供 RF-DETR 的骨干 |
| `eomt` | `transformers` |
| `midas` | `timm` 1.0.x，它提供 MiDaS 的 ViT-L/16 和 EfficientNet-Lite3 编码器 |
| `vlm` | `transformers`、`num2words`、`decord`、`lmdb`、`peft` |
| `sam` | `transformers`、`timm` |
| `openvocab` | `transformers`、`timm`、`regex`、`ftfy` |
| `sensenova` | `transformers`、`accelerate`，以及非 macOS 上的 `bitsandbytes` |
| `modus` | `transformers`、`accelerate` |
| `clip` | `regex` 和 `ftfy`，自带的 CLIP 文本分词器需要它们 |
| `siglip2` | `sentencepiece`，多语言 SigLIP 2 分词器需要它 |
| `gaze` | `gdown`，它会开启 L2CS 检查点（checkpoint）的自动下载 |
| `rtdetr` | 无。RT-DETR 不需要额外的依赖；保留这个名字是为了让它保持稳定 |

### 导出与运行时

| Extra | 新增 |
|---|---|
| `onnx` | `onnx`、`onnxsim`、`onnxruntime` |
| `tensorrt` | `tensorrt-cu12` 10.16.1.11 和 `pycuda`，macOS 上除外 |
| `openvino` | `openvino` |
| `coreml` | `coremltools` |
| `coreai` | `coreai-torch`，仅限 macOS |
| `tflite`，别名 `litert` | `libreyolo[onnx]` 再加上 `onnx2tf`、`ai-edge-litert`、`onnx-graphsurgeon` 和 `onnx-simplifier` |
| `mnn` | `libreyolo[onnx]` 再加上 `MNN` |
| `ncnn` | `pnnx` 和 `ncnn` |
| `paddle` | `libreyolo[onnx]` 再加上 `paddlepaddle` 2.6.2 和 `x2paddle` 1.6.0 |
| `executorch` | `executorch` |
| `triton` | `tritonclient[http]`，用于 HTTP 和 HTTPS 的 V2 推理 |

### 训练、评估与日志

| Extra | 新增 |
|---|---|
| `lora` | `libreyolo[rfdetr]` 再加上 `peft`，用于 `lora=True` 微调 |
| `plots` | `matplotlib` |
| `fast-eval` | `faster-coco-eval`，C++ 实现的 COCO 评估后端 |
| `tensorboard` | `tensorboard` |
| `mlflow` | `mlflow` |
| `wandb` | `wandb` |
| `comet` | `comet-ml` |
| `clearml` | `clearml` |
| `neptune` | `neptune-scale` |
| `dvclive`，别名 `dvc` | `dvclive` |

`fast-eval` 是选装的，而不是硬依赖，这样一个没有预编译 wheel 的平台就不会让一次普通
安装失败。这个包不在时，COCO 评估会回退到 pycocotools，运行照常继续。

### 工具

| Extra | 新增 |
|---|---|
| `stream` | `yt-dlp`，只有解析 YouTube 页面 URL 时才需要 |
| `tracking` | 无。跟踪用到的依赖全都已经是核心依赖 |
| `label` | `libreyolo[sam]`，它让 `libreyolo label` 里的点击生成掩码辅助可用 |
| `hub-kernels` | `kernels`，编译好的 Hub kernel 的可选加载器。参见 [kernels](/docs/reference/kernels)，那里说明了装上它可能让 RF-DETR 的预测在浮点容差范围内发生变化 |
| `clip-convert` | `libreyolo[clip]` 再加上 `open_clip_torch`，用于权重转换和一致性核对 |
| `siglip2-convert` | `libreyolo[siglip2]` 再加上 `transformers`，出于同样的理由 |

摄像头、RTSP、RTMP、TCP、UDP、HLS 以及本地的多路流列表都不需要 extra。只有 YouTube
页面 URL 需要。

### 聚合 extra

`libreyolo[all]` 一条命令就装上模型、导出、跟踪和日志这几类 extra。有一些是特意留在
外面的。`neptune` 被排除，是因为稳定版 `neptune-scale` 要求 protobuf 低于 7，而
TFLite 那条路要求 protobuf 7。`executorch` 被排除，是因为 ExecuTorch 会限定它能搭配
的 PyTorch 版本；`coreai` 被排除，是因为 `coreai-torch` 把 PyTorch 锁在 2.11.x，会把
整个环境拖到那个版本上。`fast-eval`、`hub-kernels`、`clip-convert` 和
`siglip2-convert` 同样被留在外面。要用哪个就按名字单独装。

## 平台约束

有三个 extra 由它们的依赖 marker 限定了平台，所以安装在哪儿都能成功，只是在没有
wheel 的地方装得少一些。

| Extra | 约束 |
|---|---|
| `coreai` | 仅限 macOS。Core AI 工具链在别处既不能转换也不能运行 |
| `tensorrt` | 在 macOS 上跳过，那里没有 CUDA |
| `tflite`、`litert` | `onnx2tf` 和 `ai-edge-litert` 需要 Python 3.12 或更高版本 |

`sensenova` 在 macOS 上跳过 `bitsandbytes`，那里没有发布 wheel；这个 extra 的其余
部分照常安装。

如果瓶颈是磁盘，那么占地方的大头是 PyTorch，而 PyTorch 里的大头又是它默认 wheel 打包
的 CUDA 负载。换成仅 CPU 的 wheel 就能去掉这部分，而且什么也不用放弃。如果一台机器
根本不该带上 torch，只想在上面跑 ONNX 检测，参见[轻量安装](/docs/lightweight-install)。

## GPU 与 CUDA

设备的选择发生在构造模型的时候。默认值 `device="auto"` 会在
`torch.cuda.is_available()` 为真时用 CUDA，其次在 `torch.backends.mps.is_available()`
为真时用 Metal Performance Shaders，否则用 CPU。库里再没有别的地方去探测硬件，所以
PyTorch 看不到 GPU，LibreYOLO 也就看不到。

想固定设备，就把 `device` 传给模型，或者传给 `predict`、`train`、`val` 和 `export`。
它接受 `"cpu"`、`"cuda"`、`"cuda:0"`、`"mps"`、像 `0` 这样的裸整数，或者像 `"0"` 这样
的数字字符串；后两者会展开成 `cuda:<n>`。

先跑 `libreyolo checks`，它会打印 Torch 版本、Torch 编译时所对应的 CUDA 和 cuDNN
版本，以及每一块可见的 GPU 及其显存。如果一台机器上有英伟达显卡而它报告没有 CUDA，
那就是 pip 解析到的 PyTorch wheel 是 CPU 构建。先从 PyTorch 的索引装一个 CUDA 构建，
再装 LibreYOLO：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install libreyolo
```

仓库自己在 Linux 和 Windows 上那套由 uv 管理的环境，锁的就是这个索引。它需要 555 或
更新的英伟达驱动，这是 CUDA 12.8 的运行时要求。macOS 继续用 PyPI 上的 wheel，因为
PyTorch 的下载站点不发布 Darwin 构建。

## 检查安装结果

<code-tabs name="verify" />

`libreyolo models` 是看一个 extra 有没有生效的最快办法：缺少依赖的家族会连同启用它的
那条确切 pip 命令一起打印出来。两条命令也都接受 `--json`，它会把同样的数据以机器可读
的对象打印到 stdout。
