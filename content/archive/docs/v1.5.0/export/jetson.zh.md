---
title: NVIDIA Jetson
seo_title: 在 NVIDIA Jetson 上安装 LibreYOLO 和 PyTorch
description: >-
  在 NVIDIA Jetson 上安装 LibreYOLO：JetPack 漏掉的四个 CUDA 库、PyTorch 需要的 --no-deps
  这一步，以及 Orin Nano 上的实测数据。
lead: >-
  NVIDIA Jetson 板子用标准的 aarch64 PyTorch wheel 就能跑 LibreYOLO。不涉及任何 Jetson 专用的
  torch 构建，但 JetPack 漏掉了 torch 所链接的四个库，安装时必须自己补上。
keywords:
  - NVIDIA Jetson
  - Jetson Orin Nano
  - JetPack 7.2
  - jetson 安装 pytorch
  - nvidia-cudnn-cu13
  - nvidia-nccl-cu13
  - nvidia-cusparselt-cu13
  - nvidia-nvshmem-cu13
  - torch.cuda.is_available
  - no kernel image is available for execution on the device
  - jetson tensorrt 加速
  - aarch64 wheel
last_verified: 1.4.0
meta:
  - label: 开发板
    value: Jetson Orin Nano Super Developer Kit，8 GB，GPU 计算能力 8.7
  - label: 平台
    value: 'JetPack 7.2 (L4T R39.2), Ubuntu 24.04, CUDA 13, Python 3.12.3, aarch64'
  - label: 测试过的技术栈
    value: >-
      libreyolo 1.4.0, torch 2.13.0+cu130, torchvision 0.28.0+cu130, opencv
      5.0.0, numpy 2.5.1，于 2026-07-27
  - label: JetPack 里缺少的
    value: >-
      nvidia-cudnn-cu13, nvidia-nccl-cu13, nvidia-cusparselt-cu13,
      nvidia-nvshmem-cu13
    mono: true
  - label: 基准测试
    value: 在这块板子上 223 次验证过的运行，12 个家族共 58 个模型，覆盖 PyTorch、ONNX Runtime 和 TensorRT
    links:
      - label: visionanalysis.org/hardware/jetson_orin
        href: 'https://www.visionanalysis.org/hardware/jetson_orin'
  - label: 跟踪于
    value: issue 648 中属于 Jetson 的那一半
    links:
      - label: issue 648
        href: 'https://github.com/LibreYOLO/libreyolo/issues/648'
verification: >-
  安装步骤和预期输出取自 2026-07-27 在一台 Jetson Orin Nano Super 上的安装过程。延迟和精度数据行来自
  visionanalysis.org 背后的已验证结果快照，按硬件 jetson_orin 过滤，2026 年 6 月在 libreyolo
  1.2.0.dev0 上测得。导出与加载器行为读自
  libreyolo/export/exporter.py、libreyolo/export/tensorrt.py 和
  libreyolo/models/__init__.py。
snippets:
  prep:
    - label: 系统软件包与虚拟环境
      language: bash
      code: |
        # JetPack 不预装 pip，也不预装 venv 模块
        sudo apt update
        sudo apt install -y python3.12-venv python3-pip

        python3 -m venv ~/libreyolo
        source ~/libreyolo/bin/activate
        pip install -U pip wheel setuptools
  torch:
    - label: PyTorch，来自 CUDA 13 的 wheel 索引
      language: bash
      code: |
        pip install torch torchvision \
          --index-url https://download.pytorch.org/whl/cu130 \
          --extra-index-url https://pypi.org/simple
    - label: JetPack 没有附带的四个库
      language: bash
      code: |
        pip install nvidia-cudnn-cu13 nvidia-nccl-cu13 \
                    nvidia-cusparselt-cu13 nvidia-nvshmem-cu13
    - label: 如果 pip 非要 cuda-toolkit 13.0.3，就用 --no-deps 安装
      language: bash
      code: |
        # --no-deps 意味着 torch 的 Python 依赖也要手动一一写出来
        pip install --no-deps \
          torch torchvision \
          nvidia-cudnn-cu13 nvidia-nccl-cu13 \
          nvidia-cusparselt-cu13 nvidia-nvshmem-cu13 \
          filelock typing_extensions sympy networkx jinja2 markupsafe mpmath \
          fsspec numpy pillow
  ldd:
    - label: 直接点出下一个缺失的库，不用猜
      language: bash
      code: >
        ldd
        "$VIRTUAL_ENV/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so" \
          | grep "not found"

        # 一次性列出 torch 所有库里仍然缺失的东西：

        ldd "$VIRTUAL_ENV"/lib/python3.12/site-packages/torch/lib/*.so
        2>/dev/null \
          | grep "not found" | sort -u
  install:
    - label: 在 torch 之后装 LibreYOLO，不要在它之前
      language: bash
      code: |
        # torch 已经满足了，所以 pip 会原样保留那个 CUDA 构建
        pip install libreyolo

        # onnx extra 只在导出时需要，TensorRT 导出要经过
        # ONNX，所以在下面的导出小节之前先装上
        pip install "libreyolo[onnx]"
  verify:
    - label: 版本与设备
      language: python
      code: |
        import cv2
        import numpy
        import torch

        import libreyolo

        print("torch", torch.__version__, "cuda", torch.cuda.is_available())
        print("gpu", torch.cuda.get_device_name(0))
        print("libreyolo", libreyolo.__version__)
        print("cv2", cv2.__version__, "numpy", numpy.__version__)
      expect: |
        torch 2.13.0+cu130 cuda True
        gpu Orin
        libreyolo 1.4.0
        cv2 5.0.0 numpy 2.5.1
    - label: 然后跑一个真正的 kernel
      language: python
      code: |
        import torch

        x = torch.rand(2000, 2000, device="cuda")
        print(float((x @ x).sum()))
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO9, SAMPLE_IMAGE

        # 首次使用时自动下载检查点
        model = LibreYOLO9("libreyolo9s.pt", size="s")

        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes)
    - label: CLI
      language: bash
      code: >
        libreyolo predict --source
        https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        --model libreyolo9s.pt --save
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, LibreYOLO9, SAMPLE_IMAGE


        # 先写出 libreyolo9s.onnx，再据此构建 libreyolo9s.engine

        LibreYOLO9("libreyolo9s.pt", size="s").export(format="tensorrt",
        half=True)


        # engine 通过同一个入口加载回来

        result = LibreYOLO("libreyolo9s.engine").predict(SAMPLE_IMAGE)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model libreyolo9s.pt --format tensorrt --half
  power:
    - label: 功耗模式与时钟
      language: bash
      code: |
        sudo nvpmodel -q      # 这块板子提供哪些模式，以及当前生效的是哪个
        sudo nvpmodel -m 0    # 这里测试的这块板子上的最高模式
        sudo jetson_clocks

        tegrastats            # 实时负载，nvidia-smi 在 Tegra 上功能有限
source_hash: c07ff908503e89b5
---

## 这个页面记录了什么

这个页面记录的是一套端到端验证过的配置，不是支持矩阵。板子是一台 Jetson Orin
Nano Super Developer Kit，8 GB 内存，运行 JetPack 7.2（L4T R39.2、Ubuntu 24.04、
CUDA 13、Python 3.12.3），在它上面跑起来的技术栈是 `libreyolo 1.4.0` 配
`torch 2.13.0+cu130`、OpenCV 5.0.0 和 NumPy 2.5.1。`torch.cuda.is_available()`
返回 `True`，GPU 自报为 `Orin`。

其他 JetPack 版本、其他 Jetson 板子和其他 CUDA 版本都没有测试。下面这套步骤就是
在那个组合上跑通的那一套。

那次运行是 2026-07-27 针对 LibreYOLO 1.4.0 做的，还没有在 1.5.0 的硬件上重跑：
这是 1.5.0 文档树里唯一一个仍然带着 1.4.0 验证的页面，所以它的 front matter 写的
是 `last_verified: "1.4.0"`。1.5.0 的改动没有一处涉及这里描述的安装路径、四个缺失
的库或导出参数，所以这些命令预计仍然成立，但下面输出里的版本号是 1.4.0 打印出来
的，不是 1.5.0 的实测。

其中有两点和大多数 Jetson 教程说的相反。这些 wheel 就是为 CUDA 13 发布的普通
aarch64 构建，所以不需要任何 Jetson 专用的 torch 构建。而 JetPack 没有附带那些
wheel 所链接的四个库，所以 `import torch` 会一个库接一个库地失败，直到四个全部装
上。

## 安装

JetPack 镜像里既没有 pip，也没有 `venv` 模块，所以这两样要先装。

<code-tabs name="prep" />

8 GB 的板子对较大的检查点（checkpoint）来说很紧张。在加载它们之前先在 NVMe 上加
一块 swap，可以避免跑到一半被内存不足杀掉。

然后是 PyTorch。CUDA 13 索引里放的是 aarch64 wheel；额外索引则从 PyPI 提供纯
Python 的依赖。

<code-tabs name="torch" />

四个 `nvidia-*-cu13` wheel 是最容易漏掉的部分。JetPack 提供的是 GPU 驱动，不是
cuDNN、NCCL、cuSPARSELt 或 NVSHMEM，而缺了它们 torch 就拒绝导入。一次把四个都装
上，比一个异常一个异常地把它们找出来要快。

第三段代码针对的是一个具体的失败：torch 的 CUDA 13 构建在依赖元数据里要求
`cuda-toolkit==13.0.3`，而 PyPI 上没有它的 aarch64 wheel，于是还没下载任何东西，
解析就已经失败了。`--no-deps` 跳过解析器，这意味着每一个依赖都得在命令行上写出来。

LibreYOLO 最后装。先装它会让 pip 自己挑一个 torch，而在这个平台上那不是 CUDA
构建。

<code-tabs name="install" />

其余每一个依赖都能解析到预编译的 aarch64 wheel，包括 OpenCV、NumPy、SciPy、
pycocotools 和 safetensors。没有任何东西需要从源码编译。

## 确认 CUDA 能用

<code-tabs name="verify" />

第二段代码和第一段一样重要。为错误的 GPU 架构编译出来的 wheel 照样会报告
`torch.cuda.is_available() == True`，然后在第一次真正的运算上以
`CUDA error: no kernel image is available for execution on the device` 失败。在设
备上做一次矩阵乘法，才是能抓住它的检查。

## 跑一次预测

<code-tabs name="predict" />

`predict` 返回的 `Results` 对象和在其他任何平台上一样，所以模型页面原样适用。

## 导出到 TensorRT

在这块板子上，对于在每个运行时里都测过的全部 55 个模型，TensorRT 都比 PyTorch 和
ONNX Runtime 快。

<code-tabs name="export" />

`format="tensorrt"` 会先写出一张 ONNX 图，再据此构建 engine，所以必须装上 `onnx`
extra。`LibreYOLO()` 按文件后缀分派，因此 `.engine` 文件和 `.pt` 检查点走的是同一
个调用。

不要在 Jetson 上用 `tensorrt` 这个 pip extra。它固定依赖 `tensorrt-cu12`，那是
CUDA 12 的构建，而这里是 CUDA 13 平台。改用 JetPack 装好的 TensorRT。如果
`import tensorrt` 在虚拟环境里失败、在环境外却正常，就用 `--system-site-packages`
重建环境，让系统模块可见。

序列化后的 TensorRT engine 与设备、GPU 架构以及构建它的 TensorRT 版本绑定。在工作
站上构建的 engine 在 Jetson 上加载不了，所以构建这一步要在板子上跑。

## 在这块板子上的实测

每张图像的延迟，批大小 1，端到端包含预处理和后处理，在 COCO val2017（500 张图像的
子集）上，`conf=0.001`、`max_det=300`。这是测过的 58 个模型里的五个：

| 模型 | 输入（px） | PyTorch FP32（毫秒） | ONNX FP32（毫秒） | TensorRT FP32（毫秒） | TensorRT FP16（毫秒） | mAP 50-95 |
|---|---:|---:|---:|---:|---:|---:|
| DEIMv2-Atto | 320 | 64.9 | 22.8 | 12.3 | 11.2 | 27.49 |
| YOLOX-Tiny | 416 | 49.2 | 31.8 | 23.0 | 19.4 | 35.45 |
| YOLO9-t | 640 | 101.2 | 53.8 | 36.0 | 29.1 | 41.78 |
| RT-DETR-r18 | 640 | 98.3 | 103.7 | 45.3 | 25.7 | 49.72 |
| D-FINE-s | 640 | 96.8 | 96.1 | 44.7 | 33.1 | 53.45 |

mAP 那一列是 TensorRT FP16 这次运行自己的分数。在四个运行时里都测过的 55 个模型
中，PyTorch FP32 分数和 TensorRT FP16 分数之间最大的差距是 0.59 个点，出现在
DEIMv2-X 上。这些运行时的差别在速度，不在精度。

在这 55 个模型上，TensorRT FP32 全都比 PyTorch 和 ONNX Runtime 快。TensorRT FP16
同样在全部 55 个上都快过 PyTorch FP32，快 1.68 倍到 6.22 倍，中位数 3.39 倍。会来
回变的是 ONNX Runtime：它在这 55 个里有 23 个比 PyTorch 慢，其中就有 RT-DETR-r18
那一行。

每一个数字背后的条件：`libreyolo 1.2.0.dev0`、`torch 2.12.0+cu130`、Python
3.12.3、CUDA 13、驱动 595.78、ONNX Runtime 1.24.0，测量于 2026 年 6 月。Jetson 上
的延迟还取决于当前生效的功耗模式，而基准测试记录里没有带上这一项。

<code-tabs name="power" />

全部 223 次运行，包括另外 53 个模型和完整的精度列，都发布在
[Vision Analysis 的 Jetson Orin 页面](https://www.visionanalysis.org/hardware/jetson_orin)。

## 故障排查

### import torch 失败并报出某个共享库

上面四个库里少了一个。与其猜是哪一个，不如直接从二进制文件里读出来：

<code-tabs name="ldd" />

每一条缺失记录对应一个 wheel：

| 缺失的库 | Wheel |
|---|---|
| cuDNN | `nvidia-cudnn-cu13` |
| NCCL | `nvidia-nccl-cu13` |
| cuSPARSELt | `nvidia-cusparselt-cu13` |
| NVSHMEM | `nvidia-nvshmem-cu13` |

### torch 警告说没有任何构建支持这块 GPU

在能正常工作的配置上，第一次 CUDA 调用会打印出这个：

```text
UserWarning: Found GPU0 Orin which is of compute capability (CC) 8.7.
The following list shows the CCs this version of PyTorch was built for and the hardware CCs it supports:
- 8.0 which supports hardware CC >=8.0,<9.0 except {8.7}
- 9.0 which supports hardware CC >=9.0,<10.0
- 10.0 which supports hardware CC >=10.0,<11.0 except {10.1}
- 11.0 which supports hardware CC >=11.0,<12.0
- 12.0 which supports hardware CC >=12.0,<13.0
No published PyTorch CUDA builds for release 2.13.0+cu130 support this GPU.
```

在这块板子上，这条警告只是表面现象。wheel 里带的是 `sm_80` kernel，Orin 能执行它
们。同一个索引里更早的那个 wheel 也出现过同样的警告，而上面每一行基准测试数据都是
它跑出来的。用 CUDA 检查里的那次矩阵乘法去确认，而不是去相信或不相信这条消息。

### CUDA error: no kernel image is available for execution on the device

装上的 wheel 是为另一种 GPU 架构编译的。英伟达 `sbsa` 索引里的 wheel 就是这种情
况，它们针对的是服务器 ARM GPU，而不是 Jetson 芯片。请从安装小节里的 CUDA 13 索引
重新安装。

### pip 找不到 cuda-toolkit 13.0.3

它没有 aarch64 wheel。用安装小节里的 `--no-deps` 写法，把 torch 的依赖显式写出来。

### libnvpl_lapack_lp64_gomp.so.0: cannot open shared object file

aarch64 的 torch wheel 链接了 NVIDIA Performance Libraries 来做 CPU 数学运算。装
上它们，并放进库路径：

```bash
pip install nvpl-lapack nvpl-blas --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130/
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvpl/lib:$LD_LIBRARY_PATH"
```

那个索引对这两个 CPU 库来说没问题。是它的 torch 构建才会导致上面那种「no kernel
image」失败。

### 不适配 JetPack 7.2 的 wheel 来源

| 来源 | 在 Orin Nano Super 上的结果 |
|---|---|
| `pypi.jetson-ai-lab.io/sbsa/cu130` 的 torch | 为服务器 ARM GPU 编译。能导入，报告 CUDA 可用，然后以「no kernel image is available for execution on the device」失败。 |
| `pypi.jetson-ai-lab.io/jp6/*` 的 torch | CUDA 12 和 Python 3.10 的构建。在这个镜像的 Python 3.12 上装不上。 |
| JetPack 6 的 PyTorch 容器 | 在 JetPack 7 宿主机上，CUDA 初始化以 error 801 失败。 |
| 从源码编译 torch | 可行，但在 8 GB 的板子上要花好几个小时，而且一旦装上 CUDA 13 wheel 就没必要了。 |

## DeepStream

如果要的是完整的视频流水线而不是一个 Python 循环，就用 `deepstream=True` 导出，再
让图通过 `nvinfer` 运行。那条路有自己的页面，包括生成的 `nvinfer` 配置文件、检测框
解析器的编译，以及已知的坑：[DeepStream](/docs/export/deepstream)。

DeepStream 流水线本身是在 x86 独立 GPU 上验证的，不是在 Jetson 上。导出契约不依赖
架构，但在 aarch64 上跑这条流水线仍然欠着。

## 尚未验证

- 7.2 以外的 JetPack 版本，以及 R39.2 以外的 L4T 版本。
- Orin Nano Super 8 GB 以外的 Jetson 板子。
- 在板子上训练。推理和导出跑过了；训练没有跑。
- INT8 engine。这块板子只有 FP32 和 FP16 的数据行。
- 大于 1 的批大小。上面每一次测量都是批 1。
