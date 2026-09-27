---
title: NVIDIA DeepStream
seo_title: 在 NVIDIA DeepStream 上运行 YOLO 模型
description: >-
  为 NVIDIA DeepStream 导出 LibreYOLO 模型：一个 ONNX 图，加上自动生成的 nvinfer
  配置。构建解析器和跑通流水线的确切命令。
lead: >-
  NVIDIA DeepStream 通过 nvinfer 元件跑推理，而 nvinfer 需要一个 ONNX
  图、一份与之匹配的配置文件和一个检测框解析器。在 ONNX 导出上设置 deepstream=True 会写出前两个，并把它们接到第三个上。
keywords:
  - deepstream yolo
  - yolo 导出 deepstream
  - nvinfer 配置文件
  - deepstream 检测框解析器
  - config_infer_primary
  - NvDsInferParseYolo
  - deepstream-app
  - deepstream tensorrt 引擎
  - jetson deepstream
meta:
  - label: 开关
    value: 'export(format="onnx", deepstream=True)'
    mono: true
  - label: 输出
    value: 一个 ONNX 图、config_infer_primary_<stem>.txt 和 <stem>_labels.txt
  - label: 覆盖范围
    value: 九项任务下的 43 个家族与任务组合
  - label: 解析器
    value: >-
      NvDsInferParseYolo，来自 Marcos Luciano 的 DeepStream-Yolo 项目，采用 MIT
      许可。每台设备编译一次。
    links:
      - label: github.com/marcoslucianops/DeepStream-Yolo
        href: 'https://github.com/marcoslucianops/DeepStream-Yolo'
  - label: 可用性
    value: 随 v1.5.0 发布。2026-08-08 通过 pull request 728 合入 dev。
    links:
      - label: pull request 728
        href: 'https://github.com/LibreYOLO/libreyolo/pull/728'
      - label: issue 648
        href: 'https://github.com/LibreYOLO/libreyolo/issues/648'
  - label: 运行时已验证
    value: DeepStream 8.0.0，跑在 RTX 5070 Ti 上，仅检测，2026-08-08
verification: >-
  根据 2026-08-08 的运行时验证写成。家族列表、配置键和默认值读自 commit 5f81e11e 的
  libreyolo/export/deepstream.py 与 libreyolo/export/exporter.py，该 commit 当天通过
  pull request 728 合入 dev。
snippets:
  install:
    - label: 安装
      language: bash
      code: |
        pip install "libreyolo[onnx]"
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO9, LibreDFINE


        # 在工作目录下写出 libreyolo9s.onnx、config_infer_primary_libreyolo9s.txt

        # 和 libreyolo9s_labels.txt

        LibreYOLO9("libreyolo9s.pt", size="s").export(format="onnx",
        deepstream=True)


        # 每个检测模型放在各自的目录里：所有检测配置指向的引擎缓存文件名都一样

        # 参见「已知陷阱」

        LibreDFINE("LibreDFINEs.pt", size="s").export(format="onnx",
        deepstream=True)
    - label: 参数
      language: python
      code: |
        model.export(
            format="onnx",     # 其他任何格式都会拒绝 deepstream=True
            deepstream=True,
            conf=0.25,         # 填 pre-cluster-threshold（以及相应任务上的 classifier-threshold、
                               # segmentation-threshold）
            iou=0.45,          # 填 nms-iou-threshold，cluster-mode=4 时省略
            batch=1,           # 填 batch-size 和引擎缓存文件名
            half=False,        # True 会把配置标成 network-mode=2（fp16 构建）
            int8=False,        # True 会把配置标成 network-mode=1
            dynamic=True,      # ONNX 图里的动态 batch 轴
            imgsz=640,         # 填 infer-dims=3;H;W
        )

        # deepstream=True 和 nms=True 互斥：DeepStream 在自己的聚类阶段做抑制，
        # 所以图里不内嵌任何东西
    - label: 先下载 D-FINE 权重
      language: bash
      code: |
        curl -L -o LibreDFINEs.pt \
          https://huggingface.co/LibreYOLO/LibreDFINEs/resolve/main/LibreDFINEs.pt
  gpu:
    - label: 动手之前先确认 GPU 直通
      language: bash
      code: |
        docker run --rm --gpus all nvcr.io/nvidia/tritonserver:26.04-py3 \
          nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
      expect: |
        name, driver_version, compute_cap
        NVIDIA GeForce RTX 5070 Ti, 591.86, 12.0
  parser:
    - label: build_parser.sh，在 DeepStream 容器内运行
      language: bash
      code: >
        set -e

        git clone --depth 1
        https://github.com/marcoslucianops/DeepStream-Yolo.git


        # 这个镜像里的 /usr/local/cuda-12 是个 stub，编译会死在它上面，报

        # "fatal error: crt/host_defines.h: No such file or directory"。找一个真正

        # 带这个头文件的 toolkit；在 8.0 镜像上就是 cuda-12.5

        CUDA_DIR=$(readlink -f /usr/local/cuda)

        [ -f "$CUDA_DIR/include/crt/host_defines.h" ] || \
          CUDA_DIR=$(ls -d /usr/local/cuda-*.* | sort -Vr | \
                     while read d; do [ -f "$d/include/crt/host_defines.h" ] && echo "$d" && break; done)

        # 镜像自带 libcublas.so.12 和 libcublas.so.12.8.4.1，却没有 -lcublas 需要的

        # 无版本号 libcublas.so，所以链接一步会失败并报

        # "/usr/bin/ld: cannot find -lcublas"。把链接器想要的名字给它

        mkdir -p /tmp/cudalibs

        for lib in cublas cublasLt cudart; do
          real=$(find /usr/local -name "lib${lib}.so.1*" | grep -v stubs | sort -V | tail -1)
          ln -sf "$real" "/tmp/cudalibs/lib${lib}.so"
        done

        export LIBRARY_PATH="/tmp/cudalibs:$LIBRARY_PATH"


        make -C DeepStream-Yolo/nvdsinfer_custom_impl_Yolo
        CUDA_VER="${CUDA_DIR##*/cuda-}"
    - label: 实例分割用的是另一个解析器
      language: bash
      code: >
        git clone --depth 1
        https://github.com/marcoslucianops/DeepStream-Yolo-Seg.git

        make -C DeepStream-Yolo-Seg/nvdsinfer_custom_impl_Yolo_seg \
          CUDA_VER="${CUDA_DIR##*/cuda-}"
  run:
    - label: deepstream_app_config.txt
      language: text
      code: >
        [application]

        enable-perf-measurement=1

        perf-measurement-interval-sec=5

        gie-kitti-output-dir=kitti


        [tiled-display]

        enable=0


        [source0]

        enable=1

        type=3

        uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4

        num-sources=1

        gpu-id=0


        [streammux]

        gpu-id=0

        batch-size=1

        batched-push-timeout=40000

        width=1920

        height=1080

        live-source=0


        [primary-gie]

        enable=1

        gpu-id=0

        gie-unique-id=1

        config-file=config_infer_primary_libreyolo9s.txt


        [osd]

        enable=1

        border-width=2

        text-size=15


        [sink0]

        enable=1

        type=1

        sync=0


        [tests]

        file-loop=0
    - label: 运行它
      language: bash
      code: |
        deepstream-app -c deepstream_app_config.txt
      expect: |
        App run successful
    - label: 两步都在一个容器里跑
      language: bash
      code: |
        docker run --rm --gpus all -v "$PWD:/work" -w /work \
          nvcr.io/nvidia/deepstream:8.0-samples-multiarch \
          bash -c "bash build_parser.sh && deepstream-app -c deepstream_app_config.txt"
source_hash: 1ee91c265753dd9a
---

## 可用性

DeepStream 导出随 v1.5.0 发布。它在 2026-08-08 通过 pull request 728 合入了
`dev`，所以现在装的版本里已经有它，不需要固定到某个分支。

<code-tabs name="install" />

如果你在 2026-08-08 之前克隆过 `deepstream-export` 分支，把它换掉。那个分支做过
rebase 和强制推送，旧历史里缺了一个修复，没有它这些导出在 CUDA 机器上根本跑不起来。

## 导出会写出什么

`model.export(format="onnx", deepstream=True)` 会并排写出三个文件。对
`libreyolo9s.pt` 来说：

- `libreyolo9s.onnx`，检测图，一个形状为 `(batch, num_detections, 6)` 的输出张量，
  每行是网络输入像素坐标下的 `[x1, y1, x2, y2, score, class_id]`。
- `config_infer_primary_libreyolo9s.txt`，一份 `nvinfer` 配置，带着该家族的预处理
  常数、类别数、阈值和解析器接线。
- `libreyolo9s_labels.txt`，每行一个类别名。

只要检查点（checkpoint）带类别名，就会有一个标签文件。深度模型没有类别名，所以既
不会有这个文件，也不会有 `labelfile-path` 键。

LibreYOLO 不产出 `.so`。DeepStream 加载的那个 `.so` 是来自
`marcoslucianops/DeepStream-Yolo` 的检测框解析器，每台设备编译一次，而且不管你把它
对到哪个 LibreYOLO 检测器上，都是同一个二进制。模型是那份 ONNX。分类和语义分割完全
不需要解析器，因为 `nvinfer` 自己会做后处理。

## 导出模型

<code-tabs name="export" />

文件不在磁盘上时，`LibreDFINE._load_weights` 会抛 `FileNotFoundError`，不会尝试下载，
所以要先自己把 `LibreDFINEs.pt` 取下来。这个缺口记在
[issue #727](https://github.com/LibreYOLO/libreyolo/issues/727) 里。YOLO9 的权重会在
首次使用时下载。

这个开关只有 Python 有。这个分支上的 `libreyolo export` 没有 `deepstream` 选项，而且
CLI 是按一份固定列表拼出导出参数的，不会把未知的键透传下去。

## 构建检测框解析器

检测需要这个解析器库，实例分割需要另一个，其余任务都不需要。DeepStream 8.0 镜像上有
两件事会让文档里那条构建命令失败，两件都是环境问题，不是 LibreYOLO 的问题。

镜像在 `/usr/local` 下自带 `cuda`、`cuda-12`、`cuda-12.5`、`cuda-12.8` 和
`cuda-12.9`。只有 `cuda-12.5` 的 toolkit 是完整的。它还自带 `libcublas.so.12` 和
`libcublas.so.12.8.4.1`，却没有 `-lcublas` 要解析到的无版本号 `libcublas.so`。下面这段
脚本把两件事都绕开了。

<code-tabs name="parser" />

然后把生成的配置里的 `custom-lib-path` 指向编译出来的
`libnvdsinfer_custom_impl_Yolo.so`。生成的值是相对路径
`nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so`，`deepstream-app` 从
`DeepStream-Yolo` 的检出目录里运行时它能解析到，其他情况下需要改。

## 运行流水线

在把时间花到别的事情上之前，先确认容器能看见 GPU。这也是那次验证运行最先做的检查，
跑在 WSL2 下的一块 Blackwell 卡上。

<code-tabs name="gpu" />

那次验证运行用一个文件源驱动 `deepstream-app`，没有显示 sink，开着屏幕显示，并设了
`gie-kitti-output-dir`，让每一帧的检测结果都以 KITTI 文本落盘。带这些设置的配置：

<code-tabs name="run" />

`nvinfer` 会在首次运行时从 ONNX 构建 TensorRT 引擎，并把它缓存在模型旁边，所以第一次
运行要付构建引擎的代价，后面的运行直接加载缓存。

## 生成的配置

下面两份配置都是导出器为那次验证运行写出来的，之后没有改过。

| 键 | YOLO9-s | D-FINE-s |
|---|---|---|
| `net-scale-factor` | 0.003921568627 | 0.003921568627 |
| `model-color-format` | 0 | 0 |
| `infer-dims` | 3;640;640 | 3;640;640 |
| `maintain-aspect-ratio` | 1 | 0 |
| `symmetric-padding` | 0 | 0 |
| `network-type` | 0 | 0 |
| `num-detected-classes` | 80 | 80 |
| `cluster-mode` | 2 | 4 |
| `parse-bbox-func-name` | NvDsInferParseYolo | NvDsInferParseYolo |
| `pre-cluster-threshold` | 0.25 | 0.25 |
| `nms-iou-threshold` | 0.45 | |
| `topk` | 300 | 300 |

两份配置有三处不同：`maintain-aspect-ratio`、`cluster-mode`，以及
`nms-iou-threshold` 到底在不在。D-FINE 的配置把那个键整个省掉了，`cluster-mode=4`
要的就是这样。

每个目标最多只出一个预测的 head 会拿到 `cluster-mode=4`，DeepStream 就不对它们做聚类；
聚类会把本来就彼此不同的检测结果合并掉。这覆盖 `rfdetr`、`dfine`、`deim`、`deimv2`、
`ec`、`rtdetr`、`rtdetrv2`、`rtdetrv4` 和 `yolo9_e2e`。网格 head 和锚框 head 拿到的是
`cluster-mode=2` 加 `nms-iou-threshold`。

检测配置里还带着 `engine-create-func-name=NvDsInferYoloCudaEngineGet`，它把构建引擎的
活儿交给解析器库。引擎缓存文件名被写死就是因为它，也是已知陷阱里那个冲突的来源。

## 支持的任务与家族

有 43 个家族与任务组合能导出。`libreyolo/export/deepstream.py` 里的
`deepstream_supported_tasks()` 和 `deepstream_supported_families(task)` 在运行时返回
同样的列表。

| 任务 | `network-type` | 解析器库 | 家族 |
|---|---|---|---|
| 检测 | 0 | DeepStream-Yolo | yolo9, yolo9_p2, yolo9_e2e, yolo1, yolo2, yolo3, yolo4, yolo7, yolox, yolonas, rtmdet, picodet, rfdetr, dfine, deim, deimv2, ec, rtdetr, rtdetrv2, rtdetrv4 |
| 分类 | 1 | 不需要 | mobilenetv4, convnext, efficientnetv2, resnet, dinov2 |
| 语义分割 | 2 | 不需要 | pidnet, eomt, dinov2, lingbotvision |
| 实例分割 | 3 | DeepStream-Yolo-Seg | rfdetr, dfine, ec |
| 姿态 | 100 | 不需要 | yolo9, yolonas, rfdetr, ec |
| 深度 | 100 | 不需要 | depth_anything, zipdepth |
| 复原 | 100 | 不需要 | nafnet, realesrgan, swinir |
| 抠图 | 100 | 不需要 | birefnet |
| 视线 | 100 | 不需要 | l2cs |

`network-type=100` 表示 DeepStream 对这个任务没有后处理器。这些配置会设
`output-tensor-meta=1`，图的原生输出原封不动地透传出去，由应用从张量元数据里解码。
多输出的图在这里没问题：每个输出层都带着和普通 ONNX 导出一样的输出名和动态轴进入
元数据。

实例分割的每一行是检测行后面跟上该实例的掩码，按 `(netH / 4, netW / 4)` 展平，也就是
seg 解析器写死的那个分辨率，以概率的形式供 `segmentation-threshold` 使用。

分类和视线估计以二级推理的方式运行。在生成的配置里设 `process-mode=2` 和
`operate-on-gie-id`，就能把分类器接在检测器后面。视线估计是只有 head 的契约，每个输入
一张人脸裁剪图，所以它前面需要一个人脸检测器。

有三个家族是故意缺席的。`segformer` 没有接到共用的语义导出契约上，任何格式都导不出
ONNX。RTMDet-Ins 和 YOLO9 的实例分割导出在 LibreYOLO 内部就被挡住了。
`depth_anything3` 没有导出实现。

表里有两行背后是检查点缺口。EoMT 的语义检查点只发布了 `l` 一个，DINOv2 分类根本没有
发布过检查点，所以那个组合需要你自己微调的权重。

## 预处理差异

`nvinfer` 按通道算 `net-scale-factor * (x - offsets)`，缩放是个标量，表达不了逐通道的
标准差。需要逐通道标准差的家族（`rfdetr`、`ec`、用 DINO 骨干的 `deimv2` 尺寸、
`rtmdet`、`picodet`，以及所有分类家族）把归一化烘进了导出的图里，生成的配置则给图喂上
与之匹配的原始输入空间。

几何上，LibreYOLO 自己的 Python 流水线和 `nvinfer` 仍然有分歧：

- letterbox 家族（`yolo9`、`yolox`、`yolonas`、`rtmdet`、`yolo2`、`yolo3`、`yolo4`、
  `yolo7`）原生用灰色填充。`nvinfer` 填黑色。
- `yolonas` 检测原生是在 640 的画布里把最长边缩到 636。`nvinfer` 的
  `maintain-aspect-ratio` 用满 640。
- 分类原生是先缩短边再中心裁剪。`nvinfer` 把整帧或目标 ROI 拉伸到网络输入尺寸，所以
  裁得很紧的主体会有差别。
- EoMT 原生对语义分割跑滑窗切块。导出的图是单张拉伸后的画布，更快，也更不准。
- `pidnet` 输出的类别图是输入分辨率的 1/8，`lingbotvision` 是 1/16。DeepStream 会把
  类别图上采样后再显示。

ONNX 一致性关卡喂进去的是已经预处理好的张量，所以它检查的是图的输出，抓不到配置里搞错
的通道顺序或填充策略。在部署要求严格一致的工作负载之前，先在你自己的数据上验证。

## 已知陷阱

### 同一目录下的两个检测模型会加载彼此的引擎

每一份检测配置都带着同一行：

```ini
model-engine-file=model_b1_gpu0_fp32.engine
```

解析器的引擎构建器要求这个文件名，而且它不随模型变化。把第二个检测模型导出到同一个
目录里，第二次运行就会加载第一个模型缓存下来的引擎。什么都不会崩，只是检测框是错的。
给每个检测模型各自一个目录。那次验证运行必须先把 D-FINE 隔离到单独目录里，才谈得上
测它。

### 一个框只能带一个类别

`nvinfer` 的行格式是 `[x1, y1, x2, y2, score, class_id]`，一个框一个类别，所以导出会把
类别分数塌缩到 argmax。`predict` 下报成两个类别的框，到这里只留下一个。实测的例子：
LibreYOLO 在同一个框上报 `vase 0.773` 和 `bottle 0.383`，DeepStream 的图留下了 `vase`。
这是解析器行格式的直接结果，不脱离那份契约就改不了，所以它是预期行为，不是回归。

## 已验证

在两种检测器 head 类型上，`deepstream-app` 都跑到了 EOS 并打出 `App run successful`，
素材是英伟达自带的 `sample_1080p_h264.mp4`（1443 帧），并开启了逐帧 KITTI 转储。

| | YOLO9-s | D-FINE-s |
|---|---|---|
| head 类型 | 网格 | 一对一 |
| `cluster-mode` | 2 | 4 |
| `maintain-aspect-ratio` | 1 | 0 |
| 有检测结果的帧数 | 1443 | 1443 |
| 检测总数 | 18031 | 71105 |

两个模型在全部 1443 帧上的类别直方图都是汽车第一、人第二，对街景来说这是对的。检测数上
四倍的差距，正是 `cluster-mode` 的差异在起作用：D-FINE 在 `cluster-mode=4` 下不做聚类，
所以每个超过阈值的 query 都会留下来，几乎重复的那些也在内。

两个各自独立训练出来的模型，把画面里最主要的目标放在了同一个位置：

```text
YOLO9  bus  [706.72,  0.82, 1916.34, 1062.97]  conf 0.965
D-FINE bus  [702.73,  2.93, 1916.24, 1069.32]  conf 0.965
```

那次运行确立了五件事：TensorRT 能在 sm_120 上从导出的 ONNX 构建引擎，`nvinfer` 接受
生成配置里的每一个键，`NvDsInferParseYolo` 能正确读出张量布局，检测框落在源分辨率
1920x1080 的坐标里，标签能在生成的标签文件里解析到。

它运行的环境：

| 组件 | 值 |
|---|---|
| 宿主机操作系统 | Windows 11 Pro 26200 |
| GPU | NVIDIA GeForce RTX 5070 Ti，16 GB |
| 驱动 | 591.86 |
| 计算能力 | 12.0（Blackwell，sm_120） |
| 容器运行时 | Docker Desktop 29.4.3，WSL2 后端 |
| DeepStream 镜像 | `nvcr.io/nvidia/deepstream:8.0-samples-multiarch` |
| DeepStream 版本 | 8.0.0 |
| 容器内 CUDA | 12.8.1 |
| 解析器 | `marcoslucianops/DeepStream-Yolo` 的 HEAD |

除了这次流水线运行之外，`tests/unit/test_deepstream_export.py` 覆盖了图适配器和生成的
配置键，它的 35 个测试在这个 commit 上全部通过。

## 未验证

写在这里，是为了不让上面的范围被读得比它实际的更宽。

- Jetson 和 aarch64。导出契约不依赖架构，但流水线只在 x86 独立 GPU 上跑过。
- 43 个组合里的 41 个。只有 `yolo9` 检测和 `dfine` 检测真正走过 DeepStream。分类、
  语义分割、实例分割和输出原始张量的那些任务由单元测试和 ONNX 一致性检查覆盖，不是靠
  一次流水线运行。
- FP16 和 INT8。只跑过 `network-mode=0`。
- 多路流和批处理。一路源，`batch-size=1`。
- 在真值（ground truth）数据集上的精度。检测结果只做了语义合理性和跨模型一致性的检查，
  没有在 DeepStream 里评成 mAP。
