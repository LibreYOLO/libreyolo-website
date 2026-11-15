---
title: Axelera Metis
seo_title: 从 LibreYOLO 导出到 Axelera Metis
description: >-
  把 LibreYOLO 模型编译成面向 Axelera Metis AIPU 的 INT8 .axm：需要你自己安装的 SDK、导出参数、一致性检查，以及哪些家族和尺寸能编译。
lead: >-
  format="axelera" 把 LibreYOLO 模型编译成面向 Axelera Metis AIPU 的 INT8 .axm，导出的目录可以通过 LibreYOLO() 重新加载，predict、val 和 track 的调用方式与 PyTorch 模型完全相同。1.8.0 新增，标记为预览。
keywords:
  - libreyolo axelera
  - axelera metis
  - metis aipu
  - voyager sdk
  - axelera-devkit
  - axelera-rt
  - yolo 导出 npu int8
  - axm 模型
last_verified: "1.8.0"
meta:
  - label: 参数
    value: 'export(format="axelera")'
    mono: true
  - label: 精度
    value: 仅 INT8。half=True、quantize=16 和 quantize=32 会抛错。
  - label: 编译主机
    value: Linux x86_64（Ubuntu 22.04 或 24.04，或 WSL2），Python 3.10 到 3.13。编译不需要 Metis 设备。
  - label: SDK
    value: 来自 Axelera 自有包索引的 axelera-devkit 和 axelera-rt 1.9.0，不是 LibreYOLO 的 extra
  - label: 输出
    value: "weights/<stem>_axelera_model/"
    mono: true
  - label: 状态
    value: 预览。23 个家族已在一块 Metis M.2 Max 上用 SDK 1.9.0 完成编译和测量。
verification: >-
  读自 dev 分支 1.8.0 候选版本上的 docs/axelera.md、libreyolo/export/axelera.py、libreyolo/export/support.py 和 libreyolo/cli/commands/export.py。精度数字来自库自己做的 INT8 与 FP32 配对测量，本站没有重跑。
snippets:
  install:
    - label: 专用环境
      language: bash
      code: |
        python -m venv ~/venvs/axelera && source ~/venvs/axelera/bin/activate
        pip install "libreyolo[onnx]"   # 编译需要 onnx 这个 extra

        # 最后安装 SDK：它的 wheel 会锁定 LibreYOLO 也在用的包
        # 编译（导出）：
        pip install --extra-index-url https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple axelera-devkit==1.9.0
        # 运行（.axm 推理）：
        pip install --extra-index-url https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple axelera-rt==1.9.0
    - label: 让 LibreYOLO 来安装
      language: bash
      code: |
        # 不设置的话，缺少 SDK 时会报错并给出上面的 pip 命令
        export LIBREYOLO_AUTOINSTALL=1
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")
        path = model.export(format="axelera")   # weights/LibreYOLO9t_axelera_model

        # 用你自己领域的数据校准，并写出一致性报告
        path = model.export(
            format="axelera",
            data="my_dataset.yaml",
            fraction=0.5,
            verify=True,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO9t.pt format=axelera
        libreyolo export model=LibreYOLO9t.pt format=axelera verify=true
  run:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        npu = LibreYOLO("weights/LibreYOLO9t_axelera_model")   # 或者其中的 .axm
        results = npu(SAMPLE_IMAGE, conf=0.25)
        metrics = npu.val(data="coco128.yaml")
        for r in npu.track("video.mp4"):
            print(r.track_id)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=weights/LibreYOLO9t_axelera_model source=image.jpg
        libreyolo val model=weights/LibreYOLO9t_axelera_model data=coco128.yaml
        libreyolo track model=weights/LibreYOLO9t_axelera_model source=video.mp4
source_hash: 19ff2c735643dec7
---

## 安装

Axelera SDK 是 Axelera 的专有软件，受 Axelera 的 EULA 约束，从 Axelera 自有的包索引分发。
它不在 PyPI 上，也不是 LibreYOLO 的 extra，LibreYOLO 从不捆绑它。`axelera-devkit` 负责编译；
`axelera-rt` 运行编译好的模型，也是部署机器上唯一需要的那个。

<code-tabs name="install" />

请使用专用的虚拟环境。SDK 1.9.0 的 wheel 锁定了几个 LibreYOLO 也在用的包，pip 安装时会把它们降级：

| 包 | 锁定为 |
|---|---|
| `torch` | `>=2.7,<2.15` |
| `typer` | `>=0.12,<0.13` |
| `click` | `==8.1.8` |
| `rich` | `<14` |
| `onnx` | `<1.19` |
| `opencv-python` | `<5` |
| `pillow` | `<13` |

LibreYOLO 在这些版本下都能工作。在 SDK 之后再安装其他 LibreYOLO extra 可能会把它们重新升级，所以最后装 SDK。
缺少某个包时，导出或加载会失败并给出确切的 `pip install` 命令；设置 `LIBREYOLO_AUTOINSTALL=1`
可以让 LibreYOLO 替你执行它，并记录到日志。

运行需要一台 Metis 设备、Axelera 的 `metis-dkms` 内核驱动（SDK 1.9 需要 1.7.0 或更高版本）以及
`axelera-rt`。`axdevice` 会列出每台设备及其固件，`libreyolo checks` 会列出它看到的 Axelera 包和 Metis 设备。

## 导出

<code-tabs name="export" />

| 参数 | Axelera 下的行为 |
|---|---|
| `imgsz` | 固化进编译后的模型；必须是 32 的倍数。默认：检查点（checkpoint）的尺寸 |
| `data` | 校准数据集 YAML。detect 和 pose 默认用 `coco128.yaml` |
| `fraction` | 使用校准划分中的比例。推荐 100 到 400 张图像；少于 100 张会记录警告 |
| `batch` | 仅用于校准加载器的批大小。编译后的模型一次只接收一张图像 |
| `device` | 用于追踪模型的 Torch 设备，不是 Metis 设备 |
| `quantize`、`int8` | INT8 始终开启。`quantize=16`、`quantize=32` 和 `half=True` 会抛错 |
| `dynamic` | 始终是静态的；`dynamic=True` 会抛错 |
| `opset` | ONNX 中间图的 opset，14 到 17（默认 17） |
| `nms` | 不支持：NMS 在主机上运行 |
| `output_path` | 默认 `weights/<checkpoint stem>_axelera_model` |
| `verify` | 写出 `<stem>.parity.json`，把量化模型与 FP32 对比，有设备时也与设备结果对比 |
| `compiler_config` | 仅限 Python：一个由 Axelera `CompilerConfig` 字段组成的 dict。未知字段和由 LibreYOLO 管理的字段会抛错 |

校准使用模型自己的预处理。对于微调过的模型，请用它所在领域的图像校准。编译需要几分钟到几十分钟，以及数 GB 内存。

```text
weights/LibreYOLO9t_axelera_model/
    LibreYOLO9t.axm               compiled model
    compiler_config_final.toml    compiler settings used for the build
    metadata.yaml                 LibreYOLO metadata
    LibreYOLO9t.parity.json       only with verify=True
```

目录按检查点文件命名。预训练的 YOLO-NAS 权重在首次使用时会被转换，所以它们的导出产物落在
`weights/LibreYOLONASs-converted_axelera_model`。重新导出到同一路径时，只有在新构建成功之后才会替换目录，
并且不会留下 ONNX 文件或编译器构建文件。

## 运行导出的产物

<code-tabs name="run" />

网络在 Metis AIPU 上运行；预处理、检测框解码和 NMS 在主机上运行，所以检测框坐标和分数从不经过 8 位张量。
结果与 PyTorch 模型给出的 `Results` 对象相同。输入尺寸由导出固定。`device="auto"`、`"axelera"`、
`"metis"`、`"aipu"` 和 `"cpu"` 都在 AIPU 上运行；CUDA 设备会被忽略并给出警告。把这个目录复制到任何一台装有
Metis 设备和 `axelera-rt` 的 Linux 机器上即可。

## 支持的家族

下面每个尺寸都能编译、能在 Metis 设备上运行，也能导出。「1.0 以内」表示在配对集上 INT8 结果与 FP32 相差不超过
1.0 个点；测得下降更大的尺寸照样能导出，导出时会把这个下降以警告形式记录一次。精度从不阻止导出。每个家族和任务在导出矩阵里
都有自己的 `validated` 或 `available` 单元格（`libreyolo formats --family yolonas` 会打印出来），
未列出的组合会在编译前抛错。

| 家族 | 任务 | 尺寸 | 测得的下降 |
|---|---|---|---|
| YOLO-NAS | detect | s、m、l | 1.0 以内 |
| YOLO-NAS | pose | n、s、m、l | n 和 s 更大 |
| YOLO-NAS-R | obb | s、m、l | 1.0 以内；需要校准数据 |
| YOLO9 | detect | t、s、m、c | 1.0 以内 |
| YOLOX | detect | n、t、s、m、l、x | n 更大 |
| YOLOv2 | detect | t、b | 1.0 以内 |
| YOLOv3 | detect | t、b、spp | 1.0 以内 |
| YOLOv4 | detect | t、b | b 更大 |
| YOLOv7 | detect | b | 1.0 以内 |
| PP-YOLOE | detect | s、m、l、x | s、m 和 x 更大 |
| PicoDet | detect | s、m、l | 更大 |
| RTMDet | detect | t、s、m、l | 更大 |
| RTMDet-Ins | segment | t、s、m、l | 更大 |
| RetinaNet | detect | r50 | 更大 |
| SSD | detect | 300 | 1.0 以内 |
| EfficientDet | detect | d0、d1、d2 | 更大 |
| ResNet | classify | 18、34、50、101 | 1.0 以内 |
| MobileNetV4 | classify | s、m、l | s 和 l 更大 |
| EfficientNetV2 | classify | b0 | 更大 |
| VGG | classify | 16、19、16bn、19bn | 19bn 更大 |
| AlexNet | classify | b | 1.0 以内 |
| FCN | semantic | r50、r101 | 1.0 以内 |
| DeepLabV3 | semantic | mv3 | 更大 |
| MiDaS | depth | s | 更大 |
| QuickSRNet | restore | m2 | 更大 |
| Real-ESRGAN | restore | x4t | 更大 |

部分实测差值，在 1000 张 COCO val2017 图像上的 mAP50-95，设备上的 INT8 对比同一批图像上的 FP32：YOLO9-s 从 0.4502
到 0.4450（-0.52），YOLO9-t -0.45，YOLO-NAS-s -0.30，YOLOX-s -0.48，YOLOX-n -1.81（300 张图像），PicoDet-m
-2.86，RTMDet-s -6.75。ResNet-50 在 ImageNetV2 top-1 上损失 0.7 个点。包含每个数字的逐尺寸表格在库的
`docs/axelera.md` 里。

因结构原因被阻止的：EfficientNetV2 b1 到 b3、EfficientDet-d3、RTMDet-x、RTMDet-Ins-x、DeepLabV3 r50 和 r101、
MiDaS-l、RetinaNet-r50v2、FCOS、HRNet、CenterNet、Faster R-CNN、ConvNeXt 和 ConvNeXt V2、YOLOv1、EC、
TinyFormer、NAFNet、TEED 和 ZipDepth。RF-DETR 和其他 transformer 家族不在范围内。YOLO9-E2E、YOLO9-P2、
EfficientDet-d4、Real-ESRGAN x4 和 x2、PIDNet、DEKR 和 UNet 尚未测量。

## 许可证

LibreYOLO 的代码采用 MIT 许可。`axelera-devkit` 和 `axelera-rt` 是 Axelera 的专有软件，需要你自己安装。
编译后的模型沿用其来源权重的许可证：YOLO-NAS 预训练权重仅限非商用且不可再分发，所以 LibreYOLO 不托管任何
YOLO-NAS `.axm` 文件，每个用户自己编译。`axelera` 单元格是如何判定的，见[导出矩阵](/docs/reference/export-matrix)。
