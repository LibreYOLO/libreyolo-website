---
title: Python API
seo_title: LibreYOLO Python API 参考
description: LibreYOLO 在包层面导出的名字：五个工厂函数、家族类、Results 载荷、后端、验证器、跟踪器和数据辅助函数。
lead: >-
  LibreYOLO 的公开 Python 接口就是 libreyolo/__init__.py 里的 __all__ 列表。这个页面上的所有名字都能用
  from libreyolo import <name> 导入；不在那个列表里的都是内部实现。
keywords:
  - libreyolo python api
  - libreyolo 导入
  - LibreYOLO 工厂函数
  - LibreSAM
  - LibreVLM
  - LibreOpenVocab
  - LibreEnsemble
  - libreyolo __all__
last_verified: 1.5.0
verification: >-
  名字与签名读取自 v1.5.0 的
  libreyolo/__init__.py、libreyolo/models/__init__.py、libreyolo/models/base/model.py、libreyolo/models/base/inference.py、libreyolo/models/sam/model.py、libreyolo/models/vlm/__init__.py、libreyolo/models/openvocab/__init__.py
  和 libreyolo/ensemble/model.py。
snippets:
  usage:
    - label: 用一个工厂函数加载任何模型
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        # 单张图像的源返回一个 Results；列表或目录
        # 返回由它们组成的列表
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
        print(result.names)
    - label: 直接导入家族类
      language: python
      code: |
        from libreyolo import LibreYOLO9, SAMPLE_IMAGE

        model = LibreYOLO9("LibreYOLO9t.pt", size="t")
        result = model(SAMPLE_IMAGE)

        print(len(result))
  factories:
    - label: 五个入口
      language: python
      code: >
        from libreyolo import LibreYOLO, LibreEnsemble


        # 面向无提示家族的权重嗅探工厂函数

        detector = LibreYOLO("LibreYOLO9t.pt")


        # 两个或更多检测器，共用一个预测接口

        ens = LibreEnsemble(["LibreYOLO9t.pt", "LibreYOLO9s.pt"])


        # 另外三个工厂函数需要安装额外依赖：

        #   pip install 'libreyolo[sam]'        -> from libreyolo import
        LibreSAM

        #   pip install 'libreyolo[vlm]'        -> from libreyolo import
        LibreVLM

        #   pip install 'libreyolo[openvocab]'  -> from libreyolo import
        LibreOpenVocab

        print(type(detector).__name__, ens.fusion)
source_hash: 66e34e78b2e0fb2d
---

## 入口

有五个可调用对象负责加载模型。它们按调用约定划分，而不是按架构划分。

| 工厂函数 | 加载什么 | 调用时的提示 | 需要的额外依赖（extra） |
|---|---|---|---|
| `LibreYOLO` | 无提示家族，靠嗅探检查点（checkpoint）或文件后缀 | | |
| `LibreSAM` | 可提示的分割器，按尺寸别名 | 点、检测框或概念文本 | `sam` |
| `LibreVLM` | 生成式视觉语言检测器，按别名 | 类别词表或自由形式的提示 | `vlm` |
| `LibreOpenVocab` | 以文本为条件的检测器，按别名 | 类别词表 | `openvocab` |
| `LibreEnsemble` | 两个或更多检测器，融合成一个接口 | | |

<code-tabs name="factories" />

`LibreYOLO` 是唯一读取文件的那个。另外三个接受一个字符串别名，并把它解析成一个
Hugging Face 仓库，所以那个参数是模型名而不是路径。

```python
LibreYOLO(
    model_path: str,
    size: str | None = None,
    reg_max: int = 16,
    nb_classes: int | None = None,
    device: str = "auto",
    task: str | None = None,
    compute_units: str = "all",
)
```

`model_path` 接受 `.pt` 检查点、ONNX 的 `.onnx` 文件、ExecuTorch 的 `.pte`、
MNN 的 `.mnn`、TensorRT 的 `.engine`、OpenVINO、Paddle 或 ncnn 目录，也接受
Triton 的 HTTP 或 HTTPS 模型 URL。省略时，`size` 和 `nb_classes` 从检查点里读取。
`compute_units` 只在加载 CoreML `.mlpackage` 时才会读取，取值是 `all`、
`cpu_only`、`cpu_and_gpu`、`cpu_and_ne` 之一。`task` 接受 `libreyolo.tasks.TASKS`
里的任意规范任务名。

<code-tabs name="usage" />

## 家族类

工厂函数能返回的每个家族也都按名字导出，所以提前知道检查点是哪个时，可以直接构造
对应的类。构造函数遵循 `BaseModel.__init__`：

```python
Family(model_path, size, nb_classes=80, device="auto", task=None, **kwargs)
```

在家族类上 `size` 没有默认值，这就是它和工厂函数的区别。YOLO9 及其变体在 `size`
之后插入 `reg_max: int = 16`。

检测与多任务家族：`LibreYOLO9`、`LibreYOLO9E2E`、
`LibreYOLO9P2`、`LibreYOLONAS`、`LibreYOLOX`、`LibreYOLO7`、`LibreYOLO4`、
`LibreYOLO3`、`LibreYOLO2`、`LibreYOLO1`、`LibreRTDETR`、`LibreRTDETRv2`、
`LibreRTDETRv4`、`LibreRFDETR`、`LibreDFINE`、`LibreDOMEDETR`、`LibreDEIM`、
`LibreDEIMv2`、`LibreDETR`、`LibreDeformableDETR`、`LibreDINODETR`、
`LibreLWDETR`、`LibreMaskRCNN`、`LibreFCOS`、`LibreFasterRCNN`、
`LibreRetinaNet`、`LibreSSD`、`LibreCenterNet`、`LibreEfficientDet`、
`LibreEC`、`LibrePICODET`、`LibreRTMDet`、`LibreFOMO`。

稠密预测家族：`LibreMiDaS`、`LibreDepthAnythingV2`、
`LibreDepthAnything3`、`LibreZipDepth`、`LibreMoGe2`、`LibreTEED`、
`LibreDexiNed`、`LibreNAFNet`、`LibreRealESRGAN`、`LibreSwinIR`、
`LibreBiRefNet`、`LibreFeyNobg`、`LibreFCN`、`LibreEoMT`、`LibreDeepLabv3`、
`LibrePIDNet`、`LibreSegformer`、`LibreLingBotVision`。

分类与嵌入向量家族：`LibreViT`、`LibreMobileNetV4`、
`LibreConvNeXt`、`LibreDeiT`、`LibreSwin`、`LibreEfficientNetV2`、`LibreVGG`、
`LibreResNet`、`LibreAlexNet`、`LibreCLIP`、`LibreSigLIP2`、`LibreDINOv2`。

其他任务：`LibreHRNet`（pose）、`LibreL2CS`（gaze）、`LibrePPOCR`（ocr）、
`LibreFaceEmbedder`（embed）。

兄弟层级也导出各自的家族类：`LibreSAM1`、`LibreSAM2`、
`LibreSAM3`、`LibreEdgeTAM`、`LibreMobileSAM`、`LibrePicoSAM3`；
`LibreGroundingDINO`、`LibreOWLv2`、`LibreOMDetTurbo`；`LibreLFM2VL`、
`LibreQwen3VL`、`LibreSmolVLM2`、`LibreInternVL3`、`LibreFlorence2`、
`LibreKosmos2`、`LibreLocateAnything`、`LibreMODUS`（也写作
`LibreModus`）。

## 预测接口

调用模型就会跑推理。`predict` 是 `__call__` 的别名，两者可以互换。

```python
model(
    source=None,
    *,
    conf=0.25,
    iou=0.45,
    imgsz=None,
    device=None,
    classes=None,
    max_det=300,
    augment=False,
    save=False,
    batch=1,
    stream=False,
    stream_buffer=False,
    vid_stride=1,
    show=False,
    output_path=None,
    color_format="auto",
    tiling=False,
    overlap_ratio=0.2,
    output_file_format=None,
    cuda_graph=False,
    **kwargs,
)
```

单张图像的源返回一个 `Results`。列表、元组或目录返回由它们组成的列表，而
`stream=True` 返回一个生成器。模型对象上的其他方法记录在[模型 API
页面](/docs/reference/model-api)。

## Results 载荷

`Results` 和它的十八个载荷类都在包层面导出：
`Results`、`Boxes`、`Masks`、`Keypoints`、`Points`、`Probs`、`OBB`、`Gaze`、
`SemanticMask`、`PanopticSegmentation`、`DepthMap`、`EdgeMap`、`NormalMap`、
`RestoredImage`、`Matte`、`Meshes`、`OCRRegions`、`Embeddings`、`Identities`。
每一个都在 [Results 类型](/docs/reference/results-types)里有说明。

## 后端

导出的产物通过 `LibreYOLO()` 按文件后缀加载，所以后端类很少需要手工构造。把它们
导出来，是为了应对必须显式选择后端的场合：`OnnxBackend`、`OpenVINOBackend`、
`PaddleBackend`、`TensorRTBackend`、`TritonBackend`、`NcnnBackend`、
`CoreMLBackend`，外加 `create_triton_config`。`BaseExporter` 是 `model.export()`
背后的导出器注册表（registry）。

## 验证器

`model.val()` 按任务分发到对应的验证器，所以这些类导出出来供直接使用和继承：
`DetectionValidator`、`SegmentationValidator`、`PoseValidator`、`SemanticValidator`、
`PanopticValidator`、`DepthValidator`、`NormalValidator`、`EdgeValidator`，以及
共用的 `ValidationConfig`。

## 跟踪

`model.track()` 按名字选择跟踪器。跟踪器类和它们的配置 dataclass 也都导出了：
`ByteTracker` 配 `TrackConfig`、`BoTSortTracker` 配 `BoTSortConfig`、
`OCSortTracker` 配 `OCSortConfig`。

## 数据辅助函数

`DATASETS_DIR` 是解析后的数据集根目录，`load_data_config` 读取数据集 YAML，
`check_dataset` 验证一份数据集。[数据集格式](/docs/reference/dataset-formats)里
点名的那些任务专用加载器住在 `libreyolo.data`，而不在包层面。

## 底库与蒸馏

`Gallery` 和 `FaceGallery` 为 `embed` 任务保存注册进来的身份向量，构成底库
（gallery），并产出 `Identities` 载荷。`Distiller` 和 `get_distill_config` 驱动
师生训练。

## 资源

`SAMPLE_IMAGE` 是随包附带的一张图像的绝对路径，所以这份文档里的每个代码片段都不用
先下载图片就能跑。

## 延迟导入与改名的类

大多数兄弟层级的名字、后端、验证器和数据辅助函数都通过模块级的 `__getattr__`
解析，所以 `import libreyolo` 不会导入它们的依赖。缺少所需的额外依赖时，导入仍然
会带着清晰的报错信息失败。

有两个类改了名，旧写法仍然可以解析，但会带一条 `DeprecationWarning`：
`LibreYOLORTDETR` 现在叫 `LibreRTDETR`，`LibreYOLORFDETR` 现在叫 `LibreRFDETR`。
