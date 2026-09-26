---
title: ncnn
seo_title: 从 LibreYOLO 导出到 ncnn
description: >-
  把 LibreYOLO 模型通过 PNNX 导出为 ncnn：param 与 bin 文件对、固定的导出画布、YOLOX 的 Focus
  改写，以及哪些家族能转换。
lead: >-
  ncnn 是腾讯面向移动端的 CPU 推理库。LibreYOLO 通过 PNNX 转换，写出一份 model.ncnn.param 图文件，旁边放上
  model.ncnn.bin 权重文件，以及一个记录家族、任务和类别名的 metadata.yaml。
keywords:
  - yolo 导出 ncnn
  - pnnx 转换
  - model.ncnn.param
  - 移动端 cpu 推理
  - ncnn 推理 python
  - focus pixel_unshuffle
last_verified: 1.5.0
meta:
  - label: 参数
    value: export(format="ncnn")
    mono: true
  - label: 输出
    value: 一个目录，里面有 model.ncnn.param、model.ncnn.bin 和 metadata.yaml
  - label: 额外依赖
    value: 'pip install "libreyolo[ncnn]"'
    mono: true
  - label: 重新加载方式
    value: LibreYOLO("weights/LibreYOLO9t_ncnn")
    mono: true
  - label: 形状
    value: 固定。无论标志怎么设置，元数据都记录 dynamic=False。
  - label: 精度
    value: 仅 FP32。half=True 和 int8=True 会被拒绝。
verification: >-
  依据 dev 分支上的
  libreyolo/export/ncnn.py、libreyolo/export/exporter.py、libreyolo/export/support.py、libreyolo/backends/ncnn.py
  和 pyproject.toml 校对。
snippets:
  install:
    - label: 安装
      language: bash
      code: |
        # pnnx 负责转换，ncnn 负责运行转换结果
        pip install "libreyolo[ncnn]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # 写出目录 weights/LibreYOLO9t_ncnn
        path = model.export(format="ncnn", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format ncnn --imgsz 640
    - label: 参数
      language: python
      code: |
        model.export(
            format="ncnn",
            imgsz=640,        # int，或 (height, width)
            batch=1,
            simplify=True,    # 只作用于 ONNX 回退路径
            opset=None,       # 自动；只作用于 ONNX 回退路径
            output_path=None, # None 表示写入 weights/<stem>_ncnn
        )

        # half=True 和 int8=True 会在校验阶段被拒绝
  run:
    - label: 通过 LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_ncnn")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: 裸 ncnn
      language: python
      code: |
        import ncnn
        import numpy as np
        import yaml

        directory = "weights/LibreYOLO9t_ncnn"
        net = ncnn.Net()
        net.load_param(f"{directory}/model.ncnn.param")
        net.load_model(f"{directory}/model.ncnn.bin")

        # ncnn 接受单张 CHW 图像，而不是一个批次
        mat_in = ncnn.Mat(np.zeros((3, 640, 640), dtype=np.float32))
        extractor = net.create_extractor()
        extractor.input("in0", mat_in)
        ret, mat_out = extractor.extract("out0")
        print(ret, np.array(mat_out).shape)

        meta = yaml.safe_load(open(f"{directory}/metadata.yaml"))
        print(meta["model_family"], meta["task"], meta["names"])

        # 这条路径上的预处理和后处理由你自己负责
  support:
    - label: 导出前先检查某个家族和任务
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 9a849a16a3b32334
---

## 安装

<code-tabs name="install" />

这个额外依赖会把工具链的两半都装上：`pnnx` 完成转换，`ncnn` 执行转换结果。主路径上两者都不经过 ONNX。

## 导出

<code-tabs name="export" />

产物是一个目录。`weights/LibreYOLO9t_ncnn` 里放着 `model.ncnn.param`、`model.ncnn.bin` 和 `metadata.yaml`；这三个文件是同一份产物，要一起移动。

转换会先尝试直接从 PyTorch 走 PNNX。如果失败，就把一张静态 ONNX 图导出到临时目录，再对它调用 `pnnx` 命令行工具，只有两条路径都失败时导出才会报错，并把两个错误一起报出来。因此 `opset` 和 `simplify` 只影响回退路径。

YOLOX 需要一处改写才能转换成功。它的 Focus 层用的是带步长的切片，PNNX 无法把它下降（lower）为底层算子，所以导出会把它换成 `pixel_unshuffle`，并对后一层卷积的输入通道做置换，以补偿通道顺序的差异。输出在数值上完全相同，导出结束后原始权重会被恢复。

## 运行产物

<code-tabs name="run" />

`LibreYOLO()` 能识别任何含有 `model.ncnn.param` 和 `model.ncnn.bin` 的目录，读取 `metadata.yaml`，并返回与检查点（checkpoint）相同的 `Results` 对象。

第二段代码是裸运行时路径，其中有两处细节和这里其他所有格式都不一样。ncnn 处理的是单张 CHW 图像而不是一个批次，所以没有前面那根批次轴。Blob 名字来自 `.param` 文件；PNNX 按惯例写的是 `in0` 和 `out0`，而后端会去解析这个文件，而不是直接假定名字就是这两个。这条路径上的预处理、解码、NMS 和坐标缩放都由你自己负责。

## 约束

FP32，固定画布。`half=True` 和 `int8=True` 在校验阶段都会被拒绝，而且不管标志怎么设置，导出的元数据都记录 `dynamic=False`，这样就不会有后端去假定一根图里根本没有的轴。

所有 DETR 风格的家族都会在预检阶段被拒绝：`detr`、`deformable_detr`、`dinodetr`、`dfine`、`lwdetr`、`deim`、`deimv2`、`rtdetr`、`rtdetrv2`、`rtdetrv4`、`rfdetr` 和 `ec`。它们的提示信息都一样，说模型需要 ncnn 里没有的解码器或采样算子，并转而指向 ONNX、OpenVINO、TorchScript 或 TensorRT。

卷积一侧能转换的范围很广：YOLO9 和 YOLO9-E2E、YOLOX、PicoDet、YOLO-NAS 的检测和姿态、更早的 YOLO1、YOLO3、YOLO4 和 YOLO7 检测器、四个 CNN 分类家族、PIDNet 语义分割、固定 96×96 的 FOMO 点检测、ZipDepth、NAFNet 和 Real-ESRGAN。

被屏蔽的条目会写明具体的失败原因。transformer 图通常会留下不受支持的 `pnnx.Expression` 节点，产生一个没有可运行输入 blob 的网络，DINOv2、CLIP、SigLIP2 和 SegFormer 就是卡在这里。BiRefNet 需要 torchvision 的可变形卷积，PNNX 同样无法下降。YOLO2 转换出来的图在 Windows 上会在输出提取时触发原生的整数除零，直接终止 ncnn 运行时。

完整的家族与任务对照表见[导出矩阵](/docs/reference/export-matrix)。若只想查某一个组合：

<code-tabs name="support" />
