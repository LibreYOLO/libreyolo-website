---
title: 推理性能
seo_title: 在 LibreYOLO 里跑得更快
description: 预测阶段的 CUDA 图、半精度、批处理、切片推理和测试时增强，附真实的默认值，以及每一项分别支持哪些家族。
lead: >-
  有五个预测阶段的控制项会改变吞吐或精度：CUDA
  图重放、数值精度、批处理、切片和测试时增强（TTA）。每一项都只作用于一批特定的家族，其中两项花掉的是精度或延迟，而不是省下它们。
keywords:
  - cuda graph pytorch 推理
  - yolo 批量推理 python
  - fp16 半精度推理
  - 小目标 切片推理
  - 大图 切片推理
  - 测试时增强 tta 目标检测
  - capture_graph
  - yolo 预测整个文件夹
last_verified: 1.5.0
verification: >-
  参数默认值读自 libreyolo/models/base/inference.py 里的 InferenceRunner.__call__。CUDA 图
  API 读自 libreyolo/models/base/model.py 里的
  BaseModel.capture_graph、graph_info、release_graphs 和
  cuda_graph_scope；家族层面的主动开启读自 SUPPORTS_CUDA_GRAPH 类变量。半精度行为读自
  libreyolo/utils/predict_args.py 里的
  NOOP_PREDICT_KWARGS、libreyolo/cli/commands/predict.py 里的 CLI 警告，以及
  libreyolo/quant/api.py 里的 CAST_RECIPES 和 SUPPORTED_FAMILIES。批处理条件读自
  InferenceRunner._process_in_batches 和 _predict_batch。切片读自 _predict_tiled 和
  _merge_tile_detections。测试时增强读自 BaseModel._predict_augment 和 _merge_tta，其中
  TTA_ENABLED、TTA_SCALES 和 TTA_FIXED_SIZE 是通读 libreyolo/models/ 得到的。
snippets:
  batch:
    - label: 对一个文件夹做批量推理
      language: python
      code: |
        from pathlib import Path
        from PIL import Image

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        folder = Path("batch_demo")
        folder.mkdir(exist_ok=True)
        image = Image.open(SAMPLE_IMAGE)
        for index in range(8):
            image.save(folder / f"frame_{index}.jpg")

        model = LibreYOLO("LibreYOLO9s.pt")

        # 在支持的家族上，每 4 张一组做一次堆叠前向
        results = model(str(folder), batch=4)
        print(len(results), "results")
    - label: 流式返回，列表不会一次性生成
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        for result in model("batch_demo", batch=4, stream=True):
            print(len(result.boxes))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt source=batch_demo batch=4
  graphs:
    - label: 先捕获，再重放（需要 CUDA）
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")

        # 预热和捕获只付一次，而且不落在第一次请求上
        model.capture_graph()

        result = model(SAMPLE_IMAGE, cuda_graph=True)
        print(len(result.boxes))
        print(model.graph_info())
    - label: 形状重复出现后才捕获（需要 CUDA）
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")

        # "auto" 会等到某个形状被看到第二次，所以一次性的活
        # 永远不用为捕获付出代价
        for _ in range(3):
            model(SAMPLE_IMAGE, cuda_graph="auto")

        print(model.graph_info())
        model.release_graphs()
  precision:
    - label: 安装导出额外依赖
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: 按默认精度导出，再加载回来
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        path = model.export(format="onnx")

        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)
        print(len(result.boxes))
    - label: FP16 导出（在有 CUDA 的机器上构建并运行）
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")
        path = model.export(format="onnx", half=True)

        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)
        print(len(result.boxes))
    - label: 在 PyTorch 里用转换配方跑 FP16（需要 CUDA）
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")

        # 转换类配方不读取校准数据
        model.quantize(recipe="fp16", calib=None)

        result = model(SAMPLE_IMAGE)
        print(len(result.boxes))
  tiling:
    - label: 对大图做切片推理
      language: python
      code: |
        from PIL import Image

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 只有图像大于输入尺寸时，切片才会生效
        large = Image.open(SAMPLE_IMAGE).resize((2048, 1536))
        large.save("large.jpg")

        model = LibreYOLO("LibreYOLO9s.pt")

        result = model("large.jpg", tiling=True, overlap_ratio=0.2)
        print(result.num_tiles, "tiles", len(result.boxes), "detections")
  tta:
    - label: 测试时增强
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        plain = model(SAMPLE_IMAGE)
        flipped = model(SAMPLE_IMAGE, augment=True)

        print(len(plain.boxes), "->", len(flipped.boxes))
source_hash: 3914665d0e7f892c
---

## 各个控制项和它们的默认值

这些全都是 `predict` 的参数，而且默认值全都是关闭。

| 参数 | 默认值 | 作用 |
|---|---|---|
| `batch` | `1` | 目录源和列表源每次前向传播处理的图像数 |
| `cuda_graph` | `False` | 从捕获好的 CUDA 图里重放前向 |
| `tiling` | `False` | 把大图切成互相重叠的切片 |
| `overlap_ratio` | `0.2` | `tiling` 打开时的切片重叠比例 |
| `augment` | `False` | 跑翻转视图并把它们合并 |
| `half` | | 接受、告警，然后忽略 |
| `device` | `None` | 预测前先把模型搬过去 |

`imgsz` 同样影响开销，因为它决定了模型运行时的分辨率，但它首先是一个精度参数，
归属于模型那边，而不是这里。

## 批处理

<code-tabs name="batch" />

`batch` 适用于目录源和列表源。`batch=1` 时，每张图像各跑一次前向传播。大于 `1`
时，每一组先做预处理，堆叠成单个张量，一次跑完，再切回去，这样每个家族现有的单图
后处理拿到的都是它预期的东西。

只有下面这些条件全部成立时，才会走堆叠这条路径：

- `batch` 大于 `1`，
- `tiling` 是关闭的，
- 测试时增强没有生效，
- 家族设置了 `SUPPORTS_BATCHED_PREDICT`，
- 底层网络不处于训练模式。

最后一条不是细枝末节。处于训练模式的网络会用跨图像的批统计量去归一化堆叠后的这一
组，让同一组里的图像互相改变对方的预测，所以这类运行保持逐张顺序执行。

`SUPPORTS_BATCHED_PREDICT` 默认为 true。下面这些家族选择退出，不管 `batch` 是多少，
每次前向都只跑一张图像：Depth Anything V2、Depth Anything 3、EoMT、Faster R-CNN、
FCOS、HRNet、L2CS-Net、LibreMODUS、MiDaS、MoGe-2、PP-OCRv5、Real-ESRGAN、RetinaNet、
SAM 3D Body、SwinIR、YOLOv1、ZipDepth、所有开放词汇检测器，以及所有视觉语言模型。

还有一层兜底。如果预处理没有在整组上返回形状、dtype 和设备都一致的统一
`(1, C, H, W)` 张量，这一组就逐张顺序运行而不是堆叠，所以正确性从不依赖于图像恰好
尺寸相同。

在大文件夹上把 `batch` 和 `stream=True` 一起用，就能拿到成批的前向，又不必把每一个
结果都留在内存里。

## CUDA 图

<code-tabs name="graphs" />

CUDA 图把一次前向传播录制下来，然后作为单次启动重放。小检测器在批大小为 1 时有很大
一部分时间花在启动 kernel 上，所以把这些启动折叠起来是吞吐上的收益，而且重放的输出
和 eager 执行逐位一致。

`cuda_graph` 接受三种取值。`False` 是默认值，什么也不做。`True` 会在每种输入形状第
一次用到时捕获。`"auto"` 会等到某个形状重复出现之后才捕获，所以一次性的、形状多变的
活永远不用付捕获的代价。

`capture_graph(imgsz=None, batch=1, dtype=None)` 把这份开销从第一次请求上挪走。一张
图只对它捕获时的那个确切形状有效，所以这里的 `batch` 必须和之后调用 `predict` 的方
式对上。

`graph_info()` 会报告已捕获的图、重放次数，以及本次运行回退到 eager 的任何原因。
`release_graphs()` 会释放它们和它们的静态缓冲区。

捕获需要 CUDA，也需要家族通过 `SUPPORTS_CUDA_GRAPH` 主动开启，因为它要求前向里没有
任何主机可见的工作，而这一点是按家族逐个核实的。在没有开启的家族上要求捕获会抛出
`NotImplementedError`，而不是悄悄按 eager 运行。

图记录的是内存地址，不是数值，所以任何会搬动参数的操作都会让它作废。通过
`predict(device=...)` 改设备、量化和反量化，都会让已捕获的图失效。

完整的逐家族支持矩阵、接缝切分和数值契约在 [CUDA 图](/docs/reference/cuda-graphs)
上。

## 数值精度

<code-tabs name="precision" />

预测时的 `half=True` 什么也不做。它是为了命令行兼容才被接受的，会抛出一条说明它是空
操作的警告，并在到达任何家族之前被丢弃。CLI 的 `--half` flag 对 `.pt` 模型打印同样
的警告。

真正降低数值精度的路子有两条。

对导出的产物来说，精度在导出时用 `export(format=..., half=True)` 选定，得到的文件照
原样通过 `LibreYOLO()` 加载回来。

对 PyTorch 执行来说，`model.quantize(recipe="fp16")` 会把模型转成 float16，并装上钩
子，让模型的输入和输出保持 float32。`"bf16"` 用 bfloat16 做同样的事。这两种转换都不
读校准数据，所以对它们来说 `calib` 会被忽略。量化目前覆盖四个家族：YOLOv9、RF-DETR、
BiRefNet 和 FeyNobg。在 CPU 设备上做转换会记录一条说它会很慢的警告，所以这些配方是
给 GPU 用的。

两条路都会改变数值。两者都不保证检测结果原样不变，所以部署前先验证。

## 切片推理

<code-tabs name="tiling" />

切片会把大图裁成互相重叠的方形切片，在每一片上预测，然后把结果合并。它是高分辨率
图像里小目标的那个选项，在那种场景下，整图缩放会把目标缩小到模型分辨不出来的程度。

切片尺寸就是模型的输入尺寸，给了 `imgsz` 就用 `imgsz`，而且必须是正方形。
`overlap_ratio` 默认为 `0.2`。重叠的切片之间用按类别的非极大值抑制在 `iou` 阈值上做
调和，合并后的列表再截断到 `max_det`。这意味着即便是自己不跑 NMS 的家族，`iou` 也会
影响切片预测的结果。

图像本身就装得下时，切片是被跳过的，而不只是开销小：如果两个维度都不超过输入尺寸，
就改跑一次普通的前向。分类、语义分割和 `embed` 任务同样会跳过，它们回退到单次前向，
因为切片在那里没有意义。

对于载荷无法再拼回去的任务，它会抛错：实例分割掩码、旋转框、点、深度、边缘和法线。
它不能和 `augment` 一起用。

结果上带有 `result.tiled` 和 `result.num_tiles`。开了 `save=True` 时，切片运行会在
`runs/tiled_detections` 下写一个目录，里面放着每一个切片、标注后的图像、一张网格
可视化图，以及一个记录了切片尺寸、重叠和阈值的 `metadata.json`，`result.tiles_path`
和 `result.grid_path` 指向它们。

## 测试时增强

<code-tabs name="tta" />

`augment=True` 会把图像跑不止一次，并用按类别的非极大值抑制在 `iou` 阈值上合并检测
结果。和切片一样，这让 `iou` 对本来会忽略它的家族变得关键。

实际上它做的是水平翻转。尺度列表 `TTA_SCALES` 默认只有 `1.0` 一个尺度，而且随包发布
的家族没有一个覆盖它，所以每个家族都跑两遍：原图和它的镜像。标记了 `TTA_FIXED_SIZE`
的家族会缩放到固定的正方形，这让多尺度对它们来说无论如何都是空操作。

语义分割和全景分割走的是另一种合并。它们的翻转视图会被翻转回来，两个 softmax 分布在
取 argmax 之前先做平均，而不是按检测框合并。

测试时增强并不是每个任务都可用。对旋转框、姿态、点、深度、法线、边缘、复原、OCR 和
嵌入向量模型，它会抛错，而且不能和切片一起用。

下面这些家族直接把它禁用了，所以 `augment=True` 只跑一次普通的前向：BiRefNet、
CenterNet、CLIP、DexiNed、FOMO、HRNet、L2CS-Net、LibreMODUS、NAFNet、PP-OCRv5、
Real-ESRGAN、RetinaNet、SAM 3D Body、SigLIP2、SwinIR、TEED、所有 SAM 变体、所有开放
词汇检测器，以及所有视觉语言模型。

## 测量

本页没有任何一个延迟数字，因为一个不带硬件、运行时、精度和批大小的毫秒数不算事实。
跨硬件和运行时的实测数据发布在 [visionanalysis.org](https://www.visionanalysis.org)，
而 `libreyolo profile` 会在你面前这台机器上测量某个具体的模型。
