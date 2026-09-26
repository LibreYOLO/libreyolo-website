---
title: 实例分割
seo_title: LibreYOLO 中的实例分割
description: 在 LibreYOLO 中分割单个物体：服务这项任务的模型家族、多边形标注格式，以及 predict、train、validate 和 export 调用。
lead: 实例分割定位每一个物体实例，并为每个实例返回一张逐像素的掩码，同时给出检测器会返回的检测框、类别和分数。任务键是 segment。
keywords:
  - 实例分割 python
  - 物体掩码预测
  - 分割模型训练
  - 多边形标注
  - MIT 分割库
  - 掩码 mAP
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 文件名里的 -seg 后缀会选中掩码 head，
        # 因此不需要传 task 参数
        model = LibreYOLO("LibreDFINEn-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)   # (N, H, W)，每个检测一张掩码
        print(result.boxes.xyxy.shape)   # (N, 4)，同样的 N 行
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDFINEn-seg.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 掩码轮廓
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDFINEn-seg.pt")
        result = model(SAMPLE_IMAGE)

        # .xy 是一组以像素为单位的 (P, 2) 轮廓，.xyn 是同样的轮廓归一化后的结果
        for name, contour in zip(result.boxes.cls, result.masks.xy):
            print(result.names[int(name)], contour.shape)
    - label: 换个家族，同样的调用
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRTMDets-seg.pt")
        result = model(SAMPLE_IMAGE)

        print(result.masks.data.shape)
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # 从已发布的分割权重继续训练，掩码 head 也包含在内

        # data 必须指向标注中带多边形的数据集

        model = LibreYOLO("LibreDFINEn-seg.pt")

        model.train(data="my-dataset.yaml", epochs=50, imgsz=640, batch=8,
        lr0=2e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDFINEn-seg.pt data=my-dataset.yaml \
          epochs=50 imgsz=640 batch=8 lr0=2e-4
    - label: 从检测权重出发
      language: bash
      code: |
        # 检测权重不带掩码 head，所以这是一次显式的迁移：
        # head 从未训练的状态开始，而指定 task=segment
        # 就是对它的授权
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          task=segment epochs=50 imgsz=640
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])       # 掩码
        print(metrics["metrics/mAP50-95(M)"])    # 掩码，显式写明
        print(metrics["metrics/mAP50-95(B)"])    # 检测框
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDFINEn-seg.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn-seg.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDFINEn-seg.pt format=onnx imgsz=640
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀路由，因此导出的产物加载起来和检查点一样，
        # 并且返回同一个 Results 对象
        model = LibreYOLO("LibreDFINEn-seg.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.masks.data.shape)
source_hash: 33e331eac0f9b0af
---

## 定义

实例分割就是检测加上形状。每个物体实例照样有一个检测框、一个类别和一个分数，
此外还有一张覆盖属于它的那些像素的二值掩码（mask）。掩码之间可以重叠，不属于
任何物体的像素则不做分配，这正是这项任务与[语义分割](/docs/tasks/semantic-segmentation)和[全景分割](/docs/tasks/panoptic-segmentation)的区别。

`segment` 是规范的任务键，检查点（checkpoint）文件名里的 `-seg` 后缀会选中它，
因此加载已发布的权重时不需要 `task=`。

`predict()` 会在填充 `result.boxes` 的同时填充 `result.masks`。`.data` 是原图
画布上一摞 `(N, H, W)` 的掩码，与检测框按行对齐，因此第 `i` 张掩码属于第 `i` 个
检测框。`.xy` 把每张掩码转换成它最大的外轮廓，形式是 `(P, 2)` 的像素数组，
`.xyn` 给出同一条轮廓归一化后的结果。

## 模型

有四个家族既能训练也能预测掩码：[RF-DETR](/docs/models/rf-detr)、
[EdgeCrafter](/docs/models/edgecrafter)、[D-FINE](/docs/models/d-fine) 和
[RTMDet](/docs/models/rtmdet)。RF-DETR 需要自己的 extra，
`pip install "libreyolo[rfdetr]"`；其余三个在基础包上就能跑。

[Mask R-CNN](/docs/models/mask-rcnn) 能预测、验证和导出掩码，但它的 `train()`
会抛出 `NotImplementedError`。

[EoMT](/docs/models/eomt) 能预测和验证掩码，同样不能训练，而且它的导出范围更窄：
`export()` 只接受语义分割任务，对 `segment` 和 `panoptic` 会抛出
`NotImplementedError`，因为这两者需要的 query-mask 运行时约定尚未定义。要拿 EoMT
做实例掩码，就在 Python 里用，而不要走导出的计算图。

另有一组模型是从提示而不是类别列表出发做分割：一次点击、一个框或者一段短语选中
物体，模型返回它的掩码。[SAM](/docs/models/sam)、[SAM 2](/docs/models/sam-2)、
[SAM 3](/docs/models/sam-3)、[MobileSAM](/docs/models/mobilesam)、
[EdgeTAM](/docs/models/edgetam) 和 [PicoSAM3](/docs/models/picosam3) 都是这样
工作的，[SenseNova-Vision](/docs/models/sensenova-vision) 也一样，它的分割属于
指代式（referring）：接受一段指名某一个物体的短语。它们通过各自的工厂和 extra
加载，每个模型页上都写着确切的调用方式。

## 预测

权重会在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

`conf` 和 `max_det` 塑造输出的方式和检测任务一样，掩码会随它们所属的检测框一起
被过滤。数据源、流式处理和结果处理见[预测](/docs/predict)。

## 数据集格式

目录布局就是检测任务的布局：每张图像一个 `.txt` 标注文件，把图像路径里的
`images` 换成 `labels` 并改掉扩展名就能找到它。

```text
dataset/
  data.yaml
  images/
    train/000001.jpg
    val/000101.jpg
  labels/
    train/000001.txt
    val/000101.txt
```

变的是每一行。一个分割实例是一个类别索引，后面跟着一个扁平的多边形：

```text
<class_id> <x1> <y1> ... <xN> <yN>
```

至少三个点，所以类别索引之后的坐标个数是偶数且不少于六个，而且多边形不能退化。
坐标是相对于原图宽高的 `[0, 1]` 浮点数。分割数据集里也接受五个字段的检测行，它
会被读成一个矩形分割实例，这样只有检测框的数据集不用转换就能直接加载。

YAML 就是检测任务的 YAML：

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: person
  1: bicycle
```

原生 COCO JSON 同样可用：加一个 `annotations` 映射，把划分名映射到 JSON 文件，
划分路径给出图像根目录。

## 训练

<code-tabs name="train" />

训练默认从已发布的 `-seg` 检查点继续。从检测权重出发也可以，但那是一次刻意的
迁移：那些权重不带掩码 head，所以 head 从未训练的状态开始，而传入 `task=segment`
就是对这次替换的授权。数据集、数据增强、多卡训练和日志记录器见[训练](/docs/train)。

## 验证

`val()` 返回一个由 `metrics/` 键构成的普通字典。检测框和掩码分开评分，都走 COCO
评估，掩码的数字是主要的那一组。

<code-tabs name="val" />

不带后缀的键保存掩码结果：`metrics/mAP50-95`、`metrics/mAP50`、`metrics/mAP75`，
然后是按物体面积划分的 `metrics/mAP_small`、`metrics/mAP_medium` 和
`metrics/mAP_large`，以及表示平均查全率的 `metrics/AR1`、`metrics/AR10`、
`metrics/AR100`、`metrics/AR_small`、`metrics/AR_medium`、`metrics/AR_large`。
`metrics/AR_max_det` 和 `metrics/max_det` 记录这次运行使用的检测数上限。

另有四组数字带显式后缀发布，`(M)` 表示掩码，`(B)` 表示检测框，这样比较就永远不
取决于某个家族把哪个数字定为主指标：`metrics/mAP50-95(M)` 和
`metrics/mAP50-95(B)`、`metrics/mAP50(M)` 和 `metrics/mAP50(B)`、
`metrics/precision(M)` 和 `metrics/precision(B)`、`metrics/recall(M)` 和
`metrics/recall(B)`。这项任务上没有不带后缀的 `metrics/precision` 或
`metrics/recall`。

查准率和查全率这两组键要仔细读。它们是为了向后兼容而保留的别名，而不是一个工作
点：`metrics/precision(M)` 保存的值与 `metrics/mAP50-95(M)` 相同，
`metrics/recall(M)` 保存的值与 100 个检测下的掩码 AR 相同，`(B)` 对检测框的行为
也一样。把其中一对画成图，等于把同一个数字报了两遍。

## 导出

<code-tabs name="export" />

导出的产物按文件后缀通过 `LibreYOLO()` 重新加载，所以一个 `.onnx` 或 `.engine`
文件的行为和检查点一样，并返回同样的 `Results`。在同一个家族上，分割的覆盖范围
比检测更窄。每个模型页上的矩阵由已验证的集合生成，并写明某个目标不可用的原因。
各种格式、它们的 extra 和限制见[导出与部署](/docs/export)。
