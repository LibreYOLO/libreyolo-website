---
title: 集成 API
seo_title: LibreEnsemble API 与融合操作
description: >-
  LibreEnsemble、ExternalDetector，以及 libreyolo.ops 里的三个融合操作：weighted boxes
  fusion、它的种子变体，以及按类别的 NMS 融合。
lead: >-
  LibreEnsemble 在同一张图像上运行多个检测器，并把它们的检测结果融合成一个
  Results。融合发生在每个成员各自的后处理之后，所以成员各自保留自己的输入尺寸、归一化和抑制方式。
keywords:
  - LibreEnsemble
  - 模型集成 目标检测
  - wbf 加权框融合
  - ExternalDetector
  - libreyolo.ops.fusion
  - min_votes 投票一致
last_verified: 1.5.0
verification: >-
  签名与默认值读取自 v1.5.0 的 libreyolo/ensemble/model.py 和
  libreyolo/ops/fusion.py。设计意图来自 docs/adr/0004-model-ensembling.md。
snippets:
  usage:
    - label: 两个成员，默认融合
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        ens = LibreEnsemble(["LibreYOLO9t.pt", "LibreYOLO9s.pt"])

        # 单张图像的源返回一个 Results，不是列表
        result = ens(SAMPLE_IMAGE, conf=0.25)

        print(result.boxes.xyxy)
        print(result.speed)
    - label: 共识与按成员设置的阈值
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        ens = LibreEnsemble(
            ["LibreYOLO9t.pt", "LibreYOLO9s.pt"],
            weights=[1.0, 2.0],
            fusion="wbf",
            fusion_iou=0.55,
            min_votes=2,
        )
        result = ens(SAMPLE_IMAGE, conf=[0.25, 0.4])
        print(len(result))
  ops:
    - label: 融合操作，不涉及模型
      language: python
      code: >
        import torch

        from libreyolo.ops import weighted_boxes_fusion


        boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0], [12.0, 11.0, 51.0,
        49.0]])

        scores = torch.tensor([0.9, 0.8])

        labels = torch.tensor([0, 0])

        model_ids = torch.tensor([0, 1])


        fused = weighted_boxes_fusion(
            boxes, scores, labels, model_ids, num_models=2, iou_thr=0.55
        )

        print(fused)
source_hash: 3834f628efb1193d
---

## LibreEnsemble

```python
LibreEnsemble(
    members,
    *,
    weights=None,
    fusion="wbf",
    fusion_iou=0.55,
    min_votes=1,
)
```

| 参数 | 默认值 | 含义 |
|---|---|---|
| `members` | | 两个或更多检测器 |
| `weights` | `None` | 每个成员的信任系数；省略时全为 `1.0` |
| `fusion` | `"wbf"` | `"wbf"`、`"wbf_seeded"`、`"nms"`，或一个可调用对象 |
| `fusion_iou` | `0.55` | 融合聚类使用的 IoU 阈值 |
| `min_votes` | `1` | 只保留至少被这么多成员确认过的检测框 |

成员可以是通过 `LibreYOLO()` 工厂解析的权重路径、一个已经构造好的模型、一个
导出的后端，或者一个 `ExternalDetector`。每个成员都必须是检测任务的模型。

<code-tabs name="usage" />

构造时会拒绝这些情况：成员少于两个、长度不对的 `weights` 列表、非正数的权重、
不是正整数的 `min_votes`，以及大于成员数量的 `min_votes`。`fusion="nms"` 搭配
`min_votes > 1` 同样会报错，因为 NMS 丢弃了聚类归属信息，无法统计票数。

`weights` 调节对每个成员的信任程度。权重越高，融合后的坐标和分数越偏向该成员。
惯例是让它们与验证 mAP 成正比。

## 类别空间

`names` 完全相同的成员直接放行。否则类别空间会按名称取并集，成员的类别 ID 通过
查找表重映射，融合后的 `Results.names` 就是这个并集。融合只在同一个统一类别内
合并检测框，所以只有一个成员认识的类别会不经融合直接通过。不一致时会在构造阶段
打一条警告日志。

`min_votes` 会按类别封顶，上限是标签空间里包含该类别的成员数量，这样在词表只有
部分重叠时共识依然有意义。

## 调用集成模型

```python
ens(
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
    output_path=None,
    color_format="auto",
    batch=1,
    stream=False,
    stream_buffer=False,
    vid_stride=1,
    show=False,
    **kwargs,
)
```

`predict` 是 `__call__` 的别名。返回值是一贯的 `Results`，它的 `speed` 会按成员
拆分耗时，并额外加上一个 `fusion` 条目。单张图像的源返回一个，列表或目录返回一个
列表，`stream=True` 返回一个生成器。

`conf`、`iou` 和 `device` 会广播到每个成员，也接受按成员各给一个值，所以
`conf=[0.25, 0.4]` 给成员 0 的阈值是 0.25，给成员 1 的阈值是 0.4。`imgsz` 是 int
或元组时广播，只有在是列表时才按成员生效，所以 `imgsz=(480, 640)` 是给所有成员的
同一个矩形尺寸，而 `imgsz=[480, 640]` 表示成员 0 用 480、成员 1 用 640。每个取值
都必须对该成员所属的家族有效。

`augment` 会广播到支持测试时增强（TTA）的成员，导出的后端会忽略它。`classes`
接受并集里的类别 ID，`max_det` 作用于融合后的结果，所以成员可以放开跑，由集成
统一裁剪一次。`batch` 只是为了 API 对齐而被接受；图像是逐张顺序处理的。

`val()` 和 `export()` 会抛出 `NotImplementedError`。请分别对各个成员做验证和导出。

## ExternalDetector

```python
ExternalDetector(fn: Callable, names: dict[int, str])
```

把任意检测用的可调用对象适配成一个成员。`fn` 接收一张 PIL 图像，返回
`(boxes, scores, labels)`，其中 boxes 是原图像素坐标下的 xyxy，labels 是在
`names` 里有效的类别 ID。张量、数组和嵌套列表都可以。LibreYOLO 不会从外部代码里
导入任何东西。

适配器会校验返回值：必须是一个三元组，boxes 的形状必须是 `(N, 4)`，三个数组的
长度必须一致，并且每个类别 ID 都要出现在 `names` 里。置信度小于等于 `conf` 的
检测结果会在融合前被丢弃。

## 融合操作

这些融合原语是 `libreyolo.ops` 里独立的 torch 操作。它们不依赖模型，可以单独
导入，这也是它们和集成分开导出的原因。

<code-tabs name="ops" />

三个操作都接收同样的位置参数 `boxes, scores, labels, model_ids`，并返回
`(boxes, scores, labels)`。

| 操作 | 注册表（registry）键 | 行为 |
|---|---|---|
| `weighted_boxes_fusion` | `wbf` | 顺序执行、忠于论文的 weighted boxes fusion |
| `wbf_seeded` | `wbf_seeded` | 同一套归约的并行单趟变体 |
| `nms_fusion` | `nms` | 把所有结果拼接起来，再做按类别的 NMS |

`FUSIONS` 把这三个注册表键映射到对应的可调用对象，`LibreEnsemble` 就是在这里查找
`fusion=` 的。

```python
weighted_boxes_fusion(
    boxes, scores, labels, model_ids,
    *,
    weights=None,
    num_models=None,
    iou_thr=0.55,
    skip_box_thr=0.0,
    conf_type="avg",
    min_votes=1,
    models_per_label=None,
    label_weights=None,
)
```

`wbf_seeded` 的签名完全相同。`nms_fusion` 除了 `conf_type` 之外接收同样的参数，
并在 `min_votes > 1` 时抛出 `ValueError`。

在 `weighted_boxes_fusion` 里，检测结果按加权后的置信度从高到低依次处理。每个
检测要么加入已有的某个簇——标签相同、IoU 高于 `iou_thr`、并且与该簇当前的融合框
重叠得最好——要么另起一个新簇。一个簇的融合框是其成员坐标按置信度加权的平均值，
分数则是成员置信度的加权平均或最大值，并会重新缩放，使得被更少模型确认的框得分
更低。

`wbf_seeded` 用 `iou_thr` 下按类别的 NMS 挑出簇的种子，把每个检测分配给标签相同、
IoU 最高的那个种子，再用同样的方式归约每个簇。簇的形状在一趟处理中不会变化，
所以整个操作就是固定形状的张量运算。只要聚类没有歧义，两个变体的结果就一致；在簇
彼此重叠成链时可能略有差异。

`nms_fusion` 会原样保留每一组重叠框里置信度最高的那个。按模型给的 `weights` 只在
抑制排序时缩放置信度，留下来的框保留原始分数。

## 自定义融合

`fusion=` 也接受与上面这些操作签名相同的可调用对象。它的名字会记录在
`ens.fusion` 上，没有名字时记为 `"custom"`。返回值同样会被校验：必须是一个形状
一致的 `(boxes, scores, labels)` 三元组。
