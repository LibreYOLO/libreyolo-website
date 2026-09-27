---
title: 阈值与过滤
seo_title: LibreYOLO 中的 conf、iou 和 max_det
description: >-
  conf、iou、max_det 和 classes 在预测时到底做了什么，哪些家族因为不跑 NMS 而忽略 iou，以及为什么 agnostic_nms
  是空操作。
lead: >-
  有四个参数决定哪些预测能留下来：conf、iou、max_det 和 classes。其中只有两个对每个家族都生效，因为集合预测器解码的是一组固定的
  query，从不跑 NMS。
keywords:
  - yolo conf 阈值
  - iou 阈值 nms
  - max_det
  - python 过滤检测类别
  - agnostic nms
  - detr 无 nms
  - 目标检测 置信度阈值
  - 推理 只保留某些类别
last_verified: 1.5.0
verification: >-
  默认值取自 libreyolo/models/base/inference.py 中的 InferenceRunner.__call__。各家族的 NMS
  行为读自 libreyolo/postprocess/ 下的每个模块，并与 libreyolo/backends/base.py 中的
  _is_nms_free_family 交叉核对。类别过滤取自 InferenceRunner._apply_classes_filter 和
  _wrap_results。agnostic_nms 的状态取自 libreyolo/utils/predict_args.py 中的
  NOOP_PREDICT_KWARGS。开放词汇的处理取自 libreyolo/models/openvocab/base.py 中的
  NMS_THRESHOLD。验证阶段的默认值取自 BaseModel.val。
snippets:
  basic:
    - label: 四个参数
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        result = model(
            SAMPLE_IMAGE,
            conf=0.25,      # 保留分数不低于该值的预测
            iou=0.45,       # NMS 重叠阈值，仅对跑 NMS 的家族有效
            max_det=300,    # 每张图的上限
            classes=None,   # 或者一个类别 id 列表
        )
        print(len(result.boxes))
    - label: 扫描 conf
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        for conf in (0.1, 0.25, 0.5, 0.75):
            result = model(SAMPLE_IMAGE, conf=conf)
            print(conf, len(result.boxes))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt conf=0.4 iou=0.5 max_det=100 \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  classes:
    - label: 过滤到指定类别
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        # 类别 id 是 model.names 的索引。在 COCO 上，0 是 person
        result = model(SAMPLE_IMAGE, classes=[0])

        print({result.names[int(c)] for c in result.boxes.cls.tolist()})
    - label: 按名称查找 id
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        wanted = {"person", "backpack"}
        ids = [i for i, name in result.names.items() if name in wanted]
        print(ids)

        filtered = model(SAMPLE_IMAGE, classes=ids)
        print(len(filtered.boxes))
  nmsfree:
    - label: 在不跑 NMS 的家族上使用 iou
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # RF-DETR 解码的是一组固定的 query，所以这里的 iou 不起任何作用
        model = LibreYOLO("LibreRFDETRs.pt")

        loose = model(SAMPLE_IMAGE, iou=0.9)
        tight = model(SAMPLE_IMAGE, iou=0.1)

        # 两种情况下数量相同。真正起作用的控制项是 conf 和 max_det
        print(len(loose.boxes), len(tight.boxes))
source_hash: 0b978963c356027d
---

## 四个参数

| 参数 | 默认值 | 适用范围 |
|---|---|---|
| `conf` | `0.25` | 每个家族 |
| `iou` | `0.45` | 跑非极大值抑制的家族 |
| `max_det` | `300` | 每个家族 |
| `classes` | `None` | 每个家族 |

<code-tabs name="basic" />

这里面有两个是通用的，另外两个不是，这是你在调任何东西之前最该知道的一件事。

验证阶段特意用了不同的默认值：`val()` 跑的是 `conf=0.001` 和 `iou=0.6`，因为平均精度（AP）是在完整的查准率-查全率曲线上算出来的，0.25 的截断会把曲线砍掉一段。

## conf

`conf` 指的是低于该分数的预测会被丢弃。它对每个家族都生效，包括那些从不跑 NMS 的家族，而且当检测结果太多或太少时，它是第一个该去调的控制项。

默认的 `0.25` 适合用来看图。喂给下游系统通常需要调高一些；测量精度则需要调得低得多。

## iou

`iou` 指的是重叠超过该值时，非极大值抑制会把同一类别的两个框中分数较低的那个去掉。只有当家族真的跑抑制时，它才有意义。

集合预测器解码固定数量的 query，然后取分数最高的那些。重复框是在训练时由架构内部抑制掉的，而不是靠一个后处理步骤，所以没有阈值可调。下面这些家族接受 `iou` 只是为了保持 API 一致，实际会忽略它：

CenterNet、DEIM、DETR、Deformable DETR、D-FINE、DINO-DETR、EdgeCrafter、
Faster R-CNN、LW-DETR、Mask R-CNN、RF-DETR、RT-DETR，以及端到端的 YOLOv9
head。基于这些解码器构建的变体也继承同样的行为。

<code-tabs name="nmsfree" />

它们大多在自己的后处理 docstring 里写明了这一点，但运行时不会有任何警告，所以在 RF-DETR 上扫一遍 `iou` 得到的是一条平线，而不是报错。Faster R-CNN 和 Mask R-CNN 的情况稍有不同：两者都已经在模型内部跑过 NMS，用的是上游固定的阈值，而 `iou` 没有受支持的办法去改它。

下面这些家族确实会用到它：YOLOv1 到 YOLOv4、YOLOv7、YOLOv9、YOLOX、YOLO-NAS、RTMDet、PicoDet、EfficientDet、FCOS、RetinaNet 和 SSD。

有两个预测时的选项会让 `iou` 即使对集合预测器也变得重要，因为它们都在模型跑完之后合并检测框：

- `tiling=True` 会以 `iou` 为阈值用分类别 NMS 调和重叠的切片
- `augment=True` 会以 `iou` 为阈值用分类别 NMS 合并翻转后的视图

这两个都在[推理性能](/docs/predict/performance)里有介绍。

开放词汇检测器有自己的规则。如果一个家族的 processor 会跑 NMS，它就会声明自己的默认阈值并遵守 `iou`，OMDet-Turbo 就是这种情况。什么都不抑制的家族——Grounding DINO、OWLv2 和 OV-DEIM——在传入 `iou` 时会发出警告。这是库里唯一一处这样的警告。

## max_det

`max_det` 限制一张图能返回多少个预测。它在所有地方都生效，但机制不同：跑 NMS 的家族是在抑制之后截断，集合预测器则把它当作 top-k 选择的大小。

有些家族会压到比你要求的更低，因为它们上游的参考配置就是这么做的。SSD 的上限是 200，RTMDet 实例分割是 100，FCOS 用的是它自己的每图检测数上限。把 `max_det` 调到这些值以上不会有任何效果。

`max_det` 唯一一处集中应用而不是按家族应用的地方是切片推理，合并后的列表会在切片调和完成之后被截断。

## 类别过滤

<code-tabs name="classes" />

`classes` 接受一个类别 id 列表，只保留类别在列表里的预测。id 是 `result.names` 的索引，最稳妥的拿法是从一个结果里读 `names`，而不是假定某个数据集的顺序。

过滤是集中做的，发生在每个家族各自的后处理之后，在所有预测路径都会经过的那个唯一入口里。这带来两个值得知道的结果。一是它对每个家族都有效，包括那些没有 NMS 的家族。二是它同时会过滤与检测框对齐的负载，所以掩码、关键点和旋转框会跟着一起被裁掉，而不是留下对不上的数据。

在命令行上，`classes` 接受一个裸整数、一个列表，或者一个逗号分隔的字符串：

```bash
libreyolo predict model=LibreYOLO9s.pt classes=0 source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
libreyolo predict model=LibreYOLO9s.pt classes="[0,2,5]" source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
```

过滤不等于白捡精度。模型仍然会把预算花在预测那些你随后要丢掉的类别上，而且 `max_det` 是由家族在过滤之前应用的，所以一张挤满了你不想要的类别的图，可能在轮到你要的类别之前就已经撞上了上限。真出现这种情况，就把 `conf` 调低或者把 `max_det` 调高。

## agnostic_nms

`agnostic_nms` 会被接受，但什么也不做。传入它会触发一条警告，说明它只是为了命令行兼容性保留的空操作（no-op），这个参数随后会被丢弃。

库里没有类别无关的抑制模式。每一次 NMS 调用都是区分类别的，所以两个不同类别的重叠框都会保留下来，`iou` 取什么值都一样。如果这成了问题，就先用 `classes` 过滤，或者自己在 `result.boxes` 上跨类别做抑制。

## predict 会拒绝什么

有两个参数会直接抛错而不是警告：`visualize` 和 `embed` 都会抛出 `NotImplementedError`。要拿嵌入向量，就用 `task="embed"` 加载模型，然后正常调用 `predict` 或 `embed`。

任何无法识别的参数都会抛出 `TypeError`，并列出支持的选项，所以打错字会立刻失败，而不是被悄悄忽略。

下面这些会被接受、发出警告然后丢弃：`agnostic_nms`、`boxes`、`dnn`、`half`、`line_width`、`retina_masks`、`show_conf`、`show_labels` 和 `verbose`。
