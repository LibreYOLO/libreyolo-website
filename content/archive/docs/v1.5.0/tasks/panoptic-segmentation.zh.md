---
title: 全景分割
seo_title: LibreYOLO 中的全景分割
description: >-
  在 LibreYOLO 中给每个像素分配唯一一个分段：服务这项任务的模型家族、COCO-panoptic 数据集格式，以及 predict 和
  validate 调用。
lead: 全景分割把每个像素分配给恰好一个互不重叠的分段（segment），把可数的目标实例与无定形的背景区域统一起来。任务键是 panoptic。
keywords:
  - 全景分割 python
  - 全景质量 pq 指标
  - thing stuff 分割区别
  - coco panoptic 格式
  - 分段 id 图
  - eomt 全景分割
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 文件名里的 -panoptic 后缀会选中任务，
        # 因此不需要传 task 参数
        model = LibreYOLO("LibreEoMTl-panoptic.pt")
        result = model(SAMPLE_IMAGE, save=True)

        pan = result.panoptic
        print(pan.data.shape)       # (H, W) 分段 id
        print(pan.segments_info)    # [{"id": ..., "category_id": ...}, ...]
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreEoMTl-panoptic.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 逐个分段处理
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreEoMTl-panoptic.pt")(SAMPLE_IMAGE)
        pan = result.panoptic

        for segment in pan.segments_info:
            pixels = pan.segment_mask(segment["id"])   # 布尔 (H, W)
            print(result.names[segment["category_id"]], int(pixels.sum()))
    - label: 更小的检查点
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEoMTs-panoptic.pt")
        result = model(SAMPLE_IMAGE)

        print(len(result.panoptic.segment_ids))
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-panoptic.pt")

        # val() 返回的是一个普通 dict，不是对象
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PQ"])
        print(metrics["metrics/SQ"], metrics["metrics/RQ"])
        print(metrics["metrics/PQ_things"], metrics["metrics/PQ_stuff"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEoMTl-panoptic.pt data=my-dataset.yaml
source_hash: b8adc9ccde7a4e6c
---

## 定义

全景分割是另外两项分割任务的并集。每个像素恰好得到一个分段，分段之间从不重叠，
而一个分段要么是 thing，也就是可数的目标实例，要么是 stuff，也就是天空或道路
这类无定形的区域。这让它比[实例分割](/docs/tasks/instance-segmentation)更严格，
后者不给背景像素做分配，还允许掩码重叠；也比
[语义分割](/docs/tasks/semantic-segmentation)更严格，后者虽然标注了每个像素，
却会把同一个类别里相接的实例合并起来。

`panoptic` 是规范的任务键，检查点（checkpoint）文件名里的 `-panoptic` 后缀会
选中它，因此加载已发布的权重时不需要 `task=`。

`predict()` 会填充 `result.panoptic`。`.data` 是原图画布上一张 `(H, W)` 的整数
分段 id 图。`.segments_info` 是一个 dict 列表，每个分段一项，每项至少带有
`{"id", "category_id"}`，其中 `id` 对应图里的某个值，`category_id` 是
`result.names` 的索引。`.segment_ids` 按从小到大列出出现过的 id，
`.segment_mask(id)` 返回某一个分段的布尔 `(H, W)` 选区。分段 id `0` 是 void 值：
未标注的像素，排除在指标之外，也不会出现在 `.segment_ids` 里。

thing 与 stuff 的区分是类别的属性，不是单个分段的属性。它记录在标签集的类别元
数据上，预测载荷为了方便可以把它以 `"isthing"` 的形式复制到每个分段上，但真正
权威的仍然是类别元数据。

## 模型

通过 `LibreYOLO()` 服务这项任务的家族是 [EoMT](/docs/models/eomt)。它在基础包上
就能跑，并提供 s、b、l 三种尺寸的全景检查点，都在 COCO 上训练。

[SenseNova-Vision](/docs/models/sensenova-vision) 也能输出全景图。它是一个提示
驱动的生成式模型，有自己的工厂函数 `LibreVLM` 和自己的 extra；没有设置词汇表
时，它会回退到自己调优时用的那套 COCO 全景类别。它的权重不可商用。它的单图延迟
远高于专门做分割的模型，因为每次预测都是一次扩散解码。

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

`conf` 过滤查询的选择。数据源、流式处理和结果处理见[预测](/docs/predict)。

## 数据集格式

LibreYOLO 原样采用 COCO-panoptic 格式，出自 Kirillov 等，CVPR 2019。不存在
LibreYOLO 专属的全景目录布局。

```text
dataset/
  data.yaml
  images/
    val/000000000139.jpg
  annotations/
    panoptic_val.json
    panoptic_val/000000000139.png
```

每张图像对应一张分辨率相同的 RGB PNG，其中每个像素的颜色编码了它所属分段的 id：

```text
segment_id = R + 256 * G + 256 * 256 * B
```

分段 id `0`，也就是 RGB 黑色，是 void：未标注的像素，既不会给预测加分，也不会
扣分。其余每个像素都恰好属于一个分段。

这份 JSON 按图像列出分段 id PNG 以及它里面的分段：

```json
{
  "images":      [{"id": 139, "file_name": "000000000139.jpg"}],
  "annotations": [{"image_id": 139, "file_name": "000000000139.png",
                   "segments_info": [
                     {"id": 3226956, "category_id": 1, "area": 2840,
                      "bbox": [413, 158, 53, 138], "iscrowd": 0}]}],
  "categories":  [{"id": 1, "name": "person", "isthing": 1}]
}
```

`annotations[].file_name` 指出全景目录里那张 PNG 的名字，`segments_info[].id`
与该 PNG 中的某个值对应。`iscrowd` 标记群组区域：它们永远不算漏检，而大部分
覆盖了其中一个的预测也不算误检。`isthing` 位于 `categories` 上，绝不在单个分段
上。

YAML 同时指向这两者：

```yaml
path: dataset
val: images/val
annotations:
  val: annotations/panoptic_val.json
panoptic_dir:
  val: annotations/panoptic_val
names:
  0: person
  1: bicycle
```

`annotations` 和 `panoptic_dir` 都可以接受单个路径，也可以接受按划分（split）
给出的映射。原始的 COCO 类别 id 通常并不连续，而模型预测的是连续的 `0..nc-1`，
所以 id 会按类别名通过 `names` 重映射。JSON 里某个类别在 `names` 中缺失会报错，
而不是被静默丢弃，因为丢掉它会被永久算作漏检。

规范加载器是 `libreyolo.data.PanopticDataset`。

## 训练

目前 LibreYOLO 里没有任何家族训练全景分割：EoMT 的 `train()` 会抛出
`NotImplementedError`，所以全景检查点只能按发布时的样子直接用。

## 验证

`val()` 返回一个由 `metrics/` 键组成的普通字典，它在真值（ground truth）分辨率
下、在数据集 YAML 里 `val` 指定的那份划分上计算。同一类别的一个预测分段和一个
真值分段在 IoU 高于 0.5 时匹配，而且这个匹配是唯一的。

<code-tabs name="val" />

`metrics/PQ` 是全景质量（Panoptic Quality），也就是最主要的那个数字。在单个类别
内部，它是两个因子的乘积。分割质量是匹配上的那些分段的平均 IoU，说明匹配到的
形状对得有多齐。识别质量是 `TP / (TP + 0.5 FP + 0.5 FN)`，也就是匹配本身的 F1
分数，说明总共找到了多少个分段。这三个数字随后在出现过的类别上取平均，报告为
`metrics/PQ`、`metrics/SQ` 和 `metrics/RQ`，所以报告出来的 PQ 是各类别乘积的
均值，而不是两个报告出来的均值的乘积。

`metrics/PQ_things` 和 `metrics/PQ_stuff` 把同样的各类别 PQ 分别在 thing 类别和
stuff 类别上取平均，`metrics/categories` 则统计出现过、因而参与了平均的类别数。
这个字典里还带有 `fitness`，它是 PQ 值的一份副本。

## 导出

全景检查点不能导出。这个任务上 `export()` 会抛出 `NotImplementedError`，因为
查询掩码输出还没有导出运行时的契约。EoMT 的语义任务可以导出，见
[语义分割](/docs/tasks/semantic-segmentation)和[导出与部署](/docs/export)。
