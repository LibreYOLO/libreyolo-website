---
title: 姿态估计
seo_title: LibreYOLO 中的姿态估计
description: >-
  在 LibreYOLO 中为每个实例预测关键点：服务这一任务的模型家族、标注格式，以及 predict、train、validate 和 export
  调用。
lead: 姿态估计定位每个实例，并为它返回一组有序的、带名字的关键点，所以输出携带的是目标的内部结构，而不只是它的范围。任务键是 pose。
keywords:
  - 姿态估计 python
  - 关键点检测
  - 人体姿态估计模型
  - COCO 关键点
  - OKS mAP
  - 训练姿态估计模型
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 文件名中的 -pose 后缀会选中关键点 head，所以不需要
        # task 参数
        model = LibreYOLO("LibreECs-pose.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.keypoints.xy.shape)   # (N, K, 2) 像素坐标
        print(result.boxes.xyxy.shape)     # (N, 4)，同样的 N 个实例
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreECs-pose.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 只取可见的关键点
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreECs-pose.pt")(SAMPLE_IMAGE)
        kpts = result.keypoints

        # .has_visible 由关键点的第三列导出，当检查点只预测
        # (x, y) 时全为 true
        for person, visible in zip(kpts.xy, kpts.has_visible):
            print(person[visible])
    - label: 改用自顶向下
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # HRNet 是自顶向下的：它先裁出每个人。没有给定人体来源时，
        # 它会给自己配一个 LibreYOLO9t 检测器并记录这个选择
        model = LibreYOLO("LibreHRNetw32-pose.pt")
        result = model(SAMPLE_IMAGE)

        print(result.keypoints.xy.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # coco8-pose.yaml 带有内嵌的下载脚本，所以除非数据已经在本地，
        # 否则需要显式授权
        model = LibreYOLO("LibreECs-pose.pt")
        model.train(
            data="coco8-pose.yaml",
            epochs=50,
            imgsz=640,
            batch=4,
            allow_download_scripts=True,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreECs-pose.pt data=coco8-pose.yaml \
          epochs=50 imgsz=640 batch=4 allow_download_scripts=True
    - label: 用自己的数据集
      language: python
      code: |
        from libreyolo import LibreYOLO

        # data.yaml 必须声明 kpt_shape，而且标注行必须正好带有
        # 5 + K * D 个字段
        model = LibreYOLO("LibreECs-pose.pt")
        model.train(data="my-pose-dataset.yaml", epochs=50, imgsz=640, batch=8)
  val:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreECs-pose.pt")


        # val() 返回一个普通的 dict，不是对象

        metrics = model.val(data="coco8-pose.yaml", allow_download_scripts=True)


        print(metrics["metrics/keypoints_mAP50-95"])

        print(metrics["metrics/keypoints_mAP50"],
        metrics["metrics/keypoints_mAP75"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreECs-pose.pt data=coco8-pose.yaml \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-pose.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreECs-pose.pt format=onnx imgsz=640
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂函数按文件后缀分发，所以导出的产物加载起来和检查点一样，
        # 返回同样的 Results 对象
        model = LibreYOLO("LibreECs-pose.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.keypoints.xy)
source_hash: 9de01d1f615bdf33
---

## 定义

姿态估计返回的是结构，而不只是范围。每个实例仍然会拿到一个检测框、一个类别和一个
分数，同时还会拿到按固定顺序排列的 `K` 个关键点，所以索引 5 在每个实例上、每张图像
里都表示同一个身体部位。这个顺序由标签集定义；输出中没有任何东西按名字标识关键点。

`pose` 是规范的任务键，检查点（checkpoint）文件名中的 `-pose` 后缀会选中它，所以
加载已发布的权重时不需要 `task=`。

`predict()` 会在 `result.boxes` 之外填充 `result.keypoints`。`.data` 是
`(N, K, 2)` 或 `(N, K, 3)`，与检测框逐行对齐，所以其中一个里的实例 `i` 就是另一个里
的实例 `i`。`.xy` 切出像素坐标，`.xyn` 按原始图像尺寸对它们做归一化。检查点预测第三
列时 `.conf` 就是这一列，不预测时它是 `None`，而 `.has_visible` 是由它导出的布尔
掩码，没有第三列时全为 true。

有两种架构能得到这个输出。单阶段模型在一次前向中同时预测检测框和关键点。自顶向下的
模型先跑一个检测器，裁出每个实例，再在裁剪图内部回归关键点，所以它的精度取决于它前面
的那个检测器。

## 模型

有三个家族既能训练也能预测：[RF-DETR](/docs/models/rf-detr)、
[EdgeCrafter](/docs/models/edgecrafter) 和 [YOLO-NAS](/docs/models/yolo-nas)，
全都是单阶段的。RF-DETR 需要它自己的额外依赖，`pip install "libreyolo[rfdetr]"`。
RF-DETR 和 EdgeCrafter 都提供已发布的姿态检查点，两者都在单类别、只含人的数据集上
微调；EdgeCrafter 的关键点 head 在构建时就固定了，会拒绝声明了不同数量的数据集，而
RF-DETR 会为这种数据集重新初始化它的 head。YOLO-NAS 从 Deci.AI 自己的 CDN 拉取权重，
采用非商用许可，LibreYOLO 一个都不发布；它的姿态 head 同样会为新的关键点数量重建，
而且它是三者中唯一类别数不固定为 1 的，所以要做多类别或非人体骨架，比如动物姿态，
就用这个家族。

[HRNet](/docs/models/hrnet) 是自顶向下的那个选项。它能预测、验证和导出，而它的
`train()` 会抛出 `NotImplementedError`。在没有给定人体来源时，它会自动给自己配一个
LibreYOLO9t 检测器；`cropped=True` 把整张图像当作一个实例，`person_boxes=` 接收你
已经拿到的检测框，`person_detector=` 指定另一个检测器。

[SenseNova-Vision](/docs/models/sensenova-vision) 也会输出关键点。它是一个靠提示驱动
的生成式模型，有自己的工厂函数 `LibreVLM`，也有自己的额外依赖；没有设置词汇表时，
`set_task("pose")` 会回退到 person 类别。它的权重不可商用，而且每张图像的延迟远高于
一个专门做姿态的 head，因为每次预测都是一次扩散解码。

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

关键点的数量和顺序是检查点的属性，而不是库的属性，所以在不同骨架上训练出来的模型会
返回不同的 `K`，每个索引的含义也不一样。关键点第三列里装的是什么，同样是检查点的
属性：EdgeCrafter 在那里写的是一个常数，而不是逐点的分数，而且它根本没有检测框
head，所以它的每个姿态检测框都是该实例自身关键点的外接范围。数据源、流式处理和结果
处理见[预测](/docs/predict)。

## 数据集格式

布局就是检测任务的布局：每张图像一个 `.txt` 标注文件，把图像路径里的 `images` 换成
`labels` 再改掉扩展名就能找到。

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

一行就是在检测行后面追加关键点：

```text
<class_id> <cx> <cy> <w> <h> <k1x> <k1y> [<k1v>] ... <kKx> <kKy> [<kKv>]
```

字段数正好是 `5 + K * D`，其中 `D` 是 `kpt_shape` 的第二个值。检测框和关键点坐标都是
相对原始图像宽高归一化后的浮点数。可见性 `v` 只在 `D` 为 3 时出现，取值是 `0`、`1`
或 `2`。

YAML 在共用的约定之上多加两个键：

```yaml
path: dataset
train: images/train
val: images/val
kpt_shape: [17, 3]
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
names:
  0: person
```

`kpt_shape` 是必填的，取值为 `[K, 2]` 或 `[K, 3]`。`flip_idx` 是可选的，它是
`0..K-1` 的一个排列，给出每个关键点在水平翻转之后所取的索引，左手腕之所以还是左手腕
就靠它。省略它，关键点上的水平翻转增强会被关掉，而不是以错误的索引顺序照样施加。

## 训练

<code-tabs name="train" />

训练从一个已发布的 `-pose` 检查点继续，它已经带着关键点 head；任务是从你加载的检查点
里读出来的，而不是训练时传进去的某个开关，所以一个检测检查点不会因为你提出要求就变成
一次姿态训练。对 EdgeCrafter 来说，你 YAML 里的 `kpt_shape` 必须和 head 完全一致，
因为它的 head 在构建时就固定了，而 RF-DETR 和 YOLO-NAS 则会为不同的数量调整 head 的
大小。数据集、数据增强、多卡训练和日志记录器见[训练](/docs/train)。

## 验证

`val()` 返回一个由 `metrics/` 键组成的普通字典。评分用的是基于 Object Keypoint
Similarity 的 COCO 关键点评测，它按实例尺度和逐关键点的容差给每个关键点的距离误差
加权，所以它扮演的是 IoU 在检测框上扮演的角色。它需要 `pycocotools`，这个包在基础
安装里。

<code-tabs name="val" />

`metrics/keypoints_mAP50-95` 是首要数字，即在 0.50 到 0.95 的 OKS 阈值上平均得到的
平均精度均值，训练也用它来挑选最佳轮次。`metrics/keypoints_mAP50` 和
`metrics/keypoints_mAP75` 是单阈值版本，`metrics/keypoints_mAP_M` 和
`metrics/keypoints_mAP_L` 按实例面积把平均值拆成中等和大两档；COCO 关键点评测没有
定义小目标那一档。与之对应的平均查全率数字是 `metrics/keypoints_AR50-95`、
`metrics/keypoints_AR50`、`metrics/keypoints_AR75`、`metrics/keypoints_AR_M` 和
`metrics/keypoints_AR_L`。这个任务上的每个键都带 `keypoints_` 前缀，所以检测器返回的
那些检测框 `mAP` 键不会出现。

## 导出

<code-tabs name="export" />

导出的产物按文件后缀经由 `LibreYOLO()` 重新加载，所以一个 `.onnx` 或 `.engine` 文件
的行为和检查点一样，返回同样的 `Results`。格式覆盖情况因家族而异；每个模型页面上的
矩阵是从验证过的集合生成的，而不是手工敲上去的。格式、它们的额外依赖和限制见
[导出与部署](/docs/export)。
