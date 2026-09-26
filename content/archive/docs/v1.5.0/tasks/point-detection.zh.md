---
title: 点检测
seo_title: LibreYOLO 中的点检测与计数
description: 在 LibreYOLO 里用单个点而不是检测框定位物体。预测中心点、统计数量、训练 FOMO，并读懂点任务的指标。
lead: >-
  点检测为每个物体返回一个 x, y 位置，而不是一个检测框。LibreYOLO 把它做成 point 任务，一次预测里每个物体占一行，包含
  x、y、类别和置信度。
keywords:
  - 点检测 python
  - 目标计数 python
  - 中心点检测
  - FOMO 点定位
  - 图像目标计数
  - 点定位
last_verified: 1.5.0
snippets:
  predict:
    - label: 预测点并计数
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # LibreFOMO 权重不会自动下载，先从
        # https://huggingface.co/LibreYOLO 拿一个检查点，再按本地路径加载
        model = LibreYOLO("./LibreFOMOs-point.pt")
        result = model(SAMPLE_IMAGE, save=True)

        points = result.points
        print(len(points))     # 物体数量
        print(points.xy)       # (N, 2) 中心点，单位是原图像素
        print(points.cls, points.conf)
    - label: 归一化坐标与分类别计数
      language: python
      code: |
        from collections import Counter

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("./LibreFOMOs-point.pt")
        result = model(SAMPLE_IMAGE)

        points = result.points.numpy()
        print(points.xyn)                          # 同样的中心点，取值在 [0, 1]
        print(Counter(points.cls.astype(int).tolist()))
  train:
    - label: 在 YOLO 数据集上训练 FOMO
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.train(data="my-dataset.yaml", epochs=40, batch=32, lr0=3e-4)
    - label: 用训练好的检查点预测
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("./LibreFOMOs-point.pt")
        results = model.train(data="my-dataset.yaml", epochs=40)

        # train() 会把最优检查点重新加载回同一个对象，所以调用一返回，
        # 模型用的就是训练好的权重
        print(results["best_checkpoint"])
        print(model(SAMPLE_IMAGE).points.xy)
  val:
    - label: 验证并读取指标键
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/precision"], metrics["metrics/recall"])
        print(metrics["metrics/f1"])
        print(metrics["metrics/mAP@[0.01:0.10]"])   # fitness
        print(metrics["metrics/MLE"])               # 平均定位误差
        print(metrics["metrics/MAE"], metrics["metrics/RMSE"])   # 计数误差
    - label: 修改距离阈值
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("./LibreFOMOs-point.pt")


        # 扫描的上下界是键名文本的一部分，所以自定义扫描会给它产生的

        # mAP 键改名

        metrics = model.val(data="my-dataset.yaml", dist_thresholds=[0.02,
        0.05])


        print(metrics["metrics/mAP@0.02"])

        print(metrics["metrics/mAP@[0.02:0.05]"])
  export:
    - label: 导出
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.export(format="onnx")
    - label: 运行导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂函数按文件后缀分发，所以导出产物的加载方式和检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("./LibreFOMOs-point.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.points.xy)
source_hash: 932153c8870d1c7c
---

## 定义

`point` 任务用一个 x, y 坐标加一个类别来定位每个物体，没有宽、没有高，也没有掩码。
因为一次预测就是一个扁平的物体列表，行数就是物体数量，这正是它成为计数任务的原因。

一次预测会填充 `result.points`，它是一个 `Points` 载荷，包裹着一个 `(N, 4)` 数组，
每行是原图像素坐标下的 `x, y, class, confidence`。`.xy` 返回坐标，`.xyn` 返回同样的
坐标除以图像尺寸后的结果，`.cls` 返回类别索引，`.conf` 返回分数；`len()` 返回点的
数量。`result.boxes` 保持为空，所以 `iou` 和 `max_det` 没有作用对象。

## 模型

有三个家族支持 `point`，而且它们之间不能互相替代。

[FOMO](/docs/models/fomo) 是固定词汇表的那个选项：一个网格分类器，把低分辨率网格的
每个格子判为背景或物体中心。它是 LibreYOLO 唯一能训练的点家族，也是唯一能导出的。

[LocateAnything](/docs/models/locate-anything) 接受文本而不是类别索引，所以词汇表就
是你写下的任意短语。它需要 `vlm` 这个额外依赖，构造方式是 `LibreLocateAnything` 而
不是走 `LibreYOLO()` 工厂函数，而且它的权重限定为非商业用途。确切条款，以及这个
检查点（checkpoint）叠加的另外两个许可证，都在它自己的页面上。

[SenseNova-Vision](/docs/models/sensenova-vision) 通过它服务另外六个任务时用的同一个
提示式生成检查点来支持 `point`，加载方式是
`LibreVLM("sensenova-vision", task="point")`。它需要 `sensenova` 这个额外依赖，而且
每次预测都是在一个 7B 模型上跑一遍生成，所以单图延迟会明显高于专门设计的检测器。它的
权重是非商业的；许可证在它自己的页面上。

## 预测

LibreFOMO 权重是本站唯一不自动下载的例外。`LibreYOLO("LibreFOMOs-point.pt")` 会在
磁盘上找这个文件，找不到就抛出一个指名它的 `ValueError`，而不是去下载它。先从
Hugging Face 上的 [LibreYOLO 组织](https://huggingface.co/LibreYOLO)下载一个检查点，
然后按本地路径加载，或者自己训练一个。

<code-tabs name="predict" />

文件名必须带上 `-point` 这个任务后缀，加载器才能识别它。`predict(..., nms_radius=1)`
控制两个 FOMO 检测点在网格上至少要相隔几个格子才能同时保留下来。输入源、流式处理和
结果处理见[预测](/docs/predict)。

## 数据集格式

`point` 没有自己的标注格式。点家族读取标准的 YOLO 检测布局，从每一行检测框里推导出
一个中心点，所以 `cx cy` 就是那个点，而 `w h` 只决定这一行是否有效。

```text
dataset/
  data.yaml
  images/
    train/scene.jpg
    val/scene.jpg
  labels/
    train/scene.txt
    val/scene.txt
```

每个标注文件里每个物体占一行，坐标是归一化的：

```text
<class_id> <cx> <cy> <w> <h>
```

```yaml
path: dataset
train: images/train
val: images/val
nc: 1
names: {0: seedling}
```

标注文件缺失或为空表示没有物体。完整的约定见
[数据集格式](/docs/reference/dataset-formats)。

## 训练

FOMO 是唯一实现了训练的点家族。LocateAnything 和 SenseNova-Vision 上的 `train()` 会
抛出 `NotImplementedError`；这两个请在上游微调，再把结果加载进来。

<code-tabs name="train" />

`imgsz` 对 FOMO 来说不能随便选：它默认取所加载检查点的原生分辨率，传入别的值会抛出
一个指明期望尺寸的 `ValueError`。数据集、日志记录器和多卡训练见[训练](/docs/train)，
这个家族的默认值见 [FOMO 页面](/docs/models/fomo)。

## 验证

`val()` 用匈牙利算法把预测点和真值（ground truth）点一对一匹配，并在一组距离阈值上
做扫描。一个阈值就是归一化图像坐标下的欧氏距离，默认扫描是从 0.01 到 0.10 的十个值。

<code-tabs name="val" />

`metrics/precision`、`metrics/recall` 和 `metrics/f1` 是在扫描里最严格的那个阈值下按
类别做宏平均得到的，默认是 0.01。`metrics/mAP@0.01` 是同一个阈值下的平均精度，
`metrics/mAP@[0.01:0.10]` 是整个扫描上的均值。这个扫描值同时也是 `fitness`，即最优
检查点选择读取的那个数字。两个 mAP 键都是用当前所用的阈值拼出来的，所以传入
`dist_thresholds=` 会给它们改名。

`metrics/MLE` 是最严格阈值下匹配点对之间的平均距离，单位同样是归一化的。
`metrics/MAE` 和 `metrics/RMSE` 是计数指标而不是定位指标：它们衡量的是每张图上预测
点数量和真值点数量之间的差。

FOMO 在这些之上还加了第二组、网格级别的指标。它会扫描置信度和 `nms_radius`，把 F1
最好的那组组合发布为 `metrics/grid_F1`、`metrics/grid_precision`、
`metrics/grid_recall`、`metrics/grid_mean_distance`、`metrics/grid_TP`、
`metrics/grid_FP` 和 `metrics/grid_FN`，产生这组结果的设置则放在 `decode/threshold`
和 `decode/nms_radius` 下面。

## 导出

FOMO 走共用的导出路径，导出产物按文件后缀通过 `LibreYOLO()` 加载回来，所以一个
`.onnx` 或 `.engine` 文件的表现和检查点一样，返回同样的 `Results`。

<code-tabs name="export" />

分格式的支持情况在 [FOMO 页面](/docs/models/fomo)和[完整导出矩阵](/docs/reference/export-matrix)上。
LocateAnything 和 SenseNova-Vision 不支持导出：`export()` 在两者上都会抛异常，因为
生成式模型没有可追踪的检测图。
