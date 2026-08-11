---
title: 深度估计
seo_title: LibreYOLO 中的单目深度估计
description: 在 LibreYOLO 中用一张图像预测稠密的相对深度图。对比各个深度模型家族，读懂深度指标，并导出深度模型。
lead: 深度估计用单张图像预测每个像素离相机有多远。LibreYOLO 把它作为 depth 任务提供，返回原图画布上的一张稠密相对逆深度图。
keywords:
  - 单目深度估计 python
  - 单张图片生成深度图
  - 相对深度模型
  - depth anything libreyolo
  - 稠密深度预测
last_verified: 1.5.0
snippets:
  predict:
    - label: 预测深度图
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.data.shape)              # 原图画布上的 (H, W)
        print(depth.min, depth.max, depth.mean)
    - label: 处理这些数值
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        result = model(SAMPLE_IMAGE)

        depth = result.depth_map
        raw = depth.data          # 值越大越近，没有度量单位，没有尺度
        gray = depth.normalized() # 重新缩放到 [0, 1] 用于可视化
        print(raw.shape, float(gray.max()))
    - label: 一个紧凑的替代方案
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 相同的任务约定，但网络小得多，为边缘运行时打造
        model = LibreYOLO("LibreZipDepthb-depth.pt")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
  val:
    - label: 验证并读取指标键
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])   # fitness
        print(metrics["metrics/delta2"], metrics["metrics/delta3"])
  export:
    - label: 导出
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        model.export(format="onnx")
    - label: 运行导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂函数按文件后缀路由，导出的产物像任何检查点一样加载，
        # 并返回同样的 Results 对象
        model = LibreYOLO("LibreDepthAnythingV2s-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
source_hash: e0612c59f9c999b4
---

## 定义

`depth` 任务从单张 RGB 图像为每个像素预测一个值。LibreYOLO 把这个值定义为相对
逆深度：值越大表示离相机越近，这些数字没有度量单位，也没有能跨两张图像成立的
尺度。比较同一次预测中两个像素的深度是有意义的；把一个值和另一张图像的值相比
则没有意义。

一次预测会填充 `result.depth_map`，这是一个 `DepthMap` 载荷，装着原图画布上的
`(H, W)` 数组。`.min`、`.max` 和 `.mean` 读取其中的有限值，`.normalized()` 把这
张图重新缩放到 `[0, 1]` 以便显示。`result.boxes` 保持为空，因此 `conf`、`iou`
和 `max_det` 不起作用，`save=True` 写出的是这张图经过色彩映射的图像，而不是带
标注的照片。

## 模型

有六个家族支持 `depth`。

[Depth Anything V2](/docs/models/depth-anything-v2) 把 DINOv2 编码器和 DPT 解码
器搭配在一起，是这里的通用默认选择。许可对尺寸的影响不亚于精度：Small 检查点
（checkpoint）采用 Apache-2.0 许可，而 Base 和 Large 仅限非商用，所以挑选之前先
看它页面上的检查点表格。

[Depth Anything 3](/docs/models/depth-anything-3) 移植了 DA3MONO-LARGE 检查点，
那是一个普通的 transformer，架构上没有针对深度做任何特化。

[ZipDepth](/docs/models/zipdepth) 是紧凑档：一个可重参数化的 CNN，从 Depth
Anything V2 Large 蒸馏而来，另有一个检查点，它的解码器避开了 gather 和 unfold
算子，以适配缺少这些算子的 NPU 编译器。

[MiDaS](/docs/models/midas) 是确立了零样本相对深度评测协议的那一系工作，其他家
族都按这套协议衡量。它是 LibreYOLO 唯一没有重新发布的深度家族：请求一个检查点
时，会从作者的 GitHub release 下载官方文件，并校验一个固定的 SHA-256。

[LibreMODUS](/docs/models/libremodus) 是把深度作为一个 any-to-any 模型的目标之
一来完成，而不是靠专门的 head。它需要 `modus` 附加依赖（extra）和你自己通过认证
的 Hugging Face 账号，而且既不提供 `val()` 也不提供 `export()`。

[SenseNova-Vision](/docs/models/sensenova-vision) 通过一次扩散解码把深度图当作
图像生成出来，用的是同一个服务于它另外六个任务的 7B 检查点。它需要 `sensenova`
附加依赖，权重仅限非商用；许可证在它的页面上。

## 预测

权重在首次使用时从 Hugging Face 下载并缓存在本地，上面提到的那两个家族除外。

<code-tabs name="predict" />

输入分辨率按家族各有限制。Depth Anything V2 和 Depth Anything 3 建立在 DINOv2
的 patch 网格之上，因此 `imgsz` 必须能被 14 整除，LibreYOLO 会在运行前检查这一
点。`Results.plot()` 不覆盖这个任务；它只为表面法线和边缘定义。关于输入源、流和
结果处理，见[预测](/docs/predict)。

## 数据集格式

深度验证为每张图像配一张同分辨率的稠密单通道深度图，其位置是把图像路径中的目录
替换为深度目录后找到的。

```text
dataset/
  data.yaml
  images/
    val/room.jpg
  depths/
    val/room.png
```

```yaml
path: dataset
val: images/val
depths_dir: depths
nc: 1
names: {0: depth}
```

深度图是单通道 PNG 或 TIF，也可以是 `.npy`。其中的值是原始深度，单位由数据集自
己保持一致，而 `0`、负值、NaN 和无穷像素标记的是无效样本，会被排除在指标之外。
整数深度图会除以 `depth_scale`，它默认为 `256.0`，即 16 位 PNG 的惯例；浮点
`.npy` 深度图则按原样使用。`depth_stem_suffix` 和 `depth_mask_suffix` 用于那些
给深度文件或有效性掩码起了别的名字的数据集。完整约定见[数据集格式](/docs/reference/dataset-formats)。

## 训练

LibreYOLO 里没有任何一个深度家族带训练实现：六个家族上的 `train()` 都会抛出
`NotImplementedError`。每个模型页面都写明了对应的转换脚本，它把上游训练好的检查
点转成 LibreYOLO 能加载的形式。

## 验证

`val()` 运行的是共用的深度验证器。相对深度没有绝对尺度，因此每次预测先用逐图像
的最小二乘缩放和平移拟合到其真值（ground truth）的倒数上，再取倒数变回深度。下
面每个指标都是在这张对齐后的深度图上逐图像计算，再在整个数据集上取平均，且只统
计数据集标记为有效的像素。

<code-tabs name="val" />

`metrics/abs_rel` 是平均绝对相对误差，即残差除以真值深度，越低越好。
`metrics/rmse` 是均方根误差，单位是数据集自己的深度单位，同样越低越好。
`metrics/delta1`、`metrics/delta2` 和 `metrics/delta3` 是阈值精度：在有效像素
中，与真值之比（取两个方向里较大的那一个）小于 1.25、1.25 的平方和 1.25 的立方
的像素占比，所以越高越好。`metrics/delta1` 同时也是 `fitness`，即最佳检查点选择
读取的那个数。

## 导出

导出后的深度模型会按文件后缀经由 `LibreYOLO()` 重新加载，因此一个 `.onnx` 或
`.engine` 文件的行为和检查点一样，返回同样的 `Results`，只是用 `depth_map` 取代
了检测框。

<code-tabs name="export" />

覆盖范围因家族而异，Depth Anything 3 会直接拒绝其验证集合之外的任何格式，而不是
去尝试一次未经验证的转换。在定下目标格式之前，先看模型页面和[完整导出矩阵](/docs/reference/export-matrix)。
LibreMODUS 和 SenseNova-Vision 完全不支持导出。[导出](/docs/export)列出了每种格
式接受的参数。
