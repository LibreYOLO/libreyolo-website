---
title: 表面法线
seo_title: LibreYOLO 中的表面法线估计
description: 在 LibreYOLO 中用一张图像预测稠密的表面法线场。读懂相机坐标系约定，验证角误差，并导出模型。
lead: 表面法线估计预测每个可见表面朝向的方向。LibreYOLO 把它作为 normal 任务提供，返回原图画布上一整片稠密的单位向量场。
keywords:
  - 表面法线估计 python
  - 图像生成法线贴图
  - 单目几何 moge-2
  - 法线角误差指标
  - 稠密法线预测
last_verified: 1.5.0
snippets:
  predict:
    - label: 预测法线场
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        result = model(SAMPLE_IMAGE, save=True)

        normals = result.normal_map
        print(normals.data.shape)      # (H, W, 3) float32 单位向量
        normals.assert_normalized()    # 只要有像素不是单位长度就抛异常
    - label: 读取单个像素
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        result = model(SAMPLE_IMAGE)

        # OpenCV 相机坐标系：+x 向右，+y 向下，+z 指向场景内部，正对相机的表面
        # 读数接近 (0, 0, -1)
        field = result.normals.data
        h, w = field.shape[:2]
        print(field[h // 2, w // 2])
    - label: 保存可视化结果
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        result = model(SAMPLE_IMAGE)

        # plot() 会把这个场渲染出来，法线结果和边缘结果都有定义
        result.plot().save("normals.png")
  val:
    - label: 验证并读取指标键
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=518)

        print(metrics["metrics/mean_angular_error"])     # 度
        print(metrics["metrics/median_angular_error"])   # 度
        print(metrics["metrics/within_11_25"])           # 像素百分比
        print(metrics["metrics/within_22_5"], metrics["metrics/within_30"])
  export:
    - label: 导出
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        model.export(format="onnx", imgsz=518)
    - label: 运行导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂函数按文件后缀路由，导出的产物像任何检查点一样加载，
        # 并返回同样的 Results 对象
        model = LibreYOLO("LibreMoGe2s-normal.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.normal_map.data.shape)
source_hash: d26d26d894b436ff
---

## 定义

`normal` 任务从单张 RGB 图像为每个像素预测一个三分量单位向量：该像素处表面朝向
的方向。和深度不同，它的输出没有自由尺度，因此两次预测无需对齐就能直接比较。

一次预测会填充 `result.normal_map`，这是一个 `NormalMap` 载荷，装着原图画布上的
`(H, W, 3)` float32 数组，也可以通过 `result.normals` 拿到。向量使用 LibreYOLO
的 OpenCV 相机坐标系，`+x` 向右、`+y` 向下、`+z` 指向场景内部，并且它们朝向相
机，所以一个正对相机的表面读数为 `(0, 0, -1)`。`.assert_normalized()` 检查每个像
素都是有限值，且在容差范围内为单位长度。`result.boxes` 保持为空，因此 `conf`、
`iou` 和 `max_det` 不起作用，而 `Results.plot()` 覆盖这个任务。

## 模型

有两个家族支持 `normal`。

[MoGe-2](/docs/models/moge-2) 是专用的那个：一个单次前向的单目几何模型，有三种编
码器尺寸。LibreYOLO 没有把这些检查点（checkpoint）复制到自己的组织下；加载其中一
个时，会按固定的修订版本（revision）从官方仓库下载对应的尺寸，并对照记录在案的
SHA-256 做校验。

[LibreMODUS](/docs/models/libremodus) 是把法线作为一个 any-to-any 模型的目标之一
产出，而且可以接受深度图而不是 RGB 图像作为输入。它需要 `modus` 附加依赖
（extra）和你自己通过认证的 Hugging Face 账号，并且既不提供 `val()` 也不提供
`export()`，所以它不参与下面的验证和导出两节。

## 预测

MoGe-2 权重在首次使用时下载并缓存在本地。

<code-tabs name="predict" />

`imgsz` 必须能被 ViT 编码器的 patch size 整除，LibreYOLO 会在运行开始前检查这一
点。对一个图像列表做预测时，每张图像跑一次前向；这个任务没有堆叠成批的快速通
道。关于输入源、流和结果处理，见[预测](/docs/predict)。

## 数据集格式

法线验证为每张图像配一张主文件名相同、分辨率相同的三通道 16 位 PNG，另加一张可
选的有效性掩码。

```text
dataset/
  data.yaml
  images/
    val/room.jpg
  normals/
    val/room.png
  masks/
    val/room.png
```

```yaml
path: dataset
train: images/train
val: images/val
normals_dir: normals
masks_dir: masks
nc: 1
names: {0: normal}
```

目标 PNG 严格是三通道 `uint16`，通道按 RGB 存储。解码方式是
`n = png / 65535 * 2 - 1`，随后对每个向量重新归一化，解码出的向量使用和预测相同
的 OpenCV 相机坐标系。掩码像素只要非零就算有效；没有掩码文件时，每个有限且非零
的解码向量都有效。无效的目标像素和填充出来的目标像素在内部保存为 `(0, 0, 0)`，
永远不会参与任何指标。完整约定见[数据集格式](/docs/reference/dataset-formats)。

## 训练

两个法线家族都没有训练实现：`train()` 在它们上面都会抛出 `NotImplementedError`。
MoGe-2 的页面指向它那些固定住的官方检查点，用于预测、验证和导出。

## 验证

`val()` 在数据集标记为有效的像素上，测量每个预测向量与其真值（ground truth）向量
之间的夹角。

<code-tabs name="val" />

`metrics/mean_angular_error` 和 `metrics/median_angular_error` 就是这个夹角，单位
是度，越低越好。`metrics/within_11_25`、`metrics/within_22_5` 和
`metrics/within_30` 是角误差落在 11.25、22.5 和 30 度以内的有效像素百分比，所以
越高越好。注意单位：这三个是百分比，不是分数。`fitness` 是
`metrics/within_11_25` 除以 100，这让最佳检查点选择和其他每一个任务处在同一个
`[0, 1]` 量纲上。

## 导出

导出后的法线模型会按文件后缀经由 `LibreYOLO()` 重新加载，因此一个 `.onnx` 文件的
行为和检查点一样，返回同样的 `Results`。

<code-tabs name="export" />

法线导出使用固定分辨率、批大小为 1 的运行时约定：`dynamic` 以及不等于 1 的
`batch` 都会被拒绝，而且 `imgsz` 必须能被编码器的 patch size 整除。各格式的覆盖
范围见 [MoGe-2 页面](/docs/models/moge-2)和[完整导出矩阵](/docs/reference/export-matrix)。
[导出](/docs/export)列出了每种格式接受的参数。
