---
title: Dome-DETR
families:
  - domedetr
seo_title: Dome-DETR：在 LibreYOLO 里做微小目标检测
description: "使用 Dome-DETR 进行微小目标检测、训练和验证。镜像中的预训练检查点保留仅限学术研究的条款。"
lead: >-
  一个建立在 D-FINE 之上的微小目标专家：一个密度 head 决定目标在哪里，编码器注意力被限制在装着目标的那些窗口内，query
  数量也由这份密度来定，而不是固定不变。LibreYOLO 支持它做检测。
keywords:
  - Dome-DETR
  - 微小目标检测
  - 小目标检测
  - 航拍 目标检测
  - 无人机 目标检测
  - 遥感目标检测
  - VisDrone
  - AI-TOD
  - DETR
  - 密度自适应 query
last_verified: "1.6.0"
snippets:
  predict:
    - label: "Python"
      language: "python"
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 预训练权重仅限学术研究
        model = LibreYOLO("LibreDOMEDETRs-visdrone.pt", device="cpu")
        print(model(SAMPLE_IMAGE).boxes)
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreDOMEDETRs-visdrone.pt")

        model.train(data="my-dataset.yaml", epochs=160, imgsz=800, batch=4,
        lr0=2e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDOMEDETRs-visdrone.pt data=my-dataset.yaml \
          epochs=160 imgsz=800 batch=4 lr0=2e-4
    - label: 多卡训练
      language: bash
      code: |
        libreyolo train model=LibreDOMEDETRs-visdrone.pt data=my-dataset.yaml \
          epochs=160 device=0,1 batch=4
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDOMEDETRs-visdrone.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDOMEDETRs-visdrone.pt data=my-dataset.yaml
source_hash: 8482301790a9b8d9
---

## 安装

Dome-DETR 不需要任何可选 extra。它导入的一切都在基础安装里。

```bash
pip install libreyolo
```

## 预测

六个转换后的检查点自动从 LibreYOLO 镜像下载。其上游条款将用途限制为学术研究。

<code-tabs name="predict" />

返回的 `Results` 对象和每个家族返回的都是同一个，所以换成另一个检测器只是一行的
改动。`conf` 和 `max_det` 过滤的是 query 的选择；`iou` 为了 API 一致性而被接受，
但没有任何作用，因为解码器是一个集合预测器，没有 NMS 步骤。数据源、流式处理和
结果处理见[预测](/docs/predict)。

这个家族有两项能力是关掉的。CUDA 图捕获被禁用，因为 PAQI 的 query 数量取决于
数据，前向过程的形状因此会逐张图片变化，而这恰恰是图捕获消化不了的东西。测试时
增强（TTA）只在单一的固定正方形尺寸上运行，所以多尺度 TTA 的请求是空操作。

## 变体

三种尺寸，s、m 和 l，都是 800 乘 800。尺寸决定骨干，而权重来自的那个数据集决定
解码器深度和 query 预算，所以单看一个尺寸代号并不能确定一张计算图。AI-TOD-V2
权重在每张图片 300 到 1500 个 query 之间取值，VisDrone 权重在 250 到 500 之间，
大模型在 AI-TOD-V2 上跑四个解码器层，在 VisDrone 上跑六个。

Dome-DETR 就是 D-FINE 加上三样东西。DeFE 预测一张密度图。MWAS 用这张图把编码器
注意力限制在真正装着目标的那些窗口内，而不是到处都算。PAQI 用同一份密度来定
query 集合的大小，而不是固定解码 300 个。收益集中在目标最小的地方，随着目标变大
而收窄：上游自己的消融实验把极微小目标上的 AP 从 14.0 抬到 17.8，而中等目标上的
AP 只从 45.4 动到 46.4。把它当作 [D-FINE](/docs/models/d-fine) 在航拍、无人机和
遥感图像上的搭档，而不是它的替代品。

这个家族没有记录 Vision Analysis 基准测试数据。

## 训练

Dome-DETR 可以训练。训练跑的是上游完整的目标函数：D-FINE 的那套损失函数，加上
DeFE 的密度与计数监督，其中填充出来的 query 会从分类项里被掩掉，去噪注意力掩码
按图片单独生成，这样一张图片的填充就泄漏不到另一张里去。

<code-tabs name="train" />

配置沿用 D-FINE 的配方，只改 MWAS 所要求的那些。`imgsz` 是 800，`lr0` 是
`2e-4`，骨干参数组按 `backbone_lr_mult=0.1` 缩放，`multi_scale` 被强制关闭，
因为 MWAS 的窗口需要输入始终能被 stride 8 整除。`batch` 默认是 4 而不是 D-FINE
的 16：PAQI 会把每个批次填充到其中最宽的那个成员，所以显存跟着批次里最忙的那张
图片走，而不是跟着平均那张。

关于精度，有一点要老实说。上游用 `MultiStepLR(milestones=[80, 120], gamma=0.8)`
训练 160 轮，而这里的默认值用 D-FINE 的 flat-cosine 调度跑同样的 160 轮。这套
调度这里没有复现，论文里的 AP 数字也没有复现，所以请把它们读作上游作者的结果，
而不是这套配方能达到那些数字的承诺。如果目标是对齐论文，就自己提供上游的调度。

数据集、数据增强、多卡训练和日志记录器见[训练](/docs/train)。

## 验证

`val()` 返回一个以指标名为键的字典，并在 `verbose` 保持开启时打印每个类别的
结果。

<code-tabs name="val" />

验证是在你自己的数据集上跑的，格式和你训练时用的一致。库里的 COCO 验证门槛在
这里不适用，因为这个家族没有可供衡量的 COCO 检查点。

## 导出

导出不受支持，所有格式都一样，请求导出会抛出异常，而不是产出一个文件。

原因在 PAQI。它按图片决定 query 数量，依据是经过密度过滤的候选和一个贪心的
密度自适应抑制循环，所以解码器输出的长度是输入的属性，而不是计算图的属性。
追踪（tracing）会把追踪时那张图片碰巧产生的数量固化进去，得到的产物对其他每一张
图片都会悄无声息地返回错误结果。要做成静态形式，就得把那个抑制循环在全部 250 到
1500 个候选上展开，而退化成固定的 top-k 又恰好会砍掉这个家族赖以存在的微小目标
查全率。如果你需要一个能导出的检测 transformer，该拿的是
[D-FINE](/docs/models/d-fine)。

## 检查点

<checkpoint-table />

## 许可证

<provenance-box>

六个镜像保留上游仅限学术研究的限制。代码的许可独立于权重。上游仓库采用 Apache-2.0，LibreYOLO 移植采用 MIT，用自己的数据训练出的权重归你所有。

</provenance-box>
