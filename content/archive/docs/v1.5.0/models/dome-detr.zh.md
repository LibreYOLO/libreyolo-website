---
title: Dome-DETR
families:
  - domedetr
seo_title: Dome-DETR：在 LibreYOLO 里做微小目标检测
description: 在 LibreYOLO 里用 Dome-DETR 对航拍和无人机图像做微小目标检测。转换上游权重，然后在采用 MIT 许可的代码下预测、微调和验证。
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
last_verified: 1.5.0
snippets:
  predict:
    - label: 先转换，再预测
      language: bash
      code: |
        # LibreYOLO 不托管任何 Dome-DETR 权重，所以检查点要从上游仓库拉取，
        # 再转换一次
        hf download RicePasteM/Dome-DETR --include 'best_ckpts_dome_2026/*' \
          --local-dir dome-ckpts

        python weights/convert_domedetr_weights.py \
          dome-ckpts/best_ckpts_dome_2026/dome-s-visdrone_converted.pth \
          LibreDOMEDETRs-visdrone.pt --size s --variant visdrone
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 这里是本地路径，不是裸名称：这个家族不会下载任何东西
        model = LibreYOLO("LibreDOMEDETRs-visdrone.pt")
        result = model("drone-frame.jpg", save=True)

        for box in result.boxes:
            print(result.names[int(box.cls)], box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDOMEDETRs-visdrone.pt
        source=drone-frame.jpg save=True
    - label: 类别名称
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 没有 COCO 检查点，所以类别来自权重训练时用的那个数据集，
        # 从检查点元数据里读出来
        aitod = LibreYOLO("LibreDOMEDETRs-aitod.pt")
        print(aitod.model.names)     # 9 个 AI-TOD-V2 类别

        visdrone = LibreYOLO("LibreDOMEDETRs-visdrone.pt")
        print(visdrone.model.names)  # 12 个 VisDrone 类别
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
source_hash: 381f01d769e7c420
---

## 安装

Dome-DETR 不需要任何可选 extra。它导入的一切都在基础安装里。

```bash
pip install libreyolo
```

## 预测

没有任何东西会自动下载。LibreYOLO 不托管这些权重，所以流程是：拉取上游的检查点
（checkpoint），转换一次，再按路径加载转换后的文件。[许可证](#licensing)解释了
为什么。

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

LibreYOLO 没有为这个家族发布任何基准测试数据行，因为它根本没有发布可供测试的
检查点。

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

这里没有可列的。LibreYOLO 不发布任何 Dome-DETR 权重，形如
`LibreDOMEDETR<size>-<dataset>.pt` 的名称也没有一个能解析到下载。

上游发布了六个检查点，两个数据集各有 s、m 和 l：AI-TOD-V2 有 9 个类别，VisDrone
有 12 个。没有 COCO 检查点，所以规范的文件名总是带着数据集后缀，类别名称随检查点
元数据一起走，而不是来自家族里的某个常量。直接要一个裸的 `LibreDOMEDETRs.pt` 会
立刻抛出异常，消息里点名两个真实的文件名和转换命令，而不是去尝试一次必然 404 的
下载。

转换由 `weights/convert_domedetr_weights.py` 完成。它会重建 LibreYOLO 的计算图，
把上游的张量加载进去，只要有一个键缺失、多余或者形状不对就拒绝写出任何东西，所以
转换出来的文件要么完全一致，要么根本不存在。把它指向一个上游的 `.pth`，再传入
尺寸和变体：

```bash
python weights/convert_domedetr_weights.py \
    dome-ckpts/best_ckpts_dome_2026/aitod-s-best.pth \
    LibreDOMEDETRs-aitod.pt --size s --variant aitod
```

关于数值保真度，`weights/parity_domedetr.py` 会在全部六个检查点上把这份移植和
上游实现做对比，先逐位检查 MWAS 的窗口掩码，然后在 `pred_logits` 和 `pred_boxes`
上都报告 `max_abs_diff == 0.0`，并单独把每一项损失函数与上游的 criterion 做差。
把这件事说清楚：那是一个手动脚本，需要上游的代码检出和已发布的检查点在磁盘上，
靠人工运行。它不属于持续集成，也没有任何 CI 任务复现它。

## 许可证

<provenance-box>

这个家族没有被镜像，原因出在权重上。上游的模型卡在元数据里没有 license 字段，
正文里既说项目采用 Apache-2.0 许可，又把材料限制为仅供学术研究使用。这两种读法
对不上，而更严格的那一种并不构成再分发许可，所以在澄清之前，LibreYOLO 链接到
上游仓库，而不是复制那些文件。这里 [YOLO-NAS](/docs/models/yolo-nas) 适用的也是
同一套理由。

代码是另一个问题，而且清楚得多。上游仓库采用 Apache-2.0 许可，LibreYOLO 的移植
采用 MIT 许可，你用自己的数据训练出来的权重是你自己的。

</provenance-box>

## 引用

Dome-DETR 发表于 ACM Multimedia 2025，标题是「Dome-DETR: DETR with
Density-Oriented Feature-Query Manipulation for Efficient Tiny Object
Detection」。预印本在 [arxiv.org/abs/2505.05741](https://arxiv.org/abs/2505.05741)。
作者没有在仓库里给出 BibTeX 块，所以这里也不再手工拼一个出来。

<citation-block />
