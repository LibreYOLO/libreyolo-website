---
title: 知识蒸馏
seo_title: LibreYOLO 中的知识蒸馏
description: 用更大的教师模型或冻结的 DINOv2 骨干来训练小检测器：MGD、CWD 和 feature-MSE 损失、抽取点，以及家族支持情况。
lead: >-
  蒸馏会加上第二个损失项，把学生的中间特征图拉向一个冻结教师的特征图。LibreYOLO 用前向 hook 抽取特征，所以教师自己的 head
  和损失从头到尾都不参与。
keywords:
  - 知识蒸馏
  - yolo 知识蒸馏
  - 特征蒸馏
  - mgd 蒸馏损失
  - cwd 通道蒸馏
  - dinov2 教师模型
  - 教师学生蒸馏
  - 蒸馏训练小模型
last_verified: 1.5.0
snippets:
  detector:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 用同一家族里更大的检查点来监督这个小模型
        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            distill_model="LibreYOLO9c.pt",
            distill_loss_type="mgd",
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 distill_model=LibreYOLO9c.pt distill_loss_type=mgd
  foundation:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 用一个冻结的自监督 ViT 来监督某一个骨干阶段
        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            distill_model="dinov2",
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 distill_model=dinov2
  tuned:
    - label: 调整损失
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            distill_model="LibreYOLO9c.pt",
            distill_loss_type="cwd",
            dis=1.0,           # 全局蒸馏权重
            distill_tau=1.0,   # CWD 的 softmax 温度
        )
source_hash: 7210031328f6826f
---

## 从更大的检查点蒸馏

设置 `distill_model` 就会打开蒸馏。这个值是一个教师检查点（checkpoint），和其他模型
一样通过同一个工厂加载。

<code-tabs name="detector" />

教师在 `no_grad` 下前向，开启 AMP 时还会在 autocast 下前向，这样冻结的模型不必在每一步
都付出全精度计算的开销。前向 hook 在命名的抽取点上捕获它的特征图，损失把这些特征图和
学生的做比较，结果加进训练损失，并作为一个名为 `distill` 的分量上报。

## 从冻结的基础模型骨干蒸馏

也可以换成用一个自监督 ViT 来监督学生骨干的单个阶段。教师的特征来自它自己的特征提取器
而不是 hook，损失会处理 patch 网格和卷积步幅之间的不匹配。

<code-tabs name="foundation" />

`distill_model` 认得 `dinov2`（也就是 DINOv2-base），以及 `dinov2_vits14`、
`dinov2_vitb14`、`dinov2_vitl14`、`dinov2-small`、`dinov2-base`、`dinov2-large`，
还有任何以 `facebook/dinov2` 开头的原始 hub id。其他任何值都会被当成教师检查点的路径。

这条路径无论 `distill_loss_type` 是什么都使用 `feat_mse`，并且需要装好 `transformers`。
加载时缺少权重键的教师会直接中止，而不是拿一个部分随机的骨干去蒸馏。

## 哪些家族支持

蒸馏支持体现为学生模型上的一个方法，一共有两个。

`get_distill_config()` 给出检测器教师所监督的多尺度抽取点。YOLOv9、YOLOX 和 RF-DETR
实现了它。

`get_backbone_distill_config()` 给出基础模型教师所监督的那一个骨干阶段。YOLOv9 实现了
它，而且是唯一实现了它的家族。

其他家族会直接报错，而不是在没有这个损失的情况下训练：

```text
LibreDFINE does not implement get_distill_config(). Distillation is not yet
supported for the 'dfine' family.
```

```text
Foundation-model distillation into the 'yolox' family is not supported yet
(no get_backbone_distill_config()).
```

## 抽取点

抽取点按家族和角色固定，所以教师和学生不必是同一种架构；它们需要的是相同的特征步幅。

| 家族 | 角色 | 抽取点 | 步幅 |
|---|---|---|---|
| YOLOv9 | 教师或学生 | `neck.elan_up2`、`neck.elan_down1`、`neck.elan_down2` | 8、16、32 |
| YOLOv9 | 基础模型学生 | `backbone.elan3` | 16 |
| YOLOX | 教师或学生 | `backbone.C3_p3`、`backbone.C3_n3`、`backbone.C3_n4` | 8、16、32 |
| RF-DETR | 教师或学生 | `model.backbone.0.projector.stages.0` | 初始化时探测 |

步幅不匹配会在训练开始前报错：

```text
Teacher and student must have matching strides. Teacher: [8, 16, 32],
Student: [16]
```

基础模型教师会跳过这项检查，因为它存在的意义本来就是两边网格不一样。

## 三种损失

`distill_loss_type` 为检测器教师选择特征损失。基础模型教师始终使用 `feat_mse`。

`mgd`，即掩码生成式蒸馏（masked generative distillation），会遮掉学生一部分空间位置，
并训练一个两层卷积的小生成器，从剩下的部分重建教师完整的特征图。`distill_mask_ratio`
设置被遮掉的比例，默认 0.65。

`cwd`，即通道级蒸馏（channel-wise distillation），把每个通道的空间激活变成一个概率
分布，然后逐通道最小化 KL 散度。`distill_tau` 是 softmax 温度，默认 1.0。

`feat_mse` 用一个 1x1 卷积把学生的通道对齐到教师的通道，用双线性插值把教师的网格缩放到
学生的尺寸，再取均方误差。`distill_normalize=True` 会先在通道维度上对两边的特征图做 L2
归一化，让匹配只看角度、与尺度无关。它默认为 `False`。

`dis` 是在此之上施加的全局权重。不设置时，每种损失使用各自论文给出的默认值：MGD 是
2e-5，CWD 是 1.0，特征 MSE 是 1.0。它们相差五个数量级，所以为某一种损失类型调好的权重
换到另一种上就毫无意义。

<code-tabs name="tuned" />

`distill_mask_ratio`、`distill_tau` 和 `distill_normalize` 没有 CLI 标志。它们是
Python 参数，或者 `cfg=` YAML 里的键。RF-DETR 的蒸馏整体上也只能用 Python，因为它的
CLI 参数映射没有带上蒸馏相关的键。

## 适配器、检查点与多卡训练

每种损失都会构建一些活在学生之外的小型可训练模块：1x1 通道适配器，以及 MGD 的生成器。
它们在优化器里有自己的参数组，用的是这次运行的实际学习率。

这些模块会以 `distiller` 键写进检查点，并在断点续训时恢复，所以续训的运行不会从零重新
开始训练它的投影层。

在 DDP 下，适配器位于被包装的学生之外，这意味着 DDP 的 reducer 永远看不到它们的梯度。
训练器每一步都显式地对它们做 all-reduce，所以每个 rank 训练的是同一份适配器。

蒸馏运行上无法使用 CUDA 图捕获。传入 `cuda_graph=True` 会打一行日志，然后以 eager 模式
训练。见[训练性能](/docs/train/performance)。

## 相关

- [层冻结](/docs/train/layer-freezing) 和
  [LoRA 微调](/docs/train/lora)，两者都没有被禁止和蒸馏一起使用。
- [超参数](/docs/train/hyperparameters) 介绍 `train()` 的其余部分。
