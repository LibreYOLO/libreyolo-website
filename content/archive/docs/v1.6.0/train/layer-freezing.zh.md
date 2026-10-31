---
title: 层冻结
seo_title: 在 LibreYOLO 中训练时冻结层
description: 为迁移学习冻结模型的一部分：家族冻结分组的整数个数、显式的索引列表，或者模块名和参数名选择器。
lead: 冻结让选中的权重保持不变，模型的其余部分照常训练。选择器寻址的是家族自己那份有序的冻结分组或它的模块名，而不是 YAML 计算图里的原始层号。
keywords:
  - yolo 冻结层
  - 迁移学习 微调
  - 冻结骨干
  - batchnorm 冻结
  - 冻结分组
  - 只训练 head
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # 前 10 个分组就是整个 YOLOv9 骨干
        model.train(data="my-dataset.yaml", epochs=50, freeze=10)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=50 freeze=10
    - label: 按名称
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.train(data="my-dataset.yaml", epochs=50, freeze="backbone")
    - label: 多个选择器
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", freeze=["backbone", "neck"])
  groups:
    - label: 按顺序列出家族的冻结分组
      language: python
      code: |
        from libreyolo import LibreYOLO9
        from libreyolo.models.yolo9.trainer import YOLO9Trainer

        model = LibreYOLO9("LibreYOLO9s.pt", size="s")
        trainer = YOLO9Trainer(model=model.model, wrapper_model=model, size="s")

        for index, (name, _module) in enumerate(trainer.get_freeze_groups()):
            print(index, name)
source_hash: 9f1e7551af6b16fe
---

## 冻结一部分

`freeze` 是可选的，默认不冻结任何东西。

<code-tabs name="train" />

冻结发生在模型构建之后、为新的类别数重建 head 之后，并且在优化器创建之前，所以优化器拿到的始终只有可训练参数。

## 选择器可以是什么

| 取值 | 含义 |
|---|---|
| `None`、`False`、`""`、`"none"` | 训练所有参数 |
| `10` 或 `"10"` | 冻结家族的前十个冻结分组 |
| `[0, 3, 7]` | 冻结这些从 0 开始编号的分组 |
| `"backbone"` | 冻结匹配的分组、模块或参数名前缀 |
| `["backbone", "neck"]` | 冻结列出的每一个选择器 |
| `["backbone", 3]` | 混合列表也可以 |

字符串会先解析再解释，所以 CLI 和 YAML 配置文件接受和 Python 一样的形式。`freeze="[0, 3, 'head']"` 会被解析成字面量列表，`freeze="backbone,neck"` 按逗号切分，而纯十进制数字字符串则变成一个计数。

`freeze=True` 因为含义不明确而被拒绝。

名称选择器会匹配冻结分组名、模块名或参数名前缀，通配符 `*`、`?` 和 `[` 都可用。开头的 `model.` 会被灵活处理，所以 `backbone` 和 `model.backbone` 都能命中家族内部实际使用的那种写法。

## 分组由家族自己定义

整数寻址的是家族自己那份有序的冻结分组列表，而不是某个共享计算图里的位置。LibreYOLO 的家族并不都是同一种按 YAML 索引的顺序模型，所以原始层号在每个家族上都会指向不同的东西。

YOLOv9 从输入侧开始排列它的分组：十个骨干阶段，然后是六个 neck 阶段，最后是 head。这就是 `freeze=10` 恰好等于整个骨干的原因。`backbone`、`neck` 和 `head` 是建立在它之上的稳定名称选择器。

RF-DETR 的分组是 `backbone.encoder`、`backbone.projector`、`decoder`、`queries`、`transformer.encoder_output` 和 `head`。这里用名称更合适，因为 transformer 组件并不能对应到一个层数。`backbone` 会按前缀同时匹配两个骨干分组。

没有定义语义分组的家族会回退到一个保守的默认值：模型的每个直接子模块，只要它至少拥有一个参数，按声明顺序排列。这通常是一个很短的列表，所以较大的整数会找不到足够多的分组：

```text
freeze index 10 is out of range for 3 available freeze groups.
```

想看到真实的列表而不是靠猜：

<code-tabs name="groups" />

## 失败会大声报错

每一种用错的方式都会直接报错，而不是去训练一个你没有要求的东西。

匹配不到任何东西的选择器会报错，并指出是哪些选择器没有命中：

```text
freeze selector(s) matched no parameters: 'backbon'
```

会导致没有任何参数可训练的冻结同样会报错，在冻结时报一次，在构建优化器时再报一次：

```text
freeze would leave no trainable parameters. Use a smaller freeze value or
target a narrower module.
```

`freeze="all"` 就是这种情况，因为 `all` 会匹配到每一个参数。

冻结成功时，会有一行记录发生了什么：

```text
Layer freezing: selectors=[10], tensors=124, params=2103776, trainable=1863456/3967232
```

## 冻结的 BatchNorm 不再更新

被冻结的参数仍然位于某个模块内部，而这个模块的滑动统计量本来还会继续变动。凡是参数落入冻结集合的 BatchNorm 类模块都会被切换到 eval 模式，训练器还会在每一轮的 `model.train()` 调用之后重新施加这一设置，所以统计量在整个训练过程中保持固定。

这项行为默认开启，也正是它让冻结骨干这件事真正把骨干冻住。

## 与 LoRA 组合

`freeze` 和 `lora=True` 可以一起用。在 RF-DETR、DEIM 和 ConvNeXt 上，即使适配器参数所在的父分组被冻结，它们也会被保留为可训练——这正是你想要的组合：一个冻结的骨干，加上在它之上学习的适配器。参见 [LoRA 微调](/docs/train/lora)。

## 范围

这里说的是在启动时就决定好的静态冻结。按计划解冻和渐进式冻结不属于这个接口。

## 相关

- [超参数](/docs/train/hyperparameters) 介绍 `train()` 的其余部分。
- [蒸馏](/docs/train/distillation) 介绍把一个大模型的知识搬进训练过程的另一种方式。
