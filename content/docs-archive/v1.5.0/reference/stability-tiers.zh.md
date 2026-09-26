---
title: 稳定性层级
seo_title: LibreYOLO 的各个支持层级分别意味着什么
description: LibreYOLO 使用的层级词汇：三种导出支持层级、四种 API 层级、六个覆盖分组，以及它们都不承诺的事情。
lead: >-
  LibreYOLO
  用「层级」这个词指三件不同的事：一条导出路径背后的证据、一个模型家族遵循的调用契约，以及这个家族被编入的覆盖分组。本页逐一给出定义，并说明它们不意味着什么。
keywords:
  - libreyolo 支持层级
  - validated available blocked
  - 导出支持层级
  - libreyolo 覆盖分组
  - g0 g1 g2 g3 g4
  - 模型分层
last_verified: 1.5.0
verification: >-
  导出层级读自 docs/adr/0011-export-support-tiers.md 与
  libreyolo/export/support.py；覆盖分组与各家族数量读自 libreyolo/models/registry.py 里的
  MODEL_GROUPS；从零训练的判断读自 libreyolo/models/base/model.py 与
  libreyolo/cli/commands/train.py；CLI 清单读自 libreyolo/models/inventory.py；API
  层级读自 libreyolo/models/sam/、openvocab/ 与 vlm/ 的包 docstring 以及 base.py 里的契约，全部基于
  v1.5.0。面向读者的分组名称（旗舰、核心、已支持、仅推理、博物馆、兄弟层级）是本站为同一批分组使用的说法，来自
  src/data/docs/registry.json。
snippets:
  usage:
    - label: 读取同一个家族的两种分类
      language: python
      code: |
        from libreyolo.models.registry import GROUPS, group_of
        from libreyolo.export.support import get_support, validated_alternatives

        family = "yolo9"

        group = group_of(family)
        print(group, GROUPS[group])

        print(get_support(family, "detect", "onnx").tier)
        print(validated_alternatives(family, "detect"))
source_hash: de545894b0d125e4
---

## 导出支持层级

决定一次调用能否成功的层级。它作用于 `(family, task, format)` 这个三元组，
每一种组合都恰好属于其中一个。

| 层级 | 含义 | 调用 `export()` 时会发生什么 |
|---|---|---|
| `validated` | 数值一致性由 CI 或有记录的 nightly 运行覆盖 | 正常运行 |
| `available` | 转换已经实现，但尚未记录运行时的数值一致性证据 | 正常运行 |
| `blocked` | 没有受支持的路径 | 在预检阶段抛出 `NotImplementedError`，并给出原因 |

validated 和 available 都会直接执行，不需要确认，也不会给出笼统的警告。区别
在于证据，而不是权限：validated 条目背后有一项一致性测试和一个 `since` 版本，
available 条目还没有。举例来说，没有在 macOS 上跑过一次预测的 CoreML 转换就是
available，而不是 validated。

blocked 的组合会在依赖检查、校准数据加载、tracing 和产物生成之前就失败，所以
不会写出任何半成品。

每个 validated 单元格都带一条约束，说明这个一致性数字出自什么配置，通常是固定
的输入画布、batch 1、FP32 和一个具名的运行时版本。把它读作针对那套配置的结论，
而不是针对这个格式本身。没有显式条目的单元格按什么规则填充，见[导出矩阵](/docs/reference/export-matrix)页面。

<code-tabs name="usage" />

## API 层级

决定一次调用长什么样的层级。每个家族恰好属于其中一个，划分依据是调用契约而不
是架构。

| 层级 | 工厂 | 契约 |
|---|---|---|
| 检测器工厂 | `LibreYOLO` | 一次无提示的前向传播就返回它找到的所有目标，并带有校准过的分数。成员通过识别检查点（checkpoint）来自行注册 |
| 可提示分割 | `LibreSAM` | 如果调用时不为每张图像提供空间提示或概念提示，一次前向传播没有意义。交互式且有状态：编码一次，可以提示很多次 |
| 开放词汇检测 | `LibreOpenVocab` | 文本条件的判别式检测器。类别列表是一种提示，由 `set_classes` 设置 |
| 视觉语言 | `LibreVLM` | 当作检测器来驱动的生成式模型。类别列表是一种提示，置信度只是占位符 |

这三个兄弟层级刻意不注册进检测器工厂，所以 `LibreYOLO("some-alias")` 到不了
它们。它们靠尺寸别名和自动下载来加载，而不是靠嗅探检查点。

四者都返回同样的 `Results`，所以下游代码在它们之间不用改。差别在于哪些方法能
用：兄弟层级的 `train()`、`val()` 和 `export()` 会抛出 `NotImplementedError`，
SAM 和开放词汇层级的 `track()` 也会抛。每个层级的页面都会列出自己排除了哪些。

## 覆盖分组

决定一次跨家族测试要跑哪些家族的分类，也是读者在模型页面上最可能碰到的那一
种。每个注册家族都恰好编入一个分组，只要有注册家族没被编入，测试就会失败。下
表「含义」列来自 `libreyolo/models/registry.py` 里的 `GROUPS`；同一个文件里的
`MODEL_GROUPS` 给每个家族做分配，「家族数」列直接数的就是这份分配。「名称」列
是本站在模型页面标题处对同一个分组使用的简称。

| 分组 | 名称 | 家族数 | 含义 |
|---|---|---|---|
| `g0` | 旗舰 | 2 | 共享特性覆盖中必须包含的旗舰标杆 |
| `g1` | 核心 | 10 | 可训练检测器的覆盖集合 |
| `g2` | 已支持 | 14 | 额外的可训练家族覆盖集合 |
| `g3` | 仅推理 | 35 | 没有训练实现的家族 |
| `g4` | 博物馆 | 5 | 有推理覆盖的历史家族 |
| `s` | 兄弟层级 | 21 | 单独覆盖的兄弟 API（SAM、开放词汇、VLM、零样本） |

一共是六个分组、87 个家族。光 `g3` 一个分组的家族数就超过其他所有分组之和，
因为注册表（registry）里大部分是仅推理的谱系和博物馆式的覆盖，而不是持续训练
的检测器。

对正在挑模型的读者来说，分组说明的是工程投入落在哪里，而不是一个家族有多精
确。`g0` 和 `g1` 是新特性设计出来、并且最先落地的地方；`g2` 在 CI 里保持绿
色，但特性是有机会才落到那里，不会跟着同一波发布。`g3` 陈述的是一种缺失，而
不是一种限制：预测、验证，以及在家族支持的情况下导出，都仍然可用；对 `g3` 或
`g4` 家族调用 `train()` 会抛出 `NotImplementedError` 并说明原因，而不是默默做
一半。`s` 家族完全不在这个取舍里，因为它们通过自己的工厂加载，而不是
`LibreYOLO()`。读检查点文件名时分组与任务、家族、尺寸怎么配合，见[核心概念](/docs/concepts)。

分组本身不会赋予或限制任何面向用户的能力。支持来自家族已实现的 API，以及针对
具体格式的能力检查，绝不会只凭分组归属。分组划分的是家族而不是任务，所以限定
了任务的覆盖运行会显式写出任务名，比如「g1 detect」。

有两个地方会在运行时读分组，而不只是在测试里读。`libreyolo/models/inventory.py`
里的 `collect_model_inventory()` 会把分组附到 CLI 清单打印的每一条记录上；而
`pretrained=False` 只对 `g0` 和 `g1` 里的家族触发那条特殊的从零重新初始化路
径。在这两个分组之外，`libreyolo/models/base/model.py` 里的检查会被整个跳过，
于是 `pretrained=False` 会作为一个普通关键字参数传到家族自己的 `train()` 里。

## 训练

`g3` 或 `g4` 里的家族没有训练实现，对它们调用 `train()` 会抛异常。这是家族代
码的性质，不是分组的性质：分组只是记录这个事实，而不是造成它。

对于确实能训练的家族，某个数据增强参数会不会真的到达流水线是另一个问题，它有
自己的一套三值词汇：`used`、`gated_by_mosaic` 和 `ignored`。见[数据增强矩阵](/docs/reference/augmentation-matrix)。

## 层级没有告诉你什么

层级不是关于精度的结论。一条 validated 的导出说明产物在给定阈值内复现了原生模
型；它没有说原生模型在某个数据集上分数如何。基准测试的数字在模型页面上。

层级也不是关于许可证的说明。同一个家族内部的权重许可各不相同，托管某个具体检
查点的仓库才是权威。一个家族在检测器工厂里，并不说明它发布的权重是否允许商用。
