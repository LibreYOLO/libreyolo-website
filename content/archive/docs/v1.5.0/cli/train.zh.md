---
title: libreyolo train
seo_title: libreyolo train 命令参考
description: 从命令行训练模型：全部 59 个参数及其默认值、家族默认值如何覆盖它们，以及哪些参数会被某个家族忽略。
lead: >-
  在一个数据集上训练一个模型，并把检查点（checkpoint）、指标和日志写进运行目录。下面每个参数都有一个来自命令定义的默认值，而模型家族自己的训练配置可能会替换它。
keywords:
  - libreyolo train 命令
  - libreyolo 训练参数
  - yolo 命令行训练
  - yolo 训练自己的数据集
  - libreyolo dry_run
  - yolo 冻结层
last_verified: 1.5.0
meta:
  - label: 命令
    value: libreyolo train
    mono: true
  - label: 必填
    value: data
    mono: true
  - label: 输出
    value: 检查点、指标和日志，位于 runs/train/exp
snippets:
  examples:
    - label: 基本用法
      language: bash
      code: >
        # coco8.yaml 随包一起分发，首次使用时会下载它的 8 张图片

        libreyolo train model=LibreYOLO9s.pt data=coco8.yaml epochs=10 imgsz=640
        batch=8
    - label: 先查看解析后的配置
      language: bash
      code: >
        # 打印这次运行会使用的配置，包括家族默认值，然后退出，

        # 既不训练也不加载数据

        libreyolo train model=LibreDFINEn.pt data=coco8.yaml epochs=10
        dry_run=true
    - label: 命名运行，显式指定超参数
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=coco8.yaml \
          epochs=50 batch=8 optimizer=adamw lr0=0.001 weight_decay=0.0001 \
          patience=20 save_period=5 project=runs/train name=yolo9s-coco8 exist_ok=true
source_hash: 3aad4298310d3081
---

## 概要

```bash
libreyolo train data=<dataset.yaml> [model=<name|path>] [key=value ...]
```

参数是 `key=value` 形式的键值对，POSIX 形式同样可用，所以 `epochs=50` 和
`--epochs 50` 是同一个参数。布尔值接受 `true` 和 `false`：对于带否定形式的
标志，`amp=false` 会变成 `--no-amp`。

## 参数

### 模型与数据

| 参数 | 默认值 | 含义 |
|---|---|---|
| `data` | | 数据集 YAML 的路径（YOLO 格式，例如 `coco8.yaml`）。必填 |
| `model` | `yolox-s` | 模型名称或权重路径 |
| `task` | | 显式覆盖任务：`detect`、`segment`、`semantic`、`pose`、`classify`、`gaze`、`obb`、`point`、`depth` |
| `pretrained` | `true` | 使用预训练权重。`false` 会构建架构并从头训练 |
| `allow_download_scripts` | `false` | 允许数据集 YAML 下载块中内嵌的 Python 代码 |

### 训练循环

| 参数 | 默认值 | 含义 |
|---|---|---|
| `epochs` | `300` | 训练轮数 |
| `batch` | `16` | 每个设备的批大小 |
| `imgsz` | `640` | 训练图像尺寸：`640`（正方形）或 `480x640`（高×宽） |
| `device` | `auto` | 设备：`0`、`cpu`、`mps`、`auto` |
| `workers` | `4` | 数据加载器的工作进程数 |
| `cache` | `false` | 缓存图像以加快数据加载：`ram`、`disk`、`true`、`false` |
| `seed` | `0` | 随机种子 |
| `resume` | | 断点续训：`true`，或指向某个检查点的路径 |
| `amp` | `true` | 自动混合精度 |
| `amp_dtype` | `float16` | CUDA AMP 的 dtype：`float16` 或 `bfloat16` |
| `cuda_graph` | `false` | 把训练的前向和反向捕获进 CUDA 图。仅限单 GPU 和受支持的家族，其余家族按 eager 模式运行 |
| `lora` | `false` | LoRA 微调，适用于「注意事项」中列出的 transformer 家族 |
| `freeze` | | 冻结层：一个整数数量、一个索引列表，或模块名 |

### 蒸馏

| 参数 | 默认值 | 含义 |
|---|---|---|
| `distill_model` | | 教师：一个检测器检查点，或用于骨干特征蒸馏的基础教师 id，例如 `dinov2` |
| `dis` | | 蒸馏损失权重。未设置时采用该损失类型公开的默认值 |
| `distill_loss_type` | `mgd` | 检测器教师使用的特征损失：`mgd`、`cwd`。基础教师始终使用 `feat_mse` |

### 优化器

| 参数 | 默认值 | 含义 |
|---|---|---|
| `optimizer` | `sgd` | 优化器：`sgd`、`adam`、`adamw` |
| `lr0` | `0.01` | 初始学习率 |
| `momentum` | `0.937` | SGD 动量，以及 Adam 系优化器的一阶矩系数 |
| `weight_decay` | `0.0005` | L2 正则化 |
| `nesterov` | `true` | Nesterov 动量 |

### 调度器

| 参数 | 默认值 | 含义 |
|---|---|---|
| `scheduler` | `yoloxwarmcos` | 学习率调度类型 |
| `warmup_epochs` | `5` | 预热时长 |
| `warmup_lr_start` | `0.0` | 预热的初始学习率 |
| `min_lr_ratio` | `0.05` | 最小学习率比例 |
| `lr_drop` | `100` | RF-DETR 阶梯式学习率下降的轮次 |

### 数据增强

| 参数 | 默认值 | 含义 |
|---|---|---|
| `mosaic` | `1.0` | Mosaic 概率 |
| `mixup` | `1.0` | Mixup 概率 |
| `hsv_prob` | `1.0` | HSV 抖动概率 |
| `flip_prob` | `0.5` | 水平翻转概率 |
| `degrees` | `10.0` | 旋转范围，正负两个方向，单位为度 |
| `translate` | `0.1` | 平移比例 |
| `shear` | `2.0` | 错切角度 |
| `mosaic_scale` | `(0.1,2.0)` | Mosaic 缩放范围 |
| `mixup_scale` | `(0.5,1.5)` | Mixup 缩放范围 |
| `no_aug_epochs` | `15` | 最后 N 轮关闭数据增强 |

### EMA

| 参数 | 默认值 | 含义 |
|---|---|---|
| `ema` | `true` | 指数移动平均 |
| `ema_decay` | `0.9998` | EMA 衰减系数 |

### 训练中的验证

| 参数 | 默认值 | 含义 |
|---|---|---|
| `val` | `true` | 训练过程中做验证 |
| `eval_interval` | `10` | 每 N 轮验证一次 |
| `max_det` | `300` | 验证 NMS 之后每张图像的最大预测数 |
| `eval_max_det` | | COCO 评估器的上限。未设置时采用 pycocotools 的 AP@100 惯例 |
| `faster_coco_eval` | `true` | 装有 faster-coco-eval 时，用它的 C++ 后端计算 COCO 指标；否则回退到 pycocotools |
| `save_plots` | `false` | 训练过程中保存最终的验证图表 |
| `patience` | `50` | 早停的耐心值。`0` 表示禁用 |

### 输出

| 参数 | 默认值 | 含义 |
|---|---|---|
| `project` | `runs/train` | 输出目录的根路径 |
| `name` | `exp` | 实验名称 |
| `exist_ok` | `false` | 复用已存在的输出目录 |
| `save_period` | `10` | 每 N 轮保存一次检查点 |
| `log_interval` | `10` | 每 N 个批次记录一次损失 |

### agent 标志

| 参数 | 默认值 | 含义 |
|---|---|---|
| `json` | `false` | 向 stdout 输出 JSON |
| `quiet` | `false` | 抑制 stderr |
| `dry_run` | `false` | 解析并打印配置，但不执行 |
| `help_json` | `false` | 把命令 schema 以 JSON 形式打印出来并退出 |

## 示例

<code-tabs name="examples" />

## 注意事项

### 上面的默认值不一定就是实际用到的值

每个模型家族都带着自己的训练配置，只要该配置与基础配置不同，对于你没有显式
设置的参数，它的值就会替换命令的默认值。你自己设置的参数始终优先。
`libreyolo cfg` 会打印基础默认值和各家族的覆盖值，这是查看某个家族实际会用
什么的办法。

`imgsz` 是这件事影响最大的参数。命令默认值是 `640`，而它并不是每个检查点的
原生输入：RF-DETR 公开的检测尺寸是 384、512、576 和 704，YOLOX 的 `n` 和
`t` 检查点是 416。RF-DETR 和 DEIMv2 的处理方式是只在显式设置时才转发
`imgsz`，否则它们自己的尺寸继续生效。其他家族会按给定值拿到它，并以该尺寸
训练。FOMO 是最严格的一个：每种尺寸只接受它自己的原生输入（96、192 和
224），所以一次 FOMO 训练需要把 `imgsz` 设成匹配的值，否则会报错停止。
RF-DETR 还要求这个值能被它的 patch 尺寸乘以窗口数量整除，不满足时会报出最
接近的两个合法尺寸。

### 家族会忽略的参数

并不是每个家族都会读取每个参数，数据增强这一组就是体现得最明显的地方。
RF-DETR、D-FINE、DEIM、DEIMv2、RT-DETRv4 和 DINOv2 通过直通式流水线训练，
没有 mosaic、没有 mixup、也没有仿射变换，所以 `mosaic`、`mixup`、
`hsv_prob`、`degrees`、`translate`、`shear`、`mosaic_scale` 和
`mixup_scale` 在那里落不到实处。EC 共用同一条流水线，但当它的任务是姿态时
确实会读取 `hsv_prob`、`degrees` 和 `translate`。分类家族、SegFormer 和
NAFNet 会忽略整组参数，连 `flip_prob` 一起，因为它们的翻转按固定概率运行，
而不是一个可配置的概率。YOLO-NAS 只忽略 `mosaic`，因为它改用一直开启的逐
样本仿射变换来做增强。RF-DETR 在这份清单之外还要再忽略三个：`optimizer`、
`momentum` 和 `nesterov`。

设置其中某一个并不是错误。运行时会向 stderr 打印一行日志，写明家族以及它将
忽略的参数，然后开始训练，那一行就是所安装版本的权威清单。它也是唯一的信
号，所以带 `quiet=true` 的脚本化运行会连同 stderr 上的其他内容一起把这条警
告压掉。

`val=false` 是与此相关的一种情况。对大多数家族来说，它会把 `eval_interval`
设为 `0`；RF-DETR 没法用这种方式关掉验证，只会记录一条日志说明它忽略了这个
请求。

### 其他值得了解的行为

`lora=true` 会被 RF-DETR、D-FINE、DEIM、DEIMv2、RT-DETR v1、v2 和 v4、EC 以
及 ConvNeXt 接受。其他任何家族都会以 `config_unsupported` 退出，而不是在没
有它的情况下训练。

`pretrained=false` 与 `resume` 同时使用，在支持从头训练的家族上会被拒绝，因
为这两者要求的是相反的事情。

`mosaic` 和 `mixup` 是配置字段 `mosaic_prob` 和 `mixup_prob` 的命令行写法。
在 mixup 只作用于 mosaic 样本的家族上，`mixup` 大于零而 `mosaic` 为零时永远
不会触发，运行时也会这样提示。

`dry_run=true` 会解析模型引用、应用家族默认值，并打印它将用来训练的配置。它
不会加载数据集，所以这是确认某个参数是否取到你预期值的廉价办法。

stdout 承载最终的结果对象；进度和警告走 stderr。退出码在成功时是 `0`，用法
或配置错误是 `2`，找不到或读不了数据集是 `3`，模型加载不了是 `4`，其他运行
时故障是 `1`。

相关：[`libreyolo doctor`](/docs/cli/doctor) 用于在正式开跑之前检查数据集，
[`libreyolo monitor`](/docs/cli/monitor) 用于在浏览器里观察一次运行，
[`libreyolo val`](/docs/cli/val) 用于衡量结果。
