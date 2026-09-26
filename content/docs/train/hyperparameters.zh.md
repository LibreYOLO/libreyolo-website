---
title: 超参数
seo_title: LibreYOLO 中的训练超参数
description: >-
  train() 里真正要紧的参数：epochs、batch、lr0、optimizer、EMA、autobatch、梯度累积和
  resume，以及为什么每个家族的默认值不一样。
lead: >-
  每个训练参数都是 TrainConfig dataclass
  上的一个字段。基类定义字段和它的默认值；每个模型家族继承这个基类，并覆盖自己已发布配方所改动的那些默认值。
keywords:
  - yolo 训练参数
  - yolo 学习率设置
  - 批大小 batch
  - autobatch 自动批大小
  - 指数移动平均 ema
  - 梯度累积
  - yolo 断点续训
  - 早停 patience
  - amp bfloat16 混合精度
  - 训练配置 yaml
last_verified: "1.6.0"
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        results = model.train(
            data="my-dataset.yaml",
            epochs=100,
            batch=16,
            imgsz=640,
            lr0=0.01,
        )

        print(results["best_mAP50_95"])
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 batch=16 imgsz=640 lr0=0.01
  defaults:
    - label: 读出某个家族解析后的默认值
      language: python
      code: |
        from dataclasses import fields

        from libreyolo import LibreYOLO9
        from libreyolo.training.config import TrainConfig

        family_cfg = LibreYOLO9.TRAIN_CONFIG()
        base_cfg = TrainConfig()

        for f in fields(family_cfg):
            family_value = getattr(family_cfg, f.name)
            base_value = getattr(base_cfg, f.name, None)
            if not hasattr(base_cfg, f.name) or family_value != base_value:
                print(f"{f.name}: {family_value}")
    - label: CLI
      language: bash
      code: |
        # 打印 train、val 和 predict 的默认值，包括家族的覆盖值
        libreyolo cfg
  autobatch:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # batch=-1 会探测 GPU 显存，并解析成一个具体的 2 的幂
        model.train(data="my-dataset.yaml", batch=-1, imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml batch=-1
  accumulate:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # 每个优化器步 4 个大小为 16 的微批，有效批大小 64
        model.train(data="my-dataset.yaml", batch=16, nbs=64)
  resume:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 先加载被中断那次运行的检查点，再要求断点续训
        model = LibreYOLO("runs/train/exp/weights/last.pt")
        model.train(data="my-dataset.yaml", epochs=100, resume=True)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=runs/train/exp/weights/last.pt \
          data=my-dataset.yaml epochs=100 resume=true
  cfg:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # yaml 里的键就是 TrainConfig 的字段名，显式传入的 kwargs 优先
        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", cfg="my-recipe.yaml", epochs=50)
source_hash: eac4e55fcf16ca15
---

## 设置参数

`train()` 接受关键字参数，CLI 用同样的名字，写成 `key=value` 的形式。

<code-tabs name="train" />

两条路径最后落在同一个地方。这些 kwargs 会交给 `TrainConfig.from_kwargs()`，由它构建出该家族的配置 dataclass。

## 拼错不会报错

`from_kwargs()` 会丢掉任何不是配置字段的键，并发出一条点名它的 `UserWarning`。训练随后就带着原来的默认值开始跑：

```python
# UserWarning: Unknown training config keys (ignored): ['learning_rate']
model.train(data="my-dataset.yaml", learning_rate=0.001)
```

什么都不会失败，这次运行会跑完，而学习率从来就不是调用方要的那个。新配方的第一轮要把警告读一遍。CLI 更严格，因为它在配置构建之前就校验标志名，所以拼错的 CLI 标志会被直接拒绝。

## 默认值按家族区分

`TrainConfig` 定义字段和一个基础默认值。每个家族继承它，并覆盖自己已发布配方所改动的部分，所以「默认学习率是多少」并没有唯一正确的答案。

基础默认值是 `optimizer="sgd"`、`lr0=0.01`、`momentum=0.937`、`weight_decay=5e-4`、`scheduler="yoloxwarmcos"`、`epochs=300`、`batch=16`、`imgsz=640` 和 `amp=True`。下面三个例子说明一个家族能离它多远：

| 字段 | 基础 | YOLOv9 | D-FINE | YOLO-NAS |
|---|---|---|---|---|
| `optimizer` | `sgd` | `sgd` | `adamw` | `adamw` |
| `lr0` | `0.01` | `0.01` | `2e-4` | `5e-4` |
| `weight_decay` | `5e-4` | `5e-4` | `1e-4` | `1e-5` |
| `scheduler` | `yoloxwarmcos` | `linear` | `flat_cosine` | `cos` |
| `epochs` | `300` | `300` | `132` | `300` |
| `amp` | `True` | `True` | `True` | `True` |

D-FINE、DEIM、RT-DETRv4 和 YOLO-NAS 检测默认使用 `amp=True` 和 `amp_dtype="float16"`。Dome-DETR、PP-YOLOE 和 YOLO-NAS OBB 保留 FP32 默认值。需要 FP32 时传入 `amp=False`。

要读到某个家族真实的默认值，而不是靠猜：

<code-tabs name="defaults" />

## 批大小

`batch` 是全局批大小。在多卡训练下，每个 rank 加载 `batch // world_size`，所以你传的数字就是每个优化器步的图片数，跟用了几块 GPU 无关。参见[多卡训练](/docs/train/multi-gpu)。

`batch=-1` 会打开 autobatch。训练器在训练模式下按 2 的幂探测模型，带一次真实的反向传播，对显存曲线拟合一条直线，然后挑出严格小于外推值、且能装进总显存 60% 以内的最大 2 的幂。

<code-tabs name="autobatch" />

在训练模式下带反向传播地探测，正是关键：推理模式的探测会漏掉保留下来的激活值和梯度张量，对一个深层 CNN 来说，它们是推理占用的好几倍。RF-DETR 把目标比例降到 45%，因为探测用的合成反向传播仍然低估了它的损失函数（criterion）和辅助解码器层的开销。

autobatch 是 CUDA 上的功能。在 CPU 或 MPS 上，它只记一行日志，然后保持默认的批大小。

`min_samples=0` 保持每轮长度不变。正值下限会对较短的检测数据集进行有放回采样。`class_balanced=False` 改为 true 时启用按图像的重复因子采样；它可以与 `min_samples` 和 DDP 一起使用。绕过共用采样器的专用加载器会拒绝启用类别平衡。

## 梯度累积

`nbs` 设置名义批大小，也就是有效批大小。训练器每个优化器步累积 `round(nbs / batch)` 个微批（micro-batch）。

<code-tabs name="accumulate" />

保持默认的 `None` 时，累积是关闭的，训练不受影响。

## 学习率与调度

`lr0` 是初始学习率，`optimizer` 接受 `sgd`、`adam` 和 `adamw`。`momentum` 是 SGD 的动量或 Adam 的 beta1，`weight_decay` 是 L2 项，`nesterov` 只对 SGD 生效。

调度的形状由 `scheduler`、`warmup_epochs`、`warmup_lr_start` 和 `min_lr_ratio` 决定。`no_aug_epochs` 设定最后多少轮不带强数据增强，而且有几个调度器也用它来塑造自己的尾段，所以它不只是一个数据增强旋钮。各家族拿它的数据增强那一半做什么，写在[数据增强](/docs/train/augmentations)里。

有些家族会加上自己的学习率旋钮。`backbone_lr_mult` 相对 head 缩放骨干那一组，`clip_max_norm` 设置梯度裁剪，SegFormer 用 `head_lr_mult` 让它的 decode head 以骨干十倍的速率跑。这些都在家族的配置子类上，不在基类上。

## EMA

`ema=True` 会在训练权重之外，再维护一份权重的指数移动平均。除了 FOMO，其他地方都默认开启。

`ema_decay` 是目标衰减率。衰减是慢慢爬上去的，而不是一开始就取目标值：第 `n` 次更新时的有效值是 `ema_decay * (1 - exp(-n / tau))`，`tau` 默认为 2000，所以早期更新更贴近模型，后期更新则把它抹平。家族默认值从 YOLO-NAS pose 上的 `0.997`，到 YOLOX 上的 `0.9998`，再到 YOLOv9 和 DETR 一系上的 `0.9999`。

拿去验证的是 EMA 权重，`best.pt` 和 `last.pt` 里装的也是它。原始的训练权重同样会存下来，放在 `train_model` 键下，这样断点续训接的是训练轨迹，而不是那份平均值。

## 数值精度

`amp=True` 让前向传播在 CUDA autocast 下运行。`amp_dtype` 选择 `float16`（默认）或 `bfloat16`；`fp16` 和 `bf16` 也是能接受的写法。

Float16 需要动态损失缩放，会拿到一个真正在工作的 `GradScaler`。Bfloat16 的指数范围更宽，不需要它，所以它的 scaler 仍会被构造出来，但处于禁用状态，这样优化器路径保持不变。在不支持 bfloat16 的 CUDA 设备上要求 bfloat16，会在初始化阶段直接报错，而不是悄悄降级。

## 输出、检查点与停止

运行结果写到 `project/name`。`project` 在各处都默认为 `runs/train`，但 `name` 属于按家族覆盖的那几项：基础默认值是 `exp`，而 YOLOv9 用 `yolo9_exp`，D-FINE 用 `dfine_exp`。在默认的 `exist_ok=False` 下，已存在的目录会被加上一个递增后缀，而不是被覆盖。

`save_period` 每 N 轮额外写一个 `weights/epoch_<N>.pt`，在此之上，每轮结束会写 `weights/last.pt`，被跟踪的指标每次变好会写 `weights/best.pt`。`eval_interval` 设置验证运行的频率，`patience` 会在连续这么多轮没有提升之后停掉这次运行，`0` 表示关闭早停。

`cache` 把解码后的图片留在内存里（`True` 或 `"ram"`），或者作为 `.npy` 文件放在源文件旁边（`"disk"`），以此加快重复的轮次。缓存读到的内容与重新读取逐位一致。用了 dataloader worker 时，`"disk"` 是两者中更稳妥的那个。

`average_best=0` 禁用检查点平均；正整数 N 会从至多 N 个最佳快照生成 `weights/average.pt`。浮点张量均匀平均，整数缓冲区取自最佳快照。可以启用默认关闭的 `export_check=False`，在 ONNX 导出失败时于第一轮前终止。匹配的原始输出按 `rtol=1e-3`、`atol=1e-4` 比较；布局不兼容时会记录跳过比较。

`precise_bn=0` 禁用最终的 BatchNorm 重新校准。正值限制最终验证前从训练加载器读取的图像数；DDP 下该预算分别应用于每个参与的 rank。冻结的 BatchNorm 保持冻结。[自定义适应度回调](/docs/train/fitness-callbacks)可以决定最佳检查点和耐心值。

## 断点续训

`resume=True` 会接着一次被中断的运行往下跑。检查点必须先加载好，因为 resume 是从模型上读它，而不是从一个单独的参数读。

<code-tabs name="resume" />

断点续训会恢复训练权重、优化器状态、EMA 权重和更新计数、最佳指标的跟踪、`GradScaler` 的缩放系数，以及 PyTorch、CUDA 和 NumPy 的随机状态。它从检查点的轮次加一开始，并把调度快进到那个位置。

有两件事它不做。`resume=True` 不能和 `pretrained` 一起用，那会报错。还有，当检查点的最佳指标键与当前这次运行的不同时，最佳指标的跟踪会带一条警告重置为零，而不是去比较两个含义并不相同的值。

## 把配方写进文件

`cfg=` 会加载一份由 `TrainConfig` 字段名组成的 YAML 映射，并把它合并在显式关键字参数之下，所以 kwarg 永远赢过文件。

<code-tabs name="cfg" />

`size` 和 `num_classes` 会从文件里剥掉，因为模型实例本身已经拥有它们。CLI 上没有 `--cfg` 标志；这个文件路径是一个 Python 参数。

## 相关内容

- [数据集](/docs/train/datasets)，讲 `data=` 接受什么。
- [数据增强](/docs/train/augmentations)，讲数据增强的旋钮以及哪些家族会遵守它们。
- [层冻结](/docs/train/layer-freezing)和 [LoRA](/docs/train/lora)，讲只训练一部分权重。
- [验证与指标](/docs/train/validation)，讲这次运行会报告什么。

## 类别选择和损失加权

对于 YOLO9、RF-DETR、EdgeCrafter、RT-DETR、D-FINE、DEIM、TinyFormer 和 YOLO-NAS 检测，`classes=None` 保留全部数据集类别；传入列表会过滤监督标签，同时保留原始 ID、`nc` 和 `names`。可以将 `single_cls=False` 改为启用，把保留的标签映射到名为 `object` 的类别 0。续训和验证继承保存的设置。模型的 OBB 训练拒绝 `single_cls`。

ResNet、ConvNeXt、ConvNeXt V2、MobileNetV4、EfficientNetV2 和 DINOv2 支持 `cls_pw=0.0`：[0, 1] 范围内的值将每个类别的逆频率提升到该幂次后作为权重，并将均值归一化为 1。`class_weights=False` 启用后改用 `N / (C * n_c)` 加权。它不能与正的 `cls_pw` 一起使用；续训要求加权设置匹配。

`plot_samples=8` 设置验证样例图像数量。0 表示不绘制，-1 表示全部；它不会减少实际评估的图像数。
