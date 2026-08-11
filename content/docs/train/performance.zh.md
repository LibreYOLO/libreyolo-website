---
title: 训练性能
seo_title: 训练提速：CUDA 图、AMP、性能剖析器
description: 让一次训练跑得更快：把训练步捕获进 CUDA 图、选一个 AMP 的 dtype，再用内置的性能剖析器看清时间到底花在了哪里。
lead: 有三根杠杆决定一个训练步跑得多快：混合精度、把网络的前向与反向捕获进 CUDA 图，以及性能剖析器指出的那个真正拖住这一步的东西。
keywords:
  - cuda graph 训练加速
  - 训练速度优化
  - 混合精度训练
  - bfloat16 训练
  - pytorch 性能分析
  - 数据加载瓶颈
  - kernel 启动开销
  - gpu 利用率低
last_verified: 1.5.0
snippets:
  profile:
    - label: 剖析并继续训练
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # 剖析一小段真实训练步的窗口，打印一条结论，然后
        # 撤掉 hook 继续这次运行
        model.train(data="my-dataset.yaml", epochs=100, profile=True)
    - label: 只测量，然后停下
      language: bash
      code: |
        # 设 no_aug_epochs=0，只跑够填满窗口的那些轮数
        libreyolo profile run coco128 --weights LibreYOLO9s.pt --size s
    - label: 深入查看结果
      language: bash
      code: |
        libreyolo profile summary runs/profile/prof/profile.json
        libreyolo profile phases runs/profile/prof/profile.json
        libreyolo profile kernels runs/profile/prof/profile.json --top 10
  graph:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", epochs=100, cuda_graph=True)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 cuda_graph=true
  amp:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", amp=True, amp_dtype="bfloat16")
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          amp_dtype=bfloat16
source_hash: ee5bb727065b6099
---

## 动手改任何东西之前先测量

下面这三根杠杆解决的是不同的问题，用错了那一根什么都不会变。你遇到的是哪个问题，
性能剖析器会告诉你。

<code-tabs name="profile" />

`profile=True` 会测量一段真实训练步的窗口，默认先丢掉五步、再测量二十步，打印一份
报告，写出它的产物，然后撤掉 hook 继续训练。关掉时没有任何开销，在分布式训练下
会被忽略。

报告最后给出四种结论中的一种：

| 结论 | 含义 | 杠杆 |
|---|---|---|
| `dataloader` | GPU 在等输入数据 | 加大 `workers`、`cache="ram"` 或 `"disk"`、更轻的数据增强、增大批大小 |
| `host / launch` | 喂给 GPU 的速度太慢，kernel 又多又小 | 增大批大小、CUDA 图、减少每步的主机同步 |
| `compute` | GPU 已经跑满 | AMP 或 bfloat16，或者接受现状 |
| `memory-pressure` | 分配器颠簸，显存到了边缘 | 降低批大小；这里的利用率数字不可信 |

利用率这个数字是 kernel 忙碌时间除以未同步的单步耗时。这个窗口是特意拆开的：前一半
不加任何额外同步，这样结论反映的是真实的重叠情况，只有后一半才在每个阶段前后各加
一次同步，用来归属 GPU 时间。给每个阶段都加同步会让 dataloader 的 worker 多出余量，
把饥饿现象藏起来，所以阶段构成的数字从不用来判定结论。

运行目录里会生成四个文件：`timeline.html`，它自己就能在浏览器里打开；给 Perfetto 或
Nsight 用的 `profile_trace.json`；`profile_summary.json`；以及 `profile.json`，那份
自包含、可以随手拷走再喂回 `libreyolo profile` 子命令的文件。

关于 `profile run` 有两件事值得知道。它会设 `no_aug_epochs=0`，因为性能剖析器测的是
第 0 轮，而一次短运行在默认 `no_aug_epochs` 下剖析到的会是更轻的、不带增强的
dataloader，而不是训练真正用的那个。还有 `--repeat N` 会报告均值和标准差，这一点很
重要，因为受启动限制的一步噪声大到单次运行会误导人；它会写出 `prof_1`、`prof_2`
这样的逐次试验目录，外加一份汇总的 `profile_repeat.json`。

## 混合精度

`amp=True` 是大多数家族的默认值，它让前向传播在 CUDA autocast 下运行。`amp_dtype`
在 `float16` 和 `bfloat16` 之间选。

<code-tabs name="amp" />

Float16 需要动态损失缩放，因此会配一个实时的梯度缩放器；bfloat16 的指数范围更宽，
不需要这个，所以它的缩放器是关掉的。有四个家族默认就带着 `amp=False`——D-FINE、
DEIM、YOLO-NAS 和 FOMO——而 DEIM 的这项设置通过继承传给了 RT-DETRv4。D-FINE 说明了
理由：它的解码器把激活值钳在 65504，也就是 float16 能表示的最大有限值。

这些参数的语义，包括在不支持 bfloat16 的硬件上请求 bfloat16 会发生什么，都在
[超参数](/docs/train/hyperparameters)。

## CUDA 图

`cuda_graph=True` 会把网络训练时的前向与反向捕获进一张 CUDA 图，消除每一步的 kernel
启动开销。

<code-tabs name="graph" />

这个开关任何时候都可以放心传。无法捕获的家族、任务或配置会写一行日志，然后照常以
eager 训练，什么都不变。

只有网络会被捕获。损失按设计留在 eager，因为检测损失要用布尔掩码筛选、跑匈牙利匹配、
根据分配结果分支，这些图一个都记录不了。优化器步进、梯度裁剪、EMA 更新和学习率
调度同样留在 eager。

这就把收益的上限锁在了一步里网络所占的比例上，而这个比例差别很大。在 RTX 5070 Ti
上、640 px、批大小 8 测得：YOLOv9-t 一步里 84% 是网络，YOLOv7-b 是 44%，YOLOX-t 是
31%，RTMDet-t 是 26%。后两者一步中的大部分时间花在各自的标签分配器里，所以捕获网络
对它们的帮助最小。

### 收益有多大

下面每个数字的条件：RTX 5070 Ti、Windows、AMP，每组一个进程、从同一份保存的状态出发，
重放同一个真实批次以便把 dataloader 排除在外，取预热之后 24 步里最快的一次。检测在
640 px，分类在 224 px。批大小按行给出。

| 家族 | 尺寸 | 批大小 | eager | 捕获后 | 加速比 |
|---|---|---:|---:|---:|---:|
| FOMO | s | 16 | 7.0 ms | 1.9 ms | 3.63x |
| MobileNetV4 | s | 16 | 14.5 ms | 5.3 ms | 2.74x |
| EfficientNetV2 | b0 | 16 | 29.0 ms | 11.9 ms | 2.44x |
| YOLOv9 | t | 8 | 93.6 ms | 47.0 ms | 1.99x |
| NAFNet | s | 8 | 132.5 ms | 105.5 ms | 1.26x |
| PicoDet | s | 8 | 145.0 ms | 118.7 ms | 1.22x |
| D-FINE | n | 4 | 185.3 ms | 159.2 ms | 1.16x |
| RF-DETR | n | 4 | 276.3 ms | 239.8 ms | 1.15x |
| YOLOX | t | 8 | 102.2 ms | 90.5 ms | 1.13x |
| RTMDet | t | 8 | 149.7 ms | 136.2 ms | 1.10x |
| YOLOv7 | b | 4 | 102.5 ms | 98.0 ms | 1.05x |

这些数字隔离出的是 GPU 上的那一步。一次完整的微调还要为 dataloader 和验证买单。同一
台机器上，YOLOv9-t 在一个 406 张图的检测集上跑 20 轮、批大小 8、640 px、4 个
dataloader worker：eager 的墙钟时间是 428.4 s，捕获后是 367.7 s，收益 1.16x，两组的
mAP50-95 都是 0.6394。

有三件事会影响这些数字。小批次受启动限制，大批次受算力限制，所以 RT-DETR-r18 在批
大小 2 上加速 1.19x，在批大小 8 上只有 1.04x。启动开销在 Windows 上最高，Linux 上的
收益大约落在这张表的三分之一到一半。还有，受 dataloader 限制的运行在墙钟时间上完全
看不到变化，这正是性能剖析器要排在最前面的原因。

在 `amp=False` 下，捕获的生效方式完全相同，但 fp32 的 kernel 跑得更久，所以一步受
启动限制的程度更低，大多数家族的收益也更小。在同一硬件上，MobileNetV4-s 在批大小 16
上从 AMP 下的 2.74x 变成 fp32 下的 3.61x，而 YOLOv9-t 在批大小 8 上从 1.99x 变成
1.69x，RT-DETR-r18 在批大小 4 上从 1.12x 变成 0.99x。

### 捕获在哪些地方适用

| 任务 | 家族 |
|---|---|
| detect | yolo9, yolo9_p2, yolo9_e2e, yolox, yolo7, yolonas, picodet, rtmdet, rfdetr, dfine, deim, deimv2, rtdetr, rtdetrv2, rtdetrv4, ec |
| classify | resnet, convnext, mobilenetv4, efficientnetv2 |
| semantic | segformer, lingbotvision |
| point | fomo |
| restore | nafnet |

其余一切都会写一行日志然后回退到 eager：这些家族上的其他任务、没有列出的家族、
分布式运行和蒸馏运行。运行时的一次捕获失败同样是把这次运行剩下的部分降为 eager，
而不是让它失败。

对编码器-解码器检测器——D-FINE、DEIM、DEIMv2、RT-DETR v1、v2 和 v4，以及 EC——只有
骨干和编码器会被捕获。它们的解码器要读真值（ground truth）来构造对比去噪查询，而这些
查询的数量取决于批次里最大的真值个数，所以它的 token 数量会随批次变化。

### 形状

一张图只对它捕获时用的那个输入形状有效。训练器会统计批次形状，等某个形状重复出现
三次之后才捕获。任何其他形状的批次都以 eager 运行：多尺度的批次，以及一轮里最后那个
不完整的批次。

这就是 DETR 家族的陷阱，它们默认给每个批次都重新缩放尺寸。在 `multi_scale=True` 下，
一次短运行可能根本没见到哪个形状出现得足够频繁，也就完全捕获不了。如果要的就是这份
加速，那就传 `multi_scale=False`。

YOLOX 会在一次运行进行到一半时改变被捕获区域的计算内容：mosaic 在 `no_aug_epochs`
处关闭时，它会打开自己的 L1 回归分支。训练器会在那里让已有的捕获失效，等新的形状
稳定下来后重新捕获。

### 数值与显存

在 AMP 下，大多数家族能逐位复现它们 eager 的损失轨迹。FOMO 和 LingBot-Vision 因为
求和顺序不同，在 float32 的最后一位上有差异。可变形注意力的检测器——D-FINE、DEIM、
DEIMv2、RT-DETR、RF-DETR 和 EC——连它们自己的 eager 运行都复现不了，因为那个反向用
原子操作累加，而 TF32 卷积每次启动都会挑一个归约顺序；捕获后的运行落在这个波动范围
之内。RTMDet 在 139 个梯度里有两个的相对偏差约为 3e-4，因为它在各金字塔层级之间共享
head 卷积，两条反向路径以不同的顺序把三份贡献加了起来。SegFormer 在被捕获的区域内有
随机深度（stochastic depth），所以重放的图会抽自己的随机流，它与 eager 是统计等价而
不是完全相同；管理器会在捕获时记一次日志说明这一点。

在 `amp=False` 下，这台硬件上不管有没有捕获，什么都给不出逐位一致。两次随机种子相同
的 eager YOLOv9-t 运行在 20 步内的相对偏离达到 36%，YOLOX-t 是 2.6%，因为 cuDNN 对
某些 fp32 卷积形状会选一个不确定的权重梯度算法。

捕获出来的图会固定住静态的输入、输出和工作区缓冲区，所以显存峰值大约会多出一份激活值。
在上面这些家族里，峰值分配量的变化落在 -5% 到 +19% 之间。相对代价最大的是小的分类
模型，它们的激活值本来就小：ResNet-18 在 224 px、批大小 16 下从 eager 的 0.48 GB 变成
捕获后的 0.57 GB。如果这让一次运行超出了上限，就降低批大小，或者干脆不开这个开关。

## 相关

- [超参数](/docs/train/hyperparameters)，讲 `batch`、`nbs`、`cache` 和
  `workers`。
- [多卡训练](/docs/train/multi-gpu)，那里 CUDA 图和性能剖析器都不可用。
- [CUDA 图](/docs/reference/cuda-graphs)，讲推理与训练合起来的支持矩阵、接缝拆分
  和数值契约。
