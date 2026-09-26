---
title: CUDA 图
seo_title: LibreYOLO CUDA 图支持矩阵
description: 哪些家族在预测时捕获前向、在训练时捕获前向与反向，数值上能保证什么，捕获在哪里被拆开，以及不支持的家族为什么会抛错。
lead: >-
  CUDA 图记录一次固定 kernel 序列的执行，并以单次启动的方式重放。LibreYOLO 在 39 个验证过的家族上捕获推理，在 24
  个上捕获训练，始终按家族划分，始终在逐位一致的校验之后，绝不作为静默回退。
keywords:
  - libreyolo cuda graph
  - cuda_graph=True
  - cuda graph 支持矩阵
  - pytorch cuda graph 训练
  - yolo 推理加速 cuda graph
  - capture_error_mode thread_local
last_verified: 1.5.0
verification: >-
  推理家族列表取自 v1.5.0 中 tests/e2e/test_cuda_graph_families.py 里的 CAPTURABLE
  矩阵。训练家族列表、一致性类别和计时数据取自 docs/training_cuda_graphs.md。API 和 NotImplementedError
  取自 libreyolo/models/base/model.py 中的
  BaseModel._require_cuda_graph_support、cuda_graph_scope 和 capture_graph，以及
  SUPPORTS_CUDA_GRAPH 类变量。接缝拆分读自 depth_anything3、birefnet、ppocr、sam 和 sensenova
  家族中的 _get_graph_runner 覆写，以及
  libreyolo/models/base/detr_cuda_graph.py。capture_error_mode 取自
  libreyolo/models/base/cuda_graph.py 和 libreyolo/training/cuda_graph.py。训练回退取自
  libreyolo/training/trainer.py，--cuda-graph 参数取自
  libreyolo/cli/commands/train.py。
meta:
  - label: 推理家族
    value: '39'
  - label: 训练家族
    value: '24'
  - label: 推理开关
    value: predict(cuda_graph=True)
    mono: true
  - label: 训练开关
    value: train(cuda_graph=True)
    mono: true
snippets:
  usage:
    - label: 预测
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        # True 会在每种输入形状首次使用时捕获
        # "auto" 会等到形状重复出现，再去付捕获的代价
        result = model(SAMPLE_IMAGE, cuda_graph=True)
    - label: 训练
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")
        model.train(data="my-dataset.yaml", epochs=100, cuda_graph=True)
    - label: 用 CLI 训练
      language: bash
      code: |
        libreyolo train model=LibreYOLO9t.pt data=my-dataset.yaml \
          epochs=100 --cuda-graph
source_hash: 67c46199939278f2
---

## 捕获了什么

图记录的是一段固定的 kernel 序列，以及它们读写的内存地址。它不记录数值、形状或控制流。重放是一次启动而不是几百次，所以在小网络、小批大小上收益最大——这时一步的开销主要来自启动，而不是来自算术运算。

两个入口捕获的工作量不同。

| | 图内 | 图外（eager） |
|---|---|---|
| 推理 | 网络前向，`model._forward(x)` | 预处理、NMS、全部后处理 |
| 训练 | 网络前向与反向 | 损失、优化器步进、梯度裁剪、EMA、学习率调度 |

NMS 和检测损失都不在候选之列。两者都用布尔掩码做筛选，跑匈牙利匹配或标签分配器，并根据结果分支，而这恰恰是图无法记录的。把它们排除在外，正是捕获之所以安全的原因，而不是一个需要绕开的限制。

<code-tabs name="usage" />

预测时 `cuda_graph` 接受三个值。默认是 `False`。`True` 会在每种输入形状第一次出现时捕获。`"auto"` 会等一个形状重复出现，这样一次性的、形状多变的任务就不必为一次用不上的捕获买单。`capture_graph(imgsz=None, batch=1, dtype=None)` 把这份开销从第一次请求里挪走，`graph_info()` 报告已捕获的图和重放次数，`release_graphs()` 负责释放。

训练时这个开关就是一个普通布尔值，在 CLI 上是 `--cuda-graph`。周边的相关控制项见[预测性能](/docs/predict/performance)和[训练性能](/docs/train/performance)。

## 推理支持

支持是按家族来的，通过 `SUPPORTS_CUDA_GRAPH` 类变量声明；只有当一个家族对两个取自不同分布的探针输入完成捕获、并逐位一致地重放之后，才会被标记为支持。这份共用的一致性矩阵覆盖九类任务下的 39 个家族。

| 任务 | 家族 |
|---|---|
| detect | yolo1, yolo2, yolo3, yolo4, yolo9, yolo9_p2, yolo9_e2e, yolox, yolo7, yolonas, picodet, rtmdet, dfine, deim, deimv2, rtdetr, rtdetrv2, rtdetrv4, rfdetr, ec |
| segment | dfine, rtmdet, rfdetr, ec |
| pose | ec, yolonas, rfdetr |
| point | fomo |
| classify | resnet, convnext, mobilenetv4, efficientnetv2, clip, dinov2, siglip2 |
| semantic | eomt, dinov2, segformer, pidnet, lingbotvision |
| depth | depth_anything, depth_anything3, zipdepth |
| restore | nafnet, realesrgan, swinir |
| matte | birefnet |

有几个家族出现在不止一个任务下，所以矩阵的行数多于不重复的家族数。另有三个家族走的是家族专属的代码路径、配有各自的专门测试，而不是走这份共用矩阵，它们不算在这 39 个里：PP-OCR、SAM 和 SenseNova。

验证是逐位的，不是近似的。协议的早期版本按相对量级判断一致性，错误地降级了三个本来正常的家族——YOLOX、EfficientNetV2 和 YOLOv7——它们从 eager 到图的差异在 1e-7 量级，但在真正重要的那个探针上仍然逐位一致。

## 训练支持

本次发布中，训练捕获从两个家族增加到 24 个，覆盖五类任务。

| 任务 | 家族 |
|---|---|
| detect | yolo9, yolo9_p2, yolo9_e2e, yolox, yolo7, yolonas, picodet, rtmdet, rfdetr, dfine, deim, deimv2, rtdetr, rtdetrv2, rtdetrv4, ec |
| classify | resnet, convnext, mobilenetv4, efficientnetv2 |
| semantic | segformer, lingbotvision |
| point | fomo |
| restore | nafnet |

其余一切都以 eager 训练：这些家族上的其他任务、没有列出的家族、分布式运行和蒸馏运行。形状还是新的时候也会跳过捕获，因为训练路径要等一个输入形状重复出现三次才捕获，这意味着 `multi_scale=True` 可能根本不会捕获。

## 不支持的家族有两种不同的回应

推理路径会抛错。对没有开启支持的家族调用 `predict(cuda_graph=True)`，会抛出点名该家族的 `NotImplementedError`，而不是转去 eager 执行、让你以为拿到了其实并不存在的加速。原因在于坏的捕获不会大声报错：一个做了不可捕获操作的前向，在重放时会静默返回错误的数值，所以支持必须是按家族逐个给出的显式断言，而不是先尝试、失败再回退。

训练路径只写日志。`train(cuda_graph=True)` 任何时候都可以放心传，无法捕获的家族、任务或配置会写一行日志，然后照常以 eager 训练。一次运行中途失败的捕获，同样是把剩下的部分降为 eager，而不是中止这次运行。这种不对称是有意的：预测是一次调用，你可以在调用处直接改；而一次训练运行不该因为一项可选的优化在第六个小时上死掉。

## 接缝拆分

有些家族没法整体捕获，因为其中某个阶段确实做了图记录不了的事情。与其放弃这个家族，不如在一个验证过的接缝处把捕获拆开：可捕获的部分走重放，其余部分以 eager 执行，合起来的输出和全部 eager 执行一样。

| 家族 | 捕获的部分 | eager 的部分，以及原因 |
|---|---|---|
| Depth Anything 3 | 网络 | `sky` 步骤，它是前向之后的、主机可见的工作 |
| BiRefNet | 编码器，`forward_enc` | 解码器，它的 `deform_conv2d` 在捕获下重放会得到不同的结果 |
| PP-OCR | 检测阶段，`forward_det` | 识别，因为每一行的裁剪宽度都不一样 |
| SAM | 图像编码器 | 提示路径（prompt），每次编码要跑很多次 |
| SenseNova | 视觉塔 | 自回归生成，KV 缓存每一步都在变大 |
| 编码器-解码器检测器 | 骨干和编码器 | 解码器和匈牙利匹配准则 |

BiRefNet 这一拆值得多读一遍：`deform_conv2d` 在捕获下的异常行为，在任何模型之外的裸调用上也能复现。用纯 PyTorch 的等价实现替换它这个方案被否掉了，因为那样也会改变 eager 的预测结果，而 eager 的数值就是契约。

编码器-解码器这一类涵盖 D-FINE、DEIM、DEIMv2、RT-DETR、RT-DETRv2、RT-DETRv4 和 EC。它们的解码器会从真值（ground truth）构造对比去噪查询，而这些查询的数量取决于一个批次里最大的真值个数，所以解码器的 token 数量会随批次变化。这正是图唯一不能容忍的东西。对这些家族来说，骨干加编码器大约占一步的五分之一到四分之一，这也是它们排在加速表末尾的原因。

PP-OCR 为每种检测输入形状捕获一张图，数量受 runner 的缓存上限约束，在没有活跃的捕获作用域时返回 eager 的结果。

## 数值

大多数家族是逐位一致的，不一致的地方也都点明了原因，而不是含糊带过。在训练的第 0 步，全部 24 个家族的损失都逐位一致，也没有任何 BatchNorm 缓冲区出现差异；把这些类别区分开的是梯度的比较。

| 类别 | 家族 | 含义 |
|---|---|---|
| 完全一致 | 24 个里的大多数 | 每个梯度都逐位一致 |
| 1 ULP | fomo, lingbotvision | float32 的最后一位，相对约 1e-7，来自不同的求和顺序 |
| eager 噪声 | DETR 一系 | 图与 eager 的差异，不超过两次 eager 运行彼此之间的差异 |
| 浮点舍入 | rtmdet | 139 个梯度里有 137 个逐位一致，两个相差约 3e-4 |
| 独立 RNG 流 | segformer | 随机深度（stochastic depth）落在被捕获的区域内 |

eager 噪声这一类最需要读对。对这些家族来说，两次固定随机种子的 eager 运行本来就对不上，所以逐位一致不是图运行没跨过的门槛，而是谁都跨不过的门槛。这在 `amp=False` 下更普遍：实测 fp32 权重梯度里 3.2e-7 的相对不确定性会不断累积——两次固定种子的 eager YOLOv9-t 运行在 20 步内会偏离 36%，关掉 TF32 也解决不了。

## 锁页内存

捕获以 `capture_error_mode="thread_local"` 运行。在 PyTorch 默认的 `"global"` 模式下，DataLoader 的锁页内存线程在预备下一个批次时会调用 `cudaHostAlloc`，这既会让进行中的捕获失效，自己也会被它污染，于是运行会在下一次取批次时死掉，错误从锁页内存线程内部抛出。这一组合在一次真实的训练任务里出现过两次，之后才被定位。

thread-local 模式只约束发起捕获的那个线程。锁页线程从不碰捕获所用的流，所以它做的任何事本来就不该进图。训练还更进一步，临时把 `torch.cuda.CUDAGraph` 替换成一个强制该模式的子类，因为 `make_graphed_callables` 没有为它暴露参数；替换过程带锁，这样两个并发的捕获就不会把替换留在原地。

## 它值多少

在 RTX 5070 Ti 上、开启 AMP 测得，每组一个进程，重放同一个真实批次以便把 dataloader 排除在外，取预热之后 24 步里最快的一次。检测在 640 px，分类在 224 px。

| 家族 | 批大小 | 加速比 |
|---|---:|---:|
| FOMO s | 16 | 3.63x |
| MobileNetV4 s | 16 | 2.74x |
| EfficientNetV2 b0 | 16 | 2.44x |
| YOLOv9-t | 8 | 1.99x |
| YOLOv9 e2e | 8 | 1.76x |
| YOLOv9 p2 | 8 | 1.49x |
| 其余全部 | 不定 | 1.04x 到 1.26x |

整个运行的收益要小一些，因为图没法加速 dataloader 或验证。在 406 张图上做 20 轮 YOLOv9-t 微调，耗时从 428.4 s 降到 367.7 s，端到端加速 1.16x，两组的 mAP50-95 都是 0.6394，每轮的损失也完全相同。

上限取决于一步里有多少比例是网络。在同一硬件、640 px、批大小 8 的条件下，YOLOv9-t 是 84%，而 RTMDet-t 只有 26%，它一步中的大部分时间花在标签分配器上。启动开销在 Windows 上最高，所以 Linux 上的收益大约落在这张表的三分之一到一半，而受 dataloader 限制的运行完全看不到墙钟时间的变化。峰值显存的变化在低 5% 到高 19% 之间。

## 注意事项

图记录的是地址而不是数值，所以任何会让参数搬家的操作都会让它失效。通过 `predict(device=...)` 换设备、量化和反量化，都会让已捕获的图作废。

批大小比家族更重要：RT-DETR-r18 在批大小 2 上加速 1.19x，在批大小 8 上只有 1.04x，因为大批次受算力限制，可以省掉的启动开销也更少。

推理一致性测试套件是在没有安装可选的 `kernels` 包的情况下跑的，所以启用编译版 Hub kernel 时的捕获安全性不在它的覆盖范围内。排查捕获问题时，设置 `LIBREYOLO_HUB_KERNELS=0` 可以把它们排除在外。见 [kernels](/docs/reference/kernels)。
