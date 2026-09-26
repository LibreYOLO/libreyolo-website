---
title: 内核
seo_title: LibreYOLO 内核注册表与 Hub 内核
description: >-
  LibreYOLO 如何挑选加速实现：libreyolo/kernels 下的内核注册表、可选的 Hugging Face Hub
  MS-deform-attn 内核，以及融合注意力开关。
lead: >-
  LibreYOLO
  里每个加速操作都有一个可移植的默认实现，有时还有一个注册在它之上的更快变体。选择在运行时按谓词进行，缺少可选依赖是回退而不是报错，导出的计算图始终走可移植路径。
keywords:
  - libreyolo kernels
  - LIBREYOLO_KERNELS
  - LIBREYOLO_HUB_KERNELS
  - hub-kernels 可选依赖
  - ms_deform_attn 内核
  - set_fused_attention
  - libreyolo triton 内核 cuda
last_verified: 1.5.0
verification: >-
  注册表 API 读取自 v1.5.0 的 libreyolo/kernels/__init__.py，注意力 API 读取自
  libreyolo/kernels/attention/__init__.py 和 sdpa.py，Hub 提供方读取自
  libreyolo/kernels/attention/ms_deform_attn.py，包括它固定的 commit 版本和适用性谓词。目录结构列自
  libreyolo/kernels/。可选依赖的定义来自 pyproject.toml。行为说明与基准测试数据来自
  docs/kernels.md。v1.4.0 的门控历史来自给 RF-DETR 接上槽位的那次提交，以及 1.5.0 的 CHANGELOG 条目。
meta:
  - label: 包
    value: libreyolo.kernels
    mono: true
  - label: 可选依赖
    value: 'libreyolo[hub-kernels]'
    mono: true
  - label: 强制走参考实现
    value: LIBREYOLO_KERNELS=off
    mono: true
snippets:
  usage:
    - label: 查看选中了什么
      language: python
      code: |
        import libreyolo.kernels as kernels

        # op 槽位到所选实现名的映射，或者 "unavailable"
        print(kernels.active())
    - label: 强制走参考路径
      language: bash
      code: |
        # off 和 reference 意思相同，而且会完全跳过
        # 加速提供方的导入
        LIBREYOLO_KERNELS=off python train.py
    - label: 不卸载也能关掉 Hub 内核
      language: bash
      code: |
        LIBREYOLO_HUB_KERNELS=0 python predict.py
    - label: 把一个家族切到融合注意力
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.kernels.attention import set_fused_attention

        model = LibreYOLO("LibreSwinIRs.pt")

        # 返回切换了多少个注意力模块
        print(set_fused_attention(model))
    - label: 注册你自己的实现
      language: python
      code: |
        import libreyolo.kernels as kernels

        kernels.register(
            "fake_quant_fp8",
            my_impl,
            name="mybackend",
            predicate=my_check,
        )
source_hash: 23d504e88b7959f8
---

## 注册表

`libreyolo/kernels/` 是一个小的运行时注册表（registry），里面放的是可插拔的实现。
op 槽位是一个名字，比如 `fake_quant_fp8` 或 `ms_deform_attn`。调用方向注册表要一个
槽位，拿回第一个通过自己谓词的已注册实现，注册越晚优先级越高，都不适用时就落到参考
实现。

这个结构的存在是为了让可选依赖永远不成为硬性要求。一台没有 Triton、没有 CUDA、也没有
`kernels` 包的机器跑的是同样的代码，得出同样的数字，只是慢一些。

| 函数 | 用途 |
|---|---|
| `active()` | op 槽位到所选实现名的映射，或者 `"unavailable"` |
| `resolve(op)` | 实际会运行的那个可调用对象，或者 `None` |
| `register(op, impl, *, name, predicate=None)` | 添加一个实现，最新的排在最前 |
| `unregister(op, name)` | 移除一个 |
| `clear_cache()` | 丢掉记忆化的解析结果 |

<code-tabs name="usage" />

抛异常的谓词会被捕获并给出警告，绝不会向外传播，所以一个有问题的第三方实现只会退化到
可移植路径，而不是让预测挂掉。

### 目录结构

这棵树先按用途组织、再按后端组织，所以找一个槽位靠的是它算什么，而不是今天碰巧由哪个
库来实现它。

| 目录 | 内容 |
|---|---|
| `kernels/quant/simulate/` | 伪量化的 Triton 内核，带直通反向，可以在任意设备上跑。QAT 和模拟的训练后量化都用它 |
| `kernels/quant/execute/` | 只给已定型模型用的真实精度路径，没有反向：FP8 张量核 GEMM、它融合的 Triton 前置与后置阶段，以及打包权重的解包内核 |
| `kernels/attention/` | 跨家族共用的注意力操作：`ms_deform_attn` 槽位，以及融合 SDPA 的策略 |

`simulate` 和 `execute` 的分界是模型是否已经定型，而不是在训练还是在部署。参考实现留在
`libreyolo/quant/`，那里定义这些数字的含义；`kernels/` 只负责让它们变快。权重打包完全
没有变体，因为它是检查点（checkpoint）契约。

GEMM 和注意力槽位没有参考实现。调用方必须检查 `resolve()` 是不是返回了东西，并且自己
保留一条可移植路径，这也是为什么 ONNX、TensorRT 和 `torch.export` 的计算图里始终装的是
可移植的数学运算。

### 选择的覆盖方式

`LIBREYOLO_KERNELS=off` 或 `=reference` 会强制使用参考实现，并且完全跳过加速提供方的
导入。其他任何值都会把选择限制在以该名字注册的实现上。`LIBREYOLO_QUANT_KERNELS` 作为
遗留别名仍然生效，来自注册表还放在 `libreyolo/quant/` 下的那个时期，并且只在
`LIBREYOLO_KERNELS` 没设置时才会读取。两个都和其余设置一起列在[设置](/docs/reference/settings)里。

## Hub 内核

发布在 Hugging Face Hub 上的编译好的 CUDA 内核，通过可选的 `kernels` 包在运行时加载。
LibreYOLO 里没有内置任何这类产物；产物由那个包负责拉取和缓存，每个提供方都固定到一个
经过审查的 commit 版本，所以改动这个固定版本之前必须先跑一遍 GPU 一致性验证。

安装这个可选依赖就是显式开启：

```bash
pip install "libreyolo[hub-kernels]"
```

没装这个包时什么都不会变，也不会发起任何网络请求。`LIBREYOLO_HUB_KERNELS=0` 可以在不
卸载任何东西的前提下关掉拉取。加载失败或运行失败的内核会在该进程剩下的时间里自我禁用，
并带一条警告回退。

今天只有一个槽位由 Hub 提供：`ms_deform_attn`，来自 Deformable DETR 的多尺度可变形注意力
前向与反向的编译实现，采用 Apache-2.0 许可。它接进了整个可变形谱系：RF-DETR、Deformable
DETR、DINO-DETR、LW-DETR、Grounding DINO、RT-DETR、RT-DETRv2、D-FINE、RT-DETRv4、DEIM、
DEIMv2、EC 和 OV-DEIM。因为反向也是编译的，训练和预测都能受益。

适用条件是故意收窄的。输入必须是 CUDA 上的 float32，执行必须是 eager 的：提供方在
`torch.jit.is_tracing()`、`torch.compiler.is_compiling()`、`torch.compiler.is_exporting()`
和 `torch.onnx.is_in_onnx_export()` 下都会拒绝。另有两种输入布局也会落到可移植路径：各层
之间不一致的每层点数，以及离散的整数索引采样。EC 的姿态变体没有接上。

### 这个内核现在才真正跑得到

在已有项目上安装这个可选依赖之前，先读这一节。

在 v1.4.0 里，这个槽位是在一个辅助函数内部被查询的，外面还有一个要求空间形状对不存在的
条件。RF-DETR 总是把这些对一路传进它的解码器，所以这个条件从来没有成立过，内核在任何
eager 前向里都没有执行过。这次查询在 v1.5.0 挪了位置，现在内核真的会跑。

实际后果是，升级到 v1.5.0 *并且*在 CUDA 上安装 `libreyolo[hub-kernels]`，意味着 RF-DETR
和它这一系第一次从编译好的二进制里取前向。预测结果和指标可能因此在浮点误差范围内发生
变化。不带这个可选依赖的默认安装不受影响。如果你要跨这次升级比较指标，就把这个可选依赖
保持不变，或者两边都设 `LIBREYOLO_HUB_KERNELS=0`。

## 融合注意力

融合的缩放点积注意力不需要任何可选依赖，只要原版 PyTorch，所以管着它的是策略而不是
可用性。有两条规则。

第一，图捕获永远不用它。每个被替换的调用点都在一个导出检查后面保留了原始算子写法，覆盖
两种情况：ONNX 导出，它的默认 opset 没有 SDPA 的 symbolic；以及 `torch.jit.trace`，
TorchScript、CoreML 和 NCNN 都要经过它。Dynamo 的捕获被故意放在这道门之外，因为
`torch.compile` 对 SDPA 的下译（lowering）好过手写的数学运算，而且 Core AI 和 ExecuTorch
都会自己把 SDPA 分解成 core ATen。

第二，把它设为默认的一致性门槛是逐位一致。达标的家族默认使用 SDPA：SegFormer、Depth
Anything 和 MoGe-2、BERT、Grounding DINO、SwinIR 以及 PP-OCR。没达标的家族保留手写数学，
转而暴露一个 `fused_attn` 开关，`set_fused_attention(model)` 翻转的就是它：Swin、DINO-DETR
的 Swin 骨干、BiRefNet 和 FeyNobg、OWLv2、LW-DETR、SigLIP 2、ZipDepth 以及 MobileSAM。
ViT 和 DeiT 带的是同一个开关，但跟随上游默认打开，所以同样的调用加上 `enabled=False`
就是把它们关掉。

在适用的地方这么做是值得的。在 RTX 5070 Ti 上、fp16 autocast 下，Swin 的窗口注意力从
1.278 毫秒降到 0.721 毫秒，提升 1.77 倍，OWLv2 的视觉注意力从 6.483 毫秒降到 1.735 毫秒，
提升 3.74 倍。

## 硬件

| 平台 | 行为 |
|---|---|
| CPU 和 MPS | 每个 CUDA 和 Triton 谓词都不通过，所以一切都跑参考实现 |
| NVIDIA CUDA | Triton 内核，以及符合条件的 Hub 内核和 GEMM 内核会启用 |
| AMD ROCm | Triton 可以启用，因为 ROCm 的 wheel 带了 Triton 的 AMD 后端，但一致性只在 CI 的英伟达卡上验证 |

## 添加一个实现

带上名字和谓词调用 `register()`。树外的编译内核可以作为一个独立的 `libreyolo_kernels`
包发布，在导入时自行注册，这样私有后端完全不用进 LibreYOLO 的代码树。

任何进树的东西都要过一致性这一关：前向与参考实现完全一致，梯度与直通估计器的差距在
1e-6 以内，覆盖测试套件带的那组形状。

内核的选择会和 [CUDA graphs](/docs/reference/cuda-graphs) 相互影响：推理一致性矩阵是在
没有安装 `kernels` 包的情况下跑的，所以编译内核处于启用状态时的捕获安全性并不在它的
覆盖范围内。
