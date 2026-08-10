---
title: 量化
seo_title: "在 PyTorch 里量化一个 LibreYOLO 模型"
description: "LibreYOLO 的 PyTorch 量化 API：九种配方、和训练数据分开的校准集、QAT 与 QAD，以及两种部署产物。"
lead: "LibreYOLO 的量化完全在 PyTorch 里跑：model.quantize() 会把模型的 Conv2d 和 Linear 模块换成量化版本并做校准。结果仍然保持普通的 predict、val、train 和 save 契约，所以量化模型和浮点模型由同一批验证器打分。"
keywords:
  - libreyolo 量化
  - yolo int8 量化
  - 量化感知训练 qat
  - qat qad
  - nvfp4 mxfp4
  - fp8 e4m3
  - 量化校准数据集
  - qdq onnx 导出
last_verified: "1.5.0"
meta:
  - label: 调用
    value: 'model.quantize(recipe="int8", calib="coco128.yaml")'
    mono: true
  - label: 命令
    value: "libreyolo quantize --model M.pt --recipe int8 --calib coco128.yaml"
    mono: true
  - label: 额外依赖
    value: "无。量化在 PyTorch 里运行。"
  - label: 家族
    value: "yolo9, rfdetr, birefnet, feynobg"
  - label: 配方
    value: "fp16, bf16, fp8, int8, w4a16, w4a8, nvfp4, mxfp4, int2"
    mono: true
  - label: 部署产物
    value: 'export(format="pt") for a packed checkpoint, export(format="onnx") for a QDQ INT8 graph'
    mono: true
verification: "读自 dev 分支上的 libreyolo/quant/api.py、libreyolo/models/base/model.py、libreyolo/cli/commands/quantize.py 和 docs/quantization.md。检查点体积数字是 docs/quantization.md 中记录的实测值。"
snippets:
  quantize:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # 结构替换加校准。calib 是一小批无标注图像，
        # 只做前向读取，用来推算激活值范围和缩放系数
        qmodel = model.quantize(recipe="int8", calib="coco128.yaml", samples=128)

        print(qmodel.quant_info())
        qmodel.val(data="coco8.yaml")          # 和浮点模型用同一批验证器
        qmodel.save("LibreYOLO9s-int8.pt")     # 检查点里带着一份量化清单
    - label: CLI
      language: bash
      code: |
        libreyolo quantize --model LibreYOLO9s.pt --recipe int8 --calib coco128.yaml
    - label: 参数
      language: python
      code: |
        model.quantize(
            recipe="int8",
            calib="coco128.yaml",      # data.yaml 路径或内置名称；None 表示跳过校准
            samples=128,               # 校准图像数量上限
            batch=8,                   # 校准批大小
            algorithm="auto",          # auto 和 minmax 等价；另一个选项是 percentile
            keep_high_precision=None,  # None 表示采用家族默认策略
            verbose=True,
        )
  reload:
    - label: 量化检查点重新加载后依然是量化模型
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 量化清单会在权重加载之前，
        # 先重建量化结构和缩放系数
        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")
        print(qmodel.quant_info())
  train:
    - label: QAT 就是在量化模型上普通地调用 train()
      language: python
      code: |
        from libreyolo import LibreYOLO

        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")

        # 这是微调，不是从头训练：用微调的学习率
        qmodel.train(data="coco8.yaml", epochs=5, lr0=1e-4)
    - label: QAD 只是加上已有的蒸馏参数
      language: python
      code: |
        qmodel.train(
            data="coco8.yaml",
            epochs=5,
            lr0=1e-4,
            distill_model="LibreYOLO9m.pt",
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train --model LibreYOLO9s-int8.pt --data coco8.yaml --epochs 5 --lr0 1e-4
  export:
    - label: 打包后的 PyTorch 检查点
      language: python
      code: |
        from libreyolo import LibreYOLO

        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")

        # 输出 LibreYOLO9s-int8-final.pt：打包后的低比特权重和缩放系数，
        # fp32 主权重被剥掉，未量化的其余部分转成 fp16
        qmodel.export(format="pt")

        # remainder="fp32" 让未量化的张量保持精确
        qmodel.export(format="pt", remainder="fp32")
    - label: QDQ INT8 ONNX
      language: python
      code: |
        from libreyolo import LibreYOLO

        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")

        # 图内嵌 QuantizeLinear/DequantizeLinear 对，带着模型
        # 自己校准出来或 QAT 训练出来的缩放系数
        qmodel.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9s-int8.pt --format onnx
  dequantize:
    - label: 回到浮点，保留 QAT 训练出的权重
      language: python
      code: |
        from libreyolo import LibreYOLO

        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")
        qmodel.dequantize()

        # 现在任何浮点导出器都可用，精度也随它支持的来
        qmodel.export(format="tensorrt", half=True)
---

## 安装

量化不需要额外依赖。模块替换、校准过程和模拟运算全都在 PyTorch 里跑，所以
`pip install libreyolo` 就是全部要求。部署产物需要各自格式所需的东西，ONNX 这条路
需要的是 `libreyolo[onnx]`。

## 量化

<code-tabs name="quantize" />

`quantize()` 会就地改造已加载的模型并把它返回。整个过程不涉及梯度：替换只是装上量化
模块，校准过程只做前向。

得到的检查点（checkpoint）就是一个普通的 LibreYOLO 检查点，只是附带了一份 `quant`
清单，所以重新加载时结构和缩放系数都完好无损：

<code-tabs name="reload" />

QAT 过程中训练器写出的检查点同样带着这份清单，也就是说这种训练跑出来的 `best.pt`
本身就是一个量化检查点。

## 配方

支持四个家族：`yolo9`、`rfdetr`、`birefnet` 和 `feynobg`。

| 配方 | 作用 | 家族 | 校准 |
|---|---|---|---|
| `fp16` | 转成半精度，输入和输出契约仍是 float32。仅推理。 | 四个都支持 | 无 |
| `bf16` | 转成 bfloat16，它保留了 float32 的指数范围。DETR 类模型上 fp16 溢出时用它解决。仅推理。 | 四个都支持 | 无 |
| `fp8` | 在 `Conv2d` 和 `Linear` 上用 E4M3 的权重和激活值：逐通道的权重缩放系数，校准出来的逐张量激活缩放系数。 | 四个都支持 | 必需 |
| `int8` | 在 `Conv2d` 和 `Linear` 上做 W8A8：逐通道对称权重，逐张量仿射激活值。 | 四个都支持 | 必需，或用 `calib=None` 只量化权重 |
| `w4a16` | 在 `Linear` 上用分组对称 INT4 权重，沿 `in_features` 每组 128，激活值保持浮点。 | rfdetr, birefnet, feynobg | 不需要 |
| `w4a8` | 在 `Linear` 上用分组 INT4 权重加校准过的 INT8 激活值。 | rfdetr, birefnet, feynobg | 必需 |
| `nvfp4` | 在 `Linear` 上做 W4A4 NVFP4：E2M1 元素，16 个元素一块，FP8 E4M3 的块缩放系数，FP32 的张量缩放系数。激活值动态缩放。 | rfdetr, birefnet, feynobg | 不需要 |
| `mxfp4` | 在 `Linear` 上做 OCP MXFP4：E2M1 元素，32 个元素一块，2 的幂次 E8M0 块缩放系数。激活值动态缩放。 | rfdetr, birefnet, feynobg | 不需要 |
| `int2` | 仅供研究：在 `Linear` 上用分组 2 比特权重，每组 64，再加 INT8 激活值。只做训练后量化完全不可用，所以必须配合 QAT 或 QAD。 | rfdetr | 必需 |

低于 8 比特的配方针对的是 `nn.Linear`，对 `yolo9` 是故意拒绝的：在当前硬件上那种加速
只发生在 GEMM 上，所以卷积留在更高的精度。YOLO9 用 `int8` 或 `fp8`。`int2` 对
`birefnet` 和 `feynobg` 也会被拒绝，因为这两个家族仅支持推理，而这个配方所依赖的
QAT 修复在那里做不了。

各家族的默认策略会把第一层和各个 head 留在浮点，YOLO9 的 DFL 卷积则从不量化：它是一个
固定的积分期望算子。有理由的话，用 `keep_high_precision=("head.",)` 覆盖它。

## 校准数据不是训练数据

`calib=` 接受几百张图像，不读标注，只做前向来估计激活值范围。`train()` 和 `val()`
里的 `data=` 则是用来算梯度和指标的带标注数据集。它们是用途不同的两个参数，而 `calib`
的默认值是 `coco128.yaml`。

`algorithm="minmax"` 保留校准各批次中见到的绝对极值，`"auto"` 选的就是它。
`"percentile"` 用的是每批 0.1 和 99.9 分位数的均值；实测它会让 DETR 家族的精度崩掉，
因为 transformer 的激活异常值是有承重作用的。真正能解决小模型 INT8 敏感问题的是校准
足够多的批次：用 `coco128` 这个默认值，YOLO9-t 落在浮点分数约一个 mAP 点以内。所选的
算法会记录在检查点清单里。

## 找回精度

<code-tabs name="train" />

量化模块保留 fp32 主权重，并用直通估计器（straight-through estimator）做伪量化，所以
梯度能到达主权重，现有的训练器不用改就能工作：EMA、AMP、断点续训和蒸馏参数都能组合
使用。

QAT 是对一个已经训练好的模型做微调。用微调的学习率，而不是从头训练的默认值，否则不管
有没有量化，一次短训练就会毁掉预训练权重。QAD 能不能用取决于家族的蒸馏支持情况，今天
指的是 `yolo9` 和 `rfdetr`。

用 `fp16` 和 `bf16` 量化出来的模型仅供推理，训练器会拒绝它们，并指向 `amp=True`。

## 导出

<code-tabs name="export" />

`format="pt"` 会把模型固化下来。打包后的低比特权重和缩放系数取代主权重，未量化的其余
部分会转成 fp16，除非传了 `remainder="fp32"`。打包的不变量是：在你做固化的那台设备上，
解包能逐位一致地复现模拟结果，所以固化出来的文件跑出的分数正是你验证过的分数。实测：
YOLO9-s int8 从 29.5 MB 降到 9.6 MB，RF-DETR-n nvfp4 从 122 MB 降到 26 MB。加载这样
一个文件得到的是可直接推理的模型，在它上面调用 `train()` 会自动从打包权重重建主权重。

`format="onnx"` 适用于 `int8` 模型，输出一张带着模型自己校准出来或 QAT 训练出来的缩放
系数的 QDQ 图，ONNX Runtime 和 TensorRT 会用真正的 INT8 kernel 来跑它。这和在浮点模型
上用 [`export(format="onnx", int8=True)`](/docs/export/onnx) 是两条不同的路，后者的缩放
系数由 ONNX Runtime 自己推导。

只做类型转换的那几个配方根本不需要量化导出器：

<code-tabs name="dequantize" />

## 限制

量化运算是在模拟中执行的，也就是即便在 AMP 下也在 float32 孤岛里算出来的伪量化。模拟
在数值上是真实的，所以在任何设备上跑出的 `val()` 分数都是关于量化运算的真实结论。它不
是速度上的结论。

有两个例外是原生执行的。`fp16` 和 `bf16` 就是普通的类型转换。固化后的 `fp8` 模块在
Ada、Hopper 和 Blackwell 级别的硬件上，通过 `torch._scaled_mm` 直接在打包的 E4M3 权重
上跑 GEMM，用的还是模拟时那套校准出来的激活缩放系数；设置 `LIBREYOLO_KERNELS=off`
可以在任何地方恢复完全模拟的路径。

部署覆盖面比配方列表要窄。这里只有 `int8` 有可部署的 ONNX 形式；`fp8` 和低于 8 比特的
线性配方在 PyTorch 里执行，通过 `format="pt"` 固化。对它们请求 ONNX 导出会报错并给出
这条提示，对一个 `int8` 模型请求任何非 ONNX 格式同样会报错：改为从 QDQ 图去构建下游
引擎。

导出一个激活值从未校准过的 `int8` 模型，会记录一条警告，并产出一张只带权重量化的图。
