---
title: LoRA 微调
seo_title: LibreYOLO 中的 LoRA 微调
description: 用 lora=True 在低显存上微调 transformer 检测器。哪九个家族支持它、每个家族的适配器配方，以及检查点的行为。
lead: LoRA 冻结模型中预训练好的重型部分，在它们旁边训练小的低秩适配器，外加那些必须保持稠密的层。在 LibreYOLO 里，整个公开接口就是一个布尔值。
keywords:
  - lora 微调
  - 参数高效微调
  - peft
  - dora 微调
  - 低显存训练
  - rf-detr lora
  - d-fine lora
  - lora 适配器合并
last_verified: 1.5.0
snippets:
  install:
    - label: pip
      language: bash
      code: |
        pip install "libreyolo[lora]"
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.train(data="my-dataset.yaml", epochs=50, lora=True)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs.pt data=my-dataset.yaml \
          epochs=50 lora=true
  merge:
    - label: 导出时会合并适配器
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("runs/train/exp/weights/best.pt")
        model.export(format="onnx")
    - label: 原地合并
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.training.lora import merge_lora_adapters

        model = LibreYOLO("runs/train/exp/weights/best.pt")
        merged = merge_lora_adapters(model.model)

        print(f"{merged} adapter layers folded into dense weights")
source_hash: 603fdddf5ec0c316
---

## 安装

LoRA 依托可选依赖 `peft`。

<code-tabs name="install" />

没有它，`lora=True` 会抛出一个点名这条命令的 `ImportError`，而不是稀里糊涂地跑成一次全量微调。

## 使用

<code-tabs name="train" />

`lora=True` 就是全部接口。秩、alpha、dropout 和目标模块按家族固定，与各自的上游参考实现保持一致，不是面向用户的旋钮。

不支持 LoRA 的家族会在初始化阶段直接报错，而不是忽略这个标志：

```text
LoRA fine-tuning (lora=True) is not supported for yolo9. LoRA targets
transformer components with nn.Linear layers (e.g. RF-DETR, D-FINE, DEIM).
```

CLI 拦得更早，在模型构建之前就用它自己那份同样九个家族的允许列表把它挡下来。

## 支持哪些家族

RF-DETR、D-FINE、DEIM、DEIMv2、RT-DETR v1、v2 和 v4、EC 和 ConvNeXt。开关是每个家族训练器类上的 `supports_lora` 属性，CLI 另外带一份与之对应的允许列表。

任务覆盖比家族覆盖更窄。D-FINE 和 EC 只支持检测，它们的分割和姿态路径会报错。RF-DETR 的语义分割路径会报错。ConvNeXt 是分类。

其余一律报错。没有部分支持，也没有静默模式。

## 每套配方各自做了什么

配方不同是因为架构不同，一套在 ViT 骨干上奏效的配方，放到卷积骨干上没有东西可挂。

RF-DETR 用的是 DoRA，即权重分解的 LoRA，以秩 16、alpha 16 挂在 DINOv2 骨干的注意力 `query`、`key` 和 `value` 投影上，与 RF-DETR 的参考实现一致。ViT 骨干冻结；投影器、解码器和检测 head 照常训练。

D-FINE、DEIM 以及 RT-DETR v1、v2 和 v4 把卷积骨干和一个 transformer 混合编码器、一个可变形解码器配在一起，所以切分的位置也跟着挪。卷积骨干整体冻结，这同时也省掉了它的反向传播。transformer 块冻结自己的基础权重，在它们的线性层上训练同样是秩 16、alpha 16 的普通 LoRA 适配器：前馈的 `linear1` 和 `linear2`、门控，以及可变形注意力的那些投影。其余部分，也就是编码器的卷积融合、输入投影、预测 head 和查询嵌入向量，都继续稠密训练。

这套配方里有两个细节是刻意为之。解码器的自注意力保持冻结、不加适配器，因为 PyTorch 的 `nn.MultiheadAttention` 会直接读取 `out_proj.weight`，从而静默绕过注入进去的适配器。另外用的是普通 LoRA 而不是 DoRA，因为解码器里有几个线性层按设计就是零初始化的，而 DoRA 的幅度归一化要除以权重范数。

DEIMv2 沿用同一套配方，目标换成它的 SwiGLU 前馈层 `w12` 和 `w3`。它的 S、M、L 和 X 尺寸还带一个 DINOv3 ViT 骨干，此时 ViT 主体冻结、其融合的注意力 `qkv` 层挂上适配器，而 Spatial Tuning Adapter 的卷积金字塔作为投影器的对应物继续训练。哪怕配置文件里发布时 ViT 就是冻结的，这些 `qkv` 适配器也照加不误，因为适配一个冻结的骨干正是目的所在。sub-S 尺寸用的是卷积骨干，走普通配方。

EC 是一个 DETR，它的骨干是一个 ViT，外面包着一圈可训练的卷积投影器金字塔。ViT 主体冻结、它的 `qkv` 层挂适配器，transformer 块套用共享的那套配方，投影器和各个 head 保持稠密。

ConvNeXt 块里带的是 channels-last 的线性 MLP，即 `fc1` 和 `fc2`，这两个挂普通适配器。深度卷积（depthwise）、各个归一化层和 layer-scale 参数冻结。分类 head 保持稠密，这样自定义类别数才能继续工作。

检测 head 和分类 head 在每套配方里都始终保持可训练，因为自定义的类别数需要一个重新训练过的 head。

## 检查点与导出

`best.pt` 和 `last.pt` 会保留适配器张量，所以一次 LoRA 训练可以像其他训练一样断点续训或者拿来检查。加载这类检查点（checkpoint）需要装上 `lora` 这个额外依赖，因为加载器会重放一遍适配器注入，好让键名对得上。

`export()` 会把适配器合并进稠密权重，所以导出的产物不带任何对 `peft` 的依赖。同样的合并也可以直接对内存中的模型做。

<code-tabs name="merge" />

合并之后模块树就完全稠密了，再合并第二次是空操作。

## 它省下了什么，又没省下什么

LoRA 削减优化器和梯度占用的显存，在那些干脆把骨干冻结掉的家族上，它还省掉了这部分骨干的反向传播。

激活值显存没有变化。只要还有东西可训练，前向的激活值就得保留下来，而峰值通常就是由它决定的。显存预算最紧的时候，还要把 `batch` 或 `imgsz` 一并调小。

## 相关

- [层冻结](/docs/train/layer-freezing)：另一种只训练部分权重的办法，它对每个家族都有效，也不需要额外依赖。`freeze` 和 `lora=True` 可以组合：即使适配器所在的骨干分组被冻结，适配器参数依然保持可训练。
- [超参数](/docs/train/hyperparameters)：`batch`、`imgsz` 以及 `train()` 的其余参数。
