---
title: 上游检查点
seo_title: 在 LibreYOLO 中加载上游检查点
description: 自动转换如何把发布的上游检查点变成 LibreYOLO v1.0 检查点：它能解开哪些结构、哪些家族识别什么，以及它到哪里为止。
lead: >-
  LibreYOLO 的家族移植自上游项目，那些项目发布的检查点几乎可以直接加载，却没有带 LibreYOLO 元数据。自动转换会识别这些文件，把它们包装成
  schema v1.0，并把结果写在源文件旁边。
keywords:
  - libreyolo 自动转换
  - 加载上游检查点
  - convert_upstream_state_dict
  - libreyolo 上游权重
  - 检查点转换
last_verified: 1.5.0
verification: >-
  行为读自 libreyolo/models/autoconvert.py 与
  BaseModel.convert_upstream_state_dict；各家族的识别逻辑通过阅读每个家族对
  convert_upstream_state_dict 的重写核对，全部基于 v1.5.0。RF-DETR 的 COCO 规则来自
  docs/checkpoint_schema.md。
snippets:
  usage:
    - label: 直接把文件传给工厂
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 被识别的上游文件会在加载时转换，转换后的
        # 检查点写在它旁边
        # model = LibreYOLO("yolov9-t-converted.pt")

        # 任何 LibreYOLO 检查点都原样加载
        model = LibreYOLO("LibreYOLO9t.pt")
        print(model.family, model.size, model.task, model.nb_classes)
source_hash: c6022771a2a207a1
---

## 加载时发生了什么

当 `LibreYOLO()` 遇到一个还不是完整 v1.0 检查点（checkpoint）的 `.pt` 文件时，它会调用自动转换器，后者会：

1. 从常见的上游结构中解开张量字典；
2. 询问每个已注册的家族是否识别这个结构，并在上游命名与原生移植不同时重映射键；
3. 把胜出者包装成严格的 v1.0 元数据检查点，直接从张量本身读取尺寸、任务和类别数，因此微调过的检查点也能正确转换；
4. 把它写在源文件旁边，命名为 `<source>-<Prefix><size>[-task].pt` 并返回该路径，于是工厂照常加载它。

调用方什么都不用做。没有任何家族认领的文件不会返回结果，工厂会报告它无法加载。

<code-tabs name="usage" />

## 它能解开的结构

按这个优先顺序寻找张量字典，EMA 在最前，并逐个尝试候选，直到某一个确实装着张量。因此一个空的、或只有元数据的 EMA 块不会遮住下面有效的权重。

| 键 | 说明 |
|---|---|
| `ema.module` | 常见的 EMA 包装 |
| `ema` | 直接存放张量的老式扁平 EMA 包装 |
| `ema_state_dict` | `module.` 前缀下的条目会被剥掉 |
| `params_ema` | |
| `params` | |
| `ema_net` | |
| `net` | |
| `model` | |
| `state_dict` | |
| 文件本身 | 一个普通的 state dict |

每个候选随后会被收窄到取值为张量的条目并做归一化：开头的 `module.` 或 `_orig_mod.` 前缀会被剥掉，键全部以 `model.model.` 开头的字典会去掉这个前缀。

## 哪些家族识别什么

识别是各家族自己的 classmethod。默认实现认领键已经与原生移植匹配的结构。上游键命名不同的家族会用一个重映射重写它，并对它不认识的结构返回空。

带重映射识别器的家族：`centernet`、`deeplabv3`、`deformable_detr`、`dexined`、`moge2`、`picodet`、`rtdetr`、`rtdetrv2`、`rtdetrv4`、`rtmdet`、`segformer`、`swin`、`teed`、`yolo7`、`yolo9`、`yolo9_e2e`、`yolo9_p2`。

直接拒绝自动转换的家族：`efficientdet`、`eomt` 和 `pidnet` 的识别器返回空，因此它们的上游文件改走转换脚本。`l2cs` 被排除在通用识别器之外，因为它只能推理，且权重有再分发限制。

RF-DETR 保留自己的识别器，因为它需要整个检查点而不只是张量字典，才能检测尺寸并重映射 COCO 类别。只有在安装了它的可选依赖时才会注册。

其余每个已注册的家族都用默认实现：当自己的加载器已经识别那些键时，它就认领这个文件。

## 哪个家族胜出

多个家族可以认领同一个文件，因此裁决方式与工厂的分派规则一致。

子类的认领胜过它的基类。注册顺序跟随类创建顺序，因此派生家族在它所细化的基类之后注册，而它的正向标志不能输给基类更宽泛的透传。

接下来由注册表（registry）顺序决定，因为它编码了特异性：最早的认领就是最具体的匹配。

注册表顺序唯一打不破的平局是 DEIM 对 D-FINE，它们的架构键完全相同。只有在那里，文件名才是决定性信号，而名字给不出线索的文件会被拒绝，而不是靠猜。其他任何地方都刻意不查看文件名，因此一个宽泛的误报认领绝不会仅凭文件叫什么，就被提到比更具体的认领更前面。

## 安全加载

上游文件通过 weights-only unpickler 加载。有些上游训练检查点里嵌入了该 unpickler 拒绝的库对象。这些对象是训练元数据而不是权重，因此每个被拦下的 global 都会用一个惰性替身类重试，它满足 unpickler 而不执行任何东西。捕获到的名字只作为字符串标签使用，绝不导入、求值或调用。

敏感的模块名会被直接拒绝、绝不打桩：`builtins`、`os`、`sys`、`posix`、`nt` 和 `subprocess`。重试循环上限为 32 次，因此刻意构造出一长串不同 global 的文件会以失败告终，而不是一直空转。只有张量能进入转换后的检查点。

## 转换后的文件去哪里

输出写在源文件旁边，命名为 `<source>-<Prefix><size>[-task].pt`。它总是重新写入而不是复用，这样既让同一个源的反复加载保持最新，又避免与官方权重、或与同一目录下同家族、同尺寸、同任务的另一次微调发生冲突。

当源目录只读时，转换会退回到每次调用新建的私有临时目录，日志行会写出它用的路径。只有当这也失败时，转换才会被放弃，并给出一条警告。

## 已有的 LibreYOLO 检查点

带有 LibreYOLO 专属标记 `libreyolo_version` 或 `model_family` 的文件属于正常加载路径，不会被重新转换。这个跳过只适用于透传认领，也就是键集没有变化的那种。转换改变了键集的认领，证明这是外来的上游结构，即使文件带标记也会被接受。

`schema_version` 刻意不被当作标记，因为其他训练和导出工具也在用这个通用名字；`names`、`nc`、`size`、`task` 和 `imgsz` 同样不算，因为上游微调也可能带着它们。因此一个只是带了通用 `names` 键的外来微调不会被标记，于是它的原生键认领会正常转换，并从张量 head 推导类别数，而不是被误加载成 80 类。

## 转换器读取的元数据

类别名取自顶层的 `names` 键，或者 `args` 或 `hyper_parameters` 块里的 `class_names`。按标签而不是按类别索引作键的 names 映射无法使用，会被替换成生成的默认值。比检测到的类别数更长的 names 列表会被截断，因为越界索引会通不过严格校验器，并悄悄中止转换。

上游的 `args` 会作为普通元数据一并带过来，其中任何不是字符串、数字、布尔、列表或字典的值都会被丢弃，因此不会有不安全的东西进入保存的文件。

## RF-DETR 的 COCO 归一化

上游 RF-DETR 检查点暴露的是 91 输出的分类 head，也就是 COCO 的 90 类加背景。自动转换把 COCO 版的 RF-DETR 归一化到 COCO-80 约定，重映射在后处理阶段应用。

当一个检查点恰好带 80 个名字、或声明类别数为 80、或有 `coco` 数据集提示、或完全没有类别与数据集元数据时，它会被当作 COCO。最后那种情况很重要：一个裸的上游 state dict 就是标准的 COCO 预训练检查点，也是分发中唯一没有元数据的 91 输出 RF-DETR。

真正自定义的 90 类 RF-DETR 会被保留为 90 类。它靠 names 列表、明确的非 80 类别数，或非 COCO 的数据集提示来识别，因此裸检查点的兜底不会对它生效。判断是否存在数据集提示时，空占位符会被忽略。

## 限制

自动转换识别的是已发布的上游结构。它不重写架构，也不能让一个没有移植的模型变得可加载。当没有家族认领某个文件时，答案是一个转换脚本而不是一个工厂参数：仓库为需要的家族提供了 `weights/convert_*.py`，包括 EoMT、PIDNet 和 EfficientDet。

转换同样不会凭空捏造它读不到的元数据。尺寸、任务和类别数来自张量；名字在文件里有时取自文件，没有时生成为 `class_i`。
