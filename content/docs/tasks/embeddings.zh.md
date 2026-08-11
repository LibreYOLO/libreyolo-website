---
title: 嵌入向量
seo_title: LibreYOLO 中的图像与区域嵌入向量
description: >-
  embed 任务为整张图像、每个检测到的区域或文本返回 L2 归一化的 float32 向量。注册一个底库，按余弦相似度匹配，从 Python 或 CLI
  里检索。
lead: >-
  一个任务覆盖 LibreYOLO 产出的每一个向量。embed 返回单位长度的 float32
  行，它们的点积就是相似度分数，无论这一行描述的是一整张图像、一张检测到的人脸还是一行文本，同一个 Gallery 都能匹配它们。
keywords:
  - 图像嵌入向量 python
  - l2 归一化 embedding
  - 余弦相似度检索
  - libreyolo embed 任务
  - 以图搜图
  - 人脸底库注册
  - clip 图像特征
  - dinov2 特征提取
  - 行人重识别特征
last_verified: 1.5.0
verification: >-
  任务键与别名读自 libreyolo/tasks.py。结果载荷来自 libreyolo/utils/results.py 中的 Embeddings 与
  Identities 类。Gallery API 来自 libreyolo/utils/gallery.py。embed 与
  _postprocess_embeddings 来自 libreyolo/models/base/model.py。支持的家族通过在
  libreyolo/models/**/model.py 中搜索 SUPPORTED_TASKS 里的 embed 定位。CLI 接口来自
  libreyolo/cli/__init__.py、libreyolo/cli/commands/special.py 和
  libreyolo/cli/commands/predict.py。设计意图来自
  docs/adr/0015-embed-generalization.md。
meta:
  - label: 任务键
    value: embed
    mono: true
  - label: 别名
    value: 'face-recognition, reid, face'
    mono: true
  - label: 结果载荷
    value: 'Embeddings, Identities'
    mono: true
  - label: 行的 dtype
    value: float32，单位长度
snippets:
  predict:
    - label: 整图
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # CLIP 默认走 classify，所以要显式索要向量
        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")
        result = model(SAMPLE_IMAGE)

        print(result.embeddings.data.shape)  # (1, 512)，每张图像一行
        print(result.boxes)                  # None：什么都没有定位
    - label: 逐区域
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("librefacerec-l.onnx")
        result = model(SAMPLE_IMAGE)

        # 第 i 行描述第 i 个检测框中的区域
        print(result.boxes.xyxy.shape)       # (N, 4)
        print(result.embeddings.data.shape)  # (N, 512)
    - label: 一次处理多张图像
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")

        # 每个结果的每一行，拼进同一个张量
        vectors = model.embed(["a.jpg", "b.jpg", "c.jpg"])
        print(vectors.shape)  # (3, 384)
    - label: 文本
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        # 文本是一个方法，绝不是预测的输入源；传给
        # model(...) 的字符串仍然表示路径或 URL
        text = model.embed_text(["a photo of a cat", "a photo of a dog"])
        print(text.shape)  # (2, 512)
  similarity:
    - label: 比较两组行
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        query = model.embed("query.jpg")          # (1, 512)
        pool = model.embed(["a.jpg", "b.jpg"])    # (2, 512)

        # 每一行都是单位长度，所以余弦相似度就是点积
        scores = model("query.jpg").embeddings.similarity(pool)
        print(scores.shape)  # (1, 2)
    - label: 图像与文本比对
      language: python
      code: |
        import torch

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        image = model.embed("photo.jpg")                       # (1, 512)
        text = model.embed_text(["a cat", "a dog", "a car"])   # (3, 512)

        print(torch.matmul(image, text.T))
  gallery:
    - label: 注册并识别
      language: python
      code: |
        from libreyolo import Gallery, LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        gallery = Gallery(model)
        gallery.enroll("ada", ["people/ada/1.jpg", "people/ada/2.jpg"])
        gallery.enroll("grace", "people/grace/1.jpg")
        gallery.save("refs.npz")

        result = model("group.jpg", gallery=gallery, threshold=0.4)
        for name, score in result.identities.data:
            print(name, score)   # 低于阈值时 name 为 None
    - label: Top-k 检索
      language: python
      code: |
        from libreyolo import Gallery
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")
        gallery = Gallery.load("refs.npz", model=model)

        result = model("query.jpg")
        matches = gallery.match(result.embeddings, top_k=5, threshold=0.4)
        print(matches[0])   # 第一行的 [(name, score), ...]
    - label: 注册一个你手上已有的向量
      language: python
      code: |
        from libreyolo import Gallery

        gallery = Gallery()
        gallery.enroll_embedding("ada", vector)  # 入库时会归一化
        print(gallery.identities, gallery.dim, len(gallery))
  cli:
    - label: 注册一整棵目录树
      language: bash
      code: >
        # source/<identity>/*.jpg，已有的底库会就地扩展

        libreyolo enroll model=librefacerec-l.onnx source=people/
        gallery=refs.npz
    - label: 预测时顺带识别
      language: bash
      code: |
        libreyolo predict model=librefacerec-l.onnx source=group.jpg \
          gallery=refs.npz gallery_threshold=0.45
    - label: 比较两张图像
      language: bash
      code: >
        libreyolo compare model=librefacerec-l.onnx \
          source=a.jpg source2=b.jpg threshold=0.4

        # verify 是同一条命令的第二个名字

        libreyolo verify model=librefacerec-l.onnx source=a.jpg source2=b.jpg
        --json
source_hash: ffbaad5599035bc7
---

## 定义

`embed` 把一张图像、图像中的一个区域或一个字符串，变成一行宽度固定、长度为一的
float32 向量。因为每一行都是单位向量，比较两行就是一次点积，比较两组行就是一次矩阵
乘法。这个任务里没有别的东西跟具体模型有关：检索、重复检测、重识别和人脸识别，都是
在不同的行上做同样的算术。

向量就是输出。这里没有类别列表，所以名字是之后通过和你提供的参考向量比对贴上去的，
而不是网络在训练时学会预测的东西。

### 三种形状

| 形状 | `Results.embeddings` | `Results.boxes` | 由什么产出 |
|---|---|---|---|
| 整图 | `(1, D)` | `None` | 把一张图像传给整图家族 |
| 区域 | `(N, D)` | `(N, 4)`，逐行对齐 | 先做定位的家族，例如人脸识别 |
| 文本 | 根本不是 `Results` | | `model.embed_text(texts)`，返回 `(M, D)` |

整图结果即使只有一张图像也保持二维。`(D,)` 不是允许的返回形状，所以调用方永远不必为
单行的情况做特判。文本返回的是一个普通张量而不是 `Results`，因为字符串不是图像输入
源：传给 `model(...)` 的字符串仍然表示路径或 URL，库不会去猜一个字符串是不是文本。

规范的任务键是 `embed`。`embedding`、`embeddings`、`face-recognition`、
`facial-recognition`、`recognition`、`face`、`faceid` 和 `reid` 都会归一到它，所以
`task="reid"` 和 `task="embed"` 选中的是完全相同的东西。

## 模型

有四个家族服务于这个任务，它们按是否先做定位干净地分成两类。

| 家族 | 形状 | 维度 | 还支持 |
|---|---|---|---|
| [LibreFaceRec](/docs/models/librefacerec) | 区域，每检测到一张人脸一行 | 512 | 没有；`embed` 是它唯一的任务 |
| [CLIP](/docs/models/clip) | 整图，配一座文本塔 | `b32` 和 `b16` 为 512，`l14` 为 768 | `classify`，并且它仍是默认任务 |
| [SigLIP 2](/docs/models/siglip2) | 整图，配一座文本塔 | `b16` 为 768，`so400m` 为 1152 | `classify`，并且它仍是默认任务 |
| [DINOv2](/docs/models/dinov2) | 整图，只吃图像 | 384 | `semantic`、`classify` |

CLIP 和 SigLIP 2 仍以 `classify` 为默认任务，所以 `task="embed"` 必须显式指定。它们
现有的 `-cls` 检查点（checkpoint）就是共享的双塔产物；相同的权重不会再发布一份重复
的 `-embed` 检查点。

`embed_text` 只存在于 CLIP 和 SigLIP 2 上，这两个家族带文本塔。DINOv2 没有。DINOv2
的嵌入向量绕过语义 head 和分类 head，在 224 像素下读取最后归一化的 CLS token；`n`、
`s`、`m` 和 `l` 变体共用同一个 DINOv2-S 编码器，所以四者返回的都是 `D = 384`。

本次发布新增的仅分类骨干 [ViT](/docs/models/vit)、[Swin](/docs/models/swin) 和
[DeiT](/docs/models/deit) 只声明 `classify`，不服务于这个任务。

<code-tabs name="predict" />

`model.embed(source, **kwargs)` 是批量的快捷方式：它跑一次 `predict`，把每个结果的每
一行拼进同一个 `(N_total, D)` 的 CPU float32 张量，各行维度不一致就抛错。支持任务里
没有 `embed` 的家族会抛 `NotImplementedError`。

## 结果载荷

`result.embeddings` 是一个 `Embeddings` 载荷。它的 `data` 永远是 `(N, D)` 的 float32，
推理路径上已经做过 L2 归一化，非二维的输入会抛错，而不是被悄悄改成别的形状。

| 成员 | 含义 |
|---|---|
| `.data` | `(N, D)` 矩阵 |
| `.dim` | `D` |
| `.normalized` | 同样的行，做了一次防御性的重新归一化 |
| `.similarity(other)` | 对另一组返回 `(N, M)`，对单个 `(D,)` 向量返回 `(N,)` |
| `.verify(i, j, threshold=0.4)` | 行 `i` 和行 `j` 是否属于同一主体 |

`result.identities` 是一个 `Identities` 载荷，只有在传入底库（gallery）时才存在。它
是一个普通容器而不是张量，所以在设备之间搬动 `Results` 不会动到它。

| 成员 | 含义 |
|---|---|
| `.name` | 名字列表，没有任何东西越过阈值的位置为 `None` |
| `.score` | `(N,)` 的 float32 最佳余弦分数，名字是 `None` 时也保留 |
| `.data` | `(name, score)` 元组的列表 |

<code-tabs name="similarity" />

向量默认不进 `summary()` 和 `to_json()`，因为一行 512 个浮点数意味着每个主体大约两
千字节。每一行改为报告 `embedding_dim`，用了底库时再加上 `identity` 和
`identity_score`。传 `summary(embeddings=True)` 可以把数值也带上。

## 底库

`Gallery` 是一组带名字的参考行。它把每条参考单独存下来，而不是取平均，所以一个名字
的分数取自它匹配得最好的那一条参考，加进一张糟糕的照片也拽不动某个身份的质心。

<code-tabs name="gallery" />

`Gallery(model)` 绑定到将要产出其向量的权重。`enroll(name, sources, select="best")`
对每个输入源跑一次预测，每个结果只保留置信度最高的那一行；`select="all"` 则保留每一
行，当一张参考图像确实包含多个主体时，你要的就是这个。
`enroll_embedding(name, vector)` 跳过推理，直接接收一个向量，对它做归一化并拒绝全零
的行。

`FaceGallery` 是同一个类的永久别名，早期只做人脸的版本写出的归档文件仍然能加载。

### 匹配与阈值

匹配是对每一条存下来的参考做一次稠密矩阵乘法，再取最大值归约成每个名字一个分数。这
里没有近似索引，这让数值保持精确，也给底库规模设了一个实际的上限。

两个入口在阈值以下的行为不同。`match()` 为每一行返回 `[(name, score), ...]`，低于阈
值的全部丢掉，所以没有匹配的行就是一个空列表。`identify()` 返回一个 `Identities`
载荷，它始终保留最佳分数，并在分数低于阈值时把名字置为 `None`。两者都不会拿最接近的
那个低于阈值的名字来顶替。

默认阈值处处都是 `0.4`。它是一个余弦值而不是概率，合适的工作点取决于你的数据和你对
误匹配的容忍度，所以请在有标注的图像对上扫一遍，而不是直接接受默认值。
`libreyolo enroll` 和预测参数 `gallery=` 用的是同一个数。

### 持久化

`save(path)` 写出一个压缩的 `.npz`，里面装着向量、名字，以及一个元数据块，块里记录
格式版本、嵌入向量维度和产出这些行的权重的指纹。`Gallery.load(path, model=...)` 在比
较任何东西之前先检查这两项，所以把一个底库指向另一个模型会抛错，而不是静默地把两个
毫不相干的空间里的向量互相打分。保存空底库会被拒绝。

## 命令行

| 命令 | 用途 |
|---|---|
| `libreyolo enroll` | 遍历每个身份一个文件夹的目录树，写出或扩展一个 `.npz` 底库 |
| `libreyolo compare` | 对两张图像里的主要主体求嵌入向量，报告余弦相似度 |
| `libreyolo verify` | 同一条命令的第二个名字 |
| `libreyolo predict gallery=...` | 给一次普通的预测运行附上身份 |

<code-tabs name="cli" />

每条 LibreYOLO 命令都同时接受 `key=value` 和 `--key value`，所以 `gallery=refs.npz`
和 `--gallery refs.npz` 是同一个参数。

`enroll` 接受 `model`、`source` 和 `gallery`，另外还有可选的 `face-detector`、
`device`、`--json` 和 `--quiet`。它按每个身份一个文件夹来读，文件夹名就是身份，里面
的每张图像都贡献参考：

```text
people/
  ada/
    1.jpg
    2.jpg
  grace/
    1.jpg
```

产出为空的图像会被跳过，只在 stderr 上打一行，而不是中断整个运行，汇总里会报告每个
名字存了多少条参考。已有的底库文件会就地扩展，所以身份可以随时间陆续加进去。

`compare` 和 `verify` 是同一个函数注册了两次。它们接受 `model`、`source`、`source2`
和一个可选的 `threshold`，打印余弦相似度、是否同一的判定，以及得出该判定的阈值。
`--json` 把同样的三个字段作为一个对象打印。

在 `predict` 上，`gallery` 指向一个保存好的 `.npz`，`gallery_threshold` 覆盖 `0.4`
这个默认值。把底库传给任务不是 `embed` 的模型是一个错误，而不是静默的空操作；底库文
件不存在时，会提示那条能创建它的 `libreyolo enroll` 命令。

## 人脸

人脸识别就是这个任务的区域形状，也是这种形状目前唯一发布的实现。它在嵌入向量 head
前面加了检测和对齐环节，另外还有 `verify()` 方法、一个自带检测框的参数、公开的精度
数字，以及阈值的校准指引。这些全都在[人脸识别](/docs/tasks/face-recognition)上，当对
象是人脸时，那才是该跟着走的完整流程。本页面上的一切原样适用于它。

## 训练、验证与导出

这个任务里没有任何东西是在 LibreYOLO 内部训练的。人脸嵌入向量 head 是一个 ONNX 产
物，它的 `train()`、`val()` 和 `export()` 全都会抛错；请在上游训练 head，再按路径加
载文件。CLIP、SigLIP 2 和 DINOv2 是通过它们的分类和分割任务来训练和导出的，而不是通
过 `embed`。

没有检索验证器。验证精度请在有标注的图像对上扫 `threshold` 来测，识别精度请注册一个
底库，在留出的图像上读 `identities.name` 和 `identities.score` 来测，把名字为
`None` 记作一次拒绝。
