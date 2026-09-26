---
title: 人脸识别
seo_title: LibreYOLO 中的人脸识别
description: 在 LibreYOLO 里检测人脸、求嵌入向量并识别身份。注册人脸库、比较两张图像，用余弦相似度匹配，Python 和 CLI 都能做。
lead: >-
  人脸识别就是把 embed 任务用在人脸上。检测器定位并对齐每张人脸，识别 head 为每张人脸返回一个 L2
  归一化向量，身份由与已注册参考向量的余弦相似度决定，而不是由一份固定的类别列表决定。
keywords:
  - 人脸识别 python
  - 人脸特征提取
  - 人脸比对 验证
  - 人脸底库 注册
  - arcface onnx
  - libreyolo 人脸识别
  - 余弦相似度 人脸
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # librefacerec-* 名称一律路由到人脸嵌入向量家族，与文件后缀无关，
        # 并在首次使用时连同默认人脸检测器一起从 LibreYOLO 的 Hugging Face
        # 组织下载
        model = LibreYOLO("librefacerec-l.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)             # (N, 4) 人脸检测框
        print(result.embeddings.data.shape)  # (N, D)，每张人脸一行
        print(result.embeddings.dim)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=librefacerec-l.onnx source=photo.jpg
    - label: 比较两张图像
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        # 对两张图都跑检测和嵌入向量，并比较各自置信度最高的人脸，
        # 余弦相似度取值范围为 [-1, 1]
        outcome = model.verify("person_a.jpg", "person_b.jpg", threshold=0.4)
        print(outcome["similarity"], outcome["same_person"])
    - label: 注册人脸库并识别
      language: python
      code: |
        from libreyolo import Gallery, LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        gallery = Gallery(model)
        gallery.enroll("ada", ["people/ada/1.jpg", "people/ada/2.jpg"])
        gallery.enroll("grace", "people/grace/1.jpg")
        gallery.save("faces.npz")

        result = model("group_photo.jpg", gallery=gallery, threshold=0.4)
        for name, score in result.identities.data:
            print(name, score)   # 低于阈值时 name 为 None
    - label: 用 CLI 注册并识别
      language: bash
      code: >
        libreyolo enroll model=librefacerec-l.onnx source=people/
        gallery=faces.npz

        libreyolo predict model=librefacerec-l.onnx source=group_photo.jpg
        gallery=faces.npz
    - label: 自带人脸检测框
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("librefacerec-l.onnx")

        # face_boxes 会完全跳过检测；face_detector 接受可调用对象、
        # LibreYOLO 检测模型或 FaceDetector 实例
        result = model(SAMPLE_IMAGE, face_boxes=[[34, 12, 90, 80]])
        print(result.embeddings.data.shape)
source_hash: d7dfcb6f812ebb2d
---

## 定义

人脸识别为每张人脸返回一个向量，而不是一个标签。预测分两个阶段：人脸检测器
定位每张人脸及其五个关键点，裁剪出来的图像被变换到标准的 112x112 对齐姿态，
识别 head 输出一个 L2 归一化的嵌入向量。

`result.embeddings` 是一个形状为 `(N, D)` 的 `Embeddings` 负载，与
`result.boxes` 按行对齐，因此第 `i` 行描述的就是第 `i` 个检测框里的人脸。由于
每一行都是单位向量，余弦相似度就是点积，`embeddings.similarity()` 一次调用就能
对另一个 `Embeddings` 或整个矩阵算出结果。

给人脸命名是单独的一步。`Gallery` 保存带名字的参考向量；给 `predict()` 传入
`gallery=` 会附加 `result.identities`，它与嵌入向量按行对齐，为每张人脸给出一个
名字和对应的最佳余弦分数。低于匹配阈值的人脸，名字保持为 `None`，绝不会用最接近
但仍低于阈值的名字顶替。

库里规范的任务键是 `embed`。`face-recognition`、`facial-recognition`、`reid` 和
`face` 都会规范化到它，所以 `task="face-recognition"` 和 `task="embed"` 选中的是
同一件事。人脸是这个更宽泛任务的区域形态；[嵌入向量](/docs/tasks/embeddings)
讲的是整图形态和文本形态、共用的 `Embeddings`、`Identities` 与 `Gallery` API，
以及那些不做任何检测就产出向量的模型。

## 模型

[LibreFaceRec](/docs/models/librefacerec) 是这个任务的模型家族。它是藏在一次调用
背后的两个 ONNX 产物：`librefacerec-l.onnx`，一个输出 512 维嵌入向量的 iResNet100
识别 head；以及 `librefacerec-det.onnx`，带五个关键点的默认人脸检测器，取自
OpenCV zoo。两者都在首次使用时从 LibreYOLO 的 Hugging Face 组织下载。任何其他遵循
ArcFace 约定的 ONNX 文件（输入对齐后的 112x112，输出 `(N, D)`）都可以替换识别
head，只要传它的路径而不是 `librefacerec-*` 名称。

`embed` 任务键比人脸更宽。[CLIP](/docs/models/clip)、
[SigLIP2](/docs/models/siglip2) 和 [DINOv2](/docs/models/dinov2) 同样支持
`task="embed"`，返回一个整图向量，那是图像检索而不是人脸身份。它们共用 `Gallery`
和 `Embeddings` API，所以下面的注册与匹配流程可以照搬，但它们不检测人脸，也不做
人脸对齐。

识别 head 通过 `onnxruntime` 运行，而基础安装并不带它：

```bash
pip install "libreyolo[onnx]"
```

## 预测

<code-tabs name="predict" />

不去动它的话，`predict()` 会下载并配上默认检测器。`face_detector` 用可调用对象、
LibreYOLO 检测模型或 `FaceDetector` 实例覆盖它，既可以在构造函数上设置，也可以
按次调用设置。`face_boxes` 用你手上已有的检测框绕过检测。在 CLI 上，
`face_detector=` 接受人脸检测器的 `.onnx` 路径或 LibreYOLO 检测器名称。

`model.verify(image_a, image_b)` 是两张图的快捷方式：它对每张图里置信度最高的
人脸求嵌入向量，返回 `{"similarity", "same_person", "threshold"}`。
`model.embed(sources)` 把一张或多张图像里的所有人脸行堆叠成单个 `(N_total, D)`
张量返回。关于输入源、流式处理和结果处理，见[预测](/docs/predict)。

## 数据集格式

注册按每个身份一个文件夹来读取。文件夹名就是身份，里面的每张图像都为这个名字
贡献参考：

```text
people/
  ada/
    1.jpg
    2.jpg
  grace/
    1.jpg
```

`libreyolo enroll` 会遍历这棵目录树，写出一个 `.npz` 人脸库。已存在的库文件是
就地扩展而不是被替换，所以身份可以随着时间陆续加入。人脸库通过嵌入向量维度和
文件指纹绑定到产出它的权重上；用另一个模型来匹配会直接抛出异常，而不是去比较
不兼容的向量空间。

默认情况下每张源图像只贡献一行参考，也就是置信度最高的那张人脸，所以一张背景里
有路人的肖像只会注册它的主体。给 `Gallery.enroll` 传 `select="all"` 可以把返回的
每一行都存下来。

## 训练

这个任务里没有任何家族能在 LibreYOLO 内部训练。`LibreFaceEmbedder.train()` 会
抛出异常：在上游训练一个识别 head，按 ArcFace 约定导出为 ONNX，再按路径加载
文件。

## 验证

这个任务没有数据集验证器，`val()` 会抛出异常，而不是假装有。验证精度用
`model.verify()` 在带标注的图像对上测量，扫一遍 `threshold` 挑出你想要的工作点。
识别精度则通过注册一个人脸库、在留出图像上读取 `result.identities.name` 和
`result.identities.score` 来测量，把 `None` 名字算作拒识。

## 导出

识别 head 本身已经是 ONNX 计算图，没有什么可转换的：`LibreFaceEmbedder.export()`
会抛出异常。直接部署这个 `.onnx` 文件，或者把 LibreYOLO 指向它，让这个家族来处理
检测、对齐和归一化。
