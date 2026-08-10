---
title: LibreFaceRec
families: [facerec]
seo_title: "LibreFaceRec：人脸识别与人脸验证"
description: "在 LibreYOLO 里用 LibreFaceRec 做人脸检测、人脸嵌入向量和人脸验证。装好就能预测；嵌入权重采用 Apache-2.0 许可。"
lead: "LibreFaceRec 是 LibreYOLO 的人脸嵌入向量任务：一个人脸检测器负责定位并对齐人脸，一个识别 head 产出经过 L2 归一化的身份嵌入向量，用来做验证或者检索。"
keywords: [LibreFaceRec, "人脸识别 python", "人脸特征向量提取", "两张人脸比对相似度", "人脸识别 onnx", ArcFace]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # librefacerec-* 这类名字不管文件后缀是什么都会路由到这个家族，并在
        # 首次使用时连同默认的人脸检测器一起，从 LibreYOLO 的 Hugging Face
        # 组织下载
        model = LibreYOLO("librefacerec-l.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.embeddings.data.shape)   # (N, D)，已做 L2 归一化
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=librefacerec-l.onnx source=face.jpg
    - label: 人脸验证
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        # 用两张图像中各自最显著的那张人脸、它们 L2 归一化嵌入向量之间的
        # 余弦相似度来做比较
        result = model.verify("person_a.jpg", "person_b.jpg", threshold=0.4)
        print(result["similarity"], result["same_person"])
    - label: 图库检索
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        query = model("query.jpg").embeddings          # 这张图像里的人脸
        gallery = model.embed(["a.jpg", "b.jpg", "c.jpg"])   # (N_total, D)

        # (query_faces, N_total) 的余弦相似度
        scores = query.similarity(gallery)
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")
        model.export(format="onnx")
---

## 安装

LibreFaceRec 的识别 head 跑在 `onnxruntime` 上，而它不属于基础安装的一部分。

```bash
pip install "libreyolo[onnx]"
```

## 预测

<code-tabs name="predict" />

检测和识别是一次调用背后的两张独立 ONNX 计算图：人脸检测器定位每一张人脸，并把
它对齐成一个标准裁剪，识别 head 则为每张人脸返回一个经过 L2 归一化的嵌入向量。
什么都不设时，`predict()` 会自动下载并配上自带的默认检测器。`face_detector` 接受
一个可调用对象、一个 LibreYOLO 检测模型，或者一个 `FaceDetector` 实例；
`face_boxes` 则用你已经有的检测框完全跳过检测这一步。`result.embeddings` 每检测到
一张人脸就存一行，和 `result.boxes` 一一对应；它的 `.similarity()` 方法一次调用就
能算出与另一个嵌入向量、或者与整个图库之间的余弦相似度。如果你要直接比较两张
图像，而不是两个已经算好的嵌入向量，`model.verify(image_a, image_b)` 会对两张图都
跑一遍检测和嵌入，然后比较各自置信度最高的那张人脸。任何其他遵循 ArcFace 约定的
ONNX 识别模型（输入对齐后的裁剪，输出 `(N, D)` 嵌入向量）都可以替换进来，只要传
它的文件路径而不是 `librefacerec-*` 名字。数据源、流式处理和结果处理见
[预测](/docs/predict)。

## 导出

<export-matrix />

LibreFaceRec 封装的本来就是一张预先导出好的 ONNX 计算图；把它再导出成另一种格式
还没有实现。

## 许可证

<provenance-box>

自带的默认人脸检测器是第二份产物，适用的是第二份许可证：OpenCV Zoo 的 YuNet，
MIT，版权归 Shiqi Yu。两个项目的架构代码都没有被移植过来；两张计算图都通过
`onnxruntime` 以不透明的方式调用，所以 LibreYOLO 自己的封装不含任何第三方代码，
从头到尾都是 MIT。

</provenance-box>

## 引用

<citation-block />
