---
title: PicoSAM3
families:
  - picosam3
seo_title: PicoSAM3：LibreYOLO 里的框提示边缘端分割
description: 在 LibreYOLO 里用 PicoSAM3 对边缘端传感器上的区域做框提示分割。安装、预测并导出这个 Apache-2.0 许可的 pico 检查点。
lead: >-
  PicoSAM3 是一个从 SAM 2.1 和 SAM 3 蒸馏出来的紧凑 CNN，专为 Sony IMX500
  这类传感器上的框提示感兴趣区域（ROI）分割而设计。LibreYOLO 通过一个专门的 LibreSAM 工厂函数支持它，与 LibreYOLO()
  检测器工厂函数分开，并且只接受框提示。
keywords:
  - PicoSAM3
  - Segment Anything
  - IMX500
  - 边缘端分割
  - 框提示分割
  - sam 蒸馏小模型
  - imx500 分割
  - 端侧分割模型
last_verified: 1.5.0
snippets:
  predict:
    - label: 框提示
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        # PicoSAM3 只有一个尺寸 "pico"，所以不需要别的别名
        model = LibreSAM("picosam3")

        # bboxes= 是唯一支持的提示：[x1, y1, x2, y2] 或者一组框，每个框对应
        # 一个掩码。每个框会先扩大 10%、变成正方形、裁剪到图像范围内，再缩放
        # 到 96x96，然后才跑 CNN
        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])
        print(result.masks.xy)      # 每个掩码一个多边形
        print(result.boxes.xyxy)    # 由掩码推出的紧致检测框
    - label: 编码一次，多次提示
      language: python
      code: |
        from libreyolo import LibrePicoSAM3, SAMPLE_IMAGE

        model = LibrePicoSAM3()

        # set_image() 缓存的是源图像；PicoSAM3 每个框都要跑一次完整的 CNN
        # 前向，所以这里省下的是图像的加载和解码，而不是像其他 SAM 家族那样
        # 省下一次编码器前向
        model.set_image(SAMPLE_IMAGE)
        a = model.predict(bboxes=[300, 200, 900, 700])
        b = model.predict(bboxes=[100, 100, 400, 400])
        model.reset_image()
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibrePicoSAM3

        model = LibrePicoSAM3()
        model.export(format="onnx", output_path="LibrePicoSAM3pico.onnx")

        # opset（默认 13）和 dynamic（默认 True，仅 batch 轴）是这个家族
        # 接受的仅有的两个导出参数
    - label: 使用导出的文件
      language: python
      code: >
        import numpy as np

        import onnxruntime as ort


        # PicoSAM3 导出的是它原始的 96x96 ROI CNN：roi_image -> mask_logits。

        # 这里没有 LibreYOLO 侧的预处理/后处理可以复用，因为 export() 不像

        # 检测器检查点那样把产物路由回 LibreYOLO()

        session = ort.InferenceSession("LibrePicoSAM3pico.onnx")

        name = session.get_inputs()[0].name

        outputs = session.run(None, {name: np.zeros((1, 3, 96, 96),
        dtype=np.float32)})


        for meta, array in zip(session.get_outputs(), outputs):
            print(meta.name, array.shape)
source_hash: 5d60ff14fe61ba29
---

## 安装

PicoSAM3 需要 `sam` 这个 extra：LibreYOLO 自己的权重下载仍然走 `transformers`
的 Hugging Face 工具，尽管推理跑的是原生的、不依赖 `transformers` 的 CNN。

```bash
pip install "libreyolo[sam]"
```

## 预测

`LibreSAM(...)`（或者家族专用的 `LibrePicoSAM3(...)`）是和 `LibreYOLO(...)`
分开的入口：它返回的是一个可提示（promptable）分割器，而不是检测器，因为在这里
没有提示的一次前向传播毫无意义。这个家族没有 `libreyolo predict` CLI 命令；请用
Python API。

<code-tabs name="predict" />

PicoSAM3 只接受 `bboxes=`；传 `points=`、`labels=`、`masks=`、`text=`、
`multimask=True`，或者省略框来分割整张图，都会抛出一个清晰的 `ValueError`，因为
上游模型里根本没有这些模式。`conf` 按预测出的掩码质量（IoU）过滤，而不是检测置信
度，取值必须在 `0.0` 和 `1.0` 之间。每个掩码的类别 id 都是 `0`，名字是
`"object"`。`train()`、`val()` 和 `track()` 会抛出 `NotImplementedError`；点提示、
文本提示、掩码提示或者分割一切，请用 LibreSAM2 或 LibreSAM3。输入源类型见[预测](/docs/predict)。

## 变体

只有一个尺寸 pico，ROI 输入固定为 96 px：PicoSAM3 每个框都跑一次完整的 CNN
前向，而不是把整张图编码一次。

## 导出

<export-matrix />

PicoSAM3 是 SAM 这一档里唯一能导出的家族：它把原始的 96x96 ROI CNN 导成 ONNX，
`roi_image -> mask_logits`，里面没有内嵌 NMS，也没有掩码后处理。其他 SAM 家族在
`export()` 上都会抛出 `NotImplementedError`，因为它们的编码器/解码器拆分还没有
定义好的运行时导出约定。导出的 PicoSAM3 计算图不能通过 `LibreYOLO()` 加载回来；
请直接用 `onnxruntime` 这类运行时去跑，并施加上面展示的同一套扩 10% 的方形 ROI
预处理。

<code-tabs name="export" />

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

## 许可

<provenance-box>

PicoSAM3 是以 SAM 2.1 和 SAM 3 作为教师模型蒸馏出来的。LibreYOLO 在这个家族里
既不打包也不再分发任一教师模型的代码或权重；随包提供的只有紧凑的学生 CNN 和它
转换后的检查点。

</provenance-box>

## 引用

<citation-block />
