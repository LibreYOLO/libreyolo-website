---
title: 视线估计
seo_title: LibreYOLO 中的视线估计
description: 在 LibreYOLO 中估计每张人脸的视线俯仰角与偏航角。用 Python 或 CLI 预测，读取弧度制的角度，并把视线 head 导出为 ONNX。
lead: >-
  视线估计为图像中的每一张人脸返回一个注视方向。LibreYOLO 把它建模成一个两阶段任务：先跑人脸检测器，再由视线 head
  从它返回的每个人脸裁剪图中读出俯仰角（pitch）和偏航角（yaw）。
keywords:
  - 视线估计 python
  - 眼动追踪
  - pitch yaw 视线角度
  - L2CS-Net
  - 视线方向估计
  - 头部姿态估计
  - libreyolo 视线任务
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 不给 face_detector 时，预测会回退到 OpenCV 自带的
        # 检测器，所以除了检查点之外不会下载任何东西
        model = LibreYOLO("LibreL2CSr50.pt")
        result = model(SAMPLE_IMAGE)

        gaze = result.gaze
        print(gaze.pitch, gaze.yaw)              # 弧度，每张人脸一行
        print(gaze.pitch_deg, gaze.yaw_deg)      # 同样的角度，换成度数
        print(gaze.direction_3d)                 # (N, 3) 单位向量
    - label: CLI
      language: bash
      code: >
        # 与 Python 路径不同，CLI 没有自动回退：视线模型必须

        # 显式给出人脸检测器，而且它必须是一个输出人脸检测框的

        # LibreYOLO 检测器

        libreyolo predict model=LibreL2CSr50.pt source=photo.jpg
        face_detector=face-detector.pt save=True
    - label: 选择人脸来源
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreL2CSr50.pt")

        # 把你已经跑过的检测器给出的检测框交给视线 head
        result = model(SAMPLE_IMAGE, face_boxes=[[34, 12, 90, 80]])

        # 或者指定一个内置的检测器
        result = model(SAMPLE_IMAGE, face_detector="yunet")
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreL2CSr50.pt")
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreL2CSr50.pt format=onnx
source_hash: 22aa3c3d87b0c730
---

## 定义

视线估计为每张人脸返回两个角度。`result.gaze` 是一个形状为 `(N, 2)` 的 `Gaze`
载荷，第 0 列是俯仰角，第 1 列是偏航角，单位是弧度，逐行与 `result.boxes`
（检测到的人脸检测框）对齐。这里的约定沿用 L2CS-Net 的那一套：偏航角为正表示
视线转向被摄者的左侧，俯仰角为正表示视线向下。

同一个载荷还通过 `pitch_deg` 和 `yaw_deg` 给出度数，以及 `direction_3d`——相机
坐标系下的一个 `(N, 3)` 单位向量，列为 `(x, y, z)`。

因为这个任务是两阶段的，一次预测依赖两个模型。检测器漏掉的人脸不会有对应的视线
行，而它放偏的检测框会让角度来自一张裁剪糟糕的人脸。任务的规范键是 `gaze`；
`gaze-estimation` 会归一化到它。

## 模型

[L2CS-Net](/docs/models/l2cs) 是唯一服务这个任务的家族。它把一个 ResNet 主干和
两个并行的角度分箱分类 head 配在一起，一个负责俯仰角，一个负责偏航角，输入是
448x448 的人脸裁剪图。架构上支持五种骨干深度，其中一种，也就是 ResNet-50，有
公开的检查点（checkpoint）。

这些权重带有许可限制。它们在 Gaze360 上训练，而 Gaze360 的许可只允许研究和非
商业用途，并且禁止再分发，所以 LibreYOLO 不为这个家族镜像任何东西。库能自动
获取的那唯一一个检查点，是在打印出许可条款之后，通过 `gdown` 直接从作者自己的
Google Drive 分发处拉取的。部署之前请先读[L2CS-Net](/docs/models/l2cs)。

这条下载路径需要 `gaze` 这个 extra：

```bash
pip install "libreyolo[gaze]"
```

没有它，库会打印手动下载说明，而不去尝试传输。对你已经拿到的检查点做预测和
导出，完全不需要任何 extra。

## 预测

<code-tabs name="predict" />

人脸来源有三种指定方式。`face_boxes` 传入你已经算好的检测框，跳过检测。
`face_detector` 接受 `"auto"`、`"haar"`、`"yunet"`、一个 LibreYOLO 检测模型，
或者一个普通的可调用对象，可以在构造函数上设置，也可以每次调用时设置。在
Python 里不设置时，预测会回退到 OpenCV 自带的检测器，所以一次裸调用不接任何东西
也能跑。在 OpenCV 4 上那是 wheel 里自带的 Haar 级联，完全不需要下载；在
OpenCV 5 上，Haar API 已被移除，用的是 YuNet，它会从 OpenCV zoo 一次性拉取一个
小模型文件。

CLI 不共享这个回退。`libreyolo predict` 会拒绝没有 `face_detector=` 的视线模型，
而它接受的值是一个 LibreYOLO 检测器名称或检查点路径。关于输入源、流式处理和结果
处理，见[预测](/docs/predict)。

## 训练

这个任务里没有任何家族能在 LibreYOLO 内部训练。`LibreL2CS.train()` 会抛出异常：
请到上游的 L2CS-Net 项目训练，然后把得到的 state dict 加载到这里。

## 验证

针对视线真值（ground truth）数据集的验证不在范围内，`val()` 会抛出异常，而不是
返回它并没有算过的指标。这个任务没有 `metrics/` 字典。请在上游评估，在这个检查点
所面向的数据集上做。

## 导出

<code-tabs name="export" />

视线的导出契约覆盖 ONNX、TorchScript、ExecuTorch、TensorRT 和 OpenVINO。离开库
的只有 ResNet 主干和那两个角度分箱 head：图接受一张预处理好的 448x448 人脸裁剪
图，返回原始的偏航角和俯仰角 logits。人脸检测、裁剪、softmax、分箱期望以及到
角度的转换全都留在 Python 里，在 `libreyolo.models.l2cs.utils` 中。关于这些格式
及其参数，见[导出](/docs/export)。
