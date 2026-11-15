---
title: InstantHMR
families:
  - instanthmr
seo_title: InstantHMR Python 用法：从一张图像实时重建人体网格
description: >-
  在 LibreYOLO 里用 InstantHMR 为图像中的每个人恢复 3D 人体网格。提供 n、t、s 和 m 尺寸，外加一个采用 Apache-2.0
  许可、可商用的 m 检查点。
lead: >-
  InstantHMR 出自 Naturalpad 和 LAAS-CNRS，是一个实时人体网格重建模型。对每个人体框，它返回 MHR 身体姿态和体型参数、70
  个 2D 和 3D 关键点、相机平移，以及一个 18,439 个顶点的网格。LibreYOLO 用 PyTorch 原生移植了它，并用 RF-DETR
  找出画面里的人。
keywords:
  - InstantHMR
  - 人体网格重建
  - HMR
  - 人体网格 mesh
  - 3D 人体姿态估计
  - MHR
  - Momentum Human Rig
  - 实时
last_verified: 1.8.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreInstantHMRm-mesh.pt")   # 用 LibreRFDETRm.pt 找人
        result = model("people.jpg")

        meshes = result.meshes            # result.boxes 里的每个框对应一行
        print(meshes.body_model)          # "mhr"
        print(meshes.vertices.shape)      # (people, 18439, 3)，单位为米，相机坐标系
        print(meshes.joints2d.shape)      # (people, 70, 2)，像素
    - label: 使用自己的框
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreInstantHMRm-mesh-commercial.pt")
        boxes = [[40, 20, 260, 600]]      # 每个人一组 x1, y1, x2, y2
        result = model("people.jpg", person_boxes=boxes)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 在 YOLO-pose 格式的 COCO-17 数据集上微调一个预训练检查点
        model = LibreYOLO("LibreInstantHMRs-mesh.pt")
        model.train(data="my-pose-dataset.yaml", epochs=20)
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreInstantHMRm-mesh.pt")
        model.export(format="onnx")   # 输入为 image、cliff_cond；面片存在元数据里
source_hash: 458443d8f7f54450
---

## 安装

InstantHMR 不需要任何可选 extra。默认的人体检测器 `LibreRFDETRm.pt` 需要 RF-DETR
的 extra。

```bash
pip install "libreyolo[rfdetr]"
```

## 预测

权重在首次使用时从 LibreYOLO 的 Hugging Face 镜像下载，下载前会显示一次许可证提示。

<code-tabs name="predict" />

人由 RF-DETR 检测器找出（默认是 `LibreRFDETRm.pt`）。传入 `person_boxes=` 可以用你
自己的框，传入 `person_detector=` 可以换掉检测器。`result.meshes` 的各行与
`result.boxes` 一一对应。所有内容都在原图的相机坐标系里：`transl` 以米为单位，
`vertices` 和 `joints3d` 是度量值并已包含 `transl`，`joints2d` 是原图上的像素坐标。
`global_orient`、`body_pose` 和 `betas` 存放 MHR 参数；MHR 的旋转是以弧度表示的欧拉
角。完整载荷见[人体网格](/docs/tasks/body-mesh)任务页。

## 变体

四种尺寸 `n`、`t`、`s` 和 `m`，人体裁剪都是 288 px，外加
`LibreInstantHMRm-mesh-commercial.pt`，这是一个只用可商用图像训练的 `m`。上游只发布
ONNX；LibreYOLO 重建了 PyTorch 检查点，全部五个发布版都与上游 ONNX 计算图的误差在
8e-5 以内。

## 训练

`train()` 在带 COCO-17 人体关键点的 YOLO-pose 数据集上微调预训练检查点，使用 2D
关键点损失、重投影损失和冻结先验损失。不支持从零训练和 `resume`。`val()` 会抛出异常：
常用的人体网格基准（3DPW、EMDB）只有研究用途许可，所以没有内置任何一个；MPJPE、
PA-MPJPE 和 PVE 指标在 `libreyolo.validation.mesh_metrics` 里，可用于你自己持有的
数据集。

<code-tabs name="train" />

## 导出

<export-matrix />

`export(format="onnx")` 写出一个带 `image` 和 `cliff_cond` 输入的计算图，网格面片存在
它的元数据里。

<code-tabs name="export" />

## 检查点

<checkpoint-table />

## 许可证

<provenance-box>

InstantHMR 出自 Naturalpad 和 LAAS-CNRS
（[Naturalpad-studio/InstantHMR](https://github.com/Naturalpad-studio/InstantHMR)）。
代码采用 Apache-2.0 许可。权重的许可证取决于检查点：

- `n`、`t`、`s` 和 `m`：CC BY-NC 4.0，**仅限非商用**，因为它们的部分训练图像
  （SA-1B、AI Challenger）仅限非商用。
- `m-commercial`：Apache-2.0，只用 Harmony4D 和获得商用许可的 COCO 图像训练。

每个检查点（checkpoint）还都用 SAM 3D Body 数据集的标注训练过，这些标注由 Meta 以
SAM License 发布，其条款随权重一并传递。网格解码是以净室方式编写的，所以没有使用任何
采用 SAM 许可的 sam-3d-body 代码。检查点镜像为
`LibreYOLO/LibreInstantHMR{n,t,s,m}-mesh` 和
`LibreYOLO/LibreInstantHMRm-mesh-commercial`，不需要申请访问。

</provenance-box>

## 引用

<citation-block />
