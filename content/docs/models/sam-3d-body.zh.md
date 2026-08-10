---
title: SAM 3D Body
families: [sam3dbody]
seo_title: "SAM 3D Body：LibreYOLO 里的全身网格恢复"
description: "在 LibreYOLO 里用 SAM 3D Body 做全身人体网格恢复。安装并跑预测；检查点受 Meta 的 SAM License 限制，且必须有 CUDA。"
lead: "SAM 3D Body 是 Meta 的可提示模型，它从单张图像加人体框恢复出包含手和脚在内的全身 3D 网格。LibreYOLO 封装的是上游的包，而不是把它移植过来。"
keywords: [SAM 3D Body, MHR, Momentum Human Rig, "人体网格恢复", "单图 3d 人体重建", "3d 人体姿态估计 python", "sam 3d body 权重下载"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.sam3dbody import LibreSAM3DBody

        # 这个家族没有注册到 LibreYOLO() 工厂函数里，所以要直接构造。
        # model_path=None 才会触发受限（gated）的 Hugging Face 下载；
        # 传字符串则会被当成已有的本地检查点路径，永远不会自动拉取。
        # 推理需要一个 CUDA 设备，没有 CPU 路径
        model = LibreSAM3DBody(None, size="d3", device="cuda")
        result = model(SAMPLE_IMAGE, person_boxes=[[34, 12, 220, 400]])

        meshes = result.meshes
        print(meshes.vertices.shape)    # (N, V, 3)，相机坐标系，单位为米
        print(meshes.joints3d.shape)    # (N, J, 3)
    - label: 配合人体检测器
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE
        from libreyolo.models.sam3dbody import LibreSAM3DBody

        # 这里没有字符串名称的快捷写法：传一个构造好的 LibreYOLO 检测器、
        # 一个普通的可调用对象，或者一个 PersonDetector 实例
        detector = LibreYOLO("LibreRFDETRn.pt")
        model = LibreSAM3DBody(None, size="d3", device="cuda")

        result = model(SAMPLE_IMAGE, person_detector=detector)
---

## 安装

```bash
pip install libreyolo
```

这样只会装上 LibreYOLO 的适配器。SAM 3D Body 本身没有打包进来，因为它的许可证
不属于 LibreYOLO 自己的代码可以从中衍生的那一类：请自行克隆上游仓库、安装它的
依赖，然后把 LibreYOLO 指向这份克隆。

```bash
git clone https://github.com/facebookresearch/sam-3d-body
pip install roma einops yacs omegaconf braceexpand pytorch-lightning timm
```

```python
from libreyolo.models.sam3dbody import LibreSAM3DBody

model = LibreSAM3DBody(
    None,
    size="d3",
    sam_3d_body_path="/path/to/sam-3d-body",
    device="cuda",
)
```

或者设置 `SAM_3D_BODY_PATH` 环境变量，而不用每次调用都传 `sam_3d_body_path`。
从来不构造这个家族的用户永远不会触发这个 import，也永远不会碰到 SAM License。
这个家族既没有接进 `LibreYOLO()` 工厂函数，也没有接进 `libreyolo predict` CLI
命令；`LibreSAM3DBody` 是唯一的入口。

## 预测

<code-tabs name="predict" />

检查点下载是受限的：首次下载成功之前，需要在 Hugging Face 模型页上接受 Meta 的
许可证，并用 `hf auth login` 完成认证。推理本身则无条件需要一个 CUDA 设备：上游
的估计器不做检查就把这一批数据搬到 GPU 上，所以只有 CPU 的机器会直接报错，而不是
回退。`result.meshes` 是一个 `Meshes` 负载，和 `result.boxes` 按行对齐（检测到的
每个人一行）：`vertices` 和 `joints3d` 以米为单位，并且已经包含估计出的相机平移，
`joints2d` 是原图上的像素坐标，旋转遵循 MHR 的约定，用的是欧拉角而不是轴角。输入
源、流式处理和结果处理见[预测](/docs/predict)。

## 变体

同一个 MHR 人体模型背后有两种骨干：`d3` 用的是 DINOv3 ViT-H/16+ 编码器，`h` 用的
是原始的 ViT-H 编码器。

## 导出

<export-matrix />

人体网格导出还没有实现：LibreYOLO 尚未为网格这个任务定义导出图的约定，其中也包括
在 PyTorch 之外怎么表示 MHR 的参数布局。

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

## 许可证

<provenance-box>

这些检查点驱动的人体模型 MHR（Momentum Human Rig）是 Meta 单独发布的，采用
Apache-2.0 许可。LibreYOLO 在运行时从 MHR 自己的公开发布里拉取它的 TorchScript
资源，并缓存到本地；这个文件不由 LibreYOLO 镜像，适用的是它自己的 Apache-2.0
条款，而不是 SAM License。

</provenance-box>

## 引用

<citation-block />
