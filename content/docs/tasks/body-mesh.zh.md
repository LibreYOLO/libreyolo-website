---
title: 人体网格
seo_title: LibreYOLO 中的人体网格重建
description: 在 LibreYOLO 中为每个人重建参数化 3D 人体网格。从人体检测框或检测器出发做预测，并读取顶点、关节和相机平移。
lead: 人体网格重建把一张图像和一组人体检测框变成每个人的参数化 3D 人体：形状与姿态参数、带姿态的顶点、3D 关节，以及把它们摆在镜头前的相机平移。
keywords:
  - 人体网格重建 python
  - 3d 人体网格
  - 单目 3d 人体姿态
  - SAM 3D Body
  - MHR
  - 参数化人体模型
  - libreyolo mesh
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.sam3dbody import LibreSAM3DBody

        # 这个家族没有注册到 LibreYOLO() 工厂，因此直接构造；
        # model_path=None 会触发受限的 Hugging Face 下载，传入字符串
        # 则视为已存在的本地检查点，永远不会去下载，
        # 推理需要 CUDA
        model = LibreSAM3DBody(None, size="d3", device="cuda")
        result = model(SAMPLE_IMAGE, person_boxes=[[34, 12, 220, 400]])

        meshes = result.meshes
        print(meshes.body_model)      # 这些张量使用的参数化方式
        print(meshes.vertices.shape)  # (N, V, 3)，相机坐标系，单位米
        print(meshes.joints3d.shape)  # (N, J, 3)
        print(meshes.joints2d.shape)  # (N, J, 2)，源图像上的像素坐标
    - label: 搭配人体检测器
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE
        from libreyolo.models.sam3dbody import LibreSAM3DBody

        # person_detector 接受已构造的 LibreYOLO 检测器、普通可调用对象，
        # 或者 PersonDetector 实例，没有名称快捷方式
        detector = LibreYOLO("LibreYOLO9s.pt")
        model = LibreSAM3DBody(None, size="d3", device="cuda")

        result = model(SAMPLE_IMAGE, person_detector=detector)
source_hash: 31c5b44171cbcd0e
---

## 定义

人体网格重建为每张图像返回一份 `Meshes` 负载，与 `result.boxes` 按行对齐：第
`i` 行描述第 `i` 个检测框里的那个人，和姿态任务对关键点采用的约定完全一样。

所有量都表达在原图的相机坐标系下。`transl` 是公制的，单位为米，+z 指向远离相机
的方向。`vertices` 和 `joints3d` 同样是公制的，并且已经包含了 `transl`，不需要
再做任何合成。`joints2d` 是原始图像画布上的像素坐标，而不是网络实际看到的那张
裁剪图上的坐标。`faces` 保存网格拓扑，对整张图像只存一份而不是每行一份，因为所
有人共用同一套拓扑。这个版本里没有世界坐标系或重力坐标系，也没有哪个字段会悄悄
充当它。

不同人体模型的参数布局不一样，所以形状没有任何一处是固定的：`body_model` 给出
参数化方式的名字，具体数量从张量里读回来。对于 `"mhr"`，也就是 Momentum Human
Rig，旋转是以弧度表示的欧拉角而不是轴角，`body_pose` 是一个扁平的逐关节参数向量
而不是每个关节一个三元组，`betas` 是身份混合形状（blendshape）系数。骨架尺度、
手部姿态和面部表情放在 `extras` 里。

规范的任务键是 `mesh`。`body-mesh`、`hmr` 和 `human-mesh-recovery` 都会归一化到
它。

## 模型

[SAM 3D Body](/docs/models/sam-3d-body) 是唯一服务这个任务的家族，而且它是一层
封装而不是移植：Meta 的 `sam-3d-body` 包以 SAM License 发布，LibreYOLO 自己的
代码不能从它派生，所以其中没有任何一行被收进仓库。两个骨干共用同一个 MHR 人体
模型，`d3` 用 DINOv3 ViT-H/16+ 编码器，`h` 用原始的 ViT-H。

第一次预测之前有三个前提条件，没有一个是可选的。

上游包要由你自己安装，不是由 LibreYOLO 安装：

```bash
git clone https://github.com/facebookresearch/sam-3d-body
pip install roma einops yacs omegaconf braceexpand pytorch-lightning timm
```

用 `sam_3d_body_path=` 或者 `SAM_3D_BODY_PATH` 环境变量把库指向这份克隆。从不
构造这个家族的用户，也就永远不会触发这次导入。

检查点（checkpoint）镜像是受限的。要在 Hugging Face 模型页上接受许可，并用
`hf auth login` 完成认证，否则第一次下载就会失败。MHR 人体模型本身是单独的
Apache-2.0 发布，从它自己的公开地址获取，并缓存在本地。

推理需要 CUDA 设备。上游的估计器不做检查就把这一批数据搬到 GPU 上，所以没有可以
回退的 CPU 路径，`device="cpu"` 会抛异常。

## 预测

<code-tabs name="predict" />

人有两条路径进到模型里。`person_boxes` 传入你已经拿到的检测框，只适用于单张
图像：一组固定的检测框没法跟着人在视频帧之间移动，所以把它和视频源一起传会抛
异常，而不是悄悄复用第一帧的检测框。`person_detector` 接受已构造的 LibreYOLO
检测器、一个可调用对象，或者一个 `PersonDetector`，这是处理视频的路径。
`focal_length` 用来提供已知的相机内参；不设置时，模型会用自己的估计值，
`meshes.focal_length` 报告的就是这个值。

这个家族没有接入 `LibreYOLO()` 工厂，也没有接入 `libreyolo predict` CLI 命令。
`LibreSAM3DBody` 是唯一的入口。数据源、流式处理和结果处理见[预测](/docs/predict)。

## 训练

这个任务里没有任何家族能在 LibreYOLO 内部训练。`LibreSAM3DBody.train()` 会抛
异常：请到上游项目训练，再把得到的检查点加载到这里。

## 验证

没有网格验证器，`val()` 会抛异常。常用的基准测试都只提供研究许可，所以既没有
内置任何一个，也没法替你下载。

指标本身以 `libreyolo.validation.mesh_metrics` 的形式提供，可以拿来在你已经
持有的数据集上做评估。它接受预测关节和目标关节，可选地接受预测顶点和目标顶点，
返回一个键名和验证器完全一致的字典：

`metrics/mpjpe` 是对齐根关节之后的平均每关节位置误差，所以它衡量的是姿态，而
不管人站在场景里的什么位置。`metrics/pa_mpjpe` 是在完整的 Procrustes 对齐，
也就是旋转、统一缩放和平移之后的同一个量，它去掉了全局朝向和身材尺寸带来的
误差，只留下关节姿态。`metrics/pve` 是按顶点质心对齐之后、网格表面上的平均每
顶点误差；和关节指标不同，它对体型敏感，而且只有在两组顶点数组都提供时才会
出现。三个指标都是越低越好。输入假定是公制的，单位为米，`scale_to_mm` 把结果
换算成文献里报告的毫米。

## 导出

网格导出没有实现。LibreYOLO 还没有为这个任务定义导出图的元数据约定，包括如何在
PyTorch 之外承载 MHR 的参数布局，所以 `export()` 会抛异常，而不是产出一张输出
无法解释的图。
