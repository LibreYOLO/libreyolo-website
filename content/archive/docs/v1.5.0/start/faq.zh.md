---
title: 常见问题
seo_title: LibreYOLO 常见问题
description: 对所有 LibreYOLO 模型都适用的那些问题的简短回答：硬件、许可、权重、设备、训练、导出覆盖范围和 CLI。
lead: 回答那些不针对某一个模型家族的问题。任何与具体家族有关的内容都在该家族自己的页面上。
keywords:
  - libreyolo 常见问题
  - libreyolo 需要 gpu 吗
  - libreyolo 许可证
  - libreyolo 权重放在哪
  - libreyolo 命令行
  - libreyolo 离线使用
last_verified: 1.5.0
source_hash: a729b43a6642f2a0
---

## 我该从哪个模型开始？

CNN 检测器选 YOLOv9，transformer 检测器选 RF-DETR。两者都在旗舰梯队里，这意味着新功能会先针对它们设计并在 GPU 上验证，然后才轮到别的模型。参见 [YOLOv9](/docs/models/yolov9) 和
[RF-DETR](/docs/models/rf-detr)，其余的见[全部模型](/docs/models)。

## 我需要 GPU 吗？

不需要。每个模型都能在 CPU 上运行，[快速上手](/docs/quickstart)里的所有内容也都是照着在 CPU 上跑写的。GPU 改变的是训练和视频推理要花多长时间，而不是它们能不能跑。

## LibreYOLO 是怎么选择设备的？

默认是 `device="auto"`：当 PyTorch 报告 CUDA 可用时用 CUDA，其次在 Metal Performance Shaders 可用时用它，否则用 CPU。要固定设备，就把 `device` 传给模型，或者传给 `predict`、`train`、`val` 和 `export`。它接受 `"cpu"`、`"cuda"`、`"cuda:0"`、`"mps"`、像 `0` 这样的裸整数，或者一个数字字符串；后两者会展开成 `cuda:<n>`。

`libreyolo checks` 会打印 Torch 的构建信息、它的 CUDA 和 cuDNN 版本，以及它能看到的每一块 GPU。如果这条命令显示没有 CUDA，说明这个 PyTorch wheel 是 CPU 构建；[安装](/docs/install)讲了怎么把它换掉。

## 下载的权重放在哪里？

放在相对于工作目录的 `weights/` 里。不带目录部分的模型引用会解析到那里，并在首次使用时下载；带目录的引用会被原样使用，永远不会去下载。参见[检查点与权重](/docs/weights)。

## 可以在没有网络的情况下运行吗？

可以。在一台联网的机器上把检查点（checkpoint）取一次，把 `weights/` 目录拷贝过去，之后就不会再有任何网络请求。共享的只读路径同样可行，因为带目录的引用会被按字面处理。数据集解析到 `~/datasets` 下，或者 `LIBREYOLO_DATASETS_DIR` 下。

## 可以把 LibreYOLO 用于商业用途吗？

代码采用 MIT 许可。预训练权重是另一个问题：它们可能继承来源项目或数据集的条款，而且这些条款即使在同一个家族内部也不统一。具体那个 Hugging Face 仓库上的许可证才具有权威性，每个模型页面都有一节许可说明把它复述出来。凡是权重受限的地方，LibreYOLO 会在下载开始之前把限制打印出来。

## 可以加载其他项目的检查点吗？

通常可以，把它的路径传给 `LibreYOLO()` 就行。能识别的上游布局会在加载时转换，保留它们的类别数量和名称，并在源文件旁边写出一个 LibreYOLO 检查点。[导入已有权重](/docs/migrate)讲了哪些能被识别、哪些需要转换脚本。

## 为什么 train 会抛出 NotImplementedError？

因为那个家族只提供推理，异常里也写明了原因。预测、验证以及（在支持的情况下）导出都能用；LibreYOLO 里没有该架构的训练循环。模型页面头部的支持梯队会在你动手之前就告诉你。参见[核心概念](/docs/concepts)。

## val 返回什么？

一个普通字典，不是对象。检测任务的键包括
`metrics/precision`、`metrics/recall`、`metrics/mAP50` 和
`metrics/mAP50-95`。其他任务返回对它们有意义的键，比如分类的 `metrics/accuracy_top1`，或者全景分割的 `metrics/PQ`、`metrics/SQ` 和 `metrics/RQ`。

## 怎么对文件夹、视频或摄像头运行？

把它作为来源传进去。文件路径是一张图片，目录是其中的每一张图片，视频路径是一段视频，整数是摄像头索引，RTSP、RTMP、TCP、UDP 或 HLS URL 是实时流。一个 `.streams` 文件可以一次列出多个来源。实时来源需要 `stream=True`，它每帧产出一个 `Results`，而不是构建一个列表；长视频和大目录也值得用这个参数。只有 YouTube 页面 URL 需要一个额外依赖 `libreyolo[stream]`。

## 怎么只保留部分类别？

把你想要的类别索引通过 `classes` 传给 `predict`，比如
`classes=[0, 2]`。`conf` 设置置信度阈值，默认 `0.25`，`max_det` 限制每张图片的检测数量，默认 `300`。

## CLI 用的是 flag 还是 key=value 对？

每条命令都是键和值用等号连接：

```bash
libreyolo predict model=yolo9-t source=my-image.jpg save=True
libreyolo train model=yolo9-t data=coco8.yaml epochs=50 imgsz=640
```

`model` 接受一个路径，或者 `family-size` 形式的短名称，后面还可以选择性地加上任务后缀，`libreyolo models` 会列出所有有效的名称。诊断和清单类命令还接受 `--json`，它会把同样的数据以机器可读的对象打印到标准输出。

## 每个模型都能导出成每种格式吗？

不能。覆盖范围按家族、按任务而定，并不统一，而且每种格式都有自己要装的额外依赖。每个模型页面都带有其家族的导出矩阵；[导出章节](/docs/export)讲格式本身。

## segment、semantic 和 panoptic 有什么区别？

三个独立的任务。`segment` 为每个检测到的目标生成一个掩码。`semantic` 给每个像素标上一个类别，不区分实例。`panoptic` 给每个像素恰好一个标签，把可数的 things 和无定形的 stuff 合在一起。它们有不同的真值（ground truth）、不同的结果字段和不同的指标，一个家族支持其中的哪几个，取决于它的任务列表里出现了哪些。

## 怎么在自己的类别上训练？

写一个带 `train`、`val` 和 `names` 的数据集 YAML。标注放在图片旁边一个平行的 `labels/` 目录树里，每张图片一个 `.txt`，坐标是归一化的。`nc` 是可选的，出现时必须与 `names` 一致。先运行
`libreyolo doctor <data.yaml>`：它会检查数据集有没有问题，发现错误时以非零状态退出，因此可以当作 CI 门禁使用。

## 为什么加载时会打印元数据警告？

因为这个检查点没有携带完整的 v1.0 元数据。加载会走兼容路径继续进行，警告里会写明到底缺了哪些键。运行 `libreyolo metadata path=<file>` 可以看到里面有什么，schema 的要求见[检查点与权重](/docs/weights)。

## 升级之后某个 import 不能用了，是什么变了？

有两个类名为了保持一致做了重命名：`LibreYOLORTDETR` 变成了
`LibreRTDETR`，`LibreYOLORFDETR` 变成了 `LibreRFDETR`。旧名称仍然能解析，并会发出一个指向新名称的 `DeprecationWarning`，所以已有代码在你更新它的这段时间里还能继续运行。
