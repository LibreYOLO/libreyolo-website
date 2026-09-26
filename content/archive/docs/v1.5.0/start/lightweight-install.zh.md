---
title: 轻量安装
seo_title: 不装 PyTorch 跑 LibreYOLO 的 ONNX 推理
description: >-
  用 --no-deps 安装 LibreYOLO，只靠 numpy 跑 ONNX 检测，磁盘上没有
  torch。这个做法本身、它的边界，以及确切的软件包清单。
lead: >-
  LibreYOLO 的 ONNX 推理路径从头到尾都是 numpy，包括解码和 NMS。这条路径上没有任何东西在运行时需要
  PyTorch，所以一次跳过依赖解析的安装，可以在机器上没有 torch 的情况下跑检测。
keywords:
  - 不装 torch 推理
  - 无 torch 推理
  - libreyolo 不装 pytorch
  - onnx 推理 不用 torch
  - libreyolo 轻量安装
  - pip install no-deps
  - libreyolo 占用磁盘空间
  - onnxruntime 推理
last_verified: 1.5.0
meta:
  - label: 适用范围
    value: ONNX 检测，七个模型家族
  - label: 入口
    value: libreyolo.backends.onnx.OnnxBackend
    mono: true
  - label: 支持级别
    value: 尽力而为（best effort），不是单独的发行版
snippets:
  install:
    - label: 轻量
      language: bash
      code: |
        # 安装这个包但不带它的依赖列表，然后自己补上 ONNX 检测路径
        # 实际会导入的那四个包
        pip install --no-deps libreyolo
        pip install numpy pillow opencv-python-headless onnxruntime
    - label: 仅 CPU 的 torch
      language: bash
      code: |
        # 先试这个，它保留全部功能，并避开 CUDA wheel，
        # 磁盘占用主要就出在那里
        pip install libreyolo --index-url https://download.pytorch.org/whl/cpu
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo.backends.onnx import OnnxBackend


        model = OnnxBackend("libreyolo9t.onnx")

        result =
        model.predict("https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg")


        # 这里的 xyxy 是 numpy ndarray，不是 torch 张量

        print(result.boxes.xyxy)

        print(result.boxes.conf)

        print(result.boxes.cls)
source_hash: e60e83d32d13026e
---

## 这为什么可行

`pip install --no-deps libreyolo` 会安装这个包，并完全跳过它的依赖列表。没有任何
东西替你解析，你要自己负责装上你实际用到的那些。

只有当你想走的代码路径确实不需要你跳过的那些依赖时，这才有用，而 ONNX 检测正是
如此。解码，包括非极大值抑制在内，是 numpy 写的。预处理配方是 numpy 写的。
PyTorch 是训练和 eager 推理的依赖，在这条路径上它从来不会被调用。

在这个版本之前，导入本身就会失败：导入 `libreyolo.models` 下的任何东西，都会构建
每一个模型类，用来填充检查点（checkpoint）自动识别的注册表（registry），而这些类
是 `torch.nn.Module` 的子类。预处理配方现在住在自己的包里，`libreyolo.preprocess`，
torch 的导入被推迟到有东西真的碰到某个 torch 属性时才发生，所以在机器上没有 torch
的情况下，ONNX 路径也能完成导入。那个包里为每个家族放了一个 numpy 原生的预处理
器：`yolo9`、`yolonas`、`yolox`、`ec`、`rtdetr`、`rfdetr`、`dfine`、`deim` 和
`deimv2`，比下文端到端验证过的那七个家族多两个。每个
`libreyolo/models/<family>/utils.py` 都从它那里重新导出，所以现有的导入路径仍然
可用。

## 先试试仅 CPU 的 wheel

大多数来问这件事的人，想避开的是一次几个 GB 的安装，而体积集中在一个地方：默认的
`torch` wheel 打包了 CUDA。仅 CPU 的构建只有它的一小部分，而且不需要什么特殊的安装
方式。

<code-tabs name="install" />

仅 CPU 这个选项保留了 LibreYOLO 的每一项功能：训练、验证、每一种任务、每一个家族、
CLI。只有当你想让机器上一点 torch 都没有，而不只是少一点时，才走轻量这条路。

## 轻量安装覆盖什么

| | |
|---|---|
| 任务 | 检测 |
| 格式 | ONNX |
| 入口 | `OnnxBackend` |
| 接口 | Python 库 |

这条路径上验证过七个家族：[YOLOv9](/docs/models/yolov9)、
[YOLO-NAS](/docs/models/yolo-nas)、[EdgeCrafter](/docs/models/edgecrafter)、
[RT-DETR](/docs/models/rt-detr)、[RF-DETR](/docs/models/rf-detr)、
[D-FINE](/docs/models/d-fine) 和 [DEIM](/docs/models/deim)，每个家族的各个变体都
算在内。

这是验证过的范围，不是库强制执行的边界。其他任务和其他家族只是不在检查过的范围
里：有些在你调用时会把 torch 拉进来，也有少数可能碰巧能跑。这份清单之外的东西，
都按尚未验证看待，而不是按受支持或按坏掉看待。

在清单之内，结果和正常安装是完全一致的，而不只是接近。每个家族都导出成 ONNX 并跑了
两次，一次正常跑，一次在 torch 被屏蔽的情况下跑；检测框、分数和类别完全对得上。
测试套件里有一个一致性测试，防止这个约定漂移。

## 五个容易踩的坑

**用 `OnnxBackend`，不要用模型类。** `LibreYOLO9("model.onnx")` 仍然需要 torch，
因为 `LibreYOLO9` 本身就是 `nn.Module` 的子类。这是最容易犯的错误，因为这套文档里
其他每一页都是通过模型类或者 `LibreYOLO()` 来加载模型的。

**在别的地方导出。** 生成 `.onnx` 文件需要 torch，所以轻量的那台机器造不出来。在
开发机或者 CI 机器上导出，再把产物送到精简的目标机上。

**Results 里装的是 numpy 数组。** 这里的 `result.boxes.xyxy` 是一个 `ndarray`。
容器两种类型都接受，所以属性名没有变化，但对结果调用 `.cpu()` 或 `.numpy()` 的
代码会失败。

**单张图片返回一个 `Results`。** `predict()` 对一张图片返回一个 `Results`，对多张
返回一个列表。用 `[0]` 索引单个结果，取到的是第一个检测框，而不是第一张图片，
这会悄无声息地给你一个只有一个框的结果，而不是报错。

**CLI 用不了。** `typer` 和 `click` 不在这四个包里，所以 `libreyolo` 命令不可用。
这是一次库的安装。

## 预测

<code-tabs name="predict" />

把 `onnxruntime` 换成 `onnxruntime-gpu` 就能跑在 CUDA 上。这四个包是一次完整的、
不带 torch 的 `predict()` 实际导入的那些，是在调用过程中记录下来的，而不是推想
出来的。`opencv-python-headless` 顶替了声明里的 `opencv-python`：同一个模块，不带
GUI 库，占的磁盘更小。

在其余声明的依赖里，`requests` 只在从 URL 加载图片时才需要，`pycocotools` 和
`scipy` 用于验证和评估，`typer` 和 `click` 是 CLI。

## 这份清单会漂移，这是设计如此

上面那份软件包清单，对应的是本页顶部标明的那个版本。`--no-deps` 让你退出了依赖
解析，所以没有任何东西替你检查它，后续的版本也可能导入这里没有列出的东西。

如果你碰到 `ModuleNotFoundError`，你已经明白这套做法了：把缺的包装上。这就是预期
的维护方式，而不是一份缺陷报告。这条路径属于尽力而为，不是单独受支持的发行版，
这也是为什么 PyPI 上没有第二个轻量包，也没有做一个的计划。

要确认你的环境真的没有 torch，而不是悄悄回退到某个已经装好的副本，用断言检查
一下：

```python
import importlib.util

assert importlib.util.find_spec("torch") is None, "torch is installed"
```

对精简镜像来说，这个检查值得留在 CI 里。没有它，一个碰巧装了 torch 的环境会通过
每一项测试，什么也告诉不了你。
