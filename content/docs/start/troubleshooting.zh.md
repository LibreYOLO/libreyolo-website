---
title: 疑难排查
seo_title: 修复 LibreYOLO 常见报错
description: LibreYOLO 最常抛出的报错、每一条的含义，以及修复办法。还包括两种以前不报错的故障——取下标和指标——以及 1.8.0 对它们的改动。
lead: 报错按你看到的信息分组。最后两条讲的是以前跑得通却返回错误结果的代码，以及 1.8.0 改了什么。
keywords:
  - libreyolo 报错
  - libreyolo modulenotfounderror
  - libreyolo cuda 显存不足
  - libreyolo notimplementederror
  - libreyolo 问题排查
last_verified: 1.6.0
source_hash: e5da0ce267fbec05
---

报错按你看到的文字分组。如果你的报错信息不在这里，[FAQ](/docs/faq) 回答的是那些
不属于故障的问题，而 `libreyolo models` 会告诉你当前这套安装到底能加载什么。

## ModuleNotFoundError 指向一个你从未导入的包

有些家族需要一个可选 extra。报错信息给出的是缺失的包名，而不是 extra 名，所以从
traceback 里不一定能直接看出该怎么修。

运行 `libreyolo models`。凡是依赖缺失的家族，都会连同启用它所需的确切 pip 命令
一起打印出来，你不必自己把包名反推回 extra。`libreyolo models --json` 会把同样的
内容打印成一个对象。

RF-DETR 和 DINOv2 需要 `libreyolo[rfdetr]`，它们的报错信息会点出这个 extra。
没有它的话，`LibreYOLO("LibreRFDETRn.pt")`、DINOv2 的权重名称和
`libreyolo predict model=rfdetr-n` 都会带着安装命令中止，CLI 则以
`model_load_failed` 退出：

```
ModuleNotFoundError: RF-DETR support requires extra dependencies.
Install with: pip install libreyolo[rfdetr]
```

[安装页面](/docs/install)列出了每一个 extra 以及它覆盖的范围。

## ONNX inference requires onnxruntime

```
ImportError: ONNX inference requires onnxruntime. Install with: pip install onnxruntime
```

基础包不依赖任何运行时，因为你需要哪一个取决于你的硬件。CPU 上装 `onnxruntime`，
CUDA 上装 `onnxruntime-gpu`。两者提供的是同一个 `onnxruntime` 模块，所以只装一个，
不要两个都装。

## ONNX model not found

```
FileNotFoundError: ONNX model not found: <path>
```

路径是相对于工作目录解析的，不是相对于脚本。导出悄悄写到了别的地方时也会出现这
一条：`export()` 会返回它实际写入的路径，所以要接住返回值，而不是假定一个文件名。

## train() 抛出 NotImplementedError

不是每个家族都能训练。有些只移植了预测、验证和导出，它们的 `train()` 会直接抛
异常，而不是假装在跑。

[FAQ 里的条目](/docs/faq)解释了这么做的理由。要在动手写训练脚本之前确认某个具体
家族，它的模型页面会写明是否支持训练。

## export() 抛出 NotImplementedError

一个家族可以支持某项任务，却仍然不支持把它导出。EoMT 是大家最常撞上的例子：
`export()` 接受语义任务，但对 `segment` 和 `panoptic` 会抛异常，因为它们需要的
query-mask 运行时契约还没有定义。

```
NotImplementedError: LibreEoMT instance and panoptic export need query-mask runtime contracts.
```

每个家族的页面都带有一张导出矩阵，标明哪些任务与格式的组合经过了验证。

## CUDA out of memory

先降 `batch`，再降 `imgsz`。两者对显存的占用大致都随自身大小变化，但 batch 是你
可以往下调而不改变模型所见内容的那一个。

如果失败发生在验证阶段而不是训练阶段，验证会用它自己的批大小，所以那个也要调低。

在 Windows 上，负责显示的 GPU 还有第二种故障模式，它看起来像是随机的 CUDA 报错，
而不是显存不足：驱动会重置一块停止响应时间超过超时阈值的 GPU，把当时正在跑的东西
一并杀掉。跑在驱动你显示器那块卡上的长核函数就可能触发它。

## 权重下载不下来

权重在首次使用时从 Hugging Face 拉取，并缓存到本地。[FAQ](/docs/faq) 讲了缓存放在
哪里，以及如何完全离线运行。

如果下载返回 404，检查一下你传进去的文件名。URL 是由它推导出来的，包括任务后缀，
所以一个跟已发布检查点（checkpoint）对不上的名字，会生成一个并不存在的 URL。每个
模型页面上的检查点表格列出了已发布的确切文件名。

## 在脚本顶层调用了 train() 或 val()

```
train() is called at the top level of your script, and DataLoader workers re-run that script on this platform, so it runs with workers=0. Put the call under `if __name__ == "__main__":` to load data in parallel.
```

Windows 没有 `fork`，macOS 上的 Python 默认也不使用它，所以 dataloader 的 worker
是靠重新导入你的脚本来启动的。`spawn` 和 `forkserver` 启动方式在任何平台上都是
如此。没有 `if __name__ == "__main__":` 这层保护，每个 worker 都会把你的 `train()` 或
`val()` 调用再跑一遍。从 1.7.1 起，LibreYOLO 会检测到这种情况，用 `workers=0` 运行该调用，
并只记录一次这行信息。脚本能跑完，但数据加载在主进程里进行，速度更慢。

把调用放进这层保护，就能恢复并行数据加载：

```python
def main():
    ...  # 构建模型并调用 train()

if __name__ == "__main__":
    main()
```

加了保护的脚本、notebook、CLI 以及 Linux 的 `fork` 都会保留各自的 worker。

## 1.8.0 改掉的两种静默故障

本页其余部分讲的都是报错。这两种以前更糟：代码照跑，还递回来一个看着没问题的东西。
从 1.8.0 起，两者的行为都与 Ultralytics 一致，所以升级后脚本给出的数字不一样时，先检查这两处。

### 对单个结果取下标

`predict()` 对一张图片返回一个 `Results`，对多张图片返回一个列表。从 1.8.0 起，
在单图的 `Results` 上用整数 `0` 或 `-1` 取下标会返回整个结果，所以照着列表形式写的
代码在两种情况下都能用：

```python
result = model.predict("image.jpg")   # 一个 Results
result.boxes                          # 全部检测结果
result[0].boxes                       # 全部检测结果，同一个 Results
result[2:3].boxes                     # 只有第三个检测结果
result.boxes[2]                       # 同一个检测结果
```

其他整数（`result[2]`）仍然选中一个检测结果，并发出一条点名 `result[2:3]` 或
`result.boxes[2]` 的 `DeprecationWarning`。切片、布尔掩码和索引数组照旧选中检测结果。
在 1.8.0 之前，`result[0]` 会悄悄地只返回第一个检测结果，所以在 1.7 上跑过的代码在
1.8 上会报出更多检测框。

### 把指标当属性读

`val()` 返回的是一个以指标名为键的字典，从 1.8.0 起，它的 `box` 属性也给出了主要的
检测指标：

```python
metrics = model.val(data="coco8.yaml")
metrics["metrics/mAP50-95"]   # 字典键
metrics.box.map               # 同一个值
metrics.box.map50, metrics.box.map75
metrics.box.mp, metrics.box.mr  # 平均查准率和查全率
metrics.box.maps              # 按类别索引的 AP50-95
```

分割任务会额外提供 `metrics.seg`，其中的 `map`、`map50`、`map75` 和 `maps` 针对掩码；
检测结果还带有 `metrics.confusion_matrix`。字典的键带有 `metrics/` 和 `speed/` 命名空间前缀。先把字典打印一次，看看你的任务产出了
什么，因为这个集合因任务而异。

## 训练前先检查数据集

大多数训练失败其实是数据集的问题。`libreyolo doctor data.yaml` 会对一个检测数据集
跑一遍健康检查，并按严重程度报告发现的问题，这比从第一轮开始读 traceback 快得多。

```python
from libreyolo import doctor

report = doctor.diagnose("data.yaml", imgsz=640)
if report.errors:
    ...
```

检查目录见 [doctor 命令](/docs/cli/doctor)。

## 持久 worker 无法看到数据集变更

启用的 `close_mosaic` 或 `set_epoch` 钩子必须作用于 worker 使用的数据集副本。如果自定义多 worker 加载器保留了不兼容的持久副本，请使用 `persistent_workers=False`，或在修改后重建 worker。默认的非持久路径不受影响。
