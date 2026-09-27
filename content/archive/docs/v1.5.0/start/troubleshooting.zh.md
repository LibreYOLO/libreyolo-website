---
title: 疑难排查
seo_title: 修复 LibreYOLO 常见报错
description: LibreYOLO 最常抛出的报错、每一条的含义，以及修复办法。还包括两种不抛异常、直接给出错误输出的故障。
lead: 报错按你看到的信息分组。最后两条讲的是相反的问题：代码跑得通，返回的东西看着也合理，但它是错的。
keywords:
  - libreyolo 报错
  - libreyolo modulenotfounderror
  - libreyolo cuda 显存不足
  - libreyolo notimplementederror
  - libreyolo 问题排查
last_verified: 1.5.0
source_hash: e271ab29b789865a
---

报错按你看到的文字分组。如果你的报错信息不在这里，[FAQ](/docs/faq) 回答的是那些
不属于故障的问题，而 `libreyolo models` 会告诉你当前这套安装到底能加载什么。

## ModuleNotFoundError 指向一个你从未导入的包

有些家族需要一个可选 extra。报错信息给出的是缺失的包名，而不是 extra 名，所以从
traceback 里不一定能直接看出该怎么修。

运行 `libreyolo models`。凡是依赖缺失的家族，都会连同启用它所需的确切 pip 命令
一起打印出来，你不必自己把包名反推回 extra。`libreyolo models --json` 会把同样的
内容打印成一个对象。

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

## Windows 上训练卡住或反复重启

Windows 没有 `fork`，所以 dataloader 的 worker 是靠重新导入你的脚本来启动的。没有
`if __name__ == "__main__":` 这层保护，每个 worker 都会把你的训练调用重跑一遍，
结果要么死锁，要么无休止地派生进程。

```python
def main():
    ...  # 构建模型并调用 train()

if __name__ == "__main__":
    main()
```

设 `workers=0` 也能绕开它，代价是吞吐下降。加上这层保护才是更好的修法。

## 两种不抛异常的故障

本页其余部分讲的都是报错。这两种更糟，因为代码照跑，还递回来一个看着没问题的东西。

### 对单个结果取下标

`predict()` 对一张图片返回一个 `Results`，对多张图片返回一个列表。对单图的返回值
取下标，选中的是一个*检测结果*，而不是一张图片：

```python
result = model.predict("image.jpg")   # 一个 Results
result.boxes                          # 全部检测结果，正确
result[0].boxes                       # 只有一个检测结果，而且悄无声息
```

什么都不会抛出，因为对 `Results` 取下标是一个返回子集的合法操作。照着列表形式写的
代码会悄悄地每张图片只报一个检测框。只对你确知是列表的东西取下标。

### 把指标当属性读

`val()` 返回的是一个以指标名为键的普通字典，而不是一个支持属性访问的对象：

```python
metrics = model.val(data="coco8.yaml")
metrics["metrics/mAP50-95"]   # 正确
metrics.box.map               # AttributeError
```

键带有 `metrics/` 和 `speed/` 命名空间前缀。先把字典打印一次，看看你的任务产出了
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
