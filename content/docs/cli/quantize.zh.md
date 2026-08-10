---
title: libreyolo quantize
seo_title: libreyolo quantize 命令参考
description: 在命令行里量化 PyTorch 检查点：配方、校准参数、默认值，以及每种配方接受的模型家族。
lead: 把模型的浮点模块换成量化模块，在配方需要统计量时用无标注图像做校准，并把结果保存为 PyTorch 检查点（checkpoint）。
keywords:
  - libreyolo quantize cli
  - int8 量化命令
  - fp8 量化
  - 训练后量化 yolo
  - libreyolo quantize 参数
last_verified: 1.5.0
meta:
  - label: 命令
    value: libreyolo quantize
    mono: true
  - label: 必填
    value: model
    mono: true
  - label: 输出
    value: 在后缀前带 -<recipe> 的源路径，例如 LibreYOLO9s-int8.pt
    mono: true
snippets:
  examples:
    - label: 基础
      language: bash
      code: |
        # 在 coco128 上校准，并写出 LibreYOLO9s-int8.pt
        libreyolo quantize model=LibreYOLO9s.pt recipe=int8
    - label: 只转换类型，不做校准
      language: bash
      code: |
        libreyolo quantize model=LibreYOLO9s.pt recipe=fp16 calib=none \
          out=weights/LibreYOLO9s-fp16.pt
    - label: 扩大校准，再恢复精度
      language: bash
      code: >
        libreyolo quantize model=LibreYOLO9s.pt recipe=int8 \
          calib=coco128.yaml samples=256 batch=16 algorithm=minmax

        # 在量化后的检查点上做量化感知训练可以恢复精度

        libreyolo train model=LibreYOLO9s-int8.pt data=coco8.yaml epochs=10
        lr0=0.001
source_hash: 7ae663e9f117826e
---

## 概要

```bash
libreyolo quantize model=<name|path> [recipe=<recipe>] [key=value ...]
```

参数是 `key=value` 形式的键值对，POSIX 形式也可以，所以 `recipe=int8` 和
`--recipe int8` 是同一个参数。

## 参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `model` | | 模型权重 `.pt`。必填 |
| `recipe` | `int8` | 量化配方：`fp16`、`bf16`、`fp8`、`int8`、`w4a16`、`w4a8`、`nvfp4`、`mxfp4`、`int2` |
| `calib` | `coco128.yaml` | 校准图像：一个数据 YAML，或内置数据集的名称。无标注，只做前向。`none` 跳过校准 |
| `samples` | `128` | 校准图像的最大数量 |
| `batch` | `8` | 校准批大小 |
| `algorithm` | `auto` | 激活范围估计：`auto`（会选择 minmax）、`minmax` 或 `percentile` |
| `out` | | 输出检查点路径。默认是在后缀前带 `-<recipe>` 的源路径 |
| `device` | `auto` | 设备 |
| `allow_download_scripts` | `false` | 允许数据集 YAML 下载块中内嵌的 Python |
| `json` | `false` | 向 stdout 输出 JSON |
| `quiet` | `false` | 屏蔽 stderr |
| `help_json` | `false` | 以 JSON 形式打印命令 schema 并退出 |

## 示例

<code-tabs name="examples" />

## 说明

### 哪些家族接受它

量化覆盖四个家族：`yolo9`、`rfdetr`、`birefnet` 和 `feynobg`。其他家族一律以
`quantize_failed` 退出，并附上这份列表。

### 每种配方改动什么

`fp16` 和 `bf16` 是类型转换（cast）。它们只改 dtype，不需要校准，对它们来说
`calib=none` 是正确的设置。

`int8` 和 `fp8` 量化 `Conv2d` 和 `Linear` 模块，所以它们适合卷积家族。

`w4a16`、`w4a8`、`nvfp4`、`mxfp4` 和 `int2` 只量化 `nn.Linear`，因此针对的是
transformer 家族。在 `yolo9` 上请求其中之一会被拒绝并给出解释，而不是悄悄产出
一个未量化的模型，因为在那里 8 比特以下的加速只作用于 GEMM，卷积会留在更高的
精度上。

`int8`、`fp8`、`w4a8` 和 `int2` 需要激活的校准统计量。`int2` 之后还需要训练来
恢复，所以在没有训练器的 `birefnet` 和 `feynobg` 上会被拒绝。

不管用哪种配方，每个家族都会保留一部分模块为浮点：首层、预测 head，以及
YOLOv9 上的 DFL 卷积——它是一个固定的积分期望算子，不能被量化。

### 校准数据不是训练数据

`calib` 指向一小组无标注图像，只做前向，用来推导激活范围。不会拿它做评估，它的
标注也从不读取。默认的 `coco128.yaml` 首次使用时从一个 URL 下载，所以不需要额外
权限；带内嵌 Python 下载脚本的 YAML 则需要 `allow_download_scripts=true`。

`algorithm=percentile` 是可用的，但在 transformer 家族上可能降低精度，这也是
`auto` 选择 minmax 的原因。

### 恢复精度

输出是一个普通的 PyTorch 检查点，所以
[`libreyolo train`](/docs/cli/train) 可以直接接受它。训练一个量化后的检查点就是
量化感知训练；加上 `distill_model=<teacher>` 就变成量化感知蒸馏。

### 输出与退出码

结果会打印保存路径、配方、执行模式、校准是否运行过，以及按类别统计的被替换模块
数量。退出码在成功时是 `0`，模型无法加载时是 `4`，量化或保存失败时是 `5`，其他
运行时失败是 `1`。

相关：[`libreyolo export`](/docs/cli/export)，它会离开 PyTorch，转而写出一个部署
产物。
