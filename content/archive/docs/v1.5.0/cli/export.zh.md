---
title: libreyolo export
seo_title: libreyolo export 命令参考
description: 把检查点导出成一种部署格式：每个参数及其默认值、产物写到哪里，以及命令会拒绝的组合。
lead: 把一个检查点（checkpoint）转换成一种部署格式，并把产物写到 weights/ 下。格式决定下面哪些参数适用。
keywords:
  - libreyolo export cli
  - libreyolo 导出命令
  - yolo 导出 onnx
  - tensorrt 导出命令
  - libreyolo export 参数
last_verified: 1.5.0
meta:
  - label: 命令
    value: libreyolo export
    mono: true
  - label: 必填
    value: model
    mono: true
  - label: 输出
    value: 'weights/<checkpoint-stem>[_fp16|_int8]<format-suffix>'
    mono: true
snippets:
  examples:
    - label: 基础
      language: bash
      code: |
        # 输出 weights/LibreYOLO9s.onnx
        libreyolo export model=LibreYOLO9s.pt format=onnx imgsz=640
    - label: 把 NMS 放进图里
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx \
          nms=true conf=0.25 iou=0.45 max_det=300
    - label: 运行导出的产物
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx imgsz=640

        # 工厂按文件后缀路由，所以导出产物会像检查点一样加载
        libreyolo predict model=weights/LibreYOLO9s.onnx \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
source_hash: ef2ca20af3814109
---

## 概要

```bash
libreyolo export model=<name|path> [format=<format>] [key=value ...]
```

参数是 `key=value` 形式，POSIX 形式也能用，所以 `format=onnx` 和
`--format onnx` 是同一个参数。

## 参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `model` | | 模型权重 `.pt`。必填 |
| `format` | `onnx` | 导出格式：`onnx`、`torchscript`、`executorch`、`tensorrt`、`openvino`、`paddle`、`mnn`、`rknn`、`ncnn`、`tflite`、`coreml`、`coreai` |
| `name` | | RKNN 目标平台，目前只有 `rk3588`。和其他任何格式一起传都会被拒绝 |
| `imgsz` | | 输入图像尺寸：`640` 或 `480x640`（HxW）。`480,640` 也接受。未设置时用模型自身的尺寸 |
| `batch` | `1` | 导出批大小 |
| `half` | `false` | FP16 精度 |
| `int8` | `false` | INT8 量化 |
| `dynamic` | `false` | 动态输入形状（ONNX） |
| `simplify` | `true` | ONNX 图简化 |
| `nms` | `false` | 把 NMS 内嵌进模型。仅 ONNX 和 CoreML |
| `conf` | `0.25` | 内嵌 NMS 的置信度阈值 |
| `iou` | `0.45` | 内嵌 NMS 的 IoU 阈值 |
| `max_det` | `300` | ONNX 内嵌 NMS 的最大检测数 |
| `opset` | | ONNX opset 版本。未设置时自动选择 |
| `data` | | INT8 的校准数据 |
| `fraction` | `1.0` | 使用校准数据的比例 |
| `device` | `auto` | 用于 tracing 的设备 |
| `allow_download_scripts` | `false` | 允许数据集 YAML 下载块里内嵌的 Python 代码 |
| `json` | `false` | 向 stdout 输出 JSON |
| `quiet` | `false` | 屏蔽 stderr |
| `verbose` | `false` | 详细的导出日志 |
| `verify` | `false` | 运行 RKNN Toolkit2 PC 模拟器并与 ONNX Runtime 对比。仅 RKNN |
| `help_json` | `false` | 以 JSON 打印命令 schema 并退出 |

`engine` 是 `tensorrt` 的别名，`litert` 是 `tflite` 的别名。两者在写入任何东西之前
都会解析成规范名称，所以 JSON 输出和那行日志里报的始终是 `tensorrt` 或 `tflite`。

## 示例

<code-tabs name="examples" />

## 说明

### 文件写到哪里

这个命令不接受输出路径。产物写到 `weights/` 下，名字取源检查点的主干名加上该格式的
后缀，如果请求了 FP16 或 INT8 中的某个精度，就在中间插入 `_fp16` 或 `_int8`。
`LibreYOLO9s.pt` 以 FP16 导出成 ONNX 会变成 `weights/LibreYOLO9s_fp16.onnx`。JSON
结果里带有解析后的 `output_path`、以 MB 为单位的文件大小，以及形如
`[batch, 3, height, width]` 的输入形状。

### 会被拒绝的组合

`nms=true` 在 ONNX 和 CoreML 上被接受，在其他所有格式上都以
`nms_unsupported_format` 拒绝。在 ONNX 上它会强制关掉 `dynamic`，因为内嵌的图固定
在批大小 1，并在 stderr 上说明这一点。在 CoreML 上它接受 `conf` 和 `iou`，但不接受
`max_det`，所以把非默认的 `max_det` 和 `format=coreml nms=true` 一起传会以
`config_unsupported` 退出。

`half=true` 和 `int8=true` 同时传不是错误。INT8 胜出，`half` 被丢弃，并向 stderr
发一条警告。

`name` 和 `verify` 目前是 RKNN 的选项。把其中任何一个和别的格式一起传会以
`config_unsupported` 退出，而不是被忽略。

### 一个家族支持哪些格式

支持情况是按家族、按任务来的，不是全局的。`libreyolo formats
family=<family> task=<task>` 会为这个组合打印每种格式的等级，附上原因和挂在上面的
约束。参数见 [`libreyolo formats`](/docs/cli/utilities)。

有些格式需要装一个可选依赖，有些需要工具链。缺少 Python 依赖会以
`export_dep_missing` 退出；该格式产不出的精度会以 `format_precision_unsupported`
退出。

### 运行你导出的东西

导出的产物和检查点走同一个模型工厂加载，按文件后缀来认，所以
`libreyolo predict model=weights/LibreYOLO9s.onnx` 不用再做任何转换就能跑。有三个
预测选项是例外，在运行时后端上会被拒绝：`tiling`、`overlap_ratio` 和
`output_file_format`。

有两个部署目标有自己的页面：
[NVIDIA DeepStream](/docs/export/deepstream) 和
[NVIDIA Jetson](/docs/export/jetson)。

### 输出与退出码

stdout 上是结果，进度走 stderr。退出码：成功是 `0`，用法或配置错误是 `2`，模型加载
不了是 `4`，未知格式、缺少导出依赖、不支持的精度或被拒绝的内嵌 NMS 请求是 `5`，其他
运行时失败是 `1`。

相关：[`libreyolo quantize`](/docs/cli/quantize)，它留在 PyTorch 里，写出的是检查点
而不是部署产物。
