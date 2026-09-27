---
title: libreyolo 实用命令
seo_title: libreyolo CLI 实用命令参考
description: >-
  LibreYOLO 的一组小命令：version、checks、models、formats、cfg、info、metadata、enroll 和
  compare，每个都附上它的参数和默认值。
lead: 九个只做报告和检查、不做计算的命令。它们打印环境信息、模型与格式清单、解析后的默认值、检查点（checkpoint）细节，并构建和查询一个人脸库。
keywords:
  - libreyolo version
  - libreyolo checks
  - libreyolo 模型列表
  - libreyolo 支持的导出格式
  - 查看 yolo 检查点元数据
  - libreyolo 默认配置
  - libreyolo 人脸注册
  - libreyolo 人脸比对
last_verified: 1.5.0
meta:
  - label: 命令
    value: 'version, checks, models, formats, cfg, info, metadata, enroll, compare'
    mono: true
  - label: 输出
    value: stdout，文本形式；或者在 json=true 时输出一个带 schema_version 的对象
snippets:
  examples:
    - label: 环境
      language: bash
      code: |
        libreyolo version
        libreyolo checks
    - label: 有哪些可用
      language: bash
      code: |
        libreyolo models
        libreyolo formats family=yolo9 task=detect
    - label: 查看检查点
      language: bash
      code: |
        libreyolo info model=LibreYOLO9s.pt
        libreyolo metadata path=weights/LibreYOLO9s.pt
source_hash: 7b5b53c46df00c06
---

## 概要

```bash
libreyolo <command> [key=value ...]
```

参数是 `key=value` 对，POSIX 形式同样可用，所以 `model=x` 和 `--model x` 是同一个
参数。这里的每个命令都把结果写到 stdout，并接受 `json=true` 和 `quiet=true`。

根命令自己带一个标志 `libreyolo --version`，它打印版本字符串然后退出。它的输出比
下面的 `version` 命令更少。

## version

打印 LibreYOLO 的版本，以及它所依赖运行的 Python、torch 和 CUDA 版本。

```bash
libreyolo version
```

| 参数 | 默认值 | 含义 |
|---|---|---|
| `json` | `false` | 向 stdout 输出 JSON |
| `quiet` | `false` | 屏蔽 stderr |

## checks

更详细地打印环境：Python、torch、CUDA、cuDNN、检测到的每块 GPU 及其名称和显存，
还有导出路径用到的每个可选包的安装版本。

```bash
libreyolo checks
```

| 参数 | 默认值 | 含义 |
|---|---|---|
| `json` | `false` | 向 stdout 输出 JSON |
| `quiet` | `false` | 屏蔽 stderr |

包列表覆盖 `onnx`、`onnxruntime`、`tensorrt`、`openvino`、`paddlepaddle`、
`x2paddle`、`mnn`、`ncnn`、`onnx2tf`、`ai-edge-litert`、`transformers` 和
`scipy`。没有安装的包会照实报告，而不是直接略过，所以一次失败的导出可以只靠这一个
命令追溯到缺失的依赖。

## models

列出每个模型家族及其任务、尺寸、能解析到它的检查点的 CLI 名称，以及每种尺寸的输入
分辨率。

```bash
libreyolo models
```

| 参数 | 默认值 | 含义 |
|---|---|---|
| `json` | `false` | 向 stdout 输出 JSON |
| `quiet` | `false` | 屏蔽 stderr |

可选依赖没有安装的家族会被列为不可用，并附上那行能让它可用的 `pip install`。这些
CLI 名称就是 `model=` 接受的简写：`yolox-s` 解析到 `LibreYOLOXs.pt`，非检测任务则
带上各自的任务后缀。

## formats

列出当前安装的环境能产出的导出格式，以及每种格式的文件扩展名和它是否支持 FP16 和
INT8。

```bash
libreyolo formats [family=<family>] [task=<task>]
```

| 参数 | 默认值 | 含义 |
|---|---|---|
| `family` | | 显示某一个模型家族的支持等级。`model=` 被当作同一个选项接受 |
| `task` | | 规范的模型任务。未设置时取该家族的默认任务 |
| `json` | `false` | 向 stdout 输出 JSON |
| `quiet` | `false` | 屏蔽 stderr |

不带 `family` 时，输出只有格式清单本身。带上它之后，每种格式还会给出针对该家族和该
任务的支持等级、这个等级背后的理由，以及附在它上面的任何约束。未知的家族，或者该
家族不支持的任务，都是用法错误。

格式别名显示在它的规范名称旁边：`engine` 对应 `tensorrt`，`litert` 对应 `tflite`。

## cfg

打印解析后的默认配置：训练默认值、验证默认值、预测默认值，以及各家族的覆盖项。

```bash
libreyolo cfg
```

| 参数 | 默认值 | 含义 |
|---|---|---|
| `json` | `false` | 向 stdout 输出 JSON |
| `quiet` | `false` | 屏蔽 stderr |

这些值读自配置 dataclass，而不是某份副本，所以当你不传参数时，一次训练会用什么，以
它为准。`family_overrides` 就是那一节，它回答为什么某个家族按你没要求过的设置训练。
这些覆盖项如何生效，见 [`libreyolo train`](/docs/cli/train)。

## info

在 CPU 上加载一个模型，报告它的家族、尺寸、参数量、类别，以及每种格式的导出等级。

```bash
libreyolo info model=<name|path>
```

| 参数 | 默认值 | 含义 |
|---|---|---|
| `model` | | 模型名称或权重路径。必填 |
| `detailed` | `false` | 包含逐参数的细节 |
| `json` | `false` | 向 stdout 输出 JSON |
| `quiet` | `false` | 屏蔽 stderr |

## metadata

不构造模型就读出一个检查点的元数据，并按 LibreYOLO 的检查点 schema 校验它。

```bash
libreyolo metadata path=<checkpoint.pt>
```

| 参数 | 默认值 | 含义 |
|---|---|---|
| `path` | | 指向一个 `.pt` 检查点的路径。必填 |
| `json` | `false` | 向 stdout 输出 JSON |
| `quiet` | `false` | 屏蔽 stderr |

带有大张量的条目只做摘要，不整个打印出来，所以面对一个完整的训练检查点，输出依然
可读。不存在的检查点会以 `checkpoint_not_found` 退出；元数据校验不通过的，会打印
出错误并以 `1` 退出。

## enroll

从一棵每人一个文件夹的目录树构建人脸库，好让后面的预测能叫出它找到的人脸的名字。

```bash
libreyolo enroll model=<embedder> source=<people-dir> gallery=<gallery.npz>
```

| 参数 | 默认值 | 含义 |
|---|---|---|
| `model` | | 人脸嵌入向量模型，路径或名称。必填 |
| `source` | | 每人一个文件夹的目录树，`source/<identity>/*.jpg`。必填 |
| `gallery` | | 输出的人脸库文件 `.npz`。已存在时就地扩充。必填 |
| `face_detector` | | 人脸检测器：一个 YuNet `.onnx`，或者一个 LibreYOLO 检测器。未设置时取该家族的默认检测器 |
| `device` | `auto` | 设备：`0`、`cpu`、`mps`、`auto` |
| `json` | `false` | 向 stdout 输出 JSON |
| `quiet` | `false` | 屏蔽 stderr |

```bash
# people/ 下每个身份一个文件夹，文件夹名就是身份
libreyolo enroll model=librefacerec-l.onnx source=people/ gallery=people.npz
```

子文件夹名就是身份。检测不到人脸的参考图会被跳过，并在 stderr 上留一行，其余的继续
处理；没有身份子文件夹的 source，或者一张人脸都没找到的 source，是错误。

把得到的文件以 `gallery=people.npz` 传给
[`libreyolo predict`](/docs/cli/predict)，检测结果就会带上一个身份和一个匹配分数。

## compare

报告两张人脸图像之间的余弦相似度，以及它有没有越过同一身份的阈值。

```bash
libreyolo compare model=<embedder> source=<a.jpg> source2=<b.jpg>
```

| 参数 | 默认值 | 含义 |
|---|---|---|
| `model` | | 人脸嵌入向量模型，路径或名称。必填 |
| `source` | | 第一张图像。必填 |
| `source2` | | 用来比对的第二张图像。必填 |
| `face_detector` | | 人脸检测器：一个 YuNet `.onnx`，或者一个 LibreYOLO 检测器 |
| `threshold` | `0.4` | 判定为同一身份的余弦相似度阈值 |
| `device` | `auto` | 设备：`0`、`cpu`、`mps`、`auto` |
| `json` | `false` | 向 stdout 输出 JSON |
| `quiet` | `false` | 屏蔽 stderr |

```bash
libreyolo compare model=librefacerec-l.onnx source=a.jpg source2=b.jpg
```

`libreyolo verify` 注册为这个命令的第二个名字，接受同样的参数。

`compare` 和 `enroll` 都需要一个任务是人脸嵌入向量的模型。其他的一律以
`config_unsupported` 退出。本地图像路径和 `http` 或 `https` URL 都可以作为源。

## 示例

<code-tabs name="examples" />

## 说明

结果走 stdout；进度和警告走 stderr。`json=true` 打印一个带 `schema_version` 的
对象，脚本要读的就是这种形式。文本输出是默认值，它是给人看的。

退出码沿用 CLI 其余部分的同一套映射：成功是 `0`，用法或配置错误是 `2`，找不到源是
`3`，模型或检查点加载不了是 `4`，其他运行时失败是 `1`。

相关：[`libreyolo doctor`](/docs/cli/doctor)，数据集一侧的检查命令；以及
[`libreyolo profile`](/docs/cli/profile)，性能一侧的那个。
