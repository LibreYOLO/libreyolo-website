---
title: libreyolo predict
seo_title: libreyolo predict 命令参考
description: 从命令行跑推理：每一个参数、从 CLI 定义里读出的默认值，以及会改变 stdout 输出内容的 flag。
lead: 在一个源上运行已加载的模型并打印预测结果。源可以是图像、目录、视频、URL 或实时流；模型可以是检查点（checkpoint），也可以是导出的产物。
keywords:
  - libreyolo predict cli
  - libreyolo 命令行推理
  - yolo 命令行预测
  - libreyolo predict 参数
  - libreyolo json 输出
last_verified: 1.5.0
meta:
  - label: 命令
    value: libreyolo predict
    mono: true
  - label: 必填
    value: source
    mono: true
  - label: 输出
    value: 预测结果打印到 stdout。save=true 时，标注后的文件写入 runs/detect/predict
snippets:
  examples:
    - label: 基础用法
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 保存标注后的图像
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt save=true \
          project=runs/detect name=parkour exist_ok=true \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 过滤类别，JSON 输出到 stdout
      language: bash
      code: >
        # 类别 0 在检查点自带的 COCO 类别列表里是 person

        libreyolo predict model=LibreYOLO9s.pt classes="[0]" conf=0.4 max_det=50
        \
          json=true quiet=true \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
source_hash: 7e46c7ed7dd9e6c4
---

## 用法概要

```bash
libreyolo predict source=<path|url|index> [model=<name|path>] [key=value ...]
```

参数是 `key=value` 形式的键值对。同一条命令也接受 POSIX 写法，所以 `conf=0.4`
和 `--conf 0.4` 可以互换，写成 `save=true` 的布尔值等价于 `--save`。带下划线的
名字两种拼写都接受：`max_det=50` 和 `--max-det 50` 指向同一个选项。

`libreyolo detect predict ...` 同样被接受，行为完全一致；任务名会在解析前被
去掉。

## 参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `source` | | 图像路径、目录或 URL。必填 |
| `model` | `yolox-s` | 模型名称或路径 |
| `conf` | `0.25` | 置信度阈值 |
| `iou` | `0.45` | NMS 的 IoU 阈值 |
| `imgsz` | | 输入图像尺寸：`640`（正方形）或 `480x640`（高×宽）。不设置时使用模型自身的输入尺寸 |
| `classes` | | 按类别 ID 过滤，例如 `[0,2,5]`。也接受单个整数 |
| `max_det` | `300` | 每张图像的最大检测数 |
| `half` | `false` | FP16 推理（仅限 CUDA，且需要模型支持） |
| `save` | `false` | 保存标注后的图像 |
| `batch` | `1` | 目录源每次前向传播处理的图像数。大于 1 时，在支持的模型上执行真正的批量推理 |
| `stream` | `false` | 增量产出结果。摄像头和实时流会自动开启 |
| `stream_buffer` | `false` | 缓冲实时流的每一帧，而不是只保留最新的一帧 |
| `vid_stride` | `1` | 每 N 帧处理一帧视频或实时流 |
| `show` | `false` | 显示视频和实时结果；按 `q` 停止 |
| `tiling` | `false` | 对大图做切片推理 |
| `overlap_ratio` | `0.2` | 切片重叠比例 |
| `output_path` | | 显式指定输出路径。不指定时，`save=true` 会写入 `project/name` |
| `color_format` | `auto` | 输入色彩：`auto`、`rgb`、`bgr` |
| `output_file_format` | | 输出格式：`jpg`、`png`、`webp` |
| `device` | `auto` | 设备：`0`、`cpu`、`mps`、`auto` |
| `face_detector` | | 人脸检测模型（路径或 CLI 名称）。视线估计模型必填 |
| `gallery` | | 由 `libreyolo enroll` 生成的人脸底库 `.npz`，用来比对识别人脸。仅限人脸嵌入向量模型 |
| `gallery_threshold` | `0.4` | 底库身份匹配的余弦阈值 |
| `project` | `runs/detect` | 输出目录根路径 |
| `name` | `predict` | 实验名称 |
| `exist_ok` | `false` | 复用已存在的输出目录 |
| `json` | `false` | 向 stdout 输出 JSON |
| `quiet` | `false` | 抑制 stderr |
| `verbose` | `false` | stderr 输出更详细的信息 |
| `help_json` | `false` | 以 JSON 形式输出命令的 schema 并退出 |

## 示例

<code-tabs name="examples" />

## 说明

导出的产物和检查点的加载方式一样，所以 `model=weights/LibreYOLO9s.onnx` 和
`model=weights/LibreYOLO9s.engine` 都是 `model` 的合法取值。有三个选项在这些
运行时上会被拒绝，而不是被忽略：运行时后端无法满足 `tiling`、`overlap_ratio`
和 `output_file_format` 时，命令会以 `config_unsupported` 退出。

`half` 则相反。导出的运行时会接受它并以 FP16 运行；原生 PyTorch 推理会在日志
里说明该选项被忽略，并继续以 FP32 运行。

视线估计模型是两阶段的，自身不带检测器，所以必须为它们指定 `face_detector`。
`gallery` 只对任务为 `embed` 的模型有效；传给其他模型会以 `config_unsupported`
退出。

stdout 只承载结果，别的什么都没有；进度、警告和错误都走 stderr。`json=true`
每次调用打印一个 JSON 对象，流式处理时每帧打印一个，每个都带
`schema_version`。`quiet=true` 会让 stderr 静默。两者合用，机器读取方就能拿到
一条干净的 stdout 流。

退出码在成功时是 `0`，用法或配置错误是 `2`，找不到源是 `3`，模型加载失败是
`4`，其他运行时故障是 `1`。

`help_json=true` 会以 JSON 形式打印该命令的参数、类型、默认值和 flag，不实际
运行任何东西，这是从已安装的版本里回读这张表最可靠的方式。

相关：[`libreyolo val`](/docs/cli/val) 用于在数据集上测量指标，
[`libreyolo export`](/docs/cli/export) 用于生成上面提到的运行时产物。
