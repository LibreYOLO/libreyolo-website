---
title: libreyolo track
seo_title: libreyolo track 命令参考
description: 在命令行里对视频或图像序列做目标跟踪：每一个参数及其默认值、四种跟踪器，以及每帧一行的 JSON 输出。
lead: 在视频文件、图像目录或图像列表上运行 model.track()，每帧打印一行，带上该帧的跟踪 ID。1.8.0 新增，检查点（checkpoint）和导出的产物都能用。
keywords:
  - libreyolo track cli
  - libreyolo 跟踪命令
  - bytetrack 命令行
  - botsort 命令行
  - 多目标跟踪 命令行
  - libreyolo track json
last_verified: "1.8.0"
meta:
  - label: 命令
    value: libreyolo track
    mono: true
  - label: 必填
    value: source
    mono: true
  - label: 输出
    value: 每帧一行输出到 stdout。save=true 时，标注后的视频写入 runs/track/
snippets:
  examples:
    - label: 基础用法
      language: bash
      code: |
        libreyolo track model=LibreYOLO9s.pt source=video.mp4
    - label: 换一个跟踪器，保存视频
      language: bash
      code: |
        libreyolo track model=LibreYOLO9s.pt source=video.mp4 \
          tracker=botsort classes="[0]" save=true
    - label: 导出的模型，每帧 JSON
      language: bash
      code: |
        libreyolo track model=weights/LibreYOLO9s.onnx source=frames/ fps=25 \
          json=true quiet=true
source_hash: 65b603ce6ee7a80f
---

## 用法概要

```bash
libreyolo track source=<video|directory|image|a.jpg,b.jpg> [model=<name|path>] [key=value ...]
```

参数是 `key=value` 形式的键值对，也接受 POSIX 写法，所以 `tracker=botsort`
和 `--tracker botsort` 是同一个参数。

## 参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `source` | | 视频文件、图像目录、图像路径，或逗号分隔的 `a.jpg,b.jpg` 图像列表。必填 |
| `model` | `yolox-s` | 模型名称或路径；导出的产物也可以 |
| `tracker` | `bytetrack` | `bytetrack`、`botsort`、`ocsort` 或 `deepocsort`。写成 `.yaml` 的形式会选中同样的内置跟踪器 |
| `track_conf` | `0.25` | 跟踪器第一阶段关联的置信度阈值 |
| `conf` | | 检测阈值。不设置时，检测器以跟踪器自己的低阈值运行，让弱检测框留作找回之用 |
| `iou` | `0.45` | NMS 的 IoU 阈值 |
| `imgsz` | | 输入图像尺寸：`640`（正方形）或 `480x640`（高×宽）。不设置时使用模型自身的输入尺寸 |
| `classes` | | 按类别 ID 过滤，例如 `[0,2,5]`。也接受单个整数 |
| `max_det` | `300` | 每帧的最大检测数 |
| `save` | `false` | 保存标注后的视频 |
| `show` | `false` | 显示跟踪后的帧；按 `q` 停止 |
| `vid_stride` | `1` | 每 N 帧处理一帧 |
| `fps` | `30` | 图像序列假定的帧率；视频会忽略它 |
| `output_path` | | 输出视频路径。默认 `runs/track/<name>.mp4`，带编号，一次运行永远不会覆盖另一次 |
| `color_format` | `auto` | 输入色彩：`auto`、`rgb`、`bgr` |
| `device` | `auto` | 设备：`0`、`cpu`、`mps`、`auto` |
| `json` | `false` | 向 stdout 输出 JSON，每帧一个对象 |
| `quiet` | `false` | 抑制 stderr |
| `verbose` | `false` | stderr 输出更详细的信息 |
| `help_json` | `false` | 以 JSON 形式输出命令的 schema 并退出 |

## 示例

<code-tabs name="examples" />

## 说明

每次调用都从第一帧开始跟踪一个源，所以没有 `persist` 选项：跟踪 ID 永远不会在命令之间延续。
实时源会以 `config_unsupported` 被拒绝：摄像头和流都是。请使用视频文件、图像目录或图像列表。

面向人读的输出是每帧一行，例如
`frame 12 video.mp4: 1920x1080 3 tracks (ids 1, 2, 5)`，stderr 最后会给出帧数和不重复的跟踪 ID 个数。
`json=true` 时，每帧是一个 JSON 对象，带有 `tracker`、`frame_index` 和一个 `detections` 列表，
其中每个检测框都有 `track_id`、`class`、`class_id`、`confidence` 和 `bbox_xyxy`。

如果模型的任务没有可供关联的检测框，或者模型类型没有 `track()`，命令会以 `config_unsupported` 退出。
退出码与 [`libreyolo predict`](/docs/cli/predict) 一致：成功是 `0`，用法或配置错误是 `2`，
找不到源是 `3`，模型加载失败是 `4`，其他运行时故障是 `1`。

跟踪器、它们的配置以及 Python 的 `track()` 调用见[目标跟踪](/docs/tasks/object-tracking)。
