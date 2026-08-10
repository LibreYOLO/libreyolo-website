---
title: libreyolo val
seo_title: "libreyolo val 命令参考"
description: "从命令行在数据集划分（split）上评估检查点（checkpoint）：每个参数及其默认值，以及每种任务返回的指标键。"
lead: "把一个模型放到一个数据集划分上评估并打印指标。指标集合取决于模型的任务，这些数字就是基准测试里一行数据的来源。"
keywords: [libreyolo val 命令, libreyolo 验证命令, yolo 命令行评估, mAP50-95 计算, libreyolo val 参数]
last_verified: "1.5.0"
meta:
  - label: 命令
    value: libreyolo val
    mono: true
  - label: 必填
    value: model, data
    mono: true
  - label: 输出
    value: "指标打印到 stdout。需要时在 runs/val/exp 下生成图表和 COCO JSON"
snippets:
  examples:
    - label: 基础用法
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml
    - label: 图表与 COCO JSON
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml \
          imgsz=640 batch=8 save_json=true save_plots=true \
          project=runs/val name=yolo9s-coco8 exist_ok=true
    - label: 机器可读
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml json=true quiet=true
---

## 概要

```bash
libreyolo val model=<name|path> data=<dataset.yaml> [key=value ...]
```

参数是 `key=value` 对，POSIX 写法同样可用，所以 `batch=8` 和 `--batch 8` 是同一个
参数。

## 参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `model` | | 模型权重路径或 CLI 名称。必填 |
| `data` | | 数据集 YAML 的路径（YOLO 格式，例如 `coco8.yaml`）。必填 |
| `data_dir` | | 直接指定数据集目录，绕过 YAML 里写的路径 |
| `split` | `val` | 数据集划分：`val`、`test`、`train` |
| `batch` | `16` | 批大小 |
| `imgsz` | | 图像尺寸：`640`（正方形）或 `480x640`（HxW，高 × 宽）。不设置时用模型自身的输入尺寸 |
| `conf` | `0.001` | 置信度阈值 |
| `iou` | `0.6` | NMS 的 IoU 阈值 |
| `max_det` | `300` | NMS 之后每张图像保留的最大预测数 |
| `eval_max_det` | | COCO 评估器的上限。不设置时采用 pycocotools 的 AP@100 惯例 |
| `faster_coco_eval` | `true` | 装了 faster-coco-eval 时用它的 C++ 后端算 COCO 指标；否则回退到 pycocotools |
| `half` | `false` | FP16 推理 |
| `amp_dtype` | `float16` | `half=true` 时 CUDA autocast 的数据类型：`float16` 或 `bfloat16` |
| `save_json` | `false` | 保存 COCO 格式的 JSON 结果 |
| `save_plots` | `false` | 保存验证图表：指标、各类别 AP、混淆矩阵、样本 |
| `workers` | `4` | 数据加载器的 worker 数 |
| `device` | `auto` | 设备 |
| `project` | `runs/val` | 输出目录根路径 |
| `name` | `exp` | 实验名称 |
| `exist_ok` | `false` | 复用输出目录 |
| `allow_download_scripts` | `false` | 允许数据集 YAML 的 download 块里内嵌 Python 代码 |
| `json` | `false` | 把 JSON 输出到 stdout |
| `quiet` | `false` | 屏蔽 stderr |
| `verbose` | `true` | 详细输出 |
| `help_json` | `false` | 把命令 schema 以 JSON 打印出来并退出 |

## 示例

<code-tabs name="examples" />

## 说明

### 指标都有哪些

打印出来的指标集合跟着模型的任务走，JSON 输出用的是同一套键。

目标检测、分割和旋转框报告 `mAP50`、`mAP50_95`、`precision` 和 `recall`。当一个
模型预测的输出不止一种时，按种类分组的指标会并排出现，分别是 `box_metrics`、
`mask_metrics` 和 `obb_metrics`，每一组都带着同样这四个键。

分类报告 `accuracy_top1` 和 `accuracy_top5`。点检测报告 `precision`、`recall`、
`f1`、`MLE`、`MAE`、`RMSE` 和 `mAP_sweep`。深度估计报告 `abs_rel`、`rmse`、
`delta1`、`delta2` 和 `delta3`。语义分割报告 `mIoU` 和 `pixel_accuracy`。图像
复原（restoration）报告 `PSNR` 和 `SSIM`。

JSON 结果里还带有 `eval_backend`，标明算出这些数字的 COCO 评估库和版本，这样比较
两次运行时就知道它们是不是由同一个后端打的分。

### 阈值

这里的默认值是评估的默认值，不是预测的默认值：`conf` 是 `0.001`，`iou` 是 `0.6`，
而 [`libreyolo predict`](/docs/cli/predict) 用的是 `0.25` 和 `0.45`。把 `conf`
提高到适合显示的阈值会降低查全率，mAP 也跟着降低，这样得出的数字没法和公开发布的
数字相比。

`imgsz` 默认不设置，也就是用模型自身的输入尺寸。设置之后会按给定的尺寸评估，这就是
在原生分辨率之外衡量一个检查点的方式。

### 会自行下载的数据集

如果数据集 YAML 的 `download` 字段是一个 URL，首次使用时就会直接下载，不需要额外
确认。如果带的是内嵌的 Python 下载脚本，就需要 `allow_download_scripts=true`，命令
会在 stderr 上警告本地代码执行已被启用。自带的 `coco8.yaml` 和 `coco128.yaml` 走的
是 URL，所以什么都不用加。

### 输出与退出码

指标走 stdout，进度走 stderr。`json=true` 会打印一个带 `schema_version` 的对象，
`quiet=true` 会让 stderr 安静下来。

退出码在成功时是 `0`，用法或配置错误是 `2`，找不到数据集是 `3`，模型加载不了是
`4`，其他运行时失败是 `1`。

相关：[`libreyolo train`](/docs/cli/train)，它会通过 `eval_interval` 按自己的节奏
运行同样这套评估。
