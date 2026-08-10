---
title: libreyolo profile
seo_title: libreyolo profile 命令参考
description: 测量训练和推理速度并读懂结果：profile 的每个子命令、它们的参数和默认值，以及每种读法各自报告什么。
lead: 一个命令组，它测量训练步或推理调用中的时间花在了哪里，写出一份自包含的剖析文件（profile），再用几种不同的读法把这份剖析文件读回来。
keywords:
  - libreyolo profile 命令
  - yolo 训练性能分析
  - 推理延迟测试
  - gpu kernel 性能分析
  - libreyolo 性能对比
last_verified: 1.5.0
meta:
  - label: 命令
    value: libreyolo profile
    mono: true
  - label: 输出
    value: profile.json and profile_trace.json under runs/profile
    mono: true
snippets:
  examples:
    - label: 测量推理
      language: bash
      code: |
        # 不带 source 参数时使用内置的示例图片
        libreyolo profile infer --device cpu --warmup 5 --runs 20
    - label: 读取结论
      language: bash
      code: |
        libreyolo profile summary runs/profile/infer/profile.json
    - label: 对比两次测量
      language: bash
      code: >
        libreyolo profile infer --device cpu --warmup 5 --runs 20 --project
        runs/profile/a

        libreyolo profile infer --device cpu --warmup 5 --runs 20 --batch 4
        --project runs/profile/b


        libreyolo profile compare runs/profile/a/infer/profile.json \
          runs/profile/b/infer/profile.json
source_hash: b967e869fd9ba418
---

## 概要

```bash
libreyolo profile <subcommand> [<positional>] [--flag value ...]
```

这个命令组不接受 `key=value` 形式的参数。它的子命令使用位置参数和 POSIX 风格的
标志，所以要写 `--weights LibreYOLO9t.pt`，而不是 `weights=LibreYOLO9t.pt`。不带
子命令运行 `libreyolo profile` 会打印子命令列表。

两个子命令负责测量并写出剖析文件，其余的负责读取。`run` 和 `infer` 都会输出同一种
自包含的 `profile.json`，所以每个读取类子命令对两者都适用。

## profile run

运行一段短的带剖析的训练，并写出一份剖析文件。

```bash
libreyolo profile run <data> [--flag value ...]
```

| 参数 | 默认值 | 含义 |
|---|---|---|
| `data` | | 位置参数。数据集 YAML 或名称，例如 `coco128`。必填 |
| `--weights` | `LibreYOLO9t.pt` | 模型权重文件或名称 |
| `--size` | `t` | 模型尺寸变体 |
| `--batch` | `16` | 微批。`-1` 自动适配约 70% 的显存 |
| `--imgsz` | `640` | 训练图像尺寸 |
| `--workers` | `8` | 数据加载器的工作进程数 |
| `--amp` | `true` | 使用该家族的 AMP 路径。`--no-amp` 关闭它 |
| `--steps` | `20` | 被剖析、也就是被测量的步数 |
| `--warmup` | `5` | 测量前的预热步数 |
| `--repeat` | `1` | 重复 N 次以得到均值和标准差 |
| `--device` | `0` | 设备 |
| `--project` | `runs/profile` | 输出目录根路径 |
| `--json` | `false` | 向 stdout 输出 JSON |

测量窗口是 `--warmup` 加 `--steps` 次迭代。数据集小到填不满这个窗口时不会产生剖析
文件，命令以退出码 `3` 结束，并点明三条出路：换更大的数据集、减少步数，或者减小
批大小。

`--repeat` 大于 1 时会写出一份汇总的 `runs/profile/profile_repeat.json`，其中的
标量指标是各次试验的平均值，而 kernel 列表来自最后一次试验。它也是 `compare` 给出
显著性结论的前提：单次运行给不出这个结论。

## profile infer

剖析推理路径，并写出一份剖析文件。

```bash
libreyolo profile infer [<source>] [--flag value ...]
```

| 参数 | 默认值 | 含义 |
|---|---|---|
| `source` | | 位置参数。图像或目录。省略时使用内置的示例图片 |
| `--weights` | `LibreYOLO9t.pt` | 模型权重文件或名称 |
| `--size` | `t` | 模型尺寸变体 |
| `--batch` | `1` | 每次前向传播的图像数 |
| `--imgsz` | `640` | 输入图像尺寸 |
| `--half` | `false` | 前向使用 autocast，仅限 CUDA。`--no-half` 关闭它 |
| `--amp-dtype` | `float16` | CUDA autocast 的 dtype：`float16` 或 `bfloat16` |
| `--warmup` | `20` | 测量前的预热迭代次数 |
| `--runs` | `100` | 被测量的迭代次数 |
| `--repeat` | `1` | 重复 N 次以得到均值和标准差 |
| `--conf` | `0.25` | 置信度阈值，它会改变 NMS 的工作量 |
| `--iou` | `0.45` | NMS 的 IoU 阈值 |
| `--max-det` | `300` | 每张图像的最大检测数，它会改变 NMS 的工作量 |
| `--device` | `0` | 设备 |
| `--trace` | `true` | 输出一份 Chrome trace，用于下钻到 kernel 和算子。`--no-trace` 跳过它 |
| `--project` | `runs/profile` | 输出目录根路径 |
| `--json` | `false` | 向 stdout 输出 JSON |

报告 p50、p90 和 p99 的延迟，以每秒图像数计的吞吐量，以及预处理、前向和后处理之间
的阶段拆分。这三个阈值参数出现在这里，是因为它们会改变后处理的数字。

## profile summary

```bash
libreyolo profile summary <trace> [--json]
```

| 参数 | 默认值 | 含义 |
|---|---|---|
| `trace` | | 位置参数。指向 `profile.json` 或 `profile_trace.json` 的路径。必填 |
| `--json` | `false` | 向 stdout 输出 JSON |

高层次的读法：单步耗时、吞吐量、GPU 利用率、Tensor Core 占比、显存峰值、主机开销、
每步的 kernel 启动次数、瓶颈结论及其理由、按类别划分的 kernel 构成，以及每步耗时
最高的那些 kernel。在推理剖析文件上，它还会打印延迟分位数和阶段拆分。

在显存颠簸下采集的剖析文件会被标记出来，因为在那种情况下测到的利用率和吞吐量
不可信。

## profile get

```bash
libreyolo profile get <trace> [<field>] [--json]
```

| 参数 | 默认值 | 含义 |
|---|---|---|
| `trace` | | 位置参数。指向剖析文件的路径。必填 |
| `field` | | 位置参数。指标名称。省略则列出可用的指标 |
| `--json` | `false` | 向 stdout 输出 JSON |

只打印一个指标，别的什么都不打印，便于在脚本循环里使用。未知的字段会以退出码 `2`
结束，并指向列出指标的用法。

## profile phases

```bash
libreyolo profile phases <trace> [--json]
```

| 参数 | 默认值 | 含义 |
|---|---|---|
| `trace` | | 位置参数。指向剖析文件的路径。必填 |
| `--json` | `false` | 向 stdout 输出 JSON |

每个阶段的 GPU 毫秒数、墙钟毫秒数、kernel 数量和算子数量：forward、backward、
dataload、to_device、optimizer。

## profile kernels

```bash
libreyolo profile kernels <trace> [--flag value ...]
```

| 参数 | 默认值 | 含义 |
|---|---|---|
| `trace` | | 位置参数。指向剖析文件的路径。必填 |
| `--top` | `20` | 按 GPU 时间显示前 N 个 |
| `--category` | | 按类别子串过滤：`gemm`、`layout`、`norm`、`elementwise` |
| `--grep` | | 按 kernel 名称的正则表达式过滤 |
| `--tensorcore` | `false` | 只看 Tensor Core kernel |
| `--sort` | `time` | `time`、`count` 或 `name` |
| `--phase` | | 限定到一个阶段：`forward`、`backward`、`dataload`、`to_device`、`optimizer` |
| `--json` | `false` | 向 stdout 输出 JSON |

分析的最底层：单个 GPU kernel 及其占 GPU 时间的比例、每步的毫秒数、每步的调用次数
和类别。未知的 `--phase` 会以退出码 `2` 结束，并列出该剖析文件里有哪些阶段。

## profile ops

```bash
libreyolo profile ops <trace> [--flag value ...]
```

| 参数 | 默认值 | 含义 |
|---|---|---|
| `trace` | | 位置参数。指向剖析文件的路径。必填 |
| `--top` | `20` | 按 CPU 时间显示前 N 个 |
| `--phase` | | 限定到一个阶段 |
| `--json` | `false` | 向 stdout 输出 JSON |

框架视角而不是设备视角：按 CPU 时间排序的 `aten` 和 autograd 算子，主机端的启动
开销就体现在这里。

## profile compare

```bash
libreyolo profile compare <before> <after> [--json]
```

| 参数 | 默认值 | 含义 |
|---|---|---|
| `before` | | 位置参数。基线剖析文件。必填 |
| `after` | | 位置参数。新的剖析文件。必填 |
| `--json` | `false` | 向 stdout 输出 JSON |

对比吞吐量、每张图像的毫秒数、GPU 利用率、主机开销、每步的 kernel 启动次数和瓶颈
结论的差异。

显著性判断要求两边都用至少为 2 的 `--repeat` 测量。满足这一点后，差异超过合并标准
误的两倍才算显著，输出会打印它所做的比较。不满足时，那一行会写明单次运行支撑不了
这个判断。

## profile what-if

```bash
libreyolo profile what-if <trace> [--flag value ...]
```

| 参数 | 默认值 | 含义 |
|---|---|---|
| `trace` | | 位置参数。指向剖析文件的路径。必填 |
| `--remove-category` | | 推算移除某一类 kernel 的效果：`gemm`、`layout`、`norm`、`elementwise` |
| `--remove-launches` | | 推算每步减少 N 次 kernel 启动的效果，例如一次算子融合带来的收益 |
| `--json` | `false` | 向 stdout 输出 JSON |

在改动写出来之前，先估算这个改动能换来什么。两个选项必须给出其中之一；一个都不给
会以退出码 `2` 结束。

推算遵循剖析文件自己的结论。GPU 利用率低于 80% 时，它把节省建模为减少的启动次数
乘以实测的单次启动主机开销；高于这个值时，则建模为减少的 GPU 工作量。结果带一个
caveat 字段，因为单次启动的开销只是一个近似值，唯一的证明是再测一次。

## 示例

<code-tabs name="examples" />

## 说明

性能剖析器只测量和报告。它不改动任何东西：读结论、改配置或改代码、重新运行、再
对比，这就是它被设计出来要支撑的循环。

`--device` 默认为 `0`，也就是 CUDA 设备 0。传 `--device cpu` 会在 CPU 上测量，产生
的剖析文件读取类子命令照样接受，只是没有 GPU kernel 的细节。

每个子命令都支持 `--json`，读取类的子命令只往 stdout 打印，这正是这个命令组能在
脚本里使用的原因。

这里的退出码是该命令组自己的：`2` 表示文件不存在或参数无法解析，`3` 表示 `run`
没有产生剖析文件，`1` 表示某个 trace 无法被分析。

相关：[`libreyolo train`](/docs/cli/train)，训练剖析文件通常就是为了调它的参数
才采集的。
