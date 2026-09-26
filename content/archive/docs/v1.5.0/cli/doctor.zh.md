---
title: libreyolo doctor
seo_title: libreyolo doctor 命令参考
description: 训练前检查检测数据集：带默认值的参数、可以跳过或单独选择的检查家族，以及 CI 可以据此设门禁的退出码。
lead: 对检测数据集跑一组健康检查，报告哪些问题会拖累训练：缺失的文件、损坏的标注、损坏的图像、划分（split）之间的泄漏，以及类别不均衡。
keywords:
  - libreyolo doctor cli
  - 数据集健康检查
  - yolo 数据集校验
  - 数据集泄漏检查
  - libreyolo doctor strict
last_verified: 1.5.0
meta:
  - label: 命令
    value: libreyolo doctor
    mono: true
  - label: 必填
    value: data
    mono: true
  - label: 输出
    value: stdout 上的一份问题报告。发现错误时退出码为 1
snippets:
  examples:
    - label: 基础
      language: bash
      code: |
        # download=true 允许内置的 coco8.yaml 在图像缺失时自行下载
        libreyolo doctor coco8.yaml download=true
    - label: 快速扫描，不解码图像
      language: bash
      code: |
        libreyolo doctor coco8.yaml download=true fast=true
    - label: 对选定的检查设 CI 门禁
      language: bash
      code: |
        libreyolo doctor coco8.yaml download=true strict=true json=true \
          only=labels,files,config
source_hash: 79e0ef471d567ea3
---

## 概要

```bash
libreyolo doctor <data.yaml> [key=value ...]
```

数据集是位置参数，也接受 `data=<path>` 这种写法。两者都给且取值不同时，会以
`config_conflict` 退出。其余一律是 `key=value` 对，POSIX 写法同样可用，因此
`imgsz=1024` 和 `--imgsz 1024` 是同一个参数。

## 参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `data` | | 位置参数。YOLO 检测格式的数据集 YAML，例如 `coco8.yaml`。必填 |
| `imgsz` | `640` | 训练图像尺寸，用于基于像素的检查，例如小目标 |
| `fast` | `false` | 跳过图像解码，因此会丢掉损坏、重复和泄漏这几项检查 |
| `skip` | | 要跳过的检查 id 或家族，逗号分隔，例如 `images,labels.tiny_object` |
| `only` | | 要单独运行的检查 id 或家族，逗号分隔 |
| `strict` | `false` | 警告也会影响退出码，用于 CI 门禁 |
| `download` | `false` | 允许在数据集缺失时通过 URL 下载。绝不执行脚本 |
| `json` | `false` | 以 JSON 输出到 stdout |
| `quiet` | `false` | 静默 stderr |
| `help_json` | `false` | 以 JSON 输出命令 schema 并退出 |

### 检查家族

`skip` 和 `only` 既接受完整的检查 id，也接受家族前缀，因此 `images` 会选中所有
`images.*` 检查。

| 家族 | 覆盖内容 |
|---|---|
| `config` | 数据集 YAML 本身：缺少 `names`、`nc` 与 `names` 对不上、缺少划分、`path` 无法解析、类别名重复 |
| `files` | 图像与标注的配对：标注缺失、图像缺失、孤立标注、不支持的扩展名、大小写冲突 |
| `labels` | 标注内容：语法、多边形行、类别 id 越界、坐标越界、退化框、小目标、超大框、极端长宽比、重复框、目标过密的图像、内容相同的文件 |
| `images` | 像素数据：文件损坏、EXIF 方向、异常色彩模式、尺寸过小或极端、纯色图像、完全重复与近似重复 |
| `splits` | 划分之间的泄漏，完全重复与近似重复 |
| `balance` | 类别分布：实例数为零或过少的类别、不均衡、划分覆盖、背景占比、划分偏斜 |

## 示例

<code-tabs name="examples" />

## 说明

### 退出码

没有发现错误时为 `0`，只要有一条问题属于错误就是 `1`。加上 `strict=true` 后，
警告同样会把退出码抬到 `1`，这正是 CI 门禁想要的设置。

用法错误有自己的码：`skip` 或 `only` 里出现未知的检查 id 或家族时为 `2`，找不到
数据集时为 `3`，数据集不是检测形态时也是 `3`。

### 选择在扫描之前解析

在从磁盘读取任何东西之前，`skip` 和 `only` 就已经对着检查注册表（registry）解析
完毕，因此拼写错误会立刻失败，而不是等一趟漫长的图像扫描跑完之后。匹配不到任何
检查的选择器是错误，报错信息会列出已知的家族。

如果 `skip`、`only` 和 `fast` 组合下来没有任何检查可跑，这同样是错误，而不是静默
通过。

### 下载

除非 `download=true`，否则不会去拉取数据集，而且只会执行基于 URL 的下载。数据集
YAML 里内嵌的 Python 下载脚本，不管这个开关怎么设，这个命令都绝不会执行。

### 适用范围

这些检查是为检测数据集写的。标注是姿态、分割或旋转框形态的数据集会被识别出来，
并以 `data_invalid` 拒绝，而不是拿错误的规则去给它打分。

### 输出

面向人读的报告走 stdout，`json=true` 会把它换成一个结构化对象，里面带有汇总计数、
数据集统计、每一条问题，以及被跳过的检查列表。

相关：[`libreyolo train`](/docs/cli/train)，这条命令就是要在它之前跑的。
