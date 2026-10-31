---
title: libreyolo monitor
seo_title: libreyolo monitor 命令参考
description: 为训练 run 提供一个实时仪表盘：带默认值的参数、服务端从磁盘读取哪些文件，以及一个服务端如何覆盖多个 run。
lead: 为训练 run 提供一个网页仪表盘，读取 run 写到磁盘上的产物。它从不挂接到训练进程，所以正在跑的、已结束的和已崩溃的 run 都能显示。
keywords:
  - libreyolo monitor cli
  - yolo 训练可视化
  - 实时查看训练过程
  - libreyolo monitor 端口
  - 训练指标面板
last_verified: 1.5.0
meta:
  - label: 命令
    value: libreyolo monitor
    mono: true
  - label: 输出
    value: 在 stdout 上打印一个服务地址，然后进程留在前台
snippets:
  examples:
    - label: 基础
      language: bash
      code: |
        # 监视 runs/，列出其下的每个 run
        libreyolo monitor
    - label: 另一个 runs 根目录
      language: bash
      code: |
        libreyolo monitor experiments/
    - label: 单个 run、固定端口、不开浏览器
      language: bash
      code: |
        libreyolo monitor runs/train/exp port=9100 no_browser=true
source_hash: 4aa178141d451728
---

## 概要

```bash
libreyolo monitor [<run-dir|runs-root>] [key=value ...]
```

目录是位置参数。其余都是 `key=value` 形式，POSIX 写法也可以，所以 `port=9100`
和 `--port 9100` 是同一个参数。

## 参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `run_dir` | `runs` | 位置参数。要监视的 runs 根目录，或者直接打开的单个 run 目录。两种写法都会列出根目录下的每个 run |
| `host` | `127.0.0.1` | 绑定的主机或网卡 |
| `port` | `8420` | 绑定的端口。被占用时顺延到下一个空闲端口 |
| `no_browser` | `false` | 不自动打开浏览器 |
| `json` | `false` | 向 stdout 输出 JSON |
| `quiet` | `false` | 抑制 stderr |
| `verbose` | `false` | 输出详细的 stderr 信息 |

## 示例

<code-tabs name="examples" />

## 说明

### 一个服务端，多个 run

服务端监视的是 runs 根目录而不是单个 run，并用 URL 来定位每个 run，所以一台机器
上的多个 run 共用一个端口。打开根 URL 看索引页，或者每个 run 开一个标签页；每个
URL 里的 `?run=` 参数标明是哪一个。

把命令指向单个 run 目录时，服务端会以该目录的父目录为根，所以同级的 run 仍会出现
在索引里，同时直接深链到指定的那个 run。

### 它读取什么

仪表盘由 `libreyolo train` 写出的文件构建：`status.json`、`metrics.jsonl`、
`train.log` 以及该 run 的图像。不会从训练进程本身读取任何东西，所以已经结束或者
已经挂掉的 run，显示效果和正在跑的完全一样。

### 前置条件与端口

至少要已经存在一个 run。不带参数且没有 `runs/` 目录时，命令以 `source_not_found`
退出；给定的目录里没有 run 时也一样。

端口被占用时会顺延到下一个，最多顺延到请求端口之后的第二十个。二十个都失败则以
`io_error` 退出。打印到 stdout 的 URL 用的是实际绑定的端口。

命令在前台一直服务，直到 Ctrl+C。`json=true` 会把 URL、监视的根目录和找到的 run
数量，作为一个带 `schema_version` 的对象打印出来。

相关：[`libreyolo train`](/docs/cli/train)，它的 `project` 和 `name` 参数决定这些
run 目录落在哪里。
