---
title: libreyolo label
seo_title: libreyolo label 命令参考
description: 启动本地检测框标注工具：带默认值的参数、AI 辅助开关，以及绑定到网络接口会暴露什么。
lead: 启动一个本地网页工具，用来绘制和编辑检测框。它写出 LibreYOLO 原生的标注文件，因此在这里标注的数据集不需要任何转换步骤就能训练。
keywords:
  - libreyolo label cli
  - 检测框标注工具
  - yolo 标注工具
  - 自动标注 cli
  - libreyolo label 共享
last_verified: 1.5.0
meta:
  - label: 命令
    value: libreyolo label
    mono: true
  - label: 输出
    value: stdout 上输出一个服务器 URL；标注写入图像旁边的 labels/*.txt
snippets:
  examples:
    - label: 基础
      language: bash
      code: |
        # 打开项目主页，在浏览器里选择或新建数据集
        libreyolo label
    - label: 仅手动，固定端口
      language: bash
      code: |
        libreyolo label no_assist=true port=9200 no_browser=true
    - label: 让同事加入
      language: bash
      code: |
        libreyolo label share=true
source_hash: bddad245877793b1
---

## 概要

```bash
libreyolo label [data=<dataset.yaml|folder>] [key=value ...]
```

参数是 `key=value` 形式的键值对，POSIX 形式同样可用，所以 `port=9200` 和
`--port 9200` 是同一个参数。

## 参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `data` | | 直接打开的数据集 YAML 或文件夹。未设置时从项目主页开始 |
| `host` | `127.0.0.1` | 要绑定的主机或网络接口 |
| `port` | `8000` | 要绑定的端口。被占用时顺延到下一个空闲端口 |
| `device` | `auto` | AI 自动标注使用的设备：`0`、`cpu`、`mps`、`auto` |
| `no_assist` | `false` | 关闭 AI 自动标注，只留下手动标注工具 |
| `no_browser` | `false` | 不自动打开浏览器 |
| `share` | `false` | 绑定 `0.0.0.0`，让同一网络里的同事可以加入 |
| `json` | `false` | 向 stdout 输出 JSON |
| `quiet` | `false` | 静默 stderr |
| `verbose` | `false` | 输出详细的 stderr 信息 |

## 示例

<code-tabs name="examples" />

## 说明

### 它写出什么

检测框保存为 LibreYOLO 原生的 `labels/*.txt` 文件，这正是 `libreyolo train`
读取的格式，所以之后不需要做任何转换。这个版本只处理检测框。在图像之间切换时，
修改会随之保存。

### 打开数据集

不带 `data` 时，工具从项目主页启动，数据集在浏览器里选择或新建。传入
`data=path/to/data.yaml` 会直接打开那个数据集，启动行会报告图像数量、类别数量，
以及数据集是否可写。只读的数据集仍然可以打开，并会说明为什么无法写入。

### 共享，以及 `host` 的作用

`share=true` 绑定通配地址，让同一网络里的其他机器能访问这个工具，而管理操作，
也就是切换或删除项目、启动计算，仍然留在本机。

把 `host` 设成某个具体接口做的是另一件事，而且更不安全：本机变得和网络客户端无法
区分，于是每个客户端都拿到管理权限。这样做时，命令会在 stderr 上打印一条警告。
优先用 `share=true`。

### 端口与退出

端口被占用时会顺延到下一个，最多试到请求端口之后的第二十个。二十个都失败则以
`io_error` 退出。stdout 上打印的 URL 用的是实际绑定的端口。带 `share=true` 时，
结果里还会带上 `lan_url`，也就是同事应该打开的地址。

命令在前台提供服务，直到按下 Ctrl+C。

相关：用 [`libreyolo doctor`](/docs/cli/doctor) 在训练前检查标注好的数据集，用
[`libreyolo train`](/docs/cli/train) 在它上面训练。
