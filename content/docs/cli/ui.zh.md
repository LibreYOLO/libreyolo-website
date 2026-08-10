---
title: libreyolo ui
seo_title: "libreyolo ui 命令参考"
description: "启动本地推理 Web 界面：绑定地址、端口行为、设备选择，以及命令如何退出。"
lead: "启动一个本地 Web 服务器，接受拖入或粘贴的图片，用你选定的模型对它们跑推理，并在浏览器里显示结果。"
keywords: [libreyolo ui cli, libreyolo web 界面, 本地推理 web 界面, 拖拽图片推理, libreyolo ui 端口]
last_verified: "1.5.0"
meta:
  - label: 命令
    value: libreyolo ui
    mono: true
  - label: 输出
    value: "在 stdout 上打印一个服务器 URL，随后进程留在前台"
snippets:
  examples:
    - label: 基本用法
      language: bash
      code: |
        libreyolo ui
    - label: 固定端口，不打开浏览器
      language: bash
      code: |
        libreyolo ui port=9000 no_browser=true
    - label: 在 CPU 上运行，机器可读输出
      language: bash
      code: |
        libreyolo ui device=cpu json=true
---

## 概要

```bash
libreyolo ui [key=value ...]
```

参数是 `key=value` 形式的键值对，POSIX 写法也能用，所以 `port=9000` 和
`--port 9000` 是同一个参数。

## 参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `host` | `127.0.0.1` | 要绑定的主机或网络接口 |
| `port` | `8000` | 要绑定的端口。被占用时顺延到下一个空闲端口 |
| `device` | `auto` | 设备：`0`、`cpu`、`mps`、`auto` |
| `no_browser` | `false` | 不自动打开浏览器 |
| `json` | `false` | 向 stdout 输出 JSON |
| `quiet` | `false` | 屏蔽 stderr |
| `verbose` | `false` | 更详细的 stderr 输出 |

## 示例

<code-tabs name="examples" />

## 说明

默认绑定的是回环地址（loopback），所以这个界面只能从本机访问。

如果请求的端口已被占用，命令会试下一个，一直往上找到请求端口之后的第二十个。
二十个都失败就以 `io_error` 退出，并提示你换一个端口。stdout 上打印的 URL 用的
是实际绑定的那个端口，所以要读它，而不是假定就是你要的那个。

除非设了 `no_browser=true`，绑定完成后不久会在该 URL 打开一个浏览器标签页。

随后命令在前台一直提供服务，直到你按 Ctrl+C，服务器会干净地关闭。没有分离运行
的模式；想把终端腾出来，就用你的 shell 把它放到后台。

`json=true` 会在服务器启动前，把 URL 和设备连同 `schema_version` 一起作为一个
对象打印出来，脚本就是这样拿到实际绑定的端口的。

相关：[`libreyolo label`](/docs/cli/label) 用于画检测框和保存标注，
[`libreyolo monitor`](/docs/cli/monitor) 用于查看训练过程。两者都是本地 Web
服务器，端口和浏览器行为与这里相同。
