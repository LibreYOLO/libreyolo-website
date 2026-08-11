---
title: 设置
seo_title: LibreYOLO 环境变量与目录
description: LibreYOLO 读取的每一个环境变量、它写入的目录、它需要的令牌，以及决定走哪条代码路径的开关。
lead: LibreYOLO 没有配置文件。凡是不通过函数参数控制的行为，都由环境变量和少量约定目录决定，这里把它们全部列了出来。
keywords:
  - LIBREYOLO_DATASETS_DIR
  - LIBREYOLO_KERNELS
  - LIBREYOLO_FASTER_COCO_EVAL
  - HF_TOKEN
  - libreyolo 权重目录
  - libreyolo 环境变量
last_verified: 1.5.0
verification: >-
  变量通过在 v1.5.0 的 libreyolo/**/*.py 中搜索 os.environ 和 os.getenv
  定位，语义在每个使用点读取。目录约定读取自
  libreyolo/data/utils.py、libreyolo/utils/download.py、libreyolo/export/exporter.py、libreyolo/models/base/model.py
  和 libreyolo/models/sam3dbody/mhr_body.py。
snippets:
  usage:
    - label: 把数据集根目录指向别处
      language: bash
      code: |
        export LIBREYOLO_DATASETS_DIR=/data/datasets
        python -c "from libreyolo.data import DATASETS_DIR; print(DATASETS_DIR)"
    - label: 从 Python 读取解析后的值
      language: python
      code: |
        from libreyolo.data import DATASETS_DIR

        # 默认为 ~/datasets，LIBREYOLO_DATASETS_DIR 在导入时覆盖它
        print(DATASETS_DIR)
source_hash: 462f1288582225ce
---

## 环境变量

| 变量 | 默认值 | 作用 |
|---|---|---|
| `LIBREYOLO_DATASETS_DIR` | `~/datasets` | 数据集根目录。导入时读取一次，写入 `libreyolo.data.DATASETS_DIR` |
| `LIBREYOLO_FASTER_COCO_EVAL` | 未设置 | 覆盖 `faster_coco_eval` 验证开关。`1`、`true`、`yes` 或 `on` 强制启用更快的后端，其他任何值强制关闭，不设置则听从配置里的开关 |
| `LIBREYOLO_KERNELS` | 未设置 | 内核选择。`off` 或 `reference` 强制使用参考实现；其他任何值只选取以该名称注册的实现 |
| `LIBREYOLO_QUANT_KERNELS` | 未设置 | `LIBREYOLO_KERNELS` 的旧别名，只有在后者未设置时才读取 |
| `LIBREYOLO_HUB_KERNELS` | 未设置 | `0`、`false`、`off` 或 `no` 会禁用 Hugging Face Hub 内核加载。其他任何值，包括未设置，都保持启用 |
| `LIBREYOLO_MHR_PATH` | `~/.cache/libreyolo/mhr/mhr_model.pt` | `mesh` 任务所用的 MHR 人体模型的位置 |
| `LIBRELABEL_ENABLE_LOCATE` | 未设置 | 必须恰好是 `1`、`true`、`yes` 或 `on`，才会在标注工具里开放 LocateAnything 助手。其他任何值都保持关闭 |
| `SAM_3D_BODY_PATH` | 未设置 | mesh 家族用的 SAM 3D Body 包路径，在没有传给构造函数时使用 |
| `HF_TOKEN` | 未设置 | Hugging Face 访问令牌，用于受限（gated）仓库 |

<code-tabs name="usage" />

`LIBREYOLO_DATASETS_DIR` 在导入时读取，所以在导入 `libreyolo.data` 之后再设置它，
对 `DATASETS_DIR` 没有影响。

Hub 内核的启用分两部分。只有装了可选的 `kernels` 包，运行时才会去拉取，所以安装
`libreyolo[hub-kernels]` 就是启用，`LIBREYOLO_HUB_KERNELS=0` 就是关闭。没装这个
额外依赖的安装，两边都不受影响。

内核选择还会短路导入：当 `LIBREYOLO_KERNELS` 强制为 `off` 或 `reference` 时，源码
树内的加速实现根本不会被导入。这三个变量控制的注册表（registry）记录在
[kernels](/docs/reference/kernels)。

## 库自己设置的变量

这些变量是被写入的，不是被读取的，所以手动设置它们并不是受支持的做法。

| 变量 | 设置方 |
|---|---|
| `RANK`、`LOCAL_RANK`、`WORLD_SIZE`、`MASTER_ADDR`、`MASTER_PORT` | DDP spawn 辅助函数，每个工作进程一个值 |
| `CUDA_VISIBLE_DEVICES` | 分布式初始化期间临时收窄，之后恢复 |
| `PYTORCH_ENABLE_MPS_FALLBACK` | 由 EC 训练器用 `setdefault` 设为 `1`，所以已有的值优先 |
| `MOMENTUM_ENABLED` | 由 mesh 家族的加载器用 `setdefault` 设置 |

`LOCAL_RANK` 同时充当分布式模式的信号：训练代码就是靠它是否出现在环境里，来判断
自己是不是跑在 DDP 下。

## 日志记录器变量

可选的训练日志记录器在项目名上会回退到环境里的默认值。

| 变量 | 默认值 | 使用方 |
|---|---|---|
| `WANDB_PROJECT` | `libreyolo` | Weights and Biases 日志记录器，在没有传入项目名时 |
| `COMET_PROJECT_NAME` | `libreyolo` | Comet 日志记录器，在没有传入项目名时 |

这些服务的认证走它们自己的工具，不归 LibreYOLO 管。

## 令牌

`HF_TOKEN` 就是 Hugging Face 的访问令牌。它未设置时，令牌从
`~/.cache/huggingface/token` 读取，那也是 Hugging Face CLI 登录写入的位置。两条
路径都可以。

只有受限仓库才需要令牌。SAM 3 是随库附带的例子：它的权重从一个采用自定义许可的
受限仓库下载，所以必须在仓库页面上接受条款，会话也必须通过认证。

## 目录

| 路径 | 内容 |
|---|---|
| `weights/` | 下载的检查点（checkpoint）、下载的 Hugging Face 快照，以及导出的产物 |
| `~/datasets` | 数据集根目录，除非 `LIBREYOLO_DATASETS_DIR` 另有指定 |
| `~/.cache/huggingface/token` | Hugging Face 令牌，在它不放进 `HF_TOKEN` 时 |
| `~/.cache/libreyolo/mhr/mhr_model.pt` | MHR 人体模型，除非 `LIBREYOLO_MHR_PATH` 另有指定 |
| `runs/track/` | `model.track(save=True)` 的默认输出位置 |

`weights/` 是相对于工作目录的。纯文件名会通过它解析，所以
`LibreYOLO("LibreYOLO9t.pt")` 会去找 `weights/LibreYOLO9t.pt`，文件不在时就下载到
那里。没有给出 `output_path` 时，`model.export()` 写入同一个目录。同级层
（sibling tier）会把多文件快照下载到 `weights/<Prefix><size>/`。

## 下载行为

权重下载会带退避重试三次，能从半截的文件续传，并由一个锁文件守着，这样两个进程
不会同时拉取同一个检查点。从第三方主机拉取的家族可以固定一个校验和，不匹配时直接
判定失败。

有些下载在开始前会打印一段许可证声明。这些声明是下载路径的一部分，无法通过配置
关掉。

## 验证后端

`model.val()` 默认接受 `faster_coco_eval=True`，在该包没有安装时回退到
pycocotools，并警告一次。设置 `LIBREYOLO_FASTER_COCO_EVAL` 会覆盖单次调用的开关，
动不了每次运行配置的基准测试框架应该用它。实际运行的后端会报告在
`model.last_eval_backend` 上。

## 数据集下载脚本

数据集 YAML 里可能带一个装着 Python 代码的 `download` 字段。除非给读取它的那次
调用传了 `allow_download_scripts=True`，否则它不会执行；这是 `val()` 和 `export()`
上的函数参数，不是环境变量。
