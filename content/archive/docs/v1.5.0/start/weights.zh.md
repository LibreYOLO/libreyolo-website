---
title: 检查点与权重
seo_title: LibreYOLO 检查点与权重
description: LibreYOLO 如何查找、下载和校验模型权重，权重托管在哪里，如何在没有网络的环境里运行，以及什么样的检查点才能安全加载。
lead: >-
  LibreYOLO 检查点是一个 torch.save 字典，里面装着一份 state
  dict，外加识别它所需的元数据。本页讲这些文件从哪里来、落到哪里，以及它们是怎么被加载的。
keywords:
  - libreyolo 权重下载
  - libreyolo 检查点
  - yolo 权重 离线
  - libreyolo hugging face
  - yolo 预训练权重
  - 检查点 元数据
last_verified: 1.5.0
meta:
  - label: 托管位置
    value: 每个检查点一个 Hugging Face 仓库：
    links:
      - label: huggingface.co/LibreYOLO
        href: 'https://huggingface.co/LibreYOLO'
  - label: 本地缓存
    value: 工作目录下的 weights/
    mono: true
  - label: 元数据 schema
    value: v1.0
snippets:
  load:
    - label: 自动下载
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 裸文件名会解析到 weights/LibreYOLO9t.pt，
        # 如果还不存在就下载到那里
        model = LibreYOLO("LibreYOLO9t.pt")
        print(model(SAMPLE_IMAGE).boxes)
    - label: 显式路径
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 带目录部分的路径会照原样使用，
        # 并且永远不会从网络获取
        model = LibreYOLO("/opt/models/LibreYOLO9t.pt")
        print(model.family, model.size, model.task)
  inspect:
    - label: CLI
      language: bash
      code: |
        # 读取元数据而不构建模型，
        # 并报告它是否满足 schema
        libreyolo metadata path=weights/LibreYOLO9t.pt
    - label: JSON
      language: bash
      code: |
        libreyolo metadata path=weights/LibreYOLO9t.pt --json
    - label: Python
      language: python
      code: >
        from libreyolo.utils.serialization import (
            load_untrusted_torch_file,
            validate_checkpoint_metadata,
        )


        loaded = load_untrusted_torch_file("weights/LibreYOLO9t.pt")


        # 返回问题列表，为空表示文件满足 v1.0

        print(validate_checkpoint_metadata(loaded))

        print(loaded["model_family"], loaded["size"], loaded["task"],
        loaded["nc"])
source_hash: 210a12baa1417cfb
---

## 检查点在哪里被查找

不带目录部分的模型引用，比如 `LibreYOLO9t.pt`，会相对当前工作目录解析到 `weights/`
下。如果 `weights/LibreYOLO9t.pt` 存在就用它；如果工作目录本身就有一个同名文件，
那就改用那个；否则 `weights/LibreYOLO9t.pt` 就成为下载目标。

带目录的引用——无论绝对还是相对——都按字面使用。当权重放在某个集中位置、并且不希望
发生任何下载时，就用这种写法。

<code-tabs name="load" />

## 自动下载

当解析出的路径不存在时，LibreYOLO 会解析文件名，还原出家族、尺寸和任务，再向对应的
家族索要一个下载 URL。多数家族基于 Hugging Face 上的 LibreYOLO 组织构建这个 URL，
那里每个检查点都有自己的仓库，仓库名就是文件名：

```text
https://huggingface.co/LibreYOLO/<name>/resolve/main/<name>.pt
```

数据集变体后缀仍然是仓库名的一部分，因此在家族默认数据集之外训练出的检查点会解析到
它自己的仓库，而不是覆盖掉默认的那个。

传输过程本身是防御性的，因为一个被截断的权重文件会在之后以一条毫无帮助的报错失败。
下载会流式写入一个 `.part` 文件，只有在完成时才原子地移动到位，因此被中断的进程绝不
会在最终路径上留下一个写了一半的检查点。中断的传输会借助 HTTP 验证器从它的字节偏移
处续传，如果服务器表明对象已经变了，就从零重新开始。失败会以指数退避重试三次。指向
同一路径的并发进程会占用一个锁文件，因此同时启动的两次训练只下载一次。当某个家族是
从第三方主机而不是 LibreYOLO 组织获取文件时，它可以固定一个校验和，并在不匹配时拒绝
该文件。

如果设置了 `HF_TOKEN`，或者 `~/.cache/huggingface/token` 下缓存了 token，它会作为
bearer token 附加上去。它只会附加到 `huggingface.co` 的 URL 上，因此从别的主机下载的
家族永远拿不到它。

并不是每个家族都会自动下载。有些故意不返回 URL，因为发布的权重不允许再分发，此时报错
会说明该改用什么。另一些会在传输开始前打印一条许可提示。那条提示就是运行时的信号，说明
这个检查点的条款比代码的更窄，值得读一读，而不是划过去。

## Hugging Face 组织

发布的权重放在 [huggingface.co/LibreYOLO](https://huggingface.co/LibreYOLO)，每个
检查点一个仓库。每个仓库都带有一份许可证，而且许可证在一个家族内部并不统一：代码采用
MIT 许可的家族，也可能有一些权重不是。以仓库为准。每个模型页都会在 Checkpoints 和
Licensing 小节列出该家族已发布的检查点及其许可证。

## 离线使用

一旦文件都在本地，这个库就没有任何地方需要网络访问。有两种做法可行：

在任务运行的位置旁边预先准备一个 `weights/` 目录。在一台联网的机器上把检查点取一次，
然后把这个目录复制过去就够了；上面那步解析会找到它们，永远不会访问网络。

或者传一个指向共享位置的绝对路径。带目录部分的引用会照原样使用，因此把整理好的权重以
只读方式挂载也是一种有效的配置。如果进程无法在它需要转换的检查点旁边写入，转换会退回
到一个私有临时目录，而不是直接失败。

数据集遵循另一套规则：它们解析到 `~/datasets` 下，或者在设置了
`LIBREYOLO_DATASETS_DIR` 变量时解析到该变量指定的目录下。

## 加载安全

检查点是 pickle，而 pickle 在被打开时可以执行任意代码。LibreYOLO 把每个权重文件都当作
不可信的，用 PyTorch 的 `weights_only=True` 路径加载它，这条路径把 unpickler 限制在
张量和一小组安全类型上。这对你传入的文件同样适用，而不只是对 LibreYOLO 下载的文件。
如果 PyTorch 的版本老到不支持这个参数，加载会被拒绝，而不是以不安全的方式执行。

有些上游训练检查点内嵌了受限 unpickler 会拒绝的对象，比如来自它们训练所用框架的一个
配置对象。这些对象是 LibreYOLO 并不需要的元数据，因此在转换过程中，每个被拦下的类都会
被替换成一个惰性替身，它能满足 unpickler 而不运行任何东西，最终只有张量保留进转换后的
文件。敏感的模块名会被直接拒绝，而不是替换成桩，重试循环也有上限，因此一个刻意构造出
无穷多被拦类的文件会以失败告终。这条路径的其余部分见[导入已有权重](/docs/migrate)。

## 检查点元数据

LibreYOLO 检查点是一个字典，它的 `model` 键放着 PyTorch 的 state dict。schema v1.0
要求九个键，它们合在一起，让工厂不必解析文件名、也不必从张量形状去猜，就能识别一个
文件。

| 键 | 含义 |
|---|---|
| `model` | PyTorch 的 state dict |
| `schema_version` | 元数据契约的版本。v1.0 用字符串 `1.0` |
| `libreyolo_version` | 生成该文件的 LibreYOLO 版本 |
| `model_family` | 一个已注册的家族标识符，比如 `yolo9` |
| `size` | 该家族内的变体，比如 `t` 或 `r18` |
| `task` | 一个规范任务名 |
| `nc` | 一个正的类别数 |
| `names` | 类别索引到标签的映射，覆盖 `0` 到 `nc - 1` |
| `imgsz` | 一个正的输入分辨率 |

带有额外结构的任务会把这部分信息记在上述键旁边。姿态检查点会加上 `num_keypoints` 和
`keypoint_dim`，还可能加上逐关键点的 OKS sigma。OCR 检查点内嵌完整的 CTC 字符集，
这样文件是自包含的。Restore 检查点可能记录退化类型和一个放大倍数。训练器检查点会加上
断点续训所需的状态，比如 `epoch`、优化器状态和 EMA 权重；发布出来的推理权重不应该带
这些。

满足全部九个键的文件走元数据路径加载。不满足的文件要么被转换——前提是有某个家族认得
它的结构——要么走兼容路径加载，并给出一条指明缺了什么的警告。

## 查看一个检查点

<code-tabs name="inspect" />

`libreyolo metadata` 从不构建模型，因此它对家族没有安装的文件、以及你拿不准的文件都
能用。
