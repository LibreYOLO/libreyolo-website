---
title: 数据集
seo_title: LibreYOLO 中的训练数据集
description: LibreYOLO 读取的数据集 YAML、它期望的目录结构、自动下载如何工作，以及训练前检查数据集的 doctor 命令。
lead: >-
  LibreYOLO 的数据集就是一个 YAML
  文件，指明一个根目录、它的划分（split）和它的类别名。其余的一切，包括标注文件放在哪里，都由这个文件按约定推导出来。
keywords:
  - yolo 数据集格式
  - data.yaml
  - yolo 训练自己的数据集
  - yolo 标注格式
  - coco json 数据集
  - 数据集自动下载
  - libreyolo doctor
  - 类别不均衡检查
  - 训练集验证集泄漏
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # 内置名称、相对路径和绝对路径都可以
        model.train(data="coco8.yaml", epochs=10)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=coco8.yaml epochs=10
  doctor:
    - label: 检查数据集
      language: bash
      code: |
        libreyolo doctor my-dataset.yaml
    - label: 让警告也能让 CI 任务失败
      language: bash
      code: |
        libreyolo doctor my-dataset.yaml strict=true json=true
    - label: 跳过图像解码这一趟
      language: bash
      code: |
        # 只读取标注和 YAML，损坏、重复和划分泄漏这几项
        # 检查都需要像素，因此会被跳过
        libreyolo doctor my-dataset.yaml fast=true
    - label: Python
      language: python
      code: |
        from libreyolo import doctor

        report = doctor.diagnose("my-dataset.yaml", imgsz=640)

        for finding in report.findings:
            print(finding.severity.value, finding.check_id, finding.message)

        raise SystemExit(report.exit_code(strict=False))
source_hash: 9a12a0551c8b56e9
---

## 把 train 指向一个数据集

`data=` 接受一个 YAML 路径，或者随包提供的某个配置的名字。

<code-tabs name="train" />

名字按固定顺序解析：先是存在的绝对路径，然后是相对工作目录、按原样给出的名字，
接着是同一个名字加上 `.yaml`，最后是内置配置目录。全都对不上时，报错信息会列出
搜索过的每一个目录，并列出全部内置配置。

## 内置配置

包内自带十三份数据集配置，位于 `libreyolo/config/datasets/`。

| 配置 | 任务 | 说明 |
|---|---|---|
| `coco8.yaml` | detect | 8 张图像，从普通 URL 下载 |
| `coco128.yaml` | detect | 128 张图像 |
| `coco1000.yaml` | detect | 800 张训练，200 张验证 |
| `coco5000.yaml` | detect | 4000 张训练，1000 张验证 |
| `coco.yaml` | detect | 完整的 COCO 2017 |
| `coco-val-only.yaml` | detect | 只有 val2017 |
| `coco8-pose.yaml` | pose | 8 张图像，COCO-17 关键点 |
| `coco-pose.yaml` | pose | COCO 2017 关键点 |
| `ade20k.yaml` | semantic | 150 个类别 |
| `cityscapes.yaml` | semantic | 19 个类别，需手动下载 |
| `cocostuff.yaml` | semantic | 182 个类别，需手动下载 |
| `gopro.yaml` | restore | 去模糊图像对 |
| `sr8.yaml` | restore | 超分辨率图像对 |

只有 `coco8.yaml` 和 `coco128.yaml` 带了一个普通的下载 URL。其余的要么带一段
Python 下载代码块，需要下面讲的那个显式开关，要么就默认数据已经在磁盘上。

## 数据集在磁盘上的位置

YAML 里的 `path` 键指定数据集根目录。绝对的 `path` 按原样使用。相对的先在数据集
目录下找，然后在 YAML 文件自己旁边找，而即将下载的数据集会放进数据集目录。

这个目录是 `~/datasets`，可以用 `LIBREYOLO_DATASETS_DIR` 环境变量覆盖。它没有配置
文件。

## YAML 的各个键

```yaml
path: my-dataset        # 数据集根目录
train: images/train     # 训练必填
val: images/val         # 验证必填
test: images/test       # 可选
nc: 3                   # 可选，必须与 names 一致
names:
  0: person
  1: helmet
  2: vest
download: https://example.com/my-dataset.zip   # 可选
```

`train`、`val` 和 `test` 各自接受一个图像目录、一个每行一条图像路径的 `.txt`
文件，或者两者混排的列表。`.txt` 列表里的行可以是相对路径，这时它们相对列表文件
自己所在的目录解析，以 `#` 开头的行会被跳过。

`names` 可以是列表，也可以是整数键的映射。`nc` 是可选的；两者都给出且对不上时，
doctor 会把它报成错误。

## 目录结构与标注文件

检测、分割、姿态和旋转框共用同一套结构。标注路径由图像路径推导：把路径里的
`images` 目录段改写成 `labels`，扩展名换成 `.txt`：

```text
my-dataset/
  images/train/0001.jpg   ->   labels/train/0001.txt
  images/val/0002.jpg     ->   labels/val/0002.txt
```

只有完整的 `images` 路径段会被改写，所以叫 `images_old` 的目录不受影响。

一行检测标注是五个字段，全部按原始图像的宽和高归一化到 `[0, 1]`：

```text
<class_id> <cx> <cy> <w> <h>
```

标注文件缺失或为空，表示这张图像没有目标，它会作为背景参与训练，而不是抛错。超过
五个字段的行会被当成多边形读取，它的框取多边形的外接范围，因此一份拿来做检测训练
的分割导出也能顺利加载。doctor 会报告有多少行走了这条路径。

## 其他任务

分割沿用同一套结构，只是每行是多边形，`<class_id> <x1> <y1> ... <xN> <yN>`，
至少三个点。五字段的检测行也接受，表示一个矩形实例。

姿态在 YAML 里加上 `kpt_shape: [K, D]`，以及可选的 `flip_idx` 置换。每一行正好是
`5 + K * D` 个字段：先是框，然后是 `K` 个关键点，形式为 `x y` 或 `x y v`，可见性
取 `0`、`1` 或 `2`。

旋转框正好用九个字段，类别之后跟四个归一化坐标下的角点。文件里不存角度。

语义分割给每张图像配一张同分辨率的单通道掩码，路径由 `masks_dir`（默认 `masks`）
替换 `images` 解析出来。像素值 `255` 表示忽略。`label_mapping` 在加载时把源 id
重映射为训练 id。

分类用的是 ImageFolder 目录树，而不是标注文件，`train/` 和 `val/` 下各自每个类别
一个目录。类别到索引的映射就是文件夹名排序后的顺序。

图像修复通过 `input_dir` 和 `target_dir`，把一张退化的输入和一张同分辨率的干净目标
配成一对。深度、表面法线和边缘各自通过自己的目录键，把图像和一张稠密图配成一对。

每个任务完整的约定，包括深度的尺度约定和全景分割 segment-id 的 PNG 编码，在库仓库
里的 `docs/dataset_schema.md`。

## 原生 COCO JSON

COCO JSON 标注文件可以直接用。加上一个 `annotations` 映射，划分路径就变成图像根
目录：

```yaml
path: my-dataset
train: images/train
val: images/val
annotations:
  train: annotations/train.json
  val: annotations/val.json
```

给出 `names` 时，JSON 里的类别名必须与之对上，并且由 `names` 决定模型预测的标签
id。不给 `names` 时，COCO 的类别 id 会先排序，再稠密映射到 `0..N-1`。

这条路径要求每个划分只有一个图像目录。给一组路径或者一个 `.txt` 图像列表会抛错，
而不是悄悄加载另一批数据。

## 自动下载

当数据集的 `train` 或 `val` 路径解析到一个非空目录或一个已存在的文件时，就算它
已经在了。解析不到，而 YAML 里又有 `download` 键时，这个键的取值决定接下来发生
什么。

`http` 或 `https` 的 URL 会被下载下来，如果是 zip 就解压到数据集根目录。其他内容
一律当作内嵌的 Python 脚本，只有在 `allow_download_scripts=True` 时才执行。没有它，
脚本会被跳过并给出一条警告，训练继续用磁盘上现有的数据跑。

```bash
libreyolo train model=LibreYOLO9s.pt data=coco.yaml allow_download_scripts=true
```

这个开关管的是代码执行，不是网络。URL 下载两种情况下都会发生，需要它的是
`download: |` 这类代码块。开关打开时 CLI 会打印一条警告，而 doctor 从不启用它。

## 训练前先检查数据集

`libreyolo doctor` 读取一个检测数据集，在动用 GPU 之前就报告哪里会出问题。发现
错误时退出码为 1，因此它可以当作 CI 门禁用。

<code-tabs name="doctor" />

检查分为六个家族：

| 家族 | 覆盖内容 |
|---|---|
| `config` | 缺少 `names`、`nc` 与 `names` 对不上、划分缺失或为空、类别名重复 |
| `files` | 没有标注文件的图像、没有图像的标注、划分里列出却不存在的图像、主文件名冲突 |
| `labels` | 格式错误的行、`[0, nc)` 之外的类别 id、`[0, 1]` 之外的坐标、面积为零的框、过小或过大的框、重复框、逐位一致的标注文件 |
| `balance` | 实例数为零或过少的类别、类别不均衡比例、只出现在一个划分里的类别、背景图像占比 |
| `images` | 无法解码的文件、EXIF 旋转、异常的通道排布、纯色图像、完全重复与近似重复 |
| `splits` | 同一张图像出现在两个划分里，完全相同或近似相同 |

`--only` 和 `--skip` 接受一个检查 id 或一个家族前缀，因此
`skip=images,labels.tiny_object` 是合法的。`--fast` 会丢掉所有需要解码像素的检查，
也就是 `images` 和 `splits` 这两个家族。

有两个行为值得知道。`--strict` 让警告和错误一样影响退出码。另外 doctor 只覆盖检测
数据集：姿态、分割或旋转框的数据集会被拒绝，报错信息会说明它识别出的是什么，而不是
拿错误的约定去检查它。

## 相关

- [超参数](/docs/train/hyperparameters)：数据就位之后 `train()` 接受的参数。
- [验证与指标](/docs/train/validation)：在 `val` 或 `test` 划分上做评估。
