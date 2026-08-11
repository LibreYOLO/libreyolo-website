---
title: 数据集格式
seo_title: LibreYOLO 各任务的数据集格式
description: 按规范任务划分的数据集文件约定：YAML 键、目录结构、标注行、掩码与各类图的约定，以及读取它们的加载器。
lead: 本页镜像了库自身 docs/dataset_schema.md 里的数据集文件约定。它涵盖每个规范任务所要求的 YAML 键和磁盘布局。
keywords:
  - libreyolo 数据集格式
  - yolo 标注格式
  - data.yaml
  - 语义分割掩码数据集
  - coco panoptic 格式
  - 深度数据集
  - pose kpt_shape
last_verified: 1.5.0
verification: 镜像 libreyolo 仓库 v1.5.0 的 docs/dataset_schema.md，加载器名称与 libreyolo/data/ 交叉核对过。
snippets:
  usage:
    - label: 解析一行检测标注
      language: python
      code: >
        from libreyolo.data import parse_yolo_label_line


        # class_id cx cy w h，归一化到 [0, 1]

        row = parse_yolo_label_line("0 0.5 0.5 0.25 0.5", 640, 480,
        num_classes=80)


        # (class_id, x1, y1, x2, y2, area)，单位为像素

        print(row)
source_hash: a8282c079624044d
---

## 通用 YAML

适用于 `detect`、`segment`、`pose` 和 `obb`。

| 键 | 是否必填 | 含义 |
|---|---|---|
| `path` | | 数据集根目录 |
| `train` | 训练时必填 | 训练图像 |
| `val` | 验证时必填 | 验证图像 |
| `test` | | 测试图像 |
| `names` | 是 | 类别列表，或以整数为键的映射 |
| `nc` | | 类别数量；存在时必须与 `names` 一致 |
| `download` | | 下载说明；Python 脚本需要显式开启 |
| `annotations` | | 把数据集划分（split）映射到原生 COCO JSON 文件，适用于 detect、segment 和 obb |

`train`、`val` 和 `test` 可以是图像目录、图像列表 `.txt` 文件，或者由它们组成的
列表。标注路径遵循一次替换：

```text
images/.../image.jpg -> labels/.../image.txt
```

对于原生 COCO JSON 数据集，`annotations` 把一个划分映射到它的 JSON 文件，划分
路径给出图像根目录：

```yaml
path: dataset
train: images/train
val: images/val
annotations:
  train: annotations/train.json
  val: annotations/val.json
```

当 `names` 存在时，原生 COCO JSON 的类别名必须与 YAML 里的类别名一致，并且这些
名称决定模型的标签 ID。没有 `names` 时，COCO 类别 ID 会被排序并稠密映射到
`0..N-1`。

数据集 YAML 不带 `task` 键。显式指定的模型和任务优先。

对每个文本标注文件都适用的规则：

- 每张图像对应一个 `.txt` 标注文件；
- 标注文件缺失或为空表示没有目标；
- `class_id` 是 `0..nc-1` 范围内的整数；
- 坐标是 `[0, 1]` 内有限的归一化浮点数；
- 坐标相对于原始图像的宽和高；
- 每行不带置信度，也不带跟踪 ID。

<code-tabs name="usage" />

## detect

每行恰好五个字段：

```text
<class_id> <cx> <cy> <w> <h>
```

`cx cy w h` 是归一化的轴对齐检测框，`w` 和 `h` 必须为正。

## segment

一行多边形：

```text
<class_id> <x1> <y1> ... <xN> <yN>
```

`N` 至少为 3，`class_id` 之后的坐标数量必须是偶数，并且多边形必须非退化。五个
字段的检测行同样接受，表示一个矩形分割区域。

## pose

YAML 增加 `kpt_shape`，它是必填的，取值为 `[K, 2]` 或 `[K, 3]`，以及可选的
`flip_idx`，它是 `0..K-1` 的一个整数排列。

```text
<class_id> <cx> <cy> <w> <h> <k1x> <k1y> [<k1v>] ... <kKx> <kKy> [<kKv>]
```

字段数量恰好是 `5 + K * D`，其中 `D` 是 `kpt_shape` 的第二个值。关键点坐标是
归一化的。可见性 `v` 出现时取值为 `0`、`1` 或 `2`。

## obb

恰好九个字段：

```text
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```

这四个点是 `[0, 1]` 内归一化的图像坐标，构成一个非退化的旋转矩形。标注文件里
不存角度。

规范解析器默认是严格的，会拒绝超出范围的坐标。数据集和验证的读取过程可以把坐标
截断到 `[0, 1]`，用于其他方面合法的裁剪边界标注，之后仍然会拒绝退化的检测框。
解析是区分任务的：九个字段只有在 `obb` 模式下才表示 `obb`，而在 `segment` 模式下
它们可能是一个四点多边形。

在内部，归一化的角点会被转换成规范的 `xywhr`，角度以弧度表示宽边绕检测框中心的
旋转。公开结果把 OBB 检测以 `xywhr, conf, cls` 行的形式暴露出来。

原生 COCO JSON 的 OBB 读取按以下优先级接受标注：`obb` 为八个像素空间角点；`obb`
为 `[cx, cy, w, h, angle]`，角度以弧度表示；COCO 的 `segmentation` 多边形或 RLE，
重新拟合为最小面积矩形；以及 COCO 的 `bbox`，按轴对齐读取并规范化。

Mosaic 和 mixup 在 OBB 训练中被禁用，直到出现角点感知的 OBB 数据增强。

规范的行解析器是 `libreyolo.data.parse_yolo_obb_label_line`。

## semantic

每张图像配一张无损格式（通常是 PNG）的稠密单通道掩码，而不是 `.txt` 文件：

```text
images/.../image.jpg -> <masks_dir>/.../image.png
```

掩码是单通道的，调色板模式的 PNG 按调色板索引读取。每个像素值是 `0..nc-1` 内的
类别 ID，像素值 `255` 表示忽略，会被排除在损失函数和指标之外，并且掩码分辨率必须
与图像分辨率相等。

有两个可选的 YAML 键叠加在通用约定之上。`masks_dir` 是在每个图像路径中替换
`images` 的掩码目录名，默认为 `masks`。`label_mapping` 是一个
`{source_id: train_id}` 重映射，在加载时作用于掩码像素值，其中未映射的源值变成
忽略，训练 ID 必须落在 `0..nc-1` 内。

省略 `masks_dir` 时，掩码会在加载时由 `segment` 多边形标注栅格化得到，标注通过
`images` 到 `labels` 的约定解析，并且会在目标类别之后追加一个 `background` 类，
所以 `nc` 加一。

规范加载器：`libreyolo.data.SemanticDataset`。

## panoptic

LibreYOLO 原样采用 COCO-panoptic 格式（Kirillov 等，CVPR 2019）。不存在
LibreYOLO 专属的全景分割格式。

每张图像对应一张与图像分辨率相同的 RGB PNG，用颜色编码每个像素的 segment ID：

```text
segment_id = R + 256 * G + 256 * 256 * B
```

每个像素恰好属于一个 segment，segment 之间不会重叠。Segment ID `0`，也就是 RGB
黑色，是 void：未标注的像素，排除在指标之外。

```json
{
  "images":      [{"id": 139, "file_name": "000000000139.jpg"}],
  "annotations": [{"image_id": 139, "file_name": "000000000139.png",
                   "segments_info": [
                     {"id": 3226956, "category_id": 1, "area": 2840,
                      "bbox": [413, 158, 53, 138], "iscrowd": 0}]}],
  "categories":  [{"id": 1, "name": "person", "isthing": 1, "supercategory": "person"}]
}
```

`annotations[].file_name` 指出 `panoptic_dir` 里那张 segment ID PNG 的名字，
`segments_info[].id` 与该 PNG 中的某个值对应。`iscrowd` 标记群组区域：它们永远
不算漏检，而大部分覆盖了其中一个的预测也不算误检。

thing 与 stuff 的区分是按类别的属性。`isthing` 位于 `categories` 上，绝不在
`segments_info` 上。

COCO-panoptic 的 `category_id` 值是数据集的原始 ID，通常并不连续。模型预测的是
连续的 `0..nc-1`，所以原始 ID 会按类别名通过 YAML 的 `names` 重映射，和原生
COCO JSON 的 detect 加载器遵循同一条规则。JSON 里某个类别在 `names` 中缺失会
报错，而不是被静默丢弃，因为否则它会被永久算作漏检。

```yaml
path: coco
val: images/val2017
annotations:
  val: annotations/panoptic_val2017.json
panoptic_dir:
  val: annotations/panoptic_val2017
names: {0: person, 1: bicycle, 132: rug-merged}
```

`annotations` 和 `panoptic_dir` 接受单个路径或按划分的映射。

验证会报告全景质量（Panoptic Quality），它在真值（ground truth）分辨率下计算，
并在出现过的类别上取平均，然后拆分为 `PQ_things` 和 `PQ_stuff`。匹配是唯一的：
同一类别的预测 segment 和真值 segment 在 IoU 高于 0.5 时匹配。

规范加载器：`libreyolo.data.PanopticDataset`。

## depth

每张图像配一张稠密的单通道深度图：

```text
images/.../image.jpg -> <depths_dir>/.../image.png
```

这张图是与图像分辨率相同的单通道 PNG 或 TIF，或者一个 `.npy` 文件。数值是以
数据集内统一的单位表示的原始深度。零、负数、NaN 和无穷值标记无效像素，会被排除
在损失函数和指标之外。

| 键 | 默认值 | 含义 |
|---|---|---|
| `depths_dir` | `depths` | 替换 `images` 的深度目录 |
| `depth_stem_suffix` | | 追加到图像主干名（stem）之后的后缀；省略时会同时尝试相同主干名和 `_depth` 后缀 |
| `depth_mask_suffix` | `_mask` | 有效性掩码的后缀；掩码值小于等于零、NaN 和无穷会让该深度像素失效 |
| `depth_scale` | `256.0` | 整数类型深度图的除数，也就是常见的 16 位 PNG 约定 |

浮点 `.npy` 图按原样使用，不应用 `depth_scale`。

规范加载器：`libreyolo.data.DepthDataset`。

## edge

每张 RGB 图像配一张同主干名的单通道无损图，以及一张可选的有效性掩码：

```text
images/val/scene.jpg -> edges/val/scene.png
                     -> masks/val/scene.png
```

这张图是单通道 PNG 或 TIF，不是 RGB 可视化结果，分辨率与图像相同。整数图会除以
其 dtype 的最大值；浮点图必须本身就是有限值且落在 `[0, 1]` 内。`0` 表示非边缘，
`1` 表示边缘。可选掩码的像素在非零时有效。缩放对目标和掩码使用最近邻插值，填充
出来的像素无效，不参与验证。

| 键 | 默认值 | 含义 |
|---|---|---|
| `edges_dir` | `edges` | 替换 `images` 的边缘图目录 |
| `edge_stem_suffix` | | 追加到图像主干名之后的后缀 |
| `edge_extension` | `.png` | 无损目标文件的扩展名 |
| `edge_invert` | | 当源图以白底黑边存储时设为 true |
| `masks_dir` | `masks` | 可选的有效性掩码目录 |

```yaml
path: edge-dataset
train: images/train
val: images/val
edges_dir: edges
masks_dir: masks
nc: 1
names: {0: edge}
```

验证用四方向梯度的非极大值抑制来细化连续预测，并在一段可配置的阈值扫描上报告
ODS 和 OIS 的 F-measure。预测像素和真值像素在 `edge_max_dist * image_diagonal`
之内一对一匹配，默认的归一化容差是 `0.0075`。

规范加载器：`libreyolo.data.EdgeDataset`。这个加载器只负责格式：它不下载也不
再分发基准测试数据。

## normal

每张图像配一张同主干名的三通道 16 位 PNG，外加一张可选的同主干名有效性掩码：

```text
images/val/room.jpg -> normals/val/room.png
                    -> masks/val/room.png
```

这张 PNG 恰好是三通道 `uint16`，通道按 RGB 存储，分辨率与图像相同。用
`n = png / 65535 * 2 - 1` 解码，然后对每个向量重新归一化。解码出的向量使用
OpenCV 相机坐标系，`+x` 向右、`+y` 向下、`+z` 指向场景内部，并且朝向相机。可选
掩码是单通道 PNG，非零表示有效；没有掩码时，每个有限且非零的解码向量都有效。
无效的和填充出来的目标像素在内部用 `(0, 0, 0)` 表示。缩放时三个分量做双线性
插值再重新归一化，有效性掩码使用最近邻插值，水平翻转还会把 x 分量取反。

| 键 | 默认值 | 含义 |
|---|---|---|
| `normals_dir` | `normals` | 替换 `images` 的法向图目录 |
| `masks_dir` | `masks` | 可选的有效性掩码目录 |

验证报告以度为单位的平均和中位角度误差，以及落在 11.25、22.5 和 30 度以内的
有效像素百分比。

规范加载器：`libreyolo.data.NormalDataset`。

## restore

每张退化的输入图像配一张干净的 RGB 目标图：

```text
inputs/.../image.jpg -> targets/.../image.jpg
```

输入和目标是 RGB 兼容的图像文件，它们的分辨率必须完全一致。验证保持原始分辨率，
只填充到刚好能堆成一个批次，指标在原始图像画布上计算。训练对输入和目标这一对
施加耦合的裁剪和水平翻转。

| 键 | 默认值 | 含义 |
|---|---|---|
| `input_dir` | `inputs` | 划分路径里使用的退化输入目录 |
| `target_dir` | `targets` | 替换 `input_dir` 的干净目标目录 |
| `target_stem_suffix` | | 查找目标之前追加到输入主干名之后的后缀 |
| `target_stem_suffixes` | | `target_stem_suffix` 的列表形式 |
| `degradation` | | 元数据标签，例如 `deblur` 或 `denoise` |
| `dataset` | | 数据集或来源标签 |

那些类别相关的 YAML 字段只是模式占位符：用 `nc: 1` 和 `names: {0: image}`。
restore 模型暴露的是 `Results.restored`，不是检测结果。

规范加载器：`libreyolo.data.RestoreDataset`。

## matte

每张 RGB 图像配一张同主干名的单通道真值 matte，其中 0 是背景，255 是前景：

```text
images/subject.jpg -> mattes/subject.png
```

接受两种布局。一种是包含 `images/` 和一个 matte 目录的目录根，matte 目录在
`mattes/`、`matte/`、`gt/`、`masks/`、`mask/` 和 `alpha/` 中自动检测，通过
`data=` 传入。另一种是一个 YAML，带 `path` 加上按划分的 `val_images` 和
`val_mattes`，以及可选的 `train_images` 和 `train_mattes`，每一项相对于 `path`
或者写成绝对路径。

matte 是灰度图，按 `[0, 1]` 内的不透明度读取，形状不一致时会用双线性插值缩放到
预测画布。指标是原始图像画布上的 MAE 和 S-measure（Fan 等，ICCV 2017），并以
S-measure 作为挑选最佳检查点（checkpoint）的 fitness。

那些类别相关的 YAML 字段只是模式占位符：用 `nc: 1` 和 `names: {0: matte}`。
matte 模型暴露 `Results.matte`。

本版本中验证只做推理。规范的配对解析器：
`libreyolo.data.matte_dataset.resolve_matte_pairs`。

## ocr

标注是每个划分一个 JSONL 文件，每张图像一个 JSON 对象：

```text
images/val/receipt.jpg -> labels/val.jsonl
```

```json
{"image": "receipt.jpg", "regions": [{"polygon": [[10, 12], [118, 14], [117, 40], [9, 38]], "text": "TOTAL 12.50"}]}
```

`polygon` 是绝对像素坐标下的四点四边形，按左上、右上、右下、左下的顺序排列。
文本不可辨认的区域使用 `"text": "###"`，也就是 ICDAR 的 do-not-care 约定：它们
被排除在识别评分之外，与它们重叠的预测在检测匹配中被忽略，而不是被扣分。

指标是：一对一多边形匹配、IoU 高于 0.5 的检测 hmean；要求 IoU 高于 0.5 且经过
NFKC 归一化和去除空白后转写完全一致、区分大小写的端到端 F1；以及匹配对上的
1-NED。挑选最佳检查点的 fitness 是端到端 F1。

接受两种布局：一种是包含 `images/<split>/` 和 `labels/<split>.jsonl` 的目录根，
通过 `data=` 传入；另一种是一个 YAML，带 `path` 加上可选的 `images` 和 `labels`
目录名。

那些类别相关的 YAML 字段只是模式占位符：用 `nc: 1` 和 `names: {0: text}`。
OCR 模型暴露 `Results.ocr`。

本版本中验证只做推理。规范的样本解析器：
`libreyolo.data.ocr_dataset.resolve_ocr_samples`。

## classify

一棵 ImageFolder 风格的目录树，而不是标注文件：

```text
dataset_root/
  train/
    class_a/*.jpg
    class_b/*.jpg
  val/
    class_a/*.jpg
    class_b/*.jpg
```

`train/` 是训练所必需的，并按排序后的文件夹名定义类别到索引的映射。`val/` 是
验证所必需的。`test/` 可以存在，但默认的 train 和 val 命令不会用它。非训练的
划分必须包含与预期的训练类别集合或检查点类别集合相同的类别文件夹名。支持的图像
扩展名定义在 `libreyolo.data.classify_dataset.IMAGE_EXTENSIONS` 里。

## gaze 与 point

`gaze` 没有实现训练或验证的数据集文件约定。

`point` 是一种模型输出任务，而不是数据集标注模式。point 家族可以在内部改造已有的
标注，例如从检测框行推导出目标中心，但没有定义只含点的文本标注格式。
