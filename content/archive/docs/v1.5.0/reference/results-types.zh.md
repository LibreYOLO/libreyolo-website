---
title: Results 类型
seo_title: LibreYOLO Results 对象参考
description: >-
  LibreYOLO Results
  对象能携带的每一种载荷，每种任务形态一个槽位：boxes、masks、keypoints、probs、obb、depth、ocr、embeddings，以及另外十个。
lead: Results 是每个 LibreYOLO 模型统一的单图返回类型。它带有十八个可选的载荷槽位，每种任务形态一个，并且只填充模型真正产出的那些。
keywords:
  - libreyolo results 对象
  - Results.boxes
  - Results.masks
  - Results.probs
  - Results.depth_map
  - Results.summary
  - libreyolo results 转 json
last_verified: 1.5.0
verification: 槽位名称、形状、属性与默认值读取自 v1.5.0 的 libreyolo/utils/results.py。语义引自各载荷类的文档字符串。
snippets:
  usage:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")
        result = model(SAMPLE_IMAGE)

        print(result.orig_shape, result.path)
        print(result.boxes.xyxy)
        print(result.boxes.conf)
        print(result.names[int(result.boxes.cls[0])])
  convert:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")
        result = model(SAMPLE_IMAGE)

        # 所有载荷一起转移
        result = result.cpu().numpy()

        # 行数据，先是普通 dict，再是 JSON
        print(result.summary()[:1])
        print(result.to_json())
source_hash: 16f654364ae6448a
---

## Results 对象

一个 `Results` 描述一张图像。单张图像的源返回一个，列表源或目录返回一个列表，
`stream=True` 则返回一个逐个产出它们的生成器。

| 属性 | 类型 | 含义 |
|---|---|---|
| `orig_shape` | `(int, int)` | 原始图像的高和宽 |
| `path` | `str` | 输入来自磁盘时的源路径 |
| `names` | `dict[int, str]` | 类别索引到类别名 |
| `speed` | `dict[str, float]` | 每个阶段的毫秒数 |
| `track_id` | 张量 | 结果来自 `track()` 时的跟踪 ID |
| `frame_idx` | `int` | 视频和流源的帧索引 |
| `restore_scale` | `int` | 复原结果输出对输入的放大倍数；其他情况下都是 `1` |

<code-tabs name="usage" />

## 载荷槽位

除非模型产出了它，否则每个槽位都是 `None`。一个家族填哪个槽位，由它的任务决定。

| 槽位 | 类 | 任务 |
|---|---|---|
| `boxes` | `Boxes` | detect |
| `masks` | `Masks` | segment |
| `keypoints` | `Keypoints` | pose |
| `probs` | `Probs` | classify |
| `obb` | `OBB` | obb |
| `gaze` | `Gaze` | gaze |
| `points` | `Points` | point |
| `semantic_mask` | `SemanticMask` | semantic |
| `panoptic` | `PanopticSegmentation` | panoptic |
| `depth_map` | `DepthMap` | depth |
| `normal_map` | `NormalMap` | normal |
| `edges` | `EdgeMap` | edge |
| `restored` | `RestoredImage` | restore |
| `matte` | `Matte` | matte |
| `ocr` | `OCRRegions` | ocr |
| `embeddings` | `Embeddings` | embed |
| `identities` | `Identities` | embed，带图库（gallery） |
| `meshes` | `Meshes` | mesh |

`result.normals` 是 `result.normal_map` 的可读写别名。

多个槽位可以同时被填上。分割模型会同时填 `boxes` 和 `masks`；视线模型用人脸框填
`boxes`，用角度填 `gaze`；网格模型用人体框填 `boxes`，并让 `meshes` 与之逐行对齐。

## Boxes

一张图像的检测框。

| 成员 | 返回 |
|---|---|
| `xyxy` | 原始图像像素下的角点坐标 |
| `xywh` | 像素下的中心点和尺寸 |
| `xyxyn` | 归一化到 `[0, 1]` 的角点 |
| `xywhn` | 归一化到 `[0, 1]` 的中心点和尺寸 |
| `conf` | 每个框的置信度 |
| `cls` | 每个框的类别索引 |
| `id` | 每个框的跟踪 ID，没有则为 `None` |
| `is_track` | 存在跟踪 ID 时为 `True` |
| `data` | 打包后的张量 |

`with_id(id)` 和 `with_orig_shape(orig_shape)` 返回一个替换了对应字段的新
`Boxes`。

## Masks

一张图像的实例掩码。`data` 是掩码张量；`xy` 返回每个实例以像素表示的轮廓，
`xyn` 返回归一化后的轮廓。

## Keypoints

姿态关键点，与 `boxes` 逐行对齐。`xy` 是每个关键点的坐标对，`xyn` 是归一化后的
坐标对。数据带第三个通道时 `conf` 就是它，否则为 `None`。`has_visible` 是一个
布尔数组，`conf > 0` 的地方为真；没有置信度通道时全为真。

## Points

一张图像的点定位。`data` 的形状是 `(N, 4)`，每行是 `x, y, class, confidence`。
坐标是绝对像素；`xy`、`cls` 和 `conf` 把这些列拆开，`xyn` 则把坐标归一化。

## Probs

分类分数。`top1` 是获胜的索引，`top5` 是最好的五个索引，`top1conf` 和
`top5conf` 是它们的分数。

## OBB

旋转框。`data` 每行存 7 或 8 个值：`xywhr`、一个可选的跟踪 ID，然后是置信度和
类别。

| 成员 | 返回 |
|---|---|
| `xywhr` | 中心点、尺寸，以及以弧度表示的旋转角 |
| `xyxyxyxy` | 以像素表示的四个角点 |
| `xyxyxyxyn` | 归一化后的四个角点 |
| `xyxy` | 以像素表示的轴对齐外接框 |
| `conf`, `cls`, `id`, `is_track` | 与 `Boxes` 上的一样 |

## Gaze

每张人脸的视线角度，以弧度表示，形状为 `(N, 2)`，与 `boxes` 里的人脸框逐行
对齐。第 0 列是俯仰角（pitch），第 1 列是偏航角（yaw），采用 L2CS 约定：偏航角
为正时视线朝被摄者的左侧转，俯仰角为正时视线向下转。`pitch_deg` 和 `yaw_deg`
换算成角度，`direction_3d` 返回单位方向向量。

## SemanticMask

稠密语义图，形状为 `(H, W)`，是原始图像画布上的整数类别 ID。`255` 是忽略值，
永远不算作一个类别（`SemanticMask.IGNORE_INDEX`）。`classes` 列出出现过的类别
ID，`class_mask(class_id)` 返回某一个类别的布尔掩码。

## PanopticSegmentation

每个像素恰好属于一个互不重叠的分段，把 stuff 区域和 thing 实例统一起来。`data`
是一张 `(H, W)` 的整数分段 ID 图；分段 ID `0` 表示无标注
（`PanopticSegmentation.IGNORE_INDEX`）。`segments_info` 是一个 dict 列表，每个
分段一项，每项至少包含 `{"id": int, "category_id": int}`，其中 `id` 对应图里的
一个值，`category_id` 是 `names` 的索引。`segment_ids` 列出出现过的 ID，
`segment_mask(segment_id)` 返回某一个分段的布尔掩码。

thing 与 stuff 的区分是类别的属性，不是分段的属性。载荷可以把它反规范化到每个
分段上，写成 `"isthing": bool`，这么做的时候，这个值必须和类别级别的映射一致。

## DepthMap

稠密的相对逆深度图，形状为 `(H, W)`，是原始图像画布上的浮点数。值越大表示离
相机越近。这些值是相对的，不是以米为单位的度量值。`min`、`max` 和 `mean` 只在
有限值上计算，`normalized()` 把整张图重新缩放到 `[0, 1]`。

## NormalMap

稠密的表面法线场，原始图像画布上的 float32 `(H, W, 3)`，采用 OpenCV 相机坐标
系：`+x` 向右，`+y` 向下，`+z` 指向场景内部。法线朝向相机，所以一个正对相机的
平面是 `(0, 0, -1)`。每个像素都是单位向量。`assert_normalized(atol=1e-4)` 会
检查这条不变式。

## EdgeMap

稠密的边缘概率图，原始图像画布上的 float32 `(H, W)`，其中 `0` 是非边缘，`1` 是
边缘。保留连续的图，是为了把阈值留给调用方决定：`binary(threshold=0.5)` 应用一
个阈值，`array` 返回 numpy 视图。

## RestoredImage

复原后的 RGB 图像，`(H, W, 3)` uint8。做超分辨率时，画布是输入的
`Results.restore_scale` 倍。`array` 返回 numpy 视图，`save(path)` 把图像写入
磁盘。

## Matte

软的不透明度遮罩（matte），原始图像画布上取值在 `[0, 1]` 的 float32 `(H, W)`。
`1` 是完全前景，`0` 是完全背景。在 0.5 处取阈值，软遮罩就涵盖了硬的去背景掩码，
同时保留了二值掩码会丢掉的抗锯齿边缘。`array` 返回 numpy 视图。

在 matte 结果上，`Results.cutout(image=None)` 返回一个 RGBA 的 `(H, W, 4)` uint8
数组，它的第四个通道就是遮罩，`Results.save(path, image=None)` 则把这份抠像写成
背景透明的 PNG。两者在给了 `image` 时从中取 RGB，否则从 `Results.path` 重新
加载。

## OCRRegions

定位到的文字连同它们的转写。`data` 是 `(N, 4, 2)` 的浮点多边形，坐标为原始图像
像素，四个点按左上、右上、右下、左下排列；区域本身按阅读顺序给出，从上到下再从
左到右。`texts` 是 N 条转写组成的列表。`conf` 是每个区域的识别分数，`det_conf`
是检测分数，两者都是 `(N,)`。

检测四边形是真正的多边形，所以它们不会填充 `Results.boxes`。`xyxy` 给出轴对齐
的外接框。

## Embeddings

来自 `embed` 任务的 L2 归一化向量，形状始终是 `(N, D)`。整图结果只带一行，也没
有检测框；区域嵌入向量与 `boxes` 逐行对齐。因为每一行都已归一化，余弦相似度就
是一次点积。

| 成员 | 返回 |
|---|---|
| `dim` | `D` |
| `normalized` | 重新归一化后的各行 |
| `similarity(other)` | 与另一个 `Embeddings` 或张量之间的成对余弦相似度 |
| `verify(i, j, threshold=0.4)` | 第 `i` 行和第 `j` 行匹配时为 `True` |

## Identities

带名字的图库匹配结果，与 `embeddings` 逐行对齐。把一个 `Gallery` 传给 `embed`
预测时产出。`name` 是一个列表，低于匹配阈值的项为 `None`，而且绝不会去猜那个
最接近但低于阈值的名字。`score` 是匹配分数数组，`data` 把两者配成对。

## Meshes

参数化的人体网格，与 `boxes` 里的人体框逐行对齐。所有东西都在原始图像的相机
坐标系下。`transl` 以米为单位，`+z` 指向背离相机的方向；`vertices` 和
`joints3d` 也是米制，并且已经包含了 `transl`；`joints2d` 是原始图像画布上的
像素坐标，而不是网络看到的那块裁剪图上的坐标。没有任何字段带世界坐标系或重力
坐标系。

不同人体模型的参数布局不一样，所以关于形状的一切都没有写死。`body_model` 指出
用的是哪种参数化，各种数量则从张量里读回来：`num_vertices`、`num_joints`、
`num_betas` 和 `has_vertices`。`params` 返回参数 dict，`save_obj(path,
index=0)` 写出一个网格。字段有 `global_orient`、`body_pose`、`betas`、
`transl`、`vertices`、`faces`、`joints3d`、`joints2d`、`conf`、`focal_length`
和 `extras`。

对 `body_model="mhr"`，旋转用的是以弧度表示的欧拉角，而不是轴角；`body_pose`
是一个扁平的按关节排列的参数向量，而不是每个关节一个三元组；`betas` 是身份
blendshape 系数。骨架尺度、手部姿态和面部表情放在 `extras` 里。

## 转换与选择

每个载荷都带有 `to(*args, **kwargs)`、`cpu()`、`cuda()` 和 `numpy()`，在
`Results` 上调用其中之一，会一次性作用到所有被填充的槽位上。

<code-tabs name="convert" />

`result[idx]` 在所有逐行对齐的载荷上按行做选择。`len(result)` 是检测数量，没有
检测框时则是点的数量。`result.update(...)` 返回一份替换了指定槽位的副本；它接受
所有槽位，外加 `track_id` 和 `restore_scale`。

## summary 与 to_json

`summary(normalize=False, decimals=5, embeddings=False)` 返回一个普通 dict 的
列表，按被填充的槽位，每个检测、分段、点或区域对应一行。`to_json(**kwargs)` 把
自己的参数转给 `summary`，返回 JSON 字符串。

`plot()` 把稠密的法线或边缘结果渲染成它们各自约定的可视化形式；对其他类型的
结果它会抛异常。其他任务的标注图像来自 `predict(save=True)`。
