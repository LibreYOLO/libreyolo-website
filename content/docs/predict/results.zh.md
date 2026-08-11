---
title: 使用 Results 对象
seo_title: LibreYOLO 的 Results 对象
description: >-
  每张图像一个 Results 对象，每种载荷类型对应一个槽位：boxes、masks、keypoints、probs、depth、panoptic、OCR
  等。绘制、保存与 JSON。
lead: >-
  每次预测都会为每张图像返回一个 Results
  对象。它为每一种载荷准备了一个具名槽位（slot），除了模型实际产出的那些之外全都是空的；导出的产物上也有同样这些槽位。
keywords:
  - yolo results 对象 python
  - results.boxes xyxy
  - yolo 结果转 json
  - yolo 保存标注图像
  - 实例分割 mask python
  - yolo 关键点结果
  - yolo 深度图输出
  - libreyolo results summary
  - onnx 推理结果一致
last_verified: 1.5.0
verification: >-
  载荷类、槽位、搬移语义、summary()、to_json()、plot()、save() 和 cutout() 读自
  libreyolo/utils/results.py。标注和写盘行为读自 libreyolo/models/base/inference.py 里的
  InferenceRunner._save_annotated_image 和 libreyolo/utils/general.py 里的
  resolve_save_path。后缀分派读自 libreyolo/models/__init__.py 里的 LibreYOLO()。
snippets:
  basic:
    - label: Boxes
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        print(result.orig_shape)   # 源图像的 (height, width)
        print(result.path)         # 源路径，内存输入时为 None

        for xyxy, conf, cls in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
        ):
            print(result.names[int(cls)], round(float(conf), 3), xyxy)
    - label: 归一化坐标
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy[:1])    # 像素，x1 y1 x2 y2
        print(result.boxes.xywh[:1])    # 像素，中心 x、中心 y、w、h
        print(result.boxes.xyxyn[:1])   # 同一个框除以宽和高
        print(result.boxes.xywhn[:1])
    - label: NumPy 与设备
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        # 下面每一个都返回新的 Results，原对象不变
        as_numpy = result.numpy()
        on_cpu = result.cpu()

        print(type(as_numpy.boxes.xyxy).__name__)
        print(type(on_cpu.boxes.xyxy).__name__)
  json:
    - label: summary 与 to_json
      language: python
      code: |
        import json

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        rows = result.summary()
        print(json.dumps(rows[:2], indent=2))

        # 同样的内容，只是以字符串返回，关键字参数也一样
        print(result.to_json(normalize=True, decimals=3)[:200])
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt --json \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  saving:
    - label: 标注后的图像
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        # save=True 会画出载荷，并写到 runs/detect/predict* 下
        result = model(SAMPLE_IMAGE, save=True)
        print(result.saved_path)
  exported:
    - label: 安装导出用的 extra
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: 从导出产物拿到同样的 Results
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        path = model.export(format="onnx")   # 返回写入的路径

        # LibreYOLO() 按文件后缀分派
        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)

        print(type(result).__name__, len(result.boxes))
source_hash: 548dbc9c7f5552ec
---

## 一个对象，每种载荷一个槽位

对一张图像做一次预测返回一个 `Results`。它带着十八个载荷槽位，模型只填充自己
这个任务会产出的那些。其余每个槽位都是 `None`，所以在检测器上读 `result.masks`
得到的是 `None`，而不是报错。

| 槽位 | 类 | 形状 | 由什么产出 |
|---|---|---|---|
| `boxes` | `Boxes` | `(N, 4)`，外加分数和类别 | 检测，以及任何先做定位的任务 |
| `masks` | `Masks` | `(N, H, W)` | 实例分割 |
| `keypoints` | `Keypoints` | `(N, K, 2)` 或 `(N, K, 3)` | 姿态 |
| `probs` | `Probs` | `(C,)` | 分类 |
| `obb` | `OBB` | `(N, 7)` 或 `(N, 8)` | 旋转框 |
| `gaze` | `Gaze` | `(N, 2)`，弧度制的俯仰角和偏航角 | 视线估计 |
| `points` | `Points` | `(N, 4)`，依次是 x、y、类别、置信度 | 点定位 |
| `semantic_mask` | `SemanticMask` | `(H, W)` 类别 id | 语义分割 |
| `panoptic` | `PanopticSegmentation` | `(H, W)` 分段 id，外加 `segments_info` | 全景分割 |
| `depth_map` | `DepthMap` | `(H, W)` 浮点数 | 深度估计 |
| `normal_map` | `NormalMap` | `(H, W, 3)` 单位向量 | 表面法线 |
| `edges` | `EdgeMap` | `[0, 1]` 区间的 `(H, W)` 浮点数 | 边缘检测 |
| `restored` | `RestoredImage` | `(H, W, 3)` uint8 RGB | 图像恢复与超分辨率 |
| `matte` | `Matte` | `[0, 1]` 区间的 `(H, W)` 浮点数 | alpha 抠图与背景移除 |
| `ocr` | `OCRRegions` | `(N, 4, 2)` 多边形，外加转写文本 | 文本检测与识别 |
| `embeddings` | `Embeddings` | `(N, D)`，行已做 L2 归一化 | `embed` 任务 |
| `identities` | `Identities` | N 个名字和分数 | 带图库的 `embed` 任务 |
| `meshes` | `Meshes` | 人体参数，以及可选的顶点 | 人体网格恢复 |

和它们并列的，是每个结果都有的那些字段：`orig_shape` 是 `(height, width)`，
`path` 是源路径（内存输入时为 `None`），`names` 把类别 id 映射到类别名，
`frame_idx` 用于视频和实时帧，`track_id` 在跟踪时出现，还有 `restore_scale`，
即恢复类结果的整数放大倍数。

`result.normals` 是 `result.normal_map` 的别名。

`result.speed` 每个结果上都有，但只有[集成](/docs/predict/ensembling)会填充它，
这时它的键是 `member_0`、`member_1` 和 `fusion`，单位是毫秒。单个模型时它一直
是一个空字典。

## Boxes

<code-tabs name="basic" />

`Boxes` 把坐标和分数分开存成几个数组，而不是打包成一个张量。

| 属性 | 内容 |
|---|---|
| `xyxy` | `(N, 4)` 绝对像素，x1 y1 x2 y2 |
| `xywh` | `(N, 4)` 绝对像素，中心 x、中心 y、宽、高 |
| `xyxyn`、`xywhn` | 同样的值除以图像宽和高 |
| `conf` | `(N,)` 置信度 |
| `cls` | `(N,)` 类别 id，浮点数形式 |
| `id` | `(N,)` 跟踪 id，或者 `None` |
| `is_track` | `id` 是否已设置 |
| `data` | 全部拼接在一起：检测框、可选的 id、conf、cls |

`cls` 是浮点数组，所以要这样用它：`result.names[int(cls)]`。

`xyxyn` 和 `xywhn` 需要 `orig_shape`，这个字段由 `Results` 替你填好。

## 稠密载荷

覆盖整张图像的载荷，行为和逐实例的载荷不同，切片的时候这一点很重要。

`SemanticMask` 存的是原始画布上的 `(H, W)` 类别 id，其中 `255` 被保留为忽略值，
永远不算作一个类别。`classes` 列出出现过的 id，并把它排除在外；`class_mask(id)`
返回一个布尔的 `(H, W)`。

`PanopticSegmentation` 存的是 `(H, W)` 分段 id，`0` 是 void id，另外还有一个
`segments_info` 字典列表，每项至少带 `id` 和 `category_id`。`segment_ids` 列出
出现过的 id，`segment_mask(id)` 选出其中一个。

`DepthMap` 存的是 `(H, W)` 相对逆深度：值越大表示越近，而且这些值不是以米为
单位的度量值。它提供在有限值上算出的 `min`、`max`、`mean`，以及把范围重新缩放
到 `[0, 1]` 的 `normalized()`。

`NormalMap` 存的是 OpenCV 相机坐标系下的 `(H, W, 3)` 单位向量，`+x` 向右，
`+y` 向下，`+z` 指向场景内部，所以正对相机的表面是 `(0, 0, -1)`。
`assert_normalized()` 检查每个像素都是有限值、且长度为 1。

`EdgeMap` 存的是 `[0, 1]` 区间的 `(H, W)` float32。这里保留的是连续图，而不是
阈值化之后的结果，所以选择截断点的地方是 `binary(threshold=0.5)`。

`Matte` 存的是 `[0, 1]` 区间的 `(H, W)` float32，`1` 表示完全是前景。`array`
返回裁剪后的 float32。

`RestoredImage` 存的是 `(H, W, 3)` uint8 RGB，`array` 给出原始 ndarray，
`save(path)` 把它写出去。

`Probs` 为整张图像存一个概率向量。`top1` 和 `top5` 是类别索引，`top1conf` 和
`top5conf` 是对应的分数。

`Embeddings` 存的是已经做过 L2 归一化的 `(N, D)` 行，所以余弦相似度就是一次
点积。`similarity(other)` 对图库返回 `(N, M)`，对单个向量返回 `(N,)`；
`verify(i, j, threshold=0.4)` 比较两行。

`OCRRegions` 按阅读顺序存 `(N, 4, 2)` 多边形，四个角点依次是左上、右上、右下、
左下。转写文本在 `texts` 里，识别分数在 `conf` 里，检测分数在 `det_conf` 里。
因为这些是真正的旋转多边形，它们不会填充 `boxes`；需要矩形时，`ocr.xyxy` 给出
轴对齐的外接框。

## 切片与搬移

`result[i]` 返回一个只装着一个实例的新 `Results`。逐实例的载荷会被切片，整图的
载荷则原样带过去，所以对一个分类结果切片不会把它的概率向量截断成单个类别，对一
个深度结果切片也不会破坏 `(H, W)` 的布局。

`len(result)` 数的是实例：检测框、点、嵌入向量、OCR 区域或网格。任何稠密的整图
载荷都算作 `1`。什么都没装的结果是 `0`。

`to()`、`cpu()`、`cuda()` 和 `numpy()` 各自返回一个新的 `Results`，其中每个已
填充的槽位都转换过。它们不会修改原对象。

`update()` 是唯一一个原地修改的方法，它替换指定名字的槽位，并返回同一个对象。

## JSON

<code-tabs name="json" />

`summary()` 返回一个由普通字典组成的列表，`to_json()` 就是把这个列表交给
`json.dumps`。两者接受同样的三个参数：`normalize=False` 把坐标切换成 `[0, 1]`，
`decimals=5` 设置四舍五入位数，`embeddings=False` 控制要不要包含嵌入向量。

每一行的形状跟着载荷走。检测行带 `name`、`class`、`confidence` 和一个 `box`
字典，有掩码时会加上 `segments`，旋转框会加上 `obb` 和 `corners`，视线会加上以
弧度和角度两种单位给出的 `gaze` 角度，跟踪时加上 `track_id`，有网格时加上
`mesh` 参数。

没有检测框时，由某一个载荷来决定这些行：OCR 每个区域输出一行，带上它的
`text`，点每个点一行，全景每个分段一行、带 `pixel_count` 和 `pixel_fraction`，
语义每个出现过的类别一行，分类输出前五个类别。深度、法线、边缘、恢复和抠图各自
只输出一行汇总，描述的是这张图本身，而不是它的像素。

有两种载荷是有意精简过的。嵌入向量只报告 `embedding_dim`，因为一行 512 个浮点数
意味着每张脸大约 2 KB；传 `embeddings=True` 才会把数值也包含进来。网格顶点则
完全不包含，因为那是每个人几万个坐标。要拿几何数据，读 `result.meshes.vertices`
或者调用 `result.meshes.save_obj(path)`。

## 绘制与保存

<code-tabs name="saving" />

`predict(save=True)` 才是那条既标注又写盘的路径。它根据哪个槽位被填充来挑选绘制
例程，所以语义结果写出来是一张上色（colorize）后的掩码，深度结果写成深度可视化
图，全景结果带上它的分段，matte 写成透明背景的 RGBA PNG，检测器写成检测框、掩码
画在框下面。写出的路径会作为 `result.saved_path` 挂在结果上。

`Results.plot()` 比它的名字听上去要窄。它只为法线图和边缘图定义，其他任何情况都
抛 `NotImplementedError`。别的任务请用 `save=True`。

`Results.save(path)` 同样窄：它把 matte 结果写成透明背景的 RGBA PNG 抠图，其他
情况一律抛 `NotImplementedError`。`Results.cutout()` 返回同一个 RGBA 数组，但
不写盘。两者都需要源图像，从 `result.path` 取，或者用 `image=` 传进来。

有两种载荷自带写盘方法：恢复后的图像用 `result.restored.save(path)`，网格用
`result.meshes.save_obj(path, index=0)`。

文件最终落在哪里，以及 `output_path` 和 `output_file_format` 的行为，见
[预测数据源](/docs/predict/sources)。

## 导出产物返回同样的对象

<code-tabs name="exported" />

`LibreYOLO()` 按文件后缀分派，所以一个导出产物和一个 `.pt` 检查点（checkpoint）
走同一个调用加载，返回同样的 `Results`。`.onnx`、`.engine`、`.pte` 和 `.mnn`
文件按后缀识别，OpenVINO、Paddle 和 ncnn 目录以及 Triton 模型 URL 也一样。把一个
模型换成它的导出版本时，读 `result.boxes.xyxy` 的代码不用改。完整的格式清单见
[导出](/docs/export)。

反过来，直接去用运行时自己的 API，就意味着预处理、后处理和类别名都得你自己管。
