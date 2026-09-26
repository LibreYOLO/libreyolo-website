---
title: 检查点结构
seo_title: LibreYOLO 检查点元数据结构 v1.0
description: 每个 LibreYOLO .pt 检查点都携带的元数据：必需键、各任务的附加字段、导出运行时键、量化清单和训练字段。
lead: >-
  LibreYOLO 的 .pt 文件是用 torch.save 保存的扁平字典。model
  键保存状态字典；其他顶层键是元数据，不用解析文件名、也不用嗅探状态字典就能识别这个检查点。
keywords:
  - libreyolo 检查点结构
  - schema_version 1.0
  - model_family
  - libreyolo 检查点元数据
  - 量化 quant 清单
  - wrap_libreyolo_checkpoint
last_verified: 1.5.0
verification: >-
  对应 libreyolo 仓库 v1.5.0 的 docs/checkpoint_schema.md，并与
  libreyolo/utils/serialization.py 和 BaseModel.save 交叉核对。
snippets:
  usage:
    - label: 读出检查点里的元数据
      language: python
      code: >
        from libreyolo import LibreYOLO

        from libreyolo.utils.serialization import unwrap_libreyolo_checkpoint

        import torch


        # 下载一个检查点，再重新保存一份，这样本地就有了路径

        LibreYOLO("LibreYOLO9t.pt").save("roundtrip.pt")


        loaded = torch.load("roundtrip.pt", map_location="cpu",
        weights_only=False)

        state_dict, metadata = unwrap_libreyolo_checkpoint(loaded)


        print(metadata["schema_version"], metadata["model_family"])

        print(metadata["size"], metadata["task"], metadata["nc"],
        metadata["imgsz"])

        print(len(state_dict), "tensors")
source_hash: ce760f1bed97bfd0
---

## 结构 v1.0

每个官方 LibreYOLO `.pt` 检查点（checkpoint）都包含：

```python
{
    "model": state_dict,
    "schema_version": "1.0",
    "libreyolo_version": "0.x.y",
    "model_family": "yolo9",
    "size": "t",
    "task": "detect",
    "nc": 80,
    "names": {0: "cat", 1: "dog"},
    "imgsz": 640,
}
```

| 键 | 类型 | 含义 |
|---|---|---|
| `model` | 状态字典 | 模型权重 |
| `schema_version` | str | 元数据契约的版本；v1.0 使用字符串 `"1.0"` |
| `libreyolo_version` | str | 产出该检查点的版本 |
| `model_family` | str | 已注册的家族，例如 `yolo9`、`rfdetr`、`dfine`、`ec` |
| `size` | str | 家族内的变体，例如 `t`、`s`、`r18`、`atto` |
| `task` | str | 规范任务名 |
| `nc` | int | 正的类别数量 |
| `names` | dict | `dict[int, str]`，键位于 `0..nc-1` |
| `imgsz` | int | 正的方形输入分辨率，或矩形契约下的旧版标量 |

`task` 取 `detect`、`segment`、`semantic`、`panoptic`、`pose`、`classify`、
`gaze`、`obb`、`point`、`depth`、`edge`、`normal`、`restore`、`matte`、`ocr`、
`embed` 或 `mesh` 之一。

官方检查点会写出每一个 `names` 键。对于旧版的稀疏映射，读取方可以用
`class_i` 标签补齐缺失的键，但超出范围的键是无效的。

矩形检查点为旧版读取方保留标量 `imgsz`，取值为 `max(imgsz_h, imgsz_w)`，并额外
写出带真实尺寸的 `imgsz_h` 和 `imgsz_w`。理解矩形字段的读取方必须优先使用它们，
而不是那个标量。有固定矩形契约的家族，例如 HRNet 姿态，会拒绝不兼容的运行时尺寸。

这个结构是刻意保持扁平的，`model` 也是刻意做成状态字典的。

<code-tabs name="usage" />

## 姿态附加字段

姿态通常是单类别的，`nc: 1`，名称为 `person`，但 YOLO-NAS 的姿态 head 也支持多
类别姿态，共用一套关键点骨架，这时 `nc` 和 `names` 像检测那样描述类别。运行时的
姿态导出会给出形状为 `[batch, anchors, nc]` 的 `scores`。

| 键 | 含义 |
|---|---|
| `num_keypoints` | 姿态 head 使用的关键点数量，必须为正 |
| `keypoint_dim` | `2` 表示 `x,y` 标注，`3` 表示 `x,y,visibility` 标注；模型输出始终暴露 `x,y,visibility` |
| `oks_sigmas` | 可选的每关键点 OKS sigma；缺失时使用 `num_keypoints` 对应的任务默认值 |
| `num_keypoints_per_class` | 可选的每类别关键点数量，用于 GroupPose 风格的 head——它们的关键点张量按类别做了填充；没有关键点的类别为 `0` |

## 网格附加字段

网格检查点使用 `task: "mesh"`、`nc: 1` 和 `names: {0: "person"}`。不同人体模型的
参数布局并不相同，因此这些维度是记录下来的，而不是假定的。

| 键 | 含义 |
|---|---|
| `body_model` | 参数化方式，例如 `mhr`；必需，下面每个字段都要靠它来解释 |
| `num_betas` | 身份与形状系数的数量；MHR 为 45 |
| `num_body_pose` | 人体姿态参数块的宽度；MHR 为 130。它是一个扁平向量，而不是每个关节一个三元组，因为骨骼关节的自由度各不相同 |
| `num_vertices` | 解码器输出的顶点数量；MHR 为 18439 |
| `num_joints` | 解码器输出的关节数量；MHR 为 127 |
| `rotation_format` | 旋转的编码方式，例如 MHR 的 `euler_zyx` 或 `axis_angle`。绝不从张量形状推断，因为三维向量是有歧义的 |

## 密集任务的占位字段

有几种任务预测的是密集图而不是类别，因此这些类别相关的槽位只是为了兼容结构而
存在。

| 任务 | `nc` | `names` |
|---|---|---|
| `depth` | 1 | `{0: "depth"}` |
| `edge` | 1 | `{0: "edge"}` |
| `restore` | 1 | `{0: "image"}` |
| `ocr` | 1 | `{0: "text"}` |

边缘预测是取值在 `[0, 1]` 内的密集 float32 概率图。

复原检查点可以额外加上 `degradation`，即简短的退化类型标签，例如 `deblur`、
`denoise` 或 `super-resolution`；`dataset`，即来源标签，例如 `GoPro` 或 `SIDD`；
以及 `scale`，即正整数的输出对输入放大倍数，例如 x4 超分辨率模型为 `4`。缺失或
为 `1` 表示复原后的图像保持输入分辨率。运行时也会从家族和尺寸推导出这个倍数，
所以 `scale` 属于来源元数据，而不是加载时的必需项。

## OCR 附加字段

`ppocr` 家族为每个档位提供一个复合检查点，它的 `model` 状态字典在 `det.*` 和
`rec.*` 两个键命名空间下保存了两个子模型。

| 键 | 含义 |
|---|---|
| `charset` | 按输出索引顺序排列的完整 CTC 字母表：索引 0 是 CTC 空白符，然后是识别字典，最后是空格字符。加载方必须从检查点里读取它，绝不能从旁路文件读取 |
| `pipeline` | 转换时固化下来的流水线默认值：`det_limit_side_len`、`det_db_thresh`、`det_db_box_thresh`、`det_db_unclip_ratio`、`rec_image_shape`。运行时参数可以按调用覆盖它们 |
| `components` | 预留给可选的流水线阶段，例如文档方向、去扭曲和文本行旋转。v1 中为空 |

## 导出运行时元数据

导出产物使用同样的矩形双写约定：`imgsz_h` 和 `imgsz_w` 写在旧版标量 `imgsz` 旁边，
不理解矩形字段的读取方不得默默把这个标量当成方形契约。

矩形的运行时支持按家族和按格式区分。YOLO9 家族、HRNet、NAFNet 和 Real-ESRGAN 的
导出可以在受支持的格式里使用非方形的 `imgsz_h` 和 `imgsz_w`；没有明确矩形支持的
家族或格式会拒绝这些元数据，而不是把这些产物按方形做预处理。HRNet 导出是固定
尺寸、batch 为 1、FP32 的人体裁剪 head，其中 W32 接受 256x192，W48 接受 384x288，
人体检测器并不内嵌在图里。

内嵌 NMS 的导出可以加上这些扁平键：

| 键 | 含义 |
|---|---|
| `nms` | 字符串布尔值；`"true"` 表示图中包含内嵌的后处理输出 |
| `nms_conf` | 固化进内嵌输出的置信度阈值 |
| `nms_iou` | 固化进内嵌输出的 IoU 阈值 |
| `max_det` | 内嵌输出最多给出的 NMS 后检测行数 |
| `nms_raw_output` | 字符串布尔值；`"true"` 表示图中还额外暴露一路原始检测器输出 |

对于 `nms=true` 的 ONNX YOLO9 检测导出，输出 `0`（名为 `output`）是按导出时阈值
得到的、可独立使用的 NMS 后张量。当 `nms_raw_output=true` 时，输出 `1`（名为
`raw`）保留给 LibreYOLO 后端，让它们能应用原生的原始画布裁剪以及运行时的
`predict(conf=..., iou=..., max_det=...)` 语义。第三方使用方应当用第一路输出。

姿态导出可以加上 `num_keypoints`；`keypoint_dim`，其中 GroupPose 风格的原始导出
在张量包含精度或类别 logit 字段时可能用更大的值，例如 `8`；
`num_keypoints_per_class`，以 JSON 编码的列表形式给出，其中关键点数为零的类别
槽位必须保留，因为它们定义了结构；以及 `pose_input`，其中 `"person_crop"` 表示
图消费的是一份已经裁好的人体裁剪，并且不含检测器。HRNet 的运行时导出要求该取值。

分类导出可以加上 `crop_pct`，即中心裁剪比例的浮点值，它裁剪前的缩放目标是
`round(imgsz / crop_pct)`，缺失时默认为 `0.875`；以及 `interpolation`，取
`"bilinear"` 或 `"bicubic"`，默认为 `"bilinear"`。

ExecuTorch 导出把扁平元数据写进必需的 `<program>.pte.json` 附属文件。v1 契约是
CPU、FP32、batch 为 1 和固定的输入画布，此外还要求 `executorch_version`、等于
`"xnnpack"` 的 `executorch_delegate`，以及为正的 `executorch_delegate_partitions`。
如果附属文件声称用的是别的 delegate、动态形状或非 FP32 精度，加载器会拒绝它。

MNN 导出把扁平元数据写进必需的 `<model>.mnn.json` 附属文件。v1 契约是 CPU、FP32、
仅检测和固定的 NCHW 输入形状，此外还要求 `mnn_version`、等于 `"cpu"` 的
`mnn_backend`、有序且非空的 `mnn_input_names` 和 `mnn_output_names`、按
`[batch, channels, height, width]` 顺序给出四个正整数的 `mnn_input_shape`，以及
等于 `mnn_input_shape[0]` 的 `mnn_batch`。动态形状、非 FP32、非检测、不受支持的
家族或前后不一致的形状元数据，都会被加载器拒绝。

`.pte` 和 `.mnn` 是特定后端的产物，不是 PyTorch 检查点。

## 量化检查点

量化模型会加上一个可选的扁平键 `quant`，里面是一个清单字典，包含 `schema`、
`recipe`、`keep_high_precision`、`execution`、校准来源信息、`module_count` 和
`state`。FP8 的清单还可以带上 `fp8_tensorwise_weights`，即权重缩放按整个张量而不
是按输出通道来做的那些 `QuantLinear` 模块名的完整列表。看到 `quant` 的加载器会在
`load_state_dict` 之前重建量化模块结构和缩放策略。

`state` 区分两种产物形态。

`"prepared"` 是默认值，保存 FP32 主权重加上 `_q_*` 缩放缓冲区，并且可以训练。
不支持量化的读取方可以忽略 `quant` 键，把主权重当成浮点模型加载。

`"finalized"` 是由 `export(format="pt")` 写出的部署形态。主权重被剥离，每个量化
模块改为携带打包后的权重：

| 方案 | 打包张量 | 反量化 |
|---|---|---|
| int8 | `weight_packed` 为原权重形状的 int8，`_q_w_scale` 为每通道一个的 FP32 | `weight_packed * scale` |
| fp8 | `weight_packed` 为原形状的 float8_e4m3fn，`_q_w_scale` 为 FP32，每个输出通道一项 | `weight_packed * scale` |
| w4a16、w4a8 | `weight_packed` 为 uint8，每字节两个 4 位码，低半字节在前，码为 `q + 8`；`_q_w_gscale` 为 FP32 `[out, ngroups]`，沿 in_features 每 128 个一组 | 按组缩放 |
| int2 | 每字节四个 2 位码，码为 `q + 2`，每 64 个一组 | 按组缩放 |
| nvfp4 | `weight_packed` 为 uint8 `[out, ceil(in/16)*8]`，码为 `sign<<3 \| E2M1 level`；`weight_block_scale` 为 float8_e4m3fn `[out, ceil(in/16)]`；`_q_w_amax` 为每张量一个的 FP32 | `block_scale * amax / (448 * 6)` |
| mxfp4 | 与 nvfp4 相同，但块是 32 个元素，另加 `weight_block_exp` int8 `[out, ceil(in/32)]` | `2 ** exponent` |

int8 会保留激活范围缓冲区 `_q_act_lo`、`_q_act_hi` 和 `_q_calibrated`。清单用
`remainder` 记录未量化张量的精度，取 `"fp16"` 或 `"fp32"`。解包逐位一致地复现
模拟结果，因此在执行 finalize 的设备上，finalized 推理与 prepared 推理完全一致。
这套布局是面向外部导出器和运行时的稳定契约。

## 训练检查点

训练器检查点使用同一套必需的元数据核心，并且可以加上扁平的训练与续训字段：

```python
{
    "model": state_dict,
    "epoch": 42,
    "optimizer": optimizer_state_dict,
    "config": {},
    "loss": 1.23,
    "best_metric_key": "metrics/mAP50-95",
    "best_metric_value": 0.51,
    "best_epoch": 39,
    "is_ema_weights": True,
    "train_model": raw_state_dict,
    "ema": ema_state_dict,
    "ema_updates": 12345,
}
```

`is_ema_weights` 声明顶层的 `model` 是否经过 EMA 平滑。启用 EMA 时，`train_model`、
`ema` 和 `ema_updates` 保存续训状态。发布出去的推理权重应当精简，不应包含优化器、
轮次、配置、损失或 EMA 续训状态，除非它们本来就是有意作为训练检查点分发的。

为了发布兼容性，读取方接受旧版的最佳指标别名 `best_mAP50_95`、`best_mAP50`、
`best_metric` 和 `best_metric_name`。

## 外部快照

这个结构管辖的是由 LibreYOLO 生成的 `.pt` 文件。它不会重命名或包装那些由独立模型
档位使用的多文件上游快照。

LibreMODUS 的 `14b-a7b` 尺寸是一个明确的例外：这个别名会经由 `LibreVLM(...)` 解析
到一个存放固定版本上游文件的目录，LibreYOLO 既不会给它添加 v1.0 元数据，也不会把
它重新发布为 `.pt`。

## 旧版与外来权重

新的写入方会严格验证，并且必须写出 v1.0 元数据。当元数据缺失或不完整时，看起来
像旧版 LibreYOLO 的检查点会走兼容路径加载，并给出警告和转换说明，外来的上游检查点
则转入自动转换。参见[上游检查点](/docs/reference/upstream-checkpoints)。

## 辅助函数

结构相关的辅助函数位于 `libreyolo.utils.serialization`：

```python
wrap_libreyolo_checkpoint(
    state_dict,
    *,
    model_family,
    size,
    task,
    nc,
    names=None,
    imgsz=None,
    libreyolo_version=None,
    schema_version="1.0",
    **extra_metadata,
) -> dict

validate_checkpoint_metadata(checkpoint, *, strict=False) -> list[str]

unwrap_libreyolo_checkpoint(loaded, *, strict=False) -> tuple[dict, dict]
```

`validate_checkpoint_metadata` 不修改输入，返回错误列表；设为 `strict=True` 时它
改为抛出 `CheckpointMetadataError`。`model.save(path)` 是写出符合本结构的检查点的
受支持方式。
