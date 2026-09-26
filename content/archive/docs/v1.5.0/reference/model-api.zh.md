---
title: 模型 API
seo_title: LibreYOLO 模型对象的方法与签名
description: >-
  已加载的 LibreYOLO
  模型上的每一个方法：predict、embed、track、val、train、export、save、quantize、info 以及 CUDA
  图控制，附真实的默认值。
lead: >-
  已加载的 LibreYOLO 模型是 BaseModel 的一个实例。本页列出该实例携带的方法，签名与默认值读取自
  libreyolo/models/base/model.py。
keywords:
  - libreyolo 模型方法
  - libreyolo predict 参数
  - libreyolo val 参数
  - libreyolo 导出参数
  - model.track
  - model.quantize
  - capture_graph
last_verified: 1.5.0
verification: >-
  签名与默认值读取自 v1.5.0 的 libreyolo/models/base/model.py 和
  libreyolo/models/base/inference.py。家族类可能收窄或扩展它们；train() 由每个家族各自定义，这里只记录共通的
  cfg= 包装。
snippets:
  usage:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        model.info()
        result = model(SAMPLE_IMAGE, conf=0.25, iou=0.45)

        print(result.boxes.xyxy)
        print(result.speed)
  stream:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        # stream=True 返回一个生成器，每帧或每张图像一个 Results
        for result in model([SAMPLE_IMAGE, SAMPLE_IMAGE], stream=True):
            print(len(result))
source_hash: da0776970ded8716
---

## 构造

工厂返回的是一个家族类实例。直接构造该类接受同样的参数，只是 `size` 是必填的：

```python
Family(model_path, size, nb_classes=80, device="auto", task=None, **kwargs)
```

`device="auto"` 在 CUDA 可用时选 CUDA，其次是 MPS，再次是 CPU。整数或数字字符串会被当作 CUDA 序号，所以 `device=0` 和 `device="0"` 都表示 `cuda:0`。`task` 会按家族的 `SUPPORTED_TASKS` 校验。传 `model_path=None` 只构建架构并让它留在训练模式；传一个 `dict` 则直接加载该 state dict。

## predict 与 \_\_call\_\_

`predict` 是 `__call__` 的别名。

```python
model(
    source=None,
    *,
    conf=0.25,
    iou=0.45,
    imgsz=None,
    device=None,
    classes=None,
    max_det=300,
    augment=False,
    save=False,
    batch=1,
    stream=False,
    stream_buffer=False,
    vid_stride=1,
    show=False,
    output_path=None,
    color_format="auto",
    tiling=False,
    overlap_ratio=0.2,
    output_file_format=None,
    cuda_graph=False,
    **kwargs,
)
```

| 参数 | 默认值 | 含义 |
|---|---|---|
| `source` | `None` | 图像、内存中图像的列表或元组、目录、视频文件，或者屏幕源，例如 `"screen"`、`"screen 1"`、`"screen 1 100 200 512 256"` |
| `conf` | `0.25` | 置信度阈值 |
| `iou` | `0.45` | NMS 的 IoU 阈值 |
| `imgsz` | `None` | 覆盖输入尺寸；`None` 使用模型的原生尺寸 |
| `device` | `None` | 覆盖本次调用的设备 |
| `classes` | `None` | 只保留这些类别 ID |
| `max_det` | `300` | 每张图像的最大检测数 |
| `augment` | `False` | 测试时增强 |
| `save` | `False` | 写出带标注的图像或视频 |
| `batch` | `1` | 目录与列表源每次前向传播处理的图像数 |
| `stream` | `False` | 返回生成器，而不是已经生成好的列表 |
| `stream_buffer` | `False` | 保留采集到的每一帧实时画面，而不是只保留最新的一帧 |
| `vid_stride` | `1` | 每隔 N 帧处理一次视频或屏幕画面 |
| `show` | `False` | 在窗口中显示带标注的帧 |
| `output_path` | `None` | `save=True` 时的输出路径 |
| `color_format` | `"auto"` | 内存数组的颜色格式提示 |
| `tiling` | `False` | 对大图做切片推理 |
| `overlap_ratio` | `0.2` | 切片重叠比例 |
| `output_file_format` | `None` | `"jpg"`、`"png"` 或 `"webp"` |
| `cuda_graph` | `False` | `True` 为每种输入形状在首次使用时捕获，`"auto"` 会等到某个形状重复出现 |

单张图像的源返回一个 `Results`。列表、元组或目录返回它们的列表，而 `stream=True` 在所有情况下都返回生成器。

实时流源是无界的，必须用 `stream=True`。`tiling` 和 `augment` 不能同时使用。对 `embed`、`point` 和 `edge` 任务，测试时增强会抛错。

<code-tabs name="usage" />

当 `batch > 1` 时，`SUPPORTS_BATCHED_PREDICT` 为真的家族每个分块跑一次堆叠后的前向；`batch=1` 则每张图像跑一次前向。

<code-tabs name="stream" />

## embed

```python
model.embed(source=None, **kwargs) -> torch.Tensor
```

对 `predict` 的便捷封装，把每一行嵌入向量堆叠成一个 `(N_total, D)` 张量。模型必须以 `task="embed"` 构造，否则会抛出 `NotImplementedError`。

## track

```python
model.track(
    source,
    *,
    track_conf=0.25,
    iou=0.45,
    imgsz=None,
    classes=None,
    max_det=300,
    save=False,
    show=False,
    vid_stride=1,
    output_path=None,
    tracker="bytetrack",
    tracker_config=None,
    augment=False,
    **tracker_kwargs,
) -> Generator[Results, None, None]
```

每帧产出一个设置了 `track_id` 的 `Results`。`tracker` 取 `"bytetrack"`、`"botsort"`、`"ocsort"` 或 `"deepocsort"`，给了 `tracker_config` 时它会被忽略，因为配置类型决定用哪个跟踪器。`track_conf` 对 ByteTrack 和 BoT-SORT 映射到 `track_high_thresh`，对 OC-SORT 和 Deep OC-SORT 映射到 `det_thresh`。`output_path` 默认是 `runs/track/<video_stem>.mp4`。

## val

```python
model.val(
    data=None,
    batch=16,
    imgsz=None,
    conf=0.001,
    iou=0.6,
    workers=4,
    allow_download_scripts=False,
    device=None,
    split="val",
    augment=False,
    save_json=False,
    verbose=True,
    *,
    plots=None,
    **kwargs,
) -> Dict
```

返回一个指标字典，键取决于任务；检测返回 `metrics/precision`、`metrics/recall`、`metrics/mAP50` 和 `metrics/mAP50-95`。`imgsz` 接受表示正方形的整数或一个 `(height, width)` 元组，默认使用模型的原生输入尺寸。`plots` 是 `save_plots` 的别名。`allow_download_scripts` 控制数据集 YAML 可能在 `download` 字段里携带的内嵌 Python。

`faster_coco_eval` 通过 `**kwargs` 接受，默认为 `True`，包没安装时回退到 pycocotools。实际运行的后端记录在 `model.last_eval_backend` 上。

对 `obb` 和 `pose` 任务，带增强的验证会抛错。

## train

`train` 由每个家族各自定义，所以参数各不相同。有两个行为是共通的，因为基类包装了每个家族的 `train`：

- `cfg=` 接受一个 YAML 路径，其中的键会合并进本次调用。显式传入的关键字参数优先于文件。
- 在覆盖分组 `g0` 或 `g1` 的家族上传 `pretrained=False`，会在训练前把模型从头重新初始化，并且不能与 `resume=True` 同时使用。

一个家族实际会遵循哪些数据增强开关是逐家族的问题；见[数据增强矩阵](/docs/reference/augmentation-matrix)。

## export

```python
model.export(format="onnx", **kwargs) -> str
```

返回写出产物的路径。`format` 通过导出器注册表（registry）解析，其中 `engine` 是 `tensorrt` 的别名，`litert` 是 `tflite` 的别名。所有导出器共有的参数：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `output_path` | `None` | 输出文件路径；省略时在 `weights/` 下生成 |
| `imgsz` | `None` | `(height, width)` 元组或单个整数；默认用原生尺寸 |
| `opset` | `None` | ONNX opset 版本 |
| `simplify` | `True` | 运行 ONNX 图简化 |
| `dynamic` | `True` | 启用动态轴 |
| `half` | `False` | FP16 精度 |
| `int8` | `False` | INT8 精度 |
| `batch` | `1` | 固化进产物的批大小 |
| `device` | `None` | 用于 trace 的设备 |
| `data` | `None` | 用于 INT8 校准的 data.yaml |
| `fraction` | `1.0` | 使用校准数据集的比例 |
| `allow_download_scripts` | `False` | 允许数据集 YAML 下载中的内嵌 Python |
| `verbose` | `False` | 导出器的详细日志 |

被禁止的组合会在 trace 之前的预检里抛出 `NotImplementedError`。覆盖情况及其规则见[导出矩阵](/docs/reference/export-matrix)页面。存在活跃的 LoRA 适配器时，它们会被折叠进稠密权重，而这个合并只在所有请求拒绝之后才发生。

## save

```python
model.save(path) -> str
```

写出一个 schema v1.0 的 LibreYOLO 检查点（checkpoint）：state dict 加上[检查点 schema](/docs/reference/checkpoint-schema) 里描述的元数据。量化后的模型还会额外带上它的 `quant` 清单，所以 `LibreYOLO(path)` 能恢复量化结构与缩放系数。

## quantize、quant_info 与 dequantize

```python
model.quantize(
    recipe,
    calib="coco128.yaml",
    samples=128,
    batch=8,
    algorithm="auto",
    keep_high_precision=None,
    allow_download_scripts=False,
    verbose=True,
)
```

原地量化并返回模型。`recipe` 可以是类型转换 `fp16` 和 `bf16`，作用于 Conv 与 Linear 的配方 `int8` 和 `fp8`，或者只作用于 Linear 的配方 `w4a16`、`w4a8`、`nvfp4`、`mxfp4` 和 `int2`——后者由 RF-DETR 这类 transformer 家族支持。`int2` 需要 QAT。`calib` 接受一个 data.yaml 路径或内置数据集名称，只做前向读取图像；标注从不读取。传 `calib=None` 可跳过校准。`algorithm` 取 `"minmax"`、`"percentile"` 或 `"auto"`。

`model.quant_info()` 返回量化状态摘要，浮点模型则返回 `None`。`model.dequantize()` 原地恢复浮点模块，同时保留量化训练得到的主权重，这是从 QAT 通往 `export(format="onnx", int8=True, data=...)` 的桥梁。

## info 与层

```python
model.info(detailed=False, verbose=True) -> Dict[str, Any]
model.get_available_layer_names() -> List[str]
model.get_distill_config() -> Dict
```

`info` 返回一个对 JSON 友好的字典，并在 `verbose` 为真时打印一份人类可读的摘要。`get_available_layer_names` 列出蒸馏或特征提取配置可以指定的层。

## CUDA 图

在 `SUPPORTS_CUDA_GRAPH` 类属性为真的家族上可用。回放与 eager 执行逐位一致。

```python
model.capture_graph(imgsz=None, batch=1, dtype=None) -> None
model.cuda_graph_scope(mode=True)          # 上下文管理器
model.graph_info() -> Dict[str, Any]
model.release_graphs() -> None
```

捕获下来的图只对捕获时的那个确切形状有效，所以 `batch` 和 `imgsz` 必须与之后的 `predict` 调用一致。`capture_graph` 把捕获开销从首次请求里挪走。`mode` 接受 `True` 或 `"on"` 表示首次使用时捕获，`"auto"` 表示等到某个形状重复出现，`False` 表示不做任何事。家族没有开启支持时 `capture_graph` 抛出 `NotImplementedError`，捕获失败时抛出 `CudaGraphUnavailable`。

## 设备与 dtype

`Results` 对象带有 `.to()`、`.cpu()`、`.cuda()` 和 `.numpy()`；见 [Results 类型](/docs/reference/results-types)。模型本身通过给 `predict` 传 `device=`，或在构造时移动。
