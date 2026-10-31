---
title: 升级到 1.7.0
seo_title: LibreYOLO 从 1.6.0 升级到 1.7.0
description: LibreYOLO 1.7.0 的迁移步骤：YOLO9 检查点、模块名称和训练配方，CLI 运行目录，混淆矩阵，以及新增的报错。
lead: 1.7.0 让旧的 YOLO9 检查点继续以相同的浮点检测框加载。它改变了 YOLO9 家族的训练、代码在网络内部看到的名称，以及少数几处验证和预测的约定。
keywords:
  - libreyolo 升级
  - libreyolo 1.6.0 迁移
  - libreyolo 破坏性变更
  - yolo9 模块名称
  - clip_max_norm
  - yolo 混淆矩阵
last_verified: 1.7.0
source_hash: 895bcd7425f0052f
---

## 从 1.6.0 升级到 1.7.0

- 由 1.7.0 保存的 YOLO9、YOLO9-E2E 和 YOLO9-P2 检查点使用新的张量名称，在 1.6.x 中无法加载：1.6.x 加载 YOLO9 和 YOLO9-P2 的检查点时会给出缺失键警告并且不预测任何检测框，加载 YOLO9-E2E 的检查点则会抛出 `model_family` 不匹配错误。请升级每一个读取它们的环境。早期版本写出的浮点、量化和训练检查点都能在 1.7.0 中加载，ONNX 导出不受影响。

- 直接访问 YOLO9 家族模型的 `model.model` 的代码，现在看到的是 MultimediaTechLab/YOLO 的模块名称，没有别名。`DDetect` 变成 `YOLO9Head`（`anchor_convs` / `class_convs`），`RepConvN` 变成 `RepConv`，`RepNBottleneck` 变成 `Bottleneck`，`DFL` 变成 `Anchor2Vec`（`anchor2vec`），`nc` 变成 `num_classes`，块的子层是 `conv1`/`conv2`/`bottleneck`。在 YOLO9-E2E 中，`YOLO9E2EDetect` 变成 `YOLO9E2EHead`，`one2one_cv2` / `one2one_cv3` 变成 `one_to_one_anchor_convs` / `one_to_one_class_convs`。`LibreYOLO9Model.fuse()`、`RepConvN.fuse_convs()`、`Conv.forward_without_bn()`、`Concat` 和 `yolo9.utils.decode_boxes` 已移除。每一项重命名都列在[更新日志](/docs/changelog)里。

- head、骨干和 neck 里的 ONNX 节点名称跟随新的模块名称，例如 `/head/anchor_convs.0/...`。请更新以节点名称为键的工具，例如 Hailo 的 end-node 配置。模型的输入、输出和输出名称不变。

- 用不带 PGI 张量的权重微调 YOLO9，其中包括已发布的 `LibreYOLO9{t,s,m,c}.pt`，只训练主 head，和 1.5.0 一样；1.6.0 会以 `aux_weight=0.25` 挂上一个随机初始化的 PGI 分支。带有 PGI 张量的权重和从头开始的运行仍然使用 PGI 训练。传入 `aux_weight=0.25` 可以强制挂上新分支，传入 `aux_weight=0` 则强制关闭。

- 挂上 PGI 时，YOLO9-m 和 YOLO9-c 会构建 MultimediaTechLab 的辅助分支：在 80 个类别下仅训练时使用的参数量为 12.8M（m）和 25.7M（c），之前分别是 7.5M 和 10.2M，所以这两个尺寸的 PGI 训练每一步需要更多显存和时间。1.6.0 的 m/c 训练检查点会沿用它训练时的分支断点续训和微调。推理和 YOLO9-t/s 不变。

- YOLO9、YOLO9-E2E 和 YOLO9-P2 把梯度 L2 范数裁剪到 10，它们的训练分配器用 fp32 给锚点打分，所以在 fp16 AMP 下选出的正样本可能不同。传入 `clip_max_norm=0` 可以得到 1.6.0 的裁剪行为；在 1.6.0 中开始、在 1.7.0 中续训的运行也会裁剪，除非你传入它。fp32 分配没有开关。

- 在所有使用共享仿射变换的家族中（YOLO9 及其 E2E 和 P2 变体、YOLOX、YOLOv7、YOLO-NAS、PP-YOLOE、RTMDet、PicoDet、RT-DETR 和 RT-DETRv2），非零的 `perspective` 现在按角点位移来做变形，包括从设置了它的 1.6.0 检查点续训的运行。取值 `0.002` 及以上给出相同的最大变形，负值会抛出 `ValueError`。没有任何设置能复现 1.6.0 的变形；默认值 `perspective=0` 不变。

- YOLO9 的 mosaic 流水线现在从 Beta(1, 1) 而不是 Beta(32, 32) 中抽取 mixup 比例，没有设置可以恢复旧值。YOLOv7 的默认训练因此改变，因为 YOLOv7 默认启用 mixup。YOLO9、YOLO9-E2E、YOLO9-P2、RT-DETR 和 RT-DETRv2 只有在启用 mixup 时才会变化。

- 不带 `name=` 的 `libreyolo train` 会把 EC、D-FINE、SegFormer 和 LingBot-Vision 的运行分别写入 `runs/train/ec_exp`、`dfine_exp`、`segformer_exp` 和 `lingbotvision_exp`，而不再是 `runs/train/exp`；V-JEPA 2 探针写入 `runs/train/train`，YOLO9-P2 的运行写入 `runs/train/yolo9_exp`。要用旧路径，传入 `name=exp`，YOLO9-P2 则传入 `name=yolo9_p2_exp`。

- `libreyolo train` 让 D-FINE、SegFormer 和 LingBot-Vision 从它们各自的配方开始。对 D-FINE 来说，就是 132 轮、`lr0=2e-4`、`weight_decay=1e-4`、2 轮预热，并且不使用 mosaic、mixup、HSV 或仿射数据增强。要发送 1.6.0 发送的那些值，请传入 `epochs=300 optimizer=sgd lr0=0.01 weight_decay=5e-4 warmup_epochs=5 mosaic=1.0 mixup=1.0 hsv_prob=1.0 degrees=10 translate=0.1 shear=2.0`。

- `libreyolo.validation.ConfusionMatrix` 按 `matrix[predicted, true]` 索引（原来是 `[actual, predicted]`），在 `conf_thres=0.25` 下统计检测（原来是 0.15），`names=` 和 `task=` 只能作为关键字参数传入，并且它的 `matrix` 不能再被重新赋值。请转置按旧矩阵索引的代码，并传入 `conf_thres=0.15` 以得到旧的计数。`plots=True` 会把计数写入 `confusion_matrix.png`，把归一化后的图写入 `confusion_matrix_normalized.png`。

- `predict(agnostic_nms=True)` 现在会跨类别抑制重叠的检测框，而不是发出警告。没有检测框的任务会抛出 `ValueError`，`LibreEnsemble` 和 HRNet 会抛出 `NotImplementedError`，SAM 层级则以 `TypeError` 拒绝这个关键字；请从这些调用中去掉这个参数。

- 未知的分类器 `interpolation` 会抛出 `ValueError`，包括 torchvision 的 `LANCZOS`、`BOX`、`HAMMING` 和 `NEAREST_EXACT` 模式。请使用 `bilinear`、`bicubic`、`nearest` 或 `bilinear_no_antialias`。

- 在与 1.6.0 的数字比较之前，请重新运行验证，涉及的情形有：在多个类别上训练的模型使用 `val(single_cls=True)`、混合了只有检测框的行和多边形行的分割数据集，以及 RTMDet-Ins（默认每张图像最多保留 `max_det=300` 个掩码；传入 `max_det=100` 可得到 MMDetection 的设置）。

- 没有 `libreyolo[rfdetr]` 时，使用 RF-DETR 或 DINOv2 模型名称的 CLI 调用会以 JSON 错误码 `model_load_failed` 退出，而不是 `model_not_found`，并且错误信息会点出 `pip install libreyolo[rfdetr]`。

完整的发布内容见[更新日志](/docs/changelog)。

## 从 1.5.0 升级到 1.6.0

- 如果环境将 ONNX Runtime 固定在 1.18 以下，安装 ONNX extra 前请升级到 `onnxruntime>=1.18.0`。

- SAM 3D Body 接受经过审查的快照资源集。安装 `libreyolo[hf]`，获得受限访问权限并使用自动获取，或在可信本地文件系统上提供未经修改的快照目录和固定的 MHR 资源。

- RF-DETR 非姿态缩放和分类器归一化修正后，请重新运行验证。与 1.5.0 结果比较前，请重新检查部署的置信度阈值；没有保留旧预处理的标志。

- D-FINE、DEIM、RT-DETRv4 和 YOLO-NAS 检测默认启用 FP16 AMP。传入 `amp=False` 可保留 FP32。

- 要使用以前的 YOLO9 微调设置，请设置 `aux_weight=0`、`max_labels=100` 和 `warmup_momentum=0.937`。新的中心对齐转换权重需要复现旧几何时，设置 `letterbox_pad="topleft"`。旧单 head 检查点继续使用原计算图续训。新的 YOLO9 运行会占用更多显存，所以如果沿用 1.5.0 的批大小出现显存不足，请调低 `batch`。

- RF-DETR 和 DINOv2 创建以家族命名、自动递增的训练目录。请更新使用产物路径的程序，或设置 `output_dir="runs/train", exist_ok=True`，保留原位置和复用行为。

- 保持分类 `mixup + cutmix <= 1`；无效组合现在会在初始化时报错。

- QAT 禁用 EMA、SyncBatchNorm 和检查点平均。请使用 QAT 的 best/last 检查点，不要依赖这些状态。

- 带数据集修改钩子的自定义加载器必须使用 `persistent_workers=False`，或在修改后重建 worker。不兼容的持久多 worker 副本现在会报错。

- NumPy 图像数组按 BGR 读取，也就是 OpenCV 的顺序。传入 RGB 数组（例如 `np.asarray(pil_image)`）时，请同时传入 `color_format="rgb"`；`cv2.imread()` 的输出和视频帧无需修改。

- 4D NumPy 数组或张量表示一批图像：`predict()` 返回一个列表，每张图像对应一个 `Results`，而不再只使用第一张图像。

- `train(resume=True)` 和 `train(resume="path/to/last.pt")` 会恢复该次运行保存的训练参数，并继续写入它的运行目录；显式传入的参数优先。对已发布的权重或已经达到 `epochs` 的运行做断点续训会抛出 `ValueError`。

- 启用验证时，最后一轮总会验证，因此短于 `eval_interval` 的运行现在也会报告指标并写出 `best.pt`。`val=False` 会关闭验证，最后一轮也包括在内。训练中的验证写入 `<run>/val`，而不是 `runs/val/`。

- CLI `train` 失败时会报告错误类型，因此配置错误以退出码 2 退出，而不是带着 `io_error` 以退出码 1 退出。

完整版本说明见[更新日志](/docs/changelog)，检查点转换见[导入权重](/docs/migrate)。

## 从 1.4.0 升级到 1.5.0

本页讲的是升级 LibreYOLO 本身。如果你要找的是如何加载上游项目的检查点
（checkpoint），那是[导入已有权重](/docs/migrate)，属于另一个话题。

这次发布的完整条目在[更新日志](/docs/changelog)里。下面只写需要你动手的那部分。

### 必须修改的代码

#### `allow_experimental=True` 已移除

那道确认关卡没有了，它背后的 `ddp_aware(experimental_key=...)` 机制也一并移除。
EC、RTMDet、PicoDet 和 FOMO 的训练与导出此前都要求传这个参数，所以任何训练这些
家族的脚本都会受影响。

```python
# 1.4.0
model.train(data="data.yaml", epochs=100, allow_experimental=True)

# 1.5.0：删掉这个参数
model.train(data="data.yaml", epochs=100)
```

没有留下弃用的兼容垫片。仍然传这个参数的调用会收到一条
`Unknown training config keys (ignored)` 警告，而这个参数不会产生任何效果。
`BaseModel.EXPERIMENTAL_WEIGHT_FILENAMES` 也随之移除。`get_download_notice()`
这个钩子保留了下来，MiDaS、SegFormer 和 YOLO9-P2 依然会覆盖它。

支持级别照常公布，只是不再是一个参数：见[稳定性层级](/docs/reference/stability-tiers)。

#### 导出等级 `"experimental"` 已移除

```python
from libreyolo.export.support import Tier

# 1.4.0: Literal["validated", "experimental", "blocked"]
# 1.5.0: Literal["validated", "available", "blocked"]
```

按层级字符串分支的代码，原先写 `"experimental"` 的地方现在要读作 `"available"`。
`BaseExporter` 不再为这些格式发出 `RuntimeWarning`。每种格式的状态列在
[导出矩阵](/docs/reference/export-matrix)里。

#### `pretrained=False` 与 `resume` 同时使用现在会被拒绝

这个组合此前会以自相矛盾的方式继续跑下去。现在它会抛出：

```
ValueError: pretrained=False cannot be combined with resume.
```

二选一。`pretrained=False` 从一份带种子的全新初始化开始，在 1.5.0 里对每个可训练
的家族都有效，而不再只对其中三个有效；`resume` 则从检查点接着跑一次被中断的训练。
两者都记录在[训练](/docs/train)里。

#### CLI `--imgsz` 是字符串，不是整数

影响面比听上去要窄。下面这两种写法都不受影响：

```bash
libreyolo predict --model yolo9-t --source img.jpg --imgsz 640   # 依然可用
```

```python
model.predict("img.jpg", imgsz=640)   # 依然可用
```

只有从 Python 直接调用 [CLI](/docs/cli) 命令函数的代码需要改，因为 `predict`、
`train` 和 `val` 把 `--imgsz` 从 `int` 放宽成了 `str`，这样它才能接受矩形尺寸：

```python
from libreyolo.cli.commands.predict import predict_cmd

predict_cmd(..., imgsz=640)      # 1.4.0
predict_cmd(..., imgsz="640")    # 1.5.0，而且 "480x640" 现在也能用了
```

`train` 的默认值现在是字符串 `"640"`。`export --imgsz` 本来就是字符串，`profile`
没有变化。

### 会变化的数值

有三处改动会在默认设置下改变指标。如果你跨版本跟踪结果，在拿 1.5.0 的运行结果和
1.4.0 的对比之前，先读这几条。

#### faster-coco-eval 是默认 COCO 指标后端

`val()` 和训练中每轮的验证，现在用 faster-coco-eval 的 C++ 后端计算 COCO 指标，
而不是 pycocotools。

这次切换的依据是在全部 100 个 RF100-VL 测试集划分上实测的一致性：1400 个指标值中
有 1381 个逐位一致，最大偏差 2.22e-16，主要指标的差值正好为 0，整体快 15.6 倍，
在检测密集的数据集上快 56 倍。你的数字不应该发生变化。但它们毕竟是由另一套实现
产出的，这正是它出现在这份清单里的原因。

没有安装 faster-coco-eval 时，pycocotools 仍然是自动的回退方案。要强制使用它：

```bash
libreyolo val --model yolo9-t --data coco.yaml --no-faster-coco-eval
```

```python
model.val(data="coco.yaml", faster_coco_eval=False)
```

`LIBREYOLO_FASTER_COCO_EVAL=0` 在全局做同样的事。实际使用的后端会以 INFO 级别写进
日志，在 `val()` 之后通过 `model.last_eval_backend` 暴露出来，并作为 `eval_backend`
包含在 [CLI](/docs/cli/val) 的 JSON 输出里。用
`pip install libreyolo[fast-eval]` 安装这条快路径。

#### 1.5.0 之前训练的 YOLOX 检查点需要覆盖 eps

这是这次发布里的坑。如果你微调过 [YOLOX](/docs/models/yolox)，就读一下这段。

YOLOX 规定 BatchNorm 使用 `eps=1e-3` 和 `momentum=0.03`。在 1.5.0 之前，这两个值
是以事后修补的方式应用的，而当你的数据集 `nc` 与检查点不同时，`train()` 会按类别数
重建模型，这份修补撑不过那次重建。这样的微调在训练和训练中验证时用的是 torch 的
默认值 `eps=1e-5`，之后为推理重新加载时又变回 `1e-3`：同一批张量，归一化却不同。

普通卷积的几个尺寸几乎不受影响。深度可分离的 `n` 变化很大，因为它逐通道的
`running_var` 小到足以让 eps 占主导。在 RF100-VL 的 `ball` 上，同一个 nano 检查点
按它训练时的 eps 评估得到 **0.566** mAP50-95，而按默认方式重新加载后只有
**0.151**。

在 1.5.0 之前训练的检查点带的是 eps=1e-5 的语义。要为它报出忠实的数字，要么在评估时
把 BN 的 eps 覆盖成 1e-5：

```python
import torch
from libreyolo import LibreYOLO

model = LibreYOLO("my-yolox-finetune.pt")
for module in model.model.modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        module.eps = 1e-5

model.val(data="data.yaml")
```

要么把 `sqrt((var + 1e-3) / (var + 1e-5))` 一次性折叠进 BN 权重并保存结果。在 1.5.0
及以后训练的检查点两者都不需要。

#### D-FINE 多尺度训练使用上游各尺寸配方

`base_size_repeat` 此前对每个尺寸都硬编码为 3。现在它按上游的规定逐尺寸取值：
**n** 关闭多尺度、按固定尺寸训练，**s** 为 20，**m** 为 6，**l** 为 4，**x** 为 3。
此前只有 x 是对得上的，所以 n、s、m 和 l 会遇到不同的尺度分布，并收敛到不同的指标。

要恢复旧行为，就显式设置它：

```python
from libreyolo.training.config import DFINEConfig

config = DFINEConfig(base_size_repeat=3)
```

DEIM 仍然使用硬编码的 3。家族细节见 [D-FINE](/docs/models/d-fine)。

### 需要了解但无需操作的变化

- **矩形 `imgsz` 的结果变了，因为它们之前是错的。** 检测框坐标、RTMDet 的掩码缩放、
  YOLO-NAS 的重新缩放，以及验证器对真值（ground truth）的缩放，现在都按高和宽分轴
  处理，而不是用一个标量。正方形 `imgsz` 逐位不变。在 1.4.0 上跑的矩形推理或验证
  缩放是错的。YOLO-NAS 现在会直接拒绝矩形 `imgsz`，而不是默默产出错误的输出。
- **指标字典多了一些键。** 来自 COCO 评估器的 `max_det`、`ar_max_det` 和
  `AR_max_det`，以及来自 FOMO 的 `metrics/loss` 和 `metrics/loss/ce`。默认设置下的
  取值没有变化，但任何遍历指标键的东西，包括自定义[日志记录器](/docs/train/loggers)
  和 CSV 表头，都会看到新的列。
- **带种子、且触发 head 重建的 YOLO9 训练**会从不同的初始化开始，因为种子现在是在
  重建之前而不是之后设置的。1.4.0 上带种子、微调到不同类别数的训练，在 1.5.0 上
  无法逐位复现。
- **`libreyolo[hub-kernels]` 在 CUDA 上现在真的会启用原生的 MS-deform-attn 内核。**
  1.4.0 把它挡在一个 RF-DETR 从来走不到的条件后面，所以那个内核从未运行过。RF-DETR
  和其他可变形注意力家族的预测结果可能在浮点容差范围内变化。默认安装不受影响，
  `LIBREYOLO_HUB_KERNELS=0` 可以关掉它。
- **`libreyolo predict` 会丢弃不支持的选项，而不是报错。** CLI 会按模型 `__call__`
  的签名过滤 kwargs，所以某个家族不接受的选项会被忽略，而不是抛出 `TypeError`。
  未知的 flag 名字仍然会被拒绝，报 `No such option`。
- **实时输入源会改变 JSON 输出的形态。** 摄像头、RTSP 流和屏幕捕获会隐式开启流式
  模式，于是每帧产出一条记录，而不是整次调用产出一条。这些
  [输入源](/docs/predict/sources)是 1.5.0 新增的，所以没有 1.4.0 的脚本会受影响。
- **把 `rfdetr-pose` 或 `yolonas-pose` 重新导出为 ONNX 会得到不同的输出名。** 1.4.0
  通过一个按输出数量判断的启发式规则，把它们的多张量姿态 head 误读成了分割。磁盘上
  已有的 `.onnx` 文件不受影响。
- **在不带 torch 的安装里**，结果里装的是 numpy 数组而不是 `torch.Tensor`，所以
  `.boxes.data` 返回的类型不同，NMS 打平局时的取舍也可能与 torchvision 不一样。
  装了 torch 的话，行为逐字节不变。见[轻量安装](/docs/lightweight-install)。
- **配置对象在构造时会做更多校验。** `TrainConfig` 原本没有 `__post_init__`，现在
  加上了，所以一份本来就不合法的配置会立刻报错，而不是跑到很深的地方才失败。
  `ValidationConfig` 的序列化多了一个 `edge_thresholds` 键，这会让从 1.4.0 的 dump
  严格做 `ValidationConfig(**dump)` 往返失效。
- **带任务后缀的家族，权重文件名的解析方式不一样了。** `segformer-b0` 现在解析为
  `LibreSegformerb0-sem.pt`。这修好了自动下载的 404，也会让任何硬编码了旧的无后缀
  文件名的脚本失效。
- **pytest 标记 `experimental_backend` 现在叫 `extended_backend`。** 只有当你用 `-m`
  跑测试套件时才有关系。

### 检查点和数据集

1.4.0 写出的检查点照常加载。[结构定义](/docs/reference/checkpoint-schema)为矩形模型
新增了 `imgsz_h` 和 `imgsz_w`，同时仍然为较旧的读取方写入标量 `imgsz = max(h, w)`。
[ExecuTorch](/docs/export/executorch) 和 [MNN](/docs/export/mnn) 导出现在需要一个
附带文件，分别是 `<program>.pte.json` 和 `<model>.mnn.json`，HRNet 导出则带上
`pose_input: "person_crop"`。数据集格式没有变化。
