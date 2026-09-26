---
title: "升级到 1.6.0"
seo_title: "LibreYOLO 从 1.5.0 升级到 1.6.0"
description: "LibreYOLO 1.6.0 的迁移步骤：预处理、训练默认值、训练目录、固定资源、QAT 和数据加载器。"
lead: "1.6.0 更改了预处理、训练默认值和检查点处理。请重新验证已保存的基线，复现旧训练时显式设置以前的默认值。"
keywords:
  - libreyolo 升级
  - libreyolo 1.5.0 迁移
  - allow_experimental 移除
  - libreyolo 破坏性变更
  - yolox bn eps
  - faster-coco-eval 默认
last_verified: "1.6.0"
source_hash: f4fda6ef286113ab
---

## 从 1.5.0 升级到 1.6.0

- 如果环境将 ONNX Runtime 固定在 1.18 以下，安装 ONNX extra 前请升级到 `onnxruntime>=1.18.0`。

- SAM 3D Body 接受经过审查的快照资源集。安装 `libreyolo[hf]`，获得受限访问权限并使用自动获取，或在可信本地文件系统上提供未经修改的快照目录和固定的 MHR 资源。

- RF-DETR 非姿态缩放和分类器归一化修正后，请重新运行验证。与 1.5.0 结果比较前，请重新检查部署的置信度阈值；没有保留旧预处理的标志。

- D-FINE、DEIM、RT-DETRv4 和 YOLO-NAS 检测默认启用 FP16 AMP。传入 `amp=False` 可保留 FP32。

- 要使用以前的 YOLO9 微调设置，请设置 `aux_weight=0`、`max_labels=100` 和 `warmup_momentum=0.937`。新的中心对齐转换权重需要复现旧几何时，设置 `letterbox_pad="topleft"`。旧单 head 检查点继续使用原计算图续训。

- RF-DETR 和 DINOv2 创建以家族命名、自动递增的训练目录。请更新使用产物路径的程序，或设置 `output_dir="runs/train", exist_ok=True`，保留原位置和复用行为。

- 保持分类 `mixup + cutmix <= 1`；无效组合现在会在初始化时报错。

- QAT 禁用 EMA、SyncBatchNorm 和检查点平均。请使用 QAT 的 best/last 检查点，不要依赖这些状态。

- 带数据集修改钩子的自定义加载器必须使用 `persistent_workers=False`，或在修改后重建 worker。不兼容的持久多 worker 副本现在会报错。

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

没有留下弃用的兼容垫片。仍然传这个参数的调用会抛出 `TypeError`。
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
from libreyolo import LibreYOLOX

model = LibreYOLOX("my-yolox-finetune.pt")
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
  flag 名字拼错了现在也会被默默忽略。
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
