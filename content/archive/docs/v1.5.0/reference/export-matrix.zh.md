---
title: 完整导出矩阵
seo_title: LibreYOLO 导出支持矩阵及其规则
description: LibreYOLO 如何判定一个家族、任务和格式的组合能不能导出：十二种格式、三个等级、兜底规则，以及一致性阈值。
lead: >-
  导出支持就是按 (family, task, format)
  三元组做的一次查表。本页讲这个矩阵的形状、没有显式条目覆盖的单元格由哪些规则填上，以及怎么去查你关心的那个组合。
keywords:
  - libreyolo 导出支持
  - 导出支持矩阵
  - onnx tensorrt openvino tflite
  - libreyolo formats 命令
  - 导出一致性阈值
  - 导出 NotImplementedError
last_verified: 1.5.0
verification: >-
  格式、等级、兜底顺序、任务和家族的屏蔽以及 NCNN 屏蔽名单读自 libreyolo/export/support.py；别名和共享参数读自
  libreyolo/export/exporter.py；等级定义读自
  docs/adr/0011-export-support-tiers.md；一致性阈值读自 docs/export_support.md，全部基于
  v1.5.0。逐个组合的单元格没有抄录在这里；用下面的代码片段去查。
snippets:
  usage:
    - label: 查询矩阵，不需要模型
      language: python
      code: |
        from libreyolo.export.support import (
            EXPORT_FORMATS,
            get_support,
            validated_alternatives,
        )

        print(EXPORT_FORMATS)

        entry = get_support("yolo9", "detect", "onnx")
        print(entry.tier, entry.since)
        print(entry.constraint)

        print(validated_alternatives("yolo9", "detect"))
    - label: CLI
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
        libreyolo formats --family yolo9 --task detect --json
  export:
    - label: 导出，并读懂一次拒绝
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.export.support import get_support

        model = LibreYOLO("LibreYOLO9t.pt")
        print(model.export(format="onnx"))

        # 调用前先查一下：被屏蔽的组合会在预检阶段抛错，
        # 报错信息里带的就是这个 reason
        blocked = get_support("domedetr", "detect", "onnx")
        print(blocked.tier)
        print(blocked.reason)
source_hash: 83de3289634888c6
---

## 矩阵的形状

矩阵以 `(family, task, format)` 为键。家族键是模型注册表（registry）里的规范名，
任务键来自 `libreyolo.tasks.TASKS`，格式一共十二种：

`onnx`、`torchscript`、`executorch`、`tensorrt`、`openvino`、`paddle`、`mnn`、
`rknn`、`ncnn`、`tflite`、`coreml`、`coreai`。

`model.export(format=...)` 还额外接受两个别名：`engine` 对应 `tensorrt`，`litert`
对应 `tflite`，后者是 TensorFlow Lite 现在的名字。格式本身和 `.tflite` 后缀都没有变。

<code-tabs name="usage" />

因为一个单元格是三个键的函数，完整表格很大，而且每个版本都会变。它是生成出来的，
不是手写的，放在库仓库的 `docs/export_support.md` 里。查这个矩阵请用 Python 或 CLI，
而不是去读某一份副本。

## 三个等级

| 等级 | 含义 |
|---|---|
| `validated` | 数值一致性（parity）已由 CI 或有文档记录的 nightly 运行覆盖 |
| `available` | 转换已经实现，但运行时的数值一致性证据没有记录过 |
| `blocked` | 预检阶段在 tracing 之前抛出 `NotImplementedError`，并给出原因 |

validated 和 available 的组合都会直接往下走，不需要额外确认，也不会给一条笼统的警告。
它们记录下来的证据和约束会在生成的文档里保持可见。blocked 的组合会在依赖检查、
校准数据加载、tracing 和产物生成之前就失败。

新增一条 validated 条目需要一个一致性测试和一个 `since` 字段。

一个 `SupportEntry` 带四个字段：`tier`、一个 `reason` 字符串、`since` 版本，
以及一个 `constraint` 字符串。约束是集成时真正要紧的那部分：一个对勾只在它写明的
条件下成立，而这些条件通常是固定的输入画布、batch 1、FP32，以及一个指定的运行时版本。

## 单元格是怎么定下来的

`get_support(family, task, fmt)` 按下面的顺序解析。第一条匹配上的规则胜出。

1. 未知的任务，或者不在这十二种之内的格式，返回 `blocked`。
2. 存在显式的 `(family, task, format)` 条目时，按记录返回。
3. 家族级别的屏蔽返回 `blocked`，附带该家族的原因。
4. 任务级别的屏蔽返回 `blocked`，附带该任务的原因。
5. 对 `ncnn`，在 NCNN 屏蔽名单上的家族返回 `blocked`。
6. `mnn` 返回 `blocked`：这个家族和任务没有运行时契约。
7. `rknn` 返回 `blocked`。这个版本的 RKNN 只限于在模拟器上测过的那几个检测变体：RK3588 上的 YOLO9-t、YOLO9-E2E-t、YOLO-NAS-s 和 PicoDet-s。
8. `tensorrt` 和 `openvino` 返回 `available`：转换路径是有的，但该家族和任务的运行时一致性没有记录过。
9. `tflite`、`paddle`、`coreai` 和 `coreml` 返回 `blocked`，各有各自的原因。
10. 其余一切返回 `available`：转换已经实现，运行时的数值一致性没有记录。

第 8 到第 10 步的不对称是有意为之。TensorRT 和 OpenVINO 是从 ONNX 通用转换的，
所以没列出来的组合值得一试。TFLite、Paddle、Core AI 和 CoreML 各自都需要一条按家族
写的路径，所以没列出来的组合是一次拒绝，而不是一份邀请。

## 被屏蔽的任务

对任何没有显式条目的家族，这些任务都是屏蔽的。

| 任务 | 原因 |
|---|---|
| `ocr` | 两个网络加上按区域动态裁剪，不符合单图导出契约 |
| `point` | 该家族没有接入共享的点热图和后端峰值解码契约 |
| `semantic` | 该家族没有接入共享的稠密 logits 和后端 argmax 契约 |
| `mesh` | 人体网格的图输出、元数据和运行时契约都没有定义 |
| `normal` | 该家族没有接入固定画布的稠密单位法线和后端重归一化契约 |
| `panoptic` | 全景分割导出没有后端运行时契约 |
| `gaze` | 该家族没有接入共享的双 head logits 和后端期望解码契约 |

显式条目会覆盖这些规则，所以比如一个已经接好线的语义分割家族照样能导出。

## 被屏蔽的家族

| 家族 | 屏蔽范围 |
|---|---|
| `depth_anything3` | 所有格式；它的深度计算图不在导出运行时契约里 |
| `domedetr` | 所有格式。PAQI 按每张图设定 query 数量，所以 trace 出来的图只对被 trace 的那张图有效。想要一个能导出的 DETR，用 D-FINE |
| `eomt` | 实例分割和全景分割导出，这两者没有运行时解析 |
| `l2cs` | ONNX、TorchScript、ExecuTorch、TensorRT 和 OpenVINO 之外的一切 |
| `hrnet` | ONNX、TorchScript、OpenVINO 和 TensorRT 之外的一切 |
| `sam`、`sam2`、`sam3`、`edgetam`、`mobilesam` | 所有格式；可提示模型的导出不在 v1 运行时契约的范围内 |
| `grounding_dino`、`owlv2`、`omdet_turbo`、`ov_deim` | 所有格式；开放词汇的运行时导出不在 v1 的范围内 |
| `florence2`、`kosmos2`、`lfm2vl`、`internvl3`、`qwen3vl`、`smolvlm2`、`locateanything` | 所有格式；生成式 VLM 的导出不在 v1 的范围内 |

PicoSAM3 是可提示这一档里的例外：它会把自己那个原始的 96 像素 ROI 网络导出成 ONNX。

## 对 NCNN 的屏蔽

DETR 风格的解码器需要 NCNN 没有实现的采样算子，所以除非有显式条目另作规定，
下面这些家族对 `ncnn` 都是屏蔽的：Deformable DETR、DETR、DINO-DETR、D-FINE、
LW-DETR、DEIM、DEIMv2、RT-DETR、RT-DETRv2、RT-DETRv4、RF-DETR 和 EC。
拒绝信息会把 ONNX、OpenVINO、TorchScript 和 TensorRT 列为替代方案。

## 一致性阈值

一个 validated 单元格意味着导出的产物在下面这些界限内复现了原生模型：

| 任务组 | 阈值 |
|---|---|
| 检测和 OBB | 匹配上的检测框 IoU 高于 0.95，分数 MAE 低于 0.01 |
| 分割和全景分割 | 掩码 IoU 高于 0.95 |
| 姿态 | 原生分辨率下关键点 L2 低于 2 像素 |
| 分类 | logits 余弦高于 0.999，且 top-1 类别相同 |
| 深度和复原 | 相对原生输出 PSNR 高于 40 dB |
| 表面法线 | 平均角度误差低于 0.1 度 |
| 点 | 峰值位置在一个输出单元格之内相等 |

DETR 的 query 行是一个无序集合，所以 DETR 家族的一致性是把 query 行当作集合来对齐，
而不是按位置对齐。

## 导出

<code-tabs name="export" />

被屏蔽的组合会在预检阶段抛出 `NotImplementedError`，报错信息里带着记录下来的原因。
`validated_alternatives(family, task)` 返回这一对组合下已验证的格式，这正是紧挨着
一条拒绝打印出来最有用的东西。

所有导出器共享的参数列在[模型 API 页面](/docs/reference/model-api)上。
格式专属的参数在各自的格式页面上。

## 读懂一条约束

一个 validated 单元格是关于某一个实测配置的断言，而不是关于这个格式整体的断言。
像 `FP32, batch 1, fixed 520x520 input` 这样的约束字符串，意思是一致性是在那个形状
和精度下记录的。换一个分辨率或者批大小去导出，照样能产出产物；只是那不再是这个数字
所来自的配置。
