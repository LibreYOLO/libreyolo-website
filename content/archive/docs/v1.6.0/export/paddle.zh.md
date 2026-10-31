---
title: Paddle
seo_title: 从 LibreYOLO 导出到 PaddlePaddle
description: >-
  通过 X2Paddle 把 LibreYOLO 检测器转换成 PaddlePaddle 推理模型：锁定的工具链、静态 batch 为 1 的 FP32
  图，以及 CPU 推理。
lead: >-
  PaddlePaddle 的推理模型由一个 model.pdmodel 计算图和一个放在旁边的 model.pdiparams
  权重文件组成。LibreYOLO 会导出一张静态的 opset-15 ONNX 图，用 X2Paddle 转换，再连同一个 metadata.yaml
  一起打包，这样它就能和其他所有运行时一样，通过同一个工厂加载。
keywords:
  - yolo 导出 paddle
  - paddle 推理模型
  - x2paddle
  - model.pdmodel
  - model.pdiparams
  - onnx opset 15
last_verified: 1.5.0
meta:
  - label: 参数
    value: export(format="paddle")
    mono: true
  - label: 输出
    value: 一个包含 model.pdmodel、model.pdiparams 和 metadata.yaml 的目录
  - label: 额外依赖
    value: 'pip install "libreyolo[paddle]"'
    mono: true
  - label: 加载方式
    value: 'LibreYOLO("weights/LibreYOLO9t_paddle", device="cpu")'
    mono: true
  - label: 后端
    value: libreyolo.backends.paddle.PaddleBackend
    mono: true
  - label: 形状
    value: 静态形状，batch 为 1，opset 15。三条都会强制校验。
  - label: 精度
    value: 仅 FP32，仅 CPU。
  - label: 工具链
    value: PaddlePaddle 2.6.2、X2Paddle 1.6.0、ONNX 1.17 或更早，逐项精确检查
verification: >-
  读自 dev 分支上的
  libreyolo/export/paddle.py、libreyolo/export/exporter.py、libreyolo/export/support.py、libreyolo/backends/paddle.py、docs/paddle.md
  和 pyproject.toml。
snippets:
  install:
    - label: 安装
      language: bash
      code: |
        # Python 3.10 到 3.12，Windows 上验证过的路径是带 Ubuntu 22.04 的 WSL2
        pip install "libreyolo[paddle]"
    - label: 确认锁定的版本
      language: bash
      code: >
        python -c "from importlib.metadata import version;
        print(version('paddlepaddle'), version('x2paddle'), version('onnx'))"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # 输出目录 weights/LibreYOLO9t_paddle
        path = model.export(format="paddle")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format paddle
    - label: 完整参数
      language: python
      code: |
        model.export(
            format="paddle",
            imgsz=640,        # int，这个家族的方形画布
            batch=1,          # 其他值都会抛 ValueError
            dynamic=False,    # True 会抛 ValueError
            simplify=True,    # False 会抛 ValueError
            opset=15,         # 其他值都会抛 ValueError
            output_path=None, # None 输出 weights/<stem>_paddle
        )
  run:
    - label: 通过 LibreYOLO 运行
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_paddle", device="cpu")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: CLI
      language: bash
      code: |
        libreyolo predict --model weights/LibreYOLO9t_paddle \
          --source https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg --device cpu --save
    - label: 直接使用后端
      language: python
      code: |
        from libreyolo.backends.paddle import PaddleBackend

        # LibreYOLO() 遇到 Paddle 目录时构造的就是它，同样的 Results
        # 对象，中间没有工厂路由
        backend = PaddleBackend("weights/LibreYOLO9t_paddle", device="cpu")
        result = backend.predict("parkour.jpg")
        print(result.boxes.xyxy[:3])
    - label: 裸 Paddle
      language: python
      code: |
        import numpy as np
        import paddle.inference as paddle_infer
        import yaml

        directory = "weights/LibreYOLO9t_paddle"
        config = paddle_infer.Config(
            f"{directory}/model.pdmodel", f"{directory}/model.pdiparams"
        )
        config.disable_gpu()
        config.disable_mkldnn()
        config.switch_ir_optim(False)

        predictor = paddle_infer.create_predictor(config)
        handle = predictor.get_input_handle(predictor.get_input_names()[0])
        handle.reshape([1, 3, 640, 640])
        handle.copy_from_cpu(np.zeros((1, 3, 640, 640), dtype=np.float32))
        predictor.run()
        for name in predictor.get_output_names():
            print(name, predictor.get_output_handle(name).copy_to_cpu().shape)

        meta = yaml.safe_load(open(f"{directory}/metadata.yaml"))
        print(meta["model_family"], meta["task"], meta["names"])

        # 这条路上预处理和后处理都归你自己做
  support:
    - label: 导出前检查某个家族和任务
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: cdd8bf12286e2f53
---

## 安装

<code-tabs name="install" />

这个额外依赖锁定的正是一致性验证实测的那套技术栈：PaddlePaddle 2.6.2、X2Paddle
1.6.0 和 ONNX 1.17 或更早。这些锁定不只在安装时检查，导出时也会检查，版本不对会抛
`ImportError` 并点出期望的版本。更新的 Paddle 版本会拒绝 X2Paddle 1.6.0 生成的部分
静态代码，所以早点失败好过产出一个没人验证过的产物。

## 导出

<code-tabs name="export" />

有四个参数是写死的，而不是给了默认值。`dynamic` 必须是 `False`，`batch` 必须是 1，
`simplify` 必须是 `True` 才能得到完全静态的转换图，`opset` 必须是 15，这是 X2Paddle
1.6.0 能接受的上限。传别的值会在 trace 之前就抛错。

中间图上会跑一次归一化。ONNX 规定省略的 MaxPool dilation 取 1，PyTorch 会把这个全 1
的属性显式写出来，而 X2Paddle 1.6.0 不接受，所以导出器会去掉这个冗余的默认值，同时
让显式指定的运算保持不变。

产物是一个目录：`model.pdmodel`、`model.pdiparams` 和 `metadata.yaml`。X2Paddle 在
转换过程中生成的那些 Python 代码不属于产物的一部分。

## 运行产物

<code-tabs name="run" />

`LibreYOLO()` 会识别任何同时包含 `model.pdmodel` 和 `model.pdiparams` 的目录，读取
`metadata.yaml`，并返回和检查点（checkpoint）一样的 `Results` 对象。传 `auto` 或
`cpu` 以外的 device 会抛错：这个后端只支持 CPU。

工厂构造出来的是 `PaddleBackend`，它从 `libreyolo` 导出，也可以按
`libreyolo.backends.paddle.PaddleBackend` 导入。当你只想要后端、不想要工厂那套按后缀
路由的逻辑时，就自己构造它，比如为一个 `metadata.yaml` 不是你写的目录显式传 `task=`。
它的 `predict()` 接受同样的输入源，返回同样的结果。

裸运行时那个片段照搬了后端的配置，其中关掉的三个选项是有意为之。Paddle 2.6 的 CPU
融合流水线在优化可变形注意力（deformable attention）产生的那些大型 gather 和 scatter
图时可能崩溃，所以一致性是拿这张可移植的、未做融合的静态图测出来的。在那条路上，
预处理、解码、NMS 和坐标缩放都归你自己做。

## 限制

没有动态形状，没有 FP16，没有 INT8，没有内嵌 NMS，没有 GPU 运行时。

验证过的组合是 YOLO9 检测，YOLO9-E2E 和 YOLO9-P2 检测，EC 的检测、姿态和分割，
RT-DETRv4、D-FINE、DEIM 和 DEIMv2 检测，以及 YOLO-NAS 的检测和姿态。每一个都覆盖了
转换、CPU 运行时重新加载、原始输出一致性，以及与公开结果的对齐。

被阻止的组合，每个都记录了原因：

| 组合 | 原因 |
|---|---|
| RF-DETR，全部任务 | 需要 ONNX opset 17 和 GridSample；X2Paddle 1.6.0 只接受 opset 15 及以下，也没有 GridSample 的映射 |
| RT-DETR 和 RT-DETRv2 检测 | 训练好的图需要 opset 16 或更高的 GridSample |
| D-FINE 分割 | 能转换也能重新加载，但掩码 logit 的相对 RMS 误差是 3.52%，匹配掩码的最小 IoU 是 0.582 |
| YOLO9 分割 | 在 LibreYOLO 里 YOLO9 只做检测 |
| RTMDet-Ins 分割 | 动态卷积核的掩码解码没有导出运行时的契约 |

凡是既没列为已验证、也没列为被阻止的组合，都会被拒绝，并附上说明：它尚未通过 ONNX
到 Paddle 的转换路径验证。

完整的家族与任务网格见[导出矩阵](/docs/reference/export-matrix)。想查单个组合：

<code-tabs name="support" />
