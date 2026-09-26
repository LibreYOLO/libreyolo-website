---
title: RKNN
seo_title: 导出到 RKNN，跑在瑞芯微 NPU 上
description: 把 LibreYOLO 检测器编译成瑞芯微的 .rknn 产物：需要你自己装的厂商 SDK、四个通过验证的 RK3588 变体，以及模拟器一致性。
lead: >-
  RKNN 是瑞芯微（Rockchip）的 NPU 编译格式。LibreYOLO 会导出一个 opset 19 的 ONNX 中间产物，用 RKNN
  Toolkit2 SDK 把它编译出来，还能在 Toolkit2 的主机模拟器里、不用板子就把编译后的图和 ONNX Runtime 做对比。
keywords:
  - yolo 导出 rknn
  - 瑞芯微 npu
  - rk3588
  - rknn-toolkit2
  - rknn 模拟器 一致性
  - orange pi yolo 推理
last_verified: 1.5.0
meta:
  - label: 参数
    value: 'export(format="rknn", name="rk3588")'
    mono: true
  - label: 输出
    value: >-
      一个 .rknn 文件、一个 .rknn.metadata.json sidecar，以及 verify=True 时的一份
      .rknn.parity.json 报告
  - label: 额外依赖
    value: PyPI 上没有。rknn-toolkit2 是厂商 SDK，要你自己安装。
  - label: 回读方式
    value: 不经过 LibreYOLO。产物在板子上由瑞芯微自己的运行时执行。
  - label: 形状
    value: 固定的正方形、批大小 1、opset 19。三条都会强制检查。
  - label: 精度
    value: 厂商的浮点构建。half=True 和 int8=True 会被拒绝。
  - label: 范围
    value: RK3588 上的四个检测变体：YOLO9-t、YOLO9-E2E-t、PicoDet-s 和 YOLO-NAS-s
verification: >-
  读自 dev 分支上的
  libreyolo/export/rknn.py、libreyolo/export/exporter.py、libreyolo/export/support.py
  和 docs/rknn.md。实测的一致性数字来自 docs/rknn.md 中日期为 2026-08-04 的验证记录。
snippets:
  install:
    - label: LibreYOLO 这一侧
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: 厂商 SDK，由你自己安装
      language: bash
      code: |
        # rknn-toolkit2 是瑞芯微的 SDK，采用单独的许可。LibreYOLO 既不打包
        # 也不安装它。仅限 x86_64 Linux；Windows 上用 WSL2 或
        # Linux 容器
        #
        # Toolkit2 2.3.2 需要 setuptools<81，并且在 ONNX 1.19 及更高版本上
        # 会失败，因为它的编译器仍然导入已被移除的 onnx.mapping
        pip install "setuptools==80.9.0" "onnx==1.18.0"

        # 然后从瑞芯微自己的 wheel 仓库安装对应的 rknn-toolkit2 wheel，
        # 并确认它能导入：
        python -c "import rknn.api; print('rknn-toolkit2 ready')"
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        # 输出 weights/LibreYOLO9t.rknn 和 weights/LibreYOLO9t.rknn.metadata.json

        path = model.export(format="rknn", name="rk3588", imgsz=640,
        verify=True)

        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format rknn --name rk3588 \
          --imgsz 640 --verify
    - label: 完整参数
      language: python
      code: |
        model.export(
            format="rknn",
            name="rk3588",     # 目标平台；target= 和 target_platform= 也可以
            imgsz=640,         # 必须和该变体记录的画布一致
            batch=1,           # 其他任何值都会抛 NotImplementedError
            dynamic=False,     # True 会抛 ValueError
            opset=19,          # 其他任何值都会抛 NotImplementedError
            verify=False,      # True 会跑 PC 模拟器，并以一致性作为门槛
        )
  parity:
    - label: 不用板子，对着已有的 ONNX 产物做一致性检查
      language: python
      code: |
        import numpy as np
        from libreyolo.export import verify_rknn_simulator_parity

        input_tensor = np.random.default_rng(0).standard_normal(
            (1, 3, 640, 640), dtype=np.float32
        )
        metrics = verify_rknn_simulator_parity(
            "weights/LibreYOLO9t.onnx",
            input_tensor,
            target_platform="rk3588",
            rtol=1e-3,
            atol=1e-4,
            raise_on_failure=False,
        )
        print(metrics)
  support:
    - label: 编译前检查某个家族和任务
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: c659713cc3c8cc9e
---

## 安装

编译需要瑞芯微的 RKNN Toolkit2，它以厂商 SDK 的形式分发，采用瑞芯微自己的许可，
不是 LibreYOLO 的依赖。没有 `libreyolo[rknn]` 这个额外依赖，这个格式也没有哪一部分
是一行命令就能装好的。

<code-tabs name="install" />

编译和检查数值一致性都不需要板子。延迟、功耗和温度的测量需要一块 RK3588 板子，而这些
数据一个都还没有记录。

## 导出

<code-tabs name="export" />

在编译任何东西之前，请求会先对照一份精确的模型变体清单做校验，画布也一样要校验：传入
的 `imgsz` 只要不是该变体记录的那个尺寸就会报错，而不会默默编译出一个没测过的东西。
LibreYOLO 会写出一个 opset 19 的 ONNX 中间产物，把它编译掉，按需做一次模拟，之后再把
中间产物删掉。

元数据是一个名为 `<model>.rknn.metadata.json` 的 sidecar，因为 RKNN 格式没有可移植的
元数据字段。

`verify=True` 会在编译出产物的同一个会话里跑 Toolkit2 的 PC 模拟器，用同一份输入把每
个输出和 ONNX Runtime 做对比，并写出带逐输出误差指标的 `<model>.rknn.parity.json`。
门槛是余弦相似度不低于 0.9999、归一化 RMSE 不高于 0.02，只施加在那些还没有逐元素接近
的输出上；厂商的浮点构建会把内部张量降到半精度，所以即便解码出的检测框是稳的，严格的
`allclose` 也不成立。失败的那一次会写出 `<model>.rknn.failed.parity.json`，丢弃候选产
物，并且不动此前成功导出在该路径上的任何文件。

想拿一个已经有的 ONNX 产物做对比，又不重新导出一遍：

<code-tabs name="parity" />

Toolkit2 的模拟器跑的是 `load_onnx` 和 `build` 产出的内存中的图。没有板子它就无法重新
加载一个面向具体目标的 `.rknn` 文件，这也是 `verify=True` 把编译、导出和模拟放在同一个
会话里做的原因。

## 运行产物

`libreyolo/backends` 里没有 RKNN 这一项，所以 `LibreYOLO()` 不会加载 `.rknn` 文件。编译
出的产物要部署到板子上，由瑞芯微自己的运行时执行，在那边预处理、解码、NMS 和坐标缩放都
归应用自己负责。

`<model>.rknn.metadata.json` 里带着类别名、输入尺寸、任务和目标平台，这些正是应用复现
LibreYOLO 后处理所需要的东西。把它和编译好的模型一起发出去。

想要一个不需要板子的主机侧检查，就留一份同样固定形状的 ONNX 产物，照上面的方式在模拟器
里做对比。

## 限制

能编译的组合有四个，而且它们是模型变体，不是家族：

| 变体 | 任务 | 画布 | 目标 |
|---|---|---:|---|
| YOLO9-t | 检测 | 640 | RK3588 |
| YOLO9-E2E-t | 检测 | 640 | RK3588 |
| PicoDet-s | 检测 | 320 | RK3588 |
| YOLO-NAS-s | 检测 | 640 | RK3588 |

其他一切都会在编译前被拒绝，报错信息会说明这个版本的 RKNN 只限于经过模拟器测试的那几
个确切检测变体。其他模型确实有「只编译」的结果，但那些结果是刻意不作为支持呈现的：在
同一次测量中，RF-DETR 有两个解码器 `GridSample` 节点没有完成 lowering，而 D-FINE、
RT-DETR、RT-DETRv2、RT-DETRv4、DEIM、DEIMv2 和 EC 虽然能编译也能模拟，解码出来的输出却
明显是错的。

批大小 1、静态形状、opset 19。`half=True` 会被拒绝，因为 RKNN 没有暴露 LibreYOLO 的
`half` 契约；`int8=True` 也会被拒绝，直到有具代表性的校准和任务精度结果为止。

其他瑞芯微目标都会被拒绝：`rk3588` 是唯一通过验证的平台。

完整的家族与任务网格见[导出矩阵](/docs/reference/export-matrix)。想查单个组合：

<code-tabs name="support" />
