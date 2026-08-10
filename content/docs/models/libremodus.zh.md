---
title: LibreMODUS
families: [libremodus]
seo_title: "LibreMODUS：在 LibreYOLO 里做 any-to-any 图像分析"
description: "在 LibreYOLO 里用 LibreMODUS 做深度、法线、边缘和检测，并用 any2any() 把它们组合起来。仅支持推理；权重从 EPFL-VILAB 加载。"
lead: "LibreMODUS 是对 MODUS 14B-A7B 检查点（checkpoint）的一个仅支持推理的集成，这是一个 any-to-any 模型，把一种由图像派生的输入变成另一种：输入 RGB，输出深度；输入深度，输出法线；上述任意一种再加一个短语，输出检测框。LibreYOLO 通过标准的 predict API 支持四个任务，通过 any2any() 支持更广的一组。"
keywords: [LibreMODUS, MODUS, any-to-any, 单目深度估计, 表面法线估计, 边缘检测, "文本提示检测物体", EPFL VILAB]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreMODUS

        model = LibreMODUS(size="14b-a7b", task="normal")
        result = model.predict("room.jpg")
        normals = result.normal_map.data

        model.set_task("edge")
        result = model.predict("room.jpg")
        edges = result.edges.data

        # 不设置自定义词汇表时，detect 把检查点的 COCO 标签 token
        # 解码成连续的 COCO-80 类别 id
        model.set_task("detect")
        result = model.predict("street.jpg")
        print(result.boxes.xyxy)
    - label: 短语定位
      language: python
      code: |
        from libreyolo import LibreMODUS

        model = LibreMODUS(task="detect")
        # set_classes() 把检测切换成短语定位：每个短语独立运行，
        # 并通过同一个 Boxes 契约返回
        model.set_classes(["red bus", "cyclist"])
        result = model.predict("street.jpg", conf=0.2)
        print(result.boxes.xyxy, result.boxes.cls)
    - label: any2any()
      language: python
      code: |
        from libreyolo import LibreMODUS

        model = LibreMODUS()

        # 一到三个由图像派生的输入（rgb、depth、normal、canny/edge），
        # 加上可选的辅助文本，组合指向一个目标
        result = model.any2any(
            inputs={"rgb": "room.jpg"},
            target="normal",
            steps=10,
            cfg=2.0,
            seed=0,
        )
        normals = result.normal_map.data

        # 通过 any2any() 做定位，需要一个点名该短语的文本输入
        result = model.any2any(
            {"rgb": "street.jpg", "text": "red bus"},
            target="grounding",
        )
        print(result.boxes.xyxy)
---

## 安装

LibreMODUS 需要自己的 extra，它会为这个检查点所需的大模型分发（dispatch）拉入 `accelerate`。

```bash
pip install "libreyolo[modus]"
```

LibreYOLO 不会重新分发也不会镜像 MODUS 权重。默认情况下，加载一个 `LibreMODUS` 模型会在一个固定的 Hugging Face 修订版上直接从 `EPFL-VILAB/MODUS` 下载所需文件，而一次全新的下载总是需要用户自己已认证的 Hugging Face 账号，哪怕上游的托管门禁暂时是开放的。请先审阅并接受上游条款，然后认证：

```bash
hf auth login
```

```python
from libreyolo import LibreMODUS

model = LibreMODUS(token="hf_...")
```

想完全避免网络请求，就指向一份你已经有的快照：

```python
model = LibreMODUS(checkpoint_path="/models/MODUS")
```

那个目录里必须包含 `model.safetensors`、`ae.safetensors`、`llm_config.json`、`vit_config.json`、`tokenizer_config.json`、`vocab.json` 和 `merges.txt`。检查点的条款允许做什么，见下面的许可证一节。

## 预测

<code-tabs name="predict" />

标准的任务 API 覆盖四个任务，每个都映射到一个 MODUS 目标：`depth` 对应相对深度（`result.depth_map`），`normal` 对应表面法线（`result.normal_map`），`edge` 对应 Canny 风格的边缘（`result.edges`），`detect` 对应 COCO-80 检测框（`result.boxes`），除非 `set_classes()` 把它切换成短语定位（phrase grounding）。`set_task()` 在同一个已加载的模型上在它们之间切换。官方发布的配方用十步流采样（flow sampling），文本引导 4.0，图像引导 2.0；在构造时用 `inference_steps=`、`inference_cfg=` 和 `inference_image_cfg=` 覆盖它们。

`any2any()` 通向更广的那片公开分析面：一到三个由图像派生的输入（`rgb`、`depth`、`normal`、`canny`/`edge`），加上可选的辅助文本，组合指向深度、法线、边缘、SAM 派生的边缘、COCO 检测或短语定位中的任意一个。所有由图像派生的输入都必须描述同一张对齐的画布；LibreMODUS 会拒绝宽高不匹配的输入，而不是各自独立地缩放它们。`chain=(...)` 生成中间目标，并把它们喂回同一个上下文，且不超出这个检查点训练时的三条件预算。`verify=N`（N >= 2）生成 N 个候选，并保留在一次带约束的自洽性检查里得分最高的那个，通过 `result.verification_score` 暴露出来。

`dtype="bf16"`（默认值）匹配已发布检查点的精度；`dtype="fp8"` 把解码器主干里符合条件的线性层权重按每输出通道的缩放存成 E4M3，一次性转换到 `~/.cache/libreyolo/modus/fp8` 下的本地缓存里，并在每次矩阵乘法时反量化回输入 dtype，所以它做的是内存上的取舍，而不是激活层面精度上的取舍。

`train()`、`val()` 和 `export()` 都会抛异常：LibreMODUS 仅支持推理，不提供数据集验证，也没有 ONNX、TensorRT 或 TFLite 的导出路径。批量 `predict()` 和测试时增强（TTA）同样不支持；每次调用处理一张图像。

## 许可证

<provenance-box>

LibreYOLO 不在任何地方托管或镜像 MODUS 检查点，包括它自己的 Hugging Face 组织：加载它总是直接从 EPFL-VILAB/MODUS 拉取那个固定的修订版，或者读取磁盘上已有的、位于 `checkpoint_path` 的快照。

</provenance-box>

## 引用

<citation-block />
