---
title: Qwen3-VL-Embedding
families:
  - qwen3vl_embedding
seo_title: Qwen3-VL-Embedding Python 用法：图像与文本检索向量
description: >-
  在 LibreYOLO 里运行 Qwen3-VL-Embedding-2B：图像和文本的嵌入向量落在同一个 2048 维空间里，用于搜索和检索。采用
  Apache-2.0 许可。
lead: >-
  Qwen3-VL-Embedding 是 Qwen 基于 Qwen3-VL
  构建的嵌入模型。它把图像和文本映射到同一个向量空间，所以一条文本查询能找到它描述的那一页或那张照片。LibreYOLO 在 embed 任务上运行它的 2B
  尺寸，这是库里第一个基于 VLM 的嵌入模型。
keywords:
  - Qwen3-VL-Embedding
  - Qwen
  - 多模态嵌入 embedding
  - 以文搜图
  - 图像检索 python
  - 文档检索
  - 嵌入向量
  - LibreVLM
last_verified: 1.8.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("qwen3-vl-embedding-2b")   # 权重在首次使用时下载
        result = model.predict(SAMPLE_IMAGE)
        print(result.embeddings.shape)              # (1, 2048)，单位长度
    - label: 检索
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("qwen3-vl-embedding-2b")
        pages = model.embed("pages/")               # (N, 2048)，每张图像一行

        # 查询带任务指令；图像保留默认指令
        q = model.embed_text(
            ["total due on the invoice"],
            prompt="Retrieve images or text relevant to the user's query.",
        )                                           # (1, 2048)
        scores = q @ pages.T                        # 余弦相似度
source_hash: 0a217649e02fb6ae
---

## 安装

Qwen3-VL-Embedding 属于 LibreYOLO 的 VLM 层级，需要 `vlm` extra。

```bash
pip install "libreyolo[vlm]"
```

## 预测

权重在首次使用时从 `LibreYOLO/LibreQwen3VLEmbedding2b` 下载，这是
`Qwen/Qwen3-VL-Embedding-2B` 的镜像，参数未作改动。架构通过 transformers 的
Qwen3-VL 实现运行。

<code-tabs name="predict" />

这个家族通过 `LibreVLM()` 或 `LibreQwen3VLEmbedding` 类加载。它和 CLIP、SigLIP2、
PE 遵循同一套 [embed](/docs/tasks/embeddings) 约定：`predict(image)` 返回形状为
`(1, 2048)` 的 `Results.embeddings`，`embed(source)` 把每张图像堆叠为一行，
`embed_text(texts, prompt=None)` 返回同一空间里的文本行。各行都经过 L2 归一化，
所以点积就是余弦相似度。

每个输入都是一段对话：一条系统指令，然后是图像或文本。最后一个 token 处的隐藏
状态经过归一化后就是向量。`prompt` 就是这条指令。默认值是上游的
`"Represent the user's input."`；在构造时设置 `prompt=` 可以同时改掉图像和文本的
指令，也可以在每次调用 `embed_text` 时传入。做检索时，给查询加上任务指令，文档
保留默认指令。

图像按 32 像素网格缩放到 4,096 到 1,843,200 像素之间，与上游的参考嵌入实现一致。
与该参考实现相比，在 CPU 上以 float32 运行时，文本向量的差异最多为 7.4e-7，图像
向量的余弦相似度至少为 0.99998。

## 变体

只有一个尺寸，`2b`。8B 尺寸没有封装。不支持的有：图像加文本的组合输入、视频片段
（视频会逐帧嵌入），以及作为选项的 Matryoshka 截断；请改为取前 `k` 列再重新归一化。

## 导出

仅支持推理：`chat()`、`set_classes()`、`train()`、`val()` 和 `export()` 会抛出异常。

## 许可证

<provenance-box>

Qwen3-VL-Embedding-2B 采用 Apache-2.0 许可，出自
[阿里巴巴的 Qwen 团队](https://github.com/QwenLM/Qwen3-VL-Embedding)。
允许商用。镜像不需要申请访问；由于上游的 Hugging Face 仓库没有附带 LICENSE 文件，
镜像附上了 GitHub 仓库里的 Apache-2.0 文本。

</provenance-box>

## 引用

<citation-block />
