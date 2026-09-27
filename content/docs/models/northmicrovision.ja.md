---
title: North Micro Vision
families:
  - northmicrovision
seo_title: LibreYOLOのNorth Micro Vision
description: North Micro Visionは、テキストの語彙を使って物体を検出します。
lead: North Micro Visionは、テキストの語彙を使って物体を検出します。
keywords:
  - North Micro Vision
  - LibreYOLO
  - 物体検出 python
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("north-micro-vision", device="cpu")
        model.set_classes(["person", "building"])
        result = model(SAMPLE_IMAGE)
        print(result.boxes)
source_hash: 9a0fe9b366c4232f
---

## インストール

```bash
pip install "libreyolo[vlm]" "transformers>=5.16.0"
```

## 推論

<code-tabs name="predict" />

アダプターは語彙のクラスごとに1つの検出クエリを実行します。共通のVLM追加パッケージの最低要件より新しいTransformers 5.16以降が必要です。学習には対応していません。

## ライセンス

<provenance-box></provenance-box>
