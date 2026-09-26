---
title: Qwen3-VLグラウンディング
families:
  - ground_qwen3vl
seo_title: LibreYOLOのQwen3-VLグラウンディング
description: Qwen3-VLグラウンディングは、指示の対象を画像上の点として特定します。
lead: Qwen3-VLグラウンディングは、指示の対象を画像上の点として特定します。
keywords:
  - Qwen3-VL grounding
  - LibreYOLO
  - 画像 グラウンディング
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreGround, SAMPLE_IMAGE

        model = LibreGround("qwen3-vl-2b", device="cpu")
        result = model(SAMPLE_IMAGE, prompt="the person")
        print(result.points.xy)
source_hash: c0e411fb5fdba6cc
---

## インストール

```bash
pip install "libreyolo[ground]"
```

## 推論

<code-tabs name="predict" />

1回の呼び出しには`prompt=`または`query=`を使い、指示を保持するには`set_query()`を使ってください。1枚の画像に複数の指示を渡すと、クエリごとに最大1つのクリック位置を返します。画像リストやフォルダーを対象に複数のクエリを渡すと、エラーになります。座標は元の画像を基準とします。学習、検証、エクスポートには対応していません。[グラウンディングAPI](/docs/reference/ground-api)を参照してください。

## ライセンス

<provenance-box></provenance-box>

## 引用

<citation-block />
