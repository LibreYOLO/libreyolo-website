---
title: V-JEPA 2
families:
  - vjepa2
seo_title: V-JEPA 2：LibreYOLOでの推論と学習
description: V-JEPA 2は動画の埋め込みベクトルを生成し、アテンションを使う分類プローブの学習に対応します。
lead: V-JEPA 2は動画の埋め込みベクトルを生成し、アテンションを使う分類プローブの学習に対応します。
keywords:
  - V-JEPA 2
  - LibreYOLO
  - 埋め込みベクトル
  - 画像分類 python
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreVJEPA2l256-embed.pt", device="cpu")
        result = model(SAMPLE_IMAGE)
        print(result.embeddings)
        tokens = model.embed_tokens(SAMPLE_IMAGE)
        print(tokens.shape)
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreVJEPA2l256-cls-ssv2.pt", device="cpu")

        # データセットYAMLまたは分類フォルダーのパスを入力

        model.train(data=input("Dataset path: "), epochs=1, device="cpu",
        workers=0)
source_hash: 1fcdf8f15c0752cd
---

## インストール

```bash
pip install "libreyolo"
```

## 推論

<code-tabs name="predict" />

静止画像はクリップに変換されます。`embed_tokens()`は`(B, T, H, W, D)`のトークンを返し、公開APIの埋め込みベクトルはその平均を正規化したものです。エクスポートしたクリップ用グラフには、5次元の入力をランタイムに直接渡す必要があります。

## 学習

学習するのはアテンションを使うプローブと線形分類器のみで、エンコーダーは凍結したままです。分類タスクと動画マニフェスト形式のデータセットを使ってください。デフォルトは10エポック、バッチサイズ2、学習率0.001、ワーカー数0です。エンコーダー全体のファインチューニングと事前学習には対応していません。

必要な学習データについては[データセットの設定](/docs/train/datasets)を参照してください。

<code-tabs name="train" />

## エクスポート

<export-matrix />

形式ごとの依存関係とエクスポートしたファイルの読み込みについては[エクスポートの設定](/docs/export)を参照してください。

## チェックポイント

<checkpoint-table />

## ライセンス

<provenance-box></provenance-box>
