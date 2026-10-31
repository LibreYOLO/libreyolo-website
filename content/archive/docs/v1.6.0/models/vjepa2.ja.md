---
title: V-JEPA 2
families:
  - vjepa2
seo_title: V-JEPA 2をLibreYOLOで使う
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

        # 動画データセットの YAML に置き換えてください (train/val マニフェストは '<video-path> <class-id>'
        形式の行)

        model.train(data="path/to/your/video_dataset.yaml", epochs=1,
        device="cpu", workers=0)
  val:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreVJEPA2l256-cls-ssv2.pt", device="cpu")

        # 動画データセットの YAML に置き換えてください (val() は '<video-path> <class-id>' 形式の val
        マニフェストを読み込みます)

        metrics = model.val(data="path/to/your/video_dataset.yaml", workers=0)

        print(metrics)
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreVJEPA2l256-embed.pt", device="cpu")
        model.export(format="onnx")
source_hash: 86cbde874e2a664a
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


## 検証

<code-tabs name="val" />

タスクに対応する形式のデータセットを使います。[検証](/docs/train/validation)でデータセットの要件と返される指標を説明しています。

## エクスポート

<export-matrix />

<code-tabs name="export" />

形式ごとの依存関係とエクスポートしたファイルの読み込みについては[エクスポートの設定](/docs/export)を参照してください。

## チェックポイント

<checkpoint-table />

## ライセンス

<provenance-box></provenance-box>
