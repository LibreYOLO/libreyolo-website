---
title: Perception Encoder
families:
  - pe
seo_title: Perception EncoderをLibreYOLOで使う
description: Perception Encoderは、画像、テキスト、有限長の動画クリップを共通の埋め込み空間に変換します。
lead: Perception Encoderは、画像、テキスト、有限長の動画クリップを共通の埋め込み空間に変換します。
keywords:
  - Perception Encoder
  - LibreYOLO
  - 画像分類 python
  - 埋め込みベクトル
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePEt16-cls.pt", device="cpu")
        model.set_classes(["person", "building"])
        result = model(SAMPLE_IMAGE)
        print(result.probs)
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePEt16-cls.pt", device="cpu")
        model.export(format="onnx")
source_hash: 6e0183876fd868e3
---

## インストール

```bash
pip install "libreyolo[clip]"
```

## 推論

<code-tabs name="predict" />

ゼロショット分類の前にクラスを設定してください。埋め込みベクトルを取得する場合は、読み込み時に`task="embed"`を指定します。動画の埋め込み処理では、デフォルトで`clip_frames=8`フレームを抽出し、そのベクトルを平均してL2正規化します。学習には対応していません。エクスポートした動画用グラフは、ランタイムへの直接入力を受け付けます。

## エクスポート

<export-matrix />

<code-tabs name="export" />

形式ごとの依存関係とエクスポートしたファイルの読み込みについては[エクスポートの設定](/docs/export)を参照してください。

## チェックポイント

<checkpoint-table />

## ライセンス

<provenance-box></provenance-box>

## 引用

<citation-block />
