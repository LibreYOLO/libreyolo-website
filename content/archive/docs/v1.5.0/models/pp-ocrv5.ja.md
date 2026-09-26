---
title: PP-OCRv5
families:
  - ppocr
seo_title: PP-OCRv5：LibreYOLOでテキスト検出と認識
description: >-
  LibreYOLOのPP-OCRv5で多言語のscene-text
  OCRを行います。Apache-2.0のtとlチェックポイントのインストール、推論、検証を説明します。
lead: >-
  PP-OCRv5はPaddleOCRのテキスト検出・認識パイプラインです。differentiable
  binarization検出器がテキストの四角形を見つけ、SVTR・CTC認識器が読み取ります。LibreYOLOは2つの層をPyTorchへ移植しています。
keywords:
  - PP-OCRv5 使い方
  - PaddleOCR
  - OCR python
  - テキスト検出
  - 文字認識
  - scene text
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPOCRl-ocr.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for text, conf in zip(result.ocr.texts, result.ocr.conf):
            print(text, float(conf))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibrePPOCRl-ocr.pt source=receipt.jpg save=True
    - label: 四角形
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPOCRl-ocr.pt")
        result = model(SAMPLE_IMAGE)

        # 読み順の(N, 4, 2)ポリゴン: 左上、右上、右下、左下
        # 検出された四角形は実際のポリゴン（回転テキスト）のため
        # result.boxesではなくresult.ocrに格納
        print(result.ocr.data.shape)
        print(result.ocr.det_conf)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePPOCRl-ocr.pt")
        metrics = model.val(data="my-dataset")

        print(metrics["metrics/det_hmean"])
        print(metrics["metrics/e2e_f1"])       # 主要指標
        print(metrics["metrics/rec_1-NED"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibrePPOCRl-ocr.pt data=my-dataset
source_hash: 9835057f8bd95bc1
---

## インストール

PP-OCRv5には基本パッケージ以外の追加パッケージは必要ありません。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

各チェックポイントは検出と認識の両段階を1個の`.pt`ファイルにまとめ、認識用文字セットと
パイプラインのデフォルト値をチェックポイントのメタデータに格納します。認識器は1個の辞書で
簡体字中国語、繁体字中国語、英語、日本語、ピンインを読み取ります。`result.ocr`は
`OCRRegions`payloadです。`.data`には4点ポリゴン、`.texts`には転写テキスト、`.conf`には
領域ごとの認識スコア、`.det_conf`には検出スコアが入ります。複数画像の入力ソースは順番に
実行されます。2段階パイプラインは画像をまたいでバッチ化しません。入力ソース、ストリーミング、
結果の処理については[推論](/docs/predict)を参照してください。

## バリアント

層は2種類です。`t`はCPU向けの軽量なPP-LCNetV3・PP-OCRv5_mobileバックボーン上に構築され、
`l`は高精度なPP-HGNetV2 serverバックボーン上に構築されます。両方の層が固定長辺上限で検出を
実行し、クロップをバッチで認識します。`rec_batch`は1回の順伝播で認識器を通過するクロップ数を
制御します。

## 検証

`val()`は、画像ディレクトリと`labels/<split>.jsonl`ファイルの組、または同等のデータセットYAMLに
対してパイプラインを測定します。各ラベルには画像ごとのテキスト領域ポリゴンと転写テキストが
記載されます。検出hmean（IoUで対応付けた適合率・再現率・F1）、エンドツーエンドF1（hmeanと
正規化後の転写テキスト完全一致を組み合わせたチェックポイントのfitness指標）、1-NED
（対応付けたペアの平均正規化編集距離）を報告します。

<code-tabs name="val" />

## エクスポート

<export-matrix />

PP-OCRv5は検出と認識が連動する2ネットワークのパイプラインで、トレース可能な単一グラフでは
ありません。そのためエクスポートは未実装で、対応する形式はまだありません。この形式以外の
チェックポイントが必要な場合は、Apache-2.0のアップストリーム学習コードを直接ファインチューニングし、
`weights/convert_ppocr_weights.py`で結果を変換してください。

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box></provenance-box>

## 引用

<citation-block />

