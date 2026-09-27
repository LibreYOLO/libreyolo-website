---
title: OCR
seo_title: OCR：LibreYOLOのテキスト検出と認識
description: >-
  LibreYOLOで画像内のテキストを見つけて読み取ります。四角形と文字列を推論し、JSONLデータセットへラベルを付け、hmean、エンドツーエンドF1、1-NEDで検証します。
lead: >-
  OCRは画像内のテキストを特定して読み取ります。LibreYOLOはocrタスクとして提供し、読み順に並べた各テキスト領域について、4点polygonと1つの文字列を返します。
keywords:
  - ocr python ライブラリ
  - シーンテキスト認識
  - テキスト検出 四角形
  - PP-OCRv5 python
  - エンドツーエンド text spotting
last_verified: 1.5.0
snippets:
  predict:
    - label: 画像内のテキストを読み取る
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # t tierは2つのうち軽量で、CPU向けに構築されている。SAMPLE_IMAGEなら
        # そのまま実行でき、独自のテキストを含む画像も指定可能
        model = LibreYOLO("LibrePPOCRt-ocr.pt")
        result = model(SAMPLE_IMAGE)

        regions = result.ocr
        print(len(regions), "regions")
        for text, score in zip(regions.texts, regions.conf):
            print(repr(text), float(score))
    - label: 四角形を読み取る
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPOCRt-ocr.pt")
        result = model(SAMPLE_IMAGE)

        regions = result.ocr
        print(regions.data.shape)   # (N, 4, 2) polygon、TL TR BR BL
        print(regions.xyxy)         # polygonのaxis-aligned hull
        print(regions.det_conf)     # .confとは別の検出スコア
    - label: 認識の信頼度で絞り込む
      language: python
      code: |
        import numpy as np
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPOCRt-ocr.pt")
        result = model(SAMPLE_IMAGE)

        # boolean maskではなく位置でindexを指定。sliceにより文字列と
        # 2つのスコア配列がgeometryとともに引き継がれる
        regions = result.ocr.numpy()
        keep = regions[np.flatnonzero(regions.conf >= 0.9)]
        print(keep.texts)
  val:
    - label: 検証して指標キーを読み取る
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePPOCRt-ocr.pt")
        metrics = model.val(data="my-ocr-dataset")

        print(metrics["metrics/det_precision"], metrics["metrics/det_recall"])
        print(metrics["metrics/det_hmean"])
        print(metrics["metrics/e2e_f1"])       # fitness
        print(metrics["metrics/rec_1-NED"])
source_hash: 58ad5305c9dd458c
---

## 定義

`ocr` タスクは1回の呼び出しで2つの処理を行います。画像内のすべてのテキスト領域を特定し、文字起こしします。scene textは回転していることが多いため、領域はaxis-aligned boxではなく4点polygonで返され、上から下、左から右の読み順に並びます。

推論によって `result.ocr` という `OCRRegions` ペイロードが埋められます。`.data` は元画像のpixel単位で表した `(N, 4, 2)` float polygon配列で、左上、右上、右下、左下の順です。`.texts` はN個の文字列のリスト、`.conf` は領域ごとの認識スコア、`.det_conf` は検出スコア、`.xyxy` は各polygonのaxis-aligned hullです。四角形は本物のpolygonなので、`result.boxes` には格納されません。`OCRRegions` をsliceすると、文字列と両方のスコア配列がgeometryとともに引き継がれます。

## モデル

2つのファミリーが `ocr` を提供します。

[PP-OCRv5](/docs/models/pp-ocrv5)は専用パイプラインです。differentiable binarization検出器がテキスト四角形を見つけ、SVTR・CTC認識器が読み取ります。両段階は認識用charsetとともに1つの `.pt` ファイルへまとめられます。CPU向けの軽量tierと高精度のserver tierの2つがあり、1つの辞書で簡体字中国語、繁体字中国語、英語、日本語、pinyinを扱います。

[SenseNova-Vision](/docs/models/sensenova-vision)は、ほかの6つのタスクも提供する同じ7Bチェックポイントから、tagged textとして単語を生成してOCRに対応します。`LibreVLM("sensenova-vision", task="ocr")` で読み込みます。`sensenova` 追加パッケージが必要で、重みは非商用利用に制限されています。ライセンスはそのページに記載されています。

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

PP-OCRv5は固定の長辺上限で検出を実行し、切り取った領域をバッチで認識します。`rec_batch` で、1回のforward passにより認識器へ渡すcrop数を制御します。2段階パイプラインは画像をまたいだバッチ処理を行わないため、複数画像ソースは順番に実行されます。ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## データセット形式

OCRラベルはsplitごとに1つのJSONLファイルで、画像自体の横に画像ごとのJSONオブジェクトを1つずつ格納します。

```text
my-ocr-dataset/
  images/
    val/receipt.jpg
  labels/
    val.jsonl
```

各行は画像を示し、その領域を列挙します。

```json
{"image": "receipt.jpg", "regions": [{"polygon": [[10, 12], [118, 14], [117, 40], [9, 38]], "text": "TOTAL 12.50"}]}
```

`polygon` は絶対pixel座標の4点四角形で、左上、右上、右下、左下の順です。テキストを読み取れない領域には、ICDARのdon't-care規約である `"text": "###"` のラベルを付けます。この領域は認識スコアから除外され、重なる推論結果はfalse positiveとして数えずに無視されます。

ルートディレクトリを `data=` として渡すだけで十分です。データセットYAMLも使えます。`path` に加えて任意の `images` および `labels` ディレクトリ名を指定し、OCRモデルは検出結果ではなく `Results.ocr` を返すため、schema placeholderとして `nc: 1` と `names: {0: text}` を指定します。完全な仕様については[データセット形式](/docs/reference/dataset-formats)を参照してください。

## 学習

どちらのOCRファミリーにも学習実装はありません。両方で `train()` が `NotImplementedError` を送出し、OCRサポートの対象は推論と検証だけです。PP-OCRv5のページには、Apache-2.0のアップストリーム学習コードと、ファインチューニング済みチェックポイントをLibreYOLOへ戻す変換スクリプトが記載されています。

## 検証

`val()` はパイプライン全体、つまり検出と認識をまとめて評価し、IoUが0.5を超える推論polygonと正解polygonを1対1で照合します。

<code-tabs name="val" />

`metrics/det_precision`、`metrics/det_recall`、`metrics/det_hmean` は位置推定だけを評価します。文字列の内容に関係なく、polygonの重なりだけで一致を判定します。`metrics/e2e_precision`、`metrics/e2e_recall`、`metrics/e2e_f1` は読み取りも加えます。一致には、同じpolygonの重なりに加え、NFKC正規化と空白除去後の完全な文字列一致が必要で、大文字と小文字は引き続き区別されます。`metrics/e2e_f1` は、best-checkpoint選択で読む値 `fitness` でもあります。

`metrics/rec_1-NED` は、検出ですでに一致した組に対して認識器だけを評価します。1から正規化編集距離を引いた値なので、1文字だけ違う文字列は、エンドツーエンドF1では0でも、こちらでは1に近いスコアとなります。

## エクスポート

このタスクで利用できるエクスポート形式はありません。PP-OCRv5は1つのtrace可能なグラフではなく、連携して動く2つのネットワークであり、両方のファミリーで、すべての形式に対して `export()` が例外を送出します。LibreYOLO外へデプロイするには、アップストリームでファインチューニングし、アップストリームのデプロイ経路を使ってください。

