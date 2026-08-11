---
title: SSD
families:
  - ssd
seo_title: SSD（SSD300）：LibreYOLOでの物体検出
description: >-
  LibreYOLOでSSD300を実行します。BSD-3-Clauseの下で推論、検証、ONNXエクスポートに対応する、1段階のVGG16検出器です。学習経路はありません。
lead: >-
  SSD（Single Shot MultiBox
  Detector）は、別の領域提案段階を使わず、密なデフォルトボックスのグリッドから1回の順伝播ですべてのボックスとクラススコアを予測します。LibreYOLOはVGG16をバックボーンとするSSD300チェックポイントを推論専用検出器として提供します。
keywords:
  - SSD
  - SSD300
  - Single Shot MultiBox Detector
  - 物体検出
  - VGG16
  - アンカーベース 検出器
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSSD300.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSSD300.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSSD300.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSSD300.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSSD300.pt")

        # imgszは意図的に省略 SSD300はチェックポイント固有のキャンバスで
        # トレースされ他の値ではエクスポート開始前にエラー
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSSD300.pt format=onnx
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO

        # ファクトリーはファイル接尾辞で振り分けるためエクスポート済み成果物も
        # チェックポイントと同様に読み込まれて同じResultsオブジェクトを返す
        model = LibreYOLO("LibreSSD300.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 3b3f9ea72291c4fa
---

## インストール

SSDにオプションの追加パッケージは不要です。インポートするものはすべて基本インストールに含まれます。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

返される `Results` オブジェクトはすべてのファミリーで共通のため、別の検出器への置き換えは1行の変更で済みます。SSDはクラスごとのスコアを使ってデフォルトボックスのグリッドをデコードし、NMSを実行します。そのため、このライブラリのクエリベース検出器とは異なり、`conf`、`iou`、`max_det` はすべて実際に効果を持ちます。ソース、ストリーミング、結果の処理については、[推論](/docs/predict)を参照してください。

## バリアント

SSDが提供するチェックポイントは1つです。固定されたネイティブキャンバスを持ち、VGG16をバックボーンとするSSD300ネットワークです。このファミリーにはサイズやスケールの選択肢がなく、推論、検証、エクスポートはすべて同じ1つのグラフを使用します。

重みファイルは `LibreSSD300.pt` で、ファミリーの接頭辞に唯一のサイズキー `"300"` が続きます。背後のクラスは `LibreSSD` なので、直接構築する場合はファイル名に基づくクラス名ではなく `LibreSSD(size="300")` を使用します。

## 検証

`val()` は `metrics/` キーの辞書を返します。内容は適合率、再現率、mAP 50、mAP 50-95で、学習に使用した形式の任意のデータセットに対して測定されます。

<code-tabs name="val" />

## エクスポート

<export-matrix />

SSDはONNXだけにエクスポートできます。その他の形式は現在、このファミリーで除外されています。エクスポートでは常にチェックポイント固有のキャンバスを使用し、グラフはNMSを統合した出力ではなくSSDの未処理の結合ヘッドを公開します。そのため、エクスポート時に `nms=True` は受け付けません。LibreYOLO独自のバックエンドは、グラフを再度読み込んだ後にデコードと抑制を実行します。

<code-tabs name="export" />

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box>

LibreYOLOのSSD300コードは、論文著者独自のCaffeリリースから移植したものではありません。torchvisionのBSD-3-Clause SSD300実装を基にしており、上でアップストリームソースとしてリンクされているのもそのリポジトリです。バックボーンのVGG16の重みは、Karen SimonyanとAndrew ZissermanがCC BY 4.0で公開したOxfordの完全畳み込み型reduced VGGNetまでさかのぼります。

</provenance-box>

## 引用

<citation-block />
