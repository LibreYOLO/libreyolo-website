---
title: EoMT
families:
  - eomt
seo_title: EoMT：セマンティック・インスタンス・パノプティックセグメンテーション
description: >-
  LibreYOLOでEoMTを使い、decoderなしの標準DINOv2視覚Transformer上でセマンティック、インスタンス、パノプティックセグメンテーションを行います。MITライセンスです。
lead: >-
  専用のpixel
  decoderを持たない標準的な視覚Transformer上に構築されたセグメンテーションネットワークです。encoder自体に追加された学習済みqueryがマスクを予測します。LibreYOLOはセマンティック、インスタンス、パノプティックセグメンテーションでEoMTに対応します。
keywords:
  - EoMT
  - encoder-only mask transformer
  - DINOv2
  - パノプティックセグメンテーション
  - インスタンスセグメンテーション
  - セマンティックセグメンテーション
last_verified: 1.5.0
snippets:
  predict:
    - label: セマンティック
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEoMTl-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # (H, W)クラスID
        print(mask.classes)      # 画像内に存在するソート済みクラスID
    - label: インスタンスセグメンテーション
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファイル名の-seg接尾辞でinstanceタスクが選ばれるため
        # ここではtask引数が不要
        model = LibreYOLO("LibreEoMTl-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.boxes.xyxy)
        print(result.masks.data.shape)
    - label: パノプティック
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreEoMTl-panoptic.pt")

        result = model(SAMPLE_IMAGE, save=True)


        pan = result.panoptic

        print(pan.data.shape)       # (H, W)セグメントID

        print(pan.segments_info)    # セグメントごとの[{"id": ..., "category_id": ...},
        ...]
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreEoMTl-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: セマンティック
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: インスタンスセグメンテーション
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # マスク
        print(metrics["metrics/mAP50-95(B)"])   # ボックス
    - label: パノプティック
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-panoptic.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PQ"])
        print(metrics["metrics/SQ"], metrics["metrics/RQ"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEoMTl-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-sem.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreEoMTl-sem.pt format=onnx
        libreyolo export model=LibreEoMTl-sem.pt format=tensorrt half=True
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリーはファイル接尾辞で経路を選ぶため、エクスポート済み成果物も
        # 通常のチェックポイントと同様に読み込まれ、同じResultsオブジェクトを返す
        model = LibreYOLO("LibreEoMTl-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: 64b2da642999f150
---

## インストール

EoMTに任意の追加パッケージは必要ありません。インポートするものはすべて基本インストールに含まれています。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。ファイル名のタスク接尾辞（`-sem`、`-seg`、`-panoptic`）でタスクが選ばれ、`LibreYOLO()` はそのファイル名から推測するため `task=` 引数は不要です。

<code-tabs name="predict" />

セマンティックセグメンテーションは `result.semantic_mask` を埋め、`.data` にクラスIDの `(H, W)` 配列を格納します。インスタンスセグメンテーションは `result.boxes` と `result.masks` を埋め、ほかのセグメンテーションファミリーと同じ形を返します。パノプティックセグメンテーションは `result.panoptic` を埋めます。`.data` に `(H, W)` のセグメントIDマップ、`.segments_info` にセグメントごとの `{"id", "category_id"}` 辞書のリストが格納されます。`conf` はquery選択を絞り込みます。semanticタスクはNMSなしでpixelごとにargmaxを取るため、`iou` は効果がありません。ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## バリアント

DINOv2をバックボーンとするs・b・lの3つのencoderサイズがあります。semanticチェックポイントはADE20Kで512 px、instanceとpanopticのチェックポイントはCOCOで640 pxを使って学習され、instanceには1280 pxで学習された2つ目のチェックポイントもあります。アップストリームでDINOv2のinstance segmentation用重みが公開されているのはサイズlだけです。sとbはsemanticおよびpanopticだけで公開されています。DINOv3をバックボーンとするEoMTバリアントもアップストリームにはありますが、アクセス制限付きの非商用DINOv3重みに依存するため、ここでは提供しません。

LibreYOLOはEoMTを学習しません。このファミリーでは `train()` が `NotImplementedError` を送出し、上の[サポート階層](/docs/models)では推論専用と示されています。

## 検証

`val()` はタスクに応じて処理を振り分けます。semanticは `metrics/mIoU` と `metrics/pixel_accuracy` を返します。インスタンスセグメンテーションは、ほかのセグメンテーションファミリーと同じマスク・ボックスmAPキーを返します。panopticはPanoptic Qualityを `metrics/PQ` として返し、`metrics/SQ`（segmentation quality）と `metrics/RQ`（recognition quality）へ分け、さらに `metrics/PQ_things` と `metrics/PQ_stuff` も返します。

<code-tabs name="val" />

## エクスポート

<export-matrix />

現在エクスポートできるのはsemanticタスクだけです。instanceとpanoptic segmentationで `export()` を呼び出すと `NotImplementedError` が送出されます。query-mask出力のランタイムエクスポート仕様がまだないためです。エクスポート済みsemantic成果物はファイル接尾辞に基づいて `LibreYOLO()` から再度読み込めるため、`.onnx` または `.engine` ファイルはチェックポイントと同様に動作し、同じ `Results` を返します。

<code-tabs name="export" />

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box></provenance-box>

## 引用

<citation-block />

