---
title: Mask R-CNN
families:
  - mask_rcnn
seo_title: LibreYOLOのMask R-CNN：推論、検証、エクスポート
description: >-
  LibreYOLOでMask
  R-CNNを実行し、物体検出とインスタンスセグメンテーションを行います。BSD-3-Clauseのtorchvision移植版をインストールし、推論、検証、エクスポートします。
lead: >-
  Mask R-CNNはFaster
  R-CNNへ領域ごとのマスク分岐を追加し、検出した各ボックスとともにセグメンテーションマスクを予測します。LibreYOLOは検出とインスタンスセグメンテーション向けにtorchvision実装を移植しています。
keywords:
  - Mask R-CNN
  - インスタンスセグメンテーション
  - 物体検出
  - Faster R-CNN
  - torchvision
  - 2段階 検出器
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMaskRCNNr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreMaskRCNNr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: ボックスのみ
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # task="detect"でマスクヘッドを省略し同じチェックポイントから
        # ボックスを返す 結果にマスクはない
        model = LibreYOLO("LibreMaskRCNNr50.pt", task="detect")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMaskRCNNr50.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])      # マスク
        print(metrics["metrics/mAP50-95(B)"])   # ボックス
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMaskRCNNr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMaskRCNNr50.pt")
        model.export(format="onnx", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreMaskRCNNr50.pt format=onnx imgsz=800
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリーはファイル接尾辞で振り分けるためエクスポート済み成果物も
        # チェックポイントと同様に読み込まれて同じResultsオブジェクトを返す
        model = LibreYOLO("LibreMaskRCNNr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.masks.data.shape)
source_hash: 9608459b801aa6d5
---

## インストール

Mask R-CNNにオプションの追加パッケージは不要です。インポートするものはすべて基本インストールに含まれます。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

返される `Results` オブジェクトはすべてのファミリーで共通のため、別の検出器への置き換えは1行の変更で済みます。`task` 引数なしでチェックポイントを読み込むと、インスタンスマスクを返します。セグメンテーションがこのファミリーのデフォルトタスクであるためです。その場合、`result.masks` はボックスとともにマスクを保持します。`task="detect"` を渡すとマスクヘッドなしで同じ重みを読み込み、ボックスだけを返します。`conf` と `iou` は信頼度とNMSのしきい値を設定します。クエリベース検出器とは異なり、Mask R-CNNはアップストリームのNMS処理を維持します。ソース、ストリーミング、結果の処理については、[推論](/docs/predict)を参照してください。

## バリアント

バックボーンは1つです。特徴ピラミッドを持つResNet-50で、torchvisionのv2 Mask R-CNNビルダーを使用します。公開チェックポイントにはBSD-3-Clauseライセンスが適用され、このファミリーの両方のタスクを提供します。そのため、選択するサイズはありません。

## 検証

`val()` は `metrics/` キーの辞書を返します。このチェックポイントのデフォルトのセグメンテーションタスクに対し、通常の `metrics/mAP50-95` キーがマスクスコアを保持し、同じ実行で `(B)` 接尾辞の下にボックスも報告されます。そのため、1回の処理で両方を利用できます。

<code-tabs name="val" />

## エクスポート

<export-matrix />

Mask R-CNNはバッチサイズ1でONNXだけにエクスポートできます。エクスポート済みグラフはアップストリームのリサイズ処理とマスク貼り付け処理を内部に保持するため、LibreYOLOは指定内容にかかわらず `dynamic=True` を強制し、正方形でないソースでもグラフが有効になるようにします。エクスポート済み `.onnx` ファイルは、ファイル接尾辞によって `LibreYOLO()` から再度読み込まれ、同じ `Results` を返します。

<code-tabs name="export" />

## チェックポイント

このファミリーで公開されているすべての重みファイルです。次の1つのチェックポイントはdetectの下に記載されていますが、同じファイルをセグメンテーションにも読み込めます。`task` 引数を渡さなければ、デフォルトでマスクを返します。

<checkpoint-table />

## ライセンス

<provenance-box>

Mask R-CNNはLibreYOLOのFaster R-CNNラッパーのサブクラスとして構築されています。同じtorchvisionソースとBSD-3-Clauseライセンスを共有し、同じ移植元コミットのマスク予測器とマスクRoIヘッドを追加します。

</provenance-box>
