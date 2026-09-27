---
title: YOLOv3
families:
  - yolo3
seo_title: LibreYOLOのYOLOv3：推論、検証、エクスポート
description: >-
  LibreYOLOでYOLOv3を実行します。tiny、base、SPPサイズを備えた固定済みの推論専用museumファミリーです。パブリックドメインライセンスの下で推論、検証、エクスポートを行います。
lead: >-
  YOLOv3はYOLO系列にmulti-scale predictionと独立したlogistic
  classifierを追加したDarknet-53検出器です。LibreYOLOはtiny、base、SPPサイズの固定済み推論専用モデルとして提供します。
keywords:
  - YOLOv3 使い方
  - Darknet
  - Darknet-53
  - 物体検出
  - マルチスケール検出
  - museum model
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO3b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO3b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: SPPサイズ
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # SPPバリアントは検出ヘッドの前にspatial pyramid poolingブロックを追加し
        # 独自のネイティブ入力サイズで実行
        model = LibreYOLO("LibreYOLO3spp.pt")
        result = model(SAMPLE_IMAGE)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO3b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO3b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO3b.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO3b.pt format=onnx
        libreyolo export model=LibreYOLO3b.pt format=tensorrt half=True
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリがファイル接尾辞で振り分けるため、エクスポートした成果物も
        # 通常のチェックポイントと同様に読み込まれ、同じResultsオブジェクトを返す
        model = LibreYOLO("LibreYOLO3b.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: a4c652bb2707fc8f
---

## インストール

YOLOv3には基本パッケージ以外の追加パッケージは必要ありません。

```bash
pip install libreyolo
```

## 推論

このファミリーは推論専用です。`train()`は`NotImplementedError`を発生させるため、このページに
「学習」セクションはありません。推論、検証、エクスポートにはすべて対応します。重みは初回使用時に
Hugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

返される`Results`オブジェクトはすべてのファミリーで共通のため、別の検出器への切り替えは
1行の変更だけで済みます。`conf`は信頼度のしきい値、`iou`はNMSのしきい値で絞り込みます。
3つのヘッドからのボックスを統合する前に、スケールごとに適用されます。入力ソース、
ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## 検証

`val()`は、検証に使う形式の任意のデータセットに対して測定した適合率、再現率、mAP 50、
mAP 50-95を含む`metrics/`キーの辞書を返します。

<code-tabs name="val" />

## エクスポート

<export-matrix />

エクスポートした成果物はファイル接尾辞に基づいて`LibreYOLO()`から再読み込みされます。そのため、
`.onnx`または`.engine`ファイルはチェックポイントと同様に動作し、同じ`Results`を返します。
LibreYOLOをインストールせずにランタイムからグラフを直接実行する方法にも対応しますが、その場合は
前処理と後処理を自分で実装します。

<code-tabs name="export" />

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box></provenance-box>

