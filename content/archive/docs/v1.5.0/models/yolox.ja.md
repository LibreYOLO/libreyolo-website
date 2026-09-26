---
title: YOLOX
families:
  - yolox
seo_title: YOLOX：Apache-2.0で推論、学習、エクスポート
description: LibreYOLOでYOLOXを物体検出に使います。Apache-2.0ライセンスのもとで、インストール、推論、学習、検証、エクスポートを行います。
lead: >-
  YOLOXは、分類と回帰を分離したヘッドを備え、SimOTAラベル割り当てで学習する、アンカーフリー（anchor-free）の1段階検出器です。LibreYOLOは物体検出でYOLOXに対応します。
keywords:
  - YOLOX
  - 物体検出
  - アンカーフリー 検出
  - decoupled head
  - SimOTA
  - リアルタイム 物体検出
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLOXs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLOXs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLOXs.pt")

        model.train(data="my-dataset.yaml", epochs=300, imgsz=640, batch=16,
        lr0=0.01)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLOXs.pt data=my-dataset.yaml \
          epochs=300 imgsz=640 batch=16 lr0=0.01
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLOXs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLOXs.pt data=my-dataset.yaml
    - label: COCOに対する検証
      language: bash
      code: |
        # 同梱のCOCO yamlにはdownload scriptが埋め込まれているため
        # データセットがローカルにない場合は明示的な許可が必要
        libreyolo val model=LibreYOLOXn.pt data=coco.yaml imgsz=416 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLOXs.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreYOLOXs.pt format=onnx imgsz=640

        libreyolo export model=LibreYOLOXs.pt format=tensorrt imgsz=640
        half=True
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリーはファイル接尾辞で経路を選ぶため、エクスポート済み成果物も
        # 通常のチェックポイントと同様に読み込まれ、同じResultsオブジェクトを返す
        model = LibreYOLO("LibreYOLOXs.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: f5ab735a29f85a95
---

## インストール

YOLOXに基本パッケージ以外の追加パッケージは必要ありません。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

返される `Results` オブジェクトはすべてのファミリーが返すものと同じなので、別の検出器への切り替えは1行の変更で済みます。`conf` は信頼度のしきい値、`iou` は分離された3つの推論スケール全体に適用するNMSのしきい値を設定します。ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## バリアント

6つのサイズが同じCSPバックボーンとPAFPNネックを共有します。もっとも小さい2つの `n` と `t` は、ほかの4つより小さな固定入力解像度で動作します。以下のベンチマーク表にそれぞれの正確な値があります。

<benchmark-table task="detect" />

<va-embed />

## 学習

<code-tabs name="train" />

設定を変更しなければ、trainerは `lr0=0.01`、SGD momentum 0.9、5エポックのウォームアップで300エポック実行し、最後の15エポックではmosaicおよびmixup拡張を無効にします。`train()` は `pretrained` 引数も受け付けますが、メソッド内でその値が読み取られることはありません。学習は常にモデル構築時の重みから続行されるため、`pretrained=False` でもネットワークは再初期化されません。

`imgsz` は、読み込んだチェックポイントのネイティブ解像度ではなく、基本学習設定の固定値をデフォルトとします。これは特に `n` と `t` のチェックポイントに影響します。`imgsz` を明示せずにいずれかの学習を続けると、公開時の小さなサイズではなく、大きい方のデフォルトへ切り替わります。

データセット、データ拡張、マルチGPU、loggerについては[学習](/docs/train)を参照してください。

## 検証

`val()` は、学習に使った形式の任意のデータセットに対して測定した適合率、再現率、mAP 50、mAP 50-95を含む `metrics/` キーの辞書を返します。

<code-tabs name="val" />

## エクスポート

<export-matrix />

エクスポート済み成果物はファイル接尾辞に基づいて `LibreYOLO()` から再度読み込めます。そのため、`.onnx` または `.engine` ファイルはチェックポイントと同様に動作し、同じ `Results` を返します。LibreYOLOをインストールしていない単独のランタイムでグラフを実行することもできますが、その場合は前処理と後処理を自身で記述する必要があります。CoreMLエクスポートでは、`nms=True` によりNMSをグラフへ埋め込めます。現在、このフラグを受け付けるファミリーはYOLOXとYOLOv9だけです。

<code-tabs name="export" />

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box></provenance-box>

## 引用

<citation-block />

