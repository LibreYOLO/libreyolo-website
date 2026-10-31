---
title: PicoDet
families:
  - picodet
seo_title: LibreYOLOのPicoDet：推論、学習、エクスポート
description: >-
  LibreYOLOでPicoDetをモバイル物体検出に使います。Apache-2.0ライセンスのもとで、インストール、推論、学習、検証、エクスポートを行います。
lead: >-
  PicoDetはモバイルおよびエッジCPU向けに構築された1段階検出器で、ESNetバックボーン、CSP-PANネック、共有Generalized
  Focal Lossヘッドを備えます。LibreYOLOは物体検出でPicoDetに対応します。
keywords:
  - PicoDet
  - PP-PicoDet
  - 物体検出
  - モバイル 物体検出
  - エッジ 物体検出
  - ESNet
  - Generalized Focal Loss
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePICODETs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibrePICODETs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePICODETs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibrePICODETs.pt data=my-dataset.yaml
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePICODETs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=300, batch=16, lr0=0.01,
        )
    - label: CLI
      language: bash
      code: >
        # imgszは明示する価値がある。CLIのデフォルトは640だが

        # sチェックポイントのネイティブ解像度は320

        libreyolo train model=LibrePICODETs.pt data=my-dataset.yaml imgsz=320
        epochs=300 batch=16 lr0=0.01
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePICODETs.pt")
        model.export(format="onnx", imgsz=320)
        model.export(format="tensorrt", imgsz=320, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibrePICODETs.pt format=onnx imgsz=320

        libreyolo export model=LibrePICODETs.pt format=tensorrt imgsz=320
        half=True
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリーはファイル接尾辞で経路を選ぶため、エクスポート済み成果物も
        # 通常のチェックポイントと同様に読み込まれ、同じResultsオブジェクトを返す
        model = LibreYOLO("LibrePICODETs.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 947aa47214abc4c0
---

## インストール

PicoDetに基本パッケージ以外の追加パッケージは必要ありません。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

返される `Results` オブジェクトはすべてのファミリーが返すものと同じなので、別の検出器への切り替えは1行の変更で済みます。`conf` は信頼度のしきい値、`iou` はNMSのしきい値を設定します。ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## バリアント

3つのサイズがあり、それぞれ独自の固定入力解像度を持ちます。`s` がもっとも小さく、`l` がもっとも大きいサイズです。サイズに伴って解像度も上がるため、大きなチェックポイントはパラメータが多いだけでなく、画像ごとの実行コストも高くなります。

<benchmark-table task="detect" />

<va-embed />

## 学習

<code-tabs name="train" />

損失成分とassignerはアップストリームの手順に従います。VFL、DFL、GIoU、SimOTAに、分類品質の重み付けとdynamic-IoU VFL targetを組み合わせます。同じチェックポイントでの推論は、アップストリームとbit単位で同等です。

`train()` 独自のdocstringによると、full-datasetでの収束、マルチGPUでの動作、水平反転以外のデータ拡張は未確認です。ネイティブ解像度320の `s` チェックポイントは、小規模ファインチューニングのテストにライブラリが使う30枚・2クラスのfixtureにおいて、LibreYOLOの精度下限を安定して超えていません。このサイズはfull COCO規模により適しています。

`train()` は `pretrained` 引数も受け付けますが、メソッド内でその値が読み取られることはありません。学習は常にモデル構築時の重みから続行されるため、`pretrained=False` でもネットワークは再初期化されません。Pythonで `imgsz` を未設定にすると、読み込んだチェックポイントのネイティブ解像度が使われます。`s` は320、`m` は416、`l` は640です。CLIは常に `imgsz` を渡し、デフォルトは640なので、チェックポイントに合わせて設定してください。

ほかの設定を変更しなければ、trainerはSGD、`lr0=0.01`、momentum 0.9、weight decay 4e-5、1エポックのウォームアップを使うcosine scheduleで300エポック実行します。適用されるデータ拡張は水平反転だけです。

データセット、データ拡張、マルチGPU、loggerについては[学習](/docs/train)を参照してください。

## 検証

`val()` は、学習に使った形式の任意のデータセットに対して測定した適合率、再現率、mAP 50、mAP 50-95を含む `metrics/` キーの辞書を返します。

<code-tabs name="val" />

## エクスポート

<export-matrix />

エクスポート済み成果物はファイル接尾辞に基づいて `LibreYOLO()` から再度読み込めます。そのため、`.onnx` または `.engine` ファイルはチェックポイントと同様に動作し、同じ `Results` を返します。LibreYOLOをインストールしていない単独のランタイムでグラフを実行することもできますが、その場合は前処理と後処理を自身で記述する必要があります。

<code-tabs name="export" />

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box>

LibreYOLOの移植は、PaddleDetectionの元のPP-PicoDetをPyTorchで再実装したBo396543018/Picodet_Pytorchに従います。mmcvを取り除き、すべてのactivationを正確に一致させているため、Boのパイプラインで変換したPaddlePaddleチェックポイントを数値的なずれなしで読み込めます。両方のソースには、論文の作者と同じApache-2.0の条件が適用されます。

</provenance-box>

## 引用

<citation-block />

