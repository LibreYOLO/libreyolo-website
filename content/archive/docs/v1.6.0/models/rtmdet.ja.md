---
title: RTMDet
families:
  - rtmdet
seo_title: LibreYOLOのRTMDet：推論、学習、エクスポート
description: >-
  LibreYOLOでRTMDetの物体検出とRTMDet-Insのインスタンスセグメンテーションを実行します。Apache-2.0でインストール、推論、学習、検証、エクスポートに対応します。
lead: >-
  RTMDetは、グリッド位置ごとにアンカーなしの点ベース事前分布を1つ使い、特徴レベル間で畳み込みを共有するヘッドから予測する1段検出器です。LibreYOLOは物体検出とRTMDet-InsのインスタンスセグメンテーションでRTMDetをサポートします。
keywords:
  - RTMDet
  - 物体検出
  - インスタンスセグメンテーション
  - RTMDet-Ins
  - アンカーフリー 検出
  - mmdetection
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRTMDets.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRTMDets.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: インスタンスセグメンテーション
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファイル名の -seg サフィックスで RTMDet-Ins マスクヘッドを選択するため
        # task 引数は不要
        model = LibreYOLO("LibreRTMDets-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRTMDets.pt data=my-dataset.yaml
    - label: インスタンスセグメンテーション
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # マスク
        print(metrics["metrics/mAP50-95(B)"])   # ボックス
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=300, imgsz=640, batch=16, lr0=0.004,
        )
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreRTMDets.pt data=my-dataset.yaml imgsz=640
        epochs=300 batch=16 lr0=0.004
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreRTMDets.pt format=onnx imgsz=640

        libreyolo export model=LibreRTMDets.pt format=tensorrt imgsz=640
        half=True
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリーはファイルサフィックスで振り分けるためエクスポート成果物も
        # 任意のチェックポイントと同様に読み込まれ同じ Results オブジェクトを返す
        model = LibreYOLO("LibreRTMDets.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 2f5033bdc1c3c931
---

## インストール

RTMDetには基本パッケージ以外の追加パッケージは不要です。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

返される`Results`オブジェクトはすべてのファミリーに共通するため、別の検出器への切り替えは1行の変更で済みます。ファイル名に`-seg`があるとRTMDet-Insタスクとして自動的に解決され、`result.masks`にはボックスとともにインスタンスマスクが格納されます。`conf`は信頼度のしきい値、`iou`はNMSのしきい値を設定します。ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## バリアント

`t`から`x`までの5つのサイズが、共通の入力解像度で1つのアーキテクチャを共有します。このファミリーにはベンチマーク表がありません。下の表にあるチェックポイントのファイルサイズで各サイズを比較してください。

## 学習

<code-tabs name="train" />

検出は`train()`で学習します。QualityFocalLoss、GIoU、DynamicSoftLabelAssignerの各コンポーネントはアップストリームのmmdetectionから移植されています。順伝播とONNXエクスポートはビット単位で同等で、後処理はval2017のサブセット上でmmdetの出力と0.001 mAP以内で一致します。

`train()`自体のdocstringによると、小規模データセットでのファインチューニング収束、ゼロからの学習による論文との一致、マルチGPUの動作、キャッシュされたMosaicとMixUpのスループット、アップストリームの厳密な2段パイプライン切り替え、正規化パラメータとバイアスパラメータの減衰をゼロにするパラメータ別weight decay上書きは未検証です。

RTMDet-Insには学習経路がありません。`-seg`チェックポイントで、または`task="segment"`を指定して`train()`を呼び出すと`NotImplementedError`が発生します。インスタンスセグメンテーションは推論と検証だけをサポートします。

`train()`は`pretrained`引数も受け付けますが、その値がメソッド内で読み取られることはありません。学習は常にモデルの構築時に使った重みから続行されるため、`pretrained=False`を指定してもネットワークは再初期化されません。

そのほかを変更しなければ、トレーナーはAdamW、`lr0=0.004`、`weight_decay=0.05`で300エポック実行します。コサインスケジュール上で1エポックのウォームアップを行い、最後の20エポックではMosaicとMixUpを無効にします。

データセット、データ拡張、マルチGPU、ロガーについては[学習](/docs/train)を参照してください。

## 検証

`val()`は、学習に使用した形式の任意のデータセットで測定した適合率、再現率、mAP 50、mAP 50-95を含む`metrics/`キーの辞書を返します。

<code-tabs name="val" />

`-seg`チェックポイントを対象にすると、通常の`metrics/mAP50-95`キーにはマスクのスコアが格納されます。同じ実行で`(B)`のボックスと`(M)`のマスクも報告されるため、1回の処理で両方を取得できます。

## エクスポート

<export-matrix />

検出はほとんどの形式にエクスポートできますが、インスタンスセグメンテーションは現在どの形式にもエクスポートできません。上のマトリクスはこの違いを反映しています。エクスポートした検出成果物は、ファイルサフィックスに基づいて`LibreYOLO()`から再度読み込めます。そのため、`.onnx`または`.engine`ファイルはチェックポイントと同様に動作し、同じ`Results`を返します。LibreYOLOをインストールせず、単独のランタイムでグラフを実行することもサポートされますが、その場合は前処理と後処理を自分で実装する必要があります。

<code-tabs name="export" />

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box></provenance-box>

## 引用

<citation-block />

