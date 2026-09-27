---
title: YOLOv7
families:
  - yolo7
seo_title: LibreYOLOのYOLOv7：MITで推論、学習、エクスポート
description: LibreYOLOでYOLOv7を物体検出に使います。MITライセンスのコードと重みを使い、インストール、推論、学習、検証、エクスポートを行います。
lead: >-
  YOLOv7はアンカーベースの1段階検出器で、ヘッドは最終畳み込みの前に学習済みimplicit knowledge
  offsetを加えます。LibreYOLOは、公開されている1つのサイズを物体検出でサポートします。
keywords:
  - YOLOv7
  - 物体検出
  - アンカーベース 検出
  - implicit knowledge
  - ImplicitA
  - リアルタイム 物体検出
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO7b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO7b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO7b.pt")

        model.train(data="my-dataset.yaml", epochs=300, imgsz=640, batch=16,
        lr0=0.01)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO7b.pt data=my-dataset.yaml \
          epochs=300 imgsz=640 batch=16 lr0=0.01
    - label: 新しいモデルからウォームスタート
      language: python
      code: |
        from libreyolo import LibreYOLO7

        # pretrained=Trueは、このインスタンスの構築方法に関係なく
        # 公開済みLibreYOLO7b.ptチェックポイントを常に読み込む
        # LibreYOLO()を通さずクラスを直接構築すると
        # 重みをまったく読み込まずに開始
        model = LibreYOLO7(None, size="b")
        model.train(data="my-dataset.yaml", epochs=300, pretrained=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO7b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO7b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO7b.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreYOLO7b.pt format=onnx imgsz=640

        libreyolo export model=LibreYOLO7b.pt format=tensorrt imgsz=640
        half=True
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリーはファイル接尾辞で経路を選ぶため、エクスポート済み成果物も
        # 通常のチェックポイントと同様に読み込まれ、同じResultsオブジェクトを返す
        model = LibreYOLO("LibreYOLO7b.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 361e81de5614a571
---

## インストール

YOLOv7に基本パッケージ以外の追加パッケージは必要ありません。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

返される `Results` オブジェクトはすべてのファミリーが返すものと同じなので、別の検出器への切り替えは1行の変更で済みます。`conf` は信頼度のしきい値、`iou` はアンカーベースのヘッドをdecodeした後に適用するNMSのしきい値を設定します。ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## バリアント

LibreYOLOは `b` という1つのサイズを提供します。アップストリームが公開するYOLOv7モデルは1つだけなので、選択するサイズはありません。

## 学習

<code-tabs name="train" />

このファミリーでは、同名の引数が何もしないほかの一部のファミリーとは異なり、`pretrained` が読み取られます。公開済みの `LibreYOLO7b.pt` チェックポイントからウォームスタートするには `True` を渡し（自動ダウンロードされます）、それ以外の場合はパスまたは名前を渡してください。公開済みチェックポイントは80クラスのCOCO用です。そのため、別のクラス数に再構築済みのモデルでこれを要求すると、まず80クラスに戻して再構築し、読み込んだ後、データセットのクラス数を読み取って対象のヘッド数に合わせる際に、shapeが一致するすべてのテンソルを転送します。`resume=True` と `pretrained` は併用できません。デフォルトの `None` のままにすると、モデル構築時の重みから学習を続行し、何も読み込まれていなければランダム初期化から始めます。

ほかの設定を変更しなければ、trainerは `lr0=0.01`、SGD momentum 0.937、3エポックのウォームアップで300エポック実行します。また、YOLOXと同じSimOTA割り当てと、最後の15エポックでデータ拡張を行わない段階をアンカーベースのヘッド向けに調整して使います。違いは1つあります。YOLOXは最後のエポックでL1ボックス回帰のrefinementを追加しますが、v7では省略します。v7のSimOTA損失には、調整する未処理offsetのL1 branchがないためです。

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

<provenance-box></provenance-box>

## 引用

<citation-block />

