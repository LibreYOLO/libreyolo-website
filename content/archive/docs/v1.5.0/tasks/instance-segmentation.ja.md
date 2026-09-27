---
title: インスタンスセグメンテーション
seo_title: LibreYOLOによるインスタンスセグメンテーション
description: LibreYOLOで個々の物体をセグメンテーションします。対応ファミリー、ポリゴンラベル形式、推論、学習、検証、エクスポートの呼び出しを解説します。
lead: >-
  インスタンスセグメンテーションは各物体インスタンスの位置を特定し、検出器が返すボックス、クラス、スコアに加えて、物体ごとにピクセル単位のマスクを返します。タスクキーはsegmentです。
keywords:
  - Python インスタンスセグメンテーション
  - 物体マスク 予測
  - セグメンテーションモデル 学習
  - ポリゴンラベル
  - MIT セグメンテーションライブラリ
  - mask mAP
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファイル名の-seg接尾辞がマスクヘッドを選択するためtask引数は不要
        model = LibreYOLO("LibreDFINEn-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)   # (N, H, W) 検出ごとに1つのマスク
        print(result.boxes.xyxy.shape)   # (N, 4) 同じN行
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDFINEn-seg.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: マスクの輪郭
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDFINEn-seg.pt")
        result = model(SAMPLE_IMAGE)

        # .xyはピクセル単位の(P, 2)輪郭リスト .xynは正規化した同じ輪郭
        for name, contour in zip(result.boxes.cls, result.masks.xy):
            print(result.names[int(name)], contour.shape)
    - label: 別ファミリーでも同じ呼び出し
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRTMDets-seg.pt")
        result = model(SAMPLE_IMAGE)

        print(result.masks.data.shape)
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # マスクヘッドを含む公開済みセグメンテーション重みから続ける

        # dataはポリゴンを持つラベルのデータセットを指す必要がある

        model = LibreYOLO("LibreDFINEn-seg.pt")

        model.train(data="my-dataset.yaml", epochs=50, imgsz=640, batch=8,
        lr0=2e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDFINEn-seg.pt data=my-dataset.yaml \
          epochs=50 imgsz=640 batch=8 lr0=2e-4
    - label: 検出用重みから開始
      language: bash
      code: |
        # 検出用重みにはマスクヘッドがないため明示的な転移となり
        # ヘッドは未学習状態から始まる task=segmentの指定がこれを許可する
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          task=segment epochs=50 imgsz=640
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])       # マスク
        print(metrics["metrics/mAP50-95(M)"])    # 明示的なマスク
        print(metrics["metrics/mAP50-95(B)"])    # ボックス
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDFINEn-seg.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn-seg.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDFINEn-seg.pt format=onnx imgsz=640
    - label: エクスポートしたファイルを使用
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリはファイルの拡張子に応じて振り分けるためエクスポートした
        # アーティファクトもチェックポイントと同様に読み込まれ同じResultsオブジェクトを返す
        model = LibreYOLO("LibreDFINEn-seg.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.masks.data.shape)
source_hash: 33e331eac0f9b0af
---

## 定義

インスタンスセグメンテーションは検出に形状を加えたものです。各物体インスタンスには引き続きボックス、クラス、スコアがあり、さらにその物体に属するピクセルを覆う二値マスクが追加されます。マスクは重なることができ、どの物体にも属さないピクセルは未割り当てのままです。この点が[セマンティックセグメンテーション](/docs/tasks/semantic-segmentation)と[パノプティックセグメンテーション](/docs/tasks/panoptic-segmentation)との違いです。

`segment`が標準タスクキーで、チェックポイントのファイル名にある`-seg`接尾辞が選択します。そのため、公開済みの重みを読み込むときに`task=`は不要です。

`predict()`は`result.boxes`とともに`result.masks`を設定します。`.data`は元画像のキャンバス上の`(N, H, W)`スタックで、ボックスと行が対応します。そのためマスク`i`はボックス`i`に属します。`.xy`は各マスクを最大の外輪郭であるピクセル単位の`(P, 2)`配列へ変換し、`.xyn`は同じ輪郭を正規化して返します。

## モデル

マスクの学習と推論の両方に対応するファミリーは4つです。[RF-DETR](/docs/models/rf-detr)、[EdgeCrafter](/docs/models/edgecrafter)、[D-FINE](/docs/models/d-fine)、[RTMDet](/docs/models/rtmdet)です。RF-DETRには専用の追加パッケージ`pip install "libreyolo[rfdetr]"`が必要で、他の3つは基本パッケージで実行できます。

[Mask R-CNN](/docs/models/mask-rcnn)はマスクの推論、検証、エクスポートに対応しますが、`train()`は`NotImplementedError`を送出します。

[EoMT](/docs/models/eomt)はマスクの推論と検証に対応しますが、学習はできません。エクスポートの範囲はさらに狭く、`export()`が受け付けるのはsemanticタスクだけです。`segment`と`panoptic`に必要なクエリマスクのランタイム契約が定義されていないため、これらでは`NotImplementedError`を送出します。EoMTのインスタンスマスクはエクスポート済みグラフではなくPythonで使用してください。

別のグループはクラスリストではなくプロンプトからセグメンテーションします。クリック、ボックス、フレーズで物体を選び、モデルがそのマスクを返します。[SAM](/docs/models/sam)、[SAM 2](/docs/models/sam-2)、[SAM 3](/docs/models/sam-3)、[MobileSAM](/docs/models/mobilesam)、[EdgeTAM](/docs/models/edgetam)、[PicoSAM3](/docs/models/picosam3)がこの方法を使います。[SenseNova-Vision](/docs/models/sensenova-vision)も同様ですが、そのセグメンテーションは参照式で、1つの物体を指すフレーズを受け取ります。これらは固有のファクトリと追加パッケージを通して読み込まれ、正確な呼び出しは各モデルページに記載されています。

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

`conf`と`max_det`は検出と同じように出力を絞り込み、マスクも対応するボックスとともにフィルタリングされます。入力ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## データセット形式

配置は検出と同じです。画像ごとに1つの`.txt`ラベルファイルを使用し、画像パス内の`images`を`labels`へ置き換えて拡張子を変更し、ファイルを検索します。

```text
dataset/
  data.yaml
  images/
    train/000001.jpg
    val/000101.jpg
  labels/
    train/000001.txt
    val/000101.txt
```

異なるのは行の内容です。セグメントはクラスインデックスの後に平坦化したポリゴンを記述します。

```text
<class_id> <x1> <y1> ... <xN> <yN>
```

点は3つ以上必要なので、クラスインデックス後の座標数は6以上の偶数となり、ポリゴンは退化していない必要があります。座標は元画像の幅と高さに対する`[0, 1]`範囲の浮動小数点数です。5フィールドの検出行もセグメンテーションデータセット内で受け付け、長方形のセグメントとして読み取ります。そのため、ボックスだけのデータセットも変換処理なしで読み込めます。

YAMLは検出用YAMLと同じです。

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: person
  1: bicycle
```

ネイティブのCOCO JSONも使用できます。分割名とJSONファイルの`annotations`マッピングを追加すると、分割パスが画像ルートになります。

## 学習

<code-tabs name="train" />

学習はデフォルトで公開済みの`-seg`チェックポイントから続行します。検出用重みから始めることもできますが、意図的な転移として扱われます。その重みにはマスクヘッドがないため未学習状態から始まり、`task=segment`を渡すことで入れ替えを許可します。データセット、データ拡張、マルチGPU、ロガーについては[学習](/docs/train)を参照してください。

## 検証

`val()`は`metrics/`キーを持つ通常の辞書を返します。ボックスとマスクはどちらもCOCO評価で別々に採点され、マスクの数値が主要な結果です。

<code-tabs name="val" />

接尾辞なしのキーにはマスクの結果が格納されます。`metrics/mAP50-95`、`metrics/mAP50`、`metrics/mAP75`、物体面積別の`metrics/mAP_small`、`metrics/mAP_medium`、`metrics/mAP_large`、平均再現率の`metrics/AR1`、`metrics/AR10`、`metrics/AR100`、`metrics/AR_small`、`metrics/AR_medium`、`metrics/AR_large`です。`metrics/AR_max_det`と`metrics/max_det`には実行時に使った検出上限が記録されます。

どの数値をファミリーが主要としたかに依存せず比較できるよう、4つの数値はマスクを表す`(M)`とボックスを表す`(B)`の明示的な接尾辞付きでも公開されます。`metrics/mAP50-95(M)`と`metrics/mAP50-95(B)`、`metrics/mAP50(M)`と`metrics/mAP50(B)`、`metrics/precision(M)`と`metrics/precision(B)`、`metrics/recall(M)`と`metrics/recall(B)`です。このタスクには接尾辞なしの`metrics/precision`や`metrics/recall`はありません。

適合率と再現率のキーは注意して解釈してください。後方互換性のために維持された別名であり、ある運用点の値ではありません。`metrics/precision(M)`は`metrics/mAP50-95(M)`と同じ値を、`metrics/recall(M)`は検出数100でのマスク平均再現率と同じ値を保持します。`(B)`もボックスについて同様です。この2つをペアとして描画すると同じ数値を二重に報告することになります。

## エクスポート

<code-tabs name="export" />

エクスポートしたアーティファクトはファイルの拡張子に基づいて`LibreYOLO()`で再読み込みできるため、`.onnx`や`.engine`ファイルもチェックポイントと同様に動作し、同じ`Results`を返します。同じファミリーでもセグメンテーションの対応形式は検出より狭くなります。各モデルページのマトリックスは検証済みの集合から生成され、利用できない理由も示します。形式、追加パッケージ、制約については[エクスポートとデプロイ](/docs/export)を参照してください。
