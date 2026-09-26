---
title: 物体検出
seo_title: LibreYOLOによる物体検出
description: LibreYOLOで軸平行のボックスとして物体を検出します。対応ファミリー、ラベル形式、推論、学習、検証、エクスポートの呼び出しを解説します。
lead: 物体検出は画像内の各物体インスタンスの位置を特定し、それぞれに軸平行の長方形、クラスラベル、スコアを返します。タスクキーはdetectです。
keywords:
  - Python 物体検出
  - 画像内 物体検出
  - バウンディングボックス 検出
  - MIT 物体検出ライブラリ
  - YOLO 代替
  - 物体検出モデル 学習
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(result.names[int(box.cls)], float(box.conf), box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9t.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 別ファミリーでも同じ呼び出し
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリはチェックポイントに応じて振り分け 全検出器が同じ
        # Resultsオブジェクトを返すためファミリーの切り替えは1行の変更で済む
        model = LibreYOLO("LibreDFINEn.pt")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy.shape)
    - label: 動画とストリーム
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # ライブラリが受け付ける任意のソース ファイル フォルダー URL
        # Webカメラのインデックス RTSPストリーム .streamsリスト
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # coco128.yamlは初回使用時に128枚のサンプルをダウンロードする
        # 実際の学習ではdataに独自データセットのYAMLを指定する
        model.train(data="coco128.yaml", epochs=50, imgsz=640, batch=8)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9t.pt data=coco128.yaml \
          epochs=50 imgsz=640 batch=8
    - label: マルチGPU
      language: bash
      code: |
        libreyolo train model=LibreYOLO9t.pt data=coco128.yaml \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # val()はオブジェクトではなく通常のdictを返す
        metrics = model.val(data="coco128.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"], metrics["metrics/mAP75"])
        print(metrics["metrics/AR100"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO9t.pt data=coco128.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO9t.pt format=onnx imgsz=640
    - label: エクスポートしたファイルを使用
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリはファイルの拡張子に応じて振り分けるためエクスポートした
        # アーティファクトもチェックポイントと同様に読み込まれ同じResultsオブジェクトを返す
        model = LibreYOLO("LibreYOLO9t.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: c735b6e3de78dd2b
---

## 定義

物体検出は、各物体がどこにあり、何であるかを答えます。1枚の画像を入力すると、インスタンスごとに1行を出力します。各行には長方形を表す4つの数値、クラスインデックス、スコアがあります。ピクセル単位の形状、向き、部位は含まれません。この点が[インスタンスセグメンテーション](/docs/tasks/instance-segmentation)、[回転ボックス](/docs/tasks/oriented-detection)、[姿勢推定](/docs/tasks/pose-estimation)との違いです。

`detect`が標準タスクキーであり、デフォルトです。ファイル名にタスク接尾辞がないチェックポイントは検出器として読み込まれます。

`predict()`は`result.boxes`を設定します。`.xyxy`は元画像のキャンバス上のピクセル単位の角座標、`.conf`はスコア、`.cls`は`result.names`を参照するクラスインデックスです。`.xywh`、`.xyxyn`、`.xywhn`は同じ行から導出したビューで、トラッカーを接続すると`.id`に追跡IDが格納されます。`Boxes`オブジェクトを反復処理すると1行のスライスが得られるため、検出ごとに`box.cls`、`box.conf`、`box.xyxy`を使用できます。

## モデル

学習と推論の両方に対応するファミリーは12個です。[YOLOv9](/docs/models/yolov9)、[RF-DETR](/docs/models/rf-detr)、[EdgeCrafter](/docs/models/edgecrafter)、[RT-DETR](/docs/models/rt-detr)、[D-FINE](/docs/models/d-fine)、[DEIM](/docs/models/deim)、[Dome-DETR](/docs/models/dome-detr)、[YOLO-NAS](/docs/models/yolo-nas)、[YOLOX](/docs/models/yolox)、[YOLOv7](/docs/models/yolov7)、[RTMDet](/docs/models/rtmdet)、[PicoDet](/docs/models/picodet)です。YOLOv9とRF-DETRは2つの主力ファミリーで、新機能は最初にこれらへ追加されます。RF-DETRには専用の追加パッケージ`pip install "libreyolo[rfdetr]"`が必要です。それ以外は基本パッケージで実行できます。

さらに11個のファミリーが推論、検証、エクスポートに対応しますが、`train()`は`NotImplementedError`を送出します。[LW-DETR](/docs/models/lw-detr)、[DETR](/docs/models/detr)、[Deformable DETR](/docs/models/deformable-detr)、[DINO-DETR](/docs/models/dino-detr)、[Faster R-CNN](/docs/models/faster-rcnn)、[Mask R-CNN](/docs/models/mask-rcnn)、[FCOS](/docs/models/fcos)、[RetinaNet](/docs/models/retinanet)、[SSD](/docs/models/ssd)、[CenterNet](/docs/models/centernet)、[EfficientDet](/docs/models/efficientdet)です。

Darknet系統の[YOLOv1](/docs/models/yolov1)、[YOLOv2](/docs/models/yolov2)、[YOLOv3](/docs/models/yolov3)、[YOLOv4](/docs/models/yolov4)は、固定された展示物として維持されています。推論、検証、エクスポートは機能しますが、学習はできません。

別のグループはチェックポイントではなく実行時にクラスリストを受け取るため、学習時に見たことのない名前も検出できます。[Grounding DINO](/docs/models/grounding-dino)、[OWLv2](/docs/models/owlv2)、[OMDet-Turbo](/docs/models/omdet-turbo)、[OV-DEIM](/docs/models/ov-deim)に加え、Vision-Languageファミリーの[Florence-2](/docs/models/florence-2)、[Kosmos-2](/docs/models/kosmos-2)、[Qwen3-VL](/docs/models/qwen3-vl)、[SmolVLM2](/docs/models/smolvlm2)、[InternVL3](/docs/models/internvl3)、[LFM2-VL](/docs/models/lfm2-vl)、[LocateAnything](/docs/models/locate-anything)、[SenseNova-Vision](/docs/models/sensenova-vision)、[LibreMODUS](/docs/models/libremodus)です。これらは固有のファクトリと追加パッケージを通して読み込まれます。正確な呼び出しは各モデルページに記載されています。

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

`conf`は信頼度のしきい値、`max_det`は行数の上限です。`iou`はNMSのしきい値なので、NMSを実行するファミリーだけに影響します。RF-DETRとエンドツーエンドYOLOv9ヘッドは固定された予測集合をデコードするため、これを無視します。入力ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## データセット形式

画像ごとに1つの`.txt`ラベルファイルを使用します。画像パス内の`images`を`labels`へ置き換え、拡張子を変更して検索します。

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

各行は正確に5つのフィールドで、クラスインデックスの後に、正規化済みの中心座標とサイズを持つボックスを記述します。

```text
<class_id> <cx> <cy> <w> <h>
```

座標は元画像の幅と高さに対する`[0, 1]`範囲の浮動小数点数です。`w`と`h`は正でなければなりません。ラベルファイルが存在しないか空の場合、その画像に物体がないことを意味します。行に信頼度や追跡IDは含まれません。

YAMLでは分割とクラスを指定します。

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: person
  1: bicycle
```

`train`と`val`には画像ディレクトリ、画像一覧の`.txt`ファイル、またはそのどちらかのリストを指定できます。`nc`は任意で、指定した場合は`names`と一致する必要があります。ネイティブのCOCO JSONも使用できます。分割名とJSONファイルの`annotations`マッピングを追加すると、分割パスが画像ルートになります。`names`がある場合はラベルIDを定義するため、JSONのカテゴリー名と一致しなければなりません。

## 学習

<code-tabs name="train" />

最初に調整する引数は`epochs`、`imgsz`、`batch`、`lr0`です。ファミリー間で流用できないのは`lr0`です。畳み込み検出器が許容する学習率でもTransformer検出器では発散する可能性があるため、別ファミリーの例ではなくモデルページの値を使用してください。ファミリーが引数を完全に無視する場合もあり、各ページにその対象が記載されています。データセット、データ拡張、マルチGPU、ロガーについては[学習](/docs/train)を参照してください。

## 検証

`val()`は、データセットYAMLの`val`で指定した分割に対してCOCO評価で計算した`metrics/`キーを持つ通常の辞書を返します。

<code-tabs name="val" />

`metrics/mAP50-95`はIoUしきい値0.50から0.95までで平均した平均適合率で、主要な数値です。`metrics/mAP50`と`metrics/mAP75`は単一しきい値版です。`metrics/mAP_small`、`metrics/mAP_medium`、`metrics/mAP_large`は同じ平均を物体面積別に分けます。`metrics/AR1`、`metrics/AR10`、`metrics/AR100`、`metrics/AR_small`、`metrics/AR_medium`、`metrics/AR_large`は対応する平均再現率です。`metrics/AR_max_det`と`metrics/max_det`には実行時に使った検出上限が記録されます。

このタスクの`metrics/precision`と`metrics/recall`は注意して解釈してください。後方互換性のために維持された別名であり、ある運用点の値ではありません。`metrics/precision`は`metrics/mAP50-95`と同じ値を、`metrics/recall`は`metrics/AR100`と同じ値を保持します。適合率と再現率のペアとして描画すると同じ数値を二重に報告することになります。モデルがマスクも予測する場合でも検出キーを同じように読めるよう、4つのキーはボックスを表す`(B)`接尾辞付きでも繰り返されます。`metrics/mAP50-95(B)`、`metrics/mAP50(B)`、`metrics/precision(B)`、`metrics/recall(B)`です。

## エクスポート

<code-tabs name="export" />

エクスポートしたアーティファクトはファイルの拡張子に基づいて`LibreYOLO()`で再読み込みできるため、`.onnx`や`.engine`ファイルもチェックポイントと同様に動作し、同じ`Results`を返します。対応形式はファミリーごとに異なり、各モデルページのマトリックスは手入力ではなく検証済みの集合から生成されます。形式、追加パッケージ、制約については[エクスポートとデプロイ](/docs/export)を参照してください。
