---
title: YOLOv9
families:
  - yolo9
seo_title: YOLOv9：MITライセンスで推論、学習、エクスポート
description: >-
  LibreYOLOでYOLOv9を実行します。NMS不要のエンドツーエンドヘッドとstride
  4の小物体用ヘッドも含まれます。インストール、推論、学習、検証、エクスポートに対応します。
lead: >-
  1回の処理で密なボックスグリッドにスコアを付け、NMSで重複を除く1段畳み込み検出器です。LibreYOLOには3つのバリアントがあり、そのうち1つはNMSステップを持ちません。
keywords:
  - YOLOv9
  - YOLO9
  - 物体検出
  - NMS不要 物体検出
  - エンドツーエンド 物体検出
  - 小物体検出
  - programmable gradient information
  - GELAN
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: NMSなし
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 同じ呼び出しで異なるチェックポイントを使用 エンドツーエンドヘッドが
        # 高スコアの予測を返すため NMS は実行されず iou は無視される
        model = LibreYOLO("LibreYOLO9E2Es.pt")
        result = model(SAMPLE_IMAGE, conf=0.25, max_det=300)

        print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 imgsz=640 batch=16
    - label: 小物体
      language: python
      code: >
        from libreyolo import LibreYOLO9P2


        # stride 4 バリアントには固有の COCO チェックポイントがないため

        # 基本検出モデルを指定 バックボーンとネックはそのまま読み込み

        # stride 4 ヘッドタワーはランダム初期化で開始

        model = LibreYOLO9P2(None, size="s")

        model.train(data="my-dataset.yaml", epochs=100,
        pretrained="LibreYOLO9s.pt")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=my-dataset.yaml
    - label: COCOで検証
      language: bash
      code: |
        # 同梱の COCO yaml はダウンロードスクリプトを内包するため
        # データセットがローカルにない場合は明示的な許可が必要
        libreyolo val model=LibreYOLO9c.pt data=coco.yaml imgsz=640 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx imgsz=640
    - label: グラフ内でNMSを実行
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx nms=True \
          conf=0.25 iou=0.45 max_det=300
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリーはファイルサフィックスで振り分けるためエクスポート成果物も
        # 任意のチェックポイントと同様に読み込まれ同じ Results オブジェクトを返す
        model = LibreYOLO("LibreYOLO9s.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: eaa6023a4a0b9e71
---

## インストール

YOLOv9には基本パッケージ以外の追加パッケージは不要です。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

返される`Results`オブジェクトはすべてのファミリーに共通するため、別の検出器への切り替えは1行の変更で済みます。基本モデルとstride 4モデルでは、`conf`が信頼度のしきい値、`iou`がNMSのしきい値を設定します。エンドツーエンドモデルはNMSを実行せず`iou`を無視するため、`conf`と`max_det`で出力が決まります。ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## バリアント

3つのバリアントがバックボーンを共有します。3つすべてが検出専用で、同じ引数を受け取ります。

基本モデルは3つの特徴スケールで予測し、NMSで重複するボックスを除去します。

エンドツーエンドモデルはそのヘッドを維持し、隣に1対1マッチングの分岐を追加します。推論では1対1の分岐だけを読み取り、高スコアの予測を取得するため、NMSは実行されません。デプロイ先のランタイムにNMS演算子がない場合に選択してください。

stride 4モデルはバックボーンのさらに上位の1レベルを表に出し、そこまでネックを拡張し、3つではなく4つのスケールで予測します。追加のスケールは少数のピクセルしか占めない物体向けです。公開されている唯一のチェックポイントは航空画像で学習されています。基本検出チェックポイントを転移できます。バックボーンとネックは変更せずに読み込まれ、3つの学習済みヘッドタワーは1スロット上にずれ、stride 4タワーはランダム初期化から開始します。

<benchmark-table task="detect" />

<va-embed />

## 学習

<code-tabs name="train" />

`pretrained`によって実行の開始点が決まります。同じモデルとサイズの公開済みチェックポイントを読み込むには`True`を渡し、それ以外の場合は名前またはパスを渡してください。形状が一致しないテンソルは拒否されずにスキップされ、読み込んだ数が実行ログに記録されます。そのため、異なるクラス数で学習したチェックポイントも開始点として使えます。

stride 4モデルには固有の公開済みCOCOチェックポイントがないため、そこで`True`を指定すると存在しないファイルに解決され、ダウンロードに失敗します。代わりに基本検出チェックポイントを指定してください。

データセット、データ拡張、マルチGPU、ロガーについては[学習](/docs/train)を参照してください。

## 検証

`val()`は、学習に使用した形式の任意のデータセットで測定した適合率、再現率、mAP 50、mAP 50-95を含む`metrics/`キーの辞書を返します。

<code-tabs name="val" />

## エクスポート

<export-matrix />

チェックマークは3つすべてのバリアントに適用されます。差がある場合、マトリクスには3つのうち最も弱いものが示されます。

エクスポートした成果物は、ファイルサフィックスに基づいて`LibreYOLO()`から再度読み込めます。そのため、`.onnx`または`.engine`ファイルはチェックポイントと同様に動作し、同じ`Results`を返します。LibreYOLOをインストールせず、単独のランタイムでグラフを実行することもサポートされますが、その場合は前処理と後処理を自分で実装する必要があります。

基本検出モデルでは、その後処理部分をグラフ内に移動できます。ONNXエクスポートで`nms=True`を指定すると抑制がモデル内部に組み込まれ、最初の出力は固定された`(1, max_det, 6)`テンソルになります。各行は`x1, y1, x2, y2, score, class`で、検出数を超える部分はゼロでパディングされます。このグラフはバッチ1で、動的軸を持ちません。エンドツーエンドモデルとstride 4モデルはこのフラグを受け付けません。

各形式では異なる追加パッケージをインストールし、それぞれ固有の引数をいくつか受け取ります。どちらも該当する形式のページに記載されています。

<code-tabs name="export" />

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box>

ここにある1つのチェックポイントはMITではありません。VisDrone2019-DETで学習したstride 4モデルは、そのデータセットのCC BY-NC-SA 3.0条件を継承します。非商用利用に限定され、派生物には継承条件が適用され、このファミリーの残りが提供される寛容なライセンスの範囲外です。COCOクラスではなくVisDroneの航空クラスを予測します。ライブラリはファイルをダウンロードする前にこれらすべてを表示します。

</provenance-box>

## 引用

<citation-block />

