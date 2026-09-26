---
title: RF-DETR
families:
  - rfdetr
seo_title: RF-DETR：MITライセンスで学習、ファインチューニング、エクスポート
description: >-
  LibreYOLOでRF-DETRを使い、検出、インスタンスセグメンテーション、姿勢推定、回転バウンディングボックスを扱います。すべてMITライセンスで、インストール、推論、学習、検証、エクスポートに対応します。
lead: >-
  密なグリッドではなく固定された物体集合を予測するため、推論時にNMSを必要としない検出Transformerです。LibreYOLOは4つのタスクでRF-DETRをサポートします。
keywords:
  - RF-DETR
  - real-time detection transformer
  - DETR
  - 物体検出
  - インスタンスセグメンテーション
  - 姿勢推定
  - 回転バウンディングボックス
last_verified: 1.5.0
hero:
  src: /showcase/parkour-detection.mp4
  poster: /showcase/parkour-detection-poster.jpg
  caption: LibreRFDETRs、512 pxでの動画検出。
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRFDETRs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRFDETRs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: 動画
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")

        # ライブラリが受け付ける任意のソース: ファイル フォルダー URL Webカメラ番号
        # RTSP ストリーム または .streams リスト
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreRFDETRs.pt")

        model.train(data="my-dataset.yaml", epochs=50, imgsz=512, batch=8,
        lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs.pt data=my-dataset.yaml \
          epochs=50 imgsz=512 batch=8 lr0=1e-4
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.train(data="my-dataset.yaml", epochs=50, lora=True)
    - label: マルチGPU
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs.pt data=my-dataset.yaml \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")

        # val() はオブジェクトではなく通常の辞書を返す
        metrics = model.val(data="my-dataset.yaml", imgsz=512)

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRFDETRs.pt data=my-dataset.yaml imgsz=512
    - label: COCOで検証
      language: bash
      code: |
        # 同梱の COCO yaml はダウンロードスクリプトを内包するため
        # データセットがローカルにない場合は明示的な許可が必要
        libreyolo val model=LibreRFDETRn.pt data=coco.yaml imgsz=384 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.export(format="onnx", imgsz=512)
        model.export(format="tensorrt", imgsz=512, half=True)

        # すべての形式で受け付ける引数:
        #
        #   format    "onnx" | "torchscript" | "executorch" | "tensorrt"
        #             | "openvino" | "paddle" | "mnn" | "rknn" | "ncnn"
        #             | "tflite" | "coreml" | "coreai".
        #             "engine" は tensorrt の別名で "litert" は tflite の別名
        #   imgsz     int または (height, width) デフォルトはチェックポイントの
        #             ネイティブ解像度
        #   batch     int デフォルト 1
        #   half      bool FP16 でエクスポート デフォルト False
        #   int8      bool INT8 でエクスポート デフォルト False `data` が必要
        #   data      int8 の較正に使うデータセット YAML のパス
        #   fraction  float 使用する較正セットの割合 デフォルト 1.0
        #   dynamic   bool 動的軸 デフォルト True
        #   simplify  bool ONNX グラフ簡略化を実行 デフォルト True
        #   opset     int ONNX opset 未指定時はファミリーごとに選択
        #   device    str トレースに使うデバイス デフォルトはモデルのデバイス
        #   output_path  str デフォルトはチェックポイント由来の名前
        #   verbose   bool デフォルト False
        #   allow_download_scripts  bool デフォルト False ダウンロードが必要な
        #             データセット YAML 内の Python 実行を許可
        #
        # 一部の形式には RKNN ターゲットプラットフォームなど固有の追加引数がある
        # それらは各形式のページに記載
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreRFDETRs.pt format=onnx imgsz=512

        libreyolo export model=LibreRFDETRs.pt format=tensorrt imgsz=512
        half=True
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO

        # ファクトリーはファイルサフィックスで振り分けるためエクスポート成果物も
        # 任意のチェックポイントと同様に読み込まれ同じ Results オブジェクトを返す
        model = LibreYOLO("LibreRFDETRs.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
    - label: LibreYOLOを使わない場合
      language: python
      code: >
        import numpy as np

        import onnxruntime as ort


        # グラフを直接実行する場合は前処理と後処理を自分で実装

        # 接続前にシグネチャを確認

        session = ort.InferenceSession("LibreRFDETRs.onnx")

        name = session.get_inputs()[0].name

        outputs = session.run(None, {name: np.zeros((1, 3, 512, 512),
        dtype=np.float32)})


        for meta, array in zip(session.get_outputs(), outputs):
            print(meta.name, array.shape)
source_hash: 8c464aa759131694
---

## インストール

RF-DETRには専用の追加パッケージが必要で、バックボーン用の`transformers`がインストールされます。

```bash
pip install "libreyolo[rfdetr]"
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

返される`Results`オブジェクトはすべてのファミリーに共通するため、別の検出器への切り替えは1行の変更で済みます。`conf`と`max_det`はクエリ選択をフィルタリングします。調整するNMSステップはありません。ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## バリアント

4つのサイズと、1つのアーキテクチャを共有する4つのタスクがあります。セグメンテーション、姿勢推定、回転バウンディングボックスは異なるヘッドで検出デコーダーを再利用するため、同じ引数を受け取ります。各サイズのパラメータ数は近く、主に入力解像度が異なります。

<benchmark-table task="detect" />

<va-embed />

## 学習

4つすべてのタスクで、学習は公開済みチェックポイントから開始します。RF-DETRのネイティブトレーナーが無視する引数の一覧に`pretrained`が含まれるため、ここで`pretrained=False`を渡してもランダムに初期化されたモデルにはなりません。

<code-tabs name="train" />

ここでは、CNN検出器の場合より2つの引数が重要です。Transformer検出器はYOLOモデルが許容する学習率では発散するため、`lr0`を`1e-4`以下に保ってください。変更する理由がない限り、`imgsz`はチェックポイントのネイティブ解像度のままにしてください。入力はバックボーンのパッチサイズとウィンドウ数の積で割り切れる必要があります。LibreYOLOは実行開始前にこれを確認し、最も近い有効なサイズを示します。

データセット、データ拡張、マルチGPU、ロガーについては[学習](/docs/train)を参照してください。

## 検証

`val()`は、学習に使用した形式の任意のデータセットで測定した適合率、再現率、mAP 50、mAP 50-95を含む`metrics/`キーの辞書を返します。

<code-tabs name="val" />

## エクスポート

<export-matrix />

エクスポートした成果物は、ファイルサフィックスに基づいて`LibreYOLO()`から再度読み込めます。そのため、`.onnx`または`.engine`ファイルはチェックポイントと同様に動作し、同じ`Results`を返します。LibreYOLOをインストールせず、単独のランタイムでグラフを実行することもサポートされますが、その場合は前処理と後処理を自分で実装する必要があります。

<code-tabs name="export" />

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box></provenance-box>

## 引用

<citation-block />

