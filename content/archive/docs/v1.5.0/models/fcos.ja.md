---
title: FCOS
families:
  - fcos
seo_title: LibreYOLOのFCOS：推論、検証、エクスポート
description: >-
  LibreYOLOのFCOSでアンカーフリー物体検出を実行します。BSD-3-Clauseのtorchvision移植版であるResNet-50・FPNについて、インストール、推論、検証、エクスポートを説明します。
lead: >-
  FCOSは定義済みのアンカーボックス群に依存せず、ピクセル単位で物体を検出します。特徴マップ上の各位置でボックスとcenternessスコアを予測します。LibreYOLOはtorchvisionの実装を物体検出向けに移植しています。
keywords:
  - FCOS 使い方
  - アンカーフリー 物体検出
  - 物体検出
  - 1段階 検出器
  - torchvision
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFCOSr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFCOSr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCOSr50.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFCOSr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCOSr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="torchscript", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreFCOSr50.pt format=onnx imgsz=800
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリがファイル接尾辞で振り分けるため、エクスポートした成果物も
        # 通常のチェックポイントと同様に読み込まれ、同じResultsオブジェクトを返す
        model = LibreYOLO("LibreFCOSr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 60bd7b8dfd903a8c
---

## インストール

FCOSに任意の追加パッケージは必要ありません。インポートするものはすべて基本インストールに含まれます。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

返される`Results`オブジェクトはすべてのファミリーで共通のため、別の検出器への切り替えは
1行の変更だけで済みます。しきい値引数を指定せずにモデルを呼び出すと、FCOS独自の公開済み
デフォルト値`conf=0.2`、`iou=0.6`、`max_det=100`が適用されます。上書きするには3つの
いずれかを渡します。FCOSはピクセル単位の推論結果に対する最終NMS処理を維持します。
入力ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## バリアント

サイズはfeature pyramidを持つResNet-50の1種類で、このファミリーが認識する唯一の
バリアントです。

## 検証

`val()`は、学習に使った形式の任意のデータセットに対して測定した適合率、再現率、mAP 50、
mAP 50-95を含む`metrics/`キーの辞書を返します。

<code-tabs name="val" />

## エクスポート

<export-matrix />

FCOSはONNX、TorchScript、OpenVINOへエクスポートできます。FCOSはグラフ実行前に入力ソースの
アスペクト比を維持するため、LibreYOLOは渡された値にかかわらずONNXとOpenVINOの経路で
`dynamic=True`を強制します。これにより、paddingされた入力形状でもグラフが有効に保たれます。
エクスポートした`.onnx`ファイルは、ファイル接尾辞に基づいて`LibreYOLO()`から再読み込みされ、
同じ`Results`を返します。

<code-tabs name="export" />

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box></provenance-box>

## 引用

<citation-block />

