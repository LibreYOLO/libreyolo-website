---
title: EfficientDet
families:
  - efficientdet
seo_title: EfficientDet：LibreYOLOで物体検出
description: >-
  LibreYOLOでEfficientDet
  D0〜D4を実行します。Apache-2.0のBiFPN検出器で、推論、検証、ONNX・TensorRT・OpenVINOへのエクスポートを行います。
lead: >-
  EfficientDetはEfficientNetバックボーンと、反復するbi-directional feature pyramid
  network（BiFPN）を組み合わせ、5種類のサイズにわたって深さ、幅、解像度を同時にスケーリングします。LibreYOLOは推論専用の検出器として提供します。
keywords:
  - EfficientDet 使い方
  - BiFPN
  - EfficientNet
  - 物体検出
  - compound scaling
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEfficientDetd0.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreEfficientDetd0.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEfficientDetd0.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEfficientDetd0.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEfficientDetd0.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreEfficientDetd0.pt format=onnx
        libreyolo export model=LibreEfficientDetd0.pt format=tensorrt half=True
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO

        # ファクトリがファイル接尾辞で振り分けるため、エクスポートした成果物も
        # 通常のチェックポイントと同様に読み込まれ、同じResultsオブジェクトを返す
        model = LibreYOLO("LibreEfficientDetd0.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 12c61fb0035437ce
---

## インストール

EfficientDetに任意の追加パッケージは必要ありません。インポートするものはすべて基本
インストールに含まれます。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

返される`Results`オブジェクトはすべてのファミリーで共通のため、別の検出器への切り替えは
1行の変更だけで済みます。EfficientDetはアンカーベースの候補をデコードした後、クラス単位の
non-maximum suppressionを実行します。そのため、`conf`、`iou`、`max_det`はすべて実際に
効果があります。入力ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## バリアント

サイズはD0〜D4の5種類です。サイズを1段階上げるたびに、より大きなEfficientNetバックボーンと、
より深く幅広いBiFPNおよび深い推論ヘッドを組み合わせます。そのため、論文のcompound-scaling
規則に従ってパラメータ数と計算量が同時に増えます。

## 検証

`val()`は、学習に使った形式の任意のデータセットに対して測定した適合率、再現率、mAP 50、
mAP 50-95を含む`metrics/`キーの辞書を返します。

<code-tabs name="val" />

## エクスポート

<export-matrix />

エクスポートした成果物はファイル接尾辞に基づいて`LibreYOLO()`から再読み込みされます。そのため、
`.onnx`または`.engine`ファイルはチェックポイントと同様に動作し、同じ`Results`を返します。

<code-tabs name="export" />

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box>

LibreYOLOのD0〜D4チェックポイントは、Apache-2.0のrwightman/efficientdet-pytorch
プロジェクトを通じて変換されています。このプロジェクト自体は、google/automlの公式TensorFlow
学習済み重みを学習済みテンソルを変更せずにミラーします。LGPLライセンスの
zylo117/Yet-Another-EfficientDet-Pytorchプロジェクトのソースは参照も使用もしていません。

</provenance-box>

