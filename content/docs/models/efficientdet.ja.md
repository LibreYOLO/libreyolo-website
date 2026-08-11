---
title: EfficientDet
families:
  - efficientdet
seo_title: EfficientDet：LibreYOLOでの物体検出
description: >-
  LibreYOLOでEfficientDet
  D0-D4を実行します。Apache-2.0の下で、推論、検証、ONNX、TensorRT、OpenVINOへのエクスポートに対応するBiFPN検出器です。
lead: >-
  EfficientDetはEfficientNetバックボーンと、反復する双方向特徴ピラミッドネットワーク（BiFPN）を組み合わせ、5つのサイズ全体で深さ、幅、解像度を同時にスケーリングします。LibreYOLOは推論専用検出器として提供します。
keywords:
  - EfficientDet
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

        # ファクトリーはファイル接尾辞で振り分けるためエクスポート済み成果物も
        # チェックポイントと同様に読み込まれて同じResultsオブジェクトを返す
        model = LibreYOLO("LibreEfficientDetd0.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 12c61fb0035437ce
---

## インストール

EfficientDetにオプションの追加パッケージは不要です。インポートするものはすべて基本インストールに含まれます。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

返される `Results` オブジェクトはすべてのファミリーで共通のため、別の検出器への置き換えは1行の変更で済みます。EfficientDetはアンカーベースの候補をデコードした後、クラス単位のNMSを実行します。そのため、`conf`、`iou`、`max_det` はすべて実際に効果を持ちます。ソース、ストリーミング、結果の処理については、[推論](/docs/predict)を参照してください。

## バリアント

サイズはD0からD4までの5つです。各段階で、より大きなEfficientNetバックボーンと、より深く幅広いBiFPNおよび予測ヘッドを組み合わせます。そのため、論文の複合スケーリング規則に従って、パラメータ数と計算量が同時に増えます。

## 検証

`val()` は `metrics/` キーの辞書を返します。内容は適合率、再現率、mAP 50、mAP 50-95で、学習に使用した形式の任意のデータセットに対して測定されます。

<code-tabs name="val" />

## エクスポート

<export-matrix />

エクスポート済み成果物はファイル接尾辞によって `LibreYOLO()` から再度読み込まれるため、`.onnx` または `.engine` ファイルはチェックポイントのように動作し、同じ `Results` を返します。

<code-tabs name="export" />

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box>

LibreYOLOのD0-D4チェックポイントは、Apache-2.0のrwightman/efficientdet-pytorchプロジェクトを通じて変換されています。このプロジェクト自体はgoogle/automlの公式TensorFlow学習済み重みを、学習済みテンソルを変更せずミラーしています。LGPLライセンスのzylo117/Yet-Another-EfficientDet-Pytorchプロジェクトのソースは参照も使用もしていません。

</provenance-box>
