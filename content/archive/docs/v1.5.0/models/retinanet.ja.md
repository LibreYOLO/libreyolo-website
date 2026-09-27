---
title: RetinaNet
families:
  - retinanet
seo_title: LibreYOLOのRetinaNet：推論、検証、エクスポート
description: >-
  LibreYOLOのRetinaNetでfocal
  lossを使う1段階物体検出を実行します。BSD-3-Clauseのtorchvision移植版について、インストール、推論、検証、エクスポートを説明します。
lead: >-
  RetinaNetはfocal
  lossで学習する1段階検出器です。簡単な負例の重みを下げることで、密なアンカーグリッドでも精度維持のための独立したproposal段階が不要になります。LibreYOLOはtorchvisionの実装を物体検出向けに移植しています。
keywords:
  - RetinaNet 使い方
  - focal loss
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

        model = LibreYOLO("LibreRetinaNetr50v2.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRetinaNetr50v2.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRetinaNetr50v2.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRetinaNetr50v2.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRetinaNetr50v2.pt")
        model.export(format="onnx", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRetinaNetr50v2.pt format=onnx imgsz=800
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリがファイル接尾辞で振り分けるため、エクスポートした成果物も
        # 通常のチェックポイントと同様に読み込まれ、同じResultsオブジェクトを返す
        model = LibreYOLO("LibreRetinaNetr50v2.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 1cc7ceb6de290bdb
---

## インストール

RetinaNetに任意の追加パッケージは必要ありません。インポートするものはすべて基本
インストールに含まれます。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

返される`Results`オブジェクトはすべてのファミリーで共通のため、別の検出器への切り替えは
1行の変更だけで済みます。`conf`と`iou`は信頼度とNMSのしきい値を設定します。RetinaNetは
密なアンカーグリッドに対するアップストリームのNMS処理を維持します。入力ソース、
ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## バリアント

サイズは2種類で、どちらもfeature pyramidを持つResNet-50です。`r50`は元のヘッドを使い、
`r50v2`はGroupNormヘッドと、FPN出力ではなくバックボーンの最終段階から入力される幅広いP6
ブロックに置き換えます。

## 検証

`val()`は、学習に使った形式の任意のデータセットに対して測定した適合率、再現率、mAP 50、
mAP 50-95を含む`metrics/`キーの辞書を返します。

<code-tabs name="val" />

## エクスポート

<export-matrix />

RetinaNetはバッチサイズ1でONNXだけにエクスポートできます。RetinaNetはアスペクト比を維持した
可変入力へリサイズするため、LibreYOLOは渡された値にかかわらず`dynamic=True`を強制します。
これにより、異なる形状の入力ソースでもグラフが有効に保たれます。エクスポートした`.onnx`
ファイルは、ファイル接尾辞に基づいて`LibreYOLO()`から再読み込みされ、同じ`Results`を返します。

<code-tabs name="export" />

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box></provenance-box>

