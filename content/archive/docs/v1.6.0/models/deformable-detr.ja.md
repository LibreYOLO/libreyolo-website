---
title: Deformable DETR
families:
  - deformable_detr
seo_title: Deformable DETR：Apache-2.0の下で推論、エクスポート
description: >-
  LibreYOLOのDeformable DETRで物体検出を行います。すべてApache-2.0ライセンスのsparse
  attentionによる5サイズをインストールし、推論、検証、エクスポートします。
lead: >-
  Deformable
  DETRはDETRの密なcross-attentionを、各参照点の周囲で行う疎なマルチスケールサンプリングへ置き換えます。これにより、Transformer検出器の学習が実用的になりました。LibreYOLOは検出向けに5つのサイズを推論専用として提供します。
keywords:
  - Deformable DETR
  - detection transformer
  - sparse attention
  - multi-scale attention
  - 物体検出
  - SenseTime
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDeformableDETRr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDeformableDETRr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeformableDETRr50.pt")

        # val()はオブジェクトではなく通常のdictを返す
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDeformableDETRr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeformableDETRr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="tensorrt", imgsz=800, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDeformableDETRr50.pt format=onnx imgsz=800

        libreyolo export model=LibreDeformableDETRr50.pt format=tensorrt
        imgsz=800 half=True
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリーはファイル接尾辞で振り分けるためエクスポート済み成果物も
        # チェックポイントと同様に読み込まれて同じResultsオブジェクトを返す
        model = LibreYOLO("LibreDeformableDETRr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 35225efc54b5ef91
---

## インストール

Deformable DETRにオプションの追加パッケージは不要です。インポートするものはすべて基本インストールに含まれ、純粋なPyTorchによるマルチスケール変形可能アテンションコアを使用します。

```bash
pip install libreyolo
```

`libreyolo[hub-kernels]` のインストールはオプションです。`kernels` パッケージが存在すると、LibreYOLOは実行時にHugging Face Hubからコンパイル済みのマルチスケール変形可能アテンションカーネルを取得し、純粋なPyTorchコアの代わりに使用します。`LIBREYOLO_HUB_KERNELS=0` で再び無効にできます。

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

返される `Results` オブジェクトはすべてのファミリーで共通のため、別の検出器への置き換えは1行の変更で済みます。`conf` と `max_det` はクエリ選択をフィルタリングします。APIの一貫性のため `iou` は受け付けますが、デコーダーがNMS処理のない集合予測器なので効果はありません。ソース、ストリーミング、結果の処理については、[推論](/docs/predict)を参照してください。

LibreYOLOのDeformable DETRは推論専用です。アップストリームはHungarian matchingとfocal classification lossで学習します。そのレシピはここで実装されていないため、`train()` は `NotImplementedError` を発生させます。

## バリアント

5つのチェックポイントが公開構成を網羅し、すべて同じ入力解像度です。`r50ss` はアテンションを1つの特徴スケールに限定します。`r50ssdc5` はそれにdilated C5バックボーン段階を追加します。`r50` はデフォルトのマルチスケール構成で、4つの特徴マップレベルをまたいでサンプリングします。`r50refine` はデコーダー層をまたぐ反復的なバウンディングボックス改善を追加し、`r50twostage` は学習済みクエリではなくエンコーダー出力から初期領域提案を生成します。

## 検証

`val()` は `metrics/` キーの辞書を返します。内容は適合率、再現率、mAP 50、mAP 50-95で、学習に使用した形式の任意のデータセットに対して測定されます。

<code-tabs name="val" />

## エクスポート

<export-matrix />

エクスポート済み成果物はファイル接尾辞によって `LibreYOLO()` から再度読み込まれるため、`.onnx` または `.engine` ファイルはチェックポイントのように動作し、同じ `Results` を返します。[エクスポート](/docs/export)では、すべての形式が受け付ける引数を説明しています。

<code-tabs name="export" />

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box></provenance-box>

## 引用

<citation-block />
