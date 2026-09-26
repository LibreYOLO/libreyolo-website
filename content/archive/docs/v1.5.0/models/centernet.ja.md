---
title: CenterNet
families:
  - centernet
seo_title: CenterNet：LibreYOLOで物体検出
description: >-
  ResDCN-18とDLA-34バックボーンを使うCenterNet（Objects as
  Points）をLibreYOLOで実行します。MITライセンスの下で推論、検証、ONNXへのエクスポートに対応します。学習経路はありません。
lead: >-
  CenterNetは物体をバウンディングボックスの中心点としてモデル化し、その他のすべての属性をヒートマップのピークから回帰します。そのため、アンカーもnon-maximum
  suppression処理も必要ありません。LibreYOLOは推論専用の検出器として提供します。
keywords:
  - CenterNet 物体検出
  - Objects as Points
  - キーポイント検出
  - アンカーフリー 検出器
  - ResDCN-18
  - DLA-34
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCenterNetresdcn18.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreCenterNetresdcn18.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: DLA-34
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCenterNetdla34.pt")
        result = model(SAMPLE_IMAGE, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCenterNetresdcn18.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreCenterNetresdcn18.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCenterNetresdcn18.pt")

        # ONNXエクスポートにはopset 16以降が必要。deformable-convolutionの
        # アップサンプリング段階はopset 16で導入されたGridSampleへ変換
        model.export(format="onnx", opset=18)
        model.export(format="tensorrt")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreCenterNetresdcn18.pt format=onnx opset=18
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO

        # ファクトリがファイル接尾辞で振り分けるため、エクスポートした成果物も
        # 通常のチェックポイントと同様に読み込まれ、同じResultsオブジェクトを返す
        model = LibreYOLO("LibreCenterNetresdcn18.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 20aaef83cc95590d
---

## インストール

CenterNetに任意の追加パッケージは必要ありません。インポートするものはすべて基本
インストールに含まれます。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

返される`Results`オブジェクトはすべてのファミリーで共通のため、別の検出器への切り替えは
1行の変更だけで済みます。`conf`と`max_det`は順位付けされたヒートマップのピークを絞り込みます。
`iou`はAPIの一貫性のため受け付けますが、効果はありません。CenterNetのtop-kピークのデコードには
ボックスIoUによる抑制処理が不要なためです。入力ソース、ストリーミング、結果の処理については
[推論](/docs/predict)を参照してください。

## バリアント

バックボーンは2種類です。`resdcn18`はResNet-18 trunkとdeformable-convolutionによる
アップサンプリングを組み合わせ、`dla34`はDLA-34 trunkと反復的なdeep-aggregation
アップサンプリングを組み合わせます。どちらも同じ3個の密なヘッド（ヒートマップ、幅・高さ、
offset）と同じ入力キャンバスを使います。

## 検証

`val()`は、学習に使った形式の任意のデータセットに対して測定した適合率、再現率、mAP 50、
mAP 50-95を含む`metrics/`キーの辞書を返します。

<code-tabs name="val" />

## エクスポート

<export-matrix />

ONNXエクスポートにはopset 16以降が必要です。両方のバックボーンにあるdeformable-convolutionの
アップサンプリング段階は、opset 16で導入されたONNXの`GridSample`演算子へ変換されます。
古いopsetを要求すると、トレース開始前に例外が発生します。

<code-tabs name="export" />

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box>

ResDCN-18グラフではMicrosoftのMITライセンス版human-pose-estimation.pytorchも
クレジットし、DLA-34グラフではFisher YuによるBSD-3-ClauseのDLA実装をクレジットします。
LibreYOLOはアップストリームプロジェクトが使った元のDCNv2拡張機能を同梱しません。ネイティブ
実行では代わりにtorchvisionのBSD-3-Clause版`deform_conv2d`を使い、エクスポート専用の
移植可能な実装はLibreYOLO向けに別途作成されています。

</provenance-box>

## 引用

<citation-block />

