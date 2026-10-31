---
title: SwinIR
families:
  - swinir
seo_title: SwinIR：LibreYOLOで4倍画像超解像を実行
description: >-
  LibreYOLOでSwinIRを使い、4倍画像超解像を行います。軽量・中・大規模チェックポイントについて、インストール、推論、検証、エクスポートの方法を説明します。
lead: >-
  画像復元用のSwin
  Transformerネットワークです。LibreYOLOは4倍超解像チェックポイントとして、公式の軽量ジェネレーター、実世界向け中規模ジェネレーター、実世界向け大規模ジェネレーターの推論と検証に対応します。
keywords:
  - SwinIR
  - Swin Transformer
  - 画像 超解像
  - 画像復元
  - residual Swin Transformer block
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSwinIRm-restore.pt")
        result = model(SAMPLE_IMAGE, save=True)

        restored = result.restored
        print(restored.array.shape, restored.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSwinIRm-restore.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: 大きな画像向けのタイル分割
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwinIRl-restore.pt")

        # tileは順伝播を重なり合うタイルに分割し、継ぎ目を再び合成する
        # tile_padは各タイルを切り戻す前に周囲へ加えるハロー領域
        # どちらもPython専用のキーワード引数であり
        # CLIフラグではない
        result = model("large-photo.jpg", tile=512, tile_pad=16, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwinIRm-restore.pt")
        metrics = model.val(data="my-restore-dataset.yaml")

        print(metrics["metrics/PSNR"])
        print(metrics["metrics/SSIM"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSwinIRm-restore.pt data=my-restore-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwinIRm-restore.pt")

        # imgszを省略すると作業解像度ではなく小さな内部パッチサイズが
        # デフォルトになるため、実際にデプロイ先からモデルへ渡すサイズを指定
        model.export(format="onnx", imgsz=512)
        model.export(format="tensorrt", imgsz=512, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSwinIRm-restore.pt format=onnx imgsz=512
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリーはファイル接尾辞で経路を選ぶため、エクスポート済み成果物も
        # 通常のチェックポイントと同様に読み込まれ、同じResultsオブジェクトを返す
        model = LibreYOLO("LibreSwinIRm-restore.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.restored.array.shape)
source_hash: 87fc3d5524480eec
---

## インストール

SwinIRに任意の追加パッケージは必要ありません。インポートするものはすべて基本インストールに含まれています。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

復元結果にボックスは含まれません。`result.restored` は、各次元が入力の4倍のキャンバスを持つ密な `(H, W, 3)` uint8 RGB画像です。`save=True` を指定すると、アノテーション付きプロットではなく、その画像を直接書き込みます。入力はリサイズされずに8の倍数へパディングされるため、推論は写真自体の解像度で実行されます。メモリに収まらない大きさのソースは `tile` と `tile_pad` で分割でき、出力時にタイルの継ぎ目が再び合成されます。ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## バリアント

3つのサイズがあり、すべて拡大率は4倍に固定されています。`s` は公式の軽量ジェネレーターで、4段のResidual Swin Transformer Block（RSTB）とpixel-shuffle-directアップサンプリングを備えます。`m` と `l` は実世界向けの中規模および大規模ジェネレーターで、それぞれ6段と9段のRSTB、およびバイキュービック縮小だけでなく実世界の劣化を想定した最近傍補間と畳み込みによるアップサンプラーを備えます。

## 検証

`val()` は復元出力ときれいな目標画像との間でPSNRとSSIMを測定します。どちらも境界の切り取りやリサイズを行わず、元のキャンバス上のRGBで計算されます。SSIMはsigma 1.5の11x11 Gaussianウィンドウを使い、3つの色チャンネル全体で平均されます。

<code-tabs name="val" />

データセット引数には、劣化した入力画像のディレクトリと同じ解像度のきれいな目標画像のディレクトリを対応付けるYAMLを指定します。正確なキーについては[データセット形式](/docs/reference/dataset-formats)を参照してください。

## エクスポート

<export-matrix />

エクスポート済み成果物はファイル接尾辞に基づいて `LibreYOLO()` から再度読み込めます。そのため、`.onnx` または `.engine` ファイルはチェックポイントと同様に動作し、同じ `Results` を返します。このファミリーでは、ExecuTorchと、マトリックスでブロック対象と示されたすべての形式を利用できません。ONNX、TorchScript、TensorRT、OpenVINO、TFLiteは利用できます。[エクスポート](/docs/export)には、すべての形式で受け付ける引数と、一部の形式で追加される引数が記載されています。

<code-tabs name="export" />

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box></provenance-box>

## 引用

<citation-block />

