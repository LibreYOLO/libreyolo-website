---
title: NAFNet
families:
  - nafnet
seo_title: NAFNet：MITでノイズ除去、学習、エクスポート
description: >-
  LibreYOLOでNAFNetを画像ノイズ除去と復元に使います。MITライセンスのSIDDチェックポイントについて、インストール、推論、学習、検証、エクスポートの方法を説明します。
lead: >-
  NAFNetは画像復元用の畳み込みネットワークで、一般的なUNet blockから非線形activation
  functionを取り除き、要素ごとの乗算に置き換えます。LibreYOLOは復元という1つのタスクで対応し、SIDDで学習した実画像ノイズ除去チェックポイントを公開しています。
keywords:
  - NAFNet
  - 画像復元
  - 画像 ノイズ除去
  - 画像 ぼけ除去
  - nonlinear activation free network
  - SIDD
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        result = model("noisy.jpg", save=True)

        restored = result.restored
        print(restored.array.shape)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreNAFNetl-restore-sidd.pt source=noisy.jpg
        save=True
    - label: 復元画像を保存
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        result = model.predict("noisy.jpg")

        result.restored.save("denoised.png")
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        model.train(data="my-dataset.yaml", epochs=100, imgsz=256, batch=16,
        lr0=1e-3)
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreNAFNetl-restore-sidd.pt data=my-dataset.yaml
        \
          epochs=100 imgsz=256 batch=16 lr0=1e-3
    - label: チェックポイントの来歴
      language: python
      code: |
        from libreyolo import LibreYOLO

        # degradationとdatasetは保存済みチェックポイントに記録されるが
        # 学習内容は変更しない
        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            degradation="denoise",
            dataset="MyDataset",
        )
    - label: マルチGPU
      language: bash
      code: >
        libreyolo train model=LibreNAFNetl-restore-sidd.pt data=my-dataset.yaml
        \
          epochs=100 device=0,1 batch=32
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # val()はオブジェクトではなく通常のdictを返す
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PSNR"])
        print(metrics["metrics/SSIM"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreNAFNetl-restore-sidd.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        model.export(format="onnx", imgsz=256)
        model.export(format="tensorrt", imgsz=256, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreNAFNetl-restore-sidd.pt format=onnx
        imgsz=256

        libreyolo export model=LibreNAFNetl-restore-sidd.pt format=tensorrt
        imgsz=256 half=True
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO

        # ファクトリーはファイル接尾辞で経路を選ぶため、エクスポート済み成果物も
        # 通常のチェックポイントと同様に読み込まれ、同じResultsオブジェクトを返す
        model = LibreYOLO("LibreNAFNetl-restore-sidd.onnx")
        result = model("noisy.jpg")

        result.restored.save("denoised.png")
source_hash: 9bae9f82bee741bf
---

## インストール

NAFNetに任意の追加パッケージは必要ありません。インポートするものはすべて基本インストールに含まれています。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

返される `Results` オブジェクトは、このファミリー用の `restored` という1つのフィールドを持ちます。元のキャンバス上の密なHWC uint8 RGB画像であり、反復対象のボックスはありません。`save=True` は入力上にアノテーションを描かず、その復元画像をディスクへ直接書き込みます。`conf`、`iou`、`max_det` はほかのすべてのファミリーとのsignatureの一貫性のために受け付けますが、復元では絞り込み対象の検出結果が生成されないため効果はありません。ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## バリアント

`s`（幅32）と `l`（幅64）の2つの幅がこのアーキテクチャを共有し、どちらも256 pxの学習patchを中心に構築されています。推論と検証はサイズに関係なく画像のネイティブ解像度で動作し、ネットワークのdownsample factorに合わせたpaddingだけを行います。現在公開されているのは、SIDDで学習した実画像ノイズ除去チェックポイントの `l` 幅だけです。

## 学習

NAFNetは独自に用意した劣化画像ときれいな画像のペアでファインチューニングします。データセットYAMLは、劣化画像の `inputs/<split>/` フォルダーと、ファイルstemで対応付けられたきれいな対象画像の `targets/<split>/` フォルダーを指定します。`degradation` と `dataset` は来歴として保存済みチェックポイントへ記録される任意の文字列で、学習には関与しません。

<code-tabs name="train" />

設定を変更しなければ、trainerはAdamW、`lr0=1e-3`、バッチ16、256 px cropで100エポック実行し、PSNRが50エポック改善しないと早期終了（early stopping）します。このファミリーにはLoRA経路がありません。`NAFNetTrainer` はadapterファインチューニングへオプトインしないため、`lora=True` は実行せずにエラーを送出します。

学習中、ネットワークは通常のglobal-average poolingで動作します。NAFNetの推論専用windowed local pooling（Test-time Local Converter）は最初のエポック前に取り外され、学習完了後に再び取り付けられます。固定window local poolを通じたbackpropagationでは、推論時のチェックポイント利用方法と一致しないためです。

データセット、データ拡張、マルチGPU、loggerについては[学習](/docs/train)を参照してください。

## 検証

`val()` は、全有効キャンバスのRGBで計算した `metrics/PSNR` と `metrics/SSIM` を含む辞書を返します。SSIMはsigma 1.5の11x11 Gaussian windowを使い、best-checkpoint選択の `fitness` はPSNR値です。`data` は学習に使うものと同じペア画像データセット形式を指します。

<code-tabs name="val" />

## エクスポート

<export-matrix />

エクスポート済み成果物はファイル接尾辞に基づいて `LibreYOLO()` から再度読み込めます。そのため、`.onnx` または `.engine` ファイルはチェックポイントと同様に動作し、`restored` に出力画像を保持する同じ `Results` を返します。NAFNetは固定空間解像度でエクスポートされます。`imgsz` はネットワークのdownsample factor（どちらのアーキテクチャ幅でも16）で割り切れる必要があり、`dynamic=True` でもdynamicになるのはbatch次元だけです。高さと幅はエクスポート時に固定されます。

<code-tabs name="export" />

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box></provenance-box>

## 引用

<citation-block />

