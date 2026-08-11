---
title: ConvNeXt
families:
  - convnext
seo_title: ConvNeXt：Apache-2.0で学習、検証、エクスポート
description: >-
  LibreYOLOのConvNeXtで画像分類を行います。LibreConvNeXt
  tiny・small・baseのインストール、推論、LoRAによるファインチューニング、検証、エクスポートを説明します。
lead: >-
  ConvNeXtは標準的な畳み込みだけで構築された画像分類器で、ResNetの各ブロックをvision
  transformerの設計方針へ近づける形で現代化しています。LibreYOLOが対応するタスクは分類だけです。
keywords:
  - ConvNeXt 使い方
  - ConvNeXt tiny
  - 画像分類
  - 畳み込みニューラルネットワーク
  - ImageNet 分類器
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.probs.top1, result.probs.top1conf)
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreConvNeXtt-cls.pt source=cat.jpg save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.train(data="imagenette160", epochs=5)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreConvNeXtt-cls.pt data=imagenette160 epochs=5
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.train(data="imagenette160", epochs=5, lora=True)
    - label: マルチGPU
      language: bash
      code: |
        libreyolo train model=LibreConvNeXtt-cls.pt data=imagenette160 \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreConvNeXtt-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreConvNeXtt-cls.pt format=onnx
        libreyolo export model=LibreConvNeXtt-cls.pt format=tensorrt half=True
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリがファイル接尾辞で振り分けるため、エクスポートした成果物も
        # 通常のチェックポイントと同様に読み込まれ、同じResultsオブジェクトを返す
        model = LibreYOLO("LibreConvNeXtt-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: 1682cc69cf2925e6
---

## インストール

ConvNeXtに任意の追加パッケージは必要ありません。インポートするものはすべて基本
インストールに含まれます。

```bash
pip install libreyolo
```

`lora=True`を使うアダプターのファインチューニングは例外で、`lora`追加パッケージが必要です。

```bash
pip install "libreyolo[lora]"
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

返される`Results`オブジェクトはすべてのファミリーで共通のため、別のモデルへの切り替えは
1行の変更だけで済みます。分類器にはボックスもマスクもありません。`result.probs`には画像全体の
推論結果が入り、`top1`、`top5`、`top1conf`、`top5conf`を利用できます。`conf`、`iou`、
`max_det`はAPIの一貫性のため受け付けますが、効果はありません。1個の確率ベクトルに対して
しきい値処理や抑制をする対象がないためです。入力ソース、ストリーミング、結果の処理については
[推論](/docs/predict)を参照してください。

## バリアント

サイズはtiny・small・baseの3種類で、すべて同じ方法で学習・評価されます。選択はパラメータ数と
精度の直接的なトレードオフです。タスクは固定され、すべてのサイズが分類だけに対応します。
どのサイズでも重みのファイル名は`-cls.pt`で終わり、ファクトリはその接尾辞を読み取って
このファミリーへ振り分けます。`task=`引数は不要です。

## 学習

ファインチューニングは公開済みのImageNetバックボーンから開始し、対象データセットのクラス数に
合わせて最終分類器層を自動的に再構築します。

<code-tabs name="train" />

設定を変更しない場合、trainerはAdamW、`lr0=1e-3`、バッチ64で100エポック実行し、改善がない
状態が50エポック続くと早期終了（early stopping）します。`data`にはデータセットのルート
（`train/`と`val/`、クラスごとに1個のフォルダー）、`imagenette160`などの既知の短縮名、
または`.zip`のURLを指定できます。ConvNeXtのブロックにはLoRAが必要とする`nn.Linear` MLPが
あるため、`lora=True`に対応します。バックボーン全体をファインチューニングする代わりに、
ブロックMLPへアダプターを挿入します。

データセット、データ拡張、マルチGPU、loggerについては[学習](/docs/train)を参照してください。

## 検証

`val()`は`metrics/`キーの辞書を返します。分類では、検証分割に対するtop-1精度とtop-5精度です。

<code-tabs name="val" />

## エクスポート

<export-matrix />

エクスポートした成果物はファイル接尾辞に基づいて`LibreYOLO()`から再読み込みされます。そのため、
`.onnx`または`.engine`ファイルはチェックポイントと同様に動作し、同じ`Results`を返します。
[エクスポート](/docs/export)には、各形式が受け付ける引数と、一部の形式で必要になる追加パッケージの一覧があります。

<code-tabs name="export" />

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box>

このファミリーで提供するのはConvNeXt V1だけです。ConvNeXt-V2の小規模な学習済み
チェックポイントはCC-BY-NC 4.0であり、意図的に除外しています。非商用の重みをMIT・商用
ライブラリ内で再配布できないためです。

</provenance-box>

## 引用

<citation-block />

