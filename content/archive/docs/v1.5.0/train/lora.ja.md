---
title: LoRAファインチューニング
seo_title: LibreYOLOのLoRAファインチューニング
description: >-
  lora=Trueを使い、少ないVRAMでTransformer検出器をファインチューニングします。対応する9ファミリー、ファミリーごとのadapter手順、チェックポイントの動作を説明します。
lead: >-
  LoRAはモデルの学習済みの大規模部分を凍結し、その横にある小さな低rank
  adapterと、密なままにする必要がある層を学習します。LibreYOLOの公開インターフェースは1つのbooleanだけです。
keywords:
  - lora ファインチューニング
  - パラメータ効率 ファインチューニング
  - peft
  - dora
  - 低 vram 学習
  - rf-detr lora
  - d-fine lora
  - adapter merge
last_verified: 1.5.0
snippets:
  install:
    - label: pip
      language: bash
      code: |
        pip install "libreyolo[lora]"
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.train(data="my-dataset.yaml", epochs=50, lora=True)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs.pt data=my-dataset.yaml \
          epochs=50 lora=true
  merge:
    - label: エクスポート時にadapterをmerge
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("runs/train/exp/weights/best.pt")
        model.export(format="onnx")
    - label: その場でmerge
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.training.lora import merge_lora_adapters

        model = LibreYOLO("runs/train/exp/weights/best.pt")
        merged = merge_lora_adapters(model.model)

        print(f"{merged} adapter layers folded into dense weights")
source_hash: 603fdddf5ec0c316
---

## インストール

LoRAは任意の `peft` 依存関係を使います。

<code-tabs name="install" />

これがない場合、誤って完全なファインチューニングを行わないよう、`lora=True` は該当するコマンドを示す `ImportError` を送出します。

## 使用方法

<code-tabs name="train" />

`lora=True` がインターフェースのすべてです。rank、alpha、dropout、対象モジュールは、各アップストリームreferenceに合わせてファミリーごとに固定され、ユーザー向けの調整項目ではありません。

LoRAに対応しないファミリーはフラグを無視せず、設定時に例外を送出します。

```text
LoRA fine-tuning (lora=True) is not supported for yolo9. LoRA targets
transformer components with nn.Linear layers (e.g. RF-DETR, D-FINE, DEIM).
```

CLIでは、モデル構築前のより早い段階で、同じ9ファミリーの独自allowlistを使って拒否します。

## 対応ファミリー

RF-DETR、D-FINE、DEIM、DEIMv2、RT-DETR v1・v2・v4、EC、ConvNeXtです。gateは各ファミリーのtrainer classにある `supports_lora` 属性で、CLIにも一致するallowlistがあります。

タスクの対象範囲はファミリーの対象範囲より狭くなります。D-FINEとECは検出だけに対応し、segmentおよびpose経路では例外が送出されます。RF-DETRのsemantic経路でも例外が送出されます。ConvNeXtは分類用です。

それ以外ではすべて例外が送出されます。部分的または通知なしで動作するモードはありません。

## 各手順の動作

アーキテクチャが異なるため、手順も異なります。ViTバックボーンで機能する手順を、接続先のない畳み込みバックボーンに適用することはできません。

RF-DETRはRF-DETR referenceに合わせ、DINOv2バックボーンのattention `query`、`key`、`value` projectionで、rank 16およびalpha 16のDoRA（weight-decomposed LoRA）を使います。ViTバックボーンは凍結され、projector、decoder、検出ヘッドは通常どおり学習を続けます。

D-FINE、DEIM、RT-DETR v1・v2・v4は、畳み込みバックボーンとTransformer hybrid encoderおよびdeformable decoderを組み合わせるため、分割位置が変わります。畳み込みバックボーンは完全に凍結され、backward passも省略されます。Transformer blockは基本の重みを凍結し、その線形層で同じrank 16・alpha 16の通常のLoRA adapterを学習します。対象はfeed-forwardの `linear1` と `linear2`、gate、deformable attention projectionです。それ以外のencoder convolution fusion、入力projection、推論ヘッド、query embeddingはすべて密に学習されます。

この手順の2つの詳細は意図的なものです。decoder self-attentionはadapterなしで凍結されます。PyTorchの `nn.MultiheadAttention` は `out_proj.weight` を直接読み取り、注入したadapterを通知なしで迂回するためです。また、DoRAではなく通常のLoRAを使います。設計上ゼロ初期化されるdecoderの線形層がいくつかあり、DoRAのmagnitude normalizationではweight normによる除算が行われるためです。

DEIMv2は、SwiGLU feed-forward層の `w12` と `w3` を対象にして同じ手順を使います。S、M、L、XサイズにはDINOv3 ViTバックボーンもあり、ViTの基本部分を凍結し、fused attentionの `qkv` 層にadapterを追加します。一方、Spatial Tuning Adapterの畳み込みpyramidはprojectorに相当するものとして学習を続けます。設定上ViTが凍結された状態で公開されていても、これらの `qkv` adapterは追加されます。凍結したバックボーンを適応させることが目的だからです。sub-Sサイズは畳み込みバックボーンを使い、通常の手順を適用します。

ECは、学習可能なconvolution projector pyramidに囲まれたViTバックボーンを持つDETRです。ViTの基本部分を凍結して `qkv` 層にadapterを追加し、Transformer blockには共通手順を適用します。projectorとヘッドは密なままです。

ConvNeXt blockにはchannels-lastのlinear MLPである `fc1` と `fc2` があり、通常のadapterを適用します。depthwise convolution、norm、layer-scaleパラメータは凍結されます。独自のクラス数に対応できるよう、分類ヘッドは密なままです。

独自のクラス数には新たに学習したヘッドが必要なため、すべての手順で検出ヘッドと分類ヘッドは常に学習可能な状態を維持します。

## チェックポイントとエクスポート

`best.pt` と `last.pt` はadapterテンソルを保持するため、LoRA実行はほかと同様に再開または調査できます。これらのチェックポイントの読み込みには `lora` 追加パッケージが必要です。loaderがadapter注入を再実行し、キーを一致させるためです。

`export()` はadapterを密な重みにmergeするため、エクスポート済み成果物は `peft` に依存しません。同じmergeをメモリ内モデルへ直接適用することもできます。

<code-tabs name="merge" />

merge後のモジュールツリーは完全に密になり、2回目のmergeは何もしません。

## 節約できるものとできないもの

LoRAはoptimizerとgradientのメモリを削減し、バックボーンを完全に凍結するファミリーでは、そのバックボーンのbackward passも省略します。

activation memoryは変わりません。学習可能な部分についてはforward activationを引き続き保持する必要があり、通常はそれがpeakを決めます。VRAMの制約が厳しい場合は、`batch` または `imgsz` も小さくしてください。

## 関連項目

- すべてのファミリーで機能し、追加の依存関係なしで重みの一部を学習するもう1つの方法については[層の凍結](/docs/train/layer-freezing)。`freeze` と `lora=True` は併用でき、親バックボーングループが凍結されていてもadapterパラメータは学習可能なままです。
- `batch`、`imgsz`、そのほかの `train()` の項目については[ハイパーパラメータ](/docs/train/hyperparameters)。

