---
title: ResNet
families:
  - resnet
seo_title: 'ResNet: Apache-2.0で学習、検証、エクスポート'
description: >-
  LibreYOLOでResNetによる画像分類を実行します。LibreResNet18、34、50、101のインストール、推論、ファインチューニング、検証、エクスポートを解説します。
lead: >-
  ResNetは残差ブロックからなる画像分類器です。スキップ接続によって、通常の深い畳み込み層の積み重ねで生じる精度低下を避けながら、ネットワークに多数のレイヤーを追加できます。LibreYOLOでは画像分類に対応します。
keywords:
  - ResNet
  - ResNet50
  - 画像分類
  - 残差学習
  - 深層残差ネットワーク
  - ImageNet分類モデル
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreResNet50-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.probs.top1, result.probs.top1conf)
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreResNet50-cls.pt source=cat.jpg save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreResNet50-cls.pt")
        model.train(data="imagenette160", epochs=5)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreResNet50-cls.pt data=imagenette160 epochs=5
    - label: マルチGPU
      language: bash
      code: |
        libreyolo train model=LibreResNet50-cls.pt data=imagenette160 \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreResNet50-cls.pt")
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreResNet50-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreResNet50-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreResNet50-cls.pt format=onnx
        libreyolo export model=LibreResNet50-cls.pt format=tensorrt half=True
    - label: エクスポートしたファイルを使用
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリはファイルの拡張子に応じて振り分けるためエクスポートした
        # アーティファクトもチェックポイントと同様に読み込まれ同じResultsオブジェクトを返す
        model = LibreYOLO("LibreResNet50-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: e2f46c73716af1b7
---

## インストール

ResNetに追加オプションは必要ありません。インポートするものはすべて基本インストールに含まれています。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

返される`Results`オブジェクトは全ファミリーで共通なので、別のモデルへの切り替えは1行の変更で済みます。分類器にはボックスやマスクはなく、`result.probs`に画像全体の予測が格納されます。利用できる属性は`top1`、`top5`、`top1conf`、`top5conf`です。単一の確率ベクトルにはしきい値処理や抑制の対象がないため、`conf`、`iou`、`max_det`はAPI互換性のため受け付けますが効果はありません。入力ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## バリアント

深さの異なる4種類があり、すべて同じ方法で学習、評価されます。選択基準はパラメーター数と精度の直接的なトレードオフです。タスクは固定されており、どのサイズも画像分類だけに対応します。各サイズの重みファイル名は`-cls.pt`で終わり、ファクトリはこの接尾辞を読み取ってこのファミリーに振り分けます。`task=`引数は不要です。

## 学習

ファインチューニングは公開済みのImageNetバックボーンから開始し、最終分類器レイヤーを対象データセットのクラス数に合わせて自動的に再構築します。

<code-tabs name="train" />

指定を変更しない場合、トレーナーはAdamW、`lr0=1e-3`、バッチサイズ64で100エポック実行し、改善のない状態が50エポック続くと早期終了します。`data`には、データセットのルート（`train/`と`val/`の下にクラスごとのフォルダーを配置）、`imagenette160`などの既知の短縮名、または`.zip`のURLを指定できます。ここでは`lora=True`に対応していません。LibreYOLOのLoRAは`nn.Linear`レイヤーを持つTransformerコンポーネントを対象とし、ResNetには該当するレイヤーがないため、指定すると例外を送出します。

データセット、データ拡張、マルチGPU、ロガーについては[学習](/docs/train)を参照してください。

## 検証

`val()`は`metrics/`キーを持つ辞書を返します。画像分類では、検証分割に対するtop-1精度とtop-5精度が含まれます。

<code-tabs name="val" />

## エクスポート

<export-matrix />

エクスポートしたアーティファクトはファイルの拡張子に基づいて`LibreYOLO()`で読み込めるため、`.onnx`や`.engine`ファイルもチェックポイントと同様に動作し、同じ`Results`を返します。[エクスポート](/docs/export)には各形式で受け付ける引数と、一部の形式で追加されるオプションが記載されています。

<code-tabs name="export" />

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box></provenance-box>

## 引用

<citation-block />
