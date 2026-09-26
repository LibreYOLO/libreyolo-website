---
title: データ拡張
seo_title: LibreYOLOの学習用データ拡張
description: TrainConfigのデータ拡張設定、その背後にある4種類のパイプライン形状、各設定が使用・制限・無視されるかを示すファミリー別表について説明します。
lead: >-
  データ拡張はTrainConfigの設定で構成しますが、各モデルファミリーは独自の学習パイプラインを実行します。mosaic分岐がないパイプラインは、mosaic_probを近似せず無視します。
keywords:
  - yolo データ拡張
  - mosaic データ拡張
  - mixup
  - hsv jitter
  - random affine
  - copy paste データ拡張
  - randaugment
  - cutmix
  - no_aug_epochs
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            mosaic_prob=1.0,
            mixup_prob=0.15,
            hsv_prob=1.0,
            flip_prob=0.5,
            no_aug_epochs=15,
        )
    - label: CLI
      language: bash
      code: |
        # CLIではmosaic_probをmosaic、mixup_probをmixupと記述
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 mosaic=1.0 mixup=0.15 hsv_prob=1.0 \
          flip_prob=0.5 no_aug_epochs=15
  support:
    - label: ファミリーの対応表を確認
      language: python
      code: |
        from libreyolo.data.augment.spec import AUG_KNOBS, aug_support

        for knob, description in AUG_KNOBS.items():
            support = aug_support("yolo9")[knob]
            print(f"{knob:16} {support.status:16} {support.note or description}")
    - label: 無視される項目のみ
      language: python
      code: |
        from libreyolo.data.augment.spec import ignored_aug_params

        print(sorted(ignored_aug_params("rfdetr")))
  classify:
    - label: 分類用設定
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.train(
            data="my-classification-dataset",
            epochs=50,
            auto_augment="randaugment",
            erasing=0.25,
            mixup=0.2,
            cutmix=0.2,
        )
source_hash: 47461cd13aab580c
---

## 設定値の指定

データ拡張の設定値は通常の`train()`引数です。

<code-tabs name="train" />

このうち2つはCLIで短い綴りを使います。`mosaic`は`mosaic_prob`に、`mixup`は`mixup_prob`に
対応します。他の設定値はすべて両方で同じ綴りです。

## 2つではなく3つの状態

設定値が機能するかどうかはファミリーによって異なります。ライブラリはその宣言的な表を保持し、
各項目は3つの状態のいずれかになります。

`used`は設定値がパイプラインに到達してサンプルを変更することを意味します。`ignored`は
パイプラインに到達せず、設定しても何も起きないことを意味します。`gated_by_mosaic`はmosaic
分岐を通ったサンプルだけに適用されることを意味します。そのため、配線されていても
`mosaic_prob=0`では一度も作動しません。

3つ目の状態が予想外になりやすいものです。YOLOX方式のパイプラインではアフィン変形がmosaic
キャンバス上で実行され、MixUpがmosaicサンプルを混合します。そのため、`mosaic_prob=0`は
`degrees`、`translate`、`shear`、`perspective`、`mosaic_scale`、`mixup_prob`、
`mixup_scale`をすべて通知なく無効にします。学習器はMixUpの場合に限り、次の警告を記録します。

```text
mixup_prob=0.15 has no effect for YOLOv9: mixup only applies to mosaic samples
and mosaic_prob=0. Set mosaic_prob > 0 to enable mixup.
```

CLIは無視される設定値についても警告し、実際に入力したものだけを一覧化します。

```text
Warning: RF-DETR ignores these parameters: degrees, mosaic
```

## 4種類のパイプライン形状

ファミリーは4つの学習パイプラインに分類され、パイプラインによってほぼすべての結果が決まります。

YOLOX方式のmosaicパイプラインはサンプルごとにHSVジッターと反転を適用し、mosaic分岐内で
アフィン変換とMixUpを実行します。YOLOX、YOLOv7、YOLOv9とそのE2EおよびP2バリアント、
RTMDet、PicoDet、RT-DETR、RT-DETRv2、FOMOが対象です。

DETR方式のパススルーパイプラインにはmosaicもアフィン変形もありません。測光歪み、ズームアウト、
IoU切り抜きは設定値ではなくレシピ定数なので、有効なのは`flip_prob`と`no_aug_epochs`だけです。
D-FINE、Dome-DETR、DEIM、DEIMv2、RT-DETRv4、ECが対象で、RF-DETRでは1点異なります。

分類用ImageFolderパイプラインは、検出用の設定値をすべて無視します。水平反転は固定の0.5で、
`flip_prob`は到達しません。代わりに、後述する独自の設定群を持ちます。

YOLO-NASは独自の形状です。mosaicは一切なく、サンプルごとのアフィン変換が常に有効で、MixUpは
制限されず独立して適用されます。`mosaic_scale`の値はアフィン変換のスケール範囲として再利用
されます。

SegFormerとNAFNetは、それぞれタスク固有のパイプラインを実行します。そのランダム性は設定可能
ではなくファミリー内で固定されています。SegFormerで有効な設定値は、`mosaic_scale`と
`hsv_prob`ではなくクラス属性`semantic_scale_jitter`と`semantic_hsv_prob`です。NAFNetの
切り抜きと反転は、入力とターゲットを組み合わせた処理として固定確率0.5で行われます。

## 各ファミリーが尊重する設定値

次の表は`libreyolo/data/augment/spec.py`に同梱された仕様で、ライブラリ自身のテストにより実際の
パイプライン配線と照合されています。アーキテクチャから推測せず、ここを参照してください。

<code-tabs name="support" />

基本設定値をパイプラインごとに要約すると次のようになります。

| 設定値 | YOLOX方式 | YOLO-NAS | DETR方式 | 分類 |
|---|---|---|---|---|
| `mosaic_prob` | 使用 | 無視 | 無視 | 無視 |
| `mixup_prob` | mosaicにより制限 | 使用 | 無視 | 無視 |
| `hsv_prob` | 使用 | 使用 | 無視 | 無視 |
| `flip_prob` | 使用 | 使用 | 使用 | 無視 |
| `flipud` | 使用 | 使用 | 無視 | 無視 |
| `degrees` | mosaicにより制限 | 使用 | 無視 | 無視 |
| `translate` | mosaicにより制限 | 使用 | 無視 | 無視 |
| `shear` | mosaicにより制限 | 使用 | 無視 | 無視 |
| `perspective` | mosaicにより制限 | 使用 | 無視 | 無視 |
| `mosaic_scale` | mosaicにより制限 | 使用 | 無視 | 無視 |
| `mixup_scale` | mosaicにより制限 | 使用 | 無視 | 無視 |
| `no_aug_epochs` | 使用 | 使用 | 使用 | 使用 |

各列内の例外は次のとおりで、すべて範囲を狭めるものです。

- RTMDet、PicoDet、RT-DETR、RT-DETRv2、FOMOには垂直反転がないため、`flipud`は無視されます。FOMOのmosaicラッパーもperspectiveなしで構築されています。
- RF-DETR固有のパイプラインにはHSVジッターがないため、DETR方式の列に加えて`hsv_prob`も無視されます。
- ECは`hsv_prob`、`degrees`、`translate`を尊重しますが、それらを読み取るキーポイント対応変換を使う`task="pose"`の場合だけです。detectとsegmentの経路は固定の測光レシピを使います。
- DINOv2はdetectおよびsemanticタスクでDETR方式の列に従い、`task="classify"`では分類用設定を追加します。

`no_aug_epochs`はどこでも`used`ですが、意味は同じではありません。mosaicパイプラインでは最後の
エポックでmosaicとMixUpを無効にします。DETR方式のパイプラインでは測光、ズームアウト、切り抜き
による拡張を停止し、スケジュール終端の形状を決めます。分類およびセマンティックパイプラインでは、
終端の形状だけを決めます。

## 分類用設定

4つの設定値が分類パイプラインだけを制御します。検出ファミリーは4つすべてを無視します。

<code-tabs name="classify" />

`auto_augment`は`"randaugment"`、`"autoaugment"`、`"augmix"`、または`None`を受け取ります。
`erasing`はRandomErasingの確率です。`mixup`と`cutmix`はソフトラベルを生成するバッチごとの確率で、
バッチごとに最大1つだけが実行されます。MixUpが先なので、2つは加算され、合計を最大1にする
必要があります。

4つともデフォルトで無効なので、明示的に要求しない限り分類学習は変わりません。

1つの名前衝突は明記する価値があります。CLIでは`mixup`が検出用`mixup_prob`の別名です。分類用の
`mixup`フィールドには独自のCLI表記がなく、Pythonの`model.train(mixup=...)`からしか到達
できません。

## ファミリー固有の設定値

一部の設定値は基底クラスではなくファミリーの設定サブクラスにあるため、そのファミリーだけに
存在し、CLIフラグはありません。

| ファミリー | 設定値 | 効果 |
|---|---|---|
| YOLOv9、YOLOv9-E2E、YOLOv9-P2 | `copy_paste` | Copy-pasteインスタンス拡張の確率。`task="segment"`のみ |
| YOLOv9、YOLOv9-E2E、YOLOv9-P2 | `copy_paste_mode` | `"flip"`は同じサンプルを反転して再利用し、`"mixup"`は2つ目のサンプルを取得 |
| YOLOv9、YOLOv9-E2E、YOLOv9-P2 | `rot90` | 90度ランダム回転の確率 |
| YOLOv9 | `max_labels` | 学習変換での画像ごとの正解データ上限。デフォルトは100 |
| RF-DETR | `copy_paste`、`copy_paste_mode` | `task="segment"`用のCopy-paste。`"flip"`モードのみ |
| RF-DETR、D-FINE、EC | `crop_resize_prob` | ランダムな切り抜きとサイズ変更の確率 |
| EC、YOLO-NAS | `brightness_contrast_prob`、`affine_prob` | 姿勢経路のジッターとキーポイント対応アフィン変換の確率 |

`max_labels`は、通知なくデータを失う設定値です。上限を超えたボックスはエラーなく破棄されるため、
航空写真など検出対象が密な画像では値を増やす必要があります。

回転ボックスに対応した隅座標のデータ拡張は実装されていないため、設定値にかかわらず回転ボックス
学習ではMosaicとMixUpが無効になります。

## 関連項目

- `no_aug_epochs`をスケジュール引数として扱う方法と、その他の`train()`設定については[ハイパーパラメータ](/docs/train/hyperparameters)を参照してください。
- これらの変換が使用するラベル形式については[データセット](/docs/train/datasets)を参照してください。
