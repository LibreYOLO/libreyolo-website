---
title: データ拡張
seo_title: LibreYOLOの学習用データ拡張
description: TrainConfigのデータ拡張設定、その背後にある4種類のパイプライン構成、各ファミリーがどの設定を使用、条件付き使用、無視するかを示す表を説明します。
lead: >-
  データ拡張はTrainConfigの設定で指定しますが、各モデルファミリーは独自の学習パイプラインを実行します。Mosaic分岐がないパイプラインは、mosaic_probを近似して使うのではなく無視します。
keywords:
  - YOLO データ拡張
  - Mosaic データ拡張
  - MixUp
  - HSV jitter
  - Random Affine
  - Copy-Paste データ拡張
  - RandAugment
  - CutMix
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
        # CLIではmosaic_probをmosaic、mixup_probをmixupと記述します。
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
    - label: 無視される設定だけを確認
      language: python
      code: |
        from libreyolo.data.augment.spec import ignored_aug_params

        print(sorted(ignored_aug_params("rfdetr")))
  classify:
    - label: 画像分類用設定
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

## 設定項目の指定

データ拡張の設定項目は通常の`train()`引数です。

<code-tabs name="train" />

2つには短いCLI表記があります。`mosaic`は`mosaic_prob`に、`mixup`は`mixup_prob`にマッピングされます。その他の設定名はどちらでも同じです。

## 2つではなく3つの状態

設定が効果を持つかどうかはファミリーによって異なります。ライブラリは宣言的な表でこれを管理し、各エントリーは3つの状態のいずれかになります。

`used`は設定がパイプラインに届き、サンプルを変更することを意味します。`ignored`はパイプラインに届かず、設定しても何も変わらないことを意味します。`gated_by_mosaic`はMosaic分岐を通ったサンプルにだけ適用されることを意味します。したがって、配線されていても`mosaic_prob=0`では実行されません。

この3番目の状態は見落とされがちです。YOLOX形式のパイプラインでは、Affine変換がMosaicキャンバス上で実行され、MixUpがMosaicサンプルをブレンドします。このため、`mosaic_prob=0`にすると`degrees`、`translate`、`shear`、`perspective`、`mosaic_scale`、`mixup_prob`、`mixup_scale`がすべて暗黙的に無効になります。MixUpの場合は、トレーナーが次の警告をログに記録します。

```text
mixup_prob=0.15 has no effect for YOLOv9: mixup only applies to mosaic samples
and mosaic_prob=0. Set mosaic_prob > 0 to enable mixup.
```

CLIは無視される設定についても警告し、実際に入力したものだけを列挙します。

```text
Warning: RF-DETR ignores these parameters: degrees, mosaic
```

## 4種類のパイプライン構成

ファミリーは4種類の学習パイプラインに分類され、ほとんどの挙動はパイプラインによって決まります。

YOLOX形式のMosaicパイプラインでは、サンプルごとにHSV jitterと反転を適用し、Mosaic分岐内でAffineとMixUpを実行します。YOLOX、YOLOv7、YOLOv9とそのE2EおよびP2バリアント、RTMDet、PicoDet、RT-DETR、RT-DETRv2、FOMOが該当します。

DETR形式のパススルーパイプラインにはMosaicもAffine変換もありません。測光歪み、ズームアウト、IoU cropは構成設定ではなくレシピの定数なので、有効なのは`flip_prob`と`no_aug_epochs`だけです。D-FINE、Dome-DETR、DEIM、DEIMv2、RT-DETRv4、ECと、1点だけ異なるRF-DETRが該当します。

画像分類のImageFolderパイプラインは、物体検出用設定をすべて無視します。水平反転は固定の0.5で、`flip_prob`は届きません。代わりに、後述する独自の設定一式があります。

YOLO-NASは独自の構成です。Mosaicはなく、サンプルごとのAffineが常に有効で、MixUpはMosaicを条件とせず独立して適用されます。`mosaic_scale`の値はAffineのスケール範囲として再利用されます。

SegFormerとNAFNetはそれぞれタスク固有のパイプラインを実行し、ランダム性は設定可能ではなくファミリー内で固定されています。SegFormerで有効な設定は、`mosaic_scale`や`hsv_prob`ではなく、クラス属性の`semantic_scale_jitter`と`semantic_hsv_prob`です。NAFNetのcropと反転は、入力とターゲットを組にして固定確率0.5で処理します。

## ファミリーごとの設定対応

下の表は`libreyolo/data/augment/spec.py`に含まれる仕様です。ライブラリ自身のテストで、実際のパイプライン接続と一致することを確認しています。アーキテクチャから推測せず、この表を参照してください。

<code-tabs name="support" />

基本設定をパイプライン別にまとめると次のようになります。

| 設定 | YOLOX形式 | YOLO-NAS | DETR形式 | 画像分類 |
|---|---|---|---|---|
| `mosaic_prob` | 使用 | 無視 | 無視 | 無視 |
| `mixup_prob` | Mosaicを条件に使用 | 使用 | 無視 | 無視 |
| `hsv_prob` | 使用 | 使用 | 無視 | 無視 |
| `flip_prob` | 使用 | 使用 | 使用 | 無視 |
| `flipud` | 使用 | 使用 | 無視 | 無視 |
| `degrees` | Mosaicを条件に使用 | 使用 | 無視 | 無視 |
| `translate` | Mosaicを条件に使用 | 使用 | 無視 | 無視 |
| `shear` | Mosaicを条件に使用 | 使用 | 無視 | 無視 |
| `perspective` | Mosaicを条件に使用 | 使用 | 無視 | 無視 |
| `mosaic_scale` | Mosaicを条件に使用 | 使用 | 無視 | 無視 |
| `mixup_scale` | Mosaicを条件に使用 | 使用 | 無視 | 無視 |
| `no_aug_epochs` | 使用 | 使用 | 使用 | 使用 |

これらの列には、すべて対応範囲を狭める次の例外があります。

- RTMDet、PicoDet、RT-DETR、RT-DETRv2、FOMOには垂直反転がないため、`flipud`は無視されます。FOMOのMosaicラッパーはPerspectiveなしで構築されます。
- RF-DETRのネイティブパイプラインにはHSV jitterがないため、DETR形式の列に加えて`hsv_prob`も無視されます。
- ECは`hsv_prob`、`degrees`、`translate`に対応しますが、これらを読み取るキーポイント対応変換を使用する`task="pose"`の場合だけです。物体検出とセグメンテーションの経路では固定の測光レシピを使用します。
- DINOv2は物体検出とセマンティックタスクでDETR形式の列に従い、`task="classify"`では画像分類用設定が加わります。

`no_aug_epochs`はすべてで`used`ですが、意味は同じではありません。Mosaicパイプラインでは最後のエポックでMosaicとMixUpを無効にします。DETR形式のパイプラインでは、測光、ズームアウト、cropのデータ拡張を停止し、スケジュール終盤を形成します。画像分類とセマンティックのパイプラインでは、終盤の形成だけを行います。

## 画像分類用設定

4つの設定が画像分類パイプラインだけを制御します。物体検出ファミリーは4つすべてを無視します。

<code-tabs name="classify" />

`auto_augment`は`"randaugment"`、`"autoaugment"`、`"augmix"`、または`None`を受け付けます。`erasing`はRandomErasingの確率です。`mixup`と`cutmix`はソフトラベルを生成するバッチごとの確率です。1バッチで実行されるのは最大1つで、MixUpが先に評価されます。このため、2つの確率は加算され、合計を1以下にする必要があります。

4つはすべてデフォルトで無効なので、明示的に指定しない限り画像分類の学習は変わりません。

名前の衝突を明確にしておきます。CLIでは`mixup`が物体検出用`mixup_prob`のエイリアスです。画像分類の`mixup`フィールドには専用のCLI表記がなく、Pythonの`model.train(mixup=...)`からだけ利用できます。

## ファミリー固有の設定

一部の設定は基本クラスではなく、ファミリーの構成サブクラスにあります。このため、そのファミリーだけに存在し、CLIフラグはありません。

| ファミリー | 設定 | 効果 |
|---|---|---|
| YOLOv9、YOLOv9-E2E、YOLOv9-P2 | `copy_paste` | Copy-Pasteインスタンス拡張の確率。`task="segment"`だけで使用 |
| YOLOv9、YOLOv9-E2E、YOLOv9-P2 | `copy_paste_mode` | `"flip"`は同じサンプルを反転して再利用し、`"mixup"`は2つ目のサンプルを取得 |
| YOLOv9、YOLOv9-E2E、YOLOv9-P2 | `rot90` | ランダムな90度回転の確率 |
| YOLOv9 | `max_labels` | 学習変換での画像ごとの正解データ上限。デフォルトは100 |
| RF-DETR | `copy_paste`、`copy_paste_mode` | `task="segment"`向けのCopy-Paste。`"flip"`モードだけに対応 |
| RF-DETR、D-FINE、EC | `crop_resize_prob` | ランダムなcropとサイズ変更の確率 |
| EC、YOLO-NAS | `brightness_contrast_prob`、`affine_prob` | 姿勢推定経路のjitterとキーポイント対応Affineの確率 |

`max_labels`は通知なくデータを失う設定です。上限を超えたボックスはエラーなしで破棄されるため、航空写真のように密度の高い画像では値を増やす必要があります。

回転ボックス対応のデータ拡張が実装されていないため、設定にかかわらず方向付きボックスの学習ではMosaicとMixUpが無効になります。

## 関連項目

- スケジュール引数としての`no_aug_epochs`とその他の`train()`設定については、[ハイパーパラメーター](/docs/train/hyperparameters)を参照してください。
- これらの変換が使用するラベル形式については、[データセット](/docs/train/datasets)を参照してください。
