---
title: データ拡張マトリックス
seo_title: LibreYOLOの各ファミリーが反映するデータ拡張
description: すべてのデータ拡張設定値の範囲とCLIでの綴り、学習可能な31のファミリーそれぞれにおける各設定値の状態、各ファミリーのデフォルト値。
lead: >-
  データ拡張の設定値を指定しても、それがパイプラインに届くとは限りません。このページでは、学習可能な各ファミリーがTrainConfigの各設定値をどう扱うかと各ファミリーの初期値を、記載対象のリリースのライブラリから生成して記録しています。
keywords:
  - libreyolo データ拡張
  - データ拡張 パラメータ
  - データ拡張 デフォルト値
  - mosaic_prob
  - mixup_prob
  - hsv_prob
  - mosaic_scale
  - no_aug_epochs
  - データ拡張 対応マトリックス
  - TrainConfig 設定値
last_verified: 1.8.0
verification: >-
  scripts/build-augmentation/collect.pyでLibreYOLO
  1.8.0から生成しました。設定値の一覧、状態、注記はlibreyolo/data/augment/spec.py（ライブラリのテストが実際のパイプラインに固定しています）から、デフォルト値は各ファミリーのTrainConfigサブクラスから、CLIでの綴りはlibreyolo
  trainコマンドから取得しています。範囲と、仕様とパイプラインが食い違う唯一の箇所は、同じスクリプトがライブラリと照合して確認しています。
snippets:
  usage:
    - label: 仕様を直接照会
      language: python
      code: |
        from libreyolo.data.augment.spec import (
            AUG_KNOBS,
            aug_support,
            ignored_aug_params,
            uses_mosaic_gating,
        )

        print(sorted(AUG_KNOBS))

        table = aug_support("yolo9")
        print(table["mixup_prob"].status, table["mixup_prob"].note)

        print(sorted(ignored_aug_params("dfine")))
        print(uses_mosaic_gating("yolo9"), uses_mosaic_gating("yolonas"))
    - label: ファミリーのデフォルト値
      language: python
      code: |
        from libreyolo.training.config import YOLO9Config

        config = YOLO9Config()
        print(config.mosaic_scale, config.degrees, config.mixup_prob)
source_hash: a88efd1537732e6c
---

## 設定値一覧

これらは`TrainConfig`のフィールド名です。`train()`とCLIはCLI列の短い綴りも受け付けるため、`mosaic`で`mosaic_prob`を設定できます。CLIでは`mixup`はタスクに応じて振り分けられ、分類モデルでは分類のバッチ混合、検出モデルでは`mixup_prob`になります。`perspective`にはCLIフラグがないため、Pythonから設定します。

<aug-knobs />

基本デフォルト値は、`TrainConfig`自体に設定されている値です。ほとんどのファミリーはその一部を上書きしており、学習が実際に使う初期値は[ファミリー別のデフォルト値](#defaults-by-family)に示しています。各設定値の働きを図で確認するには、[データ拡張](/docs/train/augmentations)を参照してください。

角括弧で示す範囲は区間で、`[0, 1)`は0を含み1を含みません。分類用の設定値はパイプラインの構築時に検査され、範囲外では`ValueError`が送出されます。検出用の設定値は`TrainConfig`では検査されないため、範囲外の値でも何の通知もなく学習が進みます。例外は、抽出されたスケールが0以下の場合と`perspective`が負の場合で、サンプルを変換するときに`ValueError`が送出されます。

<code-tabs name="usage" />

## 3つの状態

| 状態 | 意味 |
|---|---|
| `used` | 設定値がファミリーの学習パイプラインに届き、サンプルを変える |
| `gated_by_mosaic` | 設定値はmosaic分岐を通ったサンプルにのみ適用されるため、`mosaic_prob == 0`では一度も作動しない |
| `ignored` | 設定値はパイプラインに届かず、設定しても何も起きない |

何も失敗しないため、実行前に確認する価値があるのは`ignored`です。明示的に設定した学習パラメータが選択したファミリーで無視されるものである場合はCLIが警告し、ファミリーがMixUpをmosaicに依存させていて`mosaic_prob`が0のために`mixup_prob > 0`が作動できない場合はトレーナーが警告します。

## ファミリーごとの反映状況

<aug-matrix />

対象となるすべてのファミリーは、6つのパイプラインのいずれかに従います。YOLOX方式のパイプラインでは、サンプルごとの前処理がHSVジッターと反転を適用し、アフィン変換とMixUpはmosaic分岐の中でのみ実行されます。一方、YOLO-NASは常に有効なサンプルごとのアフィン変換を実行し、mosaicを無視し、MixUpを独立して適用し、`mosaic_scale`をアフィン変換のスケール範囲として再利用します。

DETR方式のパイプラインは、mosaicのないパススルー変換です。その測光歪み、ズームアウト、IoUクロップは設定値ではなくレシピの定数であるため、`hsv_prob`と幾何系の設定値はこのパイプラインに届きません。分類は、水平反転に`flip_prob`、垂直反転に`flipud`を使います。セマンティックセグメンテーションのスケールジッターとHSVは設定値ではなくファミリーのクラス属性から決まり、復元の反転は入力とターゲットを組にした操作で、確率は0.5に固定されています。

`no_aug_epochs`はすべてのパイプラインで反映されますが、無効にする対象は、YOLOX方式ではmosaicとMixUp、YOLO-NASではアフィン変換とMixUp、DETR方式では強い測光とクロップのデータ拡張に加えて学習率の終盤、分類では自動データ拡張、消去、MixUp、CutMixと異なります。分類のクロップと反転は有効なままです。

ファミリーが同じパイプラインの他のファミリーと異なる場合、その理由は次のいずれかです。

| ファミリー | 差異 |
|---|---|
| `rtmdet`、`picodet`、`rtdetr`、`rtdetrv2` | `flipud`を無視：変換に垂直反転がない |
| `fomo` | `perspective`と`flipud`を無視 |
| `ppyoloe` | `flipud`を無視：元のレシピでは代わりにランダムな90度回転を使う |
| `yolo9`、`yolo9_e2e`、`yolo9_p2`、`yolo7`、`rtdetr`、`rtdetrv2` | `mixup_scale`を無視：MixUpはフレーム全体をランダムな比率で混合し、どちらもリスケールしない |
| `gtr` | `mosaic_prob`と`mixup_prob`を最初の`mosaic_epochs`エポックの間使用。`degrees`、`translate`、`mosaic_scale`はmosaicに続くアフィン変換を決める |
| `rfdetr` | `hsv_prob`を無視：ネイティブパイプラインにHSVジッターがない |
| `ec` | `hsv_prob`、`degrees`、`translate`を`task="pose"`の場合のみ使用。detectとsegmentは固定の測光レシピを使う |
| `dinov2` | `task="classify"`の場合のみ分類パックを使用 |

`ec`と`dinov2`はマルチタスクのファミリーなので、設定値が無視とマークされるのは、そのファミリーの学習可能なすべてのタスクがそれを無視する場合だけです。これにより、CLIの警告が、あるタスクでは誤りで別のタスクでは正しい、ということが起こりません。

Dome-DETRはD-FINEの変換をそのまま継承します。利用できない唯一のものはマルチスケール学習で、これはデータ拡張の仕様ではなく、自身の設定で無効にされています。

対象は31のファミリーです。この一覧にないファミリーでは`ignored_aug_params`が空の集合を返すため、警告は出ません。

## ファミリー別のデフォルト値

各検出ファミリーの設定の初期値。空欄は、そのファミリーがその設定値を無視するため、デフォルト値が何も変えないことを示します。

<aug-defaults />

個々の値よりも重要なパターンが2つあります。RT-DETR、RT-DETRv2、PicoDet、FOMOは`mosaic_prob=0`から始まるため、これらのファミリーでは、マトリックスでmosaic依存とされている設定値は、mosaicを有効にするまで何もしません。YOLOv9は、回転、せん断、MixUpを無効にし、基本デフォルト値の`(0.1, 2.0)`に対して`mosaic_scale=(0.5, 1.5)`で始まります。

EdgeCrafterは`task="pose"`の場合にだけ`hsv_prob`、`degrees`、`translate`を読み取るため、その行にはこの3つについて姿勢推定のデフォルト値を示しています。

分類ファミリーは基本デフォルト値から始まり、`flip_prob=0.5`、`flipud=0.0`、`scale=(0.5, 1.0)`、ファミリー独自の`crop_pct`で、`auto_augment`、`erasing`、`mixup`、`cutmix`は無効です。

## ファミリー固有の設定値

一部のファミリーは、基底クラスではなく独自の`TrainConfig`サブクラスにデータ拡張の設定値を持ちます。CLIはこれらを公開していないため、Python APIから設定してください。

| ファミリー | 設定値 | 意味 |
|---|---|---|
| `yolo9`、`yolo9_e2e`、`yolo9_p2` | `copy_paste` | copy-pasteインスタンス拡張の確率。`task="segment"`のみ |
| `yolo9`、`yolo9_e2e`、`yolo9_p2` | `copy_paste_mode` | copy-pasteのソース：`flip`は同じサンプルを鏡像化し、`mixup`は2つ目のサンプルを使う |
| `yolo9`、`yolo9_e2e`、`yolo9_p2` | `rot90` | ランダムな90度回転の確率 |
| `rfdetr` | `copy_paste` | `task="segment"`用のcopy-pasteの確率。`flip`モードのみ |
| `rfdetr` | `copy_paste_mode` | `task="segment"`用のcopy-pasteのソースモード |
| `rfdetr` | `crop_resize_prob` | ネイティブパイプラインでのランダムなクロップ・リサイズの確率 |
| `dfine` | `crop_resize_prob` | ランダムなクロップ・リサイズの確率。`task="segment"` |
| `ec` | `crop_resize_prob` | ランダムなクロップ・リサイズの確率。`task="segment"` |
| `ec`、`yolonas` | `brightness_contrast_prob` | 明るさとコントラストのジッターの確率。`task="pose"` |
| `ec`、`yolonas` | `affine_prob` | キーポイント対応アフィン変換の確率。`task="pose"` |

`rot90`は`yolo9`のdetectとOBBに適用されます。

## 仕様の照会

| ヘルパー | 戻り値 |
|---|---|
| `aug_support(family)` | 設定値から`Support`への表。不明なファミリーでは`None` |
| `ignored_aug_params(family)` | ファミリーが無視する設定値名の集合。不明なファミリーでは空 |
| `uses_mosaic_gating(family)` | ファミリーのMixUpがmosaicサンプルでのみ作動するかどうか |
| `display_name(family)` | 警告で使う、人が読むためのファミリー名 |
| `mixup_gating_warning(family, mosaic_prob, mixup_prob)` | MixUpが一切作動できない場合の警告テキスト。それ以外は`None` |

`Support`は`status`と`note`からなる名前付きタプルで、noteはそのファミリーで設定値が無視される、またはmosaicに依存する理由を説明します。1.8.0の仕様はYOLOv9のmosaicラッパーで学習するファミリーについて`mixup_scale`をまだmosaic依存として記載しているため、パイプラインと照合した上のマトリックスでは無視とされているにもかかわらず、`aug_support`はそれらのファミリーでmosaic依存と報告します。

## mosaicゲート

YOLOX方式のファミリーでは、`mosaic_prob=0`とともに`mixup_prob=0.5`を指定すると、MixUpはmosaicサンプルにしか適用されないため、MixUpが完全に無効になります。この組み合わせは、学習の後半でmosaicを無効にするときに簡単に生じます。トレーナーはファミリー名を示す警告をログに記録し、`mixup_gating_warning`はその背後にある純粋関数です。
