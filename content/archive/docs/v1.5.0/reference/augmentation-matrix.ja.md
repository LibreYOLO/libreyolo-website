---
title: データ拡張マトリクス
seo_title: LibreYOLOの各ファミリーが反映するデータ拡張
description: >-
  ファミリーごとのデータ拡張ノブ対応状況を示します。16個のTrainConfigノブ、3つの状態、6つのパイプライン構成、およびファミリーが暗黙に無視するノブを扱います。
lead: >-
  データ拡張ノブを設定しても、必ずパイプラインに届くとは限りません。このページでは、ライブラリが唯一の信頼できる情報源として提供する宣言的な表を使い、学習可能な各ファミリーがTrainConfig上の各ノブをどのように扱うかを記録します。
keywords:
  - libreyolo データ拡張
  - mosaic_prob
  - mixup_prob
  - hsv_prob
  - no_aug_epochs
  - データ拡張 対応マトリクス
  - TrainConfig ノブ
last_verified: 1.5.0
verification: >-
  ノブ一覧、状態、構成、ファミリーごとの差異、ヘルパー関数はv1.5.0のlibreyolo/data/augment/spec.pyで確認しました。この表と実際のパイプラインとの一致はtests/unit/test_augment_spec.pyで固定されています。
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
source_hash: d2e1b9f5c81072e1
---

## ノブ

これらはCLIでの表記ではなく、`TrainConfig`のフィールド名です。CLIは固有の別名をこれらにマッピングするため、`--mosaic`は`mosaic_prob`を設定します。

| ノブ | 意味 |
|---|---|
| `mosaic_prob` | 4画像のmosaicサンプルを構築する確率 |
| `mixup_prob` | 2つ目のサンプルを混ぜる確率 |
| `hsv_prob` | HSVカラージッターを行う確率 |
| `flip_prob` | 水平反転の確率 |
| `degrees` | アフィン変換におけるランダム回転範囲（度） |
| `translate` | アフィン変換におけるランダム平行移動の割合 |
| `mosaic_scale` | アフィン変換におけるランダムスケール範囲 |
| `mixup_scale` | MixUp相手の画像に適用するジッタースケール範囲 |
| `shear` | アフィン変換におけるランダムせん断範囲（度） |
| `perspective` | アフィン変換における射影変換の大きさ |
| `flipud` | 垂直反転の確率 |
| `no_aug_epochs` | 強いデータ拡張を無効にして学習する最後のエポック数 |
| `auto_augment` | 分類用AutoAugmentポリシー：randaugment、autoaugment、augmix |
| `erasing` | 分類用RandomErasingの確率 |
| `mixup` | ソフトラベルを使う分類用バッチMixUpの確率 |
| `cutmix` | ソフトラベルを使う分類用バッチCutMixの確率 |

最後の4つが分類パックです。検出ファミリーはこれらを無視します。`mixup`はAPI専用ノブです。CLIの`--mixup`は検出用`mixup_prob`の別名です。

<code-tabs name="usage" />

## 3つの状態

| 状態 | 意味 |
|---|---|
| `used` | ノブがファミリーの学習パイプラインに届き、サンプルを変更 |
| `gated_by_mosaic` | mosaic分岐に入ったサンプルだけにノブを適用。`mosaic_prob == 0`なら一切発動しない |
| `ignored` | ノブはパイプラインに届かず、設定しても何も変わらない |

実行前に確認する価値があるのは`ignored`です。何も失敗しないためです。明示的に設定した学習パラメータを選択したファミリーが無視する場合、CLIは警告します。ファミリーがMixUpをmosaicに依存させ、`mosaic_prob`がゼロのため`mixup_prob > 0`が発動できない場合も、トレーナーが警告します。

## パイプライン構成

対象となるすべてのファミリーは、以下に示す少数のファミリー固有差を除き、6つのパイプラインのいずれかに従います。

| ノブ | YOLOX形式 | YOLO-NAS | DETR形式 | 分類 | セマンティック | 復元 |
|---|---|---|---|---|---|---|
| `mosaic_prob` | 使用 | 無視 | 無視 | 無視 | 無視 | 無視 |
| `mixup_prob` | mosaic依存 | 使用 | 無視 | 無視 | 無視 | 無視 |
| `hsv_prob` | 使用 | 使用 | 無視 | 無視 | 無視 | 無視 |
| `flip_prob` | 使用 | 使用 | 使用 | 無視 | 無視 | 無視 |
| `degrees` | mosaic依存 | 使用 | 無視 | 無視 | 無視 | 無視 |
| `translate` | mosaic依存 | 使用 | 無視 | 無視 | 無視 | 無視 |
| `mosaic_scale` | mosaic依存 | 使用 | 無視 | 無視 | 無視 | 無視 |
| `mixup_scale` | mosaic依存 | 使用 | 無視 | 無視 | 無視 | 無視 |
| `shear` | mosaic依存 | 使用 | 無視 | 無視 | 無視 | 無視 |
| `perspective` | mosaic依存 | 使用 | 無視 | 無視 | 無視 | 無視 |
| `flipud` | 使用 | 使用 | 無視 | 無視 | 無視 | 無視 |
| `no_aug_epochs` | 使用 | 使用 | 使用 | 使用 | 使用 | 使用 |
| `auto_augment` | 無視 | 無視 | 無視 | 使用 | 無視 | 無視 |
| `erasing` | 無視 | 無視 | 無視 | 使用 | 無視 | 無視 |
| `mixup` | 無視 | 無視 | 無視 | 使用 | 無視 | 無視 |
| `cutmix` | 無視 | 無視 | 無視 | 使用 | 無視 | 無視 |

YOLOX形式のパイプラインでは、サンプルごとの前処理がHSVジッターと反転を適用し、アフィン変換とMixUpはmosaic分岐の中だけで実行されます。一方、YOLO-NASは常に有効なサンプルごとのアフィン変換を実行し、mosaicを無視し、MixUpを独立して適用します。`mosaic_scale`はアフィン変換のスケール範囲として再利用されます。

DETR形式のパイプラインはmosaicなしの通過型変換です。測光変形、zoom-out、IoU-cropは設定可能なノブではなくレシピ定数なので、`hsv_prob`と幾何学ノブは一切届きません。分類パイプラインは`flip_prob`ではなく固定値0.5の水平反転を持つImageFolder変換を使います。セマンティックのスケールジッターとHSVは設定ノブではなくファミリーのクラス属性に由来し、復元の反転は入力とターゲットを連動させる固定確率0.5の処理です。

`no_aug_epochs`はすべての場所で反映されますが、無効にするものは異なります。YOLOX形式ではmosaicとMixUp、YOLO-NASではアフィン変換とMixUp、DETR形式では強い測光拡張とcrop拡張および学習率の末尾、そのほかではスケジューラーの末尾です。

## 構成別のファミリー

| 構成 | ファミリー |
|---|---|
| YOLOX形式 | `yolox`、`yolo7`、`yolo9`、`yolo9_e2e`、`yolo9_p2`、`rtmdet`、`picodet`、`rtdetr`、`rtdetrv2`、`fomo` |
| YOLO-NAS | `yolonas` |
| DETR形式 | `dfine`、`domedetr`、`deim`、`deimv2`、`rtdetrv4`、`rfdetr`、`ec`、`dinov2` |
| 分類 | `resnet`、`convnext`、`mobilenetv4`、`efficientnetv2` |
| セマンティック | `segformer` |
| 復元 | `nafnet` |

25のファミリーが対象です。この一覧にないファミリーでは無視される集合が空で返されるため、警告は表示されません。

## 差異

| ファミリー | 構成との差異 |
|---|---|
| `rtmdet` | `flipud`を無視。変換に垂直反転がない |
| `picodet` | `flipud`を無視 |
| `rtdetr` | `flipud`を無視 |
| `rtdetrv2` | `flipud`を無視 |
| `fomo` | `perspective`と`flipud`を無視 |
| `ec` | `task="pose"`の場合だけ`hsv_prob`、`degrees`、`translate`を使用。detectとsegmentでは固定の測光レシピを使用 |
| `dinov2` | `task="classify"`の場合だけ分類パックを使用 |

`ec`と`dinov2`はマルチタスクファミリーなので、ファミリーの学習可能なすべてのタスクが無視する場合だけノブを無視とマークします。これにより、あるタスクでは誤りで別のタスクでは正しいCLI警告が出ることはありません。

Dome-DETRはD-FINEの変換を変更せずに継承します。利用できないのはマルチスケール学習だけで、データ拡張仕様ではなく設定によって無効にされます。

## ファミリー固有のノブ

一部のファミリーは、基本クラスではなく固有の`TrainConfig`サブクラスにデータ拡張ノブを持ちます。CLIはこれらを公開しないため、Python APIから設定してください。

| ファミリー | ノブ | 意味 |
|---|---|---|
| `yolo9`、`yolo9_e2e`、`yolo9_p2` | `copy_paste` | `task="segment"`だけで使うcopy-pasteインスタンス拡張の確率 |
| `yolo9`、`yolo9_e2e`、`yolo9_p2` | `copy_paste_mode` | copy-pasteのソース。`flip`は同じサンプルを鏡像化し、`mixup`は2つ目のサンプルを使用 |
| `yolo9`、`yolo9_e2e`、`yolo9_p2` | `rot90` | ランダムな90度回転の確率 |
| `rfdetr` | `copy_paste` | `task="segment"`向けcopy-pasteの確率。`flip`モードだけ |
| `rfdetr` | `copy_paste_mode` | `task="segment"`向けcopy-pasteソースモード |
| `rfdetr` | `crop_resize_prob` | ネイティブパイプライン内のランダムcrop-resize確率 |
| `dfine` | `crop_resize_prob` | `task="segment"`向けランダムcrop-resize確率 |
| `ec` | `crop_resize_prob` | `task="segment"`向けランダムcrop-resize確率 |
| `ec`、`yolonas` | `brightness_contrast_prob` | `task="pose"`向け明るさとコントラストのジッター確率 |
| `ec`、`yolonas` | `affine_prob` | `task="pose"`向けキーポイント対応アフィン変換の確率 |

`rot90`は`yolo9`のdetectとOBBに適用されます。

## 仕様の照会

| ヘルパー | 返り値 |
|---|---|
| `aug_support(family)` | ノブから`Support`への表。不明なファミリーでは`None` |
| `ignored_aug_params(family)` | ファミリーが無視するノブ名の集合。不明なファミリーでは空 |
| `uses_mosaic_gating(family)` | ファミリーのMixUpがmosaicサンプルだけで発動するか |
| `display_name(family)` | 警告で使うユーザー向けファミリー名 |
| `mixup_gating_warning(family, mosaic_prob, mixup_prob)` | MixUpが一切発動できない場合の警告テキスト。それ以外は`None` |

`Support`は`status`と`note`からなる名前付きタプルで、noteはそのファミリーでノブが無視される、またはmosaicに依存する理由を説明します。

## mosaicゲート

YOLOX形式のファミリーでは、`mosaic_prob=0`で`mixup_prob=0.5`を指定するとMixUpが完全に無効になります。MixUpはmosaicサンプルだけに適用されるためです。この組み合わせは学習後半でmosaicを無効にすると簡単に発生します。トレーナーはファミリー名を示す警告をログに記録し、その背後にある純粋関数が`mixup_gating_warning`です。

