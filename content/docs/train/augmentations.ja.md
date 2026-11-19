---
title: データ拡張
seo_title: LibreYOLOの学習用データ拡張（例付き）
description: >-
  LibreYOLOの各データ拡張設定値が何をするかを、ライブラリ自身の学習コードで実際の画像に適用して示し、範囲、デフォルト値、その設定値を読み取るファミリーも記載します。
lead: >-
  学習用データ拡張は、すべての学習サンプルに適用されるランダムな変換です。TrainConfigの設定値で構成し、各モデルファミリーは独自のパイプラインを実行するため、設定値がサンプルを変えるのは、そのパイプラインが設定値を読み取る場合だけです。
keywords:
  - yolo データ拡張
  - mosaic データ拡張
  - mixup
  - hsv jitter
  - random affine
  - random erasing
  - 小物体検出
  - データ拡張 例
  - copy paste データ拡張
  - randaugment
  - cutmix
  - no_aug_epochs
last_verified: 1.8.0
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
  small:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # YOLOXの初期値mosaic_scale=(0.1, 2.0)では物体が元の10分の1まで縮小しうる
        # (0.5, 1.5)なら縮小は半分まで
        model = LibreYOLO("LibreYOLOXs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            mosaic_scale=(0.5, 1.5),
        )
    - label: CLI
      language: bash
      code: |
        # CLIでは、ペアをカンマ区切りの2つの数値で指定
        libreyolo train model=LibreYOLOXs.pt data=my-dataset.yaml \
          epochs=100 mosaic_scale=0.5,1.5
  support:
    - label: ファミリーの対応表を確認
      language: python
      code: |
        from libreyolo.data.augment.spec import AUG_KNOBS, aug_support

        for knob, description in AUG_KNOBS.items():
            support = aug_support("yolo9")[knob]
            print(f"{knob:16} {support.status:16} {support.note or description}")
    - label: 無視される設定値のみ
      language: python
      code: |
        from libreyolo.data.augment.spec import ignored_aug_params

        print(sorted(ignored_aug_params("rfdetr")))
  classify:
    - label: 分類パック
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
source_hash: 72d22a38d647de0a
---

## 設定値の指定

データ拡張の設定値は、通常の`train()`引数です。

<code-tabs name="train" />

このうち3つには短い綴りがあり、`train()`とCLIのどちらでも使えます。`mosaic`は`mosaic_prob`に、`fliplr`は`flip_prob`に対応し、物体検出では`mixup`が`mixup_prob`に対応します。短い綴りとそのフィールドを異なる値で両方渡すと、例外が送出されます。1.8.0では`perspective`にCLIフラグがないため、Pythonから設定します。

分類用の設定値はパイプラインの構築時に検証され、CLIはさらに`flip_prob`、`flipud`、`cutmix`、`erasing`も検査します。`TrainConfig`は検出用の設定値の範囲を検査しないため、`degrees=100`では最大100度の回転で学習します。各設定値の範囲は[データ拡張マトリックス](/docs/reference/augmentation-matrix)に記載しています。

## 2つではなく3つの状態

設定値が効果を持つかどうかは、ファミリーによって異なります。ライブラリはこれを宣言的な表として保持しており、各項目は3つの状態のいずれかです。

`used`は、設定値がパイプラインに届いてサンプルを変えることを意味します。`ignored`は、設定値がパイプラインに届かないため、設定しても何も起きないことを意味します。`gated_by_mosaic`は、mosaic分岐を通ったサンプルにだけ適用されることを意味し、配線されていても`mosaic_prob=0`では一度も作動しません。

意外に思われやすいのは、3つ目の状態です。YOLOX方式のパイプラインではアフィン変換がmosaicキャンバス上で実行され、MixUpはmosaicサンプルを混合するため、`mosaic_prob=0`にすると`degrees`、`translate`、`shear`、`perspective`、`mosaic_scale`、`mixup_prob`、`mixup_scale`がまとめて通知なく無効になります。トレーナーは、MixUpの場合に限って次の警告をログに記録します。

```text
mixup_prob=0.15 has no effect for YOLOv9: mixup only applies to mosaic samples
and mosaic_prob=0. Set mosaic_prob > 0 to enable mixup.
```

CLIも無視される設定値について警告し、実際に入力したものだけを一覧にします。

```text
Warning: RF-DETR ignores these parameters: degrees, mosaic
```

Pythonから呼び出した場合、`train()`は無視される設定値について警告しません。[データ拡張マトリックス](/docs/reference/augmentation-matrix)には、すべてのファミリーにおける各設定値の状態と、各ファミリーの初期値を記載しています。この表はライブラリに同梱された`libreyolo/data/augment/spec.py`の仕様そのもので、ライブラリ自身のテストがパイプラインと照合しており、次のように直接読み取ることもできます。

<code-tabs name="support" />

## 各設定値の働き

以下の図は、LibreYOLO 1.8.0自身の学習用変換を、1枚の画像だけのデータセットを通してサンプル画像に適用して生成したもので、mosaicの各タイルもMixUpの各相手もすべて同じ写真です。ボックスは、LibreYOLO9tがこの画像で検出した人物です。各パネルの下には、設定と最大ボックスのサイズを、640ピクセルの学習解像度でのピクセル数で示しています。データ拡張なしでは、人物のサイズは291 × 217 pxです。

幾何系の設定値のいくつかは範囲で、`degrees=10`は-10〜+10度のどこかから抽出した角度を意味します。これらのパネルでは抽出値をキャプションの値に固定しているため、典型的なサンプルではなく、その設定で起こりうる最も強い効果を示しています。検出の図はYOLOv9のパイプラインによるものです。YOLOv7はこのパイプラインでそのまま学習し、RT-DETRとRT-DETRv2はそのmosaicラッパーを独自のサンプルごとの変換と組み合わせて使います。YOLOX、RTMDet、PicoDet、FOMOは、別のラッパーでmosaicとアフィン変換を同じ方法で構築しますが、そのラッパーのMixUpは異なります。

### mosaic_prob

サンプルがmosaicになる確率：mosaicは入力サイズの2倍のキャンバスに4枚の画像を配置したもので、これをアフィン変換が入力サイズまで縮小して写します。YOLOX方式のパイプラインでは、アフィン変換とMixUpはmosaicサンプルでのみ実行されるため、この値が0だと両方とも停止します。RT-DETR、RT-DETRv2、PicoDet、FOMOのデフォルトは0です。

<aug-figure name="mosaic" />

2枚目のパネルでは、アフィン変換のスケールを、キャンバス全体が収まる0.5×に固定しています。最後の2枚はYOLOv9のデフォルト値による通常の抽出で、キャプションにはそれぞれが抽出したスケールを示しています。

### mosaic_scale

アフィン変換の拡大縮小率を抽出する範囲。1×では物体は推論時と同じサイズのままで、0.5×ではその半分になります。基本デフォルト値は`(0.1, 2.0)`で、YOLOX、YOLOv7、RTMDet、PicoDet、RT-DETR、RT-DETRv2、FOMOはこれをそのまま使うため、物体が元の10分の1のサイズまで縮小することがあります。YOLOv9のデフォルトは`(0.5, 1.5)`です。YOLO-NASとPP-YOLOEにはmosaicがなく、この範囲をサンプルごとのアフィン変換のスケールとして再利用します。

<aug-figure name="mosaic_scale" />

1×を超えると出力にはキャンバスの左上部分しか映らないため、物体は端で切れるか、フレームの外に出ます。

### degrees

回転の範囲：アフィン変換される各サンプルは、`-degrees`から`+degrees`までの範囲で抽出した角度だけ回転します。回転の中心はキャンバスの左上隅なので、角度が大きいと画像の大部分がフレームの外に出ます。YOLOXとYOLOv7は基本デフォルト値の10をそのまま使い、YOLOv9のデフォルトは0です。

<aug-figure name="degrees" />

### translate

入力の辺に対する割合で表した平行移動の範囲：`imgsz=640`では、アフィン変換される各サンプルが各軸で最大`translate`×640 px移動します。基本デフォルト値は0.1です。YOLO-NASとPP-YOLOEのデフォルトは0.25、RTMDet、PicoDet、FOMOのデフォルトは0です。

<aug-figure name="translate" />

### shear

軸ごとに個別に抽出される、度単位のせん断範囲。YOLOXとYOLOv7は基本デフォルト値の2をそのまま使い、これを読み取る他のすべてのファミリーのデフォルトは0です。

<aug-figure name="shear" />

### perspective

回転、スケール、せん断、平行移動に重ねる射影変換。キャンバスの各隅は軸ごとに、最大でキャンバスの辺の半分の`min(perspective * 100, 0.2)`倍のランダムな量だけ内側へ移動するため、`perspective=0.001`はどの解像度でも隅を最大で辺の5パーセントだけ動かし、`0.002`以上の値はすべて同じ最大値に達します。デフォルトの`0`では変換はアフィンのままで、負の値を指定すると、最初のサンプルを変換するときに`ValueError`が送出されます。パネルは1回のランダムな抽出を共有し、値に応じてスケールしています。YOLOv9とそのE2EおよびP2バリアント、YOLOX、YOLOv7、YOLO-NAS、PP-YOLOE、RTMDet、PicoDet、RT-DETR、RT-DETRv2がこれを適用します。

<aug-figure name="perspective" />

### hsv_prob

HSVカラージッターの確率。設定値になっているのは確率だけです。ジッターは、OpenCVの0〜179の色相スケールで色相を最大5、0〜255のスケールで彩度と明度を最大30ずらし、3つのずらしはそれぞれ確率2分の1でスキップされます。`hsv_prob`を読み取る検出パイプラインはすべて、これらの量を使います。最後の3枚のパネルは、119回の抽出のうち最も変化がわかりやすいものです。

<aug-figure name="hsv" />

### flip_probとflipud

水平反転と垂直反転の確率。`flip_prob`はFOMOを除くすべてのファミリーで0.5から始まり、FOMOのデフォルトは0です。`flipud`は0から始まります。RTMDet、PicoDet、RT-DETR、RT-DETRv2、FOMO、PP-YOLOE、およびDETR方式のファミリーには垂直反転がなく、これを無視します。PP-YOLOEのレシピでは、代わりにランダムな90度回転を使います。

<aug-figure name="flip" />

### mixup_probとmixup_scale

MixUpは、2つ目のサンプルを現在のサンプルに混ぜ合わせ、両方のボックスを保持します。YOLOv9のパイプラインでは、相手はmosaicなしのサンプル全体で、混合比率は0〜1の間で一様に抽出され、`mixup_scale`は、ライブラリの仕様ではmosaicに依存すると記載されているものの、一度も読み取られません。YOLOX、RTMDet、PicoDet、FOMOが使うYOLOXラッパーと、YOLO-NASのラッパーでは、相手をまず`mixup_scale`から抽出した倍率でリスケールし、2枚の画像を固定の0.5で平均します。`mixup_prob`の基本デフォルト値は1.0で、YOLOv9のデフォルトは0です。

<aug-figure name="mixup" />

最後の2枚のパネルは、2つ目のサンプルを0.5×のmosaicに混合しています。そのキャプションには、mosaicサンプルの重みである混合比率λを示しています。

### no_aug_epochs

`no_aug_epochs`はどこでも`used`ですが、どこでも同じ意味というわけではありません。mosaicパイプラインでは、最後のエポックでmosaicとMixUpを無効にします。DETR方式のパイプラインでは、測光、ズームアウト、クロップのデータ拡張を停止し、スケジュールの終盤も形作ります。分類パイプラインでは、自動データ拡張、消去、MixUp、CutMixを無効にし、クロップと反転は残ります。

## DETR方式のレシピ

D-FINE、Dome-DETR、DEIM、DEIMv2、RT-DETRv4、TinyFormerは、mosaicもアフィン変換もない1つのtorchvision変換で学習し、EdgeCrafter（EC）も検出では同様です。その3つの強い操作は、設定値ではなくレシピの定数です。

- `RandomPhotometricDistort`は、各色変化（明るさ0.875〜1.125、コントラストと彩度0.5〜1.5、色相±0.05、チャネルのシャッフル）をそれぞれ確率0.5で適用します。
- `RandomZoomOut`は確率0.5で、画像を最大で辺の4倍の黒いキャンバスに配置するため、物体が元の4分の1のサイズまで縮小することがあります。
- `RandomIoUCrop`は確率0.8で適用され、各辺の0.3〜1.0倍の領域を、ボックスの少なくとも一部が残るように選んでクロップします。

これらの処理の後、いずれかの辺が1 px以下のボックスは破棄されます。3つの操作はいずれもファミリーの停止エポックで停止し、遅くとも最後の`no_aug_epochs`が始まる時点で停止します。このパイプラインが読み取る設定値は`flip_prob`と`no_aug_epochs`だけです。

<aug-figure name="detr_ops" />

3つの操作がすべて有効な状態での、D-FINE学習の開始直後の3サンプル：

<aug-figure name="detr_samples" />

RF-DETRのパイプラインは独自のもので、水平反転を行い、次に確率`crop_resize_prob`で短辺を400、500、600 pxのいずれかにリサイズして384〜600 pxの正方形をランダムにクロップし、最後に正方形の学習解像度へ直接リサイズして、レターボックスではなく引き伸ばします。`crop_resize_prob`のデフォルトは0.5で、Pythonから設定します。

<aug-figure name="rfdetr_crop" />

## 小さな物体

`erasing`は物体検出には適用されません。Random Erasingは分類パックに属し、消去の処理を持つ検出パイプラインはないため、どの値でも検出器の学習はまったく同じで、無効にするものもありません。小さな物体を縮小または除去するのは、次の処理です。

- `mosaic_scale`。mosaicパイプラインでは物体のサイズに抽出されたスケールが掛かるため、下限で最小サイズが決まります。基本デフォルト値`(0.1, 2.0)`では、20 pxの物体が2 pxになることがあります。その後、YOLOv9のパイプラインはいずれかの辺が2 px以下のボックスを破棄し、その最終変換もYOLOXと同様に、いずれかの辺が1 px以下のボックスを破棄します。物体のピクセルはラベルのないまま画像に残ります。YOLOv9の`(0.5, 1.5)`のように下限を0.5にすると、すべての物体が推論時のサイズの半分以上に保たれます。
- `degrees`、`translate`、`shear`、`perspective`は物体を端へ押し出し、そこでボックスがクリップされて同じサイズフィルターを通ります。
- DETR方式のレシピでは、`RandomZoomOut`が物体を元の4分の1のサイズまで縮小し、`RandomIoUCrop`が物体を切り取ってしまうことがあります。どちらも設定値ではなく、どちらも停止エポックで停止します。
- RF-DETRのクロップは、その端で物体を切ることがあります。`crop_resize_prob=0`で無効になります。
- 学習解像度はすべての物体の大きさに影響し、レターボックスは`imgsz`を画像の長辺で割った倍率で画像をリサイズするため、`imgsz=1280`ではすべての物体の幅と高さが640のときの2倍になります。[YOLOv9](/docs/models/yolov9)には、小さな物体向けにストライド4のヘッドを持つP2バリアントもあります。

<code-tabs name="small" />

## 分類パック

6つの設定値は分類パイプラインにだけ作用します。検出ファミリーは6つすべてを無視します。

<code-tabs name="classify" />

`scale`はランダムクロップの面積範囲です。`scale=0.5`は`(0.5, 1.0)`を意味し、ペアを明示すると両方の境界を設定します。3枚目のパネルのように、下限が小さいと物体が完全に切り取られることがあります。

<aug-figure name="cls_crop" />

`auto_augment`は`"randaugment"`、`"autoaugment"`、`"augmix"`、`None`のいずれかを受け取り、各サンプルにそのポリシーから1回抽出した変換を適用します。

<aug-figure name="auto_augment" />

`erasing`はRandom Erasingの確率で、範囲は`[0, 1)`です。正規化の後に実行され、画像の2〜33パーセントを覆う矩形を、正規化後にはImageNetの平均色にあたるゼロで埋めます。

<aug-figure name="erasing" />

`mixup`と`cutmix`は、ソフトラベルを生成するバッチごとの確率です。1つのバッチで実行されるのは多くとも1つで、MixUpが先に判定されるため、2つの確率は加算され、合計は1以下でなければなりません。パネルのAとBは同じ写真の2つのクロップで、各キャプションには、混合した画像の学習に使うソフトラベルを示しています。

<aug-figure name="batch_mix" />

`auto_augment`、`erasing`、`mixup`、`cutmix`はデフォルトで無効なので、明示的に指定しない限り分類の学習は変わりません。`train()`とCLIはタスクに応じて`mixup`を振り分け、分類ではバッチの混合、検出では`mixup_prob`を使います。

`crop_pct=None`は、ファミリーの評価用リサイズ比率を維持します。上書きすると、学習中に実行される検証の評価用クロップが変わりますが、エクスポートではファミリー本来の前処理が維持されます。

CLIは`auto_augment`、`erasing`、`cutmix`、`scale`、`crop_pct`、`fliplr`、`flipud`を公開しています。分類の`mixup`はバッチの混合を意味し、デフォルトは0.0です。水平反転のデフォルトは0.5、垂直反転は0.0です。

`no_aug_epochs`は最後のエポックで自動データ拡張、消去、MixUp、CutMixを無効にし、クロップと反転は維持します。分類レシピでは、この末尾期間のデフォルトは0です。

## ファミリー固有の設定値

一部の設定値は基底クラスではなくファミリーの設定サブクラスにあるため、そのファミリーにだけ存在し、CLIフラグはありません。

| ファミリー | 設定値 | 効果 |
|---|---|---|
| YOLOv9、YOLOv9-E2E、YOLOv9-P2 | `copy_paste` | copy-pasteインスタンス拡張の確率。`task="segment"`のみ |
| YOLOv9、YOLOv9-E2E、YOLOv9-P2 | `copy_paste_mode` | `"flip"`は同じサンプルを鏡像にして再利用し、`"mixup"`は2つ目のサンプルを取得 |
| YOLOv9、YOLOv9-E2E、YOLOv9-P2 | `rot90` | ランダムな90度回転の確率 |
| YOLOv9、YOLOv9-E2E、YOLOv9-P2 | `max_labels` | 学習用変換における画像ごとの正解データ（ground truth）の上限。デフォルトは300 |
| RF-DETR | `copy_paste`、`copy_paste_mode` | `task="segment"`用のcopy-paste。`"flip"`モードのみ |
| RF-DETR | `crop_resize_prob` | ネイティブパイプラインでのランダムなクロップ・リサイズの確率。デフォルトは0.5 |
| D-FINE、EdgeCrafter | `crop_resize_prob` | ランダムなクロップ・リサイズの確率。`task="segment"`のみ |
| EdgeCrafter、YOLO-NAS | `brightness_contrast_prob`、`affine_prob` | 姿勢推定経路のジッターと、キーポイント対応アフィン変換の確率 |

データを失うのは`max_labels`です。上限を超えたボックスはエラーではなく1回だけの警告とともに破棄されるため、航空写真のような密な画像では値を上げる必要があります。ラベル割り当てのメモリは、上限ではなくバッチ内で最も密な画像に従います。

回転バウンディングボックスの学習では、回転したボックスの隅を考慮するデータ拡張が実装されていないため、設定値にかかわらずmosaicとMixUpが無効になります。

## 関連項目

- 各ファミリーにおける全設定値の状態、範囲、各ファミリーのデフォルト値については[データ拡張マトリックス](/docs/reference/augmentation-matrix)を参照してください。
- スケジュール引数としての`no_aug_epochs`と、`train()`のその他の引数については[ハイパーパラメータ](/docs/train/hyperparameters)を参照してください。
- これらの変換が使うラベル形式については[データセット](/docs/train/datasets)を参照してください。
