---
title: 画像分類
seo_title: LibreYOLOによる画像分類
description: >-
  LibreYOLOで画像全体にラベルを付けます。対応ファミリー、ImageFolderデータセットの配置、推論、学習、検証、エクスポートの呼び出しを解説します。
lead: 画像分類は画像全体に1つのラベル分布を割り当て、画像内の位置は特定しません。タスクキーはclassifyです。
keywords:
  - Python 画像分類
  - 画像分類モデル 学習
  - ImageFolder データセット
  - top-1精度
  - ゼロショット分類
  - MIT 画像分類ライブラリ
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファイル名の-cls接尾辞がタスクを選択するためtask引数は不要
        model = LibreYOLO("LibreResNet50-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.names[result.probs.top1], float(result.probs.top1conf))
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreResNet50-cls.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 分布全体
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreResNet50-cls.pt")(SAMPLE_IMAGE)
        probs = result.probs

        # .dataは完全な(C,)ベクトル top5とtop5confは順序付きのビュー
        print(probs.data.shape)
        for index, score in zip(probs.top5, probs.top5conf):
            print(result.names[index], float(score))
    - label: 学習不要のゼロショット
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # CLIPは画像をテキストプロンプトと比較するためラベル集合は

        # チェックポイントに埋め込まず呼び出し時に設定する

        model = LibreYOLO("LibreCLIPb32-cls.pt")

        model.set_classes(["a person jumping", "an empty street", "a parked
        car"])

        result = model(SAMPLE_IMAGE)


        print(model.names[result.probs.top1], float(result.probs.top1conf))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # imagenette160は既知のデータセット名で初回使用時にダウンロードされる
        # 独自データではtrain/分割を含むディレクトリを渡す
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

        # val()はオブジェクトではなく通常のdictを返す
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
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreResNet50-cls.pt format=onnx
    - label: エクスポートしたファイルを使用
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリはファイルの拡張子に応じて振り分けるためエクスポートした
        # アーティファクトもチェックポイントと同様に読み込まれ同じResultsオブジェクトを返す
        model = LibreYOLO("LibreResNet50-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1, result.probs.top1conf)
source_hash: 836bea76cd2cdf92
---

## 定義

画像分類は画像全体に対してクラスごとに1つのスコアを生成し、座標はまったく生成しません。画像に何が写っているかを答え、どこにあるかは答えません。この点が[物体検出](/docs/tasks/object-detection)との違いです。

`classify`が標準タスクキーで、チェックポイントのファイル名にある`-cls`接尾辞が選択します。画像分類ファミリーではこの接尾辞は任意ではなく必須なので、`LibreResNet50.pt`は分類器として読み取られず、`LibreResNet50-cls.pt`だけが該当します。

`predict()`は`result.probs`を設定し、`boxes`を空のままにします。`.data`は完全なスコアベクトル、`.top1`は最高スコアのインデックス、`.top1conf`はその値です。`.top5`は降順の上位5インデックス、`.top5conf`はそのスコアです。インデックスは`result.names`を参照します。`Results`オブジェクトをスライスしても`probs`は切り詰められません。このベクトルは1つの行ではなく画像全体に属するためです。

## モデル

学習と推論の両方に対応するファミリーは5つです。[ResNet](/docs/models/resnet)、[ConvNeXt](/docs/models/convnext)、[MobileNetV4](/docs/models/mobilenetv4)、[EfficientNetV2](/docs/models/efficientnetv2)、[DINOv2](/docs/models/dinov2)です。最初の4つは基本パッケージで実行でき、公開済みの重みがあります。DINOv2には`pip install "libreyolo[rfdetr]"`が必要で、LibreYOLOがホストするチェックポイントはありません。ランダムに初期化された線形ヘッドを持つアップストリームのバックボーンを読み込むため、すぐに使える推論器ではなくファインチューニングの出発点です。

さらに5つのファミリーが推論、検証、エクスポートに対応しますが、`train()`は`NotImplementedError`を送出します。[ViT](/docs/models/vit)、[Swin](/docs/models/swin)、[VGG](/docs/models/vgg)、[AlexNet](/docs/models/alexnet)、[DeiT](/docs/models/deit)です。

[CLIP](/docs/models/clip)と[SigLIP2](/docs/models/siglip2)は固定ラベル集合なしで分類します。画像をテキストプロンプトと比較するため、`set_classes()`で呼び出し時にクラスを定義でき、新しいラベル集合のための学習手順はありません。どちらも`embed`タスクに対応します。

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

ここではしきい値処理や抑制を行う候補はなく、分布が1つだけなので、`conf`、`iou`、`max_det`は効果がありません。入力ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## データセット形式

画像分類ではラベルファイルやYAMLではなくディレクトリツリーを使用し、`data`にはデータセットのルートを指定します。

```text
dataset/
  train/
    tench/000001.jpg
    parachute/000002.jpg
  val/
    tench/000101.jpg
    parachute/000102.jpg
```

学習には`train/`が必須で、並べ替えたフォルダー名によってクラスとインデックスの対応を定義します。そのため、アルファベット順で最初のフォルダーがクラス0になります。検証には`val/`が必須です。`test/`分割が存在しても構いませんが、デフォルトの学習と検証コマンドでは使用しません。`train`以外の分割には、期待されるクラス集合と同じクラスフォルダー名が必要です。この仕組みにより、不一致を誤った予測として採点せず、明示的に失敗させます。対応する画像の拡張子は`.jpg`、`.jpeg`、`.png`、`.bmp`、`.webp`、`.tif`、`.tiff`です。

`data`には3種類の値を指定できます。`train/`分割を含むディレクトリへのパス、`.zip`のURL、または既知のデータセット名である`imagenette160`と`smoke10`です。既知のデータセットは初回使用時にダウンロードされ、キャッシュされます。

標準ローダーは`libreyolo.data.classify_dataset`です。

## 学習

<code-tabs name="train" />

`nc`を宣言する必要はありません。クラス数は`train/`下のフォルダー名から取得され、最終線形レイヤーがその数に合わせて再構築される一方、バックボーンは変更せず転用されます。データセット、データ拡張、マルチGPU、ロガーについては[学習](/docs/train)を参照してください。

## 検証

`val()`はデータセットルートの`val/`分割に対して計算した`metrics/`キーを持つ通常の辞書を返します。

<code-tabs name="val" />

`metrics/accuracy_top1`は、最高スコアのクラスが正解だった画像の割合です。主要な数値であり、学習でも最良エポックの選択に使用します。`metrics/accuracy_top5`は、正解クラスが上位5クラスのどこかに含まれた画像の割合です。データセットのクラス数が少ないほど情報量も少なくなります。辞書にはtop-1値のコピーである`fitness`も含まれます。

## エクスポート

<code-tabs name="export" />

エクスポートしたアーティファクトはファイルの拡張子に基づいて`LibreYOLO()`で再読み込みできるため、`.onnx`や`.engine`ファイルもチェックポイントと同様に動作し、同じ`Results`を返します。対応形式はファミリーごとに異なり、各モデルページのマトリックスは手入力ではなく検証済みの集合から生成されます。形式、追加パッケージ、制約については[エクスポートとデプロイ](/docs/export)を参照してください。
