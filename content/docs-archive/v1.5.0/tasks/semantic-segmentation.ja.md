---
title: セマンティックセグメンテーション
seo_title: LibreYOLOによるセマンティックセグメンテーション
description: LibreYOLOで各ピクセルにクラスを割り当てます。対応ファミリー、密なマスク形式、推論、学習、検証、エクスポートの呼び出しを解説します。
lead: セマンティックセグメンテーションは画像の各ピクセルにクラスを割り当て、同じクラスのインスタンス同士を区別しません。タスクキーはsemanticです。
keywords:
  - Python セマンティックセグメンテーション
  - ピクセル分類
  - 密な予測
  - セグメンテーションモデル 学習
  - mIoU
  - MIT セグメンテーションライブラリ
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファイル名の-sem接尾辞がタスクを選択するためtask引数は不要
        model = LibreYOLO("LibreSegformerb0-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # 元キャンバス上の(H, W)クラスID
        print(mask.classes)      # 255を除く存在クラスIDを並べ替えたもの
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreSegformerb0-sem.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: クラスごとに処理
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreSegformerb0-sem.pt")(SAMPLE_IMAGE)
        mask = result.semantic_mask

        for class_id in mask.classes:
            pixels = mask.class_mask(class_id)   # boolean (H, W)
            print(result.names[class_id], int(pixels.sum()))
    - label: 別ファミリーでも同じ呼び出し
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePIDNets-sem.pt")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.train(data="my-dataset.yaml", epochs=160, imgsz=512, batch=8)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreSegformerb0-sem.pt data=my-dataset.yaml \
          epochs=160 imgsz=512 batch=8
    - label: ADE20Kで学習
      language: bash
      code: |
        # ade20k.yamlには約1 GBのアーカイブ用ダウンロードスクリプトが
        # 埋め込まれているためデータがローカルになければ明示的な許可が必要
        libreyolo train model=LibreSegformerb0-sem.pt data=ade20k.yaml \
          epochs=160 imgsz=512 batch=8 allow_download_scripts=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")

        # val()はオブジェクトではなく通常のdictを返す
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSegformerb0-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.export(format="onnx", imgsz=512)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSegformerb0-sem.pt format=onnx imgsz=512
    - label: エクスポートしたファイルを使用
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリはファイルの拡張子に応じて振り分けるためエクスポートした
        # アーティファクトもチェックポイントと同様に読み込まれ同じResultsオブジェクトを返す
        model = LibreYOLO("LibreSegformerb0-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: 44b92d8ba6062f04
---

## 定義

セマンティックセグメンテーションは物体ではなくピクセルにラベルを付けます。各ピクセルに1つのクラスIDを割り当てるため、画像内で2台の車が接していると、境界のない1つの車クラス領域になります。インスタンスを数えるのは[インスタンスセグメンテーション](/docs/tasks/instance-segmentation)、すべてのピクセルへラベルを付けながらインスタンスも分離するのは[パノプティックセグメンテーション](/docs/tasks/panoptic-segmentation)です。

`semantic`が標準タスクキーで、チェックポイントのファイル名にある`-sem`接尾辞が選択します。そのため、公開済みの重みを読み込むときに`task=`は不要です。

`predict()`は`result.semantic_mask`を設定します。`.data`は元画像のキャンバス上の`(H, W)`整数クラスマップ、`.classes`は存在するIDを並べ替えたリスト、`.class_mask(id)`は1クラスを選択するブール値の`(H, W)`配列です。値`255`は無視ラベルです。クラスとして扱われず、損失と指標から除外され、`.classes`にも含まれません。

## モデル

学習と推論の両方に対応するファミリーは3つです。[SegFormer](/docs/models/segformer)、[LingBot-Vision](/docs/models/lingbot-vision)、[DINOv2](/docs/models/dinov2)です。SegFormerとLingBot-Visionは基本パッケージで実行でき、公開済みの重みがあります。DINOv2には`pip install "libreyolo[rfdetr]"`が必要で、LibreYOLOがホストするチェックポイントはありません。アップストリームのバックボーンを読み込み、密なヘッドはランダム初期化から始まるため、すぐに使える推論器ではなく学習の出発点です。

さらに4つのファミリーが推論、検証、エクスポートに対応しますが、`train()`は`NotImplementedError`を送出します。[FCN](/docs/models/fcn)、[DeepLabv3](/docs/models/deeplabv3)、[PIDNet](/docs/models/pidnet)、[EoMT](/docs/models/eomt)です。

クラス集合はファミリーではなくチェックポイントごとに異なります。公開済みの重みは、ADE20Kの150クラスやCityscapesの19クラスなど、ほとんど共通点のないラベル空間を持つデータセットから得られています。何にラベルを付けられるかはチェックポイントの`names`で確認し、2つのチェックポイントを比較できるのは同じデータセットで学習した場合だけです。

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

マップはピクセルごとのargmaxなのでNMS処理がなく、`iou`は効果を持ちません。SegFormer、PIDNet、その他の密な予測器では、`conf`と`max_det`をAPI互換性のため受け付けますが何もしません。EoMTは例外で、`conf`がクエリ選択を絞り込みます。入力ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## データセット形式

各画像は`.txt`ラベルファイルではなく、密な単一チャンネルマスクと対応付けます。画像パス内の`images`をマスクディレクトリへ置き換えて検索します。

```text
dataset/
  data.yaml
  images/
    train/000001.jpg
    val/000101.jpg
  masks/
    train/000001.png
    val/000101.png
```

マスクは通常PNGの可逆圧縮単一チャンネル画像です。パレットモードのPNGはパレットインデックスとして読み込まれます。各ピクセル値は`0..nc-1`範囲のクラスID、値`255`は無視を意味します。マスク解像度は対応する画像と一致する必要があります。

YAMLでは共通契約に加えて2つのキーを使用します。

```yaml
path: dataset
train: images/train
val: images/val
masks_dir: masks
nc: 19
names:
  0: road
  1: sidewalk
```

`masks_dir`は`images`の代わりに使うディレクトリ名で、デフォルトは`masks`です。`label_mapping`は任意の`{source_id: train_id}`マッピングで、読み込み時にマスクのピクセル値へ適用されます。1から150の番号を使うデータセットを0から149へ変換する場合などに使用します。マッピングされていないソース値はすべて無視になり、各学習IDは`0..nc-1`範囲でなければなりません。

`masks_dir`を省略すると、ローダーはフォールバックへ切り替わります。通常の`images`から`labels`への規約で解決したポリゴンラベルを読み込み時にラスタライズし、物体クラスの後に`background`クラスを追加するため、`nc`が1増えます。

標準ローダーは`libreyolo.data.SemanticDataset`です。

## 学習

<code-tabs name="train" />

ここでの`imgsz`には検出器にはない制約があります。各ファミリーはパッチグリッドまたは出力ストライドに基づく除数を宣言し、`imgsz`が割り切れない場合は、学習と検証の両方で実行前に`ValueError`を送出します。除数はSegFormerで32、LingBot-VisionとEoMTで16、DINOv2で14、FCNとPIDNetで8です。データセット、データ拡張、マルチGPU、ロガーについては[学習](/docs/train)を参照してください。

## 検証

`val()`は、データセットYAMLの`val`で指定した分割に対して計算した`metrics/`キーを持つ通常の辞書を返します。

<code-tabs name="val" />

`metrics/mIoU`は平均Intersection over Unionです。各クラスについて予測ピクセルと正解ピクセルの共通部分を和集合で割り、クラス間で平均します。主要な数値であり、学習中の最良エポック選択にも使用されます。`metrics/pixel_accuracy`は正しいクラスを割り当てたピクセルの割合ですが、大きな背景クラスによって高く見える場合があるため、比較にはmIoUを使用してください。`255`と記されたピクセルはどちらの指標にも含まれません。辞書にはmIoU値のコピーである`fitness`も含まれます。

## エクスポート

<code-tabs name="export" />

エクスポートしたアーティファクトはファイルの拡張子に基づいて`LibreYOLO()`で再読み込みできるため、`.onnx`や`.engine`ファイルもチェックポイントと同様に動作し、同じ`Results`を返します。対応形式はファミリーごとに異なり、各モデルページのマトリックスは手入力ではなく検証済みの集合から生成されます。形式、追加パッケージ、制約については[エクスポートとデプロイ](/docs/export)を参照してください。
