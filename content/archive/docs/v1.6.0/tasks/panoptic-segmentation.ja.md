---
title: パノプティックセグメンテーション
seo_title: LibreYOLOによるパノプティックセグメンテーション
description: >-
  LibreYOLOで各ピクセルに1つのセグメントを割り当てます。対応ファミリー、COCO-panopticデータセット形式、推論と検証の呼び出しを解説します。
lead: >-
  パノプティックセグメンテーションは、数えられる物体インスタンスと不定形の背景領域を統合し、各ピクセルを重なりのない1つのセグメントへ割り当てます。タスクキーはpanopticです。
keywords:
  - Python パノプティックセグメンテーション
  - Panoptic Quality
  - things stuff セグメンテーション
  - COCO panoptic 形式
  - セグメントIDマップ
  - PQ 指標
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファイル名の-panoptic接尾辞がタスクを選択するためtask引数は不要
        model = LibreYOLO("LibreEoMTl-panoptic.pt")
        result = model(SAMPLE_IMAGE, save=True)

        pan = result.panoptic
        print(pan.data.shape)       # (H, W) セグメントID
        print(pan.segments_info)    # [{"id": ..., "category_id": ...}, ...]
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreEoMTl-panoptic.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: セグメントごとに処理
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreEoMTl-panoptic.pt")(SAMPLE_IMAGE)
        pan = result.panoptic

        for segment in pan.segments_info:
            pixels = pan.segment_mask(segment["id"])   # boolean (H, W)
            print(result.names[segment["category_id"]], int(pixels.sum()))
    - label: 小型チェックポイント
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEoMTs-panoptic.pt")
        result = model(SAMPLE_IMAGE)

        print(len(result.panoptic.segment_ids))
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-panoptic.pt")

        # val()はオブジェクトではなく通常のdictを返す
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PQ"])
        print(metrics["metrics/SQ"], metrics["metrics/RQ"])
        print(metrics["metrics/PQ_things"], metrics["metrics/PQ_stuff"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEoMTl-panoptic.pt data=my-dataset.yaml
source_hash: b8adc9ccde7a4e6c
---

## 定義

パノプティックセグメンテーションは他の2つのセグメンテーションタスクを統合したものです。各ピクセルは正確に1つのセグメントへ割り当てられ、セグメントは重なりません。セグメントはthing（数えられる物体インスタンス）か、stuff（空や道路などの不定形領域）のどちらかです。そのため、背景ピクセルを未割り当てのままにし、マスクの重なりを許す[インスタンスセグメンテーション](/docs/tasks/instance-segmentation)より厳密です。また、各ピクセルにラベルを付けるものの、同じクラスの接するインスタンスを統合する[セマンティックセグメンテーション](/docs/tasks/semantic-segmentation)よりも厳密です。

`panoptic`が標準タスクキーで、チェックポイントのファイル名にある`-panoptic`接尾辞が選択します。そのため、公開済みの重みを読み込むときに`task=`は不要です。

`predict()`は`result.panoptic`を設定します。`.data`は元画像のキャンバス上の`(H, W)`整数セグメントIDマップです。`.segments_info`はセグメントごとの辞書リストで、各辞書には少なくとも`{"id", "category_id"}`が含まれます。`id`はマップ内の値と一致し、`category_id`は`result.names`を参照します。`.segment_ids`は存在するIDを並べ替えたリスト、`.segment_mask(id)`は1つのセグメントを選択するブール値の`(H, W)`配列です。セグメントID`0`はvoid値です。ラベルのないピクセルとして指標から除外され、`.segment_ids`にも含まれません。

thingとstuffの区別は個々のセグメントではなくカテゴリーの属性です。ラベル集合のカテゴリーメタデータに保持され、推論ペイロードが便宜上各セグメントへ`"isthing"`としてコピーする場合もありますが、正式な情報源はカテゴリーメタデータです。

## モデル

[EoMT](/docs/models/eomt)は`LibreYOLO()`を通してこのタスクに対応するファミリーです。基本パッケージで実行でき、COCOで学習したs、b、lの3サイズのパノプティックチェックポイントを提供します。

[SenseNova-Vision](/docs/models/sensenova-vision)もパノプティックマップを出力します。固有のファクトリ`LibreVLM`と固有の追加パッケージを持つプロンプト生成モデルです。ボキャブラリーを設定しない場合は、ファインチューニングに使ったCOCO panopticカテゴリーへフォールバックします。重みは非商用です。推論ごとに拡散デコードを実行するため、画像ごとのレイテンシは専用セグメンターよりはるかに長くなります。

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

`conf`はクエリ選択を絞り込みます。入力ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## データセット形式

LibreYOLOはKirillovら（CVPR 2019）のCOCO-panoptic形式をそのまま採用しています。LibreYOLO固有のパノプティック配置はありません。

```text
dataset/
  data.yaml
  images/
    val/000000000139.jpg
  annotations/
    panoptic_val.json
    panoptic_val/000000000139.png
```

各画像は同じ解像度の1枚のRGB PNGと対応付けられ、各ピクセルの色が所属するセグメントのIDをエンコードします。

```text
segment_id = R + 256 * G + 256 * 256 * B
```

セグメントID`0`、つまりRGBの黒はvoidです。ラベルのないピクセルであり、推論結果への加点も減点もありません。それ以外の各ピクセルは正確に1つのセグメントへ属します。

JSONには画像ごとにセグメントID PNGと、その中のセグメントを記述します。

```json
{
  "images":      [{"id": 139, "file_name": "000000000139.jpg"}],
  "annotations": [{"image_id": 139, "file_name": "000000000139.png",
                   "segments_info": [
                     {"id": 3226956, "category_id": 1, "area": 2840,
                      "bbox": [413, 158, 53, 138], "iscrowd": 0}]}],
  "categories":  [{"id": 1, "name": "person", "isthing": 1}]
}
```

`annotations[].file_name`はパノプティックディレクトリ内のPNG名で、`segments_info[].id`はそのPNG内の値と一致します。`iscrowd`はグループ領域を示します。偽陰性として数えられることはなく、領域の大半を覆う推論も偽陽性になりません。`isthing`は`categories`にあり、個々のセグメントにはありません。

YAMLは両方を参照します。

```yaml
path: dataset
val: images/val
annotations:
  val: annotations/panoptic_val.json
panoptic_dir:
  val: annotations/panoptic_val
names:
  0: person
  1: bicycle
```

`annotations`と`panoptic_dir`はそれぞれ単一パスまたは分割ごとのマッピングを受け付けます。生のCOCOカテゴリーIDは通常連続していませんが、モデルは連続する`0..nc-1`を予測するため、カテゴリー名に基づいて`names`を通してIDを再マッピングします。`names`にないJSONカテゴリーは暗黙に破棄せずエラーになります。破棄すると恒常的な偽陰性として採点されるためです。

標準ローダーは`libreyolo.data.PanopticDataset`です。

## 学習

現在、LibreYOLO内でパノプティックセグメンテーションを学習できるファミリーはありません。EoMTの`train()`は`NotImplementedError`を送出するため、パノプティックチェックポイントは公開済みの状態で使用します。

## 検証

`val()`はデータセットYAMLの`val`で指定した分割に対して、正解解像度で計算した`metrics/`キーを持つ通常の辞書を返します。同じカテゴリーの予測セグメントと正解セグメントはIoUが0.5を超えると一致し、その対応は一意です。

<code-tabs name="val" />

`metrics/PQ`はPanoptic Qualityであり、主要な数値です。1つのカテゴリー内では2つの要素の積です。Segmentation Qualityは一致したセグメント間の平均IoUで、対応した形状がどれだけ揃っているかを示します。Recognition Qualityは`TP / (TP + 0.5 FP + 0.5 FN)`で、対応自体のF1値です。見つけられたセグメント数を示します。この3つの数値は出現したカテゴリー間で平均され、`metrics/PQ`、`metrics/SQ`、`metrics/RQ`として報告されます。そのため、報告されるPQはカテゴリーごとの積の平均であり、報告された2つの平均値の積ではありません。

`metrics/PQ_things`と`metrics/PQ_stuff`は同じカテゴリーごとのPQをthingカテゴリーとstuffカテゴリーで別々に平均します。`metrics/categories`は出現し、平均対象になったカテゴリー数です。辞書にはPQ値のコピーである`fitness`も含まれます。

## エクスポート

パノプティックチェックポイントはエクスポートできません。このタスクのクエリマスク出力にはランタイムのエクスポート契約がまだないため、`export()`は`NotImplementedError`を送出します。EoMTのセマンティックタスクはエクスポートできます。[セマンティックセグメンテーション](/docs/tasks/semantic-segmentation)と[エクスポートとデプロイ](/docs/export)を参照してください。
