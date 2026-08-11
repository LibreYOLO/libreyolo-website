---
title: 姿勢推定
seo_title: LibreYOLOの姿勢推定
description: LibreYOLOでインスタンスごとのキーポイントを予測します。対応するファミリー、ラベル形式、予測、学習、検証、エクスポートの呼び出しを説明します。
lead: >-
  姿勢推定は各インスタンスの位置を特定し、名前付きキーポイントの順序付きセットを返します。このため、出力には物体の範囲だけでなく内部構造も含まれます。タスクキーはposeです。
keywords:
  - 姿勢推定 Python
  - キーポイント検出
  - 人体姿勢推定 モデル
  - COCO キーポイント
  - OKS mAP
  - 姿勢推定モデル 学習
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファイル名の-poseサフィックスでキーポイントヘッドが選択されるため、
        # task引数は不要です。
        model = LibreYOLO("LibreECs-pose.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.keypoints.xy.shape)   # (N, K, 2)のピクセル座標
        print(result.boxes.xyxy.shape)     # (N, 4)、同じN個のインスタンス
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreECs-pose.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 可視キーポイントのみ
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreECs-pose.pt")(SAMPLE_IMAGE)
        kpts = result.keypoints

        # .has_visibleはキーポイントの第3列から導出され、チェックポイントが
        # (x, y)だけを予測する場合はすべてTrueになります。
        for person, visible in zip(kpts.xy, kpts.has_visible):
            print(person[visible])
    - label: トップダウン方式
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # HRNetはトップダウン方式で、最初に各人物を切り抜きます。人物ソースを指定しない場合は、
        # LibreYOLO9t検出器と自動的に組み合わせ、その選択をログへ記録します。
        model = LibreYOLO("LibreHRNetw32-pose.pt")
        result = model(SAMPLE_IMAGE)

        print(result.keypoints.xy.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # coco8-pose.yamlにはダウンロードスクリプトが埋め込まれているため、データが
        # すでにローカルにない場合は明示的な許可が必要です。
        model = LibreYOLO("LibreECs-pose.pt")
        model.train(
            data="coco8-pose.yaml",
            epochs=50,
            imgsz=640,
            batch=4,
            allow_download_scripts=True,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreECs-pose.pt data=coco8-pose.yaml \
          epochs=50 imgsz=640 batch=4 allow_download_scripts=True
    - label: 独自のデータセット
      language: python
      code: |
        from libreyolo import LibreYOLO

        # data.yamlでkpt_shapeを宣言し、ラベル行には正確に
        # 5 + K * D個のフィールドを含める必要があります。
        model = LibreYOLO("LibreECs-pose.pt")
        model.train(data="my-pose-dataset.yaml", epochs=50, imgsz=640, batch=8)
  val:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreECs-pose.pt")


        # val()はオブジェクトではなく通常のdictを返します。

        metrics = model.val(data="coco8-pose.yaml", allow_download_scripts=True)


        print(metrics["metrics/keypoints_mAP50-95"])

        print(metrics["metrics/keypoints_mAP50"],
        metrics["metrics/keypoints_mAP75"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreECs-pose.pt data=coco8-pose.yaml \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-pose.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreECs-pose.pt format=onnx imgsz=640
    - label: エクスポート済みファイルを使用
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリーはファイルサフィックスに基づいて振り分けるため、エクスポート済み
        # アーティファクトもチェックポイントと同様に読み込まれ、同じResultsオブジェクトを返します。
        model = LibreYOLO("LibreECs-pose.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.keypoints.xy)
source_hash: 9de01d1f615bdf33
---

## 定義

姿勢推定は範囲だけでなく構造を返します。各インスタンスには引き続きボックス、クラス、スコアが割り当てられ、さらに固定順序の`K`個のキーポイントが割り当てられます。このため、インデックス5はすべてのインスタンスと画像で同じ身体部位を意味します。その順序はラベルセットで定義され、出力内にキーポイントを名前で識別する情報はありません。

`pose`が正規タスクキーで、チェックポイントのファイル名にある`-pose`サフィックスがタスクを選択します。このため、公開済みの重みを読み込むときに`task=`は不要です。

`predict()`は`result.boxes`とともに`result.keypoints`を設定します。`.data`の形状は`(N, K, 2)`または`(N, K, 3)`で、ボックスと行が対応します。一方のインスタンス`i`は他方のインスタンス`i`と同じです。`.xy`はピクセル座標を取り出し、`.xyn`は元画像のサイズで正規化します。チェックポイントが第3列を予測する場合は`.conf`に格納され、予測しない場合は`None`です。`.has_visible`はそこから導出される真偽値マスクで、第3列がない場合はすべてTrueです。

2種類のアーキテクチャがこの出力を生成します。1段階モデルは1回の処理でボックスとキーポイントを予測します。トップダウンモデルは最初に検出器を実行し、各インスタンスを切り抜き、その領域内でキーポイントを回帰します。このため、精度は前段の検出器に依存します。

## モデル

学習と予測の両方に対応する1段階ファミリーは、[RF-DETR](/docs/models/rf-detr)、[EdgeCrafter](/docs/models/edgecrafter)、[YOLO-NAS](/docs/models/yolo-nas)の3つです。RF-DETRには専用の追加パッケージ`pip install "libreyolo[rfdetr]"`が必要です。RF-DETRとEdgeCrafterは公開済みの姿勢推定チェックポイントを提供し、どちらも単一クラスの人物専用データセットでファインチューニングします。EdgeCrafterのキーポイントヘッドは構築時に固定され、異なるキーポイント数を宣言するデータセットを拒否します。RF-DETRはキーポイント数に合わせてヘッドを再初期化します。YOLO-NASはDeci.AI自身のCDNから非商用ライセンスの重みを取得し、LibreYOLOはそれらを一切公開していません。姿勢推定ヘッドは新しいキーポイント数に合わせて再構築されます。また、3つのうちクラス数が1に固定されない唯一のファミリーなので、動物の姿勢など、複数クラスまたは人物以外のスケルトンに適しています。

[HRNet](/docs/models/hrnet)はトップダウン方式です。予測、検証、エクスポートに対応し、`train()`は`NotImplementedError`を発生させます。人物ソースを指定しない場合はLibreYOLO9t検出器と自動的に組み合わされます。`cropped=True`は画像全体を1つのインスタンスとして扱い、`person_boxes=`は手元にあるボックスを受け取り、`person_detector=`は別の検出器を指定します。

[SenseNova-Vision](/docs/models/sensenova-vision)もキーポイントを出力します。独自のファクトリー`LibreVLM`と追加パッケージを持つ、プロンプト型の生成モデルです。語彙を設定せずに`set_task("pose")`を呼ぶと人物カテゴリーへフォールバックします。重みは非商用で、予測ごとにdiffusion decodeを行うため、画像あたりのレイテンシは姿勢推定専用ヘッドよりはるかに大きくなります。

## 予測

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

キーポイントの数と順序はライブラリではなくチェックポイントの属性です。異なるスケルトンで学習したモデルは異なる`K`を返し、各インデックスの意味も異なります。キーポイントの第3列に入る内容もチェックポイントの属性です。EdgeCrafterは点ごとのスコアではなく定数を書き込みます。またボックスヘッドがないため、各姿勢推定ボックスはそのインスタンス自身のキーポイントを囲む範囲になります。入力ソース、ストリーミング、結果の処理については[予測](/docs/predict)を参照してください。

## データセット形式

レイアウトは物体検出と同じです。画像ごとに1つの`.txt`ラベルファイルを用意し、画像パスの`images`を`labels`へ置き換え、拡張子を変更した場所に保存します。

```text
dataset/
  data.yaml
  images/
    train/000001.jpg
    val/000101.jpg
  labels/
    train/000001.txt
    val/000101.txt
```

各行は物体検出の行にキーポイントを追加した形式です。

```text
<class_id> <cx> <cy> <w> <h> <k1x> <k1y> [<k1v>] ... <kKx> <kKy> [<kKv>]
```

フィールド数は正確に`5 + K * D`で、`D`は`kpt_shape`の2番目の値です。ボックスとキーポイント座標は、元画像の幅と高さに対して正規化した浮動小数点値です。可視性`v`は`D`が3の場合だけ存在し、`0`、`1`、`2`のいずれかです。

YAMLでは共有仕様に2つのキーを追加します。

```yaml
path: dataset
train: images/train
val: images/val
kpt_shape: [17, 3]
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
names:
  0: person
```

`kpt_shape`は必須で、`[K, 2]`または`[K, 3]`です。`flip_idx`は任意で、水平反転後に各キーポイントが取るインデックスを示す`0..K-1`の順列です。これにより、左手首は反転後も左手首として扱われます。省略した場合、誤ったインデックス順序で適用せず、キーポイントの水平反転データ拡張を無効にします。

## 学習

<code-tabs name="train" />

学習は、キーポイントヘッドをすでに持つ公開済み`-pose`チェックポイントから継続します。タスクは学習時に渡すフラグではなく、読み込むチェックポイントから取得します。このため、姿勢推定を要求しても物体検出チェックポイントが姿勢推定の実行に変わることはありません。EdgeCrafterでは構築時にヘッドが固定されるため、YAMLの`kpt_shape`がヘッドと正確に一致する必要があります。RF-DETRとYOLO-NASは異なる数に合わせてヘッドのサイズを変更します。データセット、データ拡張、マルチGPU、ロガーについては[学習](/docs/train)を参照してください。

## 検証

`val()`は`metrics/`キーを持つ通常の辞書を返します。評価にはObject Keypoint Similarityに基づくCOCOキーポイント評価を使用します。各キーポイントの距離誤差をインスタンスのスケールとキーポイントごとの許容値で重み付けし、ボックスに対するIoUと同じ役割を果たします。基本インストールに含まれる`pycocotools`が必要です。

<code-tabs name="val" />

`metrics/keypoints_mAP50-95`は主要指標です。OKSしきい値0.50から0.95までで平均したmean average precisionで、学習時の最良エポック選択にも使われます。`metrics/keypoints_mAP50`と`metrics/keypoints_mAP75`は単一しきい値版です。`metrics/keypoints_mAP_M`と`metrics/keypoints_mAP_L`はインスタンス面積別に平均を中規模と大規模へ分けます。COCOキーポイント評価には小規模の区分がありません。対応するaverage recallは`metrics/keypoints_AR50-95`、`metrics/keypoints_AR50`、`metrics/keypoints_AR75`、`metrics/keypoints_AR_M`、`metrics/keypoints_AR_L`です。このタスクのすべてのキーには`keypoints_`プレフィックスが付くため、検出器が返すボックスの`mAP`キーは現れません。

## エクスポート

<code-tabs name="export" />

エクスポートしたアーティファクトは、ファイルサフィックスに基づいて`LibreYOLO()`から読み戻せます。このため、`.onnx`や`.engine`ファイルもチェックポイントと同様に動作し、同じ`Results`を返します。形式への対応状況はファミリーごとに異なります。各モデルページのマトリックスは手作業で入力されたものではなく、検証済みセットから生成されます。形式、追加パッケージ、制約については[エクスポートとデプロイ](/docs/export)を参照してください。
