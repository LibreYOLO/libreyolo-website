---
title: 姿勢推定
seo_title: LibreYOLOの姿勢推定
description: >-
  LibreYOLOでインスタンスごとのキーポイントを推論します：このタスクを提供するファミリー、ラベル形式、predict、train、validate、exportの呼び出しについて説明します。
lead: >-
  姿勢推定は各インスタンスの位置を特定し、順序付きの名前付きキーポイント集合を返します。そのため、出力には物体の範囲だけでなく内部構造も含まれます。タスクキーはposeです。
keywords:
  - 姿勢推定 python
  - キーポイント 検出
  - 人体 姿勢推定 モデル
  - COCO keypoints
  - OKS mAP
  - 姿勢推定 モデル 学習
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファイル名の-pose接尾辞がキーポイントヘッドを選ぶため、
        # task引数は不要
        model = LibreYOLO("LibreECs-pose.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.keypoints.xy.shape)   # (N, K, 2)ピクセル座標
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

        # .has_visibleはキーポイントの第3列から導出され、
        # チェックポイントが(x, y)だけを推論する場合はすべてtrue
        for person, visible in zip(kpts.xy, kpts.has_visible):
            print(person[visible])
    - label: 代わりにトップダウン方式
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # HRNetはトップダウン方式で、最初に各人物を切り抜く。人物ソースが
        # 未指定ならLibreYOLO9t検出器と自身を組み合わせ、選択をログに記録
        model = LibreYOLO("LibreHRNetw32-pose.pt")
        result = model(SAMPLE_IMAGE)

        print(result.keypoints.xy.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # coco8-pose.yamlにはダウンロードスクリプトが埋め込まれているため、
        # データがローカルにない限り明示的な許可が必要
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

        # data.yamlではkpt_shapeの宣言が必要で、ラベル行には
        # 正確に5 + K * D個のフィールドが必要
        model = LibreYOLO("LibreECs-pose.pt")
        model.train(data="my-pose-dataset.yaml", epochs=50, imgsz=640, batch=8)
  val:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreECs-pose.pt")


        # val()はオブジェクトではなく通常のdictを返す

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
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリーはファイル拡張子で振り分けるため、エクスポートした
        # 成果物もチェックポイントと同様に読み込まれ、同じResultsオブジェクトを返す
        model = LibreYOLO("LibreECs-pose.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.keypoints.xy)
source_hash: 9de01d1f615bdf33
---

## 定義

姿勢推定は範囲だけでなく構造を返します。各インスタンスは引き続きボックス、クラス、スコアを
持ち、さらに固定順序の`K`個のキーポイントを持ちます。そのため、インデックス5はすべての
インスタンスと画像で同じ身体部位を意味します。ラベル集合がその順序を定義し、出力内には
キーポイントを名前で識別するものはありません。

`pose`は正規のタスクキーです。チェックポイントのファイル名にある`-pose`接尾辞がこのタスクを
選択するため、公開済みの重みを読み込むときに`task=`は不要です。

`predict()`は`result.boxes`とともに`result.keypoints`を埋めます。`.data`は`(N, K, 2)`または
`(N, K, 3)`で、ボックスと行が対応します。そのため、一方のインスタンス`i`はもう一方の
インスタンス`i`と同じです。`.xy`はピクセル座標を切り出し、`.xyn`は元画像のサイズで正規化
します。チェックポイントが第3列を推論する場合は`.conf`がその値となり、しない場合は`None`です。
`.has_visible`はそこから導出されるブールマスクで、第3列がない場合はすべてtrueです。

2種類のアーキテクチャがこの出力を生成します。1段階モデルはボックスとキーポイントを1回の
処理で推論します。トップダウンモデルは最初に検出器を実行し、各インスタンスを切り抜き、
切り抜き内のキーポイントを回帰するため、精度は前段の検出器に依存します。

## モデル

3つのファミリーが学習と推論の両方に対応します：
[RF-DETR](/docs/models/rf-detr)、[EdgeCrafter](/docs/models/edgecrafter)、
[YOLO-NAS](/docs/models/yolo-nas)で、すべて1段階方式です。RF-DETRには専用のextraである
`pip install "libreyolo[rfdetr]"`が必要です。RF-DETRとEdgeCrafterは公開済みの姿勢
チェックポイントを提供し、どちらも単一クラスの人物専用データセットでファインチューニング
します。EdgeCrafterのキーポイントヘッドは構築時に固定され、異なる個数を宣言するデータセットを
拒否しますが、RF-DETRはそれに合わせてヘッドを再初期化します。YOLO-NASは非商用ライセンスの
下でDeci.AI独自のCDNから重みを取得し、LibreYOLOはどの重みも公開しません。その姿勢ヘッドも
新しいキーポイント数に合わせて再構築されます。また、3つのうちクラス数が1に固定されていない
唯一のファミリーなので、動物の姿勢など、マルチクラスまたは人物以外のスケルトンに適しています。

[HRNet](/docs/models/hrnet)はトップダウン方式の選択肢です。推論、検証、エクスポートに対応し、
`train()`は`NotImplementedError`を発生させます。人物ソースを指定しない場合は、
LibreYOLO9t検出器と自動的に組み合わされます。`cropped=True`は画像全体を1つのインスタンスとして
扱い、`person_boxes=`は指定済みのボックスを受け取り、`person_detector=`は別の検出器を
指定します。

[SenseNova-Vision](/docs/models/sensenova-vision)もキーポイントを出力します。独自のファクトリー
`LibreVLM`と独自のextraを持つプロンプト型生成モデルです。語彙が設定されていない場合、
`set_task("pose")`は人物カテゴリーへフォールバックします。重みは非商用で、すべての推論が
拡散デコードを行うため、画像あたりのレイテンシは姿勢専用ヘッドよりはるかに高くなります。

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

キーポイントの個数と順序はライブラリではなくチェックポイントのプロパティです。そのため、
異なるスケルトンで学習したモデルは異なる`K`を返し、インデックスごとの意味も異なります。
キーポイントの第3列が保持する値もチェックポイントのプロパティです。EdgeCrafterは点ごとの
スコアではなく定数を書き込み、ボックスヘッドを一切持たないため、各姿勢ボックスはその
インスタンス自身のキーポイントを囲む範囲になります。ソース、ストリーミング、結果処理については
[推論](/docs/predict)を参照してください。

## データセット形式

レイアウトは検出と同じです。画像パスの`images`を`labels`へ置き換え、拡張子を変更した場所に、
画像ごとに1つの`.txt`ラベルファイルを置きます。

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

1行は、末尾にキーポイントを追加した検出行です。

```text
<class_id> <cx> <cy> <w> <h> <k1x> <k1y> [<k1v>] ... <kKx> <kKy> [<kKv>]
```

フィールド数は正確に`5 + K * D`で、`D`は`kpt_shape`の2番目の値です。ボックスと
キーポイントの座標は、元画像の幅と高さに対して正規化された浮動小数点数です。可視性`v`は
`D`が3の場合だけ存在し、`0`、`1`、`2`のいずれかです。

YAMLは共有規約に2つのキーを追加します。

```yaml
path: dataset
train: images/train
val: images/val
kpt_shape: [17, 3]
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
names:
  0: person
```

`kpt_shape`は必須で、`[K, 2]`または`[K, 3]`です。`flip_idx`はオプションで、各キーポイントに
対し水平反転後に取るインデックスを示す`0..K-1`の順列です。これにより、左手首が左手首のままに
なります。省略した場合、誤ったインデックス順で適用するのではなく、キーポイントの水平反転
データ拡張が無効になります。

## 学習

<code-tabs name="train" />

学習は、キーポイントヘッドをすでに持つ公開済み`-pose`チェックポイントから続行します。
タスクは学習時に渡すフラグではなく、読み込むチェックポイントから取得されるため、検出
チェックポイントに姿勢を要求しても姿勢の実行にはなりません。YAMLの`kpt_shape`は、ヘッドが
構築時に固定されるEdgeCrafterでは完全に一致する必要があります。一方、RF-DETRとYOLO-NASは
異なる個数に合わせてヘッドのサイズを変更します。データセット、データ拡張、マルチGPU、
ロガーについては[学習](/docs/train)を参照してください。

## 検証

`val()`は`metrics/`キーを持つ通常の辞書を返します。採点はObject Keypoint Similarityを使った
COCOキーポイント評価です。各キーポイントの距離誤差をインスタンスのスケールとキーポイントごとの
許容値で重み付けするため、ボックスに対するIoUの役割を果たします。基本インストールに含まれる
`pycocotools`が必要です。

<code-tabs name="val" />

`metrics/keypoints_mAP50-95`は主要指標で、OKSしきい値0.50〜0.95にわたって平均した平均適合率です。
学習ではこれを使って最良のエポックを選択します。`metrics/keypoints_mAP50`と
`metrics/keypoints_mAP75`は単一しきい値版です。`metrics/keypoints_mAP_M`と
`metrics/keypoints_mAP_L`は平均をインスタンス面積のmediumとlargeに分けます。COCOの
キーポイント評価にはsmall区分がありません。対応する平均再現率の値は
`metrics/keypoints_AR50-95`、`metrics/keypoints_AR50`、`metrics/keypoints_AR75`、
`metrics/keypoints_AR_M`、`metrics/keypoints_AR_L`です。このタスクのすべてのキーには
`keypoints_`プレフィックスが付くため、検出器が返すボックスの`mAP`キーは現れません。

## エクスポート

<code-tabs name="export" />

エクスポートした成果物は、ファイル拡張子に応じて`LibreYOLO()`から再び読み込まれます。
そのため、`.onnx`または`.engine`ファイルはチェックポイントのように動作し、同じ`Results`を
返します。形式の対応範囲はファミリーごとに異なります。各モデルページのマトリックスは手作業で
入力されるのではなく、検証済みの集合から生成されます。形式、extra、制約については
[エクスポートとデプロイ](/docs/export)を参照してください。
