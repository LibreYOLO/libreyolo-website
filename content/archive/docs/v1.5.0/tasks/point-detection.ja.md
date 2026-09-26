---
title: 点検出
seo_title: LibreYOLOによる点検出とカウント
description: LibreYOLOでボックスではなく単一の点として物体を特定します。重心の推論、物体数のカウント、FOMOの学習、点検出指標の読み方を解説します。
lead: >-
  点検出はバウンディングボックスの代わりに、物体ごとに1つのx、y位置を返します。LibreYOLOではpointタスクとして提供され、推論結果は物体ごとにx、y、クラス、信頼度を1行に保持します。
keywords:
  - Python 点検出
  - Python 物体カウント
  - 重心検出
  - FOMO 点位置推定
  - 画像 物体数 カウント
  - 点位置推定
last_verified: 1.5.0
snippets:
  predict:
    - label: 点を推論して数える
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # LibreFOMOの重みは自動ダウンロードされない
        # https://huggingface.co/LibreYOLOから先に取得しローカルパスで読み込む
        model = LibreYOLO("./LibreFOMOs-point.pt")
        result = model(SAMPLE_IMAGE, save=True)

        points = result.points
        print(len(points))     # 物体数
        print(points.xy)       # 元画像のピクセル単位の(N, 2)中心座標
        print(points.cls, points.conf)
    - label: 正規化座標とクラスごとの個数
      language: python
      code: |
        from collections import Counter

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("./LibreFOMOs-point.pt")
        result = model(SAMPLE_IMAGE)

        points = result.points.numpy()
        print(points.xyn)                          # [0, 1]範囲の同じ中心座標
        print(Counter(points.cls.astype(int).tolist()))
  train:
    - label: YOLOデータセットでFOMOを学習
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.train(data="my-dataset.yaml", epochs=40, batch=32, lr0=3e-4)
    - label: 学習済みチェックポイントで推論
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("./LibreFOMOs-point.pt")
        results = model.train(data="my-dataset.yaml", epochs=40)

        # train()は同じオブジェクトへ最良チェックポイントを再読み込みするため
        # 呼び出しが戻った時点で学習済みの重みを使って推論する
        print(results["best_checkpoint"])
        print(model(SAMPLE_IMAGE).points.xy)
  val:
    - label: 検証して指標キーを確認
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/precision"], metrics["metrics/recall"])
        print(metrics["metrics/f1"])
        print(metrics["metrics/mAP@[0.01:0.10]"])   # fitness
        print(metrics["metrics/MLE"])               # 平均位置誤差
        print(metrics["metrics/MAE"], metrics["metrics/RMSE"])   # 個数の誤差
    - label: 距離しきい値を変更
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("./LibreFOMOs-point.pt")


        # 探索範囲の境界がキー文字列の一部なので

        # カスタム探索では生成するmAPキーの名前も変わる

        metrics = model.val(data="my-dataset.yaml", dist_thresholds=[0.02,
        0.05])


        print(metrics["metrics/mAP@0.02"])

        print(metrics["metrics/mAP@[0.02:0.05]"])
  export:
    - label: エクスポート
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.export(format="onnx")
    - label: エクスポートしたファイルを実行
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリはファイルの拡張子に応じて振り分けるためエクスポートした
        # アーティファクトもチェックポイントと同様に読み込まれ同じResultsオブジェクトを返す
        model = LibreYOLO("./LibreFOMOs-point.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.points.xy)
source_hash: 932153c8870d1c7c
---

## 定義

`point`タスクは幅、高さ、マスクを持たず、1つのx、y座標とクラスで各物体の位置を特定します。推論結果は物体の平坦なリストなので、その行数が物体数になります。この性質によりカウント用のタスクとして使えます。

推論結果では`result.points`に値が格納されます。これは元画像のピクセル単位で`x, y, class, confidence`を各行に持つ`(N, 4)`配列をラップした`Points`ペイロードです。`.xy`は座標、`.xyn`は画像サイズで割った同じ座標、`.cls`はクラスインデックス、`.conf`はスコアを返し、`len()`は点の数を返します。`result.boxes`は空のままなので、`iou`と`max_det`が作用する対象はありません。

## モデル

`point`には3つのファミリーが対応し、相互に置き換えられるものではありません。

[FOMO](/docs/models/fomo)は固定ボキャブラリーの選択肢です。低解像度グリッドの各セルを背景または物体中心として分類するグリッド分類器です。LibreYOLOで学習できる唯一の点検出ファミリーであり、エクスポートできる唯一のファミリーでもあります。

[LocateAnything](/docs/models/locate-anything)はクラスインデックスの代わりにテキストを受け取るため、ボキャブラリーは入力した任意のフレーズになります。`vlm`追加パッケージが必要で、`LibreYOLO()`ファクトリではなく`LibreLocateAnything`として構築します。重みは非商用利用に制限されています。正確な条件と、チェックポイントが組み合わせるさらに2つのライセンスはモデルページに記載されています。

[SenseNova-Vision](/docs/models/sensenova-vision)は、他の6タスクでも使う同じプロンプト生成チェックポイントを通して`point`に対応し、`LibreVLM("sensenova-vision", task="point")`で読み込みます。`sensenova`追加パッケージが必要です。推論ごとに7Bモデルの生成処理を実行するため、専用検出器より画像ごとのレイテンシが明らかに長くなります。重みは非商用で、ライセンスはモデルページに記載されています。

## 推論

このサイトでLibreFOMOの重みだけは自動ダウンロードの例外です。`LibreYOLO("LibreFOMOs-point.pt")`はディスク上のファイルを探し、取得せずファイル名を示す`ValueError`を送出します。先にHugging Faceの[LibreYOLO組織](https://huggingface.co/LibreYOLO)からチェックポイントをダウンロードし、ローカルパスで読み込むか、独自に学習してください。

<code-tabs name="predict" />

ローダーが認識するにはファイル名に`-point`タスク接尾辞が必要です。`predict(..., nms_radius=1)`は、FOMOの2つの検出を両方残すために必要なグリッドセル間の距離を制御します。入力ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## データセット形式

`point`に固有のラベル形式はありません。点検出ファミリーは標準のYOLO検出レイアウトを読み取り、各ボックス行から1つの中心を導出します。そのため`cx cy`が点になり、`w h`は行が有効かどうかの判定にだけ使われます。

```text
dataset/
  data.yaml
  images/
    train/scene.jpg
    val/scene.jpg
  labels/
    train/scene.txt
    val/scene.txt
```

各ラベルファイルには物体ごとに1行の正規化座標を記述します。

```text
<class_id> <cx> <cy> <w> <h>
```

```yaml
path: dataset
train: images/train
val: images/val
nc: 1
names: {0: seedling}
```

ラベルファイルが存在しないか空の場合、その画像に物体がないことを意味します。完全な契約については[データセット形式](/docs/reference/dataset-formats)を参照してください。

## 学習

学習実装を持つ点検出ファミリーはFOMOだけです。LocateAnythingとSenseNova-Visionの`train()`は`NotImplementedError`を送出します。これらはアップストリームでファインチューニングし、その結果を読み込んでください。

<code-tabs name="train" />

FOMOの`imgsz`は自由に選べません。デフォルトは読み込んだチェックポイントのネイティブ解像度で、別の値を渡すと期待されるサイズを示す`ValueError`を送出します。データセット、ロガー、マルチGPUについては[学習](/docs/train)、このファミリーのデフォルト値については[FOMOページ](/docs/models/fomo)を参照してください。

## 検証

`val()`は距離しきい値を変えて、予測点と正解点をハンガリアン法で1対1に対応付けます。しきい値は正規化画像座標におけるユークリッド距離で、デフォルトの探索範囲は0.01から0.10までの10個の値です。

<code-tabs name="val" />

`metrics/precision`、`metrics/recall`、`metrics/f1`は、探索範囲で最も厳しいしきい値（デフォルトは0.01）においてクラス間でマクロ平均した値です。`metrics/mAP@0.01`は同じしきい値での平均適合率、`metrics/mAP@[0.01:0.10]`は探索範囲全体の平均です。この探索範囲の値は、最良チェックポイントの選択で使う`fitness`にもなります。どちらのmAPキーも使用するしきい値から構築されるため、`dist_thresholds=`を渡すと名前が変わります。

`metrics/MLE`は最も厳しいしきい値で対応したペア間の平均距離で、同じ正規化単位を使います。`metrics/MAE`と`metrics/RMSE`は位置ではなく個数の指標で、画像ごとの予測点数と正解点数の差を測定します。

FOMOはこれらに加えてグリッド単位の第2の指標群を追加します。信頼度と`nms_radius`を変えて探索し、F1が最良となる組み合わせを`metrics/grid_F1`、`metrics/grid_precision`、`metrics/grid_recall`、`metrics/grid_mean_distance`、`metrics/grid_TP`、`metrics/grid_FP`、`metrics/grid_FN`として公開します。そのときの設定は`decode/threshold`と`decode/nms_radius`に記録されます。

## エクスポート

FOMOは共通のエクスポート経路を使用します。エクスポートしたアーティファクトはファイルの拡張子に基づいて`LibreYOLO()`で再読み込みできるため、`.onnx`や`.engine`ファイルもチェックポイントと同様に動作し、同じ`Results`を返します。

<code-tabs name="export" />

形式ごとの対応範囲は[FOMOページ](/docs/models/fomo)と[完全なエクスポートマトリックス](/docs/reference/export-matrix)に記載されています。LocateAnythingとSenseNova-Visionはエクスポートできません。生成モデルにはトレース可能な検出グラフがないため、どちらも`export()`が例外を送出します。
