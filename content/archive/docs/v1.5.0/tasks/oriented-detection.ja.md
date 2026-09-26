---
title: 回転物体検出
seo_title: LibreYOLOによる回転物体検出
description: LibreYOLOで回転した物体を検出します。回転ボックスに対応するファミリー、4頂点のラベル行、推論、学習、検証、エクスポートの呼び出しを解説します。
lead: >-
  回転物体検出は軸平行の長方形ではなく回転した長方形で各インスタンスの位置を特定します。そのため傾いた物体も背景を多く含むボックスではなく、密に囲めます。タスクキーはobbです。
keywords:
  - 回転バウンディングボックス 検出
  - 回転物体検出
  - Python OBB
  - DOTA データセット
  - 航空画像 物体検出
  - rotated IoU
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        # rfdetr追加パッケージが必要 pip install "libreyolo[rfdetr]"
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファイル名の-obb接尾辞がタスクを選択するためtask引数は不要
        model = LibreYOLO("LibreRFDETRs-obb.pt")
        result = model(SAMPLE_IMAGE, save=True)

        obb = result.obb
        print(obb.xywhr)   # (N, 5) 中心x 中心y 幅 高さ ラジアン
        print(obb.conf, obb.cls)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreRFDETRs-obb.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 角度ではなく頂点
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreRFDETRs-obb.pt")(SAMPLE_IMAGE)
        obb = result.obb

        print(obb.xyxyxyxy.shape)    # (N, 4, 2) ピクセル単位の頂点
        print(obb.xyxyxyxyn.shape)   # 正規化した同じ値
        print(obb.xyxy.shape)        # (N, 4) 内包する軸平行ボックス
    - label: 小型チェックポイント
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRFDETRn-obb.pt")
        result = model(SAMPLE_IMAGE)

        print(result.obb.xywhr.shape)
    - label: RT-DETRv2
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 1024 pxで15個の航空画像クラスを持つDOTA v1.0の重み
        # 回転グラフはチェックポイント自身のテンソルから認識されるためtask引数は不要
        model = LibreYOLO("LibreRTDETRv2n-obb.pt")
        result = model("aerial.png", save=True)

        obb = result.obb
        print(obb.xywhr)
        print(result.names)   # plane ship harbor helicopter その他11クラス
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # 公開済み回転ボックス用重みから続ける dataは4頂点を持つ

        # ラベル行のデータセットを指す必要がある

        model = LibreYOLO("LibreRFDETRs-obb.pt")

        model.train(data="my-obb-dataset.yaml", epochs=50, imgsz=512, batch=8,
        lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs-obb.pt data=my-obb-dataset.yaml \
          epochs=50 imgsz=512 batch=8 lr0=1e-4
    - label: 検出用重みから開始
      language: bash
      code: |
        # 検出用重みは角度を予測しないため明示的な転移となる
        # task=obbの指定がこれを許可する
        libreyolo train model=LibreRFDETRs.pt data=my-obb-dataset.yaml \
          task=obb epochs=50 imgsz=512
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs-obb.pt")

        # val()はオブジェクトではなく通常のdictを返す
        metrics = model.val(data="my-obb-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"], metrics["metrics/mAP75"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRFDETRs-obb.pt data=my-obb-dataset.yaml
    - label: RT-DETRv2
      language: bash
      code: |
        libreyolo val model=LibreRTDETRv2n-obb.pt data=my-obb-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs-obb.pt")
        model.export(format="onnx", imgsz=512)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRFDETRs-obb.pt format=onnx imgsz=512
    - label: RT-DETRv2
      language: bash
      code: >
        # ONNXとTorchScriptがここで検証済みの対象でFP32 バッチ1

        # 固定1024x1024キャンバスを使用する

        libreyolo export model=LibreRTDETRv2n-obb.pt format=onnx imgsz=1024

        libreyolo export model=LibreRTDETRv2n-obb.pt format=torchscript
        imgsz=1024
    - label: エクスポートしたファイルを使用
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリはファイルの拡張子に応じて振り分けるためエクスポートした
        # アーティファクトもチェックポイントと同様に読み込まれ同じResultsオブジェクトを返す
        model = LibreYOLO("LibreRFDETRs-obb.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.obb.xywhr)
source_hash: 0d605d956f3ea025
---

## 定義

回転物体検出は検出に角度という1つの数値を追加します。各インスタンスに回転した長方形、クラス、スコアを割り当てます。利点は密に囲めることです。45度に傾いた船、倉庫の屋根、並んで駐車したトラックなどは、軸平行ボックスで囲むと大半が背景になり、物体同士が重ならなくても隣のボックス同士が重なります。このため航空画像や文書レイアウトでは標準的なタスクであり、参照データセットとしてDOTAが使われます。

`obb`が標準タスクキーで、チェックポイントのファイル名にある`-obb`接尾辞が選択します。そのため、公開済みの重みを読み込むときに`task=`は不要です。

`predict()`は`result.obb`を設定します。`.xywhr`は標準的な`(N, 5)`形式で、中心x、中心y、幅、高さ、中心の周りに幅側が回転する角度をラジアンで表します。`.conf`と`.cls`にはスコアと`result.names`を参照するクラスインデックス、追跡時の`.id`には追跡IDが格納されます。`.xyxyxyxy`は各行をピクセル単位の4頂点`(N, 4, 2)`へ変換し、`.xyxyxyxyn`はその頂点を正規化します。`.xyxy`は内包する軸平行ボックスを返し、長方形だけを扱える下流コードではこれを使用します。`result.boxes`にも軸平行形式が格納されます。

## モデル

このタスクには2つのファミリーが対応し、学習が必要かどうかで選択肢が変わります。

[RF-DETR](/docs/models/rf-detr)は学習できるファミリーです。回転ボックスの推論、学習、検証、エクスポートに対応し、n、s、m、lの4サイズで公開済みの回転チェックポイントを提供します。専用の追加パッケージ`pip install "libreyolo[rfdetr]"`が必要で、モデルページに重みのライセンスと来歴が記載されています。

使用計画を立てる前に、公開済みチェックポイントが実際に何を予測するかについて後述のセクションを確認してください。

[RT-DETRv2](/docs/models/rt-detr)は航空画像用の重みを持つファミリーです。`LibreRTDETRv2n-obb.pt`から`LibreRTDETRv2x-obb.pt`までを公開しています。これは公式DOTA v1.0の単一スケールチェックポイントをLibreYOLO形式へ変換したもので、1024pxでDOTAの15クラスに対応します。基本パッケージ以外の追加パッケージは不要で、回転グラフはチェックポイント自身のテンソルから認識されます。推論、検証、ONNXとTorchScriptへのエクスポートに対応します。学習には対応せず、このファミリーの回転タスクは推論専用です。`train()`は例外を送出し、異なるバックボーンを使う検出用重みからの転移もありません。回転ボックスでは追跡とテスト時データ拡張も利用できません。

つまり、DOTAカテゴリーをすぐに使うならRT-DETRv2、独自の回転ラベルで学習するならRF-DETRです。

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

実行前にRF-DETRの公開済みチェックポイントの内容を把握してください。DOTAはこのタスクの参照ベンチマークですが、これらの重みはDOTAで学習されていません。4つすべてがRF-DETRの検出用重みから初期化され、UAV映像を収録した1つのRoboflow Universeデータセットでファインチューニングされています。クラスはbike、bus、car、other_vehicle、taxi、truckの6つです。モデルカードでは、回転ボックス学習への対応を検証する間に生成した開発用の重みと説明され、製品用またはベンチマーク公式の重みとして解釈しないよう記載されています。

実際には、上空から見た車両の回転ボックスや、パイプラインが最初から最後まで動作することの確認に使える出発点です。他の分野では独自の回転ラベルで学習する必要があります。DOTAで知られる航空画像カテゴリーには、そのデータで実際に学習されたRT-DETRv2チェックポイントを使用してください。`conf`と`max_det`は検出と同じように出力を調整します。入力ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## データセット形式

配置は検出と同じです。画像ごとに1つの`.txt`ラベルファイルを使用し、画像パス内の`images`を`labels`へ置き換えて拡張子を変更し、ファイルを検索します。

```text
dataset/
  data.yaml
  images/
    train/P0001.png
    val/P0101.png
  labels/
    train/P0001.txt
    val/P0101.txt
```

1行は正確に9つのフィールドで、クラスインデックスの後に4つの頂点を順番に記述します。

```text
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```

4つの点は`[0, 1]`範囲に正規化した浮動小数点数で、退化していない回転長方形を形成する必要があります。ラベルファイルには角度を保存しません。ローダーが頂点から標準的な`xywhr`を導出します。パーサーはデフォルトで厳密に範囲外座標を拒否します。一方、データセットと検証への取り込みでは、クロップ境界にある有効なラベルを先に`[0, 1]`へクリップできますが、その後も退化したボックスは拒否します。

行の解析はタスクを認識します。9つのフィールドを回転ボックスとして扱うのは`obb`モードだけです。`segment`モードでは同じ行を4点のポリゴンとして読み取ります。

YAMLは検出用YAMLと同じです。

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: plane
  1: ship
```

ネイティブのCOCO JSONも読み込めます。分割名とJSONファイルの`annotations`マッピングを使用します。アノテーションは次の優先順位で読み取ります。ピクセル空間の8頂点を持つ`obb`フィールド、ラジアン単位の角度を持つ`[cx, cy, w, h, angle]`形式の`obb`フィールド、最小面積長方形へ再適合する`segmentation`ポリゴンまたはRLE、最後に軸平行長方形として扱い`xywhr`へ正規化する通常のCOCO `bbox`です。

標準の行パーサーは`libreyolo.data.parse_yolo_obb_label_line`です。

## 学習

<code-tabs name="train" />

このタスクの学習にはRF-DETRを使用します。デフォルトでは公開済みの`-obb`チェックポイントから続行します。検出用重みからの開始は意図的な転移です。その重みは角度を予測しないため、`task=obb`を渡すことで入れ替えを許可します。このファミリーの他タスクと同様、`lr0`は`1e-4`以下にしてください。RT-DETRv2の回転チェックポイントはファインチューニングできません。そのまま使用するか、独自ラベルでRF-DETRモデルを学習してください。データセット、データ拡張、マルチGPU、ロガーについては[学習](/docs/train)を参照してください。

## 検証

`val()`は`metrics/`キーを持つ通常の辞書を返します。照合には、内包する軸平行ボックス間ではなく回転長方形間で計算したrotated IoUを使用します。そのため、位置が正しくても角度が誤った推論は不一致になります。

<code-tabs name="val" />

`metrics/mAP50-95`はIoUしきい値0.50から0.95まで0.05刻みで平均した平均適合率で、主要な数値です。検出で使うCOCO経路と異なり、このタスクは検証設定の`iou_thresholds`を適用するため、探索範囲を変更できます。`metrics/mAP50`と`metrics/mAP75`は単一しきい値版です。`metrics/precision`と`metrics/recall`はIoU 0.50での実際の適合率と再現率です。最も緩い運用点、つまり信頼度しきい値を通過したすべての予測を数えて計算します。検証時のデフォルトしきい値は0.001です。そのため`conf`を上げるとこれらは変わりますが、適合率と再現率の完全な曲線を使うmAP値は変わりません。4つの値は`(OBB)`接尾辞付きでも繰り返されます。`metrics/mAP50-95(OBB)`、`metrics/mAP50(OBB)`、`metrics/precision(OBB)`、`metrics/recall(OBB)`です。これにより、同じ表に軸平行と回転の結果がある場合も呼び出し側で区別できます。`metrics/mAP75`には接尾辞付きの値がありません。

このタスクでは2つのオプションが何もしません。`save_json`と`save_plots`は受け付けられますが警告を記録します。回転予測のダンプと検証プロットは実装されていません。

## エクスポート

<code-tabs name="export" />

エクスポートしたアーティファクトはファイルの拡張子に基づいて`LibreYOLO()`で再読み込みできるため、`.onnx`や`.engine`ファイルもチェックポイントと同様に動作し、同じ`Results`を返します。同じファミリーでも対応形式はタスクごとに異なり、モデルページのマトリックスは検証済みの集合から生成され、対象が利用できない理由も示します。形式、追加パッケージ、制約については[エクスポートとデプロイ](/docs/export)を参照してください。
