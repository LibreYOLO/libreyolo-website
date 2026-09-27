---
title: Propheseeイベントヒストグラム
seo_title: LibreYOLOでPropheseeイベントヒストグラムを使う
description: >-
  LibreYOLO v1.6の開発ガイド：Prophesee
  SDKのヒストグラム変換、ラベルの準備、YOLO9またはRF-DETRの学習、ONNX推論の実行方法を説明します。
lead: 正負のイベント数を数値平面として検出器に入力し、イベント取得はProphesee、学習と推論はLibreYOLOで行います。
keywords:
  - Prophesee
  - Metavision SDK
  - OpenEB
  - イベントヒストグラム
  - イベントカメラ 物体検出
  - LibreYOLO v1.6
verification: 入力ワークフローはv1.6開発実装と照合済みです。SDK APIはMetavision 5.3.1と照合済みです。SDKランタイム検証は保留中です。
snippets:
  install:
    - label: 開発版をインストール
      language: bash
      code: >
        python -m pip install "libreyolo[rfdetr,onnx] @
        git+https://github.com/LibreYOLO/libreyolo.git@event-histogram-input"
  pack:
    - label: 既存の数値ヒストグラムを変換
      language: python
      code: |
        import numpy as np


        def pack_histogram(frame, *, layout, polarity, count_divisor=1.0):
            """Convert known producer output to HWC positive/negative counts."""
            values = np.asarray(frame)
            if values.dtype.kind not in "uif" or values.ndim != 3:
                raise ValueError("Expected a numerical three-dimensional histogram")
            if layout == "CHW":
                values = values.transpose(1, 2, 0)
            elif layout != "HWC":
                raise ValueError("Declare layout as HWC or CHW")
            if values.shape[2] != 2 or min(values.shape[:2]) == 0:
                raise ValueError("Expected two nonempty polarity planes")
            if polarity == "negative_positive":
                values = values[..., [1, 0]]
            elif polarity != "positive_negative":
                raise ValueError("Declare the producer's polarity order")
            if not np.isfinite(count_divisor) or count_divisor <= 0:
                raise ValueError("count_divisor must be positive and finite")
            with np.errstate(over="ignore", invalid="ignore"):
                counts = values.astype(np.float32) * count_divisor
            if not np.isfinite(counts).all() or (counts < 0).any():
                raise ValueError("Counts must be finite, nonnegative float32 values")
            return np.ascontiguousarray(counts)


        # Replace this path with a numerical histogram from your producer.
        sdk_histogram = np.load("sdk-histogram.npy", allow_pickle=False)
        histogram = pack_histogram(
            sdk_histogram, layout="HWC", polarity="negative_positive",
            count_divisor=1.0,  # Counts already: no producer normalization to undo.
        )
        np.save("histogram.npy", histogram, allow_pickle=False)
  sdk:
    - label: SDK Core APIでヒストグラムを1つ作成
      language: python
      code: |
        import numpy as np
        from metavision_sdk_core import EventPreprocessor


        def histogram_from_sdk_chunks(chunks, *, width, height, start_us,
                                      window_us=40_000, scale=16.0):
            """chunks must cover one known window, using SDK EventCD arrays."""
            processor = EventPreprocessor.create_HistoProcessor(
                input_event_width=width,
                input_event_height=height,
                max_incr_per_pixel=1.0,  # 1増分を1カウントとして扱う
                clip_value_after_normalization=scale,
                use_CHW=False,          # SDK出力はHWC、負の極性が先
                scale_width=1.0,
                scale_height=1.0,
            )
            frame = processor.init_output_tensor()
            frame.fill(0)
            end_us = start_us + window_us
            for events in chunks:
                if len(events) == 0:
                    continue
                if ((events["t"] < start_us) | (events["t"] >= end_us)).any():
                    raise ValueError("Split events into the declared window first")
                if ((events["x"] < 0) | (events["x"] >= width)
                        | (events["y"] < 0) | (events["y"] >= height)).any():
                    raise ValueError("Event coordinates do not match the sensor canvas")
                if not np.isin(events["p"], [0, 1]).all():
                    raise ValueError("Expected SDK polarity values 0 and 1")
                processor.process_events(start_us, np.ascontiguousarray(events), frame)
            return np.ascontiguousarray(frame[..., [1, 0]], dtype=np.float32)
  dataset:
    - label: data.yaml
      language: yaml
      code: |
        path: /absolute/path/to/event-dataset
        train: images/train
        val: images/val
        names:
          0: person
        input_profile:
          format: event_histogram
          layout: HWC
          polarity: positive_negative
          encoding: counts
          scale: 16.0
          window_us: 40000
  labels:
    - label: labels/train/frame000001.txt
      language: text
      code: |
        0 0.5 0.5 0.25 0.6
  inspect:
    - label: 数値入力を検査して可視化を保存
      language: python
      code: >
        import numpy as np

        from PIL import Image

        from libreyolo.utils.event_histogram import load_histogram,
        visualize_histogram


        histogram = load_histogram("histogram.npy")

        print(histogram.shape, histogram.dtype)

        print(float(histogram.min()), float(histogram.max()))

        rgb = visualize_histogram(histogram, scale=16.0)

        Image.fromarray(rgb).save("histogram-view.png")
  yolo9:
    - label: YOLO9を学習
      language: python
      code: |
        from libreyolo import LibreYOLO, LibreYOLO9

        model = LibreYOLO9(None, size="t", device="cpu")
        run = model.train(
            data="data.yaml", pretrained=False,
            epochs=100, batch=4, imgsz=128, workers=0, device="cpu",
        )
        trained = LibreYOLO(run["best_checkpoint"], device="cpu")
        trained.save("event-detector.pt")
  rfdetr:
    - label: RF-DETRをゼロから学習
      language: bash
      code: >
        libreyolo train model=LibreRFDETRn.pt pretrained=False data=data.yaml
        epochs=100 batch=4 imgsz=128 workers=0 device=cpu project=runs
        name=event-rfdetr exist_ok=True
    - label: 選択したRF-DETRチェックポイントを保存
      language: python
      code: |
        from libreyolo import LibreYOLO

        trained = LibreYOLO("runs/event-rfdetr/weights/best.pt", device="cpu")
        trained.save("event-detector.pt")
  predict:
    - label: ファイルまたは配列から推論
      language: python
      code: |
        import numpy as np
        from libreyolo import LibreYOLO

        model = LibreYOLO("event-detector.pt", device="cpu")
        result = model.predict("histogram.npy", save=True)
        print(result.boxes.xyxy)
        print(result.boxes.conf)

        histogram = np.load("histogram.npy", allow_pickle=False)
        result = model.predict(histogram)
        results = model.predict([histogram, histogram.copy()])
    - label: CLIから推論
      language: bash
      code: |
        libreyolo predict model=event-detector.pt source=histogram.npy save=True
  val:
    - label: 検証分割を評価
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("event-detector.pt", device="cpu")
        metrics = model.val(data="data.yaml", split="val", batch=4, workers=0)
        print(metrics["metrics/mAP50-95"])
  onnx:
    - label: ONNXをエクスポートして読み込む
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("event-detector.pt", device="cpu")
        onnx_path = model.export(
            format="onnx", imgsz=128, dynamic=False, simplify=False,
            output_path="event-detector.onnx",
        )
        runtime = LibreYOLO(onnx_path, device="cpu")
        result = runtime.predict("histogram.npy", save=True)
source_hash: 02ffe9df882336f4
---

**利用可能性：`dev`向けの開発ドキュメントで、LibreYOLO v1.6でのリリースを予定しています。** ヒストグラム実装は、`dev`へのマージ待ちの[PR #865](https://github.com/LibreYOLO/libreyolo/pull/865)にあります。マージされるまでは、以下の機能ブランチ版をインストールしてください。この機能は現在のPyPIリリースには含まれていません。

## 開発版をインストール

検出器を実行するPython環境で、次を実行してください。

<code-tabs name="install" />

PR #865がマージされたら、このコマンドの`@event-histogram-input`を`@dev`に置き換えてください。v1.6のリリース後は、対応するPyPIバージョンをインストールしてください。

アプリケーションですでにProphesee Metavision SDKを使っている場合は、イベント取得とデコードの設定をそのまま使ってください。準備済みの配列またはファイルを受け取る場合、LibreYOLOにSDKは必要ありません。SDKの生成側と検出器は別々の環境で実行し、`.npy`ファイルを受け渡せます。配列を直接渡すには、同じインタープリターに両方のパッケージが必要です。新規にSDKをインストールする場合は、提供元の[OpenEBインストールガイド](https://docs.prophesee.ai/stable/installation/index.html)に従ってください。

## SDKとの受け渡し形式

ワークフローは次のとおりです。

```text
SDK event window → two numerical count planes → LibreYOLO detector → boxes
```

サポートされる入力は、形状が**`(height, width, 2)`**のNumPy配列、またはその配列を含む`.npy`ファイルです。チャンネル0は正のカウント、チャンネル1は負のカウントです。値は有限かつ非負である必要があります。整数と浮動小数点のカウントを受け付け、float32として処理します。

レイアウト、チャンネル順、スケールを明示すれば、正負2極性の数値ヒストグラムと互換性があります。SDKで表示できるあらゆる画像を受け付けるわけではありません。RGBイベント可視化画像、再構成された輝度フレーム、符号付き差分画像、タイムサーフェス、多ビンイベントキューブは、同じ入力表現を含みません。元の情報が統合または破棄されている場合は、2つのカウント平面を再生成してください。

たとえば、正のイベントが3つ、負のイベントが1つあるピクセルは`[3, 1]`として保存されます。イベントのないウィンドウは、センサー全体の寸法を持つゼロ配列です。

## 既存のヒストグラムを使う

変換する前に生成元の設定を確認してください。`HWC`は高さ、幅、チャンネル、`CHW`はチャンネル、高さ、幅を意味します。色や、どちらのチャンネルの値が大きいかから極性の順序を推測しないでください。

このヘルパーを前処理スクリプトに追加してください。宣言された生成元のレイアウトと極性順を変換し、既知の正規化除数を元に戻します。

<code-tabs name="pack" />

`clip(counts / 16, 0, 1)`として計算されたSDKヒストグラムでは、`count_divisor=16.0`と`input_profile.scale: 16.0`を使ってください。生成されるファイルには16で飽和したカウントが含まれ、同じ飽和レベルの生カウントと同じ正規化済み検出器入力になります。SDKがすでにカウントを保存している場合は、`count_divisor=1.0`を使ってください。

クリッピングを元に戻すことはできません。5でクリップされたカウントでは、16までのカウントを区別するモデル入力を再現できません。ヒストグラムを再生成するか、一致する生成元の設定で学習してください。この例はセンサー座標がまだリスケールされていないことを前提としています。入力とラベルの両方で同じキャンバスを維持してください。

## Prophesee Core APIでヒストグラムを作成

SDKは、公開されている[Core Python API](https://docs.prophesee.ai/stable/api/python/core/bindings.html#metavision_sdk_core.EventPreprocessor.create_HistoProcessor)を通じて`EventPreprocessor.create_HistoProcessor`を提供します。以下の例ではHWC出力を選択し、イベントあたり1カウントを使い、SDKの負・正の平面をLibreYOLOの正・負の順序に並べ替えます。データセットYAMLで使う`scale`までのカウントを保持します。

**SDKの例は公開APIと照合済みですが、この検証環境にインストールされたSDKでは実行していません。** 次のセクションにある準備済み配列とファイルのワークフローは、両方のモデルで動作確認済みです。

<code-tabs name="sdk" />

`x`、`y`、`p`、`t`フィールドを持つデコード済みSDK `EventCD`配列を渡してください。SDKのdtypeを維持してください。任意のNx4整数行列は`EventCD`配列ではありません。SDKの極性0は負、1は正です。`width`と`height`はイベントの座標系を表す必要があります。

`chunks`には、既知の1つのウィンドウに属するすべてのイベントが含まれている必要があります。取得またはアノテーションのタイムラインから`start_us`を設定し、学習サンプルと推論サンプルすべてで同じ長さを使ってください。最初または最後に観測されたイベントから境界を推定しないでください。このヘルパーは、`[start_us, start_us + window_us)`の範囲外のイベントを含むチャンクを拒否します。

出力テンソルはウィンドウごとに1回クリアし、各チャンク間で保持します。空のウィンドウでは空のイテラブルを渡し、生成されたゼロフレームを維持してください。イベントがある場合、まとめたバッチを1回処理しても、同じイベントを複数のチャンクに分けて処理しても、同一のヒストグラムになるはずです。データセット全体を準備する前に、使用するSDKバージョンでこの性質を確認してください。

返された配列を`np.save("histogram.npy", histogram, allow_pickle=False)`で保存するか、学習済み検出器に直接渡してください。RAWファイルとライブカメラはSDKアプリケーションの担当です。この機能では、それらをLibreYOLOの推論ソースとして追加しません。

## データセットとラベルを準備

アノテーション済みのイベントウィンドウごとに、ヒストグラム1つとYOLO物体検出ラベルファイル1つを用意してください。

```text
event-dataset/
  data.yaml
  images/
    train/frame000001.npy
    val/frame000002.npy
  labels/
    train/frame000001.txt
    val/frame000002.txt
```

以下を`data.yaml`として保存し、`path`をデータセットの場所に、`names`をクラス名に置き換えてください。ここでは40,000マイクロ秒は40 msのイベントウィンドウ、16はカウントの飽和レベルを意味します。これは設定例であり、あらゆるシーンに推奨される取得時間ではありません。

<code-tabs name="dataset" />

`input_profile`の6つのフィールドはすべて必須です。LibreYOLOはリサイズ前に`clip(counts / scale, 0, 1)`を計算します。RGB/BGR変換、ImageNet正規化、255による除算は適用しません。スケールはfloat32で正かつ有限である必要があります。蓄積時間は正の整数でなければなりません。

ラベルは標準の`class_id cx cy width height`形式で、4つのボックス値は**元のヒストグラムキャンバス**に対して正規化します。

<code-tabs name="labels" />

この例では、画像中央にクラス0のボックスを1つラベル付けしています。ボックスはヒストグラムと同じイベントウィンドウに関連付けてください。近接するウィンドウが学習と検証の両方に入らないよう、録画またはシーン単位で分割してください。ラベルファイルがないか空の場合、ローダーは物体なしとして扱います。物体がないことが分かっている場合を除き、未アノテーションのウィンドウを負例サンプルとして含めないでください。

ほかのデータセット規則については[データセット](/docs/train/datasets)を参照してください。

## 入力を検査

<code-tabs name="inspect" />

可視化では、正の活動を赤、負の活動を青、活動なしを黒で表示します。この画像でラベルと推論結果を確認しつつ、モデル入力には数値の`.npy`配列を使ってください。

## 検出器を学習

[YOLO9](/docs/models/yolov9)または[RF-DETR](/docs/models/rf-detr)を選択してください。どちらの例もランダムな重みから開始し、選択されたチェックポイントを`event-detector.pt`として保存します。最初のワークフロー確認用に小さい128ピクセルのキャンバスを使います。最終的な解像度と学習時間はデータに合わせて選んでください。適切な場合は`device="cpu"`を学習デバイスに置き換えてください。

### YOLO9

<code-tabs name="yolo9" />

### RF-DETR

CLIの`pretrained=False`経路は、RGB重みを読み込まずにアーキテクチャを構築します。モデルファイル名でファミリーとサイズを指定します。

<code-tabs name="rfdetr" />

YOLO9はアスペクト比を維持し、ゼロでパディングします。RF-DETRは指定されたキャンバスに引き伸ばし、パッチとウィンドウの分割可能性に関する要件を維持します。ヒストグラム学習では水平・垂直反転に対応しています。RGBの色変更、mosaic、mixup、ランダムクロップ、アフィン変換および射影変換によるデータ拡張は無効です。サポートされないデータ拡張設定を明示すると、エラーになります。

転移学習では、互換性のあるRGBチェックポイントを読み込み、`pretrained=False`を省略してヒストグラムデータセットで`train()`を呼び出してください。LibreYOLOは入力畳み込み層を2チャンネルに適応し、互換性のある重みを保持します。それらの重みでもイベントデータによる学習が必要です。チェックポイントには、入力層がランダムに初期化されたか、RGBから適応されたかが記録されます。

現在の学習で必要なのは、1つのデバイスと固定の正のバッチサイズです。自動バッチ、CUDAグラフ学習、RF-DETRのマルチスケール学習、LoRA、蒸留、量子化ヒストグラムモデルは、このワークフローの対象外です。

## 再読み込み、推論、検証

保存されたモデルには入力プロファイルが含まれます。そのため、推論にデータセットYAMLや、スケールまたは極性の再指定は必要ありません。

<code-tabs name="predict" />

ボックスは元のキャンバス座標で返されます。単一の入力は`Results`を1つ返し、リスト入力はリストを返します。`save=True`を指定すると、別のカラー可視化画像に検出結果を描画します。`.npy`ファイルのディレクトリにも対応しています。

同じデータセットプロファイルと別の検証分割を使って検証してください。

<code-tabs name="val" />

異なるスケール、長さ、エンコーディング、チャンネル順は異なる入力契約です。検証と学習では、宣言されたプロファイルが読み込んだチェックポイントと一致しない場合に拒否します。推論では、誤ったラベルが付いた数値配列を検出できません。保存したモデルに合わせて生成元の設定を維持してください。

## ONNXをエクスポートして実行

<code-tabs name="onnx" />

ONNXファイルには入力プロファイルが保持されるため、LibreYOLOランタイムは同じ前処理を適用します。グラフ自体は2チャンネルの正規化済みfloat32 NCHWテンソルを受け取り、イベントのデコードや蓄積は行いません。PyTorchをインストールしていない環境では、`libreyolo.backends.onnx.OnnxBackend`を直接使ってください。

初期のヒストグラムエクスポート経路は、NMSを埋め込まないFP32 ONNXに対応しています。ほかのエクスポートランタイム、TTA、タイル分割はこのワークフローではサポートされません。

## 検証済みの内容

両方のファミリーで、準備済み配列と`.npy`ファイル、学習、チェックポイントの再読み込み、CLI推論、可視化の保存、ネイティブとONNXの比較を動作確認しています。準備済みデータの推論にSDKのインストールは必要ありません。小規模なPEDRoの例も変換して検査しましたが、これらの確認によってデータセット全体の精度や推奨される学習レシピが確立されたわけではありません。

OpenEBのランタイム実行、RAWデコード、ライブカメラでの動作、録画単位で分離した大規模な評価は未検証です。[実装PR](https://github.com/LibreYOLO/libreyolo/pull/865)に現在の検証範囲が記録されています。SDK連携の詳細は、[Metavision 5.3.1の前処理ドキュメント](https://docs.prophesee.ai/stable/guides/events_preprocessing.html)と、リビジョン[`9003b5416676e78ba994d912087486cfa94fae73`](https://github.com/prophesee-ai/openeb/tree/9003b5416676e78ba994d912087486cfa94fae73)の公開OpenEB Core APIを参照しています。
