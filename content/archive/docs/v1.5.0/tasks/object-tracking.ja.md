---
title: 物体追跡
seo_title: LibreYOLOの物体追跡
description: >-
  LibreYOLOでByteTrack、BoT-SORT、OC-SORT、Deep
  OC-SORTを使い、任意の検出・セグメンテーション・姿勢モデルから動画フレーム間の物体を追跡します。
lead: >-
  追跡では、動画フレームをまたいで各検出結果に安定したIDを割り当てます。LibreYOLOは追跡を独自の重みを持つタスクとして扱いません。追跡は推論モードの
  `model.track()` であり、選択したトラッカーを検出・セグメンテーション・姿勢モデルのフレームごとの出力に適用します。
keywords:
  - 物体追跡 python
  - マルチオブジェクトトラッキング
  - bytetrack
  - botsort
  - ocsort
  - deep ocsort
  - track id
  - reid tracking
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # track()はジェネレーターで、処理したフレームごとに1つのResultsを返す
        for result in model.track("video.mp4"):
            print(result.track_id)        # ボックスと対応する(N,)整数テンソル
            print(result.boxes.xyxy)
    - label: トラッカーを選ぶ
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # "bytetrack" (デフォルト)、"botsort"、"ocsort"、"deepocsort"のいずれか
        for result in model.track("video.mp4", tracker="botsort"):
            print(result.track_id)
    - label: アノテーション付き動画を保存
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # output_pathを省略するとruns/track/<video_stem>.mp4に保存
        for result in model.track("video.mp4", save=True, vid_stride=2):
            pass
    - label: トラッカーを調整
      language: python
      code: >
        from libreyolo import BoTSortConfig, LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # 設定の型でトラッカーが選ばれるため、ここではtracker=は重複指定

        config = BoTSortConfig(track_buffer=60, frame_rate=25, enable_cmc=False)

        for result in model.track("video.mp4", tracker_config=config):
            print(result.track_id)

        # 同じフィールドをキーワード引数で渡し、track()に構築させる方法もある

        for result in model.track("video.mp4", tracker="botsort",
        track_buffer=60):
            print(result.track_id)
source_hash: f1fa7dcf60597d6b
---

## 定義

追跡はLibreYOLOのタスクキーの1つではなく、ダウンロードする追跡チェックポイントもありません。これはモデルのメソッド `model.track(source)` であり、各フレームで検出を実行して、時間の経過に沿って結果を関連付けます。このメソッドはジェネレーターです。処理したフレームごとに1つの `Results` を生成し、`result.boxes` と対応する `(N,)` 整数テンソルが `result.track_id` に設定されます。同じIDは `result.boxes.id` にもあります。

確認済みで現在追跡中の物体だけが生成されます。関連付けで見失ったトラックは、設定されたフレーム数だけ存続してから破棄されます。ByteTrackとBoT-SORTでは `track_buffer`、2つのOC-SORTバリアントでは `max_age` で設定するため、この期間内に再検出された物体は元のIDを保持します。

関連付けは検出後に行われるため、フレームのほかのペイロードも維持されます。追跡後の `Results` は検出時の `Results` を一致した行に絞り込んだもので、マスクとキーポイントもボックスとともに引き継がれます。

## モデル

追跡の実行には、フレームごとにボックスを生成するモデルと、それらを結び付けるトラッカーという2つの独立した選択肢があります。

タスクが検出、セグメンテーション、姿勢推定のいずれかであるLibreYOLOネイティブモデルは、すべて `track()` を提供します。そのため、検出器は通常どおり選べます。全リストについては[モデル一覧](/docs/models)を参照するか、[YOLO9](/docs/models/yolov9)、[RF-DETR](/docs/models/rf-detr)、[D-FINE](/docs/models/d-fine)、[RTMDet](/docs/models/rtmdet)から始めてください。関連付けるボックスが結果にないタスクは、意味のないIDを返さず、呼び出しを拒否します。分類、回転ボックス、点、深度、サーフェス法線、エッジ、セマンティックおよびパノプティックセグメンテーション、復元、OCR、ボディメッシュでは、すべて `track()` から例外が送出されます。

LibreYOLOの2つのモデル階層も追跡に対応しません。`LibreSAM` で読み込むモデルは画像セグメンターで、`LibreOpenVocab` で読み込むモデルはフレームごとの検出器です。どちらも `track()` から例外を送出するため、代わりにフレームごとに `predict()` を使います。

追跡はネイティブPyTorchモデルで実行されます。`LibreYOLO("model.onnx")` で読み込んだエクスポート済み成果物はランタイムバックエンドオブジェクトを返します。このオブジェクトは `predict()` を提供しますが、`track()` は提供しません。

ライブラリには4つのトラッカーが含まれ、`tracker` 引数で選択します。

`"bytetrack"` がデフォルトです。動きだけを使い、Kalman filterと3段階の関連付けを備えます。まず信頼度の高い検出結果、次に破棄前の信頼度の低い検出結果を既存トラックと一致させる第2パス、最後に未確認トラックを処理します。`TrackConfig` で設定します。

`"botsort"` はByteTrackの3段階のライフサイクルを維持しながら、中心・幅・高さのKalman状態を使い、照合前に予測トラックをカメラの動きに合わせて補正します。これはBoT-SORTの動きだけを使うバリアントであり、外観モデルは実行しません。`BoTSortConfig` で設定し、`enable_cmc`、`cmc_method`、`cmc_downscale` が追加されています。

`"ocsort"` も動きだけを使い、関連付けコストに速度方向の項を加えます。また、各トラックで最後に得た実際の観測結果に対する第2の関連付けパスと、トラックを再検出したときに仮想軌跡に沿ってKalman状態を平滑化する処理を追加します。`OCSortConfig` で設定します。

`"deepocsort"` はOC-SORTを外観情報で拡張します。各トラックは再識別用埋め込みベクトルの信頼度加重移動平均を保持し、cosine類似度の項が関連付けコストに加わるため、長時間の遮蔽や対象の交差があってもIDを維持できます。フレームごとに小規模な埋め込みネットワークの順伝播を1回行うコストがかかり、OSNetの重みは初回使用時にダウンロードされます。`DeepOCSortConfig` で設定します。

## 推論

<code-tabs name="predict" />

`track_conf` は最初の関連付け段階のしきい値を設定します。ByteTrackとBoT-SORTでは `track_high_thresh`、OC-SORTとDeep OC-SORTでは `det_thresh` に対応します。これは `predict()` の `conf` ではありません。またByteTrack、BoT-SORT、OC-SORTでは、弱い検出結果を回復パスで利用できるよう、内部で検出器をより低いしきい値で実行します。Deep OC-SORTでは、検出器自体を `det_thresh` で実行します。ByteTrackとBoT-SORTでは、`track_conf` を `track_low_thresh` 以上にする必要があり、後者のデフォルトは0.1です。

トラッカー設定は2つの方法で渡せます。設定インスタンスを `tracker_config=` に渡すと、その型でトラッカーが選ばれるため、`tracker=` は重複指定になります。または、フィールドをキーワード引数として渡し、`track()` で指定したトラッカーの設定を構築できます。不明なキーは黙って適用されず、警告が表示されます。どちらの方法でも、対応するキーを明示的に設定すると `track_conf` は無視されます。

残りの引数は推論と同じです。`iou`、`imgsz`、`classes`、`max_det`、`vid_stride`、`show`、および `output_path` と組み合わせる `save` を指定できます。ソースは動画ファイルのパスです。結果の処理については[推論](/docs/predict)を参照してください。

## 学習

トラッカーは学習しません。4つのうち3つは学習パラメータをまったく持たない純粋な動きモデルであり、Deep OC-SORTの外観ネットワークは初回使用時にダウンロードされる公開済みの再識別チェックポイントです。追跡品質を向上させるには、検出器を改善するか、上記の関連付けしきい値を調整します。

