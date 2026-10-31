---
title: NVIDIA DeepStream
seo_title: NVIDIA DeepStreamでYOLOモデルを動かす
description: >-
  LibreYOLOモデルをNVIDIA
  DeepStream向けにエクスポートすると、ONNXグラフと生成されたnvinfer設定が出力されます。パーサーのビルドとパイプラインの正確なコマンドを掲載します。
lead: >-
  NVIDIA
  DeepStreamはnvinferエレメントを通して推論を実行し、そこにはONNXグラフ、対応する設定ファイル、バウンディングボックスパーサーが必要です。ONNXエクスポートでdeepstream=Trueを指定すると、前の2つが出力され、3つ目に接続されます。
keywords:
  - NVIDIA DeepStream
  - deepstream yolo 物体検出
  - nvinfer 設定
  - deepstream カスタムパーサー
  - config_infer_primary
  - NvDsInferParseYolo
  - deepstream-app 使い方
  - TensorRT エンジン 変換
  - Jetson 推論 デプロイ
meta:
  - label: フラグ
    value: 'export(format="onnx", deepstream=True)'
    mono: true
  - label: 出力
    value: ONNXグラフ、config_infer_primary_<stem>.txt、<stem>_labels.txt
  - label: 対応範囲
    value: 9タスクにまたがる43通りのファミリーとタスクの組み合わせ
  - label: パーサー
    value: >-
      Marcos
      LucianoによるMITライセンスのDeepStream-YoloプロジェクトのNvDsInferParseYoloです。デバイスごとに1回ビルドします。
    links:
      - label: github.com/marcoslucianops/DeepStream-Yolo
        href: 'https://github.com/marcoslucianops/DeepStream-Yolo'
  - label: 提供状況
    value: v1.5.0で提供されます。2026-08-08にプルリクエスト728でdevへマージされました。
    links:
      - label: プルリクエスト728
        href: 'https://github.com/LibreYOLO/libreyolo/pull/728'
      - label: issue 648
        href: 'https://github.com/LibreYOLO/libreyolo/issues/648'
  - label: 実機検証
    value: RTX 5070 Ti上のDeepStream 8.0.0、検出のみ、2026-08-08
verification: >-
  2026-08-08の実機検証をもとに執筆しました。ファミリー一覧、設定キー、デフォルト値は、同日にプルリクエスト728でdevへマージされたコミット5f81e11eのlibreyolo/export/deepstream.pyおよびlibreyolo/export/exporter.pyから読み取っています。
snippets:
  install:
    - label: インストール
      language: bash
      code: |
        pip install "libreyolo[onnx]"
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO9, LibreDFINE


        # libreyolo9s.onnx、config_infer_primary_libreyolo9s.txt、

        # libreyolo9s_labels.txt を作業ディレクトリに出力

        LibreYOLO9("libreyolo9s.pt", size="s").export(format="onnx",
        deepstream=True)


        # 検出モデルはそれぞれ別のディレクトリに置く。どの検出用設定も

        # 同じエンジンキャッシュファイル名を指すため。「よくある落とし穴」を参照

        LibreDFINE("LibreDFINEs.pt", size="s").export(format="onnx",
        deepstream=True)
    - label: 引数
      language: python
      code: |
        model.export(
            format="onnx",     # deepstream=True は他のどの形式でも拒否される
            deepstream=True,
            conf=0.25,         # pre-cluster-threshold の初期値（該当タスクでは
                               # classifier-threshold、segmentation-threshold も）
            iou=0.45,          # nms-iou-threshold の初期値、cluster-mode=4 では省略
            batch=1,           # batch-size とエンジンキャッシュのファイル名の初期値
            half=False,        # True で設定に network-mode=2 を記録（fp16 ビルド）
            int8=False,        # True で設定に network-mode=1 を記録
            dynamic=True,      # ONNX グラフの動的バッチ軸
            imgsz=640,         # infer-dims=3;H;W の初期値
        )

        # deepstream=True と nms=True は排他。DeepStream はクラスタリング段階で
        # 抑制を行うため、グラフには何も埋め込まれない
    - label: 先にD-FINEの重みを取得
      language: bash
      code: |
        curl -L -o LibreDFINEs.pt \
          https://huggingface.co/LibreYOLO/LibreDFINEs/resolve/main/LibreDFINEs.pt
  gpu:
    - label: 何よりも先にGPUパススルーを確認
      language: bash
      code: |
        docker run --rm --gpus all nvcr.io/nvidia/tritonserver:26.04-py3 \
          nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
      expect: |
        name, driver_version, compute_cap
        NVIDIA GeForce RTX 5070 Ti, 591.86, 12.0
  parser:
    - label: build_parser.sh、DeepStreamコンテナー内で実行
      language: bash
      code: >
        set -e

        git clone --depth 1
        https://github.com/marcoslucianops/DeepStream-Yolo.git


        # このイメージの /usr/local/cuda-12 はスタブで、ビルドは

        # "fatal error: crt/host_defines.h: No such file or directory" で失敗する。

        # ヘッダーを実際に含むツールキットを解決する。8.0 イメージでは cuda-12.5

        CUDA_DIR=$(readlink -f /usr/local/cuda)

        [ -f "$CUDA_DIR/include/crt/host_defines.h" ] || \
          CUDA_DIR=$(ls -d /usr/local/cuda-*.* | sort -Vr | \
                     while read d; do [ -f "$d/include/crt/host_defines.h" ] && echo "$d" && break; done)

        # イメージには libcublas.so.12 と libcublas.so.12.8.4.1 はあるが、-lcublas が

        # 必要とするバージョンなしの libcublas.so がないため、リンク段階が

        # "/usr/bin/ld: cannot find -lcublas" で失敗する。リンカーが求める名前を用意する

        mkdir -p /tmp/cudalibs

        for lib in cublas cublasLt cudart; do
          real=$(find /usr/local -name "lib${lib}.so.1*" | grep -v stubs | sort -V | tail -1)
          ln -sf "$real" "/tmp/cudalibs/lib${lib}.so"
        done

        export LIBRARY_PATH="/tmp/cudalibs:$LIBRARY_PATH"


        make -C DeepStream-Yolo/nvdsinfer_custom_impl_Yolo
        CUDA_VER="${CUDA_DIR##*/cuda-}"
    - label: インスタンスセグメンテーションは別のパーサーを使う
      language: bash
      code: >
        git clone --depth 1
        https://github.com/marcoslucianops/DeepStream-Yolo-Seg.git

        make -C DeepStream-Yolo-Seg/nvdsinfer_custom_impl_Yolo_seg \
          CUDA_VER="${CUDA_DIR##*/cuda-}"
  run:
    - label: deepstream_app_config.txt
      language: text
      code: >
        [application]

        enable-perf-measurement=1

        perf-measurement-interval-sec=5

        gie-kitti-output-dir=kitti


        [tiled-display]

        enable=0


        [source0]

        enable=1

        type=3

        uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4

        num-sources=1

        gpu-id=0


        [streammux]

        gpu-id=0

        batch-size=1

        batched-push-timeout=40000

        width=1920

        height=1080

        live-source=0


        [primary-gie]

        enable=1

        gpu-id=0

        gie-unique-id=1

        config-file=config_infer_primary_libreyolo9s.txt


        [osd]

        enable=1

        border-width=2

        text-size=15


        [sink0]

        enable=1

        type=1

        sync=0


        [tests]

        file-loop=0
    - label: 実行する
      language: bash
      code: |
        deepstream-app -c deepstream_app_config.txt
      expect: |
        App run successful
    - label: 両方の手順を1つのコンテナーで
      language: bash
      code: |
        docker run --rm --gpus all -v "$PWD:/work" -w /work \
          nvcr.io/nvidia/deepstream:8.0-samples-multiarch \
          bash -c "bash build_parser.sh && deepstream-app -c deepstream_app_config.txt"
source_hash: 1ee91c265753dd9a
---

## 提供状況

DeepStreamエクスポートはv1.5.0で提供されます。2026-08-08にプルリクエスト728で
`dev`へマージされたため、最新のインストールにはすでに含まれており、ブランチを
固定する必要はありません。

<code-tabs name="install" />

2026-08-08より前に`deepstream-export`ブランチをクローンしている場合は、取り直して
ください。このブランチはリベースして強制プッシュされており、古い履歴にはこれらの
エクスポートをCUDAマシンで動かすための修正が入っていません。

## エクスポートが出力するもの

`model.export(format="onnx", deepstream=True)`は3つのファイルを並べて出力します。
`libreyolo9s.pt`の場合は次のとおりです。

- `libreyolo9s.onnx`。検出グラフで、出力テンソルは`(batch, num_detections, 6)`の
  形状が1つ、各行は`[x1, y1, x2, y2, score, class_id]`で、ネットワーク入力の
  ピクセル座標です。
- `config_infer_primary_libreyolo9s.txt`。ファミリーの前処理定数、クラス数、
  しきい値、パーサーの接続設定を持つ`nvinfer`の設定ファイルです。
- `libreyolo9s_labels.txt`。1行に1つのクラス名です。

ラベルファイルは、チェックポイントがクラス名を持っている場合に出力されます。深度
推定モデルはクラス名を持たないため、ファイルも`labelfile-path`キーも出力されません。

LibreYOLOは`.so`を出力しません。DeepStreamが読み込む`.so`は
`marcoslucianops/DeepStream-Yolo`のバウンディングボックスパーサーで、デバイスごとに
1回コンパイルすればよく、どのLibreYOLO検出器に向けても同じバイナリです。モデルに
あたるのがONNXです。`nvinfer`が自前で後処理を行うため、分類とセマンティック
セグメンテーションはパーサーをまったく必要としません。

## モデルをエクスポートする

<code-tabs name="export" />

`LibreDFINE._load_weights`は、ファイルがまだディスク上にない場合にダウンロードを
試みずに`FileNotFoundError`を送出するため、先に`LibreDFINEs.pt`を自分で取得して
ください。この不足は
[issue #727](https://github.com/LibreYOLO/libreyolo/issues/727)として追跡されて
います。YOLO9の重みは初回使用時にダウンロードされます。

このフラグはPythonのみです。このブランチの`libreyolo export`に`deepstream`
オプションはなく、CLIは未知のキーをそのまま渡すのではなく、固定のリストから
エクスポート引数を組み立てます。

## バウンディングボックスパーサーをビルドする

検出にはパーサーライブラリが必要で、インスタンスセグメンテーションには別のものが
必要、残りのタスクには不要です。DeepStream 8.0イメージには、ドキュメントに載って
いるビルドコマンドを壊す点が2つあり、どちらもLibreYOLOではなく環境側の問題です。

イメージには`/usr/local`の下に`cuda`、`cuda-12`、`cuda-12.5`、`cuda-12.8`、
`cuda-12.9`が入っています。ツールキットが完全なのは`cuda-12.5`だけです。また、
`libcublas.so.12`と`libcublas.so.12.8.4.1`は入っていますが、`-lcublas`が解決先と
するバージョンなしの`libcublas.so`はありません。以下のスクリプトはその両方を回避
します。

<code-tabs name="parser" />

そのうえで、生成された設定の`custom-lib-path`を、ビルドした
`libnvdsinfer_custom_impl_Yolo.so`に向けてください。生成される値は相対パスの
`nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so`で、`deepstream-app`を
`DeepStream-Yolo`のチェックアウト先から実行する場合は解決できますが、それ以外では
編集が必要です。

## パイプラインを実行する

ほかに時間をかける前に、コンテナーからGPUが見えることを確認してください。これは
検証時に最初に行ったチェックで、WSL2上のBlackwellカードで実施しました。

<code-tabs name="gpu" />

検証では、ファイルソース1つ、表示シンクなし、オンスクリーンディスプレイ有効、
`gie-kitti-output-dir`を設定して全フレームの検出結果をKITTI形式のテキストとして
ディスクに書き出す構成で`deepstream-app`を動かしました。その設定を反映した設定
ファイルは次のとおりです。

<code-tabs name="run" />

`nvinfer`は初回実行時にONNXからTensorRTエンジンをビルドしてモデルの隣にキャッシュ
するため、初回はエンジンのビルド分の時間がかかり、2回目以降はキャッシュを読み込み
ます。

## 生成される設定ファイル

以下の2つの設定はいずれも検証時にエクスポーターが生成したもので、あとから編集して
いません。

| キー | YOLO9-s | D-FINE-s |
|---|---|---|
| `net-scale-factor` | 0.003921568627 | 0.003921568627 |
| `model-color-format` | 0 | 0 |
| `infer-dims` | 3;640;640 | 3;640;640 |
| `maintain-aspect-ratio` | 1 | 0 |
| `symmetric-padding` | 0 | 0 |
| `network-type` | 0 | 0 |
| `num-detected-classes` | 80 | 80 |
| `cluster-mode` | 2 | 4 |
| `parse-bbox-func-name` | NvDsInferParseYolo | NvDsInferParseYolo |
| `pre-cluster-threshold` | 0.25 | 0.25 |
| `nms-iou-threshold` | 0.45 | |
| `topk` | 300 | 300 |

2つの設定が異なるのは3か所、`maintain-aspect-ratio`、`cluster-mode`、そして
`nms-iou-threshold`がそもそも存在するかどうかです。D-FINEの設定はこのキーを完全に
省いており、これは`cluster-mode=4`が求める形です。

クラスタリングは本来別々の検出結果を統合してしまうため、物体ごとに最大1つしか
予測を出さないヘッドには`cluster-mode=4`が設定され、DeepStreamはそれらに対して
クラスタリングを行いません。対象は`rfdetr`、`dfine`、`deim`、`deimv2`、`ec`、
`rtdetr`、`rtdetrv2`、`rtdetrv4`、`yolo9_e2e`です。グリッドヘッドとアンカーヘッドに
は`cluster-mode=2`と`nms-iou-threshold`が設定されます。

検出用の設定にはさらに`engine-create-func-name=NvDsInferYoloCudaEngineGet`が入って
おり、エンジンのビルドをパーサーライブラリに委ねます。これによってエンジン
キャッシュのファイル名が固定され、よくある落とし穴で説明する衝突の原因になって
います。

## 対応するタスクとファミリー

エクスポートできるファミリーとタスクの組み合わせは43通りです。
`libreyolo/export/deepstream.py`の`deepstream_supported_tasks()`と
`deepstream_supported_families(task)`は、実行時に同じ一覧を返します。

| タスク | `network-type` | パーサーライブラリ | ファミリー |
|---|---|---|---|
| 検出 | 0 | DeepStream-Yolo | yolo9, yolo9_p2, yolo9_e2e, yolo1, yolo2, yolo3, yolo4, yolo7, yolox, yolonas, rtmdet, picodet, rfdetr, dfine, deim, deimv2, ec, rtdetr, rtdetrv2, rtdetrv4 |
| 分類 | 1 | 不要 | mobilenetv4, convnext, efficientnetv2, resnet, dinov2 |
| セマンティックセグメンテーション | 2 | 不要 | pidnet, eomt, dinov2, lingbotvision |
| インスタンスセグメンテーション | 3 | DeepStream-Yolo-Seg | rfdetr, dfine, ec |
| 姿勢推定 | 100 | 不要 | yolo9, yolonas, rfdetr, ec |
| 深度推定 | 100 | 不要 | depth_anything, zipdepth |
| 復元 | 100 | 不要 | nafnet, realesrgan, swinir |
| マッティング | 100 | 不要 | birefnet |
| 視線推定 | 100 | 不要 | l2cs |

`network-type=100`は、そのタスク向けの後処理器をDeepStreamが持たないことを意味
します。これらの設定は`output-tensor-meta=1`を指定し、グラフ本来の出力はそのまま
素通りして、アプリケーション側がテンソルのメタデータからデコードします。複数出力の
グラフでも問題ありません：すべての出力レイヤーが、通常のONNXエクスポートと同じ
出力名と動的軸のままメタデータに届きます。

インスタンスセグメンテーションの行は、検出行のあとにそのインスタンスのマスクが
続く形で、マスクはsegパーサーがハードコードしている解像度`(netH / 4, netW / 4)`に
平坦化され、`segmentation-threshold`用の確率として並びます。

分類と視線推定はセカンダリ推論として動作します。検出器の後段に分類器を置くには、
生成された設定で`process-mode=2`と`operate-on-gie-id`を指定してください。視線推定
はヘッドのみの契約で、入力は顔のクロップ1枚ずつになるため、前段に顔検出器が必要
です。

3つのファミリーは意図的に外してあります。`segformer`は共通のセマンティック
エクスポート契約につながっておらず、どの形式でもONNXへエクスポートできません。
RTMDet-InsとYOLO9は、インスタンスセグメンテーションのエクスポートがLibreYOLO側で
ブロックされています。`depth_anything3`にはエクスポートの実装がありません。

表の2行にはチェックポイントの不足が隠れています。EoMTのセマンティック
チェックポイントは`l`だけが公開されており、DINOv2の分類は公開チェックポイントが
まったくないため、この組み合わせには自分でファインチューニングした重みが必要です。

## 前処理の違い

`nvinfer`はチャンネルごとに`net-scale-factor * (x - offsets)`をスカラーのスケール
で計算するため、チャンネルごとの標準偏差を表現できません。それが必要なファミリー
（`rfdetr`、`ec`、DINOバックボーンの`deimv2`サイズ、`rtmdet`、`picodet`、および
すべての分類ファミリー）は、正規化をエクスポートしたグラフに焼き込んであり、生成
された設定はそれに合う生の入力空間をグラフへ渡します。

LibreYOLO自身のPythonパイプラインと`nvinfer`が依然として食い違うのは、ジオメトリ
の部分です。

- レターボックス方式のファミリー（`yolo9`、`yolox`、`yolonas`、`rtmdet`、`yolo2`、
  `yolo3`、`yolo4`、`yolo7`）は、本来グレーでパディングします。`nvinfer`は黒で
  パディングします。
- `yolonas`の検出は、本来640のキャンバス内で長辺を636にリサイズします。`nvinfer`の
  `maintain-aspect-ratio`は640をそのまま使います。
- 分類は本来、短辺をリサイズしてから中央でクロップします。`nvinfer`はフレーム
  または物体のROIをネットワーク入力へ引き伸ばすため、被写体が詰まったクロップでは
  結果が変わります。
- EoMTは本来、セマンティックセグメンテーションでスライディングウィンドウのタイル
  処理を行います。エクスポートしたグラフは引き伸ばした単一のキャンバスで、高速です
  が精度は下がります。
- `pidnet`は入力解像度の1/8、`lingbotvision`は1/16でクラスマップを出力します。
  DeepStreamは表示のためにクラスマップをアップサンプリングします。

ONNXの一致検証は前処理済みのテンソルを入力するため、グラフの出力は確認できますが、
設定内の誤ったカラー順序やパディング方針は捕まえられません。厳密な一致が要る
ワークロードをデプロイする前に、自分のデータで検証してください。

## よくある落とし穴

### 同じディレクトリの2つの検出モデルが互いのエンジンを読み込む

どの検出用設定にも同じ行が入っています。

```ini
model-engine-file=model_b1_gpu0_fp32.engine
```

パーサーのエンジンビルダーはこのベース名を要求し、モデルによって変わりません。
2つ目の検出モデルを同じディレクトリにエクスポートすると、2回目の実行では1つ目の
モデルのキャッシュ済みエンジンが読み込まれます。クラッシュはせず、ボックスが誤る
だけです。検出モデルにはそれぞれ専用のディレクトリを用意してください。検証時も、
D-FINEを別のディレクトリに隔離してはじめてテストできました。

### 1つのボックスは1つのクラスしか持てない

`nvinfer`の行フォーマットは`[x1, y1, x2, y2, score, class_id]`で、1ボックスにつき
1クラスのため、エクスポートではクラススコアをargmaxに畳み込みます。`predict`が
2つのクラスで報告するボックスは、片方のクラスだけが残ります。実測例として、
LibreYOLOは同じボックスに`vase 0.773`と`bottle 0.383`を報告しますが、DeepStreamの
グラフは`vase`を残します。これはパーサーの行フォーマットから生じるもので、その契約
を外れない限り変えられないため、リグレッションではなく想定どおりの動作です。

## 検証済み

`deepstream-app`は、NVIDIA同梱の`sample_1080p_h264.mp4`（1443フレーム）に対して
フレームごとのKITTI出力を有効にした状態で、どちらの検出ヘッド型でもEOSまで走り、
`App run successful`を出しました。

| | YOLO9-s | D-FINE-s |
|---|---|---|
| ヘッド型 | grid | one-to-one |
| `cluster-mode` | 2 | 4 |
| `maintain-aspect-ratio` | 1 | 0 |
| 検出のあったフレーム数 | 1443 | 1443 |
| 総検出数 | 18031 | 71105 |

1443フレーム全体のクラスヒストグラムは、どちらのモデルでも1位が車、2位が人で、
街の風景として妥当です。検出数の4倍の開きは`cluster-mode`の違いが働いた結果です：
`cluster-mode=4`のD-FINEはクラスタリングを行わないため、しきい値を超えたクエリは
ほぼ重複しているものも含めてすべて残ります。

独立に学習された2つのモデルが、主要な物体を同じ位置に置いています。

```text
YOLO9  bus  [706.72,  0.82, 1916.34, 1062.97]  conf 0.965
D-FINE bus  [702.73,  2.93, 1916.24, 1069.32]  conf 0.965
```

この実行で5点が確認できました：sm_120でTensorRTがエクスポートしたONNXから
エンジンをビルドすること、`nvinfer`が生成された設定のすべてのキーを受け付けること、
`NvDsInferParseYolo`がテンソルのレイアウトを正しく読むこと、ボックスがソース解像度
である1920x1080の座標に収まること、そしてラベルが生成されたラベルファイルで解決
されることです。

実行環境は次のとおりです。

| 構成要素 | 値 |
|---|---|
| ホストOS | Windows 11 Pro 26200 |
| GPU | NVIDIA GeForce RTX 5070 Ti、16 GB |
| ドライバー | 591.86 |
| Compute Capability | 12.0（Blackwell、sm_120） |
| コンテナーランタイム | Docker Desktop 29.4.3、WSL2バックエンド |
| DeepStreamイメージ | `nvcr.io/nvidia/deepstream:8.0-samples-multiarch` |
| DeepStreamバージョン | 8.0.0 |
| コンテナーのCUDA | 12.8.1 |
| パーサー | `marcoslucianops/DeepStream-Yolo`のHEAD |

パイプラインの実行と併せて、`tests/unit/test_deepstream_export.py`がグラフ
アダプターと生成される設定キーをカバーしており、その35件のテストはこのコミットで
通過します。

## 未検証

上記の範囲を実際より広く読まれないよう、明記しておきます。

- Jetsonとaarch64。エクスポートの契約はアーキテクチャに依存しませんが、パイプライン
  はx86のディスクリートGPUでしか実行していません。
- 43通りのうち41通り。DeepStreamを通したのは`yolo9`の検出と`dfine`の検出だけです。
  分類、セマンティックセグメンテーション、インスタンスセグメンテーション、および
  生テンソルのタスクは、パイプラインの実行ではなくユニットテストとONNXの一致
  チェックでカバーしています。
- FP16とINT8。動かしたのは`network-mode=0`だけです。
- マルチストリームとバッチ処理。ソースは1つ、`batch-size=1`です。
- 正解データ（ground truth）のデータセットに対する精度。検出結果は意味的な妥当性と
  モデル間の一致を確認しただけで、DeepStreamを通してmAPとしてスコア化しては
  いません。
