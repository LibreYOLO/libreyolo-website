---
title: TensorRT
seo_title: LibreYOLOからTensorRTへエクスポート
description: >-
  LibreYOLOモデルからTensorRTエンジンを構築します。ONNX中間ファイル、FP16とINT8のビルド、動的バッチプロファイル、エンジンの移植性制限を解説します。
lead: >-
  TensorRTはグラフを1つのGPU向けに調整されたエンジンへコンパイルします。LibreYOLOは最初にONNX中間ファイルをエクスポートし、TensorRTのONNXパーサーで解析してエンジンを構築し、モデルのメタデータをJSONサイドカーとして隣に保存します。
keywords:
  - YOLO TensorRT エクスポート
  - TensorRT エンジン
  - TensorRT FP16
  - TensorRT INT8 キャリブレーション
  - 最適化プロファイル
  - TensorRT 動的バッチ
  - ハードウェア互換レベル
last_verified: 1.5.0
meta:
  - label: フラグ
    value: export(format="tensorrt")
    mono: true
  - label: 出力
    value: 1つの.engineファイルと.engine.jsonメタデータサイドカー
  - label: 追加パッケージ
    value: 'pip install "libreyolo[onnx,tensorrt]"'
    mono: true
  - label: 再読み込み
    value: LibreYOLO("weights/LibreYOLO9t.engine")
    mono: true
  - label: 形状
    value: デフォルトは静的。dynamic=Trueでバッチ軸の最適化プロファイルを追加します
  - label: 精度
    value: FP32、FP16（half=True）、INT8（int8=Trueとdata=を指定）
  - label: 必要条件
    value: ビルド時と実行時にNVIDIA GPUが必要です。GPUアーキテクチャが異なる環境へエンジンを移動することはできません。
verification: >-
  devブランチのlibreyolo/export/tensorrt.py、libreyolo/export/exporter.py、libreyolo/export/support.py、libreyolo/backends/tensorrt.py、pyproject.tomlを参照しました。
snippets:
  install:
    - label: インストール
      language: bash
      code: |
        # ONNX中間ファイルからエンジンを構築するため両方の追加パッケージが必要
        pip install "libreyolo[onnx,tensorrt]"
    - label: ビルド前にツールチェーンを確認
      language: bash
      code: >
        python -c "import tensorrt, torch; print(tensorrt.__version__,
        torch.cuda.is_available())"
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        #
        weights/LibreYOLO9t_fp16.engineとweights/LibreYOLO9t_fp16.engine.jsonを書き込む

        path = model.export(format="tensorrt", half=True)

        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format tensorrt --half
    - label: 引数
      language: python
      code: |
        model.export(
            format="tensorrt",
            imgsz=640,
            batch=1,
            half=False,
            int8=False,
            data=None,                      # int8=Trueの場合に必須
            dynamic=False,
            workspace=4.0,                  # ビルド時の作業領域 GiB
            min_batch=1,                    # 動的プロファイルの境界
            opt_batch=1,
            max_batch=8,
            hardware_compatibility="none",  # または"ampere_plus"
            gpu_device=0,                   # マルチGPUホストでのビルドデバイス
            verbose=False,
        )
  dynamic:
    - label: 動的バッチエンジン
      language: python
      code: |
        from libreyolo import LibreYOLO

        # プロファイルのバインド先を作るにはONNX中間ファイルに
        # 動的バッチ軸が必要
        LibreYOLO("LibreYOLO9t.pt").export(
            format="tensorrt",
            dynamic=True,
            min_batch=1,
            opt_batch=4,
            max_batch=8,
            half=True,
        )
  int8:
    - label: キャリブレーションデータを使うINT8
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="tensorrt",
            int8=True,
            data="coco128.yaml",   # 必須 この形式にはデフォルトがない
            fraction=1.0,
        )
  run:
    - label: LibreYOLO経由
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_fp16.engine")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: TensorRT単体
      language: python
      code: |
        import json

        import tensorrt as trt

        path = "weights/LibreYOLO9t_fp16.engine"
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(path, "rb") as handle:
            engine = runtime.deserialize_cuda_engine(handle.read())

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            print(engine.get_tensor_mode(name), name, engine.get_tensor_shape(name))

        # クラス名 タスク 入力サイズはエンジンではなくサイドカーにある
        # ここではバッファ確保 前処理 後処理を自分で実装する
        print(json.load(open(path + ".json"))["names"])
  support:
    - label: ビルド前にファミリーとタスクの組み合わせを確認
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: cb90fc98ab735233
---

## インストール

ビルドと実行の両方で、正常に動作するCUDA環境を備えたNVIDIA GPUが必要です。この形式にはCPUフォールバックがありません。

<code-tabs name="install" />

`tensorrt`追加パッケージは`tensorrt-cu12`と`pycuda`のバージョンを固定し、マーカーによってmacOSでは両方を除外します。Jetsonではこの追加パッケージを使用しないでください。CUDA 13プラットフォームに対してCUDA 12ビルドが固定されるためです。代わりに、[NVIDIA Jetson](/docs/export/jetson)の説明に従ってJetPackがインストールするTensorRTを使用してください。

## エクスポート

<code-tabs name="export" />

エクスポートは2段階で実行されます。第1段階で一時パスにONNX中間ファイルを書き込み、第2段階でそれを解析してエンジンを構築し、その後に中間ファイルを削除します。`workspace`はGiB単位のビルド時作業メモリです。値を大きくするとビルダーがより多くのカーネルを試せますが、推論時のメモリには影響しません。

メタデータサイドカーは`<engine>.json`としてエンジンの隣に書き込まれ、ビルドで実際に実現した精度を記録します。GPUに高速なFP16やINT8がない場合、ビルダーは警告してフォールバックします。サイドカーには要求した精度ではなく、実際に生成された精度が記録されます。

FP16では、グラフ内のViTバックボーンを検出し、その浮動小数点レイヤーをFP32に固定します。DINOv2形式のバックボーンはFP16でオーバーフローしてNaNを生成するため、ビルドは`OBEY_PRECISION_CONSTRAINTS`を設定し、`FP16 (FP32 ViT backbone)`と報告します。CNNバックボーンではこの処理は何も変更しません。

### 動的バッチ

<code-tabs name="dynamic" />

`dynamic=True`は`min_batch`から`max_batch`までを対象とし、`opt_batch`で最適化する1つの最適化プロファイルを追加します。これら3つの値はサイドカーにも記録されます。プロファイルが追加されるのは、ONNX中間ファイルに実際に動的バッチ次元がある場合だけです。それ以外では、静的最適化を使用することをログに記録して処理を続行します。

### INT8

<code-tabs name="int8" />

INT8はLibreYOLOのキャリブレーションローダーに対してTensorRTのエントロピーキャリブレーターを使用します。`data`は必須で、この形式には8枚の画像を使うフォールバックがありません。キャリブレーションのデバイスバッファには`cuda-python`または`pycuda`が必要です。キャリブレーションキャッシュのキーはONNXバイト列のハッシュなので、同じ出力パスに書き込む別モデルへスケールが再利用されることはありません。

`half=True`と`int8=True`を同時に指定すると警告し、TensorRTで量子化できないレイヤー用にFP16フォールバックを保持したINT8をビルドします。

## アーティファクトを実行

<code-tabs name="run" />

`LibreYOLO()`は`.engine`接尾辞に基づいて振り分け、サイドカーからクラス名、タスク、姿勢スキーマを読み取り、チェックポイントと同じ`Results`オブジェクトを返します。CUDAデバイスが存在しない場合は直ちに例外を送出します。

2つ目のスニペットはランタイムだけを使う方法です。ホストとデバイスのバッファ確保、前処理、デコード、NMS、座標の再スケーリングをすべて自分で実装する必要があります。エンジン自体にはクラス名が含まれないため、サイドカーも一緒に移動する必要があります。

## 制約

シリアライズ済みエンジンは、ビルドに使用したGPUアーキテクチャ、ドライバースタック、TensorRTバージョンに依存します。ワークステーションで構築したエンジンは別のアーキテクチャでは読み込めません。そのため、ビルド手順はデプロイ先のマシンで実行します。`hardware_compatibility="ampere_plus"`は、一部の性能と引き換えにAmpere以降の間で移植性を高めます。`"same_compute_capability"`の値は`NONE`にマッピングされ、警告を出します。エンジンは現在のGPUだけに最適化され、エクスポートは適用していない移植性を主張せず、その事実を明示します。

プロファイルの対象はバッチ軸だけです。動的な空間次元を持つビルドはこの契約の対象外なので、FCOSはブロックされます。FCOSは800×1333のアスペクト変換を維持するため、動的にパディングされた高さと幅を必要とします。

トレース前にブロックされる組み合わせは、YOLO9のセグメンテーション、RTMDet-Insのセグメンテーション、SSD、Faster R-CNN、RetinaNetの検出、BiRefNetまたはFeyNobgのマッティングです。これらではTensorRT 10.16が共有ONNXの`DeformConv`ノードに到達しますが、プラグインレジストリに`ModulatedDeformConv2d`がないため解析できません。

検証済みにもブロック対象にも分類されていない組み合わせでは変換処理を利用できますが、プロジェクトはTensorRTランタイムでの一致を記録していません。これは根拠の有無に関する記述であり、ビルドが成功するかどうかを示すものではありません。

ファミリーとタスクの完全なグリッドについては、[エクスポートマトリックス](/docs/reference/export-matrix)を参照してください。1つの組み合わせを確認するには、次のようにします。

<code-tabs name="support" />
