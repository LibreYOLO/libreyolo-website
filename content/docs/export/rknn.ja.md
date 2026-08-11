---
title: RKNN
seo_title: Rockchip NPU向けRKNNへエクスポート
description: >-
  LibreYOLOの検出器をRockchipの.rknnアーティファクトへコンパイルします。別途導入するベンダーSDK、検証済みの4つのRK3588バリアント、シミュレーターでの一致確認を解説します。
lead: >-
  RKNNはRockchipのコンパイル済みNPU形式です。LibreYOLOはopset 19のONNX中間ファイルをエクスポートし、RKNN
  Toolkit2 SDKでコンパイルします。ボードなしでも、Toolkit2のホストシミュレーターでコンパイル済みグラフとONNX
  Runtimeの結果を比較できます。
keywords:
  - YOLO RKNN エクスポート
  - Rockchip NPU
  - RK3588
  - RKNN Toolkit2
  - RKNN シミュレーター 精度比較
  - Orange Pi Rockchip 推論
last_verified: 1.5.0
meta:
  - label: フラグ
    value: 'export(format="rknn", name="rk3588")'
    mono: true
  - label: 出力
    value: 1つの.rknnファイル、.rknn.metadata.jsonサイドカー、verify=Trueの場合は.rknn.parity.jsonレポート
  - label: 追加パッケージ
    value: PyPIにはありません。rknn-toolkit2は自分でインストールするベンダーSDKです。
  - label: 再読み込み
    value: LibreYOLO経由では読み込めません。アーティファクトはボード上でRockchipのランタイムを使って実行します。
  - label: 形状
    value: 固定正方形、バッチ1、opset 19。3つすべてが強制されます。
  - label: 精度
    value: ベンダーの浮動小数点ビルド。half=Trueとint8=Trueは拒否されます。
  - label: 対象範囲
    value: 'RK3588上の4つの検出バリアント: YOLO9-t、YOLO9-E2E-t、PicoDet-s、YOLO-NAS-s'
verification: >-
  devブランチのlibreyolo/export/rknn.py、libreyolo/export/exporter.py、libreyolo/export/support.py、docs/rknn.mdを参照しました。測定された一致度の数値は、docs/rknn.mdにある2026-08-04付けの検証記録に基づきます。
snippets:
  install:
    - label: LibreYOLO側
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: 自分でインストールするベンダーSDK
      language: bash
      code: |
        # rknn-toolkit2は別ライセンスのRockchip SDK LibreYOLOには
        # 同梱もインストールもされない x86_64 Linux専用でWindowsでは
        # WSL2またはLinuxコンテナを使用する
        #
        # Toolkit2 2.3.2はsetuptools<81が必要でONNX 1.19以降では失敗する
        # コンパイラーが引き続きインポートするonnx.mappingが削除されたため
        pip install "setuptools==80.9.0" "onnx==1.18.0"

        # 次にRockchip独自のwheelリポジトリから一致するrknn-toolkit2の
        # wheelをインストールしインポートできることを確認する
        python -c "import rknn.api; print('rknn-toolkit2 ready')"
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        # weights/LibreYOLO9t.rknnとweights/LibreYOLO9t.rknn.metadata.jsonを書き込む

        path = model.export(format="rknn", name="rk3588", imgsz=640,
        verify=True)

        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format rknn --name rk3588 \
          --imgsz 640 --verify
    - label: 引数
      language: python
      code: |
        model.export(
            format="rknn",
            name="rk3588",     # 対象プラットフォーム target=とtarget_platform=も使用可能
            imgsz=640,         # バリアントに記録されたキャンバスと一致させる必要がある
            batch=1,           # 他の値ではNotImplementedErrorを送出
            dynamic=False,     # TrueではValueErrorを送出
            opset=19,          # 他の値ではNotImplementedErrorを送出
            verify=False,      # TrueではPCシミュレーターを実行し一致度で判定
        )
  parity:
    - label: 既存ONNXアーティファクトとのボード不要の一致確認
      language: python
      code: |
        import numpy as np
        from libreyolo.export import verify_rknn_simulator_parity

        input_tensor = np.random.default_rng(0).standard_normal(
            (1, 3, 640, 640), dtype=np.float32
        )
        metrics = verify_rknn_simulator_parity(
            "weights/LibreYOLO9t.onnx",
            input_tensor,
            target_platform="rk3588",
            rtol=1e-3,
            atol=1e-4,
            raise_on_failure=False,
        )
        print(metrics)
  support:
    - label: コンパイル前にファミリーとタスクの組み合わせを確認
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: c659713cc3c8cc9e
---

## インストール

コンパイルにはRockchipのRKNN Toolkit2が必要です。これはRockchip独自のライセンスで配布されるベンダーSDKであり、LibreYOLOの依存関係ではありません。`libreyolo[rknn]`という追加パッケージはなく、この形式に必要なものを1行ですべてインストールする方法もありません。

<code-tabs name="install" />

コンパイルや数値の一致確認にボードは必要ありません。レイテンシ、消費電力、温度の測定にはRK3588ボードが必要ですが、これらの測定結果はまだ記録されていません。

## エクスポート

<code-tabs name="export" />

コンパイルを開始する前に、リクエストはモデルバリアントの厳密なリストと照合され、キャンバスも検証されます。バリアントに記録されたものと異なる`imgsz`を渡すと、未検証のものを暗黙にコンパイルせず例外を送出します。LibreYOLOはopset 19のONNX中間ファイルを書き込み、それをコンパイルし、任意でシミュレーションした後、中間ファイルを削除します。

RKNN形式には移植可能なメタデータフィールドがないため、メタデータは`<model>.rknn.metadata.json`という名前のサイドカーに保存されます。

`verify=True`は、アーティファクトをコンパイルしたセッション内でToolkit2のPCシミュレーターを実行します。同じ入力に対する全出力をONNX Runtimeと比較し、出力ごとの誤差指標を`<model>.rknn.parity.json`へ書き込みます。要素ごとの値がすでに近い出力を除き、コサイン類似度0.9999以上、正規化RMSE 0.02以下を合格条件として適用します。ベンダーの浮動小数点ビルドは内部テンソルを半精度に下げるため、デコード後のボックスが安定していても厳密な`allclose`にはなりません。失敗した実行は`<model>.rknn.failed.parity.json`を書き込み、候補を破棄し、そのパスにある以前の成功済みエクスポートは変更しません。

すでに持っているONNXアーティファクトを再エクスポートせず比較するには、次のようにします。

<code-tabs name="parity" />

Toolkit2のシミュレーターは、`load_onnx`と`build`で生成したメモリ内グラフを実行します。ボードなしでは対象固有の`.rknn`ファイルを再読み込みできません。このため、`verify=True`ではコンパイル、エクスポート、シミュレーションを1つのセッションで実行します。

## アーティファクトを実行

`libreyolo/backends`にはRKNNの項目がないため、`LibreYOLO()`は`.rknn`ファイルを読み込みません。コンパイル済みアーティファクトをボードへデプロイし、Rockchip独自のランタイムで実行します。その環境では前処理、デコード、NMS、座標の再スケーリングをアプリケーション側で実装する必要があります。

`<model>.rknn.metadata.json`にはクラス名、入力サイズ、タスク、対象プラットフォームが含まれます。これらはアプリケーションがLibreYOLOの後処理を再現するために必要な情報です。コンパイル済みモデルと一緒に配布してください。

ボードを必要としないホスト側の確認には、同じ固定形状のONNXアーティファクトを保持し、上記のようにシミュレーターで比較します。

## 制約

コンパイルできる組み合わせは4つで、ファミリーではなくモデルバリアント単位です。

| バリアント | タスク | キャンバス | 対象 |
|---|---|---:|---|
| YOLO9-t | detect | 640 | RK3588 |
| YOLO9-E2E-t | detect | 640 | RK3588 |
| PicoDet-s | detect | 320 | RK3588 |
| YOLO-NAS-s | detect | 640 | RK3588 |

それ以外はコンパイル前に拒否され、このバージョンのRKNNはシミュレーターで検証済みの検出バリアントだけに限定されることが示されます。他のモデルにもコンパイルのみの結果はありますが、意図的に対応対象として提示していません。同じ測定では、RF-DETRにデコーダーの`GridSample`ノードが2つ未変換で残り、D-FINE、RT-DETR、RT-DETRv2、RT-DETRv4、DEIM、DEIMv2、ECはコンパイルとシミュレーションに成功したものの、デコード後の出力が実用上正しくありませんでした。

バッチ1、静的形状、opset 19です。RKNNはLibreYOLOの`half`契約を公開していないため、`half=True`は拒否されます。代表的なキャリブレーションとタスク精度の結果が得られるまで、`int8=True`も拒否されます。

他のRockchip対象は拒否されます。検証済みのプラットフォームは`rk3588`だけです。

ファミリーとタスクの完全なグリッドについては、[エクスポートマトリックス](/docs/reference/export-matrix)を参照してください。1つの組み合わせを確認するには、次のようにします。

<code-tabs name="support" />
