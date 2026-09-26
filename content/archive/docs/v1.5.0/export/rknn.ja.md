---
title: RKNN
seo_title: Rockchip NPU向けRKNNへエクスポート
description: >-
  LibreYOLO検出器をRockchipの.rknn成果物へcompileします。自身でインストールするvendor
  SDK、検証済みの4つのRK3588バリアント、simulatorでの一致を説明します。
lead: >-
  RKNNはRockchipのcompile済みNPU形式です。LibreYOLOはopset-19のONNX中間ファイルをエクスポートし、RKNN
  Toolkit2 SDKでcompileします。boardなしでも、Toolkit2のhost simulatorを使い、compile済みグラフをONNX
  Runtimeと比較できます。
keywords:
  - yolo rknn エクスポート
  - rockchip npu
  - rk3588
  - rknn-toolkit2
  - rknn simulator parity
  - orange pi rockchip 推論
last_verified: 1.5.0
meta:
  - label: フラグ
    value: 'export(format="rknn", name="rk3588")'
    mono: true
  - label: 出力
    value: >-
      1つの.rknnファイル、.rknn.metadata.json sidecar、verify=Trueの場合は.rknn.parity.json
      report
  - label: 追加パッケージ
    value: PyPIにはありません。rknn-toolkit2は自身でインストールするvendor SDKです。
  - label: 再読み込み
    value: LibreYOLOからは読み込めません。成果物はRockchipのruntimeを使ってboard上で実行します。
  - label: shape
    value: 固定正方形、batch 1、opset 19。3つすべてが強制されます。
  - label: 精度
    value: vendorのfloating build。half=Trueとint8=Trueは拒否されます。
  - label: 対象範囲
    value: RK3588上の4つの検出バリアント：YOLO9-t、YOLO9-E2E-t、PicoDet-s、YOLO-NAS-s
verification: >-
  dev
  branchのlibreyolo/export/rknn.py、libreyolo/export/exporter.py、libreyolo/export/support.py、docs/rknn.mdから確認しました。測定済みの一致値は、docs/rknn.mdにある2026-08-04付けの検証記録から取得しています。
snippets:
  install:
    - label: LibreYOLO側
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: 自身でインストールするvendor SDK
      language: bash
      code: |
        # rknn-toolkit2は個別ライセンスのRockchip SDKで、LibreYOLOは
        # 同梱もインストールもしない。x86_64 Linux専用で、Windowsでは
        # WSL2またはLinux containerを使用
        #
        # Toolkit2 2.3.2にはsetuptools<81が必要で、ONNX 1.19以降では失敗
        # compilerが引き続きimportするonnx.mappingが削除されたため
        pip install "setuptools==80.9.0" "onnx==1.18.0"

        # 次に、Rockchip独自のwheel repositoryから一致する
        # rknn-toolkit2 wheelをインストールし、importを確認
        python -c "import rknn.api; print('rknn-toolkit2 ready')"
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        # weights/LibreYOLO9t.rknnとweights/LibreYOLO9t.rknn.metadata.jsonを出力

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
            name="rk3588",     # target platform。target=とtarget_platform=も利用可能
            imgsz=640,         # variantで記録されたcanvasと一致が必要
            batch=1,           # ほかの値ではNotImplementedErrorを送出
            dynamic=False,     # TrueではValueErrorを送出
            opset=19,          # ほかの値ではNotImplementedErrorを送出
            verify=False,      # TrueはPC simulatorを実行し、一致でgate
        )
  parity:
    - label: 既存ONNX成果物に対するboard不要の一致確認
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
    - label: compile前に1つのファミリーとタスクを確認
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: c659713cc3c8cc9e
---

## インストール

compileにはRockchip独自のライセンスでvendor SDKとして配布されるRockchip RKNN Toolkit2が必要で、LibreYOLOの依存関係ではありません。`libreyolo[rknn]` 追加パッケージはなく、この形式に必要なものを1行でインストールする方法はありません。

<code-tabs name="install" />

compileや数値的一致の確認にboardは必要ありません。レイテンシ、消費電力、温度の測定にはRK3588 boardが必要ですが、いずれも記録されていません。

## エクスポート

<code-tabs name="export" />

compile前に要求が正確なモデルバリアントのリストに照らして検証され、canvasも検証されます。バリアントで記録されたものと異なる `imgsz` を渡すと、未検証のものを通知なしでcompileせずに例外を送出します。LibreYOLOはopset-19のONNX中間ファイルを書き込み、compileし、任意でsimulationを行い、その後中間ファイルを削除します。

RKNN形式にはportable metadata fieldがないため、メタデータは `<model>.rknn.metadata.json` というsidecarです。

`verify=True` は成果物をcompileしたのと同じsession内でToolkit2のPC simulatorを実行し、同じ入力に対するすべての出力をONNX Runtimeと比較して、出力ごとの誤差指標を `<model>.rknn.parity.json` へ書き込みます。すでに要素ごとの誤差が許容範囲内でない出力に対し、cosine類似度0.9999以上、正規化RMSE 0.02以下というgateを適用します。vendorのfloating buildは内部テンソルをhalf precisionへ下げるため、decode済みボックスが安定していても厳密な `allclose` にはなりません。失敗した実行では `<model>.rknn.failed.parity.json` を書き込み、候補を破棄し、そのパスにある以前の成功済みエクスポートには触れません。

すでに持っているONNX成果物を再エクスポートせずに比較するには、次を使います。

<code-tabs name="parity" />

Toolkit2のsimulatorは `load_onnx` と `build` で生成されたメモリ内グラフを実行します。boardなしでは対象固有の `.rknn` ファイルを再読み込みできないため、`verify=True` は1つのsessionでcompile、エクスポート、simulationを行います。

## 成果物を実行

`libreyolo/backends` にRKNNの項目はないため、`LibreYOLO()` は `.rknn` ファイルを読み込みません。compile済み成果物はboardへデプロイし、Rockchip独自のruntimeで実行します。そこでの前処理、decode、NMS、座標の再スケーリングはアプリケーション側の責任です。

`<model>.rknn.metadata.json` はクラス名、入力サイズ、タスク、対象platformを保持し、アプリケーションがLibreYOLOの後処理を再現するために必要な情報を提供します。compile済みモデルと一緒に配置してください。

boardを必要としないhost側の確認には、同じ固定shapeのONNX成果物を保持し、上記のようにsimulatorで比較します。

## 制約

4つの組み合わせがcompileでき、ファミリーではなくモデルバリアント単位です。

| バリアント | タスク | canvas | 対象 |
|---|---|---:|---|
| YOLO9-t | detect | 640 | RK3588 |
| YOLO9-E2E-t | detect | 640 | RK3588 |
| PicoDet-s | detect | 320 | RK3588 |
| YOLO-NAS-s | detect | 640 | RK3588 |

それ以外はcompile前に拒否され、このバージョンのRKNNがsimulatorで正確にテストされた検出バリアントに限定されると表示されます。ほかのモデルについてcompileのみの結果はありますが、意図的にサポートとして示していません。同じ測定実行で、RF-DETRはloweringされないdecoderの `GridSample` nodeを2つ残しました。D-FINE、RT-DETR、RT-DETRv2、RT-DETRv4、DEIM、DEIMv2、ECはcompileとsimulationに成功しましたが、decode済み出力が大幅に誤っていました。

batch 1、static shape、opset 19です。RKNNはLibreYOLOの `half` 仕様を公開しないため `half=True` は拒否され、代表的なキャリブレーションとタスク精度の結果が得られるまで `int8=True` も拒否されます。

ほかのRockchip対象は拒否されます。検証済みplatformは `rk3588` だけです。

ファミリーとタスクの完全なgridについては[エクスポートマトリックス](/docs/reference/export-matrix)を参照してください。1つの組み合わせを確認するには次を使います。

<code-tabs name="support" />

