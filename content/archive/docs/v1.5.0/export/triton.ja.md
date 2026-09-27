---
title: Triton Inference Server
seo_title: NVIDIA TritonでLibreYOLOモデルを提供
description: >-
  LibreYOLOのONNXエクスポートをNVIDIA
  Tritonから提供します。モデルリポジトリの構成、生成されるconfig.pbtxt、HTTPモデルURLに対する推論を説明します。
lead: >-
  Triton Inference
  Serverはモデルリポジトリをホストし、HTTP経由の推論要求へ応答します。LibreYOLOはONNXグラフをエクスポートし、エクスポートメタデータを1つのTritonパラメータとして保持するconfig.pbtxtを生成し、モデルURLを読み込み可能なモデルパスとして扱います。
keywords:
  - libreyolo triton
  - triton inference server
  - config.pbtxt
  - tritonclient http
  - model repository
  - remote yolo inference
last_verified: 1.5.0
meta:
  - label: 呼び出し
    value: 'LibreYOLO("http://127.0.0.1:8000/yolo9")'
    mono: true
  - label: helper
    value: >-
      create_triton_config(onnx_path, config_path, model_name=...,
      max_batch_size=8)
    mono: true
  - label: 追加パッケージ
    value: 'pip install "libreyolo[onnx,triton]"'
    mono: true
  - label: protocol
    value: HTTPおよびHTTPS V2推論のみ。gRPC、認証、共有メモリ、モデルの読み込み・取り外しは対象外。
  - label: timeout
    value: 接続とネットワークのtimeoutはデフォルトで30秒
verification: >-
  dev
  branchのlibreyolo/backends/triton.py、libreyolo/models/__init__.py、docs/triton.md、pyproject.tomlから確認しました。containerコマンドはdocs/triton.mdに固定されたものです。
snippets:
  install:
    - label: インストール
      language: bash
      code: |
        pip install "libreyolo[onnx,triton]"
  repo:
    - label: リポジトリ構成へエクスポート
      language: python
      code: |
        from pathlib import Path

        from libreyolo import LibreYOLO

        model_dir = Path("triton_repo/yolo9/1")
        model_dir.mkdir(parents=True, exist_ok=True)

        LibreYOLO("LibreYOLO9t.pt").export(
            format="onnx",
            output_path=str(model_dir / "model.onnx"),
            dynamic=True,
            simplify=False,
        )
    - label: config.pbtxtを生成
      language: python
      code: |
        from libreyolo import create_triton_config

        create_triton_config(
            "triton_repo/yolo9/1/model.onnx",
            "triton_repo/yolo9/config.pbtxt",
            model_name="yolo9",
            max_batch_size=8,
        )
    - label: 生成される構成
      language: text
      code: |
        triton_repo/
          yolo9/
            config.pbtxt
            1/
              model.onnx
  serve:
    - label: serverを起動
      language: bash
      code: |
        docker run --rm --name libreyolo-triton \
          -p 8000:8000 -p 8002:8002 \
          -v "$(pwd)/triton_repo:/models:ro" \
          nvcr.io/nvidia/tritonserver:26.04-py3 \
          tritonserver --model-repository=/models --exit-on-error=true
    - label: 準備完了を待つ
      language: bash
      code: >
        until curl --fail --silent http://127.0.0.1:8000/v2/health/ready; do
        sleep 1; done
    - label: 停止
      language: bash
      code: |
        docker stop libreyolo-triton
  run:
    - label: 提供中のモデルに対して推論
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        remote = LibreYOLO("http://127.0.0.1:8000/yolo9")
        result = remote.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: ローカルモデルと比較
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        remote = LibreYOLO("http://127.0.0.1:8000/yolo9").predict(SAMPLE_IMAGE)
        native = LibreYOLO("LibreYOLO9t.pt").predict(SAMPLE_IMAGE)

        print(len(remote.boxes), len(native.boxes))
        print(remote.boxes.xyxy[:3])
        print(native.boxes.xyxy[:3])
    - label: versionを固定またはtimeoutを変更
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.backends.triton import TritonBackend

        # 2つ目のpath segmentでmodel versionを選択。省略した場合は
        # Tritonで設定されたversion policyによって選択
        pinned = LibreYOLO("http://127.0.0.1:8000/yolo9/1")

        # 接続とネットワークのtimeoutはデフォルトで30秒
        patient = TritonBackend("http://127.0.0.1:8000/yolo9", timeout=120)
source_hash: 0652e4faf0224df3
---

## インストール

<code-tabs name="install" />

`triton` 追加パッケージは `tritonclient[http]` をインストールします。gRPCと共有メモリの追加パッケージは意図的に除外されています。この統合はHTTPおよびHTTPS V2推論専用です。提供する成果物とconfig generatorはどちらもONNXグラフから動作するため、`onnx` も必要です。

## モデルリポジトリを構築

Tritonが想定するディレクトリ構成へ、dynamic batch axisを指定してエクスポートします。

<code-tabs name="repo" />

Tritonはmodel configの応答でONNXのcustom metadataを維持しないため、完全なエクスポートメタデータを別の方法で渡す必要があります。`create_triton_config` は、`config.pbtxt` 内で `libreyolo_metadata` という1つのJSON文字列パラメータとしてエンコードし、グラフ順で入力・出力宣言を生成し、JSON escapeを処理し、モデルを `KIND_CPU` に固定します。

helperは書き込み前に検証します。ONNXグラフの入力が正確に1つ、出力が1つ以上、テンソルshapeが解決可能、メタデータの `names` mapが0から `nc - 1` までのすべてのクラスインデックスを定義している必要があります。いずれかのチェックに失敗するモデルは、最初の要求時ではなくconfig生成時に拒否されます。

`max_batch_size: 8` はdynamic exportに対応し、serverが要求ごとに最大8枚の画像をバッチ処理できるようにします。固定batch 1のONNXグラフでは `max_batch_size=0` を使います。その場合、LibreYOLOは画像を順番に送信します。

## serverを起動

<code-tabs name="serve" />

コマンドはTriton Server 26.04に固定し、DockerのGPUフラグを意図的に省略しています。生成されたconfigの `KIND_CPU` により、GPUへの配置はどちらにしても防止されるためです。

## 成果物を実行

TritonモデルURLはモデルパスです。`LibreYOLO()` はローカルパスを処理する前に `http` または `https` schemeを確認し、serverと通信するbackendを返します。そのため、呼び出し側はローカルチェックポイントと同一であり、返される `Results` オブジェクトも同じです。

<code-tabs name="run" />

URLの形式は任意のversion segmentを持つ `http(s)://host:port/model` です。portは明示する必要があります。埋め込みcredentials、query string、fragment、3つ以上のpath segmentはすべて拒否されます。

`device` は受け付けられますが、配置はserverが決めるため、ログを1行出して無視されます。

## 制約

仕様が満たされない場合、backendは劣化した結果ではなく直接エラーを返します。model configにLibreYOLOメタデータがない、モデル入力が2つ以上ある、設定済み出力とモデルメタデータが一致しない、対応していない入力datatype、serverまたはモデルの準備ができていない場合です。

このバージョンの仕様外となるものは、gRPC、認証、共有メモリ、API経由のモデル読み込み・取り外しです。

Triton自体が対応する任意の形式を提供できますが、ここでのメタデータパラメータと生成configはONNX向けなので、LibreYOLOの経路は[ONNX](/docs/export/onnx)からリポジトリへ配置するものです。request-response serverではなく完全な動画パイプラインについては[DeepStream](/docs/export/deepstream)を参照してください。

