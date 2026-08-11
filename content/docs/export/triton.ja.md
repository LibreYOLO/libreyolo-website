---
title: Triton Inference Server
seo_title: NVIDIA TritonでLibreYOLOモデルを配信
description: >-
  LibreYOLOのONNXエクスポートをNVIDIA
  Triton経由で配信します。モデルリポジトリの配置、生成されるconfig.pbtxt、HTTPモデルURLに対する推論を解説します。
lead: >-
  Triton Inference
  Serverはモデルリポジトリをホストし、HTTP経由の推論リクエストに応答します。LibreYOLOはONNXグラフをエクスポートし、エクスポートのメタデータを1つのTritonパラメーターとして保持するconfig.pbtxtを生成して、モデルURLを読み込み可能なモデルパスとして扱います。
keywords:
  - LibreYOLO Triton 使い方
  - Triton Inference Server
  - config.pbtxt
  - tritonclient HTTP
  - モデルリポジトリ
  - YOLO リモート推論
last_verified: 1.5.0
meta:
  - label: 呼び出し
    value: 'LibreYOLO("http://127.0.0.1:8000/yolo9")'
    mono: true
  - label: ヘルパー
    value: >-
      create_triton_config(onnx_path, config_path, model_name=...,
      max_batch_size=8)
    mono: true
  - label: 追加パッケージ
    value: 'pip install "libreyolo[onnx,triton]"'
    mono: true
  - label: プロトコル
    value: HTTPおよびHTTPS V2推論のみ。gRPC、認証、共有メモリ、モデルの読み込みとアンロードには対応しません。
  - label: タイムアウト
    value: 接続とネットワークのタイムアウトはデフォルトで30秒です
verification: >-
  devブランチのlibreyolo/backends/triton.py、libreyolo/models/__init__.py、docs/triton.md、pyproject.tomlを参照しました。コンテナコマンドはdocs/triton.mdに固定されているものです。
snippets:
  install:
    - label: インストール
      language: bash
      code: |
        pip install "libreyolo[onnx,triton]"
  repo:
    - label: リポジトリ構造へエクスポート
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
    - label: 生成される配置
      language: text
      code: |
        triton_repo/
          yolo9/
            config.pbtxt
            1/
              model.onnx
  serve:
    - label: サーバーを起動
      language: bash
      code: |
        docker run --rm --name libreyolo-triton \
          -p 8000:8000 -p 8002:8002 \
          -v "$(pwd)/triton_repo:/models:ro" \
          nvcr.io/nvidia/tritonserver:26.04-py3 \
          tritonserver --model-repository=/models --exit-on-error=true
    - label: 準備完了まで待機
      language: bash
      code: >
        until curl --fail --silent http://127.0.0.1:8000/v2/health/ready; do
        sleep 1; done
    - label: サーバーを停止
      language: bash
      code: |
        docker stop libreyolo-triton
  run:
    - label: 配信中のモデルで推論
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
    - label: バージョン固定またはタイムアウト変更
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.backends.triton import TritonBackend

        # 2番目のパスセグメントでモデルバージョンを選択する 省略時は
        # Tritonに設定されたバージョンポリシーが選択する
        pinned = LibreYOLO("http://127.0.0.1:8000/yolo9/1")

        # 接続とネットワークのタイムアウトはデフォルトで30秒
        patient = TritonBackend("http://127.0.0.1:8000/yolo9", timeout=120)
source_hash: 0652e4faf0224df3
---

## インストール

<code-tabs name="install" />

`triton`追加パッケージは`tritonclient[http]`をインストールします。この統合はHTTPおよびHTTPS V2推論専用なので、gRPCと共有メモリの追加パッケージは意図的に除外しています。配信するアーティファクトと設定ジェネレーターはどちらもONNXグラフを使用するため、`onnx`も必要です。

## モデルリポジトリを構築

動的バッチ軸を指定し、Tritonが要求するディレクトリ構造へエクスポートします。

<code-tabs name="repo" />

Tritonのモデル設定レスポンスにはONNXのカスタムメタデータが保持されないため、エクスポート済みの完全なメタデータを別の方法で渡す必要があります。`create_triton_config`は、それを`config.pbtxt`内の`libreyolo_metadata`という名前の1つのJSON文字列パラメーターとしてエンコードします。さらに、グラフの順序で入出力宣言を生成し、JSONのエスケープを処理して、モデルを`KIND_CPU`に固定します。

ヘルパーは書き込み前に検証します。ONNXグラフの入力がちょうど1つであること、出力が1つ以上あること、テンソル形状を解決できること、メタデータの`names`マップに0から`nc - 1`までの各クラスインデックスが定義されていることを要求します。いずれかの確認に失敗したモデルは、最初のリクエスト時ではなく設定時に拒否されます。

`max_batch_size: 8`は動的エクスポートに対応し、サーバーがリクエストごとに最大8枚の画像をバッチ処理できるようにします。固定バッチ1のONNXグラフでは`max_batch_size=0`を使用してください。その場合、LibreYOLOは画像を順次送信します。

## サーバーを起動

<code-tabs name="serve" />

コマンドはTriton Server 26.04に固定され、DockerのGPUフラグを意図的に省略しています。生成された設定の`KIND_CPU`によって、いずれにせよGPUへの配置が防止されるためです。

## アーティファクトを実行

TritonモデルのURLはモデルパスとして扱われます。`LibreYOLO()`はローカルパスを処理する前に`http`または`https`スキームを確認し、サーバーと通信するバックエンドを返します。そのため、呼び出し側はローカルチェックポイントと同じになり、返される`Results`オブジェクトも同じです。

<code-tabs name="run" />

URLの形式は`http(s)://host:port/model`で、任意でバージョンを表すセグメントを追加できます。ポートは明示する必要があります。埋め込み認証情報、クエリ文字列、フラグメントはすべて拒否され、3つ以上のセグメントを持つパスも拒否されます。

配置はサーバー側で決定するため、`device`は受け付けられますがログを1行出して無視されます。

## 制約

次の条件を満たさない場合、バックエンドは機能を縮退させた結果ではなく、直接エラーを返します。モデル設定にLibreYOLOのメタデータがない、モデル入力が複数ある、設定済みの出力とモデルのメタデータが一致しない、未対応の入力データ型である、またはサーバーかモデルの準備が完了していない場合です。

このバージョンでは、gRPC、認証、共有メモリ、API経由でのモデルの読み込みとアンロードは契約対象外です。

Triton自体が対応する形式はすべて配信できますが、ここでのメタデータパラメーターと生成設定はONNX向けに作られています。そのため、LibreYOLOではリポジトリへ[ONNX](/docs/export/onnx)を配置します。リクエストとレスポンス形式のサーバーではなく、完全な動画パイプラインについては[DeepStream](/docs/export/deepstream)を参照してください。
