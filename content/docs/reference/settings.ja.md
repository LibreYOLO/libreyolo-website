---
title: 設定
seo_title: LibreYOLOの環境変数とディレクトリ
description: LibreYOLOが読み取るすべての環境変数、書き込み先ディレクトリ、必要なトークン、実行するコード経路を変更する切り替えについて説明します。
lead: LibreYOLOに設定ファイルはありません。関数の引数ではない動作は、環境変数と少数の慣例的なディレクトリで制御されます。すべてをここに列挙します。
keywords:
  - LIBREYOLO_DATASETS_DIR
  - LIBREYOLO_KERNELS
  - LIBREYOLO_FASTER_COCO_EVAL
  - HF_TOKEN
  - libreyolo 重み 保存先
  - libreyolo キャッシュ
last_verified: 1.5.0
verification: >-
  v1.5.0のlibreyolo/**/*.pyでos.environとos.getenvを検索して変数を特定し、各使用箇所で意味を確認しました。ディレクトリ規約はlibreyolo/data/utils.py、libreyolo/utils/download.py、libreyolo/export/exporter.py、libreyolo/models/base/model.py、libreyolo/models/sam3dbody/mhr_body.pyから参照しました。
snippets:
  usage:
    - label: データセットルートを別の場所に指定
      language: bash
      code: |
        export LIBREYOLO_DATASETS_DIR=/data/datasets
        python -c "from libreyolo.data import DATASETS_DIR; print(DATASETS_DIR)"
    - label: 解決済みの値をPythonから読み取る
      language: python
      code: |
        from libreyolo.data import DATASETS_DIR

        # デフォルトは~/datasetsでインポート時にLIBREYOLO_DATASETS_DIRが上書き
        print(DATASETS_DIR)
source_hash: 462f1288582225ce
---

## 環境変数

| 変数 | デフォルト | 効果 |
|---|---|---|
| `LIBREYOLO_DATASETS_DIR` | `~/datasets` | データセットルート。インポート時に1回だけ読み取り、`libreyolo.data.DATASETS_DIR` に設定 |
| `LIBREYOLO_FASTER_COCO_EVAL` | 未設定 | 検証の `faster_coco_eval` フラグを上書き。`1`、`true`、`yes`、`on` では高速なバックエンドを強制的に有効化し、その他の値では強制的に無効化。未設定では設定フラグに従う |
| `LIBREYOLO_KERNELS` | 未設定 | カーネルの選択。`off` または `reference` は参照実装を強制し、その他の値はその名前で登録された実装だけを選択 |
| `LIBREYOLO_QUANT_KERNELS` | 未設定 | `LIBREYOLO_KERNELS` の旧別名。そちらが未設定の場合だけ読み取り |
| `LIBREYOLO_HUB_KERNELS` | 未設定 | `0`、`false`、`off`、`no` ではHugging Face Hubからのカーネル読み込みを無効化。未設定を含むその他の値では有効 |
| `LIBREYOLO_MHR_PATH` | `~/.cache/libreyolo/mhr/mhr_model.pt` | `mesh` タスクが使用するMHR身体モデルの場所 |
| `LIBRELABEL_ENABLE_LOCATE` | 未設定 | ラベル付けツールでLocateAnythingアシスタントを公開するには、正確に `1`、`true`、`yes`、`on` のいずれかが必要。その他の値では無効 |
| `SAM_3D_BODY_PATH` | 未設定 | コンストラクターへ渡さない場合の、メッシュファミリー用SAM 3D Bodyパッケージへのパス |
| `HF_TOKEN` | 未設定 | アクセス制限されたリポジトリに使用するHugging Faceアクセストークン |

<code-tabs name="usage" />

`LIBREYOLO_DATASETS_DIR` はインポート時に読み取られるため、`libreyolo.data` のインポート後に設定しても `DATASETS_DIR` には影響しません。

Hubカーネルは2段階のオプトインです。実行時の取得はオプションの `kernels` パッケージがインストールされている場合だけ行われます。そのため、`libreyolo[hub-kernels]` のインストールがオプトインで、`LIBREYOLO_HUB_KERNELS=0` がオプトアウトです。追加パッケージのないインストールはどちらの設定でも影響を受けません。

カーネル選択はインポートも短絡します。`LIBREYOLO_KERNELS` で `off` または `reference` を強制すると、ツリー内の高速化プロバイダーは一切インポートされません。これら3つの変数が制御するレジストリについては、[カーネル](/docs/reference/kernels)を参照してください。

## ライブラリが設定する変数

これらは読み取られるのではなく書き込まれるため、手動設定は対応する経路ではありません。

| 変数 | 設定元 |
|---|---|
| `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` | DDP起動ヘルパー。ワーカープロセスごとに1つの値 |
| `CUDA_VISIBLE_DEVICES` | 分散設定中に一時的に範囲を限定し、その後復元 |
| `PYTORCH_ENABLE_MPS_FALLBACK` | EC学習器が `setdefault` で `1` に設定。既存の値を優先 |
| `MOMENTUM_ENABLED` | メッシュファミリーのローダーが `setdefault` で設定 |

`LOCAL_RANK` は分散モードの信号としても機能します。環境内にこの変数が存在することで、学習コードはDDP下で実行中だと判断します。

## ロガー変数

オプションの学習ロガーは、プロジェクト名に環境変数のデフォルト値を使用します。

| 変数 | デフォルト | 使用元 |
|---|---|---|
| `WANDB_PROJECT` | `libreyolo` | プロジェクトが渡されない場合のWeights and Biasesロガー |
| `COMET_PROJECT_NAME` | `libreyolo` | プロジェクトが渡されない場合のCometロガー |

これらのサービスの認証には、LibreYOLOではなく各サービス独自のツールを使用します。

## トークン

`HF_TOKEN` はHugging Faceのアクセストークンです。未設定の場合、Hugging Face CLIへのログインが書き込む `~/.cache/huggingface/token` からトークンを読み取ります。どちらの経路も使用できます。

トークンが必要なのはアクセス制限されたリポジトリだけです。提供されている例はSAM 3です。重みはカスタムライセンス下のアクセス制限されたリポジトリからダウンロードされるため、リポジトリページで条件に同意し、セッションを認証する必要があります。

## ディレクトリ

| パス | 内容 |
|---|---|
| `weights/` | ダウンロード済みチェックポイント、Hugging Faceスナップショット、エクスポート済み成果物 |
| `~/datasets` | `LIBREYOLO_DATASETS_DIR` で変更しない場合のデータセットルート |
| `~/.cache/huggingface/token` | `HF_TOKEN` を使用しない場合のHugging Faceトークン |
| `~/.cache/libreyolo/mhr/mhr_model.pt` | `LIBREYOLO_MHR_PATH` で変更しない場合のMHR身体モデル |
| `runs/track/` | `model.track(save=True)` のデフォルト出力 |

`weights/` は作業ディレクトリからの相対パスです。単独のファイル名はこのディレクトリを通じて解決されます。そのため、`LibreYOLO("LibreYOLO9t.pt")` は `weights/LibreYOLO9t.pt` を探し、存在しない場合はそこへダウンロードします。`output_path` を指定しない場合、`model.export()` も同じディレクトリへ書き出します。関連する階層は、複数ファイルのスナップショットを `weights/<Prefix><size>/` へダウンロードします。

## ダウンロードの動作

重みのダウンロードはバックオフを伴って3回再試行され、部分的なファイルから再開します。また、ロックファイルで保護されるため、2つのプロセスが同じチェックポイントを同時に取得することはありません。サードパーティーのホストから取得するファミリーはチェックサムを固定でき、不一致では安全側に倒れて失敗します。

一部のダウンロードは開始前にライセンス通知を表示します。これらの通知はダウンロード経路の一部であり、設定から抑制できません。

## 検証バックエンド

`model.val()` はデフォルトで `faster_coco_eval=True` を受け付け、パッケージがインストールされていない場合は1回だけ警告してpycocotoolsへフォールバックします。`LIBREYOLO_FASTER_COCO_EVAL` を設定すると呼び出しごとのフラグを上書きします。実行ごとの設定を変更できないベンチマークハーネスでは、この方法を使用します。実際に実行されたバックエンドは `model.last_eval_backend` で報告されます。

## データセットダウンロードスクリプト

データセットYAMLにはPythonを含む `download` フィールドを記載できます。読み取る呼び出しへ `allow_download_scripts=True` を渡さない限り、実行されません。これは環境変数ではなく、`val()` と `export()` の関数引数です。
