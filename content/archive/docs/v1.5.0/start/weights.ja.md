---
title: チェックポイントと重み
seo_title: LibreYOLOのチェックポイントと重み
description: >-
  LibreYOLOがモデルの重みを検索、ダウンロード、検証する仕組み、ホスト先、オフラインでの実行方法、チェックポイントを安全に読み込むための条件を解説します。
lead: >-
  LibreYOLOのチェックポイントは、state
  dictとモデルを識別するためのメタデータを保持するtorch.save辞書です。このページでは、ファイルの取得元、保存先、読み込み方法を解説します。
keywords:
  - LibreYOLO 重み
  - LibreYOLO チェックポイント
  - LibreYOLO 重み ダウンロード
  - LibreYOLO オフライン
  - LibreYOLO Hugging Face
  - チェックポイント メタデータ
last_verified: 1.5.0
meta:
  - label: ホスト先
    value: 'チェックポイントごとに1つのHugging Faceリポジトリ:'
    links:
      - label: huggingface.co/LibreYOLO
        href: 'https://huggingface.co/LibreYOLO'
  - label: ローカルキャッシュ
    value: 作業ディレクトリ内のweights/
    mono: true
  - label: メタデータスキーマ
    value: v1.0
snippets:
  load:
    - label: 自動ダウンロード
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファイル名だけの場合はweights/LibreYOLO9t.ptへ解決され
        # まだ存在しなければそこへダウンロードされる
        model = LibreYOLO("LibreYOLO9t.pt")
        print(model(SAMPLE_IMAGE).boxes)
    - label: 明示的なパス
      language: python
      code: |
        from libreyolo import LibreYOLO

        # ディレクトリを含むパスは記述どおりに使用され
        # ネットワークから取得されることはない
        model = LibreYOLO("/opt/models/LibreYOLO9t.pt")
        print(model.family, model.size, model.task)
  inspect:
    - label: CLI
      language: bash
      code: |
        # モデルを構築せずにメタデータを読み取り
        # スキーマを満たすかどうか報告する
        libreyolo metadata path=weights/LibreYOLO9t.pt
    - label: JSON
      language: bash
      code: |
        libreyolo metadata path=weights/LibreYOLO9t.pt --json
    - label: Python
      language: python
      code: >
        from libreyolo.utils.serialization import (
            load_untrusted_torch_file,
            validate_checkpoint_metadata,
        )


        loaded = load_untrusted_torch_file("weights/LibreYOLO9t.pt")


        # 問題のリストを返す 空ならファイルはv1.0を満たす

        print(validate_checkpoint_metadata(loaded))

        print(loaded["model_family"], loaded["size"], loaded["task"],
        loaded["nc"])
source_hash: 210a12baa1417cfb
---

## チェックポイントの検索場所

`LibreYOLO9t.pt`のようにディレクトリ部分を含まないモデル参照は、現在の作業ディレクトリを基準とする`weights/`内で解決されます。`weights/LibreYOLO9t.pt`が存在すればそれを使用します。作業ディレクトリ自体に同名のファイルがあれば、代わりにそれを使用します。どちらも存在しない場合は`weights/LibreYOLO9t.pt`がダウンロード先になります。

絶対パスか相対パスかにかかわらず、ディレクトリを含む参照は記述どおりに扱われます。重みを中央の保存場所に置き、何も取得したくない場合はこの形式を使用してください。

<code-tabs name="load" />

## 自動ダウンロード

解決したパスが存在しない場合、LibreYOLOはファイル名を解析してファミリー、サイズ、タスクを取得し、対応するファミリーへダウンロードURLを問い合わせます。ほとんどのファミリーでは、LibreYOLOのHugging Face組織にあるURLを構築します。そこでは各チェックポイントにファイル名と同名の専用リポジトリがあります。

```text
https://huggingface.co/LibreYOLO/<name>/resolve/main/<name>.pt
```

データセットバリアントの接尾辞はリポジトリ名の一部として保持されます。そのため、ファミリーのデフォルトとは異なるデータで学習したチェックポイントは、デフォルトを上書きせず専用リポジトリへ解決されます。

重みファイルが途中で切れていると、後になって分かりにくいエラーが発生するため、転送処理には防御的な仕組みがあります。ダウンロードは`.part`ファイルへストリーミングされ、完了時だけアトミックに所定の場所へ移動します。そのため、中断した処理が最終パスに書きかけのチェックポイントを残すことはありません。中断した転送はHTTPバリデーターを使用してバイトオフセットから再開し、サーバーがオブジェクトの変更を示した場合は最初からやり直します。失敗時は指数バックオフで3回再試行します。同じパスを対象とする並行プロセスはロックファイルを取得するため、2つの学習処理が同時に開始してもダウンロードは1回です。ファミリーがLibreYOLO組織以外のホストから取得する場合は、チェックサムを固定し、不一致のファイルを拒否できます。

`HF_TOKEN`が設定されているか、`~/.cache/huggingface/token`にトークンがキャッシュされている場合はBearerトークンとして付加されます。付加するのは`huggingface.co`のURLだけなので、別のホストからダウンロードするファミリーへトークンが渡ることはありません。

すべてのファミリーが自動ダウンロードに対応するわけではありません。公開済みの重みを再配布できないため意図的にURLを返さず、代わりに何を指定すべきかエラーで説明するものもあります。転送開始前にライセンスの告知を表示するものもあります。この告知はチェックポイントの条件がコードより制限されていることを示す実行時の合図なので、読み飛ばさず確認してください。

## Hugging Face組織

公開済みの重みは[huggingface.co/LibreYOLO](https://huggingface.co/LibreYOLO)に置かれ、チェックポイントごとに1つのリポジトリがあります。各リポジトリにはライセンスがあり、同じファミリー内でも統一されているとは限りません。コードがMITでも、一部の重みがそうではないファミリーがあります。リポジトリが正式な情報源です。各モデルページの「チェックポイント」と「ライセンス」セクションには、そのファミリーで公開されているチェックポイントとライセンスが記載されています。

## オフラインでの使用

ファイルがローカルにあれば、ライブラリの動作にネットワークアクセスは必要ありません。次の2つの方法があります。

ジョブを実行する場所の隣に`weights/`ディレクトリをあらかじめ用意します。ネットワーク接続のあるマシンでチェックポイントを一度取得してからディレクトリをコピーすれば十分です。前述の解決処理がそれらを検出し、ネットワークには接続しません。

または、共有場所への絶対パスを渡します。ディレクトリ部分を含む参照は指定どおりに使用されるため、選定済みの重みを格納した読み取り専用マウントも有効です。変換が必要なチェックポイントの隣にプロセスが書き込めない場合は、失敗する代わりにプライベートな一時ディレクトリで変換します。

データセットには別の規則が適用されます。`LIBREYOLO_DATASETS_DIR`を設定した場合はそのディレクトリ、設定しない場合は`~/datasets`の下で解決されます。

## 読み込みの安全性

チェックポイントはpickleであり、pickleは開いたときに任意のコードを実行できます。LibreYOLOはすべての重みファイルを信頼できないものとして扱い、PyTorchの`weights_only=True`経路で読み込みます。これによりアンピクラーはテンソルと少数の安全な型だけに制限されます。この処理はLibreYOLOがダウンロードしたファイルだけでなく、ユーザーが渡したファイルにも適用されます。この引数に対応しないほど古いPyTorchビルドでは、安全でない読み込みを行う代わりに拒否します。

一部のアップストリームの学習用チェックポイントには、学習に使用したフレームワークの設定オブジェクトなど、制限付きアンピクラーが拒否するオブジェクトが埋め込まれています。LibreYOLOには不要なメタデータなので、変換時にはブロックされた各クラスを、コードを実行せずアンピクラーの条件を満たす不活性な代替物へ置き換えます。変換済みファイルにはテンソルだけが残ります。機密性の高いモジュール名は代替せず完全に拒否し、再試行回数にも上限を設けるため、ブロック対象クラスを際限なく追加するよう細工されたファイルは安全側で失敗します。その他の処理については[既存の重みをインポート](/docs/migrate)を参照してください。

## チェックポイントのメタデータ

LibreYOLOのチェックポイントは、`model`キーにPyTorchのstate dictを保持する辞書です。スキーマv1.0では9つのキーが必須で、これらを組み合わせることにより、ファクトリはファイル名の解析やテンソル形状からの推測を行わずファイルを識別できます。

| キー | 意味 |
|---|---|
| `model` | PyTorchのstate dict |
| `schema_version` | メタデータ契約のバージョン。v1.0では文字列`1.0`を使用 |
| `libreyolo_version` | ファイルを生成したLibreYOLOのバージョン |
| `model_family` | `yolo9`など、登録済みのファミリー識別子 |
| `size` | `t`や`r18`など、そのファミリー内のバリアント |
| `task` | 標準化された1つのタスク名 |
| `nc` | 正のクラス数 |
| `names` | `0`から`nc - 1`までを網羅する、クラスインデックスとラベルのマッピング |
| `imgsz` | 正の入力解像度 |

追加構造を持つタスクでは、これらのキーとともにその情報も記録します。姿勢推定のチェックポイントは`num_keypoints`と`keypoint_dim`を追加し、キーポイントごとのOKSシグマを追加することもあります。OCRのチェックポイントには完全なCTC文字セットが埋め込まれ、ファイルだけで完結します。画像復元のチェックポイントには劣化の種類やアップスケール倍率が記録される場合があります。トレーナーのチェックポイントは`epoch`、オプティマイザーの状態、EMAの重みなど再開用の状態を追加します。公開する推論用の重みにはこれらを含めるべきではありません。

9つすべてのキーを満たすファイルはメタデータ経路で読み込まれます。満たさないファイルは、ファミリーがその構造を認識できれば変換されます。それ以外では、不足項目を示す警告とともに互換性経路で読み込まれます。

## チェックポイントを調べる

<code-tabs name="inspect" />

`libreyolo metadata`はモデルを構築しないため、ファミリーがインストールされていないファイルや、安全性をまだ確認していないファイルにも使用できます。
