---
title: libreyolo label
seo_title: libreyolo label コマンドリファレンス
description: >-
  ローカルのバウンディングボックスアノテーションツールを起動します：デフォルト値付きの引数、AIアシストの切り替え、そしてネットワークインターフェースにバインドすると何が公開されるか。
lead: >-
  バウンディングボックスの描画と編集を行うローカルのWebツールを起動します。LibreYOLOネイティブのラベルファイルを書き出すため、ここでアノテーションしたデータセットは変換の手順なしでそのまま学習できます。
keywords:
  - libreyolo label cli
  - アノテーションツール バウンディングボックス
  - yolo アノテーションツール
  - 自動ラベリング cli
  - libreyolo label 共有
last_verified: 1.5.0
meta:
  - label: コマンド
    value: libreyolo label
    mono: true
  - label: 出力
    value: stdoutに出力されるサーバーURL。ラベルは画像と同じ場所にlabels/*.txtとして書き出されます
snippets:
  examples:
    - label: 基本
      language: bash
      code: |
        # プロジェクトホームを開いてブラウザーでデータセットを選択または作成
        libreyolo label
    - label: 手動のみ、ポート固定
      language: bash
      code: |
        libreyolo label no_assist=true port=9200 no_browser=true
    - label: チームメンバーが参加できるようにする
      language: bash
      code: |
        libreyolo label share=true
source_hash: bddad245877793b1
---

## 書式

```bash
libreyolo label [data=<dataset.yaml|folder>] [key=value ...]
```

引数は`key=value`のペアで、POSIX形式も使えるので、`port=9200`と`--port 9200`は同じ引数です。

## 引数

| 引数 | デフォルト | 意味 |
|---|---|---|
| `data` | | 直接開くデータセットYAMLまたはフォルダー。未指定の場合はプロジェクトホームから開始します |
| `host` | `127.0.0.1` | バインドするホストまたはインターフェース |
| `port` | `8000` | バインドするポート。使用中の場合は次の空きポートに繰り上がります |
| `device` | `auto` | AI自動ラベル付けに使うデバイス：`0`、`cpu`、`mps`、`auto` |
| `no_assist` | `false` | AI自動ラベル付けを無効にし、手動のラベリングツールとして使います |
| `no_browser` | `false` | ブラウザーを自動で開きません |
| `share` | `false` | `0.0.0.0`にバインドし、同じネットワーク上のチームメンバーが参加できるようにします |
| `json` | `false` | stdoutへのJSON出力 |
| `quiet` | `false` | stderrを抑制します |
| `verbose` | `false` | stderrへの詳細出力 |

## 例

<code-tabs name="examples" />

## 補足

### 出力されるもの

ボックスはLibreYOLOネイティブの`labels/*.txt`ファイルとして保存され、これは`libreyolo train`が読み込む形式なので、あとから変換する必要はありません。このバージョンが扱うのはバウンディングボックスだけです。編集内容は画像を切り替えるたびに保存されます。

### データセットを開く

`data`を指定しない場合、ツールはプロジェクトホームから起動し、データセットはブラウザーから選択または作成します。`data=path/to/data.yaml`を渡すとそのデータセットが直接開き、起動時の行に画像数、クラス数、そしてデータセットが書き込み可能かどうかが表示されます。読み取り専用のデータセットも開くことはでき、書き込めない理由が表示されます。

### 共有と`host`の動作

`share=true`はワイルドカードアドレスにバインドするため、同じネットワーク上の他のマシンからツールにアクセスできますが、プロジェクトの切り替えや削除、計算の開始といった管理操作はこのマシンに限定されます。

`host`に特定のインターフェースを指定すると、動作が変わり、安全性も下がります：ホストがネットワーククライアントと区別できなくなるため、すべてのクライアントが管理権限を持ちます。この指定をすると、コマンドはstderrに警告を出力します。`share=true`を使ってください。

### ポートと終了

使用中のポートは次のポートへずれ、要求したポートから最大20個先まで試します。20個すべてが失敗した場合は`io_error`で終了します。stdoutに表示されるURLは、実際にバインドされたポートです。`share=true`を指定した場合、結果には`lan_url`も含まれ、これがチームメンバーが開くべきアドレスです。

コマンドはCtrl+Cを押すまでフォアグラウンドで動作し続けます。

関連：学習前にラベル付けしたデータセットを確認するには[`libreyolo doctor`](/docs/cli/doctor)、そのデータセットで学習するには[`libreyolo train`](/docs/cli/train)。
