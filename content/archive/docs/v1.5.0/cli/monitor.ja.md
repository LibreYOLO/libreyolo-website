---
title: libreyolo monitor
seo_title: libreyolo monitor コマンドリファレンス
description: 学習ランのライブダッシュボードを配信します：デフォルト値付きの引数、サーバーがディスクから読み取る内容、そして1つのサーバーで複数のランをまとめて扱う方法。
lead: >-
  学習ラン（run）がディスクに書き出した成果物を読み取り、Webダッシュボードとして配信します。学習プロセスにアタッチすることはないため、実行中のラン、完了したラン、クラッシュしたランのいずれも表示できます。
keywords:
  - libreyolo monitor cli
  - 学習 ダッシュボード
  - yolo 学習 進捗 監視
  - libreyolo monitor ポート
  - 学習 メトリクス 可視化
last_verified: 1.5.0
meta:
  - label: コマンド
    value: libreyolo monitor
    mono: true
  - label: 出力
    value: stdoutにサーバーのURLを出力し、その後プロセスはフォアグラウンドにとどまります
snippets:
  examples:
    - label: 基本
      language: bash
      code: |
        # runs/を監視して配下のランをすべて一覧表示
        libreyolo monitor
    - label: 別のrunsルート
      language: bash
      code: |
        libreyolo monitor experiments/
    - label: 1つのラン、固定ポート、ブラウザーを開かない
      language: bash
      code: |
        libreyolo monitor runs/train/exp port=9100 no_browser=true
source_hash: 4aa178141d451728
---

## 書式

```bash
libreyolo monitor [<run-dir|runs-root>] [key=value ...]
```

ディレクトリは位置引数です。それ以外はすべて`key=value`のペアで、POSIX形式も使えるため、`port=9100`と`--port 9100`は同じ引数になります。

## 引数

| 引数 | デフォルト | 意味 |
|---|---|---|
| `run_dir` | `runs` | 位置引数。監視するrunsルート、または直接開く単一のランディレクトリ。どちらの場合もルート配下のすべてのランが一覧表示されます |
| `host` | `127.0.0.1` | バインドするホストまたはインターフェイス |
| `port` | `8420` | バインドするポート。使用中の場合は次の空きポートにずらします |
| `no_browser` | `false` | ブラウザーを自動で開かない |
| `json` | `false` | stdoutにJSONを出力 |
| `quiet` | `false` | stderrを抑制 |
| `verbose` | `false` | stderrへの詳細出力 |

## 使用例

<code-tabs name="examples" />

## 補足

### 1つのサーバーで複数のラン

サーバーは単一のランではなくrunsルートを監視し、各ランをURLで指定するため、1台のマシン上の複数のランが1つのポートを共有します。一覧を見るにはルートのURLを開き、ランごとに1つのタブを開くこともできます。各URLの`?run=`パラメータがどのランかを示します。

単一のランディレクトリを指定した場合、サーバーはその親ディレクトリをルートとするため、同じ階層のランも一覧に表示され、指定したランへ直接ディープリンクします。

### 読み取る対象

ダッシュボードは`libreyolo train`が書き出すファイル、つまり`status.json`、`metrics.jsonl`、`train.log`、およびランの画像から構築されます。学習プロセス自体からは何も読み取らないため、完了したランや異常終了したランも、実行中のランとまったく同じように表示されます。

### 前提条件とポート

少なくとも1つのランがすでに存在している必要があります。引数を指定せず`runs/`ディレクトリもない場合、コマンドは`source_not_found`で終了します。指定したディレクトリにランが1つもない場合も同様です。

ポートが使用中の場合は次のポートに移り、要求したポートから20個先まで試します。20個すべてで失敗した場合は`io_error`で終了します。stdoutに出力されるURLは、実際にバインドされたポートです。

コマンドはCtrl+Cを押すまでフォアグラウンドで配信を続けます。`json=true`を指定すると、URL、監視しているルート、見つかったランの数を、`schema_version`を含む1つのオブジェクトとして出力します。

関連：[`libreyolo train`](/docs/cli/train)。その`project`引数と`name`引数が、これらのランディレクトリの出力先を決めます。
