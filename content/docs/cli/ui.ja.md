---
title: libreyolo ui
seo_title: libreyolo ui コマンドリファレンス
description: ローカル推論用web UIの起動：バインドアドレス、ポートの挙動、デバイスの選択、コマンドの終了方法。
lead: ドロップまたは貼り付けた画像を受け取り、選んだモデルで推論して、結果をブラウザーに表示するローカルwebサーバーを起動します。
keywords:
  - libreyolo ui コマンド
  - libreyolo web ui
  - ローカル推論 web ui
  - ドラッグアンドドロップ 推論
  - libreyolo ui ポート
last_verified: 1.5.0
meta:
  - label: コマンド
    value: libreyolo ui
    mono: true
  - label: 出力
    value: 標準出力にサーバーのURL、その後プロセスはフォアグラウンドのまま
snippets:
  examples:
    - label: 基本
      language: bash
      code: |
        libreyolo ui
    - label: ポート固定、ブラウザーを開かない
      language: bash
      code: |
        libreyolo ui port=9000 no_browser=true
    - label: CPUで実行、機械可読
      language: bash
      code: |
        libreyolo ui device=cpu json=true
source_hash: b0eebd33fd0f463b
---

## 書式

```bash
libreyolo ui [key=value ...]
```

引数は`key=value`のペアで、POSIX形式も使えるため、`port=9000`と`--port 9000`は
同じ引数です。

## 引数

| 引数 | デフォルト | 意味 |
|---|---|---|
| `host` | `127.0.0.1` | バインドするホストまたはインターフェース |
| `port` | `8000` | バインドするポート。使用中なら次の空きポートに繰り上げ |
| `device` | `auto` | デバイス：`0`、`cpu`、`mps`、`auto` |
| `no_browser` | `false` | ブラウザーを自動で開かない |
| `json` | `false` | 標準出力にJSONを出力 |
| `quiet` | `false` | 標準エラー出力を抑制 |
| `verbose` | `false` | 標準エラー出力を詳細に |

## 例

<code-tabs name="examples" />

## 補足

デフォルトのバインド先はループバックなので、UIにアクセスできるのはこのマシンから
だけです。

指定したポートが使用中の場合、コマンドは次のポートを試し、指定値から20ポート先まで
繰り返します。20個すべてに失敗すると`io_error`で終了し、別のポートを指定するよう
提案します。標準出力に表示されるURLは実際にバインドされたポートなので、指定した
ポートを前提にせず、そのURLを読み取ってください。

`no_browser=true`でない限り、バインドの直後にそのURLでブラウザーのタブが開きます。

その後、コマンドはCtrl+Cが押されるまでフォアグラウンドで動作し、Ctrl+Cを押すと
サーバーは正常に停止します。デタッチモードはないため、ターミナルを解放したい場合は
シェルの機能でバックグラウンドに回してください。

`json=true`を指定すると、サーバーの起動前にURLとデバイスが`schema_version`付きの
1つのオブジェクトとして出力され、スクリプトはこれでバインドされたポートを取得します。

関連：ボックスを描いてラベルを保存するには[`libreyolo label`](/docs/cli/label)、
学習の実行を監視するには[`libreyolo monitor`](/docs/cli/monitor)。どちらも同じ
ポートとブラウザーの挙動を持つローカルwebサーバーです。
