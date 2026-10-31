---
title: libreyolo doctor
seo_title: libreyolo doctorコマンドリファレンス
description: 学習の前に物体検出データセットを確認します：デフォルト付きの引数、スキップまたは選択できるチェックファミリー、CIでゲートにできる終了コード。
lead: >-
  物体検出データセットに一連のヘルスチェックを実行し、学習の妨げになるものを報告します：ファイルの欠落、壊れたラベル、破損した画像、スプリット間のリーク、クラスの不均衡。
keywords:
  - libreyolo doctor cli
  - データセット 健全性チェック
  - yolo データセット 検証
  - データセット リーク 確認
  - libreyolo doctor strict
last_verified: 1.5.0
meta:
  - label: コマンド
    value: libreyolo doctor
    mono: true
  - label: 必須
    value: data
    mono: true
  - label: 出力
    value: stdoutに指摘レポート。エラーが見つかった場合は終了コード1
snippets:
  examples:
    - label: 基本
      language: bash
      code: |
        # download=trueにすると同梱のcoco8.yamlが画像を欠く場合に自動ダウンロード
        libreyolo doctor coco8.yaml download=true
    - label: 画像デコードなしの高速パス
      language: bash
      code: |
        libreyolo doctor coco8.yaml download=true fast=true
    - label: 選択したチェックでCIゲート
      language: bash
      code: |
        libreyolo doctor coco8.yaml download=true strict=true json=true \
          only=labels,files,config
source_hash: 79e0ef471d567ea3
---

## 書式

```bash
libreyolo doctor <data.yaml> [key=value ...]
```

データセットは位置引数で、代わりに`data=<path>`も指定できます。両方を異なる値で
指定すると`config_conflict`で終了します。それ以外はすべて`key=value`のペアで、
POSIX形式も使えるため、`imgsz=1024`と`--imgsz 1024`は同じ引数です。

## 引数

| 引数 | デフォルト | 意味 |
|---|---|---|
| `data` | | 位置引数。YOLO物体検出形式のデータセットYAML。例：`coco8.yaml`。必須 |
| `imgsz` | `640` | 微小物体などピクセル基準のチェックで使う学習時の画像サイズ |
| `fast` | `false` | 画像のデコードをスキップし、破損・重複・リークのチェックを外す |
| `skip` | | スキップするチェックIDまたはファミリーをカンマ区切りで指定。例：`images,labels.tiny_object` |
| `only` | | 排他的に実行するチェックIDまたはファミリーをカンマ区切りで指定 |
| `strict` | `false` | CIゲート向けに、警告も終了コードを失敗扱いにする |
| `download` | `false` | 見つからない場合にURLベースのデータセットダウンロードを許可。スクリプトは実行しない |
| `json` | `false` | stdoutへのJSON出力 |
| `quiet` | `false` | stderrを抑制 |
| `help_json` | `false` | コマンドのスキーマをJSONで出力して終了 |

### チェックファミリー

`skip`と`only`は完全なチェックIDでもファミリーのプレフィックスでも受け付けるため、
`images`は`images.*`のチェックをすべて選択します。

| ファミリー | 対象 |
|---|---|
| `config` | データセットYAML自体：`names`の欠落、`names`に対する`nc`、スプリットの欠落、解決できない`path`、クラス名の重複 |
| `files` | 画像とラベルの対応：ラベルの欠落、画像の欠落、孤立したラベル、非対応の拡張子、大文字小文字の衝突 |
| `labels` | ラベルの内容：構文、ポリゴン行、範囲外のクラスID、範囲外の座標、退化したボックス、微小物体、巨大なボックス、極端なアスペクト比、重複ボックス、密集した画像、同一ファイル |
| `images` | ピクセルデータ：破損ファイル、EXIFの向き、異常なカラーモード、極小または極端な寸法、一様な画像、完全重複とほぼ重複 |
| `splits` | スプリット間のリーク、完全一致とほぼ一致 |
| `balance` | クラス分布：インスタンスがゼロまたは少ないクラス、不均衡、スプリットのカバレッジ、背景比率、スプリットの偏り |

## 例

<code-tabs name="examples" />

## 補足

### 終了コード

エラーが見つからなければ`0`、いずれかの指摘がエラーであれば`1`です。
`strict=true`を指定すると警告でも終了コードが`1`に上がり、これがCIゲートで
使いたい設定です。

使い方の問題には専用のコードがあります：`skip`や`only`に未知のチェックIDまたは
ファミリーを指定した場合は`2`、データセットが見つからない場合は`3`、データセットが
物体検出の形式でない場合も`3`です。

### 選択はスキャンの前に解決

`skip`と`only`はディスクから何かを読み込む前にチェックのレジストリに対して
解決されるため、入力ミスは長い画像パスのあとではなくその場で失敗します。何にも
一致しないセレクターはエラーになり、メッセージには既知のファミリーが列挙されます。

`skip`、`only`、`fast`の組み合わせで実行するチェックが1つも残らない場合も、
黙って成功するのではなくエラーになります。

### ダウンロード

`download=true`でない限りデータセットは取得されず、実行されるのはURLからの
ダウンロードだけです。データセットYAMLに埋め込まれたPythonのダウンロード
スクリプトは、フラグの値にかかわらずこのコマンドが実行することはありません。

### 対象範囲

チェックは物体検出のデータセット向けに書かれています。ラベルが姿勢推定・
セグメンテーション・回転ボックスの形式であるデータセットは検出され、誤ったルールで
採点されるのではなく`data_invalid`で拒否されます。

### 出力

人間向けのレポートはstdoutに出力され、`json=true`を指定すると、サマリーの件数、
データセットの統計、すべての指摘、スキップされたチェックの一覧を持つ構造化
オブジェクトに置き換わります。

関連：[`libreyolo train`](/docs/cli/train)。このコマンドは、その実行の前に
走らせることを想定しています。
