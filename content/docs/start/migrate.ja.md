---
title: 既存の重みをインポート
seo_title: LibreYOLOでアップストリームの重みを読み込む
description: LibreYOLOにアップストリームプロジェクトのチェックポイントを指定します。自動変換により、クラス数と名前を維持したまま読み込み時に再ラップします。
lead: >-
  LibreYOLOはアップストリームプロジェクトからモデルファミリーを移植しているため、公開済みチェックポイントはほぼそのまま読み込めます。不足しているのはメタデータです。自動変換が読み込み時に補います。
keywords:
  - libreyolo 重み 変換
  - upstream checkpoint 読み込み
  - libreyolo 移行
  - pth libreyolo 変換
  - 自動変換
last_verified: 1.5.0
meta:
  - label: エントリーポイント
    value: LibreYOLO("path/to/upstream.pth")
    mono: true
  - label: ソースの隣に書き込む名前
    value: '<source>-<Prefix><size>[-task].pt'
    mono: true
  - label: スクリプト式converter
    value: リポジトリ内のweights/
    mono: true
snippets:
  convert:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # すでにあるチェックポイントのパスに置き換える。認識された
        # アップストリーム形式はその場で変換され、ソースの隣に書き込まれて
        # 読み込まれる
        model = LibreYOLO("path/to/upstream-checkpoint.pth")

        # クラス数と名前はテンソルとファイル独自のメタデータから得られるため
        # ファインチューニング結果ではCOCOではなく独自のラベルセットを維持
        print(model.family, model.size, model.task, model.nb_classes)
        print(model.names)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=path/to/upstream-checkpoint.pth \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 結果を確認
      language: bash
      code: |
        # 変換後のファイルは公開済みファイルと同じスキーマを満たす
        libreyolo metadata path=path/to/upstream-checkpoint-LibreYOLO9t.pt
source_hash: bf9d7c7d168fd2c0
---

このページでは、ほかのプロジェクトのチェックポイントを扱います。独自のコードを古いLibreYOLOから移行する場合は、[1.5.0へのアップグレード](/docs/upgrade)を参照してください。

## 外部ファイルを読み込むときの動作

`LibreYOLO()` は、まず制限付きのweights-only経路から重みファイルを読み込みます。結果に完全なLibreYOLOメタデータが含まれていれば、そのまま使われます。含まれていなければ、ほかの処理を試みる前にファイルが自動変換機能へ渡されます。チェックポイントにpickle化されたサードパーティーオブジェクトが含まれている場合など、制限付き読み込みが完全に失敗すると、それらのオブジェクトを無効化するloaderを使って自動変換を試します。

自動変換は4つの処理を行います。アップストリームプロジェクトが使った形式からテンソル辞書を取り出します。登録済みのすべてのファミリーに、得られたキーを認識するか問い合わせ、アップストリームの命名がLibreYOLOの移植と異なる場合は名前を再対応付けします。選ばれたファミリーをメタデータスキーマv1.0を満たすチェックポイントにラップし、テンソル自体からサイズ、タスク、クラス数を読み取ります。最後に結果をソースファイルの隣へ書き込み、読み込みます。

<code-tabs name="convert" />

変換は通知なしでは行われません。変換したファイルは、ファミリー、ソース名、出力名、得られたクラス数とともにログへ記録されるため、実行ログに読み込まれたものが正確に残ります。

## 取り出しに対応する形式

アップストリームのチェックポイントでは、いくつかの一般的な場所に重みがネストされます。converterはテンソルを保持するものが見つかるまで順番に試します。`ema.module` 以下のEMA blockまたは平坦な `ema`、`module.` prefixを除いた `ema_state_dict`、次に `params_ema`、`params`、`ema_net`、`net`、`model`、`state_dict`、最後にオブジェクト自体です。最初のものだけでなく複数を試すことで、counterしか持たない `ema` blockが、その下にある実際の重みを隠すことを防ぎます。

wrapperのprefixも除去されます。分散学習による `module.`、compile済みモデルによる `_orig_mod.`、一部の再配布で追加される `model.model.` のネストです。

## 読み取る内容と取得元

サイズ、タスク、クラス数はファイル名ではなくテンソルから取得します。そのため、ファインチューニング済みチェックポイントは、アーキテクチャのデフォルトではなく独自のクラス数で変換されます。クラス名は、存在する場合はチェックポイント独自のメタデータから取得します。名前が `args` または `hyper_parameters` blockにあれば、そこから取得します。また、基本のラベルセットを保持したファインチューニング結果に、ヘッドからなくなったインデックスが残らないよう、検出したクラス数に合わせて切り詰めます。

密出力タスクには架空のラベルを与えず、明示的に処理します。深度チェックポイントには `depth` という1つのクラス、復元チェックポイントには `image` という1つのクラスが設定されます。姿勢チェックポイントからは、テンソルまたはファミリーのどちらかによってキーポイント数を取得する必要があります。どちらからも得られない場合は、不完全なファイルを書き込まず、変換を拒否します。

RF-DETRには専用のrecognizerがあります。サイズ検出にチェックポイント全体が必要であり、LibreYOLOが80クラスのCOCO規約を使うのに対して、そのヘッドは91出力を持つためです。チェックポイントに正確に80個の名前がある場合、クラス数80を宣言している場合、データセットとしてCOCOを指定している場合、またはクラスやデータセットのメタデータがまったくない場合は、80クラスに正規化されます。名前、80以外の明示的なクラス数、COCO以外のデータセットを示す情報から本物の90クラスモデルと判断された場合は、そのまま維持されます。

## 変換済みファイルの保存先

出力はソースの隣に、ソースに基づく名前で書き込まれます。

```text
<source stem>-<FilenamePrefix><size>[-<task suffix>].pt
```

そのため、`upstream-checkpoint.pth` として保存されたtiny YOLOv9検出器は `upstream-checkpoint-LibreYOLO9t.pt` になります。ファミリーではなくソースに基づいて命名するため、同じディレクトリにある同一ファミリー・サイズの2つのファインチューニング結果が互いを上書きせず、公式チェックポイントとも衝突しません。ファイルは読み込むたびに書き直されるため、ソースに対して古くなることはありません。ディレクトリが読み取り専用の場合、変換済みファイルは新しいprivate temporary directoryへ保存され、場所がログに表示されます。

以後は通常のLibreYOLOチェックポイントです。メタデータ経路から読み込まれ、`libreyolo metadata` で有効と報告されます。

## 手動対応が必要な場合

2つのファミリーはgeneric recognizerの対象外です。gazeファミリーは完全に除外されます。推論専用であり、公開済みの重みに再配布制限があるためです。RF-DETRは、代わりに処理を行う上記の専用recognizerを持つため除外されます。

アップストリームの未加工PIDNetチェックポイントは拒否され、エラーから `weights/convert_pidnet_weights.py` が案内されます。このスクリプトは、チェックポイントに必要なCityscapesのsemanticメタデータを書き込みます。

D-FINEとDEIMは同じアーキテクチャキーを共有するため、テンソルだけでは区別できません。両方がファイルを候補とし、識別用のmarkerを持つ兄弟ファミリーが候補にない場合は、ファイル名で決定します。`dfine_hgnetv2_n_coco.pth` または `deim_hgnetv2_n_coco.pth` の形の名前なら確定し、何も示さない名前は推測せず、その説明とともに拒否されます。`LibreDFINE` または `LibreDEIM` を直接instantiateしても解決できます。

複数のファミリーが正当に1つのファイルを候補とする場合、改良元の基底クラスよりsubclassが優先され、それ以外はレジストリの順序で決まります。その順序は各ファミリーのチェックがどれだけ具体的かを表すためです。ファイル名を参照するのはD-FINEとDEIMが同点の場合だけなので、ファイル名によって幅広い一致が正確な一致より優先されることはありません。

## スクリプト式converter

リポジトリの `weights/` 以下にはファミリーごとの変換スクリプトと、繰り返し使う処理の共通helperがあります。ランタイム経路が拒否するファイル、読み込み時ではなく事前にチェックポイントを作成する場合、テンソルから推測できずメタデータを指定する必要があるファミリーでは、この経路を使います。

これらのスクリプトはインストール済みパッケージではなくリポジトリの一部なので、使うにはcloneが必要です。

```bash
git clone https://github.com/LibreYOLO/libreyolo.git
cd libreyolo
python weights/convert_pidnet_weights.py --help
```

すべてのスクリプトはスキーマv1.0を満たすチェックポイントを書き込みます。これは自動変換および公開済み重みと同じ基準です。そのスキーマに含まれる内容については[チェックポイントと重み](/docs/weights)を参照してください。

