---
title: データセット
seo_title: LibreYOLOの学習用データセット
description: >-
  LibreYOLOが読み取るデータセットYAML、必要なフォルダー構成、自動ダウンロードの仕組み、学習前にデータセットを検査するdoctorコマンドを説明します。
lead: >-
  LibreYOLOのデータセットは、ルート、分割、クラス名を指定するYAMLファイルです。ラベルファイルの保存場所を含むその他の情報は、規則に従ってこのファイルから導出されます。
keywords:
  - YOLO データセット形式
  - data.yaml
  - カスタムデータセット 学習
  - YOLO ラベル形式
  - COCO JSON データセット
  - データセット 自動ダウンロード
  - LibreYOLO doctor
  - クラス不均衡 チェック
  - train val データ漏洩
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # 同梱名、相対パス、絶対パスのすべてを使用できます。
        model.train(data="coco8.yaml", epochs=10)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=coco8.yaml epochs=10
  doctor:
    - label: データセットを検査
      language: bash
      code: |
        libreyolo doctor my-dataset.yaml
    - label: 警告でもCIジョブを失敗させる
      language: bash
      code: |
        libreyolo doctor my-dataset.yaml strict=true json=true
    - label: 画像デコード処理をスキップ
      language: bash
      code: |
        # ラベルとYAMLだけを読み取ります。破損、重複、分割間のデータ漏洩の
        # 検査にはピクセルが必要なので、すべてスキップされます。
        libreyolo doctor my-dataset.yaml fast=true
    - label: Python
      language: python
      code: |
        from libreyolo import doctor

        report = doctor.diagnose("my-dataset.yaml", imgsz=640)

        for finding in report.findings:
            print(finding.severity.value, finding.check_id, finding.message)

        raise SystemExit(report.exit_code(strict=False))
source_hash: 9a12a0551c8b56e9
---

## 学習にデータセットを指定する

`data=`はYAMLパス、またはパッケージに同梱された構成の名前を受け取ります。

<code-tabs name="train" />

名前は固定順序で解決されます。存在する絶対パス、作業ディレクトリを基準に指定どおりの名前、`.yaml`を追加した同じ名前、同梱の構成ディレクトリの順です。一致するものがない場合、検索したすべてのディレクトリと同梱構成の一覧がエラーに表示されます。

## 同梱構成

パッケージ内の`libreyolo/config/datasets/`には13個のデータセット構成が同梱されています。

| 構成 | タスク | 注記 |
|---|---|---|
| `coco8.yaml` | detect | 8枚の画像。通常のURLからダウンロード |
| `coco128.yaml` | detect | 128枚の画像 |
| `coco1000.yaml` | detect | 学習800枚、検証200枚 |
| `coco5000.yaml` | detect | 学習4000枚、検証1000枚 |
| `coco.yaml` | detect | 完全なCOCO 2017 |
| `coco-val-only.yaml` | detect | val2017のみ |
| `coco8-pose.yaml` | pose | 8枚の画像。COCO-17キーポイント |
| `coco-pose.yaml` | pose | COCO 2017キーポイント |
| `ade20k.yaml` | semantic | 150クラス |
| `cityscapes.yaml` | semantic | 19クラス。手動でダウンロード |
| `cocostuff.yaml` | semantic | 182クラス。手動でダウンロード |
| `gopro.yaml` | restore | ぶれ画像と正解画像の組 |
| `sr8.yaml` | restore | 超解像用の組 |

通常のダウンロードURLを持つのは`coco8.yaml`と`coco128.yaml`だけです。それ以外は、後述する明示的な許可が必要なPythonダウンロードブロックを持つか、データがすでにディスク上にあることを前提とします。

## データセットの保存場所

YAMLの`path`キーはデータセットルートを指定します。絶対`path`はそのまま使われます。相対パスは最初にデータセットディレクトリの下、次にYAMLファイル自体の隣で検索されます。ダウンロード予定のデータセットはデータセットディレクトリの下に保存されます。

そのディレクトリは`~/datasets`で、`LIBREYOLO_DATASETS_DIR`環境変数により上書きできます。設定ファイルはありません。

## YAMLのキー

```yaml
path: my-dataset        # データセットルート
train: images/train     # 学習に必須
val: images/val         # 検証に必須
test: images/test       # 任意
nc: 3                   # 任意。namesと一致する必要がある
names:
  0: person
  1: helmet
  2: vest
download: https://example.com/my-dataset.zip   # 任意
```

`train`、`val`、`test`はそれぞれ、画像ディレクトリ、1行に1つの画像パスを記載した`.txt`ファイル、または両方を混在させたリストを受け付けます。`.txt`リストの行は相対パスでもよく、その場合はリストファイル自身のディレクトリを基準に解決されます。`#`で始まる行はスキップされます。

`names`にはリストまたは整数をキーとするマッピングを指定できます。`nc`は任意です。両方が存在して一致しない場合、doctorはエラーとして報告します。

## ディレクトリ構成とラベルファイル

物体検出、セグメンテーション、姿勢推定、方向付きボックスは同じ構成を使用します。ラベルパスは、画像パス内の`images`ディレクトリ要素を`labels`へ書き換え、拡張子を`.txt`へ変更して導出されます。

```text
my-dataset/
  images/train/0001.jpg   ->   labels/train/0001.txt
  images/val/0002.jpg     ->   labels/val/0002.txt
```

完全な`images`パス要素だけを書き換えるため、`images_old`という名前のディレクトリは変更されません。

物体検出の1行は5フィールドで、すべて元画像の幅と高さに対して`[0, 1]`へ正規化されます。

```text
<class_id> <cx> <cy> <w> <h>
```

ラベルファイルがないか空の場合は物体がない画像を意味し、例外を発生させず背景として学習します。5フィールドを超える行はポリゴンとして読み取られ、その外接範囲がボックスになります。このため、セグメンテーション用のエクスポートを物体検出の学習に使用しても問題なく読み込まれます。doctorはこの経路で処理された行数を報告します。

## その他のタスク

セグメンテーションは同じ構成を維持し、少なくとも3点を持つ`<class_id> <x1> <y1> ... <xN> <yN>`のポリゴン行を使用します。5フィールドの物体検出行も受け付け、長方形インスタンスを意味します。

姿勢推定では、YAMLに`kpt_shape: [K, D]`と任意の`flip_idx`順列を追加します。各行は正確に`5 + K * D`フィールドで、ボックスに続き、`x y`または`x y v`で表す`K`個のキーポイントを並べます。可視性は`0`、`1`、`2`のいずれかです。

方向付きボックスは正確に9フィールドで、クラスの後に正規化座標の4つの角点を並べます。角度はファイルに保存しません。

セマンティックセグメンテーションでは、各画像を同じ解像度のシングルチャンネルマスクと組み合わせます。`images`を`masks_dir`（デフォルトは`masks`）へ置き換えて解決します。ピクセル値`255`は無視を意味します。`label_mapping`は読み込み時にソースIDを学習IDへ再マッピングします。

画像分類はラベルファイルではなくImageFolderツリーを使用し、`train/`と`val/`のそれぞれにクラスごとのディレクトリを置きます。クラスからインデックスへのマッピングは、フォルダー名の並べ替え順です。

画像復元では、`input_dir`と`target_dir`を通じて、劣化した入力と同一解像度の正解ターゲットを組み合わせます。深度、表面法線、エッジでは、それぞれ固有のディレクトリキーを通じて画像と密なマップを組み合わせます。

深度スケールの規則やpanopticのsegment-id PNGエンコーディングを含むタスク別の完全な仕様は、ライブラリリポジトリの`docs/dataset_schema.md`にあります。

## ネイティブCOCO JSON

COCO JSONアノテーションファイルを直接使用できます。`annotations`マッピングを追加すると、分割パスが画像ルートになります。

```yaml
path: my-dataset
train: images/train
val: images/val
annotations:
  train: annotations/train.json
  val: annotations/val.json
```

`names`がある場合、JSONのカテゴリー名と一致する必要があり、`names`がモデルの予測するラベルIDを定義します。`names`がなければ、COCOカテゴリーIDを並べ替え、`0..N-1`へ密にマッピングします。

この経路では分割ごとに1つの画像ディレクトリを前提とします。パスのリストや`.txt`画像リストを指定すると、通知なく別のセットを読み込まず、例外が発生します。

## 自動ダウンロード

`train`または`val`パスが空ではないディレクトリか既存ファイルへ解決されると、データセットは存在すると見なされます。存在せず、YAMLに`download`キーがある場合、その値によって次の処理が決まります。

`http`または`https`のURLは取得され、zipの場合はデータセットルートへ展開されます。それ以外は埋め込みPythonスクリプトとして扱われ、`allow_download_scripts=True`の場合だけ実行されます。指定がなければ、警告とともにスクリプトをスキップし、ディスク上にあるデータを使って学習を続行します。

```bash
libreyolo train model=LibreYOLO9s.pt data=coco.yaml allow_download_scripts=true
```

このフラグはコード実行のゲートであり、ネットワークのゲートではありません。URLのダウンロードはどちらの場合も行われ、許可が必要なのは`download: |`ブロックです。フラグが有効な場合、CLIは警告を表示します。doctorが有効にすることはありません。

## 学習前にデータセットを検査する

`libreyolo doctor`は物体検出データセットを読み取り、GPUを使う前に問題になる箇所を報告します。エラーがあると終了コード1を返すため、CIゲートとして使用できます。

<code-tabs name="doctor" />

検査は6つのファミリーに分かれます。

| ファミリー | 検査対象 |
|---|---|
| `config` | `names`の欠落、`names`と一致しない`nc`、分割の欠落または空、重複クラス名 |
| `files` | ラベルファイルのない画像、画像のないラベル、分割に記載された画像の欠落、stemの衝突 |
| `labels` | 不正な行、`[0, nc)`の範囲外にあるクラスID、`[0, 1]`の範囲外にある座標、面積ゼロのボックス、極小または巨大なボックス、重複ボックス、バイト単位で同一のラベルファイル |
| `balance` | インスタンス数がゼロまたは少ないクラス、クラス不均衡率、1つの分割だけに存在するクラス、背景画像の割合 |
| `images` | デコードできないファイル、EXIF回転、異常なチャンネル構成、単一色の画像、完全または近似重複 |
| `splits` | 2つの分割に完全またはほぼ同一の画像が存在すること |

`--only`と`--skip`は検査IDまたはファミリープレフィックスを受け取るため、`skip=images,labels.tiny_object`も有効です。`--fast`はピクセルのデコードが必要な検査をすべて除外します。対象は`images`と`splits`ファミリーです。

知っておくべき動作が2つあります。`--strict`はエラーだけでなく警告でも終了コードを失敗にします。また、doctorが対象とするのは物体検出データセットだけです。姿勢推定、セグメンテーション、方向付きボックスのデータセットは、誤った仕様で検査せず、検出した形式を示すメッセージとともに拒否されます。

## 関連項目

- データを準備した後に`train()`が受け取る引数については[ハイパーパラメーター](/docs/train/hyperparameters)を参照してください。
- `val`または`test`分割で評価する方法については[検証と指標](/docs/train/validation)を参照してください。
