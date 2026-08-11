---
title: データセット形式
seo_title: すべてのタスク向けLibreYOLOデータセット形式
description: >-
  正規タスクごとのデータセットファイル規約を説明します。YAMLキー、フォルダー配置、ラベル行、マスクとマップの規約、および各形式を読み取るローダーを扱います。
lead: >-
  このページは、ライブラリ自身のdocs/dataset_schema.mdにあるデータセットファイル規約に対応します。各正規タスクが想定するYAMLキーとディスク上の配置を扱います。
keywords:
  - libreyolo データセット 形式
  - yolo ラベル 形式
  - data.yaml
  - セグメンテーション マスク データセット
  - coco panoptic 形式
  - 深度 データセット
  - pose kpt_shape
last_verified: 1.5.0
verification: >-
  libreyoloリポジトリv1.5.0のdocs/dataset_schema.mdに対応し、ローダー名はlibreyolo/data/に照らして確認しました。
snippets:
  usage:
    - label: 検出ラベルの1行を解析
      language: python
      code: >
        from libreyolo.data import parse_yolo_label_line


        # class_id cx cy w h を [0, 1] に正規化

        row = parse_yolo_label_line("0 0.5 0.5 0.25 0.5", 640, 480,
        num_classes=80)


        # ピクセル単位の (class_id, x1, y1, x2, y2, area)

        print(row)
source_hash: a8282c079624044d
---

## 共通YAML

`detect`、`segment`、`pose`、`obb`に適用されます。

| キー | 必須 | 意味 |
|---|---|---|
| `path` | | データセットルート |
| `train` | 学習時 | 学習画像 |
| `val` | 検証時 | 検証画像 |
| `test` | | テスト画像 |
| `names` | 必須 | クラス一覧、または整数キーのマッピング |
| `nc` | | クラス数。存在する場合は`names`と一致する必要がある |
| `download` | | ダウンロード手順。Pythonスクリプトには明示的な許可が必要 |
| `annotations` | | detect、segment、obb向けの、splitからネイティブCOCO JSONファイルへのマッピング |

`train`、`val`、`test`には画像ディレクトリ、画像一覧の`.txt`ファイル、またはそれらのリストを指定できます。ラベルパスは1つの置換規則に従います。

```text
images/.../image.jpg -> labels/.../image.txt
```

ネイティブCOCO JSONデータセットでは、`annotations`がsplitをJSONファイルへマッピングし、splitパスが画像ルートを指定します。

```yaml
path: dataset
train: images/train
val: images/val
annotations:
  train: annotations/train.json
  val: annotations/val.json
```

`names`が存在する場合、ネイティブCOCO JSONのカテゴリ名はYAMLのクラス名と一致する必要があり、その名前がモデルのラベルIDを定義します。`names`がない場合、COCOカテゴリIDを並べ替えて`0..N-1`へ密にマッピングします。

データセットYAMLは`task`キーを持ちません。明示的なモデルとタスクの選択が優先されます。

すべてのテキストラベルファイルに共通する規則は次のとおりです。

- 画像ごとに1つの`.txt`ラベルファイル。
- ラベルファイルがない、または空の場合は物体なし。
- `class_id`は`0..nc-1`内の整数。
- 座標は`[0, 1]`内の有限な正規化浮動小数点数。
- 座標は元画像の幅と高さを基準とする。
- 行は信頼度もトラックIDも持たない。

<code-tabs name="usage" />

## detect

各行は正確に5フィールドです。

```text
<class_id> <cx> <cy> <w> <h>
```

`cx cy w h`は正規化された軸平行ボックスで、`w`と`h`は正でなければなりません。

## segment

ポリゴンの行は次の形式です。

```text
<class_id> <x1> <y1> ... <xN> <yN>
```

`N`は3以上、`class_id`後の座標数は偶数で、ポリゴンは非縮退でなければなりません。5フィールドの検出行も受け付けられ、長方形セグメントを表します。

## pose

YAMLには必須の`kpt_shape`を追加します。値は`[K, 2]`または`[K, 3]`です。任意の`flip_idx`は`0..K-1`の整数順列です。

```text
<class_id> <cx> <cy> <w> <h> <k1x> <k1y> [<k1v>] ... <kKx> <kKy> [<kKv>]
```

フィールド数は正確に`5 + K * D`で、`D`は`kpt_shape`の2番目の値です。キーポイント座標は正規化されます。存在する場合、可視性`v`は`0`、`1`、`2`のいずれかです。

## obb

正確に9フィールドです。

```text
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```

4つの点は`[0, 1]`内の正規化画像座標で、非縮退の回転長方形を形成します。ラベルファイルに角度は格納されません。

正規パーサーはデフォルトで厳密に動作し、範囲外の座標を拒否します。データセットと検証の取り込みでは、それ以外は有効なcrop境界のラベルについて座標を`[0, 1]`へクリップできますが、その後も縮退ボックスは拒否します。解析はタスクを考慮します。9フィールドは`obb`モードだけで`obb`を意味し、`segment`モードでは4点ポリゴンとして扱われることがあります。

内部では正規化コーナーを正規`xywhr`へ変換し、角度はボックス中心を軸とする幅方向の回転をラジアンで表します。公開結果では、OBB検出を`xywhr, conf, cls`行として公開します。

ネイティブCOCO JSONのOBB読み込みは、次の優先順位でアノテーションを受け付けます。ピクセル空間の8コーナーとしての`obb`、角度をラジアンとする`[cx, cy, w, h, angle]`としての`obb`、最小面積長方形へ再フィッティングするCOCO `segmentation`ポリゴンまたはRLE、軸平行として読み取り正規化するCOCO `bbox`です。

コーナーを考慮したOBBデータ拡張が存在するまで、OBB学習ではmosaicとmixupが無効です。

正規行パーサーは`libreyolo.data.parse_yolo_obb_label_line`です。

## semantic

各画像は`.txt`ファイルではなく、通常はPNGのロスレス形式の密な単一チャンネルマスクと対になります。

```text
images/.../image.jpg -> <masks_dir>/.../image.png
```

マスクは単一チャンネルで、パレットモードPNGはパレットインデックスとして読み取られます。各ピクセル値は`0..nc-1`内のクラスIDで、ピクセル値`255`は無視を意味し、損失とメトリクスから除外されます。マスク解像度は画像解像度と一致する必要があります。

共通規約に2つの任意YAMLキーを追加できます。`masks_dir`は各画像パス内の`images`を置き換えるマスクディレクトリ名で、デフォルトは`masks`です。`label_mapping`は読み込み時にマスクのピクセル値へ適用する`{source_id: train_id}`再マッピングです。マッピングされないソース値は無視となり、学習IDは`0..nc-1`内でなければなりません。

`masks_dir`を省略すると、`images`から`labels`への規則で解決される`segment`ポリゴンラベルを読み込み時にラスタライズし、物体クラスの後に`background`クラスを追加するため、`nc`が1増えます。

正規ローダーは`libreyolo.data.SemanticDataset`です。

## panoptic

LibreYOLOはCOCO-panoptic形式（Kirillovほか、CVPR 2019）をそのまま採用します。LibreYOLO固有のpanoptic形式はありません。

画像ごとに画像解像度のRGB PNGを1つ用意し、各ピクセルのセグメントIDを色でエンコードします。

```text
segment_id = R + 256 * G + 256 * 256 * B
```

各ピクセルは正確に1つのセグメントに属し、セグメントは重なりません。セグメントID `0`、RGBの黒はvoidです。ラベルなしのピクセルとしてメトリクスから除外されます。

```json
{
  "images":      [{"id": 139, "file_name": "000000000139.jpg"}],
  "annotations": [{"image_id": 139, "file_name": "000000000139.png",
                   "segments_info": [
                     {"id": 3226956, "category_id": 1, "area": 2840,
                      "bbox": [413, 158, 53, 138], "iscrowd": 0}]}],
  "categories":  [{"id": 1, "name": "person", "isthing": 1, "supercategory": "person"}]
}
```

`annotations[].file_name`は`panoptic_dir`内のセグメントID PNGを示し、`segments_info[].id`はそのPNG内の値と一致します。`iscrowd`はグループ領域を示します。これらはfalse negativeにはならず、大部分を覆う予測も検出照合でfalse positiveにはなりません。

thingとstuffの区別はカテゴリごとの属性です。`isthing`は`categories`にあり、`segments_info`にはありません。

COCO-panopticの`category_id`値はデータセットの未加工IDで、通常は連続していません。モデルは連続した`0..nc-1`を予測するため、未加工IDはカテゴリ名に基づいてYAMLの`names`を通じて再マッピングされます。これはネイティブCOCO JSON検出ローダーと同じ規則です。`names`にないJSONカテゴリは暗黙に削除せず、エラーになります。そうしなければ常にfalse negativeとして評価されるためです。

```yaml
path: coco
val: images/val2017
annotations:
  val: annotations/panoptic_val2017.json
panoptic_dir:
  val: annotations/panoptic_val2017
names: {0: person, 1: bicycle, 132: rug-merged}
```

`annotations`と`panoptic_dir`は単一パスまたはsplitごとのマッピングを受け付けます。

検証は正解データの解像度で計算し、出現するカテゴリで平均したPanoptic Qualityを報告し、さらに`PQ_things`と`PQ_stuff`へ分けます。照合は一意です。同じカテゴリの予測セグメントと正解セグメントは、IoUが0.5を超えると一致します。

正規ローダーは`libreyolo.data.PanopticDataset`です。

## depth

各画像は密な単一チャンネル深度マップと対になります。

```text
images/.../image.jpg -> <depths_dir>/.../image.png
```

マップは画像解像度の単一チャンネルPNGまたはTIF、あるいは`.npy`ファイルです。値はデータセット内で一貫した単位による通常の深度です。ゼロ、負、NaN、無限の値は無効ピクセルを示し、損失とメトリクスから除外されます。

| キー | デフォルト | 意味 |
|---|---|---|
| `depths_dir` | `depths` | `images`を置き換える深度ディレクトリ |
| `depth_stem_suffix` | | 画像stemに追加するサフィックス。省略時は同じstemと`_depth`サフィックスの両方を試行 |
| `depth_mask_suffix` | `_mask` | 有効性マスクのサフィックス。ゼロ以下、NaN、無限のマスク値は深度ピクセルを無効化 |
| `depth_scale` | `256.0` | 整数型深度マップの除数。一般的な16ビットPNG規約 |

浮動小数点の`.npy`マップはそのまま使われ、`depth_scale`は適用されません。

正規ローダーは`libreyolo.data.DepthDataset`です。

## edge

各RGB画像は同じstemの単一チャンネルロスレスマップと、任意の有効性マスクに対応します。

```text
images/val/scene.jpg -> edges/val/scene.png
                     -> masks/val/scene.png
```

マップはRGB表示画像ではなく、画像解像度の単一チャンネルPNGまたはTIFです。整数マップはdtypeの最大値で除算され、浮動小数点マップはすでに有限で`[0, 1]`内でなければなりません。`0`は非エッジ、`1`はエッジを意味します。任意マスクではゼロでないピクセルが有効です。リサイズではターゲットとマスクにnearest-neighbor補間を使い、パディングされたピクセルは無効で検証に寄与しません。

| キー | デフォルト | 意味 |
|---|---|---|
| `edges_dir` | `edges` | `images`を置き換えるエッジマップディレクトリ |
| `edge_stem_suffix` | | 画像stemに追加するサフィックス |
| `edge_extension` | `.png` | ロスレスなターゲット拡張子 |
| `edge_invert` | | ソースマップが白地に黒いエッジを格納する場合にtrueを設定 |
| `masks_dir` | `masks` | 任意の有効性マスクディレクトリ |

```yaml
path: edge-dataset
train: images/train
val: images/val
edges_dir: edges
masks_dir: masks
nc: 1
names: {0: edge}
```

検証は4方向の勾配NMSで連続予測を細線化し、設定可能なしきい値走査にわたるODSとOISのF値を報告します。予測ピクセルと正解ピクセルは`edge_max_dist * image_diagonal`内で1対1に照合され、デフォルトの正規化許容値は`0.0075`です。

正規ローダーは`libreyolo.data.EdgeDataset`です。ローダーは形式だけを扱い、ベンチマークデータをダウンロードまたは再配布しません。

## normal

各画像は同じstemの3チャンネル16ビットPNGと、任意の同じstemの有効性マスクに対応します。

```text
images/val/room.jpg -> normals/val/room.png
                    -> masks/val/room.png
```

PNGは画像解像度で、チャンネルをRGBとして格納した正確に3チャンネルの`uint16`です。`n = png / 65535 * 2 - 1`でデコードし、各ベクトルを再正規化します。デコード済みベクトルはOpenCVカメラ座標系を使い、`+x`は右、`+y`は下、`+z`はシーンの奥を向き、カメラに正対します。任意マスクは単一チャンネルPNGで、ゼロでない値が有効です。マスクがない場合、有限でゼロでないデコード済みベクトルがすべて有効です。無効またはパディングされたターゲットピクセルは、内部で`(0, 0, 0)`として表されます。リサイズでは3成分をbilinear補間してから再正規化し、有効性マスクにはnearest-neighbor補間を使います。水平反転ではx成分の符号も反転します。

| キー | デフォルト | 意味 |
|---|---|---|
| `normals_dir` | `normals` | `images`を置き換える法線マップディレクトリ |
| `masks_dir` | `masks` | 任意の有効性マスクディレクトリ |

検証は角度誤差の平均値と中央値（度）、および11.25度、22.5度、30度以内にある有効ピクセルの割合を報告します。

正規ローダーは`libreyolo.data.NormalDataset`です。

## restore

各劣化入力画像はきれいなRGBターゲットと対になります。

```text
inputs/.../image.jpg -> targets/.../image.jpg
```

入力とターゲットはRGB互換の画像ファイルで、解像度が正確に一致する必要があります。検証はネイティブ解像度を維持し、バッチに積み重ねるために必要な分だけパディングします。メトリクスは元画像のキャンバス上で計算されます。学習では入力とターゲットの組に連動したcropと水平反転を適用します。

| キー | デフォルト | 意味 |
|---|---|---|
| `input_dir` | `inputs` | splitパスで使う劣化入力ディレクトリ |
| `target_dir` | `targets` | `input_dir`を置き換えるきれいなターゲットディレクトリ |
| `target_stem_suffix` | | ターゲット検索前に入力stemへ追加するサフィックス |
| `target_stem_suffixes` | | `target_stem_suffix`のリスト形式 |
| `degradation` | | `deblur`や`denoise`などのメタデータラベル |
| `dataset` | | データセットまたは出所ラベル |

クラス相当のYAMLフィールドはスキーマのプレースホルダーです。`nc: 1`と`names: {0: image}`を使ってください。復元モデルは検出ではなく`Results.restored`を公開します。

正規ローダーは`libreyolo.data.RestoreDataset`です。

## matte

各RGB画像は、同じstemを共有する単一チャンネルの正解matteと対になります。0が背景、255が前景です。

```text
images/subject.jpg -> mattes/subject.png
```

2つの配置を受け付けます。1つ目は`images/`とmatteディレクトリを含むディレクトリルートを`data=`として渡す形式で、matteディレクトリは`mattes/`、`matte/`、`gt/`、`masks/`、`mask/`、`alpha/`から自動検出されます。2つ目は`path`とsplitごとの`val_images`、`val_mattes`、任意の`train_images`、`train_mattes`を持つYAMLで、各値は`path`からの相対パスまたは絶対パスです。

matteはグレースケールで`[0, 1]`内の不透明度として読み取られ、形状が異なる場合はbilinear補間で予測キャンバスへリサイズされます。メトリクスは元画像キャンバス上のMAEとS-measure（Fanほか、ICCV 2017）で、S-measureが最良チェックポイントの適合度です。

クラス相当のYAMLフィールドはスキーマのプレースホルダーです。`nc: 1`と`names: {0: matte}`を使ってください。matteモデルは`Results.matte`を公開します。

このバージョンの検証は推論専用です。正規ペアリゾルバーは`libreyolo.data.matte_dataset.resolve_matte_pairs`です。

## ocr

ラベルはsplitごとに1つのJSONLファイルで、画像ごとに1つのJSONオブジェクトを記述します。

```text
images/val/receipt.jpg -> labels/val.jsonl
```

```json
{"image": "receipt.jpg", "regions": [{"polygon": [[10, 12], [118, 14], [117, 40], [9, 38]], "text": "TOTAL 12.50"}]}
```

`polygon`は絶対ピクセル座標の4点四角形で、左上、右上、右下、左下の順です。読み取れないテキストを含む領域では、ICDARの評価対象外規約である`"text": "###"`を使います。これらは認識評価から除外され、重なる予測も検出照合でペナルティを受けず無視されます。

メトリクスは、IoU 0.5を超える1対1ポリゴン照合による検出hmean、IoUが0.5を超え、NFKC正規化と空白除去後の転記テキストが大文字と小文字を区別して完全一致することを求めるエンドツーエンドF1、照合済みペアの1-NEDです。最良チェックポイントの適合度はエンドツーエンドF1です。

2つの配置を受け付けます。1つ目は`images/<split>/`と`labels/<split>.jsonl`を含むディレクトリルートを`data=`として渡す形式です。2つ目は`path`と任意の`images`および`labels`ディレクトリ名を持つYAMLです。

クラス相当のYAMLフィールドはスキーマのプレースホルダーです。`nc: 1`と`names: {0: text}`を使ってください。OCRモデルは`Results.ocr`を公開します。

このバージョンの検証は推論専用です。正規サンプルリゾルバーは`libreyolo.data.ocr_dataset.resolve_ocr_samples`です。

## classify

ラベルファイルではなく、ImageFolder形式のディレクトリツリーを使います。

```text
dataset_root/
  train/
    class_a/*.jpg
    class_b/*.jpg
  val/
    class_a/*.jpg
    class_b/*.jpg
```

学習には`train/`が必要で、並べ替えたフォルダー名によりクラスからインデックスへのマッピングを定義します。検証には`val/`が必要です。`test/`も置けますが、デフォルトの学習と検証コマンドは使用しません。学習以外のsplitは、想定される学習またはチェックポイントのクラス集合と同じクラスフォルダー名を含む必要があります。対応する画像拡張子は`libreyolo.data.classify_dataset.IMAGE_EXTENSIONS`で定義されます。

## gazeとpoint

`gaze`には学習または検証用データセットファイル規約が実装されていません。

`point`はデータセットラベルスキーマではなくモデル出力タスクです。pointファミリーは、ボックス行から物体中心を導出するなど、既存ラベルを内部で適応させることがありますが、point専用のテキストラベル形式は定義されていません。

