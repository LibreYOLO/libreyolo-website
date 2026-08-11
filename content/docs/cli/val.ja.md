---
title: libreyolo val
seo_title: libreyolo valコマンドリファレンス
description: コマンドラインからデータセットの分割（split）でチェックポイントを評価します：すべての引数とそのデフォルト値、および各タスクが返す指標キー。
lead: >-
  1つのモデルを1つのデータセット分割で評価し、指標を出力します。指標の組み合わせはモデルのタスクによって決まり、これらの数値はベンチマークの1行を組み立てる元になります。
keywords:
  - libreyolo val コマンド
  - libreyolo 検証 コマンド
  - yolo 評価 cli
  - mAP50-95 コマンドライン
  - libreyolo val 引数
last_verified: 1.5.0
meta:
  - label: コマンド
    value: libreyolo val
    mono: true
  - label: 必須
    value: 'model, data'
    mono: true
  - label: 出力
    value: 指標はstdoutに出力。指定すればruns/val/exp配下にプロットとCOCO JSONを保存
snippets:
  examples:
    - label: 基本
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml
    - label: プロットとCOCO JSON
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml \
          imgsz=640 batch=8 save_json=true save_plots=true \
          project=runs/val name=yolo9s-coco8 exist_ok=true
    - label: 機械可読
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml json=true quiet=true
source_hash: f6507840568c3725
---

## 構文

```bash
libreyolo val model=<name|path> data=<dataset.yaml> [key=value ...]
```

引数は`key=value`のペアで、POSIX形式も使えるため、`batch=8`と`--batch 8`は同じ
引数です。

## 引数

| 引数 | デフォルト | 説明 |
|---|---|---|
| `model` | | モデルの重みのパスまたはCLI名。必須 |
| `data` | | データセットYAMLのパス（YOLO形式、例：`coco8.yaml`）。必須 |
| `data_dir` | | データセットディレクトリの直接指定。YAML内のパスを無視 |
| `split` | `val` | データセットの分割：`val`、`test`、`train` |
| `batch` | `16` | バッチサイズ |
| `imgsz` | | 画像サイズ：`640`（正方形）または`480x640`（HxW）。未指定ならモデル自身の入力サイズ |
| `conf` | `0.001` | 信頼度のしきい値 |
| `iou` | `0.6` | NMSのIoUしきい値 |
| `max_det` | `300` | NMS後の画像1枚あたりの最大予測数 |
| `eval_max_det` | | COCO評価器の上限。未指定ならpycocotoolsのAP@100の慣例 |
| `faster_coco_eval` | `true` | インストール済みならCOCO指標にfaster-coco-evalのC++バックエンドを使用。なければpycocotoolsにフォールバック |
| `half` | `false` | FP16推論 |
| `amp_dtype` | `float16` | `half=true`のときのCUDA autocastのdtype：`float16`または`bfloat16` |
| `save_json` | `false` | COCO形式のJSON結果を保存 |
| `save_plots` | `false` | 検証プロットを保存：指標、クラスごとのAP、混同行列、サンプル |
| `workers` | `4` | データローダーのワーカー数 |
| `device` | `auto` | デバイス |
| `project` | `runs/val` | 出力ディレクトリのルート |
| `name` | `exp` | 実験名 |
| `exist_ok` | `false` | 出力ディレクトリを再利用 |
| `allow_download_scripts` | `false` | データセットYAMLのdownloadブロックに埋め込まれたPythonを許可 |
| `json` | `false` | stdoutへのJSON出力 |
| `quiet` | `false` | stderrを抑制 |
| `verbose` | `true` | 詳細出力 |
| `help_json` | `false` | コマンドスキーマをJSONで出力して終了 |

## 使用例

<code-tabs name="examples" />

## 補足

### 指標の内容

出力される指標のセットはモデルのタスクに従い、JSON出力も同じキーを使います。

物体検出、セグメンテーション、回転バウンディングボックスは`mAP50`、`mAP50_95`、
`precision`、`recall`を報告します。モデルが複数種類の出力を予測する場合は、種類
ごとのグループが`box_metrics`、`mask_metrics`、`obb_metrics`として併せて出力され、
それぞれが同じ4つのキーを持ちます。

分類は`accuracy_top1`と`accuracy_top5`を報告します。ポイント検出は`precision`、
`recall`、`f1`、`MLE`、`MAE`、`RMSE`、`mAP_sweep`を報告します。深度推定は
`abs_rel`、`rmse`、`delta1`、`delta2`、`delta3`を報告します。セマンティック
セグメンテーションは`mIoU`と`pixel_accuracy`を報告します。復元は`PSNR`と`SSIM`
を報告します。

JSON結果には`eval_backend`も含まれ、その数値を出したCOCO評価ライブラリと
バージョンが示されるため、2回の実行を比較するときに同じバックエンドで採点された
かどうかを確認できます。

### しきい値

ここでのデフォルト値は推論用ではなく評価用です：`conf`は`0.001`、`iou`は`0.6`で、
[`libreyolo predict`](/docs/cli/predict)では`0.25`と`0.45`を使います。`conf`を
表示用のしきい値まで上げると再現率が下がり、それに伴ってmAPも下がるため、その
ようにして得た数値は公開されている数値と比較できません。

`imgsz`はデフォルトでは未指定で、モデル自身の入力サイズを意味します。値を設定
するとそのサイズで評価され、これがチェックポイントをネイティブ解像度から離れた
条件で測る方法です。

### ダウンロードを伴うデータセット

`download`フィールドがURLのデータセットYAMLは、追加の許可なしに初回使用時に取得
されます。Pythonのダウンロードスクリプトが埋め込まれている場合は
`allow_download_scripts=true`が必要で、ローカルコードの実行が有効になったことが
stderrに警告として出ます。同梱の`coco8.yaml`と`coco128.yaml`はURL方式なので、何も
必要ありません。

### 出力と終了コード

指標はstdoutに、進捗はstderrに出力されます。`json=true`は`schema_version`を含む
1つのオブジェクトを出力し、`quiet=true`はstderrを止めます。

終了コードは、成功時が`0`、使い方や設定の誤りが`2`、データセットが見つからない
場合が`3`、モデルを読み込めない場合が`4`、その他の実行時エラーが`1`です。

関連：[`libreyolo train`](/docs/cli/train)は、`eval_interval`によって独自の
スケジュールで同じ評価を実行します。
