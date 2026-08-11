---
title: 検証と指標
seo_title: LibreYOLOの検証と指標
description: 任意のモデルでval()を実行し、各タスクが返す指標キーを読み、評価バックエンドを選択して、精度指標とともに検証lossを有効にします。
lead: >-
  検証はval()を通じてデータセット分割上でモデルを実行し、指標キーと浮動小数点値のフラットな辞書を返します。キーはリテラル文字列で、取得できるものはファミリーではなくタスクによって決まります。
keywords:
  - mAP50-95
  - COCO 評価
  - 検証 指標
  - faster-coco-eval
  - pycocotools
  - 検証 loss
  - mIoU
  - Panoptic Quality
  - Top-1 Accuracy
last_verified: 1.5.0
snippets:
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="coco8.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["speed/total_ms"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml
    - label: 別の分割で実行
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="coco8.yaml", split="train", batch=4)

        print(metrics)
  valloss:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="coco8.yaml", epochs=10, val_loss=True)
  json:
    - label: COCO形式の予測を書き込み
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.val(data="coco8.yaml", save_json=True, save_dir="runs/val/exp")
source_hash: d907183492fa3f57
---

## 検証の実行

`val()`はデータセットを受け取り、指標を返します。

<code-tabs name="val" />

戻り値は通常の`dict[str, float]`です。すべてのキーはリテラルなので、位置ではなく名前で読み取ってください。

主な引数は`data`、`split`、`batch`、`imgsz`、`conf`、`iou`、`workers`、`device`、`augment`、`save_json`、`verbose`です。`conf`のデフォルトは`0.001`、`iou`は`0.6`で、どちらも予測時のデフォルトより大幅に緩くなっています。mAPのsweepには信頼度の低い末尾まで必要なためです。`imgsz`は固定値ではなくモデル自身の入力サイズがデフォルトです。`split`は`val`、`test`、`train`だけを受け付けます。

検証構成のその他のフィールドはキーワード引数として渡せます。`save_dir`、`max_det`、`eval_max_det`、`half`、`amp_dtype`、`cache`、`save_plots`も含まれます。

## タスク別の指標キー

物体検出はCOCO系の数値を返します。

```text
metrics/mAP50-95   metrics/mAP50    metrics/mAP75
metrics/mAP_small  metrics/mAP_medium  metrics/mAP_large
metrics/AR1  metrics/AR10  metrics/AR100  metrics/AR_max_det
metrics/AR_small  metrics/AR_medium  metrics/AR_large
metrics/precision  metrics/recall
metrics/precision(B)  metrics/recall(B)  metrics/mAP50(B)  metrics/mAP50-95(B)
```

このうち2つは注意が必要です。`metrics/precision`と`metrics/recall`は後方互換性のため維持されるエイリアスで、precisionとrecallの組ではなく、mAP 50-95とAR@100の値を保持します。名前が明確なキーを使用してください。

インスタンスセグメンテーションでは、上記のmAPとARをサフィックスなしのキーでマスク指標として返し、ボックス版を`(B)`サフィックス、マスク版のコピーを`(M)`サフィックスで返します。このタスクのprecisionとrecallはサフィックス付きだけで、`metrics/precision(B)`と`metrics/recall(B)`、`metrics/precision(M)`と`metrics/recall(M)`になります。どちらの組も物体検出と同じエイリアス値を保持します。`(B)`の組はボックスのmAP50-95とAR@100、`(M)`の組はマスクのmAP50-95とAR@100です。

| タスク | キー |
|---|---|
| detect | `metrics/mAP50-95`、`metrics/mAP50`、`metrics/mAP75`と、上記のサイズ別およびrecall内訳 |
| segment | 上記のdetectキーのマスク版（サフィックスなしはマスク）。`precision`と`recall`は`(B)`と`(M)`だけで、どちらも同じ規則のエイリアス |
| pose | `metrics/keypoints_mAP50-95`、`metrics/keypoints_mAP50`、`metrics/keypoints_mAP75`、`metrics/keypoints_mAP_M`、`metrics/keypoints_mAP_L`と、対応する`keypoints_AR`キー |
| obb | `metrics/mAP50-95`、`metrics/mAP50`、`metrics/mAP75`、`metrics/precision`、`metrics/recall`と、`(OBB)`サフィックス付きのコピー |
| classify | `metrics/accuracy_top1`、`metrics/accuracy_top5` |
| semantic | `metrics/mIoU`、`metrics/pixel_accuracy` |
| panoptic | `metrics/PQ`、`metrics/SQ`、`metrics/RQ`、`metrics/PQ_things`、`metrics/PQ_stuff`、`metrics/categories` |
| depth | `metrics/abs_rel`、`metrics/rmse`、`metrics/delta1`、`metrics/delta2`、`metrics/delta3` |
| normal | `metrics/mean_angular_error`、`metrics/median_angular_error`、`metrics/within_11_25`、`metrics/within_22_5`、`metrics/within_30` |
| edge | `metrics/ODS`、`metrics/OIS`、`metrics/best_threshold` |
| restore | `metrics/PSNR`、`metrics/SSIM` |
| matte | `metrics/MAE`、`metrics/Smeasure` |
| ocr | `metrics/det_precision`、`metrics/det_recall`、`metrics/det_hmean`、`metrics/e2e_precision`、`metrics/e2e_recall`、`metrics/e2e_f1`、`metrics/rec_1-NED` |
| point | `metrics/precision`、`metrics/recall`、`metrics/f1`、`metrics/MLE`、`metrics/MAE`、`metrics/RMSE`と、mAP sweepキー |

OBBの`metrics/precision`と`metrics/recall`はエイリアスではありません。IoU 0.50における実際のprecisionとrecallで、最も緩い動作点、つまり`conf`（デフォルト`0.001`）を通過したすべての予測から取得します。`(OBB)`サフィックス付きのコピーは、上記の`(B)`や`(M)`と同じ規則で、同じ4つの値をタスク固有名の下に繰り返します。

`accuracy_top5`は実際にはtop-`min(5, num_classes)`です。3クラスのデータセットではtop-3となり、すべてのサンプルが条件を満たすため1.0になります。

pointタスクのsweepキーは距離しきい値から構築されます。デフォルトでは`metrics/mAP@[0.01:0.10]`となり、単一しきい値のキーは`metrics/mAP@0.01`です。`dist_thresholds`を渡すと両方の文字列が変わります。

ほとんどのタスクは、最良チェックポイント選択でデフォルト使用する単一値の`fitness`キーも返します。物体検出、セグメンテーション、OBBはこのキーを持ちません。それらのファミリーでは辞書が返す`metrics/mAP50-95`を使って選択します。姿勢推定は`fitness`も`metrics/mAP50-95`も返さず、代わりにトレーナーが`best_metric_key`を`metrics/keypoints_mAP50-95`へ設定します。

## 速度キー

すべてのバリデーターが処理時間を追加します。

```text
speed/preprocess_ms   speed/inference_ms   speed/postprocess_ms
speed/total_ms        speed/total_s        speed/images_seen
```

これらは実行全体で平均した画像あたりのミリ秒です。実行したマシンと設定を表すため、この値を報告するときはハードウェア、バッチサイズ、精度も併記しなければ意味がありません。

## 評価バックエンド

物体検出とセグメンテーションの指標はCOCO評価器で計算され、デフォルトの`faster_coco_eval=True`は`faster-coco-eval`パッケージがインストールされている場合にC++バックエンドを選択します。ない場合はプロセスごとに1回警告し、pycocotoolsへフォールバックします。

```text
faster_coco_eval requested but not installed; falling back to pycocotools.
Install with: pip install faster-coco-eval
```

実際に使用したバックエンドはモデルの`last_eval_backend`に記録され、物体検出系タスクではCLIも出力に含めます。環境から構成値を上書きするには`LIBREYOLO_FASTER_COCO_EVAL`を設定します。

`iou_thresholds`が使われるのはOBB経路だけです。COCO経路は独自の固定0.50から0.95のsweepで評価し、この値を無視します。

## 検証loss

デフォルトの検証は精度だけを報告します。`val_loss=True`を指定すると、検証バッチに対してファミリーの学習目的関数も計算します。

<code-tabs name="valloss" />

`metrics/loss`と項ごとの`metrics/loss/<component>`を出力します。各成分には学習とまったく同じ重みが付くため、合計するとtotalになります。ロガーでは`val/loss`と`val/loss/<component>`として表示され、`libreyolo monitor`は`metrics/loss`を`train/loss`へ重ねて表示します。

成分はファミリー固有です。

| タスク | ファミリー | 成分 |
|---|---|---|
| detect | `yolo9`、`yolo9_p2`、`yolo9_e2e` | `box`、`cls`、`dfl` |
| detect | `yolonas` | `cls`、`iou`、`dfl` |
| detect | `rfdetr` | `ce`、`bbox`、`giou` |
| detect | `rtdetr`、`rtdetrv2` | `vfl`、`bbox`、`giou` |
| detect | `dfine` | `vfl`、`bbox`、`giou`、`fgl`、`ddf` |
| detect | `domedetr` | `vfl`、`bbox`、`giou`、`fgl`、`ddf`、`defe_density`、`defe_reg` |
| detect | `deim`、`deimv2`、`rtdetrv4`、`ec` | `mal`、`bbox`、`giou`、`fgl`、`ddf` |
| detect | `rtmdet` | `cls`、`bbox` |
| detect | `picodet` | `cls`、`bbox`、`dfl` |
| detect | `yolox` | `iou`、`obj`、`cls`、`l1` |
| detect | `yolo7` | `iou`、`obj`、`cls` |
| point | `fomo` | `ce` |
| classify | `resnet`、`convnext`、`mobilenetv4`、`efficientnetv2` | `ce` |
| semantic | `segformer`、`lingbotvision`、`dinov2` | `sem` |
| restore | `nafnet` | `restore` |

ターゲット割り当てによって検証の時間とメモリが増えるため、デフォルトでは無効です。精度指標用にすでに生成したモデル出力を再利用し、2回目の順伝播は実行しません。評価モデルまたはEMAモデルに対して`no_grad`下で実行し、マルチGPU学習ではcollectiveを使わずrank 0上でローカル計算します。最良チェックポイントの選択は引き続き精度指標に基づきます。

意図的に行わないことが3つあります。1つ目は対照的ノイズ除去の項を含めないことです。これには順伝播時の正解データが必要ですが、検証の順伝播では渡さないためです。2つ目は評価モードのモデルを報告することです。BatchNorm統計やStochastic Depthなど、ファミリーの学習時と評価時の順伝播が実際に異なる箇所では、数値は評価モードを反映します。これが意図する比較です。3つ目は、ファミリーで未実装のタスクを通知なくスキップしないことです。セットアップ時に構成エラーを発生させます。

```text
val_loss=True currently supports RF-DETR detection only; segment, pose, OBB,
classify, and semantic tasks are not supported
```

FOMOは動作が変わらない例外です。バリデーターが常にこのlossを計算しており、`val_loss=True`は公開するキーだけに影響します。

拡張付き検証と検証lossは併用できず、両方を指定すると例外が発生します。

## 検証が書き込むファイル

`val()`は必ず保存ディレクトリに`config.yaml`を書き込みます。`save_dir`を指定しない場合、デフォルトは`runs/val/<model>_<size>_<timestamp>`です。

<code-tabs name="json" />

`save_json=True`は、物体検出では`predictions.json`、セグメンテーションでは`predictions_bbox.json`と`predictions_masks.json`を書き込みます。OBBは対応しておらず、その旨を報告します。

`save_plots=True`は`plots/`サブディレクトリへ書き込みます。OpenCVがインストールされている場合、物体検出では`box_metrics.png`、クラスごとのAPとrecallのチャート、precision-recallと信頼度の曲線、confusion matrix、アノテーション付きサンプル画像を生成します。セグメンテーションは各項目のマスク版を追加し、姿勢推定には独自の指標と曲線一式があります。その他のバリデーターはプロットを実装していません。画像分類、セマンティック、panoptic、深度、法線、エッジ、復元、matte、OCR、OBB、pointはいずれも何も書き込みません。プロットの失敗は警告され、実行を中止しません。

## 学習中の検証

学習は`eval_interval`エポックごとにデータセットの`val`分割で検証し、生成された指標が`best.pt`の選択、`patience`による早期停止、すべてのロガーの`val/`キーを駆動します。EMAが有効な場合、検証はEMA重み上で実行されます。

`eval_interval`、`patience`、`save_plots`については[ハイパーパラメーター](/docs/train/hyperparameters)を、数値の送信先については[実験ロガー](/docs/train/loggers)を参照してください。

## 関連項目

- バリデーターが読み取る分割キーと形式については[データセット](/docs/train/datasets)を参照してください。
