---
title: 完全なエクスポートマトリックス
seo_title: LibreYOLOのエクスポート対応マトリックスと規則
description: >-
  LibreYOLOがファミリー、タスク、形式の組み合わせをエクスポートできるか判断する仕組みを解説します。12形式、3ティア、フォールバック規則、一致度のしきい値を扱います。
lead: >-
  エクスポート対応状況は（ファミリー、タスク、形式）の3つ組で検索されます。このページではマトリックスの構造、明示的な項目がないセルを埋める規則、目的の組み合わせを問い合わせる方法を解説します。
keywords:
  - LibreYOLO エクスポート 対応形式
  - エクスポート マトリックス
  - ONNX TensorRT OpenVINO TFLite
  - LibreYOLO formats コマンド
  - エクスポート 一致度 しきい値
  - NotImplementedError export
last_verified: 1.5.0
verification: >-
  形式、ティア、フォールバック順、タスクとファミリーのブロック、NCNNのブロックはlibreyolo/export/support.py、別名と共通引数はlibreyolo/export/exporter.py、ティア定義はdocs/adr/0011-export-support-tiers.md、一致度しきい値はdocs/export_support.mdから確認しました。すべてv1.5.0時点です。組み合わせごとのセルはここへ転記していません。以下のスニペットで問い合わせてください。
snippets:
  usage:
    - label: モデルなしでマトリックスを問い合わせ
      language: python
      code: |
        from libreyolo.export.support import (
            EXPORT_FORMATS,
            get_support,
            validated_alternatives,
        )

        print(EXPORT_FORMATS)

        entry = get_support("yolo9", "detect", "onnx")
        print(entry.tier, entry.since)
        print(entry.constraint)

        print(validated_alternatives("yolo9", "detect"))
    - label: CLI
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
        libreyolo formats --family yolo9 --task detect --json
  export:
    - label: エクスポートして拒否理由を読む
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.export.support import get_support

        model = LibreYOLO("LibreYOLO9t.pt")
        print(model.export(format="onnx"))

        # 呼び出し前に確認する ブロック済みの組み合わせは事前確認で例外となり
        # メッセージにこの理由が含まれる
        blocked = get_support("domedetr", "detect", "onnx")
        print(blocked.tier)
        print(blocked.reason)
source_hash: 83de3289634888c6
---

## マトリックスの構造

マトリックスのキーは`(family, task, format)`です。ファミリーキーはモデルレジストリの標準名、タスクキーは`libreyolo.tasks.TASKS`から取得し、形式は次の12種類です。

`onnx`、`torchscript`、`executorch`、`tensorrt`、`openvino`、`paddle`、`mnn`、`rknn`、`ncnn`、`tflite`、`coreml`、`coreai`。

`model.export(format=...)`はさらに2つの別名を受け付けます。`tensorrt`を表す`engine`と、TensorFlow Liteの現在の名称である`tflite`を表す`litert`です。形式と`.tflite`接尾辞は変わりません。

<code-tabs name="usage" />

1つのセルが3つのキーから決まるため、完全なグリッドは大きく、リリースごとに変化します。手作業で記述せず生成され、ライブラリリポジトリの`docs/export_support.md`にあります。コピーを読むのではなく、PythonまたはCLIでマトリックスへ問い合わせてください。

## 3つのティア

| ティア | 意味 |
|---|---|
| `validated` | 数値の一致がCIまたは文書化された夜間実行で検証済み |
| `available` | 変換は実装済みだが、ランタイムでの数値的一致を示す根拠は未記録 |
| `blocked` | トレース前の事前確認で、理由とともに`NotImplementedError`を送出 |

validatedとavailableの組み合わせはどちらも確認操作や一律の警告なしで処理を続行します。記録済みの根拠と制約は生成ドキュメントに表示されます。blockedの組み合わせは、依存関係の確認、キャリブレーションの読み込み、トレース、アーティファクト作成より前に失敗します。

validatedの項目を追加するには一致度テストと`since`フィールドが必要です。

`SupportEntry`は`tier`、理由を表す`reason`文字列、導入リリースを表す`since`、`constraint`文字列という4つのフィールドを持ちます。統合時に重要なのは制約です。チェックマークは記載条件の範囲内だけに適用され、通常は固定入力キャンバス、バッチ1、FP32、特定のランタイムバージョンが示されます。

## セルの決定方法

`get_support(family, task, fmt)`は次の順序で解決します。最初に一致した規則が適用されます。

1. 未知のタスク、または12形式以外は`blocked`を返します。
2. 明示的な`(family, task, format)`項目があれば、記録どおりに返します。
3. ファミリー全体のブロックがあれば、そのファミリーの理由とともに`blocked`を返します。
4. タスク全体のブロックがあれば、そのタスクの理由とともに`blocked`を返します。
5. `ncnn`では、NCNNブロックリストにあるファミリーが`blocked`を返します。
6. `mnn`は`blocked`を返します。このファミリーとタスクのランタイム契約がありません。
7. `rknn`は`blocked`を返します。このバージョンのRKNNは、RK3588上でシミュレーター検証済みの正確な検出バリアント、YOLO9-t、YOLO9-E2E-t、YOLO-NAS-s、PicoDet-sだけに限定されます。
8. `tensorrt`と`openvino`は`available`を返します。変換経路は存在しますが、このファミリーとタスクでのランタイム一致は未記録です。
9. `tflite`、`paddle`、`coreai`、`coreml`は、それぞれ固有の理由とともに`blocked`を返します。
10. それ以外は`available`を返します。変換は実装済みですが、数値的なランタイム一致は未記録です。

手順8から10の非対称性は意図的です。TensorRTとOpenVINOはONNXから汎用的に変換するため、一覧にない組み合わせも試す価値があります。TFLite、Paddle、Core AI、CoreMLにはファミリーごとの経路が必要なので、一覧にない組み合わせは試行の案内ではなく拒否になります。

## ブロックされるタスク

次のタスクは、明示的な項目がないすべてのファミリーでブロックされます。

| タスク | 理由 |
|---|---|
| `ocr` | 動的な領域ごとのクロップを持つ2つのネットワークは単一グラフのエクスポート契約に収まらない |
| `point` | ファミリーが共通の点ヒートマップとバックエンドのピークデコード契約に接続されていない |
| `semantic` | ファミリーが共通の密なロジットとバックエンドのargmax契約に接続されていない |
| `mesh` | 人体メッシュのグラフ出力、メタデータ、ランタイム契約が未定義 |
| `normal` | ファミリーが固定キャンバスの密な単位法線とバックエンドの再正規化契約に接続されていない |
| `panoptic` | パノプティックエクスポートにバックエンドのランタイム契約がない |
| `gaze` | ファミリーが共通の2ヘッドロジットとバックエンドの期待値デコード契約に接続されていない |

明示的な項目はこれらを上書きします。そのため、たとえば接続済みのsemanticファミリーはエクスポートできます。

## ブロックされるファミリー

| ファミリー | ブロック対象 |
|---|---|
| `depth_anything3` | 全形式。深度グラフがエクスポート済みランタイム契約の対象外 |
| `domedetr` | 全形式。PAQIが画像ごとにクエリ数を設定するため、トレース済みグラフはトレースに使用した画像だけで有効。エクスポート可能なDETRにはD-FINEを使用 |
| `eomt` | ランタイム解析のないインスタンスおよびパノプティックのエクスポート |
| `l2cs` | ONNX、TorchScript、ExecuTorch、TensorRT、OpenVINO以外 |
| `hrnet` | ONNX、TorchScript、OpenVINO、TensorRT以外 |
| `sam`、`sam2`、`sam3`、`edgetam`、`mobilesam` | 全形式。プロンプト可能モデルのエクスポートはv1ランタイム契約の対象外 |
| `grounding_dino`、`owlv2`、`omdet_turbo`、`ov_deim` | 全形式。オープンボキャブラリーのランタイムエクスポートはv1の対象外 |
| `florence2`、`kosmos2`、`lfm2vl`、`internvl3`、`qwen3vl`、`smolvlm2`、`locateanything` | 全形式。生成VLMのエクスポートはv1の対象外 |

PicoSAM3はプロンプト可能ティアの例外で、生の96px ROIネットワークをONNXへエクスポートします。

## NCNNでのブロック

DETR形式のデコーダーはNCNNにないサンプリング演算を必要とするため、明示的な項目で別途指定されない限り、次のファミリーは`ncnn`でブロックされます。Deformable DETR、DETR、DINO-DETR、D-FINE、LW-DETR、DEIM、DEIMv2、RT-DETR、RT-DETRv2、RT-DETRv4、RF-DETR、ECです。拒否メッセージには代替としてONNX、OpenVINO、TorchScript、TensorRTが示されます。

## 一致度のしきい値

validatedのセルは、エクスポート済みアーティファクトが次の範囲内でネイティブモデルを再現したことを意味します。

| タスクグループ | しきい値 |
|---|---|
| 検出とOBB | 対応ボックスのIoUが0.95超、スコアMAEが0.01未満 |
| セグメンテーションとパノプティック | マスクIoUが0.95超 |
| 姿勢推定 | ネイティブ解像度でキーポイントL2が2ピクセル未満 |
| 画像分類 | ロジットのコサイン類似度が0.999超、top-1クラスが一致 |
| 深度と画像復元 | ネイティブ出力に対するPSNRが40 dB超 |
| サーフェス法線 | 平均角度誤差が0.1度未満 |
| 点検出 | ピーク位置が出力セル1個以内で一致 |

DETRのクエリ行は順序のない集合なので、DETRファミリーの一致度確認では位置順ではなく集合としてクエリ行を対応付けます。

## エクスポート

<code-tabs name="export" />

ブロックされた組み合わせは事前確認で`NotImplementedError`を送出し、メッセージに記録済みの理由が含まれます。`validated_alternatives(family, task)`はその組み合わせで検証済みの形式を返すため、拒否理由の隣に表示すると便利です。

全エクスポーター共通の引数は[モデルAPIページ](/docs/reference/model-api)に記載されています。形式固有の引数は各形式のページを参照してください。

## 制約の読み方

validatedのセルが示すのは1つの測定済み設定であり、形式全般に対する主張ではありません。`FP32, batch 1, fixed 520x520 input`のような制約文字列は、その形状と精度で一致度が記録されたことを意味します。異なる解像度やバッチサイズでエクスポートしてもアーティファクトは生成されますが、その数値の根拠となった設定ではありません。
