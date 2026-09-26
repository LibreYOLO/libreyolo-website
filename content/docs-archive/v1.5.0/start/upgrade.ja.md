---
title: 1.5.0へのアップグレード
seo_title: LibreYOLO 1.4.0から1.5.0へのアップグレード
description: 1.5.0で必要になる4つのコード変更、指標値が変わる3つの変更、実行結果を比較する前に把握しておきたい小さな動作変更を説明します。
lead: >-
  公開モデルAPIから削除されたものはありません。1.4.0で動作していたクラスと関数はすべて引き続きインポートできます。4つの引数の形式が変わり、3つのデフォルト変更によって比較対象の数値が変わる可能性があります。
keywords:
  - LibreYOLO アップグレード
  - LibreYOLO 1.5.0 移行
  - allow_experimental 廃止
  - LibreYOLO 破壊的変更
  - YOLOX BN eps
  - faster-coco-eval デフォルト
last_verified: 1.5.0
meta:
  - label: 対象バージョン
    value: 1.4.0から1.5.0
  - label: 必要なコード変更
    value: 限定的な4項目
  - label: 数値が変わる結果
    value: COCOバックエンド、YOLOX BN eps、D-FINEマルチスケール
  - label: 公開APIの削除
    value: なし
source_hash: ab38d8ef7b53f596
---

このページではLibreYOLO自体のアップグレードについて説明します。アップストリームプロジェクトのチェックポイントを読み込む方法を探している場合は、別のトピックである[既存の重みのインポート](/docs/migrate)を参照してください。

このリリースの完全な記録は[変更履歴](/docs/changelog)にあります。以下では、利用者側の対応が必要な部分だけを説明します。

## 必須のコード変更

### `allow_experimental=True`は廃止されました

確認用ゲートと、その背後にあった`ddp_aware(experimental_key=...)`の仕組みが廃止されました。以前はEC、RTMDet、PicoDet、FOMOの学習とエクスポートでこの引数が必要だったため、これらのファミリーを学習するスクリプトが影響を受けます。

```python
# 1.4.0
model.train(data="data.yaml", epochs=100, allow_experimental=True)

# 1.5.0: この引数を削除
model.train(data="data.yaml", epochs=100)
```

非推奨互換レイヤーはありません。引数を渡し続ける呼び出しでは`TypeError`が発生します。`BaseModel.EXPERIMENTAL_WEIGHT_FILENAMES`もあわせて削除されました。`get_download_notice()`フックは残っており、MiDaS、SegFormer、YOLO9-P2で引き続きオーバーライドされています。

サポートレベルは今も公開されていますが、引数ではなくなりました。[安定性レベル](/docs/reference/stability-tiers)を参照してください。

### エクスポートレベル`"experimental"`は廃止されました

```python
from libreyolo.export.support import Tier

# 1.4.0: Literal["validated", "experimental", "blocked"]
# 1.5.0: Literal["validated", "available", "blocked"]
```

レベル文字列で分岐するコードは、`"experimental"`を読んでいた箇所で`"available"`を読むように変更してください。`BaseExporter`はこれらの形式に対して`RuntimeWarning`を発行しなくなりました。形式ごとの状態は[エクスポートマトリックス](/docs/reference/export-matrix)に掲載されています。

### `pretrained=False`と`resume`の併用は拒否されます

以前は矛盾した状態のまま処理が進んでいました。現在は次のエラーが発生します。

```
ValueError: pretrained=False cannot be combined with resume.
```

どちらか一方を選んでください。`pretrained=False`は新しいシード済み初期値から開始します。1.5.0では3つだけでなく、学習可能なすべてのファミリーで動作します。`resume`は中断した実行をチェックポイントから再開します。どちらも[学習](/docs/train)に記載されています。

### CLIの`--imgsz`は整数ではなく文字列です

影響範囲は見た目より限定的です。次の2つは影響を受けません。

```bash
libreyolo predict --model yolo9-t --source img.jpg --imgsz 640   # 引き続き有効
```

```python
model.predict("img.jpg", imgsz=640)   # 引き続き有効
```

Pythonから[CLI](/docs/cli)のコマンド関数を直接呼び出すコードだけを変更する必要があります。長方形サイズを受け付けられるよう、`predict`、`train`、`val`の`--imgsz`が`int`から`str`へ拡張されたためです。

```python
from libreyolo.cli.commands.predict import predict_cmd

predict_cmd(..., imgsz=640)      # 1.4.0
predict_cmd(..., imgsz="640")    # 1.5.0では"480x640"も使用可能
```

`train`のデフォルトは文字列`"640"`になりました。`export --imgsz`は以前から文字列で、`profile`に変更はありません。

## 変化する数値

デフォルト設定の指標値が変わる変更は3つあります。バージョンをまたいで結果を追跡する場合は、1.5.0の実行結果と1.4.0を比較する前に確認してください。

### faster-coco-evalがデフォルトのCOCO指標バックエンドになりました

`val()`と各エポックの学習検証は、pycocotoolsではなくfaster-coco-evalのC++バックエンドでCOCO指標を計算するようになりました。

この切り替えは、RF100-VLの100個すべてのテスト分割で測定した同等性に基づいて決定されました。1400個の指標値のうち1381個がビット単位で同一、最大差は2.22e-16、主要指標の差は正確に0でした。全体では15.6倍、検出密度の高いデータセットでは56倍高速です。数値は変わらないはずですが、異なる実装で生成されるため、この一覧に含めています。

faster-coco-evalがインストールされていない場合、pycocotoolsが自動フォールバックとして残ります。強制的に使用するには次のようにします。

```bash
libreyolo val --model yolo9-t --data coco.yaml --no-faster-coco-eval
```

```python
model.val(data="coco.yaml", faster_coco_eval=False)
```

`LIBREYOLO_FASTER_COCO_EVAL=0`でもグローバルに同じ設定を行えます。実際に使用されたバックエンドはINFOレベルでログに記録され、`val()`の後に`model.last_eval_backend`で参照できます。[CLI](/docs/cli/val)のJSONペイロードにも`eval_backend`として含まれます。高速経路は`pip install libreyolo[fast-eval]`でインストールできます。

### 1.5.0より前に学習したYOLOXチェックポイントにはepsのオーバーライドが必要です

これは今回のリリースで注意が必要な点です。[YOLOX](/docs/models/yolox)をファインチューニングしている場合は確認してください。

YOLOXはBatchNormの`eps=1e-3`と`momentum=0.03`を指定します。1.5.0より前は、これらの値が後処理の修正として適用されており、データセットの`nc`がチェックポイントと異なるときに`train()`が行うクラス数の再構築では維持されませんでした。そのようなファインチューニングでは、PyTorchのデフォルト`eps=1e-5`で学習され、学習中の検証も行われた後、推論時には`1e-3`で再読み込みされていました。同じテンソルに異なる正規化が適用されていたことになります。

通常の畳み込みを使うサイズでは変化はごくわずかです。Depthwiseの`n`では、チャンネルごとの`running_var`が小さくepsの影響が支配的になるため、大きく変化します。RF100-VLの`ball`では、同じnanoチェックポイントを学習時のepsで評価するとmAP50-95は**0.566**ですが、通常どおり再読み込みすると**0.151**になります。

1.5.0より前に学習したチェックポイントはeps=1e-5の意味を持ちます。正しい数値を報告するには、BNのepsを1e-5へオーバーライドして評価します。

```python
import torch
from libreyolo import LibreYOLOX

model = LibreYOLOX("my-yolox-finetune.pt")
for module in model.model.modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        module.eps = 1e-5

model.val(data="data.yaml")
```

または、`sqrt((var + 1e-3) / (var + 1e-5))`をBNの重みに一度組み込んで結果を保存してください。1.5.0以降で学習したチェックポイントでは、どちらも不要です。

### D-FINEのマルチスケール学習はアップストリームのサイズ別レシピを使用します

以前はすべてのサイズで`base_size_repeat`が3に固定されていました。現在はアップストリームの指定に従い、サイズごとに解決されます。**n**は固定サイズでマルチスケール無効、**s**は20、**m**は6、**l**は4、**x**は3です。以前と一致するのはxだけなので、n、s、m、lではスケール分布が変わり、収束する指標値も変わります。

以前の動作へ戻すには明示的に設定します。

```python
from libreyolo.training.config import DFINEConfig

config = DFINEConfig(base_size_repeat=3)
```

DEIMは引き続き3に固定されています。ファミリーの詳細は[D-FINE](/docs/models/d-fine)にあります。

## 知っておくべき変更（対応不要）

- **長方形の`imgsz`で結果が変わるのは、以前の結果が誤っていたためです。** ボックス座標、RTMDetのマスクサイズ変更、YOLO-NASの再スケーリング、バリデーターの正解データスケーリングで、単一のスカラーではなく高さと幅を軸ごとに使用するようになりました。正方形の`imgsz`はビット単位で変わりません。1.4.0での長方形の推論または検証は誤ってスケーリングされていました。YOLO-NASは誤った出力を通知なく生成せず、長方形の`imgsz`を明示的に拒否するようになりました。
- **指標辞書にキーが追加されました。** COCO評価器から`max_det`、`ar_max_det`、`AR_max_det`が、FOMOから`metrics/loss`と`metrics/loss/ce`が追加されました。デフォルトでの値は変わりませんが、カスタム[ロガー](/docs/train/loggers)やCSVヘッダーなど、指標キーを反復処理するものには新しい列が表示されます。
- **ヘッド再構築が発生するシード付きYOLO9実行**では、再構築後ではなく再構築前にシードを適用するようになったため、異なる初期値から始まります。異なるクラス数へファインチューニングしたシード付き1.4.0実行を、1.5.0でビット単位に再現することはできません。
- **CUDA上の`libreyolo[hub-kernels]`で、ネイティブのMS-deform-attnカーネルが実際に使われるようになりました。** 1.4.0ではRF-DETRが通らない条件の背後に置かれていたため、カーネルが実行されませんでした。RF-DETRやほかの変形可能アテンションファミリーでは、浮動小数点の許容範囲内で予測が変わる可能性があります。標準インストールには影響せず、`LIBREYOLO_HUB_KERNELS=0`で無効にできます。
- **`libreyolo predict`は未対応のオプションで例外を発生させず、破棄します。** CLIはモデルの`__call__`シグネチャに照らしてキーワード引数をフィルタリングするため、ファミリーが受け付けないオプションは`TypeError`を発生させず無視されます。フラグ名のタイプミスも通知なく無視されるようになりました。
- **ライブソースではJSON出力の形が変わります。** Webカメラ、RTSPストリーム、画面キャプチャは暗黙的にストリーミングを有効にし、呼び出し全体で1レコードではなくフレームごとに1レコードを出力します。これらの[入力ソース](/docs/predict/sources)は1.5.0で追加されたため、1.4.0のスクリプトには影響しません。
- **`rfdetr-pose`または`yolonas-pose`をONNXへ再エクスポートすると、出力名が変わります。** 1.4.0では出力数に基づく判定により、複数テンソルの姿勢推定ヘッドをセグメンテーションと誤認していました。ディスク上の既存`.onnx`ファイルは変更されません。
- **PyTorchなしのインストールでは**、結果が`torch.Tensor`ではなくNumPy配列を保持するため、`.boxes.data`が返す型が変わり、NMSの同点処理がtorchvisionと異なる場合があります。PyTorchがインストールされていれば、動作はバイト単位で変わりません。[軽量インストール](/docs/lightweight-install)を参照してください。
- **構成オブジェクトは構築時により多くの検証を行います。** `TrainConfig`には以前存在しなかった`__post_init__`が追加されたため、すでに無効だった構成は実行の途中ではなく即座に例外を発生させます。`ValidationConfig`のシリアライズには`edge_thresholds`キーが追加され、1.4.0のダンプに対する厳密な`ValidationConfig(**dump)`往復変換は機能しません。
- **タスクサフィックス付きファミリーの重みファイル名は異なる方法で解決されます。** `segformer-b0`は`LibreSegformerb0-sem.pt`へ解決されるようになりました。これにより自動ダウンロードの404が修正されますが、以前のサフィックスなしファイル名をハードコードしたスクリプトは動作しなくなります。
- **pytestマーカー`experimental_backend`は`extended_backend`になりました。** `-m`を指定してテストスイートを実行する場合だけ関係します。

## チェックポイントとデータセット

1.4.0で書き込まれたチェックポイントは変更なしで読み込めます。[スキーマ](/docs/reference/checkpoint-schema)には長方形モデル向けの`imgsz_h`と`imgsz_w`が追加され、古いリーダー向けにスカラーの`imgsz = max(h, w)`も引き続き書き込まれます。[ExecuTorch](/docs/export/executorch)と[MNN](/docs/export/mnn)のエクスポートでは、それぞれ`<program>.pte.json`と`<model>.mnn.json`というサイドカーが必要になりました。HRNetのエクスポートには`pose_input: "person_crop"`が含まれます。データセット形式に変更はありません。
