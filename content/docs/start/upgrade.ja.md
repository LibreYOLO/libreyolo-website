---
title: 1.7.0へのアップグレード
seo_title: LibreYOLO 1.6.0から1.7.0へのアップグレード
description: LibreYOLO 1.7.0のYOLO9チェックポイント、モジュール名と学習レシピ、CLIの実行フォルダー、混同行列、新しいエラーの移行手順を説明します。
lead: >-
  バージョン1.7.0でも、以前のYOLO9チェックポイントは同じ浮動小数点のボックスで読み込めます。一方で、YOLO9ファミリーの学習、ネットワーク内部でコードから見える名前、検証と推論のいくつかの契約が変わります。
keywords:
  - LibreYOLO アップグレード
  - LibreYOLO 1.6.0 移行
  - LibreYOLO 破壊的変更
  - YOLO9 モジュール名
  - clip_max_norm
  - YOLO 混同行列
last_verified: 1.7.0
source_hash: 895bcd7425f0052f
---

## 1.6.0から1.7.0

- 1.7.0で保存したYOLO9、YOLO9-E2E、YOLO9-P2のチェックポイントは新しいテンソル名を使うため、1.6.xでは読み込めません。1.6.xはYOLO9とYOLO9-P2のチェックポイントを欠落キーの警告付きで読み込んでボックスを1つも推論せず、YOLO9-E2Eのチェックポイントには`model_family`の不一致エラーを送出します。これらを読み込むすべての環境をアップグレードしてください。以前のリリースが書き出した浮動小数点、量子化済み、学習用のチェックポイントは1.7.0で読み込め、ONNXエクスポートには影響しません。

- YOLO9ファミリーのモデルの`model.model`に直接アクセスするコードからは、MultimediaTechLab/YOLOのモジュール名がエイリアスなしで見えます。`DDetect`は`YOLO9Head`（`anchor_convs` / `class_convs`）、`RepConvN`は`RepConv`、`RepNBottleneck`は`Bottleneck`、`DFL`は`Anchor2Vec`（`anchor2vec`）、`nc`は`num_classes`になり、ブロックのサブレイヤーは`conv1`/`conv2`/`bottleneck`になりました。YOLO9-E2Eでは、`YOLO9E2EDetect`は`YOLO9E2EHead`に、`one2one_cv2` / `one2one_cv3`は`one_to_one_anchor_convs` / `one_to_one_class_convs`になりました。`LibreYOLO9Model.fuse()`、`RepConvN.fuse_convs()`、`Conv.forward_without_bn()`、`Concat`、`yolo9.utils.decode_boxes`は削除されました。すべての名前変更は[変更履歴](/docs/changelog)に掲載されています。

- ONNXのノード名は、ヘッド、バックボーン、ネックで新しいモジュール名に従います。たとえば`/head/anchor_convs.0/...`のようになります。Hailoのエンドノード設定など、ノード名をキーにするツールを更新してください。モデルの入力、出力、出力名は変わりません。

- PGIテンソルを持たない重み（公開済みの`LibreYOLO9{t,s,m,c}.pt`を含む）からのYOLO9のファインチューニングは、1.5.0と同様に主ヘッドだけを学習します。1.6.0は`aux_weight=0.25`でランダムに初期化したPGI分岐を付けていました。PGIテンソルを含む重みと、ゼロからの実行は、引き続きPGI付きで学習します。それでも新しい分岐を付けるには`aux_weight=0.25`を、強制的に無効にするには`aux_weight=0`を渡してください。

- PGIを付けた場合、YOLO9-mとYOLO9-cはMultimediaTechLabの補助分岐を構築します。80クラスでの学習時のみのパラメータは12.8M（m）と25.7M（c）で、以前の7.5Mと10.2Mから増えるため、これらのサイズのPGI学習ではステップあたりのメモリと時間が増えます。1.6.0のm/cの学習チェックポイントは、学習時の分岐のまま再開とファインチューニングができます。推論とYOLO9-t/sは変わりません。

- YOLO9、YOLO9-E2E、YOLO9-P2は勾配のL2ノルムを10でクリップし、学習用アサイナーはアンカーをfp32でスコアリングするため、fp16のAMPで選ばれる正例が変わる場合があります。1.6.0のクリッピングにするには`clip_max_norm=0`を渡してください。1.6.0で開始して1.7.0で再開した実行も、この引数を渡さない限りクリップします。fp32の割り当てを切り替える設定はありません。

- 0以外の`perspective`は、共有アフィン変換を使うすべてのファミリー（YOLO9とそのE2E・P2バリアント、YOLOX、YOLOv7、YOLO-NAS、PP-YOLOE、RTMDet、PicoDet、RT-DETR、RT-DETRv2）で、これを設定した1.6.0のチェックポイントから再開した実行も含め、コーナーの変位によって変形します。`0.002`以上の値は同じ最大変形になり、負の値は`ValueError`を送出します。1.6.0の変形を再現する設定はありません。デフォルトの`perspective=0`は変わりません。

- YOLO9のモザイクパイプラインは、mixupの比率をBeta(32, 32)ではなくBeta(1, 1)から抽出し、以前の値に戻す設定はありません。YOLOv7はデフォルトでmixupを有効にするため、YOLOv7のデフォルトの学習が変わります。YOLO9、YOLO9-E2E、YOLO9-P2、RT-DETR、RT-DETRv2は、mixupを有効にした場合だけ変わります。

- `name=`なしの`libreyolo train`は、EC、D-FINE、SegFormer、LingBot-Visionの実行を`runs/train/exp`ではなく`runs/train/ec_exp`、`dfine_exp`、`segformer_exp`、`lingbotvision_exp`に、V-JEPA 2のプローブを`runs/train/train`に、YOLO9-P2の実行を`runs/train/yolo9_exp`に書き込みます。以前のパスにするには`name=exp`を、YOLO9-P2では`name=yolo9_p2_exp`を渡してください。

- `libreyolo train`は、D-FINE、SegFormer、LingBot-Visionをそれぞれ独自のレシピで開始します。D-FINEでは132エポック、`lr0=2e-4`、`weight_decay=1e-4`、2エポックのウォームアップで、モザイク、mixup、HSV、アフィンのデータ拡張は使いません。1.6.0が送っていた値を送るには、`epochs=300 optimizer=sgd lr0=0.01 weight_decay=5e-4 warmup_epochs=5 mosaic=1.0 mixup=1.0 hsv_prob=1.0 degrees=10 translate=0.1 shear=2.0`を渡してください。

- `libreyolo.validation.ConfusionMatrix`は`matrix[predicted, true]`でインデックスし（以前は`[actual, predicted]`）、`conf_thres=0.25`で検出を数え（以前は0.15）、`names=`と`task=`をキーワード引数としてのみ受け取り、その`matrix`は再代入できなくなりました。以前の行列をインデックスしていたコードは転置し、以前の件数を得るには`conf_thres=0.15`を渡してください。`plots=True`は件数を`confusion_matrix.png`に、正規化したプロットを`confusion_matrix_normalized.png`に書き出します。

- `predict(agnostic_nms=True)`は警告を出す代わりに、クラスをまたいで重なるボックスを抑制します。ボックスを持たないタスクは`ValueError`を、`LibreEnsemble`とHRNetは`NotImplementedError`を送出し、SAM系のモデルはこのキーワードを`TypeError`で拒否します。これらの呼び出しからは引数を削除してください。

- torchvisionのモードである`LANCZOS`、`BOX`、`HAMMING`、`NEAREST_EXACT`を含め、分類器の未知の`interpolation`は`ValueError`を送出します。`bilinear`、`bicubic`、`nearest`、`bilinear_no_antialias`のいずれかを使ってください。

- 複数クラスで学習したモデルでの`val(single_cls=True)`、ボックスだけの行とポリゴンの行が混在するセグメンテーションデータセット、デフォルトで画像あたり最大`max_det=300`個のマスクを保持するRTMDet-Insについては、1.6.0の数値と比較する前に検証をやり直してください。MMDetectionの設定にするには`max_det=100`を渡してください。

- `libreyolo[rfdetr]`がない場合、RF-DETRまたはDINOv2のモデル名を指定したCLI呼び出しは、`model_not_found`ではなくJSONエラーコード`model_load_failed`で終了し、メッセージには`pip install libreyolo[rfdetr]`が示されます。

リリース全体は[変更履歴](/docs/changelog)を参照してください。

## 1.5.0から1.6.0

- ONNX Runtimeを1.18未満に固定した環境では、ONNX追加パッケージをインストールする前に`onnxruntime>=1.18.0`へ更新してください。

- SAM 3D Bodyは、確認済みのスナップショットアセット一式を受け付けます。`libreyolo[hf]`をインストールしてアクセス制限付きモデルへの権限を取得し、自動取得を使ってください。または、変更していないスナップショットディレクトリと固定されたMHRアセットを、信頼できるローカルファイルシステム上で指定してください。

- RF-DETRの姿勢推定以外のリサイズと分類器の正規化の修正に伴い、検証をやり直してください。1.5.0の結果と比較する前に、デプロイ済みの信頼度しきい値を再確認してください。従来の前処理を使うフラグはありません。

- D-FINE、DEIM、RT-DETRv4、YOLO-NASの物体検出は、デフォルトでFP16のAMPを有効にします。FP32を維持するには`amp=False`を渡してください。

- 以前のYOLO9のファインチューニング設定を使うには、`aux_weight=0`、`max_labels=100`、`warmup_momentum=0.937`を設定してください。centerが記録された新しい変換済みモデルで従来の配置を再現する場合は、`letterbox_pad="topleft"`を設定してください。古い単一ヘッドのチェックポイントは、元のグラフで再開します。新しいYOLO9の実行はより多くのGPUメモリを使うため、1.5.0のバッチサイズでメモリ不足になる場合は`batch`を下げてください。

- RF-DETRとDINOv2は、ファミリー名を含む連番付きの実行ディレクトリを作成します。ファイルパスを使う処理を更新するか、`output_dir="runs/train", exist_ok=True`を設定して、従来の保存場所と再利用の動作を維持してください。

- 分類では`mixup + cutmix <= 1`を守ってください。不正な組み合わせは、設定時に例外を送出するようになりました。

- QATはEMA、SyncBatchNorm、チェックポイントの平均化を無効にします。これらの状態に依存せず、QATのbestまたはlastチェックポイントを使ってください。

- データセットを変更するフックを持つ独自ローダーでは、`persistent_workers=False`を使うか、変更後にワーカーを再構築してください。互換性のない永続的な複数ワーカーのコピーは、例外を送出するようになりました。

- NumPyの画像配列は、OpenCVの順序であるBGRとして読み取られます。`np.asarray(pil_image)`のようなRGB配列を渡す箇所では`color_format="rgb"`を渡してください。`cv2.imread()`の出力と動画フレームは変更不要です。

- 4DのNumPy配列またはテンソルはバッチとして扱われます。`predict()`は最初の画像だけを使うのではなく、画像ごとに1つの`Results`を含むリストを返します。

- `train(resume=True)`と`train(resume="path/to/last.pt")`は、その実行で保存された学習引数を復元し、その実行ディレクトリに引き続き書き込みます。明示的に渡した引数が優先されます。公開済みの重みや、すでに`epochs`に達した実行を再開すると`ValueError`を送出します。

- 検証が有効な場合は最終エポックで必ず検証するため、`eval_interval`より短い実行でも指標を報告し、`best.pt`を書き出すようになりました。`val=False`は最終エポックも含めて検証を無効にします。学習中の検証は`runs/val/`ではなく`<run>/val`に書き込みます。

- CLIの`train`の失敗はエラーの種類を報告するため、設定エラーは`io_error`付きのコード1ではなく、コード2で終了します。

リリース全体は[変更履歴](/docs/changelog)、チェックポイントの変換は[重みのインポート](/docs/migrate)を参照してください。

## 1.4.0から1.5.0

このページではLibreYOLO自体のアップグレードについて説明します。アップストリームプロジェクトのチェックポイントを読み込む方法を探している場合は、別のトピックである[既存の重みのインポート](/docs/migrate)を参照してください。

このリリースの完全な記録は[変更履歴](/docs/changelog)にあります。以下では、利用者側の対応が必要な部分だけを説明します。

### 必須のコード変更

#### `allow_experimental=True`は廃止されました

確認用ゲートと、その背後にあった`ddp_aware(experimental_key=...)`の仕組みが廃止されました。以前はEC、RTMDet、PicoDet、FOMOの学習とエクスポートでこの引数が必要だったため、これらのファミリーを学習するスクリプトが影響を受けます。

```python
# 1.4.0
model.train(data="data.yaml", epochs=100, allow_experimental=True)

# 1.5.0: この引数を削除
model.train(data="data.yaml", epochs=100)
```

非推奨互換レイヤーはありません。引数を渡し続ける呼び出しでは`Unknown training config keys (ignored)`という警告が出て、その引数は効果を持ちません。`BaseModel.EXPERIMENTAL_WEIGHT_FILENAMES`もあわせて削除されました。`get_download_notice()`フックは残っており、MiDaS、SegFormer、YOLO9-P2で引き続きオーバーライドされています。

サポートレベルは今も公開されていますが、引数ではなくなりました。[安定性レベル](/docs/reference/stability-tiers)を参照してください。

#### エクスポートレベル`"experimental"`は廃止されました

```python
from libreyolo.export.support import Tier

# 1.4.0: Literal["validated", "experimental", "blocked"]
# 1.5.0: Literal["validated", "available", "blocked"]
```

レベル文字列で分岐するコードは、`"experimental"`を読んでいた箇所で`"available"`を読むように変更してください。`BaseExporter`はこれらの形式に対して`RuntimeWarning`を発行しなくなりました。形式ごとの状態は[エクスポートマトリックス](/docs/reference/export-matrix)に掲載されています。

#### `pretrained=False`と`resume`の併用は拒否されます

以前は矛盾した状態のまま処理が進んでいました。現在は次のエラーが発生します。

```
ValueError: pretrained=False cannot be combined with resume.
```

どちらか一方を選んでください。`pretrained=False`は新しいシード済み初期値から開始します。1.5.0では3つだけでなく、学習可能なすべてのファミリーで動作します。`resume`は中断した実行をチェックポイントから再開します。どちらも[学習](/docs/train)に記載されています。

#### CLIの`--imgsz`は整数ではなく文字列です

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

### 変化する数値

デフォルト設定の指標値が変わる変更は3つあります。バージョンをまたいで結果を追跡する場合は、1.5.0の実行結果と1.4.0を比較する前に確認してください。

#### faster-coco-evalがデフォルトのCOCO指標バックエンドになりました

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

#### 1.5.0より前に学習したYOLOXチェックポイントにはepsのオーバーライドが必要です

これは今回のリリースで注意が必要な点です。[YOLOX](/docs/models/yolox)をファインチューニングしている場合は確認してください。

YOLOXはBatchNormの`eps=1e-3`と`momentum=0.03`を指定します。1.5.0より前は、これらの値が後処理の修正として適用されており、データセットの`nc`がチェックポイントと異なるときに`train()`が行うクラス数の再構築では維持されませんでした。そのようなファインチューニングでは、PyTorchのデフォルト`eps=1e-5`で学習され、学習中の検証も行われた後、推論時には`1e-3`で再読み込みされていました。同じテンソルに異なる正規化が適用されていたことになります。

通常の畳み込みを使うサイズでは変化はごくわずかです。Depthwiseの`n`では、チャンネルごとの`running_var`が小さくepsの影響が支配的になるため、大きく変化します。RF100-VLの`ball`では、同じnanoチェックポイントを学習時のepsで評価するとmAP50-95は**0.566**ですが、通常どおり再読み込みすると**0.151**になります。

1.5.0より前に学習したチェックポイントはeps=1e-5の意味を持ちます。正しい数値を報告するには、BNのepsを1e-5へオーバーライドして評価します。

```python
import torch
from libreyolo import LibreYOLO

model = LibreYOLO("my-yolox-finetune.pt")
for module in model.model.modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        module.eps = 1e-5

model.val(data="data.yaml")
```

または、`sqrt((var + 1e-3) / (var + 1e-5))`をBNの重みに一度組み込んで結果を保存してください。1.5.0以降で学習したチェックポイントでは、どちらも不要です。

#### D-FINEのマルチスケール学習はアップストリームのサイズ別レシピを使用します

以前はすべてのサイズで`base_size_repeat`が3に固定されていました。現在はアップストリームの指定に従い、サイズごとに解決されます。**n**は固定サイズでマルチスケール無効、**s**は20、**m**は6、**l**は4、**x**は3です。以前と一致するのはxだけなので、n、s、m、lではスケール分布が変わり、収束する指標値も変わります。

以前の動作へ戻すには明示的に設定します。

```python
from libreyolo.training.config import DFINEConfig

config = DFINEConfig(base_size_repeat=3)
```

DEIMは引き続き3に固定されています。ファミリーの詳細は[D-FINE](/docs/models/d-fine)にあります。

### 知っておくべき変更（対応不要）

- **長方形の`imgsz`で結果が変わるのは、以前の結果が誤っていたためです。** ボックス座標、RTMDetのマスクサイズ変更、YOLO-NASの再スケーリング、バリデーターの正解データスケーリングで、単一のスカラーではなく高さと幅を軸ごとに使用するようになりました。正方形の`imgsz`はビット単位で変わりません。1.4.0での長方形の推論または検証は誤ってスケーリングされていました。YOLO-NASは誤った出力を通知なく生成せず、長方形の`imgsz`を明示的に拒否するようになりました。
- **指標辞書にキーが追加されました。** COCO評価器から`max_det`、`ar_max_det`、`AR_max_det`が、FOMOから`metrics/loss`と`metrics/loss/ce`が追加されました。デフォルトでの値は変わりませんが、カスタム[ロガー](/docs/train/loggers)やCSVヘッダーなど、指標キーを反復処理するものには新しい列が表示されます。
- **ヘッド再構築が発生するシード付きYOLO9実行**では、再構築後ではなく再構築前にシードを適用するようになったため、異なる初期値から始まります。異なるクラス数へファインチューニングしたシード付き1.4.0実行を、1.5.0でビット単位に再現することはできません。
- **CUDA上の`libreyolo[hub-kernels]`で、ネイティブのMS-deform-attnカーネルが実際に使われるようになりました。** 1.4.0ではRF-DETRが通らない条件の背後に置かれていたため、カーネルが実行されませんでした。RF-DETRやほかの変形可能アテンションファミリーでは、浮動小数点の許容範囲内で予測が変わる可能性があります。標準インストールには影響せず、`LIBREYOLO_HUB_KERNELS=0`で無効にできます。
- **`libreyolo predict`は未対応のオプションで例外を発生させず、破棄します。** CLIはモデルの`__call__`シグネチャに照らしてキーワード引数をフィルタリングするため、ファミリーが受け付けないオプションは`TypeError`を発生させず無視されます。未知のフラグ名は引き続き`No such option`で拒否されます。
- **ライブソースではJSON出力の形が変わります。** Webカメラ、RTSPストリーム、画面キャプチャは暗黙的にストリーミングを有効にし、呼び出し全体で1レコードではなくフレームごとに1レコードを出力します。これらの[入力ソース](/docs/predict/sources)は1.5.0で追加されたため、1.4.0のスクリプトには影響しません。
- **`rfdetr-pose`または`yolonas-pose`をONNXへ再エクスポートすると、出力名が変わります。** 1.4.0では出力数に基づく判定により、複数テンソルの姿勢推定ヘッドをセグメンテーションと誤認していました。ディスク上の既存`.onnx`ファイルは変更されません。
- **PyTorchなしのインストールでは**、結果が`torch.Tensor`ではなくNumPy配列を保持するため、`.boxes.data`が返す型が変わり、NMSの同点処理がtorchvisionと異なる場合があります。PyTorchがインストールされていれば、動作はバイト単位で変わりません。[軽量インストール](/docs/lightweight-install)を参照してください。
- **構成オブジェクトは構築時により多くの検証を行います。** `TrainConfig`には以前存在しなかった`__post_init__`が追加されたため、すでに無効だった構成は実行の途中ではなく即座に例外を発生させます。`ValidationConfig`のシリアライズには`edge_thresholds`キーが追加され、1.4.0のダンプに対する厳密な`ValidationConfig(**dump)`往復変換は機能しません。
- **タスクサフィックス付きファミリーの重みファイル名は異なる方法で解決されます。** `segformer-b0`は`LibreSegformerb0-sem.pt`へ解決されるようになりました。これにより自動ダウンロードの404が修正されますが、以前のサフィックスなしファイル名をハードコードしたスクリプトは動作しなくなります。
- **pytestマーカー`experimental_backend`は`extended_backend`になりました。** `-m`を指定してテストスイートを実行する場合だけ関係します。

### チェックポイントとデータセット

1.4.0で書き込まれたチェックポイントは変更なしで読み込めます。[スキーマ](/docs/reference/checkpoint-schema)には長方形モデル向けの`imgsz_h`と`imgsz_w`が追加され、古いリーダー向けにスカラーの`imgsz = max(h, w)`も引き続き書き込まれます。[ExecuTorch](/docs/export/executorch)と[MNN](/docs/export/mnn)のエクスポートでは、それぞれ`<program>.pte.json`と`<model>.mnn.json`というサイドカーが必要になりました。HRNetのエクスポートには`pose_input: "person_crop"`が含まれます。データセット形式に変更はありません。
