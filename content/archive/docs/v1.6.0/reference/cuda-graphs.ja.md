---
title: CUDAグラフ
seo_title: LibreYOLO CUDAグラフ対応マトリクス
description: >-
  推論時に順伝播をキャプチャするファミリーと、学習時に順伝播と逆伝播をキャプチャするファミリーを示し、数値の保証、キャプチャの分割位置、非対応ファミリーで例外が発生する理由を説明します。
lead: >-
  CUDAグラフは固定されたカーネル列の1回の実行を記録し、単一の起動として再生します。LibreYOLOは検証済み39ファミリーで推論、24ファミリーで学習をキャプチャします。常にファミリー単位で、ビット単位の一致確認後にだけ有効化され、暗黙のフォールバックは行いません。
keywords:
  - libreyolo cuda graph
  - cuda_graph=True
  - cuda graph 対応マトリクス
  - torch cuda graph 学習
  - capture_error_mode thread_local
  - cuda graph ビット一致
last_verified: 1.5.0
verification: >-
  推論ファミリー一覧はv1.5.0のtests/e2e/test_cuda_graph_families.pyにあるCAPTURABLEマトリクスから導出しました。学習ファミリー一覧、一致クラス、時間はdocs/training_cuda_graphs.mdで確認しました。APIとNotImplementedErrorはlibreyolo/models/base/model.pyのBaseModel._require_cuda_graph_support、cuda_graph_scope、capture_graph、およびSUPPORTS_CUDA_GRAPHクラス変数で確認しました。シーム分割はdepth_anything3、birefnet、ppocr、sam、sensenovaファミリーの_get_graph_runner上書きとlibreyolo/models/base/detr_cuda_graph.pyで確認しました。capture_error_modeはlibreyolo/models/base/cuda_graph.pyとlibreyolo/training/cuda_graph.pyで確認しました。学習のフォールバックはlibreyolo/training/trainer.py、--cuda-graphフラグはlibreyolo/cli/commands/train.pyで確認しました。
meta:
  - label: 推論ファミリー
    value: '39'
  - label: 学習ファミリー
    value: '24'
  - label: 推論フラグ
    value: predict(cuda_graph=True)
    mono: true
  - label: 学習フラグ
    value: train(cuda_graph=True)
    mono: true
snippets:
  usage:
    - label: 推論
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        # True は入力形状ごとの初回使用時にキャプチャ
        # "auto" は形状が繰り返されてからキャプチャコストを負担
        result = model(SAMPLE_IMAGE, cuda_graph=True)
    - label: 学習
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")
        model.train(data="my-dataset.yaml", epochs=100, cuda_graph=True)
    - label: CLIから学習
      language: bash
      code: |
        libreyolo train model=LibreYOLO9t.pt data=my-dataset.yaml \
          epochs=100 --cuda-graph
source_hash: 67c46199939278f2
---

## キャプチャされるもの

グラフは固定されたカーネル列と、その読み書き先のメモリアドレスを記録します。値、形状、制御フローは記録しません。数百回の起動ではなく1回の起動で再生するため、演算ではなく起動オーバーヘッドがステップの大部分を占める、小さいバッチサイズの小型ネットワークで最大の効果があります。

2つのエントリーポイントでは、キャプチャする処理量が異なります。

| | グラフ内 | eager実行 |
|---|---|---|
| 推論 | ネットワークの順伝播、`model._forward(x)` | 前処理、NMS、すべての後処理 |
| 学習 | ネットワークの順伝播と逆伝播 | 損失、オプティマイザーのステップ、勾配クリップ、EMA、学習率スケジュール |

NMSも検出損失も候補ではありません。どちらもブールマスクで選択し、Hungarian matchingまたはassignerを実行し、結果に応じて分岐します。これはグラフが記録できない処理そのものです。これらを外に保つのは、回避すべき制限ではなくキャプチャを安全にする仕組みです。

<code-tabs name="usage" />

推論時の`cuda_graph`は3つの値を受け付けます。デフォルトは`False`です。`True`は各入力形状を最初に見たときにキャプチャします。`"auto"`は形状が繰り返されるまで待つため、1回限りの処理や形状が変化する処理では、再利用しないキャプチャのコストがかかりません。`capture_graph(imgsz=None, batch=1, dtype=None)`は最初のリクエストからコストを外し、`graph_info()`はキャプチャ済みグラフと再生回数を報告し、`release_graphs()`はそれらを解放します。

学習時のフラグは通常のブール値で、CLIでは`--cuda-graph`です。関連する制御については[推論性能](/docs/predict/performance)と[学習性能](/docs/train/performance)を参照してください。

## 推論の対応状況

対応状況はファミリーごとに`SUPPORTS_CUDA_GRAPH`クラス変数で宣言されます。異なる分布から取得した2つのプローブ入力に対し、キャプチャと再生がビット単位で同じ結果を返した後でのみ、ファミリーが有効化されます。この共通の一致マトリクスは、9つのタスクにわたる39ファミリーを対象とします。

| タスク | ファミリー |
|---|---|
| detect | yolo1、yolo2、yolo3、yolo4、yolo9、yolo9_p2、yolo9_e2e、yolox、yolo7、yolonas、picodet、rtmdet、dfine、deim、deimv2、rtdetr、rtdetrv2、rtdetrv4、rfdetr、ec |
| segment | dfine、rtmdet、rfdetr、ec |
| pose | ec、yolonas、rfdetr |
| point | fomo |
| classify | resnet、convnext、mobilenetv4、efficientnetv2、clip、dinov2、siglip2 |
| semantic | eomt、dinov2、segformer、pidnet、lingbotvision |
| depth | depth_anything、depth_anything3、zipdepth |
| restore | nafnet、realesrgan、swinir |
| matte | birefnet |

複数のタスクに現れるファミリーがあるため、マトリクスの行数は異なるファミリー数を上回ります。さらに3つのファミリーが共通マトリクスではなく、専用テストを持つファミリー固有のコード経路でキャプチャし、39には含まれません。PP-OCR、SAM、SenseNovaです。

検証は近似ではなくビット単位です。以前のプロトコルは相対的な大きさで一致を判定し、正常な3ファミリー、YOLOX、EfficientNetV2、YOLOv7を誤って降格させました。eager実行とグラフの差は約`1e-7`と測定されましたが、重要なプローブではビット単位で同一でした。

## 学習の対応状況

このリリースで、学習キャプチャは2ファミリーから5タスクにわたる24ファミリーへ増えました。

| タスク | ファミリー |
|---|---|
| detect | yolo9、yolo9_p2、yolo9_e2e、yolox、yolo7、yolonas、picodet、rtmdet、rfdetr、dfine、deim、deimv2、rtdetr、rtdetrv2、rtdetrv4、ec |
| classify | resnet、convnext、mobilenetv4、efficientnetv2 |
| semantic | segformer、lingbotvision |
| point | fomo |
| restore | nafnet |

そのほかはすべてeager実行で学習します。同じファミリーのそのほかのタスク、一覧にないファミリー、分散実行、蒸留実行が該当します。形状がまだ新しい場合もキャプチャをスキップします。学習経路は入力形状が3回繰り返されるまで待つため、`multi_scale=True`では一度もキャプチャされないことがあります。

## 非対応ファミリーに対する2つの異なる動作

推論経路では例外が発生します。有効化されていないファミリーで`predict(cuda_graph=True)`を使うと、eager実行へ移行して実際には得ていない高速化を得たと思わせる代わりに、ファミリー名を示して`NotImplementedError`を発生させます。不正なキャプチャは明確に失敗するとは限りません。キャプチャできない処理を行う順伝播を再生すると、誤った数値を警告なしで返すため、対応はフォールバック付きの試行ではなくファミリーごとの明示的な表明でなければなりません。

学習経路ではログを記録します。`train(cuda_graph=True)`は常に安全に渡せます。キャプチャできないファミリー、タスク、設定では1行を書き込み、変更せずeager実行で学習します。実行途中でキャプチャに失敗した場合も、処理を中止せず残りをeager実行に切り替えます。この非対称性は意図的です。推論は呼び出し場所で修正できますが、学習実行は任意の最適化が原因で6時間目に停止すべきではありません。

## シーム分割

一部のファミリーでは、ある段階がグラフで記録できない処理を実際に行うため、全体をキャプチャできません。ファミリーを除外する代わりに、検証済みのシームでキャプチャを分割します。キャプチャ可能な部分を再生し、残りをeager実行し、結合後の出力はすべてをeager実行した場合と同じです。

| ファミリー | キャプチャ | eager実行とその理由 |
|---|---|---|
| Depth Anything 3 | ネットワーク | 順伝播後にホスト側から見える処理を行う空のステップ |
| BiRefNet | エンコーダー、`forward_enc` | デコーダー。`deform_conv2d`がキャプチャ下で異なる結果を再生するため |
| PP-OCR | 検出段階、`forward_det` | 行ごとにcrop幅が変わるため認識はeager実行 |
| SAM | 画像エンコーダー | 1回のエンコードに対して何度も実行するプロンプト経路 |
| SenseNova | Vision Tower | ステップごとに増大するKV cacheを持つ自己回帰生成 |
| エンコーダー・デコーダー検出器 | バックボーンとエンコーダー | デコーダーとHungarian criterion |

BiRefNetの分割は特に重要です。キャプチャ下での`deform_conv2d`の誤動作は、モデル外の単独呼び出しでも再現します。純粋なPyTorchの等価処理への置き換えは、eager予測の値も変えるため採用されませんでした。eager実行の数値が規約です。

エンコーダー・デコーダーの場合はD-FINE、DEIM、DEIMv2、RT-DETR、RT-DETRv2、RT-DETRv4、ECを含みます。これらのデコーダーは正解データから対照的ノイズ除去クエリを構築し、その数はバッチ内で最大の正解データ数に由来するため、デコーダーのトークン数がバッチごとに変わります。これはグラフが許容できない変化です。これらのファミリーではバックボーンとエンコーダーがステップのおよそ5分の1から4分の1を占めるため、高速化表の下位に位置します。

PP-OCRは、runnerのキャッシュ上限に従って検出入力形状ごとに1つのグラフをキャプチャし、キャプチャスコープが有効でない場合はeager実行の結果を返します。

## 数値

ほとんどのファミリーはビット単位で一致し、一致しない場合も理由を曖昧にせず明記します。学習のステップ0では24ファミリーすべてで損失がビット単位で同一で、異なるBatchNormバッファもありません。カテゴリを分けるのは勾配の比較です。

| クラス | ファミリー | 意味 |
|---|---|---|
| 完全一致 | 24ファミリーの大半 | すべての勾配がビット単位で同一 |
| 1 ULP | fomo、lingbotvision | 加算順序の違いによるfloat32の最後の1ビット。相対値で約`1e-7` |
| eagerノイズ | DETR系統 | グラフとeager実行の差は2回のeager実行間の差以下 |
| 浮動小数点丸め | rtmdet | 139個の勾配中137個はビット単位で同一、2個は約`3e-4`の差 |
| 固有のRNGストリーム | segformer | stochastic depthがキャプチャ領域内に存在 |

eagerノイズのクラスは正しく理解することが重要です。これらのファミリーではseedを固定した2回のeager実行がすでに一致しないため、ビット単位の同一性はグラフ実行だけが満たせなかった基準ではなく、どの実行も満たせない基準です。これは`amp=False`でさらに広く発生します。fp32重みの勾配で測定された相対値`3.2e-7`の非決定性が蓄積し、seedを固定した2回のeager YOLOv9-t実行は20ステップで36パーセント乖離します。TF32を無効にしても解決しません。

## pin memory

キャプチャは`capture_error_mode="thread_local"`で実行されます。PyTorchのデフォルト`"global"`モードでは、次のバッチを準備するDataLoaderのpin-memoryスレッドが`cudaHostAlloc`を呼び出します。これにより進行中のキャプチャが無効になると同時に、そのスレッド自体も影響を受け、次のバッチ取得時にpin-memoryスレッド内から発生したエラーで実行が停止します。この組み合わせは、診断される前に実際の学習キャンペーンで2回観測されました。

thread-localモードはキャプチャを行うスレッドだけを制限します。pinスレッドはキャプチャストリームに触れないため、その処理は元からグラフに含まれません。学習ではさらに、モードを強制する`torch.cuda.CUDAGraph`サブクラスを一時的に置き換えます。`make_graphed_callables`はモードの引数を公開していないためです。置き換えはロック下で行われ、2つの同時キャプチャが置き換えを残すことはありません。

## 得られる効果

RTX 5070 Ti上でAMPを使い、各armに1プロセス、実際の1バッチを再生してdataloaderを処理外に置き、ウォームアップ後24ステップの最速値を測定しました。検出は640 px、分類は224 pxです。

| ファミリー | バッチ | 高速化 |
|---|---:|---:|
| FOMO s | 16 | 3.63x |
| MobileNetV4 s | 16 | 2.74x |
| EfficientNetV2 b0 | 16 | 2.44x |
| YOLOv9-t | 8 | 1.99x |
| YOLOv9 e2e | 8 | 1.76x |
| YOLOv9 p2 | 8 | 1.49x |
| そのほかすべて | さまざま | 1.04x〜1.26x |

グラフはdataloaderや検証を高速化できないため、実行全体の向上率は小さくなります。406画像で20エポックのYOLOv9-tファインチューニングは428.4秒から367.7秒になり、エンドツーエンドで1.16x向上しました。両方の実行でmAP50-95は同じ0.6394、エポックごとの損失も同一でした。

上限はステップ内でネットワークが占める割合によって決まります。同じハードウェアで640 px、バッチ8の場合、YOLOv9-tでは84パーセントですが、RTMDet-tでは26パーセントだけです。RTMDet-tはステップの大半をラベルassignerに費やします。起動オーバーヘッドはWindowsで最も高いため、Linuxでの向上率はこの表のおよそ3分の1から半分になり、dataloaderがボトルネックの実行では実時間がまったく変わりません。ピークメモリは5パーセント減少から19パーセント増加の範囲で変動します。

## 注意点

グラフは値ではなくアドレスを記録するため、パラメータを移動する操作を行うと破棄されます。`predict(device=...)`によるデバイス変更、量子化、量子化解除はすべてキャプチャ済みグラフを無効にします。

ファミリーよりもバッチサイズの影響が大きく、RT-DETR-r18はバッチ2で1.19x、バッチ8で1.04x向上します。大きいバッチは演算がボトルネックとなり、除去できる起動オーバーヘッドが少ないためです。

推論の一致テスト一式はオプションの`kernels`パッケージをインストールせずに実行されたため、コンパイル済みHubカーネルが有効な状態でのキャプチャの安全性は対象外です。キャプチャ問題を切り分ける際は`LIBREYOLO_HUB_KERNELS=0`を設定してカーネルを除外してください。[カーネル](/docs/reference/kernels)を参照してください。

