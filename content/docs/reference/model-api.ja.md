---
title: モデルAPI
seo_title: LibreYOLOモデルオブジェクトのメソッドとシグネチャ
description: >-
  読み込んだLibreYOLOモデルの全メソッド：predict、embed、track、val、train、export、save、quantize、info、CUDAグラフ制御について、実際のデフォルト値とともに説明します。
lead: >-
  読み込んだLibreYOLOモデルはBaseModelのインスタンスです。このページでは、そのインスタンスが持つメソッドを、libreyolo/models/base/model.pyから読み取ったシグネチャとデフォルト値とともに一覧化します。
keywords:
  - libreyolo モデル メソッド
  - libreyolo predict 引数
  - libreyolo val 引数
  - libreyolo export 引数
  - model.track
  - model.quantize
  - capture_graph
last_verified: 1.5.0
verification: >-
  v1.5.0のlibreyolo/models/base/model.pyとlibreyolo/models/base/inference.pyからシグネチャとデフォルト値を確認しました。ファミリークラスでは、これらが制限または拡張される場合があります。train()はファミリーごとに定義されているため、ここでは共有のcfg=ラッパーだけを説明します。
snippets:
  usage:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        model.info()
        result = model(SAMPLE_IMAGE, conf=0.25, iou=0.45)

        print(result.boxes.xyxy)
        print(result.speed)
  stream:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        # stream=Trueはジェネレーターを返し、フレームまたは画像ごとにResultsを1つ生成
        for result in model([SAMPLE_IMAGE, SAMPLE_IMAGE], stream=True):
            print(len(result))
source_hash: da0776970ded8716
---

## 構築

ファクトリーはファミリークラスのインスタンスを返します。そのクラスを直接構築する場合も
同じ引数を取りますが、`size`は必須です。

```python
Family(model_path, size, nb_classes=80, device="auto", task=None, **kwargs)
```

`device="auto"`は、利用できる場合はCUDA、次にMPS、最後にCPUを選択します。整数または
数字の文字列はCUDAの序数として読み取られるため、`device=0`と`device="0"`はいずれも
`cuda:0`を意味します。`task`はファミリーの`SUPPORTED_TASKS`に照らして検証されます。
`model_path=None`を渡すとアーキテクチャを構築して学習モードのままにし、`dict`を渡すと
そのstate dictを直接読み込みます。

## predictと\_\_call\_\_

`predict`は`__call__`の別名です。

```python
model(
    source=None,
    *,
    conf=0.25,
    iou=0.45,
    imgsz=None,
    device=None,
    classes=None,
    max_det=300,
    augment=False,
    save=False,
    batch=1,
    stream=False,
    stream_buffer=False,
    vid_stride=1,
    show=False,
    output_path=None,
    color_format="auto",
    tiling=False,
    overlap_ratio=0.2,
    output_file_format=None,
    cuda_graph=False,
    **kwargs,
)
```

| 引数 | デフォルト | 意味 |
|---|---|---|
| `source` | `None` | 画像、メモリ内画像のリストまたはタプル、ディレクトリ、動画ファイル、または`"screen"`、`"screen 1"`、`"screen 1 100 200 512 256"`などの画面ソース |
| `conf` | `0.25` | 信頼度のしきい値 |
| `iou` | `0.45` | NMSのIoUしきい値 |
| `imgsz` | `None` | 入力サイズの上書き。`None`はモデル固有のサイズを使用 |
| `device` | `None` | この呼び出しで使用するデバイスの上書き |
| `classes` | `None` | 指定したクラスIDのみを保持 |
| `max_det` | `300` | 画像あたりの最大検出数 |
| `augment` | `False` | テスト時データ拡張 |
| `save` | `False` | アノテーション付き画像または動画を書き出す |
| `batch` | `1` | ディレクトリおよびリストのソースで、順伝播1回あたりの画像数 |
| `stream` | `False` | 実体化したリストの代わりにジェネレーターを返す |
| `stream_buffer` | `False` | 取り込んだライブフレームを最新のものだけでなくすべて保持 |
| `vid_stride` | `1` | 動画または画面のN番目ごとのフレームを処理 |
| `show` | `False` | アノテーション付きフレームをウィンドウに表示 |
| `output_path` | `None` | `save=True`の場合の出力パス |
| `color_format` | `"auto"` | メモリ内配列のカラーフォーマット指定 |
| `tiling` | `False` | 大きな画像に対するタイル分割推論 |
| `overlap_ratio` | `0.2` | タイルの重複率 |
| `output_file_format` | `None` | `"jpg"`、`"png"`、または`"webp"` |
| `cuda_graph` | `False` | `True`は入力形状ごとの初回使用時にキャプチャし、`"auto"`は同じ形状が繰り返されるまで待機 |

単一の画像ソースは1つの`Results`を返します。リスト、タプル、またはディレクトリは
それらのリストを返し、`stream=True`はどの場合もジェネレーターを返します。

ライブストリームのソースには終端がなく、`stream=True`が必要です。`tiling`と`augment`は
同時に使用できません。テスト時データ拡張は、`embed`、`point`、`edge`タスクでは例外を
発生させます。

<code-tabs name="usage" />

`batch > 1`では、`SUPPORTS_BATCHED_PREDICT`がtrueのファミリーはチャンクごとに
スタックした順伝播を1回実行します。`batch=1`では画像ごとに順伝播を1回実行します。

<code-tabs name="stream" />

## embed

```python
model.embed(source=None, **kwargs) -> torch.Tensor
```

`predict`の便利なラッパーで、すべての埋め込みベクトルの行を1つの`(N_total, D)`テンソルに
スタックします。モデルは`task="embed"`で構築されている必要があり、それ以外の場合は
`NotImplementedError`を発生させます。

## track

```python
model.track(
    source,
    *,
    track_conf=0.25,
    iou=0.45,
    imgsz=None,
    classes=None,
    max_det=300,
    save=False,
    show=False,
    vid_stride=1,
    output_path=None,
    tracker="bytetrack",
    tracker_config=None,
    augment=False,
    **tracker_kwargs,
) -> Generator[Results, None, None]
```

`track_id`を設定した`Results`をフレームごとに1つ生成します。`tracker`には
`"bytetrack"`、`"botsort"`、`"ocsort"`、または`"deepocsort"`を指定できます。
`tracker_config`を指定した場合は設定の型がトラッカーを選択するため、`tracker`は無視されます。
`track_conf`は、ByteTrackとBoT-SORTでは`track_high_thresh`に、OC-SORTとDeep OC-SORTでは
`det_thresh`に対応します。`output_path`のデフォルトは`runs/track/<video_stem>.mp4`です。

## val

```python
model.val(
    data=None,
    batch=16,
    imgsz=None,
    conf=0.001,
    iou=0.6,
    workers=4,
    allow_download_scripts=False,
    device=None,
    split="val",
    augment=False,
    save_json=False,
    verbose=True,
    *,
    plots=None,
    **kwargs,
) -> Dict
```

タスクに応じたキーを持つ指標辞書を返します。検出では`metrics/precision`、
`metrics/recall`、`metrics/mAP50`、`metrics/mAP50-95`を返します。`imgsz`は正方形の整数値
または`(height, width)`タプルを受け取り、デフォルトではモデル固有の入力サイズを使います。
`plots`は`save_plots`の別名です。`allow_download_scripts`は、データセットYAMLの`download`
フィールドに含まれる可能性がある埋め込みPythonの実行可否を制御します。

`faster_coco_eval`は`**kwargs`を通じて受け入れられ、デフォルトは`True`です。パッケージが
インストールされていない場合はpycocotoolsへフォールバックします。実行されたバックエンドは
`model.last_eval_backend`で報告されます。

拡張検証は、`obb`と`pose`タスクでは例外を発生させます。

## train

`train`はファミリーごとに定義されているため、引数が異なります。基底クラスが各ファミリーの
`train`をラップするため、2つの動作は共通です。

- `cfg=`は、キーが呼び出しへマージされるYAMLパスを受け取ります。明示的なキーワード引数がファイルより優先されます。
- カバレッジグループ`g0`または`g1`のファミリーで`pretrained=False`を指定すると、学習前にモデルを最初から再初期化します。`resume=True`とは併用できません。

ファミリーが実際にどのデータ拡張設定を尊重するかは、ファミリーごとに異なります。
[データ拡張マトリックス](/docs/reference/augmentation-matrix)を参照してください。

## export

```python
model.export(format="onnx", **kwargs) -> str
```

書き出された成果物のパスを返します。`format`はエクスポーターレジストリによって解決されます。
ここでは`engine`が`tensorrt`の別名、`litert`が`tflite`の別名です。すべてのエクスポーターに
共通する引数は次のとおりです。

| 引数 | デフォルト | 意味 |
|---|---|---|
| `output_path` | `None` | 出力ファイルのパス。省略時は`weights/`以下に生成 |
| `imgsz` | `None` | `(height, width)`タプルまたは単一の整数。デフォルトは固有サイズ |
| `opset` | `None` | ONNX opsetバージョン |
| `simplify` | `True` | ONNXグラフの簡略化を実行 |
| `dynamic` | `True` | 動的軸を有効化 |
| `half` | `False` | FP16精度 |
| `int8` | `False` | INT8精度 |
| `batch` | `1` | 成果物に組み込むバッチサイズ |
| `device` | `None` | トレースに使うデバイス |
| `data` | `None` | INT8キャリブレーション用のdata.yaml |
| `fraction` | `1.0` | 使用するキャリブレーションデータセットの割合 |
| `allow_download_scripts` | `False` | データセットYAMLのダウンロード処理に埋め込まれたPythonを許可 |
| `verbose` | `False` | エクスポーターの詳細ログ |

禁止された組み合わせは、トレース前の事前検査で`NotImplementedError`を発生させます。
カバレッジとその規則は[エクスポートマトリックス](/docs/reference/export-matrix)のページに
あります。有効なLoRAアダプターが存在する場合は密な重みに統合されます。この統合は、すべての
要求拒否を確認した後にのみ行われます。

## save

```python
model.save(path) -> str
```

スキーマv1.0のLibreYOLOチェックポイントを書き出します。これはstate dictと、
[チェックポイントスキーマ](/docs/reference/checkpoint-schema)で説明されているメタデータです。
量子化モデルには`quant`マニフェストも含まれるため、`LibreYOLO(path)`は量子化された構造と
スケールを復元します。

## quantize、quant_info、dequantize

```python
model.quantize(
    recipe,
    calib="coco128.yaml",
    samples=128,
    batch=8,
    algorithm="auto",
    keep_high_precision=None,
    allow_download_scripts=False,
    verbose=True,
)
```

その場で量子化し、モデルを返します。`recipe`にはキャストの`fp16`と`bf16`、ConvとLinearの
レシピ`int8`と`fp8`、またはLinear専用のレシピ`w4a16`、`w4a8`、`nvfp4`、`mxfp4`、`int2`の
いずれかを指定します。後者はRF-DETRなどのTransformerファミリーでサポートされます。
`int2`にはQATが必要です。`calib`はdata.yamlのパスまたは組み込みデータセット名を受け取り、
順伝播専用で画像を読み込みます。ラベルは一切読みません。キャリブレーションを省略するには
`calib=None`を渡します。`algorithm`は`"minmax"`、`"percentile"`、または`"auto"`です。

`model.quant_info()`は量子化状態の概要を返し、浮動小数点モデルでは`None`を返します。
`model.dequantize()`は、量子化学習済みのマスター重みを維持しながら、その場で浮動小数点
モジュールを復元します。これはQATから`export(format="onnx", int8=True, data=...)`への
橋渡しになります。

## infoとlayers

```python
model.info(detailed=False, verbose=True) -> Dict[str, Any]
model.get_available_layer_names() -> List[str]
model.get_distill_config() -> Dict
```

`info`はJSON互換の辞書を返し、`verbose`がtrueの場合は人間が読める概要をログへ記録します。
`get_available_layer_names`は、蒸留または特徴量抽出の設定で名前を指定できる層を一覧化します。

## CUDAグラフ

`SUPPORTS_CUDA_GRAPH`クラス属性がtrueのファミリーで利用できます。リプレイはeager実行と
ビット単位で同一です。

```python
model.capture_graph(imgsz=None, batch=1, dtype=None) -> None
model.cuda_graph_scope(mode=True)          # コンテキストマネージャー
model.graph_info() -> Dict[str, Any]
model.release_graphs() -> None
```

キャプチャしたグラフは、キャプチャ時と完全に同じ形状でのみ有効です。そのため、`batch`と
`imgsz`は後続の`predict`呼び出しと一致する必要があります。`capture_graph`はキャプチャの
コストを最初の要求から切り離します。`mode`には、初回使用時にキャプチャする`True`または
`"on"`、形状が繰り返されるまで待つ`"auto"`、何もしない`False`を指定できます。
ファミリーがオプトインしていない場合、`capture_graph`は`NotImplementedError`を発生させ、
キャプチャに失敗した場合は`CudaGraphUnavailable`を発生させます。

## デバイスとdtype

`Results`オブジェクトには`.to()`、`.cpu()`、`.cuda()`、`.numpy()`があります。
[Resultsの型](/docs/reference/results-types)を参照してください。モデル自体を移動するには、
`predict`または構築時に`device=`を渡します。
