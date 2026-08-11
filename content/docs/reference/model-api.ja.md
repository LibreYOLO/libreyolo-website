---
title: モデルAPI
seo_title: LibreYOLOモデルオブジェクトのメソッドとシグネチャ
description: >-
  読み込んだLibreYOLOモデルの全メソッドを説明します。predict、embed、track、val、train、export、save、quantize、info、CUDA
  Graph制御を実際のデフォルト値とともに扱います。
lead: >-
  読み込んだLibreYOLOモデルはBaseModelのインスタンスです。このページでは、libreyolo/models/base/model.pyから確認したシグネチャとデフォルト値を使い、インスタンスが持つメソッドを一覧にします。
keywords:
  - LibreYOLO model メソッド
  - LibreYOLO predict 引数
  - LibreYOLO val 引数
  - LibreYOLO export 引数
  - model.track
  - model.quantize
  - capture_graph
last_verified: 1.5.0
verification: >-
  v1.5.0のlibreyolo/models/base/model.pyとlibreyolo/models/base/inference.pyからシグネチャとデフォルト値を確認しました。ファミリークラスはこれらを限定または拡張する場合があります。train()はファミリーごとに定義されるため、共有のcfg=ラッパーだけをここで説明します。
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

        # stream=Trueはフレームまたは画像ごとに1つのResultsを返すジェネレーターです。
        for result in model([SAMPLE_IMAGE, SAMPLE_IMAGE], stream=True):
            print(len(result))
source_hash: da0776970ded8716
---

## 構築

ファクトリーはファミリークラスのインスタンスを返します。そのクラスを直接構築する場合も同じ引数を受け取りますが、`size`は必須です。

```python
Family(model_path, size, nb_classes=80, device="auto", task=None, **kwargs)
```

`device="auto"`は利用可能な場合にCUDA、次にMPS、最後にCPUを選択します。整数または数字の文字列はCUDAのordinalとして解釈されるため、`device=0`と`device="0"`はどちらも`cuda:0`を意味します。`task`はファミリーの`SUPPORTED_TASKS`に照らして検証されます。`model_path=None`を渡すとアーキテクチャを構築して学習モードのままにし、`dict`を渡すとそのstate dictを直接読み込みます。

## predictと\_\_call\_\_

`predict`は`__call__`のエイリアスです。

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
| `source` | `None` | 画像、メモリ上の画像のリストまたはタプル、ディレクトリ、動画ファイル、`"screen"`、`"screen 1"`、`"screen 1 100 200 512 256"`のような画面ソース |
| `conf` | `0.25` | 信頼度しきい値 |
| `iou` | `0.45` | NMSのIoUしきい値 |
| `imgsz` | `None` | 入力サイズのオーバーライド。`None`ではモデルのネイティブサイズを使用 |
| `device` | `None` | この呼び出し用のデバイスオーバーライド |
| `classes` | `None` | 指定したクラスIDだけを保持 |
| `max_det` | `300` | 画像ごとの最大検出数 |
| `augment` | `False` | テスト時拡張 |
| `save` | `False` | アノテーション付き画像または動画を書き込み |
| `batch` | `1` | ディレクトリおよびリストソースでの順伝播1回あたりの画像数 |
| `stream` | `False` | 実体化したリストではなくジェネレーターを返す |
| `stream_buffer` | `False` | 最新フレームだけでなく、取得したすべてのライブフレームを保持 |
| `vid_stride` | `1` | 動画または画面のNフレームごとに処理 |
| `show` | `False` | アノテーション付きフレームをウィンドウに表示 |
| `output_path` | `None` | `save=True`の場合の出力パス |
| `color_format` | `"auto"` | メモリ上の配列に対する色形式のヒント |
| `tiling` | `False` | 大きな画像向けのタイル推論 |
| `overlap_ratio` | `0.2` | タイルの重複率 |
| `output_file_format` | `None` | `"jpg"`、`"png"`、`"webp"`のいずれか |
| `cuda_graph` | `False` | `True`では入力形状ごとの初回使用時にキャプチャし、`"auto"`では形状の再出現を待機 |

単一の画像ソースでは1つの`Results`を返します。リスト、タプル、ディレクトリではリストを返し、`stream=True`ではすべての場合にジェネレーターを返します。

ライブストリームソースには終端がないため、`stream=True`が必要です。`tiling`と`augment`は併用できません。`embed`、`point`、`edge`タスクではテスト時拡張を指定すると例外が発生します。

<code-tabs name="usage" />

`batch > 1`の場合、`SUPPORTS_BATCHED_PREDICT`がtrueのファミリーはchunkごとにstackした順伝播を1回実行します。`batch=1`では画像ごとに1回の順伝播を維持します。

<code-tabs name="stream" />

## embed

```python
model.embed(source=None, **kwargs) -> torch.Tensor
```

すべての埋め込み行を単一の`(N_total, D)`テンソルへstackする、`predict`の簡易ラッパーです。モデルが`task="embed"`で構築されている必要があり、それ以外では`NotImplementedError`が発生します。

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

`track_id`を設定した`Results`をフレームごとに1つyieldします。`tracker`は`"bytetrack"`、`"botsort"`、`"ocsort"`、`"deepocsort"`のいずれかです。`tracker_config`を指定すると、その構成の型がトラッカーを選ぶため`tracker`は無視されます。`track_conf`はByteTrackとBoT-SORTでは`track_high_thresh`に、OC-SORTとDeep OC-SORTでは`det_thresh`にマッピングされます。`output_path`のデフォルトは`runs/track/<video_stem>.mp4`です。

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

タスクによってキーが異なる指標辞書を返します。物体検出では`metrics/precision`、`metrics/recall`、`metrics/mAP50`、`metrics/mAP50-95`を返します。`imgsz`は正方形の整数または`(height, width)`タプルを受け取り、デフォルトはモデルのネイティブ入力サイズです。`plots`は`save_plots`のエイリアスです。`allow_download_scripts`は、データセットYAMLの`download`フィールドに含まれる可能性がある埋め込みPythonの実行を制御します。

`faster_coco_eval`は`**kwargs`を通じて受け付け、デフォルトは`True`です。パッケージがインストールされていない場合はpycocotoolsへフォールバックします。実際に使用したバックエンドは`model.last_eval_backend`で報告されます。

`obb`と`pose`タスクでは、拡張付き検証を指定すると例外が発生します。

## train

`train`はファミリーごとに定義されるため、引数も異なります。基本クラスが各ファミリーの`train`をラップするため、2つの動作は共通です。

- `cfg=`はYAMLパスを受け取り、キーを呼び出しに統合します。明示的なキーワード引数がファイルより優先されます。
- coverage group `g0`または`g1`のファミリーで`pretrained=False`を指定すると、学習前にモデルをスクラッチから再初期化します。`resume=True`とは併用できません。

ファミリーが実際に対応するデータ拡張設定はファミリーごとに異なります。[データ拡張マトリックス](/docs/reference/augmentation-matrix)を参照してください。

## export

```python
model.export(format="onnx", **kwargs) -> str
```

書き込んだアーティファクトのパスを返します。`format`はエクスポーターレジストリーから解決され、`engine`は`tensorrt`、`litert`は`tflite`のエイリアスです。すべてのエクスポーターに共通する引数は次のとおりです。

| 引数 | デフォルト | 意味 |
|---|---|---|
| `output_path` | `None` | 出力ファイルパス。省略すると`weights/`の下に生成 |
| `imgsz` | `None` | `(height, width)`タプルまたは単一の整数。デフォルトはネイティブサイズ |
| `opset` | `None` | ONNX opsetバージョン |
| `simplify` | `True` | ONNXグラフの簡略化を実行 |
| `dynamic` | `True` | 動的軸を有効化 |
| `half` | `False` | FP16精度 |
| `int8` | `False` | INT8精度 |
| `batch` | `1` | アーティファクトへ固定するバッチサイズ |
| `device` | `None` | traceを実行するデバイス |
| `data` | `None` | INT8 calibration用のdata.yaml |
| `fraction` | `1.0` | 使用するcalibrationデータセットの割合 |
| `allow_download_scripts` | `False` | データセットYAMLのダウンロードに含まれる埋め込みPythonを許可 |
| `verbose` | `False` | 詳細なエクスポーターログ |

禁止された組み合わせはtrace前のpreflightで`NotImplementedError`を発生させます。対応状況とルールは[エクスポートマトリックス](/docs/reference/export-matrix)のページにあります。有効なLoRAアダプターがある場合は密な重みへ統合されますが、そのmergeはすべてのリクエスト拒否判定後にだけ行われます。

## save

```python
model.save(path) -> str
```

v1.0スキーマのLibreYOLOチェックポイントを書き込みます。state dictと[チェックポイントスキーマ](/docs/reference/checkpoint-schema)に記載されたメタデータです。量子化済みモデルでは`quant`manifestも追加されるため、`LibreYOLO(path)`が量子化構造とscaleを復元します。

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

その場で量子化し、モデルを返します。`recipe`はcastの`fp16`と`bf16`、ConvとLinear向けの`int8`と`fp8`、またはLinear専用の`w4a16`、`w4a8`、`nvfp4`、`mxfp4`、`int2`のいずれかです。最後のLinear専用レシピにはRF-DETRなどのTransformerファミリーが対応します。`int2`にはQATが必要です。`calib`はdata.yamlのパスまたは組み込みデータセット名を受け取り、画像を順伝播だけで読み込みます。ラベルは読み取りません。`calib=None`でcalibrationをスキップします。`algorithm`は`"minmax"`、`"percentile"`、`"auto"`のいずれかです。

`model.quant_info()`は量子化状態の要約を返し、浮動小数点モデルでは`None`を返します。`model.dequantize()`は量子化学習済みのmaster weightを維持しながら、その場で浮動小数点モジュールを復元します。これはQATから`export(format="onnx", int8=True, data=...)`へつなぐ橋渡しです。

## infoとlayers

```python
model.info(detailed=False, verbose=True) -> Dict[str, Any]
model.get_available_layer_names() -> List[str]
model.get_distill_config() -> Dict
```

`info`はJSONで扱える辞書を返し、`verbose`がtrueの場合は人が読める要約をログへ出力します。`get_available_layer_names`は、蒸留または特徴抽出の構成で指定できる層を一覧にします。

## CUDA Graph

`SUPPORTS_CUDA_GRAPH`クラス属性がtrueのファミリーで利用できます。再生結果はeager実行とビット単位で同一です。

```python
model.capture_graph(imgsz=None, batch=1, dtype=None) -> None
model.cuda_graph_scope(mode=True)          # コンテキストマネージャー
model.graph_info() -> Dict[str, Any]
model.release_graphs() -> None
```

キャプチャしたグラフはキャプチャ時とまったく同じ形状でだけ有効なので、`batch`と`imgsz`は後の`predict`呼び出しと一致する必要があります。`capture_graph`はキャプチャコストを最初のリクエストより前に移します。`mode`は、初回使用時にキャプチャする`True`または`"on"`、同じ形状が現れるまで待つ`"auto"`、何もしない`False`を受け付けます。ファミリーが対応していない場合、`capture_graph`は`NotImplementedError`を発生させ、キャプチャに失敗した場合は`CudaGraphUnavailable`を発生させます。

## デバイスとdtype

`Results`オブジェクトは`.to()`、`.cpu()`、`.cuda()`、`.numpy()`を備えます。[Resultsの型](/docs/reference/results-types)を参照してください。モデル自体は、構築時または`predict`に`device=`を渡して移動します。
