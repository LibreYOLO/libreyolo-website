---
title: 軽量インストール
seo_title: PyTorchなしでLibreYOLOのONNX推論を実行
description: >-
  LibreYOLOを--no-depsでインストールし、ディスク上にtorchを置かず、numpyだけでONNX検出を実行します。手法、制約、正確なパッケージリストを説明します。
lead: >-
  LibreYOLOのONNX推論経路は、decodeとNMSを含めて最初から最後までnumpyで動作します。実行時にPyTorchを必要とする部分がないため、依存関係の解決を省略したインストールでは、マシンにtorchがなくても検出を実行できます。
keywords:
  - torch なし 推論
  - pytorch なし
  - libreyolo pytorch なし
  - onnx 推論 torch 不要
  - libreyolo 軽量インストール
  - pip install no-deps
  - libreyolo 容量
  - onnxruntime 推論
last_verified: 1.5.0
meta:
  - label: 適用対象
    value: ONNX検出、7つのモデルファミリー
  - label: エントリーポイント
    value: libreyolo.backends.onnx.OnnxBackend
    mono: true
  - label: サポートレベル
    value: ベストエフォート、個別のディストリビューションではない
snippets:
  install:
    - label: 軽量
      language: bash
      code: |
        # 依存関係リストを使わずにパッケージをインストールし
        # ONNX検出経路が実際にインポートする4つのパッケージを追加
        pip install --no-deps libreyolo
        pip install numpy pillow opencv-python-headless onnxruntime
    - label: CPU専用torch
      language: bash
      code: |
        # 最初にこの方法を試す。すべての機能を維持しつつ
        # ディスク容量の大半を占めるCUDA wheelを回避
        pip install libreyolo --index-url https://download.pytorch.org/whl/cpu
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo.backends.onnx import OnnxBackend


        model = OnnxBackend("libreyolo9t.onnx")

        result =
        model.predict("https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg")


        # ここでxyxyはtorchテンソルではなくnumpy ndarray

        print(result.boxes.xyxy)

        print(result.boxes.conf)

        print(result.boxes.cls)
source_hash: e60e83d32d13026e
---

## この方法が機能する理由

`pip install --no-deps libreyolo` はパッケージをインストールし、その依存関係リストを完全に省略します。依存関係は自動で解決されず、実際に使うものを自身でインストールする必要があります。

必要なコード経路が、省略した依存関係を本当に必要としない場合にだけ、この方法は役立ちます。ONNX検出では必要ありません。non-maximum suppressionを含むdecodeはnumpyです。前処理手順もnumpyです。PyTorchは学習とeager推論の依存関係であり、この経路では一度も呼び出されません。

このリリースより前は、それでもインポートに失敗していました。`libreyolo.models` 以下のものをインポートすると、チェックポイント自動検出レジストリへ登録するためにすべてのモデルクラスが構築され、それらのクラスは `torch.nn.Module` のsubclassだからです。前処理手順は独自のパッケージ `libreyolo.preprocess` に移され、torch属性に何かが触れるまでtorchのインポートが遅延されるようになりました。そのため、マシンにtorchがなくてもONNX経路をインポートできます。このパッケージは `yolo9`、`yolonas`、`yolox`、`ec`、`rtdetr`、`rfdetr`、`dfine`、`deim`、`deimv2` のファミリーごとにnumpyネイティブのpreprocessorを持ち、以下でエンドツーエンド検証されている7ファミリーより2つ多くなっています。各 `libreyolo/models/<family>/utils.py` はそこから再エクスポートするため、既存のインポートパスも引き続き機能します。

## 最初にCPU専用wheelを試す

この方法を求める多くの人は、数GBに及ぶインストールを避けようとしています。その容量は1か所に集中しています。デフォルトの `torch` wheelにはCUDAが同梱されます。CPU専用ビルドなら容量はその一部で済み、特別なインストール経路も必要ありません。

<code-tabs name="install" />

CPU専用オプションでは、学習、検証、すべてのタスク、すべてのファミリー、CLIというLibreYOLOの全機能を維持します。torchを単に減らすのではなく、マシンから完全になくしたい場合に軽量経路を使ってください。

## 軽量インストールの対象

| | |
|---|---|
| タスク | 検出 |
| 形式 | ONNX |
| エントリーポイント | `OnnxBackend` |
| インターフェース | Pythonライブラリ |

この経路では7つのファミリーが検証されています。[YOLOv9](/docs/models/yolov9)、[YOLO-NAS](/docs/models/yolo-nas)、[EdgeCrafter](/docs/models/edgecrafter)、[RT-DETR](/docs/models/rt-detr)、[RF-DETR](/docs/models/rf-detr)、[D-FINE](/docs/models/d-fine)、[DEIM](/docs/models/deim)であり、それぞれのファミリーのバリアントも含みます。

これは検証済みの範囲であり、ライブラリが強制する境界ではありません。ほかのタスクとファミリーは単に未確認です。呼び出すとtorchを読み込むものも、一部は偶然動作するものもあります。このリストを超えるものは、サポート済みや故障ではなく、未検証として扱ってください。

この範囲内では、結果は通常のインストールと近いだけでなく、同一です。各ファミリーをONNXへエクスポートして、通常の状態とtorchをブロックした状態で1回ずつ実行したところ、ボックス、スコア、クラスが完全に一致しました。テストスイートの一致テストにより、この仕様のずれを防いでいます。

## よく問題になる5つの点

**モデルクラスではなく `OnnxBackend` を使います。** `LibreYOLO9("model.onnx")` は、それ自体が `nn.Module` のsubclassであるため、引き続きtorchを必要とします。このドキュメントのほかのすべてのページではクラスまたは `LibreYOLO()` からモデルを読み込むため、もっとも起こりやすい間違いです。

**別の場所でエクスポートします。** `.onnx` ファイルの生成にはtorchが必要なので、軽量マシンでは作成できません。開発環境またはCIマシンでエクスポートし、成果物を軽量な対象環境へ配置してください。

**Resultsにはnumpy配列が格納されます。** ここでは `result.boxes.xyxy` が `ndarray` です。containerはどちらの型も受け付けるため属性名は変わりませんが、結果に対して `.cpu()` や `.numpy()` を呼び出すコードは失敗します。

**1枚の画像は1つの `Results` を返します。** `predict()` は、画像が1枚なら1つの `Results`、複数ならリストを返します。1つの結果に `[0]` でインデックスを指定すると、最初の画像ではなく最初の検出結果が選択され、例外を送出せずに1つのボックスだけを持つ結果になります。

**CLIは動作しません。** `typer` と `click` は4つのパッケージに含まれないため、`libreyolo` コマンドは利用できません。これはライブラリとしてのインストールです。

## 推論

<code-tabs name="predict" />

CUDAで実行するには、`onnxruntime` を `onnxruntime-gpu` に置き換えてください。この4つのパッケージは、torchなしの完全な `predict()` が実際にインポートするものです。推測ではなく、呼び出し中に記録されています。`opencv-python-headless` は宣言済みの `opencv-python` の代わりになります。同じモジュールを提供し、GUIライブラリを含まず、ディスク上の容量も小さくなります。

残りの宣言済み依存関係のうち、`requests` はURLから画像を読み込む場合だけ必要です。`pycocotools` と `scipy` は検証と評価用、`typer` と `click` はCLI用です。

## このリストは意図的に変化する

上記のパッケージリストは、このページの先頭に記載されたリリースに対して正確です。`--no-deps` を指定すると依存関係の解決を無効にするため、何も自動確認されず、将来のリリースではここにないものをインポートする可能性があります。

`ModuleNotFoundError` が発生した場合は、すでに手法を理解しています。不足しているパッケージをインストールしてください。これはバグ報告の対象ではなく、想定された保守方法です。この経路はベストエフォートで、個別にサポートされるディストリビューションではありません。PyPIに2つ目の軽量パッケージがなく、作成予定もない理由も同じです。

環境がインストール済みのコピーへ黙ってフォールバックしておらず、本当にtorchなしであることを確認するには、assertしてください。

```python
import importlib.util

assert importlib.util.find_spec("torch") is None, "torch is installed"
```

軽量イメージのCIにこのチェックを残しておく価値があります。このチェックがないと、たまたまtorchがある環境ではすべてのテストが通過し、何も確認できません。

