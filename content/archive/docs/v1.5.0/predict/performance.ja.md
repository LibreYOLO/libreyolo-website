---
title: 推論性能
seo_title: LibreYOLOで推論を高速化
description: 推論時のCUDAグラフ、半精度、バッチ処理、タイル分割推論、テスト時拡張について、実際のデフォルトと各機能をサポートするファミリーを説明します。
lead: >-
  推論時の5つの制御によって、スループットまたは精度が変わります。CUDAグラフの再生、精度、バッチ処理、タイル分割、テスト時拡張です。それぞれが特定のファミリー集合に適用され、そのうち2つは節約ではなく精度またはレイテンシのコストを伴います。
keywords:
  - cuda graphs pytorch 推論
  - yolo バッチ推論 python
  - fp16 推論
  - タイル分割推論 小物体
  - 大きい画像 分割推論
  - test time augmentation 物体検出
  - capture_graph
  - フォルダー 一括推論
last_verified: 1.5.0
verification: >-
  引数のデフォルトはlibreyolo/models/base/inference.pyのInferenceRunner.__call__で確認しました。CUDAグラフAPIはlibreyolo/models/base/model.pyのBaseModel.capture_graph、graph_info、release_graphs、cuda_graph_scopeで確認し、ファミリーごとの有効化はSUPPORTS_CUDA_GRAPHクラス変数で確認しました。半精度の動作はlibreyolo/utils/predict_args.pyのNOOP_PREDICT_KWARGS、libreyolo/cli/commands/predict.pyのCLI警告、libreyolo/quant/api.pyのCAST_RECIPESとSUPPORTED_FAMILIESで確認しました。バッチ処理の条件はInferenceRunner._process_in_batchesと_predict_batchで確認しました。タイル分割は_predict_tiledと_merge_tile_detectionsで確認しました。テスト時拡張はBaseModel._predict_augmentと_merge_ttaで確認し、TTA_ENABLED、TTA_SCALES、TTA_FIXED_SIZEはlibreyolo/models/全体から確認しました。
snippets:
  batch:
    - label: フォルダーに対するバッチ推論
      language: python
      code: |
        from pathlib import Path
        from PIL import Image

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        folder = Path("batch_demo")
        folder.mkdir(exist_ok=True)
        image = Image.open(SAMPLE_IMAGE)
        for index in range(8):
            image.save(folder / f"frame_{index}.jpg")

        model = LibreYOLO("LibreYOLO9s.pt")

        # 対応ファミリーでは4個のチャンクごとに1回の積み重ねた順伝播を実行
        results = model(str(folder), batch=4)
        print(len(results), "results")
    - label: リストを生成しないストリーミング
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        for result in model("batch_demo", batch=4, stream=True):
            print(len(result.boxes))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt source=batch_demo batch=4
  graphs:
    - label: 事前にキャプチャして再生（CUDAが必要）
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")

        # 最初のリクエストとは別にウォームアップとキャプチャを一度だけ実行
        model.capture_graph()

        result = model(SAMPLE_IMAGE, cuda_graph=True)
        print(len(result.boxes))
        print(model.graph_info())
    - label: 形状が繰り返されたときだけキャプチャ（CUDAが必要）
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")

        # "auto" は形状が2回現れるまで待つため1回限りの処理は
        # キャプチャのコストを負担しない
        for _ in range(3):
            model(SAMPLE_IMAGE, cuda_graph="auto")

        print(model.graph_info())
        model.release_graphs()
  precision:
    - label: エクスポート用追加パッケージをインストール
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: デフォルト精度でエクスポートして再読み込み
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        path = model.export(format="onnx")

        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)
        print(len(result.boxes))
    - label: FP16エクスポート（CUDAマシンで構築、実行）
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")
        path = model.export(format="onnx", half=True)

        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)
        print(len(result.boxes))
    - label: キャストレシピによるPyTorchのFP16（CUDAが必要）
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")

        # キャストレシピは較正データを読み取らない
        model.quantize(recipe="fp16", calib=None)

        result = model(SAMPLE_IMAGE)
        print(len(result.boxes))
  tiling:
    - label: 大きい画像でタイル分割推論
      language: python
      code: |
        from PIL import Image

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 画像が入力サイズより大きい場合だけタイル分割を実行
        large = Image.open(SAMPLE_IMAGE).resize((2048, 1536))
        large.save("large.jpg")

        model = LibreYOLO("LibreYOLO9s.pt")

        result = model("large.jpg", tiling=True, overlap_ratio=0.2)
        print(result.num_tiles, "tiles", len(result.boxes), "detections")
  tta:
    - label: テスト時拡張
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        plain = model(SAMPLE_IMAGE)
        flipped = model(SAMPLE_IMAGE, augment=True)

        print(len(plain.boxes), "->", len(flipped.boxes))
source_hash: 3914665d0e7f892c
---

## 制御とデフォルト

これらはすべて`predict`の引数で、デフォルトではすべて無効です。

| 引数 | デフォルト | 効果 |
|---|---|---|
| `batch` | `1` | フォルダーとリストのソースで、順伝播1回あたりの画像数 |
| `cuda_graph` | `False` | キャプチャ済みCUDAグラフから順伝播を再生 |
| `tiling` | `False` | 大きい画像を重なり合うタイルに分割 |
| `overlap_ratio` | `0.2` | `tiling`が有効な場合のタイルの重なり |
| `augment` | `False` | 反転したビューを実行して統合 |
| `half` | | 受け付けて警告し、無視 |
| `device` | `None` | 推論前にモデルを移動 |

`imgsz`はモデルが動作する解像度を設定するため、コストにも影響します。ただし、まず精度に関する引数であり、ここではなくモデル側に属します。

## バッチ処理

<code-tabs name="batch" />

`batch`はフォルダーとリストのソースに適用されます。`batch=1`では、画像ごとに1回の順伝播を実行します。`1`を超えると、各チャンクを前処理して1つのテンソルに積み重ね、1回実行した後、各ファミリーの既存の単一画像後処理が想定どおりの入力を受け取れるように再度分割します。

積み重ねた経路は、次のすべてを満たす場合だけ使用されます。

- `batch`が`1`より大きい。
- `tiling`が無効。
- テスト時拡張が有効でない。
- ファミリーが`SUPPORTS_BATCHED_PREDICT`を設定している。
- 基盤となるネットワークが学習モードでない。

最後の条件は単なる細部ではありません。学習モードのネットワークでは、積み重ねたチャンクを画像横断のバッチ統計で正規化し、同じチャンク内の画像が互いの予測を変えてしまいます。そのため、その場合は順次実行を維持します。

`SUPPORTS_BATCHED_PREDICT`のデフォルトはtrueです。次のファミリーは対象外となり、`batch`に関係なく順伝播1回につき1画像を実行します。Depth Anything V2、Depth Anything 3、EoMT、Faster R-CNN、FCOS、HRNet、L2CS-Net、LibreMODUS、MiDaS、MoGe-2、PP-OCRv5、Real-ESRGAN、RetinaNet、SAM 3D Body、SwinIR、YOLOv1、ZipDepth、すべてのオープンボキャブラリ検出器、すべてのVision Language Modelです。

もう1つフォールバックがあります。チャンク全体で前処理が形状、dtype、デバイスの一致する均一な`(1, C, H, W)`テンソルを返さない場合、積み重ねずに順次実行します。そのため、正しさが偶然同じサイズになった画像に依存することはありません。

大きいフォルダーで`batch`と`stream=True`を組み合わせると、すべての結果をメモリに保持せずにバッチ順伝播を実行できます。

## CUDAグラフ

<code-tabs name="graphs" />

CUDAグラフは順伝播を1回記録し、単一の起動として再生します。小型検出器ではバッチ1の時間の大きな割合がカーネル起動に費やされるため、それらの起動をまとめることでスループットが向上し、再生出力はeager実行とビット単位で一致します。

`cuda_graph`は3つの値を取ります。デフォルトの`False`では何も行いません。`True`では、各入力形状の初回使用時にキャプチャします。`"auto"`では形状が繰り返されるまで待ってからキャプチャするため、1回限りの処理や形状が変化する処理はキャプチャのコストを負担しません。

`capture_graph(imgsz=None, batch=1, dtype=None)`を使うと、そのコストを最初のリクエストから外せます。グラフはキャプチャした正確な形状にだけ有効なので、ここでの`batch`は後から`predict`を呼び出す方法と一致させる必要があります。

`graph_info()`は、キャプチャ済みグラフ、再生回数、eager実行へフォールバックした理由を報告します。`release_graphs()`はグラフと静的バッファを解放します。

キャプチャにはCUDAと、`SUPPORTS_CUDA_GRAPH`で有効化されたファミリーが必要です。ホスト側から見える処理を含まない順伝播が必要であり、ファミリーごとに検証されているためです。有効化されていないファミリーで要求すると、暗黙にeager実行へ移行せず`NotImplementedError`が発生します。

グラフは値ではなくメモリアドレスを記録するため、パラメータを移動する操作を行うと破棄されます。`predict(device=...)`によるデバイス変更、量子化、量子化解除はすべてキャプチャ済みグラフを無効にします。

ファミリーごとの完全な対応マトリクス、シームの分割、数値の規約については[CUDAグラフ](/docs/reference/cuda-graphs)を参照してください。

## 精度

<code-tabs name="precision" />

推論時の`half=True`は何も行いません。コマンドライン互換性のため受け付けられ、何もしないことを示す警告を発し、どのファミリーにも届く前に破棄されます。CLIの`--half`フラグも`.pt`モデルに対して同じ警告を表示します。

低精度化には実際に動作する2つの経路があります。

エクスポートした成果物では、`export(format=..., half=True)`を使ってエクスポート時に精度を選択し、生成されたファイルを変更せずに`LibreYOLO()`から再読み込みできます。

PyTorch実行では、`model.quantize(recipe="fp16")`がモデルをfloat16へキャストし、モデルの入力と出力をfloat32に保つフックをインストールします。`"bf16"`はbfloat16で同じ処理を行います。どちらのキャストも較正データを読み取らないため、`calib`は無視されます。現在、量子化はYOLOv9、RF-DETR、BiRefNet、FeyNobgの4ファミリーを対象とします。CPUデバイス上のキャストでは遅くなるという警告がログに記録されるため、これらのレシピはGPU向けです。

どちらの経路も数値を変えます。検出結果が同じになることをそのまま保証するものではないため、デプロイ前に検証してください。

## タイル分割推論

<code-tabs name="tiling" />

タイル分割では、大きい画像を重なり合う正方形タイルに切り出し、それぞれで推論し、結果を統合します。画像全体のリサイズによって対象がモデルで解像できないほど小さくなる高解像度画像で、小物体を扱うための選択肢です。

タイルサイズはモデルの入力サイズ、または指定した場合は`imgsz`で、正方形でなければなりません。`overlap_ratio`のデフォルトは`0.2`です。重なり合うタイルは、`iou`のしきい値でクラスごとのNMSを使って調整され、統合後の一覧が`max_det`までに切り詰められます。つまり、独自にはNMSを実行しないファミリーでも、タイル分割推論では`iou`が効果を持ちます。

画像がすでに収まる場合、タイル分割は単に低コストになるのではなくスキップされます。両方の寸法が入力サイズ以下なら、代わりに通常の順伝播を1回実行します。分類、セマンティックセグメンテーション、`embed`タスクでもスキップされます。そこではタイル分割に意味がないため、1回の処理へフォールバックします。

再結合できないペイロードを持つタスクでは例外が発生します。対象はインスタンスセグメンテーションのマスク、回転バウンディングボックス、点、深度、エッジ、法線です。`augment`と組み合わせることもできません。

結果には`result.tiled`と`result.num_tiles`が含まれます。`save=True`を指定すると、タイル分割実行は`runs/tiled_detections`の下にディレクトリを作成し、すべてのタイル、アノテーション済み画像、グリッド表示、タイルサイズ、重なり、しきい値を記録した`metadata.json`を保存します。`result.tiles_path`と`result.grid_path`はそれらを指します。

## テスト時拡張

<code-tabs name="tta" />

`augment=True`では画像を複数回実行し、`iou`のしきい値でクラスごとのNMSを使って検出結果を統合します。タイル分割と同様に、通常は`iou`を無視するファミリーでも、この処理では`iou`が重要になります。

実際には水平反転です。スケール一覧`TTA_SCALES`のデフォルトは単一スケールの`1.0`で、公開されているどのファミリーも上書きしていません。そのため、各ファミリーは元画像とその鏡像の2回の処理を実行します。`TTA_FIXED_SIZE`が設定されたファミリーは固定された正方形にリサイズするため、どの場合でもマルチスケールは何も行いません。

セマンティックセグメンテーションとパノプティックセグメンテーションは異なる統合処理を使います。反転したビューを元に戻し、ボックスとして統合する代わりに、2つのsoftmax分布をargmaxの前に平均します。

テスト時拡張はすべてのタスクで利用できるわけではありません。回転バウンディングボックス、姿勢推定、点、深度、法線、エッジ、復元、OCR、埋め込みベクトルのモデルでは例外が発生し、タイル分割とも組み合わせられません。

次のファミリーでは完全に無効になっているため、`augment=True`でも通常の処理を1回だけ実行します。BiRefNet、CenterNet、CLIP、DexiNed、FOMO、HRNet、L2CS-Net、LibreMODUS、NAFNet、PP-OCRv5、Real-ESRGAN、RetinaNet、SAM 3D Body、SigLIP2、SwinIR、TEED、すべてのSAMバリアント、すべてのオープンボキャブラリ検出器、すべてのVision Language Modelです。

## 測定

ハードウェア、ランタイム、精度、バッチサイズが示されていないミリ秒値は事実ではないため、このページにはレイテンシの数値を載せていません。ハードウェアとランタイムを横断した測定値は[visionanalysis.org](https://www.visionanalysis.org)で公開され、`libreyolo profile`を使うと手元のマシンで特定のモデルを測定できます。

