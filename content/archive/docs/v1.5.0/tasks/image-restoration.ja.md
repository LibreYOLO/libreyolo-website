---
title: 画像復元
seo_title: LibreYOLOによる画像復元とアップスケール
description: >-
  LibreYOLOで画像のノイズ除去、ぼけ除去、アップスケールを実行します。復元済みRGB画像の推論、ペアデータによるNAFNetの学習、PSNRとSSIMキーの読み方を解説します。
lead: >-
  画像復元は劣化した画像を受け取り、きれいな画像を返します。LibreYOLOではrestoreタスクとして提供され、ノイズ除去、ぼけ除去、超解像を1つの出力契約で扱います。1枚のRGB画像を入力し、1枚のRGB画像を出力します。
keywords:
  - Python 画像復元
  - 画像ノイズ除去 モデル
  - Python 超解像
  - 画像ぼけ除去 モデル
  - PSNR SSIM 検証
last_verified: 1.5.0
snippets:
  predict:
    - label: 画像をアップスケール
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 小型の4倍生成器 tileは大きなソースでピークメモリを制限する
        model = LibreYOLO("LibreRealESRGANx4t-restore.pt")
        result = model(SAMPLE_IMAGE, tile=512, tile_pad=10)

        result.restored.save("upscaled.png")
        print(result.restored.array.shape)   # 各軸で入力の4倍
    - label: 画像のノイズを除去
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # SIDDの実画像ノイズで学習され出力は入力サイズを維持する
        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        result = model(SAMPLE_IMAGE)

        result.restored.save("denoised.png")
        print(result.restore_scale)   # 1 このチェックポイントではアップスケールなし
  train:
    - label: ペア画像でNAFNetをファインチューニング
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        model.train(data="my-dataset.yaml", epochs=100, imgsz=256, batch=16,
        lr0=1e-3)
    - label: チェックポイントに来歴を記録
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # degradationとdatasetは来歴として保存済みチェックポイントへ
        # 書き込まれるが学習には関与しない
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            degradation="denoise",
            dataset="MyDataset",
        )
  val:
    - label: 検証して指標キーを確認
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # val()はオブジェクトではなく通常のdictを返す
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PSNR"])   # fitness
        print(metrics["metrics/SSIM"])
  export:
    - label: エクスポート
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # imgszはグラフへ固定されるためデプロイ先で実際にモデルへ渡す
        # サイズを指定する
        model.export(format="onnx", imgsz=256)
    - label: エクスポートしたファイルを実行
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリはファイルの拡張子に応じて振り分けるためエクスポートした
        # アーティファクトもチェックポイントと同様に読み込まれ同じResultsオブジェクトを返す
        model = LibreYOLO("LibreNAFNetl-restore-sidd.onnx")
        result = model(SAMPLE_IMAGE)

        result.restored.save("denoised.png")
source_hash: 9dc81cadb3ebf18b
---

## 定義

`restore`タスクは1枚の画像を別の画像へ変換します。ノイズ除去、ぼけ除去、超解像は、同じ契約を共有するため、ここではすべて同じタスクです。モデルはRGB画像を受け取り、RGB画像を返します。元に戻すため学習した劣化の種類はAPIではなくチェックポイントの属性です。

推論結果では`result.restored`に値が格納されます。これは`(H, W, 3)`のuint8 RGB配列を保持する`RestoredImage`ペイロードです。`.array`はNumPy配列として返し、`.save(path)`はディスクへ書き込みます。`result.restore_scale`には出力キャンバスのアップスケール倍率が記録され、解像度を維持するチェックポイントでは`1`です。`result.boxes`は空のままなので、`conf`、`iou`、`max_det`はシグネチャの互換性のため受け付けますが効果はありません。`save=True`は注釈付き写真ではなく復元済み画像を直接書き込みます。

## モデル

`restore`には、取り除く劣化の種類で分かれた3つのファミリーが対応します。

[NAFNet](/docs/models/nafnet)はノイズ除去器であり、LibreYOLOで学習できる唯一の復元ファミリーです。そのアーキテクチャはUNetブロックの非線形活性化を要素ごとの乗算へ置き換えます。公開済みチェックポイントはSIDDの実画像ノイズで学習され、出力は入力解像度を維持します。

[Real-ESRGAN](/docs/models/real-esrgan)は実用的なアップスケーラーです。バイキュービックによる縮小だけでなく合成劣化に対して学習された3つのチェックポイントがあり、4倍、2倍、低レイテンシ向けの小さく高速な4倍生成器を提供します。

[SwinIR](/docs/models/swinir)はSwin Transformerバックボーンを使って4倍にアップスケールします。公式の軽量生成器と2つの実環境向け生成器を含む3サイズがあります。

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

画像復元は固定されたネットワークキャンバスではなくソース画像本来の解像度で実行し、ネットワークのダウンサンプリング係数に合わせるためのパディングだけを行います。そのため、時間とメモリは入力のピクセル数に応じて増えます。`tile`は順伝播を重なり合うタイルへ分割し、境界を再び混合します。`tile_pad`は各タイルへ追加し、後で切り取る周辺領域です。どちらもPythonのキーワード引数です。入力ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## データセット形式

画像復元では、劣化した各入力画像を、まったく同じ解像度かつ同じファイル名の基幹部を持つきれいな対象画像と対応付けます。

```text
dataset/
  data.yaml
  inputs/
    train/photo.jpg
    val/photo.jpg
  targets/
    train/photo.jpg
    val/photo.jpg
```

```yaml
path: dataset
train: inputs/train
val: inputs/val
input_dir: inputs
target_dir: targets
degradation: denoise
dataset: MyDataset
nc: 1
names: {0: image}
```

`nc`と`names`はスキーマ上のプレースホルダーです。復元モデルは検出ではなく`Results.restored`を返します。`degradation`と`dataset`は任意の来歴ラベルです。`target_stem_suffix`は、きれいな画像に劣化画像とは異なる名前を使うデータセットに対応します。検証ではネイティブ解像度を維持し、バッチを積み重ねるために必要な分だけパディングするため、指標は元のキャンバス上で計算されます。完全な契約については[データセット形式](/docs/reference/dataset-formats)を参照してください。

## 学習

学習実装を持つ復元ファミリーはNAFNetだけです。`Real-ESRGAN.train()`と`SwinIR.train()`はどちらも`NotImplementedError`を送出します。これらのチェックポイントは合成劣化パイプラインを使うGAN学習から得られたもので、ペア画像の復元トレーナーではその手法を再現できないためです。

<code-tabs name="train" />

トレーナーは入力と対象のペアを連動してクロップするため、両側の位置関係が維持されます。データセット、マルチGPU、ロガーについては[学習](/docs/train)、このファミリーのデフォルト値と学習時に切り離す推論用プーリングについては[NAFNetページ](/docs/models/nafnet)を参照してください。

## 検証

`val()`は元のキャンバス上で、境界のクロップやサイズ変更を行わず、復元済み出力をきれいな対象とRGBで比較します。

<code-tabs name="val" />

`metrics/PSNR`はデシベル単位のピーク信号対雑音比で、最良チェックポイントの選択で使う`fitness`にもなります。`metrics/SSIM`は`[0, 1]`範囲の構造的類似度で、シグマ1.5の11×11ガウシアン窓を使用し、3つのカラーチャンネルで平均します。どちらも大きいほど良い指標です。

## エクスポート

エクスポートした復元モデルはファイルの拡張子に基づいて`LibreYOLO()`で再読み込みできるため、`.onnx`や`.engine`ファイルもチェックポイントと同様に動作します。返される`Results`では`restored`に出力画像が格納されます。

<code-tabs name="export" />

復元モデルのエクスポートでは空間解像度をグラフへ固定するため、デプロイ先で実際にモデルへ渡す`imgsz`を指定してください。NAFNetではそのサイズがネットワークのダウンサンプリング係数で割り切れる必要があり、`dynamic=True`でも動的になるのはバッチ次元だけです。Real-ESRGANとSwinIRで`imgsz`を省略すると、実際の処理解像度ではなく小さな内部パッチサイズが使われます。形式ごとの対応範囲は各モデルページと[完全なエクスポートマトリックス](/docs/reference/export-matrix)に記載されています。[エクスポート](/docs/export)には各形式で受け付ける引数が記載されています。
