---
title: Python API
seo_title: LibreYOLO Python APIリファレンス
description: >-
  LibreYOLOがパッケージレベルでエクスポートする名前を説明します。5つのファクトリー、ファミリークラス、Resultsペイロード、バックエンド、バリデーター、トラッカー、データヘルパーを扱います。
lead: >-
  LibreYOLOの公開Pythonインターフェースはlibreyolo/__init__.pyの__all__リストです。このページにあるものはすべてfrom
  libreyolo import <name>でインポートできます。リストにないものは内部実装です。
keywords:
  - LibreYOLO Python API
  - LibreYOLO import
  - LibreYOLO ファクトリー
  - LibreSAM
  - LibreVLM
  - LibreOpenVocab
  - LibreEnsemble
  - libreyolo __all__
last_verified: 1.5.0
verification: >-
  v1.5.0のlibreyolo/__init__.py、libreyolo/models/__init__.py、libreyolo/models/base/model.py、libreyolo/models/base/inference.py、libreyolo/models/sam/model.py、libreyolo/models/vlm/__init__.py、libreyolo/models/openvocab/__init__.py、libreyolo/ensemble/model.pyから名前とシグネチャを確認しました。
snippets:
  usage:
    - label: 1つのファクトリーで任意のモデルを読み込む
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        # 単一の画像ソースでは1つのResultsを返し、リストやディレクトリでは
        # Resultsのリストを返します。
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
        print(result.names)
    - label: ファミリークラスを直接インポート
      language: python
      code: |
        from libreyolo import LibreYOLO9, SAMPLE_IMAGE

        model = LibreYOLO9("LibreYOLO9t.pt", size="t")
        result = model(SAMPLE_IMAGE)

        print(len(result))
  factories:
    - label: 5つのエントリーポイント
      language: python
      code: >
        from libreyolo import LibreYOLO, LibreEnsemble


        # プロンプト不要のファミリーを重みの内容から判別するファクトリーです。

        detector = LibreYOLO("LibreYOLO9t.pt")


        # 2つ以上の検出器を1つの予測インターフェースにまとめます。

        ens = LibreEnsemble(["LibreYOLO9t.pt", "LibreYOLO9s.pt"])


        # 残りの3つのファクトリーには追加パッケージが必要です。

        #   pip install 'libreyolo[sam]'        -> from libreyolo import
        LibreSAM

        #   pip install 'libreyolo[vlm]'        -> from libreyolo import
        LibreVLM

        #   pip install 'libreyolo[openvocab]'  -> from libreyolo import
        LibreOpenVocab

        print(type(detector).__name__, ens.fusion)
source_hash: 66e34e78b2e0fb2d
---

## エントリーポイント

5つのcallableがモデルを読み込みます。アーキテクチャではなく、呼び出し規約によって分かれています。

| ファクトリー | 読み込む対象 | 呼び出し時のプロンプト | 必要な追加パッケージ |
|---|---|---|---|
| `LibreYOLO` | チェックポイントまたはファイルサフィックスから判別する、プロンプト不要のファミリー | | |
| `LibreSAM` | サイズエイリアスで指定するプロンプト可能セグメンター | 点、ボックス、概念テキスト | `sam` |
| `LibreVLM` | エイリアスで指定する生成型視覚言語検出器 | クラス語彙または自由形式のプロンプト | `vlm` |
| `LibreOpenVocab` | エイリアスで指定するテキスト条件付き検出器 | クラス語彙 | `openvocab` |
| `LibreEnsemble` | 1つのインターフェースへ融合する2つ以上の検出器 | | |

<code-tabs name="factories" />

ファイルを読み取るのは`LibreYOLO`だけです。ほかの3つは文字列エイリアスを受け取り、Hugging Faceリポジトリへ解決します。このため、引数はパスではなくモデル名です。

```python
LibreYOLO(
    model_path: str,
    size: str | None = None,
    reg_max: int = 16,
    nb_classes: int | None = None,
    device: str = "auto",
    task: str | None = None,
    compute_units: str = "all",
)
```

`model_path`は、`.pt`チェックポイント、ONNXの`.onnx`ファイル、ExecuTorchの`.pte`、MNNの`.mnn`、TensorRTの`.engine`、OpenVINO、Paddle、ncnnのディレクトリ、またはTritonのHTTPかHTTPSモデルURLを受け付けます。`size`と`nb_classes`は省略するとチェックポイントから読み取られます。`compute_units`はCoreMLの`.mlpackage`読み込み時だけ参照され、`all`、`cpu_only`、`cpu_and_gpu`、`cpu_and_ne`のいずれかです。`task`には`libreyolo.tasks.TASKS`にある正規タスク名を指定できます。

<code-tabs name="usage" />

## ファミリークラス

ファクトリーが返せるすべてのファミリーは名前でもエクスポートされるため、チェックポイントが事前に分かっている場合はクラスを直接構築できます。コンストラクターは`BaseModel.__init__`に従います。

```python
Family(model_path, size, nb_classes=80, device="auto", task=None, **kwargs)
```

ファミリークラスの`size`にはデフォルト値がありません。これがファクトリーとの違いです。YOLO9とそのバリアントでは、`size`の後に`reg_max: int = 16`が入ります。

物体検出およびマルチタスクファミリーは、`LibreYOLO9`、`LibreYOLO9E2E`、`LibreYOLO9P2`、`LibreYOLONAS`、`LibreYOLOX`、`LibreYOLO7`、`LibreYOLO4`、`LibreYOLO3`、`LibreYOLO2`、`LibreYOLO1`、`LibreRTDETR`、`LibreRTDETRv2`、`LibreRTDETRv4`、`LibreRFDETR`、`LibreDFINE`、`LibreDOMEDETR`、`LibreDEIM`、`LibreDEIMv2`、`LibreDETR`、`LibreDeformableDETR`、`LibreDINODETR`、`LibreLWDETR`、`LibreMaskRCNN`、`LibreFCOS`、`LibreFasterRCNN`、`LibreRetinaNet`、`LibreSSD`、`LibreCenterNet`、`LibreEfficientDet`、`LibreEC`、`LibrePICODET`、`LibreRTMDet`、`LibreFOMO`です。

密な予測のファミリーは、`LibreMiDaS`、`LibreDepthAnythingV2`、`LibreDepthAnything3`、`LibreZipDepth`、`LibreMoGe2`、`LibreTEED`、`LibreDexiNed`、`LibreNAFNet`、`LibreRealESRGAN`、`LibreSwinIR`、`LibreBiRefNet`、`LibreFeyNobg`、`LibreFCN`、`LibreEoMT`、`LibreDeepLabv3`、`LibrePIDNet`、`LibreSegformer`、`LibreLingBotVision`です。

画像分類および埋め込みのファミリーは、`LibreViT`、`LibreMobileNetV4`、`LibreConvNeXt`、`LibreDeiT`、`LibreSwin`、`LibreEfficientNetV2`、`LibreVGG`、`LibreResNet`、`LibreAlexNet`、`LibreCLIP`、`LibreSigLIP2`、`LibreDINOv2`です。

その他のタスクには、`LibreHRNet`（姿勢推定）、`LibreL2CS`（視線推定）、`LibrePPOCR`（OCR）、`LibreFaceEmbedder`（埋め込み）があります。

兄弟レベルでもファミリークラスをエクスポートします。`LibreSAM1`、`LibreSAM2`、`LibreSAM3`、`LibreEdgeTAM`、`LibreMobileSAM`、`LibrePicoSAM3`、`LibreGroundingDINO`、`LibreOWLv2`、`LibreOMDetTurbo`、`LibreLFM2VL`、`LibreQwen3VL`、`LibreSmolVLM2`、`LibreInternVL3`、`LibreFlorence2`、`LibreKosmos2`、`LibreLocateAnything`、`LibreMODUS`（`LibreModus`とも表記）です。

## 予測インターフェース

モデルを呼び出すと推論を実行します。`predict`は`__call__`のエイリアスなので、相互に置き換えられます。

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

単一の画像ソースでは1つの`Results`を返します。リスト、タプル、ディレクトリではリストを返し、`stream=True`ではジェネレーターを返します。モデルオブジェクトのその他のメソッドは[モデルAPIページ](/docs/reference/model-api)に記載されています。

## Resultsペイロード

`Results`と18個のペイロードクラスがパッケージレベルでエクスポートされます。`Results`、`Boxes`、`Masks`、`Keypoints`、`Points`、`Probs`、`OBB`、`Gaze`、`SemanticMask`、`PanopticSegmentation`、`DepthMap`、`EdgeMap`、`NormalMap`、`RestoredImage`、`Matte`、`Meshes`、`OCRRegions`、`Embeddings`、`Identities`です。それぞれの説明は[Resultsの型](/docs/reference/results-types)にあります。

## バックエンド

エクスポート済みアーティファクトはファイルサフィックスに基づいて`LibreYOLO()`から読み込むため、バックエンドクラスを手動で構築することはほとんどありません。バックエンドを明示的に選ぶ必要がある場合のため、`OnnxBackend`、`OpenVINOBackend`、`PaddleBackend`、`TensorRTBackend`、`TritonBackend`、`NcnnBackend`、`CoreMLBackend`、`create_triton_config`がエクスポートされています。`BaseExporter`は`model.export()`の背後にあるエクスポーターレジストリーです。

## バリデーター

`model.val()`はタスクに応じて適切なバリデーターへディスパッチします。直接使用したりサブクラス化したりできるよう、`DetectionValidator`、`SegmentationValidator`、`PoseValidator`、`SemanticValidator`、`PanopticValidator`、`DepthValidator`、`NormalValidator`、`EdgeValidator`、共有の`ValidationConfig`がエクスポートされています。

## トラッキング

`model.track()`は名前でトラッカーを選択します。トラッカークラスと構成用データクラスもエクスポートされています。`ByteTracker`と`TrackConfig`、`BoTSortTracker`と`BoTSortConfig`、`OCSortTracker`と`OCSortConfig`です。

## データヘルパー

`DATASETS_DIR`は解決済みのデータセットルート、`load_data_config`はデータセットYAMLを読み込む関数、`check_dataset`はデータセットを検証する関数です。[データセット形式](/docs/reference/dataset-formats)に記載されたタスク別ローダーは、パッケージレベルではなく`libreyolo.data`にあります。

## ギャラリーと蒸留

`Gallery`と`FaceGallery`は`embed`タスク用に登録された識別ベクトルを保持し、`Identities`ペイロードを生成します。`Distiller`と`get_distill_config`は教師生徒学習を実行します。

## アセット

`SAMPLE_IMAGE`はパッケージに同梱された画像への絶対パスです。このため、ドキュメント内の各スニペットは最初に画像をダウンロードせず実行できます。

## 遅延インポートと名称変更されたクラス

兄弟レベルの名前、バックエンド、バリデーター、データヘルパーの多くは、モジュールレベルの`__getattr__`を通じて解決されます。このため、`libreyolo`をインポートしてもそれらの依存関係はインポートされません。必要な追加パッケージがない場合も、明確なメッセージとともにインポートに失敗します。

2つのクラス名が変更されましたが、古い表記も`DeprecationWarning`とともに引き続き解決されます。`LibreYOLORTDETR`は`LibreRTDETR`に、`LibreYOLORFDETR`は`LibreRFDETR`になりました。
