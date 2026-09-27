---
title: Python API
seo_title: LibreYOLO Python APIリファレンス
description: >-
  LibreYOLOがパッケージレベルで公開する名前：5つのファクトリー、ファミリークラス、Resultsペイロード、バックエンド、バリデーター、トラッカー、データヘルパー。
lead: >-
  LibreYOLOの公開Pythonインターフェースは、libreyolo/__init__.pyの__all__リストです。このページにあるすべての名前はfrom
  libreyolo import <name>としてインポートできます。そのリストにないものは内部用です。
keywords:
  - libreyolo python api
  - libreyolo import
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

        # 単一画像のソースはResultsを1つ返し、リストまたはディレクトリは
        # Resultsのリストを返す
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


        # プロンプト不要ファミリー向けの重み判別ファクトリー

        detector = LibreYOLO("LibreYOLO9t.pt")


        # 2つ以上の検出器を1つの推論インターフェースに統合

        ens = LibreEnsemble(["LibreYOLO9t.pt", "LibreYOLO9s.pt"])


        # 残り3つのファクトリーにはextraのインストールが必要

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

5つの呼び出し可能オブジェクトがモデルを読み込みます。これらはアーキテクチャではなく、
呼び出し規約によって分かれています。

| ファクトリー | 読み込むもの | 呼び出し時のプロンプト | 必要なextra |
|---|---|---|---|
| `LibreYOLO` | チェックポイントまたはファイル拡張子を判別するプロンプト不要ファミリー | | |
| `LibreSAM` | サイズの別名で指定するプロンプト対応セグメンター | 点、ボックス、または概念テキスト | `sam` |
| `LibreVLM` | 別名で指定する生成型ビジョン言語検出器 | クラス語彙または自由形式のプロンプト | `vlm` |
| `LibreOpenVocab` | 別名で指定するテキスト条件付き検出器 | クラス語彙 | `openvocab` |
| `LibreEnsemble` | 1つのインターフェースに融合した2つ以上の検出器 | | |

<code-tabs name="factories" />

ファイルを読み取るのは`LibreYOLO`だけです。他の3つは文字列の別名を受け取り、それを
Hugging Faceリポジトリへ解決するため、引数はパスではなくモデル名です。

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

`model_path`は、`.pt`チェックポイント、ONNXの`.onnx`ファイル、ExecuTorchの`.pte`、
MNNの`.mnn`、TensorRTの`.engine`、OpenVINO、Paddle、ncnnのディレクトリ、または
TritonのHTTPかHTTPSモデルURLを受け取ります。`size`と`nb_classes`は、省略時に
チェックポイントから読み取られます。`compute_units`はCoreMLの`.mlpackage`を読み込む
場合にのみ参照され、`all`、`cpu_only`、`cpu_and_gpu`、`cpu_and_ne`のいずれかです。
`task`には`libreyolo.tasks.TASKS`の正規タスク名を指定できます。

<code-tabs name="usage" />

## ファミリークラス

ファクトリーが返せる各ファミリーは名前でも公開されているため、チェックポイントが事前に
分かっている場合はクラスを直接構築できます。コンストラクターは`BaseModel.__init__`に
従います。

```python
Family(model_path, size, nb_classes=80, device="auto", task=None, **kwargs)
```

ファミリークラスでは`size`にデフォルト値がない点が、ファクトリーとの違いです。YOLO9と
そのバリアントでは、`size`の後に`reg_max: int = 16`が入ります。

検出およびマルチタスクファミリー：`LibreYOLO9`、`LibreYOLO9E2E`、
`LibreYOLO9P2`、`LibreYOLONAS`、`LibreYOLOX`、`LibreYOLO7`、`LibreYOLO4`、
`LibreYOLO3`、`LibreYOLO2`、`LibreYOLO1`、`LibreRTDETR`、`LibreRTDETRv2`、
`LibreRTDETRv4`、`LibreRFDETR`、`LibreDFINE`、`LibreDOMEDETR`、`LibreDEIM`、
`LibreDEIMv2`、`LibreDETR`、`LibreDeformableDETR`、`LibreDINODETR`、
`LibreLWDETR`、`LibreMaskRCNN`、`LibreFCOS`、`LibreFasterRCNN`、
`LibreRetinaNet`、`LibreSSD`、`LibreCenterNet`、`LibreEfficientDet`、
`LibreEC`、`LibrePICODET`、`LibreRTMDet`、`LibreFOMO`。

密な推論ファミリー：`LibreMiDaS`、`LibreDepthAnythingV2`、
`LibreDepthAnything3`、`LibreZipDepth`、`LibreMoGe2`、`LibreTEED`、
`LibreDexiNed`、`LibreNAFNet`、`LibreRealESRGAN`、`LibreSwinIR`、
`LibreBiRefNet`、`LibreFeyNobg`、`LibreFCN`、`LibreEoMT`、`LibreDeepLabv3`、
`LibrePIDNet`、`LibreSegformer`、`LibreLingBotVision`。

分類および埋め込みベクトルのファミリー：`LibreViT`、`LibreMobileNetV4`、
`LibreConvNeXt`、`LibreDeiT`、`LibreSwin`、`LibreEfficientNetV2`、`LibreVGG`、
`LibreResNet`、`LibreAlexNet`、`LibreCLIP`、`LibreSigLIP2`、`LibreDINOv2`。

その他のタスク：`LibreHRNet`（姿勢）、`LibreL2CS`（視線）、`LibrePPOCR`（OCR）、
`LibreFaceEmbedder`（埋め込みベクトル）。

同系列の階層もファミリークラスを公開します：`LibreSAM1`、`LibreSAM2`、
`LibreSAM3`、`LibreEdgeTAM`、`LibreMobileSAM`、`LibrePicoSAM3`、
`LibreGroundingDINO`、`LibreOWLv2`、`LibreOMDetTurbo`、`LibreLFM2VL`、
`LibreQwen3VL`、`LibreSmolVLM2`、`LibreInternVL3`、`LibreFlorence2`、
`LibreKosmos2`、`LibreLocateAnything`、`LibreMODUS`（`LibreModus`という綴りも可能）。

## 推論インターフェース

モデルを呼び出すと推論を実行します。`predict`は`__call__`の別名なので、両者は同じように
使用できます。

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

単一の画像ソースは1つの`Results`を返します。リスト、タプル、またはディレクトリは
それらのリストを返し、`stream=True`はジェネレーターを返します。モデルオブジェクトの
他のメソッドは[モデルAPIページ](/docs/reference/model-api)で説明しています。

## Resultsペイロード

`Results`と18個のペイロードクラスはパッケージレベルで公開されます：
`Results`、`Boxes`、`Masks`、`Keypoints`、`Points`、`Probs`、`OBB`、`Gaze`、
`SemanticMask`、`PanopticSegmentation`、`DepthMap`、`EdgeMap`、`NormalMap`、
`RestoredImage`、`Matte`、`Meshes`、`OCRRegions`、`Embeddings`、`Identities`。
それぞれの説明は[Resultsの型](/docs/reference/results-types)にあります。

## バックエンド

エクスポートした成果物は、ファイル拡張子に応じて`LibreYOLO()`を通じて読み込まれるため、
バックエンドクラスを手動で構築することはほとんどありません。バックエンドを明示的に選択する
必要がある場合に備えて、`OnnxBackend`、`OpenVINOBackend`、`PaddleBackend`、
`TensorRTBackend`、`TritonBackend`、`NcnnBackend`、`CoreMLBackend`、および
`create_triton_config`が公開されています。`BaseExporter`は`model.export()`の背後にある
エクスポーターレジストリです。

## バリデーター

`model.val()`はタスクに応じたバリデーターへ処理を振り分けます。直接使用する場合や
サブクラス化する場合に備えて、`DetectionValidator`、`SegmentationValidator`、
`PoseValidator`、`SemanticValidator`、`PanopticValidator`、`DepthValidator`、
`NormalValidator`、`EdgeValidator`、共有の`ValidationConfig`が公開されています。

## トラッキング

`model.track()`は名前でトラッカーを選択します。トラッカークラスとその設定dataclassも
公開されます：`TrackConfig`を使う`ByteTracker`、`BoTSortConfig`を使う`BoTSortTracker`、
`OCSortConfig`を使う`OCSortTracker`です。

## データヘルパー

`DATASETS_DIR`は解決済みのデータセットルートです。`load_data_config`はデータセットYAMLを
読み込み、`check_dataset`は検証します。[データセット形式](/docs/reference/dataset-formats)で
名前を挙げたタスク固有のローダーは、パッケージレベルではなく`libreyolo.data`にあります。

## ギャラリーと蒸留

`Gallery`と`FaceGallery`は`embed`タスクに登録された人物識別ベクトルを保持し、
`Identities`ペイロードを生成します。`Distiller`と`get_distill_config`は教師・生徒学習を
駆動します。

## アセット

`SAMPLE_IMAGE`はパッケージに同梱された画像への絶対パスです。そのため、このドキュメントの
すべてのスニペットは最初に画像をダウンロードしなくても実行できます。

## 遅延インポートと名前が変更されたクラス

同系列階層のほとんどの名前、バックエンド、バリデーター、データヘルパーはモジュールレベルの
`__getattr__`を通じて解決されます。そのため、`libreyolo`をインポートしても、それらの
依存関係はインポートされません。必要なextraがない場合も、インポートは明確なメッセージと
ともに失敗します。

2つのクラス名が変更されましたが、古い綴りも`DeprecationWarning`とともに解決されます：
`LibreYOLORTDETR`は現在`LibreRTDETR`、`LibreYOLORFDETR`は現在`LibreRFDETR`です。
