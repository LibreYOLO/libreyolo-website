---
title: API Python
seo_title: Referensi API Python LibreYOLO
description: >-
  Nama yang diekspor LibreYOLO pada level paket: lima factory, kelas family,
  payload Results, backend, validator, tracker, dan helper data.
lead: >-
  Antarmuka Python publik LibreYOLO adalah list __all__ dalam
  libreyolo/__init__.py. Semua yang ada di halaman ini dapat diimpor dengan from
  libreyolo import <name>; semua yang tidak ada dalam list tersebut bersifat
  internal.
keywords:
  - api python libreyolo
  - import libreyolo
  - factory LibreYOLO
  - LibreSAM
  - LibreVLM
  - LibreOpenVocab
  - LibreEnsemble
  - libreyolo __all__
last_verified: 1.5.0
verification: >-
  Nama dan signature dibaca dari libreyolo/__init__.py,
  libreyolo/models/__init__.py, libreyolo/models/base/model.py,
  libreyolo/models/base/inference.py, libreyolo/models/sam/model.py,
  libreyolo/models/vlm/__init__.py, libreyolo/models/openvocab/__init__.py, dan
  libreyolo/ensemble/model.py pada v1.5.0.
snippets:
  usage:
    - label: Muat apa pun melalui satu factory
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        # Sumber satu gambar mengembalikan satu Results; list atau direktori
        # mengembalikan list Results.
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
        print(result.names)
    - label: Impor kelas family secara langsung
      language: python
      code: |
        from libreyolo import LibreYOLO9, SAMPLE_IMAGE

        model = LibreYOLO9("LibreYOLO9t.pt", size="t")
        result = model(SAMPLE_IMAGE)

        print(len(result))
  factories:
    - label: Lima entry point
      language: python
      code: >
        from libreyolo import LibreYOLO, LibreEnsemble


        # Factory dengan sniffing bobot untuk family tanpa prompt.

        detector = LibreYOLO("LibreYOLO9t.pt")


        # Dua detektor atau lebih di balik satu antarmuka prediksi.

        ens = LibreEnsemble(["LibreYOLO9t.pt", "LibreYOLO9s.pt"])


        # Tiga factory lainnya memerlukan ekstra yang diinstal:

        #   pip install 'libreyolo[sam]'        -> from libreyolo import
        LibreSAM

        #   pip install 'libreyolo[vlm]'        -> from libreyolo import
        LibreVLM

        #   pip install 'libreyolo[openvocab]'  -> from libreyolo import
        LibreOpenVocab

        print(type(detector).__name__, ens.fusion)
source_hash: 66e34e78b2e0fb2d
---

## Entry point

Lima callable memuat model. Pemisahannya berdasarkan kontrak pemanggilan, bukan
arsitektur.

| Factory | Memuat | Prompt saat pemanggilan | Ekstra yang diperlukan |
|---|---|---|---|
| `LibreYOLO` | Family tanpa prompt, dengan sniffing checkpoint atau suffix berkas | | |
| `LibreSAM` | Segmenter yang dapat menerima prompt, berdasarkan alias ukuran | Titik, bounding box, atau teks konsep | `sam` |
| `LibreVLM` | Detektor vision-language generatif, berdasarkan alias | Vocabulary kelas atau prompt bebas | `vlm` |
| `LibreOpenVocab` | Detektor yang dikondisikan oleh teks, berdasarkan alias | Vocabulary kelas | `openvocab` |
| `LibreEnsemble` | Dua detektor atau lebih yang digabungkan menjadi satu antarmuka | | |

<code-tabs name="factories" />

`LibreYOLO` adalah satu-satunya yang membaca berkas. Tiga factory lain menerima
alias string dan me-resolve-nya ke repositori Hugging Face, sehingga argumennya
berupa nama model, bukan path.

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

`model_path` menerima checkpoint `.pt`, berkas ONNX `.onnx`, ExecuTorch `.pte`,
MNN `.mnn`, TensorRT `.engine`, direktori OpenVINO, Paddle, atau ncnn, maupun
URL model HTTP atau HTTPS Triton. `size` dan `nb_classes` dibaca dari checkpoint
ketika tidak diberikan. `compute_units` hanya dibaca untuk pemuatan `.mlpackage`
CoreML dan nilainya salah satu dari `all`, `cpu_only`, `cpu_and_gpu`,
`cpu_and_ne`. `task` menerima nama task kanonis dari `libreyolo.tasks.TASKS`.

<code-tabs name="usage" />

## Kelas family

Setiap family yang dapat dikembalikan factory juga diekspor berdasarkan nama,
sehingga kelas dapat dibuat secara langsung ketika checkpoint telah diketahui.
Constructor mengikuti `BaseModel.__init__`:

```python
Family(model_path, size, nb_classes=80, device="auto", task=None, **kwargs)
```

`size` tidak memiliki default pada kelas family, yang membedakannya dari
factory. YOLO9 dan variannya menyisipkan `reg_max: int = 16` setelah `size`.

Family deteksi dan multi-task: `LibreYOLO9`, `LibreYOLO9E2E`,
`LibreYOLO9P2`, `LibreYOLONAS`, `LibreYOLOX`, `LibreYOLO7`, `LibreYOLO4`,
`LibreYOLO3`, `LibreYOLO2`, `LibreYOLO1`, `LibreRTDETR`, `LibreRTDETRv2`,
`LibreRTDETRv4`, `LibreRFDETR`, `LibreDFINE`, `LibreDOMEDETR`, `LibreDEIM`,
`LibreDEIMv2`, `LibreDETR`, `LibreDeformableDETR`, `LibreDINODETR`,
`LibreLWDETR`, `LibreMaskRCNN`, `LibreFCOS`, `LibreFasterRCNN`,
`LibreRetinaNet`, `LibreSSD`, `LibreCenterNet`, `LibreEfficientDet`,
`LibreEC`, `LibrePICODET`, `LibreRTMDet`, `LibreFOMO`.

Family dense prediksi: `LibreMiDaS`, `LibreDepthAnythingV2`,
`LibreDepthAnything3`, `LibreZipDepth`, `LibreMoGe2`, `LibreTEED`,
`LibreDexiNed`, `LibreNAFNet`, `LibreRealESRGAN`, `LibreSwinIR`,
`LibreBiRefNet`, `LibreFeyNobg`, `LibreFCN`, `LibreEoMT`, `LibreDeepLabv3`,
`LibrePIDNet`, `LibreSegformer`, `LibreLingBotVision`.

Family classification dan embedding: `LibreViT`, `LibreMobileNetV4`,
`LibreConvNeXt`, `LibreDeiT`, `LibreSwin`, `LibreEfficientNetV2`, `LibreVGG`,
`LibreResNet`, `LibreAlexNet`, `LibreCLIP`, `LibreSigLIP2`, `LibreDINOv2`.

Task lain: `LibreHRNet` (pose), `LibreL2CS` (gaze), `LibrePPOCR` (ocr),
`LibreFaceEmbedder` (embed).

Tingkat saudara juga mengekspor kelas family-nya: `LibreSAM1`, `LibreSAM2`,
`LibreSAM3`, `LibreEdgeTAM`, `LibreMobileSAM`, `LibrePicoSAM3`;
`LibreGroundingDINO`, `LibreOWLv2`, `LibreOMDetTurbo`; `LibreLFM2VL`,
`LibreQwen3VL`, `LibreSmolVLM2`, `LibreInternVL3`, `LibreFlorence2`,
`LibreKosmos2`, `LibreLocateAnything`, `LibreMODUS` (juga ditulis
`LibreModus`).

## Antarmuka prediksi

Memanggil model menjalankan inferensi. `predict` adalah alias untuk `__call__`,
sehingga keduanya dapat saling menggantikan.

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

Sumber satu gambar mengembalikan satu `Results`. List, tuple, atau direktori
mengembalikan list `Results`, sedangkan `stream=True` mengembalikan generator.
Metode lain pada objek model didokumentasikan di
[halaman API model](/docs/reference/model-api).

## Muatan Results

`Results` dan delapan belas kelas payload-nya diekspor pada level paket:
`Results`, `Boxes`, `Masks`, `Keypoints`, `Points`, `Probs`, `OBB`, `Gaze`,
`SemanticMask`, `PanopticSegmentation`, `DepthMap`, `EdgeMap`, `NormalMap`,
`RestoredImage`, `Matte`, `Meshes`, `OCRRegions`, `Embeddings`, `Identities`.
Setiap kelas dijelaskan dalam [Jenis Results](/docs/reference/results-types).

## Backend

Artefak hasil ekspor dimuat melalui `LibreYOLO()` berdasarkan suffix berkas,
sehingga kelas backend jarang dibuat secara manual. Kelas tersebut diekspor
untuk kasus ketika backend harus dipilih secara eksplisit: `OnnxBackend`,
`OpenVINOBackend`, `PaddleBackend`, `TensorRTBackend`, `TritonBackend`,
`NcnnBackend`, `CoreMLBackend`, ditambah `create_triton_config`.
`BaseExporter` adalah registry exporter di balik `model.export()`.

## Validator

`model.val()` melakukan dispatch ke validator yang tepat berdasarkan task,
sehingga kelas berikut diekspor untuk penggunaan langsung dan subclassing:
`DetectionValidator`, `SegmentationValidator`, `PoseValidator`,
`SemanticValidator`, `PanopticValidator`, `DepthValidator`, `NormalValidator`,
`EdgeValidator`, serta `ValidationConfig` bersama.

## Tracking

`model.track()` memilih tracker berdasarkan nama. Kelas tracker dan dataclass
konfigurasinya juga diekspor: `ByteTracker` dengan `TrackConfig`,
`BoTSortTracker` dengan `BoTSortConfig`, serta `OCSortTracker` dengan
`OCSortConfig`.

## Helper data

`DATASETS_DIR` adalah root dataset hasil resolve, `load_data_config` membaca YAML
dataset, dan `check_dataset` memvalidasinya. Loader khusus task yang disebutkan
dalam [Format dataset](/docs/reference/dataset-formats) berada di
`libreyolo.data`, bukan pada level paket.

## Gallery dan distillation

`Gallery` dan `FaceGallery` menyimpan vektor identitas terdaftar untuk task
`embed` dan menghasilkan payload `Identities`. `Distiller` dan
`get_distill_config` menjalankan pelatihan teacher-student.

## Aset

`SAMPLE_IMAGE` adalah path absolut ke gambar yang disertakan dalam paket,
sehingga setiap snippet dalam dokumentasi ini berjalan tanpa mengunduh gambar
terlebih dahulu.

## Lazy import dan kelas yang diganti namanya

Sebagian besar nama sibling tier, backend, validator, dan helper data diselesaikan
melalui `__getattr__` pada level modul, sehingga mengimpor `libreyolo` tidak
mengimpor dependency-nya. Import tetap gagal dengan pesan jelas ketika ekstra
yang diperlukan tidak ada.

Dua nama kelas telah diubah dan ejaan lama masih diselesaikan, disertai
`DeprecationWarning`: `LibreYOLORTDETR` kini menjadi `LibreRTDETR`, dan
`LibreYOLORFDETR` kini menjadi `LibreRFDETR`.



