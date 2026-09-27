---
title: API Pythona
seo_title: Dokumentacja API Pythona LibreYOLO
description: >-
  Nazwy eksportowane przez LibreYOLO na poziomie pakietu: pięć fabryk, klasy
  rodzin, dane Results, backendy, walidatory, trackery i funkcje pomocnicze
  danych.
lead: >-
  Publiczną powierzchnię Pythona w LibreYOLO wyznacza lista __all__ w
  libreyolo/__init__.py. Wszystko na tej stronie można importować przez from
  libreyolo import <name>. Wszystko, czego nie ma na tej liście, jest
  wewnętrzne.
keywords:
  - API Pythona LibreYOLO
  - import LibreYOLO
  - fabryka LibreYOLO
  - LibreSAM
  - LibreVLM
  - LibreOpenVocab
  - LibreEnsemble
  - libreyolo __all__
last_verified: 1.5.0
verification: >-
  Nazwy i sygnatury odczytano z libreyolo/__init__.py,
  libreyolo/models/__init__.py, libreyolo/models/base/model.py,
  libreyolo/models/base/inference.py, libreyolo/models/sam/model.py,
  libreyolo/models/vlm/__init__.py, libreyolo/models/openvocab/__init__.py i
  libreyolo/ensemble/model.py w wersji 1.5.0.
snippets:
  usage:
    - label: Wczytywanie dowolnego modelu przez jedną fabrykę
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9t.pt")


        # Pojedyncze źródło obrazu zwraca jeden obiekt Results, a lista lub
        katalog

        # zwraca ich listę.

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)

        print(result.names)
    - label: Bezpośredni import klasy rodziny
      language: python
      code: |
        from libreyolo import LibreYOLO9, SAMPLE_IMAGE

        model = LibreYOLO9("LibreYOLO9t.pt", size="t")
        result = model(SAMPLE_IMAGE)

        print(len(result))
  factories:
    - label: Pięć punktów wejścia
      language: python
      code: >
        from libreyolo import LibreYOLO, LibreEnsemble


        # Fabryka rodzin bez podpowiedzi, rozpoznająca typ po wagach.

        detector = LibreYOLO("LibreYOLO9t.pt")


        # Co najmniej dwa detektory za jednym interfejsem predykcji.

        ens = LibreEnsemble(["LibreYOLO9t.pt", "LibreYOLO9s.pt"])


        # Pozostałe trzy fabryki wymagają zainstalowania dodatku:

        #   pip install 'libreyolo[sam]'        -> from libreyolo import
        LibreSAM

        #   pip install 'libreyolo[vlm]'        -> from libreyolo import
        LibreVLM

        #   pip install 'libreyolo[openvocab]'  -> from libreyolo import
        LibreOpenVocab

        print(type(detector).__name__, ens.fusion)
source_hash: 66e34e78b2e0fb2d
---

## Punkty wejścia

Model wczytuje pięć obiektów wywoływalnych. Są rozdzielone według kontraktu
wywołania, a nie architektury.

| Fabryka | Wczytuje | Podpowiedź w czasie wywołania | Wymagany dodatek |
|---|---|---|---|
| `LibreYOLO` | Rodziny bez podpowiedzi, rozpoznawane na podstawie checkpointu lub sufiksu pliku | | |
| `LibreSAM` | Segmentatory sterowane podpowiedziami, według aliasu rozmiaru | Punkty, ramki lub tekst koncepcji | `sam` |
| `LibreVLM` | Generatywne detektory wizyjno-językowe, według aliasu | Słownik klas lub swobodna podpowiedź | `vlm` |
| `LibreOpenVocab` | Detektory warunkowane tekstem, według aliasu | Słownik klas | `openvocab` |
| `LibreEnsemble` | Co najmniej dwa detektory scalone w jeden interfejs | | |

<code-tabs name="factories" />

`LibreYOLO` jest jedyną fabryką odczytującą plik. Pozostałe trzy przyjmują alias
jako ciąg i rozwiązują go do repozytorium Hugging Face, dlatego argument jest
nazwą modelu, a nie ścieżką.

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

`model_path` przyjmuje checkpoint `.pt`, plik ONNX `.onnx`, ExecuTorch `.pte`,
MNN `.mnn`, TensorRT `.engine`, katalog OpenVINO, Paddle lub ncnn albo adres URL
modelu Triton przez HTTP lub HTTPS. W razie pominięcia `size` i `nb_classes` są
odczytywane z checkpointu. `compute_units` jest odczytywane tylko przy
wczytywaniu `.mlpackage` CoreML i przyjmuje `all`, `cpu_only`, `cpu_and_gpu`
lub `cpu_and_ne`. `task` przyjmuje dowolną kanoniczną nazwę zadania z
`libreyolo.tasks.TASKS`.

<code-tabs name="usage" />

## Klasy rodzin

Każda rodzina, którą może zwrócić fabryka, jest również eksportowana według
nazwy. Klasę można więc utworzyć bezpośrednio, gdy checkpoint jest znany
z wyprzedzeniem. Konstruktory są zgodne z `BaseModel.__init__`:

```python
Family(model_path, size, nb_classes=80, device="auto", task=None, **kwargs)
```

W klasie rodziny `size` nie ma wartości domyślnej i tym różni się ona od
fabryki. YOLO9 i jego warianty wstawiają `reg_max: int = 16` po `size`.

Rodziny detekcyjne i wielozadaniowe: `LibreYOLO9`, `LibreYOLO9E2E`,
`LibreYOLO9P2`, `LibreYOLONAS`, `LibreYOLOX`, `LibreYOLO7`, `LibreYOLO4`,
`LibreYOLO3`, `LibreYOLO2`, `LibreYOLO1`, `LibreRTDETR`, `LibreRTDETRv2`,
`LibreRTDETRv4`, `LibreRFDETR`, `LibreDFINE`, `LibreDOMEDETR`, `LibreDEIM`,
`LibreDEIMv2`, `LibreDETR`, `LibreDeformableDETR`, `LibreDINODETR`,
`LibreLWDETR`, `LibreMaskRCNN`, `LibreFCOS`, `LibreFasterRCNN`,
`LibreRetinaNet`, `LibreSSD`, `LibreCenterNet`, `LibreEfficientDet`,
`LibreEC`, `LibrePICODET`, `LibreRTMDet`, `LibreFOMO`.

Rodziny gęstej predykcji: `LibreMiDaS`, `LibreDepthAnythingV2`,
`LibreDepthAnything3`, `LibreZipDepth`, `LibreMoGe2`, `LibreTEED`,
`LibreDexiNed`, `LibreNAFNet`, `LibreRealESRGAN`, `LibreSwinIR`,
`LibreBiRefNet`, `LibreFeyNobg`, `LibreFCN`, `LibreEoMT`, `LibreDeepLabv3`,
`LibrePIDNet`, `LibreSegformer`, `LibreLingBotVision`.

Rodziny klasyfikacji i embeddingów: `LibreViT`, `LibreMobileNetV4`,
`LibreConvNeXt`, `LibreDeiT`, `LibreSwin`, `LibreEfficientNetV2`, `LibreVGG`,
`LibreResNet`, `LibreAlexNet`, `LibreCLIP`, `LibreSigLIP2`, `LibreDINOv2`.

Inne zadania: `LibreHRNet` (pose), `LibreL2CS` (gaze), `LibrePPOCR` (ocr),
`LibreFaceEmbedder` (embed).

Sąsiednie poziomy również eksportują klasy rodzin: `LibreSAM1`, `LibreSAM2`,
`LibreSAM3`, `LibreEdgeTAM`, `LibreMobileSAM`, `LibrePicoSAM3`;
`LibreGroundingDINO`, `LibreOWLv2`, `LibreOMDetTurbo`; `LibreLFM2VL`,
`LibreQwen3VL`, `LibreSmolVLM2`, `LibreInternVL3`, `LibreFlorence2`,
`LibreKosmos2`, `LibreLocateAnything`, `LibreMODUS` (również zapisywane jako
`LibreModus`).

## Interfejs predykcji

Wywołanie modelu uruchamia inferencję. `predict` jest aliasem `__call__`, więc
można używać ich zamiennie.

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

Pojedyncze źródło obrazu zwraca jeden `Results`. Lista, krotka lub katalog
zwraca ich listę, a `stream=True` zwraca generator. Pozostałe metody obiektu
modelu opisano na stronie [API modelu](/docs/reference/model-api).

## Dane wyników

`Results` i osiemnaście klas jego danych są eksportowane na poziomie pakietu:
`Results`, `Boxes`, `Masks`, `Keypoints`, `Points`, `Probs`, `OBB`, `Gaze`,
`SemanticMask`, `PanopticSegmentation`, `DepthMap`, `EdgeMap`, `NormalMap`,
`RestoredImage`, `Matte`, `Meshes`, `OCRRegions`, `Embeddings`, `Identities`.
Każdą opisano na stronie [typów Results](/docs/reference/results-types).

## Backendy

Wyeksportowane artefakty są wczytywane przez `LibreYOLO()` według sufiksu
pliku, dlatego klasy backendów rzadko tworzy się ręcznie. Są eksportowane na
potrzeby sytuacji, w których backend trzeba wybrać jawnie: `OnnxBackend`,
`OpenVINOBackend`, `PaddleBackend`, `TensorRTBackend`, `TritonBackend`,
`NcnnBackend`, `CoreMLBackend` oraz `create_triton_config`. `BaseExporter` jest
rejestrem eksporterów stojącym za `model.export()`.

## Walidatory

`model.val()` kieruje wywołanie do odpowiedniego walidatora według zadania,
dlatego poniższe klasy są eksportowane do bezpośredniego użycia i tworzenia
podklas: `DetectionValidator`, `SegmentationValidator`, `PoseValidator`,
`SemanticValidator`, `PanopticValidator`, `DepthValidator`, `NormalValidator`,
`EdgeValidator` oraz wspólna `ValidationConfig`.

## Śledzenie

`model.track()` wybiera tracker według nazwy. Eksportowane są również klasy
trackerów i dataclasses ich konfiguracji: `ByteTracker` z `TrackConfig`,
`BoTSortTracker` z `BoTSortConfig` oraz `OCSortTracker` z `OCSortConfig`.

## Funkcje pomocnicze danych

`DATASETS_DIR` jest rozwiązaną ścieżką główną zbiorów danych,
`load_data_config` odczytuje plik YAML zbioru danych, a `check_dataset` go
waliduje. Loadery właściwe dla zadań, wymienione na stronie
[formatów zbiorów danych](/docs/reference/dataset-formats), znajdują się
w `libreyolo.data`, a nie na poziomie pakietu.

## Galerie i destylacja

`Gallery` i `FaceGallery` przechowują zarejestrowane wektory tożsamości dla
zadania `embed` i tworzą dane `Identities`. `Distiller` i `get_distill_config`
sterują trenowaniem nauczyciel-uczeń.

## Zasoby

`SAMPLE_IMAGE` jest bezwzględną ścieżką obrazu dołączonego do pakietu, dzięki
czemu każdy fragment kodu w tej dokumentacji działa bez wcześniejszego
pobierania obrazu.

## Leniwe importy i klasy o zmienionych nazwach

Większość nazw sąsiednich poziomów, backendy, walidatory i funkcje pomocnicze
danych są rozwiązywane przez `__getattr__` na poziomie modułu, dzięki czemu
import `libreyolo` nie importuje ich zależności. Import nadal kończy się
czytelnym komunikatem, gdy brakuje wymaganego dodatku.

Nazwy dwóch klas zmieniono, a stara pisownia nadal jest rozwiązywana
z `DeprecationWarning`: `LibreYOLORTDETR` to obecnie `LibreRTDETR`,
a `LibreYOLORFDETR` to obecnie `LibreRFDETR`.
