---
title: API modelu
seo_title: Metody i sygnatury obiektu modelu LibreYOLO
description: >-
  Wszystkie metody wczytanego modelu LibreYOLO: predict, embed, track, val,
  train, export, save, quantize, info i sterowanie grafami CUDA, wraz z
  rzeczywistymi wartościami domyślnymi.
lead: >-
  Wczytany model LibreYOLO jest instancją BaseModel. Na tej stronie wymieniono
  metody tej instancji wraz z sygnaturami i wartościami domyślnymi odczytanymi z
  libreyolo/models/base/model.py.
keywords:
  - metody modelu LibreYOLO
  - argumenty predict LibreYOLO
  - argumenty val LibreYOLO
  - argumenty export LibreYOLO
  - model.track
  - model.quantize
  - capture_graph
last_verified: 1.5.0
verification: >-
  Sygnatury i wartości domyślne odczytano z libreyolo/models/base/model.py i
  libreyolo/models/base/inference.py w wersji 1.5.0. Klasy rodzin mogą je
  zawężać lub rozszerzać. train() jest definiowane dla każdej rodziny, dlatego
  opisano tu tylko wspólną otoczkę cfg=.
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
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9t.pt")


        # stream=True zwraca generator, po jednym obiekcie Results na klatkę lub
        obraz.

        for result in model([SAMPLE_IMAGE, SAMPLE_IMAGE], stream=True):
            print(len(result))
source_hash: da0776970ded8716
---

## Tworzenie

Fabryka zwraca instancję klasy rodziny. Bezpośrednie utworzenie tej klasy
przyjmuje te same argumenty, z wyjątkiem wymaganego `size`:

```python
Family(model_path, size, nb_classes=80, device="auto", task=None, **kwargs)
```

`device="auto"` wybiera CUDA, gdy jest dostępne, następnie MPS, a na końcu CPU.
Liczba całkowita lub ciąg cyfr jest interpretowany jako numer porządkowy CUDA,
więc zarówno `device=0`, jak i `device="0"` oznacza `cuda:0`. `task` jest
walidowane względem `SUPPORTED_TASKS` rodziny. Przekazanie `model_path=None`
buduje architekturę i pozostawia ją w trybie trenowania, a przekazanie `dict`
wczytuje bezpośrednio ten słownik stanu.

## predict i \_\_call\_\_

`predict` jest aliasem `__call__`.

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

| Argument | Wartość domyślna | Znaczenie |
|---|---|---|
| `source` | `None` | Obraz, lista lub krotka obrazów w pamięci, katalog, plik wideo albo źródło ekranu, takie jak `"screen"`, `"screen 1"`, `"screen 1 100 200 512 256"` |
| `conf` | `0.25` | Próg pewności |
| `iou` | `0.45` | Próg IoU dla NMS |
| `imgsz` | `None` | Nadpisanie rozmiaru wejścia; `None` używa natywnego rozmiaru modelu |
| `device` | `None` | Nadpisanie urządzenia dla tego wywołania |
| `classes` | `None` | Zachowanie tylko tych identyfikatorów klas |
| `max_det` | `300` | Maksymalna liczba detekcji na obraz |
| `augment` | `False` | Augmentacja podczas testu |
| `save` | `False` | Zapis obrazu lub wideo z adnotacjami |
| `batch` | `1` | Liczba obrazów na przebieg w przód dla źródeł będących katalogiem lub listą |
| `stream` | `False` | Zwracanie generatora zamiast zmaterializowanej listy |
| `stream_buffer` | `False` | Zachowanie każdej przechwyconej klatki transmisji zamiast tylko najnowszej |
| `vid_stride` | `1` | Przetwarzanie co N-tej klatki wideo lub ekranu |
| `show` | `False` | Wyświetlanie klatek z adnotacjami w oknie |
| `output_path` | `None` | Ścieżka wyjściowa przy `save=True` |
| `color_format` | `"auto"` | Wskazówka formatu kolorów dla tablic w pamięci |
| `tiling` | `False` | Inferencja kafelkowa dla dużych obrazów |
| `overlap_ratio` | `0.2` | Współczynnik nakładania kafelków |
| `output_file_format` | `None` | `"jpg"`, `"png"` lub `"webp"` |
| `cuda_graph` | `False` | `True` przechwytuje graf przy pierwszym użyciu każdego kształtu wejścia, `"auto"` czeka na powtórzenie kształtu |

Pojedyncze źródło obrazu zwraca jeden `Results`. Lista, krotka lub katalog
zwraca ich listę, a `stream=True` w każdym przypadku zwraca generator.

Źródła transmisji na żywo są nieograniczone i wymagają `stream=True`. Nie można
łączyć `tiling` z `augment`. Augmentacja podczas testu zgłasza błąd dla zadań
`embed`, `point` i `edge`.

<code-tabs name="usage" />

Przy `batch > 1` rodziny, których `SUPPORTS_BATCHED_PREDICT` ma wartość true,
wykonują jeden przebieg w przód na stos dla każdego fragmentu. `batch=1`
zachowuje jeden przebieg w przód na obraz.

<code-tabs name="stream" />

## embed

```python
model.embed(source=None, **kwargs) -> torch.Tensor
```

Wygodna otoczka `predict`, która układa każdy wiersz embeddingu w jeden tensor
`(N_total, D)`. Model musi być utworzony z `task="embed"`, w przeciwnym razie
zgłaszany jest `NotImplementedError`.

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

Zwraca kolejno po jednym `Results` na klatkę z ustawionym `track_id`. `tracker`
przyjmuje `"bytetrack"`, `"botsort"`, `"ocsort"` lub `"deepocsort"` i jest
ignorowane, gdy podano `tracker_config`, ponieważ typ konfiguracji wybiera
tracker. `track_conf` jest mapowane na `track_high_thresh` dla ByteTrack
i BoT-SORT oraz na `det_thresh` dla OC-SORT i Deep OC-SORT. `output_path`
domyślnie przyjmuje `runs/track/<video_stem>.mp4`.

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

Zwraca słownik metryk, którego klucze zależą od zadania. Detekcja zwraca
`metrics/precision`, `metrics/recall`, `metrics/mAP50` i `metrics/mAP50-95`.
`imgsz` przyjmuje liczbę całkowitą dla kwadratu lub krotkę `(height, width)`,
a domyślnie używa natywnego rozmiaru wejścia modelu. `plots` jest aliasem
`save_plots`. `allow_download_scripts` steruje osadzonym kodem Pythona, który
plik YAML zbioru danych może zawierać w polu `download`.

`faster_coco_eval` jest przyjmowane przez `**kwargs` i domyślnie ma wartość
`True`, a jeśli pakiet nie jest zainstalowany, używane jest pycocotools.
Uruchomiony backend jest raportowany w `model.last_eval_backend`.

Walidacja z augmentacją zgłasza błąd dla zadań `obb` i `pose`.

## train

`train` jest definiowane osobno dla każdej rodziny, dlatego jego argumenty się
różnią. Dwa zachowania są wspólne, ponieważ klasa bazowa otacza metodę `train`
każdej rodziny:

- `cfg=` przyjmuje ścieżkę YAML, której klucze są scalane z wywołaniem. Jawne argumenty nazwane mają pierwszeństwo przed plikiem.
- `pretrained=False` dla rodziny w grupie pokrycia `g0` lub `g1` ponownie inicjalizuje model od zera przed trenowaniem i nie można go łączyć z `resume=True`.

Obsługa parametrów augmentacji zależy od rodziny. Zobacz
[macierz augmentacji](/docs/reference/augmentation-matrix).

## export

```python
model.export(format="onnx", **kwargs) -> str
```

Zwraca ścieżkę zapisanego artefaktu. `format` jest rozwiązywane przez rejestr
eksporterów, gdzie `engine` jest aliasem `tensorrt`, a `litert` aliasem
`tflite`. Argumenty wspólne dla wszystkich eksporterów:

| Argument | Wartość domyślna | Znaczenie |
|---|---|---|
| `output_path` | `None` | Ścieżka pliku wyjściowego; w razie pominięcia generowana w `weights/` |
| `imgsz` | `None` | Krotka `(height, width)` lub pojedyncza liczba całkowita; domyślnie natywny rozmiar |
| `opset` | `None` | Wersja opset ONNX |
| `simplify` | `True` | Uruchomienie upraszczania grafu ONNX |
| `dynamic` | `True` | Włączenie osi dynamicznych |
| `half` | `False` | Precyzja FP16 |
| `int8` | `False` | Precyzja INT8 |
| `batch` | `1` | Rozmiar batcha zapisany w artefakcie |
| `device` | `None` | Urządzenie używane do śledzenia |
| `data` | `None` | data.yaml do kalibracji INT8 |
| `fraction` | `1.0` | Część zbioru kalibracyjnego do użycia |
| `allow_download_scripts` | `False` | Zezwolenie na osadzony kod Pythona podczas pobierania zbioru przez YAML |
| `verbose` | `False` | Szczegółowe komunikaty eksportera |

Zablokowane kombinacje zgłaszają `NotImplementedError` w kontroli wstępnej,
przed śledzeniem. Pokrycie i jego reguły opisano na stronie
[macierzy eksportu](/docs/reference/export-matrix). Jeśli występują aktywne
adaptery LoRA, są scalane w gęste wagi, a scalenie następuje dopiero po
odrzuceniu wszystkich nieprawidłowych żądań.

## save

```python
model.save(path) -> str
```

Zapisuje checkpoint LibreYOLO zgodny ze schematem v1.0: słownik stanu wraz
z metadanymi opisanymi w [schemacie checkpointu](/docs/reference/checkpoint-schema).
Skwantyzowany model dodatkowo zawiera manifest `quant`, dzięki czemu
`LibreYOLO(path)` odtwarza skwantyzowaną strukturę i skale.

## quantize, quant_info i dequantize

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

Kwantyzuje w miejscu i zwraca model. `recipe` jest jednym z rzutowań `fp16`
i `bf16`, receptur `int8` i `fp8` dla Conv i Linear albo receptur `w4a16`,
`w4a8`, `nvfp4`, `mxfp4` i `int2` tylko dla Linear, obsługiwanych przez rodziny
transformerowe, takie jak RF-DETR. `int2` wymaga QAT. `calib` przyjmuje ścieżkę
data.yaml lub nazwę wbudowanego zbioru danych i odczytuje obrazy tylko w przód.
Etykiety nigdy nie są odczytywane. Aby pominąć kalibrację, przekaż
`calib=None`. `algorithm` przyjmuje `"minmax"`, `"percentile"` lub `"auto"`.

`model.quant_info()` zwraca podsumowanie stanu kwantyzacji albo `None` dla
modelu zmiennoprzecinkowego. `model.dequantize()` odtwarza moduły
zmiennoprzecinkowe w miejscu, zachowując główne wagi wytrenowane z kwantyzacją.
Stanowi to pomost od QAT do `export(format="onnx", int8=True, data=...)`.

## info i layers

```python
model.info(detailed=False, verbose=True) -> Dict[str, Any]
model.get_available_layer_names() -> List[str]
model.get_distill_config() -> Dict
```

`info` zwraca słownik zgodny z JSON i zapisuje czytelne dla człowieka
podsumowanie, gdy `verbose` ma wartość true. `get_available_layer_names`
wymienia warstwy, które może wskazać konfiguracja destylacji lub ekstrakcji cech.

## Grafy CUDA

Dostępne dla rodzin, których atrybut klasowy `SUPPORTS_CUDA_GRAPH` ma wartość
true. Odtworzenie jest identyczne bitowo z wykonaniem eager.

```python
model.capture_graph(imgsz=None, batch=1, dtype=None) -> None
model.cuda_graph_scope(mode=True)          # menedżer kontekstu
model.graph_info() -> Dict[str, Any]
model.release_graphs() -> None
```

Przechwycony graf jest poprawny tylko dla dokładnego kształtu, z którym został
przechwycony, więc `batch` i `imgsz` muszą odpowiadać późniejszemu wywołaniu
`predict`. `capture_graph` przenosi koszt przechwycenia poza pierwsze żądanie.
`mode` przyjmuje `True` lub `"on"`, aby przechwycić graf przy pierwszym użyciu,
`"auto"`, aby czekać na powtórzenie kształtu, oraz `False`, aby nic nie robić.
`capture_graph` zgłasza `NotImplementedError`, gdy rodzina nie zadeklarowała
obsługi, oraz `CudaGraphUnavailable`, gdy przechwytywanie się nie powiedzie.

## Urządzenie i dtype

Obiekty `Results` mają metody `.to()`, `.cpu()`, `.cuda()` i `.numpy()`.
Zobacz [typy Results](/docs/reference/results-types). Sam model jest przenoszony
przez przekazanie `device=` do `predict` albo podczas tworzenia.
