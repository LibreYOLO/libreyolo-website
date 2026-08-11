---
title: Wydajność wnioskowania
seo_title: Szybsze wnioskowanie w LibreYOLO
description: >-
  Grafy CUDA, połowiczna precyzja, przetwarzanie partiami, wnioskowanie
  kafelkowe i augmentacja podczas testowania w trakcie predykcji, z
  rzeczywistymi wartościami domyślnymi i listą obsługujących je rodzin.
lead: >-
  Pięć ustawień predykcji zmienia przepustowość lub dokładność: odtwarzanie
  grafu CUDA, precyzja, przetwarzanie partiami, kafelkowanie i augmentacja
  podczas testowania. Każde dotyczy określonego zestawu rodzin, a dwa zwiększają
  koszt dokładności lub opóźnienia, zamiast go zmniejszać.
keywords:
  - grafy CUDA wnioskowanie PyTorch
  - YOLO predykcja partiami Python
  - wnioskowanie FP16
  - wnioskowanie kafelkowe małe obiekty
  - detekcja na dużych obrazach kafelki
  - augmentacja podczas testowania detekcja
  - capture_graph
  - predykcja folderu partiami
last_verified: 1.5.0
verification: >-
  Wartości domyślne argumentów pochodzą z InferenceRunner.__call__ w
  libreyolo/models/base/inference.py. API grafów CUDA pochodzi z
  BaseModel.capture_graph, graph_info, release_graphs i cuda_graph_scope w
  libreyolo/models/base/model.py; obsługę rodzin odczytano ze zmiennej klasowej
  SUPPORTS_CUDA_GRAPH. Zachowanie połowicznej precyzji pochodzi z
  NOOP_PREDICT_KWARGS w libreyolo/utils/predict_args.py, ostrzeżenia CLI w
  libreyolo/cli/commands/predict.py oraz CAST_RECIPES i SUPPORTED_FAMILIES w
  libreyolo/quant/api.py. Warunki przetwarzania partiami pochodzą z
  InferenceRunner._process_in_batches i _predict_batch. Kafelkowanie pochodzi z
  _predict_tiled i _merge_tile_detections. Augmentacja podczas testowania
  pochodzi z BaseModel._predict_augment i _merge_tta, a TTA_ENABLED, TTA_SCALES
  i TTA_FIXED_SIZE odczytano w libreyolo/models/.
snippets:
  batch:
    - label: Wnioskowanie partiami na folderze
      language: python
      code: >
        from pathlib import Path

        from PIL import Image


        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        folder = Path("batch_demo")

        folder.mkdir(exist_ok=True)

        image = Image.open(SAMPLE_IMAGE)

        for index in range(8):
            image.save(folder / f"frame_{index}.jpg")

        model = LibreYOLO("LibreYOLO9s.pt")


        # Jeden przebieg w przód dla każdego fragmentu po 4 w obsługujących go
        rodzinach.

        results = model(str(folder), batch=4)

        print(len(results), "results")
    - label: Strumieniowanie bez materializowania listy
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
    - label: Wstępne przechwycenie i odtwarzanie (wymaga CUDA)
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")

        # Zapłać raz za rozgrzewkę i przechwycenie, poza pierwszym żądaniem.
        model.capture_graph()

        result = model(SAMPLE_IMAGE, cuda_graph=True)
        print(len(result.boxes))
        print(model.graph_info())
    - label: Przechwycenie dopiero po powtórzeniu kształtu (wymaga CUDA)
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")

        # "auto" czeka, aż kształt pojawi się dwa razy, więc praca jednorazowa
        # nigdy nie ponosi kosztu przechwycenia.
        for _ in range(3):
            model(SAMPLE_IMAGE, cuda_graph="auto")

        print(model.graph_info())
        model.release_graphs()
  precision:
    - label: Instalacja dodatku eksportu
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: Eksport i ponowne ładowanie z domyślną precyzją
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        path = model.export(format="onnx")

        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)
        print(len(result.boxes))
    - label: Eksport FP16 (zbuduj i uruchom na komputerze z CUDA)
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")
        path = model.export(format="onnx", half=True)

        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)
        print(len(result.boxes))
    - label: FP16 w PyTorch przez recepturę rzutowania (wymaga CUDA)
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")

        # Receptura rzutowania nie odczytuje danych kalibracyjnych.
        model.quantize(recipe="fp16", calib=None)

        result = model(SAMPLE_IMAGE)
        print(len(result.boxes))
  tiling:
    - label: Wnioskowanie kafelkowe na dużym obrazie
      language: python
      code: >
        from PIL import Image


        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Kafelkowanie włącza się tylko wtedy, gdy obraz jest większy od
        rozmiaru wejścia.

        large = Image.open(SAMPLE_IMAGE).resize((2048, 1536))

        large.save("large.jpg")


        model = LibreYOLO("LibreYOLO9s.pt")


        result = model("large.jpg", tiling=True, overlap_ratio=0.2)

        print(result.num_tiles, "tiles", len(result.boxes), "detections")
  tta:
    - label: Augmentacja podczas testowania
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        plain = model(SAMPLE_IMAGE)
        flipped = model(SAMPLE_IMAGE, augment=True)

        print(len(plain.boxes), "->", len(flipped.boxes))
source_hash: 3914665d0e7f892c
---

## Ustawienia i ich wartości domyślne

Każda z poniższych opcji jest argumentem `predict` i domyślnie jest wyłączona.

| Argument | Wartość domyślna | Działanie |
|---|---|---|
| `batch` | `1` | Obrazy na przebieg w przód dla źródeł folderowych i list |
| `cuda_graph` | `False` | Odtwarzanie przebiegu w przód z przechwyconego grafu CUDA |
| `tiling` | `False` | Podział dużego obrazu na nakładające się kafelki |
| `overlap_ratio` | `0.2` | Nakładanie kafelków po włączeniu `tiling` |
| `augment` | `False` | Uruchamianie odbitych widoków i ich scalanie |
| `half` | | Przyjmowane, powoduje ostrzeżenie i jest ignorowane |
| `device` | `None` | Przeniesienie modelu przed predykcją |

`imgsz` również wpływa na koszt, ponieważ ustala rozdzielczość działania modelu,
ale przede wszystkim jest argumentem dokładności i należy do modelu, a nie tutaj.

## Przetwarzanie partiami

<code-tabs name="batch" />

`batch` dotyczy źródeł folderowych i list. Przy `batch=1` każdy obraz wykonuje
osobny przebieg w przód. Powyżej `1` każdy fragment jest wstępnie przetwarzany,
układany w jeden tensor, uruchamiany raz, a następnie dzielony ponownie tak, aby
istniejące przetwarzanie końcowe pojedynczego obrazu każdej rodziny otrzymało
oczekiwane dane.

Ścieżka stosu jest używana tylko po spełnieniu wszystkich warunków:

- `batch` jest większe niż `1`
- `tiling` jest wyłączone
- augmentacja podczas testowania nie jest aktywna
- rodzina ustawia `SUPPORTS_BATCHED_PREDICT`
- sieć bazowa nie jest w trybie trenowania

Ostatni warunek nie jest formalnością. Sieć w trybie trenowania normalizowałaby
ułożony fragment za pomocą statystyk partii pochodzących z wielu obrazów, przez
co obrazy w tym samym fragmencie zmieniałyby wzajemnie swoje predykcje. Takie
przebiegi pozostają sekwencyjne.

`SUPPORTS_BATCHED_PREDICT` ma domyślnie wartość true. Następujące rodziny wyłączają
tę funkcję i wykonują jeden obraz na przebieg niezależnie od `batch`: Depth
Anything V2, Depth Anything 3, EoMT, Faster R-CNN, FCOS, HRNet, L2CS-Net,
LibreMODUS, MiDaS, MoGe-2, PP-OCRv5, Real-ESRGAN, RetinaNet, SAM 3D Body,
SwinIR, YOLOv1, ZipDepth, wszystkie detektory z otwartym słownikiem i wszystkie
modele wizyjno-językowe.

Istnieje jeszcze jedna ścieżka zapasowa. Jeśli przetwarzanie wstępne nie zwraca
jednolitych tensorów `(1, C, H, W)` o zgodnym kształcie, typie danych i urządzeniu
w całym fragmencie, fragment działa sekwencyjnie zamiast tworzyć stos. Poprawność
nigdy nie zależy więc od przypadkowo jednakowego rozmiaru obrazów.

Połącz `batch` ze `stream=True` dla dużego folderu, aby wykonywać przebiegi
partiami bez przechowywania wszystkich wyników w pamięci.

## Grafy CUDA

<code-tabs name="graphs" />

Graf CUDA rejestruje przebieg w przód raz i odtwarza go jako jedno uruchomienie.
Małe detektory przeznaczają dużą część czasu partii 1 na uruchamianie kerneli,
dlatego połączenie tych uruchomień zwiększa przepustowość, a wynik odtwarzania
jest bitowo identyczny z wykonaniem natychmiastowym.

`cuda_graph` przyjmuje trzy wartości. `False` jest wartością domyślną i nic nie
robi. `True` przechwytuje przy pierwszym użyciu każdego kształtu wejścia. `"auto"`
czeka, aż kształt się powtórzy, dzięki czemu praca jednorazowa i zmieniająca
kształt nigdy nie ponosi kosztu przechwycenia.

`capture_graph(imgsz=None, batch=1, dtype=None)` przenosi ten koszt poza pierwsze
żądanie. Graf jest prawidłowy tylko dla dokładnego przechwyconego kształtu,
dlatego `batch` musi tutaj odpowiadać późniejszemu wywołaniu `predict`.

`graph_info()` raportuje przechwycone grafy, liczby odtworzeń i powód ewentualnego
powrotu do wykonania natychmiastowego. `release_graphs()` zwalnia grafy i ich
bufory statyczne.

Przechwytywanie wymaga CUDA oraz rodziny, która włączyła obsługę przez
`SUPPORTS_CUDA_GRAPH`, ponieważ potrzebuje przebiegu w przód bez pracy widocznej
dla hosta, co jest sprawdzane osobno dla każdej rodziny. Żądanie go dla rodziny,
która nie włączyła obsługi, powoduje `NotImplementedError` zamiast cichego
wykonania natychmiastowego.

Graf zapisuje adresy pamięci, a nie wartości, więc każda operacja przenosząca
parametry go usuwa. Zmiana urządzenia przez `predict(device=...)`, kwantyzacja
i dekwantyzacja unieważniają przechwycone grafy.

Pełna macierz obsługi dla rodzin, miejsca podziału i kontrakt numeryczny znajdują
się w sekcji [grafy CUDA](/docs/reference/cuda-graphs).

## Precyzja

<code-tabs name="precision" />

`half=True` podczas predykcji nic nie robi. Jest przyjmowane dla zgodności z
wierszem poleceń, powoduje ostrzeżenie o braku działania i zostaje odrzucone
przed dotarciem do którejkolwiek rodziny. Flaga `--half` interfejsu CLI wyświetla
to samo ostrzeżenie dla modelu `.pt`.

Istnieją dwie rzeczywiste drogi do niższej precyzji.

Dla wyeksportowanego artefaktu precyzję wybiera się podczas eksportu za pomocą
`export(format=..., half=True)`, a wynikowy plik jest ładowany bez zmian przez
`LibreYOLO()`.

Dla wykonania PyTorch `model.quantize(recipe="fp16")` rzutuje model na float16
i instaluje zaczepy zachowujące float32 na wejściach i wyjściach modelu. `"bf16"`
robi to samo z bfloat16. Żadne z tych rzutowań nie odczytuje danych kalibracyjnych,
więc `calib` jest dla nich ignorowane. Kwantyzacja obejmuje obecnie cztery rodziny:
YOLOv9, RF-DETR, BiRefNet i FeyNobg. Rzutowanie na CPU zapisuje ostrzeżenie o
powolnym działaniu, dlatego receptury te są przeznaczone dla GPU.

Obie ścieżki zmieniają wyniki numeryczne. Żadna nie gwarantuje identycznych
detekcji bez dodatkowych działań, dlatego przed wdrożeniem należy przeprowadzić walidację.

## Wnioskowanie kafelkowe

<code-tabs name="tiling" />

Kafelkowanie dzieli duży obraz na nakładające się kwadratowe kafelki, wykonuje
predykcję na każdym i scala wyniki. Jest to opcja dla małych obiektów na obrazach
o wysokiej rozdzielczości, gdzie zmniejszenie całego obrazu redukuje cele poniżej
poziomu rozpoznawalnego przez model.

Rozmiar kafelka jest rozmiarem wejścia modelu albo wartością `imgsz`, jeśli ją
podano, i musi być kwadratowy. `overlap_ratio` ma domyślnie wartość `0.2`.
Nakładające się kafelki są uzgadniane za pomocą tłumienia niemaksymalnego na klasę
przy progu `iou`, a scalona lista jest następnie przycinana do `max_det`. Oznacza
to, że `iou` wpływa na predykcje kafelkowe nawet w rodzinach, które nie wykonują
własnego NMS.

Kafelkowanie jest całkowicie pomijane, a nie tylko tanie, gdy obraz już się mieści.
Jeśli oba wymiary są mniejsze lub równe rozmiarowi wejścia, wykonywany jest jeden
zwykły przebieg w przód. Jest też pomijane dla klasyfikacji, segmentacji semantycznej
i zadania `embed`, które wracają do pojedynczego przebiegu, ponieważ kafelkowanie
nie ma dla nich znaczenia.

Powoduje błąd dla zadań, których danych wynikowych nie można ponownie zszyć:
masek segmentacji instancji, ramek zorientowanych, punktów, głębi, krawędzi i
normalnych. Nie można go łączyć z `augment`.

Wynik zawiera `result.tiled` i `result.num_tiles`. Przy `save=True` przebiegi
kafelkowe zapisują katalog w `runs/tiled_detections`, który zawiera każdy kafelek,
obraz z adnotacjami, wizualizację siatki oraz `metadata.json` z rozmiarem kafelka,
nakładaniem i progami. `result.tiles_path` i `result.grid_path` wskazują te pliki.

## Augmentacja podczas testowania

<code-tabs name="tta" />

`augment=True` uruchamia obraz więcej niż raz i scala detekcje za pomocą tłumienia
niemaksymalnego na klasę przy progu `iou`. Tak jak kafelkowanie sprawia to, że
`iou` ma kluczowe znaczenie dla rodzin, które w innym przypadku je ignorują.

W praktyce jest to odbicie poziome. Lista skal `TTA_SCALES` ma domyślnie jedną
skalę `1.0` i żadna dostarczana rodzina jej nie nadpisuje, dlatego każda rodzina
wykonuje dwa przebiegi: oryginalny obraz i jego odbicie. Rodziny oznaczone
`TTA_FIXED_SIZE` zmieniają rozmiar do stałego kwadratu, co i tak eliminuje wpływ
wielu skal.

Segmentacja semantyczna i panoptyczna używają innego scalania. Odbity widok jest
odwracany z powrotem, a dwa rozkłady softmax są uśredniane przed argmax zamiast
scalania w postaci ramek.

Augmentacja podczas testowania nie jest dostępna dla każdego zadania. Powoduje
błąd dla ramek zorientowanych, pozy, punktów, głębi, normalnych, krawędzi,
rekonstrukcji, OCR i modeli osadzeń, a także nie może być łączona z kafelkowaniem.

Następujące rodziny całkowicie ją wyłączają, więc `augment=True` wykonuje jeden
zwykły przebieg: BiRefNet, CenterNet, CLIP, DexiNed, FOMO, HRNet, L2CS-Net,
LibreMODUS, NAFNet, PP-OCRv5, Real-ESRGAN, RetinaNet, SAM 3D Body, SigLIP2,
SwinIR, TEED, wszystkie warianty SAM, wszystkie detektory z otwartym słownikiem
i wszystkie modele wizyjno-językowe.

## Pomiary

Ta strona nie zawiera żadnej wartości opóźnienia, ponieważ milisekunda bez
informacji o sprzęcie, środowisku uruchomieniowym, precyzji i rozmiarze partii
nie jest faktem. Wyniki zmierzone na różnych urządzeniach i środowiskach są
publikowane w serwisie [visionanalysis.org](https://www.visionanalysis.org),
a `libreyolo profile` mierzy określony model na bieżącym komputerze.
