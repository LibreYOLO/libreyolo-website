---
title: Detekcja punktów
seo_title: Detekcja i zliczanie punktów w LibreYOLO
description: >-
  Lokalizuj obiekty w LibreYOLO jako pojedyncze punkty zamiast ramek. Przewiduj
  centroidy, zliczaj obiekty, trenuj FOMO i odczytuj metryki punktów.
lead: >-
  Detekcja punktów zwraca dla każdego obiektu jedną lokalizację x, y zamiast
  ramki ograniczającej. LibreYOLO udostępnia ją jako zadanie point, a predykcja
  zawiera dla każdego obiektu jeden wiersz z x, y, klasą i pewnością.
keywords:
  - detekcja punktów python
  - zliczanie obiektów python
  - detekcja centroidów
  - lokalizacja punktów FOMO
  - liczenie obiektów na zdjęciu
  - point localization
last_verified: 1.5.0
snippets:
  predict:
    - label: Predykcja punktów i ich zliczanie
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Wagi LibreFOMO nie są pobierane automatycznie. Najpierw pobierz
        checkpoint z

        # https://huggingface.co/LibreYOLO i wczytaj go ze ścieżki lokalnej.

        model = LibreYOLO("./LibreFOMOs-point.pt")

        result = model(SAMPLE_IMAGE, save=True)


        points = result.points

        print(len(points))     # liczba obiektów

        print(points.xy)       # środki (N, 2) w pikselach oryginalnego obrazu

        print(points.cls, points.conf)
    - label: Znormalizowane współrzędne i liczby według klas
      language: python
      code: >
        from collections import Counter


        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("./LibreFOMOs-point.pt")

        result = model(SAMPLE_IMAGE)


        points = result.points.numpy()

        print(points.xyn)                          # te same środki w zakresie
        [0, 1]

        print(Counter(points.cls.astype(int).tolist()))
  train:
    - label: Trenowanie FOMO na zbiorze danych YOLO
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.train(data="my-dataset.yaml", epochs=40, batch=32, lr0=3e-4)
    - label: Predykcja wytrenowanym checkpointem
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("./LibreFOMOs-point.pt")

        results = model.train(data="my-dataset.yaml", epochs=40)


        # train() ponownie wczytuje najlepszy checkpoint do tego samego obiektu,
        więc po

        # zakończeniu wywołania model przewiduje za pomocą wytrenowanych wag.

        print(results["best_checkpoint"])

        print(model(SAMPLE_IMAGE).points.xy)
  val:
    - label: Walidacja i odczytywanie kluczy metryk
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("./LibreFOMOs-point.pt")

        metrics = model.val(data="my-dataset.yaml")


        print(metrics["metrics/precision"], metrics["metrics/recall"])

        print(metrics["metrics/f1"])

        print(metrics["metrics/mAP@[0.01:0.10]"])   # fitness

        print(metrics["metrics/MLE"])               # średni błąd lokalizacji

        print(metrics["metrics/MAE"], metrics["metrics/RMSE"])   # błąd
        zliczania
    - label: Zmiana progów odległości
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("./LibreFOMOs-point.pt")


        # Granice zakresu są częścią tekstu klucza, więc niestandardowy zakres

        # zmienia nazwy generowanych kluczy mAP.

        metrics = model.val(data="my-dataset.yaml", dist_thresholds=[0.02,
        0.05])


        print(metrics["metrics/mAP@0.02"])

        print(metrics["metrics/mAP@[0.02:0.05]"])
  export:
    - label: Eksport
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.export(format="onnx")
    - label: Uruchamianie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Fabryka wybiera ścieżkę na podstawie sufiksu pliku, więc
        wyeksportowany artefakt

        # wczytuje się jak dowolny checkpoint i zwraca ten sam obiekt Results.

        model = LibreYOLO("./LibreFOMOs-point.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.points.xy)
source_hash: 932153c8870d1c7c
---

## Definicja

Zadanie `point` lokalizuje każdy obiekt za pomocą jednej współrzędnej x, y i
klasy, bez szerokości, wysokości ani maski. Ponieważ predykcja jest płaską listą
obiektów, liczba wierszy odpowiada liczbie obiektów, dzięki czemu jest to zadanie
zliczania.

Predykcja wypełnia `result.points`, czyli strukturę `Points` opakowującą tablicę
`(N, 4)` z wierszami `x, y, class, confidence` w pikselach oryginalnego obrazu.
Pole `.xy` zwraca współrzędne, `.xyn` te same współrzędne podzielone przez
rozmiar obrazu, `.cls` indeksy klas, a `.conf` wyniki. Funkcja `len()` zwraca
liczbę punktów. Pole `result.boxes` pozostaje puste, więc `iou` i `max_det` nie
mają na co działać.

## Modele

Zadanie `point` obsługują trzy rodziny, których nie można stosować zamiennie.

[FOMO](/docs/models/fomo) jest opcją ze stałym słownikiem. To klasyfikator
siatkowy, który oznacza każdą komórkę siatki o niskiej rozdzielczości jako tło
lub środek obiektu. Jest jedyną rodziną punktową, którą LibreYOLO może trenować,
i jedyną, którą można eksportować.

[LocateAnything](/docs/models/locate-anything) przyjmuje tekst zamiast indeksu
klasy, więc słownikiem jest dowolna podana fraza. Wymaga dodatku `vlm`, jest
tworzony jako `LibreLocateAnything`, a nie przez fabrykę `LibreYOLO()`, a jego
wagi są ograniczone do użytku niekomercyjnego. Dokładne warunki oraz dwie
dodatkowe licencje składające się na checkpoint znajdują się na stronie modelu.

[SenseNova-Vision](/docs/models/sensenova-vision) realizuje zadanie `point` za
pomocą tego samego checkpointu generowania sterowanego promptem, którego używa
do sześciu innych zadań. Wczytuje się go przez
`LibreVLM("sensenova-vision", task="point")`. Wymaga dodatku `sensenova`, a każda
predykcja jest przebiegiem generowania na modelu 7B, dlatego należy oczekiwać
wyraźnie większego opóźnienia na obraz niż w wyspecjalizowanym detektorze. Jego
wagi są przeznaczone do użytku niekomercyjnego. Informacje o licencji znajdują
się na stronie modelu.

## Predykcja

Wagi LibreFOMO są jedynym wyjątkiem od automatycznego pobierania w tej witrynie.
`LibreYOLO("LibreFOMOs-point.pt")` szuka tego pliku na dysku i zamiast go pobrać,
zgłasza `ValueError` z jego nazwą. Najpierw należy pobrać checkpoint z
[organizacji LibreYOLO](https://huggingface.co/LibreYOLO) w Hugging Face i
wczytać go ze ścieżki lokalnej albo wytrenować własny.

<code-tabs name="predict" />

Nazwa pliku musi zawierać sufiks zadania `-point`, aby moduł wczytujący go
rozpoznał. `predict(..., nms_radius=1)` określa minimalną odległość w komórkach
siatki między dwiema detekcjami FOMO, aby obie zostały zachowane. Informacje o
źródłach, streamingu i obsłudze wyników znajdują się w sekcji
[predykcja](/docs/predict).

## Format zbioru danych

Zadanie `point` nie ma własnego formatu etykiet. Rodziny punktowe odczytują
standardowy układ detekcji YOLO i wyprowadzają jeden środek z każdego wiersza
ramki, więc `cx cy` określa punkt, a `w h` decyduje jedynie o poprawności wiersza.

```text
dataset/
  data.yaml
  images/
    train/scene.jpg
    val/scene.jpg
  labels/
    train/scene.txt
    val/scene.txt
```

Każdy plik etykiet zawiera jeden wiersz na obiekt ze znormalizowanymi
współrzędnymi:

```text
<class_id> <cx> <cy> <w> <h>
```

```yaml
path: dataset
train: images/train
val: images/val
nc: 1
names: {0: seedling}
```

Brakujący lub pusty plik etykiet oznacza brak obiektów. Pełny kontrakt opisano w
sekcji [formaty zbiorów danych](/docs/reference/dataset-formats).

## Trenowanie

FOMO jest jedyną rodziną punktową z implementacją trenowania. `train()` w
LocateAnything i SenseNova-Vision zgłasza `NotImplementedError`. Modele te
należy dostroić w projekcie nadrzędnym i wczytać wynik.

<code-tabs name="train" />

`imgsz` nie jest dowolnym wyborem dla FOMO. Domyślnie przyjmuje natywną
rozdzielczość wczytanego checkpointu, a przekazanie innej wartości zgłasza
`ValueError` z oczekiwanym rozmiarem. Informacje o zbiorach danych, loggerach i
wielu GPU znajdują się w sekcji [trenowanie](/docs/train), a wartości domyślne
tej rodziny na stronie [FOMO](/docs/models/fomo).

## Walidacja

`val()` dopasowuje przewidziane punkty jeden do jednego do punktów danych
referencyjnych (ground truth) za pomocą algorytmu węgierskiego, dla zakresu
progów odległości. Próg jest odległością euklidesową w znormalizowanych
współrzędnych obrazu, a domyślny zakres obejmuje dziesięć wartości od 0.01 do
0.10.

<code-tabs name="val" />

`metrics/precision`, `metrics/recall` i `metrics/f1` są uśredniane makro dla klas
przy najbardziej rygorystycznym progu zakresu, domyślnie 0.01.
`metrics/mAP@0.01` jest średnią precyzją przy tym samym progu, a
`metrics/mAP@[0.01:0.10]` jest średnią dla całego zakresu. Wartość zakresu jest
także `fitness`, używaną przy wyborze najlepszego checkpointu. Oba klucze mAP są
budowane z używanych progów, więc przekazanie `dist_thresholds=` zmienia ich
nazwy.

`metrics/MLE` jest średnią odległością między dopasowanymi parami przy
najbardziej rygorystycznym progu, wyrażoną w tych samych znormalizowanych
jednostkach. `metrics/MAE` i `metrics/RMSE` są metrykami zliczania, a nie
lokalizacji. Mierzą różnicę dla każdego obrazu między liczbą przewidzianych
punktów a liczbą punktów danych referencyjnych.

FOMO dodaje do nich drugą grupę na poziomie siatki. Przeszukuje wartości
pewności i `nms_radius`, a najlepszą kombinację F1 publikuje jako
`metrics/grid_F1`, `metrics/grid_precision`, `metrics/grid_recall`,
`metrics/grid_mean_distance`, `metrics/grid_TP`, `metrics/grid_FP` i
`metrics/grid_FN`. Ustawienia, które ją utworzyły, znajdują się w
`decode/threshold` i `decode/nms_radius`.

## Eksport

FOMO eksportuje się przez wspólną ścieżkę eksportu, a wyeksportowany artefakt
jest ponownie wczytywany przez `LibreYOLO()` na podstawie sufiksu pliku. Dzięki
temu plik `.onnx` lub `.engine` zachowuje się jak checkpoint i zwraca ten sam
obiekt `Results`.

<code-tabs name="export" />

Zakres poszczególnych formatów znajduje się na stronie
[FOMO](/docs/models/fomo) i w [pełnej macierzy eksportu](/docs/reference/export-matrix).
LocateAnything i SenseNova-Vision nie obsługują eksportu. `export()` zgłasza
błąd w obu przypadkach, ponieważ model generatywny nie ma możliwego do
prześledzenia grafu detekcji.

