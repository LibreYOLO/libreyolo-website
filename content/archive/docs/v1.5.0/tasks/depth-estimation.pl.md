---
title: Estymacja głębi
seo_title: Monokularna estymacja głębi w LibreYOLO
description: >-
  Przewiduj gęstą mapę względnej głębi na podstawie jednego obrazu w LibreYOLO.
  Porównaj rodziny głębi, odczytuj metryki i eksportuj model głębi.
lead: >-
  Estymacja głębi przewiduje na podstawie jednego obrazu, jak daleko od kamery
  znajduje się każdy piksel. LibreYOLO udostępnia ją jako zadanie depth, które
  zwraca gęstą mapę względnej odwrotności głębi na płótnie oryginalnego obrazu.
keywords:
  - monokularna estymacja głębi python
  - mapa głębi z jednego obrazu
  - model względnej głębi
  - depth anything libreyolo
  - gęsta predykcja głębi
last_verified: 1.5.0
snippets:
  predict:
    - label: Predykcja mapy głębi
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.data.shape)              # (H, W) na oryginalnym płótnie
        print(depth.min, depth.max, depth.mean)
    - label: Praca z wartościami
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")

        result = model(SAMPLE_IMAGE)


        depth = result.depth_map

        raw = depth.data          # większa wartość oznacza bliżej; bez
        jednostki i skali metrycznej

        gray = depth.normalized() # przeskalowane do [0, 1] na potrzeby
        wizualizacji

        print(raw.shape, float(gray.max()))
    - label: Kompaktowa alternatywa
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Ten sam kontrakt zadania, znacznie mniejsza sieć dla środowisk
        brzegowych.

        model = LibreYOLO("LibreZipDepthb-depth.pt")

        result = model(SAMPLE_IMAGE)


        print(result.depth_map.data.shape)
  val:
    - label: Walidacja i odczyt kluczy metryk
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])   # fitness
        print(metrics["metrics/delta2"], metrics["metrics/delta3"])
  export:
    - label: Eksport
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        model.export(format="onnx")
    - label: Uruchomienie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Funkcja fabrykująca wybiera ścieżkę na podstawie sufiksu pliku, więc

        # wyeksportowany artefakt wczytuje się jak checkpoint i zwraca ten sam
        obiekt Results.

        model = LibreYOLO("LibreDepthAnythingV2s-depth.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.depth_map.data.shape)
source_hash: e0612c59f9c999b4
---

## Definicja

Zadanie `depth` przewiduje jedną wartość na piksel na podstawie pojedynczego
obrazu RGB. LibreYOLO definiuje tę wartość jako względną odwrotność głębi.
Większa wartość oznacza położenie bliżej kamery, a liczby nie mają jednostki
metrycznej ani skali zachowywanej między dwoma obrazami. Porównywanie głębi
dwóch pikseli tej samej predykcji ma sens, ale porównywanie wartości z wartościami
z innego obrazu już nie.

Predykcja wypełnia `result.depth_map`, czyli strukturę `DepthMap` zawierającą
tablicę `(H, W)` na płótnie oryginalnego obrazu. Pola `.min`, `.max` i `.mean`
odczytują skończone wartości, a `.normalized()` przeskalowuje mapę do zakresu
`[0, 1]` na potrzeby wyświetlania. `result.boxes` pozostaje pusty, więc `conf`,
`iou` i `max_det` nie mają wpływu, a `save=True` zapisuje obraz mapy z nałożoną
skalą kolorów zamiast zdjęcia z adnotacjami.

## Modele

Zadanie `depth` obsługuje sześć rodzin.

[Depth Anything V2](/docs/models/depth-anything-v2) łączy enkoder DINOv2 z
dekoderem DPT i jest tutaj domyślnym rozwiązaniem ogólnego przeznaczenia.
Licencja wpływa na wybór rozmiaru równie mocno jak dokładność. Checkpoint Small
jest objęty licencją Apache-2.0, natomiast Base i Large są przeznaczone do
użytku niekomercyjnego. Przed wyborem należy sprawdzić tabelę checkpointów na
stronie modelu.

[Depth Anything 3](/docs/models/depth-anything-3) przenosi checkpoint
DA3MONO-LARGE, zwykły transformer bez specjalizacji architektury do głębi.

[ZipDepth](/docs/models/zipdepth) stanowi kompaktowy poziom. Jest to
reparametryzowalna sieć CNN destylowana z Depth Anything V2 Large, z drugim
checkpointem, którego dekoder unika operacji gather i unfold dla kompilatorów
NPU, które ich nie obsługują.

[MiDaS](/docs/models/midas) to linia prac, która ustanowiła protokół względnej
głębi zero-shot używany do oceny pozostałych rodzin. Jest to jedyna rodzina
głębi, której LibreYOLO nie publikuje ponownie. Żądanie checkpointu pobiera
oficjalny zasób z wydania GitHub autorów i sprawdza ustalony skrót SHA-256.

[LibreMODUS](/docs/models/libremodus) obsługuje głębię jako jeden z celów modelu
any-to-any zamiast dedykowanej głowicy. Wymaga zestawu zależności `modus` i
własnego uwierzytelnionego konta Hugging Face, a nie udostępnia ani `val()`, ani
`export()`.

[SenseNova-Vision](/docs/models/sensenova-vision) generuje mapę głębi jako obraz
przez dekodowanie dyfuzyjne, korzystając z tego samego checkpointu 7B, który
obsługuje sześć pozostałych zadań tej rodziny. Wymaga zestawu zależności
`sensenova`, a jego wagi są ograniczone do użytku niekomercyjnego. Licencję
podano na stronie modelu.

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane w lokalnej
pamięci podręcznej, z wyjątkiem dwóch rodzin opisanych powyżej.

<code-tabs name="predict" />

Rozdzielczość wejściowa podlega ograniczeniom zależnym od rodziny. Depth
Anything V2 i Depth Anything 3 opierają się na siatce fragmentów DINOv2,
dlatego `imgsz` musi dzielić się bez reszty przez 14, co LibreYOLO sprawdza przed
uruchomieniem. `Results.plot()` nie obsługuje tego zadania. Jest zdefiniowane
wyłącznie dla normalnych powierzchni i krawędzi. Informacje o źródłach,
streamingu i obsłudze wyników zawiera strona [predykcji](/docs/predict).

## Format zbioru danych

Walidacja głębi łączy każdy obraz z gęstą jednokanałową mapą głębi o tej samej
rozdzielczości, znajdowaną przez zastąpienie katalogu głębi w ścieżce obrazu.

```text
dataset/
  data.yaml
  images/
    val/room.jpg
  depths/
    val/room.png
```

```yaml
path: dataset
val: images/val
depths_dir: depths
nc: 1
names: {0: depth}
```

Mapy są jednokanałowymi plikami PNG lub TIF albo plikami `.npy`. Wartości
oznaczają zwykłą głębię w jednostce zachowywanej spójnie przez zbiór danych.
Piksele o wartości `0`, ujemnej, NaN lub nieskończonej oznaczają nieprawidłowe
próbki wykluczane z metryk. Mapy całkowitoliczbowe są dzielone przez
`depth_scale`, którego wartość domyślna to `256.0`, zgodnie z konwencją
16-bitowych plików PNG. Mapy float `.npy` są używane bez zmian.
`depth_stem_suffix` i `depth_mask_suffix` obsługują zbiory danych, które inaczej
nazywają pliki głębi lub maski poprawności. Pełny kontrakt opisują [formaty
zbiorów danych](/docs/reference/dataset-formats).

## Trenowanie

Żadna rodzina głębi w LibreYOLO nie ma implementacji trenowania. Funkcja
`train()` zgłasza `NotImplementedError` dla wszystkich sześciu. Strona każdego
modelu wskazuje skrypt konwersji, który zmienia checkpoint wytrenowany w
projekcie źródłowym na plik możliwy do wczytania przez LibreYOLO.

## Walidacja

Funkcja `val()` uruchamia współdzielony walidator głębi. Głębia względna nie ma
skali bezwzględnej, dlatego każda predykcja jest najpierw dopasowywana do
odwrotności danych referencyjnych za pomocą wyznaczonej dla obrazu skali i
przesunięcia metodą najmniejszych kwadratów, a następnie ponownie odwracana do
głębi. Każda poniższa metryka jest obliczana osobno dla obrazu na tej wyrównanej
mapie, a następnie uśredniana w zbiorze danych. Uwzględniane są tylko piksele
oznaczone przez zbiór danych jako prawidłowe.

<code-tabs name="val" />

`metrics/abs_rel` jest średnim bezwzględnym błędem względnym, czyli resztą
podzieloną przez głębię referencyjną. Im mniej, tym lepiej. `metrics/rmse` jest
pierwiastkiem błędu średniokwadratowego w jednostce głębi danego zbioru danych.
Tutaj również mniejsza wartość jest lepsza. `metrics/delta1`, `metrics/delta2` i
`metrics/delta3` oznaczają dokładność progową: część prawidłowych pikseli,
których stosunek do danych referencyjnych, liczony w kierunku dającym większą
wartość, jest mniejszy odpowiednio od 1.25, kwadratu 1.25 i sześcianu 1.25.
Większa wartość jest lepsza. `metrics/delta1` jest także wartością `fitness`,
używaną przy wyborze najlepszego checkpointu.

## Eksport

Wyeksportowany model głębi wczytuje się ponownie przez `LibreYOLO()` na
podstawie sufiksu pliku. Plik `.onnx` lub `.engine` działa więc jak checkpoint i
zwraca ten sam obiekt `Results`, z `depth_map` zamiast ramek.

<code-tabs name="export" />

Zakres różni się między rodzinami, a Depth Anything 3 odrzuca każdy format spoza
zweryfikowanego zestawu zamiast próbować niezweryfikowanej konwersji. Przed
wyborem formatu docelowego należy sprawdzić stronę modelu i [pełną macierz
eksportu](/docs/reference/export-matrix). LibreMODUS i SenseNova-Vision w ogóle
nie obsługują eksportu. Strona [Eksport](/docs/export) wymienia argumenty
przyjmowane przez każdy format.
