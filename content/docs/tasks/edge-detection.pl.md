---
title: Detekcja krawędzi
seo_title: Detekcja krawędzi w LibreYOLO
description: >-
  Przewiduj gęstą mapę prawdopodobieństwa krawędzi na podstawie jednego obrazu w
  LibreYOLO. Konwertuj checkpoint, stosuj próg, waliduj za pomocą ODS i OIS oraz
  eksportuj.
lead: >-
  Detekcja krawędzi przewiduje prawdopodobieństwo, że każdy piksel leży na
  granicy obiektu. LibreYOLO udostępnia ją jako zadanie edge, które zwraca gęstą
  mapę prawdopodobieństwa na płótnie oryginalnego obrazu zamiast zestawu
  odcinków.
keywords:
  - detekcja krawędzi python
  - wykrywanie granic deep learning
  - mapa prawdopodobieństwa krawędzi
  - ODS OIS F-measure
  - gęsta predykcja krawędzi
last_verified: 1.5.0
snippets:
  predict:
    - label: Predykcja mapy krawędzi
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # LibreYOLO nie udostępnia checkpointu krawędzi; najpierw go
        przekonwertuj (poniżej).

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")

        result = model(SAMPLE_IMAGE, save=True)


        edges = result.edges

        print(edges.array.shape)          # (H, W) float32 w zakresie [0, 1]

        print(edges.binary(0.5).sum())    # liczba pikseli krawędzi przy progu
        0.5
    - label: Wybór własnego progu
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")

        result = model(SAMPLE_IMAGE)


        # Mapa ciągła jest zachowywana, więc wybór progu pozostaje po stronie
        wywołującego.

        for t in (0.3, 0.5, 0.7):
            print(t, int(result.edges.binary(t).sum()))
    - label: Zapis wizualizacji
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")

        result = model(SAMPLE_IMAGE)


        # plot() renderuje mapę; funkcja jest zdefiniowana dla wyników krawędzi
        i normalnych.

        result.plot().save("edges.png")
  val:
    - label: Walidacja i odczyt kluczy metryk
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=352)

        print(metrics["metrics/ODS"])              # fitness
        print(metrics["metrics/OIS"])
        print(metrics["metrics/best_threshold"])
    - label: Zmiana przeszukiwania i tolerancji dopasowania
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        metrics = model.val(
            data="my-dataset.yaml",
            imgsz=352,
            edge_thresholds=(0.1, 0.2, 0.3, 0.4, 0.5),
            edge_max_dist=0.0075,
        )

        print(metrics["metrics/ODS"], metrics["metrics/best_threshold"])
  export:
    - label: Eksport
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        model.export(format="onnx", imgsz=352)
    - label: Uruchomienie wyeksportowanego pliku
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Funkcja fabrykująca wybiera ścieżkę na podstawie sufiksu pliku, więc

        # wyeksportowany artefakt wczytuje się jak checkpoint i zwraca ten sam
        obiekt Results.

        model = LibreYOLO("weights/LibreDexiNedb-edge.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.edges.array.shape)
source_hash: bc286345540ed966
---

## Definicja

Zadanie `edge` przewiduje jedno prawdopodobieństwo na piksel pojedynczego obrazu
RGB. Wartość `0` oznacza brak krawędzi, a `1` krawędź. Mapa pozostaje ciągła,
więc wybór progu przekształcającego ją w binarny obraz granic należy do kodu
wywołującego, a właściwa wartość zależy od zbioru danych i dalszego zastosowania.

Predykcja wypełnia `result.edges`, czyli strukturę `EdgeMap` zawierającą tablicę
float32 `(H, W)` w zakresie `[0, 1]` na płótnie oryginalnego obrazu. `.array`
zwraca tę mapę jako NumPy, a `.binary(threshold)` zwraca maskę boolowską.
`result.boxes` pozostaje pusty, więc `conf`, `iou` i `max_det` nie mają wpływu.
`Results.plot()` obsługuje to zadanie i bezpośrednio renderuje mapę.

## Modele

Zadanie `edge` obsługują trzy rodziny.

[DexiNed](/docs/models/dexined), czyli Dense Extreme Inception Network, łączy
kilka wyjść bocznych w jedną mapę prawdopodobieństwa i działa w natywnej
rozdzielczości 352 px.

[TEED](/docs/models/teed), czyli Tiny and Efficient Edge Detector, jest małą
siecią o tej samej natywnej rozdzielczości 352 px. Jej krok próbkowania w dół
wynosi 4 wobec 16 w DexiNed, dlatego akceptuje więcej wartości `imgsz`.

[LibreMODUS](/docs/models/libremodus) tworzy krawędzie w stylu Canny jako jeden
z celów modelu any-to-any. Wymaga zestawu zależności `modus` i własnego
uwierzytelnionego konta Hugging Face, a nie udostępnia ani `val()`, ani
`export()`. Dlatego nie uczestniczy w opisanej poniżej walidacji i eksporcie.

## Predykcja

LibreYOLO nie publikuje żadnego checkpointu krawędzi. Oficjalnie wydane wagi
DexiNed i TEED wytrenowano na BIPED, którego opublikowane warunki ograniczają
użycie zbioru danych do celów niekomercyjnych. Z tego powodu LibreYOLO nie
udostępnia ich kopii. Należy przekonwertować checkpoint, na którego użycie
pozwala posiadana licencja, a następnie wczytać przekonwertowany plik ze ścieżki:

```bash
python weights/convert_dexined_weights.py upstream.pth weights/LibreDexiNedb-edge.pt --verify
```

<code-tabs name="predict" />

Nazwa pliku musi zawierać sufiks zadania `-edge`, aby moduł wczytujący ją
rozpoznał. `imgsz` musi być podzielne przez krok próbkowania w dół sieci. Jeśli
tak nie jest, LibreYOLO zgłasza jasny błąd zawierający wymagany dzielnik.
Informacje o źródłach, streamingu i obsłudze wyników zawiera strona
[predykcji](/docs/predict).

## Format zbioru danych

Walidacja krawędzi łączy każdy obraz RGB z jednokanałową mapą o tej samej nazwie
bazowej i rozdzielczości oraz opcjonalną maską poprawności.

```text
dataset/
  data.yaml
  images/
    val/scene.jpg
  edges/
    val/scene.png
  masks/
    val/scene.png
```

```yaml
path: dataset
train: images/train
val: images/val
edges_dir: edges
masks_dir: masks
nc: 1
names: {0: edge}
```

Dane docelowe muszą być jednokanałowym plikiem PNG lub TIF, a nie wizualizacją
RGB. Mapy całkowitoliczbowe są dzielone przez maksymalną wartość ich typu danych.
Mapy float muszą już zawierać wyłącznie skończone wartości w zakresie `[0, 1]`.
Niezerowe piksele maski są uznawane za prawidłowe, a piksele dopełnienia nigdy
nie wpływają na metrykę. Opcja `edge_invert: true` obsługuje źródła zapisujące
czarne krawędzie na białym tle. Pełny kontrakt opisują [formaty zbiorów
danych](/docs/reference/dataset-formats).

## Trenowanie

Żadna rodzina krawędzi w LibreYOLO nie ma implementacji trenowania. Funkcja
`train()` zgłasza `NotImplementedError` dla wszystkich trzech. Strona każdego
modelu wskazuje skrypt konwersji, który zmienia checkpoint wytrenowany gdzie
indziej na plik możliwy do wczytania przez LibreYOLO.

## Walidacja

Funkcja `val()` zgłasza miary F w stylu BSDS. Predykcje ciągłe są najpierw
ścieńczane za pomocą non-maximum suppression gradientu w czterech kierunkach,
a następnie piksele przewidywanych i referencyjnych krawędzi są dopasowywane
jeden do jednego w granicach tolerancji odległości.

<code-tabs name="val" />

`metrics/ODS` jest miarą F optymalną w skali zbioru danych. Liczby dopasowań są
łączone w całym zbiorze dla każdego progu i zgłaszana jest najlepsza z tak
połączonych miar F. Jest to również wartość `fitness`, używana przy wyborze
najlepszego checkpointu. `metrics/OIS` jest miarą F optymalną w skali obrazu,
czyli średnią z najlepszej miary F każdego obrazu. Pozwala więc każdemu obrazowi
wybrać własny próg. `metrics/best_threshold` jest pojedynczym progiem, który dał
ODS, i to jego należy ponownie użyć w `edges.binary()` podczas inferencji.

Przeszukiwanie kształtują dwa argumenty. `edge_thresholds` jest zestawem
sprawdzanych progów, domyślnie od 0.01 do 0.99 w odstępach co jedną setną.
`edge_max_dist` określa tolerancję dopasowania jako część przekątnej obrazu,
domyślnie `0.0075`. Para oddalona bardziej nie stanowi dopasowania.

## Eksport

Wyeksportowany model krawędzi wczytuje się ponownie przez `LibreYOLO()` na
podstawie sufiksu pliku. Plik `.onnx` działa więc jak checkpoint i zwraca ten
sam obiekt `Results`.

<code-tabs name="export" />

Eksport krawędzi korzysta z kontraktu środowiska uruchomieniowego o stałej
rozdzielczości i batchu równym 1. Opcja `dynamic` i `batch` inny niż 1 są
odrzucane, a wyeksportowany graf emituje jedną połączoną mapę
prawdopodobieństwa. Zakres poszczególnych formatów podano na stronach
[DexiNed](/docs/models/dexined) i [TEED](/docs/models/teed) oraz w [pełnej
macierzy eksportu](/docs/reference/export-matrix). Strona
[Eksport](/docs/export) wymienia argumenty przyjmowane przez każdy format.
