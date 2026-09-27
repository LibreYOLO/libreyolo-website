---
title: DexiNed
families:
  - dexined
seo_title: 'DexiNed: wykrywanie krawędzi, przynieś własny checkpoint'
description: >-
  Użyj DexiNed w LibreYOLO do gęstego przewidywania prawdopodobieństwa krawędzi.
  Przekonwertuj licencjonowany checkpoint, a następnie przewiduj, waliduj i
  eksportuj go.
lead: >-
  DexiNed (Dense Extreme Inception Network) to sieć konwolucyjna, która
  przewiduje gęstą mapę prawdopodobieństwa krawędzi z jednego obrazu RGB.
  LibreYOLO otacza jej architekturę tylko w celu wykrywania krawędzi; żaden
  checkpoint nie jest dołączony do biblioteki.
keywords:
  - DexiNed
  - Gęsta Ekstremalna Sieć Incepcji
  - wykrywanie krawędzi
  - BIPED
  - gęste predykcja
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        result = model(SAMPLE_IMAGE, save=True)

        edges = result.edges
        print(edges.array.shape)        # (H, W) float32 w [0, 1]
        print(edges.binary(0.5).sum())  # liczba pikseli krawędzi po progowaniu
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=weights/LibreDexiNedb-edge.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=352)

        print(metrics["metrics/ODS"])   # optymalna skala zbioru danych miara F
        print(metrics["metrics/OIS"])   # miara F-skali optymalnego obrazu
    - label: CLI
      language: bash
      code: >
        libreyolo val model=weights/LibreDexiNedb-edge.pt data=my-dataset.yaml
        imgsz=352
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        model.export(format="onnx", imgsz=352)
        model.export(format="tensorrt", imgsz=352, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=weights/LibreDexiNedb-edge.pt format=onnx
        imgsz=352

        libreyolo export model=weights/LibreDexiNedb-edge.pt format=tensorrt
        imgsz=352 half=True
    - label: Użyj wyeksportowanego pliku
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreDexiNedb-edge.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.edges.array.shape)
source_hash: 342597fde3c4ba65
---
## Instalacja

DexiNed nie potrzebuje żadnych dodatkowych opcji. Wszystko, co importuje, znajduje się w instalacji bazowej.

```bash
pip install libreyolo
```

## Predykcja

LibreYOLO nie wysyła checkpointu DexiNed. Oficjalnie opublikowane wagi są trenowane na BIPED, którego opublikowane warunki użytkowania zbioru danych ograniczają wykorzystanie do celów niekomercyjnych, więc LibreYOLO ich nie odzwierciedla. Skonwertuj checkpoint, do którego masz licencję, aby używać go z `weights/convert_dexined_weights.py`, który sprawdza klucze tensorów w stosunku do architektury w czasie wykonania przed zapisaniem pliku, który LibreYOLO może wczytać bezpośrednio:

```bash
python weights/convert_dexined_weights.py upstream.pth weights/LibreDexiNedb-edge.pt --verify
```

<code-tabs name="predict" />

`result.edges` przechowuje wynik: tablicę `(H, W)` float32 w `[0, 1]`, z `.binary(threshold)` zwracającą maskę krawędzi typu boolean. Nie ma żadnych pudełek, więc `conf`, `iou` i `max_det` nie mają wpływu. Zobacz [predykcja](/docs/predict) dla źródeł, streaming i obsługi wyników.

## Warianty

DexiNed jest wysyłany w jednym rozmiarze w LibreYOLO. Testowy pas LibreYOLO nie mierzył tej rodziny, więc nie ma opublikowanych wyników do porównania.

## Walidacja

`val()` raportuje BSDS-style ODS i OIS miary F w odniesieniu do sparowanego zbioru danych krawędzi: obrazy obok map krawędzi tego samego pnia, z opcjonalną maską ważności, tak aby wypełnione piksele nigdy się nie liczyły. `imgsz` musi być podzielny przez współczynnik próbkowania sieci, a LibreYOLO rzuca wyraźny błąd, jeśli tak nie jest.

<code-tabs name="val" />

## Eksport

<export-matrix />

Eksport brzegowy używa kontraktu uruchomieniowego o stałej rozdzielczości i partii 1: `dynamic`, a wartości inne niż 1 są odrzucane, a eksportowany graf generuje pojedynczą złączoną mapę prawdopodobieństwa. Eksportowany artefakt można ponownie załadować przez `LibreYOLO()` na jego sufiksie pliku, więc plik `.onnx` działa jak checkpoint i zwraca te same `Results`.

<code-tabs name="export" />

## Licencjonowanie

<provenance-box>

LibreYOLO nie publikuje żadnego checkpointu DexiNed. Nic nie jest odwzorowane w organizacji LibreYOLO; zamiast tego skonwertuj checkpoint, na który posiadasz licencję w `weights/convert_dexined_weights.py`.

</provenance-box>

## Cytowanie

<citation-block />
